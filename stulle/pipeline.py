"""Data pipeline for stulle.app: records -> thumbnails -> CLIP embeddings.

Every stage is incremental and idempotent: it only does work that isn't on disk yet, so
running it again picks up newly created sandwiches. The CLI lives in ../pipeline.py.

Files (all under data/):
    stullen.jsonl        one slim JSON record per sandwich (append-only)
    missing.txt          IDs that returned 404 (deleted / reported)
    thumbs/{id}.jpg      224px thumbnails
    thumbs_missing.txt   IDs whose image 404'd
    blank.txt            IDs whose photo is blank: the black placeholder shown for moderated
                         (banned) sandwiches, or a broken render with no visible content
    emb/clip_ids.npy     int64 IDs, aligned with ...
    emb/clip.npy         float16 L2-normalised CLIP ViT-L/14 image embeddings
    emb/umap_ids.npy     IDs, aligned with ...
    emb/umap.npy         float32 2D UMAP layout of the embeddings (for the notebook's map)
"""
from __future__ import annotations

import asyncio
import heapq
import io
import json
import os
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

DATA = Path("data")
RECORDS = DATA / "stullen.jsonl"
MISSING = DATA / "missing.txt"
THUMBS = DATA / "thumbs"
THUMBS_MISSING = DATA / "thumbs_missing.txt"
BLANK = DATA / "blank.txt"
EMB_DIR = DATA / "emb"
EMB_IDS = EMB_DIR / "clip_ids.npy"
EMB = EMB_DIR / "clip.npy"
MAP_IDS = EMB_DIR / "umap_ids.npy"
MAP = EMB_DIR / "umap.npy"

API = "https://stulle.app/api/photos/{}"
THUMB_URL = "https://img.stulle.app/{}/thumb.jpg"
HEADERS = {"User-Agent": "stulle-stats/0.2 (personal data analysis; polite, throttled)"}
BUCKET = 500  # coverage is balanced across blocks of this many IDs
CLIP_MODEL = "clip-ViT-L-14"


def log(msg: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}", flush=True)


# --- small file helpers --------------------------------------------------------------
def read_ids(path: Path) -> set[int]:
    if not path.exists():
        return set()
    return {int(x) for x in path.read_text().split() if x.strip().isdigit()}


def record_ids() -> set[int]:
    ids = set()
    if RECORDS.exists():
        with RECORDS.open() as f:
            for line in f:
                try:
                    ids.add(json.loads(line)["id"])
                except (json.JSONDecodeError, KeyError):
                    pass
    return ids


def repair_jsonl(path: Path = RECORDS) -> None:
    """Drop a half-written last line (left behind if a run was killed mid-write)."""
    if not path.exists() or path.stat().st_size == 0:
        return
    with path.open("rb+") as f:
        f.seek(-1, os.SEEK_END)
        if f.read(1) == b"\n":
            return
        f.seek(0)
        content = f.read()
        cut = content.rfind(b"\n") + 1
        f.truncate(cut)
    log(f"repaired {path}: dropped a partial last line")


def slim(d: dict) -> dict:
    """Keep everything except brush-stroke geometry (replaced by its segment count)."""
    s = d.get("stulle") or {}
    ops = [{"spread": op["r"]} if "r" in op else {"stroke": len(op.get("c", []))} for op in s.get("ops", [])]
    return {
        "id": d["id"],
        "parentId": d.get("parentId"),
        "replyTo": d.get("replyTo"),
        "createdAt": d.get("createdAt"),
        "timezone": d.get("timezone"),
        "camera": d.get("camera"),
        "v": s.get("v"),
        "bread": s.get("bread"),
        "ops": ops,
        "toppings": s.get("toppings", []),
    }


# --- polite rate limiting --------------------------------------------------------------
@dataclass
class Limiter:
    """Global requests/second limiter that backs off on HTTP 429 and recovers slowly."""

    rate: float
    max_rate: float
    name: str = "api"
    _next: float = field(default_factory=time.monotonic)
    _pause_until: float = 0.0
    _streak: int = 0
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def wait(self) -> None:
        async with self._lock:
            now = time.monotonic()
            t = max(self._next, self._pause_until, now)
            self._next = t + 1.0 / self.rate
        await asyncio.sleep(max(0.0, t - time.monotonic()))

    async def throttled(self) -> None:
        async with self._lock:
            if time.monotonic() > self._pause_until:
                self.rate = max(0.3, self.rate * 0.7)
                self._pause_until = time.monotonic() + 60
                self._streak = 0
                log(f"[{self.name}] 429 -> pausing 60s, rate now {self.rate:.2f}/s")

    def ok(self) -> None:
        self._streak += 1
        if self._streak >= 600 and self.rate < self.max_rate:
            self.rate = min(self.max_rate, self.rate * 1.1)
            self._streak = 0


async def _get(client: httpx.AsyncClient, url: str, lim: Limiter, method: str = "GET") -> httpx.Response | None:
    """GET (or HEAD) with throttling and retries. Returns the final response (200/404/other) or None."""
    for attempt in range(20):
        await lim.wait()
        try:
            r = await client.request(method, url)
        except httpx.HTTPError:
            await asyncio.sleep(2 ** min(attempt, 5))
            continue
        if r.status_code == 429:
            await lim.throttled()
            continue
        if r.status_code >= 500:
            await asyncio.sleep(2 ** min(attempt, 5))
            continue
        lim.ok()
        return r
    return None


# --- stage 1: records ---------------------------------------------------------------------
async def latest_id(client: httpx.AsyncClient) -> int:
    html = (await client.get("https://stulle.app/alle")).text
    return max(int(x) for x in re.findall(r'href="/(\d+)"', html))


def balanced_order(todo: list[int], done: set[int], latest: int, seed: int = 42) -> list[int]:
    """Order IDs so coverage stays even across ID blocks at every point in the run.

    Always takes the next ID from the BUCKET-sized block with the lowest coverage, in a
    random order within each block. New IDs (0% covered) therefore catch up first, and a
    partial scrape stays an (almost) uniform random sample of the whole ID range.
    """
    rng = random.Random(seed)
    blocks: dict[int, list[int]] = {}
    for i in todo:
        blocks.setdefault((i - 1) // BUCKET, []).append(i)
    heap = []
    for b, ids in blocks.items():
        rng.shuffle(ids)
        size = min(BUCKET, latest - b * BUCKET)
        have = size - len(ids)  # already done in this block
        heapq.heappush(heap, (have / size, rng.random(), b, have, size))
    order = []
    while heap:
        _, tie, b, have, size = heapq.heappop(heap)
        order.append(blocks[b].pop())
        have += 1
        if blocks[b]:
            heapq.heappush(heap, (have / size, tie, b, have, size))
    return order


async def scrape(rate: float = 2.5, max_rate: float = 3.5, concurrency: int = 3, until: int | None = None) -> int:
    """Fetch every not-yet-seen sandwich up to the newest ID. Returns how many were fetched."""
    DATA.mkdir(exist_ok=True)
    repair_jsonl()
    done = record_ids() | read_ids(MISSING) | read_ids(BLANK)  # published state keeps no records for banned ones
    async with httpx.AsyncClient(headers=HEADERS, timeout=30) as client:
        latest = until or await latest_id(client)
        todo = [i for i in range(1, latest + 1) if i not in done]
        log(f"[scrape] newest #{latest:,}; {len(done):,} known; {len(todo):,} to fetch")
        if not todo:
            return 0
        q: asyncio.Queue[int] = asyncio.Queue()
        for i in balanced_order(todo, done, latest):
            q.put_nowait(i)
        lim = Limiter(rate, max_rate, "api")
        n = 0
        with RECORDS.open("a", buffering=1) as out, MISSING.open("a", buffering=1) as miss:

            async def worker():
                nonlocal n
                while not q.empty():
                    i = q.get_nowait()
                    r = await _get(client, API.format(i), lim)
                    if r is not None and r.status_code == 200:
                        out.write(json.dumps(slim(r.json()), separators=(",", ":")) + "\n")
                    elif r is not None and r.status_code == 404:
                        miss.write(f"{i}\n")
                    n += 1
                    if n % 500 == 0:
                        log(f"[scrape] {n:,}/{len(todo):,} rate={lim.rate:.2f}/s")

            await asyncio.gather(*(worker() for _ in range(concurrency)))
    log(f"[scrape] done: {n:,} fetched")
    return n


# --- stage 2: thumbnails ------------------------------------------------------------------
async def thumbs(rate: float = 8.0, concurrency: int = 6, size: int = 224) -> int:
    from PIL import Image

    THUMBS.mkdir(parents=True, exist_ok=True)
    have = {int(p.stem) for p in THUMBS.glob("*.jpg")} | read_ids(THUMBS_MISSING)
    # Thumbnails are only needed until a photo is embedded, so a fresh checkout restored from
    # the published dataset (embeddings, no thumbnails) only downloads the new ones.
    embedded = set(_load_aligned()[0].tolist()) if EMB_IDS.exists() and EMB.exists() else set()
    todo = sorted(record_ids() - have - embedded - read_ids(BLANK))
    log(f"[thumbs] {len(todo):,} to fetch")
    if not todo:
        return 0
    q: asyncio.Queue[int] = asyncio.Queue()
    for i in todo:
        q.put_nowait(i)
    lim = Limiter(rate, rate * 1.5, "img")
    n = 0
    async with httpx.AsyncClient(headers=HEADERS, timeout=30) as client:
        with THUMBS_MISSING.open("a", buffering=1) as miss:

            async def worker():
                nonlocal n
                while not q.empty():
                    i = q.get_nowait()
                    r = await _get(client, THUMB_URL.format(i), lim)
                    if r is not None and r.status_code == 200:
                        img = Image.open(io.BytesIO(r.content)).convert("RGB")
                        img.thumbnail((size, size))
                        tmp = THUMBS / f".{i}.tmp.jpg"
                        img.save(tmp, quality=88)
                        tmp.replace(THUMBS / f"{i}.jpg")
                        if is_blank(THUMBS / f"{i}.jpg"):
                            _append(BLANK, i)
                    elif r is not None and r.status_code == 404:
                        miss.write(f"{i}\n")
                    n += 1
                    if n % 1000 == 0:
                        log(f"[thumbs] {n:,}/{len(todo):,}")

            await asyncio.gather(*(worker() for _ in range(concurrency)))
    log(f"[thumbs] done: {n:,}")
    return n


def is_blank(path: Path) -> bool:
    """True for an image with no visible content: one flat colour or fine noise.

    Catches the all-black placeholder that replaces moderated sandwiches and the odd broken
    render (a single flat colour or thin stripes). Real photos always show the wooden board,
    so their brightness varies far more.
    """
    import numpy as np
    from PIL import Image

    im = Image.open(path)
    im.draft("L", (28, 28))
    return float(np.asarray(im.convert("L"), dtype=np.float32).std()) < 2.0


def _append(path: Path, i: int) -> None:
    with path.open("a") as f:
        f.write(f"{i}\n")


def find_blank() -> int:
    """Scan every thumbnail for blank images and add them to blank.txt (about 15 s for 50k).

    Entries are only ever added: a checkout restored from the published dataset has few
    thumbnails, and the IDs it doesn't have images for must stay flagged.
    """
    from concurrent.futures import ThreadPoolExecutor

    paths = sorted(THUMBS.glob("*.jpg"))
    with ThreadPoolExecutor(8) as ex:
        flags = list(ex.map(is_blank, paths))
    ids = sorted({int(p.stem) for p, b in zip(paths, flags) if b} | read_ids(BLANK))
    tmp = BLANK.with_suffix(".tmp")
    tmp.write_text("".join(f"{i}\n" for i in ids))
    tmp.replace(BLANK)
    log(f"[blank] {len(ids):,} blank photos flagged ({len(paths):,} thumbnails scanned)")
    return len(ids)


async def recheck(last: int | None = 2000, rate: float = 8.0, concurrency: int = 6) -> int:
    """Look at the newest `last` sandwiches again (None = all); moderation often happens after we fetched them.

    A HEAD request per thumbnail is enough: a deleted sandwich returns 404, and the black
    placeholder is ~4 KB while real thumbnails are ~100 KB, so only small ones get downloaded.
    """
    from PIL import Image

    THUMBS.mkdir(parents=True, exist_ok=True)
    gone, blank = read_ids(MISSING), read_ids(BLANK)
    todo = sorted(record_ids() - gone - blank)
    todo = todo[-last:] if last else todo
    if not todo:
        return 0
    lim = Limiter(rate, rate * 1.5, "img")
    q: asyncio.Queue[int] = asyncio.Queue()
    for i in todo:
        q.put_nowait(i)
    n_gone = n_blank = 0
    async with httpx.AsyncClient(headers=HEADERS, timeout=30) as client:

        async def worker():
            nonlocal n_gone, n_blank
            while not q.empty():
                i = q.get_nowait()
                r = await _get(client, THUMB_URL.format(i), lim, method="HEAD")
                if r is None:
                    continue
                if r.status_code == 404:
                    _append(MISSING, i)
                    n_gone += 1
                elif r.status_code == 200 and int(r.headers.get("content-length", 10**6)) < 10_000:
                    g = await _get(client, THUMB_URL.format(i), lim)
                    if g is not None and g.status_code == 200:
                        tmp = THUMBS / f".{i}.tmp.jpg"
                        Image.open(io.BytesIO(g.content)).convert("RGB").save(tmp, quality=88)
                        if is_blank(tmp):
                            tmp.replace(THUMBS / f"{i}.jpg")
                            _append(BLANK, i)
                            n_blank += 1
                        else:
                            tmp.unlink()

        await asyncio.gather(*(worker() for _ in range(concurrency)))
    log(f"[recheck] newest {len(todo):,}: {n_gone} deleted since, {n_blank} blanked since")
    return n_gone + n_blank


# --- stage 3: CLIP embeddings ----------------------------------------------------------------
_model = None


def _clip():
    global _model
    if _model is None:
        import torch
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(CLIP_MODEL, device="mps" if torch.backends.mps.is_available() else "cpu")
    return _model


def _save_npy(path: Path, arr) -> None:
    import numpy as np

    tmp = path.with_suffix(".tmp.npy")
    with tmp.open("wb") as f:
        np.save(f, arr)
    tmp.replace(path)


def embed(batch: int = 64, checkpoint_every: int = 2000) -> int:
    """Embed thumbnails that aren't embedded yet. Checkpoints so a crash loses little."""
    import numpy as np
    from PIL import Image

    EMB_DIR.mkdir(parents=True, exist_ok=True)
    if EMB_IDS.exists() and EMB.exists():
        ids, emb = _load_aligned()
    else:
        ids, emb = np.zeros(0, dtype=np.int64), np.zeros((0, 768), dtype=np.float16)
    have = set(ids.tolist())
    todo = sorted(int(p.stem) for p in THUMBS.glob("*.jpg") if int(p.stem) not in have)
    log(f"[embed] {len(have):,} embedded; {len(todo):,} new")
    if not todo:
        return 0
    model = _clip()
    new_ids, new_emb = [], []

    def flush():
        nonlocal ids, emb, new_ids, new_emb
        if new_ids:
            ids = np.concatenate([ids, np.array(new_ids, dtype=np.int64)])
            emb = np.concatenate([emb, np.concatenate(new_emb)])
            _save_npy(EMB, emb)  # embeddings first, ids last: readers trim to the shorter one
            _save_npy(EMB_IDS, ids)
            new_ids, new_emb = [], []

    for k in range(0, len(todo), batch):
        chunk = todo[k : k + batch]
        imgs = [Image.open(THUMBS / f"{i}.jpg").convert("RGB") for i in chunk]
        e = model.encode(imgs, batch_size=batch, convert_to_numpy=True, normalize_embeddings=True)
        new_ids.extend(chunk)
        new_emb.append(e.astype(np.float16))
        if len(new_ids) >= checkpoint_every:
            flush()
            log(f"[embed] {k + len(chunk):,}/{len(todo):,}")
    flush()
    log(f"[embed] done: {len(ids):,} total")
    return len(todo)


# --- stage 4: 2D map ---------------------------------------------------------------------------
def project(min_new: int = 500, force: bool = False) -> int:
    """UMAP layout of all embeddings, recomputed when at least `min_new` new ones arrived.

    Runs here rather than in the notebook: it takes a minute or two on ~40k photos, and
    Numba (used by UMAP) aborts when two notebook sessions run it at the same time.
    """
    import numpy as np

    if not EMB_IDS.exists():
        return 0
    ids, emb = _load_aligned()
    old = np.load(MAP_IDS) if MAP_IDS.exists() else np.zeros(0, dtype=np.int64)
    new = len(set(ids.tolist()) - set(old.tolist()))
    if not force and (new == 0 or (len(old) and new < min_new)):
        log(f"[map] up to date ({len(old):,} placed, {new:,} new)")
        return 0
    import umap

    log(f"[map] projecting {len(ids):,} embeddings")
    xy = umap.UMAP(n_neighbors=25, min_dist=0.05, metric="cosine", random_state=42).fit_transform(emb.astype(np.float32))
    _save_npy(MAP, xy.astype(np.float32))
    _save_npy(MAP_IDS, ids)
    log("[map] done")
    return len(ids)


def _load_aligned():
    """Embeddings and their IDs, trimmed to the same length (writers save embeddings first)."""
    import numpy as np

    ids, emb = np.load(EMB_IDS), np.load(EMB)
    n = min(len(ids), len(emb))
    return ids[:n], emb[:n]


# --- status ---------------------------------------------------------------------------------
def coverage_by_block(ids_seen: set[int], latest: int) -> list[float]:
    counts: dict[int, int] = {}
    for i in ids_seen:
        if i <= latest:
            counts[(i - 1) // BUCKET] = counts.get((i - 1) // BUCKET, 0) + 1
    nb = (latest - 1) // BUCKET + 1
    return [counts.get(b, 0) / min(BUCKET, latest - b * BUCKET) for b in range(nb)]


async def status() -> dict:
    import numpy as np

    recs, miss = record_ids(), read_ids(MISSING)
    async with httpx.AsyncClient(headers=HEADERS, timeout=30) as client:
        latest = await latest_id(client)
    cov = coverage_by_block(recs | miss, latest)
    n_emb = len(np.load(EMB_IDS)) if EMB_IDS.exists() else 0
    return {
        "newest_on_site": latest,
        "records": len(recs),
        "deleted_404": len(miss),
        "coverage_overall": (len(recs) + len(miss)) / latest,
        "coverage_min_block": min(cov),
        "coverage_max_block": max(cov),
        "thumbs": len(list(THUMBS.glob("*.jpg"))) if THUMBS.exists() else 0,
        "blank_photos": len(read_ids(BLANK)),
        "embeddings": n_emb,
        "map_points": len(np.load(MAP_IDS)) if MAP_IDS.exists() else 0,
    }
