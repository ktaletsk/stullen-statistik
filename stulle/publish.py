"""Publish the pipeline's state as a git-friendly dataset, and restore state from it.

The dataset lives in the repository (dataset/) and is what the notebooks read, locally or
straight from GitHub. It is split into partitions of PART consecutive IDs, so a daily update
only rewrites the newest one or two partitions and git history grows slowly; files that did
not change are not touched at all.

dataset/
    meta.json                          when, how many, which partitions
    catalog.json                       ingredients, breads, spreads, classics: German + English names, groups
    ids_missing.txt                    IDs that 404 (deleted by their maker or the moderator)
    ids_blank.txt                      IDs whose photo is blank (blacked out by moderation, or a broken render)
    records/part-00000.jsonl.gz        raw records from the API, stroke geometry dropped (IDs 1-2000, ...)
    stullen/part-00000.parquet         one row per sandwich, with height estimates and classic matches
    toppings/part-00000.parquet        one row per topping piece: type, position, rotation, stacking order
    spreads/part-00000.parquet         one row per spread pass
    clip/part-00000.ids.npy            IDs of ...
    clip/part-00000.emb.npy            ... their CLIP ViT-L/14 photo embeddings (float16, L2-normalised)
    map.parquet                        id, x, y (UMAP layout of the embeddings), category, confidence (zero-shot CLIP)
    hunts.parquet                      hunt, rank, id, score: best photo matches for the preset searches
    positions.parquet                  type, x_cm, y_cm, pieces: where each topping lands on the bread

Banned (blank photo) and deleted sandwiches are left out of every file except the two ID
lists. Photos are not copied: they stay at img.stulle.app/{id}/thumb.jpg, so anything
removed there disappears here too.
"""
from __future__ import annotations

import gzip
import io
import json
import time
from pathlib import Path

import numpy as np
import polars as pl

from stulle import model
from stulle import pipeline as p

OUT = Path("dataset")
PART = 2000

# Zero-shot photo categories and preset searches. Prompts are embedded with CLIP's text
# encoder and compared with every photo; precomputing them here means the browser (WASM)
# versions of the notebook never need the 1.7 GB model.
TAXONOMY = {
    "Normal Stulle": "a photo of an ordinary open sandwich with some toppings",
    "Just a spread": "a photo of a slice of bread with only a spread on it",
    "Bare bread": "a photo of an empty plain slice of bread",
    "Tower": "a photo of a very tall tower of stacked food",
    "Chaos heap": "a photo of a chaotic heap of many random foods on bread",
    "Drawing / symbol": "a picture or symbol painted with spread on bread",
    "Words": "letters and words written on bread",
    "Face": "a face made of food on bread",
    "Flag": "a national flag painted on bread",
    "Rude": "a phallic shape made of food",
}
HUNTS = {
    "☭ Hammer and sickle": "a hammer and sickle symbol drawn on bread",
    "✍️ Words and slogans": "words and letters written on bread",
    "🍆 Rude shapes": "a phallic shape made of food",
    "🏳️ Flags": "a national flag painted on bread",
    "🙂 Faces": "a smiley face made of food on bread",
    "❤️ Hearts": "a heart shape on bread",
    "🐱 Animals": "a drawing of an animal on bread",
    "🗼 Towers": "a very tall tower of stacked food",
    "🌀 Pure chaos": "a chaotic heap of many random foods",
    "🍞 Perfectly normal Stulle": "a normal German open sandwich with cheese and cucumber",
}
HUNT_TOP = 96


def part_name(k: int) -> str:
    return f"part-{k:05d}"


def part_of(ids) -> np.ndarray:
    return (np.asarray(ids) - 1) // PART


# --- writers that leave unchanged files alone ---------------------------------------------
def _write_bytes(path: Path, data: bytes) -> bool:
    if path.exists() and path.read_bytes() == data:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_bytes(data)
    tmp.replace(path)
    return True


def _write_parquet(path: Path, df: pl.DataFrame) -> bool:
    if path.exists():
        try:
            if pl.read_parquet(path).equals(df):
                return False
        except Exception:
            pass
    buf = io.BytesIO()
    df.write_parquet(buf, compression="zstd", compression_level=9, statistics=True)
    return _write_bytes(path, buf.getvalue())


def _write_npy(path: Path, arr: np.ndarray) -> bool:
    buf = io.BytesIO()
    np.save(buf, arr)
    return _write_bytes(path, buf.getvalue())


def _write_text(path: Path, text: str) -> bool:
    return _write_bytes(path, text.encode())


# --- publish -----------------------------------------------------------------------------
def catalog() -> dict:
    return {
        "toppings": {k: {"de": v["name"], "en": model.EN.get(k, v["name"]), "kind": v["kind"]} for k, v in model.TOPPINGS.items()},
        "spreads": {k: {"de": v, "en": model.EN.get(k, v), "kind": "spread"} for k, v in model.SPREADS.items()},
        "breads": {k: {"de": v, "en": model.EN.get(k, v)} for k, v in model.BREADS.items()},
        "kinds": {k: {"de": model.KIND_DE[k], "en": model.KIND_EN[k]} for k in model.KINDS},
        # rules for a strict match: all of `need`, at least one of `spreads` (if given), spreads
        # limited to `only_spreads` (if given), and no toppings beyond `need` + `allow`
        "classics": {
            k: {
                "de": k,
                "en": model.FAMOUS_EN.get(k, k),
                "need": sorted(r["need"]),
                "allow": sorted(r["allow"]),
                "spreads": sorted(r["spreads"]) if r.get("spreads") else None,
                "only_spreads": sorted(r["only_spreads"]) if r.get("only_spreads") else None,
            }
            for k, r in model.FAMOUS.items()
        },
        "taxonomy": list(TAXONOMY),
        "hunts": list(HUNTS),
    }


def _tables(rows: list[dict]):
    stullen, toppings, spreads = model.build_tables(rows)
    stullen = stullen.with_columns(
        pl.struct("topping_types", "spread_types")
        .map_elements(lambda r: model.classify(r["topping_types"], r["spread_types"]), return_dtype=pl.List(pl.Utf8))
        .alias("classics")
    )
    # float32 is plenty for positions and heights and halves the files
    f32 = [c for c, t in stullen.schema.items() if t == pl.Float64]
    stullen = stullen.with_columns(pl.col(f32).cast(pl.Float32))
    toppings = toppings.with_columns(pl.col("x", "y", "rot").cast(pl.Float32), pl.col("order").cast(pl.Int32))
    spreads = spreads.with_columns(pl.col("order").cast(pl.Int32))
    return stullen, toppings, spreads


def _text_model():
    return p._clip()


def publish(force: bool = False) -> dict:
    """Write dataset/ from the pipeline state in data/. Returns a summary of what changed."""
    t0 = time.time()
    missing, blank = p.read_ids(p.MISSING), p.read_ids(p.BLANK)
    drop = missing | blank
    rows = [r for r in model.load_raw(p.RECORDS) if r["id"] not in drop]
    rows.sort(key=lambda r: r["id"])
    by_part: dict[int, list[dict]] = {}
    for r in rows:
        by_part.setdefault((r["id"] - 1) // PART, []).append(r)
    changed: list[str] = []

    # records + tables, one partition at a time; tables are rebuilt only when records changed
    for k, part in sorted(by_part.items()):
        name = part_name(k)
        lines = "".join(json.dumps(r, separators=(",", ":"), ensure_ascii=False) + "\n" for r in part)
        rec_changed = _write_bytes(OUT / "records" / f"{name}.jsonl.gz", gzip.compress(lines.encode(), compresslevel=9, mtime=0))
        if rec_changed or force or not (OUT / "stullen" / f"{name}.parquet").exists():
            s, t, sp = _tables(part)
            for sub, df in (("stullen", s), ("toppings", t), ("spreads", sp)):
                if _write_parquet(OUT / sub / f"{name}.parquet", df):
                    changed.append(f"{sub}/{name}")
        if rec_changed:
            changed.append(f"records/{name}")

    # embeddings, partitioned the same way
    kept = np.array(sorted(r["id"] for r in rows), dtype=np.int64)
    if p.EMB_IDS.exists():
        ids, emb = p._load_aligned()
        keep = np.isin(ids, kept)
        ids, emb = ids[keep], emb[keep]
        order = np.argsort(ids)
        ids, emb = ids[order], emb[order].astype(np.float16)
        parts = part_of(ids)
        for k in np.unique(parts):
            m = parts == k
            name = part_name(int(k))
            if _write_npy(OUT / "clip" / f"{name}.emb.npy", emb[m]) | _write_npy(OUT / "clip" / f"{name}.ids.npy", ids[m]):
                changed.append(f"clip/{name}")
        changed += _derived(ids, emb)

    changed += _positions()
    for name, text in (
        ("ids_missing.txt", "".join(f"{i}\n" for i in sorted(missing))),
        ("ids_blank.txt", "".join(f"{i}\n" for i in sorted(blank))),
        ("catalog.json", json.dumps(catalog(), ensure_ascii=False, indent=1) + "\n"),
    ):
        if _write_text(OUT / name, text):
            changed.append(name)
    # The timestamp moves only when the data does, so a run with nothing new commits nothing.
    old = json.loads((OUT / "meta.json").read_text()) if (OUT / "meta.json").exists() else {}
    meta = {
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) if changed or not old else old["updated_at"],
        "source": "https://stulle.app/api/photos/{id}",
        "photos": "https://img.stulle.app/{id}/thumb.jpg",
        "max_id_checked": int(max(drop | {r["id"] for r in rows})) if rows else 0,
        "sandwiches": len(rows),
        "deleted": len(missing),
        "blank_photos": len(blank - missing),
        "part_size": PART,
        "parts": [part_name(k) for k in sorted(by_part)],
        "clip_model": p.CLIP_MODEL,
    }
    _write_text(OUT / "meta.json", json.dumps(meta, indent=1) + "\n")
    p.log(f"[publish] {len(rows):,} sandwiches in {len(by_part)} partitions; {len(changed)} files changed ({time.time() - t0:.0f}s)")
    return {"changed": changed, **meta}


def _derived(ids: np.ndarray, emb: np.ndarray) -> list[str]:
    """Map layout + zero-shot categories + preset searches (needs the CLIP text encoder)."""
    changed = []
    if not len(ids):
        return changed
    model_ = _text_model()
    e = emb.astype(np.float32)

    def text(prompts):
        return model_.encode(list(prompts), convert_to_numpy=True, normalize_embeddings=True)

    logits = 100 * e @ text(TAXONOMY.values()).T
    prob = np.exp(logits - logits.max(axis=1, keepdims=True))
    prob /= prob.sum(axis=1, keepdims=True)
    names = list(TAXONOMY)
    cats = pl.DataFrame({"id": ids, "category": [names[k] for k in prob.argmax(axis=1)], "confidence": prob.max(axis=1).astype(np.float32)})
    if p.MAP_IDS.exists():
        mids, xy = np.load(p.MAP_IDS), np.load(p.MAP)
        n = min(len(mids), len(xy))
        xy_df = pl.DataFrame({"id": mids[:n], "x": xy[:n, 0].astype(np.float32), "y": xy[:n, 1].astype(np.float32)})
        cats = cats.join(xy_df, on="id", how="left")
    else:
        cats = cats.with_columns(pl.lit(None, pl.Float32).alias("x"), pl.lit(None, pl.Float32).alias("y"))
    if _write_parquet(OUT / "map.parquet", cats.select("id", "x", "y", "category", "confidence").sort("id")):
        changed.append("map")

    scores = e @ text(HUNTS.values()).T
    frames = []
    for j, hunt in enumerate(HUNTS):
        top = np.argsort(-scores[:, j])[:HUNT_TOP]
        frames.append(pl.DataFrame({"hunt": hunt, "rank": np.arange(1, len(top) + 1, dtype=np.int32), "id": ids[top], "score": scores[top, j].astype(np.float32)}))
    if _write_parquet(OUT / "hunts.parquet", pl.concat(frames)):
        changed.append("hunts")
    return changed


def _positions(bins: int = 40, half_cm: float = 6.6) -> list[str]:
    """Where each topping lands: a 2D histogram in cm from the centre of the 11 cm slice."""
    files = sorted((OUT / "toppings").glob("part-*.parquet"))
    if not files:
        return []
    t = pl.concat([pl.read_parquet(f, columns=["type", "x", "y"]) for f in files])
    edges = np.linspace(-half_cm, half_cm, bins + 1)
    centres = ((edges[:-1] + edges[1:]) / 2).round(3).astype(np.float32)
    frames = []
    for typ, g in t.group_by("type"):
        h, _, _ = np.histogram2d(11 * g["x"].to_numpy(), 11 * g["y"].to_numpy(), bins=[edges, edges])
        ix, iy = np.nonzero(h)
        frames.append(pl.DataFrame({"type": typ[0], "x_cm": centres[ix], "y_cm": centres[iy], "pieces": h[ix, iy].astype(np.int32)}))
    df = pl.concat(frames).sort("type", "x_cm", "y_cm")
    return ["positions"] if _write_parquet(OUT / "positions.parquet", df) else []


# --- restore -----------------------------------------------------------------------------
def restore() -> None:
    """Rebuild the pipeline state in data/ from dataset/ (for a fresh checkout, e.g. in CI)."""
    p.DATA.mkdir(exist_ok=True)
    p.EMB_DIR.mkdir(parents=True, exist_ok=True)
    with p.RECORDS.open("wb") as f:
        for path in sorted((OUT / "records").glob("part-*.jsonl.gz")):
            f.write(gzip.decompress(path.read_bytes()))
    (OUT / "ids_missing.txt").exists() and p.MISSING.write_text((OUT / "ids_missing.txt").read_text())
    (OUT / "ids_blank.txt").exists() and p.BLANK.write_text((OUT / "ids_blank.txt").read_text())
    id_files = sorted((OUT / "clip").glob("part-*.ids.npy"))
    if id_files:
        ids = np.concatenate([np.load(f) for f in id_files])
        emb = np.concatenate([np.load(f.with_name(f.name.replace(".ids.", ".emb."))) for f in id_files])
        p._save_npy(p.EMB, emb)
        p._save_npy(p.EMB_IDS, ids)
    if (OUT / "map.parquet").exists():
        m = pl.read_parquet(OUT / "map.parquet").drop_nulls("x")
        p._save_npy(p.MAP, np.stack([m["x"].to_numpy(), m["y"].to_numpy()], axis=1).astype(np.float32))
        p._save_npy(p.MAP_IDS, m["id"].to_numpy().astype(np.int64))
    p.log(f"[restore] {len(p.record_ids()):,} records, {len(p.read_ids(p.MISSING)):,} deleted, {len(p.read_ids(p.BLANK)):,} blank")
