# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "httpx>=0.28.1",
#     "numpy>=2.5.5",
#     "pillow>=13.0.0",
#     "polars>=2.1.0",
#     "sentence-transformers>=6.3.0",
#     "torch>=2.15.0",
#     "umap-learn>=0.6.0",
# ]
#
# # On Linux (GitHub Actions) use the CPU-only PyTorch build: the default one pulls in
# # ~2.5 GB of CUDA libraries the runner can't use.
# [tool.uv.sources]
# torch = [{ index = "pytorch-cpu", marker = "sys_platform == 'linux'" }]
#
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
# ///
"""stulle.app data pipeline: records -> thumbnails -> CLIP embeddings -> 2D map.

Steps: scrape (new records), recheck (newest sandwiches deleted or blacked out since),
thumbs, blank (flag blank photos), embed (CLIP), map (UMAP).

    uv run pipeline.py              one pass over everything new, then exit
    uv run pipeline.py --steps scrape,thumbs
    uv run pipeline.py watch        keep going: new sandwiches every --every seconds
    uv run pipeline.py status       coverage and counts, no downloads
    uv run pipeline.py publish      write the shareable dataset/ (partitioned, git-friendly)
    uv run pipeline.py restore      rebuild data/ from dataset/ (fresh checkout, CI)

Every step only does what isn't on disk yet, so re-running is cheap and safe. Only one
instance runs at a time (data/.pipeline.lock). For a long run, detach it:

    nohup uv run pipeline.py watch >> data/pipeline.log 2>&1 &
"""
import argparse
import asyncio
import json
import os
import signal
import sys
from pathlib import Path

from stulle import pipeline as p

LOCK = Path("data/.pipeline.lock")
STEPS = ("scrape", "recheck", "thumbs", "blank", "embed", "map")


def acquire_lock() -> None:
    LOCK.parent.mkdir(exist_ok=True)
    if LOCK.exists():
        try:
            pid = int(LOCK.read_text())
            os.kill(pid, 0)
            sys.exit(f"pipeline already running (pid {pid}); stop it or wait. Lock: {LOCK}")
        except (ValueError, ProcessLookupError, PermissionError):
            pass  # stale lock
    LOCK.write_text(str(os.getpid()))


def release_lock() -> None:
    if LOCK.exists() and LOCK.read_text() == str(os.getpid()):
        LOCK.unlink()


async def run_once(steps, a) -> None:
    if "scrape" in steps:
        await p.scrape(rate=a.rate, max_rate=a.max_rate)
    if "recheck" in steps:
        await p.recheck(last=a.recheck_last or None)
    if "thumbs" in steps:
        await p.thumbs()
    if "blank" in steps:
        await asyncio.to_thread(p.find_blank)
    if "embed" in steps:
        await asyncio.to_thread(p.embed)
    if "map" in steps:
        await asyncio.to_thread(p.project, 1)


async def watch(steps, a) -> None:
    """Scrape on a timer while thumbnails and embeddings follow continuously behind it."""

    async def scrape_loop():
        while True:
            await p.scrape(rate=a.rate, max_rate=a.max_rate)
            if "recheck" in steps:
                await p.recheck(last=a.recheck_last or None)
            await asyncio.sleep(a.every)

    async def media_loop():
        while True:
            if "thumbs" in steps:
                await p.thumbs()  # flags blank photos as it downloads them
            if "blank" in steps and not p.BLANK.exists():
                await asyncio.to_thread(p.find_blank)
            if "embed" in steps:
                await asyncio.to_thread(p.embed)
            if "map" in steps:
                await asyncio.to_thread(p.project)
            await asyncio.sleep(120)

    tasks = []
    if "scrape" in steps:
        tasks.append(scrape_loop())
    if {"thumbs", "blank", "embed", "map"} & set(steps):
        tasks.append(media_loop())
    await asyncio.gather(*tasks)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", nargs="?", default="run", choices=["run", "watch", "status", "publish", "restore"])
    ap.add_argument("--steps", default=",".join(STEPS), help="comma-separated subset of: " + ",".join(STEPS))
    ap.add_argument("--every", type=int, default=900, help="watch: seconds between checks for new sandwiches")
    ap.add_argument("--rate", type=float, default=2.5, help="initial API requests/second")
    ap.add_argument("--max-rate", type=float, default=3.5)
    ap.add_argument("--recheck-last", type=int, default=2000, help="recheck: how many of the newest sandwiches to look at again (0 = all)")
    a = ap.parse_args()
    os.chdir(Path(__file__).resolve().parent)

    if a.command == "status":
        print(json.dumps(asyncio.run(p.status()), indent=2))
        return
    if a.command in ("publish", "restore"):
        from stulle import publish

        if a.command == "restore":
            publish.restore()
        else:
            out = publish.publish()
            print(json.dumps({k: v for k, v in out.items() if k != "parts"}, indent=1))
        return
    steps = [s.strip() for s in a.steps.split(",") if s.strip()]
    bad = set(steps) - set(STEPS)
    if bad:
        sys.exit(f"unknown steps: {', '.join(sorted(bad))}")
    acquire_lock()
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))  # `kill` releases the lock too
    try:
        asyncio.run(watch(steps, a) if a.command == "watch" else run_once(steps, a))
    except KeyboardInterrupt:
        p.log("stopped")
    finally:
        release_lock()


if __name__ == "__main__":
    main()
