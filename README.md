# Stullen-Statistik 🥪

[![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/ktaletsk/stullen-statistik/blob/main/stullen.py)
[![Update data](https://github.com/ktaletsk/stullen-statistik/actions/workflows/update-data.yml/badge.svg)](https://github.com/ktaletsk/stullen-statistik/actions/workflows/update-data.yml)

A serious data analysis of [stulle.app](https://stulle.app), a very unserious German
sandwich simulator by [Gregor Weichbrodt](https://ggor.de): what people put on bread, what
goes together, how tall it gets, and what CLIP finds when you search the photos.
Background on the site is in [research.md](research.md).

- **[stullen.py](stullen.py)**: the full notebook. [Open it on molab](https://molab.marimo.io/github/ktaletsk/stullen-statistik/blob/main/stullen.py),
  or run it locally with `uvx marimo edit --sandbox stullen.py`.
- **[blog/stullen_blog.py](blog/stullen_blog.py)**: a browser-only cut of it, published as
  marimo islands in a blog post.
- **[dataset/](dataset/)**: the data, refreshed daily by GitHub Actions.

## The dataset

Everything the notebooks need is in `dataset/`, split into partitions of 2,000 consecutive
IDs. A daily update only rewrites the newest one or two, so the repository grows slowly.

| Path | What |
|------|------|
| `stullen/part-*.parquet` | one row per sandwich: bread, spreads, toppings, remix parent, estimated height, classic-recipe matches |
| `toppings/part-*.parquet` | one row per topping piece: type, position, rotation, stacking order |
| `spreads/part-*.parquet` | one row per pass of the knife |
| `records/part-*.jsonl.gz` | the raw API records (stroke geometry dropped) |
| `clip/part-*.{ids,emb}.npy` | CLIP ViT-L/14 photo embeddings, float16 |
| `map.parquet` | 2D UMAP layout of the embeddings + zero-shot CLIP category |
| `hunts.parquet`, `positions.parquet` | precomputed photo searches and topping-position grids |
| `catalog.json`, `meta.json` | names (German + English), groups, classic recipes; counts and freshness |
| `ids_missing.txt`, `ids_blank.txt` | deleted sandwiches, and banned ones (photo blacked out) |

Deleted and banned sandwiches are left out of everything else. Photos are not copied: they
stay at `img.stulle.app/{id}/thumb.jpg`, so anything removed there disappears here too.

Read it from anywhere with fsspec, as the notebook does on molab:

```python
import json
import fsspec, polars as pl

base = "https://raw.githubusercontent.com/ktaletsk/stullen-statistik/main/dataset/"
fs = fsspec.filesystem("https")  # needs aiohttp
meta = json.loads(fs.cat(base + "meta.json"))
files = fs.cat([f"{base}stullen/{p}.parquet" for p in meta["parts"]])  # fetched concurrently
stullen = pl.concat(pl.read_parquet(b) for b in files.values())
```

(`github://` works too, but it goes through the GitHub API, which allows 60 unauthenticated
requests an hour, and the full dataset is ~130 files.)

## Pipeline

`pipeline.py` keeps its working state in `data/` (git-ignored) and publishes `dataset/`.
Every step is incremental: it only fetches or computes what isn't on disk yet.

| Step     | What it does                                                  |
|----------|---------------------------------------------------------------|
| `scrape` | JSON record for each new sandwich from `stulle.app/api/photos/{id}` |
| `recheck`| Looks at the newest 2,000 again: deleted (404) or blacked out since |
| `thumbs` | 224px thumbnails of photos that aren't embedded yet            |
| `blank`  | Flags blank photos: the black placeholder for moderated sandwiches, or a broken render |
| `embed`  | CLIP ViT-L/14 image embeddings (Mac GPU via MPS, or CPU)        |
| `map`    | 2D UMAP layout of the embeddings                              |

```bash
uv run pipeline.py restore     # rebuild data/ from dataset/ (fresh clone)
```

```bash
uv run pipeline.py             # one pass over everything new
```

```bash
uv run pipeline.py publish     # write dataset/ (unchanged files are left alone)
```

```bash
uv run pipeline.py status      # counts and coverage, no downloads
```

That is exactly what [the daily workflow](.github/workflows/update-data.yml) does, followed
by a commit of `dataset/`. Locally you can also keep it running with
`uv run pipeline.py watch` (new sandwiches every 15 minutes).

Things to know:

- **Polite by default.** The API gets ~2–3 requests per second. The scraper backs off for
  60 seconds on HTTP 429 and speeds up again slowly.
- **Partial scrapes are fair samples.** IDs are fetched one 500-ID block at a time, always
  from the least-covered block, so coverage stays even across the whole ID range.
- **Moderation lands late.** Bans and deletions often come after a sandwich was fetched, so
  `recheck` looks at recent ones again (`--recheck-last 0` rechecks everything).
- **One instance at a time**, and safe to kill: files are written atomically.
- **Self-contained.** `pipeline.py`, `stullen.py` and the blog notebook declare their
  dependencies inline (PEP 723).

## Notebook notes

`stullen.py` reads `dataset/` next to it when it's there, and otherwise straight from
GitHub, so it also runs from a bare URL:

```bash
uvx marimo edit --sandbox https://github.com/ktaletsk/stullen-statistik/blob/main/stullen.py
```

Free-text photo search needs the CLIP text model (PyTorch), so it's available locally and on
molab but not in WebAssembly; the preset searches are precomputed and work everywhere.
molab's static preview shows the outputs committed in `__marimo__/session/`, which the
workflow refreshes weekly.
