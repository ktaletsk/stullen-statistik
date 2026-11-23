# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "altair>=6.3.0",
#     "aiohttp>=3.11.0",
#     "fsspec>=2025.3.0",
#     "ipython>=9.18.1",
#     "marimo>=0.25.2",
#     "numpy>=2.5.5",
#     "pillow>=13.0.0",
#     "polars>=2.1.0",
#     "pyarrow>=26.0.1",
#     "scipy>=1.18.1",
#     "sentence-transformers>=6.3.0",
# ]
#
# [tool.marimo.package_management]
# manager = "uv"
#
# [tool.marimo.experimental]
# rtc_v2 = false
# ///

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="Stullen-Statistik")

with app.setup:
    import asyncio
    import json
    import io
    import math
    import sys
    from collections import Counter
    from itertools import combinations
    from pathlib import Path

    import altair as alt
    import marimo as mo
    import numpy as np
    import polars as pl

    # Where the data comes from: dataset/ next to this notebook when it is there (a clone of
    # the repo, refreshed daily by GitHub Actions); otherwise the same files straight from
    # GitHub: fsspec on a server (molab, `uvx marimo edit <url>`), fetch() in the browser (WASM).
    # fsspec reads raw.githubusercontent.com over HTTPS rather than through `github://`, which
    # goes via the GitHub API and is capped at 60 unauthenticated requests an hour.
    REPO_URL = "https://github.com/ktaletsk/stullen-statistik"
    RAW_URL = "https://raw.githubusercontent.com/ktaletsk/stullen-statistik/main/"
    IN_BROWSER = sys.platform == "emscripten"
    if IN_BROWSER:
        # loaded into Pyodide for polars: parquet goes through pyarrow, time zones need tzdata
        import pyarrow  # noqa: F401
        import tzdata  # noqa: F401


    def _find_local_dataset():
        _bases = []
        try:
            _bases.append(Path(mo.notebook_dir()))
        except Exception:
            pass
        _bases.append(Path.cwd())
        for _b in _bases:
            if (_b / "dataset" / "meta.json").exists():
                return _b / "dataset"
        return None


    LOCAL_DATASET = None if IN_BROWSER else _find_local_dataset()
    DATA_SOURCE = f"[`dataset/`]({REPO_URL}/tree/main/dataset) ({'local copy' if LOCAL_DATASET else 'read from GitHub'})"


    async def read_files(paths):
        """Bytes of dataset files (paths relative to dataset/), from wherever the data lives."""
        if LOCAL_DATASET is not None:
            return [(LOCAL_DATASET / _p).read_bytes() for _p in paths]
        if IN_BROWSER:
            from pyodide.http import pyfetch

            async def _one(_p):
                _r = await pyfetch(RAW_URL + "dataset/" + _p)
                return await _r.bytes()

            return list(await asyncio.gather(*(_one(_p) for _p in paths)))
        import fsspec

        _fs = fsspec.filesystem("https")
        _urls = [f"{RAW_URL}dataset/{_p}" for _p in paths]
        _got = _fs.cat(_urls)  # fetched concurrently
        return [_got[_u] for _u in _urls]

    alt.data_transformers.disable_max_rows()

    # Charts must read equally well in light and dark mode (marimo, molab and the blog can all
    # be either). So no text, axis or grid colour is set anywhere: marimo hands Vega its light
    # or dark theme to match, and Vega supplies the ink. Data colours are mid-tone steps with
    # at least 3:1 contrast against both white and near-black (validated in both modes), and
    # shading uses the opacity of one hue, so "low" fades into whichever background is there.
    # All text is a title, axis or legend: Vega-Lite bakes a fixed colour into text marks.
    BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#3987e5", "#d95926", "#199e70", "#c98500", "#d55181"
    GREEN, VIOLET, RED, GRAY, GRAY_DARK = "#0ca30c", "#9085e9", "#e34948", "#8a8984", "#6b6a65"
    KIND_COLORS = {"spread": BLUE, "savory": ORANGE, "veg": AQUA, "junk": YELLOW, "sweet": MAGENTA}
    SOURCE = "Source: stulle.app public API, uniform random sample of sandwiches, 20–23 Nov 2026"

    @alt.theme.register("stullen_any_mode", enable=True)
    def _stullen_theme():
        # 12 px minimum text, left-aligned title with a subtitle line; colours come from Vega's theme
        return alt.theme.ThemeConfig(
            {
                "config": {
                    "background": "transparent",
                    "axis": {"labelFontSize": 12, "titleFontSize": 12, "titleFontWeight": "normal", "gridOpacity": 0.5},
                    "legend": {"labelFontSize": 12, "titleFontSize": 12, "titleFontWeight": "normal"},
                    "title": {"fontSize": 15, "subtitleFontSize": 12, "anchor": "start", "offset": 12},
                    "view": {"stroke": None},
                }
            }
        )

    def with_values(bars, data, field, labels):
        """Print each bar's value in a column on the right, as a second category axis.

        `labels` maps category -> text, in the bars' order. Axis labels follow the light/dark
        theme; text marks would not.
        """
        values = alt.Chart(data).mark_tick(opacity=0).encode(
            y=alt.Y(field, sort=list(labels), title=None, axis=alt.Axis(
                orient="right", domain=False, ticks=False, labelPadding=6,
                labelExpr=f"{json.dumps(labels, ensure_ascii=False)}[datum.value]",
            ))
        )
        return alt.layer(bars, values).resolve_axis(y="independent")

    def fmt(v, f):
        return format(v, f)

    def ttl(text, *sub):
        """Chart title stating the takeaway, with what/units/when and the source underneath."""
        return alt.Title(text, subtitle=[*sub, SOURCE])


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Stullen-Statistik: better than GTA VI 🥪

    *A serious data analysis of a very unserious sandwich simulator.*

    [stulle.app](https://stulle.app) is a German sandwich simulator. You pick a bread, spread
    something on it, drop toppings on top, and the result gets a serial number and a
    permalink. Every sandwich is also served as JSON at `stulle.app/api/photos/{id}`: the
    bread, each spread pass, and every topping with its position and rotation, in stacking
    order.

    The data comes from `pipeline.py` in [the repo](https://github.com/ktaletsk/stullen-statistik),
    which GitHub Actions runs every day: new sandwiches, recently banned or deleted ones,
    CLIP embeddings of the photos and a 2D map. It fetches IDs block by block so that
    coverage stays even, which means a partial scrape is a uniform random sample of all
    sandwiches. Names are given in German with the English in brackets.
    """)
    return


@app.cell(hide_code=True)
def _():
    _img = Path("assets/stulle-app.jpg")
    mo.image(
        str(_img) if _img.exists() else RAW_URL + "assets/stulle-app.jpg",
        alt="The stulle.app editor: a slice of rye bread and a butter knife on a wooden board, with spreads and toppings to pick from below",
        caption="The editor at stulle.app: spread with the mouse, drop toppings from the tray, press Foto.",
        width="100%",
        rounded=True,
    )
    return


@app.cell(hide_code=True)
def _(catalog):
    # Display names "Deutsch (English)" and ingredient groups, from the dataset's catalog.json
    _names = {**catalog["toppings"], **catalog["spreads"], **catalog["breads"]}
    KIND_OF = {_k: _v["kind"] for _k, _v in {**catalog["toppings"], **catalog["spreads"]}.items()}
    CLASSICS = catalog["classics"]


    def _both(de, en):
        return de if de.casefold() == en.casefold() else f"{de} ({en})"


    def lab(key):
        """Display name for an ingredient / bread / spread key: German (English)."""
        _n = _names.get(key)
        return _both(_n["de"], _n["en"]) if _n else key


    def fam(name):
        """Display name for a classic Stulle."""
        _c = CLASSICS.get(name)
        return _both(_c["de"], _c["en"]) if _c else name


    KIND_LABELS = {_k: _both(_v["de"], _v["en"]) for _k, _v in catalog["kinds"].items()}
    kind_scale = alt.Scale(domain=list(KIND_LABELS.values()), range=[KIND_COLORS[k] for k in KIND_LABELS])
    return CLASSICS, KIND_LABELS, KIND_OF, fam, kind_scale, lab


@app.cell
def _():
    reload = mo.ui.button(label="Reload data", kind="neutral")
    reload
    return (reload,)


@app.cell
async def _(reload):
    reload
    meta, catalog = [json.loads(_b) for _b in await read_files(["meta.json", "catalog.json"])]
    _tables = ("stullen", "toppings", "spreads")
    _paths = [f"{_t}/{_p}.parquet" for _t in _tables for _p in meta["parts"]]
    _files = await read_files(_paths + ["ids_missing.txt", "ids_blank.txt"])
    _n = len(meta["parts"])
    stullen, toppings, spreads = (
        pl.concat([pl.read_parquet(io.BytesIO(_b)) for _b in _files[_i * _n : (_i + 1) * _n]]) for _i in range(len(_tables))
    )
    # The dataset already leaves out deleted sandwiches (404) and banned ones (photo blacked
    # out by moderation, or a broken render); they still count as "checked" for coverage.
    _missing_ids, _blank_ids = ({int(_l) for _l in _b.decode().split()} for _b in _files[-2:])
    n_missing = len(_missing_ids)
    n_blank = len(_blank_ids - _missing_ids)
    _all_ids = set(stullen["id"].to_list()) | _blank_ids

    # Keep the sample uniform across the ID range. The pipeline fetches IDs block by block
    # (500 IDs each) so that coverage stays even, but older runs and the newest IDs can
    # still be uneven. Thin every block down to the coverage of the least-covered "normal"
    # block, using a deterministic hash so the same subset survives reloads. Blocks that
    # are still catching up (under half the median coverage) are kept whole and reported.
    _B = 500
    _seen = np.array(sorted(_all_ids | _missing_ids))
    _max_id = int(_seen.max())
    _nb = (_max_id - 1) // _B + 1
    _size = np.minimum(_B, _max_id - np.arange(_nb) * _B)
    _cov = np.bincount((_seen - 1) // _B, minlength=_nb) / _size
    _normal = _cov >= 0.5 * np.median(_cov)
    sample_rate = float(_cov[_normal].min())
    _p_keep = np.where(_normal, np.minimum(1.0, sample_rate / np.maximum(_cov, 1e-9)), 1.0)
    _ids = stullen["id"].to_numpy()
    _u = ((_ids.astype(np.uint64) * np.uint64(2654435761)) % np.uint64(2**32)) / 2**32
    stullen = stullen.filter(pl.Series(_u < _p_keep[(_ids - 1) // _B]))
    toppings = toppings.filter(pl.col("id").is_in(stullen["id"].implode()))
    spreads = spreads.filter(pl.col("id").is_in(stullen["id"].implode()))
    _lagging = int(_size[~_normal].sum())
    sampling_note = (
        f"{_lagging:,} IDs in the newest blocks are still being fetched, so they are under-represented for now."
        if _lagging
        else ""
    )
    return catalog, meta, n_blank, n_missing, sample_rate, sampling_note, spreads, stullen, toppings


@app.cell
def _(stullen):
    # Ingredient sets per sandwich, and the strict "famous sandwich" label(s)
    df = stullen.with_columns(
        pl.concat_list("spread_types", "topping_types").alias("ingredients"),
        pl.col("classics").alias("famous"),
        pl.col("created_at").dt.convert_time_zone("Europe/Berlin").alias("created_berlin"),
    )
    return (df,)


@app.cell(hide_code=True)
def _(df, meta, n_blank, n_missing, sample_rate, sampling_note):
    _max_id = int(df["id"].max())
    _seen = round(sample_rate * _max_id)
    _remixed = df["parent_id"].is_not_null().mean()

    def _tile(label, value, note=""):
        return mo.stat(value=value, label=label, caption=note, bordered=True)

    mo.vstack(
        [
            mo.md(f"**Data:** {DATA_SOURCE}, updated {meta['updated_at'].replace('T', ' ').rstrip('Z')} UTC. **Coverage:** {sample_rate:.0%} of ~{_max_id:,} IDs checked so far; {n_missing:,} were 404 (deleted or reported) and {n_blank:,} more show a blank photo (blacked out by moderation, or a broken render), so they are left out. All percentages are estimates from this uniform random sample." + (" " + sampling_note if sampling_note else "")),
            mo.hstack(
                [
                    _tile("Sandwiches", f"{df.height:,}", "in the analysed sample"),
                    _tile("Median toppings", f"{df['n_toppings'].median():.0f}", "pieces per sandwich"),
                    _tile("Median height", f"{df['height_cm'].median():.1f} cm", "estimated topping pile"),
                    _tile("Bare bread", f"{(df['n_toppings'] == 0).mean():.0%}", "no toppings at all"),
                    _tile("Remixes", f"{_remixed:.0%}", "built on someone else's Stulle"),
                ],
                widths="equal",
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 1 · Butter Late Than Never
    ### What goes on a Stulle?

    Share of sandwiches that contain each ingredient at least once. Spreads and toppings are
    on the same scale. Switch the metric to see how much of an ingredient people pile on.
    """)
    return


@app.cell
def _():
    pop_metric = mo.ui.radio(
        options={
            "% of sandwiches using it": "share",
            "Total pieces placed": "pieces",
            "Median pieces, when used": "median_pieces",
        },
        value="% of sandwiches using it",
        inline=True,
    )
    pop_metric
    return (pop_metric,)


@app.cell
def _(KIND_LABELS, KIND_OF, df, kind_scale, lab, pop_metric, spreads, toppings):
    _n = df.height
    _top = (
        toppings.group_by("id", "type").len("pieces")
        .group_by("type")
        .agg(pl.len().alias("sandwiches"), pl.col("pieces").sum(), pl.col("pieces").median().alias("median_pieces"))
        .rename({"type": "key"})
    )
    _spr = (
        spreads.group_by("id", "spread").len("pieces")
        .group_by("spread")
        .agg(pl.len().alias("sandwiches"), pl.col("pieces").sum(), pl.col("pieces").median().alias("median_pieces"))
        .rename({"spread": "key"})
    )
    popularity = (
        pl.concat([_top.with_columns(pl.lit("topping").alias("what")), _spr.with_columns(pl.lit("spread").alias("what"))])
        .with_columns(
            (pl.col("sandwiches") / _n).alias("share"),
            pl.col("key").map_elements(lab, return_dtype=pl.Utf8).alias("ingredient"),
            pl.col("key")
            .map_elements(lambda k: KIND_LABELS[KIND_OF[k]], return_dtype=pl.Utf8)
            .alias("group"),
        )
        .filter(pl.col("sandwiches") >= 10)  # hides toppings seen only in a test sandwich or two
        .sort("share", descending=True)
    )
    _m = pop_metric.value
    _title = {"share": "% of sandwiches", "pieces": "pieces placed (all sandwiches)", "median_pieces": "median pieces per sandwich that uses it"}[_m]
    _fmt = ".0%" if _m == "share" else ",.0f"
    _base = alt.Chart(popularity).encode(
        y=alt.Y("ingredient:N", sort=popularity.sort(_m, descending=True)["ingredient"].to_list(), title=None),
        x=alt.X(f"{_m}:Q", title=None, axis=None),  # value labels replace the axis
        tooltip=[
            alt.Tooltip("ingredient:N", title="Ingredient"),
            alt.Tooltip("share:Q", format=".1%", title="% of sandwiches"),
            alt.Tooltip("sandwiches:Q", format=",.0f", title="Sandwiches"),
            alt.Tooltip("pieces:Q", format=",.0f", title="Pieces placed"),
            alt.Tooltip("median_pieces:Q", title="Median pieces when used"),
        ],
    )
    _bars = _base.mark_bar(cornerRadiusEnd=4, height=14).encode(color=alt.Color("group:N", scale=kind_scale, title=None, legend=alt.Legend(orient="top", columns=3, labelLimit=400)))
    _lead = popularity.sort(_m, descending=True).row(0, named=True)
    _ttl = {
        "share": f"{_lead['ingredient']} is on {_lead['share']:.0%} of all Stullen",
        "pieces": f"{_lead['ingredient']} is the most-placed piece: {_lead['pieces']:,} in the sample",
        "median_pieces": f"Whoever uses {_lead['ingredient']} uses a lot: median {_lead['median_pieces']:.0f} pieces",
    }[_m]
    _sorted = popularity.sort(_m, descending=True)
    _pyfmt = ".0%" if _m == "share" else ",.0f"
    with_values(_bars, popularity, "ingredient:N", {r["ingredient"]: fmt(r[_m], _pyfmt) for r in _sorted.iter_rows(named=True)}).properties(
        width=560, height=alt.Step(18), title=ttl(_ttl, f"{_title.capitalize()}, by ingredient; colour = group. n = {_n:,} sandwiches.")
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 2 · The Spread Sheet
    ### Bread and spread
    """)
    return


@app.cell
def _(df, lab):
    _bread = (
        df.group_by("bread").len("n")
        .with_columns((pl.col("n") / df.height).alias("share"), pl.col("bread").map_elements(lab, return_dtype=pl.Utf8).alias("Bread"))
    )
    _b0 = _bread.sort("n", descending=True).row(0, named=True)
    _bread_base = alt.Chart(_bread).encode(
        y=alt.Y("Bread:N", sort=_bread.sort("n", descending=True)["Bread"].to_list(), title=None),
        x=alt.X("share:Q", axis=None, title=None),
        tooltip=["Bread", alt.Tooltip("n:Q", format=",.0f"), alt.Tooltip("share:Q", format=".1%")],
    )
    _bread_chart = with_values(
        _bread_base.mark_bar(cornerRadiusEnd=4, color=BLUE, height=16),
        _bread, "Bread:N", {r["Bread"]: f"{r['share']:.0%}" for r in _bread.sort("n", descending=True).iter_rows(named=True)},
    ).properties(width=220, height=alt.Step(22), title=ttl(f"{_b0['Bread']}: {_b0['share']:.0%}", "% of sandwiches by bread"))
    _combo = (
        df.with_columns(
            pl.col("spread_types")
            .list.eval(pl.element().map_elements(lab, return_dtype=pl.Utf8))
            .list.join(" + ")
            .replace("", "(nothing)")
            .alias("Spreads")
        )
        .group_by("Spreads").len("n")
        .with_columns((pl.col("n") / df.height).alias("share"))
        .sort("n", descending=True)
        .head(12)
    )
    _c0 = _combo.row(0, named=True)
    _combo_base = alt.Chart(_combo).encode(
        y=alt.Y("Spreads:N", sort=_combo["Spreads"].to_list(), title=None, axis=alt.Axis(labelLimit=320)),
        x=alt.X("share:Q", axis=None, title=None),
        tooltip=["Spreads", alt.Tooltip("n:Q", format=",.0f"), alt.Tooltip("share:Q", format=".1%")],
    )
    _combo_chart = with_values(
        _combo_base.mark_bar(cornerRadiusEnd=4, color=BLUE, height=14),
        _combo, "Spreads:N", {r["Spreads"]: f"{r['share']:.0%}" for r in _combo.iter_rows(named=True)},
    ).properties(width=260, height=alt.Step(18), title=ttl(f"{_c0['Spreads']} alone: {_c0['share']:.0%}", "% of sandwiches, 12 most common spread combinations"))
    mo.hstack([_bread_chart, _combo_chart], justify="start", gap=2)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 3 · Jam Sessions & Wurst-Case Scenarios
    ### What goes with what

    Each cell compares two ingredients across all sandwiches.

    * **Count**: number of sandwiches that contain both.
    * **Jaccard**: |A∩B| / |A∪B|, so rare ingredients are on a fair footing.
    * **Lift**: P(A∩B) / (P(A)·P(B)). Red means they appear together more often than chance
      (a pairing), blue means less often (they avoid each other). Shown on a log₂ scale.
      Raw lift is red almost everywhere, because a long tail of "kitchen-sink" Stullen
      contains nearly every ingredient.
    * **Lift (size-adjusted)**, the default: the same ratio, but chance is computed among
      sandwiches with the same number of distinct ingredients. This answers "given that
      someone uses 3 ingredients, are these two picked together more than chance?"

    **Order.** *By group* (the default) puts spreads first, then savory, vegetables, junk
    and sweet, with the colour strip marking each block. Red squares *inside* a block mean
    a group sticks together; blue squares *between* blocks mean two groups avoid each
    other. *By similarity* uses hierarchical clustering on Jaccard similarity instead, so
    the data decides what sits next to what.
    """)
    return


@app.cell
def _():
    cooc_metric = mo.ui.dropdown(options=["Lift (size-adjusted)", "Lift", "Jaccard", "Count"], value="Lift (size-adjusted)", label="Metric")
    cooc_min = mo.ui.slider(start=0, stop=200, step=10, value=30, label="Hide ingredients used in fewer than N sandwiches", show_value=True)
    cooc_sort = mo.ui.dropdown(options=["By group", "By similarity (clustering)"], value="By group", label="Order")
    mo.hstack([cooc_metric, cooc_sort, cooc_min], justify="start", gap=2)
    return cooc_metric, cooc_min, cooc_sort


@app.cell
def _(KIND_LABELS, KIND_OF, cooc_min, cooc_sort, df, lab):
    from scipy.cluster.hierarchy import leaves_list, linkage
    from scipy.spatial.distance import squareform

    _sets = df["ingredients"].to_list()
    _counts = {}
    for _s in _sets:
        for _k in set(_s):
            _counts[_k] = _counts.get(_k, 0) + 1
    cooc_keys = sorted(k for k, c in _counts.items() if c >= cooc_min.value)
    _idx = {k: i for i, k in enumerate(cooc_keys)}
    _X = np.zeros((len(_sets), len(cooc_keys)), dtype=np.float32)
    for _r, _s in enumerate(_sets):
        for _k in set(_s):
            if _k in _idx:
                _X[_r, _idx[_k]] = 1
    co_count = _X.T @ _X  # diagonal = sandwiches containing ingredient
    _occ = np.diag(co_count)
    _union = _occ[:, None] + _occ[None, :] - co_count
    co_jaccard = np.divide(co_count, _union, out=np.zeros_like(co_count), where=_union > 0)
    _p = _occ / len(_sets)
    co_lift = np.divide(co_count / len(_sets), np.outer(_p, _p), out=np.ones_like(co_count), where=np.outer(_p, _p) > 0)
    # size-adjusted lift: expected co-count assuming independence *within* sandwiches that
    # have the same number of distinct ingredients (capped at 12), so kitchen-sink
    # sandwiches don't make every pair look like a pairing.
    _size = np.minimum(np.array([len(set(_s)) for _s in _sets]), 12)
    _expected = np.zeros_like(co_count)
    for _k in np.unique(_size):
        _Xk = _X[_size == _k]
        _ck = _Xk.sum(axis=0)
        _expected += np.outer(_ck, _ck) / len(_Xk)
    co_lift_adj = np.divide(co_count, _expected, out=np.ones_like(co_count), where=_expected > 0)
    # Group of each ingredient: spreads, then savory, vegetables, junk, sweet.
    _kind = {k: KIND_OF[k] for k in cooc_keys}
    _kind_rank = {k: i for i, k in enumerate(["spread", "savory", "veg", "junk", "sweet"])}
    cooc_group_of = {lab(k): KIND_LABELS[_kind[k]] for k in cooc_keys}
    # Order: by group (most-used first inside each group), or cluster order on Jaccard distance
    if cooc_sort.value == "By group":
        _order = sorted(range(len(cooc_keys)), key=lambda i: (_kind_rank[_kind[cooc_keys[i]]], -_occ[i]))
    elif len(cooc_keys) > 2:
        _d = 1 - co_jaccard.copy()
        np.fill_diagonal(_d, 0)
        _order = leaves_list(linkage(squareform(_d, checks=False), method="average"))
    else:
        _order = np.arange(len(cooc_keys))
    cooc_order = [lab(cooc_keys[i]) for i in _order]
    return (
        co_count,
        co_jaccard,
        co_lift,
        co_lift_adj,
        cooc_group_of,
        cooc_keys,
        cooc_order,
    )


@app.cell
def _(
    co_count,
    co_jaccard,
    co_lift,
    co_lift_adj,
    cooc_group_of,
    cooc_keys,
    cooc_metric,
    cooc_order,
    kind_scale,
    lab,
):
    _rows = []
    for _i, _a in enumerate(cooc_keys):
        for _j, _b in enumerate(cooc_keys):
            if _i == _j:
                continue
            _rows.append(
                dict(
                    a=lab(_a), b=lab(_b),
                    Count=int(co_count[_i, _j]),
                    Jaccard=float(co_jaccard[_i, _j]),
                    Lift=float(co_lift[_i, _j]),
                    AdjLift=float(co_lift_adj[_i, _j]),
                    log2lift=float(np.log2(max(co_lift[_i, _j], 1 / 16))),
                    log2adj=float(np.log2(max(co_lift_adj[_i, _j], 1 / 16))),
                )
            )
    cooc_long = pl.DataFrame(_rows)
    _m = cooc_metric.value
    _col = {"Lift (size-adjusted)": "AdjLift", "Lift": "Lift", "Jaccard": "Jaccard"}.get(_m, "Count")
    _top = cooc_long.filter(pl.col("a") < pl.col("b"), pl.col("Count") >= 30).sort(_col, descending=True).row(0, named=True)
    _val = f"{_top[_col]:.1f}× chance" if "Lift" in _m else (f"{_top[_col]:.2f}" if _m == "Jaccard" else f"{_top[_col]:,} sandwiches")
    _title = ttl(
        f"Strongest pair: {_top['a']} + {_top['b']} ({_val})",
        *(["Red = together more often than chance, blue = less often; the stronger, the more solid"] if "Lift" in _m else ["The more, the more solid"]),
        f"{_m} for every pair of ingredients (headline: pairs on 30+ sandwiches); strip on the right = group",
    )
    # Legends sit under the matrix, which keeps the group strip flush with it. Shading is the
    # opacity of one hue (two for lift: direction), so weak cells fade into any background.
    _leg = dict(orient="bottom", direction="horizontal")
    if "Lift" in _m:
        _l2 = "log2adj" if _m == "Lift (size-adjusted)" else "log2lift"
        cooc_long = cooc_long.with_columns(
            pl.when(pl.col(_l2) >= 0).then(pl.lit("together")).otherwise(pl.lit("apart")).alias("direction"),
            pl.col(_l2).abs().clip(upper_bound=1.0).alias("strength"),
        )
        _color = alt.Color("direction:N", title=None, scale=alt.Scale(domain=["together", "apart"], range=[RED, BLUE]), legend=alt.Legend(**_leg))
        _opacity = alt.Opacity("strength:Q", title="× or ÷ chance", scale=alt.Scale(domain=[0, 1], range=[0.05, 1]),
                               legend=alt.Legend(**_leg, values=[0.25, 0.5, 1], labelExpr="format(pow(2, datum.value), '.1f') + '×'", symbolFillColor=GRAY, symbolType="square"))
    else:
        _color = alt.value(BLUE)
        _q = "Jaccard" if _m == "Jaccard" else "Count"
        _opacity = alt.Opacity(f"{_q}:Q", scale=alt.Scale(type="sqrt", range=[0.05, 1]), legend=alt.Legend(**_leg, symbolFillColor=BLUE, symbolType="square"))
    _heat = (
        alt.Chart(cooc_long)
        .mark_rect()
        .encode(
            x=alt.X("a:N", sort=cooc_order, title=None, axis=alt.Axis(labelAngle=-50), scale=alt.Scale(paddingInner=0.08)),
            y=alt.Y("b:N", sort=cooc_order, title=None, scale=alt.Scale(paddingInner=0.08)),
            color=_color,
            opacity=_opacity,
            tooltip=[
                alt.Tooltip("a:N", title="Ingredient"),
                alt.Tooltip("b:N", title="with"),
                alt.Tooltip("Count:Q", format=",.0f"),
                alt.Tooltip("Jaccard:Q", format=".2f"),
                alt.Tooltip("Lift:Q", format=".2f"),
                alt.Tooltip("AdjLift:Q", format=".2f", title="Size-adjusted lift"),
            ],
        )
        .properties(width=alt.Step(20), height=alt.Step(20))
    )
    # a thin colour strip on the right shows which group each row belongs to
    _strip = (
        alt.Chart(pl.DataFrame({"b": list(cooc_group_of), "group": list(cooc_group_of.values())}))
        .mark_rect()
        .encode(
            y=alt.Y("b:N", sort=cooc_order, axis=None, scale=alt.Scale(paddingInner=0.08)),
            color=alt.Color("group:N", scale=kind_scale, legend=None),
            tooltip=[alt.Tooltip("b:N", title="Ingredient"), alt.Tooltip("group:N", title="Group")],
        )
        .properties(width=10, height=alt.Step(20))
    )
    # key for the strip, drawn under the matrix so long group names never clip on the right
    _key = (
        alt.Chart(pl.DataFrame({"group": list(kind_scale.domain)}))
        .mark_square(size=140, opacity=1)
        .encode(
            y=alt.Y("group:N", sort=list(kind_scale.domain), title=None, axis=alt.Axis(orient="right", domain=False, ticks=False, labelLimit=500)),
            color=alt.Color("group:N", scale=kind_scale, legend=None),
        )
        .properties(width=14, height=alt.Step(18), title=alt.Title("Group (strip colour)", fontSize=12, fontWeight="normal"))
    )
    alt.vconcat(alt.hconcat(_heat, _strip, spacing=3).resolve_scale(color="independent"), _key, spacing=16, title=_title).resolve_scale(color="independent")
    return (cooc_long,)


@app.cell
def _(cooc_long):
    _common = cooc_long.filter(pl.col("a") < pl.col("b"), pl.col("Count") >= 30)
    mo.hstack(
        [
            mo.vstack([mo.md("**Strongest pairings** (highest size-adjusted lift, ≥30 sandwiches)"), mo.ui.table(_common.sort("AdjLift", descending=True).head(12).select("a", "b", "Count", pl.col("AdjLift").round(2).alias("Adj. lift")), selection=None, show_column_summaries=False)]),
            mo.vstack([mo.md("**Strongest avoidance** (lowest size-adjusted lift, ≥30 sandwiches)"), mo.ui.table(_common.sort("AdjLift").head(12).select("a", "b", "Count", pl.col("AdjLift").round(2).alias("Adj. lift")), selection=None, show_column_summaries=False)]),
        ],
        widths="equal",
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 4 · Déjà Stulle
    ### The most common exact recipes

    The same set of ingredients regardless of piece count or position, with the bread
    ignored.
    """)
    return


@app.cell
def _(df, lab):
    _recipes = (
        df.with_columns(
            pl.col("ingredients")
            .list.eval(pl.element().map_elements(lab, return_dtype=pl.Utf8))
            .list.sort()
            .list.join(" · ")
            .replace("", "(bare bread)")
            .alias("Recipe"),
            pl.col("ingredients").list.len().alias("ingredients_n"),
        )
        .group_by("Recipe")
        .agg(pl.len().alias("Sandwiches"), pl.col("id").min().alias("first_id"), pl.col("ingredients_n").first().alias("Ingredients"))
        .with_columns((pl.col("Sandwiches") / df.height * 100).round(2).alias("% of all"))
        .sort("Sandwiches", descending=True)
    )
    _n_unique = _recipes.height
    mo.vstack(
        [
            mo.md(f"{_n_unique:,} distinct ingredient sets among {df.height:,} sandwiches. **{(_recipes['Sandwiches'] == 1).sum():,}** of them appear exactly once."),
            mo.ui.table(
                _recipes.head(30)
                .with_columns(pl.format("https://img.stulle.app/{}/thumb.jpg", "first_id").alias("Example"))
                .select("Example", "Recipe", "Sandwiches", "% of all", "Ingredients", pl.col("first_id").alias("ID")),
                selection=None,
                page_size=15,
                show_column_summaries=False,
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(CLASSICS):
    _rules = "\n".join(
        f"| {name} | {', '.join(sorted(r['need'])) or '–'} | {', '.join(sorted(r.get('spreads') or r.get('only_spreads') or [])) or 'any'} | {', '.join(sorted(r['allow'])) or '–'} |"
        for name, r in CLASSICS.items()
    )
    mo.vstack(
        [
            mo.md(r"""
    ## 5 · Let's Go Ham
    ### How common are the classics?

    A sandwich counts as a classic only on a **strict** match: the defining toppings, plus at
    most the usual garnish, and nothing else. A Käsebrot with gummy bears on top does not
    count. The rules are in `stulle/model.py` (`FAMOUS`) and in the dataset's `catalog.json`.
    """),
            mo.accordion({"Matching rules": mo.md("| Classic | Must have | Spread | Allowed garnish |\n|---|---|---|---|\n" + _rules)}),
        ]
    )
    return


@app.cell
def _(CLASSICS, df, fam):
    _ex = df.explode("famous").filter(pl.col("famous").is_not_null())
    _fam = (
        _ex.group_by("famous")
        .agg(pl.len().alias("n"), pl.col("id").sample(n=1, seed=1).first().alias("example"))
        .with_columns((pl.col("n") / df.height).alias("share"))
    )
    _fam = pl.DataFrame({"famous": list(CLASSICS)}).join(_fam, on="famous", how="left").fill_null(0)
    _fam = _fam.with_columns(pl.col("famous").map_elements(fam, return_dtype=pl.Utf8).alias("Classic"))
    _any = df["famous"].list.len().gt(0).mean()
    _f0 = _fam.sort("n", descending=True).row(0, named=True)
    _chart = (
        alt.Chart(_fam)
        .mark_bar(cornerRadiusEnd=4, color=BLUE, height=14)
        .encode(
            y=alt.Y("Classic:N", sort="-x", title=None, axis=alt.Axis(labelLimit=320)),
            x=alt.X("share:Q", axis=alt.Axis(format=".1%"), title="% of all sandwiches (strict match)"),
            tooltip=[alt.Tooltip("Classic:N", title="Classic"), alt.Tooltip("n:Q", format=",.0f", title="Sandwiches"), alt.Tooltip("share:Q", format=".2%"), alt.Tooltip("example:Q", title="Example ID")],
        )
        .properties(
            width=480,
            height=alt.Step(18),
            title=ttl(f"{_f0['Classic']} is the top classic, yet only {_f0['share']:.1%} of all Stullen", f"Strict recipe match; {_any:.0%} of sandwiches match any classic. n = {df.height:,}."),
        )
    )
    _gallery = (
        _fam.filter(pl.col("n") > 0)
        .sort("n", descending=True)
        .select(
            pl.format("https://img.stulle.app/{}/thumb.jpg", "example").alias("Example"),
            pl.col("Classic"),
            pl.col("n").alias("Sandwiches"),
            (pl.col("share") * 100).round(2).alias("% of all"),
            pl.col("example").alias("ID"),
        )
    )
    mo.vstack(
        [
            mo.md(f"Only **{_any:.1%}** of sandwiches are a recognisable classic. The rest are freestyle."),
            _chart,
            mo.md("**One random example of each classic**"),
            mo.ui.table(_gallery, selection=None, page_size=10, show_column_summaries=False),
        ]
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 6 · Stairway to Leberkäse
    ### How tall do they get?

    The API records where each piece was dropped and in what order, but not where it came
    to rest. Height is therefore estimated from two simple models that bracket the truth:

    * **Rigid stack** (upper bound): each piece lands on the highest point under its
      footprint, as if nothing slides or nests.
    * **Packed** (lower bound): total topping volume divided by covered area, i.e. the
      whole pile squashed flat with no air.

    **Height** is the geometric mean of the two. Against the photos of three towers
    ([#7646](https://stulle.app/7646) fries, [#8939](https://stulle.app/8939) chips,
    [#36045](https://stulle.app/36045)) it lands within about 20% of what you can see.
    Topping sizes and thicknesses come from the app's own physics table. The bread itself
    is not included.
    """)
    return


@app.cell
def _():
    height_measure = mo.ui.radio(
        options={"Height estimate (cm)": "height_cm", "Rigid stack (cm)": "rigid_height_cm", "Packed (cm)": "packed_height_cm", "Number of pieces": "n_toppings"},
        value="Height estimate (cm)",
        inline=True,
    )
    height_measure
    return (height_measure,)


@app.cell
def _(df, height_measure):
    _col = height_measure.value
    _v = df.filter(pl.col("n_toppings") > 0)[_col].to_numpy()
    _edges = np.unique(np.round(np.geomspace(max(_v.min(), 0.05), _v.max() * 1.01, 40), 3))
    _h, _ = np.histogram(_v, bins=_edges)
    _hist = pl.DataFrame({"lo": _edges[:-1], "hi": _edges[1:], "n": _h})
    _q = np.quantile(_v, [0.5, 0.9, 0.99])
    _unit = "" if _col == "n_toppings" else " cm"
    _chart = (
        alt.Chart(_hist)
        .mark_bar(color=BLUE, binSpacing=1)
        .encode(
            x=alt.X("lo:Q", bin="binned", scale=alt.Scale(type="log"), title={"height_cm": "Height estimate (cm)", "rigid_height_cm": "Rigid stack (cm)", "packed_height_cm": "Packed (cm)", "n_toppings": "Number of pieces"}[_col] + " (log scale)"),
            x2="hi:Q",
            y=alt.Y("n:Q", title="Sandwiches", axis=alt.Axis(format=",.0f")),
            tooltip=[alt.Tooltip("lo:Q", format=".2f", title="from"), alt.Tooltip("hi:Q", format=".2f", title="to"), alt.Tooltip("n:Q", format=",.0f", title="Sandwiches")],
        )
        .properties(
            width=620,
            height=260,
            title=ttl(
                f"Half stay under {_q[0]:.1f}{_unit}; the top 1% pass {_q[2]:.0f}{_unit}",
                f"Sandwiches per bin, log-spaced bins; sandwiches with at least one topping (n = {len(_v):,})",
            ),
        )
    )
    mo.vstack([_chart, mo.md(f"Median **{_q[0]:.1f}{_unit}** · 90th pct **{_q[1]:.1f}{_unit}** · 99th pct **{_q[2]:.1f}{_unit}** · max **{_v.max():.1f}{_unit}**")])
    return


@app.cell
def _(df, lab):
    _tall = df.sort("height_cm", descending=True).head(8)
    _cards = [
        mo.vstack(
            [
                mo.image(f"https://img.stulle.app/{r['id']}/og.jpg", width=260, rounded=True),
                mo.md(f"[**Nr. {r['id']:,}**](https://stulle.app/{r['id']}) · ~{r['height_cm']:.0f} cm · {r['n_toppings']} pieces"),
                mo.Html(
                    f'<div title="{", ".join(lab(t) for t in r["topping_types"])}" style="height:3.9em;overflow:hidden;'
                    f'display:-webkit-box;-webkit-line-clamp:3;-webkit-box-orient:vertical;font-size:12px;opacity:.72">'
                    f'{", ".join(lab(t) for t in r["topping_types"])}</div>'
                ),
            ],
            gap=0.25,
        )
        for r in _tall.iter_rows(named=True)
    ]
    mo.vstack([mo.md("### The tallest Stullen"), mo.hstack(_cards[:4], widths="equal"), mo.hstack(_cards[4:], widths="equal")])
    return


@app.cell
def _(df, lab, toppings):
    # Which toppings make tall sandwiches: median height of sandwiches containing each topping
    _h = (
        toppings.select("id", "type").unique()
        .join(df.select("id", "height_cm", "n_toppings"), on="id")
        .group_by("type")
        .agg(pl.col("height_cm").median().alias("median_height"), pl.col("height_cm").quantile(0.9).alias("p90_height"), pl.len().alias("n"))
        .with_columns(pl.col("type").map_elements(lab, return_dtype=pl.Utf8).alias("Topping"))
        .filter(pl.col("n") >= 20)
    )
    _base = alt.Chart(_h).encode(y=alt.Y("Topping:N", sort=_h.sort("median_height", descending=True)["Topping"].to_list(), title=None))
    _chart = (
        _base.mark_rule(color=GRAY, strokeWidth=2).encode(x=alt.X("median_height:Q", title="Height of sandwiches containing it (cm)"), x2="p90_height:Q")
        + _base.mark_circle(size=80, color=BLUE, opacity=1).encode(
            x="median_height:Q",
            tooltip=["Topping", alt.Tooltip("median_height:Q", format=".1f", title="Median cm"), alt.Tooltip("p90_height:Q", format=".1f", title="90th pct cm"), alt.Tooltip("n:Q", format=",.0f", title="Sandwiches")],
        )
    ).properties(
        width=520,
        height=alt.Step(18),
        title=ttl(
            f"{_h.sort('median_height', descending=True)['Topping'][0]} comes with the tallest Stullen",
            "Estimated height of sandwiches containing each topping, cm: dot = median, line runs to the 90th percentile. Toppings on 20+ sandwiches.",
        ),
    )
    _chart
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 7 · The Lay of the Ham
    ### Where on the bread do things land?

    Where people drop each topping, in cm from the centre of the 11 cm slice.
    """)
    return


@app.cell
def _(lab, toppings):
    _types = toppings["type"].value_counts().sort("count", descending=True)["type"].to_list()
    where_topping = mo.ui.dropdown(options={lab(t): t for t in _types}, value=lab(_types[0]) if _types else None, label="Topping")
    where_topping
    return (where_topping,)


@app.cell
def _(toppings, where_topping):
    _t = toppings.filter(pl.col("type") == where_topping.value)
    # normalised coordinates (bread = -0.5..0.5) to cm from the centre of the 11 cm slice
    _H, _xe, _ye = np.histogram2d(11 * _t["x"].to_numpy(), 11 * _t["y"].to_numpy(), bins=40, range=[[-6.6, 6.6], [-6.6, 6.6]])
    _xc, _yc = np.meshgrid((_xe[:-1] + _xe[1:]) / 2, (_ye[:-1] + _ye[1:]) / 2, indexing="ij")
    _w = (_xe[1] - _xe[0]) / 2  # cells tile exactly: overlapping translucent cells would show a grid
    _grid = pl.DataFrame({"x": _xc.ravel().round(3), "y": _yc.ravel().round(3), "pieces": _H.ravel()}).filter(pl.col("pieces") > 0).with_columns(
        (pl.col("x") - _w).alias("x0"), (pl.col("x") + _w).alias("x1"), (pl.col("y") - _w).alias("y0"), (pl.col("y") + _w).alias("y1")
    )
    _bread = alt.Chart(pl.DataFrame({"x": [-5.5], "y": [-5.5], "x2": [5.5], "y2": [5.5]})).mark_rect(fill=None, stroke=GRAY, strokeDash=[4, 3]).encode(x="x:Q", y="y:Q", x2="x2:Q", y2="y2:Q")
    _heat = (
        alt.Chart(_grid)
        .mark_rect()
        .encode(
            x=alt.X("x0:Q", scale=alt.Scale(domain=[-6.6, 6.6]), title="cm from centre", axis=alt.Axis(grid=False)),
            x2="x1:Q",
            y=alt.Y("y0:Q", scale=alt.Scale(domain=[-6.6, 6.6]), title="cm from centre", axis=alt.Axis(grid=False)),
            y2="y1:Q",
            color=alt.value(BLUE),
            opacity=alt.Opacity("pieces:Q", scale=alt.Scale(type="sqrt", range=[0.08, 1]), title="Pieces", legend=alt.Legend(symbolFillColor=BLUE, symbolType="square")),
            tooltip=[alt.Tooltip("x:Q", format=".1f", title="x (cm)"), alt.Tooltip("y:Q", format=".1f", title="y (cm)"), alt.Tooltip("pieces:Q", format=",.0f")],
        )
    )
    _centre = ((_t["x"] ** 2 + _t["y"] ** 2) < 0.25**2).mean() if _t.height else 0
    (_heat + _bread).properties(
        width=420,
        height=420,
        title=ttl(
            f"{where_topping.selected_key}: {_centre:.0%} of pieces land within 2.75 cm of the centre",
            f"Pieces dropped per cell, {_t.height:,} pieces; dashed square = the bread (11 cm wide). A uniform spread would put 20% there.",
        ),
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 8 · Viral Loaf
    ### When were they made, and from where?
    """)
    return


@app.cell(hide_code=True)
def _(df, hourly_chart):
    # IDs are serial, so a sandwich's ID is the running total at the moment it was made:
    # plotting ID against creation time gives an exact cumulative curve, no scaling needed.
    _start = pl.datetime(2026, 11, 20, 12, time_zone="Europe/Berlin")
    # One point per 5 minutes (the highest ID so far) keeps the chart small: plotting every
    # sandwich made the cell output ~16 MB, over marimo's 8 MB limit, with no visible gain.
    _cum = (
        df.filter(pl.col("created_berlin") >= _start)
        .sort("created_berlin")
        .group_by_dynamic("created_berlin", every="5m", label="right")
        .agg(pl.col("id").max().alias("total"))
        .select(pl.col("created_berlin").dt.replace_time_zone(None).alias("t"), pl.col("total").cum_max())
    )
    # Vega draws times in the viewer's own time zone. Naive Berlin wall-clock times (no zone)
    # are read as local and shown unchanged, so the axis really is Berlin time for everyone.
    _events = pl.DataFrame(
        {
            "t": ["2026-11-22T00:07:59Z"],
            "label": ["Posted to Hacker News"],
        }
    ).with_columns(pl.col("t").str.to_datetime(time_zone="UTC").dt.convert_time_zone("Europe/Berlin").dt.replace_time_zone(None))
    _checks = pl.DataFrame(
        {
            "t": ["2026-11-21T16:46:00Z"],
            "total": [3834],
            "label": ["Creator's Mastodon count: 3,834"],
        }
    ).with_columns(pl.col("t").str.to_datetime(time_zone="UTC").dt.convert_time_zone("Europe/Berlin").dt.replace_time_zone(None))

    _line = (
        alt.Chart(_cum)
        .mark_line(color=BLUE, strokeWidth=2, interpolate="monotone")
        .encode(
            x=alt.X("t:T", title=None),
            y=alt.Y("total:Q", title="Sandwiches made so far", axis=alt.Axis(format=",.0f")),
        )
    )
    _hover = (
        alt.Chart(_cum)
        .mark_point(opacity=0, size=60)
        .encode(
            x="t:T",
            y="total:Q",
            tooltip=[alt.Tooltip("t:T", format="%a %d %b %H:%M", title="Berlin time"), alt.Tooltip("total:Q", format=",.0f", title="Running total (ID)")],
        )
    )
    _rule = alt.Chart(_events).mark_rule(color=GRAY, strokeDash=[4, 3]).encode(x="t:T", tooltip=["label", alt.Tooltip("t:T", format="%a %d %b %H:%M")])
    _check = alt.Chart(_checks).mark_point(shape="diamond", size=110, filled=True, opacity=1, color=ORANGE).encode(
        x="t:T", y="total:Q", tooltip=["label", alt.Tooltip("t:T", format="%a %d %b %H:%M")]
    )
    _last = _cum.row(-1, named=True)
    _cum_chart = (_line + _hover + _rule + _check).properties(
        width=620,
        height=280,
        title=ttl(
            f"{_last['total']:,} Stullen by {_last['t']:%a %d %b %H:%M}",
            "Running total of sandwiches made (serial ID at creation time; includes deleted ones), Berlin time",
            "Dashed line: posted to Hacker News. Orange diamond: the creator's own count on Mastodon (3,834).",
        ),
    )
    mo.ui.tabs(
        {
            "Per hour": hourly_chart,
            "Cumulative": mo.vstack(
                [
                    _cum_chart,
                    mo.md(
                        "IDs are serial, so this curve is exact. The slope is the creation rate: bursts are steep stretches, overnight lulls are plateaus. "
                        "The orange diamond is a count the creator posted on Mastodon. It sits about 135 below the curve because "
                        "it counts sandwiches that still exist, while serial numbers also count deleted ones (~3%, matching the 404s we see)."
                    ),
                ]
            ),
        }
    )
    return


@app.cell
def _(df, sample_rate):
    _daily = (
        df.sort("created_berlin").group_by_dynamic("created_berlin", every="1h").agg(pl.len().alias("n"))
        .filter(pl.col("created_berlin") >= pl.datetime(2026, 11, 20, 12, time_zone="Europe/Berlin"))
    )
    _scale = sample_rate  # sample -> full population
    _daily = _daily.with_columns((pl.col("n") / _scale).round(0).alias("est_total"))
    # naive Berlin wall-clock time, so Vega shows Berlin time whatever the viewer's zone
    _daily_plot = _daily.with_columns(pl.col("created_berlin").dt.replace_time_zone(None))
    hourly_chart = (
        alt.Chart(_daily_plot)
        .mark_area(color=BLUE, fillOpacity=0.25, line={"color": BLUE, "strokeWidth": 2})
        .encode(
            x=alt.X("created_berlin:T", title=None),
            y=alt.Y("est_total:Q", title="Sandwiches per hour (est.)", axis=alt.Axis(format=",.0f")),
            tooltip=[alt.Tooltip("created_berlin:T", format="%a %d %b %H:00", title="Hour (Berlin)"), alt.Tooltip("est_total:Q", format=",.0f", title="Est. sandwiches")],
        )
        .properties(
            width=620,
            height=220,
            title=ttl(
                f"Peak: ~{int(_daily['est_total'].max()):,} Stullen an hour, {_daily.sort('est_total')['created_berlin'][-1]:%a %d %b %H:00}",
                "Sandwiches made per hour, Berlin time; sample counts scaled up by the sampling rate. The last hour may be incomplete.",
            ),
        )
    )
    _tz = (
        df.group_by("timezone").len("n").sort("n", descending=True).head(12)
        .with_columns((pl.col("n") / df.height).alias("share"))
    )
    _tz_base = alt.Chart(_tz).encode(
        y=alt.Y("timezone:N", sort=_tz["timezone"].to_list(), title=None),
        x=alt.X("share:Q", axis=None, title=None),
        tooltip=["timezone", alt.Tooltip("n:Q", format=",.0f"), alt.Tooltip("share:Q", format=".1%")],
    )
    _tz_chart = with_values(
        _tz_base.mark_bar(cornerRadiusEnd=4, color=BLUE, height=14),
        _tz, "timezone:N", {r["timezone"]: f"{r['share']:.0%}" for r in _tz.iter_rows(named=True)},
    ).properties(
        width=300,
        height=alt.Step(18),
        title=ttl(f"{_tz['share'][0]:.0%} were made in {_tz['timezone'][0]} time", "% of sandwiches by the maker's browser time zone, 12 most common"),
    )
    _hot = _daily.filter(pl.col("est_total") >= 100)
    _launch = _hot["created_berlin"].min() if _hot.height else None
    _pre = df.filter(pl.col("created_berlin") < _launch) if _launch is not None else df.head(0)
    _note = (
        f"Before **{_launch:%a %d %b, %H:00}** Berlin time the counter barely moves: {_pre.height:,} sampled sandwiches "
        f"(IDs up to ~{_pre['id'].max():,}) between {_pre['created_berlin'].min():%d %b} and launch, which looks like pre-launch testing. "
        f"After that it runs at hundreds to thousands per hour."
        if _launch is not None and _pre.height
        else ""
    )
    mo.vstack([mo.md(_note), _tz_chart])
    return (hourly_chart,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 9 · Hand-Me-Down Sandwiches
    ### Remixes

    *Weiterbelegen* ("keep topping") loads someone else's Stulle into your editor, and the
    result records it as `parentId`. Following parents back gives remix chains.
    """)
    return


@app.cell
def _(df):
    _parent = dict(zip(df["id"].to_list(), df["parent_id"].to_list()))

    def _depth(i, seen=0):
        d = 0
        while i is not None and d < 500:
            p = _parent.get(i)
            if p is None:
                # parent missing from sample still counts as one more hop if referenced
                break
            d += 1
            i = p
        return d

    _remixed = df.filter(pl.col("parent_id").is_not_null())
    _most = df.filter(pl.col("parent_id").is_not_null()).group_by("parent_id").len("remixes").sort("remixes", descending=True).head(10)
    _added = _remixed.join(df.select(pl.col("id").alias("parent_id"), pl.col("n_toppings").alias("parent_toppings")), on="parent_id", how="inner").with_columns((pl.col("n_toppings") - pl.col("parent_toppings")).alias("added"))
    mo.vstack(
        [
            mo.md(
                f"**{_remixed.height / df.height:.1%}** of sandwiches are remixes. When both the remix and its parent are in the sample "
                f"({_added.height:,} pairs), the remix adds a median of **{_added['added'].median():.0f}** pieces. "
                f"The longest chain found in the sample is **{max((_depth(i) for i in _remixed['id'].to_list()), default=0)}** hops."
            ),
            mo.md("**Most-remixed Stullen** (counts are within the sample)"),
            mo.ui.table(
                    _most.select(
                        pl.format("https://img.stulle.app/{}/thumb.jpg", "parent_id").alias("Original"),
                        pl.col("parent_id").alias("ID"),
                        pl.col("remixes").alias("Remixes in sample"),
                    ),
                    selection=None,
                    show_column_summaries=False,
                ),
        ]
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 10 · Nutella-Salami Theory
    ### Sweet, savory, or chaos?
    """)
    return


@app.cell
def _(KIND_OF, df):
    def _profile(tops, spr):
        kinds = {KIND_OF[t] for t in tops if t in KIND_OF}
        sweet = bool(kinds & {"sweet"}) or bool(set(spr) & {"jam", "nutella"})
        savory = bool(kinds & {"savory", "veg", "junk"}) or bool(set(spr) & {"leberwurst"})
        if sweet and savory:
            return "Chaos (sweet + savory)"
        if sweet:
            return "Sweet"
        if savory:
            return "Savory"
        return "Plain (butter/quark only)"

    _p = (
        df.with_columns(pl.struct("topping_types", "spread_types").map_elements(lambda r: _profile(r["topping_types"], r["spread_types"]), return_dtype=pl.Utf8).alias("profile"))
        .group_by("profile").len("n")
        .with_columns((pl.col("n") / df.height).alias("share"))
    )
    _order = ["Savory", "Sweet", "Chaos (sweet + savory)", "Plain (butter/quark only)"]
    _sh = dict(zip(_p["profile"], _p["share"]))
    _pb = alt.Chart(_p).encode(
        y=alt.Y("profile:N", sort=_order, title=None),
        x=alt.X("share:Q", axis=None, title=None),
        tooltip=["profile", alt.Tooltip("n:Q", format=",.0f"), alt.Tooltip("share:Q", format=".1%")],
    )
    _chart = with_values(
        # one accent: the chaos bar is the story, the rest stay grey
        _pb.mark_bar(cornerRadiusEnd=4, height=18).encode(
            color=alt.condition(alt.datum.profile == "Chaos (sweet + savory)", alt.value(MAGENTA), alt.value(GRAY))
        ),
        _p, "profile:N", {o: f"{_sh.get(o, 0):.0%}" for o in _order},
    ).properties(
        width=480,
        height=alt.Step(26),
        title=ttl(
            f"{_sh.get('Chaos (sweet + savory)', 0):.0%} of Stullen mix sweet and savory",
            f"% of sandwiches by flavour profile; savory {_sh.get('Savory', 0):.0%}, sweet {_sh.get('Sweet', 0):.0%}. n = {df.height:,}.",
        ),
    )
    mo.vstack([mo.md("Sweet = jam, Nutella, fruit or gummy bears; savory = everything else except plain butter/quark."), _chart])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 11 · Window-Shopping at the Deli
    ### Browse the sample

    Every scraped Stulle with its photo. Use search, column sorting and filters to dig
    around, e.g. sort by height or filter to one bread.
    """)
    return


@app.cell(hide_code=True)
def _(df, fam, lab):
    _browse = df.select(
        pl.format("https://img.stulle.app/{}/thumb.jpg", "id").alias("Photo"),
        pl.col("id").alias("ID"),
        pl.col("created_berlin").dt.strftime("%d %b %H:%M").alias("Made (Berlin)"),
        pl.col("bread").map_elements(lab, return_dtype=pl.Utf8).alias("Bread"),
        pl.col("ingredients")
        .list.eval(pl.element().map_elements(lab, return_dtype=pl.Utf8))
        .list.join(", ")
        .alias("Ingredients"),
        pl.col("n_toppings").alias("Pieces"),
        pl.col("height_cm").round(1).alias("Height cm"),
        pl.col("famous").list.eval(pl.element().map_elements(fam, return_dtype=pl.Utf8)).list.join(", ").alias("Classic"),
        pl.col("timezone").alias("Time zone"),
    ).sort("ID", descending=True)
    mo.ui.table(_browse, selection=None, page_size=10, wrapped_columns=["Ingredients"])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 12 · CLIP Art
    ### Searching the photos by meaning

    The JSON says *what* is on a Stulle, not what it *looks like*. Spreads in particular
    are free-hand brush strokes, and people use them to draw, write slogans, and paint
    flags. To find those we embed every thumbnail with **CLIP ViT-L/14**, a model trained to
    put images and their descriptions close together. Then we can search the photos with
    plain text, sort every photo into a rough taxonomy, and lay them all out on a map.

    The pipeline embeds every photo with CLIP ViT-L/14 and precomputes the census, the preset
    searches and the map, so all of this also works in the browser. Free-text search needs
    the CLIP text model, so it appears only where the notebook runs on a real Python server
    (locally, or on molab).
    """)
    return


@app.cell
async def _(df, meta, reload):
    reload
    # Precomputed by the pipeline: map position, zero-shot category and preset searches
    _map_b, _hunts_b = await read_files(["map.parquet", "hunts.parquet"])
    _ids_in_sample = df["id"].implode()
    clip_map_all = pl.read_parquet(io.BytesIO(_map_b)).filter(pl.col("id").is_in(_ids_in_sample))
    hunt_hits = pl.read_parquet(io.BytesIO(_hunts_b))

    # Free-text search embeds your query with CLIP's text encoder, so it needs PyTorch and
    # sentence-transformers: available locally and on molab, not in the browser.
    try:
        if IN_BROWSER:
            raise ImportError
        import torch as _torch
        from sentence_transformers import SentenceTransformer

        HAS_CLIP = True
    except ImportError:
        HAS_CLIP = False

    if HAS_CLIP:
        _parts = meta["parts"]
        _files = await read_files([f"clip/{_p}.{_k}.npy" for _p in _parts for _k in ("ids", "emb")])
        _ids = np.concatenate([np.load(io.BytesIO(_b)) for _b in _files[0::2]])
        _emb = np.concatenate([np.load(io.BytesIO(_b)) for _b in _files[1::2]]).astype(np.float32)
        _keep = np.isin(_ids, df["id"].to_numpy())
        clip_ids, clip_emb = _ids[_keep], _emb[_keep]

        @mo.cache
        def _load_clip():
            return SentenceTransformer(meta["clip_model"], device="mps" if _torch.backends.mps.is_available() else "cpu")

        clip_model = _load_clip()

        def embed_text(texts):
            return clip_model.encode(list(texts), normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
    else:
        clip_ids, clip_emb, embed_text = np.zeros(0, dtype=np.int64), np.zeros((0, 768), dtype=np.float32), None


    def photo_grid(ids, captions=None, size=150):
        """Clickable thumbnail grid; each photo links to its permalink."""
        captions = captions or [""] * len(ids)
        tiles = "".join(
            f'<a href="https://stulle.app/{i}" target="_blank" style="text-decoration:none;color:inherit">'
            f'<img src="https://img.stulle.app/{i}/thumb.jpg" loading="lazy" style="width:{size}px;height:{size}px;object-fit:cover;border-radius:6px;display:block">'
            f'<div style="font-size:11px;opacity:.72;margin-top:2px">#{i:,} {c}</div></a>'
            for i, c in zip(ids, captions)
        )
        return mo.Html(f'<div style="display:flex;flex-wrap:wrap;gap:8px">{tiles}</div>')


    mo.md(
        f"**{clip_map_all.height:,}** photos embedded ({clip_map_all.height / max(df.height, 1):.0%} of the sandwiches in the sample)."
        + ("" if HAS_CLIP else " Free-text search is off here (no CLIP text model in this runtime); the presets still work.")
    )
    return HAS_CLIP, clip_emb, clip_ids, clip_map_all, embed_text, hunt_hits, photo_grid


@app.cell
def _(HAS_CLIP, catalog):
    hunt_preset = mo.ui.dropdown(options=catalog["hunts"], value="☭ Hammer and sickle", label="Hunt for")
    hunt_text = mo.ui.text(placeholder="…or describe anything, e.g. 'a cat made of salami'", full_width=True, disabled=not HAS_CLIP)
    hunt_n = mo.ui.slider(12, 96, step=12, value=36, label="Results", show_value=True)
    mo.vstack([mo.hstack([hunt_preset, hunt_n], justify="start", gap=2)] + ([hunt_text] if HAS_CLIP else []))
    return hunt_n, hunt_preset, hunt_text


@app.cell
def _(
    HAS_CLIP,
    clip_emb,
    clip_ids,
    clip_map_all,
    embed_text,
    hunt_hits,
    hunt_n,
    hunt_preset,
    hunt_text,
    photo_grid,
):
    _free = hunt_text.value.strip() if HAS_CLIP else ""
    if _free:
        _scores = clip_emb @ embed_text([_free])[0]
        _top = np.argsort(-_scores)[: hunt_n.value]
        _ids, _sc, _query = [int(clip_ids[i]) for i in _top], [float(_scores[i]) for i in _top], _free
    else:
        _h = hunt_hits.filter(pl.col("hunt") == hunt_preset.value, pl.col("id").is_in(clip_map_all["id"].implode())).sort("rank").head(hunt_n.value)
        _ids, _sc, _query = _h["id"].to_list(), _h["score"].to_list(), hunt_preset.value
    _out = mo.vstack(
        [
            mo.md(f"Top {len(_ids)} matches for *\u201c{_query}\u201d*. The number is cosine similarity; anything above ~0.27 is usually a real hit."),
            photo_grid(_ids, [f"· {v:.2f}" for v in _sc]),
        ]
    )
    _out
    return


@app.cell
def _(catalog, clip_map_all, df):
    # Zero-shot taxonomy (computed by the pipeline): each photo goes to the prompt it is most
    # similar to. The prompts are in stulle/publish.py (TAXONOMY).
    TAXONOMY = catalog["taxonomy"]
    clip_cats = clip_map_all.select("id", "category", "confidence").join(
        df.select("id", "n_toppings", "spread_types", "brush_segments", "height_cm"), on="id"
    )
    return TAXONOMY, clip_cats


@app.cell
def _(clip_cats):
    if clip_cats.height == 0:
        _view = mo.md("No embeddings yet.")
    else:
        _share = clip_cats.group_by("category").len("n").with_columns((pl.col("n") / clip_cats.height).alias("share"))
        _share = _share.sort("n", descending=True)
        _s0 = _share.row(0, named=True)
        _cb = alt.Chart(_share).encode(
            y=alt.Y("category:N", sort=_share["category"].to_list(), title=None),
            x=alt.X("share:Q", axis=None, title=None),
            tooltip=["category", alt.Tooltip("n:Q", format=",.0f"), alt.Tooltip("share:Q", format=".1%")],
        )
        _chart = with_values(
            _cb.mark_bar(cornerRadiusEnd=4, color=BLUE, height=14),
            _share, "category:N", {r["category"]: f"{r['share']:.0%}" for r in _share.iter_rows(named=True)},
        ).properties(
            width=480,
            height=alt.Step(18),
            title=ttl(
                f"CLIP calls {_s0['share']:.0%} of photos “{_s0['category']}”",
                f"% of {clip_cats.height:,} photos by zero-shot CLIP category (model estimate, not hand-labelled; see the check below)",
            ),
        )
        # Validation against the structured data we know to be true
        _bare_truth = (clip_cats["n_toppings"] == 0) & (clip_cats["spread_types"].list.len() == 0)
        _bare_pred = clip_cats["category"] == "Bare bread"
        _tp = int((_bare_truth & _bare_pred).sum())
        _prec = _tp / max(int(_bare_pred.sum()), 1)
        _rec = _tp / max(int(_bare_truth.sum()), 1)
        _tower_h = clip_cats.filter(pl.col("category") == "Tower")["height_cm"].median()
        _other_h = clip_cats.filter(pl.col("category") != "Tower")["height_cm"].median()
        _draw = clip_cats.group_by(pl.col("category").is_in(["Drawing / symbol", "Words", "Flag", "Face"]).alias("art")).agg(
            pl.col("brush_segments").median().alias("brush"), pl.col("n_toppings").median().alias("pieces")
        ).sort("art")
        _art_brush, _rest_brush = _draw.filter(pl.col("art"))["brush"].item(), _draw.filter(~pl.col("art"))["brush"].item()
        _view = mo.vstack(
            [
                _chart,
                mo.callout(
                    mo.md(
                        f"""**Is the census any good?** Some of it can be checked against the JSON:

    * **Bare bread**: of the photos CLIP calls bare bread, **{_prec:.0%}** really have no toppings and no spread (precision).
      It finds **{_rec:.0%}** of the truly bare ones (recall).
    * **Towers**: photos CLIP calls towers have a median estimated height of **{_tower_h:.1f} cm**, against **{_other_h:.1f} cm** for everything else.
    * **Art** (drawings, words, flags, faces): a median of **{_art_brush:.0f}** brush segments of spread, against **{_rest_brush:.0f}** for the rest. People draw with the knife.

    Treat the shares as rough. The categories overlap, and every photo is forced into exactly one."""
                    ),
                    kind="info",
                ),
            ]
        )
    _view
    return


@app.cell
def _(TAXONOMY):
    cat_pick = mo.ui.dropdown(options=list(TAXONOMY), value="Drawing / symbol", label="Show category")
    cat_pick
    return (cat_pick,)


@app.cell
def _(cat_pick, clip_cats, photo_grid):
    _c = clip_cats.filter(pl.col("category") == cat_pick.value).sort("confidence", descending=True).head(36)
    photo_grid(_c["id"].to_list(), [f"· {v:.0%}" for v in _c["confidence"].to_list()]) if _c.height else mo.md("Nothing in this category yet.")
    return


@app.cell
def _(TAXONOMY, clip_map_all):
    # The 2D layout is computed by the pipeline (`uv run pipeline.py --steps map`), not here:
    # UMAP on ~50k photos takes minutes, and Numba aborts if two sessions run it at once.
    clip_map = clip_map_all.drop_nulls("x").select("id", "category", "x", "y")
    _cats = list(TAXONOMY)
    _palette = [GRAY, BLUE, GRAY_DARK, ORANGE, YELLOW, AQUA, VIOLET, MAGENTA, GREEN, RED]
    map_chart = mo.ui.altair_chart(
        alt.Chart(clip_map)
        .mark_circle(size=8, opacity=0.6)
        .encode(
            x=alt.X("x:Q", axis=None),
            y=alt.Y("y:Q", axis=None),
            color=alt.Color("category:N", scale=alt.Scale(domain=_cats, range=_palette), legend=alt.Legend(orient="right", title=None)),
            tooltip=["id", "category"],
        )
        .properties(
            width=620,
            height=480,
            title=ttl(
                "Similar-looking Stullen form islands",
                f"{clip_map.height:,} photos placed by CLIP similarity (UMAP; the axes have no units), coloured by CLIP category. Drag a box to see the photos.",
            ),
        ),
        chart_selection="interval",
    )
    map_chart
    return clip_map, map_chart


@app.cell
def _(clip_map, map_chart, photo_grid):
    _sel = map_chart.value
    if _sel is None or len(_sel) == 0 or len(_sel) == clip_map.height:
        _out = mo.md("*Drag a box on the map to see those photos. Islands tend to be one kind of thing: towers, flags, drawings, fruit faces…*")
    else:
        _ids = pl.DataFrame(_sel)["id"].to_list()
        _out = mo.vstack([mo.md(f"**{len(_ids):,} photos selected** (showing up to 48)"), photo_grid(_ids[:48], size=120)])
    _out
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 13 · Roll for Stulle
    ### Natural 20 Gurken

    In the spirit of *Roll for Sandwich* (Jacob Pauwels rolls D&D dice against ingredient
    tables, then builds and rates the result): roll a Stulle from dice tables, then check the
    real dataset. Has anyone already made it? Which real Stullen are closest?
    """)
    return


@app.cell(hide_code=True)
def _():
    roll = mo.ui.button(label="🎲 Roll for Stulle", value=0, on_click=lambda v: v + 1, kind="success")
    roll
    return (roll,)


@app.cell(hide_code=True)
def _():
    DICE = {
        "Bread (d4)": ["roggenmisch", "graubrot", "vollkorn", "toast"],
        "Spread (d6)": ["butter", "quark", "leberwurst", "jam", "nutella", None],
        "Main (d8)": ["kaese", "salami", "schinken", "leberkaese", "lachs", "blutwurst", "ei", "spiegelei"],
        "Roughage (d8)": ["gurke", "tomate", "radieschen", "gewuerzgurke", "senfgurke", "rotebete", None, None],
        "Wild magic (d10)": ["gummibaer", "pommes", "chips", "fischstaebchen", "banane", "erdbeere", "kiwi", "harzer", None, "DOUBLE"],
    }
    return (DICE,)


@app.cell(hide_code=True)
def _(DICE, df, lab, photo_grid, roll, sample_rate):
    _rng = np.random.default_rng(None if roll.value else 20)  # first view is a fixed roll, then truly random
    _rolls = {t: int(_rng.integers(len(opts))) for t, opts in DICE.items()}
    _pick = {t: DICE[t][k] for t, k in _rolls.items()}
    _bread = _pick["Bread (d4)"]
    _spreads = {_pick["Spread (d6)"]} - {None}
    _tops = {_pick["Main (d8)"], _pick["Roughage (d8)"], _pick["Wild magic (d10)"]} - {None, "DOUBLE"}
    _crit = _pick["Wild magic (d10)"] == "DOUBLE"


    def _face(t, k):
        v = DICE[t][k]
        name = "nothing" if v is None else ("⚡ natural 10: double main!" if v == "DOUBLE" else lab(v))
        return f"| {t} | **{k + 1}** | {name} |"


    _target = _spreads | _tops
    _sets = df.select("id", "bread", pl.col("ingredients").list.unique().alias("ing"))
    _sims = _sets.with_columns(
        pl.col("ing").map_elements(lambda s: len(set(s) & _target) / max(len(set(s) | _target), 1), return_dtype=pl.Float64).alias("jaccard")
    )
    _exact = _sims.filter(pl.col("jaccard") == 1.0)
    _near = _sims.sort(["jaccard", pl.col("bread") == _bread], descending=True).head(12)
    _verdict = (
        f"🎯 **{_exact.height:,}** sandwiches in the sample have exactly these ingredients, "
        f"which scales to about **{_exact.height / sample_rate:,.0f}** on the whole site."
        if _exact.height
        else "✨ **Critical hit: nobody in the sample has built this one yet.** Go make it and claim it!"
    )
    mo.vstack(
        [
            mo.md("| Table | Roll | Result |\n|---|---|---|\n" + "\n".join(_face(t, k) for t, k in _rolls.items())),
            mo.md(
                f"**Your Stulle:** {lab(_bread)} with {', '.join(lab(s) for s in _spreads) or 'no spread'}"
                f"{' and ' + ', '.join(lab(t) for t in sorted(_tops)) if _tops else ''}{' (double main!)' if _crit else ''}.\n\n{_verdict}"
            ),
            mo.md("**Closest real Stullen** (Jaccard similarity of the ingredient sets):"),
            photo_grid(_near["id"].to_list(), [f"· {j:.0%}" for j in _near["jaccard"].to_list()], size=130),
        ]
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ---
    ### Method notes (the crusts: boring, but they hold it together)
    * **Source.** `GET https://stulle.app/api/photos/{id}` for every ID up to the newest one
      listed on `/alle`. 404s are deleted or reported sandwiches. The scraper is throttled
      to about 2–3 requests per second and backs off when it gets an HTTP 429.
    * **Sampling.** IDs are fetched in shuffled order, so until the scrape finishes every
      percentage here is an estimate from a uniform random sample. Absolute counts (pieces
      placed, remix counts) only cover the sample.
    * **Spreads.** A spread is "used" if it was picked at least once. The brush strokes are
      dropped when scraping; only their count is kept.
    * **Physics.** Topping diameters and thicknesses come from the app's JS bundle. The
      bread is 11 cm across.
    """)
    return


if __name__ == "__main__":
    app.run()
