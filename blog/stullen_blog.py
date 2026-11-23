# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "altair>=6.0.0",
#     "marimo>=0.25.2",
#     "numpy>=2.2.0",
#     "polars>=1.33.0",
#     "pyarrow>=22.0.0",
#     "tzdata>=2025.2",
# ]
# ///
"""The interactive pieces of the blog post, as marimo islands.

This is a browser-friendly cut of ../stullen.py: no PyTorch, no 9 MB toppings table, only
the small precomputed files from dataset/. Cells whose first line is `# island: <name>`
become islands that the post places by name; all other cells run invisibly.

Build (from the website repo):  uv run scripts/embed-islands.py <this file> stullen
"""

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")

with app.setup:
    import asyncio
    import io
    import json
    import sys
    from pathlib import Path

    import altair as alt
    import marimo as mo
    import numpy as np
    import polars as pl
    # Imported only so that marimo loads them into Pyodide: polars reads parquet through
    # pyarrow in WebAssembly, and needs the tzdata package for Europe/Berlin.
    import pyarrow  # noqa: F401
    import tzdata  # noqa: F401

    DATA_URL = "https://raw.githubusercontent.com/ktaletsk/stullen-statistik/main/dataset/"
    IN_BROWSER = sys.platform == "emscripten"

    # Charts must read equally well on a light page and a dark one (this post, the editor,
    # molab in either theme). So no text, axis or grid colour is set anywhere: marimo hands
    # Vega its light or dark theme to match the page, and Vega supplies the ink. Data colours
    # are mid-tone steps with at least 3:1 contrast against both white and near-black, and
    # shading uses opacity of one hue, so "low" fades into whichever background is there.
    BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#3987e5", "#d95926", "#199e70", "#c98500", "#d55181"
    GREEN, VIOLET, RED, GRAY = "#0ca30c", "#9085e9", "#e34948", "#8a8984"
    KIND_COLORS = {"spread": BLUE, "savory": ORANGE, "veg": AQUA, "junk": YELLOW, "sweet": MAGENTA}

    def with_values(bars, data, field, labels):
        """Print each bar's value in a column on the right, as a second category axis.

        Text marks can't follow the light/dark theme (Vega-Lite bakes their colour in), but
        axis labels can; so every piece of text in these charts is a title, axis or legend.
        """
        values = alt.Chart(data).mark_tick(opacity=0).encode(
            y=alt.Y(field, sort=list(labels), title=None, axis=alt.Axis(
                orient="right", domain=False, ticks=False, labelPadding=6,
                labelExpr=f"{json.dumps(labels, ensure_ascii=False)}[datum.value]",
            ))
        )
        return alt.layer(bars, values).resolve_axis(y="independent")

    @alt.theme.register("stullen_any_mode", enable=True)
    def _theme():
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

    def ttl(text, *sub):
        return alt.Title(text, subtitle=list(sub))

    def _local_dataset():
        bases = []
        try:
            bases.append(Path(mo.notebook_dir()))
        except Exception:
            pass
        bases.append(Path.cwd())
        for b in bases:
            for c in (b / "dataset", b.parent / "dataset"):
                if (c / "meta.json").exists():
                    return c
        return None

    async def read_files(paths):
        """Bytes of dataset files: local while the post is being built, GitHub in the browser."""
        if IN_BROWSER:
            from pyodide.http import pyfetch

            async def one(p):
                r = await pyfetch(DATA_URL + p)
                return await r.bytes()

            return list(await asyncio.gather(*(one(p) for p in paths)))
        root = _local_dataset()
        return [(root / p).read_bytes() for p in paths]

    def photo_grid(ids, captions=None, size=120):
        captions = captions or [""] * len(ids)
        tiles = "".join(
            f'<a href="https://stulle.app/{i}" target="_blank" rel="noopener" style="text-decoration:none;color:inherit">'
            f'<img src="https://img.stulle.app/{i}/thumb.jpg" loading="lazy" alt="Stulle #{i}" style="width:{size}px;height:{size}px;object-fit:cover;border-radius:6px;display:block;margin:0">'
            f'<div style="font-size:11px;opacity:.7;margin-top:2px">#{i:,} {c}</div></a>'
            for i, c in zip(ids, captions)
        )
        return mo.Html(f'<div style="display:flex;flex-wrap:wrap;gap:8px">{tiles}</div>')


@app.cell
async def _():
    meta, catalog = [json.loads(b) for b in await read_files(["meta.json", "catalog.json"])]
    _files = await read_files([f"stullen/{p}.parquet" for p in meta["parts"]] + ["map.parquet", "hunts.parquet", "positions.parquet"])
    stullen = pl.concat([pl.read_parquet(io.BytesIO(b)) for b in _files[:-3]])
    clip_map, hunt_hits, positions = (pl.read_parquet(io.BytesIO(b)) for b in _files[-3:])
    df = stullen.with_columns(
        pl.concat_list("spread_types", "topping_types").alias("ingredients"),
        pl.col("created_at").dt.convert_time_zone("Europe/Berlin").alias("created_berlin"),
    )
    return catalog, clip_map, df, hunt_hits, meta, positions


@app.cell
def _(catalog):
    _names = {**catalog["toppings"], **catalog["spreads"], **catalog["breads"]}
    KIND_OF = {k: v["kind"] for k, v in {**catalog["toppings"], **catalog["spreads"]}.items()}
    CLASSICS = catalog["classics"]

    def _both(de, en):
        return de if de.casefold() == en.casefold() else f"{de} ({en})"

    def lab(key):
        n = _names.get(key)
        return _both(n["de"], n["en"]) if n else key

    def fam(name):
        c = CLASSICS.get(name)
        return _both(c["de"], c["en"]) if c else name

    KIND_LABELS = {k: _both(v["de"], v["en"]) for k, v in catalog["kinds"].items()}
    kind_scale = alt.Scale(domain=list(KIND_LABELS.values()), range=[KIND_COLORS[k] for k in KIND_LABELS])
    return CLASSICS, KIND_LABELS, KIND_OF, fam, kind_scale, lab


@app.cell
def _(df, meta):
    # island: kpis
    # plain HTML tiles that take the page's own text colour, so they work in light and dark
    def _tile(label, value, note):
        return (
            '<div style="flex:1 1 140px;border:1px solid color-mix(in srgb, currentColor 22%, transparent);border-radius:10px;padding:12px 14px">'
            f'<div style="font-size:13px;opacity:.72">{label}</div>'
            f'<div style="font-size:26px;font-weight:600;line-height:1.3">{value}</div>'
            f'<div style="font-size:12px;opacity:.72">{note}</div></div>'
        )

    _tiles = "".join(
        [
            _tile("Stullen", f"{df.height:,}", "still on the site"),
            _tile("Median toppings", f"{df['n_toppings'].median():.0f}", "pieces per sandwich"),
            _tile("Bare bread", f"{(df['n_toppings'] == 0).mean():.0%}", "no toppings at all"),
            _tile("Remixes", f"{df['parent_id'].is_not_null().mean():.0%}", "built on someone else's"),
        ]
    )
    mo.vstack(
        [
            mo.Html(f'<div style="display:flex;flex-wrap:wrap;gap:10px">{_tiles}</div>'),
            mo.md(
                f"<small>Live data, updated {meta['updated_at'][:10]}: {meta['deleted']:,} deleted and "
                f"{meta['blank_photos']:,} banned sandwiches left out.</small>"
            ),
        ]
    )
    return


@app.cell
def _(KIND_LABELS, KIND_OF, df, kind_scale, lab):
    # island: popularity
    _n = df.height
    _pop = (
        df.select("id", pl.col("ingredients").list.unique()).explode("ingredients").drop_nulls()
        .group_by("ingredients").len("sandwiches")
        .filter(pl.col("sandwiches") >= 10)
        .with_columns(
            (pl.col("sandwiches") / _n).alias("share"),
            pl.col("ingredients").replace_strict({k: lab(k) for k in KIND_OF}, default=pl.col("ingredients")).alias("ingredient"),
            pl.col("ingredients").replace_strict({k: KIND_LABELS[v] for k, v in KIND_OF.items()}, default="?").alias("group"),
        )
        .sort("share", descending=True)
    )
    _lead = _pop.row(0, named=True)
    _base = alt.Chart(_pop).encode(
        y=alt.Y("ingredient:N", sort=_pop["ingredient"].to_list(), title=None),
        x=alt.X("share:Q", axis=None),
        tooltip=[alt.Tooltip("ingredient:N", title="Ingredient"), alt.Tooltip("share:Q", format=".1%", title="% of Stullen"), alt.Tooltip("sandwiches:Q", format=",.0f", title="Stullen")],
    )
    with_values(
        _base.mark_bar(cornerRadiusEnd=4, height=14).encode(color=alt.Color("group:N", scale=kind_scale, title=None, legend=alt.Legend(orient="top", columns=2, labelLimit=400))),
        _pop, "ingredient:N", {r["ingredient"]: f"{r['share']:.0%}" for r in _pop.iter_rows(named=True)},
    ).properties(width="container", height=alt.Step(18), title=ttl(f"{_lead['ingredient']} is on {_lead['share']:.0%} of all Stullen", f"% of {_n:,} sandwiches using each ingredient; colour = group"))
    return


@app.cell
def _(KIND_LABELS, KIND_OF, df, kind_scale, lab):
    # island: pairs
    _sets = [set(s) for s in df["ingredients"].to_list()]
    _counts = {}
    for _s in _sets:
        for _k in _s:
            _counts[_k] = _counts.get(_k, 0) + 1
    _rank = {k: i for i, k in enumerate(["spread", "savory", "veg", "junk", "sweet"])}
    _keys = sorted((k for k, c in _counts.items() if c >= 30 and k in KIND_OF), key=lambda k: (_rank[KIND_OF[k]], -_counts[k]))
    _idx = {k: i for i, k in enumerate(_keys)}
    _X = np.zeros((len(_sets), len(_keys)), dtype=np.float32)
    for _r, _s in enumerate(_sets):
        for _k in _s:
            if _k in _idx:
                _X[_r, _idx[_k]] = 1
    _co = _X.T @ _X
    # size-adjusted lift: chance is computed among sandwiches with the same number of ingredients
    _size = np.minimum(_X.sum(axis=1), 12)
    _exp = np.zeros_like(_co)
    for _k in np.unique(_size):
        _Xk = _X[_size == _k]
        _c = _Xk.sum(axis=0)
        _exp += np.outer(_c, _c) / len(_Xk)
    _lift = np.divide(_co, _exp, out=np.ones_like(_co), where=_exp > 0)
    _names = [lab(k) for k in _keys]
    _rows = [
        {"a": _names[i], "b": _names[j], "n": int(_co[i, j]), "lift": float(_lift[i, j]), "l2": float(np.log2(max(_lift[i, j], 1 / 16))),
         "dir": "together" if _lift[i, j] >= 1 else "apart", "strength": float(min(abs(np.log2(max(_lift[i, j], 1 / 16))), 2))}
        for i in range(len(_keys)) for j in range(len(_keys)) if i != j
    ]
    _long = pl.DataFrame(_rows)
    _top = _long.filter(pl.col("a") < pl.col("b"), pl.col("n") >= 30).sort("lift", descending=True).row(0, named=True)
    _heat = (
        alt.Chart(_long).mark_rect()
        .encode(
            x=alt.X("a:N", sort=_names, title=None, axis=alt.Axis(labelAngle=-50), scale=alt.Scale(paddingInner=0.08)),
            y=alt.Y("b:N", sort=_names, title=None, scale=alt.Scale(paddingInner=0.08)),
            color=alt.Color("dir:N", title=None, scale=alt.Scale(domain=["together", "apart"], range=[RED, BLUE]), legend=alt.Legend(orient="bottom", direction="horizontal")),
            opacity=alt.Opacity("strength:Q", title="× or ÷ chance", scale=alt.Scale(domain=[0, 1], range=[0.05, 1], clamp=True), legend=alt.Legend(orient="bottom", direction="horizontal", values=[0.25, 0.5, 1], labelExpr="format(pow(2, datum.value), '.1f') + '×'", symbolFillColor=GRAY, symbolType="square")),
            tooltip=[alt.Tooltip("a:N", title="Ingredient"), alt.Tooltip("b:N", title="with"), alt.Tooltip("n:Q", format=",.0f", title="Stullen with both"), alt.Tooltip("lift:Q", format=".2f", title="× chance")],
        )
        .properties(width=alt.Step(17), height=alt.Step(17))
    )
    _strip = (
        alt.Chart(pl.DataFrame({"b": _names, "group": [KIND_LABELS[KIND_OF[k]] for k in _keys]})).mark_rect()
        .encode(y=alt.Y("b:N", sort=_names, axis=None, scale=alt.Scale(paddingInner=0.08)), color=alt.Color("group:N", scale=kind_scale, legend=None), tooltip=["b:N", "group:N"])
        .properties(width=8, height=alt.Step(17))
    )
    # key for the strip, under the matrix so long group names never clip
    _key = (
        alt.Chart(pl.DataFrame({"group": list(kind_scale.domain)})).mark_square(size=120, opacity=1)
        .encode(y=alt.Y("group:N", sort=list(kind_scale.domain), title=None, axis=alt.Axis(orient="right", domain=False, ticks=False, labelLimit=500)), color=alt.Color("group:N", scale=kind_scale, legend=None))
        .properties(width=12, height=alt.Step(17))
    )
    alt.vconcat(alt.hconcat(_heat, _strip, spacing=3).resolve_scale(color="independent"), _key, spacing=14, title=ttl(
        f"Strongest pair: {_top['a']} + {_top['b']}, {_top['lift']:.1f}× chance",
        "Red: together more often than chance, blue: less often; the stronger, the more solid.",
        "Chance accounts for how many ingredients a Stulle has.",
        "The strip on the right marks each ingredient's group.",
    )).resolve_scale(color="independent")
    return


@app.cell
def _(CLASSICS, df, fam):
    # island: classics
    _ex = df.select("id", "classics").explode("classics").drop_nulls()
    _fam = pl.DataFrame({"c": list(CLASSICS)}).join(_ex.group_by("classics").len("n").rename({"classics": "c"}), on="c", how="left").fill_null(0)
    _fam = _fam.with_columns((pl.col("n") / df.height).alias("share"), pl.col("c").replace_strict({k: fam(k) for k in CLASSICS}).alias("Classic"))
    _any = df["classics"].list.len().gt(0).mean()
    _f0 = _fam.sort("n", descending=True).row(0, named=True)
    alt.Chart(_fam).mark_bar(cornerRadiusEnd=4, color=BLUE, height=13).encode(
        y=alt.Y("Classic:N", sort="-x", title=None, axis=alt.Axis(labelLimit=320)),
        x=alt.X("share:Q", axis=alt.Axis(format=".1%"), title="% of all Stullen (strict match)"),
        tooltip=["Classic", alt.Tooltip("n:Q", format=",.0f", title="Stullen"), alt.Tooltip("share:Q", format=".2%")],
    ).properties(width="container", height=alt.Step(17), title=ttl(
        f"Only {_any:.0%} of Stullen are a recognisable classic",
        f"Strict recipe match: the defining toppings plus the usual garnish, nothing else. {_f0['Classic']} leads with {_f0['share']:.1%}.",
    ))
    return


@app.cell
def _(df):
    # island: heights
    _v = df.filter(pl.col("n_toppings") > 0)["height_cm"].to_numpy()
    _edges = np.unique(np.round(np.geomspace(max(_v.min(), 0.05), _v.max() * 1.01, 40), 3))
    _h, _ = np.histogram(_v, bins=_edges)
    _q = np.quantile(_v, [0.5, 0.99])
    alt.Chart(pl.DataFrame({"lo": _edges[:-1], "hi": _edges[1:], "n": _h})).mark_bar(color=BLUE, binSpacing=1).encode(
        x=alt.X("lo:Q", bin="binned", scale=alt.Scale(type="log"), title="Estimated height of the topping pile, cm (log scale)"),
        x2="hi:Q",
        y=alt.Y("n:Q", title="Stullen", axis=alt.Axis(format=",.0f")),
        tooltip=[alt.Tooltip("lo:Q", format=".2f", title="from cm"), alt.Tooltip("hi:Q", format=".2f", title="to cm"), alt.Tooltip("n:Q", format=",.0f", title="Stullen")],
    ).properties(width="container", height=240, title=ttl(
        f"Half stay under {_q[0]:.1f} cm; the top 1% pass {_q[1]:.0f} cm",
        f"Stullen per bin, log-spaced bins; {len(_v):,} sandwiches with at least one topping",
    ))
    return


@app.cell
def _(df, lab):
    # island: tallest
    _tall = df.sort("height_cm", descending=True).head(4)
    _cards = "".join(
        f'<a href="https://stulle.app/{r["id"]}" target="_blank" rel="noopener" style="flex:1 1 200px;text-decoration:none;color:inherit">'
        f'<img src="https://img.stulle.app/{r["id"]}/og.jpg" loading="lazy" alt="Stulle #{r["id"]}" style="width:100%;aspect-ratio:1.9;object-fit:cover;border-radius:8px;margin:0">'
        f'<div style="font-size:13px;margin-top:4px"><b>#{r["id"]:,}</b> · ~{r["height_cm"]:.0f} cm · {r["n_toppings"]} pieces</div>'
        f'<div style="font-size:12px;opacity:.7;height:3em;overflow:hidden">{", ".join(lab(t) for t in r["topping_types"])}</div></a>'
        for r in _tall.iter_rows(named=True)
    )
    mo.Html(f'<div style="display:flex;flex-wrap:wrap;gap:12px">{_cards}</div>')
    return


@app.cell
def _(lab, positions):
    # island: where_pick
    _types = positions.group_by("type").agg(pl.col("pieces").sum()).sort("pieces", descending=True)["type"].to_list()
    where_topping = mo.ui.dropdown(options={lab(t): t for t in _types}, value=lab("gummibaer") if "gummibaer" in _types else lab(_types[0]))
    where_topping
    return (where_topping,)


@app.cell
def _(positions, where_topping):
    # island: where_chart
    _g = positions.filter(pl.col("type") == where_topping.value)
    _n = int(_g["pieces"].sum())
    _centre = _g.filter(pl.col("x_cm") ** 2 + pl.col("y_cm") ** 2 < 2.75**2)["pieces"].sum() / max(_n, 1)
    _half = 13.2 / 40 / 2  # the pipeline's grid: 40 bins over ±6.6 cm
    _heat = alt.Chart(_g.with_columns((pl.col("x_cm") - _half).alias("x0"), (pl.col("x_cm") + _half).alias("x1"), (pl.col("y_cm") - _half).alias("y0"), (pl.col("y_cm") + _half).alias("y1"))).mark_rect().encode(
        x=alt.X("x0:Q", scale=alt.Scale(domain=[-6.6, 6.6]), title="cm from centre", axis=alt.Axis(grid=False)),
        x2="x1:Q",
        y=alt.Y("y0:Q", scale=alt.Scale(domain=[-6.6, 6.6]), title="cm from centre", axis=alt.Axis(grid=False)),
        y2="y1:Q",
        color=alt.value(BLUE),
        opacity=alt.Opacity("pieces:Q", scale=alt.Scale(type="sqrt", range=[0.08, 1]), title="Pieces", legend=alt.Legend(symbolFillColor=BLUE, symbolType="square")),
        tooltip=[alt.Tooltip("x_cm:Q", format=".1f", title="x cm"), alt.Tooltip("y_cm:Q", format=".1f", title="y cm"), alt.Tooltip("pieces:Q", format=",.0f")],
    )
    _bread = alt.Chart(pl.DataFrame({"x": [-5.5], "y": [-5.5], "x2": [5.5], "y2": [5.5]})).mark_rect(fill=None, stroke=GRAY, strokeDash=[4, 3]).encode(x="x:Q", y="y:Q", x2="x2:Q", y2="y2:Q")
    (_heat + _bread).properties(width=380, height=380, title=ttl(
        f"{where_topping.selected_key}: {_centre:.0%} of pieces land within 2.75 cm of the centre",
        f"{_n:,} pieces; dashed square = the 11 cm slice. A uniform spread would put 20% there.",
    ))
    return


@app.cell
def _(df):
    # island: timeline
    _start = pl.datetime(2026, 11, 20, 12, time_zone="Europe/Berlin")
    _d = df.filter(pl.col("created_berlin") >= _start).sort("created_berlin")
    # naive Berlin wall-clock times, so the axis reads Berlin time in every viewer's time zone
    _hourly = _d.group_by_dynamic("created_berlin", every="1h").agg(pl.len().alias("n")).with_columns(pl.col("created_berlin").dt.replace_time_zone(None))
    _cum = (
        _d.group_by_dynamic("created_berlin", every="15m", label="right").agg(pl.col("id").max().alias("total"))
        .select(pl.col("created_berlin").dt.replace_time_zone(None).alias("t"), pl.col("total").cum_max())
    )
    _peak = _hourly.sort("n").row(-1, named=True)
    _hn = pl.DataFrame({"t": ["2026-11-22T01:07:59"], "label": ["Posted to Hacker News"]}).with_columns(pl.col("t").str.to_datetime())
    _rule = alt.Chart(_hn).mark_rule(color=GRAY, strokeDash=[4, 3]).encode(x="t:T")
    _per_hour = (
        alt.Chart(_hourly).mark_area(color=BLUE, fillOpacity=0.3, line={"color": BLUE, "strokeWidth": 2}).encode(
            x=alt.X("created_berlin:T", title=None),
            y=alt.Y("n:Q", title="Stullen per hour", axis=alt.Axis(format=",.0f")),
            tooltip=[alt.Tooltip("created_berlin:T", format="%a %d %b %H:00", title="Hour (Berlin)"), alt.Tooltip("n:Q", format=",.0f", title="Stullen")],
        )
        + _rule
    ).properties(width="container", height=220, title=ttl(
        f"Peak: {_peak['n']:,} Stullen in one hour, {_peak['created_berlin']:%a %d %b %H:00}",
        "Sandwiches made per hour (still on the site), Berlin time. Dashed line: posted to Hacker News.",
    ))
    _total = (
        alt.Chart(_cum).mark_line(color=BLUE, strokeWidth=2).encode(
            x=alt.X("t:T", title=None),
            y=alt.Y("total:Q", title="Made so far", axis=alt.Axis(format=",.0f")),
            tooltip=[alt.Tooltip("t:T", format="%a %d %b %H:%M", title="Berlin time"), alt.Tooltip("total:Q", format=",.0f", title="Running total")],
        )
        + _rule
    ).properties(width="container", height=220, title=ttl(
        f"{_cum['total'][-1]:,} Stullen made by {_cum['t'][-1]:%a %d %b}",
        "Running total (the serial ID at creation time, so it also counts deleted ones), Berlin time. Dashed line: posted to Hacker News.",
    ))
    mo.ui.tabs({"Per hour": _per_hour, "Running total": _total})
    return


@app.cell
def _(KIND_OF, df):
    # island: flavour
    def _profile(tops, spr):
        kinds = {KIND_OF.get(t) for t in tops}
        sweet = "sweet" in kinds or bool(set(spr) & {"jam", "nutella"})
        savory = bool(kinds & {"savory", "veg", "junk"}) or "leberwurst" in spr
        return "Chaos (sweet + savory)" if sweet and savory else "Sweet" if sweet else "Savory" if savory else "Plain (butter/quark only)"

    _p = (
        df.with_columns(pl.struct("topping_types", "spread_types").map_elements(lambda r: _profile(r["topping_types"], r["spread_types"]), return_dtype=pl.Utf8).alias("profile"))
        .group_by("profile").len("n").with_columns((pl.col("n") / df.height).alias("share"))
    )
    _sh = dict(zip(_p["profile"], _p["share"]))
    _order = ["Savory", "Sweet", "Chaos (sweet + savory)", "Plain (butter/quark only)"]
    _b = alt.Chart(_p).encode(y=alt.Y("profile:N", sort=_order, title=None), x=alt.X("share:Q", axis=None), tooltip=["profile", alt.Tooltip("n:Q", format=",.0f"), alt.Tooltip("share:Q", format=".1%")])
    with_values(
        _b.mark_bar(cornerRadiusEnd=4, height=18).encode(color=alt.condition(alt.datum.profile == "Chaos (sweet + savory)", alt.value(MAGENTA), alt.value(GRAY))),
        _p, "profile:N", {o: f"{_sh.get(o, 0):.0%}" for o in _order},
    ).properties(width="container", height=alt.Step(26), title=ttl(
        f"{_sh.get('Chaos (sweet + savory)', 0):.0%} of Stullen mix sweet and savory",
        "Sweet = jam, Nutella, fruit or gummy bears; savory = everything else except plain butter or quark",
    ))
    return


@app.cell
def _(catalog):
    # island: hunt_pick
    hunt = mo.ui.dropdown(options=catalog["hunts"], value="☭ Hammer and sickle")  # no label: it renders dark-on-dark in the blog
    hunt
    return (hunt,)


@app.cell
def _(clip_map, hunt, hunt_hits):
    # island: hunt_grid
    _h = hunt_hits.filter(pl.col("hunt") == hunt.value, pl.col("id").is_in(clip_map["id"].implode())).sort("rank").head(24)
    photo_grid(_h["id"].to_list(), [f"· {s:.2f}" for s in _h["score"].to_list()], size=110)
    return


@app.cell
def _(catalog, clip_map):
    # island: map
    _cats = catalog["taxonomy"]
    _palette = [GRAY, BLUE, "#6b6a65", ORANGE, YELLOW, AQUA, VIOLET, MAGENTA, GREEN, RED]
    # rounded positions keep the pre-rendered island (all points inline) small
    _m = clip_map.drop_nulls("x").select("id", pl.col("x", "y").round(2), "category")
    map_chart = mo.ui.altair_chart(
        alt.Chart(_m).mark_circle(size=6, opacity=0.7).encode(
            x=alt.X("x:Q", axis=None), y=alt.Y("y:Q", axis=None),
            color=alt.Color("category:N", scale=alt.Scale(domain=_cats, range=_palette), legend=alt.Legend(orient="bottom", columns=4, title=None)),
            tooltip=["id", "category"],
        ).properties(width="container", height=420, title=ttl(
            "Similar-looking Stullen form islands",
            f"{_m.height:,} photos placed by CLIP similarity (axes have no units). Drag a box to see them.",
        )),
        chart_selection="interval",
        legend_selection=False,
    )
    map_chart
    return (map_chart,)


@app.cell
def _(map_chart):
    # island: map_pick
    _sel = map_chart.value
    if _sel is None or len(_sel) == 0 or len(_sel) > 40000:
        _out = mo.md("*Drag a box on the map to see those photos.*")
    else:
        _ids = pl.DataFrame(_sel)["id"].to_list()
        _out = mo.vstack([mo.md(f"**{len(_ids):,} photos selected** (showing up to 30)"), photo_grid(_ids[:30], size=100)])
    _out
    return


@app.cell
def _():
    # island: roll_button
    roll = mo.ui.button(label="🎲 Roll for Stulle", value=0, on_click=lambda v: v + 1, kind="success")
    roll
    return (roll,)


@app.cell
def _(df, lab, roll):
    # island: roll_result
    DICE = {
        "Bread (d4)": ["roggenmisch", "graubrot", "vollkorn", "toast"],
        "Spread (d6)": ["butter", "quark", "leberwurst", "jam", "nutella", None],
        "Main (d8)": ["kaese", "salami", "schinken", "leberkaese", "lachs", "blutwurst", "ei", "spiegelei"],
        "Roughage (d8)": ["gurke", "tomate", "radieschen", "gewuerzgurke", "senfgurke", "rotebete", None, None],
        "Wild magic (d10)": ["gummibaer", "pommes", "chips", "fischstaebchen", "banane", "erdbeere", "kiwi", "harzer", None, "DOUBLE"],
    }
    _rng = np.random.default_rng(None if roll.value else 20)
    _rolls = {t: int(_rng.integers(len(o))) for t, o in DICE.items()}
    _pick = {t: DICE[t][k] for t, k in _rolls.items()}
    _bread = _pick["Bread (d4)"]
    _target = ({_pick["Spread (d6)"], _pick["Main (d8)"], _pick["Roughage (d8)"], _pick["Wild magic (d10)"]} - {None, "DOUBLE"})

    def _face(t, k):
        v = DICE[t][k]
        return f"| {t} | **{k + 1}** | {'nothing' if v is None else ('⚡ natural 10: double main!' if v == 'DOUBLE' else lab(v))} |"

    _sims = df.select("id", "bread", pl.col("ingredients").list.unique().alias("ing")).with_columns(
        pl.col("ing").map_elements(lambda s: len(set(s) & _target) / max(len(set(s) | _target), 1), return_dtype=pl.Float64).alias("j")
    )
    _exact = _sims.filter(pl.col("j") == 1.0).height
    _near = _sims.sort(["j", pl.col("bread") == _bread], descending=True).head(8)
    mo.vstack([
        mo.md("| Table | Roll | Result |\n|---|---|---|\n" + "\n".join(_face(t, k) for t, k in _rolls.items())),
        mo.md(
            f"**Your Stulle:** {lab(_bread)} with {', '.join(lab(x) for x in sorted(_target))}. "
            + (f"🎯 **{_exact:,}** real Stullen have exactly these ingredients." if _exact else "✨ **Critical hit: nobody has built this one yet.** [Go make it](https://stulle.app) and claim it!")
        ),
        mo.md("**Closest real Stullen:**"),
        photo_grid(_near["id"].to_list(), [f"· {j:.0%}" for j in _near["j"].to_list()], size=100),
    ])
    return


if __name__ == "__main__":
    app.run()
