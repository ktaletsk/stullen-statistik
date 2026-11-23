"""Shared data model for the stulle.app analysis: catalog, loading, stack height."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import polars as pl

BREAD_CM = 11.0  # the app scales topping sizes by cm / 11 -> bread spans [-0.5, 0.5]

# Topping physics, copied from the app bundle: footprint size (cm) and slice thickness (cm).
TOPPINGS = {
    "kaese": dict(name="Käse", cm=8.8, thick=0.25, kind="savory"),
    "salami": dict(name="Salami", cm=4.5, thick=0.25, kind="savory"),
    "schinken": dict(name="Schinken", cm=8.5, thick=0.15, kind="savory"),
    "gurke": dict(name="Gurke", cm=4.5, thick=0.5, kind="veg"),
    "tomate": dict(name="Tomate", cm=6.5, thick=0.5, kind="veg"),
    "ei": dict(name="Ei", cm=4.5, thick=0.45, kind="savory"),
    "radieschen": dict(name="Radieschen", cm=2.8, thick=0.25, kind="veg"),
    "gewuerzgurke": dict(name="Gewürzgurke", cm=6.0, thick=0.5, kind="veg"),
    "leberkaese": dict(name="Leberkäse", cm=9.0, thick=0.8, kind="savory"),
    "harzer": dict(name="Harzer", cm=4.5, thick=1.0, kind="savory"),
    "blutwurst": dict(name="Blutwurst", cm=4.3, thick=0.4, kind="savory"),
    "spiegelei": dict(name="Spiegelei", cm=9.5, thick=0.1, kind="savory"),
    "rotebete": dict(name="Rote Bete", cm=5.5, thick=0.4, kind="veg"),
    "senfgurke": dict(name="Senfgurke", cm=5.5, thick=0.25, kind="veg"),
    "lachs": dict(name="Lachs", cm=9.0, thick=0.12, kind="savory"),
    "bismarckhering": dict(name="Bismarckhering", cm=10.0, thick=0.6, kind="savory"),
    "schnittlauch": dict(name="Schnittlauch", cm=5.0, thick=0.2, kind="veg"),
    "banane": dict(name="Banane", cm=3.2, thick=0.4, kind="sweet"),
    "erdbeere": dict(name="Erdbeere", cm=2.8, thick=0.4, kind="sweet"),
    "kiwi": dict(name="Kiwi", cm=4.7, thick=0.4, kind="sweet"),
    "fischstaebchen": dict(name="Fischstäbchen", cm=9.0, thick=1.2, kind="junk", w=2.5),
    "pommes": dict(name="Pommes", cm=5.7, thick=0.72, kind="junk", w=0.9),
    "chips": dict(name="Chips", cm=5.0, thick=0.8, kind="junk"),
    "marshmallow": dict(name="Marshmallow", cm=3.0, thick=2.4, kind="sweet"),
    "schokolinse": dict(name="Schokolinse", cm=1.2, thick=0.5, kind="sweet"),
    "gummibaer": dict(name="Gummibär", cm=1.1, thick=0.75, kind="sweet", w=0.8),
}
SPREADS = {
    "butter": "Butter",
    "quark": "Quark",
    "leberwurst": "Leberwurst",
    "jam": "Marmelade",
    "nutella": "Nutella",
}
BREADS = {"roggenmisch": "Roggenmisch", "graubrot": "Graubrot", "vollkorn": "Vollkorn", "toast": "Toast"}


# English names, for the bilingual "Deutsch (English)" labels
EN = {
    "kaese": "cheese", "salami": "salami", "schinken": "ham", "gurke": "cucumber", "tomate": "tomato",
    "ei": "egg", "radieschen": "radish", "gewuerzgurke": "pickle", "leberkaese": "Bavarian meatloaf",
    "harzer": "Harz sour-milk cheese", "blutwurst": "blood sausage", "spiegelei": "fried egg",
    "rotebete": "beetroot", "senfgurke": "mustard pickle", "lachs": "salmon",
    "bismarckhering": "pickled herring", "schnittlauch": "chives", "banane": "banana",
    "erdbeere": "strawberry", "kiwi": "kiwi", "fischstaebchen": "fish sticks", "pommes": "fries",
    "chips": "crisps", "marshmallow": "marshmallow", "schokolinse": "chocolate drops",
    "gummibaer": "gummy bear",
    "butter": "butter", "quark": "quark", "leberwurst": "liver sausage", "jam": "jam", "nutella": "Nutella",
    "roggenmisch": "rye-wheat", "graubrot": "light rye", "vollkorn": "wholegrain", "toast": "toast",
}
FAMOUS_EN = {
    "Butterbrot": "bread & butter", "Käsebrot": "cheese sandwich", "Schinkenbrot": "ham sandwich",
    "Salamibrot": "salami sandwich", "Strammer Max": "ham & fried egg", "Leberwurst + Gewürzgurke": "liver sausage & pickle",
    "Quark + Radieschen": "quark & radish", "Eierbrot": "egg sandwich", "Tomatenbrot": "tomato sandwich",
    "Lachsbrot": "salmon sandwich", "Fischbrot (Bismarck)": "herring sandwich", "Harzer-Brot": "Harz cheese sandwich",
    "Leberkäsebrot": "meatloaf sandwich", "Blutwurstbrot": "blood sausage sandwich", "Nutellabrot": "Nutella sandwich",
    "Marmeladenbrot": "jam sandwich", "Erdbeerbrot": "strawberry sandwich", "Bananenbrot": "banana sandwich (not the cake)",
    "Fischstäbchenbrot": "fish stick sandwich", "Pommes-Stulle": "chip butty", "Chips-Stulle": "crisp sandwich",
}
KIND_DE = {"spread": "Aufstrich", "savory": "Wurst, Käse, Ei, Fisch", "veg": "Gemüse", "junk": "Pommes, Chips, Fischstäbchen", "sweet": "Süßes & Obst"}
KIND_EN = {"spread": "spreads", "savory": "meat, cheese, egg, fish", "veg": "vegetables", "junk": "fries, crisps, fish sticks", "sweet": "sweets & fruit"}
KINDS = list(KIND_DE)  # display order of ingredient groups


def label(key: str) -> str:
    if key in TOPPINGS:
        return TOPPINGS[key]["name"]
    return SPREADS.get(key) or BREADS.get(key) or key


def load_raw(path: str | Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass  # half-written last line while the scraper runs
    # de-duplicate by id (a resumed scrape could append twice)
    return list({r["id"]: r for r in rows}.values())


# --- stack height ---------------------------------------------------------------
GRID = 96
_ax = (np.arange(GRID) + 0.5) / GRID - 0.5
_GX, _GY = np.meshgrid(_ax, _ax)


def _footprint(t: dict) -> np.ndarray:
    spec = TOPPINGS.get(t["type"])
    if spec is None:
        return np.zeros_like(_GX, dtype=bool)
    dx, dy = _GX - t.get("x", 0.0), _GY - t.get("y", 0.0)
    L = spec["cm"] / BREAD_CM / 2
    if "w" in spec:  # elongated piece: rotated rectangle
        c, s = math.cos(t.get("rot", 0.0)), math.sin(t.get("rot", 0.0))
        u, v = dx * c + dy * s, -dx * s + dy * c
        W = spec["w"] / BREAD_CM / 2
        m = (np.abs(u) <= L) & (np.abs(v) <= W)
    else:
        m = dx * dx + dy * dy <= L * L
    if not m.any():  # tiny piece smaller than a cell: hit the nearest cell
        i = int(np.clip((t.get("y", 0) + 0.5) * GRID, 0, GRID - 1))
        j = int(np.clip((t.get("x", 0) + 0.5) * GRID, 0, GRID - 1))
        m[i, j] = True
    return m


def stack_height(toppings: list[dict]) -> tuple[float, float]:
    """Rigid-stack approximation of the topping pile.

    Each piece lands on the highest point under its footprint and raises that whole
    footprint to (landing height + its thickness). Returns (max height cm, mean height
    over covered area cm). Overestimates for floppy items (Schinken, Lachs drape),
    which is fine for comparing sandwiches with each other.
    """
    h = np.zeros_like(_GX)
    for t in toppings:
        m = _footprint(t)
        h[m] = h[m].max() + TOPPINGS.get(t["type"], {"thick": 0})["thick"]
    covered = h > 0
    return float(h.max()), float(h[covered].mean()) if covered.any() else 0.0


def coverage_and_packed(toppings: list[dict]) -> tuple[float, float]:
    """(fraction of the bread square covered, packed height cm).

    Packed height = total topping volume / covered area: the pile height if every
    piece were squashed flat with no air gaps. A lower-ish bound that tracks the
    photos better than the rigid stack for loose piles (Pommes, Chips).
    """
    cov = np.zeros_like(_GX, dtype=bool)
    vol = 0.0  # in grid-cell units * cm
    for t in toppings:
        m = _footprint(t)
        cov |= m
        vol += m.sum() * TOPPINGS.get(t["type"], {"thick": 0})["thick"]
    n = cov.sum()
    return float(cov.mean()), float(vol / n) if n else 0.0


# --- tables -----------------------------------------------------------------------
def build_tables(rows: list[dict]):
    """Return (stullen, toppings_long, spreads_long) polars DataFrames."""
    s_rows, t_rows, sp_rows = [], [], []
    for r in rows:
        tops = r.get("toppings") or []
        spreads = [o["spread"] for o in r.get("ops", []) if "spread" in o]
        strokes = sum(o.get("stroke", 0) for o in r.get("ops", []) if "stroke" in o)
        hmax, _ = stack_height(tops)
        cov, packed = coverage_and_packed(tops)
        s_rows.append(
            dict(
                id=r["id"],
                created_at=r.get("createdAt"),
                timezone=r.get("timezone"),
                bread=r.get("bread"),
                parent_id=r.get("parentId"),
                reply_to=r.get("replyTo"),
                n_toppings=len(tops),
                n_topping_types=len({t["type"] for t in tops}),
                spread_types=sorted(set(spreads)),
                topping_types=sorted({t["type"] for t in tops}),
                n_spread_passes=len(spreads),
                brush_segments=strokes,
                rigid_height_cm=hmax,
                packed_height_cm=packed,
                height_cm=math.sqrt(hmax * packed),
                coverage=cov,
            )
        )
        for k, t in enumerate(tops):
            t_rows.append(dict(id=r["id"], order=k, type=t["type"], x=t.get("x"), y=t.get("y"), rot=t.get("rot")))
        for k, sp in enumerate(spreads):
            sp_rows.append(dict(id=r["id"], order=k, spread=sp))
    stullen = (
        pl.DataFrame(s_rows)
        .with_columns(pl.col("created_at").str.to_datetime(time_zone="UTC"))
        .sort("id")
    )
    toppings = pl.DataFrame(t_rows, schema=dict(id=pl.Int64, order=pl.Int64, type=pl.Utf8, x=pl.Float64, y=pl.Float64, rot=pl.Float64))
    spreads = pl.DataFrame(sp_rows, schema=dict(id=pl.Int64, order=pl.Int64, spread=pl.Utf8))
    return stullen, toppings, spreads


# --- famous sandwiches ----------------------------------------------------------------
# need: toppings that must be present; spreads: at least one of these spreads (None = any/no
# requirement); allow: extra topping types that don't break the classic (garnish).
FAMOUS = {
    "Butterbrot": dict(need=set(), spreads={"butter"}, allow=set(), only_spreads={"butter"}),
    "Käsebrot": dict(need={"kaese"}, spreads=None, allow={"gurke", "tomate", "radieschen", "schnittlauch"}),
    "Schinkenbrot": dict(need={"schinken"}, spreads=None, allow={"gurke", "gewuerzgurke", "tomate"}),
    "Salamibrot": dict(need={"salami"}, spreads=None, allow={"gurke", "gewuerzgurke"}),
    "Strammer Max": dict(need={"schinken", "spiegelei"}, spreads=None, allow={"gewuerzgurke", "gurke", "tomate", "schnittlauch"}),
    "Leberwurst + Gewürzgurke": dict(need={"gewuerzgurke"}, spreads={"leberwurst"}, allow={"gurke", "radieschen", "senfgurke"}),
    "Quark + Radieschen": dict(need={"radieschen"}, spreads={"quark"}, allow={"schnittlauch", "gurke", "tomate"}),
    "Eierbrot": dict(need={"ei"}, spreads=None, allow={"schnittlauch", "tomate", "gurke", "radieschen"}),
    "Tomatenbrot": dict(need={"tomate"}, spreads=None, allow={"schnittlauch"}),
    "Lachsbrot": dict(need={"lachs"}, spreads=None, allow={"gurke", "ei", "schnittlauch", "radieschen"}),
    "Fischbrot (Bismarck)": dict(need={"bismarckhering"}, spreads=None, allow={"gurke", "gewuerzgurke"}),
    "Harzer-Brot": dict(need={"harzer"}, spreads=None, allow={"radieschen", "schnittlauch"}),
    "Leberkäsebrot": dict(need={"leberkaese"}, spreads=None, allow={"gewuerzgurke", "gurke", "spiegelei"}),
    "Blutwurstbrot": dict(need={"blutwurst"}, spreads=None, allow={"gewuerzgurke", "gurke"}),
    "Nutellabrot": dict(need=set(), spreads={"nutella"}, allow={"banane"}, only_spreads={"nutella", "butter"}),
    "Marmeladenbrot": dict(need=set(), spreads={"jam"}, allow=set(), only_spreads={"jam", "butter"}),
    "Erdbeerbrot": dict(need={"erdbeere"}, spreads=None, allow={"banane"}),
    "Bananenbrot": dict(need={"banane"}, spreads=None, allow=set()),
    "Fischstäbchenbrot": dict(need={"fischstaebchen"}, spreads=None, allow={"gurke", "tomate"}),
    "Pommes-Stulle": dict(need={"pommes"}, spreads=None, allow=set()),
    "Chips-Stulle": dict(need={"chips"}, spreads=None, allow=set()),
}


def classify(topping_types: list[str], spread_types: list[str]) -> list[str]:
    """Strict match: the classic's toppings plus at most its usual garnish, nothing else."""
    tops, sp = set(topping_types), set(spread_types)
    hits = []
    for name, rule in FAMOUS.items():
        if not rule["need"] <= tops:
            continue
        if tops - rule["need"] - rule["allow"]:
            continue
        if rule.get("spreads") and not (rule["spreads"] & sp):
            continue
        if "only_spreads" in rule and not sp <= rule["only_spreads"]:
            continue
        if not rule["need"] and not rule["allow"] and tops:
            continue
        hits.append(name)
    return hits


def load_tables(path: str | Path = "data/stullen.jsonl", cache_dir: str | Path = "data/cache"):
    """build_tables() with a parquet cache keyed on the input file size."""
    path, cache_dir = Path(path), Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = f"{path.stat().st_size}"
    files = {n: cache_dir / f"{n}_{key}.parquet" for n in ("stullen", "toppings", "spreads")}
    if all(f.exists() for f in files.values()):
        return tuple(pl.read_parquet(f) for f in files.values())
    for old in cache_dir.glob("*.parquet"):
        old.unlink()
    tables = build_tables(load_raw(path))
    for f, df in zip(files.values(), tables):
        df.write_parquet(f)
    return tables
