"""Phase 3: search index + per-game files for the page.

site/index.bin   every match's 200-point normalised tension curve, int8 (x32, clipped to +-127)
site/index.json  {n, scale, fields, rows}: one row per match, row i lives at byte offset i*200
site/games/{id}.json  WP (<=400 points), moments, scoring events, title, final; loaded lazily
usage: python scripts/build_index.py
"""
import json, pickle, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from match import N, QSCALE, normalise, quantise

SITE = ROOT / "site"
FIELDS = ["id", "league", "title", "famous", "offset", "nickname", "season"]


def load_curves():
    games = []
    for lg in ("nfl", "ipl"):
        p = ROOT / f"data/curves/{lg}.pkl"
        if p.exists():
            games += pickle.load(open(p, "rb"))
    return games


def downsample(x, m=400):
    x = np.asarray(x, dtype=float)
    if len(x) <= m:
        return x
    return x[np.round(np.linspace(0, len(x) - 1, m)).astype(int)]


def main():
    games = load_curves()
    rows, curves, flat = [], [], 0
    # overwrite files in place and prune stale ones afterwards, so a running local server never sees an
    # empty games/ (renaming a freshly written folder is often denied on Windows while it's being scanned)
    gdir = SITE / "games"
    gdir.mkdir(parents=True, exist_ok=True)
    written = set()
    for g in games:
        z = normalise(g["tension"])
        if z is None:
            flat += 1
            continue
        rows.append([g["id"], g["league"], g["title"], int(g["famous"]), len(curves) * N, g.get("nickname"), g["season"]])
        curves.append(quantise(z))
        doc = {k: g.get(k) for k in ("id", "league", "title", "final", "nickname", "season", "home", "away",
                                     "home_name", "away_name", "n_steps", "moments", "scoring", "break_step")}
        doc["wp"] = [round(float(v), 3) for v in downsample(g["wp"])]
        # scoreboard arrays picked at the same steps as wp, so the page can index both with one position
        doc["sb"] = {k: [int(v) for v in downsample(arr)] for k, arr in (g.get("sb") or {}).items()}
        doc["chase"] = g.get("chase")
        (gdir / f"{g['id']}.json").write_text(json.dumps(doc, separators=(",", ":")), encoding="utf-8")
        written.add(f"{g['id']}.json")
    for f in gdir.glob("*.json"):
        if f.name not in written:
            f.unlink()
    Q = np.stack(curves)
    (SITE / "index.bin").write_bytes(Q.tobytes())
    (SITE / "index.json").write_text(json.dumps({"n": N, "scale": QSCALE, "fields": FIELDS, "rows": rows},
                                                separators=(",", ":")), encoding="utf-8")
    # IPL model metrics for the page (empty until Phase 2 has run)
    mp = ROOT / "logs/ipl_metrics.json"
    m = json.loads(mp.read_text(encoding="utf-8")) if mp.exists() else {}
    keep = {k: m[k] for k in ("test_seasons", "test_matches", "shipped", "models") if k in m}
    (SITE / "metrics.json").write_text(json.dumps(keep, separators=(",", ":")), encoding="utf-8")
    size = (SITE / "index.bin").stat().st_size + (SITE / "index.json").stat().st_size
    counts = {lg: sum(r[1] == lg for r in rows) for lg in ("NFL", "IPL")}
    print(json.dumps({"matches": len(rows), **counts, "flat_unmatchable": flat, "index_bytes": size}))


if __name__ == "__main__":
    main()
