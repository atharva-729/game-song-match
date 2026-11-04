"""Checkpoint 3: synthetic songs + 'a game's own curve finds itself' + fit the match-% curve.
Writes logs/match_selftest.md, logs/match_selftest.png, and adds the match-% curve to site/index.json.
"""
import json, math, sys, time
from pathlib import Path
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import pickle
from match import N, normalise, dequantise, search

SITE = ROOT / "site"
idx = json.loads((SITE / "index.json").read_text(encoding="utf-8"))
meta = [dict(zip(idx["fields"], r)) for r in idx["rows"]]
Q = np.frombuffer((SITE / "index.bin").read_bytes(), dtype=np.int8).reshape(-1, N)
t = np.linspace(0, 1, 1000)

SHAPES = {
    "Slow build": np.where(t < 0.9, t ** 3, 0.729 - 2 * (t - 0.9)),
    "Explosive start": np.exp(-4 * t) + 0.05,
    "Rollercoaster": np.exp(-((t - 0.35) / 0.07) ** 2) + np.exp(-((t - 0.85) / 0.06) ** 2) + 0.1,
}


def peak_profile(i):
    z = dequantise(Q[i])
    thirds = [round(float(z[a:b].mean()), 2) for a, b in ((0, 67), (67, 134), (134, 200))]
    return thirds, round(int(np.argmax(z)) / (N - 1), 2)


lines = ["# Matcher self-test", "", f"Index: {len(meta)} matches "
         f"({sum(m['league'] == 'NFL' for m in meta)} NFL, {sum(m['league'] == 'IPL' for m in meta)} IPL).", ""]
fig, axes = plt.subplots(len(SHAPES), 1, figsize=(10, 2.6 * len(SHAPES)))
examples = {}
lines += ["## Synthetic songs", "", "Mean z-tension of the top match per third of the game, and where its peak sits (0 = start, 1 = end).", ""]
for ax, (name, song) in zip(axes, SHAPES.items()):
    t0 = time.time()
    r = search(song, Q, meta)
    dt = time.time() - t0
    lines.append(f"**{name}** ({dt:.1f} s)")
    for rank, (i, score, dist) in enumerate(r["top"][:3], 1):
        thirds, pk = peak_profile(i)
        lines.append(f"{rank}. {meta[i]['title']}{' ★' if meta[i]['famous'] else ''}: dist {dist:.3f}, thirds {thirds}, peak at {pk}")
    for lg in ("nfl", "ipl"):
        b = r[f"best_{lg}"]
        if b:
            lines.append(f"   best {lg.upper()}: {meta[b[0]]['title']} (dist {b[2]:.3f})")
    lines.append("")
    i = r["top"][0][0]
    examples[name] = {"song": [round(v, 4) for v in normalise(song)], "top": [meta[k]["id"] for k, _, _ in r["top"]],
                      "best_nfl": r["best_nfl"] and meta[r["best_nfl"][0]]["id"],
                      "best_ipl": r["best_ipl"] and meta[r["best_ipl"][0]]["id"]}
    ax.plot(normalise(song), color="#e4572e", lw=2, label=f"song: {name}")
    ax.plot(dequantise(Q[i]), color="#2a6fdb", lw=1.2, label=f"#1 {meta[i]['title']}")
    ax.legend(fontsize=8, loc="upper left"); ax.set_yticks([])
fig.tight_layout(); fig.savefig(ROOT / "logs/match_selftest.png", dpi=110)

# test 4: a real game's own (unquantised) tension curve must come back as #1
games = {}
for lg in ("nfl", "ipl"):
    p = ROOT / f"data/curves/{lg}.pkl"
    if p.exists():
        games.update({g["id"]: g for g in pickle.load(open(p, "rb"))})
rng = np.random.default_rng(20261103)
pick = rng.choice(len(meta), 20, replace=False)
passed, fails = 0, []
for i in pick:
    r = search(games[meta[i]["id"]]["tension"], Q, meta)
    if r["top"][0][0] == i:
        passed += 1
    else:
        fails.append(f"{meta[i]['title']} -> {meta[r['top'][0][0]]['title']}")
lines += ["## Test 4: a game's own curve returns that game as #1", "", f"**{passed}/20 passed.**"] + \
         [f"- FAIL: {f}" for f in fails] + [""]

# match-% curve: best distances for random smooth 'songs'; median maps to 80%
best = []
for s in range(120):
    k = rng.integers(1, 5)
    song = sum(rng.uniform(0.3, 1) * np.exp(-((t - rng.uniform(0, 1)) / rng.uniform(0.04, 0.3)) ** 2) for _ in range(k))
    song = song + rng.uniform(-0.5, 0.5) * t
    r = search(song, Q, meta)
    if r:
        best.append(r["top"][0][2])
best = np.array(best)
d50 = float(np.median(best))
s_ = d50 / math.log(100 / 80)
pct = {"formula": "round(100*exp(-dist/s))", "s": round(s_, 5), "best_dist_p10_p50_p90":
       [round(float(np.quantile(best, q)), 4) for q in (0.1, 0.5, 0.9)]}
lines += ["## Match % curve", "", f"Best-match distance over 120 random smooth songs: p10/p50/p90 = {pct['best_dist_p10_p50_p90']}.",
          f"match % = round(100 * exp(-dist / {pct['s']})), so the median song shows 80%.", ""]
idx["pct"] = pct
idx["examples"] = examples
(SITE / "index.json").write_text(json.dumps(idx, separators=(",", ":")), encoding="utf-8")
(ROOT / "logs/match_selftest.md").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
