"""Checkpoint 1: sanity plot of two known games + a check that leverage fixes 'early 50/50 looks tense'."""
import json, pickle
from pathlib import Path
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
games = pickle.load(open(ROOT / "data/curves/nfl.pkl", "rb"))
by_gid = {g["game_id"]: g for g in games}
picks = [("2023_01_BUF_NYJ", "close, OT: should be high late with OT spikes"),
         ("2023_11_NYJ_BUF", "blowout: should be mostly low")]

fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=False)
for ax, (gid, note) in zip(axes, picks):
    g = by_gid[gid]
    x = np.arange(g["n_steps"])
    ax.fill_between(x, 0, g["wp"], color="#2a6fdb", alpha=0.15, label="home WP")
    ax.plot(x, g["tension"], color="#e4572e", lw=1.8, label="tension (leverage + 3*swing)")
    ax.plot(x, g["tension_v1"], color="#888", lw=1, ls="--", label="tension_v1 (README close*stake)")
    for m in g["moments"]:
        ax.axvline(m["step"], color="#e4572e", alpha=0.25, lw=1)
    ax.set_title(f"{g['title']}  ({note})", fontsize=10)
    ax.set_ylim(0, max(1.3, max(g["tension"]) * 1.05))
    ax.legend(fontsize=8, loc="upper left")
axes[-1].set_xlabel("play (step)")
fig.tight_layout()
fig.savefig(ROOT / "logs/nfl_sanity.png", dpi=110)

# Does a 50/50 game score higher late than early? Compare z-scored tension at close states.
early, late, early_v1, late_v1 = [], [], [], []
for g in games:
    wp, n = np.array(g["wp"]), g["n_steps"]
    close = np.abs(wp - 0.5) < 0.1
    for key, e, l in (("tension", early, late), ("tension_v1", early_v1, late_v1)):
        t = np.array(g[key]); s = t.std()
        if s < 1e-6:
            continue
        z = (t - t.mean()) / s
        i_early = np.flatnonzero(close[: int(0.2 * n)])
        i_late = np.flatnonzero(close[int(0.9 * n):]) + int(0.9 * n)
        e.extend(z[i_early]); l.extend(z[i_late])
res = {"leverage": {"early_close_z": float(np.mean(early)), "late_close_z": float(np.mean(late))},
       "readme_v1": {"early_close_z": float(np.mean(early_v1)), "late_close_z": float(np.mean(late_v1))}}
for k in res:
    res[k]["gap"] = res[k]["late_close_z"] - res[k]["early_close_z"]
(ROOT / "logs/nfl_tension_check.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
print(json.dumps(res, indent=1))
for gid, _ in picks:
    g = by_gid[gid]; t = np.array(g["tension"]); n = g["n_steps"]
    print(gid, "mean first half", round(t[: n // 2].mean(), 3), "mean last 15%", round(t[int(.85 * n):].mean(), 3),
          "max", round(t.max(), 3))
