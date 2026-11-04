"""Phase 2b: IPL win-probability model + one tension curve per IPL match.

Two models (one per innings) predict P(batting team wins), trained on all men's T20 in Cricsheet up to
2021 (with an is_ipl flag), isotonic-calibrated on IPL 2022, tested on IPL 2023+. Splits are by season,
never by ball. A logistic-regression baseline uses the same features; whichever has the lower test log
loss ships.

Curve: one step per delivery. WP is always the team batting first's chance; innings 2 is 1 - p.
Tension = 1.2*clip(lev/p99, 0, 1) + 3*swing, EWM alpha 0.3, where lev = |WP if the next ball is a six
- WP if it is a wicket|.
usage: python scripts/build_ipl.py [raw_dir] [out_root]
"""
import json, pickle, sys
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from famous import resolve_ipl

RAW = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data/raw"
OUTR = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT
TRAIN_MAX, VAL_SEASON, TEST_MIN = 2021, 2022, 2023
CRR_PRIOR_RPB, CRR_PRIOR_BALLS = 8 / 6, 12

F1 = ["runs", "wickets", "balls_left", "crr", "season", "momentum", "perf_index", "is_ipl"]
M1 = [1, -1, 1, 0, 0, 0, 0, 0]          # more runs/balls never hurt, more wickets lost never help
F2 = ["runs_needed", "balls_left", "wickets_left", "rrr", "crr", "need_per_ball", "momentum", "perf_index", "is_ipl"]
M2 = [-1, 1, 1, -1, 0, -1, 0, 0, 0]     # more runs needed never helps
ABBR = {"Chennai Super Kings": "CSK", "Mumbai Indians": "MI", "Royal Challengers Bangalore": "RCB",
        "Royal Challengers Bengaluru": "RCB", "Kolkata Knight Riders": "KKR", "Rajasthan Royals": "RR",
        "Delhi Daredevils": "DD", "Delhi Capitals": "DC", "Kings XI Punjab": "KXIP", "Punjab Kings": "PBKS",
        "Sunrisers Hyderabad": "SRH", "Deccan Chargers": "DCH", "Gujarat Titans": "GT",
        "Lucknow Super Giants": "LSG", "Pune Warriors": "PWI", "Rising Pune Supergiant": "RPS",
        "Rising Pune Supergiants": "RPS", "Gujarat Lions": "GL", "Kochi Tuskers Kerala": "KTK"}


def features(df):
    """State after each delivery -> model features. Works on the parsed balls frame or hypothetical states."""
    bowled = df.legal.clip(lower=0)
    balls_left = (df.balls_total - df.legal).clip(lower=0)
    # run rate shrunk toward 8 an over (worth 12 balls), so ball 1 isn't 0 or 36 an over
    crr = 6 * (df.runs + CRR_PRIOR_RPB * CRR_PRIOR_BALLS) / (bowled + CRR_PRIOR_BALLS)
    out = pd.DataFrame({"runs": df.runs, "wickets": df.wickets, "balls_left": balls_left, "crr": crr,
                        "season": df.season, "momentum": df.momentum, "is_ipl": df.is_ipl.astype(int)})
    out["runs_needed"] = df.target - df.runs if "target" in df else np.nan
    out["wickets_left"] = 10 - df.wickets
    out["rrr"] = 6 * out.runs_needed / balls_left.clip(lower=1)
    out["need_per_ball"] = out.runs_needed / balls_left.clip(lower=1)
    out["perf_index"] = crr * out.wickets_left
    return out


def add_momentum(balls):
    """Runs in the last 12 deliveries of the innings (yorkr's runsMomentum, roughly)."""
    g = balls.groupby(["match_id", "innings"]).ball_runs
    balls["momentum"] = g.transform(lambda s: s.rolling(12, min_periods=1).sum())
    return balls


class WPModel:
    """Per-innings predictor with optional isotonic calibration and hard rules at innings 2 endings."""
    def __init__(self, kind, m1, m2, cal1=None, cal2=None):
        self.kind, self.m1, self.m2, self.cal1, self.cal2 = kind, m1, m2, cal1, cal2

    def p1(self, X):
        p = self.m1.predict_proba(X[F1])[:, 1]
        return self.cal1.predict(p) if self.cal1 is not None else p

    def p2(self, X):
        p = self.m2.predict_proba(X[F2])[:, 1]
        p = self.cal2.predict(p) if self.cal2 is not None else p
        rn, bl, wl = X.runs_needed.to_numpy(), X.balls_left.to_numpy(), X.wickets_left.to_numpy()
        p = np.where((bl == 0) | (wl == 0), np.where(rn == 1, 0.5, 0.0), p)  # out of balls or wickets: lost or tied
        return np.where(rn <= 0, 1.0, p)                                       # target reached


def fit_lgb(X, y, mono):
    m = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31, min_child_samples=200,
                           subsample=0.7, subsample_freq=1, monotone_constraints=mono, verbose=-1)
    return m.fit(X, y)


def fit_lr(X, y):
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(X, y)


def train(balls):
    X = features(balls)
    y = balls.batting_won.astype(int).to_numpy()
    clean = ~balls.dls.to_numpy()
    season, ipl, inn = balls.season.to_numpy(), balls.is_ipl.to_numpy(), balls.innings.to_numpy()
    tr = clean & (season <= TRAIN_MAX)
    va = clean & ipl & (season == VAL_SEASON)
    te = clean & ipl & (season >= TEST_MIN)
    print(f"rows: train {tr.sum()} (IPL {(tr & ipl).sum()}), val {va.sum()}, test {te.sum()}", flush=True)

    fitted = {}
    for name, rows in (("pooled", tr), ("ipl_only", tr & ipl)):
        ms = []
        for k, F, M in ((1, F1, M1), (2, F2, M2)):
            r = rows & (inn == k)
            ms.append(fit_lgb(X.loc[r, F], y[r], M))
        cals = []
        for k, F, m in ((1, F1, ms[0]), (2, F2, ms[1])):
            r = va & (inn == k)
            cals.append(IsotonicRegression(out_of_bounds="clip", y_min=0.001, y_max=0.999)
                        .fit(m.predict_proba(X.loc[r, F])[:, 1], y[r]))
        fitted[f"lgbm_{name}"] = WPModel("lgbm", ms[0], ms[1], cals[0], cals[1])
        print("fitted", name, flush=True)
    fitted["logreg_pooled"] = WPModel("logreg", *(fit_lr(X.loc[tr & (inn == k), F], y[tr & (inn == k)])
                                                  for k, F in ((1, F1), (2, F2))))

    metrics, preds = {}, {}
    for name, model in fitted.items():
        p = np.empty(te.sum())
        Xt, it = X[te], inn[te]
        p[it == 1] = model.p1(Xt[it == 1])
        p[it == 2] = model.p2(Xt[it == 2])
        p = np.clip(p, 1e-4, 1 - 1e-4)
        yt = y[te]
        metrics[name] = {"log_loss": log_loss(yt, p), "brier": brier_score_loss(yt, p),
                         **{f"log_loss_inn{k}": log_loss(yt[it == k], p[it == k], labels=[0, 1]) for k in (1, 2)}}
        preds[name] = p
    ship = min(("lgbm_pooled", "logreg_pooled"), key=lambda n: metrics[n]["log_loss"])
    metrics = {"test_seasons": f"IPL {TEST_MIN}+", "test_balls": int(te.sum()),
               "test_matches": int(balls[te].match_id.nunique()), "train_balls": int(tr.sum()),
               "shipped": ship, "models": metrics}
    calibration_plot(y[te], preds, ship)
    return fitted[ship], metrics


def calibration_plot(y, preds, ship):
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot([0, 1], [0, 1], color="#999", lw=1, ls="--")
    bins = np.linspace(0, 1, 11)
    for name, color in (("lgbm_pooled", "#e4572e"), ("logreg_pooled", "#2a6fdb"), ("lgbm_ipl_only", "#76b041")):
        p = preds[name]
        idx = np.clip(np.digitize(p, bins) - 1, 0, 9)
        xs = [p[idx == b].mean() for b in range(10) if (idx == b).sum() > 30]
        ys = [y[idx == b].mean() for b in range(10) if (idx == b).sum() > 30]
        ax.plot(xs, ys, "o-", color=color, label=name + (" (shipped)" if name == ship else ""))
    ax.set_xlabel("predicted P(batting team wins)"); ax.set_ylabel("observed win rate")
    ax.set_title(f"IPL {TEST_MIN}+ test balls: calibration", fontsize=10)
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(OUTR / "logs/ipl_calibration.png", dpi=110)


def first_wp(model, states):
    """P(team batting first wins) for a frame of states (mix of innings)."""
    X = features(states)
    p = np.empty(len(states))
    i1 = states.innings.to_numpy() == 1
    if i1.any():
        p[i1] = model.p1(X[i1])
    if (~i1).any():
        p[~i1] = 1 - model.p2(X[~i1])
    return p


def build_curves(model, balls, matches):
    ipl = matches[matches.is_ipl].copy()
    famous, misses = resolve_ipl(ipl)
    b = balls[balls.match_id.isin(ipl.match_id)].copy()
    b["wp"] = first_wp(model, b)

    # leverage of the NEXT ball from each state: WP(six) - WP(wicket)
    six, wkt = b.copy(), b.copy()
    six["runs"] += 6; six["legal"] += 1; six["momentum"] += 6
    wkt["wickets"] = (wkt.wickets + 1).clip(upper=10); wkt["legal"] += 1
    lev = np.abs(first_wp(model, six) - first_wp(model, wkt))
    done = ((b.innings == 2) & ((b.runs >= b.target) | (b.wickets >= 10) | (b.legal >= b.balls_total))).to_numpy()
    b["lev"] = np.where(done, 0.0, lev)
    lev_scale = float(np.quantile(b.lev, 0.99))

    curves, jumps = [], []
    for mid, g in b.groupby("match_id", sort=False):
        mt = ipl[ipl.match_id == mid].iloc[0]
        g = g.reset_index(drop=True)
        start = g.iloc[[0]].copy()
        start[["runs", "wickets", "legal", "momentum"]] = 0
        start["innings"] = 1
        start_wp, start_lev = first_wp(model, start)[0], lev_scale * 0.3
        outcome = 1.0 if mt.winner == mt.team1 else 0.0
        wp = np.concatenate([[start_wp], g.wp.to_numpy(), [outcome]])
        levs = np.concatenate([[start_lev], g.lev.to_numpy(), [0.0]])
        # innings break: last innings-1 value vs first innings-2 value
        brk = int(np.flatnonzero(g.innings.to_numpy() == 2)[0]) + 1
        jumps.append(abs(wp[brk] - wp[brk - 1]))
        curves.append(dict(mt=mt, g=g, wp=wp, lev=levs, brk=brk, famous=famous, outcome=outcome))

    mean_jump = float(np.mean(jumps))
    out = []
    for c in curves:
        wp, brk = c["wp"].copy(), c["brk"]
        if mean_jump > 0.15:  # README rule: smooth the five balls after the break
            w = np.linspace(0, 1, 7)[1:-1]
            wp[brk:brk + 5] = (1 - w) * wp[brk - 1] + w * wp[brk + 5]
        out.append(make_curve(c, wp, lev_scale))
    return out, {"mean_break_jump": mean_jump, "smoothed_break": mean_jump > 0.15, "lev_p99": lev_scale,
                 "famous_found": sum(c["famous"] for c in out), "famous_missing": misses}


def is_wicket(r):
    return isinstance(r.wicket_kind, str)  # missing kinds come back from parquet as NaN, which is truthy


def ball_label(r):
    if is_wicket(r):
        return f"WICKET – {r.wicket_kind}"
    if r.extra in ("wides", "noballs"):
        return f"{'wide' if r.extra == 'wides' else 'no-ball'}" + (f" +{r.ball_runs - 1}" if r.ball_runs > 1 else "")
    if r.batter_runs == 6:
        return "SIX"
    if r.batter_runs == 4:
        return "4"
    return f"{int(r.ball_runs)} run{'' if r.ball_runs == 1 else 's'}"


def make_curve(c, wp, lev_scale):
    mt, g, famous = c["mt"], c["g"], c["famous"]
    n = len(wp)
    swing = np.abs(np.diff(wp, prepend=wp[0]))
    lev = 1.2 * np.clip(c["lev"] / lev_scale, 0, 1)
    tension = pd.Series(lev + 3 * swing).ewm(alpha=0.3).mean().to_numpy()
    bowled = np.concatenate([[0], np.where(g.innings == 1, g.legal, g.legal + mt.sched_balls), [240]])
    close = 1 - 2 * np.abs(wp - 0.5)
    tension_v1 = pd.Series(close * (0.6 + 0.6 * np.clip(bowled / 240, 0, 1)) + 3 * swing).ewm(alpha=0.3).mean().to_numpy()

    moments = []
    for i in np.argsort(-swing[1:n - 1])[:5] + 1:  # skip the pre-match step and the appended outcome
        r = g.iloc[int(i) - 1]
        moments.append({"step": int(i), "clock": f"{'1st' if r.innings == 1 else '2nd'} inns {r.over}.{r.ball}",
                        "swing": round(float(swing[i]), 4),
                        "desc": f"{r.bowler} to {r.batter}: {ball_label(r)} ({r.runs}/{r.wickets})"[:140]})
    moments.sort(key=lambda m: m["step"])
    events = [[int(i) + 1, "W" if is_wicket(r) else str(int(r.batter_runs))]
              for i, r in enumerate(g.itertuples()) if is_wicket(r) or r.batter_runs in (4, 6)]

    # scoreboard per step: pre-match 0/0, then the state after each delivery, then the final (same as the last ball)
    col = lambda name, first: [first] + g[name].astype(int).tolist() + [int(g[name].iloc[-1])]
    sb = {"i": col("innings", 1), "r": col("runs", 0), "w": col("wickets", 0), "b": col("legal", 0)}
    i2 = g[g.innings == 2]
    chase = {"target": int(i2.target.iloc[0]), "balls": int(i2.balls_total.iloc[0])} if len(i2) else None

    a1, a2 = ABBR.get(mt.team1, mt.team1), ABBR.get(mt.team2, mt.team2)
    stage = mt.stage if isinstance(mt.stage, str) else (f"Match {int(mt.match_number)}" if pd.notna(mt.match_number) else mt.date)
    win_abbr, lose_abbr = (a1, a2) if mt.winner == mt.team1 else (a2, a1)
    win_score, lose_score = (mt.inn1, mt.inn2) if mt.winner == mt.team1 else (mt.inn2, mt.inn1)
    how = " (super over)" if mt.super_over else (" (DLS)" if mt.dls else "")
    return {"id": f"ipl_{mt.match_id}", "match_id": mt.match_id, "league": "IPL", "season": int(mt.season),
            "date": mt.date, "home": a1, "away": a2, "home_name": mt.team1, "away_name": mt.team2,
            "final": f"{win_abbr} {win_score} beat {lose_abbr} {lose_score} {mt.margin}{how}".strip(),
            "title": f"{a1} vs {a2} · IPL {mt.season} {stage}", "famous": mt.match_id in famous,
            "nickname": famous.get(mt.match_id), "n_steps": n, "break_step": c["brk"],
            "wp": wp.round(4).tolist(), "tension": tension.round(4).tolist(),
            "tension_v1": tension_v1.round(4).tolist(), "moments": moments, "scoring": events,
            "dls": bool(mt.dls), "super_over": bool(mt.super_over), "sb": sb, "chase": chase}


def sanity(curves):
    picks = [c for c in curves if c["nickname"] and ("Final" in c["nickname"] and ("1 run" in c["nickname"] or "last-ball" in c["nickname"]))][:3]
    if not picks:
        picks = sorted(curves, key=lambda c: -max(np.abs(np.diff(c["wp"][-12:-1]))))[:2]
    fig, axes = plt.subplots(len(picks), 1, figsize=(11, 3.4 * len(picks)), squeeze=False)
    for ax, c in zip(axes[:, 0], picks):
        x = np.arange(c["n_steps"])
        ax.fill_between(x, 0, c["wp"], color="#2a6fdb", alpha=0.18, label=f"P({c['home']} wins), batting first")
        ax.plot(x, c["tension"], color="#e4572e", lw=1.6, label="tension")
        ax.axvline(c["break_step"], color="#555", lw=1, ls=":", label="innings break")
        ax.set_title(f"{c['title']}: {c['final']}", fontsize=10); ax.legend(fontsize=8, loc="upper left")
        ax.set_ylim(0, 1.3)
    fig.tight_layout(); fig.savefig(OUTR / "logs/ipl_sanity.png", dpi=110)
    return [c["title"] for c in picks]


def main():
    balls = add_momentum(pd.read_parquet(RAW / "t20_balls.parquet"))
    matches = pd.read_parquet(RAW / "t20_matches.parquet")
    model, metrics = train(balls)
    (OUTR / "models").mkdir(exist_ok=True)
    with open(OUTR / "models/ipl_wp.pkl", "wb") as f:  # plain dict so it loads without this module
        pickle.dump({**vars(model), "features_inn1": F1, "features_inn2": F2}, f)
    curves, info = build_curves(model, balls, matches)
    metrics["curves"] = {"ipl_matches": len(curves), **info}
    # checkpoint: every curve ends at its outcome, and the last real ball is on the right side
    wrong_side = [c["title"] for c in curves if not c["dls"] and not c["super_over"]
                  and abs(c["wp"][-2] - c["wp"][-1]) > 0.5]
    metrics["curves"]["last_ball_wrong_side"] = wrong_side
    metrics["curves"]["sanity_plot"] = sanity(curves)
    (OUTR / "logs/ipl_metrics.json").write_text(json.dumps(metrics, indent=1, default=str), encoding="utf-8")
    with open(OUTR / "data/curves/ipl.pkl", "wb") as f:
        pickle.dump(curves, f)
    print(json.dumps(metrics, indent=1, default=str))


if __name__ == "__main__":
    main()
