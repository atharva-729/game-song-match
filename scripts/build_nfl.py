"""Phase 1: one tension curve per NFL game, 1999-2025.

Tension = leverage + 3*swing, smoothed (EWM alpha 0.3).
  leverage: how much the next play is expected to move win probability, E|dWP| given the game
            state, learned with a small LightGBM model over every play. A 50/50 first quarter
            scores low, a 50/50 final drive scores high. (The README's close*stake formula is
            kept as tension_v1 for comparison.)
  swing:    |wp_t - wp_{t-1}|, what actually happened.
usage: python scripts/build_nfl.py
"""
import json, pickle, re, sys
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from famous import resolve_nfl

RAW = ROOT / "data/raw/nfl"
OUT = ROOT / "data/curves/nfl.pkl"
COLS = ["game_id", "play_id", "season", "week", "season_type", "home_team", "away_team",
        "home_score", "away_score", "total_home_score", "total_away_score", "home_wp",
        "play_type", "game_seconds_remaining", "qtr", "time", "desc", "down", "ydstogo",
        "yardline_100", "sp"]
NAMES = {"ARI": "Cardinals", "ATL": "Falcons", "BAL": "Ravens", "BUF": "Bills", "CAR": "Panthers",
         "CHI": "Bears", "CIN": "Bengals", "CLE": "Browns", "DAL": "Cowboys", "DEN": "Broncos",
         "DET": "Lions", "GB": "Packers", "HOU": "Texans", "IND": "Colts", "JAX": "Jaguars",
         "KC": "Chiefs", "LA": "Rams", "STL": "Rams", "LAC": "Chargers", "SD": "Chargers",
         "LV": "Raiders", "OAK": "Raiders", "MIA": "Dolphins", "MIN": "Vikings", "NE": "Patriots",
         "NO": "Saints", "NYG": "Giants", "NYJ": "Jets", "PHI": "Eagles", "PIT": "Steelers",
         "SEA": "Seahawks", "SF": "49ers", "TB": "Buccaneers", "TEN": "Titans", "WAS": "Washington"}
ROUND_LABEL = {"WC": "Wild Card", "DIV": "Divisional", "CON": "Conference Championship", "SB": "Super Bowl"}
LEV_FEATS = ["dist", "game_seconds_remaining", "qtr", "down", "ydstogo", "yardline_100", "abs_margin"]


def clean_desc(d, limit=140):
    """nflverse play text, minus the noise: clock prefix, formation tags, eligibility notes,
    snap/hold credits, tackler lists and jersey numbers."""
    d = re.sub(r"^\(\d*:\d+\)\s*", "", str(d))
    d = re.sub(r"\((Shotgun|No Huddle|No Huddle, Shotgun|[\w ]*formation)\)\s*", "", d)
    d = re.sub(r"\d+-[A-Z][\w.'-]* reported in as eligible( receiver)?\.?\s*", "", d)
    d = re.sub(r",?\s*Center-\d+-[\w.'-]+,\s*Holder-\d+-[\w.'-]+\.?", ".", d)
    d = re.sub(r",\s*Center-\d+-[\w'-]+(\.[\w'-]+)*", "", d)
    d = re.sub(r"\s*\((\d+-[A-Z][\w.'-]*[;,]?\s*)+\)", "", d)
    d = re.sub(r"\b\d+-([A-Z][\w.'-]*)", r"\1", d)
    d = re.sub(r"\s+", " ", d).strip()
    return d if len(d) <= limit else d[: limit - 1].rsplit(" ", 1)[0] + "…"


def team_name(abbr, season):
    if abbr == "WAS" and season >= 2022:
        return "Commanders"
    return NAMES.get(abbr, abbr)


def load():
    frames = []
    for f in sorted(RAW.glob("play_by_play_*.parquet")):
        d = pd.read_parquet(f, columns=COLS)
        frames.append(d[d.home_wp.notna() & d.play_type.notna()])
    pbp = pd.concat(frames, ignore_index=True).sort_values(["game_id", "play_id"], kind="stable")
    # playoff rounds: the season's distinct postseason weeks, in order
    rounds = {}
    for season, weeks in pbp[pbp.season_type == "POST"].groupby("season").week.unique().items():
        for w, r in zip(sorted(weeks), ["WC", "DIV", "CON", "SB"]):
            rounds[(season, w)] = r
    pbp["round"] = [rounds.get((s, w), w) for s, w in zip(pbp.season, pbp.week)]
    return pbp


def fit_leverage(pbp):
    """E|dWP| for the next play given the game state, home-team agnostic."""
    nxt = pbp.groupby("game_id").home_wp.shift(-1)
    # the last play of a game moves WP to the final outcome
    last = nxt.isna()
    outcome = np.where(pbp.home_score > pbp.away_score, 1.0, np.where(pbp.home_score < pbp.away_score, 0.0, 0.5))
    nxt = nxt.where(~last, outcome)
    X = pd.DataFrame({
        "dist": (pbp.home_wp - 0.5).abs(),
        "game_seconds_remaining": pbp.game_seconds_remaining,
        "qtr": pbp.qtr.clip(upper=5),
        "down": pbp.down,
        "ydstogo": pbp.ydstogo,
        "yardline_100": pbp.yardline_100,
        "abs_margin": (pbp.total_home_score - pbp.total_away_score).abs(),
    })
    y = (nxt - pbp.home_wp).abs()
    model = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=63,
                              min_child_samples=500, subsample=0.5, subsample_freq=1, verbose=-1)
    model.fit(X, y)
    lev = np.clip(model.predict(X), 0, None)
    return lev, model


def build_game(g, lev_scale, famous):
    h, a = g.home_team.iloc[0], g.away_team.iloc[0]
    hs, as_ = int(g.home_score.iloc[0]), int(g.away_score.iloc[0])
    season, rnd = int(g.season.iloc[0]), g["round"].iloc[0]
    outcome = 1.0 if hs > as_ else (0.0 if hs < as_ else 0.5)
    wp = np.append(g.home_wp.to_numpy(), outcome)
    n = len(wp)
    swing = np.abs(np.diff(wp, prepend=wp[0]))

    lev = np.append(g.lev.to_numpy(), 0.0)
    lev = 1.2 * np.clip(lev / lev_scale, 0, 1)
    tension = pd.Series(lev + 3 * swing).ewm(alpha=0.3).mean().to_numpy()

    secs = np.append(g.game_seconds_remaining.fillna(0).to_numpy(), 0)
    ot = np.append(g.qtr.to_numpy() >= 5, g.qtr.iloc[-1] >= 5)
    stake = np.where(ot, 1.2, 0.6 + 0.6 * (1 - np.clip(secs, 0, 3600) / 3600))
    close = 1 - 2 * np.abs(wp - 0.5)
    tension_v1 = pd.Series(close * stake + 3 * swing).ewm(alpha=0.3).mean().to_numpy()

    desc, qtr, clock = g["desc"].to_numpy(), g.qtr.to_numpy(), g.time.to_numpy()
    # home_wp is the WP *before* each play, so the swing into step i is the result of play i-1
    # (and the appended outcome step is the result of the final play).
    moments = []
    for i in np.argsort(-swing[1:])[:5] + 1:
        i, p = int(i), int(i) - 1
        q = "OT" if qtr[p] >= 5 else f"Q{int(qtr[p])}"
        moments.append({"step": i, "clock": f"{q} {clock[p]}", "swing": round(float(swing[i]), 4),
                        "desc": clean_desc(desc[p])})
    moments.sort(key=lambda m: m["step"])
    sp = np.flatnonzero(g.sp.fillna(0).to_numpy() == 1)
    scoring = [[int(i) + 1, int(g.total_home_score.iloc[i]), int(g.total_away_score.iloc[i])] for i in sp]

    # scoreboard per step: step i is the state after play i-1 (total_*_score is the score after each play);
    # the clock is the time at the snap of play i, and the appended outcome step is the final (q = 0)
    secs_left = [int(m) * 60 + int(s) for m, s in (str(t).split(":") if ":" in str(t) else ("0", "0") for t in clock)]
    sb = {"h": [0] + g.total_home_score.fillna(0).astype(int).tolist()[:-1] + [hs],
          "a": [0] + g.total_away_score.fillna(0).astype(int).tolist()[:-1] + [as_],
          "q": [int(q) for q in qtr] + [0], "c": secs_left + [0]}

    is_ot = bool(g.qtr.max() >= 5)
    when = f"{season} {ROUND_LABEL[rnd]}" if isinstance(rnd, str) else f"{season} Week {int(rnd)}"
    title = f"{team_name(a, season)} {as_} @ {team_name(h, season)} {hs}{' (OT)' if is_ot else ''} · {when}"
    gid = g.game_id.iloc[0]
    return {"id": f"nfl_{gid}", "game_id": gid, "league": "NFL", "season": season,
            "week": rnd if isinstance(rnd, str) else int(rnd), "home": h, "away": a,
            "home_name": team_name(h, season), "away_name": team_name(a, season),
            "final": f"{team_name(a, season)} {as_}, {team_name(h, season)} {hs}{' (OT)' if is_ot else ''}",
            "title": title, "famous": gid in famous, "nickname": famous.get(gid),
            "n_steps": n, "wp": wp.round(4).tolist(), "tension": tension.round(4).tolist(),
            "tension_v1": tension_v1.round(4).tolist(), "moments": moments, "scoring": scoring, "sb": sb}


def main():
    pbp = load()
    print("plays", len(pbp), "games", pbp.game_id.nunique(), flush=True)
    pbp["lev"], model = fit_leverage(pbp)
    lev_scale = float(np.quantile(pbp.lev, 0.99))
    print("leverage model fitted; p99 =", round(lev_scale, 4), flush=True)

    games = pbp.groupby("game_id").agg(season=("season", "first"), round=("round", "first"),
                                       home=("home_team", "first"), away=("away_team", "first")).reset_index()
    famous = resolve_nfl(games)
    out, short = [], 0
    for gid, g in pbp.groupby("game_id", sort=False):
        if len(g) + 1 < 100:
            short += 1
            continue
        out.append(build_game(g.reset_index(drop=True), lev_scale, famous))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "wb") as f:
        pickle.dump(out, f)
    per_season = pd.Series([g["season"] for g in out]).value_counts().sort_index()
    (ROOT / "logs/nfl_counts.json").write_text(json.dumps({
        "games": len(out), "dropped_under_100_steps": short, "famous_found": len(famous),
        "leverage_p99": lev_scale, "per_season": {int(k): int(v) for k, v in per_season.items()}}, indent=1), encoding="utf-8")
    print("games written", len(out), "dropped", short, "famous", len(famous))


if __name__ == "__main__":
    main()
