"""Phase 2a: Cricsheet JSON zips -> one row per delivery (men's T20 only) + one row per match.

Reads every *.zip in data/raw/cricsheet (all_json.zip, or per-league zips; duplicates are dropped).
Keeps matches with a winner. Ties settled by a super over keep the two main innings and take the
super-over winner as the outcome. DLS/VJD matches are flagged (dls=True): they never enter training,
and are kept for curves only when innings 2 carries a revised target.
usage: python scripts/parse_cricsheet.py [raw_dir] [out_dir]
"""
import json, sys, zipfile
from collections import Counter
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data/raw/cricsheet"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "data/raw"
DLS = {"D/L", "DLS", "VJD"}


def iter_files():
    seen = set()
    for zp in sorted(RAW.glob("*.zip")):
        if not zipfile.is_zipfile(zp):
            print("skipping", zp.name, "(not a zip, e.g. a saved bot-check page)", flush=True)
            continue
        with zipfile.ZipFile(zp) as z:
            for name in z.namelist():
                mid = Path(name).stem
                if name.endswith(".json") and mid not in seen:
                    seen.add(mid)
                    yield mid, json.loads(z.read(name))


def overs_to_balls(o):
    """Cricsheet overs like 20, 15.0 or 16.2 (16 overs and 2 balls)."""
    whole = int(o)
    return whole * 6 + round((float(o) - whole) * 10)


def parse_match(mid, m, skip):
    info = m["info"]
    if info.get("match_type") not in ("T20", "IT20") or info.get("gender") != "male":
        return skip("not men's T20")
    if info.get("balls_per_over", 6) != 6:
        return skip("not 6-ball overs")
    oc = info.get("outcome", {})
    method = oc.get("method")
    winner = oc.get("winner") or oc.get("eliminator")
    if not winner:
        return skip(f"no winner ({oc.get('result', 'unknown')})")
    super_over = "eliminator" in oc or method == "Super Over"
    dls = method in DLS
    if method and not (dls or super_over):
        return skip(f"method {method}")
    inns = [i for i in m.get("innings", []) if not i.get("super_over")]
    if len(inns) != 2 or any(i.get("forfeited") or i.get("absent_hurt") for i in inns):
        return skip("not two main innings")
    if dls and "target" not in inns[1]:
        return skip("DLS without revised target")

    date = info["dates"][0]
    event = info.get("event", {})
    is_ipl = event.get("name") == "Indian Premier League"
    sched = overs_to_balls(info.get("overs", 20))
    team1, team2 = inns[0]["team"], inns[1]["team"]

    rows, totals = [], []
    for k, inn in enumerate(inns, start=1):
        if k == 1:
            balls_total, target = sched, None
        else:
            t = inn.get("target")
            target = t["runs"] if t else totals[0][0] + 1
            balls_total = overs_to_balls(t["overs"]) if t and t.get("overs") else sched
        runs = wkts = legal = 0
        for ov in inn.get("overs", []):
            in_over = 0
            for d in ov["deliveries"]:
                ex = d.get("extras", {})
                is_legal = not ("wides" in ex or "noballs" in ex)
                legal += is_legal
                in_over += is_legal
                r = d["runs"]["total"]
                runs += r
                w = d.get("wickets", [])
                wkts += len(w)
                rows.append({
                    "match_id": mid, "innings": k, "batting_team": inn["team"],
                    "over": ov["over"], "ball": max(in_over, 1), "runs": runs, "wickets": min(wkts, 10),
                    "legal": legal, "balls_total": balls_total, "target": target, "ball_runs": r,
                    "batter_runs": d["runs"]["batter"], "extra": next(iter(ex), None),
                    "wicket_kind": w[0]["kind"] if w else None, "player_out": w[0]["player_out"] if w else None,
                    "batter": d["batter"], "bowler": d["bowler"], "batting_won": inn["team"] == winner})
        totals.append((runs, wkts, legal))
        if not inn.get("overs"):
            return skip("empty innings")

    # incomplete without a method (e.g. abandoned mid-match but a result recorded): drop
    r1, w1, l1 = totals[0]
    r2, w2, l2 = totals[1]
    target = rows[-1]["target"]
    bt2 = rows[-1]["balls_total"]
    if not dls:
        if l1 < sched and w1 < 10:
            return skip("innings 1 incomplete")
        if r2 < target and w2 < 10 and l2 < bt2 and not super_over:
            return skip("innings 2 incomplete")

    by = oc.get("by", {})
    margin = (f"by {by['runs']} run{'s' * (by['runs'] != 1)}" if "runs" in by else
              f"by {by['wickets']} wicket{'s' * (by['wickets'] != 1)}" if "wickets" in by else "")
    match = {"match_id": mid, "date": date, "season": int(date[:4]), "event": event.get("name", ""),
             "is_ipl": is_ipl, "stage": event.get("stage"), "match_number": event.get("match_number"),
             "team1": team1, "team2": team2, "winner": winner, "dls": dls, "super_over": super_over,
             "margin": margin, "inn1": f"{r1}/{w1}", "inn2": f"{r2}/{w2}", "target": target,
             "balls_total2": bt2, "sched_balls": sched}
    for r in rows:
        r["season"], r["is_ipl"], r["dls"] = match["season"], is_ipl, dls
    return match, rows


def main():
    skipped = Counter()
    def skip(reason):
        skipped[reason] += 1
        return None
    matches, balls = [], []
    for mid, m in iter_files():
        res = parse_match(mid, m, skip)
        if res:
            matches.append(res[0]); balls.extend(res[1])
    mt, bl = pd.DataFrame(matches), pd.DataFrame(balls)
    OUT.mkdir(parents=True, exist_ok=True)
    mt.to_parquet(OUT / "t20_matches.parquet")
    bl.to_parquet(OUT / "t20_balls.parquet")
    summary = {"matches": len(mt), "balls": len(bl), "ipl_matches": int(mt.is_ipl.sum()),
               "ipl_dls_kept": int((mt.is_ipl & mt.dls).sum()), "skipped": dict(skipped.most_common())}
    print(json.dumps(summary, indent=1))
    return summary


if __name__ == "__main__":
    main()
