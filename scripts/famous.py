"""Hand-tagged iconic games. Famous games get a small ranking bonus in the matcher.

NFL entries are (season, round, {team, team}, nickname). round is a week number or one of
WC/DIV/CON/SB; resolve_nfl() turns them into nflverse game_ids and fails loudly on a miss.
IPL entries are (season, {team, team}, date or None, nickname); resolve_ipl() does the same
against Cricsheet match ids.
"""
import pandas as pd

# nflverse uses current team codes for every season (LA = Rams, LV = Raiders, LAC = Chargers).
NFL = [
    (1999, "WC",  {"BUF", "TEN"}, "Music City Miracle"),
    (1999, "SB",  {"LA", "TEN"}, "Super Bowl XXXIV: The Tackle"),
    (2001, "DIV", {"LV", "NE"},  "The Tuck Rule game"),
    (2001, "SB",  {"NE", "LA"},  "Super Bowl XXXVI: Vinatieri at the gun"),
    (2003, "SB",  {"NE", "CAR"},  "Super Bowl XXXVIII"),
    (2006, "WC",  {"DAL", "SEA"}, "Romo's bobbled snap"),
    (2007, "SB",  {"NYG", "NE"},  "Super Bowl XLII: the Helmet Catch"),
    (2008, "SB",  {"PIT", "ARI"}, "Super Bowl XLIII: Holmes in the corner"),
    (2010, 15,    {"PHI", "NYG"}, "Miracle at the New Meadowlands"),
    (2011, "WC",  {"PIT", "DEN"}, "Tebow's overtime strike"),
    (2012, "DIV", {"BAL", "DEN"}, "The Mile High Miracle"),
    (2013, 12,    {"DEN", "NE"},  "Patriots erase 24-0 in the snow"),
    (2014, "CON", {"GB", "SEA"},  "Seattle's NFC title comeback"),
    (2014, "SB",  {"NE", "SEA"},  "Super Bowl XLIX: Butler's interception"),
    (2015, "DIV", {"GB", "ARI"},  "Rodgers' Hail Mary, then overtime"),
    (2016, "SB",  {"ATL", "NE"},  "Super Bowl LI: 28-3"),
    (2017, "DIV", {"NO", "MIN"},  "The Minneapolis Miracle"),
    (2018, 11,    {"KC", "LA"},   "Chiefs 51, Rams 54"),
    (2018, "CON", {"LA", "NO"},   "The no-call NFC title game"),
    (2018, "CON", {"NE", "KC"},   "AFC title in overtime"),
    (2020, 10,    {"BUF", "ARI"}, "The Hail Murray"),
    (2021, "DIV", {"BUF", "KC"},  "13 seconds"),
    (2022, "WC",  {"LAC", "JAX"}, "Jaguars erase 27-0"),
    (2022, "SB",  {"KC", "PHI"},  "Super Bowl LVII"),
    (2023, 1,     {"BUF", "NYJ"}, "Gipson's walk-off punt return"),
    (2023, "SB",  {"KC", "SF"},   "Super Bowl LVIII in overtime"),
]

# Cricsheet team names; date disambiguates when the pair met more than once that season.
IPL = [
    (2008, {"Rajasthan Royals", "Chennai Super Kings"}, "2008-06-01", "IPL 2008 Final: last-ball win"),
    (2009, {"Deccan Chargers", "Royal Challengers Bangalore"}, "2009-05-24", "IPL 2009 Final"),
    (2010, {"Chennai Super Kings", "Kings XI Punjab"}, "2010-03-21", "The first IPL super over"),
    (2012, {"Kolkata Knight Riders", "Chennai Super Kings"}, "2012-05-27", "IPL 2012 Final: 191 chased"),
    (2014, {"Kings XI Punjab", "Kolkata Knight Riders"}, "2014-06-01", "IPL 2014 Final"),
    (2014, {"Mumbai Indians", "Rajasthan Royals"}, "2014-05-25", "Mumbai chase 190 in 14.3 overs"),
    (2016, {"Sunrisers Hyderabad", "Royal Challengers Bangalore"}, "2016-05-29", "IPL 2016 Final"),
    (2017, {"Mumbai Indians", "Rising Pune Supergiant"}, "2017-05-21", "IPL 2017 Final: won by 1 run"),
    (2019, {"Mumbai Indians", "Chennai Super Kings"}, "2019-05-12", "IPL 2019 Final: won by 1 run"),
    (2020, {"Kings XI Punjab", "Mumbai Indians"}, "2020-10-18", "Two super overs"),
    (2020, {"Rajasthan Royals", "Kings XI Punjab"}, "2020-09-27", "Tewatia's five sixes"),
    (2022, {"Gujarat Titans", "Sunrisers Hyderabad"}, "2022-04-27", "Rashid's last-over finish"),
    (2023, {"Kolkata Knight Riders", "Gujarat Titans"}, "2023-04-09", "Rinku's five sixes"),
    (2023, {"Chennai Super Kings", "Gujarat Titans"}, None, "IPL 2023 Final: last-ball finish"),
    (2024, {"Royal Challengers Bengaluru", "Chennai Super Kings"}, "2024-05-18", "RCB's qualifying heist"),
]


def resolve_nfl(games):
    """games: DataFrame with game_id, season, round, home, away. Returns {game_id: nickname}."""
    out = {}
    for season, rnd, teams, name in NFL:
        g = games[(games.season == season) & (games["round"] == rnd)]
        g = g[pd.Series([{h, a} == teams for h, a in zip(g.home, g.away)], index=g.index, dtype=bool)]
        if len(g) != 1:
            raise ValueError(f"famous NFL game not found exactly once: {season} {rnd} {teams} ({len(g)})")
        out[g.game_id.iloc[0]] = name
    return out


def resolve_ipl(matches):
    """matches: DataFrame with match_id, season, date, team1, team2.
    Returns ({match_id: nickname}, [misses]). Misses are logged, not fatal."""
    out, misses = {}, []
    for season, teams, date, name in IPL:
        m = matches[matches.season == season]
        m = m[pd.Series([{a, b} == teams for a, b in zip(m.team1, m.team2)], index=m.index, dtype=bool)]
        if date:
            m = m[m.date == date]
        elif len(m) > 1:
            m = m.sort_values("date").tail(1)  # the final is the pair's last meeting
        if len(m) != 1:
            misses.append(f"{season} {sorted(teams)} {date}: {len(m)} found")
            continue
        out[m.match_id.iloc[0]] = name
    return out, misses
