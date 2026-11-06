# Which Match Is Your Song? Status

**Live: <https://atharva-729.github.io/game-song-match/>** (repo: <https://github.com/atharva-729/game-song-match>)

Handoff, 2026-11-04. **Phases 0–5 and 7 pass**, with NFL and IPL data. Phase 6 (album) is on hold as asked.

## Where things stand

| Phase | Status | Key numbers |
|---|---|---|
| 0 Setup | PASS | Python 3.11 venv and FluidSynth live in `~/.venvs/ipl-song-match/`, because the `;` in this folder's path breaks venvs and DLL loading |
| 1 NFL curves | PASS | 7,273 games (1999–2025), 26/26 famous games found |
| 2 IPL model | PASS | 1,216 IPL matches, 15/15 famous found. Logistic regression ships (test log loss 0.545 vs LightGBM 0.607–0.632) |
| 3 Matcher | PASS | Self-test 20/20; index 2.48 MB for 8,489 games |
| 4 Browser analysis | PASS | Python and JS top 5 identical (curves agree to 2e-15) |
| 5 Page | PASS | 380/1280 px, light/dark, no console errors; 4-min mp3 matched in 2.3–2.5 s (plugged in) |
| 6 Album | On hold | As asked |
| 7 Publish | PASS | Live on GitHub Pages; the full browser check passes against the live URL (demos, seeking, your own songs, v1) |

Try it locally: `python scripts/serve.py`, then open <http://localhost:8000> (Ctrl+F5 after a rebuild). Don't use `python -m http.server`:
it can't serve byte ranges, so seeking in the demo audio breaks. The previous page design is at <http://localhost:8000/v1.html>.

**Page v2:** five demo pieces with real audio (credited in full), a live scoreboard that follows the playhead, and a game that
unfolds as the music plays. The demo audio overrides the README's no-audio rule, as you asked. To revert, copy `site/v1.html` over `site/index.html`.

## The IPL win-probability model
- Trained on all men's T20 matches in Cricsheet up to 2021 (1.19M balls), calibrated on IPL 2022, and tested on 281 IPL matches from 2023 on.
- **Logistic regression beat LightGBM**, so it ships, per the README rule. Its log loss is 0.545 and Brier score 0.186; a coin flip scores 0.693 and 0.25.
  LightGBM was worse than a coin flip in the first innings. It learned "this total wins" thresholds that the
  higher-scoring Impact Player era broke.
- Calibration is close to the diagonal. Every match ends on the right side. The 2017 and 2019 finals, both won by 1 run, swing wildly in the last over.
- Known limitation: the README's linear `season` feature makes 2023+ first innings too pessimistic for the team batting first
  (pre-match 0.38 vs 0.47 actual). It isn't tuned away, per "don't tune endlessly".

## Changes from the README plan (agreed in chat, or forced by the data)
- **Tension uses leverage**, meaning how much the next play or ball could move win probability, instead of `close*stake`.
  On NFL data this halves the "a 50/50 first quarter looks tense" effect (z-score +0.51 down to +0.25) and keeps close finishes high (+2.16).
- **The IPL model trains on all men's T20** with an IPL flag, plus yorkr's momentum and run rate × wickets features.
- **DLS matches** stay out of training but are kept as curves when Cricsheet gives a revised target, so the 2023 final stays in.
- **The run rate is shrunk toward 8 an over** for the first couple of overs. Raw runs/balls is 0 or 36 after one ball, which
  gave every innings a fake opening tension spike.
- **The famous-game bonus is proportional**: `dist × (1 − 0.4·famous)` instead of `dist − 0.05·famous`. With 8,489 games packed this closely,
  the flat bonus let famous games beat a game's own curve. Famous games still take about 6.5% of #1 spots for random songs, and
  carry a "★ classic" tag on the page.

## Needs you
- **To update the live site:** commit, then `bash scripts/publish.sh` (pushes `main`, then `site/` as the `gh-pages` branch).
- **Phase 6:** say when.

## Fun example matches (synthetic shapes)
The synthetic-shape matches are in `logs/match_selftest.md` (the v1 page shows them). The v2 page uses five real demo recordings instead;
for example the last two minutes of Beethoven's *Egmont* ("the comeback") land on Titans 17 @ Jets 24, 2009 Week 3.
The three long pieces (Egmont, Coriolan, Danse Macabre) are cut to their final 2 minutes by `scripts/cut_demos.py`;
the full recordings sit in `data/raw/demo_full/`.

Full logs: `logs/progress.md`. Screenshots: `logs/screens/`. One-command rebuild: `bash scripts/rebuild.sh [--nfl] [--ipl]`.
