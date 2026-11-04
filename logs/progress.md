# Progress log

## Agreed changes to the README plan (2026-11-03)
- Python 3.11 venv at `~/.venvs/ipl-song-match` (the project path contains `;`, which venv refuses).
- FluidSynth 2.6.2 + FluidR3_GM.sf2 live in `~/.venvs/ipl-song-match/tools/` for the same reason (`;` breaks DLL loading).
- Tension uses **leverage** instead of `close*stake`: how much the next play/ball is expected to move WP.
  NFL: E|dWP| from a small LightGBM model over all plays. IPL: WP(next ball is a six) - WP(next ball is a wicket).
  `tension = 1.2*clip(lev/p99, 0, 1) + 3*swing`, EWM alpha 0.3. The README formula is kept as `tension_v1` for comparison.
- IPL WP model trains on **all men's T20** in Cricsheet (<=2021) with an `is_ipl` flag, is calibrated on IPL 2022, tested on IPL 2023+.
  Extra features borrowed from yorkr: `momentum` (runs in the last 12 legal balls) and `perf_index` (run rate x wickets in hand).
- DLS matches are excluded from training, but kept as curves when Cricsheet gives a revised target (keeps the 2023 final).

## Phase 0: Setup — PASS
- Imports OK (pandas 3.1.0, lightgbm 4.7.0, librosa, playwright, ...). `fluidsynth --version` -> 2.6.2. `engine/` has both compose scripts.
- NFL play-by-play 1999-2025 downloaded (27 files, 466 MB).
- **Blocker:** cricsheet.org serves a JavaScript bot-check page to scripted downloads, so the Cricsheet zip could not be fetched.
  It needs a manual browser download (see SUMMARY.md). No workaround attempted.

## Phase 1: NFL tension curves — PASS
- 7,273 games written (0 dropped for <100 steps), 1,236,524 plays. Per-season counts in `logs/nfl_counts.json` (258-285 per season).
- 26/26 famous games resolved (nflverse uses current team codes for all seasons: LA = Rams, LV = Raiders).
- Leverage model: LightGBM E|dWP|, p99 = 0.095 WP per play.
- `logs/nfl_sanity.png`: BUF@NYJ 2023 W1 rises late with OT spikes; NYJ@BUF 2023 W11 decays to ~0. Looks right.
- Leverage check (`logs/nfl_tension_check.json`, z-scored tension at |wp-0.5|<0.1): early-game close states
  +0.25 sd (README formula: +0.51), late close states +2.16 sd (README: +2.03). Leverage halves the "dull 50/50 looks tense" effect.

## Phase 2: IPL WP model + curves — PASS
- Data: `all_json.zip` downloaded by hand from cricsheet.org (bot check blocks scripts). Parsed 10,239 men's T20 matches
  (2.39M deliveries); 1,216 IPL matches written as curves (2008-2026). Train 1.19M balls (all men's T20 <= 2021), calibrate on
  IPL 2022, test on 281 IPL 2023+ matches (67,880 balls).
- Test log loss / Brier (lower is better; coin flip = 0.693 / 0.25):
  | model | log loss | Brier | inn 1 | inn 2 |
  |---|---|---|---|---|
  | **logistic regression, pooled (shipped)** | **0.545** | **0.186** | 0.646 | 0.437 |
  | LightGBM, IPL only | 0.607 | 0.197 | 0.745 | 0.458 |
  | LightGBM, pooled | 0.632 | 0.195 | 0.805 | 0.446 |
  Per the README rule, LightGBM did not beat logistic regression, so **logistic regression ships**. LightGBM's innings-1
  loss is worse than a coin flip: it learned sharp "this total wins" thresholds from pre-2023 cricket, which the Impact
  Player era (2023+, much higher totals) broke. The smoother linear model degrades gracefully. Pooling all T20 helped innings 2 (0.446 vs 0.458) but not innings 1.
- `logs/ipl_calibration.png`: the shipped model hugs the diagonal (slightly under-confident above 0.7).
- `logs/ipl_sanity.png`: 2017 and 2019 finals (won by 1 run) swing hard in the final over; the 2023 final (DLS, last-ball four) too.
- Every match ends at its correct outcome (`last_ball_wrong_side` empty). Mean innings-break jump 0.075 (< 0.15, no smoothing).
- 15/15 famous IPL matches found.
- Fixes on the real data:
  - The current run rate is shrunk toward 8 an over (weighted as 12 balls). Raw `6*runs/balls` is 0 or 36 after one ball,
    which gave every innings a fake opening tension spike. The spike fell from about 0.70 to 0.57, and the log loss improved slightly (0.548 to 0.545).
  - Cricsheet's missing wicket kinds come back from parquet as NaN, which is truthy, so every ball was labelled
    "WICKET – nan" and drawn as a wicket dot. Fixed with an explicit string check.
- Known limitations (not tuned, per "don't tune endlessly"):
  - The README's linear `season` feature extends the pre-2023 "chasing gets easier" trend. Before 2023 the pre-match P(batting first wins)
    is 0.43 vs 0.45 actual, but in 2023+ it is 0.38 vs 0.47 actual.
  - Logistic regression has no wickets × balls-left interaction, so first-innings leverage is fairly flat. A small
    opening bump remains, and IPL first innings read as calmer than second innings (which is roughly how fans feel them anyway).

## Phase 3: Matching engine — PASS (NFL only for now)
- Index: 7,273 matches, 2.14 MB (index.bin + index.json). IPL will add ~0.3 MB.
- Batch DTW == single-pair DTW bit for bit; path stays inside the +-10 band. Python search over 7,273: ~1.6 s.
- Self-test (`logs/match_selftest.md`, `logs/match_selftest.png`): test 4 **20/20**. Slow build -> late OT games (peak at 0.92-0.98);
  Explosive start -> early drama then blowout (peak at 0.0-0.08); Rollercoaster -> mid swing + late peak. Curves overlay closely.
- Match % = round(100*exp(-dist/0.438)); median random song = 80%.

## Phase 4: Song analysis in the browser — parity PASS
- `site/core.js` (inline radix-2 FFT, energy, banded DTW) vs `scripts/song_energy.py` + `scripts/match.py` on the test tone:
  energy curves differ by <= 1.8e-15, **top 5 identical** with identical distances (`logs/parity.md`).
- Node timing: energy 0.84 s for 90 s of audio, search 0.89 s over 7,273 matches. Browser timing below (Phase 5 check).

## Phase 5: The page — PASS (NFL only until IPL data lands)
- `site/index.html` + `site/core.js`: hero + file input/drag-drop, three example shapes, NFL/IPL/Both scope, WP chart (shaded by side,
  scoring dots, moment flags), song<->game DTW alignment panel, playhead through both, seek/±10 s/click-to-seek, moments on the song's
  clock with pop-up cards, runners-up with sparklines, best NFL/IPL, share line + copy, how-it-works, metrics, credits.
- `scripts/check_page.py` (Playwright): 380 and 1280 px x light/dark screenshots in `logs/screens/`, no horizontal overflow,
  **no console errors**. Test tone via the file input: result renders, 5 moments, audio advanced 2.4 s in 2.5 s, 4 runners-up.
  4-minute mp3: **matched in 3.7-3.9 s** (Checkpoint 4 target: < 5 s). Browser #1 for the test tone == Python #1.
- Fixed along the way: nflverse `home_wp` is pre-play, so moments/scoring dots now credit play i-1 for the swing into step i
  (the walk-off play is now eligible: e.g. Gipson's OT punt-return TD, Diggs' Minneapolis Miracle). Play text cleaned of
  clock prefixes, formation tags, eligibility notes, snap/hold credits, tackler lists and jersey numbers.
- One-command rebuild: `bash scripts/rebuild.sh [--nfl] [--ipl]`.

## Phases 3-5 re-run with IPL included — PASS
- Index: 8,489 matches (7,273 NFL + 1,216 IPL), 2.48 MB (target < 3 MB).
- **Famous bonus changed** from the README's `dist - 0.05*famous` to `dist * (1 - 0.4*famous)`. With 8,489 games, nearest neighbours
  sit 0.02-0.03 apart, so a flat 0.05 let famous games beat a game's own curve (self-test fell to 18/20). The proportional
  bonus can never beat an exact match, and over 200 random songs famous games (0.5% of the index) still take 6.5% of #1 spots
  (the flat bonus gave 9%). Famous runners-up now show a "★ classic" tag so their slightly lower match % isn't confusing.
- Self-test 4: **20/20**. Parity: **top 5 identical**. Page checks: PASS, no console errors, no overflow, IPL scope works.
- Timing: the 4-min mp3 matched in 4.3-10.1 s across three runs in headless Chrome, **but the laptop was on battery (34%, CPU throttled)**. The same
  unchanged energy code ran 1.8x slower than earlier (1.52 s vs 0.84 s on the same audio). Plugged in, I expect about 4.4 s
  (3.8 s × 1.17 for the extra games). Re-time when plugged in: `python scripts/check_page.py`.
- Page fixes: a missing game file now says so instead of "Couldn't read that file"; `build_index.py` overwrites game files
  in place and prunes stale ones, so a running local server never sees an empty `site/games/`.

## Page v2 (2026-11-04, asked for in chat). The previous page is kept as `site/v1.html`.
- **Demo templates with real audio.** This overrides the README's "no song audio" rule, at the user's request: five free classical
  recordings from classicals.de, re-encoded to 128 kbps / 44.1 kHz (27 MB in total) in `site/demo/`, each credited on the page.
  - Rollercoaster: Saint-Saëns, Danse Macabre (Kevin MacLeod, CC BY 4.0)
  - Boring, then brilliance: Grieg, Mountain King (Czech National SO / Musopen, public domain)
  - Explosive start, fizzles out: Beethoven, Coriolan (same orchestra, public domain)
  - The comeback: Beethoven, Egmont (same orchestra, public domain)
  - Intense all the way: Wagner, Valkyries (Philip Milman, CC BY 3.0)
  - The Rossini file was only the strings-arranged Finale. It measured loud all the way through, not a rollercoaster, so it was dropped and Danse Macabre took its slot.
    (It is also CC BY-NC.)
  - The Grieg download had no licence file. Its tags name the Musopen Kickstarter recording, which is released as public domain.
  - `scripts/build_demos.py` precomputes each curve with the same energy code as the browser, plus the matches, into `site/demo/demos.json`.
- **Live scoreboard** (a "jumbotron") follows the playhead. NFL: score, quarter and clock. IPL: runs/wickets, overs, target, and "need X off Y".
  Per-step `sb` arrays were added to the NFL/IPL curves and downsampled with the same indices as `wp`. `site/games/` is now 51 MB.
- **Excitement pass:** floodlight sweeps in the hero, a WP chart that "unfolds" during playback (the future is ghosted), a glowing playhead,
  a big "TOUCHDOWN!" / "SIX!" flash on the scoreboard when each moment hits, score bumps, and a count-up on the match %. All of it respects reduced-motion.
- **`scripts/serve.py`:** a local server with HTTP Range support. `python -m http.server` can't serve byte ranges, so browsers can't seek
  in the demo mp3s. GitHub Pages supports ranges natively.
- **Checks:** self-test 20/20, parity top 5 identical, and page checks PASS. All five demos play and the scoreboard updates; seeking to
  62% of Egmont shows Chiefs 30-33 Rams in Q3 (2:24 left). `v1.html` loads with no errors. Timing (plugged in, idle): the 4-min mp3 matched in 2.3-2.5 s.

## Phase 7: published (2026-11-04) — PASS
- Repo <https://github.com/atharva-729/game-song-match> (the name Atharva chose; the README said `which-match-is-your-song`). No `gh`, so
  it was pushed with plain git: `main` holds everything, and `site/` goes out as the `gh-pages` branch, which made GitHub turn Pages on by
  itself. Live at <https://atharva-729.github.io/game-song-match/> about 60 s after the push. `scripts/publish.sh` repeats this.
- Not committed: `data/raw`, `data/curves`, test tones and any song audio (`.gitignore`). Only the five credited demo mp3s are in. A scan of the
  staged files found no emails, tokens or local paths. `.gitattributes` keeps `index.bin` and the mp3s binary; `site/.nojekyll` skips Jekyll.
- Share text now ends with the page link when served from a real host (not localhost).
- `check_page.py <url>` runs the full browser check against the live site and adds a seek check (jump to 60% of a demo; this needs HTTP
  Range). Live: no console errors, all five demos play with the scoreboard, seeking works, both test files match with 5 moments, the share
  text carries the link, and v1.html loads. Matching took 2.1 s / 3.7 s (tone / 4-min mp3) at 1280 px, and 5.0 s / 8.7 s on the phone-width run.
  Both runs start from a cold cache, so the gap is likely this laptop's load during the run, not the page; worth a re-time on a real phone.

## Phase 6: on hold (as asked).
