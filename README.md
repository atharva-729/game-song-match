# Which Match Is Your Song?

**Drop in a song. We find the NFL game or IPL match that *felt* exactly like it.**

Every song has a shape: it goes quiet, builds, explodes, drops. Every match has one too: it can be dull, then tense, then a collapse, then a comeback. We turn both into a line over time and search about 7,000 matches for the one whose tension rises and falls most like the song. Then we play the song with the matched game's win-probability chart underneath and a playhead running through both, so the drop lands on the interception.

> **To Claude Code:** work through every phase in order. Each checkpoint is a self-check: verify it, append the result to `logs/progress.md`, and continue. Only stop if a checkpoint fails twice after a real fix attempt. Then write `SUMMARY.md` explaining what failed, and end. Where this file gives a rule, follow it rather than improvising.

---

## The core ideas (read before building)

| Thing | What it is | How we get it |
|---|---|---|
| **Win probability (WP)** | After each play or ball, the chance team A wins | NFL: shipped in nflverse data. IPL: **we train a model** (Phase 2) |
| **Tension curve** | How much is at stake, moment by moment | Computed from WP: close and swinging is high, decided is low |
| **Energy curve** | How intense the song is, moment by moment | Computed from the audio: loudness plus how busy it is |
| **Match** | The game whose tension curve is the same *shape* as the song's energy curve | Resample both to 200 points, normalise, compare with banded DTW |

**Time axis:** one play (NFL) or one ball (IPL) is one step. This is the same convention the music engine in `engine/` uses. Don't use the game clock, which stops and skips.

**Copyright:** songs are never hosted or uploaded. The visitor picks a file on their own device, and the page analyses it in the browser. The repo and the published page must contain **no song audio**.

---

## Phase 0: Setup

**Do:**
- Create the repo layout:
  ```
  data/raw/        # downloads (gitignored)
  data/curves/     # per-match curves (gitignored)
  site/            # the published page: index.html, index.bin/json, games/*.json
  scripts/         # pipeline scripts
  engine/          # compose_vivaldi.py, compose_zimmer.py (already provided, reuse as is)
  models/          # IPL WP model
  logs/
  ```
- Install the dependencies:
  - `pip install pandas pyarrow numpy scipy scikit-learn lightgbm matplotlib mido soundfile librosa`
  - Install `fluidsynth` and the `FluidR3_GM.sf2` soundfont (apt `fluidsynth fluid-soundfont-gm`, or brew `fluid-synth` plus a downloaded FluidR3_GM).

**Don't:** don't add a backend, a database or a build tool. The final site is static files.

**✅ Checkpoint 0:** imports work, `fluidsynth --version` runs, and `engine/` has both compose scripts.

---

## Phase 1: NFL tension curves

**Do:**
- Download play-by-play for **1999–2025**: `https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{YEAR}.parquet`
- For each game:
  1. Keep rows with `home_wp` not null and `play_type` not null.
  2. `wp = home_wp`. Append the final outcome (1 or 0, or 0.5 for a tie) as the last step.
  3. **Tension per step:**
     ```
     close  = 1 - 2*|wp - 0.5|                       # 1 = coin flip, 0 = decided
     stake  = 0.6 + 0.6 * (1 - seconds_remaining/3600) # later = bigger stakes (OT counts as 1.2)
     swing  = |wp_t - wp_{t-1}|
     tension = close*stake + 3*swing                 # then smooth: EWM alpha 0.3
     ```
  4. Save per game: `id, league="NFL", season, week, home, away, final score, n_steps, wp[], tension[]`, plus `moments`: the 5 biggest |swing| steps with play `desc` (truncated to 140 characters), the clock (e.g. "Q4 1:56") and the step index.
  5. Add `title` like "Bills 16 @ Jets 22 (OT) · 2023 Week 1" and a `famous` flag (see below).
- Hand-tag a `famous` list of about 25 iconic games by `game_id`. Look them up in the data by teams and season, e.g. Super Bowl LI (ATL–NE, 28–3 comeback), the Minneapolis Miracle (NO@MIN, 2017 divisional), "13 seconds" (BUF@KC, 2021 divisional), the Helmet Catch (Super Bowl XLII), and Jets 2023 Week 1 vs BUF. Store the list in `scripts/famous.py`. Famous games get a small ranking bonus (Phase 3).

**Don't:**
- Don't drop overtime.
- Don't use `wp` from the possession team's perspective. Always use the home team.
- Don't keep games with fewer than 100 steps (data glitches).

**✅ Checkpoint 1:**
- About 7,000 NFL games written. Log the count per season.
- Plot the tension curves for 2023 Week 1 BUF@NYJ (should be high late, with spikes in OT) and 2023 Week 11 NYJ@BUF (a blowout, mostly low) to `logs/nfl_sanity.png`. Check visually that they look like that.

---

## Phase 2: IPL win-probability model + tension curves

There's no free per-ball IPL win probability, so we build one. This is the ML part of the project; do it properly but keep it simple.

**Do:**
- Download Cricsheet IPL ball-by-ball JSON: `https://cricsheet.org/downloads/ipl_json.zip`
- Parse every delivery into rows: `match_id, season, innings, over, ball, batting_team, runs_total, wickets, legal_balls_bowled, target (innings 2)`, and the match winner.
- **Exclude:** no-results, matches decided by DLS/rain (`outcome.method` present), and abandoned or incomplete matches. For tied matches settled by a super over, use the super-over winner as the outcome but model only the two main innings.
- Build **two models** (one per innings), each predicting P(batting team wins):
  - **Innings 1 features:** runs, wickets, balls_left, current run rate, season (as a number).
  - **Innings 2 features:** runs_needed, balls_left, wickets_left, required_rate, current_rate, runs_needed / balls_left (guard against divide-by-zero).
  - **Model:** LightGBM with monotone constraints that make sense: more wickets lost never raises your chance; more runs needed never raises it. Then calibrate with isotonic regression on a validation split.
  - **Split by season:** train on ≤2021, validate on 2022, test on 2023+. Never split by random ball, because balls from the same match would leak across splits.
- **Report** on the test seasons: log loss, Brier score, and a calibration plot (`logs/ipl_calibration.png`). Compare against a plain logistic-regression baseline with the same features and log both.
- **Join the innings.** Innings 1 WP is P(team batting first wins). In innings 2, convert to the same team's perspective (`1 - p`) so each match has one continuous WP line. At the innings break, the innings-1 model's final value should flow smoothly into the innings-2 model's first value. If the jump is bigger than 0.15 on average, log it and smooth the five balls after the break.
- Build tension curves the same way as the NFL. For `stake`, use `0.6 + 0.6*(balls_bowled_in_match / 240)`.
- Moments: the 5 biggest swings, each with over.ball, batter/bowler and what happened ("SIX", "WICKET – bowled", "4"), built from Cricsheet fields.
- Titles look like "CSK vs MI · IPL 2019 Final". The `famous` list includes about 15 IPL classics, e.g. 2019 final MI–CSK (won by 1 run) and the 2023 final CSK–GT (last-ball finish).
- Save the model to `models/ipl_wp.pkl` and the metrics to `logs/ipl_metrics.json`.

**Don't:**
- Don't tune endlessly. Two models with sensible features is enough. If LightGBM doesn't beat logistic regression on log loss, ship logistic regression and say so.
- Don't use any future information as a feature (e.g. the final score of innings 1 while still in innings 1).

**✅ Checkpoint 2:**
- About 1,000+ IPL matches written.
- Test log loss is below the baseline, or the reason is documented.
- The calibration plot hugs the diagonal.
- In the last-ball-finish sanity plot (`logs/ipl_sanity.png`), WP swings wildly in the final over.
- Every match ends at its correct outcome.

---

## Phase 3: The matching engine (Python reference)

**Do:**
- Write `scripts/match.py` with the reference matcher. It must match the JS version exactly (Phase 4).
- **Normalise a curve:** resample linearly to **N = 200** points, then z-score it (subtract the mean, divide by the std; if std < 1e-6, the curve is flat, so mark it unmatchable).
- **Distance:** DTW with a Sakoe–Chiba band of **±10 points** (5%), using squared difference and dividing by path length. The band is the key rule: without it DTW can bend any song into any game.
- **Score** = `distance − 0.05 * famous` (a small nudge toward games people know). Return the top 5 overall, plus the best NFL match and the best IPL match separately.
- **Build the search index** in `site/index.bin`: every match's 200-point normalised tension curve, quantised to int8 (scale ×32, clip to ±127). Add `site/index.json` with each match's `id`, `league`, `title`, `famous` and the byte offset. Target total size is under about 3 MB.
- Write `site/games/{id}.json` for each match: `wp[]` downsampled to at most 400 points, `moments[]`, `title`, `final`. Load these lazily for the result view only.
- **Self-test with synthetic "songs"** (no real audio needed):
  1. A slow build to a late peak should match games decided late.
  2. Loud from the start and then fading should match early-drama-then-decided games.
  3. Two peaks should match a mid-game swing plus a late swing.
  4. The tension curve of a real game, fed in as a "song", must return **that same game as #1**.

  Save the results to `logs/match_selftest.md`, with the top 3 titles for each.

**Don't:**
- Don't use plain Euclidean distance as the main metric (any small timing offset ruins it), and don't use unbanded DTW (everything matches everything).
- Don't put full per-play data into the index. The index is curves only.

**✅ Checkpoint 3:** self-test 4 passes for 20 random games out of 20, the index is under about 3 MB, and the synthetic songs return plausible games (look at their curves).

---

## Phase 4: Song analysis in the browser

**Do:**
- In the page's JS, use the Web Audio `decodeAudioData` API to read the visitor's file (mp3/m4a/wav/ogg). Mix to mono.
- **Energy per frame** (frames of 2048 samples, hop 1024):
  ```
  loud  = RMS in dB, clipped to [-60, 0], mapped to 0..1
  flux  = spectral flux (sum of positive magnitude changes), normalised by its 95th percentile
  energy = 0.6*loud + 0.4*flux
  ```
  Use a small radix-2 FFT written inline (no library).
- Smooth with a moving average over about 3 s, skip silence at the start and end (energy below 5% of max), then resample to 200 points and z-score.
- Port the banded DTW and scoring to JS. Search the whole index in a Web Worker created from a `blob:` URL, so the page doesn't freeze. Target: under 3 s for 7,000 matches on a laptop.
- **Parity test:** put the energy curve of one test WAV (generated by `scripts/make_test_tone.py`: a synthetic sine/noise piece with a known build, not a real song) through both Python and JS. The top-5 results must be identical. Log it.

**Don't:**
- Don't upload the file anywhere or `fetch()` it to any server.
- Don't use a JS audio-analysis library. The above is about 100 lines.
- Don't store the song in localStorage.

**✅ Checkpoint 4:** the Python and JS top-5 match on the test tone, and a 4-minute mp3 analyses plus matches in under 5 s locally.

---

## Phase 5: The page

One static `site/index.html`: plain HTML/CSS/JS, no framework.

**Flow:**
1. **Hero:** "Which match is your song?" and one line explaining it. A big **Choose a song** button (file input, with drag-and-drop too). Small print: "Your song never leaves your device."
2. **Before any song is chosen**, show three built-in **example shapes** so the page is never empty: *Slow build*, *Explosive start*, *Rollercoaster*. These are synthetic curves. Each shows its matched game, with that game's WP chart.
3. **Result view:**
   - The match title, final score and league badge (NFL / IPL).
   - **Top panel:** the matched game's win-probability chart (style it like the previous project's player page: shaded above/below 50%, dots on scoring plays).
   - **Middle panel:** the song's energy curve and the game's tension curve overlaid, after alignment. Draw the DTW alignment so the moments line up.
   - **A playhead** runs through both panels while the song plays (from the local file, via an `<audio>` element with an object URL), with seek bar, ±10 s buttons and click-to-seek.
   - **Moments list:** the 5 biggest swings, each mapped to its timestamp in the song through the DTW path, e.g. "2:41, Gipson 65-yard punt-return TD (OT)". Clicking one jumps there. A moment card pops up as the playhead passes it.
   - **Runners-up:** #2–5 as small cards with mini sparklines. Clicking one swaps the result.
   - Toggle the search scope: **NFL · IPL · Both** (default: Both).
   - **Share line:** a copy button with text like `"Bohemian Rhapsody" is Bills @ Jets, 2023 Week 1. Which match is your song?` The song name comes from the file name with the extension and track numbers stripped.
4. **How it works:** four short lines plus the IPL model's metrics (log loss, Brier) with one sentence explaining what they mean.
5. **Credits:** nflverse/nflfastR, Cricsheet, and a note that WP for IPL is our own model.

**Design:** use stadium-night energy. Make it bold and fun, not corporate. It must work at 380 px wide and support light and dark mode via `prefers-color-scheme`.

**Don't:**
- Don't add accounts, analytics or any network calls besides the page's own static files.
- Don't autoplay.
- Don't show raw DTW distances to users. If you show a match quality at all, show it as a "match %" (map distance through a fixed curve fitted on the self-test distribution, and log the curve you used).

**✅ Checkpoint 5:**
- Open the page in a headless browser (Playwright) at 380 px and 1280 px, in light and dark mode, and screenshot each.
- Load the test tone through the file input programmatically and confirm a result renders, with the playhead moving and the moments list filled.
- There are no console errors.

---

## Phase 6 (bonus): "The Season as an Album"

Only start this after Phase 5 passes.

**Do:**
- For the **Jets 2023 season** (all games), render each game with `engine/compose_vivaldi.py` **and** `engine/compose_zimmer.py`. Both take `pbp.parquet GAME_ID out_prefix` and write MIDI. Render with `fluidsynth -ni -g 0.7 -r 44100 -F out.wav FluidR3_GM.sf2 out.mid`, then convert to mp3 (128k, loudnorm I=-16).
  - The scripts assume the Jets' perspective (`NYJ`). Keep that.
  - They embed `marker` meta events "play{i}" for syncing. Use these to build the playhead timeline, as in the earlier player page.
- **Cover art per track:** generate on a canvas from the game's WP curve (a filled ribbon in team colours, opponent abbreviation, week number). Draw it from code; no images or logos.
- Build `site/album.html` with:
  - a tracklist (Week, opponent, result, length)
  - a player with the WP chart, a piano roll and a playhead (re-implement the earlier player page's idea)
  - a Vivaldi/Interstellar style toggle
- Add a link from the main page: "Hear the Jets' 2023 season as an album →".

**Don't:**
- Don't change the composition logic in `engine/` beyond what's needed to run (paths, arguments).
- Don't use NFL or Jets logos or official artwork. Text and generated graphics only.

**✅ Checkpoint 6:** all tracks render, every track's playhead stays in sync (spot-check 3 tracks at the start, middle and end), and the album page passes the same headless checks as Phase 5.

---

## Phase 7: Publish and summarise

**Do:**
- Commit `site/`, `scripts/`, `engine/`, `models/` (if under 20 MB) and `logs/` (metrics and plots only). End the commit message with the attribution lines Claude Code normally adds.
- If `gh auth status` succeeds:
  1. Create a public repo `which-match-is-your-song`.
  2. Push it.
  3. Enable Pages from `/site` (or move the site to `/docs` if Pages can't serve `/site`).
  4. Confirm the URL returns 200.
- If not, leave everything committed locally and put the exact publish commands in `SUMMARY.md`.
- Write `SUMMARY.md` for Atharva (short and plain):
  1. The live link
  2. Counts: NFL games, IPL matches
  3. IPL model metrics vs the baseline, in one sentence each
  4. Self-test results
  5. 3 fun example matches using the synthetic shapes
  6. Anything that deviated from this plan
  7. A two-line pitch: *"Every song and every game has a shape. We matched 7,000 NFL and IPL games to the shape of any song, and the drop lands on the interception. A fan-engagement toy that runs entirely in the browser, on free data and our own cricket win-probability model."*

**Don't:** don't commit `data/raw`, any song audio, or anything over 50 MB.

**✅ Checkpoint 7:** the page is live (or ready to publish with commands), and `SUMMARY.md` exists.