# Matcher self-test

Index: 8489 matches (7273 NFL, 1216 IPL).

## Synthetic songs

Mean z-tension of the top match per third of the game, and where its peak sits (0 = start, 1 = end).

**Slow build** (3.3 s)
1. Bengals 23 @ Browns 20 (OT) · 2009 Week 4: dist 0.044, thirds [-0.76, -0.5, 1.27], peak at 0.92
2. Steelers 10 @ Ravens 13 (OT) · 2003 Week 17: dist 0.050, thirds [-0.82, -0.36, 1.2], peak at 0.99
3. Colts 22 @ Ravens 19 (OT) · 2023 Week 3: dist 0.052, thirds [-0.89, -0.27, 1.18], peak at 0.94
   best NFL: Bengals 23 @ Browns 20 (OT) · 2009 Week 4 (dist 0.044)
   best IPL: DC vs RR · IPL 2025 Match 32 (dist 0.060)

**Explosive start** (2.4 s)
1. LSG vs PBKS · IPL 2023 Match 38: dist 0.018, thirds [1.12, -0.5, -0.63], peak at 0.03
2. KKR vs RCB · IPL 2020 Match 39: dist 0.021, thirds [1.08, -0.4, -0.69], peak at 0.05
3. SRH vs RCB · IPL 2019 Match 11: dist 0.022, thirds [1.19, -0.32, -0.87], peak at 0.01
   best NFL: Falcons 24 @ Raiders 0 · 2008 Week 9 (dist 0.022)
   best IPL: LSG vs PBKS · IPL 2023 Match 38 (dist 0.018)

**Rollercoaster** (2.4 s)
1. Giants 21 @ Steelers 14 · 2008 Week 8: dist 0.152, thirds [-0.51, -0.03, 0.54], peak at 0.8
2. Steelers 19 @ Falcons 16 · 2022 Week 13: dist 0.201, thirds [0.13, -0.53, 0.41], peak at 0.88
3. Texans 16 @ Ravens 23 · 2017 Week 12: dist 0.210, thirds [-0.31, -0.17, 0.49], peak at 0.9
   best NFL: Giants 21 @ Steelers 14 · 2008 Week 8 (dist 0.152)
   best IPL: GT vs KKR · IPL 2023 Match 13 (dist 0.631)

## Test 4: a game's own curve returns that game as #1

**20/20 passed.**

## Match % curve

Best-match distance over 120 random smooth songs: p10/p50/p90 = [0.0237, 0.0842, 0.2566].
match % = round(100 * exp(-dist / 0.37737)), so the median song shows 80%.
