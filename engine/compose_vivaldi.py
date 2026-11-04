"""Compose an orchestral piece (cello lead, strings, bass, timpani) from a Jets game's win probability.
Writes MIDI; render with fluidsynth + FluidR3_GM.
usage: python compose.py pbp.parquet GAME_ID out_prefix"""
import sys, numpy as np, pandas as pd, mido
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

pbp, gid, out = sys.argv[1], sys.argv[2], sys.argv[3]
d = pd.read_parquet(pbp)
g = d[(d.game_id == gid) & d.home_wp.notna()].copy()
g = g[g.play_type.notna()]
jets_home = g.home_team.iloc[0] == "NYJ"
wp = (g.home_wp if jets_home else 1 - g.home_wp).clip(0.001, 0.999).to_numpy()
fh, fa = g.home_score.iloc[0], g.away_score.iloc[0]
jets_won = (fh > fa) == jets_home
jp = (g.total_home_score if jets_home else g.total_away_score).to_numpy()
op = (g.total_away_score if jets_home else g.total_home_score).to_numpy()
secs = g.game_seconds_remaining.fillna(0).to_numpy()
n = len(wp)

# ---- per-bar state (4 plays = 1 bar of 4/4) ----
TPB = 480
bars = [list(range(i, min(i + 4, n))) for i in range(0, n, 4)]
smooth = pd.Series(wp).ewm(alpha=0.35).mean().to_numpy()

def bar_state(ix):
    w = smooth[ix].mean()
    vol = np.abs(np.diff(wp[[max(ix[0] - 1, 0)] + ix])).sum()
    close = 1 - 2 * abs(w - 0.5)
    late = 1 - min(1, secs[ix[0]] / 1800)           # second half ramps the stakes
    tension = float(np.clip(0.45 * close * (0.6 + 0.6 * late) + 2.2 * vol, 0, 1))
    return w, tension

# ---- harmony: D major when ahead, D minor (harmonic) when behind ----
D = 50  # D3
MAJ = [0, 2, 4, 5, 7, 9, 11]
MIN = [0, 2, 3, 5, 7, 8, 11]  # harmonic minor: the raised C# gives the "something's wrong" pull
# chords as scale degrees (0-based)
PROG = {
    "triumph":  [[0, 2, 4], [3, 5, 0], [4, 6, 1], [0, 2, 4]],     # I IV V I
    "hopeful":  [[0, 2, 4], [5, 0, 2], [3, 5, 0], [4, 6, 1]],     # I vi IV V
    "knife":    [[0, 2, 4], [4, 6, 1], [0, 2, 4], [4, 6, 1, 3]],  # i V i V7  (Vivaldi storm pedal)
    "sinking":  [[0, 2, 4], [3, 5, 0], [5, 0, 2], [4, 6, 1]],     # i iv VI V
    "despair":  [[0, 2, 4], [3, 5, 0], [0, 2, 4], [0, 2, 4]],     # i iv i i, static and heavy
}
def mood(w, t):
    if w >= 0.72: return "triumph", MAJ
    if w >= 0.55: return "hopeful", MAJ
    if w >= 0.40 or t > 0.65: return "knife", MIN
    if w >= 0.18: return "sinking", MIN
    return "despair", MIN

def deg_pitch(scale, deg, base):
    o, k = divmod(deg, 7); return base + 12 * o + scale[k]

def scale_tones(scale, lo, hi):
    return [p for p in range(lo, hi + 1) if (p - D) % 12 in scale]

# ---- tracks ----
mid = mido.MidiFile(ticks_per_beat=TPB)
def track(name, ch, prog, vol=100, pan=64):
    t = mido.MidiTrack(); mid.tracks.append(t)
    t.append(mido.MetaMessage("track_name", name=name))
    t.append(mido.Message("program_change", channel=ch, program=prog))
    t.append(mido.Message("control_change", channel=ch, control=7, value=vol))
    t.append(mido.Message("control_change", channel=ch, control=10, value=pan))
    t.append(mido.Message("control_change", channel=ch, control=91, value=70))  # reverb
    return t
events = {}  # track -> list of (abs_tick, msg)
def ev(tr, tick, msg): events.setdefault(id(tr), (tr, []))[1].append((int(tick), msg))
open_off = {}
def note(tr, ch, tick, dur, pitch, vel):
    vel = int(np.clip(vel, 1, 127)); pitch = int(pitch); tick = int(tick)
    key = (ch, pitch); prev = open_off.get(key)
    if prev is not None and prev[0] > tick - 1:      # same pitch still sounding: end it just before
        prev[0] = tick - 1
    ev(tr, tick, mido.Message("note_on", channel=ch, note=pitch, velocity=vel))
    off = [tick + max(1, int(dur))]; open_off[key] = off
    events[id(tr)][1].append((off, mido.Message("note_off", channel=ch, note=pitch, velocity=0)))
def cc(tr, ch, tick, c, v): ev(tr, tick, mido.Message("control_change", channel=ch, control=c, value=int(np.clip(v, 0, 127))))

meta = mido.MidiTrack(); mid.tracks.append(meta)
cello = track("Cello", 0, 42, 118, 54)
pad   = track("Strings", 1, 48, 82, 74)
trem  = track("Tremolo", 2, 44, 78, 84)
vlns  = track("Violins", 3, 40, 70, 90)
bass  = track("Contrabass", 4, 43, 100, 50)
timp  = track("Timpani", 5, 47, 110, 64)

tick = 0
cur = 57  # cello starts on A3
tempo_ticks = []
for bi, ix in enumerate(bars):
    w, t = bar_state(ix)
    name, scale = mood(w, t)
    chord = PROG[name][bi % 4]
    bar_len = TPB * len(ix)
    bpm = 96 + 48 * t                      # calm = slow and singing, tense = driving
    ev(meta, tick, mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(bpm)))
    base_vel = 52 + 55 * t

    # swells: expression follows tension within the bar
    for k in range(4):
        cc(cello, 0, tick + k * bar_len / 4, 11, 70 + 55 * t + (10 if k == 2 else 0))
        cc(pad, 1, tick + k * bar_len / 4, 11, 60 + 40 * (1 - t) if t < 0.55 else 40)

    # --- harmony ---
    chord_p = [deg_pitch(scale, dgr, D + 12) for dgr in chord]          # around D4
    root = deg_pitch(scale, chord[0], D - 12)                           # D2 region
    if t < 0.55:
        for p in chord_p: note(pad, 1, tick, bar_len + 30, p, 45 + 30 * t)
        note(bass, 4, tick, bar_len / 2, root, 70); note(bass, 4, tick + bar_len / 2, bar_len / 2, root, 60)
    else:
        for p in chord_p: note(trem, 2, tick, bar_len, p, 50 + 50 * t)
        # Vivaldi-style driving 16ths in the violins: chord tones up high, repeated
        sixteenth = TPB // 4
        vp = [p + 12 for p in chord_p]
        for s in range(len(ix) * 4):
            note(vlns, 3, tick + s * sixteenth, sixteenth * 0.9, vp[(s // 2) % len(vp)], 55 + 50 * t + (12 if s % 4 == 0 else 0))
        for s in range(len(ix) * 2):       # repeated eighth-note bass, the engine of the storm
            note(bass, 4, tick + s * TPB / 2, TPB / 2 * 0.9, root, 75 + 35 * t)

    # --- cello melody: walks toward a target pitch set by win chance ---
    target = int(45 + 26 * w)              # A2 when hopeless ... B4 when cruising
    tones = scale_tones(scale, 40, 76)
    chord_pcs = {(p - D) % 12 for p in chord_p}
    if t < 0.3:   rhythm = [2, 2]
    elif t < 0.55: rhythm = [1, 0.5, 0.5, 1.5, 0.5]
    elif t < 0.8: rhythm = [0.5] * 8
    else:         rhythm = [0.25] * 8 + [0.5] * 4
    scale_r = len(ix) / 4; pos = 0.0
    for r in rhythm:
        r *= scale_r
        strong = abs(pos - round(pos)) < 1e-6 and round(pos) % 2 == 0
        if strong:   # land on a chord tone near the path to target
            cands = [p for p in tones if (p - D) % 12 in chord_pcs and abs(p - cur) <= 7]
            cands = cands or tones
            cur = min(cands, key=lambda p: abs(p - (cur + np.sign(target - cur) * 2)))
        else:        # step toward target; if there, ornament with a neighbour
            i = min(range(len(tones)), key=lambda j: abs(tones[j] - cur))
            if abs(target - cur) <= 1: i = i + (1 if int(pos * 2) % 2 else -1)
            else: i += 1 if target > cur else -1
            cur = tones[int(np.clip(i, 0, len(tones) - 1))]
        note(cello, 0, tick + pos * TPB, r * TPB * 1.04, cur, base_vel + (10 if strong else 0))
        pos += r

    # --- events inside the bar ---
    for k, i in enumerate(ix):
        at = tick + k * TPB
        ev(meta, at, mido.MetaMessage("marker", text=f"play{i}"))
        sw = wp[i] - (wp[i - 1] if i else wp[i])
        if sw < -0.08:      # disaster: diminished 7th stab + timpani hit, fortissimo
            dim = [D + 13, D + 16, D + 19, D + 22]  # C#dim7 over D: the classic horror chord
            for p in dim: note(trem, 2, at, TPB * 2, p, 120)
            note(timp, 5, at, TPB, 38, 127); note(bass, 4, at, TPB * 2, 37, 120)
        if sw > 0.08:       # turn of fortune: timpani roll + bright stab
            for s in range(8): note(timp, 5, at + s * TPB / 8, TPB / 8, 38, 70 + s * 7)
            for p in [D + 12, D + 16, D + 19, D + 24]: note(pad, 1, at + TPB, TPB * 2, p, 115)
        if i and jp[i] > jp[i - 1]:   # Jets score: cello leaps up an arpeggio
            for s, m in enumerate([0, 4, 7, 12]):
                note(vlns, 3, at + s * TPB / 4, TPB / 4, D + 24 + m, 110)
        if i and op[i] > op[i - 1]:   # opponent scores: descending chromatic sigh in the violins
            for s in range(4):
                note(vlns, 3, at + s * TPB / 3, TPB / 3, D + 29 - s, 100)
    tick += bar_len

# --- ending ---
ev(meta, tick, mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(66)))
if jets_won:
    for s in range(16): note(timp, 5, tick + s * TPB / 8, TPB / 8, 38, 80 + s * 2)
    for p in [D - 12, D, D + 7, D + 12, D + 16, D + 19, D + 24]: note(pad, 1, tick + TPB * 2, TPB * 8, p, 110)
    note(cello, 0, tick, TPB, 69, 110); note(cello, 0, tick + TPB, TPB, 71, 110); note(cello, 0, tick + TPB * 2, TPB * 8, 74, 120)
    note(bass, 4, tick + TPB * 2, TPB * 8, D - 12, 120); note(timp, 5, tick + TPB * 2, TPB, 38, 127)
else:
    for p in [D, D + 3, D + 7]: note(pad, 1, tick, TPB * 10, p, 60)
    fall = [p for p in scale_tones(MIN, 38, cur) if p <= cur][::-1][:5]
    for k, p in enumerate(fall): note(cello, 0, tick + k * TPB * 1.5, TPB * 1.6, p, 75 - 6 * k)
    note(cello, 0, tick + TPB * 7.5, TPB * 6, 38, 55); note(bass, 4, tick, TPB * 12, D - 12, 60)
tick += TPB * 12

for tr, lst in events.values():
    lst = [(a[0] if isinstance(a, list) else a, m) for a, m in lst]
    lst.sort(key=lambda x: (x[0], 0 if x[1].type == "note_off" else 1))
    last = 0
    for a, m in lst: tr.append(m.copy(time=a - last)); last = a
mid.save(out + ".mid")
print(gid, "plays", n, "bars", len(bars), "jets_won", jets_won, "length_s", round(mid.length, 1))