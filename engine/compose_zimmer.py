"""Interstellar-style score (Zimmer-inspired, original material) from a Jets game's win probability.
Language borrowed, not notes: church organ, a repeating arpeggio ostinato, a ticking clock,
slow harmony, and tension built by stacking layers rather than speeding up.
usage: python compose_zimmer.py pbp.parquet GAME_ID out_prefix"""
import sys, numpy as np, pandas as pd, mido

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
smooth = pd.Series(wp).ewm(alpha=0.3).mean().to_numpy()

TPB = 480
BPM = 108                       # Zimmer keeps the pulse constant; tension comes from layers
mid = mido.MidiFile(ticks_per_beat=TPB)
events = {}
def ev(tr, tick, msg): events.setdefault(id(tr), (tr, []))[1].append((int(tick), msg))
open_off = {}
def note(tr, ch, tick, dur, pitch, vel):
    vel = int(np.clip(vel, 1, 127)); pitch = int(pitch); tick = int(tick)
    prev = open_off.get((ch, pitch))
    if prev is not None and prev[0] > tick - 1: prev[0] = tick - 1
    ev(tr, tick, mido.Message("note_on", channel=ch, note=pitch, velocity=vel))
    off = [tick + max(1, int(dur))]; open_off[(ch, pitch)] = off
    events[id(tr)][1].append((off, mido.Message("note_off", channel=ch, note=pitch, velocity=0)))
def cc(tr, ch, tick, c, v): ev(tr, tick, mido.Message("control_change", channel=ch, control=c, value=int(np.clip(v, 0, 127))))
def track(name, ch, prog, vol=100, pan=64, rev=90):
    t = mido.MidiTrack(); mid.tracks.append(t)
    t.append(mido.MetaMessage("track_name", name=name))
    if ch != 9: t.append(mido.Message("program_change", channel=ch, program=prog))
    for c, v in ((7, vol), (10, pan), (91, rev)): t.append(mido.Message("control_change", channel=ch, control=c, value=v))
    return t

meta  = mido.MidiTrack(); mid.tracks.append(meta)
ev(meta, 0, mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(BPM)))
lead  = track("Organ lead", 0, 19, 96, 64, 110)    # church organ, the melody you follow
organ = track("Organ pad", 1, 19, 70, 64, 110)
ost   = track("Ostinato", 2, 0, 86, 46, 100)       # piano arpeggios
strs  = track("Strings", 3, 48, 80, 84, 100)
low   = track("Low brass", 4, 57, 100, 64, 90)     # trombones for the low "braaam"
bell  = track("Bells", 5, 14, 80, 70, 110)
drum  = track("Clock+drums", 9, 0, 110, 64, 60)
taiko = track("Taiko", 6, 116, 115, 64, 80)

A = 45  # A2
# chords as semitones over A: pairs of bars share a chord, harmony moves slowly
PROG = {
    "ahead":  [[0, 4, 7, 11], [-3, 2, 5, 9], [5, 9, 12, 16], [7, 11, 14, 19]],    # Amaj7  F#m  D  E   (lift)
    "edge":   [[0, 3, 7, 10], [0, 3, 7, 10], [-4, 0, 3, 7], [-5, -1, 2, 7]],      # Am Am F E(sus) : waiting
    "behind": [[0, 3, 7, 10], [-4, 0, 3, 7], [3, 7, 10, 14], [-2, 2, 5, 10]],     # Am F C G   (descending gravity)
}
def mood(w): return "ahead" if w >= 0.58 else ("behind" if w <= 0.38 else "edge")
MAJ = [0, 2, 4, 6, 7, 9, 11]   # A lydian when winning: that floating, hopeful Interstellar colour
MIN = [0, 2, 3, 5, 7, 8, 10]   # A aeolian when losing

tick = 0
cur = 64
for bi in range(0, n, 8):                # one harmony every 8 plays (two bars)
    ix = list(range(bi, min(bi + 8, n)))
    w = smooth[ix].mean()
    vol = np.abs(np.diff(wp[[max(ix[0] - 1, 0)] + ix])).sum()
    close = 1 - 2 * abs(w - 0.5)
    late = 1 - min(1, secs[ix[0]] / 1800)
    t = float(np.clip(0.5 * close * (0.6 + 0.6 * late) + 1.4 * vol, 0, 1))
    m = mood(w); chord = PROG[m][(bi // 8) % 4]
    seg = TPB * len(ix)
    scale = MAJ if m == "ahead" else MIN

    # layer 1: organ pad, always there; swells with tension
    for c in chord: note(organ, 1, tick, seg + 20, A + 12 + c, 45 + 45 * t)
    note(organ, 1, tick, seg + 20, A - 12 + chord[0] % 12, 60 + 40 * t)          # pedal note
    cc(organ, 1, tick, 11, 70 + 50 * t)

    # layer 2: piano ostinato, 8ths when engaged, 16ths at peak tension, silent when it's over
    if t > 0.2:
        step = TPB // 4 if t > 0.7 else TPB // 2
        pat = [0, 2, 3, 1, 3, 2, 0, 3]   # a simple rolling pattern over chord tones (our own)
        tones = [A + 12 + c for c in chord] + [A + 24 + chord[0]]
        for s in range(seg // step):
            p = tones[pat[s % len(pat)] % len(tones)] + (12 if t > 0.85 and s % 2 else 0)
            note(ost, 2, tick + s * step, step * 0.95, p, 45 + 50 * t + (10 if s % 4 == 0 else 0))

    # layer 3: strings enter as the stakes rise, long crescendo across the segment
    if t > 0.45:
        for c in chord[:3]: note(strs, 3, tick, seg, A + 24 + c, 50 + 60 * t)
        for k in range(8): cc(strs, 3, tick + k * seg / 8, 11, 40 + 85 * t * (k + 1) / 8)

    # the ticking clock: one tick per play; doubles when the game clock is short and it's close
    for k, i in enumerate(ix):
        at = tick + k * TPB
        ev(meta, at, mido.MetaMessage("marker", text=f"play{i}"))
        tick_vel = 55 + 45 * t
        note(drum, 9, at, TPB / 4, 76, tick_vel)                                  # hi wood block
        if secs[i] < 300 and close > 0.4: note(drum, 9, at + TPB / 2, TPB / 4, 77, tick_vel * 0.7)
        if t > 0.6 and k % 2 == 0: note(taiko, 6, at, TPB, 41, 50 + 50 * t)        # heartbeat

    # lead: long organ notes, one per bar, pitch follows win chance (the line you follow on the roll)
    for half in range(0, len(ix), 4):
        w2 = smooth[ix[half:half + 4]].mean()
        target = int(52 + 26 * w2)
        cand = [p for p in range(50, 82) if (p - A) % 12 in scale and (p - A - chord[0]) % 12 in [c % 12 for c in chord] + [2]]
        cur = min(cand, key=lambda p: abs(p - target) + 0.3 * abs(p - cur))
        L = TPB * len(ix[half:half + 4])
        note(lead, 0, tick + half * TPB, L * 1.02, cur, 70 + 40 * t)

    # events
    for k, i in enumerate(ix):
        at = tick + k * TPB
        sw = wp[i] - (wp[i - 1] if i else wp[i])
        if sw < -0.08:     # the braaam: low brass cluster + taiko + crash
            for p in (A - 12, A - 5, A, A + 1): note(low, 4, at, TPB * 3, p, 127)
            note(taiko, 6, at, TPB, 36, 127); note(drum, 9, at, TPB, 49, 110)
        if sw > 0.08:      # lift: organ blooms upward, bells
            for s, p in enumerate([A + 24, A + 28, A + 31, A + 35, A + 36]): note(organ, 1, at + s * TPB / 4, TPB * 3, p, 105)
            note(drum, 9, at, TPB, 57, 90)
        if i and jp[i] > jp[i - 1]:
            for s, p in enumerate([A + 36, A + 43, A + 40]): note(bell, 5, at + s * TPB / 2, TPB, p, 110)
        if i and op[i] > op[i - 1]:
            note(low, 4, at, TPB * 2, A - 12, 115); note(taiko, 6, at, TPB, 36, 115)
    tick += seg

# ending: win = full organ in A major, sustained; loss = the clock keeps ticking alone, then stops
if jets_won:
    for p in (A - 12, A, A + 7, A + 12, A + 16, A + 19, A + 24, A + 28): note(organ, 1, tick, TPB * 12, p, 120)
    note(lead, 0, tick, TPB * 12, A + 36, 120); note(taiko, 6, tick, TPB, 36, 127); note(drum, 9, tick, TPB * 2, 49, 120)
    for c in (0, 4, 7): note(strs, 3, tick, TPB * 12, A + 24 + c, 110)
else:
    for p in (A - 12, A, A + 3, A + 7): note(organ, 1, tick, TPB * 10, p, 55)
    note(lead, 0, tick, TPB * 8, A + 12, 55)
    for k in range(10): note(drum, 9, tick + k * TPB, TPB / 4, 76, 70 - 6 * k)
tick += TPB * 14

for tr, lst in events.values():
    lst = [(a[0] if isinstance(a, list) else a, m) for a, m in lst]
    lst.sort(key=lambda x: (x[0], 0 if x[1].type == "note_off" else 1))
    last = 0
    for a, m in lst: tr.append(m.copy(time=a - last)); last = a
mid.save(out + ".mid")
print(gid, "plays", n, "jets_won", jets_won, "length_s", round(mid.length, 1))