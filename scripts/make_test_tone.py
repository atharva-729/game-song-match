"""Synthetic test 'song' (sine chords + noise, not music) with a known shape:
quiet intro, long build, peak, sudden drop, late final hit. 44.1 kHz mono 16-bit WAV.
usage: python scripts/make_test_tone.py out.wav [seconds]
"""
import sys
import numpy as np, soundfile as sf

out = sys.argv[1]
dur = float(sys.argv[2]) if len(sys.argv) > 2 else 90.0
sr = 44100
t = np.arange(int(dur * sr)) / sr
u = t / dur  # 0..1 through the piece
rng = np.random.default_rng(0)

# amplitude envelope: build to 0.65, peak until 0.82, drop, final hit at 0.93-1.0
env = np.interp(u, [0, 0.05, 0.65, 0.82, 0.84, 0.92, 0.93, 0.99, 1.0],
                   [0, 0.03, 0.55, 0.6, 0.06, 0.06, 0.7, 0.7, 0])
chord = sum(np.sin(2 * np.pi * f * t) for f in (220, 277.18, 329.63, 440)) / 4
# busy-ness: noise bursts on a pulse whose density rises with the envelope (drives spectral flux)
pulse = (np.sin(2 * np.pi * (2 + 6 * env) * t) > 0.6).astype(float)
noise = rng.normal(0, 1, len(t)) * pulse * env
x = env * chord * 0.8 + 0.35 * noise
x = 0.9 * x / np.abs(x).max()
sf.write(out, x.astype(np.float32), sr, subtype="PCM_16")
print(out, f"{dur:.0f}s")
