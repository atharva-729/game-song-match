"""Python reference for the page's song analysis (site/core.js energyCurve). Same frames, window,
loudness mapping, flux normalisation, smoothing and silence trim.
usage: python scripts/song_energy.py file.wav  -> prints JSON {curve, t0, t1, frames}
"""
import json, math, sys
import numpy as np, soundfile as sf

FRAME, HOP = 2048, 1024


def energy_curve(x, sr):
    x = np.asarray(x, dtype=np.float64)
    nf = (len(x) - FRAME) // HOP + 1 if len(x) >= FRAME else 0
    if nf < 4:
        return None
    hann = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(FRAME) / FRAME)
    frames = np.stack([x[f * HOP: f * HOP + FRAME] for f in range(nf)])
    rms = np.sqrt((frames ** 2).sum(axis=1) / FRAME)
    loud = (np.clip(20 * np.log10(rms + 1e-10), -60, 0) + 60) / 60
    mag = np.abs(np.fft.rfft(frames * hann, axis=1))
    flux = np.concatenate([[0.0], np.clip(np.diff(mag, axis=0), 0, None).sum(axis=1)])
    p95 = np.sort(flux)[int(math.floor(0.95 * (nf - 1)))] or 1.0
    e = 0.6 * loud + 0.4 * np.minimum(flux / p95, 1)
    half = int(round(3 * sr / HOP)) // 2
    csum = np.concatenate([[0.0], np.cumsum(e)])
    a = np.maximum(0, np.arange(nf) - half)
    b = np.minimum(nf, np.arange(nf) + half + 1)
    sm = (csum[b] - csum[a]) / (b - a)
    mx = sm.max()
    first, last = 0, nf - 1
    while first < nf and sm[first] < 0.05 * mx:
        first += 1
    while last > first and sm[last] < 0.05 * mx:
        last -= 1
    return {"curve": sm[first:last + 1].tolist(), "t0": (first * HOP + FRAME / 2) / sr,
            "t1": (last * HOP + FRAME / 2) / sr, "frames": nf}


def load_mono(path):
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    return x.mean(axis=1), sr


if __name__ == "__main__":
    x, sr = load_mono(sys.argv[1])
    print(json.dumps(energy_curve(x, sr)))
