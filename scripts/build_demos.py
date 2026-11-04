"""Demo songs for the page: five free classical recordings, one per tension shape.

Reads site/demo/{key}.mp3 (re-encoded to 128 kbps / 44.1 kHz from the classicals.de downloads), computes each
energy curve with the same code the browser uses, and writes site/demo/demos.json with the curve (resampled to
N points), credits, and the precomputed matches so the cards can show their game before anyone clicks.
Run after build_index.py (the matches depend on the index).
usage: python scripts/build_demos.py
"""
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from song_energy import energy_curve, load_mono
from match import N, resample, search

SITE = ROOT / "site"
CLASSICALS = "https://www.classicals.de"
MUSOPEN = {"performer": "Czech National Symphony Orchestra",
           "producer": "Musopen Kickstarter Recording Project",
           "producer_url": "https://www.kickstarter.com/projects/Musopen/record-and-release-free-music-without-copyrights",
           "licence": "Public Domain Mark 1.0", "licence_url": "https://creativecommons.org/publicdomain/mark/1.0/"}
DEMOS = [
    {"key": "danse-macabre", "shape": "Rollercoaster", "composer": "Camille Saint-Saëns", "piece": "Danse Macabre, Op. 40",
     "blurb": "Midnight strikes, skeletons dance into a frenzy, then a rooster crows at dawn and it all fizzles to nothing.",
     "performer": "Kevin MacLeod", "producer": "incompetech.com", "producer_url": "https://incompetech.com",
     "licence": "CC BY 4.0", "licence_url": "https://creativecommons.org/licenses/by/4.0/"},
    {"key": "mountain-king", "shape": "Boring, then brilliance", "composer": "Edvard Grieg",
     "piece": "Peer Gynt Suite No. 1, Op. 46: IV. In the Hall of the Mountain King",
     "blurb": "One tiny tiptoeing tune, over and over, louder and faster, until it goes completely mad at the end.", **MUSOPEN},
    {"key": "coriolan", "shape": "Explosive start, fizzles out", "composer": "Ludwig van Beethoven", "piece": "Coriolan Overture, Op. 62",
     "blurb": "Violent hammer-blow chords, a long fight, then it dies away into quiet plucked notes.", **MUSOPEN},
    {"key": "egmont", "shape": "The comeback", "composer": "Ludwig van Beethoven", "piece": "Egmont Overture, Op. 84",
     "blurb": "Heavy, grim chords, a struggle, a sudden hush, then an explosive triumphant ending.", **MUSOPEN},
    {"key": "valkyries", "shape": "Intense all the way", "composer": "Richard Wagner", "piece": "Ride of the Valkyries, WWV 86B",
     "blurb": "Loud almost from first note to last. The control: it should find a game that never let up.",
     "performer": "Philip Milman (conductor)", "producer": "pmmusic.pro", "producer_url": "https://pmmusic.pro/",
     "licence": "CC BY 3.0", "licence_url": "https://creativecommons.org/licenses/by/3.0/"},
]


def main():
    idx = json.loads((SITE / "index.json").read_text(encoding="utf-8"))
    meta = [dict(zip(idx["fields"], r)) for r in idx["rows"]]
    Q = np.fromfile(SITE / "index.bin", dtype=np.int8).reshape(-1, N)
    out = []
    for d in DEMOS:
        f = SITE / "demo" / f"{d['key']}.mp3"
        x, sr = load_mono(f)
        e = energy_curve(x, sr)
        curve = resample(np.asarray(e["curve"]), N)
        r = search(curve, Q, meta)
        pick = lambda t: meta[t[0]]["id"] if t else None
        out.append({**d, "file": f"demo/{f.name}", "source": CLASSICALS, "duration": round(len(x) / sr, 2),
                    "t0": e["t0"], "t1": e["t1"], "curve": [round(float(v), 5) for v in curve],
                    "top": [meta[t[0]]["id"] for t in r["top"]], "best_nfl": pick(r["best_nfl"]), "best_ipl": pick(r["best_ipl"])})
        print(f"{d['shape']:<30} {d['key']:<14} -> {meta[r['top'][0][0]]['title']}", flush=True)
    (SITE / "demo/demos.json").write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")


if __name__ == "__main__":
    main()
