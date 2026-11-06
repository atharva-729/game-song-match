"""Cut the long demo recordings down to under 2 minutes for the page.

Each clip is one unbroken stretch of the recording that still tells its card's story (no splices), always running
to the piece's real ending. Full-length sources live in data/raw/demo_full/{key}.mp3 (gitignored); the clips are
written over site/demo/{key}.mp3. build_demos.py imports CUTS to say on the page which part is played.
Run build_demos.py afterwards: the curves and matches change with the audio.
usage: python scripts/cut_demos.py
"""
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FULL = ROOT / "data/raw/demo_full"

# key: (start s, end s, fade-in s). Picked from the loudness profile so each clip opens on a clean edge.
CUTS = {
    # final 2 min: the heavy chords, a dead silence, the hushed woodwinds, then the triumphant ending
    "egmont": (421.0, 540.8, 0.05),
    # coda: starts straight after a general pause, storms up to full force, then dies into plucked notes
    "coriolan": (376.5, 496.3, 0.05),
    # final 2 min: the dance whips into its frenzy, the rooster crows, one last flare, then it fades to nothing
    "danse-macabre": (283.8, 403.6, 0.8),
}


def clock(s):
    return f"{int(s // 60)}:{int(s % 60):02d}"


def main():
    for key, (a, b, fin) in CUTS.items():
        src, dst = FULL / f"{key}.mp3", ROOT / "site/demo" / f"{key}.mp3"
        assert b - a < 120, key
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-ss", str(a), "-to", str(b),
                        "-af", f"afade=t=in:d={fin},areverse,afade=t=in:d=0.3,areverse",
                        "-map_metadata", "0", "-ac", "2", "-ar", "44100", "-b:a", "128k", str(dst)], check=True)
        print(f"{key:<14} {clock(a)}-{clock(b)}  ({b - a:.1f} s)", flush=True)


if __name__ == "__main__":
    main()
