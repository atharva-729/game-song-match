"""Checkpoint 4 parity: the test tone through Python (song_energy + match) and JS (core.js under Node)
must give the same top 5. Writes logs/parity.md.
usage: python scripts/parity_test.py
"""
import json, subprocess, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from match import N, search
from song_energy import energy_curve, load_mono

wav = ROOT / "data/test_tone.wav"
subprocess.run([sys.executable, str(ROOT / "scripts/make_test_tone.py"), str(wav)], check=True)
idx = json.loads((ROOT / "site/index.json").read_text(encoding="utf-8"))
meta = [dict(zip(idx["fields"], r)) for r in idx["rows"]]
Q = np.frombuffer((ROOT / "site/index.bin").read_bytes(), dtype=np.int8).reshape(-1, N)

x, sr = load_mono(wav)
py = energy_curve(x, sr)
r = search(py["curve"], Q, meta)
py_top = [(meta[i]["id"], d) for i, _, d in r["top"]]

js = json.loads(subprocess.run(["node", str(ROOT / "scripts/parity.js"), str(wav)], check=True,
                               capture_output=True, text=True).stdout)
js_top = [tuple(t) for t in js["top"]]
curve_diff = float(np.max(np.abs(np.array(py["curve"]) - np.array(js["curve"])))) if len(py["curve"]) == len(js["curve"]) else None
same = [a[0] for a in py_top] == [b[0] for b in js_top]
lines = ["# Python vs JS parity (test tone)", "",
         f"- Energy frames: Python {len(py['curve'])}, JS {len(js['curve'])}; max |diff| {curve_diff}",
         f"- Trim window: Python {py['t0']:.3f}-{py['t1']:.3f} s, JS {js['t0']:.3f}-{js['t1']:.3f} s",
         f"- JS timing (Node): energy {js['ms_energy']} ms, search {js['ms_search']} ms over {len(meta)} matches",
         f"- **Top 5 identical: {same}**", "", "| # | Python | dist | JS | dist |", "|---|---|---|---|---|"]
for k, (a, b) in enumerate(zip(py_top, js_top), 1):
    lines.append(f"| {k} | {a[0]} | {a[1]:.10f} | {b[0]} | {b[1]:.10f} |")
(ROOT / "logs/parity.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
sys.exit(0 if same else 1)
