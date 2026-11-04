#!/usr/bin/env bash
# Waits for a real Cricsheet zip in data/raw/cricsheet/ (not the bot-check HTML page), then runs the IPL rebuild.
# usage: bash scripts/wait_for_cricsheet.sh
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/.venvs/ipl-song-match/Scripts/python.exe}
valid() {
  "$PY" -c "import sys, zipfile, glob; z = [f for f in glob.glob('data/raw/cricsheet/*.zip') if zipfile.is_zipfile(f)]; sys.exit(0 if z else 1)"
}
until valid; do sleep 30; done
# let a browser download finish writing
prev=-1; size=$(du -sb data/raw/cricsheet | cut -f1)
while [ "$size" != "$prev" ]; do sleep 10; prev=$size; size=$(du -sb data/raw/cricsheet | cut -f1); done
echo "cricsheet zip found ($size bytes); running IPL rebuild"
bash scripts/rebuild.sh --ipl > logs/rebuild_ipl_run.txt 2>&1
code=$?
grep -E "ipl_matches|shipped|log_loss\"|passed|Top 5|page checks|Error|Traceback" logs/rebuild_ipl_run.txt | head -20
exit $code
