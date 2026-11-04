#!/usr/bin/env bash
# Rebuild the site: [NFL curves] [IPL parse + model + curves] -> index -> self-test -> parity -> browser checks.
# usage: bash scripts/rebuild.sh [--nfl] [--ipl]
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/.venvs/ipl-song-match/Scripts/python.exe}
export PYTHONUTF8=1
for arg in "$@"; do
  case $arg in
    --nfl) $PY scripts/build_nfl.py && $PY scripts/nfl_sanity.py ;;
    --ipl) $PY scripts/parse_cricsheet.py && $PY scripts/build_ipl.py ;;
  esac
done
$PY scripts/build_index.py
$PY scripts/match_selftest.py
$PY scripts/build_demos.py
$PY scripts/parity_test.py
$PY scripts/check_page.py > /dev/null && echo "page checks: PASS" || { echo "page checks: FAIL (see logs/page_check.json)"; exit 1; }
