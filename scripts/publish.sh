#!/usr/bin/env bash
# Publish site/ to GitHub Pages: pushes main, then site/ as the gh-pages branch (Pages serves that branch).
# usage: bash scripts/publish.sh        (commit your changes first)
set -euo pipefail
cd "$(dirname "$0")/.."
git push origin main
git push origin "$(git subtree split --prefix site):refs/heads/gh-pages"
echo "Pushed. GitHub Pages rebuilds in about a minute: https://atharva-729.github.io/game-song-match/"
