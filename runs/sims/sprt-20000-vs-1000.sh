#!/bin/bash
# Is 20,000 simulations a move (the human-play setting) stronger than 1,000? net:A with the opening
# model, no trades (as in the Catanatron web UI), one 20,000-sim seat against three 1,000-sim seats,
# SPRT p0 0.25 / p1 0.30.
set -euo pipefail
cd "$(dirname "$0")/../.."
.venv/bin/alphasettler sprt --candidate "net:A@20000+o" --baseline "net:A@1000+o" --p0 0.25 --p1 0.30 \
  --alpha 0.05 --beta 0.05 --batch 12 --max-seeds 400 --threads 8 --no-trades --nets-dir runs/play/nets \
  --out runs/sims/sprt-20000-vs-1000.jsonl 2>&1 | tee runs/sims/sprt-20000-vs-1000.log
