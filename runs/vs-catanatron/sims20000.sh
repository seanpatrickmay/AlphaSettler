#!/bin/bash
# net:A (3d-2 gen-040, the current best) with the opening model, no domestic trades (as in the
# Catanatron web UI the user plays in), against three Catanatron AlphaBeta bots: 20,000 simulations a
# move (the human-play setting) and 1,000 (control) on the same 13 seeds, 52 games each.
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p runs/play/nets
cp -n runs/3d2/nets/A.bin runs/3c/nets/opening.bin runs/3c/nets/opening.bin.json runs/play/nets/ 2>/dev/null || true
for sims in 20000 1000; do
  .venv/bin/alphasettler arena --candidate "net:A@${sims}+o" --baseline catanatron:alphabeta --seeds 13 \
    --seed-start 0 --threads 8 --no-trades --nets-dir runs/play/nets \
    --out "runs/vs-catanatron/A-${sims}o-vs-alphabeta-notrades.jsonl" \
    2>&1 | tee "runs/vs-catanatron/A-${sims}o-vs-alphabeta-notrades.log"
done
