#!/bin/bash
# Final-review ruling: D2 from the seat to act's view. Reruns Task 6 Steps 2-3 at d5818cf; the own-view runs are kept as *-ownview.*.
set -o pipefail
cd /Users/seanmay/Desktop/CurrentProjects/AlphaSettler
A=.venv/bin/alphasettler
$A opening-diagnose \
  --games runs/3c/gen-030 runs/3c/gen-031 runs/3c/gen-032 runs/3c/gen-033 runs/3c/gen-034 \
  --net gen-034 --nets-dir runs/3c/nets --audit 200 --out runs/3d1/diagnose-3c.md 2>&1 | tee runs/3d1/diagnose-3c.log || { echo "FAILED 2" >> runs/3d1/rerun.out; exit 1; }
$A opening-diagnose --games runs/3b-v2/gen-005 --net gen-034 --nets-dir runs/3c/nets \
  --audit 0 --out runs/3d1/diagnose-3b-gen005.md 2>&1 | tee runs/3d1/diagnose-3b-gen005.log || { echo "FAILED 3" >> runs/3d1/rerun.out; exit 1; }
echo "=== rerun done" >> runs/3d1/rerun.out
