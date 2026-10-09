#!/usr/bin/env bash
# Plan 12 Task 5 Step 4 with preflight C21: arm F through runs/p2/arm.sh unchanged, the best arm's set-up
# arguments (A: only the v4 start net) plus --compound. Waits until criterion 1 has finished.
set -euo pipefail
cd "$(dirname "$0")/../.."
while pgrep -f 'bash runs/3d4/c[1].sh' > /dev/null; do sleep 30; done
set -a; . runs/3d4/arm.env; set +a
test ! -e runs/p2/F
pgrep -fl "arm.sh|alphasettler|settler" || echo idle
grep -q '^F_RUNS=' runs/3d4/arm.env || echo "F_RUNS=runs/p2" >> runs/3d4/arm.env
echo "F set up with: $ARGS --compound" | tee -a runs/3d4/finals.log
nohup caffeinate -dims bash runs/p2/arm.sh F $ARGS --compound >> runs/p2/F.out 2>&1 &
echo "arm F launched, pid $!"
