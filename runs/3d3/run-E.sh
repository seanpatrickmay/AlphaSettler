#!/usr/bin/env bash
# Plan 11 Task 3 Step 3: arm E (the control's settings, the upgraded start, --league with the pool).
set -euo pipefail
cd "$(dirname "$0")/../.."
set -a; . runs/3d3/control.env; set +a
args="--start-net runs/3d2/gen-034-v4.bin"
if [ "$O" = "+o" ]; then args="$args --opening runs/3c/nets/opening.bin"; fi
echo "E set up with: $args --league $(tr '\n' ' ' < runs/3d3/pool.txt)" | tee -a runs/3d3/finals.log
# Fixed after the run (preflight C13): arm.sh passes its own --start-net first, so A's line was blank.
echo "$CTRL set up with: --start-net runs/3c/nets/gen-034.bin (arm.sh's own) $(head -1 runs/p2/$CTRL.log | cut -d' ' -f4-)" | tee -a runs/3d3/finals.log
pgrep -fl "alphasettler|settler" || echo idle
nohup caffeinate -dims bash runs/p2/arm.sh E $args --league $(cat runs/3d3/pool.txt) >> runs/p2/E.out 2>&1 &
echo "arm E launched, pid $!"
