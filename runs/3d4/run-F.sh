#!/usr/bin/env bash
# Plan 12 Task 5 Step 4 with preflight C21: arm F through runs/p2/arm.sh unchanged, the best arm's set-up
# arguments (A: only the v4 start net) plus --compound. Waits until criterion 1 has finished.
set -euo pipefail
cd "$(dirname "$0")/../.."
while pgrep -f 'bash runs/3d4/c[1].sh' > /dev/null; do sleep 30; done
set -a; . runs/3d4/arm.env; set +a
test ! -e runs/p2/F
pgrep -fl "arm.sh" && { echo "another arm runs" >&2; exit 1; }
# ALLOW_BUSY=1: the user allowed niced Plan 13 work (its own worktree) beside arm F (2026-10-09).
if [ "${ALLOW_BUSY:-}" != 1 ] && pgrep -fl "arm.sh|\.venv/bin/alphasettler|cargo|rustc|maturin|pytest|net_speed"; then
  echo "busy: arm F needs an idle machine (ledger C1)" >&2; exit 1
fi
pmset -g batt | head -1 | grep -q "AC Power" || { echo "on battery: arm F starts on AC only" >&2; exit 1; }
grep -q '^F_RUNS=' runs/3d4/arm.env || echo "F_RUNS=runs/p2" >> runs/3d4/arm.env
echo "F set up with: $ARGS --compound" | tee -a runs/3d4/finals.log
pmset -g batt | head -1 | tee -a runs/3d4/finals.log
nohup caffeinate -dims bash runs/p2/arm.sh F $ARGS --compound >> runs/p2/F.out 2>&1 &
echo "arm F launched, pid $!"
