#!/bin/bash
# Copy of runs/p2/arm.sh run from the v3 worktree /Users/seanmay/Desktop/CurrentProjects/AlphaSettler-v3 (Plan 9's code at 57e416a with its own extension), so arm C can be timed after main moves to encoder v4.
# One arm of P2 criterion 2 (plan 8 Task 6 Step 5, with ledger ruling I1): set up a fresh fast-loop
# from gen-034 in runs/p2/$1 UNTIMED (extra arguments go to the set-up, e.g. --gumbel 100), then run
# its generations and stop at the first 10-second check at or past BUDGET seconds after the
# generations start. runs/p2/$1-state-at-cut.json is the state at the cut: only generations finished
# by then count. Every log line is prefixed with its epoch second.
set -u
cd /Users/seanmay/Desktop/CurrentProjects/AlphaSettler-v3
arm=$1
shift
root=runs/p2/$arm
log=runs/p2/$arm.log
budget=10800
if [ -e "$root" ]; then
  echo "$root exists: each arm starts in a fresh root" >&2
  exit 2
fi
echo "$(date +%s) setup $arm $*" > "$log"
# Set-up only: --generations 34 is below --first 35, so no generation runs.
PYTHONPATH=/Users/seanmay/Desktop/CurrentProjects/AlphaSettler-v3 /Users/seanmay/Desktop/CurrentProjects/AlphaSettler/.venv/bin/python3 -m alphasettler.cli fast-loop --dir "$root" --generations 34 --first 35 --trades \
  --seed-dirs runs/3b-v2/gen-002 runs/3b-v2/gen-003 runs/3b-v2/gen-004 runs/3b-v2/gen-005 \
              runs/3c/gen-030 runs/3c/gen-031 runs/3c/gen-032 runs/3c/gen-033 runs/3c/gen-034 \
  --start-net runs/3c/nets/gen-034.bin "$@" >> "$log" 2>&1 || { echo "$(date +%s) setup failed" >> "$log"; exit 1; }
start=$(date +%s)
echo "$start start $arm (generations; the budget starts here)" >> "$log"
PYTHONPATH=/Users/seanmay/Desktop/CurrentProjects/AlphaSettler-v3 /Users/seanmay/Desktop/CurrentProjects/AlphaSettler/.venv/bin/python3 -m alphasettler.cli fast-loop --dir "$root" --generations 99 \
  > >(while IFS= read -r line; do echo "$(date +%s) $line"; done >> "$log") 2>&1 &
pid=$!
while kill -0 "$pid" 2>/dev/null; do
  if (( $(date +%s) - start >= budget )); then
    cp "$root/state.json" "runs/p2/$arm-state-at-cut.json"
    cut=$(date +%s)
    # SIGTERM: a background job of a non-interactive bash ignores SIGINT.
    kill -TERM "$pid"
    wait "$pid"
    echo "$cut cut $arm after $((cut - start)) s" >> "$log"
    exit 0
  fi
  sleep 10
done
wait "$pid"
code=$?
echo "$(date +%s) loop exited early with $code" >> "$log"
exit 1
