#!/usr/bin/env bash
# Plan 13 Task 9 (preflight C5): the JSettlers arms are wall-clock sensitive (an answer from 4.5 s is `late`,
# robots wait 5 s for answers, a stalled turn is force-ended at 60 s, and 8 workers each run a JVM beside
# net:B@1000 on a 10-core machine), so each step starts only on an idle machine, on AC power, with the lid open.
# Exit 0 and print "idle" when all three hold; else print why and exit 1.
set -uo pipefail

# This script's own shell and its callers (a `bash runs/jsettlers/idle.sh && .venv/bin/alphasettler ...` line
# names the CLI in its own command line) are not other work.
ancestors() {
  local p=$$
  while [ -n "$p" ] && [ "$p" -gt 1 ]; do
    echo "$p"
    p=$(ps -o ppid= -p "$p" | tr -d ' ')
  done
}

busy=$(pgrep -f "arm.sh|\.venv/bin/alphasettler|cargo|rustc|maturin|pytest|net_speed|oracle\.play" \
  | grep -vxF -f <(ancestors) || true)
if [ -n "$busy" ]; then
  ps -o pid=,command= -p "$(echo "$busy" | paste -sd, -)" | cut -c1-160 >&2
  echo "busy: the JSettlers arms are wall-clock sensitive and need an idle machine" >&2
  exit 1
fi
pmset -g batt | head -1 | grep -q "AC Power" || { echo "on battery: the arms start on AC only" >&2; exit 1; }
ioreg -r -k AppleClamshellState | grep -q '"AppleClamshellState" = No' \
  || { echo "lid closed: the arms start with the lid open (a clamshell sleep voided arm F of 3d-4)" >&2; exit 1; }
echo idle
