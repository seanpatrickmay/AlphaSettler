#!/bin/bash
# Plan 13 Task 9 Steps 5-6, unattended: the three arms (native, aligned, no trades), 100 seeds each, then the
# 20-seed native-boards check. Each arm waits for runs/jsettlers/idle.sh (polling every 5 minutes). After the
# native arm the native-arm condition is checked (late at most 5% of offers_to_us); a breach keeps that file as
# native-8-threads.jsonl and moves every arm to --threads 4 on 8,000,001,000 so the arms stay paired. A failed
# arm (exit 3) stops the script with its .partial file kept for triage.
set -uo pipefail
cd "$(dirname "$0")/../.."
J=runs/jsettlers
L=$J/logs
NETS=$J/nets

wait_idle() {
  until bash $J/idle.sh >/dev/null 2>&1; do sleep 300; done
}

move_logs() {  # the CLI writes JVM logs loose under $L; Step 7 greps them per arm
  mkdir -p "$L/$1" && find $L -maxdepth 1 -name '*.log' -exec mv {} "$L/$1/" \;
}

arm() {  # arm NAME THREADS SEED_START ARGS...
  local name=$1 threads=$2 start=$3
  shift 3
  wait_idle
  echo "$(date '+%F %T') start $name (threads $threads, seeds from $start)" >> $J/arms.out
  .venv/bin/alphasettler arena --baseline jsettlers:smart --threads "$threads" --seed-start "$start" \
    --nets-dir $NETS --out "$J/$name.jsonl" "$@" 2>&1 | tee "$J/$name.log"
  local rc=${PIPESTATUS[0]}
  move_logs "$name"
  echo "$(date '+%F %T') end $name exit $rc" >> $J/arms.out
  if [ "$rc" -ne 0 ]; then
    echo "$name failed (exit $rc); see $J/$name.log and $J/$name.jsonl.partial" >> $J/arms.out
    exit "$rc"
  fi
}

count() {  # count KEY LOGFILE: the number after KEY on the trades: line
  grep '^trades:' "$2" | grep -oE "$1 [0-9]+" | awk '{print $2}'
}

mkdir -p $L/before && find $L -maxdepth 1 -name '*.log' -exec mv {} $L/before/ \;

threads=8
start=8000000000
arm native $threads $start --candidate net:B@1000+t+o --trades --seeds 100
late=$(count late $J/native.log)
offers=$(count offers_to_us $J/native.log)
echo "native-arm condition: late $late of offers_to_us $offers" >> $J/arms.out
if [ $((late * 20)) -gt "$offers" ]; then
  echo "native-arm condition BREACHED at 8 threads; rerunning every arm with 4 threads on 8000001000" >> $J/arms.out
  mv $J/native.jsonl $J/native-8-threads.jsonl
  mv $J/native.log $J/native-8-threads.log
  mv $L/native $L/native-8-threads
  threads=4
  start=8000001000
  arm native $threads $start --candidate net:B@1000+t+o --trades --seeds 100
  late=$(count late $J/native.log)
  offers=$(count offers_to_us $J/native.log)
  echo "native-arm condition (rerun): late $late of offers_to_us $offers" >> $J/arms.out
fi
arm aligned $threads $start --candidate net:B@1000+t+o --trades --aligned --seeds 100
arm no-trades $threads $start --candidate net:B@1000+o --seeds 100
arm native-boards $threads $start --candidate net:B@1000+t+o --trades --native-boards --seeds 20
echo "$(date '+%F %T') all arms done" >> $J/arms.out
