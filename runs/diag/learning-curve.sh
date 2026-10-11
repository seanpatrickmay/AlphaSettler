#!/bin/bash
# Diagnostic 5: is training data-limited or target-limited? Trains 48/2 nets from scratch for 45 minutes each on
# the newest 4k, 8k, 16k, 32k and 54k self-play games (L's gens, then arm A's, then 3c's, newest first) and scores
# every net, plus A and L, on one fixed held-out set (the seed % 20 games of L's gen-053..056). If the held-out
# loss keeps falling with more games, more games a generation would help; if it is flat, the targets are the limit.
# Waits for the noise-floor diagnostic to finish (wall-clock training must not share the machine).
set -uo pipefail
cd "$(dirname "$0")/../.."
D=runs/diag/curve
mkdir -p $D
while pgrep -f noise_floor.py >/dev/null; do sleep 60; done
until bash runs/jsettlers/idle.sh >/dev/null 2>&1; do sleep 300; done

L=runs/train/L
A=runs/p2/A
gens() { for g in "$@"; do echo -n "$g "; done; }
POOL_4=$(gens $L/gen-055 $L/gen-056)
POOL_8=$(gens $L/gen-05{3,4,5,6})
POOL_16=$(gens $L/gen-049 $L/gen-05{0,1,2,3,4,5,6})
POOL_32=$(gens $L/gen-04{1,2,3,4,5,6,7,8,9} $L/gen-05{0,1,2,3,4,5,6})
POOL_54=$(gens runs/3c/gen-03{0,1,2,3,4} $A/gen-03{5,6,7,8,9} $A/gen-040 $POOL_32)

for n in 4 8 16 32 54; do
  pool_var=POOL_$n
  out=$D/scratch-${n}k.bin
  [ -e "$out" ] && continue
  echo "$(date '+%F %T') train ${n}k" | tee -a $D/curve.log
  caffeinate -dims .venv/bin/alphasettler fast-train --out "$out" --minutes 45 --seed $n ${!pool_var} \
    > $D/train-${n}k.log 2>&1 || { echo "train ${n}k failed" | tee -a $D/curve.log; exit 1; }
  tail -1 $D/train-${n}k.log | tee -a $D/curve.log
done
echo "$(date '+%F %T') score" | tee -a $D/curve.log
.venv/bin/alphasettler fast-score --nets $D/scratch-{4,8,16,32,54}k.bin runs/train/nets/A.bin runs/train/nets/L.bin \
  -- $L/gen-05{3,4,5,6} 2>&1 | tee $D/score.log
echo "$(date '+%F %T') done" | tee -a $D/curve.log
