#!/bin/bash
# Targets v2 follow-up: does the gain hold with more search? After runs/tv2/arms.sh finishes, one SPRT of the
# targets v2 best (E if E's SPRT against T was H1, else T) against three A at 600 simulations, trading game, the
# arms' SPRT settings, seeds from 15,000,000,000. Resumable: skipped if its log already has a verdict.
set -uo pipefail
cd "$(dirname "$0")/../.."
D=runs/tv2
while pgrep -f runs/tv2/arms.sh >/dev/null; do sleep 300; done
grep -q ' done$' $D/arms.log || { echo "arms.sh did not finish" >> $D/arms.log; exit 1; }
until bash runs/jsettlers/idle.sh >/dev/null 2>&1; do sleep 300; done
if grep -q '^sprt: H1' $D/sprt-E-vs-T.log 2>/dev/null; then c=E; else c=T; fi
out=$D/sprt-$c-vs-A-600
if [ -e "$out.log" ] && grep -q '^sprt: H' "$out.log"; then exit 0; fi
echo "$(date '+%F %T') sprt $c vs A @600" >> $D/arms.log
rm -f "$out.jsonl"
caffeinate -dims .venv/bin/alphasettler sprt --candidate "net:$c@600+t" --baseline "net:A@600+t" --trades \
  --p0 0.25 --p1 0.30 --alpha 0.05 --beta 0.05 --batch 12 --max-seeds 400 --threads 8 \
  --seed-start 15000000000 --nets-dir $D/nets --out "$out.jsonl" 2>&1 | tee "$out.log"
echo "$(date '+%F %T') confirm done" >> $D/arms.log
