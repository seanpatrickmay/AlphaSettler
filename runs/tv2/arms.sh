#!/bin/bash
# Targets v2 (spec docs/superpowers/specs/2026-10-11-targets-v2-design.md, Section 2): arm T is long-A.sh's
# continuation of arm A (gen-041..056, the same seed directories and start net) with --targets-v2, then one SPRT of
# T's best against A from seed 12,000,000,000. Then diagnostic D5 (runs/diag/learning-curve.sh, deferred so its
# timed trainings get an idle machine). Then arm E: T's settings plus --ema, SPRT'd against T's best if T's SPRT
# was H1, else against A, from seed 13,000,000,000. Waits for an idle machine (runs/jsettlers/idle.sh) before
# each part. Resumable: rerunning skips finished set-ups, loops, copies and SPRTs.
set -uo pipefail
cd "$(dirname "$0")/../.."
D=runs/tv2
log=$D/arms.log
mkdir -p $D/nets

wait_idle() {
  while pgrep -f noise_floor.py >/dev/null; do sleep 60; done
  until bash runs/jsettlers/idle.sh >/dev/null 2>&1; do sleep 300; done
}

SEEDS=(runs/3c/gen-030 runs/3c/gen-031 runs/3c/gen-032 runs/3c/gen-033 runs/3c/gen-034
       runs/p2/A/gen-035 runs/p2/A/gen-036 runs/p2/A/gen-037 runs/p2/A/gen-038 runs/p2/A/gen-039 runs/p2/A/gen-040)

arm() {  # arm NAME FLAGS...: set up (once) and run generations 41..56, then copy the best net to nets/NAME.bin
  local name=$1; shift
  local root=$D/$name
  wait_idle
  if [ ! -e "$root/state.json" ]; then
    echo "$(date '+%F %T') $name: setup ($*)" >> "$log"
    # Set-up only: --generations 40 is below --first 41, so no generation runs.
    .venv/bin/alphasettler fast-loop --dir "$root" --generations 40 --first 41 --trades "$@" \
      --seed-dirs "${SEEDS[@]}" --start-net runs/3d2/nets/A.bin >> "$log" 2>&1 \
      || { echo "$name: setup failed" >> "$log"; exit 1; }
  fi
  echo "$(date '+%F %T') $name: generations" >> "$log"
  caffeinate -dims .venv/bin/alphasettler fast-loop --dir "$root" --generations 56 >> "$log" 2>&1 \
    || { echo "$(date '+%F %T') $name: loop exited with $?" >> "$log"; exit 1; }
  cp "$root/nets/best.bin" "$D/nets/$name.bin"
  shasum -a 256 "$D/nets/$name.bin" >> "$log"
}

sprt() {  # sprt CANDIDATE BASELINE SEED_START: one SPRT @100+t, skipped if its log already has a verdict
  local out=$D/sprt-$1-vs-$2
  if [ -e "$out.log" ] && grep -q '^sprt: H' "$out.log"; then return; fi
  wait_idle
  echo "$(date '+%F %T') sprt $1 vs $2" >> "$log"
  rm -f "$out.jsonl"
  caffeinate -dims .venv/bin/alphasettler sprt --candidate "net:$1@100+t" --baseline "net:$2@100+t" --trades \
    --p0 0.25 --p1 0.30 --alpha 0.05 --beta 0.05 --batch 12 --max-seeds 400 --threads 8 \
    --seed-start "$3" --nets-dir $D/nets --out "$out.jsonl" 2>&1 | tee "$out.log"
}

cp runs/3d2/nets/A.bin $D/nets/A.bin
arm T --targets-v2
sprt T A 12000000000

if [ ! -e runs/diag/curve/score.log ]; then
  wait_idle
  echo "$(date '+%F %T') D5 learning curve" >> "$log"
  bash runs/diag/learning-curve.sh > runs/diag/learning-curve.out 2>&1
fi

arm E --targets-v2 --ema
if grep -q '^sprt: H1' $D/sprt-T-vs-A.log; then base=T; else base=A; fi
sprt E $base 13000000000
echo "$(date '+%F %T') done" >> "$log"
