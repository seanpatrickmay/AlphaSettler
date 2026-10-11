#!/bin/bash
# Does more training help? Arm A's settings continued from its best net (gen-040 at encoder v4) for 16 more
# generations (gen-041 to gen-056) on A's own data window, then one SPRT of the final best net against A, both
# @100+t as in every equal-wall-time criterion, on fresh seeds from 10,000,000,000. Waits for an idle machine
# (runs/jsettlers/idle.sh) before each part. Resumable: rerunning skips the set-up and continues the loop.
set -uo pipefail
cd "$(dirname "$0")/../.."
root=runs/train/L
log=runs/train/L.log
mkdir -p runs/train/nets

wait_idle() {
  until bash runs/jsettlers/idle.sh >/dev/null 2>&1; do sleep 300; done
}

wait_idle
if [ ! -e "$root/state.json" ]; then
  echo "$(date '+%F %T') setup" >> "$log"
  # Set-up only: --generations 40 is below --first 41, so no generation runs.
  .venv/bin/alphasettler fast-loop --dir "$root" --generations 40 --first 41 --trades \
    --seed-dirs runs/3c/gen-030 runs/3c/gen-031 runs/3c/gen-032 runs/3c/gen-033 runs/3c/gen-034 \
                runs/p2/A/gen-035 runs/p2/A/gen-036 runs/p2/A/gen-037 runs/p2/A/gen-038 runs/p2/A/gen-039 \
                runs/p2/A/gen-040 \
    --start-net runs/3d2/nets/A.bin >> "$log" 2>&1 || { echo "setup failed" >> "$log"; exit 1; }
fi
echo "$(date '+%F %T') generations" >> "$log"
caffeinate -dims .venv/bin/alphasettler fast-loop --dir "$root" --generations 56 >> "$log" 2>&1 \
  || { echo "$(date '+%F %T') loop exited with $?" >> "$log"; exit 1; }
cp "$root/nets/best.bin" runs/train/nets/L.bin
cp runs/3d2/nets/A.bin runs/train/nets/A.bin
shasum -a 256 runs/train/nets/L.bin runs/train/nets/A.bin >> "$log"
wait_idle
echo "$(date '+%F %T') sprt" >> "$log"
caffeinate -dims .venv/bin/alphasettler sprt --candidate net:L@100+t --baseline net:A@100+t --trades \
  --p0 0.25 --p1 0.30 --alpha 0.05 --beta 0.05 --batch 12 --max-seeds 400 --threads 8 \
  --seed-start 10000000000 --nets-dir runs/train/nets --out runs/train/sprt-L-vs-A.jsonl \
  2>&1 | tee runs/train/sprt-L-vs-A.log
echo "$(date '+%F %T') done" >> "$log"
