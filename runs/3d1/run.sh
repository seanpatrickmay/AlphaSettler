#!/bin/bash
# Plan 7 Task 6 Steps 2-7 (controller runs this), in order. Each step's output is tee'd as the plan says;
# a failing step stops the script with its name in runs/3d1/run.out.
set -o pipefail
cd /Users/seanmay/Desktop/CurrentProjects/AlphaSettler
A=.venv/bin/alphasettler
step() { echo "$(date +%s) step $1" >> runs/3d1/run.out; }
fail() { echo "$(date +%s) FAILED $1" >> runs/3d1/run.out; exit 1; }
step 2
$A opening-diagnose \
  --games runs/3c/gen-030 runs/3c/gen-031 runs/3c/gen-032 runs/3c/gen-033 runs/3c/gen-034 \
  --net gen-034 --nets-dir runs/3c/nets --audit 200 --out runs/3d1/diagnose-3c.md 2>&1 | tee runs/3d1/diagnose-3c.log || fail 2
step 3
$A opening-diagnose --games runs/3b-v2/gen-005 --net gen-034 --nets-dir runs/3c/nets \
  --audit 0 --out runs/3d1/diagnose-3b-gen005.md 2>&1 | tee runs/3d1/diagnose-3b-gen005.log || fail 3
step 4
$A opening-train \
  --games runs/3b-v2/gen-001 runs/3b-v2/gen-002 runs/3b-v2/gen-003 runs/3b-v2/gen-004 runs/3b-v2/gen-005 \
          runs/3c/gen-030 runs/3c/gen-031 runs/3c/gen-032 runs/3c/gen-033 runs/3c/gen-034 \
  --out runs/3c/nets/opening.bin --compare-net gen-034 --nets-dir runs/3c/nets 2>&1 | tee runs/3d1/train.log || fail 4
step 5
$A sprt --candidate net:gen-034@100+t+o --baseline net:gen-034@100+t --trades \
  --nets-dir runs/3c/nets --out runs/3d1/sprt-o.jsonl 2>&1 | tee runs/3d1/sprt.log || fail 5
.venv/bin/python3 -c "
import json
rs = [json.loads(l) for l in open('runs/3d1/sprt-o.jsonl')]
print(f'{len(rs)} games, belief resets {sum(r[\"belief_resets\"] for r in rs)}')
" | tee -a runs/3d1/sprt.log
if grep -q '^sprt: H1' runs/3d1/sprt.log; then
  step 6
  $A arena --candidate net:gen-034@600+t+o --baseline net:gen-034@600+t --trades --seeds 200 \
    --seed-start 2000001000 --nets-dir runs/3c/nets --out runs/3d1/confirm-600.jsonl 2>&1 | tee runs/3d1/confirm-600.log || fail 6
else
  echo "$(date +%s) step 6 skipped (no H1)" >> runs/3d1/run.out
fi
step 7
$A opening-diagnose --bot net:gen-034@100+t --seeds 200 --trades --nets-dir runs/3c/nets \
  --out runs/3d1/d4-gen034-100t.md 2>&1 | tee runs/3d1/d4.log || fail 7
$A opening-diagnose --bot net:gen-034@1000+t --seeds 200 --trades --nets-dir runs/3c/nets \
  --out runs/3d1/d4-gen034-1000t.md 2>&1 | tee -a runs/3d1/d4.log || fail 7
$A opening-diagnose --bot net:gen-034@100+t+o --seeds 200 --trades --nets-dir runs/3c/nets \
  --out runs/3d1/d4-gen034-100to.md 2>&1 | tee -a runs/3d1/d4.log || fail 7
echo "$(date +%s) === done" >> runs/3d1/run.out
