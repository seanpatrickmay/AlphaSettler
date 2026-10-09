#!/bin/bash
# Plan 8 Task 6 Steps 2-4: criterion 1, Gumbel (+g, no draws) vs PUCT at 32, 100 and 600 simulations.
cd "$(dirname "$0")/../.."
.venv/bin/alphasettler sprt --candidate net:gen-034@32+g --baseline net:gen-034@32 --no-trades \
  --seed-start 3000000000 --nets-dir runs/3c/nets --out runs/p2/sprt-32.jsonl 2>&1 | tee runs/p2/sprt-32.log
.venv/bin/alphasettler sprt --candidate net:gen-034@100+g --baseline net:gen-034@100 --no-trades \
  --seed-start 3000001000 --nets-dir runs/3c/nets --out runs/p2/sprt-100.jsonl 2>&1 | tee runs/p2/sprt-100.log
.venv/bin/alphasettler arena --candidate net:gen-034@600+g --baseline net:gen-034@600 --no-trades --seeds 200 \
  --seed-start 3000002000 --nets-dir runs/3c/nets --out runs/p2/arena-600.jsonl 2>&1 | tee runs/p2/arena-600.log
for f in runs/p2/sprt-32.jsonl runs/p2/sprt-100.jsonl; do
  .venv/bin/python3 -c "
import json, sys
rs = [json.loads(l) for l in open(sys.argv[1])]
print(f'{sys.argv[1]}: {len(rs)} games, belief resets {sum(r[\"belief_resets\"] for r in rs)}')
" "$f" | tee -a runs/p2/resets.log
done
echo "=== crit1 done"
