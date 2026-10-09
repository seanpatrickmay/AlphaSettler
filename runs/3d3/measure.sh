#!/usr/bin/env bash
# Plan 11 Task 3 Steps 5-6: the SPRT against the control, the extra SPRT against D (ledger ruling),
# then each net against three copies of each style on 200 fixed seeds, and the pooled summary (preflight C11).
set -euo pipefail
cd "$(dirname "$0")/../.."
set -a; . runs/3d3/control.env; set +a
.venv/bin/alphasettler sprt --candidate "net:E@100+t$O" --baseline "net:$CTRL@100+t$O" --trades \
  --seed-start 6000000000 --nets-dir runs/3d3/nets --out runs/3d3/sprt-E.jsonl 2>&1 | tee runs/3d3/sprt-E.log
.venv/bin/python3 -c "
import json
rs = [json.loads(l) for l in open('runs/3d3/sprt-E.jsonl')]
print(f'{len(rs)} games, belief resets {sum(r[\"belief_resets\"] for r in rs)}')
" | tee -a runs/3d3/sprt-E.log
.venv/bin/alphasettler sprt --candidate "net:E@100+t$O" --baseline "net:D@100+t$O" --trades \
  --seed-start 6002000000 --nets-dir runs/3d3/nets --out runs/3d3/sprt-E-vs-D.jsonl 2>&1 | tee runs/3d3/sprt-E-vs-D.log
.venv/bin/python3 -c "
import json
rs = [json.loads(l) for l in open('runs/3d3/sprt-E-vs-D.jsonl')]
print(f'{len(rs)} games, belief resets {sum(r[\"belief_resets\"] for r in rs)}')
" | tee -a runs/3d3/sprt-E-vs-D.log
for net in E "$CTRL"; do
  for style in roads army production; do
    .venv/bin/alphasettler arena --candidate "net:$net@100+t$O" --baseline "style:$style@100+t" --trades \
      --seeds 200 --seed-start 6001000000 --nets-dir runs/3d3/nets --out "runs/3d3/robust-$net-$style.jsonl" \
      2>&1 | tee "runs/3d3/robust-$net-$style.log"
  done
done
.venv/bin/python3 - <<'PY' | tee runs/3d3/robust.log
import json, os
from alphasettler.arena import format_summary
from alphasettler.stats import summarize
ctrl, o = os.environ["CTRL"], os.environ["O"]
for net in ("E", ctrl):
    recs = []
    for i, style in enumerate(("roads", "army", "production")):
        rs = [json.loads(line) for line in open(f"runs/3d3/robust-{net}-{style}.jsonl")]
        print(format_summary(f"net:{net}@100+t{o} vs style:{style}@100+t", summarize(rs)))
        recs += [{**r, "seed": r["seed"] + i * 1_000_000} for r in rs]
    print(format_summary(f"net:{net}@100+t{o} vs the three styles (pooled, seed x style units)", summarize(recs)))
    print(f"belief resets: {sum(r['belief_resets'] for r in recs)}")
PY
echo "measure.sh done"
