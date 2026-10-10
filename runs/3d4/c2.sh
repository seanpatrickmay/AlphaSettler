#!/usr/bin/env bash
# Plan 12 Task 5 Steps 4 (post-cut) to 7 with preflight C21/C22: arm F checked against arm A, the tables,
# F's best, then criterion 2: net:F@100+t+c vs three net:B@100+t+c from 7,001,000,000.
set -euo pipefail
cd "$(dirname "$0")/../.."
set -a; . runs/3d4/arm.env; set +a
tail -1 "$F_RUNS/F.log" | grep -E 'cut F after 108[0-9][0-9] s'
.venv/bin/python3 -c "
import json
a = json.load(open('$RUNS/$ARM-state-at-cut.json'))
f = json.load(open('$F_RUNS/F-state-at-cut.json'))
for key in a.keys() - {'generations', 'best', 'latest', 'compound'}:
    assert a[key] == f.get(key), key
assert not a.get('compound') and f['compound'] is True
for ga, gf in zip(a['generations'], f['generations']):
    assert ga['gen'] == gf['gen'] and ga['selfplay']['seed_start'] == gf['selfplay']['seed_start'], gf['gen']
    assert ga['selfplay']['games'] == gf['selfplay']['games'] and ga['gate']['seeds'] == gf['gate']['seeds'], gf['gen']
suffix = '+t+c' + ('' if f.get('opening') is None else '+o')
for g in f['generations']:
    assert g['selfplay']['compound'] is True and g['selfplay']['belief_resets'] == 0 and g['gate']['belief_resets'] == 0, g['gen']
    assert g['gate']['candidate'].endswith('@100' + suffix) and g['gate']['baseline'].endswith('@100' + suffix), g['gate']
print(f'F matches $ARM: {len(f[\"generations\"])} generations, the same set-up, seeds and gate; {suffix} gates; no belief resets')
" | tee -a runs/3d4/finals.log
cmp runs/3d2/gen-034-v4.bin "$F_RUNS/F/nets/start.bin" \
  && echo "F started from runs/3d2/gen-034-v4.bin (A's start net at encoder v4, as arms D and E)" | tee -a runs/3d4/finals.log
.venv/bin/python3 runs/p2/tables.py "$ARM" F | tee runs/3d4/tables.md
fbest=$(.venv/bin/python3 -c "import json; print(json.load(open('$F_RUNS/F-state-at-cut.json'))['best'])")
cp "$F_RUNS/F/nets/$fbest.bin" runs/3d4/nets/F.bin
echo "F: $fbest ($F_RUNS/F/nets/$fbest.bin, runs/3d4/nets/F.bin)" | tee -a runs/3d4/finals.log
shasum -a 256 runs/3d4/nets/B.bin runs/3d4/nets/F.bin | tee -a runs/3d4/finals.log
if cmp -s runs/3d4/nets/B.bin runs/3d4/nets/F.bin; then
  echo "F's best is B: criterion 2 is H0 without a match" | tee -a runs/3d4/finals.log; exit 0
fi
.venv/bin/alphasettler sprt --candidate "net:F@100+t+c$O" --baseline "net:B@100+t+c$O" --trades \
  --seed-start 7001000000 --nets-dir runs/3d4/nets --out runs/3d4/sprt-c2.jsonl 2>&1 | tee runs/3d4/sprt-c2.log
.venv/bin/python3 -c "
import json
rs = [json.loads(l) for l in open('runs/3d4/sprt-c2.jsonl')]
print(f'{len(rs)} games, belief resets {sum(r[\"belief_resets\"] for r in rs)}')
" | tee -a runs/3d4/sprt-c2.log
grep -E '^net:B@100\+t\+c(\+o)? vs net:B@100\+t(\+o)?: |^sprt: ' runs/3d4/sprt-c1.log
grep -E '^net:F@100\+t\+c(\+o)? vs net:B@100\+t\+c(\+o)?: |^sprt: ' runs/3d4/sprt-c2.log
echo "c2.sh done"
