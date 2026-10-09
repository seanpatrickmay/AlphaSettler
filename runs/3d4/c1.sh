#!/usr/bin/env bash
# Plan 12 Task 5 Steps 1-3 with preflight C19/C20: the best arm from the recorded verdicts (3d-1b, 3d-2
# and 3d-3 all H0, so arm A), B = A's best at encoder v4, then criterion 1: net:B@100+t+c vs net:B@100+t.
set -euo pipefail
cd "$(dirname "$0")/../.."
ls runs/p2/arm.sh runs/p2/tables.py docs/bot/results-3d1b.md docs/bot/results-3d2.md docs/bot/results-3d3.md \
  runs/3d3/sprt-E.log runs/p2/E-state-at-cut.json runs/3d2/nets/A.bin runs/3d2/gen-034-v4.bin > /dev/null
test "$(ls runs/3d3/pool/*.bin | wc -l | tr -d ' ')" = 10
grep -n '^sprt: ' runs/p2/sprt-C-vs-A.log runs/3d2/sprt-D.log runs/3d3/sprt-E.log | tee runs/3d4/verdicts.log
grep -q '^sprt: H1' runs/3d3/sprt-E.log && { echo "3d-3 was H1: this script is the A branch only" >&2; exit 1; }
mkdir -p runs/3d4/nets
printf 'ARM=A\nRUNS=runs/p2\nB_SRC=runs/3d2/nets/A.bin\n' > runs/3d4/arm.env
set -a; . runs/3d4/arm.env; set +a
test "$RUNS" = runs/p2 && test -d "$RUNS/$ARM" && test -f "$RUNS/$ARM.log" && test -f "$RUNS/$ARM-state-at-cut.json" \
  && test -f "$B_SRC"
best=$(.venv/bin/python3 -c "import json; print(json.load(open('$RUNS/$ARM-state-at-cut.json'))['best'])")
O=$(.venv/bin/python3 -c "import json; print('' if json.load(open('$RUNS/$ARM-state-at-cut.json')).get('opening') is None else '+o')")
cp "$B_SRC" runs/3d4/nets/B.bin
test "$best" = gen-040 && grep -q 'wrote runs/3d2/nets/A.bin' runs/3d2/finals.log
ARGS="--start-net runs/3d2/gen-034-v4.bin"
if [ -n "$O" ]; then cp runs/3c/nets/opening.bin runs/3d4/nets/opening.bin; fi
printf 'O=%s\nARGS="%s"\n' "$O" "$ARGS" >> runs/3d4/arm.env
.venv/bin/python3 -c "
import json
st = json.load(open('$RUNS/$ARM-state-at-cut.json'))
assert not st.get('compound') and st.get('gumbel') is None, 'the best arm used neither --compound nor --gumbel'
print('best arm $ARM ($RUNS/$ARM): B is', st['best'], 'at encoder v4 ($B_SRC) | opening', st.get('opening'),
      '| arm F set-up arguments: $ARGS --compound')
" | tee runs/3d4/best-arm.log
shasum -a 256 runs/3d4/nets/*.bin runs/3d2/gen-034-v4.bin | tee -a runs/3d4/best-arm.log
set -a; . runs/3d4/arm.env; set +a
.venv/bin/alphasettler sprt --candidate "net:B@100+t+c$O" --baseline "net:B@100+t$O" --trades \
  --seed-start 7000000000 --nets-dir runs/3d4/nets --out runs/3d4/sprt-c1.jsonl 2>&1 | tee runs/3d4/sprt-c1.log
.venv/bin/python3 -c "
import json
rs = [json.loads(l) for l in open('runs/3d4/sprt-c1.jsonl')]
n = len(rs)
seat = [r['candidate_seat'] for r in rs]
searched = sum(r['searches'][c] for r, c in zip(rs, seat))
compound = sum(r['compound_decisions'][c] for r, c in zip(rs, seat))
with_c = sum(r['bank_trades'][c] for r, c in zip(rs, seat)) / n
without = sum(sum(r['bank_trades']) - r['bank_trades'][c] for r, c in zip(rs, seat)) / (3 * n)
print(f'{n} games, belief resets {sum(r[\"belief_resets\"] for r in rs)}')
print(f'decisions with a compound move: {compound} of {searched} searched decisions of the +c seat ({compound / searched:.3f})')
print(f'bank trades per game: {with_c:.2f} by the +c seat, {without:.2f} by each seat without +c')
" | tee -a runs/3d4/sprt-c1.log
echo "c1.sh done"
