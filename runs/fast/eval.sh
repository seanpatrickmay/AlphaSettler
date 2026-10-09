#!/bin/bash
# usage: eval.sh NAME [SIMS] [BASELINE] [SEEDS] -- NAME@SIMS vs three BASELINE@SIMS, no trades, fresh seeds from 2e9 (never used by training or gates)
cd /Users/seanmay/Desktop/CurrentProjects/AlphaSettler
N=$1; S=${2:-100}; B=${3:-gen-004}; K=${4:-200}
echo "=== $(date +%H:%M:%S) $N@$S vs $B@$S, $K seeds" >> runs/fast/eval.log
/usr/bin/time -p .venv/bin/alphasettler arena --candidate net:$N@$S --baseline net:$B@$S --seeds $K --seed-start 2000000000 --no-trades \
  --nets-dir runs/fast/nets --out runs/fast/eval-$N@$S-vs-$B.jsonl >> runs/fast/eval.log 2>&1
