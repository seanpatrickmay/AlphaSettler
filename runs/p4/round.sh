#!/bin/bash
# usage: round.sh TAG CANDIDATE_FOR_CATANATRON [curve sims...]
tag=$1; cand=$2; shift 2
cd /Users/seanmay/Desktop/CurrentProjects/AlphaSettler
for n in "$@"; do
  { time .venv/bin/alphasettler arena --candidate ismcts@$n --baseline greedy --seeds 200 --no-trades --out runs/p4/$tag-ismcts@$n-greedy.jsonl; } 2>&1
done
for b in value alphabeta; do
  { time .venv/bin/alphasettler arena --candidate $cand --baseline catanatron:$b --seeds 200 --out runs/p4/$tag-$cand-$b.jsonl; } 2>&1
done
