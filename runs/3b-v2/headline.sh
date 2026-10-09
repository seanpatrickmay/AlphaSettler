#!/bin/bash
cd /Users/seanmay/Desktop/CurrentProjects/AlphaSettler
echo "=== $(date) best=gen-004 vs ismcts@1000" >> runs/3b-v2/headline.log
/usr/bin/time -p .venv/bin/alphasettler arena --candidate net:best --baseline ismcts@1000 --seeds 200 --no-trades --nets-dir runs/3b-v2/nets --out runs/3b-v2/headline-vs-ismcts-$(date +%Y%m%d-%H%M).jsonl >> runs/3b-v2/headline.log 2>&1
echo "=== $(date) best=gen-004 vs catanatron:alphabeta" >> runs/3b-v2/headline.log
/usr/bin/time -p .venv/bin/alphasettler arena --candidate net:best --baseline catanatron:alphabeta --seeds 200 --nets-dir runs/3b-v2/nets --out runs/3b-v2/headline-vs-alphabeta-$(date +%Y%m%d-%H%M).jsonl >> runs/3b-v2/headline.log 2>&1
echo "=== $(date) done" >> runs/3b-v2/headline.log
