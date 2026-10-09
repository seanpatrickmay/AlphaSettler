#!/bin/bash
cd /Users/seanmay/Desktop/CurrentProjects/AlphaSettler
echo "=== $(date) best=gen-004 vs ismcts@1000, fresh seeds from 2000000000" >> runs/3b-v2/headline.log
/usr/bin/time -p nice -n 10 .venv/bin/alphasettler arena --candidate net:best --baseline ismcts@1000 --seeds 200 --seed-start 2000000000 --no-trades --nets-dir runs/3b-v2/nets --out runs/3b-v2/headline-vs-ismcts-fresh-$(date +%Y%m%d-%H%M).jsonl >> runs/3b-v2/headline.log 2>&1
echo "=== $(date) fresh done" >> runs/3b-v2/headline.log
