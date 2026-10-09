#!/bin/bash
cd /Users/seanmay/Desktop/CurrentProjects/AlphaSettler
for k in 10 30; do
  echo "=== $(date +%H:%M:%S) gen-004@100+s$k vs gen-004@100, 200 fresh seeds" >> runs/fast/setup-test.log
  /usr/bin/time -p .venv/bin/alphasettler arena --candidate net:gen-004@100+s$k --baseline net:gen-004@100 --seeds 200 \
    --seed-start 2000000000 --no-trades --nets-dir runs/fast/nets --out runs/fast/setup-s$k@100.jsonl >> runs/fast/setup-test.log 2>&1
done
echo "=== setup test done" >> runs/fast/setup-test.log
