#!/bin/bash
cd /Users/seanmay/Desktop/CurrentProjects/AlphaSettler
while kill -0 56722 2>/dev/null; do sleep 15; done
A=.venv/bin/alphasettler
echo "=== $(date +%H:%M:%S) gen-004@200 vs gen-004@100 (uniform 2x)" >> runs/search/sweep.log
$A sprt --candidate net:gen-004@200 --baseline net:gen-004@100 --no-trades --nets-dir runs/fast/nets --out runs/search/u200.jsonl >> runs/search/sweep.log 2>&1
