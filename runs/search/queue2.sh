#!/bin/bash
cd /Users/seanmay/Desktop/CurrentProjects/AlphaSettler
A=.venv/bin/alphasettler
run() { echo "=== $(date +%H:%M:%S) $1 vs $2" >> runs/search/sweep.log
  $A sprt --candidate $1 --baseline $2 --no-trades --nets-dir runs/fast/nets --out runs/search/$3.jsonl >> runs/search/sweep.log 2>&1; }
run net:gen-004@400 net:gen-004@200 u400v200
run net:gen-004@100+r4 net:gen-004@100 r4
echo "=== queue2 done" >> runs/search/sweep.log
