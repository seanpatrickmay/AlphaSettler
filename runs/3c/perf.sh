#!/bin/bash
# gen-034 measured three ways, sequentially (spec 3c criteria at 100 simulations, quick read).
cd "$(dirname "$0")/../.."
A=".venv/bin/alphasettler sprt --nets-dir runs/3c/nets"
echo "=== $(date +%H:%M) trading pays: gen-034@100+t vs answer-only gen-034@100 (trading game)"
$A --candidate net:gen-034@100+t --baseline net:gen-034@100 --trades --out runs/3c/perf-offers.jsonl
echo "=== $(date +%H:%M) progress: gen-034@100+t vs gen-004@100 (trading game, gen-4 answers only)"
$A --candidate net:gen-034@100+t --baseline net:gen-004@100 --trades --out runs/3c/perf-vs-gen4.jsonl
echo "=== $(date +%H:%M) no-trade game: gen-034@100 vs gen-004@100"
$A --candidate net:gen-034@100 --baseline net:gen-004@100 --no-trades --out runs/3c/perf-notrade.jsonl
echo "=== $(date +%H:%M) perf done"
