#!/bin/bash
cd /Users/seanmay/Desktop/CurrentProjects/AlphaSettler
for n in ft5lr1e3v256-10m n48-10m n96r3v256-30m gen-005; do runs/fast/eval.sh $n 100 gen-004 200; done
echo "=== evalq1 done" >> runs/fast/eval.log
