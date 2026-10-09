#!/bin/bash
cd /Users/seanmay/Desktop/CurrentProjects/AlphaSettler
for n in n96r3v256-90m n96r3v256-60m; do runs/fast/eval.sh $n 100 gen-004 200; done
echo "=== evalq2 done" >> runs/fast/eval.log
