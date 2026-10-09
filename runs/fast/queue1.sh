#!/bin/bash
cd /Users/seanmay/Desktop/CurrentProjects/AlphaSettler
P="runs/3b-v2/gen-001 runs/3b-v2/gen-002 runs/3b-v2/gen-003 runs/3b-v2/gen-004 runs/3b-v2/gen-005"
A=.venv/bin/alphasettler
$A fast-train $P --out runs/fast/nets/n48-10m.bin --minutes 10 --threads 4 > runs/fast/n48-10m.log 2>&1
$A fast-train $P --out runs/fast/nets/ft5-10m.bin --minutes 10 --threads 4 --init runs/fast/nets/gen-005.bin --lr 3e-4 --warmup 0 > runs/fast/ft5-10m.log 2>&1
$A fast-train $P --out runs/fast/nets/n96r3-10m.bin --minutes 10 --threads 4 --width 96 --rounds 3 > runs/fast/n96r3-10m.log 2>&1
$A fast-score runs/3b-v2/gen-005 --nets runs/fast/nets/{n48-10m,ft5-10m,n96r3-10m}.bin > runs/fast/score-q1.log 2>&1
echo done >> runs/fast/score-q1.log
