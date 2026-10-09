#!/bin/bash
cd /Users/seanmay/Desktop/CurrentProjects/AlphaSettler
while kill -0 42201 2>/dev/null; do sleep 20; done
.venv/bin/alphasettler fast-loop --dir runs/fast-loop --generations 30 --threads 10 \
  --seed-dirs runs/3b-v2/gen-001 runs/3b-v2/gen-002 runs/3b-v2/gen-003 runs/3b-v2/gen-004 runs/3b-v2/gen-005 \
  --start-net runs/fast/nets/ft5lr1e3v256-10m.bin --first 6 >> runs/fast-loop.log 2>&1
