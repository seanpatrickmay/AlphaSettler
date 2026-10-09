#!/bin/bash
cd /Users/seanmay/Desktop/CurrentProjects/AlphaSettler
if [ -e runs/3c/state.json ]; then
  .venv/bin/alphasettler fast-loop --dir runs/3c --generations 39 >> runs/3c/fast-loop.log 2>&1
else
  .venv/bin/alphasettler fast-loop --dir runs/3c --generations 39 --first 30 --trades \
    --seed-dirs runs/3b-v2/gen-002 runs/3b-v2/gen-003 runs/3b-v2/gen-004 runs/3b-v2/gen-005 \
    --start-net runs/3c/gen-004-v3.bin >> runs/3c/fast-loop.log 2>&1
fi
echo "=== loop exit $?" >> runs/3c/fast-loop.log
