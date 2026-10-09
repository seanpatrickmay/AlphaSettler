#!/bin/bash
# Plan 12, the Task 1 review's Minor 2 (ledger ruling): plain search speed (no +c) at HEAD against the
# commit before Task 1, same net, alternating builds, one core, batch 8, 1,000 simulations.
set -euo pipefail
cd "$(dirname "$0")/../.."
base=37406c6
rm -rf runs/3d4/base-src && mkdir -p runs/3d4/base-src
git archive "$base" | tar -x -C runs/3d4/base-src
PATH=$HOME/.cargo/bin:$PATH cargo build -q --release --manifest-path runs/3d4/base-src/Cargo.toml \
  --target-dir runs/3d4/base-target -p settler-bots --example net_speed
PATH=$HOME/.cargo/bin:$PATH cargo build -q --release -p settler-bots --example net_speed
echo "base $(git log -1 --format='%h %s' $base)" | tee runs/3d4/speed-plain.log
echo "head $(git log -1 --format='%h %s')" | tee -a runs/3d4/speed-plain.log
pgrep -fl "arm.sh|alphasettler (sprt|arena|fast-loop)|net_speed" || echo idle | tee -a runs/3d4/speed-plain.log
for i in 1 2 3; do
  echo "base run $i: $(runs/3d4/base-target/release/examples/net_speed speed runs/3d2/gen-034-v4.bin 500 1000 8)" | tee -a runs/3d4/speed-plain.log
  echo "head run $i: $(target/release/examples/net_speed speed runs/3d2/gen-034-v4.bin 500 1000 8)" | tee -a runs/3d4/speed-plain.log
done
