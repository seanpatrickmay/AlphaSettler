#!/bin/bash
# Plan 10 Task 4 Steps 1-4 (controller runs this): build, the v3 build, upgrade + output comparison, speed.
set -o pipefail
cd /Users/seanmay/Desktop/CurrentProjects/AlphaSettler
fail() { echo "FAILED $1" | tee -a runs/3d2/steps.out; exit 1; }
echo "== step 1" | tee -a runs/3d2/steps.out
git log -1 --format=%s | tee -a runs/3d2/steps.out
PATH=$HOME/.cargo/bin:$PATH cargo build --release -p settler-bots --example net_speed || fail 1
shasum -a 256 runs/3c/nets/gen-034.bin runs/p2/nets/A.bin runs/3c/nets/opening.bin | tee -a runs/3d2/steps.out
grep -q '^sprt: H1' runs/p2/sprt-C-vs-A.log && echo "branch O: control C, --opening" | tee -a runs/3d2/steps.out || echo "branch N: control A" | tee -a runs/3d2/steps.out
rm -rf target/wheels-3d2 && PATH=$HOME/.cargo/bin:$PATH .venv/bin/maturin build --release -q -o target/wheels-3d2 \
  && unzip -q -o target/wheels-3d2/alphasettler-*.whl 'alphasettler/_engine*.so' -d target/wheels-3d2/x \
  && cp target/wheels-3d2/x/alphasettler/_engine.cpython-312-darwin.so alphasettler/._engine.new.so \
  && mv -f alphasettler/._engine.new.so alphasettler/_engine.cpython-312-darwin.so || fail 1-rebuild
echo "== step 2" | tee -a runs/3d2/steps.out
base=$(git rev-parse "$(git log -1 --format=%H --grep='^engine: longest_trail')^")
mkdir -p runs/3d2/v3-src
git log -1 --format='%h %s' "$base" | tee runs/3d2/base.txt | tee -a runs/3d2/steps.out
git archive "$base" | tar -x -C runs/3d2/v3-src
PATH=$HOME/.cargo/bin:$PATH cargo build --release --manifest-path runs/3d2/v3-src/Cargo.toml \
  --target-dir runs/3d2/v3-target -p settler-bots --example net_speed || fail 2
(cd runs/3d2/v3-src && CARGO_TARGET_DIR=../v3-target PATH=$HOME/.cargo/bin:$PATH ../../../.venv/bin/maturin build --release -q -o ../v3-wheels) || fail 2-wheel
unzip -q -o runs/3d2/v3-wheels/alphasettler-*.whl 'alphasettler/_engine*.so' -d runs/3d2/v3-src
PYTHONPATH=runs/3d2/v3-src .venv/bin/python3 -c "import alphasettler, alphasettler._engine as e; print(alphasettler.__file__, hex(e.NET_ENCODING_HASH))" | tee -a runs/3d2/steps.out
.venv/bin/python3 -c "import alphasettler, alphasettler._engine as e; print(alphasettler.__file__, hex(e.NET_ENCODING_HASH))" | tee -a runs/3d2/steps.out
echo "== step 3" | tee -a runs/3d2/steps.out
.venv/bin/alphasettler net-upgrade runs/3c/nets/gen-034.bin runs/3d2/gen-034-v4.bin | tee runs/3d2/upgrade.log || fail 3
shasum -a 256 runs/3c/nets/gen-034.bin runs/3d2/gen-034-v4.bin | tee -a runs/3d2/upgrade.log
PYTHONPATH=runs/3d2/v3-src .venv/bin/python3 runs/3d2/forward_dump.py runs/3c/nets/gen-034.bin runs/3d2/fwd-v3 | tee runs/3d2/forward.log || fail 3-v3
.venv/bin/python3 runs/3d2/forward_dump.py runs/3d2/gen-034-v4.bin runs/3d2/fwd-v4 | tee -a runs/3d2/forward.log || fail 3-v4
.venv/bin/python3 runs/3d2/forward_compare.py runs/3d2/fwd-v3 runs/3d2/fwd-v4 | tee -a runs/3d2/forward.log || fail 3-compare
echo "== step 4" | tee -a runs/3d2/steps.out
pgrep -fl "alphasettler|settler" | grep -v steps1-4 || echo idle
for i in 1 2 3; do
  echo "v3 run $i: $(runs/3d2/v3-target/release/examples/net_speed speed runs/3c/nets/gen-034.bin 500 1000 8)" | tee -a runs/3d2/speed.log
  echo "v4 run $i: $(target/release/examples/net_speed speed runs/3d2/gen-034-v4.bin 500 1000 8)" | tee -a runs/3d2/speed.log
done
echo "== steps 1-4 done" | tee -a runs/3d2/steps.out
