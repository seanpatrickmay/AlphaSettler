#!/usr/bin/env bash
# Head-to-head random-player throughput: AlphaSettler vs catan-rl vs Catanatron.
# Third-party engines are cloned into bench/compare/.cache (gitignored), never vendored.
# Usage: bench/compare/run.sh [games_per_engine]   (default 20000; Catanatron runs 1/50th)
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
CACHE="$HERE/.cache"
GAMES="${1:-20000}"
source "$HOME/.cargo/env"
mkdir -p "$CACHE"

[ -d "$CACHE/catan-rl" ] || git clone -q --depth 1 https://github.com/Eli6th/catan-rl.git "$CACHE/catan-rl"
[ -d "$CACHE/catanatron" ] || git clone -q --depth 1 https://github.com/bcollazo/catanatron.git "$CACHE/catanatron"
if [ ! -x "$CACHE/venv/bin/python3" ]; then
  python3 -m venv "$CACHE/venv"
  "$CACHE/venv/bin/pip3" install -q "$CACHE/catanatron"
fi

echo "== machine: $(sysctl -n machdep.cpu.brand_string), $(sysctl -n hw.ncpu) cores"
echo "== catan-rl $(git -C "$CACHE/catan-rl" rev-parse --short HEAD), catanatron $(git -C "$CACHE/catanatron" rev-parse --short HEAD), alphasettler $(git -C "$ROOT" rev-parse --short HEAD)"

echo "== AlphaSettler, domestic trades off"
(cd "$ROOT" && cargo run -q --release -p settler-engine --example throughput "$GAMES" 0)
echo "== AlphaSettler, domestic trades on (<=3 offers/turn, <=2 cards per side: up to 400 offers)"
(cd "$ROOT" && cargo run -q --release -p settler-engine --example throughput "$GAMES" 3 2)
echo "== AlphaSettler, domestic trades on (<=3 offers/turn, 1 card for 1: 20 offers)"
(cd "$ROOT" && cargo run -q --release -p settler-engine --example throughput "$GAMES" 3 1)

echo "== catan-rl, random players (its trade menu: 1-2 of one resource for 1 of another, <=3 offers)"
(cd "$CACHE/catan-rl/rust" && cargo build -q --release -p catan-sim \
  && ./target/release/catan-sim --games "$GAMES" --players R,R,R,R --single-thread --seed 1 | sed -n '5,10p' \
  && ./target/release/catan-sim --games "$GAMES" --players R,R,R,R --seed 1 | sed -n '4,6p')

echo "== Catanatron, random players"
"$CACHE/venv/bin/python3" "$HERE/catanatron_bench.py" "$((GAMES / 50))"
