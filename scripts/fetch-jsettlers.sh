#!/usr/bin/env bash
# Download JSettlers2 2.6.10's server jar (GPL-3.0-or-later; never committed) and check its sha256.
set -euo pipefail
cd "$(dirname "$0")/.."
URL=https://github.com/jdmonin/JSettlers2/releases/download/release-2.6.10/jsettlers-2.6.10-server.jar
SHA=9b26cb20ec943213600795d660fad2da32fc72629c348e180bd171cc26d6b4e2
OUT=jsettlers/lib/jsettlers-2.6.10-server.jar
mkdir -p jsettlers/lib
if [ -f "$OUT" ] && echo "$SHA  $OUT" | shasum -a 256 -c --status; then
  echo "$OUT is present and checks out"
  exit 0
fi
curl -fsSL "$URL" -o "$OUT.part"
echo "$SHA  $OUT.part" | shasum -a 256 -c --status || { rm -f "$OUT.part"; echo "sha256 mismatch for $URL" >&2; exit 1; }
mv -f "$OUT.part" "$OUT"
echo "fetched $OUT"
