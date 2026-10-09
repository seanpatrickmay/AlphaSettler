#!/usr/bin/env bash
# Fail, naming the line, unless every member jsettlers/api.txt (or the file given) lists is in the pinned jar.
set -euo pipefail
cd "$(dirname "$0")/.."
JAR=jsettlers/lib/jsettlers-2.6.10-server.jar
LIST=${1:-jsettlers/api.txt}
[ -f "$JAR" ] || { echo "missing $JAR; run: bash scripts/fetch-jsettlers.sh" >&2; exit 1; }
cache=$(mktemp -d)
trap 'rm -rf "$cache"' EXIT
fail=0
n=0
while IFS= read -r line; do
  case "$line" in ''|'#'*) continue ;; esac
  cls=${line%% | *}
  want=${line#* | }
  [ -f "$cache/$cls" ] || javap -p -cp "$JAR" "$cls" > "$cache/$cls" 2>&1 || true
  n=$((n + 1))
  if ! grep -qF -- "$want" "$cache/$cls"; then
    echo "check-api: $cls has no member matching: $want" >&2
    fail=1
  fi
done < "$LIST"
[ "$fail" = 0 ] && echo "check-api: $n members present"
exit "$fail"
