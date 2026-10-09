#!/usr/bin/env bash
# Compile the arena harness against the pinned JSettlers2 jar (plain javac; no Gradle or Maven).
set -euo pipefail
cd "$(dirname "$0")/.."
JAR=jsettlers/lib/jsettlers-2.6.10-server.jar
bash jsettlers/check-api.sh
rm -rf jsettlers/build/classes
mkdir -p jsettlers/build/classes
find jsettlers/src -name '*.java' | sort > jsettlers/build/sources.txt
for f in $(cat jsettlers/build/sources.txt); do
  [ "$(sed -n 2p "$f")" = "// SPDX-License-Identifier: GPL-3.0-or-later" ] || { echo "$f lacks the GPL header" >&2; exit 1; }
done
javac --release 17 -Xlint:all,-serial,-deprecation,-removal,-processing,-path -Werror -encoding UTF-8 \
  -cp "$JAR" -d jsettlers/build/classes @jsettlers/build/sources.txt
echo "built $(wc -l < jsettlers/build/sources.txt | tr -d ' ') sources into jsettlers/build/classes"
