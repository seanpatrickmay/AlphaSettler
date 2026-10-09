"""JSettlers2 adapter. Python talks to the GPL harness under jsettlers/ only through JSON lines
on its JVM's stdin and stdout (jvm.py); nothing here imports Catanatron or the Catanatron modules
of `oracle`. This module only names paths, so a caller can say what is missing before any JVM runs."""

from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VERSION = "2.6.10"
JAR = ROOT / "jsettlers" / "lib" / f"jsettlers-{VERSION}-server.jar"
JAR_SHA256 = "9b26cb20ec943213600795d660fad2da32fc72629c348e180bd171cc26d6b4e2"
CLASSES = ROOT / "jsettlers" / "build" / "classes"
LOGS = ROOT / "runs" / "jsettlers" / "logs"
PACKAGE = "alphasettler.jsettlers"


def java_command(main: str, *args: str) -> list[str]:
    """The command that runs harness class `main` (e.g. "ArenaMain") with `args`."""
    return ["java", "-Xss4m", "-cp", f"{CLASSES}:{JAR}", f"{PACKAGE}.{main}", *args]


def missing(main: str = "ArenaMain") -> str | None:
    """What is missing to run harness class `main`, with the command that provides it, or None."""
    if shutil.which("java") is None:
        return "java is not on PATH (install a JDK, version 17 or later)"
    if not JAR.is_file():
        return f"{JAR} is missing; run: bash scripts/fetch-jsettlers.sh && bash jsettlers/build.sh"
    if not (CLASSES / Path(*PACKAGE.split(".")) / f"{main}.class").is_file():
        return f"the harness class {main} is not built; run: bash jsettlers/build.sh"
    return None
