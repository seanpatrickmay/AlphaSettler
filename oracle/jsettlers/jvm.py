"""One harness JVM (jsettlers/ArenaMain) driven over JSON lines on its stdin and stdout. Its stderr,
with JSettlers' own output, goes to a log under runs/jsettlers/logs/ that every error names. The JVM
exits when its stdin closes, so it never outlives the process that started it."""

from __future__ import annotations

import itertools
import json
import os
import queue
import subprocess
import threading
import time
from pathlib import Path

from oracle.jsettlers import LOGS, java_command, missing

READY_TIMEOUT = 120.0
_COUNT = itertools.count()


class JvmError(RuntimeError):
    """The harness reported a failure (and exited)."""


class Jvm:
    def __init__(self, mode: str = "stock", log: Path | None = None):
        why = missing()
        if why is not None:
            raise RuntimeError(f"the JSettlers harness is not ready: {why}")
        LOGS.mkdir(parents=True, exist_ok=True)
        self.log = Path(log) if log is not None else LOGS / f"{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}-{next(_COUNT)}.log"
        self._err = open(self.log, "w")
        self.proc = subprocess.Popen(java_command("ArenaMain", mode), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=self._err, text=True, encoding="utf-8", bufsize=1)
        self._lines: queue.Queue[str | None] = queue.Queue()
        threading.Thread(target=self._read, daemon=True).start()
        try:
            self.ready = self.recv(READY_TIMEOUT)
        except BaseException:
            self.close()
            raise
        if self.ready.get("t") != "ready":
            self.close()
            raise JvmError(f"the JSettlers JVM did not start: {self.ready}; see {self.log}")

    def _read(self) -> None:
        for line in self.proc.stdout:
            self._lines.put(line)
        self._lines.put(None)

    def send(self, msg: dict) -> None:
        try:
            self.proc.stdin.write(json.dumps(msg) + "\n")
            self.proc.stdin.flush()
        except (BrokenPipeError, ValueError, OSError):
            raise EOFError(f"the JSettlers JVM is gone (exit code {self.proc.poll()}); see {self.log}") from None

    def recv(self, timeout: float) -> dict:
        try:
            line = self._lines.get(timeout=timeout)
        except queue.Empty:
            raise TimeoutError(f"the JSettlers JVM sent nothing for {timeout:.1f} s; see {self.log}") from None
        if line is None:
            self.proc.wait(timeout=30)
            raise EOFError(f"the JSettlers JVM exited (code {self.proc.returncode}); see {self.log}")
        msg = json.loads(line)
        if msg.get("t") == "error":
            first = (msg.get("what") or "").strip().splitlines()[:1]
            raise JvmError(f"the JSettlers JVM failed: {first[0] if first else 'no detail'}; see {self.log}")
        return msg

    def close(self) -> None:
        if self.proc.poll() is None:
            try:
                self.proc.stdin.close()
            except OSError:
                pass
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        self._err.close()

    def __enter__(self) -> "Jvm":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
