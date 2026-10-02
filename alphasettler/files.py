"""Small file helpers: content hashes and atomic (temp-then-rename) writes.

A temporary file is `.NAME.tmp-PID` beside `NAME`: hidden (readers skip dot files) and unique to
the writing process, so two processes never share one and a reader never mistakes a half-written
file for a finished one."""

from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def temp_path(path: str | Path) -> Path:
    path = Path(path)
    return path.with_name(f".{path.name}.tmp-{os.getpid()}")


def remove_temps(directory: str | Path, name_glob: str) -> None:
    """Delete leftover temporary files of `name_glob` (an interrupted writer's) in `directory`."""
    for p in Path(directory).glob(f".{name_glob}.tmp-*"):
        p.unlink()


def atomic_write_bytes(path: str | Path, data: bytes) -> None:
    tmp = temp_path(path)
    tmp.write_bytes(data)
    os.replace(tmp, path)


def atomic_write_text(path: str | Path, text: str) -> None:
    atomic_write_bytes(path, text.encode())


def atomic_copy(src: str | Path, dst: str | Path) -> None:
    tmp = temp_path(dst)
    shutil.copyfile(src, tmp)
    os.replace(tmp, dst)
