"""Small file helpers: content hashes and atomic (temp-then-rename) writes.

A temporary file is `.NAME.tmp-HOST-PID` beside `NAME`: hidden (readers skip dot files) and unique
to the writing process on the writing machine, so two writers never share one (even on a directory
several machines share) and a reader never mistakes a half-written file for a finished one. A
temp's writer is identified by its name: HOST is the last `.tmp-` field up to the final `-`, since
a host name may itself hold `-` and `.`."""

from __future__ import annotations

import glob
import hashlib
import os
import shutil
import socket
from pathlib import Path


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def host() -> str:
    """This machine's name as it appears in temp names (no `/`, so it stays one path component)."""
    return socket.gethostname().replace("/", "") or "localhost"


def temp_path(path: str | Path) -> Path:
    path = Path(path)
    return path.with_name(f".{path.name}.tmp-{host()}-{os.getpid()}")


def temp_writer(path: str | Path) -> tuple[str, int] | None:
    """The (HOST, PID) of a `temp_path` name, or None for a name that is not one (a hostless
    `.NAME.tmp-PID` from before temp names carried the host is not)."""
    name = Path(path).name
    if not name.startswith(".") or ".tmp-" not in name:
        return None
    writer = name.rsplit(".tmp-", 1)[1]
    if "-" not in writer:
        return None
    hostname, pid = writer.rsplit("-", 1)
    if not (hostname and pid.isdecimal()):
        return None
    return hostname, int(pid)


def _running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:  # another user's process
        return True
    except OverflowError:  # no such PID can exist
        return False
    return True


def remove_dead_temps(directory: str | Path, name_glob: str) -> list[Path]:
    """Delete the temps of `name_glob` in `directory` that a dead writer on this machine left (an
    interrupted run's), and this process's own (its caller is between writes); return them. A
    temp of a running process, of another machine (its PID means nothing here) or without a host
    in its name is kept: it may be a write still in progress."""
    me = host()
    removed = []
    for p in Path(directory).glob(f".{name_glob}.tmp-{glob.escape(me)}-*"):
        writer = temp_writer(p)
        if writer is None or writer[0] != me:
            continue
        pid = writer[1]
        if pid == os.getpid() or not _running(pid):
            p.unlink(missing_ok=True)
            removed.append(p)
    return removed


def fsync_file(path: str | Path) -> None:
    """Flush `path`'s data to the disk, so a rename after it never publishes a file a power cut
    could leave empty."""
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_write_bytes(path: str | Path, data: bytes) -> None:
    tmp = temp_path(path)
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_write_text(path: str | Path, text: str) -> None:
    atomic_write_bytes(path, text.encode())


def atomic_copy(src: str | Path, dst: str | Path) -> None:
    tmp = temp_path(dst)
    shutil.copyfile(src, tmp)
    fsync_file(tmp)
    os.replace(tmp, dst)
