"""Training data (spec Section 4): self-play record files -> the Rust sampler -> tensors.

A generation directory holds finished record files `*.jsonl.gz` (other machines may drop more in:
the scale-out path). Hidden temp files (`files.temp_path`) are never read. A file cut short or
garbled is kept up to the damage; one holding a record that does not replay (or is malformed) is
skipped whole; empty files and files without decisions are noted. Each case prints a notice.
A seed already read in the same generation (a second writer playing the same seeds) is read
once. Records stream into the `TrainingSet` a file at a time, so the window is never held as
Python dicts; the Rust side checks every field of every record."""

from __future__ import annotations

import gzip
import json
import zlib
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import torch

from alphasettler import files, records
from alphasettler._engine import NET_HIDDEN_LEN, NET_PUBLIC_LEN, TrainingSet, action_space_size

ACTIONS = action_space_size()
SEATS = 4
# Seeds a duplicate notice lists before it elides the rest.
NOTICE_SEEDS = 5

# What reading a cut-short or garbled file raises: gzip framing, deflate data, text, JSON. Only
# these end a file early; anything else (an OSError such as a permission error, MemoryError, a
# bug) propagates.
_READ_ERRORS = (EOFError, gzip.BadGzipFile, zlib.error, UnicodeDecodeError, json.JSONDecodeError)


def record_files(gen_dir: str | Path) -> list[Path]:
    """Finished record files in `gen_dir`, by name; temporary (dot) files are not."""
    return sorted(p for p in Path(gen_dir).glob("*.jsonl.gz") if not p.name.startswith("."))


@dataclass
class _FileStats:
    added: set[int] = field(default_factory=set)
    duplicates: list[int] = field(default_factory=list)
    read_error: str | None = None
    # An exception raised inside the stream other than a read error. `TrainingSet.extend` wraps
    # it in a ValueError whose `__cause__` is this object, which is how a bug is told apart from
    # a bad record.
    bug: BaseException | None = None


def _games(path: Path, seen: set[int], stats: _FileStats) -> Iterator[dict]:
    """The file's records as read, minus seeds in `seen` or already streamed (collected in
    `stats.added`). A read error ends the stream and is kept in `stats.read_error`. Records are
    passed on unexamined beyond their seed: the Rust side checks every field."""
    try:
        reader = records.read(path)
        while True:
            try:
                g = next(reader)
            except StopIteration:
                return
            except _READ_ERRORS as e:
                stats.read_error = f"{type(e).__name__}: {e}"
                return
            seed = g.get("seed") if isinstance(g, dict) else None
            if isinstance(seed, int) and (seed in seen or seed in stats.added):
                stats.duplicates.append(seed)
                continue
            if isinstance(seed, int):
                stats.added.add(seed)
            yield g
    except Exception as e:
        stats.bug = e
        raise


def _seed_list(seeds: list[int]) -> str:
    shown = ", ".join(str(s) for s in seeds[:NOTICE_SEEDS])
    return shown + (f", ... ({len(seeds) - NOTICE_SEEDS} more)" if len(seeds) > NOTICE_SEEDS else "")


def _read_file(path: Path, gen_name: str, data: TrainingSet, seen: set[int], notices: list[str]) -> None:
    stats = _FileStats()
    games, decisions = data.games, data.decisions
    failure = None
    try:
        data.extend(_games(path, seen, stats))
    except ValueError as e:
        failure = e
    if failure is not None:
        if stats.bug is not None and failure.__cause__ is stats.bug:
            raise stats.bug  # our own error inside the stream, not a bad record
        # A record that is malformed or does not replay: `extend` added nothing from this file.
        notices.append(f"{path}: skipped the whole file: {failure}")
        return
    seen |= stats.added
    kept, new_decisions = data.games - games, data.decisions - decisions
    if stats.duplicates:
        n = len(stats.duplicates)
        what = f"all {n} games duplicate earlier seeds" if kept == 0 and stats.read_error is None \
            else f"skipped {n} games duplicating earlier seeds"
        notices.append(f"{path}: {what} (already in {gen_name}: {_seed_list(stats.duplicates)})")
    if stats.read_error is not None:
        notices.append(f"{path}: kept {kept} games, skipped an unreadable rest ({stats.read_error})")
    elif kept == 0 and not stats.duplicates:
        notices.append(f"{path}: no games")
    if kept and new_decisions == 0:
        notices.append(f"{path}: {kept} games but no decisions")


def read_generation(gen_dir: str | Path, data: TrainingSet, notices: list[str]) -> list[Path]:
    """Add every record file of `gen_dir` to `data`, one file at a time; returns the files read.
    ValueError if `gen_dir` is not a directory."""
    gen_dir = Path(gen_dir)
    if not gen_dir.is_dir():
        raise ValueError(f"generation directory {gen_dir} does not exist (or is not a directory)")
    seen: set[int] = set()
    paths = record_files(gen_dir)
    for path in paths:
        _read_file(path, gen_dir.name, data, seen, notices)
    return paths


@dataclass
class Window:
    data: TrainingSet
    new_positions: int
    files: dict[str, str]
    notices: list[str]


def load_window(root: str | Path, gen_dirs: list[Path]) -> Window:
    """The training set over `gen_dirs` (oldest first; the last is the newest generation), the
    newest generation's decision count, and every record file read (path relative to `root` ->
    sha256). ValueError if there is nothing to train on."""
    root = Path(root)
    gen_dirs = [Path(d) for d in gen_dirs]
    if not gen_dirs:
        raise ValueError("no generation directories to train on")
    notices: list[str] = []
    data = TrainingSet()
    read: dict[str, str] = {}
    new = 0
    for d in gen_dirs:
        before = data.decisions
        for p in read_generation(d, data, notices):
            read[str(p.relative_to(root))] = files.sha256_file(p)
        new = data.decisions - before
    if data.decisions == 0:
        raise ValueError(f"no training positions in {', '.join(str(d) for d in gen_dirs)}")
    if new == 0:
        raise ValueError(f"the newest generation {gen_dirs[-1]} has no decisions to train on")
    return Window(data, new, read, notices)


@dataclass
class Tensors:
    public: torch.Tensor
    hidden: torch.Tensor
    mask: torch.Tensor
    policy: torch.Tensor
    full: torch.Tensor
    outcome: torch.Tensor
    vp: torch.Tensor


def to_tensors(batch: dict, device: torch.device) -> Tensors:
    """Wrap a sampler batch's little-endian buffers (`torch.frombuffer`, no numpy)."""
    n = batch["n"]

    def f32(key: str, width: int) -> torch.Tensor:
        return torch.frombuffer(bytearray(batch[key]), dtype=torch.float32).view(n, width).to(device)

    def u8(key: str, width: int) -> torch.Tensor:
        return torch.frombuffer(bytearray(batch[key]), dtype=torch.uint8).view(n, width).to(device)

    return Tensors(
        public=f32("public", NET_PUBLIC_LEN),
        hidden=f32("hidden", NET_HIDDEN_LEN),
        mask=u8("mask", ACTIONS).bool(),
        policy=f32("policy", ACTIONS),
        full=u8("full", 1).view(n).float(),
        outcome=f32("outcome", SEATS),
        vp=f32("vp", SEATS),
    )
