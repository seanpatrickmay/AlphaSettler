"""Training data (spec Section 4): self-play record files -> the Rust sampler -> tensors.

A generation directory holds finished record files `*.jsonl.gz` (other machines may drop more in:
the scale-out path). Hidden temp files (`files.temp_path`) are never read. Files that are empty,
cut short, hold no decisions or do not replay are skipped (or kept up to the damage) with a
notice, and a seed already read in the same generation (a second writer playing the same seeds)
is read once. Records go into the `TrainingSet` a file at a time, as a stream, so the window is
never held as Python dicts."""

from __future__ import annotations

import json
import zlib
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import torch

from alphasettler import files, records
from alphasettler._engine import NET_HIDDEN_LEN, NET_PUBLIC_LEN, TrainingSet, action_space_size

ACTIONS = action_space_size()
SEATS = 4

# What a cut-short or garbled file raises while it is read (gzip, zlib, JSON, a non-record line).
_READ_ERRORS = (EOFError, OSError, zlib.error, json.JSONDecodeError, UnicodeDecodeError, KeyError, TypeError)


def record_files(gen_dir: str | Path) -> list[Path]:
    """Finished record files in `gen_dir`, by name; temporary (dot) files are not."""
    return sorted(p for p in Path(gen_dir).glob("*.jsonl.gz") if not p.name.startswith("."))


def compact(game: dict) -> dict:
    """Only what the sampler needs."""
    return {
        "seed": game["seed"],
        "config": game["config"],
        "actions": game["actions"],
        "decisions": [{"index": d["index"], "actor": d["actor"], "visits": d["visits"], "full_search": d["full_search"]}
                      for d in game["decisions"]],
        "winner": game["winner"],
        "vp": game["vp"],
    }


@dataclass
class _FileStats:
    duplicates: int = 0
    kept: int = 0
    decisions: int = 0
    error: str | None = None


def _games(path: Path, gen_name: str, seen: set[int], added: set[int], stats: _FileStats,
           notices: list[str]) -> Iterator[dict]:
    """The file's games, compacted, skipping seeds in `seen` or `added` (and adding the rest to
    `added`); a read error ends the stream and is kept in `stats`."""
    try:
        for g in records.read(path):
            seed = g["seed"]
            if seed in seen or seed in added:
                stats.duplicates += 1
                notices.append(f"{path}: seed {seed} is already in {gen_name}; skipped the duplicate")
                continue
            c = compact(g)
            added.add(seed)
            stats.kept += 1
            stats.decisions += len(c["decisions"])
            yield c
    except _READ_ERRORS as e:
        stats.error = f"{type(e).__name__}: {e}"


def read_generation(gen_dir: str | Path, data: TrainingSet, notices: list[str]) -> list[Path]:
    """Add every record file of `gen_dir` to `data`, one file at a time; returns the files read."""
    gen_dir = Path(gen_dir)
    seen: set[int] = set()
    paths = record_files(gen_dir)
    for path in paths:
        stats, added = _FileStats(), set()
        try:
            data.extend(_games(path, gen_dir.name, seen, added, stats, notices))
        except ValueError as e:  # a record that does not replay: `extend` added nothing
            notices.append(f"{path}: skipped the whole file: {e}")
            continue
        seen |= added
        if stats.error is not None:
            notices.append(f"{path}: kept {stats.kept} games, skipped an unreadable rest ({stats.error})")
        elif stats.kept == 0:
            notices.append(f"{path}: all {stats.duplicates} games duplicate earlier seeds" if stats.duplicates
                           else f"{path}: no games")
        if stats.kept and stats.decisions == 0:
            notices.append(f"{path}: {stats.kept} games but no decisions")
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
