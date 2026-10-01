"""Candidate-vs-baseline matches in the native Rust arena, with per-game JSONL output."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TextIO

from alphasettler._engine import MAX_SEEDS, run_match
from alphasettler.stats import Summary, summarize

# Seeds per native call. Python handles Ctrl-C between calls, so this bounds how long an
# interrupt waits; records are unchanged because every game depends only on its seed.
BATCH_SEEDS = 10_000

# The engine computes the range end seed_start + seeds as a u64, so the end itself must fit.
SEED_LIMIT = 2**64 - 1


def _write_records(f: TextIO, candidate: str, baseline: str, records: list[dict]) -> None:
    for r in records:
        f.write(json.dumps({"candidate": candidate, "baseline": baseline, **r}) + "\n")
    f.flush()


def open_output(path: str | Path) -> TextIO:
    """Create the parent directory and open `path` exclusively; an existing file is an OSError."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.open("x")


def check(
    candidate: str,
    baseline: str,
    seeds: int,
    seed_start: int = 0,
    threads: int | None = None,
    config: dict | None = None,
) -> None:
    """Raise ValueError for anything `run` would reject, without playing a game."""
    if seeds <= 0:
        raise ValueError("seeds must be positive")
    if seeds > MAX_SEEDS:
        raise ValueError(f"at most {MAX_SEEDS} seeds per match, got {seeds}")
    if seed_start + seeds > SEED_LIMIT:
        raise ValueError(f"seed_start {seed_start} + seeds {seeds} exceeds the u64 seed range")
    # A zero-seed native call checks bot names, config and integer ranges and plays nothing.
    run_match(candidate, baseline, seed_start, 0, threads or os.cpu_count() or 1, config)


def run(
    candidate: str,
    baseline: str,
    seeds: int,
    seed_start: int = 0,
    threads: int | None = None,
    config: dict | None = None,
    out: str | Path | TextIO | None = None,
) -> tuple[list[dict], Summary]:
    """Play `seeds` seeds from `seed_start` in batches of `BATCH_SEEDS`, appending each batch to
    `out` (a path, opened exclusively, or an open file the caller owns). Inputs are checked and a
    path is opened before any game is played, so a bad argument or path costs nothing."""
    check(candidate, baseline, seeds, seed_start, threads, config)
    threads = threads or os.cpu_count() or 1
    owned = isinstance(out, (str, os.PathLike))
    f = open_output(out) if owned else out
    records: list[dict] = []
    try:
        end = seed_start + seeds
        for lo in range(seed_start, end, BATCH_SEEDS):
            batch = run_match(candidate, baseline, lo, min(BATCH_SEEDS, end - lo), threads, config)
            if f is not None:
                _write_records(f, candidate, baseline, batch)
            records.extend(batch)
    finally:
        if owned:
            f.close()
    return records, summarize(records)


def format_summary(label: str, s: Summary) -> str:
    if s.ci_low == s.ci_high:  # only when the per-seed SE is 0
        ci = "CI n/a (zero variance)"
    else:
        ci = f"[{s.ci_low:.3f}, {s.ci_high:.3f}] 95% CI"
    return (
        f"{label}: win rate {s.win_rate:.3f} {ci} over {s.seeds} seeds "
        f"({s.games} games, {s.wins} wins, {s.draws} draws), z vs {s.null:.2f} = {s.z_vs_null:.2f}, "
        f"mean VP {s.mean_vp:.2f}, mean turns {s.mean_turns:.1f}"
    )
