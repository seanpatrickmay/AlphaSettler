"""Self-play records: one gzipped JSONL line per game (spec Section 4)."""

from __future__ import annotations

import gzip
import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TextIO

from alphasettler._engine import Game


class Writer:
    """Appends games to an open record file; see `open_writer`."""

    def __init__(self, f: TextIO):
        self._f = f

    def write_game(self, game: dict, config: dict) -> None:
        self._f.write(json.dumps({"config": config, **game}) + "\n")

    def flush(self) -> None:
        """Push everything written so far through the compressor to disk."""
        self._f.flush()


@contextmanager
def open_writer(path: str | Path) -> Iterator[Writer]:
    """Create `path` (and its parent directory) exclusively and yield a `Writer` for it; an
    existing file is an error. Leaving the block, by an exception too, closes the gzip stream
    cleanly, so the games written so far stay readable."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "xt") as f:
        yield Writer(f)


def write(path: str | Path, games: list[dict], config: dict) -> None:
    """Write `games` (as returned by `selfplay`) to a new file; an existing file is an error."""
    with open_writer(path) as w:
        for g in games:
            w.write_game(g, config)


def read(path: str | Path) -> Iterator[dict]:
    with gzip.open(path, "rt") as f:
        for line in f:
            yield json.loads(line)


def _plain(x):
    """As it would read back from JSON (tuples become lists)."""
    return json.loads(json.dumps(x))


def replay_mismatches(game: dict) -> list[str]:
    """Replay `game` from its seed and actions; describe every stored decision that differs."""
    g = Game(game["seed"], game["config"])
    decisions = {d["index"]: d for d in game["decisions"]}
    out = []
    for i, a in enumerate(game["actions"]):
        d = decisions.get(i)
        if d is not None:
            if g.current_actor != d["actor"]:
                out.append(f"move {i}: actor {g.current_actor}, record says {d['actor']}")
            if g.legal_actions() != d["legal"]:
                out.append(f"move {i}: legal actions differ")
            if _plain(g.observation(d["actor"])) != d["observation"]:
                out.append(f"move {i}: observation differs")
        g.apply(a)
    if g.winner != game["winner"]:
        out.append(f"winner {g.winner}, record says {game['winner']}")
    return out
