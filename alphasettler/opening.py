"""Openings (spec 3d-1): replaying a recorded game's setup, and the opening model file.

A game's setup is its first 16 actions: (settlement, road) for each seat in snake order.
`replay_setup` replays them from the record's seed and config and keeps, for each of the 8
settlements, the seat, the node and the nodes legal at that pick and, on request, the position
before the pick, the opening features right after it (before its road) and a net's view of that
same position.

The model file `<nets_dir>/opening.bin` (bots/src/opening_model.rs has the byte table): magic,
format version, OPENING_HASH, layer count, each layer's rows and columns, then each layer's f32
weights (row-major, rows = outputs) and biases, little-endian. `write_model` has the Rust loader
read the file back before it takes the name, so a file the bots would refuse is never left at the
path. No torch here: the diagnosis imports this module."""

from __future__ import annotations

import os
import struct
import sys
from array import array
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from alphasettler import files
from alphasettler._engine import (
    OPENING_FEATURES, OPENING_FORMAT_VERSION, OPENING_HASH, Game, opening_features, opening_forward,
)

MAGIC = b"ASOPEN\0\0"
SETUP_ACTIONS = 16
# BuildSettlement(n) is action 25 + n and BuildRoad(e) is 133 + e (engine/src/action.rs).
SETTLEMENT_BASE = 25
ROAD_BASE = 133
NODES = 54
EDGES = 72

Layer = tuple[int, int, Sequence[float], Sequence[float]]


@dataclass
class Pick:
    """One setup settlement."""
    seat: int
    node: int
    legal: list[int]  # the nodes legal at this pick, in engine order
    before: Game | None = None  # the position at this pick (keep_positions)
    features: list[float] | None = None  # opening_features for `seat` right after it, before its road
    value: list[float] | None = None  # value(game, seat) at that same position


@dataclass
class Setup:
    seed: int
    trading: bool  # the game allowed domestic offers
    winner: int | None
    board: dict  # Game.snapshot()["board"]
    picks: list[Pick]  # the 8 settlements in pick order
    after: Game  # the setup done, before the first roll

    def nodes(self, seat: int) -> list[int]:
        """`seat`'s settlements in the order it placed them."""
        return [p.node for p in self.picks if p.seat == seat]


def replay_setup(record: dict, *, features: bool = False, value: Callable[[Game, int], list[float]] | None = None,
                 keep_positions: bool = False) -> Setup:
    """The setup of `record` (a self-play record: `seed`, `config`, `actions`, `winner`). ValueError
    naming the game's seed if its first 16 actions are not a legal setup, or naming the missing
    fields if the record lacks any."""
    missing = [k for k in ("seed", "config", "actions", "winner") if k not in record]
    if missing:
        raise ValueError(f"a record has no {', '.join(missing)} field" + ("s" if len(missing) > 1 else ""))
    seed = record["seed"]
    actions = record["actions"]
    if len(actions) < SETUP_ACTIONS:
        raise ValueError(f"game {seed}: {len(actions)} actions, fewer than the {SETUP_ACTIONS} of the setup")
    g = Game(seed, record["config"])
    board = g.snapshot()["board"]
    picks: list[Pick] = []
    for k, a in enumerate(actions[:SETUP_ACTIONS]):
        settlement = k % 2 == 0
        base, count = (SETTLEMENT_BASE, NODES) if settlement else (ROAD_BASE, EDGES)
        if not (isinstance(a, int) and base <= a < base + count):
            kind = "settlement" if settlement else "road"
            raise ValueError(f"game {seed}: action {k} ({a!r}) is not a setup {kind}")
        seat = g.current_actor
        if settlement:
            legal = [x - SETTLEMENT_BASE for x in g.legal_actions()]
            before = g.copy() if keep_positions else None
        try:
            g.apply(a)
        except ValueError as e:
            raise ValueError(f"game {seed}: action {k} ({a}): {e}") from None
        if settlement:
            picks.append(Pick(seat, a - SETTLEMENT_BASE, legal, before,
                              opening_features(g, seat) if features else None,
                              value(g, seat) if value is not None else None))
    trading = record["config"].get("max_offers_per_turn", 0) > 0
    return Setup(seed, trading, record["winner"], board, picks, g)


def model_bytes(layers: Sequence[Layer]) -> bytes:
    """An opening model file holding `layers`: (rows, cols, rows x cols weights row-major, rows
    biases) each, in order."""
    assert sys.byteorder == "little", "the opening model file is little-endian"
    out = [MAGIC, struct.pack("<IQI", OPENING_FORMAT_VERSION, OPENING_HASH, len(layers))]
    out += [struct.pack("<II", rows, cols) for rows, cols, _, _ in layers]
    for rows, cols, w, b in layers:
        if len(w) != rows * cols or len(b) != rows:
            raise ValueError(f"a {rows} x {cols} layer needs {rows * cols} weights and {rows} biases, "
                             f"got {len(w)} and {len(b)}")
        out += [array("f", w).tobytes(), array("f", b).tobytes()]
    return b"".join(out)


def write_model(path: str | Path, layers: Sequence[Layer]) -> None:
    """Write the model to `path` through a temp file the Rust loader reads first (raising its error,
    with nothing left behind, for a file the bots would refuse)."""
    path = Path(path)
    tmp = files.temp_path(path)
    try:
        tmp.write_bytes(model_bytes(layers))
        opening_forward(str(tmp), [[0.0] * OPENING_FEATURES])
        files.fsync_file(tmp)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
