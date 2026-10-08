"""A fixed setup on seed 0's board, shared by the opening tests (records need not come from
self-play: every opening tool reads only a record's seed, config, first 16 actions and winner).

Settlements in pick order (seats 0 1 2 3 3 2 1 0): nodes 9, 12, 18, 41, 31, 10, 4, 29, each
followed by its first legal road. What they touch (resource number):
  seat 0: 9 (wheat 10, wheat 8, brick 5), 29 (wood 12, ore 11, sheep 6): all five, 20 pips
  seat 1: 12 (ore 10, brick 9, sheep 8), 4 (ore 10, wheat 10): no wood or no brick, 18 pips
  seat 2: 18 (sheep 8, brick 5, ore 11), 10 (wheat 8, wood 11): all five, 18 pips
  seat 3: 41 (wood 2, wheat 9, sheep 6), 31 (brick 5, sheep 4, wheat 9): no ore, 21 pips
The distance rule does not depend on the board, so these actions replay on any seed:
`fixed_game(seed=...)` makes as many distinct games as a test needs (only seed 0 has this board)."""

from pathlib import Path

from alphasettler import records
from alphasettler._engine import OPENING_FEATURES, resolve_config
from alphasettler.arena import TRADES
from alphasettler.opening import write_model

FIXED_NODES = [9, 12, 18, 41, 31, 10, 4, 29]
FIXED_ACTIONS = [34, 141, 37, 144, 43, 153, 66, 185, 56, 170, 35, 142, 29, 134, 54, 168]


def fixed_game(seed: int = 0, winner: int | None = 3, config: dict = TRADES) -> dict:
    """A record holding only the fixed setup, played under `config` (stored resolved, as the
    record writer stores it)."""
    return {"seed": seed, "config": resolve_config(config), "actions": list(FIXED_ACTIONS), "winner": winner,
            "vp": [0, 0, 0, 0], "turns": 0, "decisions": [], "belief_resets": 0}


def write_dir(path: Path, games: list[dict]) -> Path:
    """`games` as the record file `path/selfplay-fixed.jsonl.gz` (each keeps its own config)."""
    records.write(Path(path) / "selfplay-fixed.jsonl.gz", games, {})
    return Path(path)


def truncate(path_or_dir: Path) -> Path:
    """Cut the record file (or the one in the directory `write_dir` made) short, as a copy that died
    mid-write would be; returns the file."""
    path = Path(path_or_dir)
    if path.is_dir():
        path = path / "selfplay-fixed.jsonl.gz"
    path.write_bytes(path.read_bytes()[:-20])
    return path


def opening_model_file(path: Path, scale: float = 0.02) -> Path:
    """A one-layer opening model (78 features -> 4 seats) with fixed weights times `scale`, written
    through the Rust loader: its spots score apart, so its picks are not engine order. `scale=0.0`
    writes a zero model (every spot ties)."""
    w = [((i * 37) % 11 - 5) * scale for i in range(4 * OPENING_FEATURES)]
    write_model(path, [(4, OPENING_FEATURES, w, [0.0] * 4)])
    return Path(path)
