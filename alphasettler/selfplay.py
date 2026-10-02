"""3b self-play records (spec Section 4): playout cap randomization, root noise on full searches
and opening sampling, for the heuristic bot (`ismcts`, the warm start) or a network (`net:NAME`).
Each game carries a `selfplay` block: the bot, the net file's sha256, both simulation caps and the
exploration settings. Decisions carry no observations; the seed and actions reproduce them."""

from __future__ import annotations

import math
import os
from pathlib import Path

from alphasettler import files, records
from alphasettler._engine import (
    DEFAULT_NETS_DIR, MAX_SIMULATIONS, NET_BATCH, NOISE_ALPHA, NOISE_WEIGHT, SAMPLE_TURNS, SEARCH_BATCH,
    SELFPLAY_MIN_SIMULATIONS, resolve_net, selfplay_explore,
)

# Games per native call. Python sees Ctrl-C only between calls, and every finished batch is on disk.
BATCH_GAMES = 100


def settings(net: str | None, nets_dir: str | Path, full_simulations: int, fast_simulations: int,
             full_probability: float) -> dict:
    """The `selfplay` block. Checks every argument, and loads the net, without playing a game."""
    for what, n in (("full", full_simulations), ("fast", fast_simulations)):
        if not SELFPLAY_MIN_SIMULATIONS <= n <= MAX_SIMULATIONS:
            raise ValueError(f"{what} simulations must be {SELFPLAY_MIN_SIMULATIONS}..={MAX_SIMULATIONS}, got {n}")
    if not (math.isfinite(full_probability) and 0 <= full_probability <= 1):
        raise ValueError(f"the full-search probability must be in [0, 1], got {full_probability}")
    path = None if net is None else resolve_net(net, str(nets_dir))
    return {
        "bot": "ismcts" if net is None else f"net:{net}",
        "net": net,
        "net_sha256": None if path is None else files.sha256_file(path),
        "full_simulations": full_simulations,
        "fast_simulations": fast_simulations,
        "full_probability": full_probability,
        "noise_weight": NOISE_WEIGHT,
        "noise_alpha": NOISE_ALPHA,
        "sample_turns": SAMPLE_TURNS,
        "batch": SEARCH_BATCH if net is None else NET_BATCH,
    }


def write_explore(out: str | Path, seed_start: int, games: int, *, config: dict, full_simulations: int,
                  fast_simulations: int, full_probability: float, net: str | None = None,
                  nets_dir: str | Path | None = None, threads: int | None = None,
                  counts: dict | None = None) -> dict:
    """Play `games` games from `seed_start` and write them to the new file `out`. Arguments are
    checked before the file is created. `counts` (if given) is updated as batches finish, so a
    caller that catches KeyboardInterrupt still knows what was kept."""
    nets_dir = DEFAULT_NETS_DIR if nets_dir is None else nets_dir
    meta = settings(net, nets_dir, full_simulations, fast_simulations, full_probability)
    counts = {} if counts is None else counts
    counts.update(games=0, decisions=0, full_decisions=0, belief_resets=0)
    threads = threads or os.cpu_count() or 1
    end = seed_start + games
    with records.open_writer(out) as w:
        for lo in range(seed_start, end, BATCH_GAMES):
            batch = selfplay_explore(lo, min(BATCH_GAMES, end - lo), threads, full_simulations, fast_simulations,
                                     full_probability, config, net, str(nets_dir))
            for g in batch:
                w.write_game({**g, "selfplay": meta}, config)
                counts["games"] += 1
                counts["decisions"] += len(g["decisions"])
                counts["full_decisions"] += sum(d["full_search"] for d in g["decisions"])
                counts["belief_resets"] += g["belief_resets"]
            w.flush()
    return counts
