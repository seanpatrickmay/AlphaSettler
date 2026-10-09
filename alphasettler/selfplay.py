"""3b self-play records (spec Section 4): playout cap randomization, root noise on full searches
and opening sampling, for the heuristic bot (`ismcts`, the warm start) or a network (`net:NAME`).
Each game carries a `selfplay` block: the bot, the net file's sha256, both simulation caps and the
exploration settings. Decisions carry no observations; the seed and actions reproduce them. With
`gumbel` (P2), every move is a Gumbel root search and each decision carries its improved policy as
`target`; the block then holds `gumbel_simulations` instead of the exploration settings. With `opening`
(3d-1b), an opening model file picks every setup settlement, epsilon-greedy over its best spots, with
no search and no decision; the block then also holds `opening_sha256`, `opening_epsilon` and
`opening_top`. With `league` (3d-3: `{"nets": [...], "styles": [...]}`, bot names), about half the games seat
two of them, a past net and a style, whose moves are not recorded; the block then also holds
`league_probability`, `league_nets` and `league_styles`, and a league game's record names its seats under
`league` (null at a learner seat). With `compound` (3d-4), every learner seat searches compound build moves;
the block then also holds `"compound": true`."""

from __future__ import annotations

import math
import os
from pathlib import Path

from alphasettler import arena, files, records
from alphasettler._engine import (
    DEFAULT_NETS_DIR, LEAGUE_PROBABILITY, MAX_SIMULATIONS, NET_BATCH, NOISE_ALPHA, NOISE_WEIGHT, OPENING_EPSILON,
    OPENING_FILE, OPENING_TOP, SAMPLE_TURNS, SEARCH_BATCH, SELFPLAY_MIN_SIMULATIONS, Game, encode, net_forward,
    opening_forward, resolve_net, selfplay_explore,
)

# Games per native call. Python sees Ctrl-C only between calls, and every finished batch is on disk.
BATCH_GAMES = 100
OPENING_WITH_GUMBEL = ("--opening and --gumbel cannot be used together: Gumbel self-play stays the P2 measurement "
                       "it was (spec 3d-1b Section 3)")
LEAGUE_WITH_GUMBEL = ("--league and --gumbel cannot be used together: Gumbel self-play stays the P2 measurement "
                      "it was (spec 3d-3 Section 4)")
COMPOUND_WITH_GUMBEL = ("--compound and --gumbel cannot be used together: compound moves are not searched with a "
                        "Gumbel root (spec 3d-4 Section 4)")


def check_gumbel(simulations: int) -> None:
    """ValueError unless `simulations` is a budget Gumbel self-play accepts."""
    if not (isinstance(simulations, int) and SELFPLAY_MIN_SIMULATIONS <= simulations <= MAX_SIMULATIONS):
        raise ValueError(f"gumbel simulations must be {SELFPLAY_MIN_SIMULATIONS}..={MAX_SIMULATIONS}, "
                         f"got {simulations!r}")


def check_opening(path: str | Path, gumbel: int | None = None) -> str:
    """The sha256 of the opening model file `path`, once the Rust loader every `+o` bot uses has read
    it (its ValueError, naming the file, otherwise). With `gumbel` it is a ValueError: the opening
    model is not used with Gumbel self-play."""
    if gumbel is not None:
        raise ValueError(OPENING_WITH_GUMBEL)
    opening_forward(str(path), [])
    return files.sha256_file(path)


def check_compound(compound: bool, gumbel: int | None = None) -> None:
    """ValueError if compound moves are asked for with Gumbel self-play."""
    if compound and gumbel is not None:
        raise ValueError(COMPOUND_WITH_GUMBEL)


def check_league(league: dict, nets_dir: str | Path, gumbel: int | None = None) -> None:
    """ValueError unless `league` ({"nets": [...], "styles": [...]}, bot names) is a pool self-play can
    seat: no Gumbel self-play, a name in each half, and every name a bot whose files load from
    `nets_dir` (checked by the arena's zero-seed call, as every match checks its bots)."""
    if gumbel is not None:
        raise ValueError(LEAGUE_WITH_GUMBEL)
    if not league.get("nets"):
        raise ValueError("a league needs at least one past net")
    if not league.get("styles"):
        raise ValueError("a league needs at least one style")
    for name in [*league["nets"], *league["styles"]]:
        arena.check(name, name, 1, nets_dir=nets_dir)


def check_league_nets(paths: list[str | Path], gumbel: int | None = None) -> list[str]:
    """The sha256 of each of a league's set-up net files, once the Rust loader every `net:` bot uses has
    read it (a ValueError starting `league net PATH:` otherwise: missing, cut short, another encoder).
    No file, or `gumbel`, is a ValueError."""
    if gumbel is not None:
        raise ValueError(LEAGUE_WITH_GUMBEL)
    if not paths:
        raise ValueError("--league needs at least one net file")
    public, hidden = encode(Game(0, {"max_offers_per_turn": 0}))
    out = []
    for path in paths:
        try:
            net_forward(str(path), public, hidden)
        except ValueError as e:
            raise ValueError(f"league net {path}: {e}") from None
        out.append(files.sha256_file(path))
    return out


def mode_mismatch(root: str | Path, stored: int | None, asked: int | None) -> str | None:
    """The error for asking `--gumbel` of a loop in `root` that was set up with `stored` (None:
    without it, as every loop from before P2 was), or None if the ask is allowed: no flag keeps the
    loop's mode, the same value repeats it. A loop keeps the self-play mode it was set up with."""
    if asked is None or asked == stored:
        return None
    was = "without --gumbel" if stored is None else f"with --gumbel {stored}"
    return f"{root} was set up {was}, and a loop keeps the self-play mode it was set up with"


def opening_mismatch(root: str | Path, stored: str | None, asked: str | Path | None) -> str | None:
    """The error for asking `--opening FILE` of a loop in `root` that was set up with the opening
    model whose sha256 is `stored` (None: without one, as every loop from before 3d-1b was), or None
    if the ask is allowed: no flag keeps the loop's model, the same file (by sha256) repeats it."""
    if asked is None:
        return None
    if files.sha256_file(asked) == stored:
        return None
    was = "without --opening" if stored is None else f"with another opening model (sha256 {stored})"
    return f"{root} was set up {was}, and a loop keeps the opening model it was set up with"


def place_opening(root: str | Path, nets: str | Path, stored: str | None, asked: str | Path | None) -> None:
    """For a loop in `root` set up with the opening model whose sha256 is `stored` (None: without
    one, nothing to do): restore `<nets>/opening.bin` from `--opening FILE` (`asked`) when it is
    missing and FILE has that sha256, then check the file there is that model (ValueError if it is
    missing or changed). A changed file is never replaced."""
    if stored is None:
        return
    path = Path(nets) / OPENING_FILE
    if not path.exists() and asked is not None and files.sha256_file(asked) == stored:
        path.parent.mkdir(parents=True, exist_ok=True)
        files.atomic_copy(asked, path)
    if not path.is_file() or files.sha256_file(path) != stored:
        raise ValueError(f"{path} is not the opening model {root} was set up with (sha256 {stored}): "
                         f"restore it, or resume with --opening and that file")


def league_mismatch(root: str | Path, stored: dict | None, asked: list[str | Path] | None) -> str | None:
    """The error for asking `--league NET ...` of a loop in `root` set up with the league `stored` (its
    state.json entry; None: without one, as every loop from before 3d-3 was), or None if the ask is
    allowed: no flag keeps the loop's pool, the same files (by sha256, in order) repeat it."""
    if asked is None:
        return None
    if stored is not None and [files.sha256_file(p) for p in asked] == [n["sha256"] for n in stored["nets"]]:
        return None
    if stored is None:
        was = "without --league"
    else:
        was = f"with another league pool ({', '.join(n['source'] for n in stored['nets'])})"
    return f"{root} was set up {was}, and a loop keeps the league it was set up with"


def settings(net: str | None, nets_dir: str | Path, full_simulations: int, fast_simulations: int,
             full_probability: float, gumbel: int | None = None, opening: str | Path | None = None,
             league: dict | None = None, compound: bool = False) -> dict:
    """The `selfplay` block. Checks every argument, loads the net, reads the opening model and checks
    the league's bots, without playing a game. With `gumbel`, the PUCT arguments are not used;
    `opening`, `league` or `compound` with `gumbel` is a ValueError."""
    check_compound(compound, gumbel)
    opening_sha256 = None if opening is None else check_opening(opening, gumbel)
    if league is not None:
        check_league(league, nets_dir, gumbel)
    if gumbel is not None:
        check_gumbel(gumbel)
    else:
        for what, n in (("full", full_simulations), ("fast", fast_simulations)):
            if not SELFPLAY_MIN_SIMULATIONS <= n <= MAX_SIMULATIONS:
                raise ValueError(
                    f"{what} simulations must be {SELFPLAY_MIN_SIMULATIONS}..={MAX_SIMULATIONS}, got {n}")
        if not (math.isfinite(full_probability) and 0 <= full_probability <= 1):
            raise ValueError(f"the full-search probability must be in [0, 1], got {full_probability}")
    path = None if net is None else resolve_net(net, str(nets_dir))
    who = {
        "bot": "ismcts" if net is None else f"net:{net}",
        "net": net,
        "net_sha256": None if path is None else files.sha256_file(path),
    }
    batch = SEARCH_BATCH if net is None else NET_BATCH
    if gumbel is not None:
        return {**who, "gumbel_simulations": gumbel, "batch": batch}
    block = {
        **who,
        "full_simulations": full_simulations,
        "fast_simulations": fast_simulations,
        "full_probability": full_probability,
        "noise_weight": NOISE_WEIGHT,
        "noise_alpha": NOISE_ALPHA,
        "sample_turns": SAMPLE_TURNS,
        "batch": batch,
    }
    if opening_sha256 is not None:  # only then: records without a model are unchanged
        block.update(opening_sha256=opening_sha256, opening_epsilon=OPENING_EPSILON, opening_top=OPENING_TOP)
    if league is not None:  # only then: records without a league are unchanged
        block.update(league_probability=LEAGUE_PROBABILITY, league_nets=list(league["nets"]),
                     league_styles=list(league["styles"]))
    if compound:  # only then: records without compound moves are unchanged
        block["compound"] = True
    return block


def write_explore(out: str | Path, seed_start: int, games: int, *, config: dict, full_simulations: int,
                  fast_simulations: int, full_probability: float, net: str | None = None,
                  nets_dir: str | Path | None = None, threads: int | None = None,
                  counts: dict | None = None, gumbel: int | None = None, opening: str | Path | None = None,
                  league: dict | None = None, compound: bool = False) -> dict:
    """Play `games` games from `seed_start` and write them to the new file `out`. Arguments are
    checked before the file is created. `counts` (if given) is updated as batches finish, so a
    caller that catches KeyboardInterrupt still knows what was kept."""
    nets_dir = DEFAULT_NETS_DIR if nets_dir is None else nets_dir
    meta = settings(net, nets_dir, full_simulations, fast_simulations, full_probability, gumbel, opening, league,
                    compound)
    counts = {} if counts is None else counts
    counts.update(games=0, decisions=0, full_decisions=0, belief_resets=0)
    threads = threads or os.cpu_count() or 1
    end = seed_start + games
    with records.open_writer(out) as w:
        for lo in range(seed_start, end, BATCH_GAMES):
            batch = selfplay_explore(lo, min(BATCH_GAMES, end - lo), threads, full_simulations, fast_simulations,
                                     full_probability, config, net, str(nets_dir), gumbel,
                                     None if opening is None else str(opening),
                                     None if league is None else list(league["nets"]),
                                     None if league is None else list(league["styles"]), compound)
            for g in batch:
                w.write_game({**g, "selfplay": meta}, config)
                counts["games"] += 1
                counts["decisions"] += len(g["decisions"])
                counts["full_decisions"] += sum(d["full_search"] for d in g["decisions"])
                counts["belief_resets"] += g["belief_resets"]
            w.flush()
    return counts
