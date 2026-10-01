"""Lockstep differential test against Catanatron.

Each game is played by Catanatron random players. Before every action, Catanatron's legal
actions are compared with ours (as tokens; see oracle.translate). The action is executed in
Catanatron, and its realized chance outcome is forced on ours. Then the full positions are
compared. A divergence an allowlist entry explains is counted, and our engine re-imports
Catanatron's position so the comparison continues. Anything else ends the game as a mismatch.
Runs use spawn workers with PYTHONHASHSEED=0, because Catanatron's move order depends on the
hash seed.
"""

from __future__ import annotations

import multiprocessing as mp
import os
from collections import Counter
from dataclasses import dataclass, field

import catanatron.game as catan_game
from catanatron import Color, Game as CatanGame, RandomPlayer
from catanatron.models.enums import ActionType

from alphasettler import Game
from oracle import allowlist, hash_seed_zero
from oracle.translate import (
    SETUP_TURNS, Untranslatable, action_token, catan_snapshot, catan_tokens, compare, our_steps, our_tokens,
    seat,
)

COLORS = [Color.RED, Color.BLUE, Color.ORANGE, Color.WHITE]
TRACE_LEN = 20


@dataclass
class GameResult:
    seed: int
    steps: int = 0
    allowlisted: Counter = field(default_factory=Counter)
    mismatch: str | None = None
    trace: list[str] = field(default_factory=list)
    final_phase: str | None = None
    winner: int | None = None
    # Steps Catanatron played while its position (reached through an allowlisted difference)
    # could not be imported into our engine; the comparison resumes once it can be.
    unimported_steps: int = 0
    unimported_reasons: Counter = field(default_factory=Counter)
    ended_unimported: bool = False  # the game ended inside such a stretch, so its end was not compared


# The one position our rules cannot express is Catanatron's Road Building through an opponent's
# settlement (allowlist entry road-from-opponent-building); it lasts at most its two free roads.
TOLERATED_UNIMPORTABLE = "road-from-opponent-building"
MAX_UNIMPORTED = 2
TOLERATED_REASON = "has nowhere to build a road"  # engine/src/invariants.rs, RoadBuilding check


def _import(seed: int, snap: dict, cfg: dict):
    """Our engine at Catanatron's position, or the reason it cannot represent it."""
    try:
        return Game.from_snapshot(seed, snap, cfg), None
    except ValueError as e:
        return None, str(e)


def config() -> dict:
    return {"max_offers_per_turn": 0, "catanatron_compat": False,
            "max_turns": catan_game.TURNS_LIMIT - SETUP_TURNS}


def _executed_kind(state, action, before: dict, actor: int) -> str | None:
    """The entry explaining an executed Catanatron action with no equivalent in ours. Its token
    (e.g. ``("year_of_plenty_one_card", r)``) is classified like a Catanatron-only legal token."""
    try:
        token = action_token(state, action)
    except Untranslatable:
        return None
    return allowlist.classify_legal(token, before, actor, "catanatron")


def run_game(seed: int) -> GameResult:
    """One lockstep game. An exception (in either engine or the translation) ends it as a
    mismatch with its trace, so a run reports the seed instead of aborting. That includes an
    engine panic, which PyO3 raises as PanicException, a BaseException."""
    res = GameResult(seed)
    try:
        _play(seed, res)
    except BaseException as e:  # noqa: BLE001 - every failure is a finding for this seed
        if isinstance(e, (KeyboardInterrupt, SystemExit)):
            raise
        res.mismatch = f"step {res.steps}: {type(e).__name__}: {e}"
    return res


def _play(seed: int, res: GameResult) -> None:
    cfg = config()
    cat = CatanGame([RandomPlayer(c) for c in COLORS], seed=seed)
    by_color = {p.color: p for p in cat.state.players}
    ours = Game.from_snapshot(seed, catan_snapshot(cat), cfg)
    theirs = catan_snapshot(cat)
    unimported = 0
    last_kind = None  # the allowlist entry behind the latest re-import
    while theirs["phase"] != "game_over":
        st = cat.state
        if ours is None:
            ours, why = _import(seed, theirs, cfg)
            if ours is None:
                # Only Catanatron's Road Building through an opponent's settlement is a position our
                # rules cannot express; any other import failure is a translation or engine bug.
                unimported += 1
                if (last_kind != TOLERATED_UNIMPORTABLE or theirs["phase"] != "road_building"
                        or TOLERATED_REASON not in why):
                    res.mismatch = f"step {res.steps}: cannot import Catanatron's position: {why}"
                    return
                if unimported > MAX_UNIMPORTED:
                    res.mismatch = f"step {res.steps}: Catanatron's position stayed unimportable: {why}"
                    return
                res.unimported_reasons[why] += 1
                record = cat.execute(by_color[st.current_color()].decide(cat, cat.playable_actions))
                res.steps += 1
                res.unimported_steps += 1
                res.trace = (res.trace + [repr(record)])[-TRACE_LEN:]
                theirs = catan_snapshot(cat)
                continue
        unimported = 0
        actor = seat(st, st.current_color())
        if ours.current_actor != actor:
            res.mismatch = f"step {res.steps}: actor is {ours.current_actor} in ours, {actor} in Catanatron"
            return
        before = ours.snapshot()
        mine, catans = our_tokens(ours), catan_tokens(st, cat.playable_actions)
        for token in sorted(mine ^ catans, key=repr):
            side = "catanatron" if token in catans else "ours"
            kind = allowlist.classify_legal(token, before, actor, side)
            if kind is None:
                res.mismatch = f"step {res.steps}: {token!r} is legal only in {side}"
                return
            res.allowlisted[kind] += 1
        record = cat.execute(by_color[st.current_color()].decide(cat, cat.playable_actions))
        res.steps += 1
        res.trace = (res.trace + [repr(record)])[-TRACE_LEN:]
        theirs = catan_snapshot(cat)
        try:
            for a, chance in our_steps(cat.state, record):
                if a not in ours.legal_actions():
                    kind = allowlist.classify_legal(a, before, actor, "catanatron")
                    raise Untranslatable(kind or "illegal-in-ours", f"action {a}")
                if chance is None:
                    ours.apply(a)
                else:
                    ours.apply_forced(a, chance)
        except Untranslatable as e:
            kind = e.kind if e.kind in allowlist.ENTRIES else _executed_kind(cat.state, record.action, before, actor)
            if kind is None:
                res.mismatch = f"step {res.steps}: {e}"
                return
            res.allowlisted[kind] += 1
            ours, last_kind = None, kind  # re-imported from Catanatron at the top of the loop
            continue
        now = ours.snapshot()
        diffs = compare(now, theirs)
        if diffs:
            at = record.action.action_type
            roll = sum(record.result) if at == ActionType.ROLL else None
            kind = allowlist.classify_state(diffs, at, before, now, theirs, roll=roll)
            if kind is None:
                res.mismatch = f"step {res.steps} after {record.action}: {diffs}"
                return
            res.allowlisted[kind] += 1
            ours, last_kind = None, kind
    res.final_phase, res.winner = theirs["phase"], theirs["winner"]
    res.ended_unimported = ours is None and unimported > 0


def run(seeds, workers: int | None = None) -> list[GameResult]:
    seeds = list(seeds)
    n = max(1, min(workers or os.cpu_count() or 1, len(seeds)))
    with hash_seed_zero():
        pool = mp.get_context("spawn").Pool(n)
    with pool:
        return pool.map(run_game, seeds, chunksize=max(1, len(seeds) // (n * 8)))


@dataclass
class Summary:
    games: int
    steps: int
    clean_games: int
    allowlisted: Counter
    mismatches: list[GameResult]
    allowlisted_games: Counter = field(default_factory=Counter)  # games with at least one event
    unimported_steps: int = 0  # Catanatron steps played while its position could not be imported
    unimported_reasons: Counter = field(default_factory=Counter)
    ended_unimported: int = 0  # games whose end fell inside such a stretch (end not compared)


def summarize(results) -> Summary:
    results = list(results)
    total, games = Counter(), Counter()
    for r in results:
        total.update(r.allowlisted)
        games.update(r.allowlisted.keys())
    return Summary(
        games=len(results),
        steps=sum(r.steps - r.unimported_steps for r in results),
        clean_games=sum(1 for r in results if not r.allowlisted and r.mismatch is None),
        allowlisted=total,
        mismatches=[r for r in results if r.mismatch is not None],
        allowlisted_games=games,
        unimported_steps=sum(r.unimported_steps for r in results),
        unimported_reasons=sum((r.unimported_reasons for r in results), Counter()),
        ended_unimported=sum(r.ended_unimported for r in results),
    )


def format_summary(s: Summary) -> str:
    lines = [f"{s.games} games, {s.steps} steps compared, {s.clean_games} games identical end to end, "
             f"{len(s.mismatches)} mismatches"]
    for kind, n in sorted(s.allowlisted.items()):
        lines.append(f"  allowlisted {kind}: {n} events in {s.allowlisted_games[kind]} games "
                     f"({allowlist.ENTRIES[kind].rule})")
    if s.unimported_steps:
        lines.append(f"  {s.unimported_steps} Catanatron steps not compared: its position after an "
                     "allowlisted difference could not be imported")
        for why, n in sorted(s.unimported_reasons.items()):
            lines.append(f"    {n}x {why}")
    if s.ended_unimported:
        lines.append(f"  {s.ended_unimported} games ended while uncompared, so their end was not checked")
    return "\n".join(lines)
