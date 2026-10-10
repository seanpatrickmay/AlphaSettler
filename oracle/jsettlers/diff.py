"""Differential replay of robots-only JSettlers games through our engine (spec done criterion 2).

Four stock smart robots play in the harness on the seed's board and dice. Every step of the stream
(oracle/jsettlers/events.py) is replayed in our engine with its chance outcomes forced (the roll,
the steal, the card drawn), and the position after it is compared with JSettlers'. A step our rules
cannot express carries an allowlist kind and our engine re-imports JSettlers' position; anything
else is a mismatch, which ends that game's comparison. Positions import with the bank limit off and
no offers (robots' trades are replayed by re-import). A game that reaches our turn cap is stopped."""

from __future__ import annotations

import multiprocessing as mp
import os
from collections import Counter
from dataclasses import dataclass, field

from alphasettler import Game
from oracle.compare import compare
from oracle.jsettlers import allowlist
from oracle.jsettlers.actions import decode
from oracle.jsettlers.boards import game_command
from oracle.jsettlers.events import Step, Tracker
from oracle.jsettlers.jvm import Jvm
from oracle.jsettlers.translate import Untranslatable, import_game, snapshot

CONFIG = {"max_offers_per_turn": 0, "bank_limit": False}
MAX_TURNS = 1000
GAME_TIMEOUT = 900.0
TRACE_LEN = 20


@dataclass
class GameResult:
    seed: int
    steps: int = 0  # steps with actions replayed and compared
    allowlisted: Counter = field(default_factory=Counter)
    mismatch: str | None = None
    trace: list = field(default_factory=list)
    winner: int | None = None
    turns: int = 0
    capped: bool = False


def _replay(res: GameResult, seed: int, tracker: Tracker, ours: Game | None, st: Step) -> Game | None:
    if st.kind == "end_turn" and tracker.ctx.turn >= MAX_TURNS:
        res.capped = True  # our engine ends the game here; JSettlers would play on
        return ours
    if st.untranslatable is not None:
        if st.untranslatable not in allowlist.ENTRIES:
            res.mismatch = f"step {res.steps} ({st.kind}): our rules cannot replay it ({st.untranslatable})"
            return ours
        res.allowlisted[st.untranslatable] += 1
        ours = None
    elif st.actions:
        try:
            for a, chance in st.actions:
                if a not in ours.legal_actions():
                    res.mismatch = f"step {res.steps} ({st.kind}): {decode(a)} is not legal in ours ({ours.phase})"
                    return ours
                if chance is None:
                    ours.apply(a)
                else:
                    ours.apply_forced(a, chance)
        except BaseException as e:  # noqa: BLE001 - a panic (PyO3 PanicException) is a finding too
            if isinstance(e, (KeyboardInterrupt, SystemExit)):
                raise
            res.mismatch = f"step {res.steps} ({st.kind}): {type(e).__name__}: {e}"
            return ours
        res.steps += 1
    if st.after is None:
        return ours
    try:
        theirs = snapshot(st.after, tracker.ctx, 0)
    except Untranslatable as e:
        res.mismatch = f"step {res.steps} ({st.kind}): JSettlers' position after it does not translate: {e}"
        return ours
    if ours is None:
        try:
            return Game.from_snapshot(seed, theirs, CONFIG)
        except ValueError as e:
            res.mismatch = f"step {res.steps} ({st.kind}): JSettlers' position does not import: {e}"
            return None
    diffs = compare(ours.snapshot(), theirs)
    if diffs:
        res.mismatch = f"step {res.steps} after {st.kind} by {st.player}: {diffs}"
    return ours


def play_game(jvm: Jvm, seed: int, k: int | None = None) -> GameResult:
    """One robots-only game on `jvm`, replayed step by step. A broken harness raises."""
    k = seed % 4 if k is None else k
    name = f"diff-{seed}-{k}"
    res = GameResult(seed)
    jvm.send(game_command(name, seed, k, bridge=False))
    tracker, ours, aborted = None, None, False
    while True:
        m = jvm.recv(GAME_TIMEOUT)
        if m.get("g") != name:
            raise RuntimeError(f"a message from another game while replaying {name}: {m}")
        if m["t"] == "start":
            tracker = Tracker(m)
            ours = import_game(seed, tracker.raw, tracker.ctx, CONFIG)
            continue
        if m["t"] == "ev":
            res.trace = (res.trace + [f"{m['a']}:{m['m']}"])[-TRACE_LEN:]
            steps = tracker.feed(m)
        else:
            steps = tracker.flush()
        for st in steps:
            if res.mismatch is None and not res.capped:
                ours = _replay(res, seed, tracker, ours, st)
        if m["t"] == "over":
            res.winner = None if m["winner"] < 0 else tracker.ctx.ours(m["winner"])
            res.turns = tracker.ctx.turn
            return res
        if (res.mismatch is not None or res.capped) and not aborted:
            jvm.send({"t": "abort", "g": name})
            aborted = True


def _chunk(seeds: list[int]) -> list[GameResult]:
    with Jvm() as jvm:
        return [play_game(jvm, s) for s in seeds]


def run(seeds, workers: int | None = None) -> list[GameResult]:
    """Each worker process runs one JVM and plays its share of the seeds in turn; one worker runs in
    this process."""
    seeds = list(seeds)
    n = max(1, min(workers or os.cpu_count() or 1, len(seeds)))
    if n == 1:
        return sorted(_chunk(seeds), key=lambda r: r.seed)
    chunks = [seeds[i::n] for i in range(n)]
    with mp.get_context("spawn").Pool(n) as pool:
        results = [r for chunk in pool.map(_chunk, chunks) for r in chunk]
    return sorted(results, key=lambda r: r.seed)


@dataclass
class Summary:
    games: int
    steps: int
    clean_games: int
    allowlisted: Counter
    allowlisted_games: Counter
    mismatches: list
    capped: int


def summarize(results) -> Summary:
    results = list(results)
    total, games = Counter(), Counter()
    for r in results:
        total.update(r.allowlisted)
        games.update(r.allowlisted.keys())
    return Summary(games=len(results), steps=sum(r.steps for r in results),
                   clean_games=sum(1 for r in results if not r.allowlisted and r.mismatch is None),
                   allowlisted=total, allowlisted_games=games,
                   mismatches=[r for r in results if r.mismatch is not None], capped=sum(r.capped for r in results))


def format_summary(s: Summary) -> str:
    lines = [f"{s.games} games, {s.steps} steps compared, {s.clean_games} games identical end to end, "
             f"{len(s.mismatches)} mismatches, {s.capped} stopped at the turn cap"]
    for kind, n in sorted(s.allowlisted.items()):
        lines.append(f"  allowlisted {kind}: {n} steps in {s.allowlisted_games[kind]} games "
                     f"({allowlist.ENTRIES[kind].rule})")
    return "\n".join(lines)
