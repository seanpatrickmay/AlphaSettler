"""Differential replay of robots-only JSettlers games through our engine (spec done criterion 2).

Four stock smart robots play in the harness on the seed's board and dice. Every step of the stream
(oracle/jsettlers/events.py) is replayed in our engine with its chance outcomes forced (the roll,
the steal, the card drawn), and the position after it is compared with JSettlers'. A step our rules
cannot express carries an allowlist kind: its translatable actions replay, the position is compared
with what that kind's rule changes done as the rule does it (`excused`), and our engine re-imports
JSettlers' position. Anything else is a mismatch, which ends that game's comparison; so does a
failure of the Tracker or of an import. Positions import with the bank limit off and no offers
(robots' trades are replayed by re-import). A game that reaches our turn cap is stopped."""

from __future__ import annotations

import copy
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
from oracle.jsettlers.tables import RESOURCE_NAMES, TILE_NODES
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


def traded(st: Step) -> tuple[list[int], list[int]]:
    """The cards a bank trade step gave and got, summed over its maritime trades."""
    gave, got = [0] * 5, [0] * 5
    for e in st.events:
        if e["type"] == "maritime_traded":
            gave = [a + b for a, b in zip(gave, e["gave"])]
            got = [a + b for a, b in zip(got, e["got"])]
    return gave, got


def undo_fault(rates: list[int], gave: list[int], got: list[int], last) -> str | None:
    """Why an allowlisted bank trade is not JSettlers' undo of the player's last bank trade this turn,
    or None. An undo is the exact reverse of whole maritime trades at the player's rates (`rates`, our
    resource order): it gets back `rate` cards of a resource for each card it gives, never a resource
    on both sides, and it reverses `last`, the player's last bank trade this turn as (gave, got)."""
    units = sum(g // rate for g, rate in zip(got, rates))
    if (not any(gave) or any(g % rate for g, rate in zip(got, rates)) or units != sum(gave)
            or any(a and b for a, b in zip(gave, got))):
        return f"{gave} for {got} is not the undo of whole maritime trades at rates {rates}"
    if last != (got, gave):
        return f"{gave} for {got} reverses no bank trade of the player this turn (last: {last})"
    return None


def excused(ours: dict, theirs: dict, st: Step) -> dict:
    """Our position `ours` (before the step, its translatable actions applied) with what the step's
    allowlist rule changes done as the rule does it: a domestic trade moves exactly the traded cards
    between the two traders, a bank trade exactly the cards traded with the bank, and a turn that ends
    (forced, or inside Road Building) passes to the next player's roll as our end of turn does. A setup turn
    the server ended loses the settlement awaiting its road and what that settlement produced; the setup
    order after it (current player, phase, step, turn) is JSettlers' own, the one field set read from `theirs`.
    Otherwise `theirs` is not read: compared with JSettlers' position, anything that differs is a mismatch."""
    exp = copy.deepcopy(ours)
    hands = [p["hand"] for p in exp["players"]]

    def move(player: int, gain, loss) -> None:
        hands[player][:] = [h + g - x for h, g, x in zip(hands[player], gain, loss)]

    if st.untranslatable == "domestic-trade":
        i = st.info
        move(i["offerer"], i["get"], i["give"])
        move(i["partner"], i["give"], i["get"])
    elif st.untranslatable == "bank-trade":
        gave, got = traded(st)
        move(st.player, got, gave)
    elif st.untranslatable == "forced-setup-end":
        node = exp["setup_road_node"] if exp["phase"] == "setup_road" else None
        if node is not None:
            exp["players"][st.player]["settlements"].remove(node)
            if exp["setup_step"] >= 4:  # a second settlement: what it produced goes back
                res = exp["board"]["tile_resource"]
                for t, nodes in enumerate(TILE_NODES):
                    if node in nodes and res[t] is not None:
                        hands[st.player][RESOURCE_NAMES.index(res[t])] -= 1
        for k in ("current", "phase", "setup_road_node", "setup_step", "turn", "return_phase", "roads_left"):
            exp[k] = theirs[k]
    if st.untranslatable in ("domestic-trade", "bank-trade", "forced-setup-end"):
        exp["bank"] = [max(0, 19 - sum(h[r] for h in hands)) for r in range(5)]  # the bank with its limit off
    if any(e["type"] == "turn_ended" for e in st.events):
        exp.update(current=(exp["current"] + 1) % 4, phase="pre_roll", return_phase="pre_roll", roads_left=None,
                   setup_road_node=None, dev_played_this_turn=False, offers_this_turn=0, turn=exp["turn"] + 1)
        for q in exp["players"]:
            q["dev_new"], q["discard_remaining"] = [0] * 5, 0
    return exp


def _replay(res: GameResult, seed: int, tracker: Tracker, ours: Game, st: Step, trades: dict) -> Game:
    """Replay one step on `ours` and compare; returns the game to go on with (JSettlers' position,
    re-imported, after an allowlisted step). `trades` keeps each player's last bank trade as
    (turn, gave, got), to check an undo against."""
    if st.kind == "end_turn" and tracker.ctx.turn >= MAX_TURNS:
        res.capped = True  # our engine ends the game here; JSettlers would play on
        return ours
    kind = st.untranslatable
    if kind is not None and kind not in allowlist.ENTRIES:
        res.mismatch = f"step {res.steps} ({st.kind}): our rules cannot replay it ({kind})"
        return ours
    turn = ours.turn
    if st.actions:  # an allowlisted step's translatable actions (a roll before a forced end) replay too
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
        if kind is None:
            res.steps += 1
    if st.after is None:
        if kind is not None:
            res.mismatch = f"step {res.steps} ({st.kind}): an allowlisted step ({kind}) without JSettlers' position"
        return ours
    try:
        theirs = snapshot(st.after, tracker.ctx, 0)
    except Untranslatable as e:
        res.mismatch = f"step {res.steps} ({st.kind}): JSettlers' position after it does not translate: {e}"
        return ours
    if st.kind == "maritime":
        last = trades.get(st.player)
        trades[st.player] = (turn, *traded(st))
        if kind == "bank-trade":
            why = undo_fault(tracker.rates(st.after, st.player), *traded(st),
                             last[1:] if last is not None and last[0] == turn else None)
            if why is not None:
                res.mismatch = f"step {res.steps} ({st.kind}) by {st.player} ({kind}): {why}"
                return ours
    if kind is None:
        diffs = compare(ours.snapshot(), theirs)
        if diffs:
            res.mismatch = f"step {res.steps} after {st.kind} by {st.player}: {diffs}"
        return ours
    res.allowlisted[kind] += 1
    diffs = compare(excused(ours.snapshot(), theirs, st), theirs)
    if diffs:
        res.mismatch = f"step {res.steps} after {st.kind} by {st.player} ({kind}): beyond what its rule changes: {diffs}"
        return ours
    try:
        return Game.from_snapshot(seed, theirs, CONFIG)
    except ValueError as e:
        res.mismatch = f"step {res.steps} ({st.kind}): JSettlers' position does not import: {e}"
        return ours


def _advance(res: GameResult, seed: int, tracker: Tracker | None, ours: Game | None, m: dict, trades: dict):
    if m["t"] == "start":
        tracker = Tracker(m)
        return tracker, import_game(seed, tracker.raw, tracker.ctx, CONFIG)
    for st in tracker.feed(m) if m["t"] == "ev" else tracker.flush():
        if res.mismatch is None and not res.capped:
            ours = _replay(res, seed, tracker, ours, st, trades)
    return tracker, ours


def play_game(jvm: Jvm, seed: int, k: int | None = None) -> GameResult:
    """One robots-only game on `jvm`, replayed step by step. A broken harness (the JVM's error, exit
    or silence) raises; a failure of ours (the Tracker, a translation, the start position's import)
    is this game's mismatch. The trace holds the stream up to the mismatch or the turn cap."""
    k = seed % 4 if k is None else k
    name = f"diff-{seed}-{k}"
    res = GameResult(seed)
    jvm.send(game_command(name, seed, k, bridge=False))
    tracker, ours, aborted, trades = None, None, False, {}
    while True:
        m = jvm.recv(GAME_TIMEOUT)
        if m.get("g") != name:
            raise RuntimeError(f"a message from another game while replaying {name}: {m}")
        if res.mismatch is None and not res.capped:
            if m["t"] == "ev":
                res.trace = (res.trace + [f"{m['a']}:{m['m']}"])[-TRACE_LEN:]
            try:
                tracker, ours = _advance(res, seed, tracker, ours, m, trades)
            except BaseException as e:  # noqa: BLE001 - a panic (PyO3 PanicException) is a finding too
                if isinstance(e, (KeyboardInterrupt, SystemExit)):
                    raise
                res.mismatch = f"step {res.steps} ({m['t']} message): {type(e).__name__}: {e}"
        if m["t"] == "over":
            res.winner = None if m["winner"] < 0 or tracker is None else tracker.ctx.ours(m["winner"])
            res.turns = 0 if tracker is None else tracker.ctx.turn
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
