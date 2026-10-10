"""An AlphaSettler bot as one seat against three JSettlers2 smart robots in JSettlers' own server
(spec Sections 2 and 5). The bridge brain asks at each decision; `Seat` imports the latest position
in the stream, asks the bot (which sees only its own observation) and answers in JSettlers' terms.
A position or choice the translation cannot express answers `fallback` and the stock brain decides
that hook, counted by reason. Our bot is shown every step's events, redacted for its seat.

Seeds: a seed fixes the board, dice and deck (common random numbers); the candidate takes turn
position k in the seed's k-th game, with bot seed `seed * 4 + k`. Each worker process runs one JVM
and plays its games in turn."""

from __future__ import annotations

import multiprocessing as mp
import os
import time
from collections import Counter
from pathlib import Path

from alphasettler import Bot, Game
from alphasettler.arena import TRADES
from oracle.jsettlers import actions as A
from oracle.jsettlers import missing
from oracle.jsettlers.boards import game_command
from oracle.jsettlers.events import Tracker
from oracle.jsettlers.jvm import Jvm
from oracle.jsettlers.tables import (
    OUR_EDGE_TO_JS_EDGE, OUR_NODE_TO_JS_NODE, OUR_RES_TO_JS, OUR_TILE_TO_JS_HEX, PLAY1, RESOURCE_NAMES, js5_to_ours,
    ours_to_js5,
)
from oracle.jsettlers.translate import Untranslatable, snapshot

BASELINES = ("smart",)
NO_TRADES = {"max_offers_per_turn": 0}
LATE = 4.5  # seconds: robots wait 5 s for answers in robots-only games
SILENCE = 900.0  # seconds without a message before the run fails
MAX_TURNS = 1000
FALLBACK_REASONS = ("position", "declined", "engine", "choice", "preroll-card", "turn-cap", "java-timeout", "victim",
                    "forced-discard")
AFTER_WIN = "after-win"  # a decision JSettlers asks once our rules have ended the game: answered, not counted
COUNTERS = ("our_offers", "our_offers_accepted", "offers_to_us", "offers_to_us_accepted", "unrepresentable",
            "counters_rejected", "late", "robot_offers", "robot_offers_over_caps")
RECORD_KEYS = ("seed", "candidate_seat", "winner", "vp", "turns", "rounds", "aborted", "robot_desync", "last_build_turn",
               "fallbacks", "fallback_reasons", "belief_resets", *COUNTERS, "bank_limit_hits", "arm")
BUILD_STEPS = frozenset({"build", "road_building", "buy"})  # a piece built (setup aside) or a card bought


class Fallback(Exception):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def config(trades: bool) -> dict:
    return {**(TRADES if trades else NO_TRADES), "bank_limit": False}


class Seat:
    def __init__(self, bot, seed: int, config: dict, tracker: Tracker, seat: int, max_turns: int = MAX_TURNS,
                 clock=time.monotonic):
        self.bot, self.seed, self.config, self.tracker, self.seat = bot, seed, config, tracker, seat
        self.max_turns, self.clock = max_turns, clock
        self.fallbacks: Counter = Counter()
        self.counts: Counter = Counter({k: 0 for k in COUNTERS})
        self.capped = False
        # Per seat of ours, the last turn it built a piece or bought a card (None if never): stock robots can stop
        # building at 8-9 VP when nobody holds Longest Road (stall-debug.md), and Task 9 reports those games apart.
        self.last_build: list[int | None] = [None] * 4
        self._declined_seen = 0
        self._virtual, self._virtual_turn = 0, 0  # offers declined inside our engine only, this turn
        self._robot_offers: Counter = Counter()  # (turn, seat) -> offers made
        # Our seat's next discard as the stream should report it: None (none asked), "stock" (the stock brain
        # decides it, unseen by our bot) or the resource names our bot was shown while it chose them.
        self._discard_expect: Counter | str | None = None

    # --- the stream ---

    def observe(self, steps) -> None:
        for st in steps:
            if st.kind in BUILD_STEPS and st.player is not None and st.actions:
                self.last_build[st.player] = self.tracker.ctx.turn
            if st.kind == "offer" and st.player != self.seat:
                self.counts["robot_offers"] += 1
                key = (self.tracker.ctx.turn, st.player)
                self._robot_offers[key] += 1
                i = st.info
                others = sorted(q for q in range(4) if q != st.player)
                within = (sum(i["give"]) == 1 and sum(i["get"]) == 1 and i["give"] != i["get"]
                          and i["to"] == others and self._robot_offers[key] <= 2)
                if not within:
                    self.counts["robot_offers_over_caps"] += 1
            if st.kind == "trade":
                if st.info["offerer"] == self.seat:
                    self.counts["our_offers_accepted"] += 1
                elif st.info["partner"] == self.seat:
                    self.counts["offers_to_us_accepted"] += 1
            self.bot.observe(self.seat, [A.redact(e, self.seat) for e in self._unseen(st.events)])

    def _unseen(self, events: list[dict]) -> list[dict]:
        """`events` less our seat's discards the bot was shown while it chose them (`_discard`). A discard of
        ours that no reply of ours asked for (the server force-discarded, before our reply or with none) is
        counted (`forced-discard`) and shown; what our bot was shown instead then resets its belief, counted.
        Either way the expectation ends with the step that reports our discard."""
        ours = Counter(e["resource"] for e in events if e["type"] == "discarded" and e["player"] == self.seat)
        if not ours:
            return events
        expect, self._discard_expect = self._discard_expect, None
        if expect == ours:
            return [e for e in events if not (e["type"] == "discarded" and e["player"] == self.seat)]
        if expect != "stock":
            self.fallbacks["forced-discard"] += 1
        return events

    def fellback(self, reason: str) -> None:
        self.fallbacks[reason] += 1

    # --- decisions ---

    def decide(self, ask: dict, received: float) -> dict:
        if ask["kind"] == "discard":
            self._discard_expect = "stock"  # unless `_discard` replies with the cards it showed
        if self.tracker.over:
            # Our rules ended the game (a knight's Largest Army reached the target) while JSettlers plays on to
            # the robber move: no decision of ours is owed, so the stock brain moves it, uncounted.
            return {"t": "reply", "id": ask["id"], "fallback": AFTER_WIN}
        try:
            if self.capped or self.tracker.ctx.turn >= self.max_turns:
                raise Fallback("turn-cap")
            reply = getattr(self, "_" + ask["kind"])(ask.get("extra") or {}, received)
        except Fallback as f:
            self.fallbacks[f.reason] += 1
            return {"t": "reply", "id": ask["id"], "fallback": f.reason}
        return {"t": "reply", "id": ask["id"], **reply}

    def _act(self, game: Game) -> int:
        try:
            return self.bot.act(game)
        except BaseException as e:  # noqa: BLE001 - an engine panic (PyO3 PanicException) is a counted fallback
            if isinstance(e, (KeyboardInterrupt, SystemExit)):
                raise
            raise Fallback("engine") from e

    def _snapshot(self) -> dict:
        if self.tracker.ctx.turn != self._virtual_turn:
            self._virtual, self._virtual_turn = 0, self.tracker.ctx.turn
        cap = self.config.get("max_offers_per_turn", 0)
        try:
            snap = snapshot(self.tracker.raw, self.tracker.ctx, cap)
        except Untranslatable:
            raise Fallback("position") from None
        snap["offers_this_turn"] = min(snap["offers_this_turn"] + self._virtual, cap)
        return snap

    def _import(self, snap: dict, *phases: str) -> Game:
        if snap["phase"] not in phases:
            raise Fallback("position")
        try:
            g = Game.from_snapshot(self.seed, snap, self.config)
        except ValueError:
            raise Fallback("position") from None
        if g.current_actor != self.seat:
            raise Fallback("position")
        return g

    def _game(self, *phases: str) -> Game:
        return self._import(self._snapshot(), *phases)

    def _check_declined(self) -> None:
        """A request of ours JSettlers declined since the last decision: the stock brain decides this one,
        so the same refused request is not sent again in a loop."""
        if self.tracker.declined[self.seat] != self._declined_seen:
            self._declined_seen = self.tracker.declined[self.seat]
            raise Fallback("declined")

    def _setup_settlement(self, extra, received) -> dict:
        self._check_declined()
        kind, *v = A.decode(self._act(self._game("setup_settlement")))
        if kind != "settlement":
            raise Fallback("choice")
        return {"node": OUR_NODE_TO_JS_NODE[v[0]]}

    def _setup_road(self, extra, received) -> dict:
        self._check_declined()
        kind, *v = A.decode(self._act(self._game("setup_road")))
        if kind != "road":
            raise Fallback("choice")
        return {"edge": OUR_EDGE_TO_JS_EDGE[v[0]]}

    def _pre_roll(self, extra, received) -> dict:
        self._check_declined()
        a = self._act(self._game("pre_roll"))
        if a == A.ROLL:
            return {"do": "roll"}
        if a == A.KNIGHT:
            return {"do": "knight"}
        raise Fallback("preroll-card")  # the stock turn machine plays only knights before the roll

    def _road_building(self, extra, received) -> dict:
        self._check_declined()
        kind, *v = A.decode(self._act(self._game("road_building")))
        if kind != "road":
            raise Fallback("choice")
        return {"edge": OUR_EDGE_TO_JS_EDGE[v[0]]}

    def _move_robber(self, extra, received) -> dict:
        self._check_declined()
        g = self._game("move_robber")
        a = self._act(g)
        kind, *v = A.decode(a)
        if kind != "move_robber":
            raise Fallback("choice")
        after = g.copy()
        after.apply(a)
        victim = -1
        if after.phase == "steal":
            k2, *w = A.decode(self._act(after))
            if k2 != "steal_from":
                raise Fallback("choice")
            victim = self.tracker.ctx.js(w[0])
        return {"hex": OUR_TILE_TO_JS_HEX[v[0]], "victim": victim}

    def _discard(self, extra, received) -> dict:
        """Our rules discard one card a decision: the bot sees each discard before it chooses the next (its belief
        checks its hand), and the same discards in the stream are not shown to it again. A choice that fails after
        the first card is shown completes with the first discards our rules allow (counted), so the reply is what
        the bot was shown."""
        self._check_declined()
        snap = self._snapshot()
        if snap["phase"] != "discard":
            raise Fallback("position")
        n = int(extra["n"])
        for i, p in enumerate(snap["players"]):
            p["discard_remaining"] = n if i == self.seat else 0
        g = self._import(snap, "discard")
        cards, shown, filling = [0] * 5, Counter(), False
        for i in range(n):
            try:
                if filling:
                    raise Fallback("filling")
                a = self._act(g)
                if A.decode(a)[0] != "discard":
                    raise Fallback("choice")
            except Fallback as f:
                if i == 0:
                    raise
                if not filling:
                    self.fallbacks[f.reason] += 1
                    filling = True
                a = next(x for x in g.legal_actions() if A.decode(x)[0] == "discard")
            r = A.decode(a)[1]
            cards[r] += 1
            seen = len(g.log(self.seat))
            g.apply(a)
            self.bot.observe(self.seat, list(g.log(self.seat))[seen:])
            shown[RESOURCE_NAMES[r]] += 1
        self._discard_expect = shown
        return {"res": ours_to_js5(cards)}

    def _main(self, extra, received) -> dict:
        self._check_declined()
        for _ in range(self.config.get("max_offers_per_turn", 0) + 1):
            g = self._game("main")
            a = self._act(g)
            if A.is_offer(a):
                reply = self._offer(g, a)
                if reply is None:
                    continue  # declined inside our engine: decide again
                return reply
            return self._main_reply(a)
        raise Fallback("choice")  # unreachable: our engine caps the offers a turn

    def _main_reply(self, a: int) -> dict:
        kind, *v = A.decode(a)
        if kind == "end_turn":
            return {"do": "end"}
        if kind == "road":
            return {"do": "build", "piece": 0, "coord": OUR_EDGE_TO_JS_EDGE[v[0]]}
        if kind in ("settlement", "city"):
            return {"do": "build", "piece": 1 if kind == "settlement" else 2, "coord": OUR_NODE_TO_JS_NODE[v[0]]}
        if kind == "buy_dev":
            return {"do": "buy"}
        if kind == "knight":
            return {"do": "knight"}
        if kind == "road_building":
            return {"do": "rb"}
        if kind == "year_of_plenty":
            x, y = A.PAIRS[v[0]]
            return {"do": "yop", "res": [OUR_RES_TO_JS[x], OUR_RES_TO_JS[y]]}
        if kind == "monopoly":
            return {"do": "mono", "res": OUR_RES_TO_JS[v[0]]}
        if kind == "maritime":
            give, get = divmod(v[0], 5)
            gave, got = [0] * 5, [0] * 5
            gave[give], got[get] = self.tracker.rates(self.tracker.raw, self.seat)[give], 1
            return {"do": "bank", "give": ours_to_js5(gave), "get": ours_to_js5(got)}
        raise Fallback("choice")

    def _offer(self, g: Game, a: int) -> dict | None:
        """Our offer goes to the opponents whose lone acceptance our bot would confirm. With none, it is
        made and declined inside our engine only (our bot sees its events), and the bot decides again."""
        self.counts["our_offers"] += 1
        give, get = A.offer_hands(a)
        to = []
        for q in range(4):
            if q == self.seat:
                continue
            h = g.copy()
            h.apply(a)
            ok = True
            while h.phase == "trade_response":
                if h.current_actor == q:
                    if A.ACCEPT not in h.legal_actions():
                        ok = False
                        break
                    h.apply(A.ACCEPT)
                else:
                    h.apply(A.REJECT)
            if ok and h.phase == "trade_confirm" and self._act(h) == A.confirm(q):
                to.append(q)
        if not to:
            h = g.copy()
            h.apply(a)
            while h.phase == "trade_response":
                h.apply(A.REJECT)
            self.bot.observe(self.seat, list(h.log(self.seat)))
            self._virtual += 1
            return None
        js_to = [False] * 4
        for q in to:
            js_to[self.tracker.ctx.js(q)] = True
        return {"do": "offer", "to": js_to, "give": ours_to_js5(give), "get": ours_to_js5(get)}

    def _answer(self, extra, received) -> dict:
        """A robot's offer to us: answered at the offerer's Main position, its earlier offers this turn
        clamped to 1, the offer applied and the seats before ours rejecting."""
        self.counts["offers_to_us"] += 1
        ctx, raw = self.tracker.ctx, self.tracker.raw
        if raw["current"] == ctx.js(self.seat):
            self.counts["counters_rejected"] += 1
            return {"accept": False}
        frm = ctx.ours(int(extra["from"]))
        give, get = js5_to_ours(extra["give"]), js5_to_ours(extra["get"])
        a = A.offer(give, get)
        cap = self.config.get("max_trade_cards", 0) if self.config.get("max_offers_per_turn", 0) else 0
        if (a is None or sum(give) > cap or sum(get) > cap or any(x and y for x, y in zip(give, get))
                or raw["state"] != PLAY1 or frm != ctx.ours(raw["current"])):
            self.counts["unrepresentable"] += 1
            return {"accept": False}
        snap = self._snapshot()
        snap["offers_this_turn"] = min(max(ctx.offers - 1, 0), 1)
        g = self._import_answer(snap, a)
        accept = self._act(g) == A.ACCEPT
        if self.clock() - received >= LATE:
            self.counts["late"] += 1
            return {"accept": False}
        return {"accept": accept}

    def _import_answer(self, snap: dict, a: int) -> Game:
        if snap["phase"] != "main":
            raise Fallback("position")
        try:
            g = Game.from_snapshot(self.seed, snap, self.config)
        except ValueError:
            raise Fallback("position") from None
        if a not in g.legal_actions():
            raise Fallback("position")
        g.apply(a)
        while g.phase == "trade_response" and g.current_actor != self.seat:
            g.apply(A.REJECT)
        if g.phase != "trade_response":
            raise Fallback("position")
        return g


def play_game(jvm: Jvm, seed: int, k: int, candidate: str, nets_dir: str | None = None, trades: bool = True,
              arm: str = "native", native_boards: bool = False, max_turns: int = MAX_TURNS) -> dict:
    """One game: the candidate at turn position k against three smart robots. A broken harness raises."""
    name = f"{arm}-{seed}-{k}"
    cfg = config(trades)
    bot = Bot(candidate, seed * 4 + k, nets_dir)
    jvm.send(game_command(name, seed, k, injected=not native_boards, no_trades=not trades, bridge=True))
    tracker = seat = None
    aborted = False
    while True:
        m = jvm.recv(SILENCE)
        if m.get("g") != name:
            # The bridge of an earlier, aborted game may still ask (or time out) after its "over": answer and drop.
            if m["t"] == "ask":
                jvm.send({"t": "reply", "id": m["id"], "fallback": "turn-cap"})
                continue
            if m["t"] == "fallback":
                continue
            raise RuntimeError(f"a message from another game while playing {name}: {m}")
        t = m["t"]
        if t == "start":
            tracker = Tracker(m)
            seat = Seat(bot, seed, cfg, tracker, tracker.ctx.ours(0), max_turns)
            if seat.seat != k:
                raise RuntimeError(f"{name}: the bridge has turn position {seat.seat}, not {k}")
            bot.observe(seat.seat, [])
            continue
        if t == "ev":
            seat.observe(tracker.feed(m))
        elif t == "ask":
            received = time.monotonic()
            seat.observe(tracker.flush())
            jvm.send(seat.decide(m, received))
        elif t == "fallback":
            seat.fellback(m["reason"])
        elif t == "over":
            seat.observe(tracker.flush())
            ctx = tracker.ctx
            return {
                "seed": seed, "candidate_seat": k,
                "winner": None if m["winner"] < 0 else ctx.ours(m["winner"]),
                "vp": [m["vp"][ctx.js(i)] for i in range(4)],
                "turns": ctx.turn, "rounds": m["rounds"], "aborted": m["aborted"],
                # A setup settlement the server cancelled by a forced end: a stock robot keeps it in its own copy of
                # the game when the turn was its own, and plays the rest of the game on a board it does not have.
                "robot_desync": bool(tracker.setup_cancelled),
                "last_build_turn": list(seat.last_build),
                "fallbacks": sum(seat.fallbacks.values()), "fallback_reasons": dict(seat.fallbacks),
                "belief_resets": bot.diagnostics().get("belief_resets", 0),
                **{c: seat.counts[c] for c in COUNTERS},
                "bank_limit_hits": tracker.bank_limit_hits, "arm": arm,
            }
        else:
            raise RuntimeError(f"{name}: unexpected message {m}")
        if not aborted and tracker.ctx.turn >= max_turns:
            jvm.send({"t": "abort", "g": name})
            aborted = seat.capped = True


def check(candidate: str, baseline: str, seeds: int, seed_start: int = 0, nets_dir: str | None = None) -> None:
    """Raise ValueError for anything `run_match` would reject, without starting a JVM."""
    if baseline not in BASELINES:
        raise ValueError(f"unknown JSettlers robot {baseline!r}; known: {list(BASELINES)}")
    Bot(candidate, 0, nets_dir)  # raises ValueError for a bad name or a missing or invalid net
    if seeds <= 0:
        raise ValueError("seeds must be positive")
    if seed_start < 0 or (seed_start + seeds) * 4 > 2**64:
        raise ValueError(f"seeds {seed_start}..{seed_start + seeds} are outside the bot seed range")
    why = missing()
    if why is not None:
        raise ValueError(f"the JSettlers harness is not ready: {why}")


def arm_name(trades: bool, aligned: bool, native_boards: bool) -> str:
    if native_boards:
        return "native-boards"
    if not trades:
        return "no-trades"
    return "aligned" if aligned else "native"


def _chunk(jobs, candidate, nets_dir, trades, aligned, native_boards, max_turns, log_dir) -> list[dict]:
    arm = arm_name(trades, aligned, native_boards)
    log = None if log_dir is None else Path(log_dir) / f"{arm}-{os.getpid()}.log"
    with Jvm("aligned" if aligned else "stock", log=log) as jvm:
        return [play_game(jvm, s, k, candidate, nets_dir, trades, arm, native_boards, max_turns) for s, k in jobs]


def run_match(candidate: str, baseline: str, seeds: int, seed_start: int = 0, workers: int | None = None,
              nets_dir: str | None = None, trades: bool = True, aligned: bool = False, native_boards: bool = False,
              max_turns: int = MAX_TURNS, log_dir: str | Path | None = None) -> list[dict]:
    """Four games a seed (the candidate in each turn position), one JVM per worker process. Each JVM's log
    goes under `log_dir` when given (else runs/jsettlers/logs/)."""
    if native_boards and not trades:
        raise ValueError("native boards need trades (that check is the native arm on JSettlers' own boards)")
    check(candidate, baseline, seeds, seed_start, nets_dir)
    jobs = [(s, k) for s in range(seed_start, seed_start + seeds) for k in range(4)]
    n = max(1, min(workers or os.cpu_count() or 1, len(jobs)))
    chunks = [jobs[i::n] for i in range(n)]
    args = [(c, candidate, nets_dir, trades, aligned, native_boards, max_turns, log_dir) for c in chunks]
    with mp.get_context("spawn").Pool(n) as pool:
        out = pool.starmap(_chunk, args)
    return sorted((r for chunk in out for r in chunk), key=lambda r: (r["seed"], r["candidate_seat"]))
