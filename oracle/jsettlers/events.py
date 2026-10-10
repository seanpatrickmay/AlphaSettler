"""The harness's stream of one game as steps of ours (spec Section 2, "Events to our bot").

A client's request line starts a stretch; the server's messages up to the next request are what it
did. Each closed stretch becomes at most one `Step`: our actions with their forced chance outcomes
(for the replay), our engine's events (for our bot, before redaction) and the JSettlers position
after it. Two stretches make one action of ours when the first leaves a position we have no phase
for (a build request and its placement, a Year of Plenty or Monopoly card and its pick). Discards
come one player at a time in any order; their actions are released in our seat order once the
last is in. A step our rules cannot replay carries an allowlist kind (`untranslatable`) instead.
Our rules end the game the moment the player on turn reaches the target: when JSettlers plays on (a
knight's Largest Army wins there only after the robber moves) the step carries the position as our
rules leave it, and nothing after our game's end becomes a step. A setup turn the server ended (a stalled
placement) is its own allowlisted step, `forced-setup-end`."""

from __future__ import annotations

from dataclasses import dataclass, field

from oracle.jsettlers import actions as A
from oracle.jsettlers.boards import our_board
from oracle.jsettlers.messages import Entry, ours5, parse
from oracle.jsettlers.tables import (
    CITY, DEV_NAMES, EDGE_NODES, JS_DEV_TO_OURS, JS_EDGE_TO_OUR_EDGE, JS_HEX_TO_OUR_TILE, JS_NODE_TO_OUR_NODE,
    JS_RES, OVER, PLACING_CITY, PLACING_FREE_ROAD1, PLACING_FREE_ROAD2, PLACING_ROAD, PLACING_SETTLEMENT,
    RESOURCE_NAMES, ROAD, SETTLEMENT, SETUP_STATES, START2B, WAITING_FOR_DISCARDS, WAITING_FOR_DISCOVERY,
    WAITING_FOR_MONOPOLY, js5_to_ours,
)
from oracle.jsettlers.translate import Context, snapshot

_HALF_DONE = frozenset({PLACING_ROAD, PLACING_SETTLEMENT, PLACING_CITY, WAITING_FOR_DISCOVERY, WAITING_FOR_MONOPOLY})
VP_TO_WIN = 10  # the arena's game option VP, and our vp_to_win


def dice_pair(total: int) -> list[int]:
    """JSettlers reports a roll's total only; our forced roll and event use this pair."""
    a = min(total - 1, 6)
    return [a, total - a]


@dataclass
class Step:
    kind: str
    player: int | None  # our seat that acted
    actions: list = field(default_factory=list)  # [(action id, chance dict or None)], in order
    events: list = field(default_factory=list)  # our engine's events, unredacted
    after: dict | None = None  # the JSettlers position after it; None while the action is incomplete
    untranslatable: str | None = None  # an allowlist kind: the replay re-imports `after`
    info: dict = field(default_factory=dict)


def _played(p: int, card: str) -> dict:
    return {"type": "played_dev", "player": p, "card": card}


def _offered(p: int, give, get) -> dict:
    return {"type": "trade_offered", "player": p, "give": list(give), "get": list(get)}


def _declined(e: Entry) -> bool:
    """The server refused the request: a declined build or card, or a refused offer or bank trade."""
    return (e.name == "SOCDeclinePlayerRequest"
            or (e.name == "SOCDevCardAction" and e.f.get("actionType") == "CANNOT_PLAY")
            or (e.name == "SOCRejectOffer" and "reasonCode" in e.f))


def _won(raw: dict) -> dict | None:
    """The position as our rules leave it when the player on turn has reached the target but JSettlers
    plays on (a knight's Largest Army wins only after the robber moves there): our game is over, won by
    that player. None when JSettlers agrees (or nobody is there)."""
    cur = raw["current"]
    if raw["state"] == OVER or raw["state"] in SETUP_STATES or cur < 0 or raw["players"][cur]["vp"] < VP_TO_WIN:
        return None
    return {**raw, "state": OVER, "winner": cur}


class Tracker:
    def __init__(self, start: dict):
        self.board = our_board(start["board"])
        self.ctx = Context(self.board, start["first"])
        self.raw = start["s"]
        self.declined = [0] * 4
        # Per seat of ours, acceptances the server refused (`SOCRejectOffer` with a reason): the offer was gone
        # when the acceptance arrived. Not a declined request: no decision of that seat repeats it, so the next one
        # is not a fallback. `beaten_accepts`: a trade executed after the offer was made (every trade clears every
        # offer), so another player's acceptance won; `late_accepts`: none did, so the offerer withdrew it (its
        # 5 s wait ended first). The arena counts them `beaten` and `late`.
        self.late_accepts = [0] * 4
        self.beaten_accepts = [0] * 4
        self._trades = 0  # trades executed so far
        self._offer_trades: dict[int, int] = {}  # per offerer (ours), `_trades` when its latest offer was made
        self.bank_limit_hits = 0
        self.setup_cancelled: list[int] = []  # our seats whose setup settlement a forced end of turn removed
        self.over = False
        self._open: list[Entry] | None = None
        self._before = self.raw
        self._discards: dict[int, list[int]] = {}

    def hand(self, raw: dict, seat: int) -> list[int]:
        return js5_to_ours(raw["players"][self.ctx.js(seat)]["res"])

    def snapshot(self, max_offers: int = 0) -> dict:
        return snapshot(self.raw, self.ctx, max_offers)

    def feed(self, msg: dict) -> list[Step]:
        e = parse(msg["a"], msg["m"], msg.get("s"))
        out: list[Step] = []
        if e.from_client:
            out = self._close()
            self._open, self._before = [e], self.raw
        else:
            if self._open is None:
                self._open, self._before = [], self.raw
            self._open.append(e)
        if e.raw is not None:
            self.raw = e.raw
        return out

    def flush(self) -> list[Step]:
        """Close the open stretch (before a decision, and at the end of the game)."""
        return self._close()

    def _close(self) -> list[Step]:
        entries, before, after = self._open, self._before, self.raw
        self._open, self._before = None, self.raw
        if not entries or self.over:  # after the game is over in ours, nothing more replays
            return []
        req = entries[0] if entries[0].from_client else None
        server = [e for e in entries if not e.from_client]
        if req is not None and req.sender is not None and any(_declined(e) for e in server):
            if req.name == "SOCAcceptOffer":
                q, o = self.ctx.ours(req.sender), self.ctx.ours(int(req.f["offering"]))
                beaten = self._trades > self._offer_trades.get(o, self._trades)
                (self.beaten_accepts if beaten else self.late_accepts)[q] += 1
            else:
                self.declined[self.ctx.ours(req.sender)] += 1
        for r in range(5):
            b = sum(self.hand(before, q)[r] for q in range(4))
            a = sum(self.hand(after, q)[r] for q in range(4))
            if a > 19 and a > b:
                self.bank_limit_hits += 1
                break
        step = self._action(req, server, before, after)
        if (before["state"] not in SETUP_STATES and any(e.name == "SOCTurn" for e in server)
                and (step is None or step.kind != "end_turn")):
            step = self._forced_end(step, before, after)
        if (step is not None and step.after is not None and step.after["state"] in _HALF_DONE
                and not step.actions and step.untranslatable is None):
            # Events only (an offer or an answer to one) inside another player's build or card pick: the
            # position after it has no phase of ours, and the step that completes the action carries it.
            step.after = None
        steps = [] if step is None else [step]
        won = _won(after)
        if won is not None and steps:
            steps[-1].after = won
        end = won or after
        if end["state"] == OVER:
            self.over = True
            ev = {"type": "game_over", "winner": None if end["winner"] < 0 else self.ctx.ours(end["winner"])}
            if steps:
                steps[-1].events.append(ev)
            else:
                steps.append(Step("game_over", None, events=[ev], after=end))
        return steps

    def _action(self, req, server, before, after) -> Step | None:
        def first(name, pred=lambda e: True):
            return next((e for e in server if e.name == name and pred(e)), None)

        if (dice := first("SOCDiceResult")) is not None:
            return self._roll(int(dice.f["param"]), before, after)
        if (piece := first("SOCPutPiece", lambda e: e.aud == "all")) is not None:
            return self._piece(piece, before, after)
        if (robber := first("SOCMoveRobber", lambda e: e.aud == "all")) is not None:
            return self._robber(robber, server, after)
        if (robbery := self._robbery(server)) is not None:
            thief, victim, r = robbery
            return Step("steal", thief, [(A.steal_from(victim), {"steal": RESOURCE_NAMES[r]})],
                        [{"type": "stole", "thief": thief, "victim": victim, "resource": RESOURCE_NAMES[r]}], after)
        if (draw := first("SOCDevCardAction", lambda e: e.f.get("actionType") == "DRAW"
                          and e.f.get("cardType", "0") != "0")) is not None:
            return self._buy(draw, after)
        if (play := first("SOCDevCardAction", lambda e: e.f.get("actionType") == "PLAY")) is not None:
            return self._play(play, after)
        if (pick := first("SOCPickResources", lambda e: e.f.get("reason") == "2")) is not None:
            return self._plenty(pick, after)
        if (mono := first("SOCSimpleAction", lambda e: e.f.get("actType") == "3")) is not None:
            return self._monopoly(mono, before, after)
        if discs := [e for e in server
                     if e.name == "SOCDiscard" and "playerNum" in e.f and e.f["resources"]["unknown"] == 0]:
            return self._discard(discs, after)
        if (bank := first("SOCBankTrade", lambda e: "pn" in e.f)) is not None:
            return self._bank(bank, before, after)
        if (accept := first("SOCAcceptOffer", lambda e: "toAccepting" in e.f)) is not None:
            return self._trade(accept, after)
        if first("SOCTurn") is not None and before["state"] not in SETUP_STATES:
            return self._end_turn(before, after, requested=req is not None and req.name == "SOCEndTurn")
        cancelled = first("SOCCancelBuildRequest", lambda e: e.aud == "all" and e.f.get("pieceType") == "1")
        if before["state"] in SETUP_STATES and (cancelled is not None or (
                first("SOCTurn") is not None and (after["current"] != before["current"]
                                                  or after["state"] not in SETUP_STATES))):
            return self._forced_setup_end(before, after, cancelled is not None)
        if (made := first("SOCMakeOffer")) is not None:
            return self._offer(made, after)
        if (rejected := first("SOCRejectOffer", lambda e: "reasonCode" not in e.f)) is not None:
            q = self.ctx.ours(int(rejected.f["playerNumber"]))
            return Step("reject", q, [], [{"type": "trade_responded", "player": q, "accepted": False}], after,
                        info={"player": q})
        if (cleared := first("SOCClearOffer", lambda e: int(e.f["playerNumber"]) >= 0)) is not None:
            q = self.ctx.ours(int(cleared.f["playerNumber"]))
            return Step("clear", q, [], [{"type": "trade_cancelled", "player": q}], after)
        if after["state"] == OVER and req is not None and (step := self._cut_short(req, before, after)) is not None:
            return step
        if after["state"] in _HALF_DONE or after == before:
            return None
        return Step("noop", None, after=after)

    def _forced_setup_end(self, before, after, cancelled: bool) -> Step:
        """The server ended a stalled setup turn (SOCForceEndTurnThread): the settlement awaiting its road, if
        any, is removed with the cards it produced, and the setup order moves on (JSettlers may move its first
        player); a settlement of JSettlers' second round (cancelled in START2B) takes back the hand it paid
        (`info["second_round"]`, read from JSettlers' state: our setup step does not say the round after a
        skip). Our engine has no such end, and no event removes a settlement, so our bot is shown nothing and
        the replay re-imports. A stock robot whose own settlement is cancelled keeps it in its copy of the game
        (`setup_cancelled`, the arena's `robot_desync`)."""
        p = self.ctx.ours(before["current"])
        if cancelled:
            self.setup_cancelled.append(p)
        return Step("forced_setup_end", p, [], [], after, untranslatable="forced-setup-end",
                    info={"second_round": before["state"] == START2B})

    def _forced_end(self, step: Step | None, before, after) -> Step:
        """The server ended the turn (SOCForceEndTurnThread) in a stretch whose first rule read another
        action (a roll, a bank trade, a forced discard, a card left unpicked): the turn ends after it."""
        if step is None:
            return self._end_turn(before, after, requested=False)
        c = self.ctx
        c.turn, c.rolled, c.drawn, c.offers = c.turn + 1, False, [0] * 5, 0
        step.events.append({"type": "turn_ended", "player": c.ours(before["current"])})
        step.after = after
        if step.untranslatable is None:  # our replay re-imports the position after it
            step.untranslatable = "forced-end-turn"
        return step

    def _cut_short(self, req: Entry, before, after) -> Step | None:
        """The winning action when the stream ends before the server reports it: the harness stops at the
        first message recorded once the game is over, and JSettlers announces a winning placement or card
        in text first. The request says what was asked; the position after it says what was done."""
        if req.sender is None:
            return None
        js = req.sender
        if req.name == "SOCPutPiece" and int(req.f["playerNumber"]) == js:
            kind, coord = int(req.f["pieceType"]), req.num("coord")
            key = {ROAD: "roads", SETTLEMENT: "settlements", CITY: "cities"}.get(kind)
            if key is not None and coord in after["players"][js][key] and coord not in before["players"][js][key]:
                return self._piece(Entry("all", req.name, req.f), before, after)
        if req.name == "SOCBuyDevCardRequest":
            new = [c for c in range(10) if after["players"][js]["dev"][c] > before["players"][js]["dev"][c]]
            if len(new) == 1:
                return self._buy(Entry("all", "SOCDevCardAction", {"playerNum": str(js), "cardType": str(new[0])}),
                                 after)
        if req.name == "SOCEndTurn" and before["current"] == js and after["current"] != js:
            return self._end_turn(before, after)
        return None

    def _gain(self, before, after, q) -> list[int]:
        return [a - b for a, b in zip(self.hand(after, q), self.hand(before, q))]

    def _roll(self, total, before, after) -> Step:
        p = self.ctx.ours(before["current"])
        events = [{"type": "rolled", "player": p, "dice": dice_pair(total)}]
        bad = None
        if total != 7:
            for q in range(4):
                g = self._gain(before, after, q)
                if any(x < 0 for x in g):
                    bad = "production"
                elif any(g):
                    events.append({"type": "produced", "player": q, "resources": g})
        self.ctx.rolled = True
        return Step("roll", p, [(A.ROLL, {"roll": dice_pair(total)})], events, after, untranslatable=bad)

    def _piece(self, e, before, after) -> Step:
        p = self.ctx.ours(int(e.f["playerNumber"]))
        kind, coord = int(e.f["pieceType"]), e.num("coord")
        if kind == ROAD:
            edge = JS_EDGE_TO_OUR_EDGE[coord]
            action, ev = A.road(edge), {"type": "built_road", "player": p, "edge": edge}
        elif kind in (SETTLEMENT, CITY):
            node = JS_NODE_TO_OUR_NODE[coord]
            name = "settlement" if kind == SETTLEMENT else "city"
            action = A.settlement(node) if kind == SETTLEMENT else A.city(node)
            ev = {"type": f"built_{name}", "player": p, "node": node}
        else:
            return Step("noop", p, after=after, untranslatable="piece")
        events = [ev]
        state = before["state"]
        if state in SETUP_STATES:
            if kind == SETTLEMENT and any(g := self._gain(before, after, p)):
                events.append({"type": "produced", "player": p, "resources": g})
            return Step("setup", p, [(action, None)], events, after)
        if state in (PLACING_FREE_ROAD1, PLACING_FREE_ROAD2):
            return Step("road_building", p, [(action, None)], events, after)
        return Step("build", p, [(action, None)], events, after)

    def _robbery(self, server):
        e = next((x for x in server if x.name == "SOCRobberyResult" and x.f.get("resType", "6") != "6"), None)
        if e is None:
            return None
        return self.ctx.ours(int(e.f["perp"])), self.ctx.ours(int(e.f["victim"])), JS_RES[int(e.f["resType"])]

    def _robber(self, e, server, after) -> Step:
        p = self.ctx.ours(int(e.f["playerNumber"]))
        tile = JS_HEX_TO_OUR_TILE[e.num("coord")]
        actions = [(A.move_robber(tile), None)]
        events = [{"type": "robber_moved", "player": p, "tile": tile}]
        if (robbery := self._robbery(server)) is not None:
            thief, victim, r = robbery
            actions.append((A.steal_from(victim), {"steal": RESOURCE_NAMES[r]}))
            events.append({"type": "stole", "thief": thief, "victim": victim, "resource": RESOURCE_NAMES[r]})
        return Step("robber", p, actions, events, after)

    def _buy(self, e, after) -> Step:
        p = self.ctx.ours(int(e.f["playerNum"]))
        card = JS_DEV_TO_OURS[int(e.f["cardType"])]
        self.ctx.drawn[DEV_NAMES.index(card)] += 1
        return Step("buy", p, [(A.BUY_DEV, {"dev": card})], [{"type": "bought_dev", "player": p, "card": card}], after)

    def _play(self, e, after) -> Step | None:
        p = self.ctx.ours(int(e.f["playerNum"]))
        card = int(e.f["cardType"])
        if card == 9:
            return Step("knight", p, [(A.KNIGHT, None)], [_played(p, "knight")], after)
        if card == 1:
            return Step("road_building", p, [(A.ROAD_BUILDING, None)], [_played(p, "road_building")], after)
        if card in (2, 3):
            return None  # completed by the pick that follows
        return Step("noop", p, after=after, untranslatable="dev-card")

    def _plenty(self, e, after) -> Step:
        p = self.ctx.ours(int(e.f["pn"]))
        got = ours5(e.f["resources"])
        cards = [r for r in range(5) for _ in range(got[r])]
        if len(cards) != 2:
            return Step("noop", p, after=after, untranslatable="year-of-plenty")
        return Step("year_of_plenty", p, [(A.year_of_plenty(*cards), None)],
                    [_played(p, "year_of_plenty"), {"type": "year_of_plenty_taken", "player": p, "resources": got}],
                    after)

    def _monopoly(self, e, before, after) -> Step:
        p = self.ctx.ours(int(e.f["pn"]))
        r = JS_RES[int(e.f["v2"])]
        taken = [0 if q == p else self.hand(before, q)[r] - self.hand(after, q)[r] for q in range(4)]
        return Step("monopoly", p, [(A.monopoly(r), None)],
                    [_played(p, "monopoly"),
                     {"type": "monopoly_taken", "player": p, "resource": RESOURCE_NAMES[r], "from": taken}], after)

    def _discard(self, discs: list[Entry], after) -> Step:
        """Every discard of the stretch (the server force-discards for each stalled player at once)."""
        events = []
        for e in discs:
            q = self.ctx.ours(int(e.f["playerNum"]))
            cards = ours5(e.f["resources"])
            self._discards[q] = [a + b for a, b in zip(self._discards.get(q, [0] * 5), cards)]
            events += [{"type": "discarded", "player": q, "resource": RESOURCE_NAMES[r]}
                       for r in range(5) for _ in range(cards[r])]
        q = self.ctx.ours(int(discs[0].f["playerNum"]))
        if after["state"] == WAITING_FOR_DISCARDS:
            return Step("discard", q, [], events, None)
        actions = [(A.discard(r), None) for s in sorted(self._discards) for r in range(5)
                   for _ in range(self._discards[s][r])]
        self._discards = {}
        return Step("discard", q, actions, events, after)

    def rates(self, raw: dict, seat: int) -> list[int]:
        """Seat's maritime rates (our resource order) from its buildings in `raw` and our board's ports."""
        p = raw["players"][self.ctx.js(seat)]
        nodes = {JS_NODE_TO_OUR_NODE[n] for n in p["settlements"] + p["cities"]}
        rates, generic = [4] * 5, False
        for e, kind in self.board["ports"]:
            a, b = EDGE_NODES[e]
            if a in nodes or b in nodes:
                if kind == "generic":
                    generic = True
                else:
                    rates[RESOURCE_NAMES.index(kind)] = 2
        return [min(r, 3) if generic else r for r in rates]

    def _bank(self, e, before, after) -> Step:
        p = self.ctx.ours(int(e.f["pn"]))
        give, get = ours5(e.f["give"]), ours5(e.f["get"])
        rates = self.rates(before, p)
        whole = [{"type": "maritime_traded", "player": p, "gave": give, "got": get}]
        units = []
        for g in range(5):
            if give[g] % rates[g]:
                return Step("maritime", p, [], whole, after, untranslatable="bank-trade")
            units += [g] * (give[g] // rates[g])
        gets = [r for r in range(5) for _ in range(get[r])]
        if len(units) != len(gets) or any(a == b for a, b in zip(units, gets)):
            return Step("maritime", p, [], whole, after, untranslatable="bank-trade")
        actions, events = [], []
        for g, r in zip(units, gets):
            gave, got = [0] * 5, [0] * 5
            gave[g], got[r] = rates[g], 1
            actions.append((A.maritime(g, r), None))
            events.append({"type": "maritime_traded", "player": p, "gave": gave, "got": got})
        return Step("maritime", p, actions, events, after)

    def _trade(self, e, after) -> Step:
        a, o = self.ctx.ours(int(e.f["accepting"])), self.ctx.ours(int(e.f["offering"]))
        give, get = ours5(e.f["toAccepting"]), ours5(e.f["toOffering"])
        self._trades += 1
        events = [_offered(o, give, get), {"type": "trade_responded", "player": a, "accepted": True},
                  {"type": "trade_confirmed", "offerer": o, "partner": a, "offerer_gave": give, "partner_gave": get}]
        return Step("trade", o, [], events, after, untranslatable="domestic-trade",
                    info={"offerer": o, "partner": a, "give": give, "get": get})

    def _end_turn(self, before, after, requested: bool = True) -> Step:
        p = self.ctx.ours(before["current"])
        cancelled = before["state"] in (PLACING_FREE_ROAD1, PLACING_FREE_ROAD2)
        c = self.ctx
        c.turn, c.rolled, c.drawn, c.offers = c.turn + 1, False, [0] * 5, 0
        if not requested:  # the server ended a stalled turn (SOCForceEndTurnThread): no action of ours replays it
            return Step("end_turn", p, [], [{"type": "turn_ended", "player": p}], after, untranslatable="forced-end-turn")
        return Step("end_turn", p, [] if cancelled else [(A.END_TURN, None)], [{"type": "turn_ended", "player": p}],
                    after, untranslatable="road-building-cancelled" if cancelled else None)

    def _offer(self, e, after) -> Step:
        o = self.ctx.ours(int(e.f["from"]))
        to = sorted(self.ctx.ours(i) for i, t in enumerate(e.f["to"].split(",")) if t == "true")
        give, get = ours5(e.f["give"]), ours5(e.f["get"])
        self._offer_trades[o] = self._trades
        if after["current"] == self.ctx.js(o):
            self.ctx.offers += 1
        return Step("offer", o, [], [_offered(o, give, get)], after, info={"from": o, "to": to, "give": give, "get": get})
