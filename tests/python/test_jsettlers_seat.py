from collections import Counter

import pytest

from oracle.jsettlers import tables as T
from oracle.jsettlers.actions import (
    ACCEPT, BUY_DEV, CANCEL, END_TURN, KNIGHT, ROAD_BUILDING, ROLL, confirm, discard, maritime, monopoly,
    move_robber, offer, road, settlement, steal_from, year_of_plenty,
)
from oracle.jsettlers.arena import COUNTERS, NO_TRADES, Seat, config
from test_jsettlers_events import BOARD, Script
from test_jsettlers_events import rset as js_rset

TRADES = config(True)
WOOD, BRICK, SHEEP, WHEAT, ORE = range(5)
HOME = 20  # our settlement
NEAR = {HOME} | {b for a, b in T.EDGE_NODES if a == HOME} | {a for a, b in T.EDGE_NODES if b == HOME}
HOME_ROAD = min(e for e, (a, b) in enumerate(T.EDGE_NODES) if HOME in (a, b))


class StubBot:
    """Chooses with `choose(game) -> action id`; records the positions it was asked at and what it saw."""

    def __init__(self, choose):
        self.choose, self.seen, self.positions, self.chosen = choose, [], [], []

    def act(self, game):
        self.positions.append((game.phase, game.current_actor))
        a = self.choose(game)
        self.chosen.append(a)
        return a

    def observe(self, seat, events):
        self.seen.extend(events)

    def diagnostics(self):
        return {}


def seat_at(state=T.PLAY1, current=0, first=0, rolled=True, hands=None, cfg=TRADES, choose=None, clock=None):
    s = Script(first=first, state=state, current=current)
    for p, h in (hands or {}).items():
        s.hand(p, h)
    s.raw["players"][s.js(0)]["settlements"] = [T.OUR_NODE_TO_JS_NODE[HOME]]
    s.raw["players"][s.js(0)]["roads"] = [T.OUR_EDGE_TO_JS_EDGE[HOME_ROAD]]
    s.send("all", f"SOCGameState:game=x|state={state}", changed=True)
    s.tracker.flush()
    s.tracker.ctx.rolled = rolled
    bot = StubBot(choose or (lambda g: g.legal_actions()[0]))
    seat = Seat(bot, 5, cfg, s.tracker, 0, clock=clock or (lambda: 0.0))
    return s, seat, bot


def ask(kind, **extra):
    return {"t": "ask", "id": 7, "g": "x", "kind": kind, "state": 0, "extra": extra}


def first_road(g):
    return next(a for a in g.legal_actions() if road(0) <= a < road(72))


def test_main_choices_become_bridge_requests():
    _, seat, bot = seat_at(hands={0: [1, 1, 0, 0, 0]}, choose=first_road)
    r = seat.decide(ask("main"), 0.0)
    edge = bot.chosen[0] - road(0)
    assert r == {"t": "reply", "id": 7, "do": "build", "piece": T.ROAD, "coord": T.OUR_EDGE_TO_JS_EDGE[edge]}
    _, seat, _ = seat_at(choose=lambda g: END_TURN)
    assert seat.decide(ask("main"), 0.0)["do"] == "end"
    _, seat, _ = seat_at(choose=lambda g: BUY_DEV, hands={0: [0, 0, 1, 1, 1]})
    assert seat.decide(ask("main"), 0.0)["do"] == "buy"
    s, seat, _ = seat_at(hands={0: [4, 0, 0, 0, 0]}, choose=lambda g: maritime(WOOD, ORE))
    rate = s.tracker.rates(s.raw, 0)[WOOD]
    r = seat.decide(ask("main"), 0.0)
    assert (r["do"], r["give"], r["get"]) == ("bank", T.ours_to_js5([rate, 0, 0, 0, 0]), T.ours_to_js5([0, 0, 0, 0, 1]))


@pytest.mark.parametrize("card,action,expected", [
    (2, year_of_plenty(WHEAT, ORE), {"do": "yop", "res": [T.OUR_RES_TO_JS[WHEAT], T.OUR_RES_TO_JS[ORE]]}),
    (3, monopoly(SHEEP), {"do": "mono", "res": T.OUR_RES_TO_JS[SHEEP]}),
    (9, KNIGHT, {"do": "knight"}),
])
def test_card_plays_become_bridge_requests(card, action, expected):
    s, seat, _ = seat_at(choose=lambda g: action)
    s.raw["players"][s.js(0)]["dev"][card] = 1
    s.raw["deck"].remove(card)  # the card in hand left the deck
    s.send("all", "SOCGameState:game=x|state=20", changed=True)
    s.tracker.flush()
    r = seat.decide(ask("main"), 0.0)
    assert {k: r[k] for k in expected} == expected


def test_unimportable_declined_or_failing_decisions_fall_back_by_reason():
    s, seat, _ = seat_at(state=T.PLACING_ROAD)
    assert seat.decide(ask("main"), 0.0) == {"t": "reply", "id": 7, "fallback": "position"}
    s, seat, _ = seat_at(choose=lambda g: END_TURN)
    s.tracker.declined[0] += 1
    assert seat.decide(ask("main"), 0.0)["fallback"] == "declined"
    assert seat.decide(ask("main"), 0.0)["do"] == "end"  # the same decline is not a reason twice

    def boom(g):
        raise RuntimeError("engine panic")
    _, seat, _ = seat_at(choose=boom)
    assert seat.decide(ask("main"), 0.0)["fallback"] == "engine"
    _, seat, _ = seat_at(state=T.ROLL_OR_CARD, rolled=False, choose=lambda g: ROAD_BUILDING)
    assert seat.decide(ask("pre_roll"), 0.0)["fallback"] == "preroll-card"
    _, seat, _ = seat_at(state=T.ROLL_OR_CARD, rolled=False, choose=lambda g: ROLL)
    assert seat.decide(ask("pre_roll"), 0.0)["do"] == "roll"
    assert seat.fallbacks == Counter()
    seat.fellback("victim")
    assert seat.fallbacks == Counter({"victim": 1})
    seat.capped = True  # the driver aborted the game at our turn cap
    assert seat.decide(ask("main"), 0.0)["fallback"] == "turn-cap"


@pytest.mark.parametrize("state,kind,extra", [
    (T.PLACING_ROBBER, "move_robber", {}),
    (T.PLACING_FREE_ROAD1, "road_building", {}),
    (T.WAITING_FOR_DISCARDS, "discard", {"n": 4}),
])
def test_a_refusal_in_the_robber_free_road_or_discard_hook_falls_back_there(state, kind, extra):
    s, seat, _ = seat_at(state=state, choose=never)
    s.tracker.declined[0] += 1
    assert seat.decide(ask(kind, **extra), 0.0)["fallback"] == "declined"
    assert seat.fallbacks == Counter({"declined": 1})


def test_a_decision_at_the_seats_turn_cap_falls_back():
    s, _, _ = seat_at(choose=never)
    seat = Seat(StubBot(never), 5, TRADES, s.tracker, 0, max_turns=3, clock=lambda: 0.0)
    s.tracker.ctx.turn = 3
    assert seat.decide(ask("main"), 0.0)["fallback"] == "turn-cap"


def test_a_declined_setup_placement_falls_back_instead_of_being_asked_again():
    # JSettlers re-arms the placement after a refusal; answering the same node again would loop until the
    # robot client leaves the game.
    s = Script(state=T.START1A, current=0)
    s.send("all", "SOCGameState:game=x|state=5", changed=True)
    s.tracker.flush()
    bot = StubBot(lambda g: g.legal_actions()[0])
    seat = Seat(bot, 5, TRADES, s.tracker, 0, clock=lambda: 0.0)
    s.tracker.declined[0] += 1
    assert seat.decide(ask("setup_settlement"), 0.0)["fallback"] == "declined"
    r = seat.decide(ask("setup_settlement"), 0.0)
    assert r["node"] == T.OUR_NODE_TO_JS_NODE[bot.chosen[0] - settlement(0)]
    s.tracker.declined[0] += 1
    assert seat.decide(ask("setup_road"), 0.0)["fallback"] == "declined"
    assert seat.fallbacks == Counter({"declined": 2})


def test_the_robber_reply_carries_the_victim_chosen_at_the_steal_position():
    s, seat, bot = seat_at(state=T.PLACING_ROBBER, hands={2: [0, 0, 1, 0, 0]})
    tile = next(t for t in range(19) if BOARD["tile_resource"][t] is not None and not NEAR & set(T.TILE_NODES[t]))
    victim_node = T.TILE_NODES[tile][0]
    s.raw["players"][s.js(2)]["settlements"] = [T.OUR_NODE_TO_JS_NODE[victim_node]]
    s.send("all", "SOCGameState:game=x|state=33", changed=True)
    s.tracker.flush()
    bot.choose = lambda g: move_robber(tile) if g.phase == "move_robber" else steal_from(2)
    r = seat.decide(ask("move_robber"), 0.0)
    assert r["hex"] == T.OUR_TILE_TO_JS_HEX[tile] and r["victim"] == s.js(2)
    assert [p for p, _ in bot.positions] == ["move_robber", "steal"]


def never(g):
    raise AssertionError("the bot must not be asked")


def test_the_robber_move_asked_after_our_winning_knight_is_answered_but_not_a_fallback():
    # Our rules end the game when the knight's Largest Army reaches 10 points; JSettlers still asks us to
    # move the robber before it awards the win.
    s, seat, _ = seat_at(choose=never)
    p = s.raw["players"][s.js(0)]
    p["vp"], p["dev"][9], p["played"] = 8, 1, [2, 0, 0, 0]
    s.raw["deck"] = s.raw["deck"][3:]
    s.send("all", "SOCGameState:game=x|state=20", changed=True)
    s.tracker.flush()
    s.send(f"f{s.js(0)}", "SOCPlayDevCardRequest:game=x|devCard=9")
    p["dev"][9], p["played"], p["played_dev"], p["vp"] = 0, [3, 0, 0, 0], True, 10
    s.raw["state"], s.raw["largest"] = T.PLACING_ROBBER, s.js(0)
    s.send("all", f"SOCDevCardAction:game=x|playerNum={s.js(0)}|actionType=PLAY|cardType=9", changed=True)
    s.send("all", "SOCGameState:game=x|state=33")
    seat.observe(s.flush())
    assert s.tracker.over
    assert seat.decide(ask("move_robber"), 0.0) == {"t": "reply", "id": 7, "fallback": "after-win"}
    assert seat.fallbacks == Counter()


def discard_at(choose=None):
    """Our seat (4 wood, 4 brick) and seat 1 each owe 4 cards after a 7."""
    s, seat, bot = seat_at(state=T.WAITING_FOR_DISCARDS, hands={0: [4, 4, 0, 0, 0], 1: [0, 0, 8, 0, 0]}, choose=choose)
    s.raw["players"][s.js(0)]["discard"], s.raw["players"][s.js(1)]["discard"] = 4, 4
    s.send("all", "SOCGameState:game=x|state=50", changed=True)
    s.tracker.flush()
    return s, seat, bot


def stream_discard(s, seat, cards, hand, forced=False):
    """The stream reports our seat's discard of `cards`: after our request, or forced by the server."""
    js0 = s.js(0)
    if not forced:
        s.send(f"f{js0}", f"SOCDiscard:game=x|resources={js_rset(cards)}")
    s.hand(0, hand)
    s.raw["players"][js0]["discard"] = 0
    s.send("all", f"SOCDiscard:game=x|playerNum={js0}|resources={js_rset(cards)}", changed=True)
    seat.observe(s.flush())
    s.steps = []


def discarded(r, n=1):
    return [{"type": "discarded", "player": 0, "resource": T.RESOURCE_NAMES[r]}] * n


def test_discards_are_chosen_one_card_at_a_time_where_only_our_seat_owes():
    s, seat, bot = discard_at(lambda g: discard(WOOD) if discard(WOOD) in g.legal_actions() else discard(BRICK))
    r = seat.decide(ask("discard", n=4), 0.0)
    assert r["res"] == T.ours_to_js5([4, 0, 0, 0, 0])
    assert all(actor == 0 for _, actor in bot.positions) and len(bot.positions) == 4
    # Each discard is shown before the next is chosen (the bot's belief checks its hand at every decision),
    # and the stream's report of the same discards is not shown again.
    ours = {"type": "discarded", "player": 0, "resource": "wood"}
    assert bot.seen == [ours] * 4
    stream_discard(s, seat, [4, 0, 0, 0, 0], [0, 4, 0, 0, 0])
    assert bot.seen == [ours] * 4 and seat.fallbacks == Counter()


def test_a_discard_cut_short_is_completed_with_the_first_discards_our_rules_allow():
    # The cards already shown stand: the reply holds them and the rest, so the bot's view stays true.
    def two_then_panic(g):
        if len(bot.chosen) >= 2:
            raise RuntimeError("engine panic")
        return discard(BRICK)
    s, seat, bot = discard_at(two_then_panic)
    r = seat.decide(ask("discard", n=4), 0.0)
    assert r["res"] == T.ours_to_js5([2, 2, 0, 0, 0]) and seat.fallbacks == Counter({"engine": 1})
    assert bot.seen == discarded(BRICK, 2) + discarded(WOOD, 2)
    stream_discard(s, seat, [2, 2, 0, 0, 0], [2, 2, 0, 0, 0])
    assert bot.seen == discarded(BRICK, 2) + discarded(WOOD, 2)


def test_a_discard_that_fails_at_its_first_card_is_the_stock_brains_and_shown_from_the_stream():
    def panic(g):
        raise RuntimeError("engine panic")
    s, seat, bot = discard_at(panic)
    assert seat.decide(ask("discard", n=4), 0.0)["fallback"] == "engine" and bot.seen == []
    stream_discard(s, seat, [0, 4, 0, 0, 0], [4, 0, 0, 0, 0])
    assert bot.seen == discarded(BRICK, 4) and seat.fallbacks == Counter({"engine": 1})


def test_discards_the_server_forces_on_our_seat_are_counted_and_shown():
    s, seat, bot = discard_at()
    stream_discard(s, seat, [1, 3, 0, 0, 0], [3, 1, 0, 0, 0], forced=True)  # we were never asked
    assert seat.fallbacks == Counter({"forced-discard": 1})
    assert bot.seen == discarded(WOOD) + discarded(BRICK, 3)


def test_a_discard_forced_before_our_reply_is_counted_and_leaves_nothing_to_swallow_later():
    s, seat, bot = discard_at(lambda g: discard(WOOD))
    seat.decide(ask("discard", n=4), 0.0)  # shown 4 wood while choosing
    stream_discard(s, seat, [0, 4, 0, 0, 0], [4, 0, 0, 0, 0], forced=True)  # the server discarded first
    assert seat.fallbacks == Counter({"forced-discard": 1})
    assert bot.seen == discarded(WOOD, 4) + discarded(BRICK, 4)  # the bot's belief then resets, counted
    s.raw["players"][s.js(0)]["discard"] = 2
    stream_discard(s, seat, [2, 0, 0, 0, 0], [2, 0, 0, 0, 0], forced=True)  # a later forced discard of wood
    assert bot.seen[-2:] == discarded(WOOD, 2) and seat.fallbacks["forced-discard"] == 2


def robot_offer(frm, give, get, to=(True, False, False, False)):
    return {"from": frm, "to": list(to), "give": T.ours_to_js5(give), "get": T.ours_to_js5(get)}


def test_answers_reject_counter_offers_and_unrepresentable_offers_without_asking():
    _, seat, _ = seat_at(current=0, choose=never)  # we are on turn: an offer to us is a counter-offer
    assert seat.decide(ask("answer", **robot_offer(2, [0, 0, 1, 0, 0], [1, 0, 0, 0, 0])), 0.0)["accept"] is False
    assert seat.counts["counters_rejected"] == 1
    s, seat, _ = seat_at(current=2, choose=never, hands={0: [1, 0, 0, 0, 0], 2: [0, 0, 2, 0, 0]})
    assert seat.decide(ask("answer", **robot_offer(2, [0, 0, 2, 0, 0], [1, 0, 0, 0, 0])), 0.0)["accept"] is False
    assert seat.counts["unrepresentable"] == 1
    _, seat, _ = seat_at(current=2, choose=never, cfg=config(False), hands={0: [1, 0, 0, 0, 0], 2: [0, 0, 1, 0, 0]})
    assert seat.decide(ask("answer", **robot_offer(2, [0, 0, 1, 0, 0], [1, 0, 0, 0, 0])), 0.0)["accept"] is False
    assert seat.counts["unrepresentable"] == 1 and seat.counts["offers_to_us"] == 1


def test_answers_import_the_offerers_main_position_with_earlier_seats_rejecting():
    def answer(g):
        assert g.phase == "trade_response" and g.current_actor == 0
        return ACCEPT
    s, seat, bot = seat_at(current=2, choose=answer, hands={0: [1, 0, 0, 0, 0], 2: [0, 0, 1, 0, 0]})
    s.tracker.ctx.offers = 3  # three offers this turn, the last of them this one: earlier ones clamp to 1
    r = seat.decide(ask("answer", **robot_offer(s.js(2), [0, 0, 1, 0, 0], [1, 0, 0, 0, 0])), 0.0)
    assert r["accept"] is True and seat.counts["offers_to_us"] == 1
    assert bot.positions == [("trade_response", 0)]


def test_late_answers_are_sent_as_rejections_and_counted():
    s, seat, _ = seat_at(current=2, choose=lambda g: ACCEPT, hands={0: [1, 0, 0, 0, 0], 2: [0, 0, 1, 0, 0]},
                         clock=lambda: 4.6)
    s.tracker.ctx.offers = 1
    r = seat.decide(ask("answer", **robot_offer(s.js(2), [0, 0, 1, 0, 0], [1, 0, 0, 0, 0])), 0.0)
    assert r["accept"] is False and seat.counts["late"] == 1


def test_our_offer_goes_to_those_we_would_confirm_and_an_unsent_one_is_declined_in_our_engine():
    give, get = [1, 0, 0, 0, 0], [0, 0, 0, 0, 1]
    a = offer(give, get)
    hands = {0: [1, 0, 0, 0, 0], 1: [0, 0, 0, 0, 1], 2: [0, 0, 0, 0, 1], 3: [0, 0, 0, 0, 0]}

    def choose(g):
        if g.phase == "trade_confirm":
            return confirm(2) if confirm(2) in g.legal_actions() else CANCEL
        return a if a in g.legal_actions() else END_TURN
    s, seat, bot = seat_at(hands=hands, choose=choose)
    r = seat.decide(ask("main"), 0.0)
    assert r["do"] == "offer" and r["give"] == T.ours_to_js5(give) and r["get"] == T.ours_to_js5(get)
    assert r["to"] == [i == s.js(2) for i in range(4)]
    assert seat.counts["our_offers"] == 1

    def refuse(g):
        if g.phase == "trade_confirm":
            return CANCEL
        return a if a in g.legal_actions() else END_TURN
    s, seat, bot = seat_at(hands=hands, choose=refuse)
    assert seat.decide(ask("main"), 0.0)["do"] == "end"  # two offers made in our engine only, then the end
    assert seat.counts["our_offers"] == 2
    kinds = [e["type"] for e in bot.seen]
    assert kinds.count("trade_offered") == 2 and "trade_cancelled" in kinds and "trade_confirmed" not in kinds


def test_robot_offers_and_their_caps_are_counted_from_the_stream():
    s, seat, _ = seat_at(current=1)
    rset = "clay=0|ore=0|sheep=0|wheat=0|wood={}|unknown=0"
    js1 = s.js(1)
    s.send(f"f{js1}", "SOCMakeOffer:game=x")
    s.send("all", f"SOCMakeOffer:game=x|from={js1}|to=true,true,true,true|give={rset.format(2)}|get={rset.format(0).replace('ore=0', 'ore=1')}")
    s.send(f"f{js1}", "SOCMakeOffer:game=x")
    s.send("all", f"SOCMakeOffer:game=x|from={js1}|to=true,false,false,true|give={rset.format(1)}|get={rset.format(0).replace('ore=0', 'ore=1')}")
    seat.observe(s.flush())  # both offer steps: the one the second request closed and the one open now
    assert seat.counts["robot_offers"] == 2 and seat.counts["robot_offers_over_caps"] == 2
    assert set(COUNTERS) >= {"robot_offers", "robot_offers_over_caps", "our_offers", "late"}
    assert NO_TRADES == {"max_offers_per_turn": 0}


def test_each_seats_last_build_or_purchase_turn_is_read_from_the_stream():
    # Task 9 reports results with and without games where the robots stopped building (stall-debug.md).
    s, seat, _ = seat_at(current=1)
    assert seat.last_build == [None] * 4
    js1 = s.js(1)
    s.tracker.ctx.turn = 5
    s.send(f"f{js1}", "SOCBuyDevCardRequest:game=x")
    s.raw["players"][js1]["dev"][9] += 1
    s.raw["deck"].remove(9)
    s.send("all", f"SOCDevCardAction:game=x|playerNum={js1}|actionType=DRAW|cardType=9", changed=True)
    seat.observe(s.flush())
    s.steps = []
    assert seat.last_build == [None, 5, None, None]
    s.tracker.ctx.turn = 9
    edge = next(e for e, (a, b) in enumerate(T.EDGE_NODES) if HOME in (a, b) and e != HOME_ROAD)
    js_edge = T.OUR_EDGE_TO_JS_EDGE[edge]
    s.raw["current"] = s.js(0)
    s.send("all", "SOCGameState:game=x|state=20", changed=True)
    s.send(f"f{s.js(0)}", f"SOCPutPiece:game=x|playerNumber={s.js(0)}|pieceType=0|coord={js_edge:x}")
    s.raw["players"][s.js(0)]["roads"].append(js_edge)
    s.send("all", f"SOCPutPiece:game=x|playerNumber={s.js(0)}|pieceType=0|coord={js_edge:x}", changed=True)
    seat.observe(s.flush())
    assert seat.last_build == [9, 5, None, None]
