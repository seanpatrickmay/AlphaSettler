import copy

import pytest

from alphasettler import Bot
from jsettlers_harness import require
from oracle.jsettlers import tables as T
from oracle.jsettlers.actions import (
    BUY_DEV, END_TURN, KNIGHT, ROLL, city, discard, maritime, monopoly, move_robber, redact, road, settlement, steal_from,
    year_of_plenty,
)
from oracle.jsettlers.boards import game_command, js_layout, native_board
from oracle.jsettlers.events import Tracker, dice_pair
from oracle.jsettlers.jvm import Jvm
from oracle.jsettlers.messages import parse
from oracle.jsettlers.translate import import_game

UNLIMITED = {"max_offers_per_turn": 0, "bank_limit": False}
BOARD = native_board(3)
WOOD, BRICK, SHEEP, WHEAT, ORE = range(5)


def rset(ours=(0, 0, 0, 0, 0), unknown=0):
    js = T.ours_to_js5(list(ours))
    return "clay={}|ore={}|sheep={}|wheat={}|wood={}|unknown={}".format(*js, unknown)


class Script:
    """A Tracker fed by hand: `raw` is the server-side position; `send` records one message."""

    def __init__(self, first=0, state=T.PLAY1, current=0):
        players = [{"res": [0] * 5, "dev": [0] * 10, "played": [0] * 4, "settlements": [], "cities": [], "roads": [],
                    "discard": 0, "played_dev": False, "vp": 0} for _ in range(4)]
        desert = T.OUR_TILE_TO_JS_HEX[BOARD["tile_resource"].index(None)]
        self.raw = {"state": state, "current": current, "robber": desert, "deck": [9] * 14 + [1, 1, 2, 2, 3, 3, 4, 5, 6, 7, 8],
                    "players": players, "longest": -1, "largest": -1, "winner": -1}
        board_raw = {"hexes": [], "ports": [], "robber": desert}
        lay = js_layout(BOARD)
        for t, h in enumerate(T.OUR_TILE_TO_JS_HEX):
            board_raw["hexes"].append([h, lay["hexes"][T.HEX_NUM[h]], max(0, lay["numbers"][T.HEX_NUM[h]])])
        for i, e in enumerate(T.PORT_EDGES):
            kind = BOARD["ports"][i][1]
            board_raw["ports"].append([e, 0 if kind == "generic" else T.OUR_RES_TO_JS[T.RESOURCE_NAMES.index(kind)]])
        self.tracker = Tracker({"t": "start", "g": "x", "first": first, "board": board_raw, "s": copy.deepcopy(self.raw)})
        self.steps = []

    def js(self, ours):
        return self.tracker.ctx.js(ours)

    def hand(self, ours, h):
        self.raw["players"][self.js(ours)]["res"] = T.ours_to_js5(list(h))

    def send(self, aud, text, changed=False):
        msg = {"t": "ev", "g": "x", "a": aud, "m": text}
        if changed:
            msg["s"] = copy.deepcopy(self.raw)
        self.steps += self.tracker.feed(msg)

    def flush(self):
        """Every step so far (the stretches closed by later requests and the one open now)."""
        self.steps += self.tracker.flush()
        return self.steps


def test_parse_reads_fields_and_resource_sets():
    e = parse("all", "SOCBankTrade:game=g|give=clay=0|ore=4|sheep=0|wheat=0|wood=0|unknown=0|get=clay=1|ore=0|sheep=0|wheat=0|wood=0|unknown=0|pn=2")
    assert e.name == "SOCBankTrade" and e.f["pn"] == "2" and e.f["give"]["ore"] == 4 and e.f["get"]["clay"] == 1
    assert parse("f3", "SOCPutPiece:game=g|playerNumber=3|pieceType=0|coord=67").num("coord") == 0x67
    assert parse("f3", "SOCEndTurn:game=g").sender == 3


@pytest.mark.parametrize("total,pair", [(2, [1, 1]), (7, [6, 1]), (12, [6, 6]), (5, [4, 1])])
def test_dice_pairs_have_the_total(total, pair):
    assert dice_pair(total) == pair and sum(pair) == total


def test_a_roll_produces_from_the_hand_differences():
    s = Script(first=1, current=1)
    s.send("f1", "SOCRollDice:game=x")
    s.send("all", "SOCDiceResult:game=x|param=9")
    s.hand(0, [0, 2, 0, 0, 0])
    s.hand(2, [1, 0, 0, 0, 1])
    s.raw["state"] = T.PLAY1
    s.send("all", "SOCGameState:game=x|state=20", changed=True)
    [step] = s.flush()
    assert step.kind == "roll" and step.player == 0 and step.actions == [(ROLL, {"roll": [6, 3]})]
    assert step.events == [{"type": "rolled", "player": 0, "dice": [6, 3]},
                           {"type": "produced", "player": 0, "resources": [0, 2, 0, 0, 0]},
                           {"type": "produced", "player": 2, "resources": [1, 0, 0, 0, 1]}]
    assert s.tracker.ctx.rolled


def test_a_build_request_and_its_placement_are_one_step():
    s = Script()
    s.hand(0, [1, 1, 0, 0, 0])
    s.raw["players"][0]["roads"] = [T.OUR_EDGE_TO_JS_EDGE[3]]
    s.send("all", "SOCGameState:game=x|state=20", changed=True)
    s.send("f0", "SOCBuildRequest:game=x|pieceType=0")
    s.hand(0, [0, 0, 0, 0, 0])
    s.raw["state"] = T.PLACING_ROAD
    s.send("all", "SOCPlayerElements:game=x|playerNum=0|actionType=LOSE|e1=1,e5=1", changed=True)
    s.send("all", "SOCGameState:game=x|state=30")
    js_edge = T.OUR_EDGE_TO_JS_EDGE[4]
    s.send("f0", f"SOCPutPiece:game=x|playerNumber=0|pieceType=0|coord={js_edge:x}")
    s.raw["players"][0]["roads"].append(js_edge)
    s.raw["state"] = T.PLAY1
    s.send("all", f"SOCPutPiece:game=x|playerNumber=0|pieceType=0|coord={js_edge:x}", changed=True)
    s.send("all", "SOCGameState:game=x|state=20")
    [step] = [x for x in s.flush() if x.actions]
    assert step.kind == "build" and step.actions == [(road(4), None)]
    assert step.events == [{"type": "built_road", "player": 0, "edge": 4}]


def test_a_trade_answer_inside_a_build_carries_no_position():
    s = Script()
    s.hand(0, [0, 0, 0, 2, 3])
    s.raw["players"][0]["settlements"] = [T.OUR_NODE_TO_JS_NODE[20]]
    s.send("all", "SOCGameState:game=x|state=20", changed=True)
    s.send("f0", "SOCBuildRequest:game=x|pieceType=2")
    s.hand(0, [0, 0, 0, 0, 0])
    s.raw["state"] = T.PLACING_CITY
    s.send("all", "SOCPlayerElements:game=x|playerNum=0|actionType=LOSE|e2=3,e4=2", changed=True)
    s.send("all", "SOCGameState:game=x|state=32")
    s.send("f2", "SOCRejectOffer:game=x|playerNumber=0")  # a late answer to an earlier offer
    s.send("all", "SOCRejectOffer:game=x|playerNumber=2")
    node = T.OUR_NODE_TO_JS_NODE[20]
    s.send("f0", f"SOCPutPiece:game=x|playerNumber=0|pieceType=2|coord={node:x}")
    s.raw["players"][0]["settlements"], s.raw["players"][0]["cities"] = [], [node]
    s.raw["state"] = T.PLAY1
    s.send("all", f"SOCPutPiece:game=x|playerNumber=0|pieceType=2|coord={node:x}", changed=True)
    s.send("all", "SOCGameState:game=x|state=20")
    reject, build = [x for x in s.flush() if x.kind in ("reject", "build")]
    assert reject.kind == "reject" and reject.after is None
    assert reject.events == [{"type": "trade_responded", "player": 2, "accepted": False}]
    assert build.kind == "build" and build.actions == [(city(20), None)] and build.after["state"] == T.PLAY1


def test_a_second_setup_settlement_produces():
    s = Script(state=T.START2A, current=0)
    node = 20
    s.send("f0", f"SOCPutPiece:game=x|playerNumber=0|pieceType=1|coord={T.OUR_NODE_TO_JS_NODE[node]:x}")
    s.raw["players"][0]["settlements"].append(T.OUR_NODE_TO_JS_NODE[node])
    s.hand(0, [1, 0, 1, 0, 0])
    s.raw["state"] = T.START2B
    s.send("all", f"SOCPutPiece:game=x|playerNumber=0|pieceType=1|coord={T.OUR_NODE_TO_JS_NODE[node]:x}", changed=True)
    s.send("all", "SOCGameState:game=x|state=11")
    [step] = s.flush()
    assert step.kind == "setup" and step.actions == [(settlement(node), None)]
    assert step.events[1] == {"type": "produced", "player": 0, "resources": [1, 0, 1, 0, 0]}


def test_a_robber_move_with_one_victim_is_the_move_and_the_true_steal():
    s = Script()
    s.hand(2, [0, 0, 0, 0, 1])
    s.raw["state"] = T.PLACING_ROBBER
    s.send("all", "SOCGameState:game=x|state=33", changed=True)
    hexes = T.OUR_TILE_TO_JS_HEX
    s.send("f0", f"SOCMoveRobber:game=x|playerNumber=0|coord={hexes[5]:x}")
    s.raw["robber"] = hexes[5]
    s.send("all", f"SOCMoveRobber:game=x|playerNumber=0|coord={hexes[5]:x}", changed=True)
    s.hand(2, [0, 0, 0, 0, 0])
    s.hand(0, [0, 0, 0, 0, 1])
    s.raw["state"] = T.PLAY1
    s.send("p0", "SOCRobberyResult:game=x|perp=0|victim=2|resType=2|amount=1|isGainLose=true", changed=True)
    s.send("p2", "SOCRobberyResult:game=x|perp=0|victim=2|resType=2|amount=1|isGainLose=true")
    s.send("!p[0, 2]", "SOCRobberyResult:game=x|perp=0|victim=2|resType=6|amount=1|isGainLose=true")
    s.send("all", "SOCGameState:game=x|state=20")
    [step] = [x for x in s.flush() if x.actions]
    assert step.actions == [(move_robber(5), None), (steal_from(2), {"steal": "ore"})]
    assert step.events[1] == {"type": "stole", "thief": 0, "victim": 2, "resource": "ore"}


def test_a_dev_card_draw_uses_the_true_card():
    s = Script(first=3, current=3)
    s.send("f3", "SOCBuyDevCardRequest:game=x")
    s.raw["players"][3]["dev"][5] = 1
    s.send("p3", "SOCDevCardAction:game=x|playerNum=3|actionType=DRAW|cardType=5", changed=True)
    s.send("!p3", "SOCDevCardAction:game=x|playerNum=3|actionType=DRAW|cardType=0")
    [step] = s.flush()
    assert step.actions == [(BUY_DEV, {"dev": "victory_point"})] and step.player == 0
    assert s.tracker.ctx.drawn == [0, 1, 0, 0, 0]


def test_year_of_plenty_and_monopoly_join_the_card_and_the_pick():
    s = Script()
    s.send("f0", "SOCPlayDevCardRequest:game=x|devCard=2")
    s.raw["state"] = T.WAITING_FOR_DISCOVERY
    s.send("all", "SOCDevCardAction:game=x|playerNum=0|actionType=PLAY|cardType=2", changed=True)
    s.send("f0", f"SOCPickResources:game=x|resources={rset([0, 0, 0, 1, 1])}")
    s.hand(0, [0, 0, 0, 1, 1])
    s.raw["state"] = T.PLAY1
    s.send("all", f"SOCPickResources:game=x|resources={rset([0, 0, 0, 1, 1])}|pn=0|reason=2", changed=True)
    s.hand(1, [0, 0, 3, 0, 0])
    s.send("f0", "SOCEndTurn:game=x")  # closes the stretch; the end-turn step follows
    steps = s.flush()
    yop = next(x for x in steps if x.kind == "year_of_plenty")
    assert yop.actions == [(year_of_plenty(WHEAT, ORE), None)]
    assert [e["type"] for e in yop.events] == ["played_dev", "year_of_plenty_taken"]
    m = Script()
    m.hand(1, [0, 0, 3, 0, 0])
    m.send("all", "SOCGameState:game=x|state=20", changed=True)
    m.send("f0", "SOCPlayDevCardRequest:game=x|devCard=3")
    m.raw["state"] = T.WAITING_FOR_MONOPOLY
    m.send("all", "SOCDevCardAction:game=x|playerNum=0|actionType=PLAY|cardType=3", changed=True)
    m.send("f0", "SOCPickResourceType:game=x|resType=3")
    m.hand(1, [0, 0, 0, 0, 0])
    m.hand(0, [0, 0, 3, 0, 0])
    m.raw["state"] = T.PLAY1
    m.send("all", "SOCSimpleAction:game=x|pn=0|actType=3|v1=3|v2=3", changed=True)
    [mono] = [x for x in m.flush() if x.actions]
    assert mono.actions == [(monopoly(SHEEP), None)]
    assert mono.events[1] == {"type": "monopoly_taken", "player": 0, "resource": "sheep", "from": [0, 3, 0, 0]}


def test_concurrent_discards_replay_in_seat_order_once_the_last_is_in():
    s = Script(first=0, current=0)
    s.hand(1, [4, 4, 0, 0, 0])
    s.hand(3, [0, 0, 8, 0, 0])
    s.raw["players"][1]["discard"], s.raw["players"][3]["discard"] = 4, 4
    s.raw["state"] = T.WAITING_FOR_DISCARDS
    s.send("all", "SOCGameState:game=x|state=50", changed=True)
    s.send("f3", f"SOCDiscard:game=x|resources={rset([0, 0, 4, 0, 0])}")
    s.hand(3, [0, 0, 4, 0, 0])
    s.raw["players"][3]["discard"] = 0
    s.send("p3", f"SOCDiscard:game=x|playerNum=3|resources={rset([0, 0, 4, 0, 0])}", changed=True)
    s.send("!p3", f"SOCDiscard:game=x|playerNum=3|resources={rset(unknown=4)}")
    s.send("f1", f"SOCDiscard:game=x|resources={rset([2, 2, 0, 0, 0])}")
    s.hand(1, [2, 2, 0, 0, 0])
    s.raw["players"][1]["discard"] = 0
    s.raw["state"] = T.PLACING_ROBBER
    s.send("p1", f"SOCDiscard:game=x|playerNum=1|resources={rset([2, 2, 0, 0, 0])}", changed=True)
    first, last = [x for x in s.flush() if x.kind == "discard"]
    assert first.actions == [] and first.after is None and len(first.events) == 4
    assert last.actions == [(discard(WOOD), None)] * 2 + [(discard(BRICK), None)] * 2 + [(discard(SHEEP), None)] * 4
    assert last.after is not None


@pytest.mark.parametrize("give,get,expected", [
    ([4, 0, 0, 0, 0], [0, 0, 0, 0, 1], [maritime(WOOD, ORE)]),
    ([8, 0, 0, 0, 0], [0, 1, 0, 0, 1], [maritime(WOOD, BRICK), maritime(WOOD, ORE)]),
    ([1, 0, 0, 0, 0], [0, 0, 0, 0, 4], None),  # an undone trade
])
def test_bank_trades_become_our_maritime_trades_or_an_allowlisted_difference(give, get, expected):
    s = Script()
    s.hand(0, give)
    s.send("all", "SOCGameState:game=x|state=20", changed=True)
    s.send("f0", f"SOCBankTrade:game=x|give={rset(give)}|get={rset(get)}")
    s.hand(0, get)
    s.send("all", f"SOCBankTrade:game=x|give={rset(give)}|get={rset(get)}|pn=0", changed=True)
    [step] = [x for x in s.flush() if x.kind == "maritime"]
    if expected is None:
        assert step.untranslatable == "bank-trade" and step.actions == []
    else:
        assert [a for a, _ in step.actions] == expected and step.untranslatable is None


def test_an_accepted_offer_is_restated_responded_and_confirmed():
    s = Script(first=0, current=0)
    s.hand(0, [1, 0, 0, 0, 0])
    s.hand(2, [0, 0, 0, 1, 0])
    s.send("all", "SOCGameState:game=x|state=20", changed=True)
    s.send("f0", f"SOCMakeOffer:game=x|from=0|to=false,true,true,true|give={rset([1, 0, 0, 0, 0])}|get={rset([0, 0, 0, 1, 0])}")
    s.send("all", f"SOCMakeOffer:game=x|from=0|to=false,true,true,true|give={rset([1, 0, 0, 0, 0])}|get={rset([0, 0, 0, 1, 0])}")
    s.send("f2", "SOCAcceptOffer:game=x|accepting=0|offering=0")
    s.hand(0, [0, 0, 0, 1, 0])
    s.hand(2, [1, 0, 0, 0, 0])
    s.send("all", f"SOCAcceptOffer:game=x|accepting=2|offering=0|toAccepting={rset([1, 0, 0, 0, 0])}|toOffering={rset([0, 0, 0, 1, 0])}", changed=True)
    s.send("all", "SOCClearOffer:game=x|playerNumber=-1")
    offer, trade = [x for x in s.flush() if x.kind in ("offer", "trade")]
    assert offer.info == {"from": 0, "to": [1, 2, 3], "give": [1, 0, 0, 0, 0], "get": [0, 0, 0, 1, 0]}
    assert trade.untranslatable == "domestic-trade"
    assert [e["type"] for e in trade.events] == ["trade_offered", "trade_responded", "trade_confirmed"]
    assert trade.events[2] == {"type": "trade_confirmed", "offerer": 0, "partner": 2,
                               "offerer_gave": [1, 0, 0, 0, 0], "partner_gave": [0, 0, 0, 1, 0]}
    assert s.tracker.ctx.offers == 1


def test_ending_a_turn_counts_it_and_ending_inside_road_building_is_allowlisted():
    s = Script(first=0, current=0)
    s.tracker.ctx.rolled = True
    s.send("f0", "SOCEndTurn:game=x")
    s.raw["current"], s.raw["state"] = 1, T.ROLL_OR_CARD
    s.send("all", "SOCClearOffer:game=x|playerNumber=-1")
    s.send("all", "SOCTurn:game=x|playerNumber=1|gameState=15", changed=True)
    [step] = s.flush()
    assert step.actions == [(END_TURN, None)] and step.events == [{"type": "turn_ended", "player": 0}]
    assert (s.tracker.ctx.turn, s.tracker.ctx.rolled) == (1, False)
    r = Script(state=T.PLACING_FREE_ROAD1)
    r.send("f0", "SOCEndTurn:game=x")
    r.raw["current"], r.raw["state"] = 1, T.ROLL_OR_CARD
    r.send("all", "SOCTurn:game=x|playerNumber=1|gameState=15", changed=True)
    [step] = r.flush()
    assert step.untranslatable == "road-building-cancelled"


def test_a_turn_the_server_ends_itself_is_counted_and_allowlisted():
    s = Script(first=0, current=0)
    c = s.tracker.ctx
    c.rolled, c.drawn, c.offers = True, [0, 1, 0, 0, 0], 1
    s.send("f0", "SOCBuildRequest:game=x|pieceType=0")  # a request the player never completes
    s.raw["current"], s.raw["state"] = 1, T.ROLL_OR_CARD
    s.send("all", "SOCTurn:game=x|playerNumber=1|gameState=15", changed=True)  # SOCForceEndTurnThread
    s.send("all", "SOCRollDicePrompt:game=x|playerNumber=1")
    [step] = s.flush()
    assert (step.kind, step.player, step.actions) == ("end_turn", 0, [])
    assert step.untranslatable == "forced-end-turn" and step.events == [{"type": "turn_ended", "player": 0}]
    assert (c.turn, c.rolled, c.drawn, c.offers) == (1, False, [0] * 5, 0)


def test_a_roll_that_takes_cards_produces_nothing_negative():
    s = Script(first=0, current=0)
    s.hand(1, [2, 0, 0, 0, 0])
    s.send("all", "SOCGameState:game=x|state=20", changed=True)
    s.send("f0", "SOCRollDice:game=x")
    s.send("all", "SOCDiceResult:game=x|param=8")
    s.hand(0, [0, 1, 0, 0, 0])
    s.hand(1, [1, 0, 0, 0, 0])  # a hand that shrank on a roll: no production of ours explains it
    s.send("all", "SOCGameState:game=x|state=20", changed=True)
    [step] = [x for x in s.flush() if x.kind == "roll"]
    assert step.untranslatable == "production"
    assert step.events[1:] == [{"type": "produced", "player": 0, "resources": [0, 1, 0, 0, 0]}]


def test_declines_and_bank_limit_payouts_are_counted():
    s = Script()
    s.send("f0", "SOCBuildRequest:game=x|pieceType=2")
    s.send("un", "SOCDeclinePlayerRequest:game=x|reason=1|detail1=0|detail2=0|text=no")
    assert s.flush() == [] and s.tracker.declined == [1, 0, 0, 0]
    s.hand(1, [0, 0, 0, 0, 15])
    s.hand(2, [0, 0, 0, 0, 4])
    s.send("all", "SOCGameState:game=x|state=20", changed=True)
    s.send("f0", "SOCRollDice:game=x")
    s.send("all", "SOCDiceResult:game=x|param=6")
    s.hand(2, [0, 0, 0, 0, 5])
    s.send("all", "SOCGameState:game=x|state=20", changed=True)
    s.flush()
    assert s.tracker.bank_limit_hits == 1


def test_a_winning_placement_ends_with_game_over():
    s = Script()
    edge = T.OUR_EDGE_TO_JS_EDGE[7]
    s.send("f0", f"SOCPutPiece:game=x|playerNumber=0|pieceType=0|coord={edge:x}")
    s.raw["players"][0]["roads"].append(edge)
    s.raw["state"], s.raw["winner"] = T.OVER, 0
    s.send("all", f"SOCPutPiece:game=x|playerNumber=0|pieceType=0|coord={edge:x}", changed=True)
    s.send("all", "SOCGameState:game=x|state=1000")
    [step] = s.flush()
    assert step.events[-1] == {"type": "game_over", "winner": 0}


def test_a_winning_action_cut_short_by_the_end_of_the_stream_is_read_from_its_request():
    # The harness stops at the first message recorded once the game is over, and JSettlers announces a winning
    # placement (or card) in text before it reports the action itself.
    s = Script()
    node = T.OUR_NODE_TO_JS_NODE[20]
    s.raw["players"][0]["settlements"] = [node]
    s.send("all", "SOCGameState:game=x|state=20", changed=True)
    s.send("f0", f"SOCPutPiece:game=x|playerNumber=0|pieceType=2|coord={node:x}")
    s.raw["players"][0]["settlements"], s.raw["players"][0]["cities"] = [], [node]
    s.raw["state"], s.raw["winner"] = T.OVER, 0
    s.send("all", "SOCGameServerText:game=x|text=robot 1 built a city.", changed=True)
    [step] = [x for x in s.flush() if x.kind != "noop"]
    assert step.kind == "build" and step.actions == [(city(20), None)]
    assert step.events == [{"type": "built_city", "player": 0, "node": 20}, {"type": "game_over", "winner": 0}]
    b = Script(first=1, current=1)
    b.send("f1", "SOCBuyDevCardRequest:game=x")
    b.raw["players"][1]["dev"][5] = 1
    b.raw["deck"] = b.raw["deck"][:15] + b.raw["deck"][16:]  # a victory point card left the deck
    b.raw["state"], b.raw["winner"] = T.OVER, 1
    b.send("all", "SOCPlayerElements:game=x|playerNum=1|actionType=LOSE|e2=1,e3=1,e4=1", changed=True)
    [step] = b.flush()
    assert step.kind == "buy" and step.player == 0 and step.actions == [(BUY_DEV, {"dev": "victory_point"})]
    assert step.events[-1] == {"type": "game_over", "winner": 0}
    e = Script(first=0, current=0)
    e.tracker.ctx.rolled = True
    e.send("f0", "SOCEndTurn:game=x")
    e.raw["current"], e.raw["state"], e.raw["winner"] = 1, T.OVER, 1  # the next player starts its turn with 10
    e.send("all", "SOCClearOffer:game=x|playerNumber=-1", changed=True)
    [step] = e.flush()
    assert step.kind == "end_turn" and step.actions == [(END_TURN, None)]
    assert step.events == [{"type": "turn_ended", "player": 0}, {"type": "game_over", "winner": 1}]


def test_a_knight_that_reaches_the_target_ends_our_game_before_the_robber_moves():
    s = Script()
    s.raw["players"][0]["vp"], s.raw["players"][0]["dev"][9] = 8, 1
    s.raw["players"][0]["settlements"] = [T.OUR_NODE_TO_JS_NODE[n] for n in (0, 10, 20, 30)]
    s.raw["players"][0]["cities"] = [T.OUR_NODE_TO_JS_NODE[n] for n in (40, 50)]
    s.raw["players"][0]["played"] = [2, 0, 0, 0]
    s.raw["deck"] = s.raw["deck"][3:]  # the three knights it held or played
    s.send("all", "SOCGameState:game=x|state=20", changed=True)
    s.send("f0", "SOCPlayDevCardRequest:game=x|devCard=9")
    p = s.raw["players"][0]
    p["dev"][9], p["played"], p["played_dev"], p["vp"] = 0, [3, 0, 0, 0], True, 10
    s.raw["state"], s.raw["largest"] = T.PLACING_ROBBER, 0  # JSettlers awards the win only after the robber moves
    s.send("all", "SOCDevCardAction:game=x|playerNum=0|actionType=PLAY|cardType=9", changed=True)
    s.send("all", "SOCGameState:game=x|state=33")
    hexes = T.OUR_TILE_TO_JS_HEX
    s.send("f0", f"SOCMoveRobber:game=x|playerNumber=0|coord={hexes[5]:x}")
    s.raw["robber"], s.raw["state"], s.raw["winner"] = hexes[5], T.OVER, 0
    s.send("all", f"SOCMoveRobber:game=x|playerNumber=0|coord={hexes[5]:x}", changed=True)
    s.send("all", "SOCGameState:game=x|state=1000")
    steps = [x for x in s.flush() if x.kind != "noop"]
    assert [x.kind for x in steps] == ["knight"] and steps[0].actions == [(KNIGHT, None)]
    assert steps[0].events == [{"type": "played_dev", "player": 0, "card": "knight"},
                               {"type": "game_over", "winner": 0}]
    g = import_game(3, steps[0].after, s.tracker.ctx, UNLIMITED)
    assert g.phase == "game_over" and g.snapshot()["robber"] == BOARD["tile_resource"].index(None)
    assert s.tracker.over


def test_a_robots_game_translates_and_four_observers_keep_their_beliefs(tmp_path):
    require()
    seed = 31
    bots = [Bot("ismcts@20", seed * 4 + i, None) for i in range(4)]
    with Jvm(log=tmp_path / "jvm.log") as jvm:  # suite runs leave no logs under runs/
        jvm.send(game_command("e-full", seed, 2, bridge=False))
        tracker, kinds, imported = None, set(), 0
        while True:
            m = jvm.recv(600.0)
            if m["t"] == "start":
                tracker = Tracker(m)
                for i, b in enumerate(bots):
                    b.observe(i, [])
                continue
            steps = tracker.feed(m) if m["t"] == "ev" else tracker.flush()
            for st in steps:
                kinds.add(st.kind)
                for i, b in enumerate(bots):
                    b.observe(i, [redact(e, i) for e in st.events])
                if st.after is not None:
                    import_game(seed, st.after, tracker.ctx, UNLIMITED)
                    imported += 1
            if m["t"] == "over":
                break
    assert {"setup", "roll", "build", "end_turn"} <= kinds and imported > 100
    assert [b.diagnostics().get("belief_resets", 0) for b in bots] == [0, 0, 0, 0]
    assert tracker.over
