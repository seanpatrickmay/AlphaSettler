import copy
import json

import pytest

import oracle.jsettlers.diff as diff
import oracle.jsettlers.jvm as jvm_module
from alphasettler import Game, cli
from jsettlers_harness import require
from oracle.jsettlers import events
from oracle.jsettlers import tables as T
from oracle.jsettlers.allowlist import ENTRIES
from oracle.jsettlers.boards import js_layout, native_board
from oracle.jsettlers.diff import GameResult, compare, format_summary, play_game, summarize
from oracle.jsettlers.jvm import Jvm

SEED = 3  # the scripted games' seed: the board of their start message is native_board(SEED)
NAME = f"diff-{SEED}-0"


def rset(ours=(0, 0, 0, 0, 0)):
    return "clay={}|ore={}|sheep={}|wheat={}|wood={}|unknown=0".format(*T.ours_to_js5(list(ours)))


def board_raw(board):
    lay = js_layout(board)
    hexes = [[h, lay["hexes"][T.HEX_NUM[h]], max(0, lay["numbers"][T.HEX_NUM[h]])] for h in T.OUR_TILE_TO_JS_HEX]
    ports = [[e, T.MISC_PORT if kind == "generic" else T.OUR_RES_TO_JS[T.RESOURCE_NAMES.index(kind)]]
             for e, (_, kind) in zip(T.PORT_EDGES, board["ports"])]
    return {"hexes": hexes, "ports": ports, "robber": lay["robber"]}


class FakeJvm:
    """The harness's side of one scripted game: `recv` hands out its messages in turn; `sent` keeps
    what the driver sent."""

    def __init__(self, msgs):
        self.msgs, self.sent = list(msgs), []

    def send(self, msg):
        self.sent.append(msg)

    def recv(self, timeout):
        return self.msgs.pop(0)


class Stream:
    """One game's harness messages written by hand. JSettlers' first player is seat 0, so our seat i is
    its seat i; `raw` is the server-side position, and `ev(..., changed=True)` attaches it."""

    def __init__(self, state=T.PLAY1, hands=(), settlements=(), roads=()):
        board = native_board(SEED)
        players = [{"res": [0] * 5, "dev": [0] * 10, "played": [0] * 4, "settlements": [], "cities": [], "roads": [],
                    "discard": 0, "played_dev": False, "vp": 0} for _ in range(4)]
        self.raw = {"state": state, "current": 0, "robber": js_layout(board)["robber"],
                    "deck": [9] * 14 + [1, 1, 2, 2, 3, 3, 4, 5, 6, 7, 8], "players": players,
                    "longest": -1, "largest": -1, "winner": -1}
        for seat, h in hands:
            self.hand(seat, h)
        for seat, node in settlements:
            players[seat]["settlements"].append(T.OUR_NODE_TO_JS_NODE[node])
            players[seat]["vp"] += 1
        for seat, edge in roads:
            players[seat]["roads"].append(T.OUR_EDGE_TO_JS_EDGE[edge])
        self.msgs = [{"t": "start", "g": NAME, "first": 0, "board": board_raw(board), "s": copy.deepcopy(self.raw)}]

    def hand(self, seat, h):
        self.raw["players"][seat]["res"] = T.ours_to_js5(list(h))

    def ev(self, aud, text, changed=False):
        msg = {"t": "ev", "g": NAME, "a": aud, "m": text}
        if changed:
            msg["s"] = copy.deepcopy(self.raw)
        self.msgs.append(msg)

    def play(self):
        jvm = FakeJvm(self.msgs + [{"t": "over", "g": NAME, "winner": -1}])
        return play_game(jvm, SEED, 0), jvm


@pytest.fixture(scope="module")
def jvm(tmp_path_factory):
    require()
    with Jvm(log=tmp_path_factory.mktemp("jvm") / "jvm.log") as j:  # suite runs leave no logs under runs/
        yield j


def test_two_robots_games_replay_with_no_untriaged_mismatch(jvm):
    results = [play_game(jvm, seed) for seed in (41, 42)]
    for r in results:
        assert r.mismatch is None, (r.mismatch, r.trace[-5:])
        assert r.steps > 150 and (r.winner is not None or r.capped)
        assert set(r.allowlisted) <= set(ENTRIES)
    assert "0 mismatches" in format_summary(summarize(results))


def test_corrupted_dice_are_caught(jvm, monkeypatch):
    monkeypatch.setattr(events, "dice_pair", lambda total: [1, 1] if total != 2 else [1, 2])
    r = play_game(jvm, 43)
    # Every roll is replayed as a 2: the first roll that pays out (or a 7) differs in hands or phase.
    assert r.mismatch is not None and " after roll by " in r.mismatch, r.mismatch
    assert "hand'" in r.mismatch or "'phase'" in r.mismatch, r.mismatch
    assert r.trace[-1].startswith("f"), r.trace  # the trace ends at the request that closed the mismatching stretch


def test_compare_names_the_fields_that_differ():
    a = {"board": 1, "robber": 2, "bank": [1] * 5, "current": 0, "phase": "main", "turn": 3, "setup_step": 8,
         "longest_road_owner": None, "largest_army_owner": None, "dev_played_this_turn": False,
         "offers_this_turn": 0, "dev_deck": ["knight", "monopoly"], "return_phase": "main",
         "setup_road_node": None, "roads_left": None, "winner": None,
         "players": [{"hand": [0] * 5, "dev_hand": [0] * 5, "dev_new": [0] * 5, "dev_played": [0] * 5,
                      "knights_played": 0, "settlements": [], "cities": [], "roads": [], "discard_remaining": 0}] * 4}
    b = json.loads(json.dumps(a))
    b["dev_deck"] = ["monopoly", "knight"]  # the deck compares as a multiset
    assert compare(a, b) == []
    b["players"][2]["hand"] = [1, 0, 0, 0, 0]
    b["turn"] = 4
    assert compare(a, b) == ["turn", "players[2].hand"]


def test_every_allowlist_entry_names_its_rule():
    assert set(ENTRIES) == {"domestic-trade", "road-building-cancelled", "bank-trade", "forced-end-turn",
                            "forced-setup-end"}
    assert all(e.rule and e.why for e in ENTRIES.values())


def test_the_command_replays_and_writes_one_line_a_game(tmp_path, capsys, monkeypatch):
    require()
    monkeypatch.setattr(jvm_module, "LOGS", tmp_path / "logs")  # one worker runs in this process: no logs under runs/
    out = tmp_path / "diff.jsonl"
    rc = cli.main(["jsettlers-diff", "--games", "1", "--seed-start", "44", "--workers", "1", "--out", str(out)])
    text = capsys.readouterr().out
    assert rc == 0 and "1 games" in text and "0 mismatches" in text
    [line] = out.read_text().splitlines()
    assert json.loads(line)["seed"] == 44 and json.loads(line)["mismatch"] is None
    assert len(list((tmp_path / "logs").iterdir())) == 1


def test_the_command_says_what_is_missing(monkeypatch, tmp_path, capsys):
    import oracle.jsettlers as js

    monkeypatch.setattr(js, "JAR", tmp_path / "absent.jar")
    assert cli.main(["jsettlers-diff", "--games", "1"]) == 2
    assert "scripts/fetch-jsettlers.sh" in capsys.readouterr().err


def test_a_result_summary_counts_allowlisted_kinds():
    rs = [GameResult(1, steps=10), GameResult(2, steps=5)]
    rs[1].allowlisted["bank-trade"] += 2
    s = summarize(rs)
    assert (s.games, s.steps, s.clean_games, s.allowlisted["bank-trade"]) == (2, 15, 1, 2)


def roll_then_forced_end(stream, gain=None, nxt=1, state=T.ROLL_OR_CARD):
    """Player 0 rolls a 9 (nobody has a building on it) and the server then ends its stalled turn,
    handing it to seat `nxt` in `state` (JSettlers' position as the stream shows it)."""
    stream.ev("f0", "SOCRollDice:game=x")
    stream.ev("all", "SOCDiceResult:game=x|param=9")
    if gain is not None:
        stream.hand(*gain)
    stream.raw["state"] = T.PLAY1
    stream.ev("all", "SOCGameState:game=x|state=20", changed=True)
    stream.raw["current"], stream.raw["state"] = nxt, state
    stream.ev("all", f"SOCTurn:game=x|playerNumber={nxt}|gameState=15", changed=True)  # SOCForceEndTurnThread
    stream.ev("all", f"SOCRollDicePrompt:game=x|playerNumber={nxt}")


def road_building_ended(stream, nxt=1):
    """Player 0 ends its turn with both free roads unplaced; the turn goes to seat `nxt`."""
    stream.ev("f0", "SOCEndTurn:game=x")
    stream.raw["current"], stream.raw["state"] = nxt, T.ROLL_OR_CARD
    stream.ev("all", f"SOCTurn:game=x|playerNumber={nxt}|gameState=15", changed=True)


def domestic_trade(stream, partner_gets):
    """Player 0 gives one wood for player 2's wheat; JSettlers' position shows player 2 holding `partner_gets`."""
    give, get = rset([1, 0, 0, 0, 0]), rset([0, 0, 0, 1, 0])
    stream.ev("f0", f"SOCMakeOffer:game=x|from=0|to=false,true,true,true|give={give}|get={get}")
    stream.ev("all", f"SOCMakeOffer:game=x|from=0|to=false,true,true,true|give={give}|get={get}")
    stream.ev("f2", "SOCAcceptOffer:game=x|accepting=0|offering=0")
    stream.hand(0, [0, 0, 0, 1, 0])
    stream.hand(2, partner_gets)
    stream.ev("all", f"SOCAcceptOffer:game=x|accepting=2|offering=0|toAccepting={give}|toOffering={get}",
              changed=True)
    stream.ev("all", "SOCClearOffer:game=x|playerNumber=-1")


def bank_trades(stream, *trades):
    """Player 0's bank trades, each (give, get, its hand after) in our resource order."""
    for give, get, hand in trades:
        stream.ev("f0", f"SOCBankTrade:game=x|give={rset(give)}|get={rset(get)}")
        stream.hand(0, hand)
        stream.ev("all", f"SOCBankTrade:game=x|give={rset(give)}|get={rset(get)}|pn=0", changed=True)


def last_setup_road():
    """The start of our engine's last setup step on SEED's board (player 0's second road, first choices
    everywhere), as a Stream position, with player 0's second settlement node and what it produced."""
    g = Game(SEED)
    while not (g.phase == "setup_settlement" and g.snapshot()["setup_step"] == 7):
        g.apply(g.legal_actions()[0])
    before = g.snapshot()["players"][0]["hand"]
    g.apply(g.legal_actions()[0])
    snap = g.snapshot()
    node, ps = snap["setup_road_node"], snap["players"]
    produced = [a - b for a, b in zip(ps[0]["hand"], before)]
    start = dict(state=T.START2B, hands=[(i, p["hand"]) for i, p in enumerate(ps)],
                 settlements=[(i, n) for i, p in enumerate(ps) for n in p["settlements"]],
                 roads=[(i, e) for i, p in enumerate(ps) for e in p["roads"]])
    return start, node, produced


LAST_SETUP, SECOND_NODE, PRODUCED = last_setup_road()


def stalled_setup_road(stream, removed=SECOND_NODE, takes_back=True, nxt=1):
    """Player 0 asks for a road JSettlers refuses and stalls; the server ends its setup turn, removing the
    settlement `removed` (and, with `takes_back`, the cards it produced), and JSettlers' play begins at `nxt`."""
    stream.ev("f0", "SOCPutPiece:game=x|playerNumber=0|pieceType=0|coord=16")
    stream.ev("p0", "SOCDeclinePlayerRequest:game=x|state=0|reason=4|detail1=0|detail2=22|text=no")
    p = stream.raw["players"][0]
    p["settlements"].remove(T.OUR_NODE_TO_JS_NODE[removed])
    p["vp"] -= 1
    if takes_back:
        stream.hand(0, [h - x for h, x in zip(T.js5_to_ours(p["res"]), PRODUCED)])
    stream.ev("all", "SOCCancelBuildRequest:game=x|pieceType=1", changed=True)
    stream.raw["current"], stream.raw["state"] = nxt, T.ROLL_OR_CARD
    stream.ev("all", "SOCClearOffer:game=x|playerNumber=-1", changed=True)
    stream.ev("all", f"SOCTurn:game=x|playerNumber={nxt}|gameState=15")


def cancelled_at(step, second):
    """Our engine's last setup position on SEED's board with `setup_step` set to `step`, and a forced setup end of
    player 0's settlement awaiting its road, placed in JSettlers' second round (START2B) or not."""
    g = Game(SEED)
    while not (g.phase == "setup_road" and g.snapshot()["setup_step"] == 7):
        g.apply(g.legal_actions()[0])
    ours = {**g.snapshot(), "setup_step": step}
    theirs = {**ours, "phase": "pre_roll", "current": 1, "setup_road_node": None, "setup_step": 8}
    st = events.Step("forced_setup_end", 0, [], [], None, untranslatable="forced-setup-end",
                     info={"second_round": second})
    return ours, diff.excused(ours, theirs, st)


def test_a_cancelled_settlements_cards_go_back_by_jsettlers_round_not_our_setup_step():
    # After a round-1 skip our setup step no longer says the round: JSettlers takes the cards back (it clears the
    # hand) exactly when the settlement was placed in its second round, START2A, the one that pays.
    ours, exp = cancelled_at(3, True)
    assert exp["players"][0]["hand"] == [0] * 5 and any(ours["players"][0]["hand"])
    assert ours["setup_road_node"] not in exp["players"][0]["settlements"]
    ours, exp = cancelled_at(7, False)
    assert exp["players"][0]["hand"] == ours["players"][0]["hand"]


def one(r, n=1):
    return [n if i == r else 0 for i in range(5)]


WOOD, ORE = 0, 4
PORT_EDGE, PORT_KIND = next((e, k) for e, k in native_board(SEED)["ports"] if k != "generic")
K = T.RESOURCE_NAMES.index(PORT_KIND)  # player 0's 2:1 port resource in the port cases
R, Z = [r for r in range(5) if r != K][:2]  # two resources it trades at 4:1
AT_PORT = dict(state=T.PLAY1, hands=((0, one(K, 2)),), settlements=((0, T.EDGE_NODES[PORT_EDGE][0]),))
FOUR_WOOD = dict(state=T.PLAY1, hands=((0, one(WOOD, 4)),))
WOOD_FOR_ORE = (one(WOOD, 4), one(ORE), one(ORE))  # a whole 4:1 maritime trade, replayed as ours
ROAD_BUILDING = dict(state=T.PLACING_FREE_ROAD1, settlements=((0, T.EDGE_NODES[0][0]),), roads=((0, 0),))
TRADERS = dict(hands=((0, [1, 0, 0, 0, 0]), (2, [0, 0, 0, 1, 0])))

# id: (start, script, its arguments, None when the step must pass, else a fragment the mismatch names)
ALLOWLISTED = {
    "forced end after a roll": (dict(state=T.ROLL_OR_CARD), roll_then_forced_end, (), None),
    "forced end after a roll that pays a card no building pays": (
        dict(state=T.ROLL_OR_CARD), roll_then_forced_end, ((2, [1, 0, 0, 0, 0]),), "players[2].hand"),
    "forced end handing the turn to the wrong seat": (
        dict(state=T.ROLL_OR_CARD), roll_then_forced_end, (None, 2), "'current'"),
    "forced end leaving the player in main": (
        dict(state=T.ROLL_OR_CARD), roll_then_forced_end, (None, 0, T.PLAY1), "'phase'"),
    "forced setup end": (LAST_SETUP, stalled_setup_road, (), None),
    "forced setup end keeping the cards the settlement produced": (
        LAST_SETUP, stalled_setup_road, (SECOND_NODE, False), "players[0].hand"),
    "forced setup end removing the settlement that has its road": (
        LAST_SETUP, stalled_setup_road, (LAST_SETUP["settlements"][0][1],), "players[0].settlements"),
    "road building ended": (ROAD_BUILDING, road_building_ended, (), None),
    "road building ended handing the turn to the wrong seat": (
        ROAD_BUILDING, road_building_ended, (2,), "'current'"),
    "domestic trade": (TRADERS, domestic_trade, ([1, 0, 0, 0, 0],), None),
    "domestic trade moving more than the traded cards": (TRADERS, domestic_trade, ([2, 0, 0, 0, 0],),
                                                         "players[2].hand"),
    "bank trade undone": (FOUR_WOOD, bank_trades, (WOOD_FOR_ORE, (one(ORE), one(WOOD, 4), one(WOOD, 4))), None),
    "bank trade undone at a port": (AT_PORT, bank_trades, ((one(K, 2), one(R), one(R)), (one(R), one(K, 2), one(K, 2))),
                                    None),
    "bank trade one for one at no port": (
        FOUR_WOOD, bank_trades, (WOOD_FOR_ORE, (one(ORE), one(WOOD), one(WOOD))), "not the undo of whole maritime"),
    "bank trade of nothing for five": (
        FOUR_WOOD, bank_trades, ((one(WOOD, 0), one(ORE, 5), [4, 0, 0, 0, 5]),), "not the undo of whole maritime"),
    "bank trade at a port the player does not have": (
        AT_PORT, bank_trades, ((one(K, 2), one(R), one(R)), (one(R), one(Z, 2), one(Z, 2))),
        "not the undo of whole maritime"),
    "bank trade undoing no trade of the player": (
        dict(state=T.PLAY1, hands=((0, one(ORE)),)), bank_trades, ((one(ORE), one(WOOD, 4), one(WOOD, 4)),),
        "reverses no bank trade"),
}


@pytest.mark.parametrize("case", list(ALLOWLISTED))
def test_an_allowlisted_step_is_compared_but_for_what_its_rule_changes(case):
    start, script, args, wrong = ALLOWLISTED[case]
    s = Stream(**start)
    script(s, *args)
    r, _ = s.play()
    if wrong is None:
        assert r.mismatch is None and sum(r.allowlisted.values()) == 1, (r.mismatch, r.allowlisted)
    else:
        assert r.mismatch is not None and wrong in r.mismatch, r.mismatch


def test_an_untranslatable_kind_off_the_allowlist_is_a_mismatch():
    s = Stream(state=T.ROLL_OR_CARD, hands=((2, [1, 0, 0, 0, 0]),))
    s.ev("f0", "SOCRollDice:game=x")
    s.ev("all", "SOCDiceResult:game=x|param=9")
    s.hand(2, [0, 0, 0, 0, 0])  # a roll that takes a card: no production of ours does that
    s.raw["state"] = T.PLAY1
    s.ev("all", "SOCGameState:game=x|state=20", changed=True)
    r, _ = s.play()
    assert "production" not in ENTRIES
    assert r.mismatch == "step 0 (roll): our rules cannot replay it (production)" and not r.allowlisted


def test_a_tracker_failure_is_the_games_mismatch_and_the_trace_stops_there():
    s = Stream(state=T.ROLL_OR_CARD)
    s.ev("f0", "SOCRollDice:game=x")
    s.ev("all", "SOCDiceResult:game=x|param=nine")  # a message the Tracker cannot read
    s.ev("f0", "SOCEndTurn:game=x")  # closes the stretch: the Tracker raises here
    s.raw["current"] = 1
    s.ev("all", "SOCTurn:game=x|playerNumber=1|gameState=15", changed=True)
    r, jvm = s.play()
    assert r.mismatch.startswith("step 0 (ev message): ValueError:"), r.mismatch
    assert r.trace[-1] == "f0:SOCEndTurn:game=x" and not any("SOCTurn" in t for t in r.trace)
    assert jvm.sent[-1] == {"t": "abort", "g": NAME} and jvm.msgs == []


def test_a_start_position_that_does_not_import_is_the_games_mismatch():
    s = Stream(state=T.PLACING_ROAD)  # a state with no phase of ours
    r, jvm = s.play()
    assert r.mismatch.startswith("step 0 (start message): Untranslatable:"), r.mismatch
    assert (r.winner, r.turns) == (None, 0) and jvm.sent[-1] == {"t": "abort", "g": NAME}


def test_a_failing_harness_exits_3_not_the_missing_harness_code(monkeypatch, capsys):
    def silent(seeds, workers=None):
        raise TimeoutError("the JSettlers JVM sent nothing for 900.0 s; see x.log")

    monkeypatch.setattr(cli, "_jsettlers_missing", lambda: None)
    monkeypatch.setattr(diff, "run", silent)
    assert cli.main(["jsettlers-diff", "--games", "1"]) == 3
    assert "harness failed: TimeoutError: the JSettlers JVM sent nothing" in capsys.readouterr().err


def test_a_mismatched_games_line_keeps_its_trace(monkeypatch, tmp_path, capsys):
    bad = GameResult(5, steps=3, mismatch="step 3 after roll by 0: ['phase']", trace=["f0:SOCRollDice:game=x"])
    monkeypatch.setattr(cli, "_jsettlers_missing", lambda: None)
    monkeypatch.setattr(diff, "run", lambda seeds, workers=None: [GameResult(4, steps=9), bad])
    out = tmp_path / "diff.jsonl"
    assert cli.main(["jsettlers-diff", "--games", "2", "--seed-start", "4", "--out", str(out)]) == 1
    clean, mismatched = (json.loads(line) for line in out.read_text().splitlines())
    assert "trace" not in clean and mismatched["trace"] == ["f0:SOCRollDice:game=x"]
    assert "mismatch seed 5: step 3 after roll by 0" in capsys.readouterr().err
