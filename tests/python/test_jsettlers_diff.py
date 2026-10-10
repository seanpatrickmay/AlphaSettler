import copy
import json

import pytest

import oracle.jsettlers.diff as diff
import oracle.jsettlers.jvm as jvm_module
from alphasettler import cli
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

    def __init__(self, state=T.PLAY1, hands=()):
        board = native_board(SEED)
        players = [{"res": [0] * 5, "dev": [0] * 10, "played": [0] * 4, "settlements": [], "cities": [], "roads": [],
                    "discard": 0, "played_dev": False, "vp": 0} for _ in range(4)]
        self.raw = {"state": state, "current": 0, "robber": js_layout(board)["robber"],
                    "deck": [9] * 14 + [1, 1, 2, 2, 3, 3, 4, 5, 6, 7, 8], "players": players,
                    "longest": -1, "largest": -1, "winner": -1}
        for seat, h in hands:
            self.hand(seat, h)
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
    assert set(ENTRIES) == {"domestic-trade", "road-building-cancelled", "bank-trade", "forced-end-turn"}
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


def roll_then_forced_end(stream, gain=None):
    """Player 0 rolls a 9 (nobody has a building on it) and the server then ends its stalled turn."""
    stream.ev("f0", "SOCRollDice:game=x")
    stream.ev("all", "SOCDiceResult:game=x|param=9")
    if gain is not None:
        stream.hand(*gain)
    stream.raw["state"] = T.PLAY1
    stream.ev("all", "SOCGameState:game=x|state=20", changed=True)
    stream.raw["current"], stream.raw["state"] = 1, T.ROLL_OR_CARD
    stream.ev("all", "SOCTurn:game=x|playerNumber=1|gameState=15", changed=True)  # SOCForceEndTurnThread
    stream.ev("all", "SOCRollDicePrompt:game=x|playerNumber=1")


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


def undone_bank_trade(stream, got):
    """Player 0 trades one wheat for four ore (undoing an earlier 4:1); JSettlers' position shows it holding `got`."""
    give, get = rset([0, 0, 0, 1, 0]), rset([0, 0, 0, 0, 4])
    stream.ev("f0", f"SOCBankTrade:game=x|give={give}|get={get}")
    stream.hand(0, got)
    stream.ev("all", f"SOCBankTrade:game=x|give={give}|get={get}|pn=0", changed=True)


EXCUSED = {
    "forced-end-turn": (dict(state=T.ROLL_OR_CARD), roll_then_forced_end, (), ((2, [1, 0, 0, 0, 0]),),
                        "players[2].hand"),
    "domestic-trade": (dict(hands=((0, [1, 0, 0, 0, 0]), (2, [0, 0, 0, 1, 0]))), domestic_trade,
                       ([1, 0, 0, 0, 0],), ([2, 0, 0, 0, 0],), "players[2].hand"),
    "bank-trade": (dict(hands=((0, [0, 0, 0, 1, 0]),)), undone_bank_trade, ([0, 0, 0, 0, 4],), ([0, 0, 0, 0, 3],),
                   "players[0].hand"),
}


@pytest.mark.parametrize("kind", sorted(EXCUSED))
def test_an_allowlisted_step_is_compared_but_for_what_its_rule_changes(kind):
    start, script, right, _, _ = EXCUSED[kind]
    s = Stream(**start)
    script(s, *right)
    r, _ = s.play()
    assert r.mismatch is None and dict(r.allowlisted) == {kind: 1}


@pytest.mark.parametrize("kind", sorted(EXCUSED))
def test_a_translation_error_on_an_allowlisted_step_is_a_mismatch(kind):
    start, script, _, wrong, field = EXCUSED[kind]
    s = Stream(**start)
    script(s, *wrong)
    r, _ = s.play()
    assert r.mismatch is not None and f"({kind}): beyond what its rule changes" in r.mismatch, r.mismatch
    assert field in r.mismatch


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
