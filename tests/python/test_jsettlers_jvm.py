import pytest

from alphasettler import Game
from jsettlers_harness import require
from oracle.jsettlers import tables as T
from oracle.jsettlers.boards import (
    first_player, game_command, js_layout, native_board, normalized, our_board, streams,
)
from oracle.jsettlers.jvm import Jvm, JvmError

GAME_TIMEOUT = 600.0  # a robots-only game at the fast pace takes well under a minute


@pytest.fixture(scope="module")
def jvm(tmp_path_factory):
    require()
    with Jvm(log=tmp_path_factory.mktemp("jvm") / "jvm.log") as j:  # suite runs leave no logs under runs/
        yield j


def play(jvm, cmd, stop_after_rolls=None):
    """Messages of one game until its "over"; with `stop_after_rolls`, abort after that many rolls."""
    jvm.send(cmd)
    msgs, rolls, aborted = [], 0, False
    while True:
        m = jvm.recv(GAME_TIMEOUT)
        assert m.get("g") == cmd["g"], m
        msgs.append(m)
        if m["t"] == "ev" and m["m"].startswith("SOCDiceResult:"):
            rolls += 1
            if stop_after_rolls is not None and rolls >= stop_after_rolls and not aborted:
                jvm.send({"t": "abort", "g": cmd["g"]})
                aborted = True
        if m["t"] == "over":
            return msgs


def rolls_of(msgs):
    return [int(m["m"].rsplit("param=", 1)[1]) for m in msgs if m["t"] == "ev" and m["m"].startswith("SOCDiceResult:")]


def test_streams_and_layouts_are_pure_functions_of_the_seed():
    assert streams(5) == streams(5) and streams(5) != streams(6)
    assert all(0 <= v < 2**63 for v in streams(5).values())
    assert [first_player(k) for k in range(4)] == [0, 3, 2, 1]
    b = native_board(11)
    assert b["tile_resource"] == Game(11).snapshot()["board"]["tile_resource"]
    assert [T.OUR_EDGE_TO_JS_EDGE[e] for e, _ in b["ports"]] == T.PORT_EDGES
    assert sorted(k for _, k in b["ports"]) == sorted(k for _, k in Game(11).snapshot()["board"]["ports"])
    lay = js_layout(b)
    desert = b["tile_resource"].index(None)
    assert lay["robber"] == T.OUR_TILE_TO_JS_HEX[desert]
    assert lay["hexes"][T.HEX_NUM[lay["robber"]]] == T.DESERT_HEX
    assert sum(1 for h in lay["hexes"] if 1 <= h <= 5) == 18
    assert all(lay["hexes"][n] >= 7 for n in T.PORT_HEXNUMS)


def test_a_robots_game_runs_to_a_winner_on_the_injected_board_with_the_scripted_first_player(jvm):
    msgs = play(jvm, game_command("j-full", 7, 1, bridge=False))
    start = next(m for m in msgs if m["t"] == "start")
    over = msgs[-1]
    assert start["first"] == first_player(1) == 3
    assert start["seats"] == ["robot 1", "robot 2", "robot 3", "robot 4"]
    assert start["smart"] == [T.SMART_STRATEGY] * 4
    assert normalized(our_board(start["board"])) == normalized(native_board(7))
    assert start["s"]["robber"] == start["board"]["robber"]
    assert over["aborted"] is False and 0 <= over["winner"] < 4 and over["vp"][over["winner"]] >= 10
    assert any(m["t"] == "ev" and m["a"].startswith("f") for m in msgs), "client requests are recorded"
    assert all("s" not in m or isinstance(m["s"]["players"], list) for m in msgs if m["t"] == "ev")


def test_dice_and_the_deck_follow_the_seed_in_every_rotation(jvm):
    a = play(jvm, game_command("j-crn-a", 21, 0, bridge=False), stop_after_rolls=12)
    b = play(jvm, game_command("j-crn-b", 21, 2, bridge=False), stop_after_rolls=12)
    c = play(jvm, game_command("j-crn-c", 22, 0, bridge=False), stop_after_rolls=12)
    assert rolls_of(a)[:12] == rolls_of(b)[:12]
    assert rolls_of(a)[:12] != rolls_of(c)[:12]
    deck = [next(m for m in g if m["t"] == "start")["s"]["deck"] for g in (a, b, c)]
    assert deck[0] == deck[1] != deck[2] and len(deck[0]) == 25
    assert all(g[-1]["aborted"] is True and g[-1]["winner"] == -1 for g in (a, b, c))


def test_native_boards_are_jsettlers_own(jvm):
    msgs = play(jvm, game_command("j-native", 7, 0, injected=False, bridge=False), stop_after_rolls=1)
    board = our_board(next(m for m in msgs if m["t"] == "start")["board"])
    snap = {**Game(7).snapshot(), "board": board, "robber": board["tile_resource"].index(None)}
    Game.from_snapshot(7, snap, {})  # a valid board of ours
    assert normalized(board) != normalized(native_board(7))


def test_errors_and_silence_fail_loudly_naming_the_log(tmp_path):
    require()
    with Jvm(log=tmp_path / "jvm.log") as j:
        with pytest.raises(TimeoutError, match=str(j.log)):
            j.recv(0.5)
        j.send({"t": "nonsense"})
        with pytest.raises(JvmError, match="unknown command") as e:
            j.recv(60.0)
        assert str(j.log) in str(e.value)
        assert j.proc.wait(timeout=30) == 1


def test_the_jvm_exits_when_its_stdin_closes(tmp_path):
    require()
    j = Jvm(log=tmp_path / "jvm.log")
    j.proc.stdin.close()
    assert j.proc.wait(timeout=30) == 0
    j.close()
