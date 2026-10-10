import pytest

from jsettlers_harness import require
from oracle.jsettlers import tables as T
from oracle.jsettlers.arena import FALLBACK_REASONS, RECORD_KEYS, check, play_game, run_match
from oracle.jsettlers.boards import first_player, js_layout, native_board
from oracle.jsettlers.jvm import Jvm
from test_jsettlers_events import Script


@pytest.fixture(scope="module")
def jvm(tmp_path_factory):
    require()
    with Jvm(log=tmp_path_factory.mktemp("jvm") / "jvm.log") as j:  # suite runs leave no logs under runs/
        yield j


def test_a_no_trades_bridge_game_records_every_counter(jvm):
    r = play_game(jvm, 51, 1, "ismcts@20", trades=False, arm="no-trades")
    assert tuple(r) == RECORD_KEYS
    assert r["candidate_seat"] == 1 and r["arm"] == "no-trades" and r["aborted"] is False
    assert r["winner"] in (0, 1, 2, 3) and r["vp"][r["winner"]] >= 10
    assert r["turns"] > 10 and r["belief_resets"] == 0
    assert set(r["fallback_reasons"]) <= set(FALLBACK_REASONS)
    assert r["fallbacks"] == sum(r["fallback_reasons"].values()) and r["fallbacks"] < 20
    assert r["our_offers"] == r["offers_to_us"] == r["robot_offers"] == 0
    assert len(r["last_build_turn"]) == 4 and r["last_build_turn"][1] is not None
    assert all(t is None or 0 <= t <= r["turns"] for t in r["last_build_turn"])


def test_a_game_at_the_turn_cap_ends_with_no_winner_and_the_jvm_plays_on(jvm):
    r = play_game(jvm, 52, 0, "ismcts@20", trades=False, arm="no-trades", max_turns=3)
    assert r["aborted"] is True and r["winner"] is None and r["turns"] >= 3
    # The aborted game's bridge may still ask after its "over"; the next game on this JVM answers it.
    r = play_game(jvm, 52, 1, "ismcts@20", trades=False, arm="no-trades", max_turns=3)
    assert r["aborted"] is True and r["candidate_seat"] == 1


class FakeJvm:
    def __init__(self, messages):
        self.messages, self.sent = list(messages), []

    def send(self, m):
        self.sent.append(m)

    def recv(self, timeout):
        return self.messages.pop(0)


def start_message(name, seed, k):
    """A "start" line as the harness writes it, for seed's injected board with our seat at turn position k."""
    s = Script(first=first_player(k), state=T.START1A, current=first_player(k))
    board, lay = native_board(seed), js_layout(native_board(seed))
    hexes = [[h, lay["hexes"][T.HEX_NUM[h]], max(0, lay["numbers"][T.HEX_NUM[h]])] for h in T.OUR_TILE_TO_JS_HEX]
    ports = [[e, 0 if kind == "generic" else T.OUR_RES_TO_JS[T.RESOURCE_NAMES.index(kind)]]
             for e, (_, kind) in zip(T.PORT_EDGES, board["ports"])]
    return {"t": "start", "g": name, "first": first_player(k), "board": {"hexes": hexes, "ports": ports}, "s": s.raw}


def test_late_messages_of_an_earlier_aborted_game_are_answered_and_dropped():
    name = "no-trades-52-2"
    over = {"t": "over", "g": name, "winner": -1, "vp": [2, 2, 2, 2], "rounds": 0, "aborted": True}
    j = FakeJvm([{"t": "ask", "id": 41, "g": "no-trades-52-1", "kind": "main", "state": 20, "extra": {}},
                 {"t": "fallback", "g": "no-trades-52-1", "reason": "java-timeout"},
                 start_message(name, 52, 2), over])
    r = play_game(j, 52, 2, "random", trades=False, arm="no-trades")
    assert j.sent[0]["t"] == "game" and j.sent[1] == {"t": "reply", "id": 41, "fallback": "turn-cap"}
    assert r["fallbacks"] == 0 and r["aborted"] is True and r["last_build_turn"] == [None] * 4
    j = FakeJvm([{"t": "ev", "g": "elsewhere", "a": "all", "m": "SOCGameState:game=elsewhere|state=20"}])
    with pytest.raises(RuntimeError, match="a message from another game"):
        play_game(j, 52, 3, "random", trades=False, arm="no-trades")


def test_run_match_rotates_the_seat_on_one_board(tmp_path):
    require()
    rs = run_match("ismcts@20", "smart", 1, seed_start=53, workers=2, trades=False, log_dir=tmp_path)
    assert [r["candidate_seat"] for r in rs] == [0, 1, 2, 3] and {r["seed"] for r in rs} == {53}
    assert all(r["belief_resets"] == 0 for r in rs)
    assert len(list(tmp_path.glob("no-trades-*.log"))) == 2


@pytest.mark.parametrize("args,match", [
    (("ismcts@20", "fast", 1), "unknown JSettlers robot"),
    (("no-such-bot", "smart", 1), "unknown bot"),
    (("ismcts@20", "smart", 0), "seeds must be positive"),
])
def test_check_rejects_bad_requests_before_any_game(args, match):
    with pytest.raises(ValueError, match=match):
        check(*args)


def test_native_boards_without_trades_are_refused_before_any_game():
    with pytest.raises(ValueError, match="native boards need trades"):
        run_match("ismcts@20", "smart", 1, trades=False, native_boards=True)
