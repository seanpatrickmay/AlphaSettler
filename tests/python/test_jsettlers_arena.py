import json
import os
import time
from pathlib import Path

import pytest

import oracle.jsettlers.arena
from alphasettler import Bot
from jsettlers_harness import require
from oracle.jsettlers import actions as A
from oracle.jsettlers import diff
from oracle.jsettlers import tables as T
from oracle.jsettlers.arena import FALLBACK_REASONS, RECORD_KEYS, check, kept, play_game, run_match, run_workers
from oracle.jsettlers.boards import first_player, js_layout, native_board, normalized
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
    assert r["candidate_seat"] == 1 and r["arm"] == "no-trades" and r["aborted"] is False and r["abort_reason"] is None
    assert r["winner"] in (0, 1, 2, 3) and r["vp"][r["winner"]] >= 10
    assert r["turns"] > 10 and r["belief_resets"] == 0
    assert set(r["fallback_reasons"]) <= set(FALLBACK_REASONS)
    assert r["fallbacks"] == sum(r["fallback_reasons"].values()) and r["fallbacks"] < 20
    assert r["our_offers"] == r["offers_to_us"] == r["robot_offers"] == 0
    assert len(r["last_build_turn"]) == 4 and r["last_build_turn"][1] is not None and r["robot_desync"] is False
    assert all(t is None or 0 <= t <= r["turns"] for t in r["last_build_turn"])


def test_a_game_at_the_turn_cap_ends_with_no_winner_and_the_jvm_plays_on(jvm):
    r = play_game(jvm, 52, 0, "ismcts@20", trades=False, arm="no-trades", max_turns=3)
    assert r["aborted"] is True and r["abort_reason"] == "turn-cap" and r["winner"] is None and r["turns"] >= 3
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
    assert r["robot_desync"] is False and r["abort_reason"] is None
    j = FakeJvm([{"t": "ev", "g": "elsewhere", "a": "all", "m": "SOCGameState:game=elsewhere|state=20"}])
    with pytest.raises(RuntimeError, match="a message from another game"):
        play_game(j, 52, 3, "random", trades=False, arm="no-trades")


SEED78 = Path(__file__).parent / "data" / "jsettlers_seed78_k1.jsonl"  # the harness's lines, bridge game y-78-1


def seed78(name=None):
    msgs = [json.loads(line) for line in SEED78.read_text().splitlines()]
    return msgs if name is None else [{**m, "g": name} for m in msgs if m["t"] in ("start", "ev", "over")]


def test_a_robots_setup_settlement_cancelled_by_a_forced_end_aborts_the_game_at_once():
    # Seed 78, k=1 (stall-debug.md): robot 3's planInitRoad returns an edge off the board, the server ends its setup
    # turn and cancels its settlement, and the robot keeps it in its own copy: the rest would be played against a
    # robot on a board it does not have, force-ended every turn. The driver aborts at the first decision after it.
    j = FakeJvm(seed78())
    r = play_game(j, 78, 1, "ismcts@20", trades=False, arm="y")
    assert [m["t"] for m in j.sent[:7]] == ["game", "reply", "reply", "reply", "reply", "abort", "reply"]
    assert j.sent[6]["fallback"] == "robot-desync"  # answered, not decided and not counted
    assert r["robot_desync"] is True and r["abort_reason"] == "robot-desync" and r["winner"] is None
    assert r["fallbacks"] == 0 and r["belief_resets"] == 0  # (the recording plays on; a real abort stops it)


def test_the_seed_78_stream_replays_with_the_forced_setup_end_allowlisted():
    res = diff.play_game(FakeJvm(seed78("diff-78-1")), 78, 1)
    assert res.mismatch is None and res.allowlisted["forced-setup-end"] == 1 and res.steps > 30


def test_seed_78_desyncs_a_robot_in_the_live_harness_and_the_jvm_plays_on(jvm):
    t = time.monotonic()
    r = play_game(jvm, 78, 1, "ismcts@20", trades=False, arm="no-trades")  # the forced setup end itself takes 60 s
    assert r["robot_desync"] is True and r["abort_reason"] == "robot-desync" and r["winner"] is None
    assert r["turns"] == 0 and time.monotonic() - t < 100
    r = play_game(jvm, 51, 1, "ismcts@20", trades=False, arm="after-desync")
    assert r["aborted"] is False and r["winner"] is not None and r["robot_desync"] is False


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


def test_a_native_trading_game_answers_and_counts_the_robots_offers(jvm):
    r = play_game(jvm, 61, 2, "ismcts@20+t", trades=True, arm="native")
    assert r["belief_resets"] == 0 and r["arm"] == "native"
    assert r["robot_offers"] > 0, "stock smart robots offer in a full trading game"
    assert r["offers_to_us"] >= r["unrepresentable"] + r["counters_rejected"] + r["late"] + r["beaten"]
    assert r["our_offers_accepted"] <= r["our_offers"] and r["offers_to_us_accepted"] <= r["offers_to_us"]
    assert set(r["fallback_reasons"]) <= set(FALLBACK_REASONS)


def test_aligned_robots_offer_only_within_our_trading_game(tmp_path):
    require()
    with Jvm("aligned", log=tmp_path / "jvm.log") as j:
        assert j.ready["mode"] == "aligned"
        r = play_game(j, 62, 0, "ismcts@20+t", trades=True, arm="aligned")
    assert r["robot_offers"] > 0, "capped robots still make one-card offers"
    assert r["robot_offers_over_caps"] == 0 and r["belief_resets"] == 0 and r["arm"] == "aligned"


class Offering:
    """`ismcts@20+t` (which itself never offers), but it plays the first legal offer at each main decision and
    the first move other than cancelling at a confirm, so our offers cross the bridge (forks included)."""

    def __init__(self, name, seed, nets_dir=None):
        self.bot = Bot(name, seed, nets_dir)

    def act(self, g):
        legal = g.legal_actions()
        if g.phase == "main" and (offers := [a for a in legal if A.is_offer(a)]):
            return offers[0]
        if g.phase == "trade_confirm":
            return next(a for a in legal if a != A.CANCEL)
        return self.bot.act(g)

    def observe(self, seat, events):
        self.bot.observe(seat, events)

    def diagnostics(self):
        return self.bot.diagnostics()


@pytest.mark.parametrize("mode,arm", [("stock", "native"), ("aligned", "aligned")])
def test_our_offers_cross_the_bridge_and_are_answered(mode, arm, tmp_path, monkeypatch):
    require()
    monkeypatch.setattr(oracle.jsettlers.arena, "Bot", Offering)
    with Jvm(mode, log=tmp_path / "jvm.log") as j:
        sent = []
        send = j.send

        def spy(m):
            if m.get("do") == "offer":
                sent.append(m)
            send(m)
        monkeypatch.setattr(j, "send", spy)
        r = play_game(j, 63, 1, "ismcts@20+t", trades=True, arm=arm)
    assert len(sent) > 0 and r["our_offers"] >= len(sent) and r["our_offers_accepted"] <= r["our_offers"]
    assert r["belief_resets"] == 0 and set(r["fallback_reasons"]) <= set(FALLBACK_REASONS)


@pytest.mark.parametrize("native_boards,arm", [(True, "native-boards"), (False, "native")])
def test_a_native_board_game_records_its_board(native_boards, arm):
    # JSettlers' own boards are unseeded, so a native-boards record names the board it was played on.
    name = f"{arm}-52-2"
    over = {"t": "over", "g": name, "winner": -1, "vp": [2, 2, 2, 2], "rounds": 0, "aborted": True}
    r = play_game(FakeJvm([start_message(name, 52, 2), over]), 52, 2, "random", arm=arm, native_boards=native_boards)
    assert tuple(r) == RECORD_KEYS
    assert r["board"] == (normalized(native_board(52)) if native_boards else None)
    assert json.loads(json.dumps(r["board"])) == r["board"]


def game(seed, seat, abort_reason=None):
    return {"seed": seed, "candidate_seat": seat, "abort_reason": abort_reason}


def test_kept_drops_every_seed_with_a_desync_game_and_keeps_turn_cap_games():
    rs = ([game(0, k) for k in range(4)] + [game(1, 0), game(1, 1, "robot-desync"), game(1, 2), game(1, 3)]
          + [game(2, 0, "turn-cap")] + [game(2, k) for k in (1, 2, 3)] + [game(3, k, "robot-desync") for k in range(4)])
    ks, dropped = kept(rs)
    assert dropped == [1, 3]
    assert ks == [r for r in rs if r["seed"] in (0, 2)]
    assert kept([]) == ([], [])


# Fake workers for `run_workers` (module level, so a spawned process can import them).

def finishes(n, pause):
    for i in range(n):
        time.sleep(pause)
        yield {"seed": i, "pid": os.getpid()}


def fails_after(n):
    yield from finishes(n, 0.0)
    raise TimeoutError("the JSettlers JVM sent nothing for 900.0 s; see x.log")


def dies_after(n):
    yield from finishes(n, 0.0)
    time.sleep(0.5)  # let the records reach the queue, then die as a killed process would
    os._exit(9)


def test_workers_records_arrive_as_each_finishes():
    got = []
    rs = run_workers(finishes, [(3, 0.0), (2, 0.0)], on_record=got.append, poll=0.1)
    assert len(rs) == 5 and got == rs and len({r["pid"] for r in rs}) == 2


@pytest.mark.parametrize("broken,match", [
    ((fails_after, (2,)), r"worker 1 \(pid \d+\) failed: TimeoutError: the JSettlers JVM sent nothing"),
    ((dies_after, (2,)), r"worker 1 \(pid \d+\) exited with code 9 before its games were done"),
])
def test_one_broken_worker_stops_every_worker_at_once(broken, match):
    # The healthy worker would take 60 s; the broken one fails within its first seconds. The match raises as soon
    # as the failure is seen, with the healthy worker terminated (not waited for: `run_workers` joins every worker
    # before it returns, and would take 10 s more to kill one that ignored the terminate) and the broken worker's
    # finished records already passed on.
    got = []
    work, args = broken
    t = time.monotonic()
    with pytest.raises(RuntimeError, match=match):
        run_workers(dispatch, [("slow",), (work.__name__, *args)], on_record=got.append, poll=0.1)
    assert time.monotonic() - t < 8
    assert len([r for r in got if not r.get("slow")]) == 2


def dispatch(name, *args):
    """One fake worker by name: "slow" is a healthy worker of 60 one-second games."""
    if name == "slow":
        for r in finishes(60, 1.0):
            yield {**r, "slow": True}
        return
    yield from globals()[name](*args)
