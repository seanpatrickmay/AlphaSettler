import random
import threading
import time

import pytest

from alphasettler import Game, action_space_size, bot_names, describe_action, run_match


def play_out(g: Game, seed: int = 0) -> None:
    rng = random.Random(seed)
    while not g.is_over:
        g.apply(rng.choice(g.legal_actions()))


def test_new_game_is_in_setup():
    g = Game(1)
    assert g.phase == "setup_settlement"
    assert g.current_actor == 0
    assert len(g.legal_actions()) == 54
    assert g.seed == 1


def test_actions_and_helpers():
    assert action_space_size() == 665
    assert describe_action(0) == "Roll"
    assert describe_action(25) == "BuildSettlement(0)"
    assert set(bot_names()) >= {"random", "greedy"}


def test_illegal_actions_raise_and_change_nothing():
    g = Game(1)
    before = g.legal_actions()
    with pytest.raises(ValueError, match="Illegal"):
        g.apply(0)  # Roll during setup
    with pytest.raises(ValueError):
        g.apply(665)
    with pytest.raises(ValueError):
        g.apply(233)  # undecodable: maritime give == get
    assert g.legal_actions() == before
    assert g.log(0) == []


def test_bad_inputs_raise_value_error():
    with pytest.raises(ValueError, match="unknown config key"):
        Game(1, {"nope": 1})
    with pytest.raises(ValueError):
        Game(1, {"max_trade_cards": 3})
    g = Game(1)
    with pytest.raises(ValueError):
        g.observation(4)
    with pytest.raises(ValueError):
        g.final_vp()
    with pytest.raises(ValueError, match="unknown bot"):
        run_match("nope", "random", 0, 1, 1)


def test_out_of_range_ints_raise_value_error():
    g = Game(1)
    for bad in [
        lambda: g.apply(-1),
        lambda: g.apply(2**70),
        lambda: describe_action(-1),
        lambda: g.observation(-1),
        lambda: g.observation(256),
        lambda: g.log(-1),
        lambda: Game(-1),
        lambda: Game(2**64),
        lambda: Game(1, {"vp_to_win": 300}),
        lambda: Game(1, {"max_turns": -1}),
        lambda: run_match("random", "random", -1, 1, 1),
        lambda: run_match("random", "random", 0, -1, 1),
        lambda: run_match("random", "random", 0, 1, -1),
    ]:
        with pytest.raises(ValueError):
            bad()
    with pytest.raises(TypeError):
        g.apply("0")  # a wrong type stays a TypeError
    assert g.legal_actions() == Game(1).legal_actions()


def test_full_u64_seed_range_and_seed_overflow():
    assert Game(2**64 - 1).seed == 2**64 - 1
    assert run_match("random", "random", 2**64 - 1, 0, 1) == []
    with pytest.raises(ValueError, match="seed"):
        run_match("random", "random", 2**64 - 1, 2, 1)


def test_huge_seed_count_is_a_value_error_not_a_crash():
    with pytest.raises(ValueError, match="at most 10000000 seeds per match"):
        run_match("random", "random", 0, 2**61, 1)


def pre_roll_game(config=None) -> Game:
    g = Game(2, config)
    while g.phase in ("setup_settlement", "setup_road"):
        g.apply(g.legal_actions()[0])
    assert g.phase == "pre_roll"
    return g


def test_malformed_chance_raises_value_error():
    g = pre_roll_game()
    for bad in [
        {"roll": [300, 1]},
        {"roll": [-1, 4]},
        {"roll": [3]},
        {"roll": [3, 4], "extra": 1},
        {"roll": [3, 4], "steal": "wood"},
        {"steal": "wood", "dev": "knight"},
        {"discards": [[0] * 5] * 4},
        {"steal": "wood", "discards": None},
        {"roll": [3, 4], "discards": [[0] * 5] * 3},
        {"roll": [3, 4], "discards": [[0] * 5, [0] * 5, [0] * 5, [0, 0, 0, 0, 300]]},
        {},
    ]:
        with pytest.raises(ValueError):
            g.apply_forced(0, bad)
    assert g.phase == "pre_roll" and g.log(0)[-1]["type"] != "rolled"
    g.apply_forced(0, {"roll": (3, 4), "discards": None})  # tuples and explicit None are fine
    assert g.phase in ("move_robber", "discard")


def test_log_hides_stolen_resource_and_bought_card_from_third_parties():
    rng = random.Random(7)
    checked = set()
    for seed in range(20):
        g = Game(seed)
        while not g.is_over and len(checked) < 2:
            g.apply(rng.choice(g.legal_actions()))
            logs = [g.log(p) for p in range(4)]
            e = logs[0][-1] if logs[0] else None
            if e is None:
                continue
            if e["type"] == "bought_dev":
                buyer = e["player"]
                other = (buyer + 1) % 4
                assert logs[buyer][-1]["card"] is not None
                assert logs[other][-1]["card"] is None
                checked.add("bought_dev")
            elif e["type"] == "stole" and logs[e["thief"]][-1]["resource"] is not None:
                thief, victim = e["thief"], e["victim"]
                other = next(p for p in range(4) if p not in (thief, victim))
                assert logs[victim][-1]["resource"] == logs[thief][-1]["resource"]
                assert logs[other][-1]["resource"] is None
                checked.add("stole")
        if len(checked) == 2:
            break
    assert checked == {"bought_dev", "stole"}


def test_full_game_and_final_vp():
    g = Game(3, {"max_offers_per_turn": 0})
    play_out(g)
    assert g.phase == "game_over"
    vp = g.final_vp()
    assert len(vp) == 4
    if g.winner is not None:
        assert vp[g.winner] >= 10


def test_observation_shape_and_privacy():
    g = Game(5)
    play_out_steps = 0
    rng = random.Random(1)
    while play_out_steps < 400 and not g.is_over:
        g.apply(rng.choice(g.legal_actions()))
        play_out_steps += 1
    o = g.observation(0)
    for key in ["phase", "my_hand", "hand_counts", "settlements", "roads", "public_vp", "board", "bank", "config"]:
        assert key in o
    assert isinstance(o["my_hand"], list) and len(o["my_hand"]) == 5
    assert isinstance(o["hand_counts"], list) and len(o["hand_counts"]) == 4
    assert sum(o["my_hand"]) == o["hand_counts"][0]
    assert "hand" not in o  # no opponent hands
    assert len(o["board"]["tile_resource"]) == 19
    assert o["config"]["vp_to_win"] == 10


def test_event_log_types():
    g = Game(1)
    g.apply(g.legal_actions()[0])
    log = g.log(0)
    assert log[0]["type"] == "built_settlement"
    assert log[0]["player"] == 0


def test_forced_roll_and_impossible_chance():
    g = Game(2)
    while g.phase in ("setup_settlement", "setup_road"):
        g.apply(g.legal_actions()[0])
    assert g.phase == "pre_roll"
    with pytest.raises(ValueError, match="ImpossibleChance"):
        g.apply_forced(0, {"roll": [0, 7]})
    g.apply_forced(0, {"roll": [3, 4]})
    assert g.phase in ("move_robber", "discard")
    rolled = [e for e in g.log(1) if e["type"] == "rolled"]
    assert rolled[-1]["dice"] == [3, 4]


def test_run_match_records():
    recs = run_match("greedy", "random", 0, 3, 2)
    assert len(recs) == 12
    assert [r["candidate_seat"] for r in recs[:4]] == [0, 1, 2, 3]
    assert set(recs[0]) == {"seed", "candidate_seat", "winner", "vp", "turns", "actions", "belief_resets"}
    assert isinstance(recs[0]["vp"], list) and len(recs[0]["vp"]) == 4


def test_run_match_is_deterministic_across_threads():
    assert run_match("greedy", "random", 10, 8, 1) == run_match("greedy", "random", 10, 8, 4)


def test_run_match_releases_the_gil():
    ticks = []
    done = threading.Event()

    def ticker():
        while not done.is_set():
            ticks.append(time.perf_counter())
            time.sleep(0.001)

    t = threading.Thread(target=ticker)
    t.start()
    start = time.perf_counter()
    run_match("greedy", "random", 0, 1500, 1)
    elapsed = time.perf_counter() - start
    done.set()
    t.join()
    during = [x for x in ticks if start < x < start + elapsed]
    # A held GIL lets the ticker in at most once or twice; a released one lets it tick every
    # ~1ms. Scale the bar with elapsed so a faster engine does not turn this into a false failure.
    assert elapsed > 0.01, "match too short to observe"
    expected = max(3, int(0.2 * elapsed / 0.001))
    assert len(during) >= expected, f"python thread starved: {len(during)} ticks in {elapsed:.3f}s"
