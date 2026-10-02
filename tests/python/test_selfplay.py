import gzip
import json

import pytest

from alphasettler import Bot, Game, records
from alphasettler._engine import fit_heuristic, selfplay

CONFIG = {"max_offers_per_turn": 0}


def test_selfplay_games_have_searched_decisions():
    games = selfplay(0, 2, 10, 2, CONFIG)
    assert [g["seed"] for g in games] == [0, 1]
    for g in games:
        assert g["decisions"]
        assert g["belief_resets"] == 0
        for d in g["decisions"]:
            assert g["actions"][d["index"]] in d["legal"]
            assert len(d["visits"]) == len(d["legal"]) and sum(d["visits"]) == 9
            assert d["full_search"] is True
            assert d["observation"]["viewer"] == d["actor"]


def test_records_round_trip_and_replay_exactly(tmp_path):
    path = tmp_path / "sp.jsonl.gz"
    games = selfplay(5, 2, 10, 2, CONFIG)
    records.write(path, games, CONFIG)
    back = list(records.read(path))
    assert [g["seed"] for g in back] == [5, 6]
    assert all(g["config"] == CONFIG for g in back)
    for g in back:
        assert records.replay_mismatches(g) == []
    with pytest.raises(FileExistsError):
        records.write(path, games, CONFIG)


def test_replay_detects_a_tampered_record(tmp_path):
    g = json.loads(json.dumps({"config": CONFIG, **selfplay(7, 1, 10, 1, CONFIG)[0]}))
    g["decisions"][0]["observation"]["my_hand"][0] += 1
    assert records.replay_mismatches(g)


def test_fit_heuristic_reports_weights_and_likelihoods():
    games = [{"config": CONFIG, **g} for g in selfplay(0, 4, 10, 4, CONFIG)]
    r = fit_heuristic(games)
    assert len(r["weights"]) == 8 and r["samples"] > 0
    assert r["log_likelihood_after"] >= r["log_likelihood_before"]


def test_fit_heuristic_arguments_are_checked():
    games = [{"config": CONFIG, **g} for g in selfplay(0, 1, 10, 1, CONFIG)]
    for bad in [dict(l2=-0.1), dict(l2=float("nan")), dict(l2=float("inf")), dict(iterations=-1),
                dict(iterations=2**40)]:
        with pytest.raises(ValueError):
            fit_heuristic(games, **bad)
    with pytest.raises(ValueError, match="no samples"):
        fit_heuristic([{**games[0], "winner": None}])
    with pytest.raises(ValueError, match="no samples"):
        fit_heuristic([])


def test_records_writer_streams_games(tmp_path):
    path = tmp_path / "w.jsonl.gz"
    games = selfplay(0, 2, 10, 2, CONFIG)
    with records.open_writer(path) as w:
        w.write_game(games[0], CONFIG)
        w.flush()
        w.write_game(games[1], CONFIG)
    back = list(records.read(path))
    assert [g["seed"] for g in back] == [0, 1]
    assert all(records.replay_mismatches(g) == [] for g in back)
    with pytest.raises(FileExistsError):
        with records.open_writer(path):
            pass


def test_selfplay_arguments_are_checked():
    for bad in [dict(simulations=0), dict(simulations=1), dict(simulations=1_000_001), dict(rollout=201),
                dict(games=0)]:
        args = {"seed_start": 0, "games": 1, "simulations": 5, "threads": 1, "config": CONFIG, "rollout": 0, **bad}
        with pytest.raises(ValueError):
            selfplay(**args)


def test_ismcts_bot_from_python_observes_and_reports():
    g = Game(3, CONFIG)
    bot = Bot("ismcts@10", 1)
    seen = 0
    while not g.is_over and g.turn < 3:
        actor = g.current_actor
        if actor == 0:
            log = g.log(0)
            bot.observe(0, log[seen:])
            seen = len(log)
            a = bot.act(g)
        else:
            a = g.legal_actions()[0]
        g.apply(a)
    d = bot.diagnostics()
    assert d["belief_resets"] == 0 and d["searches"] > 0


def test_a_reused_bot_survives_a_new_game():
    bot = Bot("ismcts@10", 2)
    for seed in (1, 2):
        g = Game(seed, CONFIG)
        for _ in range(60):
            if g.is_over:
                break
            a = bot.act(g)
            assert a in g.legal_actions()
            g.apply(a)
    assert bot.diagnostics()["belief_resets"] >= 1


def test_bad_ismcts_names():
    for bad in ["ismcts@0", "ismcts@abc", "ismcts@300+r0"]:
        with pytest.raises(ValueError, match="unknown bot"):
            Bot(bad, 0)
