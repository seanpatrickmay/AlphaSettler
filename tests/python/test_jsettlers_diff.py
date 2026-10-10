import json

import pytest

import oracle.jsettlers.jvm as jvm_module
from alphasettler import cli
from jsettlers_harness import require
from oracle.jsettlers import events
from oracle.jsettlers.allowlist import ENTRIES
from oracle.jsettlers.diff import GameResult, compare, format_summary, play_game, summarize
from oracle.jsettlers.jvm import Jvm


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
    assert r.mismatch is not None and r.mismatch.startswith("step ")


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
