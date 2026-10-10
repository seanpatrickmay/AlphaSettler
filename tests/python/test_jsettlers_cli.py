import json

import pytest

from alphasettler import cli
from jsettlers_harness import require
from oracle.jsettlers.arena import COUNTERS, RECORD_KEYS


def test_arena_plays_four_games_a_seed_and_reports_the_counters(tmp_path, capsys):
    require()
    out = tmp_path / "js.jsonl"
    rc = cli.main(["arena", "--candidate", "ismcts@20", "--baseline", "jsettlers:smart", "--seeds", "1",
                   "--seed-start", "71", "--threads", "2", "--out", str(out)])
    text = capsys.readouterr().out
    assert rc == 0
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert [r["candidate_seat"] for r in rows] == [0, 1, 2, 3]
    assert all(r["candidate"] == "ismcts@20" and r["baseline"] == "jsettlers:smart" and r["arm"] == "no-trades"
               for r in rows)
    for line in ("ismcts@20 vs jsettlers:smart (no-trades): win rate", "dropped (robot desync in setup):",
                 "belief resets: 0", "fallbacks:", "trades: our_offers", "bank limit hits:",
                 "aborted at the turn cap:", f"wrote {out}"):
        assert line in text, line


@pytest.mark.parametrize("args,message", [
    (["--baseline", "jsettlers:fast"], "unknown JSettlers robot"),
    (["--baseline", "jsettlers:smart", "--aligned"], "--aligned needs --trades"),
    (["--baseline", "jsettlers:smart", "--native-boards"], "--native-boards needs --trades"),
    (["--baseline", "random", "--aligned", "--trades"], "only with a jsettlers: baseline"),
    (["--baseline", "random", "--native-boards"], "only with a jsettlers: baseline"),
])
def test_bad_requests_fail_before_any_game(args, message, tmp_path, capsys):
    require()
    out = tmp_path / "x.jsonl"
    rc = cli.main(["arena", "--candidate", "ismcts@20", "--seeds", "1", "--out", str(out), *args])
    assert rc == 2 and message in capsys.readouterr().err and not out.exists()


def test_a_missing_harness_is_named(monkeypatch, tmp_path, capsys):
    import oracle.jsettlers as js

    monkeypatch.setattr(js, "JAR", tmp_path / "absent.jar")
    rc = cli.main(["arena", "--candidate", "ismcts@20", "--baseline", "jsettlers:smart", "--seeds", "1",
                   "--out", str(tmp_path / "y.jsonl")])
    assert rc == 2 and "scripts/fetch-jsettlers.sh" in capsys.readouterr().err


def record(seed, seat, winner, abort_reason=None, turns=100, last_build=(90, 95, 80, None)):
    r = {"seed": seed, "candidate_seat": seat, "winner": winner, "vp": [10, 4, 5, 6], "turns": turns, "rounds": 25,
         "aborted": abort_reason is not None, "abort_reason": abort_reason,
         "robot_desync": abort_reason == "robot-desync", "last_build_turn": list(last_build), "fallbacks": 0,
         "fallback_reasons": {}, "belief_resets": 0, "bank_limit_hits": 0, "arm": "no-trades"}
    r.update({c: 0 for c in COUNTERS})
    assert set(r) == set(RECORD_KEYS)
    return r


def test_a_desync_seed_is_left_out_of_the_win_rate_and_counted(capsys):
    # seed 0: the candidate wins in seats 0 and 1 (rate 0.5). seed 1: seat 2's game desynced in setup (no winner, which
    # would count as a loss in its seed's rate). seed 2: one win, in seat 0 (0.25). Kept: seeds 0 and 2, 0.375.
    seed0 = [record(0, k, 0 if k == 0 else 1) for k in range(4)]
    seed1 = [record(1, 0, 1), record(1, 1, 0), record(1, 2, None, "robot-desync", turns=3, last_build=(None,) * 4),
             record(1, 3, 0)]
    seed2 = [record(2, k, 0) for k in range(4)]
    code = cli._print_jsettlers_summary("ismcts@20", "jsettlers:smart", seed0 + seed1 + seed2, COUNTERS)
    text = capsys.readouterr().out
    assert code == 0
    assert "win rate 0.375" in text and "over 2 seeds (8 games, 3 wins, 0 draws)" in text
    assert "dropped (robot desync in setup): 1 games, 1 seeds left out of the summary" in text
    assert "aborted at the turn cap: 0 (mean turns since the last build: n/a)" in text


def test_the_turn_cap_games_stay_in_as_losses_and_report_the_build_stall(capsys):
    games = [record(0, k, None, "turn-cap", turns=1000, last_build=(700, 400, None, 800)) for k in range(3)]
    games.append(record(0, 3, 3))
    code = cli._print_jsettlers_summary("ismcts@20", "jsettlers:smart", games, COUNTERS)
    text = capsys.readouterr().out
    assert code == 0
    assert "over 1 seeds (4 games, 1 wins, 3 draws)" in text
    assert "dropped (robot desync in setup): 0 games, 0 seeds" in text
    assert "aborted at the turn cap: 3 (mean turns since the last build: 200.0)" in text


def test_every_seed_desynced_is_no_summary_and_exit_1(capsys):
    games = [record(0, k, None, "robot-desync", turns=3, last_build=(None,) * 4) for k in range(4)]
    assert cli._print_jsettlers_summary("ismcts@20", "jsettlers:smart", games, COUNTERS) == 1
    text = capsys.readouterr().out
    assert "no seed left to summarize" in text and "dropped (robot desync in setup): 4 games" in text
