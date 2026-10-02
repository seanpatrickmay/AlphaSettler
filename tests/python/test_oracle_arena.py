import os
import subprocess
import sys

import pytest

pytest.importorskip("catanatron")

from catanatron import Color  # noqa: E402

from alphasettler.stats import summarize  # noqa: E402
from oracle.arena import AlphaSettlerPlayer, _force_seating, run_match  # noqa: E402
from oracle.translate import Untranslatable, action_token  # noqa: E402


def test_records_have_the_native_shape():
    recs = run_match("random", "random", seeds=2, workers=2)
    assert len(recs) == 8
    assert [r["candidate_seat"] for r in recs[:4]] == [0, 1, 2, 3]
    assert {"seed", "candidate_seat", "winner", "vp", "turns", "actions", "fallbacks", "belief_resets"} <= set(recs[0])
    for r in recs:
        if r["winner"] is not None:
            assert r["vp"][r["winner"]] >= 10


def test_greedy_beats_catanatron_random():
    recs = run_match("greedy", "random", seeds=40)
    s = summarize(recs)
    assert s.win_rate > 0.9, s
    assert s.z_vs_null > 3, s
    assert sum(r["fallbacks"] for r in recs) == 0


@pytest.mark.slow
def test_alphabeta_smoke():
    recs = run_match("greedy", "alphabeta", seeds=1, workers=4)
    assert len(recs) == 4 and all(r["turns"] > 0 for r in recs)


def test_unknown_names():
    with pytest.raises(ValueError, match="unknown Catanatron bot"):
        run_match("greedy", "nope", seeds=1)
    with pytest.raises(ValueError, match="unknown bot"):
        run_match("nope", "random", seeds=1)


class StubBot:
    """Returns a fixed action id, or raises a fixed exception."""

    def __init__(self, result):
        self.result = result

    def act(self, game):
        if isinstance(self.result, BaseException):
            raise self.result
        return self.result


def _setup_position(stub_result):
    """A fresh game with `p` forced into seat 0, so `p` is the player to act."""
    from catanatron import Game as CatanGame, RandomPlayer

    p = AlphaSettlerPlayer(Color.RED, "random", 1)
    p.bot = StubBot(stub_result)
    players = [p] + [RandomPlayer(c) for c in (Color.BLUE, Color.ORANGE, Color.WHITE)]
    game = CatanGame(players, seed=3)
    _force_seating(game, players)
    assert game.state.current_color() == Color.RED
    return p, game


def test_offered_choice_is_played_without_fallback():
    p, game = _setup_position(None)
    target = game.playable_actions[-1]  # not playable_actions[0], so a fallback would show
    p.bot.result = action_token(game.state, target)
    assert p.decide(game, game.playable_actions) == target and p.fallbacks == 0


def test_fallback_is_counted_not_fatal():
    p, game = _setup_position(1)  # EndTurn: never offered during setup, a by_token miss
    assert p.decide(game, game.playable_actions) == game.playable_actions[0] and p.fallbacks == 1


class PanicException(BaseException):
    """Stands in for PyO3's PanicException, which derives from BaseException, not Exception."""


@pytest.mark.parametrize("error", [ValueError("no import"), KeyError("node"), Untranslatable("kind", "detail"),
                                   PanicException("engine panic")])
def test_import_or_act_errors_are_counted_fallbacks(error):
    p, game = _setup_position(error)
    assert p.decide(game, game.playable_actions) == game.playable_actions[0] and p.fallbacks == 1


@pytest.mark.parametrize("error", [KeyboardInterrupt(), SystemExit(1)])
def test_interrupts_are_not_fallbacks(error):
    p, game = _setup_position(error)
    with pytest.raises(type(error)):
        p.decide(game, game.playable_actions)


def test_run_match_leaves_the_parent_environment_alone(monkeypatch):
    monkeypatch.delenv("PYTHONHASHSEED", raising=False)
    run_match("random", "random", seeds=1, workers=1)
    assert "PYTHONHASHSEED" not in os.environ
    monkeypatch.setenv("PYTHONHASHSEED", "7")
    run_match("random", "random", seeds=1, workers=1)
    assert os.environ["PYTHONHASHSEED"] == "7"


def test_cli_catanatron_baseline(tmp_path):
    out = tmp_path / "c.jsonl"
    r = subprocess.run([sys.executable, "-m", "alphasettler", "arena", "--candidate", "greedy",
                        "--baseline", "catanatron:random", "--seeds", "2", "--out", str(out)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "greedy vs catanatron:random: win rate" in r.stdout and "fallbacks: 0" in r.stdout
    assert len(out.read_text().splitlines()) == 8
    r = subprocess.run([sys.executable, "-m", "alphasettler", "arena", "--candidate", "greedy",
                        "--baseline", "catanatron:nope", "--seeds", "1"], capture_output=True, text=True)
    assert r.returncode == 2 and "unknown Catanatron bot" in r.stderr
