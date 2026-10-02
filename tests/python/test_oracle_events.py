import pytest

pytest.importorskip("catanatron")

from catanatron import Color, Game as CatanGame, RandomPlayer  # noqa: E402

from alphasettler import Game  # noqa: E402
from oracle.arena import run_match  # noqa: E402
from oracle.diff import config, run_game  # noqa: E402
from oracle.events import record_events, redact  # noqa: E402
from oracle.translate import catan_snapshot, our_steps  # noqa: E402

COLORS = [Color.RED, Color.BLUE, Color.ORANGE, Color.WHITE]


def _clean_seeds(n):
    """Seeds whose lockstep game needs no allowlist entry at all, so both logs must agree."""
    out = []
    seed = 0
    while len(out) < n:
        r = run_game(seed)
        if not r.allowlisted and r.mismatch is None:
            out.append(seed)
        seed += 1
    return out


def test_translated_events_equal_our_log_in_clean_games():
    # Classification and replay run in this process, under the same hash seed, so they agree.
    for seed in _clean_seeds(12):
        cat = CatanGame([RandomPlayer(c) for c in COLORS], seed=seed)
        ours = Game.from_snapshot(seed, catan_snapshot(cat), config())
        by_color = {p.color: p for p in cat.state.players}
        translated = []
        before = catan_snapshot(cat)
        while before["phase"] != "game_over":
            st = cat.state
            record = cat.execute(by_color[st.current_color()].decide(cat, cat.playable_actions))
            for a, chance in our_steps(cat.state, record):
                if chance is None:
                    ours.apply(a)
                else:
                    ours.apply_forced(a, chance)
            after = catan_snapshot(cat)
            translated += record_events(before, after, record, cat.state)
            before = after
        for viewer in range(4):
            assert ours.log(viewer) == [redact(e, viewer) for e in translated], f"seed {seed} viewer {viewer}"


def test_redaction():
    stole = {"type": "stole", "thief": 1, "victim": 2, "resource": "ore"}
    assert redact(stole, 1) == stole and redact(stole, 2) == stole
    assert redact(stole, 0)["resource"] is None
    bought = {"type": "bought_dev", "player": 3, "card": "knight"}
    assert redact(bought, 3) == bought and redact(bought, 0)["card"] is None


def test_ismcts_plays_inside_catanatron_without_fallbacks_or_resets():
    records = run_match("ismcts@10", "random", seeds=1, workers=4)
    assert len(records) == 4
    for r in records:
        assert r["fallbacks"] == 0, r
        assert r["belief_resets"] == 0, r
