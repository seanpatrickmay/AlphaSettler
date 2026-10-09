import random

import pytest

from alphasettler import Game
from alphasettler._engine import resolve_config

UNLIMITED = {"bank_limit": False}


def test_resolved_configs_name_the_bank_limit_only_when_it_is_off():
    assert "bank_limit" not in resolve_config({})
    assert "bank_limit" not in resolve_config({"bank_limit": True})
    assert resolve_config(UNLIMITED)["bank_limit"] is False
    assert resolve_config(UNLIMITED) == {**resolve_config({}), "bank_limit": False}


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_an_unlimited_game_shows_19_less_the_cards_in_hands(seed):
    g, rng = Game(seed, UNLIMITED), random.Random(seed)
    assert g.observation(0)["config"]["bank_limit"] is False
    while not g.is_over:
        g.apply(rng.choice(g.legal_actions()))
        s = g.snapshot()
        held = [sum(p["hand"][r] for p in s["players"]) for r in range(5)]
        assert s["bank"] == [max(0, 19 - h) for h in held]


def test_a_position_with_20_cards_of_a_resource_imports_only_without_the_limit():
    snap = Game(4).snapshot()  # the opening position, with player 0 given twenty wood
    snap["players"][0]["hand"] = [20, 0, 0, 0, 0]
    snap["bank"] = [0, 19, 19, 19, 19]
    Game.from_snapshot(4, snap, UNLIMITED)
    with pytest.raises(ValueError, match="hands and bank hold"):
        Game.from_snapshot(4, snap, {})
