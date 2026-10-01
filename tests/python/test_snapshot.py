import copy
import random

import pytest

from alphasettler import Bot, Game

NO_TRADES = {"max_offers_per_turn": 0}
TRADE_PHASES = {"trade_response", "trade_confirm"}


def play(g, steps, seed=0):
    rng = random.Random(seed)
    for _ in range(steps):
        if g.is_over:
            return
        g.apply(rng.choice(g.legal_actions()))


def test_snapshot_shape_at_start():
    snap = Game(1).snapshot()
    assert snap["phase"] == "setup_settlement" and snap["setup_step"] == 0
    assert len(snap["players"]) == 4 and snap["players"][0]["hand"] == [0] * 5
    assert len(snap["dev_deck"]) == 25 and snap["bank"] == [19] * 5
    assert set(snap["board"]) == {"tile_resource", "tile_number", "ports"}
    assert snap["winner"] is None and snap["return_phase"] == "main"
    assert snap["offers_this_turn"] == 0 and len(snap) == 17


@pytest.mark.parametrize("config", [NO_TRADES, None], ids=["no_trades", "default"])
def test_snapshot_round_trip_mid_game(config):
    offers_seen = 0
    for seed in range(5):
        g = Game(seed, config)
        play(g, 150 + 97 * seed, seed)
        # Snapshots cannot hold a pending domestic trade; play on until it resolves.
        rng = random.Random(seed)
        while g.phase in TRADE_PHASES:
            g.apply(rng.choice(g.legal_actions()))
        snap = g.snapshot()
        offers_seen += snap["offers_this_turn"]
        h = Game.from_snapshot(seed, snap, config)
        assert h.snapshot() == snap
        assert h.legal_actions() == g.legal_actions()
        assert h.observation(2) == g.observation(2)
    if config is NO_TRADES:
        assert offers_seen == 0


def test_round_trip_keeps_offers_made_this_turn():
    # Find a position after a resolved domestic offer, so offers_this_turn > 0.
    for seed in range(50):
        g = Game(seed)
        rng = random.Random(seed)
        while not g.is_over:
            snap = g.snapshot()
            if snap["offers_this_turn"] > 0 and g.phase not in TRADE_PHASES:
                h = Game.from_snapshot(seed, snap)
                assert h.snapshot() == snap
                assert h.legal_actions() == g.legal_actions()
                return
            g.apply(rng.choice(g.legal_actions()))
    pytest.fail("no position with an offer made this turn")


def test_observation_has_return_phase():
    assert Game(1).observation(0)["return_phase"] == "main"


def test_copy_is_independent():
    g = Game(3)
    play(g, 12, 3)
    log, snap = g.log(0), g.snapshot()
    assert log
    h = g.copy()
    assert h.log(0) == log and h.snapshot() == snap
    play(h, 20, 4)
    assert h.snapshot() != snap
    assert g.log(0) == log and g.snapshot() == snap
    for c in (copy.copy(g), copy.deepcopy(g)):
        assert c.snapshot() == snap and c.log(0) == log
        play(c, 5, 5)
        assert g.snapshot() == snap


@pytest.mark.parametrize(
    "mutate, needle",
    [
        (lambda s: s["bank"].__setitem__(0, 20), "resource 0"),
        (lambda s: s.__setitem__("phase", "trade_response"), "pending domestic trade"),
        (lambda s: s.__setitem__("phase", "nonsense"), "unknown phase"),
        (lambda s: s["players"][0]["roads"].append(999), "edge"),
        (lambda s: s["players"][0]["settlements"].extend([3, 3]), "twice"),
        (lambda s: s.__setitem__("extra", 1), "unknown snapshot key"),
        (lambda s: s["players"][0].__setitem__("extra", 1), "unknown player 0 key"),
        (lambda s: s.pop("robber"), 'missing "robber"'),
        (lambda s: s.pop("winner"), 'missing "winner"'),
        (lambda s: s.pop("offers_this_turn"), 'missing "offers_this_turn"'),
        (
            lambda s: s.__setitem__("roads_left", 1),
            "roads_left must be None unless phase is road_building",
        ),
        (
            lambda s: s.__setitem__("setup_road_node", 0),
            "setup_road_node must be None unless phase is setup_road",
        ),
        (lambda s: s.__setitem__("winner", 0), "winner must be None unless phase is game_over"),
        (
            lambda s: s.update(phase="road_building", setup_step=8),
            "roads_left is required when phase is road_building",
        ),
        (lambda s: s.__setitem__("offers_this_turn", 4), "offers_this_turn 4 exceeds"),
        (lambda s: s["players"].pop(), "players"),
        (lambda s: s["dev_deck"].__setitem__(0, "joker"), "dev card"),
        (lambda s: s["board"]["tile_resource"].__setitem__(0, "gold"), "unknown resource"),
        (lambda s: s.__setitem__("robber", -1), "robber -1 out of range"),
    ],
)
def test_bad_snapshots_raise_value_error(mutate, needle):
    snap = Game(2).snapshot()
    mutate(snap)
    with pytest.raises(ValueError, match=needle):
        Game.from_snapshot(2, snap)


def test_bools_are_not_ints():
    snap = Game(2).snapshot()
    snap["robber"] = True
    with pytest.raises(TypeError, match="robber must be an int, not bool"):
        Game.from_snapshot(2, snap)
    with pytest.raises(TypeError, match="not bool"):
        Game(True)
    with pytest.raises(TypeError, match="not bool"):
        Bot("random", False)


def test_bot_plays_a_full_game():
    g = Game(5, NO_TRADES)
    bots = [Bot("greedy", i) for i in range(4)]
    while not g.is_over:
        a = bots[g.current_actor].act(g)
        assert a in g.legal_actions()
        g.apply(a)
    assert g.phase == "game_over"
    with pytest.raises(ValueError, match="over"):
        bots[0].act(g)


def test_bot_names_and_errors():
    assert Bot("random", 1).name == "random"
    with pytest.raises(ValueError, match="unknown bot"):
        Bot("nope", 0)
