import pytest

catanatron = pytest.importorskip("catanatron")

from catanatron import Color, Game as CatanGame, RandomPlayer  # noqa: E402
from catanatron.models.enums import ActionType  # noqa: E402
from catanatron.models.map import PORT_DIRECTION_TO_NODEREFS, build_map  # noqa: E402

from alphasettler import Game, action_space_size, describe_action  # noqa: E402
from oracle import tables  # noqa: E402
from oracle.translate import (  # noqa: E402
    BUY_DEV, END_TURN, PLAY_KNIGHT, PLAY_ROAD_BUILDING, ROLL, build_city, build_road, build_settlement,
    CUBE_TO_OUR_TILE, catan_snapshot, catan_tokens, compare, discard, edge_nodes, maritime, move_robber,
    our_steps, our_tokens, play_monopoly, play_year_of_plenty, steal_from,
)

COLORS = [Color.RED, Color.BLUE, Color.ORANGE, Color.WHITE]
CONFIG = {"max_offers_per_turn": 0, "catanatron_compat": False, "max_turns": 994}


def new_catan(seed):
    return CatanGame([RandomPlayer(c) for c in COLORS], seed=seed)


def test_python_action_ids_match_the_engine():
    expected = {}
    expected.update({ROLL: "Roll", END_TURN: "EndTurn", BUY_DEV: "BuyDev",
                     PLAY_KNIGHT: "PlayKnight", PLAY_ROAD_BUILDING: "PlayRoadBuilding"})
    names = ["Wood", "Brick", "Sheep", "Wheat", "Ore"]
    for r in range(5):
        expected[play_monopoly(r)] = f"PlayMonopoly({names[r]})"
        expected[discard(r)] = f"Discard({names[r]})"
        for g in range(5):
            if g != r:
                expected[maritime(r, g)] = f"MaritimeTrade {{ give: {names[r]}, get: {names[g]} }}"
        for b in range(r, 5):
            expected[play_year_of_plenty(r, b)] = f"PlayYearOfPlenty({names[r]}, {names[b]})"
    for n in range(54):
        expected[build_settlement(n)] = f"BuildSettlement({n})"
        expected[build_city(n)] = f"BuildCity({n})"
    for e in range(72):
        expected[build_road(e)] = f"BuildRoad({e})"
    for t in range(19):
        expected[move_robber(t)] = f"MoveRobber({t})"
    for p in range(4):
        expected[steal_from(p)] = f"StealFrom({p})"
    for a, text in expected.items():
        assert 0 <= a < action_space_size()
        assert describe_action(a) == text


def test_tables_are_bijections():
    assert sorted(tables.CAT_NODE_TO_OUR_NODE) == list(range(54))
    assert [tables.CAT_NODE_TO_OUR_NODE[c] for c in tables.OUR_NODE_TO_CAT_NODE] == list(range(54))
    assert len(set(tables.OUR_EDGE_TO_CAT_EDGE)) == 72
    assert len(set(tables.OUR_TILE_TO_CAT_CUBE)) == 19
    assert len(set(tables.CAT_PORT_ID_TO_OUR_EDGE)) == 9


@pytest.mark.parametrize("build", range(3))
def test_land_tile_ids_match_catanatron(build):
    # Each build shuffles resources and numbers; the tile ids belong to positions.
    for cube, tile in build_map("BASE").land_tiles.items():
        assert tables.CAT_LAND_TILE_ID_TO_OUR_TILE[tile.id] == CUBE_TO_OUR_TILE[cube], (tile.id, cube)


def test_port_edges_match_catanatron_port_nodes():
    m = build_map("BASE")
    by_resource = {}
    for port in m.ports_by_id.values():
        a, b = PORT_DIRECTION_TO_NODEREFS[port.direction]  # how Catanatron fills port_nodes
        theirs = {port.nodes[a], port.nodes[b]}
        ours = {tables.OUR_NODE_TO_CAT_NODE[n] for n in edge_nodes(tables.CAT_PORT_ID_TO_OUR_EDGE[port.id])}
        assert ours == theirs, port.id
        by_resource.setdefault(port.resource, set()).update(ours)
    assert by_resource == dict(m.port_nodes)


@pytest.mark.parametrize("seed", range(10))
def test_initial_position_imports_and_matches(seed):
    cat = new_catan(seed)
    theirs = catan_snapshot(cat)
    ours = Game.from_snapshot(seed, theirs, CONFIG)
    assert compare(ours.snapshot(), theirs) == []
    assert our_tokens(ours) == catan_tokens(cat.state, cat.playable_actions)


def test_forced_outcomes_translate():
    # Play a Catanatron game to completion; every executed record must translate and the
    # translated position must import cleanly at every step.
    cat = new_catan(7)
    while cat.winning_color() is None and cat.state.num_turns < 1000:
        cat.play_tick()
        Game.from_snapshot(7, catan_snapshot(cat), CONFIG)
    assert any(r.action.action_type.name == "ROLL" for r in cat.state.action_records)


class SortedRandom(RandomPlayer):
    """Random play that does not depend on PYTHONHASHSEED: Catanatron orders playable_actions by
    set iteration, so choose from a sorted list. ``PREFER`` lists action types taken first."""

    PREFER = ()

    def decide(self, game, playable_actions):
        playable = sorted(playable_actions, key=repr)
        for kind in self.PREFER:
            ok = [a for a in playable if a.action_type == kind]
            if ok:
                return game.state.random.choice(ok)
        return game.state.random.choice(playable)


class DevHungry(SortedRandom):
    PREFER = (ActionType.PLAY_KNIGHT_CARD, ActionType.PLAY_MONOPOLY, ActionType.PLAY_YEAR_OF_PLENTY,
              ActionType.PLAY_ROAD_BUILDING, ActionType.BUY_DEVELOPMENT_CARD)


def short_free_roads(st):
    # Catanatron grants 2 free roads even when the player has 1 road piece left.
    return st.is_road_building and st.free_roads_available > st.player_state[
        f"P{st.current_turn_index}_ROADS_AVAILABLE"]


def bought_after_playing(st):
    # The only card of a kind was just bought, yet {DEV}_OWNED_AT_START says it was held at the
    # start of the turn: the held one was played earlier this turn.
    record = st.action_records[-1]
    if record.action.action_type != ActionType.BUY_DEVELOPMENT_CARD:
        return False
    key = f"P{st.current_turn_index}_{record.result}"
    return st.player_state.get(f"{key}_OWNED_AT_START", False) and st.player_state[f"{key}_IN_HAND"] == 1


@pytest.mark.parametrize("player, seed, edge_case", [
    (SortedRandom, 8, short_free_roads),
    (DevHungry, 1, bought_after_playing),
], ids=["short_free_roads", "bought_after_playing"])
def test_lockstep_replay_through_translation_edge_cases(player, seed, edge_case):
    # Replays a whole Catanatron game in our engine through our_steps. These seeds reach the
    # edge case and hit none of the known rule differences, so every step matches exactly.
    cat = CatanGame([player(c) for c in COLORS], seed=seed)
    ours = Game.from_snapshot(seed, catan_snapshot(cat), CONFIG)
    hits = 0
    while cat.winning_color() is None and cat.state.num_turns < 1000:
        assert our_tokens(ours) == catan_tokens(cat.state, cat.playable_actions)
        cat.play_tick()
        st = cat.state
        for action, chance in our_steps(st, st.action_records[-1]):
            if chance is None:
                ours.apply(action)
            else:
                ours.apply_forced(action, chance)
        theirs, mine = catan_snapshot(cat), ours.snapshot()
        assert compare(mine, theirs) == []
        assert [p["dev_new"] for p in mine["players"]] == [p["dev_new"] for p in theirs["players"]]
        Game.from_snapshot(seed, theirs, CONFIG)
        hits += bool(edge_case(st))
    assert hits > 0
    assert ours.is_over
