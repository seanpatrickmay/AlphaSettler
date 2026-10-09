import json
import subprocess
from pathlib import Path

import pytest

from alphasettler import Game
from alphasettler._engine import net_adjacency
from jsettlers_harness import require
from oracle.jsettlers import java_command, missing
from oracle.jsettlers import tables as T
from oracle.jsettlers.actions import (
    PAIRS, bundle, bundle_index, decode, maritime, offer, offer_hands, redact, year_of_plenty,
)

PROBE = json.loads((Path(__file__).parent / "data" / "jsettlers_board.json").read_text())


def test_the_committed_probe_is_what_the_jar_prints():
    require("Probe")
    out = subprocess.run(java_command("Probe"), capture_output=True, text=True, check=True).stdout
    assert json.loads(out) == PROBE


def test_our_topology_rebuilt_in_python_is_the_engines():
    adj = net_adjacency()
    assert [sorted(ns) for ns in T.TILE_NODES] == [sorted(ns) for ns in adj["tile_nodes"]]
    assert [tuple(sorted(p)) for p in adj["edge_nodes"]] == list(T.EDGE_NODES)
    assert len(T.COASTAL_EDGES) == 30


def test_tiles_map_onto_the_land_hexes():
    assert sorted(T.OUR_TILE_TO_JS_HEX) == sorted(PROBE["land_hexes"])
    assert T.OUR_TILE_TO_JS_HEX[9] == 0x77  # our centre tile (q, r) = (0, 0)
    assert all(T.JS_HEX_TO_OUR_TILE[h] == t for t, h in enumerate(T.OUR_TILE_TO_JS_HEX))


def test_hex_corners_agree_in_order():
    corners = {row[0]: row[1:] for row in PROBE["hex_nodes"]}
    for t, h in enumerate(T.OUR_TILE_TO_JS_HEX):
        assert [T.OUR_NODE_TO_JS_NODE[n] for n in T.TILE_NODES[t]] == corners[h]


def test_nodes_and_edges_are_bijections_matching_the_probe():
    assert sorted(T.JS_NODE_TO_OUR_NODE.values()) == list(range(54))
    probe_edges = {e: {a, b} for e, a, b in PROBE["edges"]}
    assert set(T.OUR_EDGE_TO_JS_EDGE) == set(probe_edges)
    for ours, js in enumerate(T.OUR_EDGE_TO_JS_EDGE):
        a, b = T.EDGE_NODES[ours]
        assert probe_edges[js] == {T.OUR_NODE_TO_JS_NODE[a], T.OUR_NODE_TO_JS_NODE[b]}
        assert set(T.js_edge_nodes(js)) == probe_edges[js]
        assert T.JS_EDGE_TO_OUR_EDGE[js] == ours


def test_ports_sit_on_our_coast_with_the_probed_facing_and_hex_numbers():
    assert T.PORT_EDGES == PROBE["port_edges"]
    assert T.PORT_FACING == PROBE["port_facing"]
    assert T.PORT_HEXNUMS == PROBE["port_hexnums"]
    assert all(T.JS_EDGE_TO_OUR_EDGE[e] in T.COASTAL_EDGES for e in T.PORT_EDGES)
    assert len({T.JS_EDGE_TO_OUR_EDGE[e] for e in T.PORT_EDGES}) == 9


def test_hex_layout_indices_match_the_probe():
    assert sorted([h, n] for h, n in PROBE["hex_nums"]) == sorted([h, n] for n, h in enumerate(T.NUM_TO_HEX))
    assert all(T.NUM_TO_HEX[T.HEX_NUM[h]] == h for h in T.NUM_TO_HEX)


def test_codes_match_the_jar():
    res = PROBE["resources"]
    assert {res[k]: T.JS_RES[res[k]] for k in ("clay", "ore", "sheep", "wheat", "wood")} == {
        res["clay"]: 1, res["ore"]: 4, res["sheep"]: 2, res["wheat"]: 3, res["wood"]: 0}
    dev = PROBE["dev_cards"]
    assert T.JS_DEV_TO_OURS == {dev["knight"]: "knight", dev["roads"]: "road_building",
                                dev["disc"]: "year_of_plenty", dev["mono"]: "monopoly",
                                **{dev[k]: "victory_point" for k in ("cap", "market", "univ", "temple", "chapel")}}
    assert (T.ROAD, T.SETTLEMENT, T.CITY) == tuple(PROBE["pieces"][k] for k in ("road", "settlement", "city"))
    assert T.STATES == PROBE["states"]
    assert (T.WATER_HEX, T.DESERT_HEX, T.MISC_PORT) == (
        PROBE["hex_types"]["water"], PROBE["hex_types"]["desert"], PROBE["hex_types"]["misc_port"])
    assert T.SMART_STRATEGY == PROBE["smart_strategy"]
    assert T.ours_to_js5([1, 0, 0, 0, 0]) == [0, 0, 0, 0, 1]  # wood is JSettlers' fifth


@pytest.mark.parametrize("hand", [[1, 0, 0, 0, 0], [0, 2, 3, 0, 1], [4, 5, 6, 7, 8]])
def test_resource_sets_round_trip(hand):
    assert T.js5_to_ours(T.ours_to_js5(hand)) == hand


def test_action_ids_follow_the_engine():
    g = Game(3)
    assert len(g.legal_actions()) == 54 and min(g.legal_actions()) == 25  # setup settlements 25..78
    assert maritime(4, 0) == 253 and year_of_plenty(3, 1) == 10 + PAIRS.index((1, 3))
    for i in range(20):
        assert bundle_index(bundle(i)) == i
    a = offer([0, 1, 0, 0, 0], [0, 0, 0, 0, 1])
    assert a == 258 + 20 * 1 + 4 and offer_hands(a) == ([0, 1, 0, 0, 0], [0, 0, 0, 0, 1])
    assert offer([1, 1, 1, 0, 0], [1, 0, 0, 0, 0]) is None
    assert decode(660 + 2) == ("confirm", 2) and decode(205 + 7) == ("move_robber", 7)


def test_redact_hides_steals_and_draws_from_others():
    stole = {"type": "stole", "thief": 1, "victim": 2, "resource": "ore"}
    assert redact(stole, 0)["resource"] is None and redact(stole, 2) == stole
    bought = {"type": "bought_dev", "player": 3, "card": "knight"}
    assert redact(bought, 1)["card"] is None and redact(bought, 3) == bought


def test_missing_names_the_command_to_run(monkeypatch, tmp_path):
    import oracle.jsettlers as js

    monkeypatch.setattr(js, "JAR", tmp_path / "absent.jar")
    assert "scripts/fetch-jsettlers.sh" in missing("Probe")
