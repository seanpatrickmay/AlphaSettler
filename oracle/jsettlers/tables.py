"""Coordinates and codes between JSettlers2 2.6.10's classic 4-player board and AlphaSettler.

Derived by geometry and checked against the probe of the jar (tests/python/data/jsettlers_board.json,
printed by jsettlers/src/alphasettler/jsettlers/Probe.java): tiles, nodes and edges are bijections.
Both engines number a hex's corners clockwise from the top. JSettlers' hex 0xRC has row R and column
C on diagonal axes, so our axial tile (q, r) is the hex ((7 + 2q + 2r) << 4) | (7 + 2q)."""

from __future__ import annotations

from collections import Counter

# --- our topology, rebuilt as engine/src/topology.rs builds it ---
_CORNER_OFFSETS = [(0, -2), (1, -1), (1, 1), (0, 2), (-1, 1), (-1, -1)]


def _our_topology():
    coords = [(q, r) for r in range(-2, 3) for q in range(-2, 3) if abs(q + r) <= 2]

    def corner(c, i):
        (q, r), (dx, dy) = c, _CORNER_OFFSETS[i]
        return (2 * q + r + dx, 3 * r + dy)

    points = sorted({corner(c, i) for c in coords for i in range(6)}, key=lambda p: (p[1], p[0]))
    node_of = {p: n for n, p in enumerate(points)}
    tile_nodes = [[node_of[corner(c, i)] for i in range(6)] for c in coords]
    sides = [(min(a, b), max(a, b)) for ns in tile_nodes for a, b in zip(ns, ns[1:] + ns[:1])]
    edges = sorted(set(sides))
    count = Counter(sides)
    coastal = frozenset(e for e, pair in enumerate(edges) if count[pair] == 1)
    return coords, tile_nodes, edges, coastal


TILE_COORDS, TILE_NODES, EDGE_NODES, COASTAL_EDGES = _our_topology()

# --- JSettlers' v1 encoding (SOCBoard, SOCBoard4p) ---
HEXNODES = (0x01, 0x12, 0x21, 0x10, -0x01, -0x10)  # N, NE, SE, S, SW, NW
OUR_TILE_TO_JS_HEX = [((7 + 2 * q + 2 * r) << 4) | (7 + 2 * q) for q, r in TILE_COORDS]
JS_HEX_TO_OUR_TILE = {h: t for t, h in enumerate(OUR_TILE_TO_JS_HEX)}


def _node_map() -> list[int]:
    js_of: dict[int, int] = {}
    for t, h in enumerate(OUR_TILE_TO_JS_HEX):
        for i, d in enumerate(HEXNODES):
            if js_of.setdefault(TILE_NODES[t][i], h + d) != h + d:
                raise AssertionError(f"our node {TILE_NODES[t][i]} has two JSettlers coordinates")
    return [js_of[n] for n in range(54)]


OUR_NODE_TO_JS_NODE = _node_map()
JS_NODE_TO_OUR_NODE = {js: n for n, js in enumerate(OUR_NODE_TO_JS_NODE)}
assert len(JS_NODE_TO_OUR_NODE) == 54, "the node map is not a bijection"


def js_edge_nodes(edge: int) -> tuple[int, int]:
    """The two nodes of JSettlers edge `edge` (SOCBoard.getAdjacentNodesToEdge_arr, v1 encoding)."""
    if ((edge & 0x0F) + (edge >> 4)) % 2 == 0:
        return edge + 0x01, edge + 0x10
    return edge, edge + 0x11


def _js_edge(a: int, b: int) -> int:
    lo, hi = min(a, b), max(a, b)
    e = lo if hi - lo == 0x11 else lo - 1
    if set(js_edge_nodes(e)) != {lo, hi}:
        raise AssertionError(f"no JSettlers edge joins nodes {lo:#x} and {hi:#x}")
    return e


OUR_EDGE_TO_JS_EDGE = [_js_edge(OUR_NODE_TO_JS_NODE[a], OUR_NODE_TO_JS_NODE[b]) for a, b in EDGE_NODES]
JS_EDGE_TO_OUR_EDGE = {e: i for i, e in enumerate(OUR_EDGE_TO_JS_EDGE)}
assert len(JS_EDGE_TO_OUR_EDGE) == 72, "the edge map is not a bijection"

# SOCBoard.numToHexID: the hex at each index of hexLayout / numberLayout (land and the water ring).
NUM_TO_HEX = [
    0x17, 0x39, 0x5B, 0x7D,
    0x15, 0x37, 0x59, 0x7B, 0x9D,
    0x13, 0x35, 0x57, 0x79, 0x9B, 0xBD,
    0x11, 0x33, 0x55, 0x77, 0x99, 0xBB, 0xDD,
    0x31, 0x53, 0x75, 0x97, 0xB9, 0xDB,
    0x51, 0x73, 0x95, 0xB7, 0xD9,
    0x71, 0x93, 0xB5, 0xD7,
]
HEX_NUM = {h: n for n, h in enumerate(NUM_TO_HEX)}
# SOCBoard4p.PORTS_EDGE_V1 / PORTS_FACING_V1 / PORTS_HEXNUM_V1, clockwise from the upper left.
PORT_EDGES = [0x27, 0x5A, 0x9C, 0xCC, 0xC9, 0xA5, 0x72, 0x42, 0x24]
PORT_FACING = [3, 4, 4, 5, 6, 6, 1, 2, 2]
PORT_HEXNUMS = [0, 2, 8, 21, 32, 35, 33, 22, 9]

WATER_HEX, DESERT_HEX, MISC_PORT = 0, 6, 0
ROAD, SETTLEMENT, CITY = 0, 1, 2
SMART_STRATEGY = 0

RESOURCE_NAMES = ("wood", "brick", "sheep", "wheat", "ore")
DEV_NAMES = ("knight", "victory_point", "road_building", "year_of_plenty", "monopoly")
# JSettlers resource (and land hex) code -> our resource index: clay 1, ore 2, sheep 3, wheat 4, wood 5.
JS_RES = {1: 1, 2: 4, 3: 2, 4: 3, 5: 0}
OUR_RES_TO_JS = {ours: js for js, ours in JS_RES.items()}
JS_DEV_TO_OURS = {9: "knight", 1: "road_building", 2: "year_of_plenty", 3: "monopoly",
                  **{c: "victory_point" for c in range(4, 9)}}


def js5_to_ours(js5) -> list[int]:
    """JSettlers' five known amounts (clay, ore, sheep, wheat, wood) in our order."""
    clay, ore, sheep, wheat, wood = js5
    return [wood, clay, sheep, wheat, ore]


def ours_to_js5(hand) -> list[int]:
    wood, brick, sheep, wheat, ore = hand
    return [brick, ore, sheep, wheat, wood]


STATES = {"START1A": 5, "START1B": 6, "START2A": 10, "START2B": 11, "ROLL_OR_CARD": 15, "PLAY1": 20,
          "PLACING_ROAD": 30, "PLACING_SETTLEMENT": 31, "PLACING_CITY": 32, "PLACING_ROBBER": 33,
          "PLACING_FREE_ROAD1": 40, "PLACING_FREE_ROAD2": 41, "WAITING_FOR_DISCARDS": 50,
          "WAITING_FOR_ROB_CHOOSE_PLAYER": 51, "WAITING_FOR_DISCOVERY": 52, "WAITING_FOR_MONOPOLY": 53,
          "OVER": 1000}
START1A, START1B, START2A, START2B = 5, 6, 10, 11
ROLL_OR_CARD, PLAY1 = 15, 20
PLACING_ROAD, PLACING_SETTLEMENT, PLACING_CITY, PLACING_ROBBER = 30, 31, 32, 33
PLACING_FREE_ROAD1, PLACING_FREE_ROAD2 = 40, 41
WAITING_FOR_DISCARDS, WAITING_FOR_ROB_CHOOSE_PLAYER = 50, 51
WAITING_FOR_DISCOVERY, WAITING_FOR_MONOPOLY = 52, 53
OVER = 1000
SETUP_STATES = frozenset({START1A, START1B, START2A, START2B})
