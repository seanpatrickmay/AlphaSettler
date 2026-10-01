"""Translation between Catanatron 3.3 and AlphaSettler: boards, seats, actions, legal sets and
full positions. Catanatron seat i (index in ``state.colors``) is our player i."""

from __future__ import annotations

from collections import Counter

import catanatron.game as catan_game
from catanatron.models.enums import CITY, RESOURCES, SETTLEMENT, ActionPrompt, ActionType

from oracle.tables import (
    CAT_EDGE_TO_OUR_EDGE, CAT_NODE_TO_OUR_NODE, CAT_PORT_ID_TO_OUR_EDGE, OUR_EDGE_TO_CAT_EDGE,
    OUR_TILE_TO_CAT_CUBE,
)

RESOURCE_NAMES = ["wood", "brick", "sheep", "wheat", "ore"]  # same order as Catanatron RESOURCES
DEV_NAMES = ["knight", "victory_point", "road_building", "year_of_plenty", "monopoly"]
CAT_DEV = ["KNIGHT", "VICTORY_POINT", "ROAD_BUILDING", "YEAR_OF_PLENTY", "MONOPOLY"]
CUBE_TO_OUR_TILE = {cube: t for t, cube in enumerate(OUR_TILE_TO_CAT_CUBE)}
SETUP_TURNS = 6  # Catanatron's num_turns counts 6 advances during setup

# Our action ids (engine/src/action.rs).
ROLL, END_TURN, BUY_DEV, PLAY_KNIGHT, PLAY_ROAD_BUILDING = 0, 1, 2, 3, 4
PAIRS = [(a, b) for a in range(5) for b in range(a, 5)]


def play_monopoly(r): return 5 + r
def play_year_of_plenty(a, b): return 10 + PAIRS.index((min(a, b), max(a, b)))
def build_settlement(n): return 25 + n
def build_city(n): return 79 + n
def build_road(e): return 133 + e
def move_robber(t): return 205 + t
def steal_from(p): return 224 + p
def discard(r): return 228 + r
def maritime(give, get): return 233 + 5 * give + get


class Untranslatable(Exception):
    """A Catanatron action with no AlphaSettler equivalent; ``kind`` names the allowlist entry."""

    def __init__(self, kind: str, detail: str):
        super().__init__(f"{kind}: {detail}")
        self.kind = kind


def _res(name: str) -> int:
    return RESOURCES.index(name)


def seat(state, color) -> int:
    return state.color_to_index[color]


def our_edge(edge) -> int:
    return CAT_EDGE_TO_OUR_EDGE[tuple(sorted(edge))]


def edge_nodes(e: int) -> tuple[int, int]:
    a, b = OUR_EDGE_TO_CAT_EDGE[e]
    return CAT_NODE_TO_OUR_NODE[a], CAT_NODE_TO_OUR_NODE[b]


def board_dict(catan_map) -> dict:
    tile_resource = [None] * 19
    tile_number = [0] * 19
    for cube, tile in catan_map.land_tiles.items():
        t = CUBE_TO_OUR_TILE[cube]
        tile_resource[t] = None if tile.resource is None else RESOURCE_NAMES[_res(tile.resource)]
        tile_number[t] = tile.number or 0
    ports = [
        (CAT_PORT_ID_TO_OUR_EDGE[p.id], "generic" if p.resource is None else RESOURCE_NAMES[_res(p.resource)])
        for p in sorted(catan_map.ports_by_id.values(), key=lambda p: p.id)
    ]
    return {"tile_resource": tile_resource, "tile_number": tile_number, "ports": ports}


def action_token(state, action):
    """The comparable token for one Catanatron action (see module docstring of oracle.diff)."""
    t, v = action.action_type, action.value
    simple = {
        ActionType.ROLL: ROLL, ActionType.END_TURN: END_TURN, ActionType.BUY_DEVELOPMENT_CARD: BUY_DEV,
        ActionType.PLAY_KNIGHT_CARD: PLAY_KNIGHT, ActionType.PLAY_ROAD_BUILDING: PLAY_ROAD_BUILDING,
    }
    if t in simple:
        return simple[t]
    if t == ActionType.PLAY_MONOPOLY:
        return play_monopoly(_res(v))
    if t == ActionType.PLAY_YEAR_OF_PLENTY:
        if len(v) == 1:
            return ("year_of_plenty_one_card", _res(v[0]))
        return play_year_of_plenty(_res(v[0]), _res(v[1]))
    if t == ActionType.BUILD_SETTLEMENT:
        return build_settlement(CAT_NODE_TO_OUR_NODE[v])
    if t == ActionType.BUILD_CITY:
        return build_city(CAT_NODE_TO_OUR_NODE[v])
    if t == ActionType.BUILD_ROAD:
        return build_road(our_edge(v))
    if t == ActionType.MOVE_ROBBER:
        cube, victim = v[0], v[1]
        return ("robber", CUBE_TO_OUR_TILE[cube], None if victim is None else seat(state, victim))
    if t == ActionType.DISCARD_RESOURCE:
        return discard(_res(v))
    if t == ActionType.MARITIME_TRADE:
        return maritime(_res(v[0]), _res(v[4]))
    raise Untranslatable("domestic-trade", f"{t} is outside the differential test")


def catan_tokens(state, playable) -> set:
    return {action_token(state, a) for a in playable}


def our_tokens(game) -> set:
    """Our legal actions as tokens; each robber move expands into one token per steal choice."""
    if game.phase != "move_robber":
        return set(game.legal_actions())
    out = set()
    for a in game.legal_actions():
        tile = a - move_robber(0)
        after = game.copy()
        after.apply(a)
        if after.phase == "steal":
            out.update(("robber", tile, v - steal_from(0)) for v in after.legal_actions())
        else:
            out.add(("robber", tile, None))
    return out


def our_steps(state, record) -> list[tuple[int, dict | None]]:
    """Our actions, with forced chance outcomes, equivalent to one executed Catanatron action."""
    action, result = record.action, record.result
    t = action.action_type
    if t == ActionType.ROLL:
        return [(ROLL, {"roll": list(result)})]
    if t == ActionType.BUY_DEVELOPMENT_CARD:
        return [(BUY_DEV, {"dev": DEV_NAMES[CAT_DEV.index(result)]})]
    token = action_token(state, action)
    if t == ActionType.MOVE_ROBBER:
        _, tile, victim = token
        steps = [(move_robber(tile), None)]
        if victim is not None:
            steps.append((steal_from(victim), {"steal": RESOURCE_NAMES[_res(result)]}))
        return steps
    if isinstance(token, tuple):
        raise Untranslatable(token[0], repr(action))
    return [(token, None)]


def _phase(game) -> tuple[str, dict]:
    st = game.state
    winner = game.winning_color()
    if winner is not None:
        return "game_over", {"winner": seat(st, winner)}
    if st.num_turns >= catan_game.TURNS_LIMIT:
        return "game_over", {"winner": None}
    prompt = st.current_prompt
    if prompt == ActionPrompt.BUILD_INITIAL_SETTLEMENT:
        return "setup_settlement", {}
    if prompt == ActionPrompt.BUILD_INITIAL_ROAD:
        node = st.action_records[-1].action.value  # the settlement just placed
        return "setup_road", {"setup_road_node": CAT_NODE_TO_OUR_NODE[node]}
    if prompt == ActionPrompt.DISCARD:
        return "discard", {}
    if prompt == ActionPrompt.MOVE_ROBBER:
        return "move_robber", {}
    if prompt == ActionPrompt.PLAY_TURN:
        if st.is_road_building:
            # Catanatron grants 2 free roads even with 1 piece left and stops when the pieces
            # run out; ours grants min(2, pieces). The roads actually placeable are the same.
            pieces = st.player_state[f"P{st.current_turn_index}_ROADS_AVAILABLE"]
            return "road_building", {"roads_left": min(st.free_roads_available, pieces)}
        rolled = st.player_state[f"P{st.current_turn_index}_HAS_ROLLED"]
        return ("main" if rolled else "pre_roll"), {}
    raise Untranslatable("domestic-trade", f"prompt {prompt}")


def _bought_this_turn(st) -> list[int]:
    """Dev cards per kind the current player bought this turn, read from the action records.
    Catanatron's ``{DEV}_OWNED_AT_START`` flags cannot give this: they are set only at END_TURN,
    so after playing a card and buying another of the same kind the flag still says "owned"."""
    out = [0] * 5
    color = st.colors[st.current_turn_index]
    for record in reversed(st.action_records):
        t = record.action.action_type
        if t == ActionType.END_TURN:
            break
        if t == ActionType.BUY_DEVELOPMENT_CARD and record.action.color == color:
            out[CAT_DEV.index(record.result)] += 1
    return out


def catan_snapshot(game) -> dict:
    """A Catanatron position in the snapshot schema of ``alphasettler.Game.snapshot``."""
    st = game.state
    board = st.board
    ps = st.player_state
    cur = st.current_turn_index
    players = []
    bought = _bought_this_turn(st)
    for i, color in enumerate(st.colors):
        key = f"P{i}_"
        dev_hand = [ps[f"{key}{d}_IN_HAND"] for d in CAT_DEV]
        players.append({
            "hand": [ps[f"{key}{r}_IN_HAND"] for r in RESOURCES],
            "dev_hand": dev_hand,
            "dev_new": bought if i == cur else [0] * 5,
            "dev_played": [ps[f"{key}PLAYED_{d}"] for d in CAT_DEV],
            "knights_played": ps[f"{key}PLAYED_KNIGHT"],
            "settlements": sorted(CAT_NODE_TO_OUR_NODE[n] for n, (c, kind) in board.buildings.items()
                                  if c == color and kind == SETTLEMENT),
            "cities": sorted(CAT_NODE_TO_OUR_NODE[n] for n, (c, kind) in board.buildings.items()
                             if c == color and kind == CITY),
            "roads": sorted({our_edge(e) for e, c in board.roads.items() if c == color}),
            "discard_remaining": st.discard_counts[i] if st.is_discarding else 0,
        })
    phase, extra = _phase(game)
    rolled = ps[f"P{cur}_HAS_ROLLED"]
    snap = {
        "board": board_dict(board.map),
        "robber": CUBE_TO_OUR_TILE[board.robber_coordinate],
        "bank": list(st.resource_freqdeck),
        "dev_deck": [DEV_NAMES[CAT_DEV.index(c)] for c in reversed(st.development_listdeck)],
        "players": players,
        "current": cur,
        "phase": phase,
        "setup_road_node": None,
        "roads_left": None,
        "winner": None,
        "return_phase": "pre_roll" if phase in ("move_robber", "road_building") and not rolled else "main",
        "turn": 0 if st.is_initial_build_phase else st.num_turns - SETUP_TURNS,
        "setup_step": len(board.roads) // 2 if st.is_initial_build_phase else 8,
        "dev_played_this_turn": ps[f"P{cur}_HAS_PLAYED_DEVELOPMENT_CARD_IN_TURN"],
        "offers_this_turn": 0,  # Catanatron's bots never offer domestic trades
        "longest_road_owner": None if board.road_color is None else seat(st, board.road_color),
        "largest_army_owner": next((i for i in range(len(st.colors)) if ps[f"P{i}_HAS_ARMY"]), None),
    }
    snap.update(extra)
    return snap


def compare(ours: dict, theirs: dict) -> list[str]:
    """Paths where two snapshots disagree. Fields the engines represent differently are compared
    by meaning: the deck as a multiset (draw order beyond the next forced card is irrelevant)
    and the return phase only where it is used."""
    diffs = [k for k in ("board", "robber", "bank", "current", "phase", "turn", "setup_step",
                         "longest_road_owner", "largest_army_owner", "dev_played_this_turn",
                         "offers_this_turn")
             if ours[k] != theirs[k]]
    used = {"setup_road": "setup_road_node", "road_building": "roads_left", "game_over": "winner"}
    if ours["phase"] == theirs["phase"] and ours["phase"] in used:
        k = used[ours["phase"]]
        if ours[k] != theirs[k]:
            diffs.append(k)
    if (ours["phase"] == theirs["phase"] and ours["phase"] in ("move_robber", "road_building", "steal")
            and ours["return_phase"] != theirs["return_phase"]):
        diffs.append("return_phase")
    if Counter(ours["dev_deck"]) != Counter(theirs["dev_deck"]):
        diffs.append("dev_deck")
    for i, (a, b) in enumerate(zip(ours["players"], theirs["players"])):
        for k in ("hand", "dev_hand", "dev_new", "dev_played", "knights_played", "settlements", "cities",
                  "roads", "discard_remaining"):
            if a[k] != b[k]:
                diffs.append(f"players[{i}].{k}")
    return diffs
