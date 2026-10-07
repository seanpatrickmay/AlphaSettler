"""Translation between Catanatron 3.3 and AlphaSettler: boards, seats, actions (domestic trades
included), legal sets and full positions. Catanatron seat i (index in ``state.colors``) is our player i."""

from __future__ import annotations

from collections import Counter

import catanatron.game as catan_game
from catanatron.models.enums import CITY, RESOURCES, SETTLEMENT, Action, ActionPrompt, ActionType

from alphasettler import Game
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


# Domestic trades (engine/src/action.rs): an offer is OFFER_BASE + 20 x give bundle + get bundle,
# where bundles 0..5 are one card of a resource and 5..20 the PAIRS.
OFFER_BASE, ACCEPT_TRADE, REJECT_TRADE, CONFIRM_BASE, CANCEL_TRADE = 258, 658, 659, 660, 664


def confirm_trade(p): return CONFIRM_BASE + p
def offer_trade(give, get): return OFFER_BASE + 20 * give + get
def is_offer(a): return OFFER_BASE <= a < ACCEPT_TRADE


def bundle(i: int) -> list[int]:
    """Bundle `i` as counts per resource."""
    counts = [0] * 5
    for r in ((i,) if i < 5 else PAIRS[i - 5]):
        counts[r] += 1
    return counts


def bundle_index(counts) -> int:
    cards = [r for r in range(5) for _ in range(counts[r])]
    if len(cards) == 1:
        return cards[0]
    if len(cards) == 2:
        return 5 + PAIRS.index((cards[0], cards[1]))
    raise Untranslatable("domestic-trade", f"a side of {len(cards)} cards (ours holds 1 or 2)")


def offer_value(a: int) -> tuple[int, ...]:
    """Catanatron's OFFER_TRADE value for our offer id: the give counts, then the get counts."""
    k = a - OFFER_BASE
    return tuple(bundle(k // 20) + bundle(k % 20))


def offer_action(color, a: int):
    return Action(color, ActionType.OFFER_TRADE, offer_value(a))


class Untranslatable(Exception):
    """A Catanatron action with no AlphaSettler equivalent; ``kind`` names the allowlist entry."""

    def __init__(self, kind: str, detail: str):
        super().__init__(f"{kind}: {detail}")
        self.kind = kind


def _res(name: str) -> int:
    return RESOURCES.index(name)


def seat(state, color) -> int:
    return state.color_to_index[color]


def answers_own_offer(state, action) -> bool:
    """An answer by the offerer to its own offer. Catanatron 3.3 asks one of an offerer outside
    seat 0: `apply_accept_trade` and `apply_reject_trade` pick the next responder skipping only
    the player who just answered. It is no answer in our rules (only the offerer's opponents
    answer), so it has no step of our engine and no event. An answer's value ends with the
    offerer's seat."""
    return (action.action_type in (ActionType.ACCEPT_TRADE, ActionType.REJECT_TRADE)
            and action.value[10] == seat(state, action.color))


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
    if t == ActionType.OFFER_TRADE:
        return offer_trade(bundle_index(v[:5]), bundle_index(v[5:10]))
    if t == ActionType.ACCEPT_TRADE:
        return ACCEPT_TRADE
    if t == ActionType.REJECT_TRADE:
        return REJECT_TRADE
    if t == ActionType.CONFIRM_TRADE:
        return confirm_trade(seat(state, v[10]))
    if t == ActionType.CANCEL_TRADE:
        return CANCEL_TRADE
    raise Untranslatable("domestic-trade", f"{t} has no translation")


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
    if answers_own_offer(state, action):
        return []
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
    if prompt in (ActionPrompt.DECIDE_TRADE, ActionPrompt.DECIDE_ACCEPTEES):
        # A pending domestic offer: the snapshot is the Main position it was made from
        # (`our_game` replays the offer and its answers onto it).
        return "main", {}
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


def _offers_made(st) -> int:
    """Domestic offers the current player made this turn, not counting one still pending (a
    snapshot is the position an offer was made from), read from the action records."""
    color = st.colors[st.current_turn_index]
    n = 0
    for record in reversed(st.action_records):
        t = record.action.action_type
        if t == ActionType.END_TURN:
            break
        if t == ActionType.OFFER_TRADE and record.action.color == color:
            n += 1
    return n - int(st.is_resolving_trade)


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
        "offers_this_turn": _offers_made(st),
        "longest_road_owner": None if board.road_color is None else seat(st, board.road_color),
        "largest_army_owner": next((i for i in range(len(st.colors)) if ps[f"P{i}_HAS_ARMY"]), None),
    }
    snap.update(extra)
    return snap


def our_game(seed: int, game, config: dict) -> Game:
    """Catanatron's position as our engine's ``Game`` under ``config``. Our snapshots cannot hold
    a pending domestic offer, so one is replayed onto the position it was made from: the offer,
    then every opponent's answer in our seat order (which leaves the same position as
    Catanatron's order). Only an offer every opponent has answered (DECIDE_ACCEPTEES: the offerer
    to confirm) is a position we play; one still collecting answers raises Untranslatable."""
    ours = Game.from_snapshot(seed, catan_snapshot(game), config)
    st = game.state
    if not st.is_resolving_trade:
        return ours
    if st.current_prompt != ActionPrompt.DECIDE_ACCEPTEES:
        raise Untranslatable("domestic-trade", "an offer still collecting answers")
    v = st.current_trade
    ours.apply(offer_trade(bundle_index(v[:5]), bundle_index(v[5:10])))
    offerer = st.current_turn_index
    for k in range(1, 4):
        ours.apply(ACCEPT_TRADE if st.acceptees[(offerer + k) % 4] else REJECT_TRADE)
    return ours


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
