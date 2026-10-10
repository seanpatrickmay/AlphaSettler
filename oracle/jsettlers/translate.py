"""JSettlers positions (the harness's RAW snapshots) in our `Game.from_snapshot` schema. The stream
adds what a position alone does not say (`Context`): our turn count, whether the current player has
rolled, the cards it bought this turn and the offers it made. Our player i is JSettlers seat
(first + i) % 4, so our player 0 moves first. The bank is derived (19 less the cards in hands, never
below 0): JSettlers' bank is unlimited, and its positions import with `bank_limit` off."""

from __future__ import annotations

from dataclasses import dataclass, field

from alphasettler import Game
from oracle.jsettlers.tables import (
    EDGE_NODES, JS_DEV_TO_OURS, JS_EDGE_TO_OUR_EDGE, JS_HEX_TO_OUR_TILE, JS_NODE_TO_OUR_NODE, OVER,
    PLACING_FREE_ROAD1, PLACING_FREE_ROAD2, PLACING_ROBBER, PLAY1, ROLL_OR_CARD, SETUP_STATES, START1A, START1B,
    START2A, START2B, WAITING_FOR_DISCARDS, WAITING_FOR_ROB_CHOOSE_PLAYER, js5_to_ours,
)

PHASES = {START1A: "setup_settlement", START2A: "setup_settlement", START1B: "setup_road", START2B: "setup_road",
          ROLL_OR_CARD: "pre_roll", PLAY1: "main", PLACING_ROBBER: "move_robber",
          WAITING_FOR_DISCARDS: "discard", WAITING_FOR_ROB_CHOOSE_PLAYER: "steal",
          PLACING_FREE_ROAD1: "road_building", PLACING_FREE_ROAD2: "road_building", OVER: "game_over"}
_NODE_EDGES = [[e for e, (a, b) in enumerate(EDGE_NODES) if n in (a, b)] for n in range(54)]


class Untranslatable(Exception):
    def __init__(self, kind: str, detail: str):
        super().__init__(f"{kind}: {detail}")
        self.kind = kind


@dataclass
class Context:
    board: dict  # our board dict
    first: int  # JSettlers' first player
    turn: int = 0  # turns ended since setup (our `turn`)
    rolled: bool = False  # the current player has rolled this turn
    drawn: list = field(default_factory=lambda: [0] * 5)  # cards it bought this turn, by our card index
    offers: int = 0  # domestic offers it made this turn

    def ours(self, js: int) -> int:
        return (js - self.first) % 4

    def js(self, ours: int) -> int:
        return (self.first + ours) % 4


def dev_counts(dev10) -> list[int]:
    """Cards held by JSettlers code (0..9) in our order: knight, victory point, road building,
    year of plenty, monopoly."""
    return [dev10[9], sum(dev10[4:9]), dev10[1], dev10[2], dev10[3]]


def _setup_road_node(p: dict) -> int:
    roads = set(p["roads"])
    for n in p["settlements"]:
        if not any(e in roads for e in _NODE_EDGES[n]):
            return n
    raise Untranslatable("setup", "no settlement of the current player awaits its road")


def snapshot(raw: dict, ctx: Context, max_offers: int = 0) -> dict:
    state = raw["state"]
    phase = PHASES.get(state)
    if phase is None:
        raise Untranslatable("intermediate-state", f"JSettlers state {state} has no phase of ours")
    cur = ctx.ours(raw["current"]) if raw["current"] >= 0 else 0
    players = []
    for i in range(4):
        p = raw["players"][ctx.js(i)]
        knights, rb, disc, mono = p["played"]
        dev = dev_counts(p["dev"])
        players.append({
            "hand": js5_to_ours(p["res"]),
            "dev_hand": dev,
            "dev_new": [min(a, b) for a, b in zip(ctx.drawn, dev)] if i == cur else [0] * 5,
            "dev_played": [knights, 0, rb, disc, mono],
            "knights_played": knights,
            "settlements": sorted(JS_NODE_TO_OUR_NODE[n] for n in p["settlements"]),
            "cities": sorted(JS_NODE_TO_OUR_NODE[n] for n in p["cities"]),
            "roads": sorted(JS_EDGE_TO_OUR_EDGE[e] for e in p["roads"]),
            "discard_remaining": p["discard"] if phase == "discard" else 0,
        })
    held = [sum(p["hand"][r] for p in players) for r in range(5)]

    def seat(js: int) -> int | None:
        return None if js < 0 else ctx.ours(js)

    setup = state in SETUP_STATES
    return {
        "board": ctx.board,
        "robber": JS_HEX_TO_OUR_TILE[raw["robber"]],
        "bank": [max(0, 19 - h) for h in held],
        "dev_deck": [JS_DEV_TO_OURS[c] for c in raw["deck"]],
        "players": players,
        "current": cur,
        "phase": phase,
        "setup_road_node": _setup_road_node(players[cur]) if phase == "setup_road" else None,
        "roads_left": (2 if state == PLACING_FREE_ROAD1 else 1) if phase == "road_building" else None,
        "winner": seat(raw["winner"]) if phase == "game_over" else None,
        "return_phase": "main" if ctx.rolled else "pre_roll",
        "turn": ctx.turn,
        "setup_step": sum(len(p["roads"]) for p in players) if setup else 8,
        "dev_played_this_turn": bool(raw["players"][raw["current"]]["played_dev"]) if raw["current"] >= 0 else False,
        "offers_this_turn": min(ctx.offers, max_offers),
        "longest_road_owner": seat(raw["longest"]),
        "largest_army_owner": seat(raw["largest"]),
    }


def import_game(seed: int, raw: dict, ctx: Context, config: dict) -> Game:
    """Our engine at this JSettlers position (ValueError if our rules cannot hold it)."""
    return Game.from_snapshot(seed, snapshot(raw, ctx, config.get("max_offers_per_turn", 0)), config)
