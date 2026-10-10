"""Boards and per-seed streams for JSettlers games. A seed's board is the native arena's (our
`Game(seed)` board): its tiles and numbers as they are and its nine port kinds, in our port order,
on JSettlers' nine port edges in their order. The dice, deck and misc seeds are fixed by the seed,
so every rotation and arm of a seed sees the same board, dice and deck (common random numbers)."""

from __future__ import annotations

from alphasettler import Game
from oracle.jsettlers.tables import (
    DESERT_HEX, HEX_NUM, JS_EDGE_TO_OUR_EDGE, JS_HEX_TO_OUR_TILE, JS_RES, MISC_PORT, OUR_EDGE_TO_JS_EDGE,
    OUR_RES_TO_JS, OUR_TILE_TO_JS_HEX, PORT_EDGES, PORT_FACING, PORT_HEXNUMS, RESOURCE_NAMES, WATER_HEX,
)

_MASK = (1 << 64) - 1
_SALTS = {"dice": 0xD1CE, "deck": 0xDEC4, "misc": 0x3155C}


def _splitmix(x: int) -> int:
    x = (x + 0x9E3779B97F4A7C15) & _MASK
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & _MASK
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & _MASK
    return x ^ (x >> 31)


def streams(seed: int) -> dict:
    """The seeds (63-bit, a Java long) of the game's dice, deck and misc random streams."""
    return {k: _splitmix((seed & _MASK) ^ salt) >> 1 for k, salt in _SALTS.items()}


def first_player(k: int) -> int:
    """The JSettlers first player that puts the bridge (JSettlers seat 0) at turn position k."""
    return (4 - k) % 4


def native_board(seed: int) -> dict:
    b = Game(seed).snapshot()["board"]
    kinds = [kind for _, kind in b["ports"]]
    # Port pairs are tuples, as `Game.snapshot()` writes them, so boards compare with ==.
    return {"tile_resource": list(b["tile_resource"]), "tile_number": list(b["tile_number"]),
            "ports": [(JS_EDGE_TO_OUR_EDGE[e], kind) for e, kind in zip(PORT_EDGES, kinds)]}


def normalized(board: dict) -> dict:
    return {"tile_resource": list(board["tile_resource"]), "tile_number": list(board["tile_number"]),
            "ports": sorted([int(e), k] for e, k in board["ports"])}


def js_layout(board: dict) -> dict:
    """`board` (ports on JSettlers' port edges) as SOCBoard's v1 hexLayout and numberLayout."""
    hexes, numbers, robber = [WATER_HEX] * 37, [-1] * 37, None
    for t, res in enumerate(board["tile_resource"]):
        h = OUR_TILE_TO_JS_HEX[t]
        if res is None:
            hexes[HEX_NUM[h]], robber = DESERT_HEX, h
        else:
            hexes[HEX_NUM[h]] = OUR_RES_TO_JS[RESOURCE_NAMES.index(res)]
            numbers[HEX_NUM[h]] = board["tile_number"][t]
    for i, (e, kind) in enumerate(board["ports"]):
        if OUR_EDGE_TO_JS_EDGE[e] != PORT_EDGES[i]:
            raise ValueError(f"port {i} is on our edge {e}, not on JSettlers' port edge {PORT_EDGES[i]:#x}")
        f = PORT_FACING[i]
        hexes[PORT_HEXNUMS[i]] = f + 6 if kind == "generic" else (f << 4) | OUR_RES_TO_JS[RESOURCE_NAMES.index(kind)]
    return {"hexes": hexes, "numbers": numbers, "robber": robber}


def our_board(raw: dict) -> dict:
    """A JSettlers board (Snapshots.board) as our board dict."""
    tile_resource, tile_number = [None] * 19, [0] * 19
    for h, kind, number in raw["hexes"]:
        t = JS_HEX_TO_OUR_TILE[h]
        if kind != DESERT_HEX:
            tile_resource[t], tile_number[t] = RESOURCE_NAMES[JS_RES[kind]], number
    ports = [(JS_EDGE_TO_OUR_EDGE[e], "generic" if kind == MISC_PORT else RESOURCE_NAMES[JS_RES[kind]])
             for e, kind in raw["ports"]]
    return {"tile_resource": tile_resource, "tile_number": tile_number, "ports": ports}


def game_command(name: str, seed: int, k: int, *, injected: bool = True, no_trades: bool = False,
                 bridge: bool = True) -> dict:
    return {"t": "game", "g": name, "layout": js_layout(native_board(seed)) if injected else None,
            "first": first_player(k), **streams(seed), "nt": no_trades, "bridge": bridge}
