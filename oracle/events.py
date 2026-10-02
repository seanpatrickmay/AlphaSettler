"""Our engine's events for each executed Catanatron action, so an AlphaSettler bot playing inside
Catanatron tracks hidden cards from the same information it would have natively (spec Section 1).
Events are read off the action record and the positions before and after it, so they follow
Catanatron's rules where those differ from ours (for example production when the bank runs short).
The shape is that of `alphasettler.Game.log` entries."""

from __future__ import annotations

from catanatron.models.enums import ActionType

from oracle.tables import CAT_NODE_TO_OUR_NODE
from oracle.translate import (
    CAT_DEV, CUBE_TO_OUR_TILE, DEV_NAMES, RESOURCE_NAMES, Untranslatable, _res, our_edge, seat,
)


def _gain(before: dict, after: dict, p: int) -> list[int]:
    return [a - b for a, b in zip(after["players"][p]["hand"], before["players"][p]["hand"])]


def _played(p: int, card: str) -> dict:
    return {"type": "played_dev", "player": p, "card": card}


def record_events(before: dict, after: dict, record, state) -> list[dict]:
    """Events for one executed action. `before`/`after` are `catan_snapshot`s around it."""
    a = record.action
    t, v, p = a.action_type, a.value, seat(state, a.color)
    ev: list[dict] = []
    if t == ActionType.ROLL:
        ev.append({"type": "rolled", "player": p, "dice": list(record.result)})
        if sum(record.result) != 7:
            for q in range(4):
                g = _gain(before, after, q)
                if any(g):
                    ev.append({"type": "produced", "player": q, "resources": g})
    elif t == ActionType.BUILD_SETTLEMENT:
        ev.append({"type": "built_settlement", "player": p, "node": CAT_NODE_TO_OUR_NODE[v]})
        g = _gain(before, after, p)
        if before["phase"] == "setup_settlement" and any(g):
            ev.append({"type": "produced", "player": p, "resources": g})
    elif t == ActionType.BUILD_ROAD:
        ev.append({"type": "built_road", "player": p, "edge": our_edge(v)})
    elif t == ActionType.BUILD_CITY:
        ev.append({"type": "built_city", "player": p, "node": CAT_NODE_TO_OUR_NODE[v]})
    elif t == ActionType.BUY_DEVELOPMENT_CARD:
        ev.append({"type": "bought_dev", "player": p, "card": DEV_NAMES[CAT_DEV.index(record.result)]})
    elif t == ActionType.PLAY_KNIGHT_CARD:
        ev.append(_played(p, "knight"))
    elif t == ActionType.PLAY_ROAD_BUILDING:
        ev.append(_played(p, "road_building"))
    elif t == ActionType.PLAY_YEAR_OF_PLENTY:
        ev.append(_played(p, "year_of_plenty"))
        ev.append({"type": "year_of_plenty_taken", "player": p, "resources": _gain(before, after, p)})
    elif t == ActionType.PLAY_MONOPOLY:
        r = _res(v)
        ev.append(_played(p, "monopoly"))
        taken = [0 if q == p else -_gain(before, after, q)[r] for q in range(4)]
        ev.append({"type": "monopoly_taken", "player": p, "resource": RESOURCE_NAMES[r], "from": taken})
    elif t == ActionType.MOVE_ROBBER:
        cube, victim = v[0], v[1]
        ev.append({"type": "robber_moved", "player": p, "tile": CUBE_TO_OUR_TILE[cube]})
        if victim is not None and record.result is not None:
            ev.append({"type": "stole", "thief": p, "victim": seat(state, victim),
                       "resource": RESOURCE_NAMES[_res(record.result)]})
    elif t == ActionType.MARITIME_TRADE:
        gave = [0] * 5
        for r in v[:4]:
            if r is not None:
                gave[_res(r)] += 1
        got = [0] * 5
        got[_res(v[4])] = 1
        ev.append({"type": "maritime_traded", "player": p, "gave": gave, "got": got})
    elif t == ActionType.DISCARD_RESOURCE:
        ev.append({"type": "discarded", "player": p, "resource": RESOURCE_NAMES[_res(v)]})
    elif t == ActionType.END_TURN:
        ev.append({"type": "turn_ended", "player": p})
    else:
        raise Untranslatable("domestic-trade", f"{t} has no event translation")
    if after["phase"] == "game_over" and before["phase"] != "game_over":
        ev.append({"type": "game_over", "winner": after["winner"]})
    return ev


def redact(event: dict, viewer: int) -> dict:
    """`event` as `viewer` may see it (the engine's `Event::redacted_for`)."""
    if event["type"] == "stole" and viewer not in (event["thief"], event["victim"]):
        return {**event, "resource": None}
    if event["type"] == "bought_dev" and viewer != event["player"]:
        return {**event, "card": None}
    return event
