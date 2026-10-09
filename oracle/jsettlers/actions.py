"""Our engine's action ids (engine/src/action.rs `encode`) and the event redaction of
`Event::redacted_for`, kept here so oracle/jsettlers needs nothing from the Catanatron modules."""

from __future__ import annotations

ROLL, END_TURN, BUY_DEV, KNIGHT, ROAD_BUILDING = 0, 1, 2, 3, 4
ACCEPT, REJECT, CANCEL = 658, 659, 664
PAIRS = [(a, b) for a in range(5) for b in range(a, 5)]


def monopoly(r: int) -> int:
    return 5 + r


def year_of_plenty(a: int, b: int) -> int:
    return 10 + PAIRS.index((min(a, b), max(a, b)))


def settlement(n: int) -> int:
    return 25 + n


def city(n: int) -> int:
    return 79 + n


def road(e: int) -> int:
    return 133 + e


def move_robber(t: int) -> int:
    return 205 + t


def steal_from(p: int) -> int:
    return 224 + p


def discard(r: int) -> int:
    return 228 + r


def maritime(give: int, get: int) -> int:
    return 233 + 5 * give + get


def confirm(p: int) -> int:
    return 660 + p


def bundle(i: int) -> list[int]:
    h = [0] * 5
    if i < 5:
        h[i] = 1
    else:
        a, b = PAIRS[i - 5]
        h[a] += 1
        h[b] += 1
    return h


def bundle_index(hand) -> int | None:
    """The bundle id of a one- or two-card hand, or None for any other hand."""
    n = sum(hand)
    if n == 1:
        return list(hand).index(1)
    if n == 2:
        cards = sorted(r for r in range(5) for _ in range(hand[r]))
        return 5 + PAIRS.index((cards[0], cards[1]))
    return None


def offer(give, get) -> int | None:
    """OfferTrade's id for these hands, or None when either side is not a bundle."""
    g, w = bundle_index(give), bundle_index(get)
    return None if g is None or w is None else 258 + 20 * g + w


def is_offer(a: int) -> bool:
    return 258 <= a < 658


def offer_hands(a: int) -> tuple[list[int], list[int]]:
    k = a - 258
    return bundle(k // 20), bundle(k % 20)


def decode(a: int) -> tuple:
    """A readable form of action id `a`, for messages and tests."""
    named = {ROLL: ("roll",), END_TURN: ("end_turn",), BUY_DEV: ("buy_dev",), KNIGHT: ("knight",),
             ROAD_BUILDING: ("road_building",), ACCEPT: ("accept",), REJECT: ("reject",), CANCEL: ("cancel",)}
    if a in named:
        return named[a]
    for lo, hi, name in ((5, 10, "monopoly"), (10, 25, "year_of_plenty"), (25, 79, "settlement"), (79, 133, "city"),
                         (133, 205, "road"), (205, 224, "move_robber"), (224, 228, "steal_from"),
                         (228, 233, "discard"), (233, 258, "maritime"), (258, 658, "offer"), (660, 664, "confirm")):
        if lo <= a < hi:
            return (name, a - lo)
    raise ValueError(f"action id {a} is not a valid action")


def redact(event: dict, viewer: int) -> dict:
    """`event` as `viewer` may see it (the engine's `Event::redacted_for`)."""
    if event["type"] == "stole" and viewer not in (event["thief"], event["victim"]):
        return {**event, "resource": None}
    if event["type"] == "bought_dev" and viewer != event["player"]:
        return {**event, "card": None}
    return event
