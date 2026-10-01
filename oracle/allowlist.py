"""Intended rule differences between AlphaSettler (which follows the Catan rulebook) and
Catanatron 3.3. Each entry says what differs and why we keep our behavior; the classifiers
recognise a divergence as one of them. Anything they do not recognise is a mismatch."""

from __future__ import annotations

from dataclasses import dataclass

from catanatron.models.enums import ActionType
from catanatron.models.map import build_map

from oracle.tables import CAT_NODE_TO_OUR_NODE
from oracle.translate import CUBE_TO_OUR_TILE, PLAY_ROAD_BUILDING, RESOURCE_NAMES, build_road, edge_nodes


@dataclass(frozen=True)
class Entry:
    id: str
    rule: str
    reason: str


ENTRIES = {e.id: e for e in [
    Entry("longest-road", "Longest road length and award",
          "Catanatron does not count a road segment that ends at an opponent's building, does not "
          "recompute lengths in every case after a cut, and after a cut gives the award to the "
          "longest road even below 5. The rulebook lets a trail end at an opponent's building "
          "and requires 5 roads for the award; we follow it."),
    Entry("bank-shortage", "Production when the bank runs short",
          "When the bank cannot pay every claimant of a resource, Catanatron pays no one, even a "
          "single claimant. The rulebook pays a single claimant what the bank has left."),
    Entry("year-of-plenty-one-card", "Year of Plenty with a short bank",
          "Catanatron offers taking one card when the bank lacks two; we offer only pairs the "
          "bank can pay."),
    Entry("road-from-opponent-building", "Building a road through an opponent's settlement",
          "After a cut, Catanatron can let a road continue from an opponent's settlement (and "
          "then offers Road Building when that is the only such edge). The rulebook forbids "
          "building through it."),
    Entry("win-off-turn", "Winning on another player's turn",
          "Catanatron declares any player at 10 VP the winner immediately, and the last seat at 10 "
          "when several are; the rulebook only lets a player win on their own turn."),
]}


# Static geometry (Catanatron node ids do not depend on the seed; reference section 1.3).
_EDGE_NODES = [edge_nodes(e) for e in range(72)]
_NODE_EDGES: dict[int, list[tuple[int, int]]] = {}
for _e, (_a, _b) in enumerate(_EDGE_NODES):
    _NODE_EDGES.setdefault(_a, []).append((_e, _b))
    _NODE_EDGES.setdefault(_b, []).append((_e, _a))
_TILE_NODES = [frozenset()] * 19
for _cube, _tile in build_map("BASE").land_tiles.items():
    _TILE_NODES[CUBE_TO_OUR_TILE[_cube]] = frozenset(CAT_NODE_TO_OUR_NODE[n] for n in _tile.nodes.values())


def _buildings(p: dict) -> set[int]:
    return set(p["settlements"]) | set(p["cities"])


def _opponent_building_nodes(snap: dict, actor: int) -> set[int]:
    return {n for i, p in enumerate(snap["players"]) if i != actor for n in _buildings(p)}


def _vp(snap: dict, i: int) -> int:
    p = snap["players"][i]
    return (len(p["settlements"]) + 2 * len(p["cities"]) + p["dev_hand"][1]
            + 2 * (snap["longest_road_owner"] == i) + 2 * (snap["largest_army_owner"] == i))


def _rulebook_road_len(roads, blocked: set[int]) -> int:
    """Longest trail; it may end at, but not pass through, an opponent's building."""
    roads = set(roads)

    def walk(v, used):
        best = 0
        for e, u in _NODE_EDGES[v]:
            if e in roads and e not in used:
                best = max(best, 1 + (0 if u in blocked else walk(u, used | {e})))
        return best

    return max((walk(n, frozenset()) for e in roads for n in _EDGE_NODES[e]), default=0)


def _rulebook_road_owner(holder, snap: dict):
    """Rulebook award: 5+; the holder keeps a tie; a unique leader takes it; else unowned."""
    lens = [_rulebook_road_len(p["roads"], _opponent_building_nodes(snap, i)) for i, p in enumerate(snap["players"])]
    top = max(lens)
    if top < 5:
        return None
    if holder is not None and lens[holder] == top:
        return holder
    leaders = [i for i, n in enumerate(lens) if n == top]
    return leaders[0] if len(leaders) == 1 else None


def _production(before: dict, roll: int, single_claimant_paid: bool):
    owed = [[0] * 5 for _ in before["players"]]
    board = before["board"]
    for t in range(19):
        if board["tile_number"][t] != roll or t == before["robber"] or board["tile_resource"][t] is None:
            continue
        r = RESOURCE_NAMES.index(board["tile_resource"][t])
        for i, p in enumerate(before["players"]):
            owed[i][r] += len(_TILE_NODES[t] & set(p["settlements"])) + 2 * len(_TILE_NODES[t] & set(p["cities"]))
    for r in range(5):
        if sum(o[r] for o in owed) <= before["bank"][r]:
            continue
        claimants = [i for i, o in enumerate(owed) if o[r]]
        for i, o in enumerate(owed):
            o[r] = before["bank"][r] if single_claimant_paid and claimants == [i] else 0
    hands = [[h + o for h, o in zip(p["hand"], owed[i])] for i, p in enumerate(before["players"])]
    bank = [before["bank"][r] - sum(o[r] for o in owed) for r in range(5)]
    return hands, bank


def _matches(snap: dict, hands_bank) -> bool:
    hands, bank = hands_bank
    return snap["bank"] == bank and [p["hand"] for p in snap["players"]] == hands


def _legit_win(snap: dict, other: dict) -> bool:
    """``snap`` ended on its current player's 10th VP, which ``other`` does not grant them."""
    cur = snap["current"]
    return snap["winner"] == cur and _vp(snap, cur) >= 10 and _vp(other, cur) < 10


def _phase_goes_on(before: dict, ours: dict) -> bool:
    """``ours`` is where a road or settlement from ``before`` leads when the game goes on: the
    same phase, or out of Road Building the next free road (same return phase) or the phase it
    returns to once the roads or the edges run out."""
    if before["phase"] != "road_building":
        return ours["phase"] == before["phase"]
    return ours["phase"] == before["return_phase"] or (
        ours["phase"] == "road_building" and ours["roads_left"] == before["roads_left"] - 1
        and ours["return_phase"] == before["return_phase"])


def classify_legal(token, before: dict, actor: int, only_in: str) -> str | None:
    """Which entry explains a token in only one engine's legal set (None: unexplained).
    ``only_in`` is "catanatron" or "ours"; every entry so far is a Catanatron-only action."""
    if only_in != "catanatron":
        return None
    me = before["players"][actor]
    if isinstance(token, tuple) and token[0] == "year_of_plenty_one_card":
        r, bank = token[1], before["bank"]
        # Catanatron adds (r,) for each pair containing r that the bank cannot pay, if it has an r.
        short_pair = bank[r] == 1 or any(bank[s] == 0 for s in range(5) if s != r)
        holds = me["dev_hand"][3] > me["dev_new"][3] and not before["dev_played_this_turn"]
        if bank[r] >= 1 and short_pair and holds:
            return "year-of-plenty-one-card"
        return None
    if isinstance(token, int) and build_road(0) <= token < build_road(72) and before["phase"] in ("main", "road_building"):
        if _road_through_opponent(before, actor, token - build_road(0)):
            return "road-from-opponent-building"
        return None
    if token == PLAY_ROAD_BUILDING:
        # Catanatron offers the card when it sees a buildable edge; ours only for a rulebook-legal
        # one. Explained only when ours has none and Catanatron's runs through an opponent.
        holds = me["dev_hand"][2] > me["dev_new"][2] and not before["dev_played_this_turn"]
        if (holds and before["phase"] in ("pre_roll", "main") and len(me["roads"]) < 15
                and not _rulebook_free_edges(before, actor)
                and any(_road_through_opponent(before, actor, e) for e in range(72))):
            return "road-from-opponent-building"
    return None


def _rulebook_free_edges(before: dict, actor: int) -> set[int]:
    """Free edges the rulebook lets ``actor`` build: touching their building, or a node their
    roads reach that no opponent holds."""
    me = before["players"][actor]
    reach = _buildings(me) | ({n for r in me["roads"] for n in _EDGE_NODES[r]}
                              - _opponent_building_nodes(before, actor))
    taken = {e for p in before["players"] for e in p["roads"]}
    return {e for e in range(72) if e not in taken and reach & set(_EDGE_NODES[e])}


def _road_through_opponent(before: dict, actor: int, e: int) -> bool:
    """Edge ``e`` is free and Catanatron would let ``actor`` build it only by continuing through
    an opponent's building: one end is that building (reached by the actor's road), and the
    actor has no road or building at the other end."""
    if any(e in p["roads"] for p in before["players"]):
        return False
    me = before["players"][actor]
    opponents = _opponent_building_nodes(before, actor)
    my_road_nodes = {n for r in me["roads"] for n in _EDGE_NODES[r]}
    for v, u in (_EDGE_NODES[e], _EDGE_NODES[e][::-1]):
        if v in opponents and v in my_road_nodes and u not in my_road_nodes and u not in _buildings(me):
            return True
    return False


def classify_state(diffs, action_type, before: dict, ours: dict, theirs: dict, roll: int | None = None) -> str | None:
    """Which entry explains the positions differing after one action (None: unexplained).
    Our side of every entry is recomputed here from the snapshots, independently of the engine;
    ``roll`` is the dice total when ``action_type`` is ROLL. Catanatron's side of longest-road
    cannot be checked: its road lengths live in board caches the snapshot does not carry."""
    d = set(diffs)
    ours_road_ok = ours["longest_road_owner"] == _rulebook_road_owner(before["longest_road_owner"], ours)
    if (action_type == ActionType.BUILD_SETTLEMENT and d <= {"phase", "longest_road_owner"}
            and theirs["phase"] == "game_over" and _phase_goes_on(before, ours)
            and theirs["winner"] != theirs["current"] and _vp(theirs, theirs["winner"]) >= 10
            and _vp(ours, ours["current"]) < 10 and ("longest_road_owner" not in d or ours_road_ok)):
        return "win-off-turn"
    if (action_type == ActionType.BUILD_SETTLEMENT
            and "winner" in d and d <= {"winner", "longest_road_owner"}
            and ours["phase"] == theirs["phase"] == "game_over"
            and ours["winner"] == ours["current"] and _vp(ours, ours["current"]) >= 10
            and theirs["winner"] != theirs["current"]
            and theirs["winner"] == max((i for i in range(4) if _vp(theirs, i) >= 10), default=None)
            and ("longest_road_owner" not in d or ours_road_ok)):
        # Both games end on the current player's settlement, but its cut also moved Catanatron's
        # longest road award (its lengths can be stale and below 5) to another player at 10 or
        # more; Catanatron names the last seat at 10, the rulebook the player on turn. Only a
        # settlement's cut can give VP off turn (roads award the builder, knights the player).
        return "win-off-turn"
    if (action_type in (ActionType.BUILD_ROAD, ActionType.BUILD_SETTLEMENT)
            and "longest_road_owner" in d and d <= {"longest_road_owner", "phase"} and ours_road_ok):
        if "phase" not in d:
            return "longest-road"
        if ours["phase"] == "game_over" and _legit_win(ours, theirs):
            return "longest-road"
        if theirs["phase"] == "game_over" and _legit_win(theirs, ours) and _phase_goes_on(before, ours):
            return "longest-road"
        return None
    if (action_type == ActionType.ROLL and roll is not None and d
            and all(k == "bank" or (k.startswith("players[") and k.endswith("].hand")) for k in d)
            and _matches(ours, _production(before, roll, True))
            and _matches(theirs, _production(before, roll, False))):
        return "bank-shortage"
    return None
