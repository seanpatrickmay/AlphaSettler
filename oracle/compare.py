"""Snapshot comparison shared by the Catanatron and JSettlers differential replays. Imports no engine of
another project, so both oracles may use it."""

from __future__ import annotations

from collections import Counter


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
