"""The rule differences between JSettlers2 2.6.10 and our engine that the differential replay
allows (spec Sections 3 and 5 and the plan's rulings on trades and the end of a turn). On such a
step only what its rule changes may differ (oracle.jsettlers.diff.excused), and our engine then
re-imports JSettlers' position; anything else our engine cannot replay is a mismatch to triage."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Entry:
    rule: str
    why: str


ENTRIES = {
    "domestic-trade": Entry(
        "a domestic trade executes when the first player accepts; the offerer never confirms, offers go to a "
        "chosen subset, may be two cards for one and are uncapped",
        "our trading game (current player offers to all, answers in seat order, offerer confirms) cannot "
        "replay it, so the position after the trade is re-imported"),
    "road-building-cancelled": Entry(
        "a player may end the turn during Road Building, forfeiting the roads not placed",
        "our Road Building has no way to stop early"),
    "bank-trade": Entry(
        "one bank trade may be any exchange the rates allow, including undoing an earlier one (1 for 2-4)",
        "only bank trades that are whole maritime trades of ours are replayed as such"),
    "forced-end-turn": Entry(
        "the server ends a stalled player's turn itself, with no end-turn request (SOCForceEndTurnThread)",
        "our engine has no forced end of turn, so the position after it is re-imported"),
    "forced-setup-end": Entry(
        "the server ends a stalled setup turn itself, removing the settlement awaiting its road with the cards it "
        "produced, and the setup order (and JSettlers' first player) moves on",
        "our setup has no forced end and no event removes a settlement, so the position after it is re-imported"),
}
