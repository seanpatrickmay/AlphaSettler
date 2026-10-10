"""The rule differences between JSettlers2 2.6.10 and our engine that the differential replay
allows (spec Section 3). Each re-imports JSettlers' position after the step; anything else that
our engine cannot replay is a mismatch to triage."""

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
}
