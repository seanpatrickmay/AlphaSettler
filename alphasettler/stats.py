"""Win-rate statistics with the seed as the unit of analysis.

Each seed is played four times with the candidate rotated through every seat, so a seed's
win rate is in {0, 0.25, 0.5, 0.75, 1}. Comparisons use per-seed rates, which also makes
two candidates evaluated on the same seeds directly paired (common random numbers).
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

Record = Mapping[str, object]


def candidate_won(r: Record) -> bool:
    """A draw (no winner at the turn limit) counts as a loss."""
    return r["winner"] is not None and r["winner"] == r["candidate_seat"]


def per_seed_win_rates(records: Iterable[Record]) -> dict[int, float]:
    """Raises ValueError unless every seed has exactly one record per seat 0..3: a truncated,
    duplicated or mixed run would otherwise skew the per-seed rates silently."""
    games: dict[int, dict[int, float]] = defaultdict(dict)
    for r in records:
        seed, seat = int(r["seed"]), int(r["candidate_seat"])
        if seat in games[seed]:
            raise ValueError(f"seed {seed} has more than one record for candidate seat {seat}")
        games[seed][seat] = 1.0 if candidate_won(r) else 0.0
    for seed, seats in games.items():
        if sorted(seats) != [0, 1, 2, 3]:
            raise ValueError(f"seed {seed} needs one record per candidate seat 0..3, got seats {sorted(seats)}")
    return {seed: sum(v.values()) / 4 for seed, v in games.items()}


def _mean_se(xs: list[float]) -> tuple[float, float]:
    if not xs:
        raise ValueError("no seeds to summarize")
    n = len(xs)
    mean = sum(xs) / n
    if n == 1:
        return mean, 0.0
    var = sum((x - mean) ** 2 for x in xs) / (n - 1)
    return mean, math.sqrt(var / n)


def _z(effect: float, se: float) -> float:
    if se > 0:
        return effect / se
    if effect == 0:
        return 0.0
    return math.copysign(math.inf, effect)


@dataclass(frozen=True)
class Summary:
    seeds: int
    games: int
    wins: int
    draws: int
    win_rate: float
    ci_low: float
    ci_high: float
    z_vs_null: float
    null: float
    mean_vp: float
    mean_turns: float


def summarize(records: Iterable[Record], null: float = 0.25, z_crit: float = 1.96) -> Summary:
    records = list(records)
    rates = per_seed_win_rates(records)
    mean, se = _mean_se(list(rates.values()))
    vp = [r["vp"][r["candidate_seat"]] for r in records]
    turns = [r["turns"] for r in records]
    return Summary(
        seeds=len(rates),
        games=len(records),
        wins=sum(candidate_won(r) for r in records),
        draws=sum(r["winner"] is None for r in records),
        win_rate=mean,
        ci_low=max(0.0, mean - z_crit * se),
        ci_high=min(1.0, mean + z_crit * se),
        z_vs_null=_z(mean - null, se),
        null=null,
        mean_vp=sum(vp) / len(vp),
        mean_turns=sum(turns) / len(turns),
    )


@dataclass(frozen=True)
class Paired:
    seeds: int
    mean_diff: float
    se: float
    z: float


def paired(a: Iterable[Record], b: Iterable[Record]) -> Paired:
    """Per-seed difference in candidate win rate, a minus b, over identical seed sets."""
    ra, rb = per_seed_win_rates(a), per_seed_win_rates(b)
    if ra.keys() != rb.keys():
        raise ValueError("paired comparison needs the same seeds on both sides")
    diffs = [ra[s] - rb[s] for s in sorted(ra)]
    mean, se = _mean_se(diffs)
    return Paired(seeds=len(diffs), mean_diff=mean, se=se, z=_z(mean, se))
