"""Sequential testing: play seeds in batches and stop as soon as the evidence decides between
H0 (the candidate's win rate is p0, no gain) and H1 (it is p1, a gain worth having).

Checking a fixed-size test's z after every batch and stopping at the first |z| > 2 inflates the
false-positive rate with every look. Wald's sequential probability ratio test keeps both error rates
at alpha and beta however often it looks. The unit is the seed, as in `stats`: a seed's four games
share a board, so they are not independent. The log-likelihood ratio is the normal approximation
over per-seed win rates with their sample variance (the generalized SPRT of Stockfish's fishtest):

    LLR = n (p1 - p0) (2 mean - p0 - p1) / (2 var)

It stops at H1 when LLR >= log((1 - beta) / alpha), at H0 when LLR <= log(beta / (1 - alpha)), and
never before MIN_SEEDS seeds (the variance needs a few)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from alphasettler.arena import open_output, run
from alphasettler.stats import Summary, per_seed_win_rates, summarize

MIN_SEEDS = 8


def bounds(alpha: float, beta: float) -> tuple[float, float]:
    return math.log(beta / (1 - alpha)), math.log((1 - beta) / alpha)


def llr(rates: list[float], p0: float, p1: float) -> float:
    n = len(rates)
    mean = sum(rates) / n
    var = sum((x - mean) ** 2 for x in rates) / n
    drift = (p1 - p0) * (2 * mean - p0 - p1)
    if var == 0:
        return 0.0 if drift == 0 else math.copysign(math.inf, drift)
    return n * drift / (2 * var)


def decide(rates: list[float], p0: float, p1: float, alpha: float, beta: float) -> str | None:
    """'H1', 'H0', or None to keep playing."""
    if len(rates) < MIN_SEEDS:
        return None
    lo, hi = bounds(alpha, beta)
    x = llr(rates, p0, p1)
    return "H1" if x >= hi else "H0" if x <= lo else None


@dataclass(frozen=True)
class SprtResult:
    decision: str | None  # None: max_seeds reached undecided
    seeds: int
    llr: float
    summary: Summary


def run_sprt(candidate: str, baseline: str, *, p0: float = 0.25, p1: float = 0.30, alpha: float = 0.05,
             beta: float = 0.05, batch: int = 12, max_seeds: int = 400, seed_start: int = 2_000_000_000,
             threads: int | None = None, config: dict | None = None, out: str | Path | None = None,
             nets_dir: str | Path | None = None, progress=None) -> SprtResult:
    """Play `batch` seeds at a time from `seed_start` until `decide` stops or `max_seeds` are played.
    Records go to `out` (opened exclusively). `progress(seeds, llr, summary)` is called per batch."""
    if not 0 <= p0 < p1 <= 1:
        raise ValueError(f"need 0 <= p0 < p1 <= 1, got p0 {p0}, p1 {p1}")
    if not (0 < alpha < 1 and 0 < beta < 1):
        raise ValueError("alpha and beta must be in (0, 1)")
    if batch < 1 or max_seeds < 1:
        raise ValueError("batch and max_seeds must be positive")
    f = None if out is None else open_output(out)
    records: list[dict] = []
    decision, x = None, 0.0
    try:
        while len(records) < 4 * max_seeds:
            played = len(records) // 4
            n = min(batch, max_seeds - played)
            new, _ = run(candidate, baseline, n, seed_start + played, threads, config, f, nets_dir)
            records += new
            rates = list(per_seed_win_rates(records).values())
            x = llr(rates, p0, p1)
            if progress is not None:
                progress(len(rates), x, summarize(records, null=p0))
            decision = decide(rates, p0, p1, alpha, beta)
            if decision is not None:
                break
    finally:
        if f is not None:
            f.close()
    return SprtResult(decision, len(records) // 4, x, summarize(records, null=p0))
