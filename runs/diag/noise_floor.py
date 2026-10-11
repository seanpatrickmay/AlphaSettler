"""Diagnostic 1: the value noise floor.

Plays GAMES source games with four net:A@SIMS+t bots in the trading game, picks POSITIONS positions in each (a
pre-roll or main decision, spread over the game), and from each plays K continuations from the snapshot with
fresh dice and steals (fresh bots, each shown only the continuation's events). Each position's empirical winner
distribution is the best prediction of a continuation's winner that any value net can make (for players like
A@SIMS); the nets' excess cross-entropy over it, scored on the same continuation winners, is how much the value
head could still learn. Writes one JSON line a position; `--report` prints the tables.

Usage:
  .venv/bin/python3 runs/diag/noise_floor.py --out runs/diag/noise-floor.jsonl
  .venv/bin/python3 runs/diag/noise_floor.py --report runs/diag/noise-floor.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from alphasettler._engine import Bot, Game, net_evaluate

SEATS = 4
TRADES = {"max_offers_per_turn": 2, "max_trade_cards": 1}
NETS_DIR = "runs/train/nets"
VALUE_NETS = ("A", "L")
PHASES = ("pre_roll", "main")


def play(g: Game, bot: str, seed: int, record: bool = False) -> list[dict] | None:
    """Play `g` to its end with four fresh `bot`s; with `record`, the snapshot at every pre-roll or main
    decision (with the turn and the actor). Returns the snapshots (or None) and leaves `g` over."""
    bots = [Bot(bot, (seed * SEATS + p) % 2**64, NETS_DIR) for p in range(SEATS)]
    seen = [0] * SEATS
    snaps = [] if record else None
    while not g.is_over:
        p = g.current_actor
        log = g.log(p)
        bots[p].observe(p, log[seen[p]:])
        seen[p] = len(log)
        if record and g.phase in PHASES:
            snaps.append({"turn": g.turn, "actor": p, "phase": g.phase, "snapshot": g.snapshot()})
        g.apply(bots[p].act(g))
    return snaps


def winner(g: Game) -> int | None:
    return g.winner


def source(seed: int, bot: str, positions: int) -> list[dict]:
    g = Game(seed, TRADES)
    snaps = play(g, bot, seed, record=True)
    z = winner(g)
    rng = random.Random(seed)
    # Spread over the game: one pick from each of `positions` equal slices of the decisions.
    n = len(snaps)
    picks = [snaps[rng.randrange(i * n // positions, max(i * n // positions + 1, (i + 1) * n // positions))]
             for i in range(positions)]
    out = []
    for s in picks:
        pos = Game.from_snapshot(0, s["snapshot"], TRADES)
        values = {net: net_evaluate(pos, s["actor"], net, NETS_DIR) for net in VALUE_NETS}
        out.append({"seed": seed, "turn": s["turn"], "actor": s["actor"], "phase": s["phase"],
                    "final_turn": g.turn, "z": z, "values": values, "snapshot": s["snapshot"]})
    return out


def continuation(pos: dict, k: int, bot: str) -> int | None:
    cseed = (pos["seed"] * 1_000_003 + pos["turn"] * 1009 + k * 7919 + 17) % 2**63
    g = Game.from_snapshot(cseed, pos["snapshot"], TRADES)
    play(g, bot, cseed)
    return winner(g)


def run(args: argparse.Namespace) -> None:
    bot = f"net:A@{args.sims}+t"
    out = Path(args.out)
    if out.exists():
        sys.exit(f"{out} exists")
    seeds = range(args.seed_start, args.seed_start + args.games)
    with ThreadPoolExecutor(args.threads) as ex:
        positions = [p for ps in ex.map(lambda s: source(s, bot, args.positions), seeds) for p in ps]
        print(f"{len(positions)} positions from {args.games} games", flush=True)
        jobs = [(i, k) for i in range(len(positions)) for k in range(args.k)]
        wins = list(ex.map(lambda j: continuation(positions[j[0]], j[1], bot), jobs))
    with out.open("w") as f:
        for i, pos in enumerate(positions):
            ws = wins[i * args.k:(i + 1) * args.k]
            counts = [sum(w == p for w in ws) for p in range(SEATS)]
            draws = sum(w is None for w in ws)
            row = {k: v for k, v in pos.items() if k != "snapshot"}
            f.write(json.dumps({**row, "counts": counts, "draws": draws, "k": args.k}) + "\n")
    print(f"wrote {out}")


def target(counts: list[float], draws: float) -> list[float]:
    """A winner distribution from counts, a draw spread evenly (as the training target does)."""
    total = sum(counts) + draws
    return [(c + draws / SEATS) / total for c in counts]


def ce(p: list[float], q: list[float]) -> float:
    return -sum(pi * math.log(max(qi, 1e-6)) for pi, qi in zip(p, q))


def report(path: str) -> None:
    rows = [json.loads(l) for l in open(path)]
    buckets = {"turns 0-19": (0, 20), "turns 20-39": (20, 40), "turns 40-59": (40, 60), "turns 60+": (60, 10**9),
               "all": (0, 10**9)}
    print(f"{len(rows)} positions, {rows[0]['k']} continuations each\n")
    print("Cross-entropy (nats) of each predictor against the continuation winners; the floor is the empirical "
          "distribution, leave-one-out with an add-0.5 prior (an upper bound on the true floor). Uniform is 1.386.\n")
    print("| positions | n | floor (LOO) | floor (in-sample) | " + " | ".join(f"net {n}" for n in VALUE_NETS)
          + " | " + " | ".join(f"excess {n}" for n in VALUE_NETS) + " | " + " | ".join(f"CE vs z, {n}" for n in VALUE_NETS)
          + " |")
    print("|---|---|---|---|" + "---|" * (3 * len(VALUE_NETS)))
    for name, (lo, hi) in buckets.items():
        rs = [r for r in rows if lo <= r["turn"] < hi]
        if not rs:
            continue
        loo = ins = 0.0
        net = dict.fromkeys(VALUE_NETS, 0.0)
        vz = dict.fromkeys(VALUE_NETS, 0.0)
        nz = 0
        for r in rs:
            c, d, k = r["counts"], r["draws"], r["k"]
            emp = target(c, d)
            ins += ce(emp, emp)
            # Leave one out: each continuation's winner scored by the other k-1, add-0.5 smoothed.
            for p in range(SEATS):
                if c[p]:
                    rest = [c[q] - (q == p) + 0.5 for q in range(SEATS)]
                    rest = target(rest, d)
                    loo += c[p] / k * -math.log(rest[p])
            if d:
                rest = target([c[q] + 0.5 for q in range(SEATS)], d - 1)
                loo += d / k * sum(-math.log(rest[q]) for q in range(SEATS)) / SEATS
            for n in VALUE_NETS:
                net[n] += ce(emp, r["values"][n])
                if r["z"] is not None:
                    vz[n] += -math.log(max(r["values"][n][r["z"]], 1e-6))
            nz += r["z"] is not None
        m = len(rs)
        cells = [f"{loo / m:.3f}", f"{ins / m:.3f}"] + [f"{net[n] / m:.3f}" for n in VALUE_NETS] \
            + [f"{(net[n] - loo) / m:+.3f}" for n in VALUE_NETS] + [f"{vz[n] / max(nz, 1):.3f}" for n in VALUE_NETS]
        print(f"| {name} | {m} | " + " | ".join(cells) + " |")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    ap.add_argument("--report")
    ap.add_argument("--games", type=int, default=128)
    ap.add_argument("--positions", type=int, default=4)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--sims", type=int, default=100)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--seed-start", type=int, default=11_000_000_000)
    args = ap.parse_args()
    if args.report:
        report(args.report)
    else:
        run(args)


if __name__ == "__main__":
    main()
