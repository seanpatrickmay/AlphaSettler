"""Command line: `alphasettler arena`, `compare`, `oracle-diff`, `selfplay` and `fit-heuristic`."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path

from alphasettler._engine import Bot, bot_names
from alphasettler.arena import SEED_LIMIT, _write_records, check, format_summary, open_output, run
from alphasettler.stats import paired, summarize


def _positive_int(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def _stamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S-%f")


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="alphasettler", description="AlphaSettler benchmark arena")
    sub = p.add_subparsers(dest="command", required=True)

    a = sub.add_parser("arena", help="one candidate vs three copies of a baseline, every seat rotated")
    a.add_argument("--candidate", required=True)
    a.add_argument("--baseline", required=True)
    a.add_argument("--seeds", type=_positive_int, required=True)
    a.add_argument("--seed-start", type=int, default=0)
    a.add_argument("--threads", type=_positive_int, default=None)
    a.add_argument("--out", default=None, help="JSONL path (default: runs/<time>-<candidate>-vs-<baseline>.jsonl)")
    a.add_argument("--no-trades", action="store_true", help="disable domestic trade offers")

    c = sub.add_parser("compare", help="two candidates against the same baseline on the same seeds")
    c.add_argument("--a", required=True)
    c.add_argument("--b", required=True)
    c.add_argument("--baseline", required=True)
    c.add_argument("--seeds", type=_positive_int, required=True)
    c.add_argument("--seed-start", type=int, default=0)
    c.add_argument("--threads", type=_positive_int, default=None)
    c.add_argument("--out-dir", default="runs", help="writes <time>-a-<a>-vs-<baseline>.jsonl and <time>-b-...")
    c.add_argument("--no-trades", action="store_true")

    o = sub.add_parser("oracle-diff", help="differential test against Catanatron (needs the oracle extra)")
    o.add_argument("--games", type=_positive_int, required=True)
    o.add_argument("--seed-start", type=int, default=0)
    o.add_argument("--workers", type=_positive_int, default=None)
    o.add_argument("--out", default=None, help="JSONL path, one line per game")

    s = sub.add_parser("selfplay", help="IsmctsBot self-play records for training and weight fits")
    s.add_argument("--games", type=_positive_int, required=True)
    s.add_argument("--simulations", type=_positive_int, required=True)
    s.add_argument("--rollout", type=int, default=0)
    s.add_argument("--seed-start", type=int, default=0)
    s.add_argument("--threads", type=_positive_int, default=None)
    s.add_argument("--out-dir", default="runs")
    s.add_argument("--trades", action="store_true", help="allow domestic offers (ismcts never makes any)")

    f = sub.add_parser("fit-heuristic", help="fit the heuristic evaluator's weights on self-play records")
    f.add_argument("records", nargs="+")
    f.add_argument("--iterations", type=_positive_int, default=25)
    f.add_argument("--l2", type=float, default=0.001)
    return p


def _check_bot_name(name: str) -> None:
    """ValueError naming `name` if the native parser rejects it (accepts `ismcts@N[+rD]` too)."""
    try:
        Bot(name, 0)
    except ValueError:
        raise ValueError(
            f"unknown bot {name!r}; known bots: {bot_names()} (or ismcts@N, ismcts@N+rD)"
        ) from None


# Games per native call. Python only sees Ctrl-C between calls (the GIL is released while games
# run), so this bounds how long an interrupt waits; every finished batch is already on disk.
SELFPLAY_BATCH = 100


def _selfplay(args) -> int:
    from alphasettler import records
    from alphasettler._engine import MAX_ROLLOUT, MAX_SIMULATIONS, SELFPLAY_MIN_SIMULATIONS, selfplay

    # Check everything the native call would reject before creating the output file.
    if not SELFPLAY_MIN_SIMULATIONS <= args.simulations <= MAX_SIMULATIONS:
        raise ValueError(f"--simulations must be {SELFPLAY_MIN_SIMULATIONS}..={MAX_SIMULATIONS}, got {args.simulations}")
    if not 0 <= args.rollout <= MAX_ROLLOUT:
        raise ValueError(f"--rollout must be 0..={MAX_ROLLOUT}, got {args.rollout}")
    if args.seed_start < 0 or args.seed_start + args.games > SEED_LIMIT:
        raise ValueError(f"seeds {args.seed_start}..{args.seed_start + args.games} are outside the u64 seed range")
    config = {} if args.trades else {"max_offers_per_turn": 0}
    out = Path(args.out_dir) / f"{_stamp()}-selfplay-s{args.simulations}.jsonl.gz"
    threads = args.threads or os.cpu_count() or 1
    games = decisions = 0
    end = args.seed_start + args.games
    # Open the output before the run, so a bad path fails before any game is played.
    with records.open_writer(out) as w:
        try:
            for lo in range(args.seed_start, end, SELFPLAY_BATCH):
                batch = selfplay(lo, min(SELFPLAY_BATCH, end - lo), args.simulations, threads, config, args.rollout)
                for g in batch:
                    w.write_game(g, config)
                    games += 1
                    decisions += len(g["decisions"])
                w.flush()
        except KeyboardInterrupt:
            print(f"interrupted: kept {games} games, {decisions} searched decisions in {out}", file=sys.stderr)
            return 130
    print(f"{games} games, {decisions} searched decisions, wrote {out}")
    return 0


def _fit_heuristic(args) -> int:
    from alphasettler import records
    from alphasettler._engine import fit_heuristic

    if not (math.isfinite(args.l2) and args.l2 >= 0):
        raise ValueError(f"--l2 must be finite and non-negative, got {args.l2}")
    games = []
    for path in args.records:
        for g in records.read(path):
            g["decisions"] = [{"index": d["index"]} for d in g["decisions"]]  # observations are not needed
            games.append(g)
    r = fit_heuristic(games, args.iterations, args.l2)
    print(f"samples: {r['samples']} from {len(games)} games")
    print(f"log-likelihood per sample: default {r['log_likelihood_before']:.4f}, fitted {r['log_likelihood_after']:.4f}")
    weights = ", ".join(f"{w:.4f}" for w in r["weights"])
    print(f"pub const DEFAULT_WEIGHTS: [f32; NUM_FEATURES] = [{weights}];")
    return 0


def _oracle_missing() -> int:
    from oracle import CATANATRON_REQUIREMENT

    print("error: the Catanatron oracle is not installed; install it with: "
          f'.venv/bin/pip3 install "{CATANATRON_REQUIREMENT}"', file=sys.stderr)
    return 2


def _oracle_diff(args) -> int:
    import oracle

    if not oracle.catanatron_available():
        return _oracle_missing()
    from oracle.diff import format_summary, run, summarize  # other import errors surface as themselves

    if args.seed_start < 0 or args.seed_start + args.games > SEED_LIMIT:
        raise ValueError(f"seeds {args.seed_start}..{args.seed_start + args.games} are outside the u64 seed range")
    with ExitStack() as stack:
        # Open the output before the run, so a bad path fails before any game is played.
        f = stack.enter_context(open_output(args.out)) if args.out else None
        results = run(range(args.seed_start, args.seed_start + args.games), args.workers)
        s = summarize(results)
        print(format_summary(s))
        if f is not None:
            for r in results:
                f.write(json.dumps({"seed": r.seed, "steps": r.steps, "allowlisted": dict(r.allowlisted),
                                    "mismatch": r.mismatch, "unimported_steps": r.unimported_steps,
                                    "ended_unimported": r.ended_unimported, "final_phase": r.final_phase,
                                    "winner": r.winner}) + "\n")
            print(f"wrote {args.out}")
    for r in s.mismatches[:5]:
        print(f"mismatch seed {r.seed}: {r.mismatch}", file=sys.stderr)
        for line in r.trace[-5:]:
            print(f"    {line}", file=sys.stderr)
    return 1 if s.mismatches else 0


def _catanatron_arena(args, out: Path) -> int:
    """Our candidate inside Catanatron against three of its bots. `--no-trades` is moot here:
    Catanatron's bots never trade, and ours plays with offers disabled."""
    import oracle

    if not oracle.catanatron_available():
        return _oracle_missing()
    import oracle.arena  # other import errors surface as themselves

    baseline = args.baseline.split(":", 1)[1]
    oracle.arena.check(args.candidate, baseline, args.seeds, args.seed_start)
    # Open the output before the run, so a bad path fails before any game is played.
    with open_output(out) as f:
        records = oracle.arena.run_match(args.candidate, baseline, args.seeds, args.seed_start, args.threads)
        _write_records(f, args.candidate, args.baseline, records)
    print(format_summary(f"{args.candidate} vs {args.baseline}", summarize(records)))
    print(f"fallbacks: {sum(r['fallbacks'] for r in records)}")
    print(f"belief resets: {sum(r['belief_resets'] for r in records)}")
    print(f"wrote {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = {"max_offers_per_turn": 0} if getattr(args, "no_trades", False) else None
    try:
        if args.command == "arena":
            name = args.baseline.replace(":", "-")
            out = Path(args.out or f"runs/{_stamp()}-{args.candidate}-vs-{name}.jsonl")
            if args.baseline.startswith("catanatron:"):
                return _catanatron_arena(args, out)
            _, summary = run(args.candidate, args.baseline, args.seeds, args.seed_start, args.threads, config, out)
            print(format_summary(f"{args.candidate} vs {args.baseline}", summary))
            print(f"wrote {out}")
        elif args.command == "oracle-diff":
            return _oracle_diff(args)
        elif args.command == "selfplay":
            return _selfplay(args)
        elif args.command == "fit-heuristic":
            return _fit_heuristic(args)
        else:
            for name in (args.a, args.b, args.baseline):
                _check_bot_name(name)
            sides = (("a", args.a), ("b", args.b))
            for _, name in sides:
                check(name, args.baseline, args.seeds, args.seed_start, args.threads, config)
            stamp = _stamp()
            outs = [Path(args.out_dir) / f"{stamp}-{side}-{name}-vs-{args.baseline}.jsonl" for side, name in sides]
            with ExitStack() as stack:
                files = [stack.enter_context(open_output(out)) for out in outs]
                results = {}
                for (side, name), out, f in zip(sides, outs, files):
                    records, summary = run(name, args.baseline, args.seeds, args.seed_start, args.threads, config, f)
                    results[side] = records
                    print(format_summary(f"{name} vs {args.baseline}", summary))
                    print(f"wrote {out}")
            pr = paired(results["a"], results["b"])
            print(
                f"paired {args.a} - {args.b}: mean diff {pr.mean_diff:+.3f} ± {pr.se:.3f} "
                f"(z = {pr.z:.2f}) over {pr.seeds} seeds"
            )
    except (ValueError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
