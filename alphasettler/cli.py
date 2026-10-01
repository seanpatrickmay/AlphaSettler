"""Command line: `alphasettler arena ...`, `alphasettler compare ...` and `alphasettler oracle-diff ...`."""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path

from alphasettler._engine import bot_names
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
    return p


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
        else:
            known = bot_names()
            for name in (args.a, args.b, args.baseline):
                if name not in known:
                    raise ValueError(f"unknown bot {name!r}; known bots: {known}")
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
