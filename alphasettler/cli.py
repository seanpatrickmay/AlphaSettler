"""Command line: `alphasettler arena`, `compare`, `oracle-diff`, `selfplay`, `fit-heuristic` and
`train`. `selfplay --simulations S` writes 3a records; `selfplay --pcr` and `selfplay --net NAME`
write 3b exploration records; `train` (the train extra) trains a network on them."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path

from alphasettler._engine import DEFAULT_NETS_DIR, Bot, bot_names
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
    a.add_argument("--nets-dir", default=DEFAULT_NETS_DIR, help="where net:NAME bots find NAME.bin")

    c = sub.add_parser("compare", help="two candidates against the same baseline on the same seeds")
    c.add_argument("--a", required=True)
    c.add_argument("--b", required=True)
    c.add_argument("--baseline", required=True)
    c.add_argument("--seeds", type=_positive_int, required=True)
    c.add_argument("--seed-start", type=int, default=0)
    c.add_argument("--threads", type=_positive_int, default=None)
    c.add_argument("--out-dir", default="runs", help="writes <time>-a-<a>-vs-<baseline>.jsonl and <time>-b-...")
    c.add_argument("--no-trades", action="store_true")
    c.add_argument("--nets-dir", default=DEFAULT_NETS_DIR, help="where net:NAME bots find NAME.bin")

    o = sub.add_parser("oracle-diff", help="differential test against Catanatron (needs the oracle extra)")
    o.add_argument("--games", type=_positive_int, required=True)
    o.add_argument("--seed-start", type=int, default=0)
    o.add_argument("--workers", type=_positive_int, default=None)
    o.add_argument("--out", default=None, help="JSONL path, one line per game")

    s = sub.add_parser("selfplay", help="IsmctsBot self-play records for training and weight fits")
    s.add_argument("--games", type=_positive_int, required=True)
    s.add_argument("--simulations", type=_positive_int, default=None,
                   help="3a self-play: every move a full search of this many simulations")
    s.add_argument("--pcr", action="store_true",
                   help="3b exploration with the heuristic bot (playout cap randomization, root noise, opening sampling)")
    s.add_argument("--net", default=None, help="3b exploration with the network <nets-dir>/NET.bin")
    s.add_argument("--nets-dir", default=DEFAULT_NETS_DIR)
    s.add_argument("--full-simulations", type=int, default=None, help="3b: full-search simulations (default 600)")
    s.add_argument("--fast-simulations", type=int, default=None, help="3b: fast-search simulations (default 100)")
    s.add_argument("--full-probability", type=float, default=None, help="3b: chance of a full search (default 0.25)")
    s.add_argument("--rollout", type=int, default=0)
    s.add_argument("--seed-start", type=int, default=0)
    s.add_argument("--threads", type=_positive_int, default=None)
    s.add_argument("--out-dir", default="runs")
    s.add_argument("--trades", action="store_true", help="allow domestic offers (ismcts never makes any)")

    f = sub.add_parser("fit-heuristic", help="fit the heuristic evaluator's weights on self-play records")
    f.add_argument("records", nargs="+")
    f.add_argument("--iterations", type=_positive_int, default=25)
    f.add_argument("--l2", type=float, default=0.001)

    t = sub.add_parser("train", help="train a network on self-play records (needs the train extra)")
    t.add_argument("records", nargs="+", help="generation directories, oldest first; the last is the newest")
    t.add_argument("--out", required=True, help="weights file to write")
    t.add_argument("--init", default=None, help="weights file to start from (default: from scratch)")
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--steps", type=_positive_int, default=None, help="default: 2 x new positions / batch")
    t.add_argument("--batch", type=_positive_int, default=1024)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--device", default="auto", choices=["auto", "mps", "cpu"])
    t.add_argument("--threads", type=_positive_int, default=None)
    t.add_argument("--trunk", type=_positive_int, default=512)
    t.add_argument("--value-width", type=_positive_int, default=256)
    return p


def _check_bot_name(name: str, nets_dir: str) -> None:
    """ValueError naming `name` if no bot answers to it (`ismcts@N[+rD]` and `net:NAME[@N[+rD]]`
    included). A well-formed `net:` name whose file is missing or invalid raises that error."""
    try:
        Bot(name, 0, nets_dir)
    except ValueError as e:
        if not str(e).startswith("unknown bot"):
            raise
        raise ValueError(
            f"unknown bot {name!r}; known bots: {bot_names()} (or ismcts@N, ismcts@N+rD, net:NAME[@N[+rD]])"
        ) from None


def _resets_line(records: list[dict]) -> str:
    """The belief resets summed over `records` (spec: every arena run must report 0)."""
    return f"belief resets: {sum(r['belief_resets'] for r in records)}"


# Games per native call. Python only sees Ctrl-C between calls (the GIL is released while games
# run), so this bounds how long an interrupt waits; every finished batch is already on disk.
SELFPLAY_BATCH = 100


def _selfplay(args) -> int:
    if args.net is not None or args.pcr:
        return _selfplay_explore(args)
    if args.simulations is None:
        raise ValueError("--simulations is required (or use --net NAME or --pcr for 3b exploration)")
    for flag in ("full_simulations", "fast_simulations", "full_probability"):
        if getattr(args, flag) is not None:
            raise ValueError(f"--{flag.replace('_', '-')} needs --net or --pcr")
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
    games = decisions = resets = 0
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
                    resets += g["belief_resets"]
                w.flush()
        except KeyboardInterrupt:
            print(f"interrupted: kept {games} games, {decisions} searched decisions in {out}", file=sys.stderr)
            print(f"belief resets: {resets}", file=sys.stderr)
            return 130
    print(f"{games} games, {decisions} searched decisions, wrote {out}")
    print(f"belief resets: {resets}")
    return 0


def _selfplay_explore(args) -> int:
    from alphasettler import files
    from alphasettler import selfplay as explore
    from alphasettler._engine import SELFPLAY_FAST_SIMULATIONS, SELFPLAY_FULL_PROBABILITY, SELFPLAY_FULL_SIMULATIONS

    if args.simulations is not None:
        raise ValueError("--simulations is for 3a self-play; with --net or --pcr use --full-simulations "
                         "and --fast-simulations")
    if args.rollout != 0:
        raise ValueError("--rollout is not supported with --net or --pcr")
    if args.seed_start < 0 or args.seed_start + args.games > SEED_LIMIT:
        raise ValueError(f"seeds {args.seed_start}..{args.seed_start + args.games} are outside the u64 seed range")
    full = SELFPLAY_FULL_SIMULATIONS if args.full_simulations is None else args.full_simulations
    fast = SELFPLAY_FAST_SIMULATIONS if args.fast_simulations is None else args.fast_simulations
    p = SELFPLAY_FULL_PROBABILITY if args.full_probability is None else args.full_probability
    config = {} if args.trades else {"max_offers_per_turn": 0}
    label = "ismcts" if args.net is None else f"net-{args.net}"
    out = Path(args.out_dir) / f"{_stamp()}-selfplay-{label}-f{full}-q{fast}.jsonl.gz"
    if out.exists():
        raise ValueError(f"{out} already exists")
    # Written under a hidden temporary name and renamed when complete, so a reader (another
    # machine's train-loop, say) never takes a half-written file for a finished one.
    tmp = files.temp_path(out)
    counts: dict = {}
    try:
        explore.write_explore(tmp, args.seed_start, args.games, config=config, full_simulations=full,
                              fast_simulations=fast, full_probability=p, net=args.net, nets_dir=args.nets_dir,
                              threads=args.threads, counts=counts)
    except KeyboardInterrupt:
        # The writer closed the gzip stream cleanly: the games so far are a readable file.
        kept = tmp.exists()
        if kept:
            os.replace(tmp, out)
        print(f"interrupted: kept {counts.get('games', 0)} games, {counts.get('decisions', 0)} searched "
              f"decisions{f' in {out}' if kept else ''}", file=sys.stderr)
        print(f"belief resets: {counts.get('belief_resets', 0)}", file=sys.stderr)
        return 130
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    os.replace(tmp, out)
    print(f"{counts['games']} games, {counts['decisions']} searched decisions "
          f"({counts['full_decisions']} full searches), wrote {out}")
    print(f"belief resets: {counts['belief_resets']}")
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


def _train(args) -> int:
    from alphasettler.train.data import load_window
    from alphasettler.train.train import TrainConfig, train

    dirs = [Path(d).resolve() for d in args.records]
    # Relative to the generations' common parent, so each file keeps its generation's name.
    root = Path(os.path.commonpath([str(d.parent) for d in dirs]))
    cfg = TrainConfig(lr=args.lr, steps=args.steps, batch=args.batch, seed=args.seed, device=args.device,
                      threads=args.threads, trunk=args.trunk, value=args.value_width)
    cfg.check()
    window = load_window(root, dirs)
    for line in window.notices:
        print(f"notice: {line}")
    print(f"window: {window.data.games} games, {window.data.decisions} positions "
          f"({window.data.full_decisions} full searches), {window.new_positions} new")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        s = train(window, out, cfg, Path(args.init) if args.init else None)
    except KeyboardInterrupt:
        print("interrupted: nothing written", file=sys.stderr)
        return 130
    except RuntimeError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(f"trained {s['steps']} steps on {s['device']}; parity max error {s['parity_max_error']:.2e}; wrote {out}")
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
    nets_dir = str(Path(args.nets_dir).resolve())
    oracle.arena.check(args.candidate, baseline, args.seeds, args.seed_start, nets_dir)
    # Open the output before the run, so a bad path fails before any game is played.
    with open_output(out) as f:
        records = oracle.arena.run_match(args.candidate, baseline, args.seeds, args.seed_start, args.threads, nets_dir)
        _write_records(f, args.candidate, args.baseline, records)
    print(format_summary(f"{args.candidate} vs {args.baseline}", summarize(records)))
    print(f"fallbacks: {sum(r['fallbacks'] for r in records)}")
    print(_resets_line(records))
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
            records, summary = run(args.candidate, args.baseline, args.seeds, args.seed_start, args.threads, config, out,
                                   args.nets_dir)
            print(format_summary(f"{args.candidate} vs {args.baseline}", summary))
            print(_resets_line(records))
            print(f"wrote {out}")
        elif args.command == "oracle-diff":
            return _oracle_diff(args)
        elif args.command == "selfplay":
            return _selfplay(args)
        elif args.command == "fit-heuristic":
            return _fit_heuristic(args)
        elif args.command == "train":
            return _train(args)
        else:
            for name in (args.a, args.b, args.baseline):
                _check_bot_name(name, args.nets_dir)
            sides = (("a", args.a), ("b", args.b))
            for _, name in sides:
                check(name, args.baseline, args.seeds, args.seed_start, args.threads, config, args.nets_dir)
            stamp = _stamp()
            outs = [Path(args.out_dir) / f"{stamp}-{side}-{name}-vs-{args.baseline}.jsonl" for side, name in sides]
            with ExitStack() as stack:
                files = [stack.enter_context(open_output(out)) for out in outs]
                results = {}
                for (side, name), out, f in zip(sides, outs, files):
                    records, summary = run(name, args.baseline, args.seeds, args.seed_start, args.threads, config, f,
                                           args.nets_dir)
                    results[side] = records
                    print(format_summary(f"{name} vs {args.baseline}", summary))
                    print(_resets_line(records))
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
