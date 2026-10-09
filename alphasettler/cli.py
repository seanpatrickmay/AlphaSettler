"""Command line: `alphasettler arena`, `compare`, `oracle-diff`, `selfplay`, `fit-heuristic`, `train`,
`sprt`, `fast-train`, `fast-score`, `fast-loop`, `train-loop`, `net-upgrade`, `opening-train` and `opening-diagnose`;
`net-upgrade` rewrites an older net (encoder v2 or v3) for this build's encoder v4. `selfplay --simulations S`
writes 3a records; `selfplay --pcr` and `selfplay --net NAME` write 3b exploration records; `train` (the train extra)
trains a network on them; `train-loop` (the train extra) runs the resumable self-play, train and
gate generations; `fast-train` and `fast-score` (the train extra) train and score candidate nets on
a fixed pool of generations, and `fast-loop` runs small generations trained that way. `opening-train`
(the train extra) fits the opening model of spec 3d-1 on recorded games' setups. `opening-diagnose` prints spec 3d-1's
opening diagnosis (D1-D3 and the verdict on record directories, or D4 for a bot)."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path

from alphasettler._engine import DEFAULT_NETS_DIR, NET_DEFAULT_ROUNDS, NET_DEFAULT_WIDTH, Bot, bot_names
from alphasettler.arena import SEED_LIMIT, TRADES, _write_records, check, format_summary, open_output, run
from alphasettler.diagnose import AUDIT_BUDGETS
from alphasettler.stats import paired, summarize


# Spec Section 4: net0 trains from scratch at 1e-3, every later net from its parent at 3e-4.
SCRATCH_LR = 1e-3
FINE_TUNE_LR = 3e-4
TRAIN_INSTALL = ".venv/bin/pip3 install -e '.[train]'"


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
    rules = a.add_mutually_exclusive_group()
    rules.add_argument("--no-trades", action="store_true", help="disable domestic trade offers")
    rules.add_argument("--trades", action="store_true",
                       help="the trading game: 2 domestic offers a turn, 1 card a side (+t bots offer)")
    a.add_argument("--nets-dir", default=DEFAULT_NETS_DIR, help="where net:NAME bots find NAME.bin")

    sp = sub.add_parser("sprt", help="candidate vs three baselines in batches until a sequential test decides")
    sp.add_argument("--candidate", required=True)
    sp.add_argument("--baseline", required=True)
    sp.add_argument("--p0", type=float, default=0.25, help="H0: the candidate's win rate (no gain)")
    sp.add_argument("--p1", type=float, default=0.30, help="H1: the win rate of a gain worth having")
    sp.add_argument("--alpha", type=float, default=0.05)
    sp.add_argument("--beta", type=float, default=0.05)
    sp.add_argument("--batch", type=_positive_int, default=12, help="seeds per batch (4 games each)")
    sp.add_argument("--max-seeds", type=_positive_int, default=400)
    sp.add_argument("--seed-start", type=int, default=2_000_000_000,
                    help="default: a range no training game or gate uses")
    sp.add_argument("--threads", type=_positive_int, default=None)
    sp.add_argument("--out", default=None, help="JSONL path (default: runs/<time>-sprt-<candidate>-vs-<baseline>.jsonl)")
    sp_rules = sp.add_mutually_exclusive_group()
    sp_rules.add_argument("--no-trades", action="store_true", help="disable domestic trade offers")
    sp_rules.add_argument("--trades", action="store_true",
                          help="the trading game: 2 domestic offers a turn, 1 card a side (+t bots offer)")
    sp.add_argument("--nets-dir", default=DEFAULT_NETS_DIR, help="where net:NAME bots find NAME.bin")

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
    s.add_argument("--trades", action="store_true",
                   help="the trading game (2 offers a turn, 1 card a side); every seat may offer")

    f = sub.add_parser("fit-heuristic", help="fit the heuristic evaluator's weights on self-play records")
    f.add_argument("records", nargs="+")
    f.add_argument("--iterations", type=_positive_int, default=25)
    f.add_argument("--l2", type=float, default=0.001)

    t = sub.add_parser("train", help="train a network on self-play records (needs the train extra)")
    t.add_argument("records", nargs="+", help="generation directories, oldest first; the last is the newest")
    t.add_argument("--out", required=True, help="weights file to write")
    t.add_argument("--init", default=None, help="weights file to start from (default: from scratch)")
    t.add_argument("--lr", type=float, default=None,
                   help=f"default: {SCRATCH_LR:g} from scratch, {FINE_TUNE_LR:g} with --init (spec Section 4)")
    t.add_argument("--steps", type=_positive_int, default=None, help="default: 2 x new positions / batch")
    t.add_argument("--batch", type=_positive_int, default=1024)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--device", default="auto", choices=["auto", "mps", "cpu"])
    t.add_argument("--threads", type=_positive_int, default=None)
    t.add_argument("--width", type=_positive_int, default=NET_DEFAULT_WIDTH,
                   help="network v2's width d, from scratch (an --init file brings its own)")
    t.add_argument("--rounds", type=_positive_int, default=NET_DEFAULT_ROUNDS,
                   help="network v2's message-passing rounds R, from scratch (an --init file brings its own)")

    ft = sub.add_parser("fast-train",
                        help="train a candidate net for a wall-clock budget on a pool of generations (train extra)")
    ft.add_argument("records", nargs="+", help="generation directories to pool")
    ft.add_argument("--out", required=True, help="weights file to write (its summary goes to OUT.json)")
    ft.add_argument("--minutes", type=float, required=True, help="the wall-clock training budget")
    ft.add_argument("--init", default=None, help="weights file to start from (default: from scratch)")
    ft.add_argument("--lr", type=float, default=SCRATCH_LR, help="the peak learning rate (cosine over the budget)")
    ft.add_argument("--warmup", type=int, default=200, help="linear warmup steps")
    ft.add_argument("--policy-batch", type=_positive_int, default=256, help="policy-target rows per step")
    ft.add_argument("--value-batch", type=_positive_int, default=64, help="value and aux rows per step")
    ft.add_argument("--width", type=_positive_int, default=NET_DEFAULT_WIDTH)
    ft.add_argument("--rounds", type=_positive_int, default=NET_DEFAULT_ROUNDS)
    ft.add_argument("--seed", type=int, default=0)
    ft.add_argument("--eval-seconds", type=float, default=60.0, help="held-out evaluation interval")
    ft.add_argument("--device", default="auto", choices=["auto", "mps", "cpu"])
    ft.add_argument("--threads", type=_positive_int, default=None)

    fs = sub.add_parser("fast-score", help="held-out losses of weights files on a pool's held-out set (train extra)")
    fs.add_argument("records", nargs="+", help="generation directories to pool")
    fs.add_argument("--nets", nargs="+", required=True, help="weights files to score")
    fs.add_argument("--device", default="auto", choices=["auto", "mps", "cpu"])

    fl = sub.add_parser("fast-loop", help="small self-play generations trained with fast-train (train extra)")
    fl.add_argument("--dir", required=True, help="the run directory")
    fl.add_argument("--generations", type=int, required=True, help="the last generation number to run")
    fl.add_argument("--seed-dirs", nargs="+", default=None,
                    help="first run only: record directories to start from, oldest first")
    fl.add_argument("--start-net", default=None, help="first run only: the starting best net")
    fl.add_argument("--first", type=int, default=None, help="first run only: the first generation's number")
    fl.add_argument("--games", type=_positive_int, default=2000)
    fl.add_argument("--window-games", type=_positive_int, default=40_000)
    fl.add_argument("--minutes", type=float, default=10.0)
    fl.add_argument("--lr", type=float, default=5e-4)
    fl.add_argument("--gate-seeds", type=_positive_int, default=100)
    fl.add_argument("--gate-simulations", type=_positive_int, default=100)
    fl.add_argument("--device", default="auto", choices=["auto", "mps", "cpu"])
    fl.add_argument("--threads", type=_positive_int, default=None)
    fl.add_argument("--trades", action="store_true",
                    help="first run only: self-play and gate in the trading game, every seat +t")
    fl.add_argument("--gumbel", type=_positive_int, default=None, metavar="SIMS",
                    help="first run only: every self-play move a Gumbel root search of SIMS simulations")
    fl.add_argument("--opening", default=None, metavar="FILE",
                    help="first run only: an opening model (ASOPEN) picks self-play's setup settlements, "
                         "and both gate sides are +o; a resume with the same file restores a missing "
                         "nets/opening.bin")

    nu = sub.add_parser("net-upgrade", help="rewrite an encoder v2 or v3 net for encoder v4, same moves")
    nu.add_argument("old", help="the encoder v2 or v3 weights file")
    nu.add_argument("new", help="the weights file to write (must not exist)")

    ot = sub.add_parser("opening-train", help="train the opening model on recorded games' setups (train extra)")
    ot.add_argument("--games", nargs="+", required=True, metavar="DIR", help="generation directories of records")
    ot.add_argument("--out", required=True, help="the model file to write, e.g. <nets-dir>/opening.bin (must not exist)")
    ot.add_argument("--compare-net", default=None, metavar="NAME",
                    help="also score this net's value head on the held-out positions")
    ot.add_argument("--nets-dir", default=DEFAULT_NETS_DIR, help="where --compare-net finds NAME.bin")
    ot.add_argument("--seed", type=int, default=0)

    od = sub.add_parser("opening-diagnose", help="spec 3d-1's opening diagnosis: D1-D3 on records, or D4 for a bot")
    od_source = od.add_mutually_exclusive_group(required=True)
    od_source.add_argument("--games", nargs="+", metavar="DIR", help="record directories: D1, D2, D3 and the verdict")
    od_source.add_argument("--bot", help="play setups with four copies of this bot: D4")
    od.add_argument("--net", default=None, metavar="NAME",
                    help="with --games: the net whose value head D2 reads and whose search D3 re-runs")
    od.add_argument("--nets-dir", default=DEFAULT_NETS_DIR)
    od.add_argument("--audit", type=int, default=200,
                    help="with --games: D3 re-chooses the setups of this many games of the last DIR (0 skips D3)")
    od.add_argument("--audit-budgets", type=_positive_int, nargs="+", default=list(AUDIT_BUDGETS), metavar="N",
                    help="with --games: D3's simulation budgets")
    od.add_argument("--seeds", type=_positive_int, default=None, help="with --bot: how many setups to play")
    od.add_argument("--seed-start", type=int, default=2_000_000_000, help="with --bot: the first seed")
    od_rules = od.add_mutually_exclusive_group()
    od_rules.add_argument("--no-trades", action="store_true", help="with --bot: domestic offers off")
    od_rules.add_argument("--trades", action="store_true", help="with --bot: the trading game")
    od.add_argument("--threads", type=_positive_int, default=None)
    od.add_argument("--out", default=None, help="also write the report to this new file")

    lp = sub.add_parser("train-loop",
                        help="self-play, train and gate generations; rerun to resume (needs the train extra)")
    lp.add_argument("--dir", required=True, help="the run directory, e.g. runs/3b (nets in <dir>/nets)")
    lp.add_argument("--generations", type=int, required=True,
                    help="learned generations after the warm start (0: the warm start only)")
    lp.add_argument("--games", type=_positive_int, default=10_000, help="self-play games per learned generation")
    lp.add_argument("--warm-games", type=_positive_int, default=100_000, help="self-play games of generation 0")
    lp.add_argument("--chunk-games", type=_positive_int, default=500, help="games per self-play record file")
    lp.add_argument("--full-simulations", type=_positive_int, default=600)
    lp.add_argument("--fast-simulations", type=_positive_int, default=100)
    lp.add_argument("--full-probability", type=float, default=0.25)
    lp.add_argument("--window", type=_positive_int, default=4, help="generations of records to train on")
    lp.add_argument("--warm-lr", type=float, default=SCRATCH_LR, help="generation 0's learning rate (from scratch)")
    lp.add_argument("--lr", type=float, default=FINE_TUNE_LR, help="later generations' learning rate")
    lp.add_argument("--steps-factor", type=float, default=2.0, help="steps = factor x new positions / batch")
    lp.add_argument("--batch", type=_positive_int, default=1024)
    lp.add_argument("--width", type=_positive_int, default=NET_DEFAULT_WIDTH, help="network v2's width d for net 0")
    lp.add_argument("--rounds", type=_positive_int, default=NET_DEFAULT_ROUNDS,
                    help="network v2's message-passing rounds R for net 0")
    lp.add_argument("--gate-seeds", type=_positive_int, default=200)
    lp.add_argument("--gate-simulations", type=_positive_int, default=600)
    lp.add_argument("--max-turns", type=_positive_int, default=1000)
    lp.add_argument("--device", default="auto", choices=["auto", "mps", "cpu"])
    lp.add_argument("--threads", type=_positive_int, default=None)
    lp.add_argument("--gumbel", type=_positive_int, default=None, metavar="SIMS",
                    help="first run only: every self-play move a Gumbel root search of SIMS simulations")
    lp.add_argument("--opening", default=None, metavar="FILE",
                    help="first run only: an opening model (ASOPEN) picks self-play's setup settlements, "
                         "and both gate sides are +o")
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
            f"unknown bot {name!r}; known bots: {bot_names()} (or ismcts@N, ismcts@N+rD, net:NAME[@N[+rD]], style:STYLE[@N[+rD]] for STYLE roads, army or production, each with +sK, +t and +g; net: names also with +o)"
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
    config = dict(TRADES) if args.trades else {"max_offers_per_turn": 0}
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
    config = dict(TRADES) if args.trades else {"max_offers_per_turn": 0}
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
            files.fsync_file(tmp)
            os.replace(tmp, out)
        print(f"interrupted: kept {counts.get('games', 0)} games, {counts.get('decisions', 0)} searched "
              f"decisions{f' in {out}' if kept else ''}", file=sys.stderr)
        print(f"belief resets: {counts.get('belief_resets', 0)}", file=sys.stderr)
        return 130
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    files.fsync_file(tmp)
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


def _sprt(args, config) -> int:
    from alphasettler.sprt import run_sprt

    name = args.baseline.replace(":", "-")
    out = Path(args.out or f"runs/{_stamp()}-sprt-{args.candidate.replace(':', '-')}-vs-{name}.jsonl")

    def progress(seeds, llr, summary):
        print(f"  {seeds} seeds: win rate {summary.win_rate:.3f}, z {summary.z_vs_null:.2f}, LLR {llr:.2f}", flush=True)

    r = run_sprt(args.candidate, args.baseline, p0=args.p0, p1=args.p1, alpha=args.alpha, beta=args.beta,
                 batch=args.batch, max_seeds=args.max_seeds, seed_start=args.seed_start, threads=args.threads,
                 config=config, out=out, nets_dir=args.nets_dir, progress=progress)
    verdict = {"H1": f"H1: a gain (win rate >= {args.p1:g})", "H0": f"H0: no gain worth {args.p1:g}",
               None: f"undecided after {r.seeds} seeds"}[r.decision]
    print(format_summary(f"{args.candidate} vs {args.baseline}", r.summary))
    print(f"sprt: {verdict} after {r.seeds} seeds ({4 * r.seeds} games), LLR {r.llr:.2f}")
    print(f"wrote {out}")
    return 0


def _train_missing() -> bool:
    """True, after printing how to install it, if PyTorch (the `train` extra) is missing. Any
    other import error surfaces as itself."""
    try:
        import alphasettler.train  # noqa: F401
    except ModuleNotFoundError as e:
        if e.name != "torch":
            raise
        print(f"error: PyTorch (the train extra) is not installed; install it with: {TRAIN_INSTALL}", file=sys.stderr)
        return True
    return False


def _train(args) -> int:
    if _train_missing():
        return 2
    from alphasettler.train import model
    from alphasettler.train.data import load_window
    from alphasettler.train.train import TrainConfig, train

    init = Path(args.init) if args.init else None
    lr = args.lr if args.lr is not None else FINE_TUNE_LR if init is not None else SCRATCH_LR
    cfg = TrainConfig(lr=lr, steps=args.steps, batch=args.batch, seed=args.seed, device=args.device,
                      threads=args.threads, width=args.width, rounds=args.rounds)
    cfg.check()
    if init is not None:
        model.load(init)  # a bad --init fails now, not after the window is read
    dirs = [Path(d).resolve() for d in args.records]
    # Relative to the generations' common parent, so each file keeps its generation's name.
    root = Path(os.path.commonpath([str(d.parent) for d in dirs]))
    window = load_window(root, dirs)
    for line in window.notices:
        print(f"notice: {line}")
    print(f"window: {window.games} games, {window.decisions} positions "
          f"({window.full_decisions} full searches), {window.new_positions} new; "
          f"{window.validation.games} held out ({window.validation.decisions} positions)")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        s = train(window, out, cfg, init)
    except KeyboardInterrupt:
        print("interrupted: nothing written", file=sys.stderr)
        return 130
    except RuntimeError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    best = s["best_held_out"]
    kept = "the last step" if best is None else \
        f"the best held-out score {best['score']} (policy {best['policy']}, value {best['value']}) after step {best['step'] + 1}"
    print(f"trained {s['steps']} steps on {s['device']} at lr {s['lr']:g}; kept {kept}; "
          f"parity max error {s['parity_max_error']:.2e}; wrote {out}")
    return 0


def _pool(records: list[str]):
    from alphasettler.train.fast import load_pool

    dirs = [Path(d).resolve() for d in records]
    pool = load_pool(Path(os.path.commonpath([str(d.parent) for d in dirs])), dirs)
    for line in pool.notices:
        print(f"notice: {line}")
    print(f"pool: {pool.train.games} games, {pool.train.decisions} positions "
          f"({pool.train.policy_decisions} policy targets); {pool.validation.games} held out "
          f"({pool.validation.decisions} positions)", flush=True)
    return pool


def _fast_train(args) -> int:
    if _train_missing():
        return 2
    from alphasettler.train import model
    from alphasettler.train.fast import FastConfig, fast_train

    cfg = FastConfig(minutes=args.minutes, lr=args.lr, warmup=args.warmup, policy_batch=args.policy_batch,
                     value_batch=args.value_batch, width=args.width, rounds=args.rounds, seed=args.seed,
                     eval_seconds=args.eval_seconds, device=args.device, threads=args.threads)
    cfg.check()
    if args.init:
        model.load(args.init)  # a bad --init fails now, not after the pool is read
    pool = _pool(args.records)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        s = fast_train(pool, out, cfg, args.init)
    except KeyboardInterrupt:
        print("interrupted: nothing written", file=sys.stderr)
        return 130
    except RuntimeError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    b = s["best"]
    print(f"trained {s['steps']} steps in {s['seconds']:.0f} s ({s['steps_per_second']} steps/s) on {s['device']}; "
          f"best held-out score {b['score']} (policy {b['policy']}, value {b['value']}) after {b['step']} steps; "
          f"parity max error {s['parity_max_error']:.2e}; wrote {out}")
    return 0


def _fast_score(args) -> int:
    if _train_missing():
        return 2
    from alphasettler.train.fast import score_net

    pool = _pool(args.records)
    for net in args.nets:
        h = score_net(pool, net, args.device)
        print(f"{net}: score {h['score']:.4f} (policy {h['policy']:.4f}, value {h['value']:.4f}, aux {h['aux']:.5f})")
    return 0


def _fast_loop(args) -> int:
    if _train_missing():
        return 2
    from alphasettler.selfplay import mode_mismatch, opening_mismatch
    from alphasettler.train.fastloop import STATE, FastLoop, FastLoopSettings

    root = Path(args.dir)
    if not (root / STATE).exists():
        if not (args.seed_dirs and args.start_net and args.first is not None):
            print("error: a new fast loop needs --seed-dirs, --start-net and --first", file=sys.stderr)
            return 2
        FastLoop.init(root, args.seed_dirs, args.start_net, args.first, trades=args.trades,
                      gumbel=args.gumbel, opening=args.opening)
    s = FastLoopSettings(generations=args.generations, games=args.games, window_games=args.window_games,
                         minutes=args.minutes, lr=args.lr, gate_seeds=args.gate_seeds,
                         gate_simulations=args.gate_simulations, device=args.device, threads=args.threads, opening=args.opening)
    loop = FastLoop(root, s)
    if args.trades and not loop.trades:
        print(f"error: {root} was set up without --trades, and a loop keeps the setting it was set up with",
              file=sys.stderr)
        return 2
    error = mode_mismatch(root, loop.gumbel, args.gumbel)
    if error is not None:
        print(f"error: {error}", file=sys.stderr)
        return 2
    error = opening_mismatch(root, loop.opening, args.opening)
    if error is not None:
        print(f"error: {error}", file=sys.stderr)
        return 2
    try:
        n = loop.run()
    except KeyboardInterrupt:
        print("interrupted: rerun the same command to resume", file=sys.stderr)
        return 130
    print(f"{n} generations run; the best net is {loop.state['best']} ({root / 'nets' / 'best.bin'})")
    return 0


def _train_loop(args) -> int:
    if _train_missing():
        return 2
    from alphasettler.train.loop import Loop, LoopSettings, gen_name

    s = LoopSettings(
        generations=args.generations, games=args.games, warm_games=args.warm_games, chunk_games=args.chunk_games,
        full_simulations=args.full_simulations, fast_simulations=args.fast_simulations,
        full_probability=args.full_probability, window=args.window, warm_lr=args.warm_lr, lr=args.lr,
        steps_factor=args.steps_factor, batch=args.batch, width=args.width, rounds=args.rounds,
        gate_seeds=args.gate_seeds, gate_simulations=args.gate_simulations, max_turns=args.max_turns,
        device=args.device, threads=args.threads,
        gumbel=args.gumbel, opening=args.opening,
    )
    loop = Loop(Path(args.dir), s)
    try:
        n = loop.run()
    except KeyboardInterrupt:
        print("interrupted: rerun the same command to resume at the first incomplete step", file=sys.stderr)
        return 130
    except RuntimeError as e:  # a RuntimeError (for example a failed parity check): the step is not recorded
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(f"{n} steps run" if n else "nothing to do: every step is complete")
    best = loop.manifest["best"]
    if best is not None:
        print(f"best net: {gen_name(best)} ({loop.nets / 'best.bin'})")
    return 0


def _net_upgrade(args) -> int:
    from alphasettler import netupgrade

    # Written to a temp file, read back by the Rust loader as every bot will read it, then renamed
    # to NEW: a file that fails any step is never left at NEW.
    s = netupgrade.upgrade_file(args.old, args.new)
    z = s["zero_inputs"]
    reset = (f"; {s['trade_rows']} trade policy rows reset (offer bias {netupgrade.OFFER_BIAS:g})"
             if s["trade_rows"] else "")
    print(f"upgraded {args.old} (encoder v{s['from']}) to encoder v4: width {s['width']}, {s['rounds']} rounds; "
          f"{z['node']} node, {z['edge']} edge and {z['global']} global inputs zeroed{reset}; wrote {args.new}")
    return 0


def _opening_train(args) -> int:
    if _train_missing():
        return 2
    from alphasettler._engine import resolve_net
    from alphasettler.train.opening import OpeningConfig, format_report, train_opening

    out = Path(args.out)
    if out.exists():
        raise ValueError(f"{out} already exists; write the opening model to a new path")
    if not out.parent.is_dir():
        raise ValueError(f"{out.parent} is not a directory; write the opening model into an existing one")
    if args.compare_net is not None:
        resolve_net(args.compare_net, args.nets_dir)  # a bad net fails now, before the games are read
    cfg = OpeningConfig(seed=args.seed)
    cfg.check()
    try:
        s = train_opening([Path(d) for d in args.games], out, cfg, args.compare_net, args.nets_dir)
    except KeyboardInterrupt:
        print("interrupted: nothing written", file=sys.stderr)
        return 130
    for line in format_report(s, out):
        print(line)
    return 0


def _opening_diagnose(args, config) -> int:
    from alphasettler import diagnose
    from alphasettler._engine import resolve_net

    threads = args.threads or os.cpu_count() or 1
    if args.bot is not None:
        if args.seeds is None:
            raise ValueError("--bot needs --seeds")
        if args.seed_start < 0 or args.seed_start + args.seeds > SEED_LIMIT:
            raise ValueError(f"seeds {args.seed_start}..{args.seed_start + args.seeds} are outside the u64 seed range")
        _check_bot_name(args.bot, args.nets_dir)
    else:
        if args.net is None:
            raise ValueError("--games needs --net NAME (the net whose value head D2 reads and whose search D3 re-runs)")
        if args.audit < 0:
            raise ValueError(f"--audit must be 0 or more, got {args.audit}")
        resolve_net(args.net, args.nets_dir)  # a bad net fails now, before any record is read
    with ExitStack() as stack:
        # Open the output before the work, so a bad path fails first.
        f = stack.enter_context(open_output(args.out)) if args.out else None
        if args.bot is not None:
            setups = diagnose.play_setups(args.bot, args.seed_start, args.seeds, config, args.nets_dir, threads)
            text = diagnose.d4_table(args.bot, setups)
        else:
            text = diagnose.diagnose([Path(d) for d in args.games], args.net, args.nets_dir, args.audit,
                                     args.audit_budgets, threads)
        print(text)
        if f is not None:
            f.write(text + "\n")
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
    """Our candidate inside Catanatron against three of its bots. `--trades` plays the trading game
    (a `+t` candidate offers, Catanatron's bots answer); otherwise ours plays with offers off, and
    `--no-trades` is moot: Catanatron's bots never offer."""
    import oracle

    if not oracle.catanatron_available():
        return _oracle_missing()
    import oracle.arena  # other import errors surface as themselves

    baseline = args.baseline.split(":", 1)[1]
    nets_dir = str(Path(args.nets_dir).resolve())
    oracle.arena.check(args.candidate, baseline, args.seeds, args.seed_start, nets_dir)
    # Open the output before the run, so a bad path fails before any game is played.
    with open_output(out) as f:
        records = oracle.arena.run_match(args.candidate, baseline, args.seeds, args.seed_start, args.threads, nets_dir,
                                         trades=args.trades)
        _write_records(f, args.candidate, args.baseline, records)
    print(format_summary(f"{args.candidate} vs {args.baseline}", summarize(records)))
    print(f"fallbacks: {sum(r['fallbacks'] for r in records)}")
    print(_resets_line(records))
    offers, accepted, confirmed = (sum(r[k] for r in records) for k in ("offers", "accepted", "confirmed"))
    print(f"trades: {offers} offers, {accepted} accepted, {confirmed} confirmed")
    print(f"wrote {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if getattr(args, "trades", False):
        config = dict(TRADES)
    elif getattr(args, "no_trades", False):
        config = {"max_offers_per_turn": 0}
    else:
        config = None
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
        elif args.command == "sprt":
            return _sprt(args, config)
        elif args.command == "oracle-diff":
            return _oracle_diff(args)
        elif args.command == "selfplay":
            return _selfplay(args)
        elif args.command == "fit-heuristic":
            return _fit_heuristic(args)
        elif args.command == "train":
            return _train(args)
        elif args.command == "fast-train":
            return _fast_train(args)
        elif args.command == "fast-score":
            return _fast_score(args)
        elif args.command == "fast-loop":
            return _fast_loop(args)
        elif args.command == "train-loop":
            return _train_loop(args)
        elif args.command == "net-upgrade":
            return _net_upgrade(args)
        elif args.command == "opening-train":
            return _opening_train(args)
        elif args.command == "opening-diagnose":
            return _opening_diagnose(args, config)
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
