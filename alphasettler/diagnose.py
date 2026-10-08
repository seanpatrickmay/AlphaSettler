"""The opening diagnosis (spec 3d-1 Section 3), printed as Markdown tables.

An opening is a seat's two setup settlements. Its group is the first of GROUPS that applies; its
pips are the dice dots of the tiles its settlements touch (a tile two of them touch counts twice).
- D1, forced or chosen: for each weak opening, at each of its two picks, the chosen spot's pips
  against the most pips of a spot legal at that pick touching the missing resource (wood or brick
  for "no wood and brick"); per group, the share of openings where such a spot was within 0..3
  pips at either pick, and the share with none at all.
- D2, misjudged: a net's win probability for each seat right after the setup (before the first
  roll), from that seat's own view, against the seat's result, per group. Games without a winner
  are left out.
- D3, search budget: the setup settlements of games drawn from one directory, re-chosen from the
  recorded position by the net at larger budgets (a fresh bot: setup positions hide nothing).
- D4: a bot's own openings over N setups (four copies of the bot; only the setup is played).
No torch here: the standard library and the extension only."""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from alphasettler import records
from alphasettler._engine import Bot, Game, net_adjacency, net_evaluate, resolve_config
from alphasettler.opening import SETTLEMENT_BASE, SETUP_ACTIONS, Setup, replay_setup

SEATS = 4
RESOURCES = ("wood", "brick", "sheep", "wheat", "ore")
# Dice dots per number token.
DOTS = {2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 8: 5, 9: 4, 10: 3, 11: 2, 12: 1}
NODE_TILES = net_adjacency()["node_tiles"]
GROUPS = ("no wood and brick", "no wheat", "no ore", "no wood or no brick", "no sheep", "all five")
WEAK = GROUPS[:3]
MISSING = {"no wood and brick": ("wood", "brick"), "no wheat": ("wheat",), "no ore": ("ore",)}
# D1's columns: a spot touching the missing resource within this many pips of the chosen one.
WITHIN = (0, 1, 2, 3)
AUDIT_BUDGETS = (1000, 8000)
AUDIT_SEED = 0
# The verdict rule (spec Section 3).
FORCED_SHARE = 1 / 3
FORCED_WITHIN = 1
MISJUDGED_SES = 2.0


def node_pips(board: dict, node: int) -> dict[str, int]:
    out = dict.fromkeys(RESOURCES, 0)
    for t in NODE_TILES[node]:
        r = board["tile_resource"][t]
        if r is not None:
            out[r] += DOTS[board["tile_number"][t]]
    return out


def spot_pips(board: dict, node: int) -> int:
    return sum(node_pips(board, node).values())


def touched(board: dict, nodes: Sequence[int]) -> set[str]:
    return {r for n in nodes for r, p in node_pips(board, n).items() if p > 0}


def group(board: dict, nodes: Sequence[int]) -> str:
    t = touched(board, nodes)
    if "wood" not in t and "brick" not in t:
        return "no wood and brick"
    if "wheat" not in t:
        return "no wheat"
    if "ore" not in t:
        return "no ore"
    if "wood" not in t or "brick" not in t:
        return "no wood or no brick"
    if "sheep" not in t:
        return "no sheep"
    return "all five"


def opening_pips(board: dict, nodes: Sequence[int]) -> int:
    return sum(spot_pips(board, n) for n in nodes)


def gap(setup: Setup, seat: int) -> int | None:
    """D1 for `seat`'s weak opening: the smallest, over its two picks, of the chosen spot's pips
    minus the most pips of a spot legal at that pick touching a missing resource; None if no such
    spot was legal at either pick."""
    missing = MISSING[group(setup.board, setup.nodes(seat))]
    gaps = []
    for p in setup.picks:
        if p.seat != seat:
            continue
        alternatives = [spot_pips(setup.board, n) for n in p.legal
                        if any(node_pips(setup.board, n)[r] > 0 for r in missing)]
        if alternatives:
            gaps.append(spot_pips(setup.board, p.node) - max(alternatives))
    return min(gaps) if gaps else None


class Diagnosis:
    """D1 and D2 gathered a game at a time, so the records are never all in memory. `value(game,
    seat)` gives four win probabilities by absolute seat; None skips D2."""

    def __init__(self, value: Callable[[Game, int], list[float]] | None):
        self.value = value
        self.gaps: dict[str, list[int | None]] = {g: [] for g in WEAK}
        self.calibration: dict[str, list[tuple[float, bool]]] = {g: [] for g in GROUPS}
        self.games = self.no_winner = self.seats = 0

    def add(self, setup: Setup) -> None:
        self.games += 1
        self.no_winner += setup.winner is None
        for seat in range(SEATS):
            self.seats += 1
            g = group(setup.board, setup.nodes(seat))
            if g in self.gaps:
                self.gaps[g].append(gap(setup, seat))
            if setup.winner is not None and self.value is not None:
                self.calibration[g].append((self.value(setup.after, seat)[seat], setup.winner == seat))


def _share(k: int, n: int) -> str:
    return f"{k / n:.3f}" if n else "-"


def _d1_row(label: str, gaps: list[int | None]) -> str:
    n = len(gaps)
    cells = [_share(sum(x is not None and x <= d for x in gaps), n) for d in WITHIN]
    cells.append(_share(sum(x is None for x in gaps), n))
    return f"| {label} | {n} | " + " | ".join(cells) + " |"


def d1_table(gaps: dict[str, list[int | None]]) -> str:
    lines = ["D1, forced or chosen: weak openings where a spot touching the missing resource was legal within "
             "d pips of the chosen spot, at either pick", "",
             "| group | openings | " + " | ".join(f"within {d}" for d in WITHIN) + " | none |",
             "|---|---|" + "---|" * (len(WITHIN) + 1)]
    lines += [_d1_row(g, gaps[g]) for g in WEAK]
    lines.append(_d1_row("weak groups", [x for g in WEAK for x in gaps[g]]))
    return "\n".join(lines)


def calibration(pairs: list[tuple[float, bool]]) -> tuple[int, float, float, float, float] | None:
    """(seats, mean predicted, actual win rate, gap = predicted - actual, the gap's standard error);
    None for no seats. One seat's standard error is inf."""
    n = len(pairs)
    if n == 0:
        return None
    predicted = sum(p for p, _ in pairs) / n
    actual = sum(w for _, w in pairs) / n
    mean = predicted - actual
    if n == 1:
        return n, predicted, actual, mean, math.inf
    var = sum((p - w - mean) ** 2 for p, w in pairs) / (n - 1)
    return n, predicted, actual, mean, math.sqrt(var / n)


def _d2_row(label: str, pairs: list[tuple[float, bool]]) -> str:
    c = calibration(pairs)
    if c is None:
        return f"| {label} | 0 | - | - | - | - |"
    n, predicted, actual, g, se = c
    return f"| {label} | {n} | {predicted:.3f} | {actual:.3f} | {g:+.3f} | {se:.3f} |"


def d2_table(cal: dict[str, list[tuple[float, bool]]]) -> str:
    lines = ["D2, misjudged: the net's win probability for each seat after the setup, from its own view, "
             "against its result (decided games)", "",
             "| group | seats | predicted | actual | gap | se |", "|---|---|---|---|---|---|"]
    lines += [_d2_row(g, cal[g]) for g in GROUPS]
    lines.append(_d2_row("weak groups", [x for g in WEAK for x in cal[g]]))
    return "\n".join(lines)


def verdict(gaps: dict[str, list[int | None]], cal: dict[str, list[tuple[float, bool]]]) -> str:
    """Spec Section 3's rule over the three weak groups pooled."""
    weak = [x for g in WEAK for x in gaps[g]]
    share = sum(x is not None and x <= FORCED_WITHIN for x in weak) / len(weak) if weak else None
    if share is not None and share < FORCED_SHARE:
        return (f"verdict: forced (a spot within {FORCED_WITHIN} pip touching the missing resource in {share:.3f} "
                f"of {len(weak)} weak openings, under 1/3)")
    w, a = calibration([x for g in WEAK for x in cal[g]]), calibration(cal["all five"])
    if share is None or w is None or a is None:
        return "verdict: undecided (no weak openings, or no decided weak or all-five seats)"
    diff, se = w[3] - a[3], math.sqrt(w[4] ** 2 + a[4] ** 2)
    ses = diff / se if se > 0 else 0.0 if diff == 0 else math.copysign(math.inf, diff)
    near = f"a spot within {FORCED_WITHIN} pip in {share:.3f} of {len(weak)} weak openings"
    if diff > MISJUDGED_SES * se:
        return (f"verdict: misjudged ({near}; the weak groups' gap {w[3]:+.3f} exceeds all five's {a[3]:+.3f} "
                f"by {ses:.2f} standard errors)")
    return (f"verdict: misdecided ({near}; the weak groups' gap {w[3]:+.3f} is {ses:.2f} standard errors from "
            f"all five's {a[3]:+.3f}, not more than {MISJUDGED_SES:g})")


def rechoose(name: str, nets_dir: str, setup: Setup, k: int) -> int:
    """The node a fresh `name` bot picks at `setup`'s pick `k` (its position kept by `replay_setup`)."""
    bot = Bot(name, (setup.seed * 8 + k) % 2**64, nets_dir)
    return bot.act(setup.picks[k].before) - SETTLEMENT_BASE


def weak_share(setups: list[Setup], nodes: list[list[int]]) -> float:
    """The share of seats whose opening is weak, with `nodes[i]` replacing setup i's picked nodes."""
    weak = seats = 0
    for s, chosen in zip(setups, nodes):
        for seat in range(SEATS):
            weak += group(s.board, [n for p, n in zip(s.picks, chosen) if p.seat == seat]) in WEAK
            seats += 1
    return weak / seats if seats else math.nan


def audit(setups: list[Setup], net: str, nets_dir: str, budgets: Sequence[int], threads: int
          ) -> list[tuple[str, int, float | None, float]]:
    """D3 rows (budget, picks, share changed, weak share of the re-chosen openings), after a
    `recorded` row for the openings as played. Each pick is re-chosen from its recorded position."""
    tasks = [(s, k) for s in setups for k in range(len(s.picks))]
    rows: list[tuple[str, int, float | None, float]] = [
        ("recorded", len(tasks), None, weak_share(setups, [[p.node for p in s.picks] for s in setups]))]
    with ThreadPoolExecutor(threads) as ex:
        for b in budgets:
            name = f"net:{net}@{b}"
            chosen = list(ex.map(lambda t: rechoose(name, nets_dir, t[0], t[1]), tasks))
            per_setup, at = [], 0
            for s in setups:
                per_setup.append(chosen[at:at + len(s.picks)])
                at += len(s.picks)
            changed = sum(c != s.picks[k].node for c, (s, k) in zip(chosen, tasks))
            rows.append((str(b), len(tasks), changed / len(tasks) if tasks else math.nan,
                         weak_share(setups, per_setup)))
    return rows


def d3_table(rows: list[tuple[str, int, float | None, float]], games: int, net: str, source: Path) -> str:
    lines = [f"D3, search budget: the setup settlements of {games} games from {source}, re-chosen from the "
             f"recorded position by net:{net}@budget (a fresh bot each)", "",
             "| budget | picks | changed | weak-group openings |", "|---|---|---|---|"]
    for budget, picks, changed, weak in rows:
        shown = "-" if changed is None else f"{changed:.3f}"
        lines.append(f"| {budget} | {picks} | {shown} | {weak:.3f} |")
    return "\n".join(lines)


def play_setup(bot: str, seed: int, config: dict | None, nets_dir: str) -> Setup:
    """Seed `seed`'s setup played by four copies of `bot`, each shown its own events."""
    g = Game(seed, config)
    bots = [Bot(bot, (seed * SEATS + p) % 2**64, nets_dir) for p in range(SEATS)]
    seen = [0] * SEATS
    actions = []
    for _ in range(SETUP_ACTIONS):
        p = g.current_actor
        log = g.log(p)
        bots[p].observe(p, log[seen[p]:])
        seen[p] = len(log)
        a = bots[p].act(g)
        g.apply(a)
        actions.append(a)
    return replay_setup({"seed": seed, "config": resolve_config(config), "actions": actions, "winner": None})


def play_setups(bot: str, seed_start: int, seeds: int, config: dict | None, nets_dir: str, threads: int
                ) -> list[Setup]:
    """D4's setups, seeds `seed_start` onwards; the result does not depend on `threads`."""
    with ThreadPoolExecutor(threads) as ex:
        return list(ex.map(lambda s: play_setup(bot, s, config, nets_dir), range(seed_start, seed_start + seeds)))


def d4_table(bot: str, setups: list[Setup]) -> str:
    counts = dict.fromkeys(GROUPS, 0)
    pips = seats = 0
    for s in setups:
        for seat in range(SEATS):
            nodes = s.nodes(seat)
            counts[group(s.board, nodes)] += 1
            pips += opening_pips(s.board, nodes)
            seats += 1
    mean = f"{pips / seats:.2f}" if seats else "-"
    lines = [f"D4, {bot}: the openings of {len(setups)} setups ({seats} seats), mean pips {mean}", "",
             "| group | seats | share |", "|---|---|---|"]
    lines += [f"| {g} | {counts[g]} | {_share(counts[g], seats)} |" for g in GROUPS]
    weak = sum(counts[g] for g in WEAK)
    lines.append(f"| weak groups | {weak} | {_share(weak, seats)} |")
    return "\n".join(lines)


def _files(dirs: Sequence[Path]) -> list[Path]:
    out = []
    for d in dirs:
        found = records.record_files(d)
        if not found:
            raise ValueError(f"no record files (*.jsonl.gz) in {d}")
        out += found
    return out


def _replay(path: Path, rec: dict, **kwargs) -> Setup:
    try:
        return replay_setup(rec, **kwargs)
    except ValueError as e:
        raise ValueError(f"{path}: {e}") from None


def diagnose(dirs: Sequence[Path], net: str, nets_dir: str, audit_games: int, budgets: Sequence[int],
             threads: int) -> str:
    """D1 and D2 over every game in `dirs`, D3 over `audit_games` games of the last directory (none
    for 0), and the verdict, as one Markdown report."""
    d = Diagnosis(lambda g, seat: net_evaluate(g, seat, net, nets_dir))
    for path in _files(dirs):
        for rec in records.read(path):
            d.add(_replay(path, rec))
    parts = [f"{d.games} games ({d.no_winner} without a winner), {d.seats} openings, from "
             f"{', '.join(str(x) for x in dirs)}; net {net}", d1_table(d.gaps), d2_table(d.calibration)]
    if audit_games > 0:
        recs = sorted(((p, r) for p in _files(dirs[-1:]) for r in records.read(p)), key=lambda pr: pr[1]["seed"])
        drawn = random.Random(AUDIT_SEED).sample(recs, min(audit_games, len(recs)))
        setups = [_replay(p, r, keep_positions=True) for p, r in drawn]
        parts.append(d3_table(audit(setups, net, nets_dir, budgets, threads), len(setups), net, dirs[-1]))
    parts.append(verdict(d.gaps, d.calibration))
    return "\n\n".join(parts)
