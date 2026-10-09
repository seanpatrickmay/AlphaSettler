"""The fast loop: small generations trained with `fast.fast_train`.

The train-loop's generations take about 6 hours of self-play and an 800-game gate that cannot
see a one-generation gain (gens 2, 3 and 5 failed it while held-out loss kept falling). Here a
generation is `games` self-play games with the best net, `minutes` of training continued from the
previous candidate on the most recent `window_games` games, and a cheap gate whose job is to stop
regressions, not to prove gains: the candidate is promoted unless it scores z <= GATE_MIN_Z
against three copies of the best net.

`<dir>/state.json` names the seed generations (record directories made elsewhere, such as
runs/3b-v2's), the best and latest nets and each finished generation's results. `<dir>/nets/`
holds `start.bin` (the starting net), `gen-NNN.bin` and `best.bin`. Generation k's records go to
`<dir>/gen-NNN/` and its self-play seeds start at k x SEEDS_PER_GENERATION, as in the train-loop,
so numbering on from the seed generations keeps every seed unique. Every step is resumable: a
finished chunk, net or gate result is a file written temp-then-rename, and state.json records each
finished generation.

With trades (set when the loop is set up: `init(..., trades=True)`, `fast-loop --trades`), self-play
and the gate play the trading game (spec 3c Section 5): every self-play seat may offer, and both
gate sides are `+t`. A resumed loop keeps the setting it was set up with.

With `gumbel` (set when the loop is set up: `init(..., gumbel=SIMS)`, `fast-loop --gumbel SIMS`;
spec P2 Section 5), every self-play move is a Gumbel root search of SIMS simulations whose improved
policy is the decision's policy target; the gate is unchanged. A resumed loop keeps that too.

With `opening` (set when the loop is set up: `init(..., opening=FILE)`, `fast-loop --opening FILE`;
spec 3d-1b), the opening model FILE is copied to `nets/opening.bin`, picks every self-play setup
settlement, and both gate sides are `+o`. A resumed loop keeps it (state.json holds its sha256, and
the copy is checked before every self-play and gate, and a missing one is restored from a
resume's `--opening FILE` with that sha256, as train-loop does); it cannot be combined with `gumbel`.

With `league` (set when the loop is set up: `init(..., league=[NET, ...])`, `fast-loop --league NET
[NET ...]`; spec 3d-3), the set-up nets are copied to `nets/league-NNN.bin` and each self-play game is,
with probability 1/2, a league game: two learner seats and two pool bots, a past net (a set-up net, or
a generation this loop promoted before this one) and a scripted style (`style:roads`, `army`,
`production`), at the fast budget; only the learner seats' decisions are recorded. The gate is
unchanged. A resumed loop keeps its pool (state.json holds each set-up net's sha256, and the copies
are checked before every self-play); it cannot be combined with `gumbel`.

With `compound` (set when the loop is set up: `init(..., compound=True)`, `fast-loop --compound`;
spec 3d-4), self-play searches compound build moves and both gate sides are `+c`. A resumed loop
keeps it; it cannot be combined with `gumbel`. League seats keep their pool names (no `+c`).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from alphasettler import files, records
from alphasettler import selfplay as selfplay_mod
from alphasettler._engine import LEAGUE_STYLES, OPENING_FILE, SELFPLAY_FAST_SIMULATIONS
from alphasettler.arena import TRADES, run as arena_run
from alphasettler.train.loop import GATE_SEED_START, SEEDS_PER_GENERATION, gen_name, json_float

STATE = "state.json"
GATE_MIN_Z = -1.0
LEAGUE_NET_PREFIX = "league-"


@dataclass(frozen=True)
class FastLoopSettings:
    generations: int  # the last generation number to run
    games: int = 2000
    chunk_games: int = 500
    full_simulations: int = 600
    fast_simulations: int = 100
    full_probability: float = 0.25
    window_games: int = 40_000
    minutes: float = 10.0
    lr: float = 5e-4
    warmup: int = 100
    value_batch: int = 256
    gate_seeds: int = 100
    gate_simulations: int = 100
    max_turns: int = 1000
    device: str = "auto"
    threads: int | None = None
    opening: str | None = None  # an opening model file: restores a missing nets/opening.bin (not a new model)
    league_simulations: int = SELFPLAY_FAST_SIMULATIONS  # league seats' budget (spec 3d-3: the fast budget)


def promoted(z: float) -> bool:
    return z > GATE_MIN_Z


def count_games(gen_dir: Path) -> int:
    """Games in a record directory's finished record files."""
    total = 0
    for path in sorted(gen_dir.glob("*.jsonl.gz")):
        if not path.name.startswith("."):
            total += sum(1 for _ in records.read(path))
    return total


class FastLoop:
    def __init__(self, root: str | Path, settings: FastLoopSettings):
        self.root = Path(root).resolve()  # the seed directories are recorded absolute
        self.s = settings
        self.nets = self.root / "nets"
        self.state = json.loads((self.root / STATE).read_text())
        self.trades = bool(self.state.get("trades", False))  # state files from before 3c have none
        self.gumbel = self.state.get("gumbel")  # state files from before P2 have none: PUCT self-play
        self.opening = self.state.get("opening")  # the model's sha256; state files from before 3d-1b have none
        self.league = self.state.get("league")  # the set-up pool; state files from before 3d-3 have none
        self.compound = bool(self.state.get("compound", False))  # state files from before 3d-4 have none
        rules = dict(TRADES) if self.trades else {"max_offers_per_turn": 0}
        self.config = {**rules, "max_turns": settings.max_turns}

    @staticmethod
    def init(root: str | Path, seed_dirs: list[str | Path], start_net: str | Path, first: int,
             trades: bool = False, gumbel: int | None = None, opening: str | Path | None = None,
             league: list[str | Path] | None = None, compound: bool = False) -> None:
        """Create `root` with `seed_dirs` (oldest first) as the starting records and `start_net`
        as the best net; generation numbering starts at `first`; `trades` plays the trading game;
        `gumbel` (simulations) makes every self-play move a Gumbel root search; `opening` (an
        opening model file, copied to `nets/opening.bin`) picks self-play's setup settlements;
        `league` (net files, copied to `nets/league-NNN.bin` in order) makes about half the self-play
        games league games against them, the nets this loop promotes and the three styles. `compound`
        searches compound build moves in self-play and gates (league seats keep their pool names)."""
        root = Path(root)
        if (root / STATE).exists():
            raise ValueError(f"{root / STATE} exists: the loop is already set up (resume it with fast-loop)")
        if gumbel is not None:
            selfplay_mod.check_gumbel(gumbel)
        selfplay_mod.check_compound(compound, gumbel)
        opening_sha256 = None if opening is None else selfplay_mod.check_opening(opening, gumbel)
        league_sha256 = None if league is None else selfplay_mod.check_league_nets(league, gumbel)
        nets = root / "nets"
        nets.mkdir(parents=True, exist_ok=True)
        files.atomic_copy(start_net, nets / "start.bin")
        files.atomic_copy(start_net, nets / "best.bin")
        if opening is not None:
            files.atomic_copy(opening, nets / OPENING_FILE)
        pool = None
        if league is not None:
            pool = {"nets": []}
            for i, (src, digest) in enumerate(zip(league, league_sha256)):
                name = f"{LEAGUE_NET_PREFIX}{i:03d}"
                files.atomic_copy(src, nets / f"{name}.bin")
                pool["nets"].append({"name": name, "source": str(Path(src).resolve()), "sha256": digest})
        dirs = [str(Path(d).resolve()) for d in seed_dirs]
        state = {"seed_dirs": [{"dir": d, "games": count_games(Path(d))} for d in dirs],
                 "first": first, "trades": trades, "gumbel": gumbel, "opening": opening_sha256, "league": pool,
                 "compound": compound, "best": "start", "latest": "start", "generations": []}
        files.atomic_write_text(root / STATE, json.dumps(state, indent=1) + "\n")

    def _save(self) -> None:
        files.atomic_write_text(self.root / STATE, json.dumps(self.state, indent=1) + "\n")

    def _opening_file(self) -> Path | None:
        """`nets/opening.bin` if the loop was set up with an opening model, restored from
        `--opening FILE` when missing and checked against the sha256 it was set up with (ValueError
        if missing or changed); None without one."""
        if self.opening is None:
            return None
        selfplay_mod.place_opening(self.root, self.nets, self.opening, self.s.opening)
        return self.nets / OPENING_FILE

    def league_pool(self) -> dict | None:
        """The pool this generation's league games draw from (spec 3d-3 Section 3), as bot names: the
        set-up nets (each copy checked against the sha256 it was set up with; ValueError if missing or
        changed), then every generation this loop has promoted, oldest first, each
        `net:NAME@SIMS[+t][+o]`; and the three styles, `style:STYLE@SIMS[+t]`. None without a league."""
        if self.league is None:
            return None
        for n in self.league["nets"]:
            path = self.nets / f"{n['name']}.bin"
            if not path.is_file() or files.sha256_file(path) != n["sha256"]:
                raise ValueError(f"{path} is not the league net {self.root} was set up with (sha256 {n['sha256']}, "
                                 f"from {n['source']}): restore that file")
        names = [n["name"] for n in self.league["nets"]]
        names += [gen_name(g["gen"]) for g in self.state["generations"] if g["gate"]["promoted"]]
        sims, offers = self.s.league_simulations, "+t" if self.trades else ""
        opening = "" if self.opening is None else "+o"
        return {"nets": [f"net:{n}@{sims}{offers}{opening}" for n in names],
                "styles": [f"style:{s}@{sims}{offers}" for s in LEAGUE_STYLES]}

    def _record_dirs(self) -> list[tuple[Path, int]]:
        """Every record directory with its game count, oldest first."""
        out = [(Path(d["dir"]), d["games"]) for d in self.state["seed_dirs"]]
        out += [(self.root / gen_name(g["gen"]), g["selfplay"]["games"]) for g in self.state["generations"]]
        return out

    def window(self, new_dir: Path, new_games: int) -> list[Path]:
        """The newest directories (with `new_dir` last) holding at least `window_games` games."""
        dirs, total = [new_dir], new_games
        for d, n in reversed(self._record_dirs()):
            if total >= self.s.window_games:
                break
            dirs.insert(0, d)
            total += n
        return dirs

    def run(self) -> int:
        """Run generations until `generations`; return how many finished."""
        done = 0
        k = self.state["first"] + len(self.state["generations"])
        while k <= self.s.generations:
            self._generation(k)
            done += 1
            k += 1
        return done

    def _generation(self, k: int) -> None:
        name, gdir = gen_name(k), self.root / gen_name(k)
        gdir.mkdir(parents=True, exist_ok=True)
        for pattern in ("selfplay-local-*", "net.bin", "gate.jsonl"):
            files.remove_dead_temps(gdir, pattern)
        files.remove_dead_temps(self.nets, "*.bin")
        best = self.state["best"]
        mode = "" if self.gumbel is None else f" (Gumbel root, {self.gumbel} simulations)"
        if self.opening is not None:
            mode += " (setup settlements from the opening model)"
        if self.league is not None:
            mode += " (league)"
        if self.compound:
            mode += " (compound build moves)"
        print(f"{name}: selfplay with net:{best}{mode}", flush=True)
        sp = self._selfplay(k, gdir, best)
        print(f"{name}: train", flush=True)
        tr = self._train(k, gdir, sp["games"])
        print(f"{name}: gate", flush=True)
        gate = self._gate(k, gdir, best)
        self.state["generations"].append({"gen": k, "selfplay": sp, "train": tr, "gate": gate})
        self.state["latest"] = name
        if gate["promoted"]:
            self.state["best"] = name
            files.atomic_copy(self.nets / f"{name}.bin", self.nets / "best.bin")
        self._save()

    def _selfplay(self, k: int, gdir: Path, best: str) -> dict:
        start = k * SEEDS_PER_GENERATION
        end = start + self.s.games
        pool = self.league_pool()  # checked first: nothing is played against a changed copy
        totals = {"games": 0, "decisions": 0, "full_decisions": 0, "belief_resets": 0}
        league_games, seats = 0, ({} if pool is None else dict.fromkeys(pool["nets"] + pool["styles"], 0))
        for i, lo in enumerate(range(start, end, self.s.chunk_games)):
            path = gdir / f"selfplay-local-{i:05d}.jsonl.gz"
            if not path.exists():
                tmp = files.temp_path(path)
                selfplay_mod.write_explore(
                    tmp, lo, min(self.s.chunk_games, end - lo), config=self.config,
                    full_simulations=self.s.full_simulations, fast_simulations=self.s.fast_simulations,
                    full_probability=self.s.full_probability, net=best, nets_dir=self.nets, threads=self.s.threads,
                    gumbel=self.gumbel, opening=self._opening_file(), league=pool,
                    compound=self.compound)
                files.fsync_file(tmp)
                os.replace(tmp, path)
                print(f"  {path.name}: seeds {lo}..{min(end, lo + self.s.chunk_games)}", flush=True)
            for g in records.read(path):
                totals["games"] += 1
                totals["decisions"] += len(g["decisions"])
                totals["full_decisions"] += sum(d["full_search"] for d in g["decisions"])
                totals["belief_resets"] += g["belief_resets"]
                if "league" in g:
                    league_games += 1
                    for name in g["league"]:
                        if name is not None:
                            seats[name] += 1
        line = f"  {totals['games']} games, {totals['decisions']} decisions, belief resets {totals['belief_resets']}"
        if pool is not None:
            line += f"; {league_games} league games"
        print(line, flush=True)
        league = None if pool is None else {**pool, "games": league_games, "seats": seats}
        return {"net": best, "seed_start": start, "gumbel": self.gumbel, "opening": self.opening, "league": league,
                "compound": self.compound, **totals}

    def _train(self, k: int, gdir: Path, games: int) -> dict:
        from alphasettler.train.fast import FastConfig, fast_train, load_pool

        out = gdir / "net.bin"
        summary_path = Path(str(out) + ".json")
        if not (out.exists() and summary_path.exists()):
            dirs = self.window(gdir, games)
            pool = load_pool(Path(os.path.commonpath([str(d.parent) for d in dirs])), dirs)
            for line in pool.notices:
                print(f"notice: {line}", flush=True)
            init = self.nets / f"{self.state['latest']}.bin"
            cfg = FastConfig(minutes=self.s.minutes, lr=self.s.lr, warmup=self.s.warmup,
                             value_batch=self.s.value_batch, seed=k, device=self.s.device, threads=self.s.threads)
            s = fast_train(pool, out, cfg, init)
            b = s["best"]
            print(f"  {s['steps']} steps from {init.name} on {len(dirs)} directories ({pool.train.games} games); "
                  f"held-out score {b['score']} (policy {b['policy']}, value {b['value']})", flush=True)
        files.atomic_copy(out, self.nets / f"{gen_name(k)}.bin")
        s = json.loads(summary_path.read_text())
        return {"init": s["init"], "steps": s["steps"], "pool_games": s["pool_games"], "best": s["best"],
                "parity_max_error": s["parity_max_error"], "net": files.sha256_file(out)}

    def _gate(self, k: int, gdir: Path, best: str) -> dict:
        sims = self.s.gate_simulations
        self._opening_file()  # a +o gate reads nets/opening.bin: check it first
        suffix = (("+t" if self.trades else "") + ("+c" if self.compound else "")
                  + ("" if self.opening is None else "+o"))
        candidate, baseline = f"net:{gen_name(k)}@{sims}{suffix}", f"net:{best}@{sims}{suffix}"
        tmp = files.temp_path(gdir / "gate.jsonl")
        recs, summary = arena_run(candidate, baseline, self.s.gate_seeds, GATE_SEED_START, self.s.threads,
                                  self.config, tmp, self.nets)
        files.fsync_file(tmp)
        os.replace(tmp, gdir / "gate.jsonl")
        z = summary.z_vs_null
        ok = promoted(z)
        print(f"  {candidate} vs {baseline}: {summary.wins} wins of {summary.games} (win rate "
              f"{summary.win_rate:.3f}, z {z:.2f}); {'promoted' if ok else 'not promoted; the best net stays'}",
              flush=True)
        return {"candidate": candidate, "baseline": baseline, "seeds": self.s.gate_seeds, "wins": summary.wins,
                "games": summary.games, "win_rate": json_float(summary.win_rate), "z": json_float(z),
                "belief_resets": sum(r["belief_resets"] for r in recs), "promoted": ok}
