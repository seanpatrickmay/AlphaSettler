"""The generation loop (spec Section 4): for each generation, self-play, train, gate.

`<dir>/gen-NNN/` holds a generation's records (`selfplay-local-NNNNN.jsonl.gz`, plus any record
files other machines drop in before training), its weights (`net.bin`) and its gating result
(`gate.jsonl`, `gate.json`). `<dir>/nets/` holds every generation's weights as `gen-NNN.bin` and
the best one as `best.bin`, so `net:gen-NNN` and `net:best` resolve with `--nets-dir <dir>/nets`.
`<dir>/manifest.json` lists the completed steps with their outputs' hashes, and a rerun resumes
at the first incomplete step. Every output is written under a temporary name and renamed when
complete (after an fsync), so a reader never takes a temp for a finished file. `run` first deletes
the temps an interrupted run left of the loop's own outputs (`GEN_OUTPUTS`, `NET_OUTPUTS` and the
manifest): those whose writer was this machine and no longer runs (`files.remove_dead_temps`).
Every other temp is left alone and named in a notice: a running writer's, another machine's (its
PID means nothing here), or one whose name the loop never writes (another machine's self-play
`<stamp>-selfplay-*.jsonl.gz`). A file a completed step recorded is checked against its sha256
before a later step uses it, and `nets/best.bin` is brought back to the manifest's best net.

Generation 0 is the warm start: 3a's `ismcts` self-plays with the same exploration, net 0 is
trained from scratch and becomes the best net without a match. Generation k >= 1 self-plays with
the best net, trains from generation k-1's weights and is gated against the best net."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

from alphasettler import files, records
from alphasettler import selfplay as selfplay_mod
from alphasettler._engine import MAX_SEEDS, MAX_SIMULATIONS, NET_MAX_WIDTH, resolve_config
from alphasettler.arena import run as arena_run

MANIFEST = "manifest.json"
MANIFEST_FORMAT = 1
STEPS = ("selfplay", "train", "gate")
SEEDS_PER_GENERATION = 10_000_000
GATE_SEED_START = 1_000_000_000
DEVICES = ("auto", "mps", "cpu")
# The loop's own outputs: in each generation directory, and in `nets/`.
GEN_OUTPUTS = ("selfplay.json", "selfplay-local-*", "net.bin", "gate.jsonl", "gate.json")
NET_OUTPUTS = ("gen-*.bin", "best.bin")
# Generation k's self-play seeds are k x SEEDS_PER_GENERATION + 0..games: below the gate's seeds.
MAX_GENERATIONS = GATE_SEED_START // SEEDS_PER_GENERATION - 1
# Temp files a notice names before it elides the rest.
NOTICE_TEMPS = 5


@dataclass(frozen=True)
class LoopSettings:
    generations: int
    games: int = 10_000
    warm_games: int = 100_000
    chunk_games: int = 500
    full_simulations: int = 600
    fast_simulations: int = 100
    full_probability: float = 0.25
    window: int = 4
    warm_lr: float = 1e-3
    lr: float = 3e-4
    steps_factor: float = 2.0
    batch: int = 1024
    trunk: int = 512
    value: int = 256
    gate_seeds: int = 200
    gate_simulations: int = 600
    max_turns: int = 1000
    device: str = "auto"
    threads: int | None = None


def gen_name(k: int) -> str:
    return f"gen-{k:03d}"


def promoted(wins: int, games: int) -> bool:
    """A win rate of at least 1/4 against three copies of the best net promotes; a tie at
    exactly 0.25 promotes. Draws count as losses."""
    return 4 * wins >= games


def json_float(x: float) -> float | None:
    """`x` rounded to 6 places, or None if it is not finite: the manifest is strict JSON
    (preflight F6), and a gate over seeds with zero spread has an infinite z."""
    return round(x, 6) if math.isfinite(x) else None


def _verify(path: Path, digest: str) -> None:
    """ValueError unless `path` exists with the sha256 `digest` a completed step recorded."""
    if not path.is_file():
        raise ValueError(f"{path} is missing, but the manifest records it as written by a completed step")
    actual = files.sha256_file(path)
    if actual != digest:
        raise ValueError(f"{path} has sha256 {actual}, not the {digest} the manifest recorded: it changed after "
                         f"its step completed; restore it, or start the run again in a new directory")


def _dumps(obj, indent: int | None = None) -> str:
    return json.dumps(obj, indent=indent, sort_keys=True, allow_nan=False) + "\n"


def _check(s: LoopSettings) -> None:
    """ValueError for a setting some step would reject, before anything is written."""
    from alphasettler.train.train import TrainConfig

    if not 0 <= s.generations <= MAX_GENERATIONS:
        raise ValueError(f"--generations must be 0..={MAX_GENERATIONS} (generation k self-plays seeds from "
                         f"k x {SEEDS_PER_GENERATION}, below the gate's from {GATE_SEED_START}), got {s.generations}")
    for name in ("games", "warm_games", "chunk_games", "window", "trunk", "value", "gate_seeds",
                 "gate_simulations", "max_turns"):
        v = getattr(s, name)
        if not (isinstance(v, int) and v >= 1):
            raise ValueError(f"{name} must be a positive integer, got {v!r}")
    for name in ("games", "warm_games"):
        if getattr(s, name) > SEEDS_PER_GENERATION:
            raise ValueError(f"{name} must be at most {SEEDS_PER_GENERATION} (a generation's seed range), "
                             f"got {getattr(s, name)}")
    for name in ("trunk", "value"):
        if getattr(s, name) > NET_MAX_WIDTH:
            raise ValueError(f"{name} must be at most {NET_MAX_WIDTH}, got {getattr(s, name)}")
    if s.gate_seeds > MAX_SEEDS:
        raise ValueError(f"gate_seeds must be at most {MAX_SEEDS}, got {s.gate_seeds}")
    if s.gate_simulations > MAX_SIMULATIONS:
        raise ValueError(f"gate_simulations must be at most {MAX_SIMULATIONS}, got {s.gate_simulations}")
    if s.device not in DEVICES:
        raise ValueError(f"unknown device {s.device!r}; use {', '.join(DEVICES)}")
    # No net is loaded here: generation 0's bot is checked, later ones load when their step runs.
    selfplay_mod.settings(None, ".", s.full_simulations, s.fast_simulations, s.full_probability)
    for lr in (s.warm_lr, s.lr):
        TrainConfig(lr=lr, steps_factor=s.steps_factor, batch=s.batch, threads=s.threads).check()


class Loop:
    def __init__(self, root: str | Path, settings: LoopSettings):
        _check(settings)
        self.root = Path(root)
        self.s = settings
        self.nets = self.root / "nets"
        self.config = {"max_offers_per_turn": 0, "max_turns": settings.max_turns}
        resolve_config(self.config)  # an out-of-range turn cap fails now
        self.manifest = self._load()

    def _load(self) -> dict:
        path = self.root / MANIFEST
        if not path.exists():
            return {"format": MANIFEST_FORMAT, "best": None, "generations": []}
        m = json.loads(path.read_text())
        if m.get("format") != MANIFEST_FORMAT:
            raise ValueError(f"{path} has format {m.get('format')!r}, expected {MANIFEST_FORMAT}")
        return m

    def _save(self) -> None:
        files.atomic_write_text(self.root / MANIFEST, _dumps(self.manifest, indent=1))

    def _entry(self, k: int) -> dict:
        gens = self.manifest["generations"]
        while len(gens) <= k:
            gens.append({"gen": len(gens)})
        return gens[k]

    def run(self, stop_after: int | None = None) -> int:
        """Run every incomplete step of generations 0..=`generations` in order and return how
        many ran. With `stop_after`, stop once that many have run. A KeyboardInterrupt
        propagates; the steps recorded so far stay done and a rerun resumes after them."""
        self.root.mkdir(parents=True, exist_ok=True)
        self._clean_temps()
        self._sync_best()
        done = 0
        for k in range(self.s.generations + 1):
            for step in STEPS:
                if step in self._entry(k):
                    continue
                if stop_after is not None and done >= stop_after:
                    return done
                print(f"{gen_name(k)}: {step}", flush=True)
                result = getattr(self, f"_{step}")(k)
                self._entry(k)[step] = result
                if step == "gate" and result["promoted"]:
                    self.manifest["best"] = k
                self._save()
                done += 1
        return done

    def _clean_temps(self) -> None:
        """Delete dead local writers' temps of the loop's own outputs; name every other temp."""
        removed = files.remove_dead_temps(self.root, MANIFEST)
        for gdir in sorted(p for p in self.root.glob("gen-*") if p.is_dir()):
            for name in GEN_OUTPUTS:
                removed += files.remove_dead_temps(gdir, name)
        for name in NET_OUTPUTS:
            removed += files.remove_dead_temps(self.nets, name)
        if removed:
            print(f"notice: deleted {len(removed)} temp files an interrupted run left", flush=True)
        others = sorted(str(p.relative_to(self.root)) for p in self.root.rglob(".*.tmp-*"))
        if others:
            shown = ", ".join(others[:NOTICE_TEMPS]) + (", ..." if len(others) > NOTICE_TEMPS else "")
            print(f"notice: {len(others)} temp files of other processes left alone (a writer still running, another "
                  f"machine's, or not a loop output; delete them by hand once no writer runs): {shown}", flush=True)

    def _net(self, j: int) -> Path:
        """`nets/gen-j.bin`, checked against the hash generation j's train step recorded."""
        path = self.nets / f"{gen_name(j)}.bin"
        _verify(path, self.manifest["generations"][j]["train"]["net"])
        return path

    def _sync_best(self) -> None:
        """Make `nets/best.bin` the manifest's best net (a promotion interrupted before the manifest
        was saved, then not repeated, leaves it holding the unpromoted net)."""
        best = self.manifest["best"]
        if best is None:
            return
        src, dst = self._net(best), self.nets / "best.bin"
        if not dst.exists() or files.sha256_file(dst) != files.sha256_file(src):
            files.atomic_copy(src, dst)
            print(f"notice: {dst} restored to the best net, {gen_name(best)}", flush=True)

    def _selfplay(self, k: int) -> dict:
        gdir = self.root / gen_name(k)
        gdir.mkdir(parents=True, exist_ok=True)
        net = None if k == 0 else gen_name(self.manifest["best"])
        if net is not None:
            self._net(self.manifest["best"])
        games = self.s.warm_games if k == 0 else self.s.games
        plan = {
            "bot": "ismcts" if net is None else f"net:{net}",
            "seed_start": k * SEEDS_PER_GENERATION,
            "games": games,
            "chunk_games": self.s.chunk_games,
            "full_simulations": self.s.full_simulations,
            "fast_simulations": self.s.fast_simulations,
            "full_probability": self.s.full_probability,
            "config": self.config,
        }
        plan_path = gdir / "selfplay.json"
        if plan_path.exists():
            if json.loads(plan_path.read_text()) != plan:
                raise ValueError(f"{gdir.name}'s self-play was started with other settings ({plan_path}); "
                                 f"rerun with those settings, or delete {gdir} to start it again")
        else:
            files.atomic_write_text(plan_path, _dumps(plan))
        start, end = plan["seed_start"], plan["seed_start"] + games
        totals = {"games": 0, "decisions": 0, "full_decisions": 0, "belief_resets": 0}
        chunks = {}
        for i, lo in enumerate(range(start, end, self.s.chunk_games)):
            path = gdir / f"selfplay-local-{i:05d}.jsonl.gz"
            if not path.exists():
                # An interrupt leaves the temp (a readable prefix of the chunk); the rerun deletes
                # it and plays the whole chunk again.
                tmp = files.temp_path(path)
                selfplay_mod.write_explore(
                    tmp, lo, min(self.s.chunk_games, end - lo), config=self.config,
                    full_simulations=self.s.full_simulations, fast_simulations=self.s.fast_simulations,
                    full_probability=self.s.full_probability, net=net, nets_dir=self.nets, threads=self.s.threads)
                files.fsync_file(tmp)
                os.replace(tmp, path)
                print(f"  {path.name}: seeds {lo}..{min(end, lo + self.s.chunk_games)}", flush=True)
            for g in records.read(path):
                totals["games"] += 1
                totals["decisions"] += len(g["decisions"])
                totals["full_decisions"] += sum(d["full_search"] for d in g["decisions"])
                totals["belief_resets"] += g["belief_resets"]
            chunks[path.name] = files.sha256_file(path)
        print(f"  {totals['games']} games, {totals['decisions']} decisions, belief resets {totals['belief_resets']}",
              flush=True)
        return {**plan, **totals, "chunks": chunks}

    def _train(self, k: int) -> dict:
        from alphasettler.train.data import HELD_OUT_EVERY, load_window
        from alphasettler.train.train import TrainConfig, train

        gens = range(max(0, k - self.s.window + 1), k + 1)
        dirs = [self.root / gen_name(j) for j in gens]
        for j, d in zip(gens, dirs):
            for name, digest in self.manifest["generations"][j]["selfplay"]["chunks"].items():
                _verify(d / name, digest)
        init = None
        if k > 0:
            init = self.root / gen_name(k - 1) / "net.bin"
            _verify(init, self.manifest["generations"][k - 1]["train"]["net"])
        window = load_window(self.root, dirs)
        for line in window.notices:
            print(f"notice: {line}", flush=True)
        gdir = self.root / gen_name(k)
        cfg = TrainConfig(lr=self.s.warm_lr if k == 0 else self.s.lr, steps_factor=self.s.steps_factor,
                          batch=self.s.batch, seed=k, device=self.s.device, threads=self.s.threads,
                          trunk=self.s.trunk, value=self.s.value)
        summary = train(window, gdir / "net.bin", cfg, init)  # temp-then-rename after the parity check
        self.nets.mkdir(parents=True, exist_ok=True)
        files.atomic_copy(gdir / "net.bin", self.nets / f"{gen_name(k)}.bin")
        return {
            "window": [d.name for d in dirs],
            "window_size": self.s.window,
            "steps_factor": self.s.steps_factor,
            "inputs": window.files,
            "games": window.data.games,
            "positions": window.data.decisions,
            "new_positions": window.new_positions,
            "held_out_every": HELD_OUT_EVERY,
            "validation_games": window.validation.games,
            "validation_positions": window.validation.decisions,
            "init": None if k == 0 else gen_name(k - 1),
            "lr": cfg.lr,
            "steps": summary["steps"],
            "batch": cfg.batch,
            "device": summary["device"],
            "history": summary["history"],
            "held_out": summary["held_out"],
            "best_step": summary["best_step"],
            "best_held_out": summary["best_held_out"],
            "parity_max_error": summary["parity_max_error"],
            "net": files.sha256_file(gdir / "net.bin"),
        }

    def _gate(self, k: int) -> dict:
        new = self._net(k)
        if k == 0:
            files.atomic_copy(new, self.nets / "best.bin")
            return {"promoted": True, "reason": "warm start: net 0 becomes the best net"}
        best = self.manifest["best"]
        self._net(best)
        gdir = self.root / gen_name(k)
        sims = self.s.gate_simulations
        candidate, baseline = f"net:{gen_name(k)}@{sims}", f"net:{gen_name(best)}@{sims}"
        tmp = files.temp_path(gdir / "gate.jsonl")
        recs, summary = arena_run(candidate, baseline, self.s.gate_seeds, GATE_SEED_START, self.s.threads,
                                  self.config, tmp, self.nets)
        files.fsync_file(tmp)
        os.replace(tmp, gdir / "gate.jsonl")
        result = {
            "candidate": candidate,
            "baseline": baseline,
            "seeds": self.s.gate_seeds,
            "seed_start": GATE_SEED_START,
            "config": self.config,
            "wins": summary.wins,
            "games": summary.games,
            "draws": summary.draws,
            "win_rate": json_float(summary.win_rate),
            "z": json_float(summary.z_vs_null),
            "belief_resets": sum(r["belief_resets"] for r in recs),
            "promoted": promoted(summary.wins, summary.games),
            "records": files.sha256_file(gdir / "gate.jsonl"),
        }
        files.atomic_write_text(gdir / "gate.json", _dumps(result, indent=1))
        if result["promoted"]:
            files.atomic_copy(new, self.nets / "best.bin")
        else:
            self._sync_best()  # an earlier, interrupted run may have promoted this net
        verdict = "promoted" if result["promoted"] else "not promoted; the best net stays"
        print(f"  {candidate} vs {baseline}: {summary.wins} wins of {summary.games} "
              f"(win rate {summary.win_rate:.3f}, z {summary.z_vs_null:.2f}); {verdict}", flush=True)
        return result
