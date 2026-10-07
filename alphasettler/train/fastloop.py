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
gate sides are `+t`. A resumed loop keeps the setting it was set up with."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from alphasettler import files, records
from alphasettler import selfplay as selfplay_mod
from alphasettler.arena import TRADES, run as arena_run
from alphasettler.train.loop import GATE_SEED_START, SEEDS_PER_GENERATION, gen_name, json_float

STATE = "state.json"
GATE_MIN_Z = -1.0


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
        rules = dict(TRADES) if self.trades else {"max_offers_per_turn": 0}
        self.config = {**rules, "max_turns": settings.max_turns}

    @staticmethod
    def init(root: str | Path, seed_dirs: list[str | Path], start_net: str | Path, first: int,
             trades: bool = False) -> None:
        """Create `root` with `seed_dirs` (oldest first) as the starting records and `start_net`
        as the best net; generation numbering starts at `first`; `trades` plays the trading game."""
        root = Path(root)
        if (root / STATE).exists():
            raise ValueError(f"{root / STATE} exists: the loop is already set up (resume it with fast-loop)")
        nets = root / "nets"
        nets.mkdir(parents=True, exist_ok=True)
        files.atomic_copy(start_net, nets / "start.bin")
        files.atomic_copy(start_net, nets / "best.bin")
        dirs = [str(Path(d).resolve()) for d in seed_dirs]
        state = {"seed_dirs": [{"dir": d, "games": count_games(Path(d))} for d in dirs],
                 "first": first, "trades": trades, "best": "start", "latest": "start", "generations": []}
        files.atomic_write_text(root / STATE, json.dumps(state, indent=1) + "\n")

    def _save(self) -> None:
        files.atomic_write_text(self.root / STATE, json.dumps(self.state, indent=1) + "\n")

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
        print(f"{name}: selfplay with net:{best}", flush=True)
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
        totals = {"games": 0, "decisions": 0, "full_decisions": 0, "belief_resets": 0}
        for i, lo in enumerate(range(start, end, self.s.chunk_games)):
            path = gdir / f"selfplay-local-{i:05d}.jsonl.gz"
            if not path.exists():
                tmp = files.temp_path(path)
                selfplay_mod.write_explore(
                    tmp, lo, min(self.s.chunk_games, end - lo), config=self.config,
                    full_simulations=self.s.full_simulations, fast_simulations=self.s.fast_simulations,
                    full_probability=self.s.full_probability, net=best, nets_dir=self.nets, threads=self.s.threads)
                files.fsync_file(tmp)
                os.replace(tmp, path)
                print(f"  {path.name}: seeds {lo}..{min(end, lo + self.s.chunk_games)}", flush=True)
            for g in records.read(path):
                totals["games"] += 1
                totals["decisions"] += len(g["decisions"])
                totals["full_decisions"] += sum(d["full_search"] for d in g["decisions"])
                totals["belief_resets"] += g["belief_resets"]
        print(f"  {totals['games']} games, {totals['decisions']} decisions, belief resets {totals['belief_resets']}",
              flush=True)
        return {"net": best, "seed_start": start, **totals}

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
        offers = "+t" if self.trades else ""
        candidate, baseline = f"net:{gen_name(k)}@{sims}{offers}", f"net:{best}@{sims}{offers}"
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
