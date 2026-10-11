"""Fast experiments: one candidate net trained for a fixed wall-clock budget on a fixed pool of
generations, scored on one held-out set every candidate shares.

Why: a train-loop generation takes about 6 hours, almost all of it self-play and gating, so it
cannot tell training choices apart. Here a candidate costs its budget plus a short arena match.

What differs from `train.train`, and why:
- Each step's rows are `policy_batch` policy targets (full searches, `sample_full`) and
  `value_batch` rows drawn over every decision for the value and aux heads. `train.train` draws
  every row over all decisions and trains the policy on the full searches among them (about a
  quarter) and the value on 1/16, so about 70% of each forward pass trains nothing.
- The pool is every listed generation as one training set, and the held-out set is every pooled
  game with seed % HELD_OUT_EVERY == 0, so held-out scores compare across candidates.
- The learning rate warms up linearly over `warmup` steps, then follows a cosine over the
  wall-clock budget down to MIN_LR_FRACTION of its peak.
- Batches are sampled on a background thread (the Rust sampler releases the GIL), so sampling
  overlaps the GPU step.

With `targets=2` (targets v2, spec 2026-10-11 Sections 4 and 5), the value target mixes in the
search value q and the training-only heads (`extras.Extras`) train on the value rows. They are not
in the weights file: `fast_train` writes them to the `sidecar` beside `out`, and a run with `init`
continues from `init`'s sidecar (fresh heads, with a notice, if it has none).

With `ema` (spec 2026-10-11 Section 6), the learning rate is a constant `lr` after the warmup, an
exponential moving average of the weights (EMA_DECAY per step) is what held-out evaluation scores and
what is exported (its best evaluation), and the sidecar carries the final step's raw weights, heads and
Adam state. A run with `init` continues the raw net, the heads and Adam from `init`'s sidecar (skipping
the warmup when it has Adam state) and starts the average from `init` itself, the previous average."""

from __future__ import annotations

import copy
import json
import math
import os
import queue
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from alphasettler import files
from alphasettler._engine import NET_DEFAULT_ROUNDS, NET_DEFAULT_WIDTH, NET_MAX_SAMPLE, TrainingSet
from alphasettler.train import model as model_mod
from alphasettler.train.data import Tensors, read_generation, to_tensors
from alphasettler.train.extras import Extras
from alphasettler.train.train import (
    EVAL_BATCH, _round, held_out_losses, losses, parity_check, pick_device,
)

MIN_LR_FRACTION = 0.02
# Fixed held-out positions: FAST_EVAL_BATCHES batches of EVAL_BATCH, from EVAL_SEED onwards.
FAST_EVAL_BATCHES = 20
EVAL_SEED = 0xFA57
PREFETCH = 4
# XORed into each step's seed for the value rows, so they never share a draw with the policy rows.
VALUE_SEED_SALT = 0x7A1E << 48
# `ema` (spec 2026-10-11 Section 6): the average's per-step decay (a horizon of about 5,000 steps) and
# the constant learning rate fast-loop trains at, about the mean of the cosine it replaces.
EMA_DECAY = 0.9998
EMA_LR = 2.5e-4


@dataclass(frozen=True)
class FastConfig:
    minutes: float
    lr: float = 1e-3
    warmup: int = 200
    policy_batch: int = 256
    value_batch: int = 64
    width: int = NET_DEFAULT_WIDTH
    rounds: int = NET_DEFAULT_ROUNDS
    seed: int = 0
    eval_seconds: float = 60.0
    device: str = "auto"
    threads: int | None = None
    targets: int = 1  # 2: targets v2, the z/q value mix and the training-only heads
    ema: bool = False  # export a weight average; a constant lr; raw weights and Adam carried in the sidecar

    def check(self) -> None:
        if not (math.isfinite(self.minutes) and self.minutes > 0):
            raise ValueError(f"the budget must be a positive number of minutes, got {self.minutes}")
        if not (math.isfinite(self.lr) and self.lr > 0):
            raise ValueError(f"the learning rate must be finite and positive, got {self.lr}")
        if self.warmup < 0:
            raise ValueError(f"warmup must be at least 0 steps, got {self.warmup}")
        for name in ("policy_batch", "value_batch"):
            if not 1 <= getattr(self, name) <= NET_MAX_SAMPLE:
                raise ValueError(f"{name} must be 1..={NET_MAX_SAMPLE}, got {getattr(self, name)}")
        if not (math.isfinite(self.eval_seconds) and self.eval_seconds > 0):
            raise ValueError(f"eval_seconds must be positive, got {self.eval_seconds}")
        if self.targets not in (1, 2):
            raise ValueError(f"targets must be 1 or 2, got {self.targets}")
        model_mod.check_arch(self.width, self.rounds)


def sidecar(path: str | Path) -> Path:
    """The training-only state beside the weights file `path`."""
    return Path(str(path) + ".train.pt")


def _load_sidecar(init: str | Path) -> dict | None:
    """`init`'s sidecar, or None if it has none; ValueError if it is another net's."""
    p = sidecar(init)
    if not p.exists():
        return None
    side = torch.load(p, weights_only=True)
    if side.get("bin_sha256") != files.sha256_file(init):
        raise ValueError(f"{p} is not the sidecar of {init} (sha256 mismatch)")
    return side


def _write_sidecar(out: Path, state: dict) -> None:
    """`state` with `out`'s sha256 to `sidecar(out)`, temp then rename."""
    path = sidecar(out)
    tmp = files.temp_path(path)
    try:
        torch.save({"bin_sha256": files.sha256_file(out), **state}, tmp)
        files.fsync_file(tmp)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    os.replace(tmp, path)


@dataclass
class Pool:
    train: TrainingSet
    validation: TrainingSet
    files: dict[str, str]
    notices: list[str]


def load_pool(root: str | Path, gen_dirs: list[Path]) -> Pool:
    """Every generation in `gen_dirs` as one training set, each one's held-out games in one
    validation set. ValueError if there is nothing to train on or nothing held out."""
    root = Path(root)
    train_set, validation = TrainingSet(), TrainingSet()
    notices: list[str] = []
    read: dict[str, str] = {}
    for d in gen_dirs:
        for p in read_generation(d, train_set, notices, validation):
            read[str(p.relative_to(root))] = files.sha256_file(p)
    if train_set.policy_decisions == 0:
        raise ValueError(f"no policy targets in {', '.join(str(d) for d in gen_dirs)}")
    if validation.decisions == 0:
        raise ValueError("no held-out games in the pool")
    return Pool(train_set, validation, read, notices)


def lr_at(cfg: FastConfig, step: int, elapsed: float, warm: bool = True) -> float:
    """Linear warmup over `cfg.warmup` steps, then a cosine over the budget to MIN_LR_FRACTION. With
    `cfg.ema`, the warmup (only when `warm`) then the constant `cfg.lr`."""
    if cfg.ema:
        return cfg.lr * (step + 1) / cfg.warmup if warm and step < cfg.warmup else cfg.lr
    if step < cfg.warmup:
        return cfg.lr * (step + 1) / cfg.warmup
    frac = min(1.0, elapsed / (cfg.minutes * 60))
    return cfg.lr * (MIN_LR_FRACTION + (1 - MIN_LR_FRACTION) * 0.5 * (1 + math.cos(math.pi * frac)))


def step_batch(data: TrainingSet, cfg: FastConfig, step: int, threads: int) -> dict:
    """Step `step`'s rows: `policy_batch` policy targets, then `value_batch` rows over every decision."""
    seed = (cfg.seed * 2**32 + step) % 2**64
    first = data.sample_full(cfg.policy_batch, seed, threads)
    second = data.sample(cfg.value_batch, seed ^ VALUE_SEED_SALT, threads)
    return {key: first[key] + second[key] for key in first}


def value_mask(cfg: FastConfig) -> torch.Tensor:
    return torch.cat([torch.zeros(cfg.policy_batch, dtype=torch.bool), torch.ones(cfg.value_batch, dtype=torch.bool)])


def held_out(validation: TrainingSet, device: torch.device, threads: int) -> list[Tensors]:
    return [to_tensors(validation.sample(EVAL_BATCH, EVAL_SEED + i, threads), device)
            for i in range(FAST_EVAL_BATCHES)]


class _Prefetch:
    """Batches for steps 0, 1, ... built on a daemon thread as CPU tensors."""

    def __init__(self, data: TrainingSet, cfg: FastConfig, threads: int):
        self.q: queue.Queue = queue.Queue(PREFETCH)
        self.stop = threading.Event()
        self.error: BaseException | None = None
        self.thread = threading.Thread(target=self._run, args=(data, cfg, threads), daemon=True)
        self.thread.start()

    def _run(self, data, cfg, threads):
        step = 0
        try:
            while not self.stop.is_set():
                t = to_tensors(step_batch(data, cfg, step, threads), torch.device("cpu"))
                while not self.stop.is_set():
                    try:
                        self.q.put(t, timeout=0.5)
                        break
                    except queue.Full:
                        pass
                step += 1
        except BaseException as e:  # surfaced by `get`
            self.error = e

    def get(self) -> Tensors:
        while True:
            if self.error is not None:
                raise self.error
            try:
                return self.q.get(timeout=0.5)
            except queue.Empty:
                pass

    def close(self):
        self.stop.set()
        self.thread.join()


def _to(t: Tensors, device: torch.device) -> Tensors:
    return Tensors(**{k: v.to(device, non_blocking=True) for k, v in vars(t).items()})


def _cpu(x):
    """`x` (a state dict, nested) with every tensor copied to the CPU."""
    if isinstance(x, torch.Tensor):
        return x.detach().to("cpu", copy=True)
    if isinstance(x, dict):
        return {k: _cpu(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(_cpu(v) for v in x)
    return x


def fast_train(pool: Pool, out: str | Path, cfg: FastConfig, init: str | Path | None = None) -> dict:
    """Train for `cfg.minutes` of wall clock, write the weights with the best held-out score to
    `out` (after the parity check) and its summary to `out` + ".json"; return the summary. With
    `cfg.targets == 2`, the training-only heads of that same evaluation go to `sidecar(out)`. With
    `cfg.ema`, the weights are the average's best and the sidecar holds the final step's raw net,
    heads and Adam state."""
    cfg.check()
    out = Path(out)
    device = pick_device(cfg.device)
    threads = cfg.threads or os.cpu_count() or 1
    side = _load_sidecar(init) if init is not None and (cfg.targets == 2 or cfg.ema) else None
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(cfg.seed)
        net = model_mod.load(init) if init is not None else model_mod.Net(cfg.width, cfg.rounds)
        extras = Extras(net.width) if cfg.targets == 2 else None
    average = None
    if cfg.ema:
        average = copy.deepcopy(net)  # init's weights (the previous average), or the fresh raw net
        average.requires_grad_(False)
        average.to(device).train()
        if side is not None and side.get("raw") is not None:
            net.load_state_dict(side["raw"])
    if extras is not None:
        if side is not None and side.get("extras") is not None:
            extras.load_state_dict(side["extras"])
        elif side is not None:
            print(f"notice: {sidecar(init)} has no training-only heads; the training-only heads start fresh",
                  flush=True)
        elif init is not None:
            print(f"notice: no sidecar for {init}; the training-only heads start fresh", flush=True)
        extras.to(device).train()
    net.to(device).train()
    params = list(net.parameters()) + ([] if extras is None else list(extras.parameters()))
    opt = torch.optim.Adam(params, lr=cfg.lr)
    warm = True
    if cfg.ema and side is not None and side.get("optimizer") is not None:
        try:
            opt.load_state_dict(side["optimizer"])  # its state moves to each parameter's device
        except ValueError as e:
            raise ValueError(f"{sidecar(init)}'s Adam state does not fit this run's parameters "
                             f"(trained with another targets setting?): {e}") from e
        warm = False
    scored = net if average is None else average
    evals = held_out(pool.validation, device, threads)
    rows = value_mask(cfg)
    history: list[dict] = []
    best_score, best_row, best_state, best_extras = math.inf, None, None, None

    def snapshot(m: torch.nn.Module) -> dict:
        return {k: v.detach().to("cpu", copy=True) for k, v in m.state_dict().items()}

    def evaluate(step: int, elapsed: float) -> None:
        nonlocal best_score, best_row, best_state, best_extras
        h = held_out_losses(scored, evals, extras)
        row = {"step": step, "seconds": round(elapsed, 1), **{k: _round(v) for k, v in h.items()}}
        history.append(row)
        better = h["score"] < best_score
        if better:
            best_score, best_row = h["score"], row
            best_state = snapshot(scored)
            best_extras = None if extras is None else snapshot(extras)
        print(f"held out after {step} steps ({elapsed / 60:.1f} min): score {h['score']:.4f} "
              f"(policy {h['policy']:.4f}, value {h['value']:.4f}, aux {h['aux']:.5f}){', best' if better else ''}",
              flush=True)

    budget = cfg.minutes * 60
    prefetch = _Prefetch(pool.train, cfg, threads)
    start = time.monotonic()
    next_eval = cfg.eval_seconds
    step = 0
    try:
        while True:
            elapsed = time.monotonic() - start
            if elapsed >= budget:
                break
            for g in opt.param_groups:
                g["lr"] = lr_at(cfg, step, elapsed, warm)
            parts = losses(net, _to(prefetch.get(), device), rows, extras)
            opt.zero_grad()
            parts["total"].backward()
            opt.step()
            if average is not None:
                with torch.no_grad():
                    for ema_p, p in zip(average.parameters(), net.parameters()):
                        ema_p.mul_(EMA_DECAY).add_(p.detach(), alpha=1 - EMA_DECAY)
            step += 1
            if elapsed >= next_eval:
                evaluate(step, elapsed)
                next_eval += cfg.eval_seconds
    finally:
        prefetch.close()
    elapsed = time.monotonic() - start
    evaluate(step, elapsed)
    final = None if average is None else {"extras": None if extras is None else snapshot(extras),
                                           "raw": snapshot(net), "optimizer": _cpu(opt.state_dict())}
    scored.load_state_dict(best_state)
    tmp = files.temp_path(out)
    try:
        tmp.write_bytes(model_mod.to_bytes(scored))
        err = parity_check(scored, tmp, pool.train)
        files.fsync_file(tmp)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    os.replace(tmp, out)
    if final is not None:  # before the summary: fast-loop takes `out` with its summary as finished
        _write_sidecar(out, final)
    elif extras is not None:
        _write_sidecar(out, {"extras": best_extras, "raw": None, "optimizer": None})
    summary = {"config": asdict(cfg), "init": None if init is None else str(init), "targets": cfg.targets,
               "ema": cfg.ema, "warmup_skipped": not warm, "device": device.type,
               "width": net.width, "rounds": net.rounds, "steps": step, "seconds": round(elapsed, 1),
               "steps_per_second": round(step / elapsed, 2),
               "policy_rows": step * cfg.policy_batch, "value_rows": step * cfg.value_batch,
               "pool_games": pool.train.games, "pool_policy_targets": pool.train.policy_decisions,
               "held_out_games": pool.validation.games, "best": best_row, "history": history,
               "parity_max_error": err}
    summary_path = Path(str(out) + ".json")
    summary_path.write_text(json.dumps(summary, indent=1) + "\n")
    return summary


def score_net(pool: Pool, path: str | Path, device: str = "auto", threads: int | None = None) -> dict:
    """The held-out losses of an existing weights file on `pool`'s held-out set."""
    dev = pick_device(device)
    threads = threads or os.cpu_count() or 1
    net = model_mod.load(path).to(dev)
    return held_out_losses(net, held_out(pool.validation, dev, threads))
