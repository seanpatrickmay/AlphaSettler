"""One training run (spec Section 4): Adam on batches from the Rust sampler, NEWEST_SHARE of each
drawn from the newest generation and the rest from the older ones, with the value and
aux losses on a seeded VALUE_FRACTION of each batch's rows; held-out losses on fixed batches of the
validation set every EVAL_EVERY steps and at the last; then export of the checkpoint with the best
held-out score, and a parity check of the exported file's Rust forward pass against PyTorch before
the file is put in place. Training on MPS is not bit-reproducible; on the CPU it is.

Why the value subsampling and the early stopping: about 240 positions of a game share its
outcome and every board is unique, so a value head trained on every row learns board -> winner.
On the first warm start (20,000 games) it scored a held-out value cross-entropy of 2.28, worse
than a uniform guess (1.39); on 1/16 of the rows its best was 0.93.

Why the newest share: drawn uniformly over the window, generation 1's batches would be 92% the
24.5M-decision warm start (generation 0) and 8% its own 2.1M, so each new position would be drawn
0.16 times instead of twice."""

from __future__ import annotations

import copy
import math
import os
from dataclasses import dataclass
from pathlib import Path

import torch

from alphasettler import files
from alphasettler._engine import NET_DEFAULT_ROUNDS, NET_DEFAULT_WIDTH, NET_MAX_SAMPLE, TrainingSet, net_forward
from alphasettler.train import model as model_mod
from alphasettler.train.data import HELD_OUT_EVERY, Tensors, Window, to_tensors

PARITY_POSITIONS = 100
PARITY_TOLERANCE = 1e-5
PARITY_SEED = 0x9A217
L2 = 1e-4
AUX_WEIGHT = 0.25
# Each batch row's value and aux targets are trained on with this probability.
VALUE_FRACTION = 1 / 16
# Held-out evaluation: every EVAL_EVERY steps and at the last, on EVAL_BATCHES batches of
# EVAL_BATCH positions drawn once from the validation set from EVAL_SEED.
EVAL_EVERY = 500
EVAL_BATCHES = 10
EVAL_BATCH = 1024
EVAL_SEED = 0xE7A1
# The share of each batch drawn from the newest generation when the window holds older ones.
NEWEST_SHARE = 0.5
# XORed into the older draw's seed: high bits only, so within a run it never equals a newest draw's.
OLDER_SEED_SALT = 0x01DE_5EED << 32


@dataclass(frozen=True)
class TrainConfig:
    lr: float
    steps: int | None = None  # None: `auto_steps`
    steps_factor: float = 2.0
    batch: int = 1024
    seed: int = 0
    device: str = "auto"
    threads: int | None = None  # sampler threads (the batches do not depend on it)
    width: int = NET_DEFAULT_WIDTH  # network v2's d and R, for a net trained from scratch (an `init`
    rounds: int = NET_DEFAULT_ROUNDS  # file brings its own)
    log_every: int = 200
    eval_every: int = EVAL_EVERY
    eval_batches: int = EVAL_BATCHES
    eval_batch: int = EVAL_BATCH

    def check(self) -> None:
        """ValueError for a setting no run could use, before anything is trained or written."""
        if not (math.isfinite(self.lr) and self.lr > 0):
            raise ValueError(f"the learning rate must be finite and positive, got {self.lr}")
        model_mod.check_arch(self.width, self.rounds)
        if not 1 <= self.batch <= NET_MAX_SAMPLE:
            raise ValueError(f"the batch must be 1..={NET_MAX_SAMPLE} positions, got {self.batch}")
        if self.steps is not None and self.steps < 1:
            raise ValueError(f"steps must be positive, got {self.steps}")
        if not (math.isfinite(self.steps_factor) and self.steps_factor > 0):
            raise ValueError(f"the steps factor must be finite and positive, got {self.steps_factor}")
        if self.threads is not None and self.threads < 1:
            raise ValueError(f"threads must be positive, got {self.threads}")
        if self.log_every < 1:
            raise ValueError(f"log_every must be positive, got {self.log_every}")
        if self.eval_every < 1 or self.eval_batches < 1:
            raise ValueError(f"eval_every and eval_batches must be positive, got {self.eval_every}, {self.eval_batches}")
        if not 1 <= self.eval_batch <= NET_MAX_SAMPLE:
            raise ValueError(f"the held-out batch must be 1..={NET_MAX_SAMPLE} positions, got {self.eval_batch}")


def _mps_problem() -> str | None:
    """Why MPS cannot be used here, or None if a tensor op on it works."""
    if not torch.backends.mps.is_available():
        return "MPS is not available"
    try:
        (torch.ones(1, device="mps") + 1).cpu()
    except Exception as e:  # a broken build: reported available, fails on first use
        return f"MPS is not available ({type(e).__name__}: {e})"
    return None


def pick_device(name: str) -> torch.device:
    """`cpu`, or `auto` / `mps`: MPS when it works, else the CPU with a printed notice."""
    if name == "cpu":
        return torch.device("cpu")
    if name not in ("auto", "mps"):
        raise ValueError(f"unknown device {name!r}; use auto, mps or cpu")
    problem = _mps_problem()
    if problem is None:
        return torch.device("mps")
    print(f"notice: {problem}; training on the CPU", flush=True)
    return torch.device("cpu")


def auto_steps(new_positions: int, batch: int, factor: float = 2.0, share: float = 1.0) -> int:
    """Enough steps to draw each of the newest generation's positions `factor` times on average,
    when `share` of each batch comes from it."""
    return max(1, math.ceil(factor * new_positions / (batch * share)))


def newest_share(window: Window) -> float:
    """The share of each batch drawn from the newest generation: all of it without older data."""
    return NEWEST_SHARE if window.older.decisions else 1.0


def batch_split(batch: int, has_older: bool) -> tuple[int, int]:
    """A batch's rows from the newest generation and from the older ones: round(batch x
    NEWEST_SHARE) (at least 1) and the rest, or every row from the newest without older data."""
    if not has_older:
        return batch, 0
    newest = min(batch, max(1, round(batch * NEWEST_SHARE)))
    return newest, batch - newest


def sample_batch(window: Window, batch: int, seed: int, step: int, threads: int) -> dict:
    """Step `step`'s batch: the newest generation's rows (drawn from (`seed`, `step`)) then the
    older generations' (from the same, salted with OLDER_SEED_SALT), concatenated as one sampler
    batch. Without older data it is the single draw from the newest generation."""
    newest, older = batch_split(batch, window.older.decisions > 0)
    first = window.newest.sample(newest, _seed(seed, step), threads)
    if not older:
        return first
    second = window.older.sample(older, _seed(seed, step) ^ OLDER_SEED_SALT, threads)
    return {key: first[key] + second[key] for key in first}  # "n" adds; the buffers concatenate


def _seed(seed: int, step: int) -> int:
    return (seed * 2**32 + step) % 2**64


def _mix32(x: int) -> int:
    """64 bits mixed (splitmix64's finaliser) and folded to 32: torch's CPU generator (mt19937)
    keeps only a seed's low 32 bits, so (seed, step) packed as `_seed` packs it would ignore the seed."""
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9 % 2**64
    x = (x ^ (x >> 27)) * 0x94D049BB133111EB % 2**64
    x ^= x >> 31
    return (x ^ (x >> 32)) & 0xFFFFFFFF


def value_rows(n: int, seed: int, step: int) -> torch.Tensor:
    """Which of a batch's `n` rows train the value and aux heads at `step` (bool, on the CPU):
    each with probability VALUE_FRACTION, drawn from a generator seeded by (`seed`, `step`);
    row 0 alone if the draw keeps none."""
    g = torch.Generator().manual_seed(_mix32(_seed(seed, step)))
    rows = torch.rand(n, generator=g) < VALUE_FRACTION
    if not rows.any():
        rows[0] = True
    return rows


def losses(net: model_mod.Net, t: Tensors, value_rows: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
    """Spec Section 4: policy cross-entropy over full-search rows only, value cross-entropy and
    AUX_WEIGHT x MSE(aux, final VP / 10) over the `value_rows` only (bool mask; None: every row),
    and L2 x the squared weights (biases excluded)."""
    logits, value, aux = net(t.public, t.hidden)
    # -1e9, not -inf: the target is 0 off the mask, and 0 x -1e9 is 0 where 0 x -inf is NaN.
    logp = torch.log_softmax(logits.masked_fill(~t.mask, -1e9), dim=1)
    per_position = -(t.policy * logp).sum(dim=1)
    policy = (per_position * t.full).sum() / t.full.sum().clamp(min=1.0)
    outcome, vp = t.outcome, t.vp
    if value_rows is not None:
        rows = value_rows.to(value.device)
        value, aux, outcome, vp = value[rows], aux[rows], outcome[rows], vp[rows]
    value_loss = -(outcome * torch.log_softmax(value, dim=1)).sum(dim=1).mean()
    aux_loss = torch.nn.functional.mse_loss(aux, vp)
    l2 = sum((layer.weight ** 2).sum() for layer in net.layers())
    total = policy + value_loss + AUX_WEIGHT * aux_loss + L2 * l2
    return {"policy": policy, "value": value_loss, "aux": aux_loss, "l2": l2, "total": total}


def held_out_batches(validation: TrainingSet, cfg: TrainConfig, device: torch.device, threads: int) -> list[Tensors]:
    """`cfg.eval_batches` batches of `cfg.eval_batch` positions from `validation`, drawn from
    EVAL_SEED (the same positions whatever the run's seed); none if it holds no positions."""
    if validation.decisions == 0:
        return []
    return [to_tensors(validation.sample(cfg.eval_batch, EVAL_SEED + i, threads), device)
            for i in range(cfg.eval_batches)]


def held_out_losses(net: model_mod.Net, batches: list[Tensors]) -> dict[str, float]:
    """The mean over `batches` of the policy, value and aux losses (value and aux on every row),
    and the selection score policy + value."""
    was_training = net.training
    net.eval()
    with torch.no_grad():
        parts = [losses(net, t) for t in batches]
    net.train(was_training)
    out = {key: sum(p[key].item() for p in parts) / len(parts) for key in ("policy", "value", "aux")}
    return {**out, "score": out["policy"] + out["value"]}


def _round(x: float) -> float | None:
    """Rounded to 6 places for the summary, None if not finite (the manifest is strict JSON)."""
    return round(x, 6) if math.isfinite(x) else None


def _worst(diff: torch.Tensor) -> float:
    """The largest entry, a NaN counting as infinitely wrong (`max` and `>` would let it pass)."""
    return torch.nan_to_num(diff, nan=float("inf")).max().item()


def parity_error(ref: tuple[torch.Tensor, torch.Tensor, torch.Tensor], rust: dict, mask: torch.Tensor) -> float:
    """The worst parity error between PyTorch's outputs `ref` and `net_forward`'s `rust`, each
    in units of PARITY_TOLERANCE's rule (preflight F9): the prior over legal moves (`mask`) and
    the value probabilities compare absolutely; raw logits and aux relative to max(1, |PyTorch|).
    A NaN anywhere on either side is an infinite error."""
    got = [torch.frombuffer(bytearray(rust[key]), dtype=torch.float32).view_as(r)
           for key, r in zip(("policy", "value", "aux"), ref)]
    errors = [_worst((g - r).abs() / r.abs().clamp(min=1.0)) for g, r in zip(got, ref)]

    def prior(logits: torch.Tensor) -> torch.Tensor:
        return torch.softmax(logits.double().masked_fill(~mask, float("-inf")), dim=1)

    errors.append(_worst((prior(got[0]) - prior(ref[0])).abs()))
    errors.append(_worst((torch.softmax(got[1].double(), dim=1) - torch.softmax(ref[1].double(), dim=1)).abs()))
    return max(errors)


def parity_check(net: model_mod.Net, path: str | Path, window: Window) -> float:
    """The parity error (`parity_error`) between the Rust forward pass of the file at `path` and
    `net` (on the CPU) over `PARITY_POSITIONS` sampled positions; RuntimeError above
    `PARITY_TOLERANCE`."""
    batch = window.newest.sample(PARITY_POSITIONS, PARITY_SEED, 1)
    cpu = copy.deepcopy(net).to("cpu").eval()
    t = to_tensors(batch, torch.device("cpu"))
    with torch.no_grad():
        ref = cpu(t.public, t.hidden)
    worst = parity_error(ref, net_forward(str(path), batch["public"], batch["hidden"]), t.mask)
    if not worst <= PARITY_TOLERANCE:  # NaN-safe
        raise RuntimeError(f"the exported weights disagree with PyTorch: max error {worst:.3g} > {PARITY_TOLERANCE}")
    return worst


def train(window: Window, out: str | Path, cfg: TrainConfig, init: str | Path | None = None) -> dict:
    """Train from `init` (a weights file) or from scratch, write `out` atomically after the
    parity check, and return a summary. `out` holds the weights at the held-out evaluation with
    the lowest score (the first, on a tie), or the last step's if the window has no held-out
    positions (a printed notice) or no evaluation scored a finite number. The summary's steps
    count from 0, as `history`'s do. The caller's torch RNG state is left as it was."""
    cfg.check()
    out = Path(out)
    device = pick_device(cfg.device)
    threads = cfg.threads or os.cpu_count() or 1
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(cfg.seed)
        net = model_mod.load(init) if init is not None else model_mod.Net(cfg.width, cfg.rounds)
    net.to(device).train()
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    share = newest_share(window)
    steps = cfg.steps or auto_steps(window.new_positions, cfg.batch, cfg.steps_factor, share)
    eval_batches = held_out_batches(window.validation, cfg, device, threads)
    if not eval_batches:
        print(f"notice: no held-out positions (the newest generation has no game with seed % {HELD_OUT_EVERY} == 0); "
              "early stopping is off and the last step's weights are exported", flush=True)
    history, held_out = [], []
    best_score, best_row, best_state = math.inf, None, None
    for step in range(steps):
        batch = sample_batch(window, cfg.batch, cfg.seed, step, threads)
        parts = losses(net, to_tensors(batch, device), value_rows(batch["n"], cfg.seed, step))
        opt.zero_grad()
        parts["total"].backward()
        opt.step()
        if step % cfg.log_every == 0 or step == steps - 1:
            row = {"step": step, **{k: round(v.detach().item(), 6) for k, v in parts.items()}}
            history.append(row)
            print(f"step {step + 1}/{steps}: total {row['total']:.4f}, policy {row['policy']:.4f}, "
                  f"value {row['value']:.4f}, aux {row['aux']:.5f}", flush=True)
        if eval_batches and (step % cfg.eval_every == 0 or step == steps - 1):
            h = held_out_losses(net, eval_batches)
            row = {"step": step, **{k: _round(v) for k, v in h.items()}}
            held_out.append(row)
            better = h["score"] < best_score  # False for NaN
            if better:
                best_score, best_row = h["score"], row
                best_state = {k: v.detach().to("cpu", copy=True) for k, v in net.state_dict().items()}
            print(f"held out after step {step + 1}/{steps}: score {h['score']:.4f} (policy {h['policy']:.4f}, "
                  f"value {h['value']:.4f}, aux {h['aux']:.5f}){', best so far' if better else ''}", flush=True)
    if best_state is not None:
        net.load_state_dict(best_state)
    tmp = files.temp_path(out)
    try:
        tmp.write_bytes(model_mod.to_bytes(net))
        err = parity_check(net, tmp, window)
        files.fsync_file(tmp)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    os.replace(tmp, out)
    return {"steps": steps, "lr": cfg.lr, "batch": cfg.batch, "newest_share": share,
            "older_positions": window.older.decisions, "init": None if init is None else str(init),
            "width": net.width, "rounds": net.rounds, "device": device.type, "history": history, "held_out": held_out,
            "best_step": None if best_row is None else best_row["step"], "best_held_out": best_row,
            "parity_max_error": err}
