"""One training run (spec Section 4): Adam on batches from the Rust sampler, then export, then
a parity check of the exported file's Rust forward pass against PyTorch before the file is put
in place. Training on MPS is not bit-reproducible; on the CPU it is."""

from __future__ import annotations

import copy
import math
import os
from dataclasses import dataclass
from pathlib import Path

import torch

from alphasettler import files
from alphasettler._engine import NET_MAX_SAMPLE, net_forward
from alphasettler.train import model as model_mod
from alphasettler.train.data import Tensors, Window, to_tensors

PARITY_POSITIONS = 100
PARITY_TOLERANCE = 1e-5
PARITY_SEED = 0x9A217
L2 = 1e-4
AUX_WEIGHT = 0.25


@dataclass(frozen=True)
class TrainConfig:
    lr: float
    steps: int | None = None  # None: steps_factor x new positions / batch, rounded up
    steps_factor: float = 2.0
    batch: int = 1024
    seed: int = 0
    device: str = "auto"
    threads: int | None = None  # sampler threads (the batches do not depend on it)
    trunk: int = 512  # for a net trained from scratch; an `init` file brings its own shape
    value: int = 256
    log_every: int = 200

    def check(self) -> None:
        """ValueError for a setting no run could use, before anything is trained or written."""
        if not (math.isfinite(self.lr) and self.lr > 0):
            raise ValueError(f"the learning rate must be finite and positive, got {self.lr}")
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


def auto_steps(new_positions: int, batch: int, factor: float = 2.0) -> int:
    """Enough steps to draw each of the newest generation's positions `factor` times on average."""
    return max(1, math.ceil(factor * new_positions / batch))


def losses(net: model_mod.Net, t: Tensors) -> dict[str, torch.Tensor]:
    """Spec Section 4: policy cross-entropy over full-search rows only, value cross-entropy,
    AUX_WEIGHT x MSE(aux, final VP / 10), and L2 x the squared weights (biases excluded)."""
    logits, value, aux = net(t.public, t.hidden)
    # -1e9, not -inf: the target is 0 off the mask, and 0 x -1e9 is 0 where 0 x -inf is NaN.
    logp = torch.log_softmax(logits.masked_fill(~t.mask, -1e9), dim=1)
    per_position = -(t.policy * logp).sum(dim=1)
    policy = (per_position * t.full).sum() / t.full.sum().clamp(min=1.0)
    value_loss = -(t.outcome * torch.log_softmax(value, dim=1)).sum(dim=1).mean()
    aux_loss = torch.nn.functional.mse_loss(aux, t.vp)
    l2 = sum((layer.weight ** 2).sum() for layer in net.layers())
    total = policy + value_loss + AUX_WEIGHT * aux_loss + L2 * l2
    return {"policy": policy, "value": value_loss, "aux": aux_loss, "l2": l2, "total": total}


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
    batch = window.data.sample(PARITY_POSITIONS, PARITY_SEED, 1)
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
    parity check, and return a summary. The caller's torch RNG state is left as it was."""
    cfg.check()
    out = Path(out)
    device = pick_device(cfg.device)
    threads = cfg.threads or os.cpu_count() or 1
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(cfg.seed)
        net = model_mod.load(init) if init is not None else model_mod.Net(cfg.trunk, cfg.value)
    net.to(device).train()
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    steps = cfg.steps or auto_steps(window.new_positions, cfg.batch, cfg.steps_factor)
    history = []
    for step in range(steps):
        batch = window.data.sample(cfg.batch, (cfg.seed * 2**32 + step) % 2**64, threads)
        parts = losses(net, to_tensors(batch, device))
        opt.zero_grad()
        parts["total"].backward()
        opt.step()
        if step % cfg.log_every == 0 or step == steps - 1:
            row = {"step": step, **{k: round(v.detach().item(), 6) for k, v in parts.items()}}
            history.append(row)
            print(f"step {step + 1}/{steps}: total {row['total']:.4f}, policy {row['policy']:.4f}, "
                  f"value {row['value']:.4f}, aux {row['aux']:.5f}", flush=True)
    tmp = files.temp_path(out)
    try:
        tmp.write_bytes(model_mod.to_bytes(net))
        err = parity_check(net, tmp, window)
        files.fsync_file(tmp)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    os.replace(tmp, out)
    return {"steps": steps, "lr": cfg.lr, "batch": cfg.batch, "init": None if init is None else str(init),
            "device": device.type, "history": history, "parity_max_error": err}
