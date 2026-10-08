"""The opening model (spec 3d-1 Section 4.4): an MLP over the 78 opening features, trained on
recorded games' setups to predict the winner relative to the seat that just placed a settlement,
written in the format of bots/src/opening_model.rs (`alphasettler.opening.write_model`).

Samples: after each of a game's 8 settlements (before its road), the features for the seat that
placed it and the winner relative to that seat; a game with no winner adds nothing. Held out:
every game whose seed % HELD_OUT_EVERY == 0, in every directory. Fit: cross-entropy, Adam, an epoch
at a time on the CPU (the model has about 10,000 weights), until the held-out loss has not improved
for `patience` epochs; the best held-out epoch is kept."""

from __future__ import annotations

import copy
import json
import math
from array import array
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from alphasettler import files, records
from alphasettler._engine import OPENING_FEATURES, Game, net_evaluate
from alphasettler.opening import Setup, replay_setup, write_model
from alphasettler.train.data import HELD_OUT_EVERY

SEATS = 4
HIDDEN = (64, 64)
# A uniform guess's cross-entropy.
UNIFORM = math.log(SEATS)
# The compare net's probability of the winner is floored here before the log, so one confident
# miss costs -ln(1e-12) rather than infinity.
PROBABILITY_FLOOR = 1e-12


@dataclass(frozen=True)
class OpeningConfig:
    lr: float = 1e-3
    batch: int = 1024
    patience: int = 3
    max_epochs: int = 100
    seed: int = 0
    hidden: tuple[int, ...] = HIDDEN

    def check(self) -> None:
        if not (math.isfinite(self.lr) and self.lr > 0):
            raise ValueError(f"the learning rate must be finite and positive, got {self.lr}")
        if self.batch < 1:
            raise ValueError(f"batch must be at least 1, got {self.batch}")
        if self.patience < 1:
            raise ValueError(f"patience must be at least 1 epoch, got {self.patience}")
        if self.max_epochs < 1:
            raise ValueError(f"max_epochs must be at least 1, got {self.max_epochs}")
        if not self.hidden or any(h < 1 for h in self.hidden):
            raise ValueError(f"hidden widths must be positive, got {self.hidden}")


class OpeningNet(nn.Module):
    """78 features -> `hidden` widths (ReLU) -> 4 seat-relative win logits."""

    def __init__(self, hidden: tuple[int, ...] = HIDDEN):
        super().__init__()
        dims = [OPENING_FEATURES, *hidden, SEATS]
        self.linears = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims, dims[1:]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.linears):
            x = layer(x)
            if i < len(self.linears) - 1:
                x = torch.relu(x)
        return x

    def layers(self) -> list[tuple[int, int, list[float], list[float]]]:
        """(rows, cols, weights row-major, biases) per layer, as `write_model` takes them."""
        return [(m.out_features, m.in_features, m.weight.detach().cpu().reshape(-1).tolist(),
                 m.bias.detach().cpu().tolist()) for m in self.linears]


@dataclass
class Samples:
    x: torch.Tensor  # n x OPENING_FEATURES, float32
    y: torch.Tensor  # n, int64: the winner relative to the sample's seat
    trading: torch.Tensor  # n, bool: the game allowed domestic offers
    compare: torch.Tensor | None  # n, float64: -ln(the compare net's probability of the winner)

    def __len__(self) -> int:
        return len(self.y)


class _Builder:
    """Samples gathered a game at a time into flat arrays (half a million rows of 78 floats would
    not fit comfortably as Python lists)."""

    def __init__(self, compare: bool):
        self.x, self.y, self.trading = array("f"), array("b"), array("b")
        self.compare = array("d") if compare else None

    def add(self, setup: Setup, winner: int) -> None:
        for pick in setup.picks:
            self.x.extend(pick.features)
            self.y.append((winner - pick.seat) % SEATS)
            self.trading.append(setup.trading)
            if self.compare is not None:
                self.compare.append(-math.log(max(pick.value[winner], PROBABILITY_FLOOR)))

    def samples(self) -> Samples:
        n = len(self.y)
        x = (torch.frombuffer(bytearray(self.x.tobytes()), dtype=torch.float32).reshape(n, OPENING_FEATURES)
             if n else torch.zeros(0, OPENING_FEATURES))
        compare = None if self.compare is None else torch.tensor(list(self.compare), dtype=torch.float64)
        return Samples(x, torch.tensor(list(self.y), dtype=torch.int64),
                       torch.tensor(list(self.trading), dtype=torch.bool), compare)


def load_samples(dirs: list[Path], compare: Callable[[Game, int], list[float]] | None = None
                 ) -> tuple[Samples, Samples, dict]:
    """(training samples, held-out samples, counts) from every record file in `dirs`. With
    `compare(game, seat)` (four win probabilities by absolute seat), each held-out sample keeps
    -ln(its probability of the winner). ValueError naming the file for a record that does not replay,
    and for a directory with no records, or no training or no held-out samples at all."""
    train, held = _Builder(False), _Builder(compare is not None)
    counts = {"games": 0, "no_winner": 0, "held_out_games": 0}
    for d in dirs:
        paths = records.record_files(d)
        if not paths:
            raise ValueError(f"no record files (*.jsonl.gz) in {d}")
        for path in paths:
            for rec in records.read(path):
                counts["games"] += 1
                if rec["winner"] is None:
                    counts["no_winner"] += 1
                    continue
                is_held = rec["seed"] % HELD_OUT_EVERY == 0
                try:
                    setup = replay_setup(rec, features=True, value=compare if is_held else None)
                except ValueError as e:
                    raise ValueError(f"{path}: {e}") from None
                (held if is_held else train).add(setup, rec["winner"])
                counts["held_out_games"] += is_held
    where = ", ".join(str(d) for d in dirs)
    if len(train.y) == 0:
        raise ValueError(f"no training games (decided, seed % {HELD_OUT_EVERY} != 0) in {where}")
    if len(held.y) == 0:
        raise ValueError(f"no held-out games (decided, seed % {HELD_OUT_EVERY} == 0) in {where}")
    return train.samples(), held.samples(), counts


def held_out_loss(net: OpeningNet, s: Samples) -> float:
    """Mean cross-entropy of `net` on `s`, in float64."""
    with torch.no_grad():
        return F.cross_entropy(net(s.x).double(), s.y).item()


def fit(train: Samples, held: Samples, cfg: OpeningConfig, log: Callable[[str], None] = print
        ) -> tuple[OpeningNet, dict]:
    """Train epoch by epoch until the held-out loss has not improved for `cfg.patience` epochs (or
    `cfg.max_epochs`); return the net at its best held-out epoch and a summary."""
    cfg.check()
    torch.manual_seed(cfg.seed)
    order_rng = torch.Generator().manual_seed(cfg.seed)
    net = OpeningNet(cfg.hidden)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    best_loss, best_epoch, best_state, stale = math.inf, 0, copy.deepcopy(net.state_dict()), 0
    history = []
    for epoch in range(1, cfg.max_epochs + 1):
        net.train()
        order = torch.randperm(len(train), generator=order_rng)
        for lo in range(0, len(train), cfg.batch):
            idx = order[lo:lo + cfg.batch]
            loss = F.cross_entropy(net(train.x[idx]), train.y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        net.eval()
        h = held_out_loss(net, held)
        history.append({"epoch": epoch, "held_out": round(h, 6)})
        log(f"epoch {epoch}: held-out cross-entropy {h:.4f}")
        if h < best_loss:
            best_loss, best_epoch, best_state, stale = h, epoch, copy.deepcopy(net.state_dict()), 0
        else:
            stale += 1
            if stale >= cfg.patience:
                break
    net.load_state_dict(best_state)
    return net, {"best_epoch": best_epoch, "epochs": len(history), "best_held_out": best_loss, "history": history}


def report(net: OpeningNet, held: Samples) -> dict:
    """Held-out cross-entropy of the model, a uniform guess and the compare net (if any), on every
    held-out sample and on the trading games' alone (None where there are none)."""

    def row(mask: torch.Tensor) -> dict | None:
        n = int(mask.sum())
        if n == 0:
            return None
        with torch.no_grad():
            model = F.cross_entropy(net(held.x[mask]).double(), held.y[mask]).item()
        compare = None if held.compare is None else held.compare[mask].mean().item()
        return {"samples": n, "model": model, "uniform": UNIFORM, "compare": compare}

    return {"all": row(torch.ones(len(held), dtype=torch.bool)), "trading": row(held.trading)}


def train_opening(dirs: list[Path], out: Path, cfg: OpeningConfig, compare_net: str | None = None,
                  nets_dir: str | None = None, log: Callable[[str], None] = print) -> dict:
    """Load the samples, fit, write the model to `out` and a summary to `out`.json; return the summary."""
    compare = None if compare_net is None else (lambda g, seat: net_evaluate(g, seat, compare_net, nets_dir))
    train, held, counts = load_samples(dirs, compare)
    log(f"samples: {len(train)} training, {len(held)} held out, from {counts['games']} games "
        f"({counts['no_winner']} without a winner)")
    net, fitted = fit(train, held, cfg, log)
    write_model(out, net.layers())
    summary = {"dirs": [str(d) for d in dirs], "config": asdict(cfg), "compare_net": compare_net, **counts,
               "train_samples": len(train), "held_out_samples": len(held), **fitted, "held_out": report(net, held)}
    files.atomic_write_text(f"{out}.json", json.dumps(summary, indent=1) + "\n")
    return summary


def format_report(s: dict, out: Path) -> list[str]:
    """The lines `opening-train` prints after training."""

    def line(label: str, row: dict | None) -> str:
        if row is None:
            return f"held-out cross-entropy, {label}: none held out"
        text = (f"held-out cross-entropy, {label} ({row['samples']} samples): model {row['model']:.4f}, "
                f"uniform {row['uniform']:.4f}")
        if row["compare"] is not None:
            text += f", net:{s['compare_net']} {row['compare']:.4f}"
        return text

    return [f"best epoch {s['best_epoch']} of {s['epochs']}", line("all games", s["held_out"]["all"]),
            line("trading games", s["held_out"]["trading"]), f"wrote {out} (summary {out}.json)"]
