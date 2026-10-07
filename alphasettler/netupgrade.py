"""`alphasettler net-upgrade OLD NEW` (spec 3c Section 4): an encoder v2 net (3b's: 91 global
inputs) rewritten for encoder v3 (115: the trade block appended), with the same moves in every
no-trade game.

- The global encoder gets zero weights for the 24 new inputs, so the net ignores them.
- The pooled head's rows for every domestic offer (OFFER_IDS), the answers and the confirmations
  (ANSWER_IDS, CancelTrade included) get zero weights, and biases of OFFER_BIAS for offers and 0
  for the rest. 3b never trained them: offers were masked out, and every answer was a hard-coded
  reject or cancel. All 400 offer rows are reset, not only the 20 one-for-one offers the trading
  game allows, because the same reason holds for all of them.
- Every other byte is copied.

No torch, so it runs without the train extra: the file is read by `alphasettler.netformat`, the
one Python reader of the weights format (net/src/weights.rs's, version 2), which the PyTorch twin
shares."""

from __future__ import annotations

import os
import struct
from array import array
from pathlib import Path

from alphasettler import files
from alphasettler._engine import (
    NET_ENCODING_HASH, NET_GLOBAL_WIDTH, NET_SETTLEMENT_BASE, NET_V2_ENCODING_HASH, NET_V2_GLOBAL_WIDTH, Game, encode,
    net_forward,
)
from alphasettler.netformat import ENTITY_SLOTS, GLOBAL_ENCODER, OTHER_POLICY, head_layer, header, read
from alphasettler.netformat import shapes  # noqa: F401  (part of this module's interface)

OFFER_BIAS = -3.0
# Engine action ids (engine/src/action.rs): offers, then accept, reject, the four confirmations, cancel.
OFFER_IDS = range(258, 658)
ANSWER_IDS = range(658, 665)
TRADE_SLOTS = range(258, 665)


def pooled_row(slot: int) -> int:
    """The pooled head's output row for policy slot `slot` (net/src/policy.rs `other_slot`, inverted)."""
    if NET_SETTLEMENT_BASE <= slot < NET_SETTLEMENT_BASE + ENTITY_SLOTS:
        raise ValueError(f"slot {slot} comes from an entity head")
    return slot if slot < NET_SETTLEMENT_BASE else slot - ENTITY_SLOTS


def _check_v2(encoding: int) -> None:
    if encoding == NET_ENCODING_HASH:
        raise ValueError("already an encoder v3 net (this build's encoding hash): nothing to upgrade")
    if encoding != NET_V2_ENCODING_HASH:
        raise ValueError(f"encoding hash {encoding:#018x} is not encoder v2's {NET_V2_ENCODING_HASH:#018x}: "
                         "only encoder v2 nets can be upgraded")


def read_v2(data: bytes) -> tuple[int, int, list[tuple[int, int, array, array]]]:
    """(width, rounds, layers) of an encoder v2 weights file, each layer (rows, cols, weights,
    biases); ValueError for any other file. Everything is checked before anything is sized from it."""
    width, rounds, declared, floats = read(data, NET_V2_GLOBAL_WIDTH, _check_v2, "an encoder v2 net")
    layers, at = [], 0
    for rows, cols in declared:
        w = floats[at:at + rows * cols]
        at += rows * cols
        b = floats[at:at + rows]
        at += rows
        layers.append((rows, cols, w, b))
    return width, rounds, layers


def upgrade(data: bytes) -> bytes:
    """The encoder v3 weights file for the encoder v2 file `data` (see the module docstring)."""
    width, rounds, layers = read_v2(data)
    extra = NET_GLOBAL_WIDTH - NET_V2_GLOBAL_WIDTH
    rows, cols, w, b = layers[GLOBAL_ENCODER]
    wide = array("f")
    for r in range(rows):
        wide.extend(w[r * cols:(r + 1) * cols])
        wide.extend(array("f", [0.0]) * extra)
    layers[GLOBAL_ENCODER] = (rows, NET_GLOBAL_WIDTH, wide, b)
    rows, cols, w, b = layers[head_layer(rounds, OTHER_POLICY)]
    for slot in TRADE_SLOTS:
        r = pooled_row(slot)
        w[r * cols:(r + 1) * cols] = array("f", [0.0]) * cols
        b[r] = OFFER_BIAS if slot in OFFER_IDS else 0.0
    out = [header(width, rounds, [(r, c) for r, c, _, _ in layers])]
    for _, _, w, b in layers:
        out += [w.tobytes(), b.tobytes()]
    return b"".join(out)


def upgrade_file(old: str | Path, new: str | Path) -> dict:
    """Write the upgrade of `old` to `new` (which must not exist, so nothing is overwritten). The
    upgrade goes to a temp file beside `new` and is published by a rename only after the Rust loader
    has read it as every bot will, so a failure at any step leaves no `new` (and no temp) behind."""
    old, new = Path(old), Path(new)
    if new.exists():
        raise ValueError(f"{new} exists: choose a new path")
    try:
        data = upgrade(old.read_bytes())
    except ValueError as e:
        raise ValueError(f"{old}: {e}") from None
    tmp = files.temp_path(new)
    try:
        with open(tmp, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        p, h = encode(Game(0, {"max_offers_per_turn": 0}))
        try:
            net_forward(str(tmp), p, h)
        except ValueError as e:
            raise ValueError(f"{old}: the Rust loader refused the upgrade: {e}") from None
        os.replace(tmp, new)
    finally:
        tmp.unlink(missing_ok=True)
    width, rounds = struct.unpack_from("<II", data, 20)
    return {"width": width, "rounds": rounds, "zero_inputs": NET_GLOBAL_WIDTH - NET_V2_GLOBAL_WIDTH,
            "trade_rows": len(TRADE_SLOTS)}
