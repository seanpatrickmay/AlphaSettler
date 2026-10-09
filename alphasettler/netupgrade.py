"""`alphasettler net-upgrade OLD NEW` (specs 3c Section 4 and 3d-2 Section 3): an older net
rewritten for this build's encoder, v4. Encoder v4 appended 5 inputs to each node, 4 to each edge
and the 24-input threat block to the globals; encoder v3 had appended the 24-input trade block to
encoder v2's 91 globals.

- Each encoder's first layer gets zero weights for the inputs the old net never saw, so the net
  ignores them.
- From encoder v2 (3b's nets), the pooled head's rows for every domestic offer (OFFER_IDS), the
  answers and the confirmations (ANSWER_IDS, CancelTrade included) get zero weights, and biases of
  OFFER_BIAS for offers and 0 for the rest. 3b never trained them: offers were masked out, and
  every answer was a hard-coded reject or cancel. All 400 offer rows are reset, not only the 20
  one-for-one offers the trading game allows, because the same reason holds for all of them. An
  upgraded v2 net makes the v2 net's moves in every no-trade game.
- From encoder v3 (3c's and P2's nets) nothing else changes: an upgraded v3 net gives the v3 net's
  outputs on every position, within the forward pass's 1e-5 (Accelerate sums the wider rows in
  another order).
- Every other byte is copied.

No torch, so it runs without the train extra: the file is read by `alphasettler.netformat`, the
one Python reader of the weights format (net/src/weights.rs's, version 2), which the PyTorch twin
shares."""

from __future__ import annotations

import os
from array import array
from pathlib import Path

from alphasettler import files
from alphasettler._engine import (
    NET_EDGE_WIDTH, NET_ENCODING_HASH, NET_GLOBAL_WIDTH, NET_NODE_WIDTH, NET_SETTLEMENT_BASE, NET_V2_ENCODING_HASH,
    NET_V2_GLOBAL_WIDTH, NET_V3_EDGE_WIDTH, NET_V3_ENCODING_HASH, NET_V3_GLOBAL_WIDTH, NET_V3_NODE_WIDTH, Game, encode,
    net_forward,
)
from alphasettler.netformat import (
    EDGE_ENCODER, ENTITY_SLOTS, GLOBAL_ENCODER, NODE_ENCODER, OTHER_POLICY, head_layer, header, read,
)
from alphasettler.netformat import shapes  # noqa: F401  (part of this module's interface)

OFFER_BIAS = -3.0
# Engine action ids (engine/src/action.rs): offers, then accept, reject, the four confirmations, cancel.
OFFER_IDS = range(258, 658)
ANSWER_IDS = range(658, 665)
TRADE_SLOTS = range(258, 665)

# The encoders a net is upgraded from: encoding hash -> (encoder version, node, edge and global input
# widths). Encoder v2's node and edge inputs are v3's.
SOURCES = {
    NET_V2_ENCODING_HASH: (2, NET_V3_NODE_WIDTH, NET_V3_EDGE_WIDTH, NET_V2_GLOBAL_WIDTH),
    NET_V3_ENCODING_HASH: (3, NET_V3_NODE_WIDTH, NET_V3_EDGE_WIDTH, NET_V3_GLOBAL_WIDTH),
}
# Each encoder layer and its input count in this build.
TARGET = {"node": (NODE_ENCODER, NET_NODE_WIDTH), "edge": (EDGE_ENCODER, NET_EDGE_WIDTH),
          "global": (GLOBAL_ENCODER, NET_GLOBAL_WIDTH)}


def pooled_row(slot: int) -> int:
    """The pooled head's output row for policy slot `slot` (net/src/policy.rs `other_slot`, inverted)."""
    if NET_SETTLEMENT_BASE <= slot < NET_SETTLEMENT_BASE + ENTITY_SLOTS:
        raise ValueError(f"slot {slot} comes from an entity head")
    return slot if slot < NET_SETTLEMENT_BASE else slot - ENTITY_SLOTS


def _check_source(encoding: int) -> None:
    if encoding == NET_ENCODING_HASH:
        raise ValueError("already an encoder v4 net (this build's encoding hash): nothing to upgrade")
    if encoding not in SOURCES:
        raise ValueError(f"encoding hash {encoding:#018x} is neither encoder v2's {NET_V2_ENCODING_HASH:#018x} "
                         f"nor encoder v3's {NET_V3_ENCODING_HASH:#018x}: only encoder v2 and v3 nets can be upgraded")


def read_old(data: bytes) -> tuple[int, int, int, list[tuple[int, int, array, array]]]:
    """(encoder version, width, rounds, layers) of a weights file this tool upgrades, each layer
    (rows, cols, weights, biases); ValueError for any other file. Everything is checked before
    anything is sized from it."""
    # The input widths follow from the encoding hash, which `read` checks only after the magic and the
    # version: peek at it, and let `read` report whatever is wrong in its own order.
    peek = int.from_bytes(data[12:20], "little") if len(data) >= 20 else None
    # (An unknown or missing hash takes this build's widths: `read` then refuses it through `_check_source`.)
    version, node, edge, glob = SOURCES.get(peek, (0, NET_NODE_WIDTH, NET_EDGE_WIDTH, NET_GLOBAL_WIDTH))
    width, rounds, declared, floats = read(data, glob, _check_source, f"an encoder v{version} net",
                                           node_width=node, edge_width=edge)
    layers, at = [], 0
    for rows, cols in declared:
        w = floats[at:at + rows * cols]
        at += rows * cols
        b = floats[at:at + rows]
        at += rows
        layers.append((rows, cols, w, b))
    return version, width, rounds, layers


def widened(layer: tuple[int, int, array, array], cols: int) -> tuple[int, int, array, array]:
    """`layer` with `cols` inputs: its own first, then zero weights for the inputs appended after them."""
    rows, old, w, b = layer
    if cols < old:
        raise ValueError(f"cannot narrow a layer of {old} inputs to {cols}")
    wide = array("f")
    for r in range(rows):
        wide.extend(w[r * old:(r + 1) * old])
        wide.extend(array("f", [0.0]) * (cols - old))
    return rows, cols, wide, b


def _upgrade(data: bytes) -> tuple[bytes, dict]:
    version, width, rounds, layers = read_old(data)
    zero = {}
    for name, (index, cols) in TARGET.items():
        zero[name] = cols - layers[index][1]
        layers[index] = widened(layers[index], cols)
    trade_rows = 0
    if version == 2:
        rows, cols, w, b = layers[head_layer(rounds, OTHER_POLICY)]
        for slot in TRADE_SLOTS:
            r = pooled_row(slot)
            w[r * cols:(r + 1) * cols] = array("f", [0.0]) * cols
            b[r] = OFFER_BIAS if slot in OFFER_IDS else 0.0
        trade_rows = len(TRADE_SLOTS)
    out = [header(width, rounds, [(r, c) for r, c, _, _ in layers])]
    for _, _, w, b in layers:
        out += [w.tobytes(), b.tobytes()]
    summary = {"from": version, "width": width, "rounds": rounds, "zero_inputs": zero, "trade_rows": trade_rows}
    return b"".join(out), summary


def upgrade(data: bytes) -> bytes:
    """The encoder v4 weights file for the older file `data` (see the module docstring)."""
    return _upgrade(data)[0]


def upgrade_file(old: str | Path, new: str | Path) -> dict:
    """Write the upgrade of `old` to `new` (which must not exist, so nothing is overwritten) and
    return what was done: the source encoder (`from`), width, rounds, the zero inputs per encoder
    and the trade rows reset. The upgrade goes to a temp file beside `new` and is published by a
    rename only after the Rust loader has read it as every bot will, so a failure at any step leaves
    no `new` (and no temp) behind."""
    old, new = Path(old), Path(new)
    if new.exists():
        raise ValueError(f"{new} exists: choose a new path")
    try:
        data, summary = _upgrade(old.read_bytes())
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
    return summary
