"""The weights file format of net/src/weights.rs, version 2, without torch: the one Python reader
of its header, shapes and floats, shared by the PyTorch twin (`alphasettler.train.model`) and
`alphasettler.netupgrade`. It lives outside `alphasettler.train`, whose package imports torch.

Little-endian: a 32-byte fixed header (magic, format version, ENCODING_HASH, width d, rounds R,
layer count), each layer's rows and columns, then each layer's f32 weights (row-major, rows =
outputs) and biases, in the layer order of net/src/weights.rs's table. `read` checks a file exactly
as the Rust loader does, and checks all of it (header, length, every weight finite) before anything
is sized from it. Only the encoding hash check is the caller's: the twin wants this encoder's, the
upgrade encoder v2's."""

from __future__ import annotations

import math
import struct
import sys
from array import array
from typing import Callable

from alphasettler._engine import (
    NET_EDGE_WIDTH, NET_ENCODING_HASH, NET_FORMAT_VERSION, NET_HIDDEN_LEN, NET_MAX_ROUNDS, NET_MAX_WIDTH,
    NET_NODE_WIDTH, NET_TILE_WIDTH, action_space_size, net_adjacency,
)

MAGIC = b"ASNET\0\0\0"
FIXED_HEADER_LEN = 8 + 4 + 8 + 4 + 4 + 4
ACTIONS = action_space_size()
SEATS = 4
ADJACENCY = net_adjacency()
NODES, EDGES, TILES = len(ADJACENCY["node_tiles"]), len(ADJACENCY["edge_nodes"]), len(ADJACENCY["tile_nodes"])
# Slots NET_SETTLEMENT_BASE .. + ENTITY_SLOTS come from the per-entity heads (net/src/policy.rs); the
# pooled head's OTHER slots are 0..NET_SETTLEMENT_BASE and everything after them.
ENTITY_SLOTS = 2 * NODES + EDGES + TILES
OTHER = ACTIONS - ENTITY_SLOTS

# Layer indices in file order (net/src/weights.rs): four encoders, four layers per round, eight heads.
GLOBAL_ENCODER = 3
ROUND_BASE = 4
LAYERS_PER_ROUND = 4
HEAD_LAYERS = 8
OTHER_POLICY = 4  # the pooled head's output layer, among the heads


def check_arch(width: int, rounds: int) -> None:
    """ValueError (with the Rust loader's wording) for a width or round count out of range."""
    if not 1 <= width <= NET_MAX_WIDTH:
        raise ValueError(f"width {width} is out of range 1..={NET_MAX_WIDTH}")
    if not 1 <= rounds <= NET_MAX_ROUNDS:
        raise ValueError(f"rounds {rounds} is out of range 1..={NET_MAX_ROUNDS}")


def layer_count(rounds: int) -> int:
    return ROUND_BASE + LAYERS_PER_ROUND * rounds + HEAD_LAYERS


def head_layer(rounds: int, head: int) -> int:
    """The file index of head layer `head` (OTHER_POLICY, say) in a net of `rounds` rounds."""
    return ROUND_BASE + LAYERS_PER_ROUND * rounds + head


def shapes(width: int, rounds: int, global_width: int) -> list[tuple[int, int]]:
    """(rows, cols) of each layer in file order, for a global encoder of `global_width` inputs; rows
    are outputs (as Rust's `Arch::shapes`)."""
    d = width
    out = [(d, NET_NODE_WIDTH), (d, NET_EDGE_WIDTH), (d, NET_TILE_WIDTH), (d, global_width)]
    out += [(d, 3 * d), (d, 5 * d), (d, 3 * d), (d, 4 * d)] * rounds
    return out + [(2, 2 * d), (1, 2 * d), (1, 2 * d), (4 * d, 4 * d), (OTHER, 4 * d),
                  (4 * d, 4 * d + NET_HIDDEN_LEN), (SEATS, 4 * d), (SEATS, 4 * d)]


def header(width: int, rounds: int, layer_shapes: list[tuple[int, int]]) -> bytes:
    """The header of a file for this build's encoder: everything before the first weight."""
    assert sys.byteorder == "little", "the weights file is little-endian"
    out = [MAGIC, struct.pack("<IQIII", NET_FORMAT_VERSION, NET_ENCODING_HASH, width, rounds, len(layer_shapes))]
    out += [struct.pack("<II", r, c) for r, c in layer_shapes]
    return b"".join(out)


def read(data: bytes, global_width: int, check_encoding: Callable[[int], None],
         layout: str) -> tuple[int, int, list[tuple[int, int]], array]:
    """(width, rounds, layer shapes, every float in file order) of the weights file `data`, whose
    global encoder takes `global_width` inputs; ValueError (with the Rust loader's wording) for a file
    it rejects. `check_encoding(hash)` raises for a hash the caller does not read; `layout` names
    the expected net in the shape error."""
    assert sys.byteorder == "little", "the weights file is little-endian"
    if len(data) < FIXED_HEADER_LEN:
        raise ValueError(f"truncated: {len(data)} bytes is shorter than the {FIXED_HEADER_LEN}-byte header")
    if data[:8] != MAGIC:
        raise ValueError("not an AlphaSettler weights file (wrong magic)")
    version, encoding, width, rounds, count = struct.unpack_from("<IQIII", data, 8)
    if version != NET_FORMAT_VERSION:
        why = "; version 1 holds the flat MLP that network v2 replaced: retrain the net" if version == 1 else ""
        raise ValueError(f"format version {version}, expected {NET_FORMAT_VERSION}{why}")
    check_encoding(encoding)
    check_arch(width, rounds)
    if count != layer_count(rounds):
        raise ValueError(f"{count} layers, expected {layer_count(rounds)} for {rounds} rounds")
    header_len = FIXED_HEADER_LEN + 8 * count
    if len(data) < header_len:
        raise ValueError(f"truncated: {len(data)} bytes is shorter than the {header_len}-byte header")
    declared = [struct.unpack_from("<II", data, FIXED_HEADER_LEN + 8 * i) for i in range(count)]
    if declared != shapes(width, rounds, global_width):
        raise ValueError(f"layer shapes {declared} do not fit {layout} with width {width} and {rounds} rounds")
    want = header_len + 4 * sum(r * c + r for r, c in declared)
    if len(data) < want:
        raise ValueError(f"truncated: {len(data)} bytes, expected {want}")
    if len(data) > want:
        raise ValueError(f"{len(data) - want} trailing bytes after the last layer")
    floats = array("f")
    floats.frombytes(data[header_len:])
    # Exact: a double sum of finite f32s cannot overflow (they are below 3.5e38, and a file holds far
    # fewer than 1e270 of them), and a NaN or an infinity makes the sum non-finite.
    if not math.isfinite(sum(floats)):
        raise ValueError("a weight is NaN or infinite")
    return width, rounds, declared, floats
