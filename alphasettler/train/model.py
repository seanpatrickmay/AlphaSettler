"""The PyTorch twin of net/src/forward.rs, network v2 (spec Section 2): an entity-structured graph
net over the encoder's 54 nodes, 72 edges, 19 tiles and 91 global scalars. Also the weights file
format of net/src/weights.rs, version 2: little-endian, a 32-byte fixed header (magic, format
version, ENCODING_HASH, width d, rounds R, layer count), each layer's rows and columns, then each
layer's f32 weights (row-major, rows = outputs) and biases. The layer order, byte for byte, is the
table in net/src/weights.rs; `Net.layers()` returns the modules in that order.

The adjacency comes from the Rust side (`net_adjacency`) as averaging and summing matrices, so both
forward passes read the board the same way. `from_bytes` checks a file exactly as the Rust loader
does, and checks all of it (header, length, every weight finite) before it builds a module, so a
corrupt header never sizes an allocation."""

from __future__ import annotations

import struct
import sys
from array import array
from pathlib import Path

import torch
from torch import nn

from alphasettler._engine import (
    NET_DEFAULT_ROUNDS, NET_DEFAULT_WIDTH, NET_EDGE_WIDTH, NET_ENCODING_HASH, NET_FORMAT_VERSION, NET_GLOBAL_WIDTH,
    NET_HIDDEN_LEN, NET_MAX_ROUNDS, NET_MAX_WIDTH, NET_NODE_WIDTH, NET_PUBLIC_LEN, NET_SETTLEMENT_BASE,
    NET_TILE_WIDTH, action_space_size, net_adjacency,
)

MAGIC = b"ASNET\0\0\0"
ACTIONS = action_space_size()
SEATS = 4
FIXED_HEADER_LEN = 8 + 4 + 8 + 4 + 4 + 4
ADJACENCY = net_adjacency()
NODES, EDGES, TILES = len(ADJACENCY["node_tiles"]), len(ADJACENCY["edge_nodes"]), len(ADJACENCY["tile_nodes"])
NODE_END = NODES * NET_NODE_WIDTH
EDGE_END = NODE_END + EDGES * NET_EDGE_WIDTH
TILE_END = EDGE_END + TILES * NET_TILE_WIDTH
assert TILE_END + NET_GLOBAL_WIDTH == NET_PUBLIC_LEN, "nodes, edges, tiles and the globals tile the public vector"
# The pooled head's slots: 0..NET_SETTLEMENT_BASE, then everything after the entity heads' slots.
OTHER = ACTIONS - 2 * NODES - EDGES - TILES


def check_arch(width: int, rounds: int) -> None:
    """ValueError (with the Rust loader's wording) for a width or round count out of range."""
    if not 1 <= width <= NET_MAX_WIDTH:
        raise ValueError(f"width {width} is out of range 1..={NET_MAX_WIDTH}")
    if not 1 <= rounds <= NET_MAX_ROUNDS:
        raise ValueError(f"rounds {rounds} is out of range 1..={NET_MAX_ROUNDS}")


def layer_count(rounds: int) -> int:
    return 4 + 4 * rounds + 8


def shapes(width: int, rounds: int) -> list[tuple[int, int]]:
    """(rows, cols) of each layer in file order; rows are outputs (as Rust's `Arch::shapes`)."""
    d = width
    out = [(d, NET_NODE_WIDTH), (d, NET_EDGE_WIDTH), (d, NET_TILE_WIDTH), (d, NET_GLOBAL_WIDTH)]
    out += [(d, 3 * d), (d, 5 * d), (d, 3 * d), (d, 4 * d)] * rounds
    return out + [(2, 2 * d), (1, 2 * d), (1, 2 * d), (4 * d, 4 * d), (OTHER, 4 * d),
                  (4 * d, 4 * d + NET_HIDDEN_LEN), (SEATS, 4 * d), (SEATS, 4 * d)]


def _links(lists: list[list[int]], cols: int, mean: bool) -> torch.Tensor:
    """The matrix whose row i averages (or sums) the columns `lists[i]`."""
    m = torch.zeros(len(lists), cols)
    for i, js in enumerate(lists):
        for j in js:
            m[i, j] += 1.0 / len(js) if mean else 1.0
    return m


def _spread(g: torch.Tensor, count: int) -> torch.Tensor:
    """A per-position row (batch x d) repeated for each of `count` entities (batch x count x d)."""
    return g.unsqueeze(1).expand(-1, count, -1)


class Round(nn.Module):
    """One round's four residual updates, applied in this order: tile, node, edge, global."""

    def __init__(self, d: int):
        super().__init__()
        self.tile = nn.Linear(3 * d, d)
        self.node = nn.Linear(5 * d, d)
        self.edge = nn.Linear(3 * d, d)
        self.glob = nn.Linear(4 * d, d)


class Net(nn.Module):
    """Encoders per entity type, `rounds` rounds of message passing, per-entity policy heads for
    settlements, cities, roads and robber moves, a pooled head for every other slot, and the value
    and aux heads on [pooled || hidden]. All value and aux outputs are seat-relative."""

    def __init__(self, width: int = NET_DEFAULT_WIDTH, rounds: int = NET_DEFAULT_ROUNDS):
        check_arch(width, rounds)
        super().__init__()
        d = width
        self.width, self.rounds = width, rounds
        self.node_encoder = nn.Linear(NET_NODE_WIDTH, d)
        self.edge_encoder = nn.Linear(NET_EDGE_WIDTH, d)
        self.tile_encoder = nn.Linear(NET_TILE_WIDTH, d)
        self.global_encoder = nn.Linear(NET_GLOBAL_WIDTH, d)
        self.round_layers = nn.ModuleList([Round(d) for _ in range(rounds)])
        self.node_policy = nn.Linear(2 * d, 2)
        self.edge_policy = nn.Linear(2 * d, 1)
        self.tile_policy = nn.Linear(2 * d, 1)
        self.other_hidden = nn.Linear(4 * d, 4 * d)
        self.other_policy = nn.Linear(4 * d, OTHER)
        self.value_hidden = nn.Linear(4 * d + NET_HIDDEN_LEN, 4 * d)
        self.value = nn.Linear(4 * d, SEATS)
        self.aux = nn.Linear(4 * d, SEATS)
        a = ADJACENCY
        # Not in the state dict: fixed by the board, rebuilt with every Net.
        self.register_buffer("tile_nodes", _links(a["tile_nodes"], NODES, True), persistent=False)
        self.register_buffer("node_tiles", _links(a["node_tiles"], TILES, True), persistent=False)
        self.register_buffer("node_edges", _links(a["node_edges"], EDGES, True), persistent=False)
        self.register_buffer("node_neighbours", _links(a["node_neighbours"], NODES, True), persistent=False)
        self.register_buffer("edge_nodes", _links(a["edge_nodes"], NODES, False), persistent=False)

    def layers(self) -> list[nn.Linear]:
        """In weights-file order."""
        out = [self.node_encoder, self.edge_encoder, self.tile_encoder, self.global_encoder]
        for r in self.round_layers:
            out += [r.tile, r.node, r.edge, r.glob]
        return out + [self.node_policy, self.edge_policy, self.tile_policy, self.other_hidden, self.other_policy,
                      self.value_hidden, self.value, self.aux]

    def forward(self, public: torch.Tensor, hidden: torch.Tensor):
        b = public.shape[0]
        relu = torch.relu
        node = relu(self.node_encoder(public[:, :NODE_END].reshape(b, NODES, NET_NODE_WIDTH)))
        edge = relu(self.edge_encoder(public[:, NODE_END:EDGE_END].reshape(b, EDGES, NET_EDGE_WIDTH)))
        tile = relu(self.tile_encoder(public[:, EDGE_END:TILE_END].reshape(b, TILES, NET_TILE_WIDTH)))
        glob = relu(self.global_encoder(public[:, TILE_END:]))
        for r in self.round_layers:
            tile = tile + relu(r.tile(torch.cat([tile, self.tile_nodes @ node, _spread(glob, TILES)], 2)))
            node = node + relu(r.node(torch.cat([node, self.node_tiles @ tile, self.node_edges @ edge,
                                                 self.node_neighbours @ node, _spread(glob, NODES)], 2)))
            edge = edge + relu(r.edge(torch.cat([edge, self.edge_nodes @ node, _spread(glob, EDGES)], 2)))
            glob = glob + relu(r.glob(torch.cat([glob, node.mean(1), edge.mean(1), tile.mean(1)], 1)))
        pooled = torch.cat([glob, node.mean(1), edge.mean(1), tile.mean(1)], 1)
        build = self.node_policy(torch.cat([node, _spread(glob, NODES)], 2))
        road = self.edge_policy(torch.cat([edge, _spread(glob, EDGES)], 2))[:, :, 0]
        robber = self.tile_policy(torch.cat([tile, _spread(glob, TILES)], 2))[:, :, 0]
        other = self.other_policy(relu(self.other_hidden(pooled)))
        s = NET_SETTLEMENT_BASE
        policy = torch.cat([other[:, :s], build[:, :, 0], build[:, :, 1], road, robber, other[:, s:]], 1)
        v = relu(self.value_hidden(torch.cat([pooled, hidden], 1)))
        return policy, self.value(v), self.aux(v)


def _floats(t: torch.Tensor) -> bytes:
    return array("f", t.detach().to("cpu", torch.float32).contiguous().view(-1).tolist()).tobytes()


def to_bytes(net: Net) -> bytes:
    assert sys.byteorder == "little", "the weights file is little-endian"
    layers = net.layers()
    out = [MAGIC, struct.pack("<IQIII", NET_FORMAT_VERSION, NET_ENCODING_HASH, net.width, net.rounds, len(layers))]
    out += [struct.pack("<II", layer.out_features, layer.in_features) for layer in layers]
    for layer in layers:
        out += [_floats(layer.weight), _floats(layer.bias)]
    return b"".join(out)


def from_bytes(data: bytes) -> Net:
    """The network in `data`; ValueError (with the Rust loader's wording) for a file it rejects."""
    assert sys.byteorder == "little", "the weights file is little-endian"
    if len(data) < FIXED_HEADER_LEN:
        raise ValueError(f"truncated: {len(data)} bytes is shorter than the {FIXED_HEADER_LEN}-byte header")
    if data[:8] != MAGIC:
        raise ValueError("not an AlphaSettler weights file (wrong magic)")
    version, encoding, width, rounds, count = struct.unpack_from("<IQIII", data, 8)
    if version != NET_FORMAT_VERSION:
        why = "; version 1 holds the flat MLP that network v2 replaced: retrain the net" if version == 1 else ""
        raise ValueError(f"format version {version}, expected {NET_FORMAT_VERSION}{why}")
    if encoding != NET_ENCODING_HASH:
        raise ValueError(f"encoding hash {encoding:#018x} does not match this encoder's {NET_ENCODING_HASH:#018x}: "
                         "the file was trained on a different feature layout")
    check_arch(width, rounds)
    if count != layer_count(rounds):
        raise ValueError(f"{count} layers, expected {layer_count(rounds)} for {rounds} rounds")
    header = FIXED_HEADER_LEN + 8 * count
    if len(data) < header:
        raise ValueError(f"truncated: {len(data)} bytes is shorter than the {header}-byte header")
    declared = [struct.unpack_from("<II", data, FIXED_HEADER_LEN + 8 * i) for i in range(count)]
    if declared != shapes(width, rounds):
        raise ValueError(f"layer shapes {declared} do not fit network v2 with width {width} and {rounds} rounds")
    want = header + 4 * sum(r * c + r for r, c in declared)
    if len(data) < want:
        raise ValueError(f"truncated: {len(data)} bytes, expected {want}")
    if len(data) > want:
        raise ValueError(f"{len(data) - want} trailing bytes after the last layer")
    params = torch.frombuffer(bytearray(data[header:]), dtype=torch.float32)
    if not torch.isfinite(params).all():
        raise ValueError("a weight is NaN or infinite")
    net = Net(width=width, rounds=rounds)
    at = 0
    with torch.no_grad():
        for layer in net.layers():
            for p in (layer.weight, layer.bias):
                n = p.numel()
                p.copy_(params[at : at + n].view_as(p))
                at += n
    return net


def save(net: Net, path: str | Path) -> None:
    Path(path).write_bytes(to_bytes(net))


def load(path: str | Path) -> Net:
    path = Path(path)
    try:
        return from_bytes(path.read_bytes())
    except ValueError as e:
        raise ValueError(f"{path}: {e}") from None
