"""The PyTorch twin of net/src/forward.rs, network v2 (spec Section 2): an entity-structured graph
net over the encoder's 54 nodes, 72 edges, 19 tiles and 139 global scalars (encoder v4), read from
and written to the weights file format of net/src/weights.rs, version 2 (`alphasettler.netformat`,
which holds the format's constants and its one reader). The layer order, byte for byte, is the
table in net/src/weights.rs; `Net.layers()` returns the modules in that order.

The adjacency comes from the Rust side (`net_adjacency`) as averaging and summing matrices, so both
forward passes read the board the same way. `from_bytes` checks a file exactly as the Rust loader
does, and checks all of it (header, length, every weight finite) before it builds a module, so a
corrupt header never sizes an allocation."""

from __future__ import annotations

from array import array
from pathlib import Path

import torch
from torch import nn

from alphasettler import netformat
from alphasettler._engine import (
    NET_DEFAULT_ROUNDS, NET_DEFAULT_WIDTH, NET_EDGE_WIDTH, NET_ENCODING_HASH, NET_GLOBAL_WIDTH, NET_HIDDEN_LEN,
    NET_NODE_WIDTH, NET_PUBLIC_LEN, NET_SETTLEMENT_BASE, NET_TILE_WIDTH, NET_V2_ENCODING_HASH,
)
# The format's names, importable from here as before.
from alphasettler.netformat import (  # noqa: F401
    ACTIONS, ADJACENCY, EDGES, FIXED_HEADER_LEN, MAGIC, NODES, OTHER, SEATS, TILES, check_arch, layer_count,
)

NODE_END = NODES * NET_NODE_WIDTH
EDGE_END = NODE_END + EDGES * NET_EDGE_WIDTH
TILE_END = EDGE_END + TILES * NET_TILE_WIDTH
assert TILE_END + NET_GLOBAL_WIDTH == NET_PUBLIC_LEN, "nodes, edges, tiles and the globals tile the public vector"


def shapes(width: int, rounds: int) -> list[tuple[int, int]]:
    """(rows, cols) of each layer in file order; rows are outputs (as Rust's `Arch::shapes`)."""
    return netformat.shapes(width, rounds, NET_GLOBAL_WIDTH)


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
    layers = net.layers()
    out = [netformat.header(net.width, net.rounds, [(layer.out_features, layer.in_features) for layer in layers])]
    for layer in layers:
        out += [_floats(layer.weight), _floats(layer.bias)]
    return b"".join(out)


def _check_encoding(encoding: int) -> None:
    if encoding != NET_ENCODING_HASH:
        why = ("; it is an encoder v2 net (before the trade block): convert it with `alphasettler net-upgrade OLD NEW`"
               if encoding == NET_V2_ENCODING_HASH else "")
        raise ValueError(f"encoding hash {encoding:#018x} does not match this encoder's {NET_ENCODING_HASH:#018x}: "
                         f"the file was trained on a different feature layout{why}")


def from_bytes(data: bytes) -> Net:
    """The network in `data`; ValueError (with the Rust loader's wording) for a file it rejects."""
    width, rounds, _, floats = netformat.read(data, NET_GLOBAL_WIDTH, _check_encoding, "network v2")
    params = torch.frombuffer(floats, dtype=torch.float32)
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
