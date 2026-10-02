"""The PyTorch twin of net/src/forward.rs, and the weights file format of net/src/weights.rs:
little-endian, an 80-byte header (magic, format version, ENCODING_HASH, layer count, each layer's
rows and columns), then each layer's f32 weights (row-major, rows = outputs) and biases.

`from_bytes` checks a file exactly as the Rust loader does, and checks all of it (header, length,
every weight finite) before it builds a module, so a corrupt header never sizes an allocation."""

from __future__ import annotations

import struct
import sys
from array import array
from pathlib import Path

import torch
from torch import nn

from alphasettler._engine import (
    NET_ENCODING_HASH, NET_FORMAT_VERSION, NET_HIDDEN_LEN, NET_MAX_WIDTH, NET_PUBLIC_LEN, action_space_size,
)

MAGIC = b"ASNET\0\0\0"
ACTIONS = action_space_size()
SEATS = 4
NUM_LAYERS = 7
HEADER_LEN = 8 + 4 + 8 + 4 + NUM_LAYERS * 8


def _check_width(what: str, width: int) -> None:
    if not 1 <= width <= NET_MAX_WIDTH:
        raise ValueError(f"the {what} width must be 1..={NET_MAX_WIDTH}, got {width}")


def shapes(trunk: int, value: int) -> list[tuple[int, int]]:
    """(rows, cols) of each layer in file order; rows are outputs (as Rust's `Arch::shapes`)."""
    return [(trunk, NET_PUBLIC_LEN), (trunk, trunk), (trunk, trunk), (ACTIONS, trunk),
            (value, trunk + NET_HIDDEN_LEN), (SEATS, value), (SEATS, value)]


class Net(nn.Module):
    """public -> trunk (3 x ReLU) -> policy logits; [trunk || hidden] -> ReLU -> value logits
    and the aux head (final VP / 10). All value and aux outputs are seat-relative."""

    def __init__(self, trunk: int = 512, value: int = 256):
        _check_width("trunk", trunk)
        _check_width("value", value)
        super().__init__()
        self.trunk_width, self.value_width = trunk, value
        self.trunk = nn.ModuleList([nn.Linear(NET_PUBLIC_LEN, trunk), nn.Linear(trunk, trunk), nn.Linear(trunk, trunk)])
        self.policy = nn.Linear(trunk, ACTIONS)
        self.value0 = nn.Linear(trunk + NET_HIDDEN_LEN, value)
        self.value1 = nn.Linear(value, SEATS)
        self.aux = nn.Linear(value, SEATS)

    def layers(self) -> list[nn.Linear]:
        """In weights-file order."""
        return [*self.trunk, self.policy, self.value0, self.value1, self.aux]

    def forward(self, public: torch.Tensor, hidden: torch.Tensor):
        h = public
        for layer in self.trunk:
            h = torch.relu(layer(h))
        v = torch.relu(self.value0(torch.cat([h, hidden], dim=1)))
        return self.policy(h), self.value1(v), self.aux(v)


def _floats(t: torch.Tensor) -> bytes:
    return array("f", t.detach().to("cpu", torch.float32).contiguous().view(-1).tolist()).tobytes()


def to_bytes(net: Net) -> bytes:
    assert sys.byteorder == "little", "the weights file is little-endian"
    layers = net.layers()
    out = [MAGIC, struct.pack("<IQI", NET_FORMAT_VERSION, NET_ENCODING_HASH, len(layers))]
    out += [struct.pack("<II", layer.out_features, layer.in_features) for layer in layers]
    for layer in layers:
        out += [_floats(layer.weight), _floats(layer.bias)]
    return b"".join(out)


def from_bytes(data: bytes) -> Net:
    """The network in `data`; ValueError (with the Rust loader's wording) for a file it rejects."""
    assert sys.byteorder == "little", "the weights file is little-endian"
    if len(data) < HEADER_LEN:
        raise ValueError(f"truncated: {len(data)} bytes is shorter than the {HEADER_LEN}-byte header")
    if data[:8] != MAGIC:
        raise ValueError("not an AlphaSettler weights file (wrong magic)")
    version, encoding, count = struct.unpack_from("<IQI", data, 8)
    if version != NET_FORMAT_VERSION:
        raise ValueError(f"format version {version}, expected {NET_FORMAT_VERSION}")
    if encoding != NET_ENCODING_HASH:
        raise ValueError(f"encoding hash {encoding:#018x} does not match this encoder's {NET_ENCODING_HASH:#018x}: "
                         "the file was trained on a different feature layout")
    if count != NUM_LAYERS:
        raise ValueError(f"{count} layers, expected {NUM_LAYERS}")
    declared = [struct.unpack_from("<II", data, 24 + 8 * i) for i in range(NUM_LAYERS)]
    trunk, value = declared[0][0], declared[4][0]
    widths_ok = 1 <= trunk <= NET_MAX_WIDTH and 1 <= value <= NET_MAX_WIDTH
    if not widths_ok or declared != shapes(trunk, value):
        raise ValueError(f"layer shapes {declared} do not fit the network (trunk and value widths must be "
                         f"1..={NET_MAX_WIDTH}, and the other dimensions are fixed by the encoder)")
    want = HEADER_LEN + 4 * sum(r * c + r for r, c in declared)
    if len(data) < want:
        raise ValueError(f"truncated: {len(data)} bytes, expected {want}")
    if len(data) > want:
        raise ValueError(f"{len(data) - want} trailing bytes after the last layer")
    params = torch.frombuffer(bytearray(data[HEADER_LEN:]), dtype=torch.float32)
    if not torch.isfinite(params).all():
        raise ValueError("a weight is NaN or infinite")
    net = Net(trunk=trunk, value=value)
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
