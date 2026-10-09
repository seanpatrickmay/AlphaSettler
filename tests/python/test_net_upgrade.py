"""`alphasettler net-upgrade` (specs 3c Section 4 and 3d-2 Section 3): an older net rewritten for
this build's encoder, v4. Needs the `train` extra for the PyTorch twin, like test_train.py."""

import json
import random
import struct
import subprocess
import sys
from array import array

import pytest
import torch

from alphasettler import Bot, Game, describe_action, netformat
from alphasettler._engine import (
    NET_GLOBAL_WIDTH, NET_HIDDEN_LEN, NET_PUBLIC_LEN, NET_V2_ENCODING_HASH, NET_V2_GLOBAL_WIDTH, NET_V3_EDGE_WIDTH,
    NET_V3_ENCODING_HASH, NET_V3_GLOBAL_WIDTH, NET_V3_NODE_WIDTH, encode, net_forward,
)
from alphasettler import netupgrade
from alphasettler.netupgrade import ANSWER_IDS, OFFER_BIAS, OFFER_IDS, TRADE_SLOTS, pooled_row, upgrade
from alphasettler.train.fastloop import STATE, FastLoop, FastLoopSettings
from alphasettler.train.model import ACTIONS, MAGIC, TILE_END, Net, _floats, from_bytes, shapes, to_bytes

from test_fast_loop import TINY
from test_train import assert_parity, generation, random_positions

NO_TRADES = {"max_offers_per_turn": 0}
# (encoding hash, node, edge and global input widths) of encoder v2; its node and edge inputs are v3's.
V2 = (NET_V2_ENCODING_HASH, NET_V3_NODE_WIDTH, NET_V3_EDGE_WIDTH, NET_V2_GLOBAL_WIDTH)
V3 = (NET_V3_ENCODING_HASH, NET_V3_NODE_WIDTH, NET_V3_EDGE_WIDTH, NET_V3_GLOBAL_WIDTH)
# Encoder v3's trade block in a v4 feature vector: globals 91 .. 115.
TRADE_BLOCK = slice(TILE_END + NET_V2_GLOBAL_WIDTH, TILE_END + NET_V3_GLOBAL_WIDTH)


def cli(*args):
    return subprocess.run([sys.executable, "-m", "alphasettler", *args], capture_output=True, text=True)


def old_net(widths, width=8, rounds=1, seed=0) -> Net:
    """A random twin with zero encoder weights for every input past an older encoder's `widths`
    (hash, node, edge, global): that older net, as this build reads it."""
    _, node, edge, glob = widths
    torch.manual_seed(seed)
    net = Net(width, rounds).eval()
    with torch.no_grad():
        for layer, keep in ((net.node_encoder, node), (net.edge_encoder, edge), (net.global_encoder, glob)):
            layer.weight[:, keep:] = 0.0
    return net


def old_bytes(net: Net, widths) -> bytes:
    """`net` written as an older encoder's file: its hash, and each encoder's first input columns only."""
    encoding, node, edge, glob = widths
    keep = {0: node, 1: edge, 3: glob}
    layers = net.layers()
    out = [MAGIC, struct.pack("<IQIII", 2, encoding, net.width, net.rounds, len(layers))]
    out += [struct.pack("<II", r, c) for r, c in netformat.shapes(net.width, net.rounds, glob, node, edge)]
    for i, layer in enumerate(layers):
        w = layer.weight.detach()
        out += [_floats(w[:, :keep[i]] if i in keep else w), _floats(layer.bias)]
    return b"".join(out)


def no_trade_positions(n: int):
    """`n` encoded positions from random no-trade games, and each one's legal-move mask."""
    rng = random.Random(1)
    public, hidden, mask = [], [], torch.zeros(n, ACTIONS, dtype=torch.bool)
    seed = 0
    while len(public) < n:
        g = Game(seed, NO_TRADES)
        seed += 1
        for k in range(300):
            if g.is_over or len(public) == n:
                break
            if k % 5 == 0:
                p, h = encode(g)
                mask[len(public), g.legal_actions()] = True
                public.append(p)
                hidden.append(h)
            g.apply(rng.choice(g.legal_actions()))
    return b"".join(public), b"".join(hidden), mask


def _close(got, ref) -> float:
    return ((got - ref).abs() / ref.abs().clamp(min=1.0)).max().item()


@pytest.mark.parametrize("width,rounds", [(48, 2), (5, 3)])
def test_an_upgraded_v2_net_keeps_every_no_trade_output(tmp_path, width, rounds):
    """`src` reads only a v2 net's inputs, so its outputs are that v2 net's. The upgrade must be
    `src` exactly but for the trade rows (zero weights, the fixed biases), so in a no-trade game,
    where the trade block is all zero, it gives `src`'s value, aux and logits on every slot but the
    trade rows, and the Rust forward pass the same prior over legal moves."""
    src = old_net(V2, width, rounds, seed=3)
    data = upgrade(old_bytes(src, V2))
    up = from_bytes(data).eval()
    for i, (a, b) in enumerate(zip(up.layers(), src.layers())):
        if a is not up.other_policy:
            assert torch.equal(a.weight, b.weight) and torch.equal(a.bias, b.bias), i
    path = tmp_path / "up.bin"
    path.write_bytes(data)
    n = 100
    public, hidden, mask = no_trade_positions(n)
    pt = torch.frombuffer(bytearray(public), dtype=torch.float32).view(n, NET_PUBLIC_LEN)
    ht = torch.frombuffer(bytearray(hidden), dtype=torch.float32).view(n, NET_HIDDEN_LEN)
    assert (pt[:, TRADE_BLOCK] == 0).all(), "no-trade games leave the trade block zero"
    with torch.no_grad():
        ref, got = src(pt, ht), up(pt, ht)
    trade = torch.zeros(ACTIONS, dtype=torch.bool)
    trade[list(TRADE_SLOTS)] = True
    assert not mask[:, trade].any(), "no trade move is legal in a no-trade game"
    assert _close(got[0][:, ~trade], ref[0][:, ~trade]) <= 1e-5
    assert (got[0][:, list(OFFER_IDS)] == OFFER_BIAS).all()
    assert (got[0][:, list(ANSWER_IDS)] == 0).all()
    assert _close(got[1], ref[1]) <= 1e-5 and _close(got[2], ref[2]) <= 1e-5
    rust = net_forward(str(path), public, hidden)
    rp = torch.frombuffer(bytearray(rust["policy"]), dtype=torch.float32).view(n, ACTIONS)
    rv = torch.frombuffer(bytearray(rust["value"]), dtype=torch.float32).view(n, 4)

    def prior(x):
        return torch.softmax(x.double().masked_fill(~mask, float("-inf")), 1)

    assert (prior(rp) - prior(ref[0])).abs().max().item() <= 1e-5
    assert (torch.softmax(rv.double(), 1) - torch.softmax(ref[1].double(), 1)).abs().max().item() <= 1e-5


@pytest.mark.parametrize("widths", [V2, V3], ids=["v2", "v3"])
def test_only_encoder_v2_and_v3_nets_are_upgraded(widths):
    old = old_bytes(old_net(widths), widths)
    with pytest.raises(ValueError, match="already an encoder v4 net"):
        upgrade(to_bytes(Net(8, 1)))
    other = bytearray(old)
    other[12] ^= 1
    with pytest.raises(ValueError, match="neither encoder v2's .* nor encoder v3's"):
        upgrade(bytes(other))
    with pytest.raises(ValueError, match="wrong magic"):
        upgrade(b"XXNET\0\0\0" + old[8:])
    one = bytearray(old)
    one[8:12] = (1).to_bytes(4, "little")
    with pytest.raises(ValueError, match="format version 1"):
        upgrade(bytes(one))
    with pytest.raises(ValueError, match="truncated"):
        upgrade(old[:20])
    with pytest.raises(ValueError, match="truncated"):
        upgrade(old[:-4])
    with pytest.raises(ValueError, match="trailing bytes"):
        upgrade(old + b"\0\0\0\0")
    wide = bytearray(old)
    struct.pack_into("<I", wide, 20, 2**31)
    with pytest.raises(ValueError, match="width 2147483648 is out of range"):
        upgrade(bytes(wide))
    nan = bytearray(old)
    struct.pack_into("<f", nan, len(nan) - 4, float("nan"))
    with pytest.raises(ValueError, match="NaN or infinite"):
        upgrade(bytes(nan))
    assert from_bytes(upgrade(old)).width == 8


@pytest.mark.parametrize("widths", [V2, V3], ids=["v2", "v3"])
def test_loaders_send_an_older_net_to_net_upgrade(tmp_path, widths):
    nets = tmp_path / "nets"
    nets.mkdir()
    (nets / "old.bin").write_bytes(old_bytes(old_net(widths), widths))
    p, h = encode(Game(0, NO_TRADES))
    loads = [lambda: net_forward(str(nets / "old.bin"), p, h),
             lambda: from_bytes((nets / "old.bin").read_bytes()),
             lambda: Bot("net:old", 0, str(nets))]
    for load in loads:
        with pytest.raises(ValueError, match="net-upgrade"):
            load()


def test_the_trade_rows_are_the_engines_offer_answer_and_confirmation_ids():
    assert len(OFFER_IDS) == 400 and all(describe_action(a).startswith("OfferTrade") for a in OFFER_IDS)
    assert [describe_action(a) for a in ANSWER_IDS] == [
        "AcceptTrade", "RejectTrade", "ConfirmTrade(0)", "ConfirmTrade(1)", "ConfirmTrade(2)", "ConfirmTrade(3)",
        "CancelTrade"]
    assert (pooled_row(0), pooled_row(24), pooled_row(258), pooled_row(664)) == (0, 24, 59, 465)
    with pytest.raises(ValueError, match="entity head"):
        pooled_row(25)
    assert netupgrade.shapes(48, 2, NET_GLOBAL_WIDTH) == shapes(48, 2)


@pytest.mark.parametrize("widths,did", [
    # 24 - 19 node, 8 - 4 edge, 139 - 91 global inputs; offers 258 .. 664: 407 rows.
    (V2, "(encoder v2) to encoder v4: width 8, 1 rounds; 5 node, 4 edge and 48 global inputs zeroed; "
         "407 trade policy rows reset (offer bias -3); wrote"),
    # 139 - 115 global inputs; v3 nets trained their trade rows: none reset.
    (V3, "(encoder v3) to encoder v4: width 8, 1 rounds; 5 node, 4 edge and 24 global inputs zeroed; wrote"),
], ids=["v2", "v3"])
def test_cli_net_upgrade(tmp_path, widths, did):
    old, new = tmp_path / "old.bin", tmp_path / "new.bin"
    old.write_bytes(old_bytes(old_net(widths), widths))
    r = cli("net-upgrade", str(old), str(new))
    assert r.returncode == 0, r.stderr
    assert f"{did} {new}" in r.stdout
    assert from_bytes(new.read_bytes()).width == 8
    r = cli("net-upgrade", str(old), str(new))
    assert r.returncode == 2 and "exists" in r.stderr
    r = cli("net-upgrade", str(new), str(tmp_path / "again.bin"))
    assert r.returncode == 2 and "already an encoder v4 net" in r.stderr
    assert not (tmp_path / "again.bin").exists()


def test_net_upgrade_leaves_nothing_behind_when_the_rust_loader_refuses(tmp_path, monkeypatch, capsys):
    """NEW is published by a rename only after the Rust loader has read the temp file, so a refusal
    leaves neither NEW nor a temp, and a rerun is not stopped by "exists"."""
    from alphasettler import cli as cli_mod

    old, new = tmp_path / "old.bin", tmp_path / "new.bin"
    old.write_bytes(old_bytes(old_net(V2), V2))

    def refuse(path, public, hidden):
        assert path != str(new) and (tmp_path / path).exists(), "the check reads the temp file, not NEW"
        raise ValueError("refused")

    monkeypatch.setattr(netupgrade, "net_forward", refuse)
    assert cli_mod.main(["net-upgrade", str(old), str(new)]) == 2
    assert "the Rust loader refused the upgrade: refused" in capsys.readouterr().err
    assert sorted(p.name for p in tmp_path.iterdir()) == ["old.bin"]
    monkeypatch.undo()
    assert cli_mod.main(["net-upgrade", str(old), str(new)]) == 0
    assert from_bytes(new.read_bytes()).width == 8


@pytest.mark.parametrize("width,rounds", [(48, 2), (5, 3)])
def test_an_upgraded_v3_net_gives_the_v3_nets_outputs_on_every_position(tmp_path, width, rounds):
    """A v3 net reads only the v3 slots of a v4 feature vector, which are its v3 features: `src` is
    that net, with zero weights on the rest. The upgrade must be `src` exactly (nothing reset), and
    its Rust forward pass must agree with `src` by the parity rule on positions of random trading
    games, where the road distances, trails, open spots and threats v4 added are all in play."""
    src = old_net(V3, width, rounds, seed=3)
    data = upgrade(old_bytes(src, V3))
    up = from_bytes(data).eval()
    for i, (a, b) in enumerate(zip(up.layers(), src.layers())):
        assert torch.equal(a.weight, b.weight) and torch.equal(a.bias, b.bias), i
    path = tmp_path / "up.bin"
    path.write_bytes(data)
    n = 100
    public, hidden, mask = random_positions(n)
    pt = torch.frombuffer(bytearray(public), dtype=torch.float32).view(n, NET_PUBLIC_LEN)
    ht = torch.frombuffer(bytearray(hidden), dtype=torch.float32).view(n, NET_HIDDEN_LEN)
    threats = pt[:, TILE_END + NET_V3_GLOBAL_WIDTH:]
    assert (threats != threats[:1]).any(), "the threat block varies"
    with torch.no_grad():
        ref = src(pt, ht)
    assert_parity(net_forward(str(path), public, hidden), ref, mask)


def test_a_fast_loop_runs_from_an_upgraded_v3_net(tmp_path):
    seed = generation(tmp_path, "seed-gen", seed_start=0)
    old, start = tmp_path / "v3.bin", tmp_path / "start.bin"
    old.write_bytes(old_bytes(old_net(V3), V3))
    r = cli("net-upgrade", str(old), str(start))
    assert r.returncode == 0, r.stderr
    root = tmp_path / "loop"
    FastLoop.init(root, [seed], start, first=1)
    assert FastLoop(root, FastLoopSettings(generations=1, **TINY)).run() == 1
    state = json.loads((root / STATE).read_text())
    assert [g["gen"] for g in state["generations"]] == [1]
    assert (root / "nets" / "start.bin").read_bytes() == start.read_bytes()


def test_a_layer_is_never_narrowed():
    """`widened` only appends zero columns: fewer columns than the layer has would drop weights."""
    layer = (2, 3, array("f", [1, 2, 3, 4, 5, 6]), array("f", [0, 0]))
    assert netupgrade.widened(layer, 4)[2].tolist() == [1, 2, 3, 0, 4, 5, 6, 0]
    assert netupgrade.widened(layer, 3)[2] == layer[2]
    with pytest.raises(ValueError, match="cannot narrow"):
        netupgrade.widened(layer, 2)
