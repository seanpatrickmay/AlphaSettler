"""`alphasettler net-upgrade` (spec 3c Section 4): an encoder v2 net rewritten for encoder v3,
with the same outputs on every no-trade position. Needs the `train` extra for the PyTorch twin,
like test_train.py."""

import random
import struct
import subprocess
import sys

import pytest
import torch

from alphasettler import Bot, Game, describe_action
from alphasettler._engine import (
    NET_GLOBAL_WIDTH, NET_HIDDEN_LEN, NET_PUBLIC_LEN, NET_V2_ENCODING_HASH, NET_V2_GLOBAL_WIDTH, encode,
    net_forward,
)
from alphasettler import netupgrade
from alphasettler.netupgrade import ANSWER_IDS, OFFER_BIAS, OFFER_IDS, TRADE_SLOTS, pooled_row, upgrade
from alphasettler.train.model import ACTIONS, MAGIC, Net, _floats, from_bytes, shapes, to_bytes

NO_TRADES = {"max_offers_per_turn": 0}


def cli(*args):
    return subprocess.run([sys.executable, "-m", "alphasettler", *args], capture_output=True, text=True)


def v2_bytes(net: Net) -> bytes:
    """`net` written as an encoder v2 file: v2's hash, and only the global encoder's first 91 input
    columns."""
    layers = net.layers()
    v2_shapes = [(r, NET_V2_GLOBAL_WIDTH if i == 3 else c) for i, (r, c) in enumerate(shapes(net.width, net.rounds))]
    out = [MAGIC, struct.pack("<IQIII", 2, NET_V2_ENCODING_HASH, net.width, net.rounds, len(layers))]
    out += [struct.pack("<II", r, c) for r, c in v2_shapes]
    for i, layer in enumerate(layers):
        w = layer.weight.detach()
        out += [_floats(w[:, :NET_V2_GLOBAL_WIDTH] if i == 3 else w), _floats(layer.bias)]
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
def test_an_upgraded_net_keeps_every_no_trade_output(tmp_path, width, rounds):
    """In a no-trade game the trade block is all zero, so the original v2 net's outputs are `src`'s
    on the v3 features (its 24 extra global inputs meet only zeros). The upgraded net must give
    the same value and aux, the same logits on every slot but the trade rows, exactly the fixed
    biases on those, and the same prior over legal moves in the Rust forward pass."""
    torch.manual_seed(3)
    src = Net(width, rounds).eval()
    data = upgrade(v2_bytes(src))
    up = from_bytes(data).eval()
    assert (up.global_encoder.weight[:, NET_V2_GLOBAL_WIDTH:] == 0).all()
    path = tmp_path / "up.bin"
    path.write_bytes(data)
    n = 100
    public, hidden, mask = no_trade_positions(n)
    pt = torch.frombuffer(bytearray(public), dtype=torch.float32).view(n, NET_PUBLIC_LEN)
    ht = torch.frombuffer(bytearray(hidden), dtype=torch.float32).view(n, NET_HIDDEN_LEN)
    block = NET_GLOBAL_WIDTH - NET_V2_GLOBAL_WIDTH
    assert (pt[:, NET_PUBLIC_LEN - block:] == 0).all(), "no-trade games leave the trade block zero"
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


def test_only_encoder_v2_nets_are_upgraded():
    torch.manual_seed(0)
    old = v2_bytes(Net(8, 1))
    with pytest.raises(ValueError, match="already an encoder v3 net"):
        upgrade(to_bytes(Net(8, 1)))
    other = bytearray(old)
    other[12] ^= 1
    with pytest.raises(ValueError, match="not encoder v2's"):
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


def test_loaders_send_an_encoder_v2_net_to_net_upgrade(tmp_path):
    torch.manual_seed(0)
    nets = tmp_path / "nets"
    nets.mkdir()
    (nets / "old.bin").write_bytes(v2_bytes(Net(8, 1)))
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


def test_cli_net_upgrade(tmp_path):
    torch.manual_seed(0)
    old, new = tmp_path / "old.bin", tmp_path / "new.bin"
    old.write_bytes(v2_bytes(Net(8, 1)))
    r = cli("net-upgrade", str(old), str(new))
    assert r.returncode == 0, r.stderr
    assert f"wrote {new}" in r.stdout and "width 8, 1 rounds" in r.stdout
    assert from_bytes(new.read_bytes()).width == 8
    r = cli("net-upgrade", str(old), str(new))
    assert r.returncode == 2 and "exists" in r.stderr
    r = cli("net-upgrade", str(new), str(tmp_path / "again.bin"))
    assert r.returncode == 2 and "already an encoder v3 net" in r.stderr
    assert not (tmp_path / "again.bin").exists()


def test_net_upgrade_leaves_nothing_behind_when_the_rust_loader_refuses(tmp_path, monkeypatch, capsys):
    """NEW is published by a rename only after the Rust loader has read the temp file, so a refusal
    leaves neither NEW nor a temp, and a rerun is not stopped by "exists"."""
    from alphasettler import cli as cli_mod

    torch.manual_seed(0)
    old, new = tmp_path / "old.bin", tmp_path / "new.bin"
    old.write_bytes(v2_bytes(Net(8, 1)))

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
