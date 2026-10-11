"""Targets v2 (spec 2026-10-11): the training-only heads, the q mix and the extra losses."""

import torch

from alphasettler._engine import TrainingSet
from alphasettler.train import model as model_mod
from alphasettler.train.data import to_tensors
from alphasettler.train.extras import Q_MIX, WEIGHTS, Extras, extras_losses
from alphasettler.train.train import held_out_losses, losses

from test_training_data import explore_games


def batch(tmp_path, n=32):
    return to_tensors(TrainingSet(explore_games(tmp_path, 2)).sample(n, 0, 1), torch.device("cpu"))


def test_forward_is_unchanged_by_the_trunk_refactor(tmp_path):
    torch.manual_seed(0)
    net = model_mod.Net(8, 1).eval()
    t = batch(tmp_path)
    p, v, a = net(t.public, t.hidden)
    node, edge, tile, glob, pooled, h = net.trunk(t.public, t.hidden)
    assert node.shape[1:] == (54, 8) and edge.shape[1:] == (72, 8) and glob.shape[1:] == (8,)
    assert torch.equal(v, net.value(h)) and torch.equal(a, net.aux(h))
    # A reloaded copy is the same function; bit-identity with the pre-refactor code is pinned by
    # the Rust parity tests (test_train.py), which compare against an unchanged Rust forward pass.
    p2, v2, a2 = model_mod.from_bytes(model_mod.to_bytes(net))(t.public, t.hidden)
    assert torch.equal(p, p2) and torch.equal(v, v2) and torch.equal(a, a2)


def test_without_extras_the_losses_are_todays(tmp_path):
    torch.manual_seed(0)
    net = model_mod.Net(8, 1)
    t = batch(tmp_path)
    plain = losses(net, t)
    assert set(plain) == {"policy", "value", "aux", "l2", "total"}


def test_the_value_target_mixes_z_and_q_on_rows_with_q(tmp_path):
    torch.manual_seed(0)
    net, ex = model_mod.Net(8, 1), Extras(8)
    t = batch(tmp_path)
    t.q = torch.zeros_like(t.q); t.q[:, 0] = 1.0  # q says relative seat 0 wins
    t.has_q = torch.ones_like(t.has_q)
    logits = net(t.public, t.hidden)[1]
    want = -(((1 - Q_MIX) * t.outcome + Q_MIX * t.q) * torch.log_softmax(logits, 1)).sum(1).mean()
    assert torch.allclose(losses(net, t, None, ex)["value"], want, atol=1e-6)
    t.has_q = torch.zeros_like(t.has_q)
    z_only = -(t.outcome * torch.log_softmax(logits, 1)).sum(1).mean()
    assert torch.allclose(losses(net, t, None, ex)["value"], z_only, atol=1e-6)


def test_the_extra_losses_are_weighted_into_the_total_and_train_the_heads(tmp_path):
    torch.manual_seed(0)
    net, ex = model_mod.Net(8, 1), Extras(8)
    t = batch(tmp_path)
    parts = losses(net, t, None, ex)
    assert {"extras_node", "extras_edge", "extras_parts", "extras_road", "extras_army", "extras_turns"} <= set(parts)
    weighted = sum(WEIGHTS[k] * parts[f"extras_{k}"] for k in WEIGHTS)
    assert torch.allclose(parts["extras"], weighted)
    base = parts["policy"] + parts["value"] + 0.25 * parts["aux"] + 1e-4 * parts["l2"]
    assert torch.allclose(parts["total"], base + parts["extras"], atol=1e-5)
    opt = torch.optim.Adam(list(net.parameters()) + list(ex.parameters()), lr=1e-2)
    first = parts["extras"].item()
    for _ in range(30):
        p = losses(net, t, None, ex); opt.zero_grad(); p["total"].backward(); opt.step()
    assert losses(net, t, None, ex)["extras"].item() < first


def test_held_out_value_is_scored_against_z_even_with_extras(tmp_path):
    torch.manual_seed(0)
    net, ex = model_mod.Net(8, 1), Extras(8)
    t = batch(tmp_path)
    plain, with_extras = held_out_losses(net, [t]), held_out_losses(net, [t], ex)
    assert with_extras["value"] == plain["value"] and with_extras["score"] == plain["score"]
    assert "extras_node" in with_extras and "extras_node" not in plain
    net.eval(); ex.eval()
    with torch.no_grad():
        assert with_extras["extras_node"] == losses(net, t, None, ex)["extras_node"].item()


def test_a_bool_mask_and_its_integer_index_give_equal_losses(tmp_path):
    torch.manual_seed(0)
    net, ex = model_mod.Net(8, 1), Extras(8)
    t = batch(tmp_path)
    t.has_q = (torch.arange(t.has_q.shape[0]) % 2).float()
    mask = torch.arange(t.has_q.shape[0]) % 3 == 0
    for e in (None, ex):
        by_mask, by_index = losses(net, t, mask, e), losses(net, t, mask.nonzero().squeeze(1), e)
        assert set(by_mask) == set(by_index) and all(torch.equal(by_mask[k], by_index[k]) for k in by_mask)


def test_value_rows_select_the_rows_of_the_value_and_extra_losses(tmp_path):
    torch.manual_seed(0)
    net, ex = model_mod.Net(8, 1).eval(), Extras(8)
    t = batch(tmp_path)
    t.q = torch.zeros_like(t.q); t.q[:, 1] = 1.0
    t.has_q = (torch.arange(t.has_q.shape[0]) % 2).float()
    rows = torch.arange(t.has_q.shape[0]) % 3 == 0
    picked = losses(net, t, rows, ex)
    sliced = type(t)(**{k: x[rows] for k, x in vars(t).items()})
    alone = losses(net, sliced, None, ex)
    for key in ("value", "aux", "extras_node", "extras_edge", "extras_parts", "extras_road", "extras_army",
                "extras_turns", "extras"):
        assert torch.allclose(picked[key], alone[key], atol=1e-6), key
    assert torch.allclose(picked["policy"], losses(net, t)["policy"])
