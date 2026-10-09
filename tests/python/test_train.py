"""Training (spec Section 4): the PyTorch twin, the data window, the training run and `train`.

These tests need the `train` extra (PyTorch: `.venv/bin/pip3 install -e '.[train]'`). Without it
they fail at collection on `import torch`, on purpose: they must be run, never skipped."""

import gzip
import math
import random
import struct
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from alphasettler import Bot, Game, files, records
from alphasettler._engine import (
    NET_DEFAULT_ROUNDS, NET_DEFAULT_WIDTH, NET_EDGE_WIDTH, NET_ENCODING_HASH, NET_GLOBAL_WIDTH, NET_HIDDEN_LEN,
    NET_MAX_ROUNDS, NET_MAX_WIDTH, NET_NODE_WIDTH, NET_PUBLIC_LEN, NET_V3_EDGE_WIDTH, NET_V3_GLOBAL_WIDTH,
    NET_V3_NODE_WIDTH, encode, net_forward,
)
from alphasettler.selfplay import write_explore
from alphasettler.train import model
from alphasettler.train.data import load_window, record_files, to_tensors
from alphasettler.train.model import ACTIONS, EDGE_END, EDGES, FIXED_HEADER_LEN, NODE_END, NODES, Net, from_bytes, shapes, to_bytes
from alphasettler.train import train as train_mod
from alphasettler.train.train import (
    NEWEST_SHARE, VALUE_FRACTION, TrainConfig, auto_steps, batch_split, held_out_batches, held_out_losses, losses,
    parity_check, parity_error, pick_device, sample_batch, train, value_rows,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG = {"max_offers_per_turn": 0}
CPU = torch.device("cpu")


def cli(*args):
    return subprocess.run([sys.executable, "-m", "alphasettler", *args], capture_output=True, text=True)


def write_games(path, seed_start, games):
    write_explore(path, seed_start, games, config=CONFIG, full_simulations=6, fast_simulations=3,
                  full_probability=0.5, threads=2)


def generation(tmp_path, name="gen-000", seed_start=0, games=2):
    d = tmp_path / name
    d.mkdir(parents=True, exist_ok=True)
    write_games(d / f"selfplay-{seed_start}.jsonl.gz", seed_start, games)
    return d


def random_positions(n, seed=0):
    """`n` encoded positions from uniformly random games (trades on), every 7th move, and each
    position's legal-move mask."""
    rng = random.Random(seed)
    public, hidden, mask = [], [], torch.zeros(n, ACTIONS, dtype=torch.bool)
    game_seed = 0
    while len(public) < n:
        g = Game(game_seed, None)
        game_seed += 1
        for k in range(400):
            if g.is_over or len(public) == n:
                break
            if k % 7 == 0:
                p, h = encode(g)
                mask[len(public), g.legal_actions()] = True
                public.append(p)
                hidden.append(h)
            g.apply(rng.choice(g.legal_actions()))
    return b"".join(public), b"".join(hidden), mask


def assert_parity(rust, outs, mask):
    """The spec's parity rule (preflight F9) between `net_forward`'s `rust` and PyTorch's `outs`
    (policy, value, aux): raw logits and aux within 1e-5 x max(1, |PyTorch|); the prior over the
    legal moves (`mask`) and the value probabilities within 1e-5 absolute."""
    got = {key: torch.frombuffer(bytearray(rust[key]), dtype=torch.float32).view_as(ref)
           for key, ref in zip(("policy", "value", "aux"), outs)}
    for key, ref in zip(("policy", "value", "aux"), outs):
        err = ((got[key] - ref).abs() / ref.abs().clamp(min=1.0)).max().item()
        assert err <= 1e-5, (key, err)

    def prior(logits):
        return torch.softmax(logits.double().masked_fill(~mask, float("-inf")), dim=1)

    assert (prior(got["policy"]) - prior(outs[0])).abs().max().item() <= 1e-5
    value = (torch.softmax(got["value"].double(), 1) - torch.softmax(outs[1].double(), 1)).abs().max().item()
    assert value <= 1e-5


@pytest.mark.parametrize("width,rounds", [(NET_DEFAULT_WIDTH, NET_DEFAULT_ROUNDS), (5, 3)])
def test_rust_and_pytorch_forward_passes_agree_on_random_weights(tmp_path, width, rounds):
    """Spec parity rule (preflight F9): the prior over legal moves and the value probabilities
    agree within 1e-5 absolute; raw logits and the aux head within 1e-5 x max(1, |PyTorch|). The
    default net, and an odd width (not a multiple of Rust's 8-lane dot product) with 3 rounds."""
    torch.manual_seed(0)
    net = Net(width, rounds).eval()
    path = tmp_path / "r.bin"
    path.write_bytes(to_bytes(net))
    public, hidden, mask = random_positions(100)
    rust = net_forward(str(path), public, hidden)
    pt = torch.frombuffer(bytearray(public), dtype=torch.float32).view(100, NET_PUBLIC_LEN)
    ht = torch.frombuffer(bytearray(hidden), dtype=torch.float32).view(100, NET_HIDDEN_LEN)
    with torch.no_grad():
        outs = net(pt, ht)
    assert_parity(rust, outs, mask)


def test_the_parity_positions_fill_every_v4_input_block():
    """The parity test above compares the two forward passes on these positions: they must reach
    every block encoder v4 added, or a twin that misread one would still pass."""
    public, _, _ = random_positions(100)
    pt = torch.frombuffer(bytearray(public), dtype=torch.float32).view(100, NET_PUBLIC_LEN)
    nodes = pt[:, :NODE_END].reshape(100, NODES, NET_NODE_WIDTH)
    edges = pt[:, NODE_END:EDGE_END].reshape(100, EDGES, NET_EDGE_WIDTH)
    distance = nodes[:, :, NET_V3_NODE_WIDTH:NET_V3_NODE_WIDTH + 4]
    assert ((distance > 0) & (distance < 1)).any(), "a node one or two roads from some seat"
    open_spot = nodes[:, :, NET_NODE_WIDTH - 1]
    assert (open_spot == 1).any() and (open_spot == 0).any()
    assert (edges[:, :, NET_V3_EDGE_WIDTH:] == 1).any(), "a road on a longest trail"
    threats = pt[:, NET_PUBLIC_LEN - (NET_GLOBAL_WIDTH - NET_V3_GLOBAL_WIDTH):]
    assert (threats != threats[:1]).any(), "the threat block changes over the games"


def test_a_nan_anywhere_is_an_infinite_parity_error():
    """`max` and `>` both let a NaN through; the parity rule must not."""
    n = 3
    mask = torch.ones(n, ACTIONS, dtype=torch.bool)
    ref = (torch.zeros(n, ACTIONS), torch.zeros(n, 4), torch.zeros(n, 4))
    clean = {k: bytes(4 * r.numel()) for k, r in zip(("policy", "value", "aux"), ref)}
    assert parity_error(ref, clean, mask) == 0.0
    for key in ("policy", "value", "aux"):
        bad = bytearray(clean[key])
        struct.pack_into("<f", bad, 4, float("nan"))
        assert parity_error(ref, {**clean, key: bytes(bad)}, mask) == float("inf"), key
    nan_ref = (ref[0], ref[1], torch.full((n, 4), float("nan")))
    assert parity_error(nan_ref, clean, mask) == float("inf")


def test_the_default_net_is_width_48_with_2_rounds():
    net = Net()
    assert (net.width, net.rounds) == (NET_DEFAULT_WIDTH, NET_DEFAULT_ROUNDS) == (48, 2)
    assert [(layer.out_features, layer.in_features) for layer in net.layers()] == shapes(48, 2)
    assert sum(p.numel() for p in net.parameters()) == 250_030, "the same count as Rust's Arch::DEFAULT"
    assert len(to_bytes(net)) == FIXED_HEADER_LEN + 8 * 20 + 4 * 250_030
    assert not any("tile_nodes" in k or "edge_nodes" in k for k in net.state_dict()), "the adjacency is not a weight"


def test_weights_round_trip_and_bad_files_are_errors():
    torch.manual_seed(1)
    net = Net(8, 1)
    data = to_bytes(net)
    back = from_bytes(data)
    assert (back.width, back.rounds) == (8, 1)
    assert all(torch.equal(a, b) for a, b in zip(net.state_dict().values(), back.state_dict().values()))

    def patch(at, value):
        b = bytearray(data)
        struct.pack_into("<I", b, at, value)
        return bytes(b)

    for bad, message in [(b"X" + data[1:], "magic"), (data[:12] + bytes([data[12] ^ 1]) + data[13:], "encoding hash"),
                         (patch(8, 1), "format version 1, expected 2; version 1 holds the flat MLP"),
                         (patch(8, 3), "format version 3, expected 2$"),
                         (patch(24, 2), "16 layers, expected 20 for 2 rounds"), (patch(28, 15), "15 layers, expected 16"),
                         (patch(FIXED_HEADER_LEN + 8, 7), "shapes"), (data[:-1], "truncated"),
                         (data[:40], "shorter than the 160-byte header"), (data[:20], "shorter than the 32-byte header"),
                         (data + b"\0", "trailing")]:
        with pytest.raises(ValueError, match=message):
            from_bytes(bad)


def header(width, rounds):
    """The 32-byte fixed header of a version 2 file declaring this width and round count."""
    return b"ASNET\0\0\0" + struct.pack("<IQIII", 2, NET_ENCODING_HASH, width, rounds, (12 + 4 * rounds) % 2**32)


def test_widths_and_rounds_are_checked_and_non_finite_weights_rejected_before_building_a_net(monkeypatch):
    """Preflight F4: same rules as the Rust loader, and no torch module is built from an
    unvalidated header."""
    torch.manual_seed(4)
    good = to_bytes(Net(3, 2))
    assert good[:FIXED_HEADER_LEN] == header(3, 2)

    def no_net(*args, **kwargs):
        raise AssertionError("a Net was built before the file was validated")

    monkeypatch.setattr(model, "Net", no_net)
    for width, rounds, message in [(NET_MAX_WIDTH + 1, 1, "width 1025 is out of range"), (0, 1, "width 0 is out"),
                                   (2**31, 1, "width 2147483648 is out"), (8, 0, "rounds 0 is out of range"),
                                   (8, NET_MAX_ROUNDS + 1, "rounds 17 is out of range")]:
        with pytest.raises(ValueError, match=message):
            from_bytes(header(width, rounds))
    with pytest.raises(ValueError, match="do not fit network v2"):  # at the bound: accepted, then the shapes differ
        from_bytes(header(NET_MAX_WIDTH, 2) + good[FIXED_HEADER_LEN:])
    for x in (float("nan"), float("inf"), float("-inf")):
        bad = bytearray(good)
        struct.pack_into("<f", bad, FIXED_HEADER_LEN + 8 * 20 + 4, x)
        with pytest.raises(ValueError, match="NaN or infinite"):
            from_bytes(bytes(bad))
    with pytest.raises(ValueError, match=f"width 1025 is out of range 1..={NET_MAX_WIDTH}"):
        Net(NET_MAX_WIDTH + 1, 1)  # the class itself, not the patched module name
    with pytest.raises(ValueError, match=f"rounds 17 is out of range 1..={NET_MAX_ROUNDS}"):
        Net(8, NET_MAX_ROUNDS + 1)


def test_the_loss_is_the_spec_sum_with_l2_over_weights_only(tmp_path):
    """Policy over full-search rows, value and 0.25 x aux over the value rows only, L2 over weights."""
    w = load_window(tmp_path, [generation(tmp_path)])
    t = to_tensors(w.newest.sample(32, 7, 2), CPU)
    torch.manual_seed(6)
    net = Net(8, 1)
    rows = torch.zeros(32, dtype=torch.bool)
    rows[[3, 17, 30]] = True
    with torch.no_grad():
        for layer in net.layers():
            layer.bias.fill_(0.5)  # biases large enough that counting them would show
        parts = losses(net, t, rows)
        _, value, aux = net(t.public, t.hidden)
    l2 = sum(layer.weight.pow(2).sum() for layer in net.layers())
    assert torch.allclose(parts["l2"], l2)
    assert not torch.allclose(parts["l2"], l2 + sum(layer.bias.pow(2).sum() for layer in net.layers()))
    value_ce = -(t.outcome * torch.log_softmax(value, 1)).sum(1)
    assert torch.allclose(parts["value"], value_ce[rows].mean())
    assert torch.allclose(parts["aux"], torch.nn.functional.mse_loss(aux[rows], t.vp[rows]))
    total = parts["policy"] + parts["value"] + 0.25 * parts["aux"] + 1e-4 * l2
    assert torch.allclose(parts["total"], total)
    with torch.no_grad():
        every = losses(net, t)  # no rows given: every row (the held-out evaluation)
    assert torch.allclose(every["value"], value_ce.mean())
    assert torch.allclose(every["aux"], torch.nn.functional.mse_loss(aux, t.vp))
    assert torch.equal(every["policy"], parts["policy"])


def test_value_and_aux_terms_use_only_the_value_rows(tmp_path):
    """Changing the outcome or final VP of a row outside the value rows leaves the loss exactly as
    it was; changing one inside moves it."""
    w = load_window(tmp_path, [generation(tmp_path)])
    t = to_tensors(w.newest.sample(64, 11, 2), CPU)
    torch.manual_seed(8)
    net = Net(8, 1)
    rows = value_rows(64, 5, 9)
    assert 0 < rows.sum().item() < 64
    base = losses(net, t, rows)
    other = torch.roll(t.outcome, 1, dims=1)  # another seat wins
    outside = replace(t, outcome=torch.where(rows[:, None], t.outcome, other),
                      vp=torch.where(rows[:, None], t.vp, 1.0 - t.vp))
    changed = losses(net, outside, rows)
    for key in ("value", "aux", "total"):
        assert torch.equal(changed[key], base[key]), key
    inside = replace(t, outcome=torch.where(rows[:, None], other, t.outcome),
                     vp=torch.where(rows[:, None], 1.0 - t.vp, t.vp))
    moved = losses(net, inside, rows)
    assert not torch.equal(moved["value"], base["value"]) and not torch.equal(moved["aux"], base["aux"])


def test_value_rows_are_a_seeded_sixteenth_of_the_batch_and_never_none():
    assert VALUE_FRACTION == 1 / 16
    assert torch.equal(value_rows(1024, 3, 7), value_rows(1024, 3, 7))
    assert not torch.equal(value_rows(1024, 3, 7), value_rows(1024, 3, 8))
    assert not torch.equal(value_rows(1024, 3, 7), value_rows(1024, 4, 7))
    draws, n = 200, 1024
    kept = sum(value_rows(n, 1, step).sum().item() for step in range(draws))
    total = draws * n
    se = math.sqrt(total * VALUE_FRACTION * (1 - VALUE_FRACTION))
    assert abs(kept - total * VALUE_FRACTION) < 4 * se
    singles = [value_rows(1, 0, step) for step in range(40)]
    assert all(r.dtype == torch.bool and r.tolist() == [True] for r in singles), "row 0 when none is drawn"
    assert value_rows(4, -1, 2**40).shape == (4,)  # any seed and step are usable


def test_steps_are_twice_the_new_positions_over_the_batch():
    assert auto_steps(1024, 1024) == 2
    assert auto_steps(1, 1024) == 1
    assert auto_steps(5000, 1024) == 10
    assert auto_steps(5000, 1024, 3.0) == 15
    # The warm start (generation 0, no older data): every row from the newest, the count unchanged.
    assert auto_steps(23_292_888, 1024, 2.0) == auto_steps(23_292_888, 1024, 2.0, 1.0) == 45_494
    # With older data half of each batch is new, so twice the steps draw each new position twice.
    assert NEWEST_SHARE == 0.5
    assert auto_steps(23_292_888, 1024, 2.0, NEWEST_SHARE) == 90_988
    assert auto_steps(5000, 1024, 2.0, 0.5) == 20


def rows(batch):
    """A sampler batch's rows as (public, hidden) byte strings."""
    pub, hid = 4 * NET_PUBLIC_LEN, 4 * NET_HIDDEN_LEN
    return [(batch["public"][i * pub:(i + 1) * pub], batch["hidden"][i * hid:(i + 1) * hid]) for i in range(batch["n"])]


def every_row(data):
    """Every position of a training set, as `rows` gives them."""
    out, game, decision = set(), 0, 0
    while game < data.games:
        try:
            out.update(rows(data.position(game, decision)))
            decision += 1
        except ValueError:  # past the game's last decision
            game, decision = game + 1, 0
    return out


def two_generations(tmp_path):
    old = generation(tmp_path, "gen-000", seed_start=1, games=2)
    new = generation(tmp_path, "gen-001", seed_start=21, games=2)
    return load_window(tmp_path, [old, new])


def test_half_of_every_batch_comes_from_the_newest_generation(tmp_path):
    """Final-review fix: round(batch x NEWEST_SHARE) rows from the newest generation's training
    games, first, and the rest from the older generations, whatever the step; each row is told
    apart by its encoded position (the two generations share none)."""
    w = two_generations(tmp_path)
    newest, older = every_row(w.newest), every_row(w.older)
    assert len(newest) > 0 and len(older) > 0 and not newest & older
    for batch in (64, 15, 13, 2):
        for step in range(3):
            got = rows(sample_batch(w, batch, 4, step, 2))
            k = round(batch * NEWEST_SHARE)
            assert batch_split(batch, True) == (k, batch - k)
            assert len(got) == batch
            assert all(r in newest for r in got[:k]), (batch, step)
            assert all(r in older for r in got[k:]), (batch, step)
    assert batch_split(1, True) == (1, 0), "a one-row batch is the newest generation's"
    assert rows(sample_batch(w, 1, 4, 0, 2))[0] in newest
    t = to_tensors(sample_batch(w, 64, 4, 0, 2), CPU)
    assert t.public.shape == (64, NET_PUBLIC_LEN) and t.mask.shape == (64, ACTIONS) and t.outcome.shape == (64, 4)


def test_without_older_generations_every_row_is_the_newest_and_the_draw_is_unchanged(tmp_path):
    w = load_window(tmp_path, [generation(tmp_path)])
    assert w.older.decisions == 0 and batch_split(64, False) == (64, 0)
    for step in range(3):
        # The single draw a window of one generation always made: seed (seed, step) packed by `_seed`.
        assert sample_batch(w, 64, 4, step, 2) == w.newest.sample(64, train_mod._seed(4, step), 2)


def test_the_same_seed_gives_byte_identical_batches(tmp_path):
    w = two_generations(tmp_path)
    a = sample_batch(w, 64, 4, 9, 2)
    assert a == sample_batch(two_generations(tmp_path / "again"), 64, 4, 9, 2), "a window read again"
    assert a == sample_batch(w, 64, 4, 9, 1), "the thread count does not matter"
    assert a != sample_batch(w, 64, 4, 10, 2) and a != sample_batch(w, 64, 5, 9, 2)
    k = round(64 * NEWEST_SHARE)
    assert rows(a)[:k] == rows(w.newest.sample(k, train_mod._seed(4, 9), 2))
    # The two draws use distinct seeds: the older draw does not reuse the newest draw's.
    assert rows(a)[k:] != rows(w.older.sample(64 - k, train_mod._seed(4, 9), 2))


def test_training_on_two_generations_draws_each_new_position_twice_and_records_the_split(tmp_path):
    w = two_generations(tmp_path)
    cfg = replace(TINY_TRAIN, steps=None, batch=512)
    s = train(w, tmp_path / "a.bin", cfg)
    assert s["steps"] == auto_steps(w.new_positions, 512, 2.0, NEWEST_SHARE) == math.ceil(4 * w.new_positions / 512)
    assert (s["newest_share"], s["older_positions"]) == (NEWEST_SHARE, w.older.decisions)
    train(w, tmp_path / "b.bin", replace(cfg, threads=1))
    assert (tmp_path / "a.bin").read_bytes() == (tmp_path / "b.bin").read_bytes(), "CPU training is reproducible"
    one = load_window(tmp_path / "one", [generation(tmp_path / "one")])
    s = train(one, tmp_path / "c.bin", replace(cfg, batch=256))
    assert s["steps"] == auto_steps(one.new_positions, 256, 2.0)
    assert (s["newest_share"], s["older_positions"]) == (1.0, 0)


def test_fast_searches_add_nothing_to_the_policy_loss(tmp_path):
    w = load_window(tmp_path, [generation(tmp_path)])
    t = to_tensors(w.newest.sample(64, 5, 2), CPU)
    assert 0 < t.full.sum().item() < 64
    torch.manual_seed(2)
    net = Net(16, 1)
    base = losses(net, t)["policy"]
    fast = t.full == 0
    noise = torch.rand(t.policy.shape) * t.mask
    other = torch.where(fast[:, None], noise / noise.sum(1, keepdim=True), t.policy)
    assert torch.equal(losses(net, replace(t, policy=other))["policy"], base)
    logits, _, _ = net(t.public, t.hidden)
    logp = torch.log_softmax(logits.masked_fill(~t.mask, -1e9), dim=1)
    assert torch.allclose(base, -(t.policy * logp).sum(1)[~fast].mean())


def test_a_tiny_dataset_can_be_overfit(tmp_path):
    w = load_window(tmp_path, [generation(tmp_path)])
    t = to_tensors(w.newest.sample(32, 1, 2), CPU)
    torch.manual_seed(3)
    net = Net(32, 2)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    # Annealed so the last step settles: at a flat 3e-3, Adam against the L2 pull swings the aux
    # loss between 1e-4 and 6e-3 late on, and the final value depends on where it stops.
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 600)
    for _ in range(600):
        parts = losses(net, t)
        opt.zero_grad()
        parts["total"].backward()
        opt.step()
        schedule.step()
    parts = losses(net, t)

    def entropy(p):
        return -(p * torch.log(p.clamp(min=1e-12))).sum(1)

    policy_floor = ((entropy(t.policy) * t.full).sum() / t.full.sum()).item()
    value_floor = entropy(t.outcome).mean().item()
    assert parts["policy"].item() - policy_floor < 0.05
    assert parts["value"].item() - value_floor < 0.05
    assert parts["aux"].item() < 1e-3


def test_without_mps_training_falls_back_to_the_cpu_with_a_notice(monkeypatch, capsys):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert pick_device("auto").type == "cpu"
    assert "MPS is not available" in capsys.readouterr().out
    assert pick_device("mps").type == "cpu"
    assert "MPS is not available" in capsys.readouterr().out
    assert pick_device("cpu").type == "cpu"
    assert capsys.readouterr().out == ""
    with pytest.raises(ValueError, match="unknown device"):
        pick_device("cuda")


def test_a_broken_mps_build_falls_back_to_the_cpu_with_a_notice(monkeypatch, capsys):
    real = torch.ones

    def ones(*args, device=None, **kwargs):
        if torch.device(device or "cpu").type == "mps":
            raise RuntimeError("MPS backend failed to initialise")
        return real(*args, device=device, **kwargs)

    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(torch, "ones", ones)
    assert pick_device("auto").type == "cpu"
    out = capsys.readouterr().out
    assert "MPS is not available" in out and "failed to initialise" in out


def test_temp_files_in_a_generation_are_not_record_files(tmp_path):
    """Preflight F3: a writer's hidden temp file (`files.temp_path`) is never read."""
    gen = generation(tmp_path)
    finished = gen / "selfplay-0.jsonl.gz"
    write_games(files.temp_path(gen / "selfplay-9.jsonl.gz"), 9, 1)
    (gen / ".hidden.jsonl.gz").write_bytes(finished.read_bytes())
    assert record_files(gen) == [finished]


def test_unreadable_empty_and_decisionless_files_are_skipped_with_a_notice(tmp_path):
    gen = generation(tmp_path)
    (gen / "empty.jsonl.gz").write_bytes(b"")
    other = tmp_path / "other"
    other.mkdir()
    write_games(other / "o.jsonl.gz", 100, 3)
    data = (other / "o.jsonl.gz").read_bytes()
    (gen / "truncated.jsonl.gz").write_bytes(data[: len(data) // 2])
    write_games(other / "z.jsonl.gz", 200, 1)
    (lone,) = records.read(other / "z.jsonl.gz")
    records.write(gen / "nodecisions.jsonl.gz", [{**lone, "decisions": []}], CONFIG)
    (first,) = [g for g in records.read(gen / "selfplay-0.jsonl.gz") if g["seed"] == 0]
    # A duplicate of seed 0 then a record that does not replay: the whole file goes, notices too.
    records.write(gen / "replayfails.jsonl.gz", [first, {**lone, "seed": 300, "actions": lone["actions"][::-1]}],
                  CONFIG)
    records.write(gen / "malformed.jsonl.gz", [{**lone, "seed": 301, "vp": [1, 2]}], CONFIG)
    with gzip.open(gen / "badjson.jsonl.gz", "wt") as f:
        f.write("{not json\n")
    (gen / ".half.jsonl.gz.tmp-123").write_bytes(b"garbage")
    w = load_window(tmp_path, [gen])
    text = "\n".join(w.notices)
    assert "empty.jsonl.gz: no games" in text
    assert "truncated.jsonl.gz: kept" in text and "unreadable" in text and "EOFError" in text
    assert "badjson.jsonl.gz: kept 0 games, skipped an unreadable rest (JSONDecodeError" in text
    assert "nodecisions.jsonl.gz: 1 games but no decisions" in text
    assert "replayfails.jsonl.gz: skipped the whole file" in text and "illegal" in text
    assert not any("replayfails" in n and "already in" in n for n in w.notices), "a skipped file leaves no duplicate notice"
    assert "malformed.jsonl.gz: skipped the whole file" in text and "vp needs 4 entries" in text
    assert ".half" not in text and not any(".tmp-" in k for k in w.files)
    assert w.validation.games == 2, "seeds 0 and 200 (which has no decisions) are held out"
    assert w.newest.games >= 1 and w.new_positions == w.newest.decisions == w.decisions > 0
    assert w.older.games == 0


@pytest.mark.parametrize("error", [AttributeError("a bug"), TypeError("a bug"), MemoryError(), PermissionError("no")])
def test_errors_that_are_not_bad_records_propagate_from_the_window(tmp_path, monkeypatch, error):
    """Only read errors and bad records skip a file; our own bugs (raised inside the stream that
    `TrainingSet.extend` consumes, which wraps them) and environment errors surface as themselves."""
    gen = generation(tmp_path)
    (good,) = [g for g in records.read(gen / "selfplay-0.jsonl.gz") if g["seed"] == 0]

    def broken_read(path):
        yield good
        raise error

    monkeypatch.setattr(records, "read", broken_read)
    with pytest.raises(type(error)) as caught:
        load_window(tmp_path, [gen])
    assert caught.value is error


def test_nothing_to_train_on_is_an_error_before_training(tmp_path):
    empty = tmp_path / "gen-001"
    empty.mkdir()
    (empty / "empty.jsonl.gz").write_bytes(b"")
    with pytest.raises(ValueError, match="no training positions"):
        load_window(tmp_path, [empty])
    old = generation(tmp_path, "gen-000")
    with pytest.raises(ValueError, match="no decisions to train on"):
        load_window(tmp_path, [old, empty])


def test_two_writers_in_one_generation_are_read_once_each(tmp_path):
    gen = generation(tmp_path)
    write_games(gen / "selfplay-elsewhere.jsonl.gz", 0, 2)
    write_games(gen / "selfplay-more.jsonl.gz", 1, 2)
    w = load_window(tmp_path, [gen])
    assert (w.newest.games, w.validation.games) == (2, 1)
    dup = [n for n in w.notices if "already in gen-000" in n]
    assert len(dup) == 2, "one notice per file, not per seed"
    assert any(n.endswith("selfplay-elsewhere.jsonl.gz: all 2 games duplicate earlier seeds (already in gen-000: 0, 1)")
               for n in dup)
    assert any(n.endswith("selfplay-more.jsonl.gz: skipped 1 games duplicating earlier seeds (already in gen-000: 1)")
               for n in dup)
    assert not any("no games" in n for n in w.notices)
    assert set(w.files) == {"gen-000/selfplay-0.jsonl.gz", "gen-000/selfplay-elsewhere.jsonl.gz",
                            "gen-000/selfplay-more.jsonl.gz"}


def test_a_duplicate_notice_lists_only_the_first_few_seeds(tmp_path):
    gen = generation(tmp_path, games=7)
    write_games(gen / "selfplay-again.jsonl.gz", 0, 7)
    (dup,) = [n for n in load_window(tmp_path, [gen]).notices if "already in" in n]
    assert dup.endswith("all 7 games duplicate earlier seeds (already in gen-000: 0, 1, 2, 3, 4, ... (2 more))")


def seed_decisions(gen_dir):
    return {g["seed"]: len(g["decisions"]) for p in record_files(gen_dir) for g in records.read(p)}


def test_the_newest_generation_holds_out_its_seeds_divisible_by_20(tmp_path):
    old = generation(tmp_path, "gen-000", seed_start=0, games=2)  # seed 0, but an older generation: trained on
    new = generation(tmp_path, "gen-001", seed_start=39, games=3)  # seed 40 is held out
    old_d, new_d = seed_decisions(old), seed_decisions(new)
    w = load_window(tmp_path, [old, new])
    assert (w.games, w.newest.games, w.older.games, w.validation.games) == (4, 2, 2, 1)
    assert w.validation.decisions == new_d[40] > 0
    assert w.older.decisions == sum(old_d.values())
    assert w.new_positions == w.newest.decisions == new_d[39] + new_d[41]
    assert w.decisions == w.older.decisions + w.newest.decisions
    assert w.full_decisions == w.older.full_decisions + w.newest.full_decisions
    again = load_window(tmp_path, [old, new])
    assert again.validation.sample(64, 1, 2) == w.validation.sample(64, 1, 2)
    assert again.newest.sample(64, 1, 2) == w.newest.sample(64, 1, 2)
    assert again.older.sample(64, 1, 2) == w.older.sample(64, 1, 2)
    flipped = load_window(tmp_path, [new, old])  # gen-000 the newest: seed 0 held out, seed 40 trained on
    assert flipped.validation.decisions == old_d[0]
    assert flipped.older.decisions == sum(new_d.values())
    assert flipped.new_positions == flipped.newest.decisions == old_d[1]


def test_a_bad_held_out_record_skips_its_whole_file(tmp_path):
    """Like a bad training record: nothing of the file enters either set, and a notice says why."""
    gen = generation(tmp_path)  # seed 0 held out, seed 1 trained on
    other = tmp_path / "other"
    other.mkdir()
    write_games(other / "x.jsonl.gz", 20, 2)
    held, trained = sorted(records.read(other / "x.jsonl.gz"), key=lambda g: g["seed"])
    records.write(gen / "bad.jsonl.gz", [{**held, "vp": [1, 2]}, trained], CONFIG)
    w = load_window(tmp_path, [gen])
    (notice,) = [n for n in w.notices if "bad.jsonl.gz" in n]
    assert "skipped the whole file" in notice and "held-out" in notice and "vp needs 4 entries" in notice
    d = seed_decisions(gen)
    assert (w.newest.games, w.validation.games) == (1, 1)
    assert (w.newest.decisions, w.validation.decisions) == (d[1], d[0])


TINY_TRAIN = TrainConfig(lr=1e-3, steps=7, batch=16, device="cpu", width=8, rounds=1, threads=2,
                         eval_every=2, eval_batches=2, eval_batch=32)


def scripted_scores(monkeypatch, scores):
    """Held-out scores forced to `scores` in turn; the other losses stay the real ones."""
    it = iter(scores)
    monkeypatch.setattr(train_mod, "held_out_losses",
                        lambda net, batches: {**held_out_losses(net, batches), "score": next(it)})


def test_the_exported_net_is_the_best_held_out_checkpoint(tmp_path, monkeypatch):
    """Evaluations after steps 0, 2, 4 and 6, the best forced at step 2: the file written is the
    weights after step 2, byte for byte with a 3-step run (CPU training is reproducible)."""
    w = load_window(tmp_path, [generation(tmp_path)])
    scripted_scores(monkeypatch, [3.0, 1.0, 2.0, 4.0])
    s = train(w, tmp_path / "best.bin", TINY_TRAIN)
    assert [r["step"] for r in s["held_out"]] == [0, 2, 4, 6]
    assert s["best_step"] == 2 and s["best_held_out"] == s["held_out"][1]
    scripted_scores(monkeypatch, [3.0, 1.0])
    short = train(w, tmp_path / "short.bin", replace(TINY_TRAIN, steps=3))
    assert short["best_step"] == 2
    assert (tmp_path / "best.bin").read_bytes() == (tmp_path / "short.bin").read_bytes()
    scripted_scores(monkeypatch, [4.0, 3.0, 2.0, 1.0])
    last = train(w, tmp_path / "last.bin", TINY_TRAIN)
    assert last["best_step"] == 6
    assert (tmp_path / "last.bin").read_bytes() != (tmp_path / "best.bin").read_bytes()


def test_held_out_losses_are_recorded_and_the_best_is_the_lowest_policy_plus_value(tmp_path):
    w = load_window(tmp_path, [generation(tmp_path)])
    s = train(w, tmp_path / "n.bin", replace(TINY_TRAIN, steps=9, lr=3e-2))
    rows = s["held_out"]
    assert [r["step"] for r in rows] == [0, 2, 4, 6, 8]
    for r in rows:
        assert set(r) == {"step", "policy", "value", "aux", "score"}
        assert abs(r["score"] - (r["policy"] + r["value"])) <= 2e-6
    assert s["best_held_out"] == min(rows, key=lambda r: r["score"]) and s["best_step"] == s["best_held_out"]["step"]
    # The held-out set: fixed batches (not the run's seed), every row's value and aux.
    batches = held_out_batches(w.validation, TINY_TRAIN, CPU, 2)
    assert len(batches) == 2 and all(t.public.shape[0] == 32 for t in batches)
    other = held_out_batches(w.validation, replace(TINY_TRAIN, seed=5), CPU, 1)
    assert all(torch.equal(a.public, b.public) and torch.equal(a.outcome, b.outcome) for a, b in zip(batches, other))
    exported = model.load(tmp_path / "n.bin")
    got = held_out_losses(exported, batches)
    with torch.no_grad():
        every = [losses(exported, t) for t in batches]
    for key in ("policy", "value", "aux"):
        assert abs(got[key] - sum(p[key].item() for p in every) / 2) <= 1e-6, key
        assert abs(got[key] - s["best_held_out"][key]) <= 1e-5, key


def test_without_held_out_games_early_stopping_is_off_with_a_notice(tmp_path, capsys):
    w = load_window(tmp_path, [generation(tmp_path, seed_start=1)])
    assert w.validation.games == 0 and w.newest.games == 2
    s = train(w, tmp_path / "n.bin", TINY_TRAIN)
    assert "notice: no held-out positions" in capsys.readouterr().out
    assert s["held_out"] == [] and s["best_step"] is None and s["best_held_out"] is None
    assert (tmp_path / "n.bin").exists()


def test_train_exports_a_net_the_rust_side_loads_and_checks_parity(tmp_path):
    w = load_window(tmp_path, [generation(tmp_path)])
    nets = tmp_path / "nets"
    nets.mkdir()
    cfg = TrainConfig(lr=1e-3, steps=5, batch=16, device="cpu", width=8, rounds=1, threads=2)
    s = train(w, nets / "t.bin", cfg)
    assert s["steps"] == 5 and s["device"] == "cpu" and s["parity_max_error"] <= 1e-5
    assert (s["width"], s["rounds"]) == (8, 1)
    assert not [p for p in nets.iterdir() if p.name.startswith(".")], "no temp file left"
    assert Bot("net:t@5", 0, nets_dir=str(nets)).name == "net"
    train(w, nets / "u.bin", replace(cfg, threads=1))
    assert (nets / "u.bin").read_bytes() == (nets / "t.bin").read_bytes(), "CPU training is reproducible"
    s = train(w, nets / "v.bin", replace(cfg, lr=3e-4, steps=2, width=4, rounds=2), init=nets / "t.bin")
    assert s["init"] == str(nets / "t.bin") and (s["width"], s["rounds"]) == (8, 1), "the init file's shape"
    assert from_bytes((nets / "v.bin").read_bytes()).width == 8
    with pytest.raises(RuntimeError, match="disagree"):
        parity_check(Net(8, 1), nets / "t.bin", w.newest)
    nan_net = model.load(nets / "t.bin")
    with torch.no_grad():
        nan_net.value.weight[0, 0] = float("nan")
    with pytest.raises(RuntimeError, match="disagree"):
        parity_check(nan_net, nets / "t.bin", w.newest)
    for bad in (replace(cfg, batch=2**16 + 1), replace(cfg, lr=float("nan")), replace(cfg, lr=0.0),
                replace(cfg, eval_every=0), replace(cfg, eval_batches=0), replace(cfg, eval_batch=0),
                replace(cfg, eval_batch=2**16 + 1), replace(cfg, width=0), replace(cfg, width=NET_MAX_WIDTH + 1),
                replace(cfg, rounds=NET_MAX_ROUNDS + 1)):
        with pytest.raises(ValueError):
            train(w, nets / "x.bin", bad)
    assert not (nets / "x.bin").exists()


def test_cli_train(tmp_path):
    gen = generation(tmp_path)
    out = tmp_path / "nets" / "c.bin"
    r = cli("train", str(gen), "--out", str(out), "--steps", "3", "--batch", "16", "--device", "cpu",
            "--width", "8", "--rounds", "1")
    assert r.returncode == 0, r.stderr
    assert f"wrote {out}" in r.stdout and "parity" in r.stdout
    assert "1 held out" in r.stdout and "kept the best held-out score" in r.stdout
    assert r.stderr == ""
    empty = tmp_path / "gen-001"
    empty.mkdir()
    r = cli("train", str(empty), "--out", str(out))
    assert r.returncode == 2 and "no training positions" in r.stderr
    r = cli("train", str(gen), "--out", str(out), "--trunk", "16")
    assert r.returncode == 2 and "unrecognized arguments: --trunk 16" in r.stderr, "network v2 has no trunk"
    r = cli("train", str(gen), "--out", str(out), "--rounds", str(NET_MAX_ROUNDS + 1))
    assert r.returncode == 2 and "rounds 17 is out of range" in r.stderr


SMALL = ["--steps", "1", "--batch", "16", "--device", "cpu", "--width", "8", "--rounds", "1"]


def test_cli_lr_defaults_to_1e3_from_scratch_and_3e4_from_an_init(tmp_path, capsys):
    """Spec Section 4: net0 (from scratch) at 1e-3, later nets (from a parent) at 3e-4; an
    explicit --lr always wins."""
    from alphasettler.cli import main

    gen = generation(tmp_path)
    nets = tmp_path / "nets"
    cases = [([], "lr 0.001;"), (["--init", str(nets / "a.bin")], "lr 0.0003;"),
             (["--lr", "0.002"], "lr 0.002;"), (["--init", str(nets / "a.bin"), "--lr", "0.002"], "lr 0.002;")]
    for i, (extra, expected) in enumerate(cases):
        assert main(["train", str(gen), "--out", str(nets / ("a.bin" if i == 0 else f"{i}.bin")), *SMALL, *extra]) == 0
        assert expected in capsys.readouterr().out, (extra, expected)


def test_cli_names_a_missing_generation_and_checks_init_before_reading_records(tmp_path):
    out = str(tmp_path / "nets" / "c.bin")
    missing = tmp_path / "gen-404"
    r = cli("train", str(missing), "--out", out)
    assert r.returncode == 2 and "gen-404 does not exist" in r.stderr
    bad = tmp_path / "bad.bin"
    bad.write_bytes(b"not a weights file")  # shorter than the 32-byte fixed header
    # The missing directory would fail too: the --init error coming first shows the order.
    r = cli("train", str(missing), "--out", out, "--init", str(bad))
    assert r.returncode == 2 and "bad.bin: truncated" in r.stderr and "gen-404" not in r.stderr
    r = cli("train", str(missing), "--out", out, "--init", str(tmp_path / "absent.bin"))
    assert r.returncode == 2 and "absent.bin" in r.stderr and "gen-404" not in r.stderr
    mlp = bytearray(to_bytes(Net(4, 1)))
    struct.pack_into("<I", mlp, 8, 1)  # a version 1 (flat MLP) file
    (tmp_path / "mlp.bin").write_bytes(bytes(mlp))
    r = cli("train", str(missing), "--out", out, "--init", str(tmp_path / "mlp.bin"))
    assert r.returncode == 2 and "format version 1, expected 2" in r.stderr and "retrain" in r.stderr
    assert "gen-404" not in r.stderr


def test_cli_train_without_pytorch_says_how_to_install_it(tmp_path):
    code = ("import sys; sys.modules['torch'] = None\n"  # what `import torch` does when it is not installed
            "from alphasettler.cli import main\n"
            f"sys.exit(main(['train', {str(tmp_path)!r}, '--out', {str(tmp_path / 'x.bin')!r}]))")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 2, r.stderr
    assert "PyTorch (the train extra) is not installed" in r.stderr and "pip3 install -e '.[train]'" in r.stderr
    assert "Traceback" not in r.stderr


def test_the_base_package_does_not_import_torch():
    code = ("import sys, alphasettler, alphasettler.cli, alphasettler.records, alphasettler.selfplay, "
            "alphasettler.files, alphasettler.arena; print('torch' in sys.modules)")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0 and r.stdout.strip() == "False", r.stderr
    outside = [p for p in ROOT.glob("alphasettler/**/*.py") if "train" not in p.relative_to(ROOT).parts]
    assert not [p for p in outside if "import torch" in p.read_text()]
