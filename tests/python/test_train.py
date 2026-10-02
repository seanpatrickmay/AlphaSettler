"""Training (spec Section 4): the PyTorch twin, the data window, the training run and `train`.

These tests need the `train` extra (PyTorch: `.venv/bin/pip3 install -e '.[train]'`). Without it
they fail at collection on `import torch`, on purpose: they must be run, never skipped."""

import gzip
import random
import struct
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from alphasettler import Bot, Game, files, records
from alphasettler._engine import NET_HIDDEN_LEN, NET_MAX_WIDTH, NET_PUBLIC_LEN, encode, net_forward
from alphasettler.selfplay import write_explore
from alphasettler.train import model
from alphasettler.train.data import load_window, record_files, to_tensors
from alphasettler.train.model import ACTIONS, HEADER_LEN, Net, from_bytes, to_bytes
from alphasettler.train.train import (
    TrainConfig, auto_steps, losses, parity_check, parity_error, pick_device, train,
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


def test_rust_and_pytorch_forward_passes_agree_on_random_weights(tmp_path):
    """Spec parity rule (preflight F9): the prior over legal moves and the value probabilities
    agree within 1e-5 absolute; raw logits and the aux head within 1e-5 x max(1, |PyTorch|)."""
    torch.manual_seed(0)
    net = Net().eval()
    path = tmp_path / "r.bin"
    path.write_bytes(to_bytes(net))
    public, hidden, mask = random_positions(100)
    rust = net_forward(str(path), public, hidden)
    pt = torch.frombuffer(bytearray(public), dtype=torch.float32).view(100, NET_PUBLIC_LEN)
    ht = torch.frombuffer(bytearray(hidden), dtype=torch.float32).view(100, NET_HIDDEN_LEN)
    with torch.no_grad():
        outs = net(pt, ht)
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


def test_weights_round_trip_and_bad_files_are_errors():
    torch.manual_seed(1)
    net = Net(16, 8)
    data = to_bytes(net)
    back = from_bytes(data)
    assert (back.trunk_width, back.value_width) == (16, 8)
    assert all(torch.equal(a, b) for a, b in zip(net.state_dict().values(), back.state_dict().values()))
    shape = bytearray(data)
    shape[32:36] = (15).to_bytes(4, "little")
    for bad, message in [(b"X" + data[1:], "magic"), (data[:12] + bytes([data[12] ^ 1]) + data[13:], "encoding hash"),
                         (bytes(shape), "shapes"), (data[:-1], "truncated"), (data[:40], "truncated"),
                         (data + b"\0", "trailing")]:
        with pytest.raises(ValueError, match=message):
            from_bytes(bad)


def header(trunk, value):
    """A weights-file header whose shapes are consistent for these widths (as Rust's Arch::shapes)."""
    shapes = [(trunk, NET_PUBLIC_LEN), (trunk, trunk), (trunk, trunk), (ACTIONS, trunk),
              (value, trunk + NET_HIDDEN_LEN), (4, value), (4, value)]
    data = bytearray(to_bytes(Net(1, 1))[:24])
    for rows, cols in shapes:
        data += struct.pack("<II", rows, cols)
    assert len(data) == HEADER_LEN
    return bytes(data)


def test_widths_are_checked_and_non_finite_weights_rejected_before_building_a_net(monkeypatch):
    """Preflight F4: same rules as the Rust loader, and no torch module is built from an
    unvalidated header."""
    torch.manual_seed(4)
    good = to_bytes(Net(3, 2))

    def no_net(*args, **kwargs):
        raise AssertionError("a Net was built before the file was validated")

    monkeypatch.setattr(model, "Net", no_net)
    for trunk, value in [(NET_MAX_WIDTH + 1, 8), (8, NET_MAX_WIDTH + 1), (0, 8), (8, 0), (2**31, 8)]:
        with pytest.raises(ValueError, match=f"widths must be 1..={NET_MAX_WIDTH}"):
            from_bytes(header(trunk, value))
    for x in (float("nan"), float("inf"), float("-inf")):
        bad = bytearray(good)
        struct.pack_into("<f", bad, HEADER_LEN + 4, x)
        with pytest.raises(ValueError, match="NaN or infinite"):
            from_bytes(bytes(bad))
    with pytest.raises(ValueError, match=f"1..={NET_MAX_WIDTH}"):
        Net(NET_MAX_WIDTH + 1, 8)  # the class itself, not the patched module name


def test_the_loss_is_the_spec_sum_with_l2_over_weights_only(tmp_path):
    w = load_window(tmp_path, [generation(tmp_path)])
    t = to_tensors(w.data.sample(32, 7, 2), CPU)
    torch.manual_seed(6)
    net = Net(16, 8)
    with torch.no_grad():
        for layer in net.layers():
            layer.bias.fill_(0.5)  # biases large enough that counting them would show
        parts = losses(net, t)
    l2 = sum(layer.weight.pow(2).sum() for layer in net.layers())
    assert torch.allclose(parts["l2"], l2)
    assert not torch.allclose(parts["l2"], l2 + sum(layer.bias.pow(2).sum() for layer in net.layers()))
    total = parts["policy"] + parts["value"] + 0.25 * parts["aux"] + 1e-4 * l2
    assert torch.allclose(parts["total"], total)


def test_steps_are_twice_the_new_positions_over_the_batch():
    assert auto_steps(1024, 1024) == 2
    assert auto_steps(1, 1024) == 1
    assert auto_steps(5000, 1024) == 10
    assert auto_steps(5000, 1024, 3.0) == 15


def test_fast_searches_add_nothing_to_the_policy_loss(tmp_path):
    w = load_window(tmp_path, [generation(tmp_path)])
    t = to_tensors(w.data.sample(64, 5, 2), CPU)
    assert 0 < t.full.sum().item() < 64
    torch.manual_seed(2)
    net = Net(32, 16)
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
    t = to_tensors(w.data.sample(32, 1, 2), CPU)
    torch.manual_seed(3)
    net = Net(64, 32)
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
    assert w.data.games >= 3 and w.new_positions == w.data.decisions > 0


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
    assert w.data.games == 3
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


def test_train_exports_a_net_the_rust_side_loads_and_checks_parity(tmp_path):
    w = load_window(tmp_path, [generation(tmp_path)])
    nets = tmp_path / "nets"
    nets.mkdir()
    cfg = TrainConfig(lr=1e-3, steps=5, batch=16, device="cpu", trunk=16, value=8, threads=2)
    s = train(w, nets / "t.bin", cfg)
    assert s["steps"] == 5 and s["device"] == "cpu" and s["parity_max_error"] <= 1e-5
    assert not [p for p in nets.iterdir() if p.name.startswith(".")], "no temp file left"
    assert Bot("net:t@5", 0, nets_dir=str(nets)).name == "net"
    train(w, nets / "u.bin", replace(cfg, threads=1))
    assert (nets / "u.bin").read_bytes() == (nets / "t.bin").read_bytes(), "CPU training is reproducible"
    s = train(w, nets / "v.bin", replace(cfg, lr=3e-4, steps=2), init=nets / "t.bin")
    assert s["init"] == str(nets / "t.bin") and from_bytes((nets / "v.bin").read_bytes()).trunk_width == 16
    with pytest.raises(RuntimeError, match="disagree"):
        parity_check(Net(16, 8), nets / "t.bin", w)
    nan_net = model.load(nets / "t.bin")
    with torch.no_grad():
        nan_net.value1.weight[0, 0] = float("nan")
    with pytest.raises(RuntimeError, match="disagree"):
        parity_check(nan_net, nets / "t.bin", w)
    for bad in (replace(cfg, batch=2**16 + 1), replace(cfg, lr=float("nan")), replace(cfg, lr=0.0)):
        with pytest.raises(ValueError):
            train(w, nets / "x.bin", bad)
    assert not (nets / "x.bin").exists()


def test_cli_train(tmp_path):
    gen = generation(tmp_path)
    out = tmp_path / "nets" / "c.bin"
    r = cli("train", str(gen), "--out", str(out), "--steps", "3", "--batch", "16", "--device", "cpu",
            "--trunk", "16", "--value-width", "8")
    assert r.returncode == 0, r.stderr
    assert f"wrote {out}" in r.stdout and "parity" in r.stdout
    assert r.stderr == ""
    empty = tmp_path / "gen-001"
    empty.mkdir()
    r = cli("train", str(empty), "--out", str(out))
    assert r.returncode == 2 and "no training positions" in r.stderr


SMALL = ["--steps", "1", "--batch", "16", "--device", "cpu", "--trunk", "16", "--value-width", "8"]


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
    bad.write_bytes(b"not a weights file at all, far too short")
    # The missing directory would fail too: the --init error coming first shows the order.
    r = cli("train", str(missing), "--out", out, "--init", str(bad))
    assert r.returncode == 2 and "bad.bin: truncated" in r.stderr and "gen-404" not in r.stderr
    r = cli("train", str(missing), "--out", out, "--init", str(tmp_path / "absent.bin"))
    assert r.returncode == 2 and "absent.bin" in r.stderr and "gen-404" not in r.stderr


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
