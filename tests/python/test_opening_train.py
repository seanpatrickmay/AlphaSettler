"""The opening model's training (spec 3d-1 Section 4.4): the setup replay, the samples, the PyTorch
model and its parity with the Rust forward pass, early stopping, and `opening-train`.

These tests need the `train` extra (PyTorch). Without it they fail at collection on purpose: they
must be run, never skipped."""

import copy
import json
import random
import re

import pytest
import torch

from alphasettler import Bot, Game
from alphasettler._engine import OPENING_FEATURES, OPENING_FILE, opening_features, opening_forward, write_random_net
from alphasettler.arena import TRADES
from alphasettler.opening import SETUP_ACTIONS, model_bytes, replay_setup, write_model
from alphasettler.selfplay import write_explore
from alphasettler.train import opening as topen
from alphasettler.train.opening import OpeningConfig, OpeningNet, Samples, fit, load_samples
from opening_games import FIXED_ACTIONS, FIXED_NODES, fixed_game, truncate, write_dir
from test_train import cli, write_games

SEATS_IN_PICK_ORDER = [0, 1, 2, 3, 3, 2, 1, 0]


def test_replay_setup_keeps_each_pick_and_the_position_after_the_setup():
    s = replay_setup(fixed_game(), features=True, keep_positions=True)
    assert [p.seat for p in s.picks] == SEATS_IN_PICK_ORDER
    assert [p.node for p in s.picks] == FIXED_NODES
    assert s.picks[0].legal == list(range(54))
    assert all(p.node in p.legal for p in s.picks)
    assert (s.after.phase, s.winner, s.trading) == ("pre_roll", 3, True)
    assert s.nodes(3) == [41, 31]
    # The features are the seat's right after its settlement, before its road.
    before = s.picks[4].before
    assert before.phase == "setup_settlement"
    g = before.copy()
    g.apply(25 + 31)
    assert s.picks[4].features == opening_features(g, 3)


@pytest.mark.parametrize("actions,message", [
    (FIXED_ACTIONS[:15], "game 0: 15 actions, fewer than the 16 of the setup"),
    ([FIXED_ACTIONS[1]] + FIXED_ACTIONS[1:], "game 0: action 0 (141) is not a setup settlement"),
    (FIXED_ACTIONS[:2] + [25 + 9] + FIXED_ACTIONS[3:], "game 0: action 2 (34): "),  # node 9 is taken
])
def test_replay_setup_refuses_records_that_do_not_replay_a_setup(actions, message):
    with pytest.raises(ValueError, match=re.escape(message)):
        replay_setup({**fixed_game(), "actions": actions})


def test_samples_are_eight_per_decided_game_with_the_winner_relative_to_the_seat(tmp_path):
    # Seed 0 is held out (0 % 20 == 0); seed 1 trains; seed 2 has no winner and adds nothing.
    d = write_dir(tmp_path / "gen", [fixed_game(0, winner=3), fixed_game(1, winner=1), fixed_game(2, winner=None)])
    train, held, counts = load_samples([d])
    assert counts == {"games": 3, "no_winner": 1, "held_out_games": 1}
    assert (len(train), len(held)) == (8, 8)
    assert held.y.tolist() == [(3 - s) % 4 for s in SEATS_IN_PICK_ORDER]
    assert train.y.tolist() == [(1 - s) % 4 for s in SEATS_IN_PICK_ORDER]
    assert held.trading.all() and held.compare is None
    assert train.x.shape == (8, OPENING_FEATURES) and train.x.dtype == torch.float32
    assert held.x[4].tolist() == replay_setup(fixed_game(), features=True).picks[4].features
    # With a compare net, each held-out sample keeps -ln(its probability of the winner): seat 3's 0.4.
    _, held, _ = load_samples([d], compare=lambda g, seat: [0.1, 0.2, 0.3, 0.4])
    assert held.compare.tolist() == pytest.approx([-torch.log(torch.tensor(0.4, dtype=torch.float64)).item()] * 8)


def test_directories_without_records_training_or_held_out_games_are_errors(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError, match="no record files"):
        load_samples([tmp_path / "empty"])
    only_held = write_dir(tmp_path / "held", [fixed_game(0, winner=0)])
    with pytest.raises(ValueError, match="no training games"):
        load_samples([only_held])
    only_train = write_dir(tmp_path / "train", [fixed_game(1, winner=0)])
    with pytest.raises(ValueError, match="no held-out games"):
        load_samples([only_train])
    bad = write_dir(tmp_path / "bad", [{**fixed_game(1), "actions": FIXED_ACTIONS[:3]}])
    with pytest.raises(ValueError, match=r"selfplay-fixed\.jsonl\.gz: game 1: 3 actions"):
        load_samples([bad])


def test_a_truncated_or_incomplete_record_file_is_an_error_naming_the_file(tmp_path):
    cut = write_dir(tmp_path / "cut", [fixed_game(s) for s in range(40)])
    path = truncate(cut)
    with pytest.raises(ValueError, match=rf"{re.escape(str(path))}: cannot read the records"):
        load_samples([cut])
    broken = {k: v for k, v in fixed_game(1).items() if k != "winner"}
    with pytest.raises(ValueError, match=r"selfplay-fixed\.jsonl\.gz: a record has no winner field"):
        load_samples([write_dir(tmp_path / "broken", [broken])])


def random_setup_features(seeds):
    """The features after every settlement of uniformly random setups."""
    rows = []
    for seed in seeds:
        rng = random.Random(seed)
        g = Game(seed, None)
        for _ in range(SETUP_ACTIONS):
            seat, settlement = g.current_actor, g.phase == "setup_settlement"
            g.apply(rng.choice(g.legal_actions()))
            if settlement:
                rows.append(opening_features(g, seat))
    return rows


@pytest.mark.parametrize("hidden", [(64, 64), (5,)])
def test_rust_and_pytorch_opening_models_agree(tmp_path, hidden):
    torch.manual_seed(1)
    net = OpeningNet(hidden)
    path = tmp_path / OPENING_FILE
    write_model(path, net.layers())
    rows = random_setup_features(range(6))
    with torch.no_grad():
        want = torch.softmax(net(torch.tensor(rows)), dim=1)
    got = torch.tensor(opening_forward(str(path), rows))
    assert (got - want).abs().max().item() < 1e-5


def test_write_model_leaves_nothing_at_the_path_when_the_rust_loader_refuses(tmp_path):
    path = tmp_path / OPENING_FILE
    with pytest.raises(ValueError, match="layer 0 takes 77 inputs"):
        write_model(path, [(4, 77, [0.0] * (4 * 77), [0.0] * 4)])
    assert list(tmp_path.iterdir()) == []
    with pytest.raises(ValueError, match="needs 312 weights and 4 biases"):
        model_bytes([(4, 78, [0.0], [0.0] * 4)])


def tiny_samples(n, seed):
    g = torch.Generator().manual_seed(seed)
    return Samples(torch.rand(n, OPENING_FEATURES, generator=g), torch.randint(0, 4, (n,), generator=g),
                   torch.zeros(n, dtype=torch.bool), None)


def test_training_stops_three_epochs_after_the_best_and_keeps_the_best(monkeypatch):
    scores = iter([1.0, 0.9, 0.95, 0.91, 0.92, 0.1])
    seen = {}

    def scripted(net, samples):
        score = next(scores)
        seen[score] = copy.deepcopy(net.state_dict())
        return score

    monkeypatch.setattr(topen, "held_out_loss", scripted)
    net, s = fit(tiny_samples(64, 0), tiny_samples(16, 1), OpeningConfig(batch=16), log=lambda _: None)
    assert (s["best_epoch"], s["epochs"], s["best_held_out"]) == (2, 5, 0.9)
    assert all(torch.equal(v, seen[0.9][k]) for k, v in net.state_dict().items())


@pytest.mark.parametrize("field,value,message", [
    ("lr", 0.0, "learning rate"), ("batch", 0, "batch"), ("patience", 0, "patience"),
    ("max_epochs", 0, "max_epochs"), ("hidden", (64, 0), "hidden"),
])
def test_bad_training_settings_are_refused(field, value, message):
    with pytest.raises(ValueError, match=message):
        OpeningConfig(**{field: value}).check()


def test_cli_opening_train_writes_a_model_the_bots_load(tmp_path):
    nets = tmp_path / "nets"
    nets.mkdir()
    write_random_net(str(nets / "tiny.bin"), 1, 8, 1)
    a, b = tmp_path / "gen-a", tmp_path / "gen-b"
    write_games(a / "selfplay-1.jsonl.gz", 1, 3)  # no-trade self-play, seeds 1-3
    write_dir(b, [fixed_game(0, winner=3)])  # the only held-out game, a trading one
    write_explore(b / "selfplay-21.jsonl.gz", 21, 2, config=TRADES, full_simulations=6, fast_simulations=3,
                  full_probability=0.5, threads=2)
    out = nets / OPENING_FILE
    args = ["opening-train", "--games", str(a), str(b), "--out", str(out), "--compare-net", "tiny",
            "--nets-dir", str(nets)]
    r = cli(*args)
    assert r.returncode == 0, r.stderr
    assert "uniform 1.3863" in r.stdout and "net:tiny" in r.stdout
    assert "held-out cross-entropy, trading games (8 samples)" in r.stdout
    assert len(opening_forward(str(out), [[0.0] * OPENING_FEATURES])[0]) == 4
    Bot("net:tiny@5+o", 0, nets_dir=str(nets))  # the bots load it
    summary = json.loads((nets / "opening.bin.json").read_text())
    assert summary["held_out"]["all"]["samples"] == 8 and summary["best_epoch"] >= 1
    again = cli(*args)
    assert again.returncode == 2 and "already exists" in again.stderr


def test_cli_opening_train_checks_the_compare_net_before_reading_games(tmp_path):
    d = write_dir(tmp_path / "gen", [fixed_game(0), fixed_game(1)])
    out = tmp_path / "opening.bin"
    r = cli("opening-train", "--games", str(d), "--out", str(out), "--compare-net", "missing",
            "--nets-dir", str(tmp_path))
    assert r.returncode == 2 and "cannot read" in r.stderr
    assert not out.exists()


def test_cli_opening_train_checks_the_output_directory_before_reading_games(tmp_path):
    d = write_dir(tmp_path / "gen", [{**fixed_game(0), "actions": []}])  # reading it would fail otherwise
    out = tmp_path / "no-such-dir" / "opening.bin"
    r = cli("opening-train", "--games", str(d), "--out", str(out), "--nets-dir", str(tmp_path))
    assert r.returncode == 2 and f"{out.parent} is not a directory" in r.stderr, r.stderr
    assert not out.parent.exists()
