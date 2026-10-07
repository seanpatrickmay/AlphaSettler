"""The fast loop (alphasettler/train/fastloop.py): set-up, generations, the window, resuming and
the regression gate. Needs the `train` extra, like test_train.py."""

import json

import pytest

from alphasettler.train.fastloop import GATE_MIN_Z, STATE, FastLoop, FastLoopSettings, count_games, promoted
from alphasettler.train.loop import SEEDS_PER_GENERATION
from alphasettler.train.model import Net, save

from test_train import cli, generation

TINY = dict(games=2, chunk_games=2, full_simulations=6, fast_simulations=3, full_probability=0.5, window_games=3,
            minutes=0.01, warmup=1, value_batch=4, gate_seeds=1, gate_simulations=6, device="cpu", threads=2)


def setup(tmp_path):
    seed = generation(tmp_path, "seed-gen", seed_start=0)
    start = tmp_path / "start.bin"
    save(Net(8, 1), start)
    root = tmp_path / "loop"
    FastLoop.init(root, [seed], start, first=1)
    return root, seed


def test_the_gate_promotes_unless_the_candidate_is_clearly_worse():
    assert promoted(0.0) and promoted(GATE_MIN_Z + 0.01)
    assert not promoted(GATE_MIN_Z) and not promoted(-3.0)


def test_init_records_the_seed_directories_and_the_start_net(tmp_path):
    root, seed = setup(tmp_path)
    state = json.loads((root / STATE).read_text())
    assert state["seed_dirs"] == [{"dir": str(seed.resolve()), "games": 2}]
    assert (state["first"], state["best"], state["latest"], state["generations"]) == (1, "start", "start", [])
    assert (root / "nets" / "start.bin").read_bytes() == (root / "nets" / "best.bin").read_bytes()
    with pytest.raises(ValueError, match="already set up"):
        FastLoop.init(root, [seed], tmp_path / "start.bin", first=1)


def test_generations_self_play_train_gate_and_resume(tmp_path):
    root, seed = setup(tmp_path)
    loop = FastLoop(root, FastLoopSettings(generations=2, **TINY))
    assert loop.run() == 2
    state = json.loads((root / STATE).read_text())
    gens = state["generations"]
    assert [g["gen"] for g in gens] == [1, 2] and state["latest"] == "gen-002"
    assert gens[0]["selfplay"]["seed_start"] == SEEDS_PER_GENERATION and gens[0]["selfplay"]["games"] == 2
    assert gens[0]["selfplay"]["net"] == "start"
    assert gens[0]["train"]["init"].endswith("start.bin") and gens[1]["train"]["init"].endswith("gen-001.bin")
    assert count_games(root / "gen-002") == 2
    for g in gens:  # one gate seed: zero spread, so z is +-inf, stored as null
        z = g["gate"]["z"]
        if z is None:
            z = float("inf") if g["gate"]["win_rate"] > 0.25 else float("-inf")
        assert g["gate"]["promoted"] == promoted(z)
    best = state["best"]
    assert (root / "nets" / "best.bin").read_bytes() == (root / "nets" / f"{best}.bin").read_bytes()
    assert FastLoop(root, FastLoopSettings(generations=2, **TINY)).run() == 0, "finished generations do not rerun"
    assert FastLoop(root, FastLoopSettings(generations=3, **TINY)).run() == 1


def test_the_window_is_the_newest_directories_holding_enough_games(tmp_path):
    root, seed = setup(tmp_path)
    loop = FastLoop(root, FastLoopSettings(generations=1, **TINY))
    new = root / "gen-001"
    assert loop.window(new, 2) == [seed.resolve(), new], "2 new games are below 3, so the seed directory joins"
    assert loop.window(new, 3) == [new]
    loop.run()
    loop2 = FastLoop(root, FastLoopSettings(generations=2, **TINY))
    assert loop2.window(root / "gen-002", 2) == [root / "gen-001", root / "gen-002"]


def test_cli_fast_loop_needs_its_set_up_arguments_the_first_time(tmp_path):
    r = cli("fast-loop", "--dir", str(tmp_path / "l"), "--generations", "1")
    assert r.returncode == 2 and "needs --seed-dirs" in r.stderr


def test_a_relative_run_directory_trains_on_absolute_seed_directories(tmp_path, monkeypatch):
    seed = generation(tmp_path, "seed-gen", seed_start=0)
    save(Net(8, 1), tmp_path / "start.bin")
    monkeypatch.chdir(tmp_path)
    FastLoop.init("loop", ["seed-gen"], "start.bin", first=1)
    assert FastLoop("loop", FastLoopSettings(generations=1, **{**TINY, "window_games": 10})).run() == 1
    state = json.loads((tmp_path / "loop" / STATE).read_text())
    assert state["seed_dirs"][0]["dir"] == str(seed.resolve())
