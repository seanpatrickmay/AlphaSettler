"""The fast loop (alphasettler/train/fastloop.py): set-up, generations, the window, resuming and
the regression gate. Needs the `train` extra, like test_train.py."""

import hashlib
import json

import pytest

from alphasettler import records
from alphasettler._engine import OPENING_FILE
from alphasettler.arena import TRADES
from alphasettler.train.fastloop import GATE_MIN_Z, STATE, FastLoop, FastLoopSettings, count_games, promoted
from alphasettler.train.loop import SEEDS_PER_GENERATION, gen_name
from alphasettler.train.model import Net, save

from opening_games import opening_model_file
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


def test_a_trading_loop_plays_and_gates_in_the_trading_game(tmp_path):
    seed = generation(tmp_path, "seed-gen", seed_start=0)  # no-trade records start the window
    save(Net(8, 1), tmp_path / "start.bin")
    root = tmp_path / "loop"
    FastLoop.init(root, [seed], tmp_path / "start.bin", first=1, trades=True)
    loop = FastLoop(root, FastLoopSettings(generations=2, **TINY))
    assert loop.trades and loop.config == {**TRADES, "max_turns": 1000}
    assert loop.run() == 2
    state = json.loads((root / STATE).read_text())
    assert state["trades"] is True
    for g in state["generations"]:
        assert g["gate"]["candidate"].endswith("+t") and g["gate"]["baseline"].endswith("+t"), g["gate"]
    for k in (1, 2):
        paths = sorted((root / gen_name(k)).glob("selfplay-local-*.jsonl.gz"))
        assert paths
        for path in paths:
            for game in records.read(path):
                assert (game["config"]["max_offers_per_turn"], game["config"]["max_trade_cards"]) == (2, 1)


def test_cli_fast_loop_keeps_the_trading_setting_it_was_set_up_with(tmp_path):
    root, _ = setup(tmp_path)  # set up without --trades
    assert FastLoop(root, FastLoopSettings(generations=0, **TINY)).trades is False
    r = cli("fast-loop", "--dir", str(root), "--generations", "0", "--trades")
    assert r.returncode == 2 and "without --trades" in r.stderr


@pytest.mark.parametrize("trades", [True, False])
def test_cli_fast_loop_records_the_trading_setting_of_a_new_loop(tmp_path, trades):
    seed = generation(tmp_path, "seed-gen", seed_start=0)
    save(Net(8, 1), tmp_path / "start.bin")
    root = tmp_path / "loop"
    flag = ["--trades"] if trades else []
    r = cli("fast-loop", "--dir", str(root), "--generations", "0", *flag, "--seed-dirs", str(seed),
            "--start-net", str(tmp_path / "start.bin"), "--first", "1")
    assert r.returncode == 0, r.stderr
    assert json.loads((root / STATE).read_text())["trades"] is trades


def test_cli_fast_loop_with_gumbel_runs_two_generations_of_gumbel_self_play(tmp_path):
    seed = generation(tmp_path, "seed-gen", seed_start=0)
    save(Net(8, 1), tmp_path / "start.bin")
    root = tmp_path / "loop"
    r = cli("fast-loop", "--dir", str(root), "--generations", "2", "--first", "1", "--gumbel", "100",
            "--seed-dirs", str(seed), "--start-net", str(tmp_path / "start.bin"), "--games", "2",
            "--window-games", "3", "--minutes", "0.01", "--gate-seeds", "1", "--gate-simulations", "6",
            "--device", "cpu", "--threads", "2")
    assert r.returncode == 0, r.stderr
    state = json.loads((root / STATE).read_text())
    assert state["gumbel"] == 100 and [g["gen"] for g in state["generations"]] == [1, 2]
    for g in state["generations"]:
        sp = g["selfplay"]
        assert sp["gumbel"] == 100 and sp["belief_resets"] == 0
        assert sp["full_decisions"] == sp["decisions"] > 0
        assert "+g" not in g["gate"]["candidate"] and "+g" not in g["gate"]["baseline"], "gates are unchanged"
    for k in (1, 2):
        paths = sorted((root / gen_name(k)).glob("selfplay-local-*.jsonl.gz"))
        assert paths
        for path in paths:
            for game in records.read(path):
                assert game["selfplay"]["gumbel_simulations"] == 100
                for d in game["decisions"]:
                    assert d["full_search"] is True and sum(d["visits"]) == 99
                    assert len(d["target"]) == len(d["legal"])
                    assert sum(d["target"]) == pytest.approx(1.0, abs=1e-4)


@pytest.mark.parametrize("set_up, resume, message", [
    (None, ["--gumbel", "100"], "without --gumbel"),
    (100, ["--gumbel", "50"], "with --gumbel 100"),
], ids=["set-up-without", "other-value"])
def test_cli_fast_loop_keeps_the_self_play_mode_it_was_set_up_with(tmp_path, set_up, resume, message):
    seed = generation(tmp_path, "seed-gen", seed_start=0)
    save(Net(8, 1), tmp_path / "start.bin")
    root = tmp_path / "loop"
    FastLoop.init(root, [seed], tmp_path / "start.bin", first=1, gumbel=set_up)
    assert FastLoop(root, FastLoopSettings(generations=0, **TINY)).gumbel == set_up
    r = cli("fast-loop", "--dir", str(root), "--generations", "0", *resume)
    assert r.returncode == 2 and message in r.stderr, r.stderr
    r = cli("fast-loop", "--dir", str(root), "--generations", "0")
    assert r.returncode == 0, r.stderr
    assert json.loads((root / STATE).read_text())["gumbel"] == set_up, "a resume without the flag keeps it"


def test_a_loop_set_up_before_gumbel_existed_reads_as_set_up_without_it(tmp_path):
    root, _ = setup(tmp_path)
    state = json.loads((root / STATE).read_text())
    del state["gumbel"]
    (root / STATE).write_text(json.dumps(state))
    assert FastLoop(root, FastLoopSettings(generations=0, **TINY)).gumbel is None
    r = cli("fast-loop", "--dir", str(root), "--generations", "0", "--gumbel", "100")
    assert r.returncode == 2 and "without --gumbel" in r.stderr


def test_a_gumbel_budget_below_two_is_refused_before_anything_is_written(tmp_path):
    seed = generation(tmp_path, "seed-gen", seed_start=0)
    save(Net(8, 1), tmp_path / "start.bin")
    with pytest.raises(ValueError, match="gumbel simulations"):
        FastLoop.init(tmp_path / "bad", [seed], tmp_path / "start.bin", first=1, gumbel=1)
    assert not (tmp_path / "bad").exists()


SETUP_SETTLEMENTS = {0, 2, 4, 6, 8, 10, 12, 14}  # the first 16 moves: settlement, road, eight times


def test_cli_fast_loop_with_an_opening_model_self_plays_and_gates_with_it(tmp_path):
    seed = generation(tmp_path, "seed-gen", seed_start=0)
    save(Net(8, 1), tmp_path / "start.bin")
    model = opening_model_file(tmp_path / "model.bin")
    digest = hashlib.sha256(model.read_bytes()).hexdigest()
    root = tmp_path / "loop"
    r = cli("fast-loop", "--dir", str(root), "--generations", "2", "--first", "1", "--trades", "--opening", str(model),
            "--seed-dirs", str(seed), "--start-net", str(tmp_path / "start.bin"), "--games", "2",
            "--window-games", "3", "--minutes", "0.01", "--gate-seeds", "1", "--gate-simulations", "6",
            "--device", "cpu", "--threads", "2")
    assert r.returncode == 0, r.stderr
    assert (root / "nets" / OPENING_FILE).read_bytes() == model.read_bytes()
    state = json.loads((root / STATE).read_text())
    assert state["opening"] == digest and [g["gen"] for g in state["generations"]] == [1, 2]
    for g in state["generations"]:
        assert g["selfplay"]["opening"] == digest and g["selfplay"]["belief_resets"] == 0
        assert g["gate"]["candidate"].endswith("@6+t+o") and g["gate"]["baseline"].endswith("@6+t+o"), g["gate"]
        assert g["gate"]["belief_resets"] == 0
    for k in (1, 2):
        paths = sorted((root / gen_name(k)).glob("selfplay-local-*.jsonl.gz"))
        assert paths
        for path in paths:
            for game in records.read(path):
                assert game["selfplay"]["opening_sha256"] == digest
                assert not SETUP_SETTLEMENTS & {d["index"] for d in game["decisions"]}


@pytest.mark.parametrize("set_up, message", [
    (False, "without --opening"),
    (True, "with another opening model"),
], ids=["set-up-without", "another-model"])
def test_cli_fast_loop_keeps_the_opening_model_it_was_set_up_with(tmp_path, set_up, message):
    seed = generation(tmp_path, "seed-gen", seed_start=0)
    save(Net(8, 1), tmp_path / "start.bin")
    model = opening_model_file(tmp_path / "model.bin")
    other = opening_model_file(tmp_path / "other.bin", scale=0.0)
    root = tmp_path / "loop"
    FastLoop.init(root, [seed], tmp_path / "start.bin", first=1, opening=model if set_up else None)
    want = hashlib.sha256(model.read_bytes()).hexdigest() if set_up else None
    assert FastLoop(root, FastLoopSettings(generations=0, **TINY)).opening == want
    r = cli("fast-loop", "--dir", str(root), "--generations", "0", "--opening", str(other if set_up else model))
    assert r.returncode == 2 and message in r.stderr, r.stderr
    r = cli("fast-loop", "--dir", str(root), "--generations", "0")
    assert r.returncode == 0, r.stderr
    assert json.loads((root / STATE).read_text())["opening"] == want, "a resume without the flag keeps it"
    if set_up:
        r = cli("fast-loop", "--dir", str(root), "--generations", "0", "--opening", str(model))
        assert r.returncode == 0, "the same file repeats the setting"


def test_a_loop_set_up_before_the_opening_model_existed_reads_as_set_up_without_it(tmp_path):
    root, _ = setup(tmp_path)
    state = json.loads((root / STATE).read_text())
    del state["opening"]
    (root / STATE).write_text(json.dumps(state))
    assert FastLoop(root, FastLoopSettings(generations=0, **TINY)).opening is None
    model = opening_model_file(tmp_path / "model.bin")
    r = cli("fast-loop", "--dir", str(root), "--generations", "0", "--opening", str(model))
    assert r.returncode == 2 and "without --opening" in r.stderr


def test_a_bad_opening_model_or_one_with_gumbel_is_refused_before_anything_is_written(tmp_path):
    seed = generation(tmp_path, "seed-gen", seed_start=0)
    save(Net(8, 1), tmp_path / "start.bin")
    model = opening_model_file(tmp_path / "model.bin")
    bad = tmp_path / "bad.bin"
    bad.write_bytes(model.read_bytes()[:-4])
    for kw, message in [({"opening": bad}, "truncated"), ({"opening": tmp_path / "missing.bin"}, "cannot read opening model"),
                        ({"opening": model, "gumbel": 100}, "--opening and --gumbel")]:
        with pytest.raises(ValueError, match=message):
            FastLoop.init(tmp_path / "bad-loop", [seed], tmp_path / "start.bin", first=1, **kw)
        assert not (tmp_path / "bad-loop").exists()
    r = cli("fast-loop", "--dir", str(tmp_path / "cli"), "--generations", "0", "--first", "1", "--seed-dirs", str(seed),
            "--start-net", str(tmp_path / "start.bin"), "--opening", str(model), "--gumbel", "100")
    assert r.returncode == 2 and "--opening and --gumbel" in r.stderr
    assert not (tmp_path / "cli").exists()


def test_a_changed_opening_model_in_the_nets_directory_stops_the_loop(tmp_path):
    seed = generation(tmp_path, "seed-gen", seed_start=0)
    save(Net(8, 1), tmp_path / "start.bin")
    root = tmp_path / "loop"
    FastLoop.init(root, [seed], tmp_path / "start.bin", first=1, opening=opening_model_file(tmp_path / "model.bin"))
    opening_model_file(root / "nets" / OPENING_FILE, scale=0.0)  # replaced in place after set-up
    with pytest.raises(ValueError, match="not the opening model"):
        FastLoop(root, FastLoopSettings(generations=1, **TINY)).run()
    assert not list((root / "gen-001").glob("selfplay-local-*.jsonl.gz")), "nothing was played with it"
