"""The fast loop (alphasettler/train/fastloop.py): set-up, generations, the window, resuming and
the regression gate. Needs the `train` extra, like test_train.py."""

import hashlib
import json
import struct

import pytest

from alphasettler import records
from alphasettler._engine import (
    LEAGUE_STYLES, NET_ENCODING_HASH, NET_V3_ENCODING_HASH, OPENING_FILE, SELFPLAY_FAST_SIMULATIONS, TrainingSet,
)
from alphasettler.arena import TRADES
from alphasettler.selfplay import write_explore
from alphasettler.train import fastloop
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


def test_a_targets_v2_loop_records_the_setting_and_carries_the_sidecar_between_generations(tmp_path, capsys):
    seed = generation(tmp_path, "seed-gen", seed_start=0)
    save(Net(8, 1), tmp_path / "start.bin")
    root = tmp_path / "loop"
    FastLoop.init(root, [seed], tmp_path / "start.bin", first=1, targets_v2=True)
    loop = FastLoop(root, FastLoopSettings(generations=2, **TINY))
    assert loop.targets == 2
    assert loop.run() == 2
    state = json.loads((root / STATE).read_text())
    assert state["targets"] == 2
    for k in (1, 2):
        assert (root / "nets" / f"{gen_name(k)}.bin.train.pt").exists()
    out = capsys.readouterr().out
    first = out[out.index("gen-001: train"):out.index("gen-001: gate")]
    second = out[out.index("gen-002: train"):out.index("gen-002: gate")]
    assert "start fresh" in first, "start.bin has no sidecar"
    assert "start fresh" not in second, "gen-002 continues from gen-001's sidecar"


def test_cli_fast_loop_keeps_the_targets_setting_it_was_set_up_with(tmp_path):
    root, _ = setup(tmp_path)  # set up without --targets-v2
    assert FastLoop(root, FastLoopSettings(generations=0, **TINY)).targets == 1
    r = cli("fast-loop", "--dir", str(root), "--generations", "0", "--targets-v2")
    assert r.returncode == 2 and (f"{root} was set up without --targets-v2, and a loop keeps the setting it was "
                                  f"set up with") in r.stderr, r.stderr


def test_a_loop_set_up_before_targets_v2_reads_as_targets_1(tmp_path):
    root, _ = setup(tmp_path)
    state = json.loads((root / STATE).read_text())
    del state["targets"]
    (root / STATE).write_text(json.dumps(state))
    assert FastLoop(root, FastLoopSettings(generations=0, **TINY)).targets == 1


def test_an_ema_loop_records_the_setting_and_trains_at_the_constant_rate(tmp_path):
    seed = generation(tmp_path, "seed-gen", seed_start=0)
    save(Net(8, 1), tmp_path / "start.bin")
    root = tmp_path / "loop"
    r = cli("fast-loop", "--dir", str(root), "--generations", "2", "--first", "1", "--targets-v2", "--ema",
            "--seed-dirs", str(seed), "--start-net", str(tmp_path / "start.bin"), "--games", "2",
            "--window-games", "3", "--minutes", "0.01", "--gate-seeds", "1", "--gate-simulations", "6",
            "--device", "cpu", "--threads", "2")
    assert r.returncode == 0, r.stderr
    state = json.loads((root / STATE).read_text())
    assert state["ema"] is True and state["targets"] == 2
    summaries = [json.loads((root / gen_name(k) / "net.bin.json").read_text()) for k in (1, 2)]
    for s in summaries:
        assert s["config"]["lr"] == 2.5e-4 and s["config"]["ema"] is True
    assert summaries[0]["warmup_skipped"] is False, "start.bin has no Adam state"
    assert summaries[1]["warmup_skipped"] is True, "gen-002 continues gen-001's Adam"
    for k in (1, 2):
        assert (root / "nets" / f"{gen_name(k)}.bin.train.pt").exists()


def test_cli_fast_loop_keeps_the_ema_setting_it_was_set_up_with(tmp_path):
    root, _ = setup(tmp_path)  # set up without --ema
    assert FastLoop(root, FastLoopSettings(generations=0, **TINY)).ema is False
    assert json.loads((root / STATE).read_text())["ema"] is False
    r = cli("fast-loop", "--dir", str(root), "--generations", "0", "--ema")
    assert r.returncode == 2 and (f"{root} was set up without --ema, and a loop keeps the setting it was "
                                  f"set up with") in r.stderr, r.stderr


def test_a_loop_set_up_before_ema_reads_as_without_it(tmp_path):
    root, _ = setup(tmp_path)
    state = json.loads((root / STATE).read_text())
    del state["ema"]
    (root / STATE).write_text(json.dumps(state))
    assert FastLoop(root, FastLoopSettings(generations=0, **TINY)).ema is False


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


def test_cli_fast_loop_restores_a_missing_opening_model_from_the_same_file_only(tmp_path):
    seed = generation(tmp_path, "seed-gen", seed_start=0)
    save(Net(8, 1), tmp_path / "start.bin")
    model = opening_model_file(tmp_path / "model.bin")
    other = opening_model_file(tmp_path / "other.bin", scale=0.0)
    root = tmp_path / "loop"
    FastLoop.init(root, [seed], tmp_path / "start.bin", first=1, opening=model)
    placed = root / "nets" / OPENING_FILE
    placed.unlink()
    r = cli("fast-loop", "--dir", str(root), "--generations", "1")
    assert r.returncode == 2 and "restore" in r.stderr and not placed.exists(), r.stderr
    r = cli("fast-loop", "--dir", str(root), "--generations", "1", "--opening", str(other))
    assert r.returncode == 2 and "another opening model" in r.stderr and not placed.exists(), r.stderr
    r = cli("fast-loop", "--dir", str(root), "--generations", "1", "--opening", str(model), "--games", "2",
            "--window-games", "3", "--minutes", "0.01", "--gate-seeds", "1", "--gate-simulations", "6",
            "--device", "cpu", "--threads", "2")
    assert r.returncode == 0, r.stderr
    assert placed.read_bytes() == model.read_bytes()
    state = json.loads((root / STATE).read_text())
    assert [g["gen"] for g in state["generations"]] == [1]
    assert state["generations"][0]["selfplay"]["opening"] == hashlib.sha256(model.read_bytes()).hexdigest()


LEAGUE_TINY = {**TINY, "games": 6, "chunk_games": 6, "league_simulations": 6}


def league_nets(tmp_path, n=2):
    """`n` tiny net files outside any loop: a league's set-up pool."""
    paths = []
    for i in range(n):
        path = tmp_path / "pool" / f"old-{i}.bin"
        path.parent.mkdir(exist_ok=True)
        save(Net(8, 1), path)
        paths.append(path)
    return paths


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_a_league_loop_self_plays_against_its_pool_and_adds_each_promoted_net(tmp_path, monkeypatch):
    seed = generation(tmp_path, "seed-gen", seed_start=0)
    save(Net(8, 1), tmp_path / "start.bin")
    pool = league_nets(tmp_path)
    root = tmp_path / "loop"
    FastLoop.init(root, [seed], tmp_path / "start.bin", first=1, trades=True, league=pool)
    state = json.loads((root / STATE).read_text())
    assert state["league"] == {"nets": [{"name": f"league-{i:03d}", "source": str(p.resolve()), "sha256": sha(p)}
                                        for i, p in enumerate(pool)]}
    for i, p in enumerate(pool):
        assert (root / "nets" / f"league-{i:03d}.bin").read_bytes() == p.read_bytes()
    monkeypatch.setattr(fastloop, "promoted", lambda z: True)  # gen-001 is promoted, so gen 2 meets it
    assert FastLoop(root, FastLoopSettings(generations=2, **LEAGUE_TINY)).run() == 2
    g1, g2 = json.loads((root / STATE).read_text())["generations"]
    assert g1["gate"]["promoted"] and g1["gate"]["candidate"] == "net:gen-001@6+t", g1["gate"]
    set_up = ["net:league-000@6+t", "net:league-001@6+t"]
    assert g1["selfplay"]["league"]["nets"] == set_up
    assert g2["selfplay"]["league"]["nets"] == set_up + ["net:gen-001@6+t"]
    total = 0
    for k, g in ((1, g1), (2, g2)):
        lg = g["selfplay"]["league"]
        assert lg["styles"] == [f"style:{s}@6+t" for s in LEAGUE_STYLES]
        games = [x for p in sorted((root / gen_name(k)).glob("selfplay-local-*.jsonl.gz")) for x in records.read(p)]
        assert len(games) == 6 and g["selfplay"]["belief_resets"] == 0
        seats = dict.fromkeys(lg["nets"] + lg["styles"], 0)
        league_games = [x for x in games if "league" in x]
        for x in league_games:
            assert x["selfplay"]["league_nets"] == lg["nets"] and x["selfplay"]["league_styles"] == lg["styles"]
            pool_seats = {i for i, n in enumerate(x["league"]) if n is not None}
            assert not pool_seats & {d["actor"] for d in x["decisions"]}, "league seats are not recorded"
            for n in x["league"]:
                if n is not None:
                    seats[n] += 1
        assert (lg["games"], lg["seats"]) == (len(league_games), seats)
        total += lg["games"]
    assert 0 < total < 12, f"{total} league games of 12"


@pytest.mark.parametrize("set_up, message", [
    (False, "without --league"),
    (True, "with another league pool"),
], ids=["set-up-without", "another-pool"])
def test_cli_fast_loop_keeps_the_league_it_was_set_up_with(tmp_path, set_up, message):
    seed = generation(tmp_path, "seed-gen", seed_start=0)
    save(Net(8, 1), tmp_path / "start.bin")
    pool = league_nets(tmp_path)
    root = tmp_path / "loop"
    r = cli("fast-loop", "--dir", str(root), "--generations", "0", "--first", "1", "--trades", "--seed-dirs", str(seed),
            "--start-net", str(tmp_path / "start.bin"), *(["--league", *map(str, pool)] if set_up else []))
    assert r.returncode == 0, r.stderr
    want = json.loads((root / STATE).read_text())["league"]
    assert (want is not None) == set_up
    assert FastLoop(root, FastLoopSettings(generations=0, **TINY)).league == want
    r = cli("fast-loop", "--dir", str(root), "--generations", "0", "--league", *map(str, pool[:1] if set_up else pool))
    assert r.returncode == 2 and message in r.stderr, r.stderr
    r = cli("fast-loop", "--dir", str(root), "--generations", "0")
    assert r.returncode == 0, r.stderr
    assert json.loads((root / STATE).read_text())["league"] == want, "a resume without the flag keeps it"
    if set_up:
        r = cli("fast-loop", "--dir", str(root), "--generations", "0", "--league", *map(str, pool))
        assert r.returncode == 0, "the same files repeat the setting"


def test_a_loop_set_up_before_the_league_existed_reads_as_set_up_without_it(tmp_path):
    root, _ = setup(tmp_path)
    state = json.loads((root / STATE).read_text())
    del state["league"]
    (root / STATE).write_text(json.dumps(state))
    loop = FastLoop(root, FastLoopSettings(generations=0, **TINY))
    assert loop.league is None and loop.league_pool() is None
    r = cli("fast-loop", "--dir", str(root), "--generations", "0", "--league", *map(str, league_nets(tmp_path)))
    assert r.returncode == 2 and "without --league" in r.stderr


def test_a_bad_league_net_or_a_league_with_gumbel_is_refused_before_anything_is_written(tmp_path):
    seed = generation(tmp_path, "seed-gen", seed_start=0)
    save(Net(8, 1), tmp_path / "start.bin")
    (good,) = league_nets(tmp_path, 1)
    cut = tmp_path / "cut.bin"
    cut.write_bytes(good.read_bytes()[:-4])
    other = tmp_path / "other-encoder.bin"
    data = bytearray(good.read_bytes())
    data[12:20] = struct.pack("<Q", NET_ENCODING_HASH ^ 1)  # another feature layout's hash
    other.write_bytes(bytes(data))
    v3 = tmp_path / "v3.bin"
    data[12:20] = struct.pack("<Q", NET_V3_ENCODING_HASH)  # an encoder v3 net: the loader names net-upgrade
    v3.write_bytes(bytes(data))
    for kw, message in [({"league": [good, cut]}, "league net .*cut.bin.*truncated"),
                        ({"league": [tmp_path / "missing.bin"]}, "league net .*missing.bin.*cannot read"),
                        ({"league": [other]}, "league net .*other-encoder.bin.*encoding hash"),
                        ({"league": [v3]}, "league net .*v3.bin.*net-upgrade"),
                        ({"league": []}, "at least one net file"),
                        ({"league": [good], "gumbel": 100}, "--league and --gumbel")]:
        with pytest.raises(ValueError, match=message):
            FastLoop.init(tmp_path / "bad-loop", [seed], tmp_path / "start.bin", first=1, **kw)
        assert not (tmp_path / "bad-loop").exists()
    r = cli("fast-loop", "--dir", str(tmp_path / "cli"), "--generations", "0", "--first", "1", "--seed-dirs", str(seed),
            "--start-net", str(tmp_path / "start.bin"), "--league", str(good), "--gumbel", "100")
    assert r.returncode == 2 and "--league and --gumbel" in r.stderr
    assert not (tmp_path / "cli").exists()


def test_a_changed_league_net_in_the_nets_directory_stops_the_loop(tmp_path):
    seed = generation(tmp_path, "seed-gen", seed_start=0)
    save(Net(8, 1), tmp_path / "start.bin")
    root = tmp_path / "loop"
    FastLoop.init(root, [seed], tmp_path / "start.bin", first=1, league=league_nets(tmp_path))
    save(Net(8, 1), root / "nets" / "league-001.bin")  # replaced in place after set-up
    with pytest.raises(ValueError, match="league-001.bin is not the league net"):
        FastLoop(root, FastLoopSettings(generations=1, **TINY)).run()
    assert not list((root / "gen-001").glob("selfplay-local-*.jsonl.gz")), "nothing was played against it"


def test_league_seats_take_the_loops_trading_and_opening_suffixes(tmp_path):
    assert SELFPLAY_FAST_SIMULATIONS == 100
    seed = generation(tmp_path, "seed-gen", seed_start=0)
    save(Net(8, 1), tmp_path / "start.bin")
    pool = league_nets(tmp_path, 1)
    model = opening_model_file(tmp_path / "model.bin")
    for trades, opening, net_suffix, style_suffix in [(True, model, "+t+o", "+t"), (False, None, "", "")]:
        root = tmp_path / f"loop-{trades}"
        FastLoop.init(root, [seed], tmp_path / "start.bin", first=1, trades=trades, opening=opening, league=pool)
        assert FastLoop(root, FastLoopSettings(generations=0, **TINY)).league_pool() == {
            "nets": [f"net:league-000@100{net_suffix}"],
            "styles": [f"style:{s}@100{style_suffix}" for s in LEAGUE_STYLES]}


def test_a_league_games_result_is_the_value_target_of_its_two_recorded_seats(tmp_path):
    """Only the learner seats' decisions are recorded, but the game's result (its winner, a pool seat
    included, and all four final vp) is each of their value targets: training and the loop's counts
    do not assume four recorded seats."""
    out = tmp_path / "league.jsonl.gz"
    league = {"nets": ["ismcts@5"], "styles": [f"style:{s}@6" for s in LEAGUE_STYLES]}
    write_explore(out, 0, 20, config={"max_offers_per_turn": 0}, full_simulations=6, fast_simulations=3,
                  full_probability=0.25, threads=2, league=league)
    games = [g for g in records.read(out) if "league" in g]
    assert len(games) >= 3
    ts = TrainingSet(games)
    assert ts.games == len(games) and ts.decisions == sum(len(g["decisions"]) for g in games)
    pool_won = learner_won = 0
    for gi, g in enumerate(games):
        learners = {i for i, n in enumerate(g["league"]) if n is None}
        assert len(learners) == 2 and {d["actor"] for d in g["decisions"]} == learners
        pool_won += g["winner"] not in learners
        learner_won += g["winner"] in learners
        for di, d in enumerate(g["decisions"]):
            b = ts.position(gi, di)
            actor = d["actor"]
            want = [float((g["winner"] - actor) % 4 == r) for r in range(4)]  # seat-relative, actor first
            assert struct.unpack("<4f", b["outcome"]) == tuple(want), f"game {g['seed']} decision {di}"
            vp = struct.unpack("<4f", b["vp"])
            assert vp == pytest.approx([g["vp"][(r + actor) % 4] / 10 for r in range(4)])
    assert pool_won and learner_won, f"{pool_won} games won by a pool seat, {learner_won} by a learner seat"


def test_cli_fast_loop_with_compound_moves_self_plays_and_gates_with_them(tmp_path):
    seed = generation(tmp_path, "seed-gen", seed_start=0)
    save(Net(8, 1), tmp_path / "start.bin")
    root = tmp_path / "loop"
    r = cli("fast-loop", "--dir", str(root), "--generations", "2", "--first", "1", "--trades", "--compound",
            "--seed-dirs", str(seed), "--start-net", str(tmp_path / "start.bin"), "--games", "2",
            "--window-games", "3", "--minutes", "0.01", "--gate-seeds", "1", "--gate-simulations", "6",
            "--device", "cpu", "--threads", "2")
    assert r.returncode == 0, r.stderr
    state = json.loads((root / STATE).read_text())
    assert state["compound"] is True and [g["gen"] for g in state["generations"]] == [1, 2]
    for g in state["generations"]:
        assert g["selfplay"]["compound"] is True and g["selfplay"]["belief_resets"] == 0
        assert g["gate"]["candidate"].endswith("@6+t+c") and g["gate"]["baseline"].endswith("@6+t+c"), g["gate"]
        assert g["gate"]["belief_resets"] == 0
    for k in (1, 2):
        paths = sorted((root / gen_name(k)).glob("selfplay-local-*.jsonl.gz"))
        assert paths
        for path in paths:
            for game in records.read(path):
                assert game["selfplay"]["compound"] is True


def test_cli_fast_loop_keeps_the_compound_setting_it_was_set_up_with(tmp_path):
    root, seed = setup(tmp_path)  # set up without --compound
    assert FastLoop(root, FastLoopSettings(generations=0, **TINY)).compound is False
    r = cli("fast-loop", "--dir", str(root), "--generations", "0", "--compound")
    assert r.returncode == 2 and "without --compound" in r.stderr, r.stderr
    on = tmp_path / "on"
    FastLoop.init(on, [seed], tmp_path / "start.bin", first=1, compound=True)
    r = cli("fast-loop", "--dir", str(on), "--generations", "0")
    assert r.returncode == 0, r.stderr
    assert json.loads((on / STATE).read_text())["compound"] is True, "a resume without the flag keeps it"
    r = cli("fast-loop", "--dir", str(on), "--generations", "0", "--compound")
    assert r.returncode == 0, "the same setting repeats"
    # A state file from before 3d-4 has no "compound" key: it reads as set up without.
    state = json.loads((root / STATE).read_text())
    del state["compound"]
    (root / STATE).write_text(json.dumps(state))
    assert FastLoop(root, FastLoopSettings(generations=0, **TINY)).compound is False


def test_compound_moves_with_gumbel_are_refused_before_anything_is_written(tmp_path):
    seed = generation(tmp_path, "seed-gen", seed_start=0)
    save(Net(8, 1), tmp_path / "start.bin")
    with pytest.raises(ValueError, match="--compound and --gumbel"):
        FastLoop.init(tmp_path / "bad", [seed], tmp_path / "start.bin", first=1, gumbel=100, compound=True)
    assert not (tmp_path / "bad").exists()
    r = cli("fast-loop", "--dir", str(tmp_path / "cli"), "--generations", "0", "--first", "1", "--seed-dirs", str(seed),
            "--start-net", str(tmp_path / "start.bin"), "--compound", "--gumbel", "100")
    assert r.returncode == 2 and "--compound and --gumbel" in r.stderr
    assert not (tmp_path / "cli").exists()


def test_a_compound_league_loop_keeps_its_pool_names_plain(tmp_path):
    seed = generation(tmp_path, "seed-gen", seed_start=0)
    save(Net(8, 1), tmp_path / "start.bin")
    root = tmp_path / "loop"
    FastLoop.init(root, [seed], tmp_path / "start.bin", first=1, trades=True, league=league_nets(tmp_path, 1),
                  compound=True)
    loop = FastLoop(root, FastLoopSettings(generations=0, **TINY))
    assert loop.compound is True
    assert loop.league_pool() == {"nets": ["net:league-000@100+t"],
                                  "styles": [f"style:{s}@100+t" for s in LEAGUE_STYLES]}


def test_a_targets_v2_generation_trained_without_its_sidecar_is_trained_again_on_resume(tmp_path, monkeypatch,
                                                                                        capsys):
    seed = generation(tmp_path, "seed-gen", seed_start=0)
    save(Net(8, 1), tmp_path / "start.bin")
    root = tmp_path / "loop"
    FastLoop.init(root, [seed], tmp_path / "start.bin", first=1, targets_v2=True)
    gate = FastLoop._gate

    def crash(*args):
        raise KeyboardInterrupt("stopped before the gate")

    monkeypatch.setattr(FastLoop, "_gate", crash)
    for _ in range(2):  # trained, then resumed with the sidecar there: not trained again
        with pytest.raises(KeyboardInterrupt):
            FastLoop(root, FastLoopSettings(generations=1, **TINY)).run()
    assert capsys.readouterr().out.count(" steps from ") == 1
    gdir = root / gen_name(1)
    assert (gdir / "net.bin").exists() and (gdir / "net.bin.json").exists()
    (gdir / "net.bin.train.pt").unlink()
    (root / "nets" / f"{gen_name(1)}.bin.train.pt").unlink()
    monkeypatch.setattr(FastLoop, "_gate", gate)
    assert FastLoop(root, FastLoopSettings(generations=1, **TINY)).run() == 1
    assert " steps from " in capsys.readouterr().out, "no sidecar: the generation trains again"
    assert (gdir / "net.bin.train.pt").exists() and (root / "nets" / f"{gen_name(1)}.bin.train.pt").exists()
