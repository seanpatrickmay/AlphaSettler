"""The generation loop (`train-loop`): manifest, atomic outputs, resume and gating.

These tests need the `train` extra (PyTorch: `.venv/bin/pip3 install -e '.[train]'`). Without it
they fail when the loop trains, on purpose: they must be run, never skipped."""

import hashlib
import json
import math
import os
import shutil
import socket
import subprocess
import sys
from dataclasses import replace

import pytest

from alphasettler import files
from alphasettler._engine import NET_MAX_ROUNDS, NET_MAX_WIDTH
from alphasettler.train.loop import Loop, LoopSettings, json_float, promoted

TINY = LoopSettings(generations=1, games=2, warm_games=2, chunk_games=1, full_simulations=4, fast_simulations=2,
                    full_probability=0.5, batch=16, width=8, rounds=1, gate_seeds=1, gate_simulations=4,
                    max_turns=60, device="cpu", threads=2)


def manifest(d):
    return json.loads((d / "manifest.json").read_text())


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def temps(d):
    return [p for p in d.rglob(".*.tmp-*")]


@pytest.fixture(scope="module")
def reference(tmp_path_factory):
    d = tmp_path_factory.mktemp("reference")
    assert Loop(d, TINY).run() == 6
    return d


def test_the_promotion_rule_includes_a_tie_at_a_quarter():
    assert promoted(200, 800) and promoted(50, 200) and promoted(1, 4)
    assert promoted(201, 800)
    assert not promoted(199, 800) and not promoted(0, 4)


def test_a_full_run_writes_every_output_and_redoes_nothing(reference):
    m = manifest(reference)
    assert [sorted(g) for g in m["generations"]] == [["gate", "gen", "selfplay", "train"]] * 2
    g0, g1 = m["generations"]
    assert g0["selfplay"]["bot"] == "ismcts" and g1["selfplay"]["bot"] == "net:gen-000"
    assert g0["selfplay"]["belief_resets"] == 0 and g1["selfplay"]["belief_resets"] == 0
    assert g0["gate"]["promoted"] is True
    assert (g0["train"]["init"], g1["train"]["init"]) == (None, "gen-000")
    assert (g0["train"]["lr"], g1["train"]["lr"]) == (1e-3, 3e-4)
    assert [(g["train"]["width"], g["train"]["rounds"]) for g in (g0, g1)] == [(8, 1), (8, 1)]
    assert m["format"] == 2
    assert g1["train"]["window"] == ["gen-000", "gen-001"]
    assert (g1["train"]["window_size"], g1["train"]["steps_factor"]) == (4, 2.0)
    assert g1["gate"]["config"] == {"max_offers_per_turn": 0, "max_turns": 60}
    assert g1["train"]["parity_max_error"] <= 1e-5
    for g, held_seed in ((g0, 0), (g1, 10_000_000)):  # each newest generation's seed divisible by 20
        t = g["train"]
        assert (t["validation_games"], t["held_out_every"]) == (1, 20) and t["validation_positions"] > 0
        assert t["held_out"] and t["held_out"][-1]["step"] == t["steps"] - 1
        assert t["best_held_out"] == min(t["held_out"], key=lambda r: r["score"])
        assert t["best_step"] == t["best_held_out"]["step"]
    # Generation 0 trains on seed 1 only; generation 1 on both of gen-000's (older) and its own seed 10,000,001.
    assert (g0["train"]["games"], g1["train"]["games"]) == (1, 3)
    # Half of each of generation 1's batches is its own; generation 0 has no older data.
    assert (g0["train"]["newest_share"], g0["train"]["older_positions"]) == (1.0, 0)
    assert g0["train"]["positions"] == g0["train"]["new_positions"]
    assert g1["train"]["newest_share"] == 0.5 and g1["train"]["older_positions"] > 0
    assert g1["train"]["positions"] == g1["train"]["older_positions"] + g1["train"]["new_positions"]
    assert g1["gate"]["games"] == 4 and g1["gate"]["promoted"] == promoted(g1["gate"]["wins"], 4)
    assert g1["gate"]["belief_resets"] == 0
    assert m["best"] == (1 if g1["gate"]["promoted"] else 0)
    nets = reference / "nets"
    assert (nets / "best.bin").read_bytes() == (nets / f"gen-00{m['best']}.bin").read_bytes()
    for k, g in enumerate(m["generations"]):
        gdir = reference / f"gen-00{k}"
        assert g["train"]["net"] == sha256(gdir / "net.bin") == sha256(nets / f"gen-00{k}.bin")
        assert g["selfplay"]["chunks"] == {name: sha256(gdir / name)
                                           for name in ("selfplay-local-00000.jsonl.gz", "selfplay-local-00001.jsonl.gz")}
    assert temps(reference) == []
    assert Loop(reference, TINY).run() == 0, "nothing is redone"


def _reject(constant):
    raise ValueError(f"not strict JSON: {constant}")


def test_the_manifest_is_strict_json_and_a_non_finite_z_is_null(reference):
    """Preflight F6. One gate seed has a zero standard error, so z is 0 at a win rate of exactly
    0.25 and infinite otherwise; an infinite z is written as null."""
    for path in (reference / "manifest.json", reference / "gen-001" / "gate.json", reference / "gen-001" / "selfplay.json"):
        json.loads(path.read_text(), parse_constant=_reject)
    gate = manifest(reference)["generations"][1]["gate"]
    assert gate["z"] == (0.0 if gate["wins"] == 1 else None)
    assert json_float(math.inf) is None and json_float(-math.inf) is None and json_float(math.nan) is None
    assert json_float(1.23456789) == 1.234568 and json_float(0.0) == 0.0


def test_stopping_after_each_step_and_resuming_gives_the_same_manifest(tmp_path, reference):
    executed = []
    while (n := Loop(tmp_path, TINY).run(stop_after=1)) > 0:
        executed.append(n)
    assert executed == [1] * 6
    assert manifest(tmp_path) == manifest(reference)


def test_ctrl_c_after_each_step_resumes_without_redoing_a_completed_step(tmp_path, reference, monkeypatch):
    """A KeyboardInterrupt right after step n is recorded propagates out of `run`; the rerun runs
    exactly the 6 - n steps left (none of the n done is redone) and ends at the uninterrupted
    run's manifest."""
    real = Loop._save
    for n in range(1, 7):
        d = tmp_path / str(n)
        saves = {"n": 0}

        def interrupting_save(self):
            real(self)
            saves["n"] += 1
            if saves["n"] == n:
                raise KeyboardInterrupt

        monkeypatch.setattr(Loop, "_save", interrupting_save)
        with pytest.raises(KeyboardInterrupt):
            Loop(d, TINY).run()
        monkeypatch.setattr(Loop, "_save", real)
        assert len([s for g in manifest(d)["generations"] for s in g if s != "gen"]) == n
        assert Loop(d, TINY).run() == 6 - n
        assert manifest(d) == manifest(reference)
        assert temps(d) == []


def _dead_pid():
    p = subprocess.Popen([sys.executable, "-c", "pass"])
    p.wait()
    return p.pid


def as_an_earlier_process(monkeypatch):
    """Make this process look like another, since-ended one (its temps carry a dead PID), the way
    a real rerun after Ctrl-C is a new process. Returns its PID and a function that undoes it."""
    dead = _dead_pid()
    monkeypatch.setattr(os, "getpid", lambda: dead)
    return dead, monkeypatch.undo


def interrupting_replace(monkeypatch, at):
    real = os.replace
    calls = {"n": 0}

    def flaky(src, dst):
        calls["n"] += 1
        if calls["n"] == at:
            raise KeyboardInterrupt
        return real(src, dst)

    monkeypatch.setattr(os, "replace", flaky)


@pytest.mark.parametrize("interrupt_at", [2, 3, 5, 9, 17])
def test_an_interrupt_during_a_rename_resumes_to_the_same_manifest(tmp_path, reference, monkeypatch, interrupt_at):
    # os.replace calls in order: 1 gen-000/selfplay.json, 2-3 its chunks, 4 manifest, 5 gen-000/net.bin,
    # 6 nets/gen-000.bin, 7 manifest, 8 nets/best.bin, 9 manifest, ..., 17 gen-001/gate.jsonl.
    dead, restore = as_an_earlier_process(monkeypatch)
    interrupting_replace(monkeypatch, interrupt_at)
    with pytest.raises(KeyboardInterrupt):
        Loop(tmp_path, TINY).run()
    restore()
    left = temps(tmp_path)
    assert left, "the interrupted write left its temp file"
    assert {files.temp_writer(p) for p in left} == {(files.host(), dead)}
    Loop(tmp_path, TINY).run()
    assert manifest(tmp_path) == manifest(reference)
    assert temps(tmp_path) == [], "the rerun deleted the dead run's temps of its own outputs"


def test_a_finished_chunk_is_not_played_again(tmp_path, reference, monkeypatch):
    """Rename 3 is chunk 1's: chunk 0 is finished, and the rerun keeps that very file."""
    _, restore = as_an_earlier_process(monkeypatch)
    interrupting_replace(monkeypatch, 3)
    with pytest.raises(KeyboardInterrupt):
        Loop(tmp_path, TINY).run()
    restore()
    chunk0 = tmp_path / "gen-000" / "selfplay-local-00000.jsonl.gz"
    before = chunk0.stat()
    Loop(tmp_path, TINY).run()
    after = chunk0.stat()
    assert (after.st_ino, after.st_mtime_ns) == (before.st_ino, before.st_mtime_ns)
    assert manifest(tmp_path) == manifest(reference)


def test_an_interrupt_inside_self_play_resumes_to_the_same_manifest(tmp_path, reference, monkeypatch):
    """Ctrl-C inside `write_explore` (in the native call of chunk 1): the chunk's temp holds a
    clean but partial file, and the rerun deletes it and plays the chunk whole."""
    import alphasettler.selfplay as selfplay_mod

    real = selfplay_mod.selfplay_explore
    calls = {"n": 0}

    def flaky(*args):
        calls["n"] += 1
        if calls["n"] == 2:
            raise KeyboardInterrupt
        return real(*args)

    _, restore = as_an_earlier_process(monkeypatch)
    monkeypatch.setattr(selfplay_mod, "selfplay_explore", flaky)
    with pytest.raises(KeyboardInterrupt):
        Loop(tmp_path, TINY).run()
    restore()
    assert [p.name.split(".tmp-")[0] for p in temps(tmp_path)] == [".selfplay-local-00001.jsonl.gz"]
    Loop(tmp_path, TINY).run()
    assert manifest(tmp_path) == manifest(reference)
    assert temps(tmp_path) == []


def test_temp_names_carry_the_host_and_the_pid(monkeypatch):
    monkeypatch.setattr(socket, "gethostname", lambda: "my-box/x.local")
    t = files.temp_path("/d/net.bin")
    assert t.name == f".net.bin.tmp-my-boxx.local-{os.getpid()}"
    assert files.temp_writer(t) == ("my-boxx.local", os.getpid())
    assert files.temp_writer(".a.tmp-h-1-2-3") == ("h-1-2", 3)
    for name in (".a.tmp-123", ".a.tmp-host-x", ".a.tmp--5", "a.tmp-h-1", ".a.tmp"):
        assert files.temp_writer(name) is None, name


def test_only_dead_local_writers_temps_of_loop_outputs_are_removed(tmp_path, capsys):
    """A temp is deleted only if its name is a loop output, its host is this machine and its PID
    is not running (or is this process). Kept: a running writer's, another machine's (a host
    that merely starts with this one's name too), another name (another machine's self-play),
    a hostless or PID-less name."""
    host, live, dead, me = files.host(), os.getppid(), _dead_pid(), os.getpid()
    gdir, nets = tmp_path / "gen-000", tmp_path / "nets"
    gdir.mkdir()
    nets.mkdir()
    kept = [gdir / f".selfplay-local-00000.jsonl.gz.tmp-{host}-{live}",
            gdir / f".selfplay.json.tmp-otherhost-{dead}",
            tmp_path / f".manifest.json.tmp-{host}-2-{dead}",
            gdir / f".20261002-selfplay-x.jsonl.gz.tmp-{host}-{dead}",
            gdir / f".selfplay-local-00001.jsonl.gz.tmp-{dead}",
            gdir / f".selfplay-local-00000.jsonl.gz.tmp-{host}-notapid"]
    gone = [tmp_path / f".manifest.json.tmp-{host}-{dead}",
            *(gdir / f".{name}.tmp-{host}-{dead}" for name in ("selfplay.json", "selfplay-local-00001.jsonl.gz",
                                                               "net.bin", "gate.jsonl", "gate.json")),
            nets / f".gen-000.bin.tmp-{host}-{dead}", nets / f".best.bin.tmp-{host}-{dead}",
            gdir / f".selfplay.json.tmp-{host}-{me}"]
    for p in kept + gone:
        p.write_text("half a file")
    assert Loop(tmp_path, replace(TINY, generations=0)).run(stop_after=1) == 1
    assert sorted(temps(tmp_path)) == sorted(kept)
    out = capsys.readouterr().out
    assert f"notice: deleted {len(gone)} temp files an interrupted run left" in out
    assert f"notice: {len(kept)} temp files of other processes left alone" in out
    assert sorted(p.name for p in gdir.glob("*.jsonl.gz")) == ["selfplay-local-00000.jsonl.gz",
                                                               "selfplay-local-00001.jsonl.gz"]
    assert manifest(tmp_path)["generations"][0]["selfplay"]["games"] == 2


def test_other_settings_on_a_started_self_play_step_are_an_error(tmp_path, monkeypatch):
    real = os.replace
    calls = {"n": 0}

    def flaky(src, dst):
        calls["n"] += 1
        if calls["n"] == 2:
            raise KeyboardInterrupt
        return real(src, dst)

    monkeypatch.setattr(os, "replace", flaky)
    with pytest.raises(KeyboardInterrupt):
        Loop(tmp_path, TINY).run()
    monkeypatch.setattr(os, "replace", real)
    with pytest.raises(ValueError, match="other settings"):
        Loop(tmp_path, replace(TINY, full_simulations=6)).run()


def test_new_flags_apply_only_to_steps_not_yet_run(tmp_path, reference):
    Loop(tmp_path, replace(TINY, generations=0)).run()
    Loop(tmp_path, replace(TINY, lr=1e-4)).run()
    m, ref = manifest(tmp_path), manifest(reference)
    assert m["generations"][0] == ref["generations"][0]
    assert m["generations"][1]["train"]["lr"] == 1e-4


@pytest.mark.parametrize("change", [{"generations": -1}, {"generations": 100}, {"lr": 0.0}, {"warm_lr": math.nan}, {"games": 0},
                                    {"window": 0}, {"gate_seeds": 0}, {"gate_simulations": 0}, {"device": "gpu"},
                                    {"warm_games": 10_000_001}, {"width": 0}, {"rounds": 0},
                                    {"width": NET_MAX_WIDTH + 1}, {"rounds": NET_MAX_ROUNDS + 1}, {"width": 2.5}])
def test_bad_settings_are_an_error_before_anything_is_written(tmp_path, change):
    with pytest.raises(ValueError):
        Loop(tmp_path / "run", replace(TINY, **change))
    assert not (tmp_path / "run").exists()


def test_the_warm_start_is_100000_games_inside_generation_0s_seed_range(tmp_path):
    from alphasettler.cli import _parser
    from alphasettler.train.loop import GATE_SEED_START, SEEDS_PER_GENERATION

    assert LoopSettings(generations=0).warm_games == 100_000
    assert _parser().parse_args(["train-loop", "--dir", "x", "--generations", "0"]).warm_games == 100_000
    assert 100_000 <= SEEDS_PER_GENERATION < GATE_SEED_START
    Loop(tmp_path / "run", LoopSettings(generations=0))  # the defaults pass every check
    assert not (tmp_path / "run").exists()


def test_self_play_seeds_stay_below_the_gate_seeds(tmp_path):
    from alphasettler.train.loop import GATE_SEED_START, MAX_GENERATIONS, SEEDS_PER_GENERATION

    assert MAX_GENERATIONS == 99
    assert MAX_GENERATIONS * SEEDS_PER_GENERATION + SEEDS_PER_GENERATION <= GATE_SEED_START
    Loop(tmp_path / "run", replace(TINY, generations=99, games=SEEDS_PER_GENERATION))
    with pytest.raises(ValueError, match="--generations must be 0..=99"):
        Loop(tmp_path / "run", replace(TINY, generations=100))
    assert not (tmp_path / "run").exists()


def test_best_bin_follows_the_manifest_after_an_unsaved_promotion(tmp_path, monkeypatch):
    """A promoting gate interrupted after the best.bin rename but before the manifest is saved:
    best.bin holds the new net while the manifest's best is the old one. A rerun puts it back,
    whether it stops short of the gate or replays the gate and this time does not promote."""
    import alphasettler.train.loop as loop_mod

    assert Loop(tmp_path, TINY).run(stop_after=5) == 5
    nets = tmp_path / "nets"
    real_save = Loop._save

    def unsaved_promotion():
        def interrupt(self):
            raise KeyboardInterrupt

        monkeypatch.setattr(loop_mod, "promoted", lambda wins, games: True)
        monkeypatch.setattr(Loop, "_save", interrupt)
        with pytest.raises(KeyboardInterrupt):
            Loop(tmp_path, TINY).run()
        monkeypatch.setattr(Loop, "_save", real_save)
        assert (nets / "best.bin").read_bytes() == (nets / "gen-001.bin").read_bytes()
        assert manifest(tmp_path)["best"] == 0

    unsaved_promotion()
    assert Loop(tmp_path, replace(TINY, generations=0)).run() == 0
    assert (nets / "best.bin").read_bytes() == (nets / "gen-000.bin").read_bytes()

    unsaved_promotion()
    monkeypatch.setattr(loop_mod, "promoted", lambda wins, games: False)
    assert Loop(tmp_path, TINY).run() == 1
    m = manifest(tmp_path)
    assert m["best"] == 0 and m["generations"][1]["gate"]["promoted"] is False
    assert (nets / "best.bin").read_bytes() == (nets / "gen-000.bin").read_bytes()


def test_a_manifest_of_another_format_is_refused_before_anything_is_written(tmp_path):
    """A format 1 run directory (the flat MLP's, such as runs/3b) is not resumed by network v2."""
    (tmp_path / "manifest.json").write_text(json.dumps({"format": 1, "best": 0, "generations": []}))
    with pytest.raises(ValueError, match="has format 1, expected 2: a format 1 run trained the flat MLP"):
        Loop(tmp_path, TINY)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["manifest.json"]


def test_linked_records_satisfy_a_new_runs_self_play_without_playing(tmp_path, reference, monkeypatch):
    """Plan 5b's warm-start retrain: generation 0's plan and chunks hard-linked into a new run
    directory complete that run's self-play step without a game, with the same manifest entry."""
    import alphasettler.selfplay as selfplay_mod

    gen = tmp_path / "run" / "gen-000"
    gen.mkdir(parents=True)
    for p in (reference / "gen-000").iterdir():
        if p.name == "selfplay.json" or p.name.startswith("selfplay-local-"):
            os.link(p, gen / p.name)

    def no_games(*args, **kwargs):
        raise AssertionError("a game was played")

    monkeypatch.setattr(selfplay_mod, "write_explore", no_games)
    assert Loop(tmp_path / "run", TINY).run(stop_after=1) == 1
    assert manifest(tmp_path / "run")["generations"][0]["selfplay"] == manifest(reference)["generations"][0]["selfplay"]


def test_a_run_whose_generation_0_was_trained_before_the_newest_share_resumes(tmp_path, reference):
    """runs/3b-v2's state when the fix landed: generation 0 complete with a train entry that has
    no `newest_share` or `older_positions`, generation 1's self-play part way. The rerun finishes
    generation 1 as an uninterrupted run does and leaves generation 0's entry as it was."""
    d = tmp_path / "run"
    shutil.copytree(reference, d)
    m = manifest(reference)
    g0 = {**m["generations"][0], "train": {k: v for k, v in m["generations"][0]["train"].items()
                                          if k not in ("newest_share", "older_positions")}}
    (d / "manifest.json").write_text(json.dumps({"format": 2, "best": 0, "generations": [g0]}, indent=1))
    for name in ("selfplay-local-00001.jsonl.gz", "net.bin", "gate.jsonl", "gate.json"):
        (d / "gen-001" / name).unlink()
    (d / "nets" / "gen-001.bin").unlink()
    assert Loop(d, TINY).run() == 3
    resumed = manifest(d)
    assert resumed["generations"] == [g0, m["generations"][1]] and resumed["best"] == m["best"]


def test_a_changed_output_of_a_completed_step_is_an_error(tmp_path):
    """Generation 1's training reads generation 0's net.bin and both generations' chunks; each
    is checked against the manifest's sha256 first, and a mismatch names the file."""
    assert Loop(tmp_path, TINY).run(stop_after=4) == 4  # through generation 1's self-play
    for path in (tmp_path / "gen-000" / "net.bin", tmp_path / "gen-000" / "selfplay-local-00001.jsonl.gz",
                 tmp_path / "gen-001" / "selfplay-local-00000.jsonl.gz"):
        good = path.read_bytes()
        path.write_bytes(good + b"x")
        with pytest.raises(ValueError, match=rf"{path.parent.name}/{path.name} has sha256 [0-9a-f]{{64}}, not the"):
            Loop(tmp_path, TINY).run()
        path.write_bytes(good)
    best = tmp_path / "nets" / "gen-000.bin"
    good = best.read_bytes()
    best.unlink()
    with pytest.raises(ValueError, match="gen-000.bin is missing"):
        Loop(tmp_path, TINY).run()
    best.write_bytes(good)
    assert Loop(tmp_path, TINY).run() == 2


def cli(*args):
    return subprocess.run([sys.executable, "-m", "alphasettler", *args], capture_output=True, text=True)


def test_cli_train_loop(tmp_path):
    args = ["train-loop", "--dir", str(tmp_path), "--generations", "0", "--warm-games", "2", "--chunk-games", "1",
            "--full-simulations", "4", "--fast-simulations", "2", "--batch", "16", "--width", "8",
            "--rounds", "1", "--max-turns", "60", "--device", "cpu", "--threads", "2"]
    r = cli(*args)
    assert r.returncode == 0, r.stderr
    assert "3 steps run" in r.stdout and "best net: gen-000" in r.stdout
    assert (tmp_path / "nets" / "best.bin").exists()
    r = cli(*args)
    assert r.returncode == 0 and "nothing to do" in r.stdout
    r = cli("train-loop", "--dir", str(tmp_path / "x"), "--generations", "-1")
    assert r.returncode == 2 and "--generations" in r.stderr
    r = cli("train-loop", "--dir", str(tmp_path / "y"), "--generations", "0", "--full-probability", "2")
    assert r.returncode == 2 and "probability" in r.stderr
    assert not (tmp_path / "y").exists()


def test_cli_train_loop_without_pytorch_says_how_to_install_it(tmp_path):
    code = ("import sys; sys.modules['torch'] = None\n"  # what `import torch` does when it is not installed
            "from alphasettler.cli import main\n"
            f"sys.exit(main(['train-loop', '--dir', {str(tmp_path / 'run')!r}, '--generations', '0']))")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 2, r.stderr
    assert "PyTorch (the train extra) is not installed" in r.stderr and "Traceback" not in r.stderr
    assert not (tmp_path / "run").exists()


def test_cli_train_loop_ctrl_c_exits_130_and_the_rerun_resumes(tmp_path, monkeypatch, capsys):
    from alphasettler.cli import main

    args = ["train-loop", "--dir", str(tmp_path), "--generations", "0", "--warm-games", "2", "--chunk-games", "1",
            "--full-simulations", "4", "--fast-simulations", "2", "--batch", "16", "--width", "8",
            "--rounds", "1", "--max-turns", "60", "--device", "cpu", "--threads", "2"]
    real = Loop._save

    def interrupting_save(self):
        real(self)
        raise KeyboardInterrupt

    monkeypatch.setattr(Loop, "_save", interrupting_save)
    assert main(args) == 130
    assert "rerun the same command to resume" in capsys.readouterr().err
    monkeypatch.setattr(Loop, "_save", real)
    assert main(args) == 0
    assert "2 steps run" in capsys.readouterr().out
