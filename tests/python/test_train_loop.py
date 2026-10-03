"""The generation loop (`train-loop`): manifest, atomic outputs, resume and gating.

These tests need the `train` extra (PyTorch: `.venv/bin/pip3 install -e '.[train]'`). Without it
they fail when the loop trains, on purpose: they must be run, never skipped."""

import hashlib
import json
import math
import os
import subprocess
import sys
from dataclasses import replace

import pytest

from alphasettler import files
from alphasettler.train.loop import Loop, LoopSettings, json_float, promoted

TINY = LoopSettings(generations=1, games=2, warm_games=2, chunk_games=1, full_simulations=4, fast_simulations=2,
                    full_probability=0.5, batch=16, trunk=16, value=8, gate_seeds=1, gate_simulations=4,
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
    assert g1["train"]["window"] == ["gen-000", "gen-001"]
    assert g1["train"]["parity_max_error"] <= 1e-5
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


@pytest.mark.parametrize("interrupt_at", [2, 5, 9, 17])
def test_an_interrupt_during_a_rename_resumes_to_the_same_manifest(tmp_path, reference, monkeypatch, interrupt_at):
    # os.replace calls in order: 1 gen-000/selfplay.json, 2-3 its chunks, 4 manifest, 5 gen-000/net.bin,
    # 6 nets/gen-000.bin, 7 manifest, 8 nets/best.bin, 9 manifest, ..., 17 gen-001/gate.jsonl.
    real = os.replace
    calls = {"n": 0}

    def flaky(src, dst):
        calls["n"] += 1
        if calls["n"] == interrupt_at:
            raise KeyboardInterrupt
        return real(src, dst)

    monkeypatch.setattr(os, "replace", flaky)
    with pytest.raises(KeyboardInterrupt):
        Loop(tmp_path, TINY).run()
    monkeypatch.setattr(os, "replace", real)
    assert temps(tmp_path), "the interrupted write left its temp file"
    Loop(tmp_path, TINY).run()
    assert manifest(tmp_path) == manifest(reference)
    assert temps(tmp_path) == []


def _dead_pid():
    p = subprocess.Popen([sys.executable, "-c", "pass"])
    p.wait()
    return p.pid


def test_only_this_process_temps_are_removed(tmp_path, capsys):
    """Another process's temp file is never deleted, live writer or dead: it may be another
    machine's self-play writing into the generation directory (its PID means nothing here). The
    rerun names them in a notice and never reads them."""
    live, dead = os.getppid(), _dead_pid()
    gdir = tmp_path / "gen-000"
    gdir.mkdir()
    kept = [gdir / f".selfplay-local-00000.jsonl.gz.tmp-{live}", gdir / f".selfplay-local-00001.jsonl.gz.tmp-{dead}",
            gdir / f".20261002-selfplay-x.jsonl.gz.tmp-{dead}", tmp_path / f".manifest.json.tmp-{dead}",
            gdir / f".selfplay.json.tmp-{dead}", gdir / ".selfplay-local-00000.jsonl.gz.tmp-notapid"]
    gone = [gdir / f".selfplay.json.tmp-{os.getpid()}", gdir / f".selfplay-local-00001.jsonl.gz.tmp-{os.getpid()}",
            tmp_path / f".manifest.json.tmp-{os.getpid()}"]
    for p in kept + gone:
        p.write_text("half a file")
    assert Loop(tmp_path, replace(TINY, generations=0)).run(stop_after=1) == 1
    assert sorted(temps(tmp_path)) == sorted(kept)
    assert f"notice: {len(kept)} temp files of other processes left alone" in capsys.readouterr().out
    assert sorted(p.name for p in gdir.glob("*.jsonl.gz")) == ["selfplay-local-00000.jsonl.gz",
                                                               "selfplay-local-00001.jsonl.gz"]
    assert manifest(tmp_path)["generations"][0]["selfplay"]["games"] == 2
    files.remove_own_temps(gdir, "selfplay-local-*")
    assert sorted(temps(tmp_path)) == sorted(kept)


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


@pytest.mark.parametrize("change", [{"generations": -1}, {"lr": 0.0}, {"warm_lr": math.nan}, {"games": 0},
                                    {"window": 0}, {"gate_seeds": 0}, {"gate_simulations": 0}, {"device": "gpu"},
                                    {"warm_games": 10_000_001}])
def test_bad_settings_are_an_error_before_anything_is_written(tmp_path, change):
    with pytest.raises(ValueError):
        Loop(tmp_path / "run", replace(TINY, **change))
    assert not (tmp_path / "run").exists()


def cli(*args):
    return subprocess.run([sys.executable, "-m", "alphasettler", *args], capture_output=True, text=True)


def test_cli_train_loop(tmp_path):
    args = ["train-loop", "--dir", str(tmp_path), "--generations", "0", "--warm-games", "2", "--chunk-games", "1",
            "--full-simulations", "4", "--fast-simulations", "2", "--batch", "16", "--trunk", "16",
            "--value-width", "8", "--max-turns", "60", "--device", "cpu", "--threads", "2"]
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
            "--full-simulations", "4", "--fast-simulations", "2", "--batch", "16", "--trunk", "16",
            "--value-width", "8", "--max-turns", "60", "--device", "cpu", "--threads", "2"]
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
