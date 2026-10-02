import json
import re
import subprocess
import sys

import pytest


def cli(*args, cwd=None):
    return subprocess.run([sys.executable, "-m", "alphasettler", *args], capture_output=True, text=True, cwd=cwd)


def test_cli_arena(tmp_path):
    out = tmp_path / "g.jsonl"
    r = cli("arena", "--candidate", "greedy", "--baseline", "random", "--seeds", "4", "--out", str(out))
    assert r.returncode == 0, r.stderr
    assert "greedy vs random: win rate" in r.stdout
    assert f"wrote {out}" in r.stdout
    assert len(out.read_text().splitlines()) == 16


def test_cli_native_arena_reports_zero_belief_resets(tmp_path):
    out = tmp_path / "i.jsonl"
    r = cli("arena", "--candidate", "ismcts@10", "--baseline", "greedy", "--seeds", "1", "--no-trades",
            "--out", str(out))
    assert r.returncode == 0, r.stderr
    assert "belief resets: 0" in r.stdout
    recs = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(recs) == 4
    assert all(rec["belief_resets"] == 0 for rec in recs)


def test_cli_arena_default_output_goes_to_runs(tmp_path):
    r = cli("arena", "--candidate", "random", "--baseline", "random", "--seeds", "2", "--no-trades", cwd=tmp_path)
    assert r.returncode == 0, r.stderr
    files = list((tmp_path / "runs").glob("*-random-vs-random.jsonl"))
    assert len(files) == 1
    assert re.fullmatch(r"\d{8}-\d{6}-\d{6}-random-vs-random\.jsonl", files[0].name), files[0].name


def test_cli_compare(tmp_path):
    r = cli("compare", "--a", "greedy", "--b", "random", "--baseline", "random", "--seeds", "6",
            "--out-dir", str(tmp_path))
    assert r.returncode == 0, r.stderr
    assert "greedy vs random: win rate" in r.stdout
    assert "random vs random: win rate" in r.stdout
    assert "paired greedy - random:" in r.stdout
    assert len(list(tmp_path.glob("*-a-greedy-vs-random.jsonl"))) == 1
    assert len(list(tmp_path.glob("*-b-random-vs-random.jsonl"))) == 1
    assert len(list(tmp_path.glob("*.jsonl"))) == 2


def test_cli_compare_same_bot_on_both_sides(tmp_path):
    r = cli("compare", "--a", "greedy", "--b", "greedy", "--baseline", "random", "--seeds", "5",
            "--out-dir", str(tmp_path))
    assert r.returncode == 0, r.stderr
    assert len(list(tmp_path.glob("*-a-greedy-vs-random.jsonl"))) == 1
    assert len(list(tmp_path.glob("*-b-greedy-vs-random.jsonl"))) == 1
    assert "paired greedy - greedy: mean diff +0.000 ± 0.000" in r.stdout


def test_cli_compare_validates_both_bots_before_running(tmp_path):
    r = cli("compare", "--a", "greedy", "--b", "nope", "--baseline", "random", "--seeds", "3",
            "--out-dir", str(tmp_path / "out"))
    assert r.returncode == 2
    assert "unknown bot 'nope'" in r.stderr
    assert r.stdout == ""
    assert not (tmp_path / "out").exists()


def test_cli_rejects_unknown_bot_and_zero_seeds():
    r = cli("arena", "--candidate", "nope", "--baseline", "random", "--seeds", "2")
    assert r.returncode == 2
    assert "unknown bot" in r.stderr
    r = cli("arena", "--candidate", "random", "--baseline", "random", "--seeds", "0")
    assert r.returncode == 2
    assert "--seeds" in r.stderr


def test_cli_rejects_unknown_baseline(tmp_path):
    r = cli("arena", "--candidate", "random", "--baseline", "nope", "--seeds", "2", cwd=tmp_path)
    assert r.returncode == 2
    assert "unknown bot" in r.stderr and "nope" in r.stderr
    assert not (tmp_path / "runs").exists()
    r = cli("compare", "--a", "random", "--b", "greedy", "--baseline", "nope", "--seeds", "2", cwd=tmp_path)
    assert r.returncode == 2
    assert "unknown bot 'nope'" in r.stderr
    assert not (tmp_path / "runs").exists()


def test_cli_rejects_a_seed_count_over_the_cap(tmp_path):
    r = cli("arena", "--candidate", "random", "--baseline", "random", "--seeds", "2305843009213693952", cwd=tmp_path)
    assert r.returncode == 2, r.stderr
    assert "at most" in r.stderr
    assert not (tmp_path / "runs").exists()


def test_cli_out_naming_a_directory_fails_before_playing(tmp_path):
    r = cli("arena", "--candidate", "random", "--baseline", "random", "--seeds", "2", "--out", str(tmp_path))
    assert r.returncode == 2
    assert r.stderr.startswith("error: ") and str(tmp_path) in r.stderr
    assert r.stdout == ""


def test_cli_never_overwrites_an_existing_output(tmp_path):
    out = tmp_path / "g.jsonl"
    out.write_text("keep\n")
    r = cli("arena", "--candidate", "random", "--baseline", "random", "--seeds", "2", "--out", str(out))
    assert r.returncode == 2
    assert "error: " in r.stderr and "exists" in r.stderr
    assert out.read_text() == "keep\n"


@pytest.mark.parametrize("command", [
    ["oracle-diff", "--games", "1"],
    ["arena", "--candidate", "greedy", "--baseline", "catanatron:random", "--seeds", "1"],
], ids=["oracle-diff", "arena"])
def test_cli_without_catanatron_explains_how_to_install(tmp_path, command):
    # Hide Catanatron from a child interpreter by shadowing it with a package that fails to import.
    fake = tmp_path / "catanatron"
    fake.mkdir()
    (fake / "__init__.py").write_text(
        "raise ModuleNotFoundError(\"No module named 'catanatron'\", name=\"catanatron\")\n")
    env = {**__import__("os").environ, "PYTHONPATH": str(tmp_path)}
    r = subprocess.run([sys.executable, "-m", "alphasettler", *command],
                       capture_output=True, text=True, env=env, cwd=tmp_path)
    assert r.returncode == 2
    assert "not installed" in r.stderr and "pip3 install" in r.stderr


def test_cli_surfaces_a_broken_catanatron_install(tmp_path):
    # Catanatron is present but one of its own imports fails: that is a bug to show, not "missing".
    fake = tmp_path / "catanatron"
    fake.mkdir()
    (fake / "__init__.py").write_text("import networkx_not_here\n")
    env = {**__import__("os").environ, "PYTHONPATH": str(tmp_path)}
    r = subprocess.run([sys.executable, "-m", "alphasettler", "oracle-diff", "--games", "1"],
                       capture_output=True, text=True, env=env, cwd=tmp_path)
    assert r.returncode != 0
    assert "not installed" not in r.stderr
    assert "No module named 'networkx_not_here'" in r.stderr


def test_cli_selfplay_and_fit(tmp_path):
    r = cli("selfplay", "--games", "3", "--simulations", "10", "--out-dir", str(tmp_path))
    assert r.returncode == 0, r.stderr
    files = list(tmp_path.glob("*-selfplay-s10.jsonl.gz"))
    assert len(files) == 1
    assert "3 games" in r.stdout
    assert "belief resets: 0" in r.stdout
    f = cli("fit-heuristic", str(files[0]))
    assert f.returncode == 0, f.stderr
    assert "pub const DEFAULT_WEIGHTS: [f32; NUM_FEATURES] = [" in f.stdout
    assert "log-likelihood per sample" in f.stdout


def test_cli_compare_accepts_ismcts_names(tmp_path):
    r = cli("compare", "--a", "ismcts@5", "--b", "greedy", "--baseline", "random", "--seeds", "1",
            "--no-trades", "--out-dir", str(tmp_path))
    assert r.returncode == 0, r.stderr
    assert r.stdout.count("belief resets: 0") == 2
    bad = cli("compare", "--a", "ismcts@0", "--b", "greedy", "--baseline", "random", "--seeds", "1")
    assert bad.returncode == 2 and "unknown bot" in bad.stderr


def test_cli_selfplay_bad_out_dir_fails_before_playing(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("not a directory")
    # Far more work than the timeout allows: playing before opening the output fails the test.
    r = subprocess.run([sys.executable, "-m", "alphasettler", "selfplay", "--games", "100000", "--simulations", "1000",
                        "--out-dir", str(blocker / "runs")], capture_output=True, text=True, timeout=60)
    assert r.returncode == 2
    assert r.stderr.startswith("error: ") and str(blocker) in r.stderr
    assert r.stdout == ""


def test_cli_selfplay_checks_arguments_before_creating_a_file(tmp_path):
    for bad in (["--simulations", "1"], ["--simulations", "1000001"], ["--simulations", "5", "--rollout", "201"],
                ["--simulations", "5", "--rollout", "-1"]):
        r = cli("selfplay", "--games", "1", *bad, "--out-dir", str(tmp_path))
        assert r.returncode == 2, bad
        assert "error" in r.stderr, bad
    assert list(tmp_path.iterdir()) == []


def test_cli_fit_heuristic_rejects_a_negative_l2(tmp_path):
    r = cli("selfplay", "--games", "1", "--simulations", "5", "--out-dir", str(tmp_path))
    assert r.returncode == 0, r.stderr
    (path,) = tmp_path.glob("*.jsonl.gz")
    for l2 in ("-0.1", "nan"):
        f = cli("fit-heuristic", str(path), "--l2", l2)
        assert f.returncode == 2 and "l2" in f.stderr, f.stderr


def test_cli_selfplay_streams_batches_and_replays(tmp_path, monkeypatch, capsys):
    from alphasettler import cli as cli_module
    from alphasettler import records

    monkeypatch.setattr(cli_module, "SELFPLAY_BATCH", 2)
    rc = cli_module.main(["selfplay", "--games", "5", "--simulations", "10", "--threads", "2",
                          "--out-dir", str(tmp_path)])
    assert rc == 0
    assert "5 games" in capsys.readouterr().out
    (path,) = tmp_path.glob("*-selfplay-s10.jsonl.gz")
    games = list(records.read(path))
    assert [g["seed"] for g in games] == [0, 1, 2, 3, 4]
    for g in games:
        assert g["config"]["vp_to_win"] == 10 and g["config"]["max_offers_per_turn"] == 0
        assert records.replay_mismatches(g) == []


def test_cli_selfplay_interrupt_keeps_written_batches(tmp_path, monkeypatch, capsys):
    import alphasettler._engine as engine
    from alphasettler import cli as cli_module
    from alphasettler import records

    real = engine.selfplay
    calls = []

    def flaky(*args):
        calls.append(args)
        if len(calls) == 2:
            raise KeyboardInterrupt
        return real(*args)

    monkeypatch.setattr(cli_module, "SELFPLAY_BATCH", 2)
    monkeypatch.setattr(engine, "selfplay", flaky)
    rc = cli_module.main(["selfplay", "--games", "6", "--simulations", "10", "--threads", "2",
                          "--out-dir", str(tmp_path)])
    assert rc == 130
    assert "interrupted" in capsys.readouterr().err
    (path,) = tmp_path.glob("*-selfplay-s10.jsonl.gz")
    games = list(records.read(path))
    assert [g["seed"] for g in games] == [0, 1]
    for g in games:
        assert records.replay_mismatches(g) == []
