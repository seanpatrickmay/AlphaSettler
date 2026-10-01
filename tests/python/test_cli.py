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
