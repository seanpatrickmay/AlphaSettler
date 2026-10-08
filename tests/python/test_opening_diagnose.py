"""The opening diagnosis (spec 3d-1 Section 3): groups, D1 to D4, the verdict and `opening-diagnose`."""

import math
import subprocess
import sys

import pytest

from alphasettler import Game
from alphasettler._engine import OPENING_FILE, write_random_net
from alphasettler.arena import TRADES
from alphasettler.diagnose import (
    GROUPS, Diagnosis, calibration, d1_table, d2_table, d4_table, gap, group, opening_pips, play_setups, verdict,
)
from alphasettler.opening import model_bytes, replay_setup
from opening_games import fixed_game, write_dir

BOARD = Game(0, None).snapshot()["board"]


def cli(*args):
    return subprocess.run([sys.executable, "-m", "alphasettler", *args], capture_output=True, text=True)


@pytest.fixture
def nets(tmp_path):
    """`tiny.bin` and an `opening.bin` that scores every spot the same."""
    d = tmp_path / "nets"
    d.mkdir()
    write_random_net(str(d / "tiny.bin"), 1, 8, 1)
    (d / OPENING_FILE).write_bytes(model_bytes([(4, 78, [0.0] * (4 * 78), [0.0] * 4)]))
    return d


@pytest.mark.parametrize("nodes,want", [
    ([9, 29], "all five"),  # wheat, wheat, brick + wood, ore, sheep
    ([4, 12], "no wood or no brick"),  # ore, wheat + ore, brick, sheep
    ([31, 41], "no ore"),  # brick, sheep, wheat + wood, wheat, sheep
    ([19, 26], "no wheat"),  # brick, wood, brick + sheep: no ore either, and no wheat comes first
    ([0, 4], "no wood and brick"),  # ore + ore, wheat
    ([53, 50], "no wood and brick"),  # sheep only: the first group that applies
    ([9, 30], "no sheep"),  # wheat, wheat, brick + ore, brick, wood
])
def test_groups_are_the_first_that_applies(nodes, want):
    assert group(BOARD, nodes) == want


def test_opening_pips_sum_over_the_settlements():
    # Node 31: brick 5 (4), sheep 4 (3), wheat 9 (4); node 41: wood 2 (1), wheat 9 (4), sheep 6 (5).
    assert opening_pips(BOARD, [31, 41]) == 21


def test_d1_on_the_fixed_game_matches_a_hand_count():
    s = replay_setup(fixed_game())
    assert [group(s.board, s.nodes(p)) for p in range(4)] == [
        "all five", "no wood or no brick", "all five", "no ore"]
    # Seat 3 (no ore) took node 41 (10 pips) when the best legal ore spot was node 44 (sheep 6, ore 3,
    # wheat 3: 9), then node 31 (11) with 44 still the best: gaps 1 and 2, so within 1 at either pick.
    assert gap(s, 3) == 1
    d = Diagnosis(None)
    d.add(s)
    table = d1_table(d.gaps)
    assert "| no ore | 1 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 |" in table
    assert "| no wheat | 0 | - | - | - | - | - |" in table
    assert "| weak groups | 1 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 |" in table


def test_d1_shares_are_cumulative_and_count_openings_with_no_alternative():
    table = d1_table({"no wood and brick": [None, 3], "no wheat": [-2, 0, 1, 2], "no ore": []})
    assert "| no wood and brick | 2 | 0.000 | 0.000 | 0.000 | 0.500 | 0.500 |" in table
    assert "| no wheat | 4 | 0.500 | 0.750 | 1.000 | 1.000 | 0.000 |" in table
    assert "| no ore | 0 | - | - | - | - | - |" in table
    assert "| weak groups | 6 | 0.333 | 0.500 | 0.667 | 0.833 | 0.167 |" in table


def test_d2_reads_each_seat_s_own_view_after_the_setup_and_skips_games_without_a_winner():
    seen = []

    def value(g, seat):
        seen.append((g.phase, seat))
        return [0.1, 0.2, 0.3, 0.4]

    d = Diagnosis(value)
    d.add(replay_setup(fixed_game(0, winner=3)))
    d.add(replay_setup(fixed_game(1, winner=None)))
    assert seen == [("pre_roll", seat) for seat in range(4)]
    assert d.calibration["no ore"] == [(0.4, True)]
    assert sorted(d.calibration["all five"]) == [(0.1, False), (0.3, False)]
    assert d.calibration["no wood or no brick"] == [(0.2, False)]
    assert (d.games, d.no_winner, d.seats) == (2, 1, 8)


def test_calibration_gap_and_standard_error():
    n, predicted, actual, g, se = calibration([(0.5, True), (0.5, False), (0.3, False), (0.3, False)])
    # Gaps -0.5, 0.5, 0.3, 0.3: mean 0.15; squared deviations sum to 0.59 over n - 1 = 3.
    assert (n, predicted, actual) == (4, pytest.approx(0.4), 0.25)
    assert g == pytest.approx(0.15) and se == pytest.approx(math.sqrt(0.59 / 3 / 4))


def test_tables_print_empty_groups_as_dashes():
    d = Diagnosis(lambda g, seat: [0.25] * 4)
    text = d2_table(d.calibration)
    for g in GROUPS:
        assert f"| {g} | 0 | - | - | - | - |" in text
    assert "| weak groups | 0 | - | - | - | - |" in text
    assert verdict(d.gaps, d.calibration).startswith("verdict: undecided")
    assert calibration([(0.3, True)])[4] == math.inf


ALL_FIVE = [(0.25, False)] * 30 + [(0.25, True)] * 10


@pytest.mark.parametrize("weak_gaps,weak_pairs,word", [
    # A spot within 1 pip in 1 of 6 weak openings, under a third: forced, whatever D2 says.
    ([2, 5, None, 1, 3, 4], [(0.25, False)] * 4, "forced"),
    # Alternatives were near; weak openings predicted 0.30 but won 4 of 44 (gap +0.21 against all
    # five's 0, about 2.5 standard errors of the difference): misjudged.
    ([0, 1, 1], [(0.3, False)] * 40 + [(0.3, True)] * 4, "misjudged"),
    # Alternatives were near and the weak groups are calibrated like all five: misdecided.
    ([0, 1, 1], ALL_FIVE, "misdecided"),
])
def test_the_verdict_rule(weak_gaps, weak_pairs, word):
    gaps = {"no wood and brick": [], "no wheat": weak_gaps, "no ore": []}
    cal = {g: [] for g in GROUPS}
    cal["no wheat"] = weak_pairs
    cal["all five"] = ALL_FIVE
    assert verdict(gaps, cal).startswith(f"verdict: {word} (")


def test_d4_plays_only_the_setup_and_does_not_depend_on_threads(nets):
    one = play_setups("net:tiny@4", 2_000_000_000, 4, TRADES, str(nets), threads=1)
    three = play_setups("net:tiny@4", 2_000_000_000, 4, TRADES, str(nets), threads=3)
    assert [s.picks for s in one] == [s.picks for s in three]
    assert all(s.after.phase == "pre_roll" and len(s.picks) == 8 for s in one)
    text = d4_table("net:tiny@4", one)
    assert "D4, net:tiny@4: the openings of 4 setups (16 seats)" in text and "| weak groups |" in text
    # A +o bot whose model scores every spot the same takes the first legal spot at every pick.
    tied = play_setups("net:tiny@4+o", 2_000_000_000, 1, TRADES, str(nets), threads=1)[0]
    assert all(p.node == p.legal[0] for p in tied.picks)


def test_cli_prints_d1_to_d3_and_the_verdict_and_writes_out(tmp_path, nets):
    d = write_dir(tmp_path / "gen", [fixed_game(0), fixed_game(1, winner=0)])
    out = tmp_path / "diagnosis.md"
    r = cli("opening-diagnose", "--games", str(d), "--net", "tiny", "--nets-dir", str(nets), "--audit", "1",
            "--audit-budgets", "2", "3", "--threads", "2", "--out", str(out))
    assert r.returncode == 0, r.stderr
    for heading in ("2 games (0 without a winner)", "D1, forced or chosen", "D2, misjudged", "D3, search budget",
                    "\nverdict: "):
        assert heading in r.stdout
    # One audited game: 8 picks at each budget, after the recorded row.
    assert "| recorded | 8 | - |" in r.stdout and "| 2 | 8 |" in r.stdout and "| 3 | 8 |" in r.stdout
    assert out.read_text() == r.stdout
    r = cli("opening-diagnose", "--bot", "net:tiny@3", "--seeds", "2", "--trades", "--nets-dir", str(nets))
    assert r.returncode == 0, r.stderr
    assert "D4, net:tiny@3: the openings of 2 setups (8 seats)" in r.stdout


@pytest.mark.parametrize("args,message", [
    (["--games", "DIR"], "--games needs --net"),
    (["--games", "DIR", "--net", "missing"], "cannot read"),
    (["--games", "DIR", "--net", "tiny", "--audit", "-1"], "--audit must be 0 or more"),
    (["--bot", "net:tiny@3"], "--bot needs --seeds"),
    (["--bot", "net:tiny@3+o+o", "--seeds", "2"], "unknown bot"),
])
def test_cli_refuses_bad_arguments_before_reading_or_playing(tmp_path, nets, args, message):
    d = write_dir(tmp_path / "gen", [fixed_game()])
    r = cli("opening-diagnose", *[str(d) if a == "DIR" else a for a in args], "--nets-dir", str(nets))
    assert r.returncode == 2 and message in r.stderr, r.stderr


def test_the_diagnosis_does_not_import_torch():
    r = subprocess.run([sys.executable, "-c",
                        "import sys, alphasettler.diagnose, alphasettler.opening; print('torch' in sys.modules)"],
                       capture_output=True, text=True)
    assert r.stdout.strip() == "False", r.stderr
