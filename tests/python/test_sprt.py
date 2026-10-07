"""The sequential test (alphasettler/sprt.py): the log-likelihood ratio, the bounds, the decision and
a short native run."""

import math

import pytest

from alphasettler.sprt import MIN_SEEDS, bounds, decide, llr, run_sprt


def test_the_bounds_are_walds():
    lo, hi = bounds(0.05, 0.05)
    assert lo == pytest.approx(math.log(0.05 / 0.95)) and hi == pytest.approx(math.log(0.95 / 0.05))


def test_the_llr_favours_the_hypothesis_nearer_the_mean():
    strong = [0.5, 0.25, 0.5, 0.75, 0.25] * 4  # mean 0.45
    null = [0.25, 0.0, 0.5, 0.25, 0.25] * 4  # mean 0.25
    assert llr(strong, 0.25, 0.30) > 0
    assert llr(null, 0.25, 0.30) < 0
    assert llr([0.275] * 3 + [0.25, 0.30], 0.25, 0.30) == pytest.approx(0.0, abs=1e-9), "midway is neutral"


def test_zero_variance_is_decided_by_the_side_of_the_midpoint():
    assert llr([0.5] * 20, 0.25, 0.30) == math.inf
    assert llr([0.0] * 20, 0.25, 0.30) == -math.inf


def test_decisions_wait_for_enough_seeds_then_stop_at_a_bound():
    assert decide([1.0] * (MIN_SEEDS - 1), 0.25, 0.30, 0.05, 0.05) is None
    assert decide([0.75, 1.0] * MIN_SEEDS, 0.25, 0.30, 0.05, 0.05) == "H1"
    assert decide([0.0, 0.25] * MIN_SEEDS, 0.25, 0.30, 0.05, 0.05) == "H0"
    assert decide([0.25, 0.5, 0.0, 0.25] * 3, 0.25, 0.30, 0.05, 0.05) is None, "too close to call yet"


def test_a_native_run_stops_early_on_a_clear_gap(tmp_path):
    out = tmp_path / "s.jsonl"
    r = run_sprt("ismcts@60", "random", p0=0.25, p1=0.30, batch=4, max_seeds=40, seed_start=0, threads=2, out=out)
    assert r.decision == "H1" and r.seeds < 40 and r.seeds % 4 == 0
    assert len(out.read_text().splitlines()) == 4 * r.seeds


def test_cli_sprt(tmp_path):
    from test_train import cli

    out = tmp_path / "s.jsonl"
    r = cli("sprt", "--candidate", "ismcts@60", "--baseline", "random", "--batch", "4", "--max-seeds", "40",
            "--seed-start", "0", "--threads", "2", "--out", str(out))
    assert r.returncode == 0, r.stderr
    assert "sprt: H1" in r.stdout and "seeds: win rate" in r.stdout
