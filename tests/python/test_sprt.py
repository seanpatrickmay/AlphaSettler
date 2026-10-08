"""The sequential test (alphasettler/sprt.py): the log-likelihood ratio, the bounds, the decision and
a short native run."""

import math

import pytest

from alphasettler.arena import TRADES
from alphasettler.sprt import MIN_SEEDS, VARIANCE_FLOOR_DIV, bounds, decide, llr, run_sprt


def test_the_bounds_are_walds():
    lo, hi = bounds(0.05, 0.05)
    assert lo == pytest.approx(math.log(0.05 / 0.95)) and hi == pytest.approx(math.log(0.95 / 0.05))


def test_the_llr_favours_the_hypothesis_nearer_the_mean():
    strong = [0.5, 0.25, 0.5, 0.75, 0.25] * 4  # mean 0.45
    null = [0.25, 0.0, 0.5, 0.25, 0.25] * 4  # mean 0.25
    assert llr(strong, 0.25, 0.30) > 0
    assert llr(null, 0.25, 0.30) < 0
    assert llr([0.275] * 3 + [0.25, 0.30], 0.25, 0.30) == pytest.approx(0.0, abs=1e-9), "midway is neutral"


def test_the_variance_is_floored_so_a_streak_cannot_fake_certainty():
    floor = 0.25 * 0.75 / VARIANCE_FLOOR_DIV
    assert llr([0.25] * 12 + [0.0], 0.25, 0.30) == pytest.approx(13 * 0.05 * (2 * 3 / 13 - 0.55) / (2 * floor))
    assert -3 < llr([0.25] * 11 + [0.0], 0.25, 0.30) < 0, "a dozen seeds near 0.25 are not yet a decision"
    assert llr([0.5] * 20, 0.25, 0.30) > 0 and math.isfinite(llr([0.5] * 20, 0.25, 0.30))


def test_decisions_wait_for_enough_seeds_then_stop_at_a_bound():
    assert decide([1.0] * (MIN_SEEDS - 1), 0.25, 0.30, 0.05, 0.05) is None
    assert decide([0.75, 1.0] * MIN_SEEDS, 0.25, 0.30, 0.05, 0.05) == "H1"
    assert decide([0.0, 0.25] * MIN_SEEDS, 0.25, 0.30, 0.05, 0.05) == "H0"
    assert decide([0.25, 0.5, 0.0, 0.25] * 3, 0.25, 0.30, 0.05, 0.05) is None, "too close to call yet"


def test_a_native_run_stops_early_on_a_clear_gap(tmp_path):
    out = tmp_path / "s.jsonl"
    r = run_sprt("ismcts@60", "random", p0=0.25, p1=0.30, batch=4, max_seeds=60, seed_start=0, threads=2, out=out)
    assert r.decision == "H1" and MIN_SEEDS <= r.seeds < 60 and r.seeds % 4 == 0
    assert len(out.read_text().splitlines()) == 4 * r.seeds


def test_cli_sprt(tmp_path):
    from test_train import cli

    out = tmp_path / "s.jsonl"
    r = cli("sprt", "--candidate", "ismcts@60", "--baseline", "random", "--batch", "4", "--max-seeds", "60",
            "--seed-start", "0", "--threads", "2", "--out", str(out))
    assert r.returncode == 0, r.stderr
    assert "sprt: H1" in r.stdout and "seeds: win rate" in r.stdout


def test_max_seeds_below_the_minimum_is_refused(tmp_path):
    from test_train import cli

    with pytest.raises(ValueError, match="max_seeds"):
        run_sprt("ismcts@60", "random", max_seeds=MIN_SEEDS - 1, out=tmp_path / "s.jsonl")
    assert not (tmp_path / "s.jsonl").exists(), "refused before any game"
    r = cli("sprt", "--candidate", "ismcts@60", "--baseline", "random", "--max-seeds", str(MIN_SEEDS - 1),
            "--out", str(tmp_path / "c.jsonl"))
    assert r.returncode == 2 and "max_seeds" in r.stderr, r.stderr
    assert not (tmp_path / "c.jsonl").exists()


@pytest.mark.parametrize("flag, expected", [("--trades", TRADES), ("--no-trades", {"max_offers_per_turn": 0}),
                                            (None, None)])
def test_cli_sprt_passes_the_chosen_rules_to_the_test(monkeypatch, tmp_path, flag, expected):
    import alphasettler.sprt
    from alphasettler.cli import main

    seen = {}

    def fake_run_sprt(candidate, baseline, **kwargs):
        seen["config"] = kwargs["config"]
        raise SystemExit(0)

    monkeypatch.setattr(alphasettler.sprt, "run_sprt", fake_run_sprt)
    argv = ["sprt", "--candidate", "ismcts@20+t", "--baseline", "ismcts@20", "--out", str(tmp_path / "s.jsonl")]
    with pytest.raises(SystemExit):
        main(argv + ([flag] if flag else []))
    assert seen["config"] == expected


def test_cli_sprt_refuses_both_rule_flags():
    from test_train import cli

    r = cli("sprt", "--candidate", "ismcts@20", "--baseline", "random", "--trades", "--no-trades")
    assert r.returncode != 0 and "not allowed with" in r.stderr
