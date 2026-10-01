import math

import pytest

from alphasettler.stats import candidate_won, paired, per_seed_win_rates, summarize


def rec(seed, seat, winner, vp=None, turns=100):
    return {"seed": seed, "candidate_seat": seat, "winner": winner,
            "vp": vp or [5, 5, 5, 5], "turns": turns, "actions": 1000}


def four(seed, wins):
    """Four rotated games for `seed`; the candidate wins in the first `wins` seats."""
    return [rec(seed, s, s if s < wins else (s + 1) % 4) for s in range(4)]


def test_candidate_won():
    assert candidate_won(rec(0, 2, 2))
    assert not candidate_won(rec(0, 2, 1))


def test_draws_count_as_losses():
    assert not candidate_won(rec(0, 0, None))
    s = summarize([rec(0, seat, None) for seat in range(4)] + four(1, 2))
    assert s.wins == 2
    assert s.win_rate == pytest.approx(0.25)


def test_per_seed_rates():
    rates = per_seed_win_rates(four(0, 2) + four(1, 0))
    assert rates == {0: 0.5, 1: 0.0}


def test_summary_numbers():
    s = summarize(four(0, 2) + four(1, 0))
    assert (s.seeds, s.games, s.wins) == (2, 8, 2)
    assert s.win_rate == pytest.approx(0.25)
    # per-seed rates 0.5 and 0.0: sample sd = 0.3536, se = 0.25
    assert s.z_vs_null == pytest.approx(0.0)
    assert s.ci_low == pytest.approx(0.0)
    assert s.ci_high == pytest.approx(0.25 + 1.96 * 0.25)
    assert s.mean_vp == pytest.approx(5.0)
    assert s.mean_turns == pytest.approx(100.0)


def test_z_against_null():
    records = four(0, 2) + four(1, 1) + four(2, 2) + four(3, 1)
    s = summarize(records)
    # rates 0.5, 0.25, 0.5, 0.25: mean 0.375, sd 0.1443, se 0.07217, z = 1.732
    assert s.win_rate == pytest.approx(0.375)
    assert s.z_vs_null == pytest.approx(0.125 / (0.1443375673 / 2), rel=1e-6)


def test_zero_variance_and_single_seed():
    all_wins = summarize(four(0, 4) + four(1, 4))
    assert all_wins.win_rate == 1.0
    assert all_wins.z_vs_null == math.inf
    assert all_wins.ci_high == 1.0
    all_losses = summarize(four(0, 0) + four(1, 0))
    assert all_losses.z_vs_null == -math.inf
    at_null = summarize(four(0, 1) + four(1, 1))
    assert at_null.z_vs_null == 0.0
    one = summarize(four(7, 3))
    assert one.seeds == 1 and one.win_rate == 0.75 and one.z_vs_null == math.inf
    with pytest.raises(ValueError):
        summarize([])


def test_paired():
    a = four(0, 3) + four(1, 2)
    b = four(0, 1) + four(1, 1)
    p = paired(a, b)
    # diffs 0.5 and 0.25: mean 0.375, sd 0.1768, se 0.125, z 3.0
    assert p.seeds == 2
    assert p.mean_diff == pytest.approx(0.375)
    assert p.se == pytest.approx(0.125)
    assert p.z == pytest.approx(3.0)
    with pytest.raises(ValueError, match="same seeds"):
        paired(four(0, 1), four(1, 1))


def test_draws_are_counted():
    s = summarize([rec(0, seat, None) for seat in range(4)] + four(1, 2))
    assert s.draws == 4
    assert summarize(four(0, 2)).draws == 0


def test_mean_vp_reads_the_candidate_seat():
    # Distinct VP per seat, and the candidate moves seat each game: mean_vp must follow it.
    vp = [3, 6, 9, 12]
    records = [rec(0, seat, 3, vp=vp) for seat in range(4)] + [rec(1, seat, 3, vp=[1, 1, 1, 1]) for seat in range(4)]
    s = summarize(records)
    assert s.mean_vp == pytest.approx((3 + 6 + 9 + 12 + 4 * 1) / 8)
    one_seat_only = [rec(2, seat, 3, vp=[0, 0, 0, 0]) for seat in range(4)]
    one_seat_only[2]["vp"] = [0, 0, 8, 0]
    assert summarize(one_seat_only).mean_vp == pytest.approx(2.0)


@pytest.mark.parametrize(
    "records",
    [
        four(0, 1)[:3],  # truncated
        four(0, 1) + [rec(0, 2, 2)],  # duplicated seat
        four(0, 1) + four(0, 2),  # two runs mixed on the same seed
        four(0, 1) + [rec(1, 0, 0)],  # a second seed cut short
        [rec(0, s, s) for s in (0, 1, 2, 4)],  # a seat outside 0..3
    ],
)
def test_incomplete_rotations_are_rejected(records):
    with pytest.raises(ValueError, match="seed"):
        per_seed_win_rates(records)
    with pytest.raises(ValueError, match="seed"):
        summarize(records)
