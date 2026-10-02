import json

import pytest

from alphasettler import arena, run_match
from alphasettler.arena import format_summary, run
from alphasettler.stats import summarize


def rotation(seed, winner_of_seat, vp=(5, 5, 5, 5)):
    """Four rotated records for `seed`; `winner_of_seat(seat)` is that game's winner."""
    return [{"seed": seed, "candidate_seat": c, "winner": winner_of_seat(c), "vp": list(vp), "turns": 50,
             "actions": 1} for c in range(4)]


def test_run_writes_jsonl_and_summarizes(tmp_path):
    out = tmp_path / "x" / "games.jsonl"
    records, summary = run("greedy", "random", seeds=5, seed_start=100, threads=2, out=out)
    assert len(records) == 20
    lines = out.read_text().splitlines()
    assert len(lines) == 20
    first = json.loads(lines[0])
    assert first["candidate"] == "greedy" and first["baseline"] == "random"
    assert first["seed"] == 100 and first["candidate_seat"] == 0
    assert first["belief_resets"] == 0  # neither bot reports any
    assert summary.seeds == 5 and summary.games == 20
    text = format_summary("greedy vs random", summary)
    assert "win rate" in text and "z vs 0.25" in text
    assert f"{summary.draws} draws" in text


def test_greedy_beats_random_with_z_above_3():
    _, summary = run("greedy", "random", seeds=300)
    assert summary.z_vs_null > 3, summary


def test_rotations_of_a_seed_are_the_same_game_when_bots_are_identical():
    # Each seat's bot stream depends on (seed, seat) only, so with candidate == baseline the
    # four rotations of a seed are one game, differing only in which seat is labelled candidate.
    records, _ = run("random", "random", seeds=30, seed_start=500)
    assert len(records) == 120
    for i in range(0, 120, 4):
        group = records[i : i + 4]
        assert [r["candidate_seat"] for r in group] == [0, 1, 2, 3]
        games = [(r["seed"], r["winner"], r["vp"], r["turns"], r["actions"]) for r in group]
        assert games == [games[0]] * 4


def test_batched_run_matches_direct_calls(tmp_path, monkeypatch):
    monkeypatch.setattr(arena, "BATCH_SEEDS", 7)
    out = tmp_path / "batched.jsonl"
    records, _ = run("greedy", "random", seeds=20, seed_start=40, threads=3, out=out)
    assert records == run_match("greedy", "random", 40, 20, 3)
    assert records == run_match("greedy", "random", 40, 7, 1) + run_match("greedy", "random", 47, 13, 2)
    lines = [json.loads(line) for line in out.read_text().splitlines()]
    assert [{k: v for k, v in r.items() if k not in ("candidate", "baseline")} for r in lines] == records


def test_bad_output_path_fails_before_playing(tmp_path, monkeypatch):
    played = []
    monkeypatch.setattr(arena, "run_match", lambda *a: played.append(a[3]) or [])
    existing = tmp_path / "games.jsonl"
    existing.write_text("keep\n")
    with pytest.raises(FileExistsError):
        run("random", "random", seeds=3, out=existing)
    with pytest.raises(OSError):
        run("random", "random", seeds=3, out=tmp_path)
    assert existing.read_text() == "keep\n"
    assert all(n == 0 for n in played)  # only the zero-seed validation calls ran


def test_format_summary_zero_variance_says_ci_na():
    text = format_summary("x", summarize(rotation(0, lambda c: c) + rotation(1, lambda c: c)))
    assert "CI n/a (zero variance)" in text
    assert "[1.000, 1.000]" not in text


def test_format_summary_counts_draws():
    records = rotation(0, lambda c: None) + rotation(1, lambda c: None) + rotation(2, lambda c: c)
    s = summarize(records)
    assert s.draws == 8
    text = format_summary("x", s)
    assert "(12 games, 4 wins, 8 draws)" in text
    assert "95% CI" in text and "n/a" not in text


def test_seed_range_end_must_fit_in_u64(tmp_path):
    out = tmp_path / "never.jsonl"
    with pytest.raises(ValueError, match="u64"):
        run("random", "random", seeds=1, seed_start=2**64 - 1, out=out)
    assert not out.exists()
