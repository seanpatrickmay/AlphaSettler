"""Fast experiments (alphasettler/train/fast.py): the pool, the step batches, the schedule and a
budgeted run. Needs the `train` extra, like test_train.py."""

import json
import math

import pytest
import torch

from alphasettler import Bot
from alphasettler.train.data import to_tensors
from alphasettler.train.fast import (
    MIN_LR_FRACTION, FastConfig, fast_train, load_pool, lr_at, score_net, step_batch, value_mask,
)

from test_train import cli, generation

TINY = FastConfig(minutes=0.02, lr=1e-3, warmup=2, policy_batch=8, value_batch=4, width=8, rounds=1,
                  eval_seconds=0.3, device="cpu", threads=2)


def two_gens(tmp_path):
    return [generation(tmp_path, "gen-000", seed_start=0), generation(tmp_path, "gen-001", seed_start=20)]


def test_the_pool_holds_out_every_generations_seeds_divisible_by_20(tmp_path):
    pool = load_pool(tmp_path, two_gens(tmp_path))
    assert (pool.train.games, pool.validation.games) == (2, 2)
    assert 0 < pool.train.policy_decisions <= pool.train.full_decisions
    assert sorted(pool.files) == ["gen-000/selfplay-0.jsonl.gz", "gen-001/selfplay-20.jsonl.gz"]


def test_a_pool_without_held_out_games_is_an_error(tmp_path):
    with pytest.raises(ValueError, match="no held-out"):
        load_pool(tmp_path, [generation(tmp_path, seed_start=1)])


def test_a_step_is_policy_targets_then_value_rows_and_seeded(tmp_path):
    pool = load_pool(tmp_path, two_gens(tmp_path))
    b = step_batch(pool.train, TINY, 3, 2)
    assert b["n"] == 12
    t = to_tensors(b, torch.device("cpu"))
    assert t.full[:8].eq(1).all(), "the first policy_batch rows are policy targets"
    assert step_batch(pool.train, TINY, 3, 1) == b
    assert step_batch(pool.train, TINY, 4, 2) != b
    assert value_mask(TINY).tolist() == [False] * 8 + [True] * 4


def test_the_learning_rate_warms_up_then_follows_a_cosine_over_the_budget():
    cfg = FastConfig(minutes=1.0, lr=1e-3, warmup=10)
    assert lr_at(cfg, 0, 0.0) == pytest.approx(1e-4)
    assert lr_at(cfg, 9, 0.0) == pytest.approx(1e-3)
    assert lr_at(cfg, 10, 0.0) == pytest.approx(1e-3)
    assert lr_at(cfg, 50, 30.0) == pytest.approx(1e-3 * (MIN_LR_FRACTION + (1 - MIN_LR_FRACTION) / 2))
    assert lr_at(cfg, 99, 60.0) == pytest.approx(1e-3 * MIN_LR_FRACTION)
    assert lr_at(cfg, 99, 600.0) == pytest.approx(1e-3 * MIN_LR_FRACTION), "past the budget it stays at the floor"


@pytest.mark.parametrize("bad", [dict(minutes=0), dict(minutes=math.nan), dict(lr=0), dict(warmup=-1),
                                 dict(policy_batch=0), dict(value_batch=2**16 + 1), dict(eval_seconds=0),
                                 dict(width=0)])
def test_bad_settings_are_errors(bad):
    with pytest.raises(ValueError):
        FastConfig(**{**TINY.__dict__, **bad}).check()


def test_a_budgeted_run_writes_the_best_net_and_its_summary(tmp_path):
    pool = load_pool(tmp_path, two_gens(tmp_path))
    nets = tmp_path / "nets"
    nets.mkdir()
    s = fast_train(pool, nets / "c.bin", TINY)
    assert s["steps"] > 0 and s["seconds"] >= TINY.minutes * 60
    assert s["parity_max_error"] <= 1e-5 and (s["width"], s["rounds"]) == (8, 1)
    assert s["best"]["score"] == min(h["score"] for h in s["history"])
    assert json.loads((nets / "c.bin.json").read_text())["steps"] == s["steps"]
    assert not [p for p in nets.iterdir() if p.name.startswith(".")], "no temp file left"
    assert Bot("net:c@5", 0, nets_dir=str(nets)).name == "net"
    h = score_net(pool, nets / "c.bin", "cpu", 2)
    assert h["score"] == pytest.approx(s["best"]["score"], abs=1e-5)
    s2 = fast_train(pool, nets / "d.bin", TINY, init=nets / "c.bin")
    assert s2["init"] == str(nets / "c.bin")


def test_cli_fast_train_and_score(tmp_path):
    gens = two_gens(tmp_path)
    out = tmp_path / "nets" / "f.bin"
    r = cli("fast-train", *map(str, gens), "--out", str(out), "--minutes", "0.01", "--device", "cpu",
            "--width", "8", "--rounds", "1", "--policy-batch", "8", "--value-batch", "4", "--threads", "2")
    assert r.returncode == 0, r.stderr
    assert "pool: 2 games" in r.stdout and f"wrote {out}" in r.stdout
    r = cli("fast-score", *map(str, gens), "--nets", str(out), "--device", "cpu")
    assert r.returncode == 0, r.stderr
    assert f"{out}: score" in r.stdout
