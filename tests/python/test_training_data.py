import hashlib
import json
import subprocess
import sys
from array import array
from pathlib import Path

import pytest

from alphasettler import Game, files, records
from alphasettler._engine import (
    NET_BATCH, NET_HIDDEN_LEN, NET_PUBLIC_LEN, NOISE_ALPHA, NOISE_WEIGHT, SAMPLE_TURNS, SEARCH_BATCH, TrainingSet,
    action_space_size, encode, net_forward, resolve_net, write_random_net,
)
from alphasettler.selfplay import write_explore

CONFIG = {"max_offers_per_turn": 0}
A = action_space_size()


def floats(b: bytes) -> list[float]:
    return array("f", b).tolist()


def explore_games(tmp_path, games, config=CONFIG, name="x.jsonl.gz", **kw):
    out = tmp_path / name
    counts = write_explore(out, 0, games, config=config, full_simulations=6, fast_simulations=3,
                           full_probability=0.25, threads=2, **kw)
    back = list(records.read(out))
    assert counts["games"] == games == len(back)
    assert counts["decisions"] == sum(len(g["decisions"]) for g in back)
    return back


def test_explore_records_carry_their_settings_and_no_observations(tmp_path):
    for g in explore_games(tmp_path, 3):
        assert g["selfplay"] == {
            "bot": "ismcts", "net": None, "net_sha256": None, "full_simulations": 6, "fast_simulations": 3,
            "full_probability": 0.25, "noise_weight": NOISE_WEIGHT, "noise_alpha": NOISE_ALPHA,
            "sample_turns": SAMPLE_TURNS, "batch": SEARCH_BATCH,
        }
        assert (NOISE_WEIGHT, NOISE_ALPHA, SAMPLE_TURNS) == (0.25, 10.0, 8)
        assert all("observation" not in d for d in g["decisions"])
        assert {d["full_search"] for d in g["decisions"]} == {True, False}
        for d in g["decisions"]:
            assert sum(d["visits"]) == (5 if d["full_search"] else 2)
        assert g["belief_resets"] == 0
        assert records.replay_mismatches(g) == []


def test_net_self_play_records_the_net_and_its_hash(tmp_path):
    nets = tmp_path / "nets"
    nets.mkdir()
    write_random_net(str(nets / "tiny.bin"), 1, 16, 8)
    (g,) = explore_games(tmp_path, 1, net="tiny", nets_dir=nets)
    assert g["selfplay"]["bot"] == "net:tiny" and g["selfplay"]["net"] == "tiny"
    assert g["selfplay"]["net_sha256"] == hashlib.sha256((nets / "tiny.bin").read_bytes()).hexdigest()
    assert g["selfplay"]["batch"] == NET_BATCH
    with pytest.raises(ValueError, match="bad net name"):
        write_explore(tmp_path / "no.jsonl.gz", 0, 1, config=CONFIG, full_simulations=6, fast_simulations=3,
                      full_probability=0.25, net="../tiny", nets_dir=nets)
    assert not (tmp_path / "no.jsonl.gz").exists()


def test_record_files_are_byte_identical_for_the_same_games(tmp_path):
    explore_games(tmp_path, 2, name="a.jsonl.gz")
    explore_games(tmp_path, 2, name="b.jsonl.gz")
    assert (tmp_path / "a.jsonl.gz").read_bytes() == (tmp_path / "b.jsonl.gz").read_bytes()


def test_a_batch_equals_the_encoder_on_the_replayed_state(tmp_path):
    games = explore_games(tmp_path, 2)
    ts = TrainingSet(games)
    assert ts.games == 2 and ts.decisions == sum(len(g["decisions"]) for g in games)
    assert 0 < ts.full_decisions < ts.decisions
    checked = 0
    for gi, g in enumerate(games):
        for di, d in enumerate(g["decisions"][:40]):
            b = ts.position(gi, di)
            game = Game(g["seed"], g["config"])
            for a in g["actions"][: d["index"]]:
                game.apply(a)
            assert game.current_actor == d["actor"]
            public, hidden = encode(game)
            assert b["n"] == 1 and b["public"] == public and b["hidden"] == hidden
            mask = list(b["mask"])
            assert [i for i in range(A) if mask[i]] == sorted(game.legal_actions())
            policy = floats(b["policy"])
            total = sum(d["visits"])
            for a, v in zip(d["legal"], d["visits"]):
                assert policy[a] == pytest.approx(v / total, rel=1e-6)
            assert list(b["full"]) == [int(d["full_search"])]
            checked += 1
    assert checked > 40
    batch = ts.sample(50, 3, 2)
    assert batch == ts.sample(50, 3, 1)
    assert (len(batch["public"]), len(batch["mask"]), len(batch["outcome"])) == (50 * len(public), 50 * A, 50 * 16)


def test_turn_cap_draws_are_a_quarter_each(tmp_path):
    games = explore_games(tmp_path, 2, config={"max_offers_per_turn": 0, "max_turns": 4})
    assert all(g["winner"] is None for g in games)
    b = TrainingSet(games).sample(20, 1, 1)
    assert floats(b["outcome"]) == [0.25] * 80


def test_a_record_that_does_not_replay_is_an_error(tmp_path):
    (g,) = explore_games(tmp_path, 1)
    bad = json.loads(json.dumps(g))
    bad["winner"] = None if g["winner"] is not None else 0
    with pytest.raises(ValueError, match="winner"):
        TrainingSet([bad])
    missing = dict(g)
    del missing["vp"]
    with pytest.raises(ValueError, match="missing"):
        TrainingSet([missing])
    bad_id = json.loads(json.dumps(g))
    bad_id["actions"][0] = 9999
    with pytest.raises(ValueError, match="9999"):
        TrainingSet([bad_id])
    with pytest.raises(ValueError, match="no positions"):
        TrainingSet([{**g, "decisions": []}]).sample(1, 0, 1)


def test_net_forward_runs_the_rust_forward_pass(tmp_path):
    path = tmp_path / "r.bin"
    write_random_net(str(path), 2, 16, 8)
    public, hidden = encode(Game(1, CONFIG))
    out = net_forward(str(path), public * 3, hidden * 3)
    assert (len(out["policy"]), len(out["value"]), len(out["aux"])) == (3 * A * 4, 3 * 16, 3 * 16)
    with pytest.raises(ValueError, match="for some n"):
        net_forward(str(path), public, hidden * 2)
    with pytest.raises(ValueError, match="cannot read"):
        net_forward(str(tmp_path / "none.bin"), public, hidden)


def test_resolve_net_checks_the_name_and_the_file(tmp_path):
    write_random_net(str(tmp_path / "tiny.bin"), 1, 16, 8)
    assert resolve_net("tiny", str(tmp_path)) == str(tmp_path / "tiny.bin")
    for bad in ["../tiny", "a/b", "", "x" * 65]:
        with pytest.raises(ValueError, match="bad net name"):
            resolve_net(bad, str(tmp_path))
    with pytest.raises(ValueError, match="cannot read"):
        resolve_net("missing", str(tmp_path))


def cli(*args):
    return subprocess.run([sys.executable, "-m", "alphasettler", *args], capture_output=True, text=True)


def test_cli_selfplay_with_a_net_or_pcr(tmp_path):
    nets = tmp_path / "nets"
    nets.mkdir()
    write_random_net(str(nets / "tiny.bin"), 1, 16, 8)
    r = cli("selfplay", "--net", "tiny", "--nets-dir", str(nets), "--games", "2", "--full-simulations", "4",
            "--fast-simulations", "2", "--out-dir", str(tmp_path / "out"))
    assert r.returncode == 0, r.stderr
    (path,) = (tmp_path / "out").glob("*-selfplay-net-tiny-f4-q2.jsonl.gz")
    assert "2 games" in r.stdout and "belief resets: 0" in r.stdout
    assert all(g["selfplay"]["net"] == "tiny" for g in records.read(path))
    r = cli("selfplay", "--pcr", "--games", "1", "--full-simulations", "4", "--fast-simulations", "2",
            "--out-dir", str(tmp_path / "out"))
    assert r.returncode == 0, r.stderr
    assert len(list((tmp_path / "out").glob("*-selfplay-ismcts-f4-q2.jsonl.gz"))) == 1
    for bad, message in [
        (["--net", "tiny", "--simulations", "10"], "--simulations"),
        (["--pcr", "--rollout", "4"], "--rollout"),
        (["--net", "../tiny"], "bad net name"),
        (["--net", "missing"], "cannot read"),
        (["--pcr", "--full-simulations", "1"], "full simulations"),
        (["--pcr", "--full-probability", "1.5"], "probability"),
        (["--full-simulations", "4"], "--simulations is required"),
        (["--simulations", "10", "--fast-simulations", "4"], "needs --net or --pcr"),
    ]:
        r = cli("selfplay", "--games", "1", "--nets-dir", str(nets), *bad, "--out-dir", str(tmp_path / "never"))
        assert r.returncode == 2 and message in r.stderr, (bad, r.stderr)
        assert not (tmp_path / "never").exists()


def test_encode_buffers_have_the_documented_lengths():
    public, hidden = encode(Game(1, CONFIG))
    assert (len(public), len(hidden)) == (4 * NET_PUBLIC_LEN, 4 * NET_HIDDEN_LEN)
    game = Game(1, {"vp_to_win": 3, "max_offers_per_turn": 0})
    while not game.is_over:
        game.apply(game.legal_actions()[0])
    with pytest.raises(ValueError, match="over"):
        encode(game)


def test_a_training_set_grows_one_file_at_a_time(tmp_path):
    """F2: the window never has to be one list of dicts; `extend` takes any iterable of games."""
    a = explore_games(tmp_path, 2, name="a.jsonl.gz")
    b = explore_games(tmp_path, 1, name="b.jsonl.gz", config={"max_offers_per_turn": 0, "max_turns": 6})
    whole = TrainingSet(a + b)
    parts = TrainingSet()
    parts.extend(a)
    parts.extend(records.read(tmp_path / "b.jsonl.gz"))
    sizes = lambda t: (t.games, t.decisions, t.full_decisions)  # noqa: E731
    assert sizes(parts) == sizes(whole)
    assert parts.sample(40, 2, 2) == whole.sample(40, 2, 2)
    bad = json.loads(json.dumps(a[1]))
    bad["vp"][0] += 1
    with pytest.raises(ValueError, match=r"game 4 \(seed 1\).*vp"):
        parts.extend([a[0], bad])
    assert sizes(parts) == sizes(whole), "a failed extend adds nothing"


def test_malformed_records_are_value_errors_not_crashes(tmp_path):
    (g,) = explore_games(tmp_path, 1)

    def broken(edit):
        bad = json.loads(json.dumps(g))
        edit(bad)
        return bad

    cases = [
        (lambda b: b.update(vp=[1, 2, 3]), "4 entries"),
        (lambda b: b.update(vp=[1, 2, 3, 300]), "vp"),
        (lambda b: b.update(winner=7), "winner"),
        (lambda b: b.update(seed=-1), "seed"),
        (lambda b: b["actions"].__setitem__(0, -1), "action id"),
        (lambda b: b["decisions"][0].update(actor=4), "actor"),
        (lambda b: b["decisions"][0].update(index=-1), "index"),
        (lambda b: b["decisions"][0].update(index=len(g["actions"]) + 5), "out of order"),
        (lambda b: b["decisions"][0]["visits"].append(-1), "visits"),
        (lambda b: b["decisions"][0]["visits"].pop(), "visit counts"),
        (lambda b: b["decisions"][0].pop("full_search"), "missing"),
        (lambda b: b.update(config={"bogus": 1}), "bogus"),
    ]
    for edit, message in cases:
        with pytest.raises(ValueError, match=message):
            TrainingSet([broken(edit)])
    ts = TrainingSet([g])
    for game, decision in [(1, 0), (0, len(g["decisions"])), (-1, 0)]:
        with pytest.raises((ValueError, OverflowError)):
            ts.position(game, decision)
    for n in [0, 1 << 17]:
        with pytest.raises(ValueError):
            ts.sample(n, 0, 1)


def test_a_temp_name_is_never_a_record_file(tmp_path):
    """F3: a half-written record is `.NAME.tmp-PID`: hidden, and not matched by `*.jsonl.gz`."""
    final = tmp_path / "20261002-selfplay-ismcts-f6-q3.jsonl.gz"
    tmp = files.temp_path(final)
    assert tmp.parent == tmp_path and tmp.name.startswith(".") and str(tmp).endswith(f".tmp-{__import__('os').getpid()}")
    tmp.write_bytes(b"half a file")
    assert list(tmp_path.glob("*.jsonl.gz")) == []
    files.remove_temps(tmp_path, final.name)
    assert list(tmp_path.iterdir()) == []


def test_file_helpers_replace_atomically(tmp_path):
    p = tmp_path / "x.txt"
    files.atomic_write_text(p, "one")
    files.atomic_write_bytes(p, b"two")
    assert p.read_bytes() == b"two"
    files.atomic_copy(p, tmp_path / "y.txt")
    assert files.sha256_file(tmp_path / "y.txt") == hashlib.sha256(b"two").hexdigest()
    (tmp_path / ".x.txt.tmp-1").write_text("left over")
    (tmp_path / ".other.tmp-1").write_text("not ours")
    files.remove_temps(tmp_path, "x.txt")
    assert sorted(q.name for q in tmp_path.iterdir()) == [".other.tmp-1", "x.txt", "y.txt"]


def _selfplay_args(out):
    return ["selfplay", "--pcr", "--games", "1", "--full-simulations", "4", "--fast-simulations", "2",
            "--out-dir", str(out)]


def test_cli_selfplay_publishes_its_file_only_when_complete(tmp_path, monkeypatch):
    """F3: the CLI writes to the temp name and renames when done; an interrupted run's games are
    renamed into place too; a failed run leaves nothing."""
    from alphasettler import cli, selfplay

    out = tmp_path / "out"
    real = selfplay.write_explore

    def complete(path, *a, **kw):
        assert Path(path).name.startswith(".")
        counts = real(path, *a, **kw)
        assert list(out.glob("*.jsonl.gz")) == [], "the final name appears only after the write"
        return counts

    monkeypatch.setattr(selfplay, "write_explore", complete)
    assert cli.main(_selfplay_args(out)) == 0
    (first,) = out.iterdir()
    assert first.name.endswith("-selfplay-ismcts-f4-q2.jsonl.gz")

    def interrupted(path, *a, **kw):
        real(path, *a, **kw)
        raise KeyboardInterrupt

    monkeypatch.setattr(selfplay, "write_explore", interrupted)
    assert cli.main(_selfplay_args(out)) == 130
    names = sorted(p.name for p in out.iterdir())
    assert len(names) == 2 and not any(n.startswith(".") for n in names)
    assert all(len(list(records.read(out / n))) == 1 for n in names)

    def failed(path, *a, **kw):
        real(path, *a, **kw)
        raise OSError("disk full")

    monkeypatch.setattr(selfplay, "write_explore", failed)
    assert cli.main(_selfplay_args(out)) == 2
    assert sorted(p.name for p in out.iterdir()) == names
