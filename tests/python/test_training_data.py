import hashlib
import json
import subprocess
import sys
from array import array
from pathlib import Path

import pytest

from alphasettler import Game, files, records
from alphasettler._engine import (
    LEAGUE_PROBABILITY, LEAGUE_SEATS, LEAGUE_STYLES, NET_BATCH, NET_HIDDEN_LEN, NET_PUBLIC_LEN, NOISE_ALPHA,
    NOISE_WEIGHT, OPENING_EPSILON, OPENING_TOP, SAMPLE_TURNS, SEARCH_BATCH, TrainingSet, action_space_size, encode,
    net_forward, resolve_net, selfplay_explore, write_random_net,
)
from alphasettler.selfplay import write_explore
from opening_games import opening_model_file

CONFIG = {"max_offers_per_turn": 0}
TRADING = {"max_offers_per_turn": 2, "max_trade_cards": 1}
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
    write_random_net(str(nets / "tiny.bin"), 1, 8, 1)
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


@pytest.mark.parametrize("config", [CONFIG, TRADING], ids=["no-trades", "trading"])
def test_gumbel_self_play_records_an_improved_policy_on_every_decision(tmp_path, config):
    games = explore_games(tmp_path, 2, config=config, gumbel=6)
    for g in games:
        assert g["selfplay"] == {"bot": "ismcts", "net": None, "net_sha256": None, "gumbel_simulations": 6,
                                 "batch": SEARCH_BATCH}
        assert g["belief_resets"] == 0 and records.replay_mismatches(g) == []
        assert g["decisions"]
        for d in g["decisions"]:
            assert d["full_search"] is True and sum(d["visits"]) == 5
            assert len(d["target"]) == len(d["legal"]) and min(d["target"]) >= 0
            assert sum(d["target"]) == pytest.approx(1.0, abs=1e-4)


def test_plain_self_play_records_carry_no_target(tmp_path):
    assert all("target" not in d for g in explore_games(tmp_path, 2) for d in g["decisions"])


def test_gumbel_self_play_needs_at_least_two_simulations(tmp_path):
    with pytest.raises(ValueError, match="gumbel simulations"):
        write_explore(tmp_path / "no.jsonl.gz", 0, 1, config=CONFIG, full_simulations=6, fast_simulations=3,
                      full_probability=0.25, gumbel=1)
    assert not (tmp_path / "no.jsonl.gz").exists()


def test_opening_self_play_records_no_setup_settlement_and_names_its_model(tmp_path):
    model = opening_model_file(tmp_path / "opening.bin")
    games = explore_games(tmp_path, 2, config=TRADING, opening=model)
    plain = explore_games(tmp_path, 2, config=TRADING, name="plain.jsonl.gz")
    assert (OPENING_EPSILON, OPENING_TOP) == (0.25, 5)
    for g, p in zip(games, plain):
        assert g["selfplay"] == {**p["selfplay"], "opening_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
                                 "opening_epsilon": OPENING_EPSILON, "opening_top": OPENING_TOP}
        assert g["belief_resets"] == 0 and records.replay_mismatches(g) == []
        recorded = {d["index"] for d in g["decisions"]}
        replay = Game(g["seed"], g["config"])
        settlements = roads = 0
        for i, a in enumerate(g["actions"]):
            if replay.phase == "setup_settlement":
                assert i not in recorded, f"game {g['seed']} move {i}: a setup settlement was recorded"
                settlements += 1
            elif replay.phase == "setup_road":
                roads += i in recorded
            replay.apply(a)
        assert settlements == 8 and roads > 0
    assert any(g["actions"][:16] != p["actions"][:16] for g, p in zip(games, plain)), "the model opens the games"


def test_opening_self_play_refuses_gumbel_and_bad_model_files(tmp_path):
    model = opening_model_file(tmp_path / "opening.bin")
    bad = tmp_path / "bad.bin"
    bad.write_bytes(model.read_bytes()[:-4])
    for kw, message in [({"opening": model, "gumbel": 6}, "--opening and --gumbel"),
                        ({"opening": bad}, "opening model .*bad.bin.*truncated"),
                        ({"opening": tmp_path / "missing.bin"}, "cannot read opening model")]:
        with pytest.raises(ValueError, match=message):
            write_explore(tmp_path / "no.jsonl.gz", 0, 1, config=CONFIG, full_simulations=6, fast_simulations=3,
                          full_probability=0.25, **kw)
        assert not (tmp_path / "no.jsonl.gz").exists()
    with pytest.raises(ValueError, match="Gumbel self-play"):  # the native check, below the Python one
        selfplay_explore(0, 1, 1, 6, 3, 0.25, CONFIG, None, None, 6, str(model))


def test_compound_self_play_records_say_so_and_put_compound_visits_on_first_trades(tmp_path):
    games = explore_games(tmp_path, 3, config=TRADING, compound=True)
    plain = explore_games(tmp_path, 3, config=TRADING, name="plain.jsonl.gz")
    on_trades = 0
    for g, p in zip(games, plain):
        assert g["selfplay"] == {**p["selfplay"], "compound": True}
        assert g["belief_resets"] == 0 and records.replay_mismatches(g) == []
        for d in g["decisions"]:
            trades = [(a, v) for a, v in zip(d["legal"], d["visits"]) if 233 <= a <= 257 and v > 0]
            assert sum(d["visits"]) == (5 if d["full_search"] else 2)
            if not trades:
                continue
            on_trades += 1
            i = d["index"]
            if 233 <= g["actions"][i] <= 257:  # a played compound: its later steps follow, unrecorded
                recorded = {e["index"] for e in g["decisions"]}
                rest = g["actions"][i + 1:i + 3]
                j = next(k for k, a in enumerate(rest) if not 233 <= a <= 257)
                assert rest[j] == 2 or 25 <= rest[j] <= 204, (g["seed"], i, rest)
                assert all(i + 1 + k not in recorded for k in range(j + 1)), (g["seed"], i)
    assert on_trades > 0, "no compound was visited"
    assert any(g["actions"] != p["actions"] for g, p in zip(games, plain)), "compound moves change the games"


def test_compound_self_play_refuses_gumbel(tmp_path):
    with pytest.raises(ValueError, match="--compound and --gumbel"):
        write_explore(tmp_path / "no.jsonl.gz", 0, 1, config=CONFIG, full_simulations=6, fast_simulations=3,
                      full_probability=0.25, gumbel=6, compound=True)
    assert not (tmp_path / "no.jsonl.gz").exists()
    with pytest.raises(ValueError, match="Gumbel"):  # the native check, below the Python one
        selfplay_explore(0, 1, 1, 6, 3, 0.25, CONFIG, None, None, 6, None, compound=True)


LEAGUE_NETS = ["net:tiny@6+t", "ismcts@5+t"]
LEAGUE_STYLE_BOTS = [f"style:{s}@6+t" for s in LEAGUE_STYLES]


def league_nets_dir(tmp_path):
    nets = tmp_path / "nets"
    nets.mkdir(exist_ok=True)
    write_random_net(str(nets / "tiny.bin"), 1, 8, 1)
    return nets


def test_league_self_play_records_its_pool_and_the_seats_of_each_league_game(tmp_path):
    assert (LEAGUE_PROBABILITY, LEAGUE_SEATS, LEAGUE_STYLES) == (0.5, 2, ["roads", "army", "production"])
    nets = league_nets_dir(tmp_path)
    league = {"nets": LEAGUE_NETS, "styles": LEAGUE_STYLE_BOTS}
    games = explore_games(tmp_path, 12, config=TRADING, nets_dir=nets, league=league)
    plain = explore_games(tmp_path, 12, config=TRADING, name="plain.jsonl.gz", nets_dir=nets)
    kinds = set()
    for g, p in zip(games, plain):
        assert g["selfplay"] == {**p["selfplay"], "league_probability": LEAGUE_PROBABILITY,
                                 "league_nets": LEAGUE_NETS, "league_styles": LEAGUE_STYLE_BOTS}
        assert "league" not in p and g["belief_resets"] == 0 and records.replay_mismatches(g) == []
        if "league" not in g:
            kinds.add("plain")
            assert ({k: v for k, v in g.items() if k != "selfplay"} == {k: v for k, v in p.items() if k != "selfplay"}), \
                f"game {g['seed']}: a plain game in a league run is the plain game"
            continue
        kinds.add("league")
        seats = g["league"]
        assert len(seats) == 4
        pool = [n for n in seats if n is not None]
        assert len(pool) == LEAGUE_SEATS and sum(n in LEAGUE_NETS for n in pool) == 1, seats
        assert sum(n in LEAGUE_STYLE_BOTS for n in pool) == 1, seats
        assert {d["actor"] for d in g["decisions"]} == {i for i, n in enumerate(seats) if n is None}, \
            f"game {g['seed']}: only the learner seats are recorded, both of them"
    assert kinds == {"plain", "league"}


def test_league_self_play_refuses_gumbel_unknown_bots_and_an_empty_half(tmp_path):
    nets = league_nets_dir(tmp_path)
    ok = {"nets": LEAGUE_NETS, "styles": LEAGUE_STYLE_BOTS}
    for kw, message in [({"league": ok, "gumbel": 6}, "--league and --gumbel"),
                        ({"league": {**ok, "nets": []}}, "at least one past net"),
                        ({"league": {**ok, "styles": []}}, "at least one style"),
                        ({"league": {**ok, "nets": ["net:missing@6"]}}, "missing"),
                        ({"league": {**ok, "styles": ["style:tempo@6"]}}, "unknown bot"),
                        ({"league": {**ok, "styles": ["style:roads@6+o"]}}, "unknown bot")]:
        with pytest.raises(ValueError, match=message):
            write_explore(tmp_path / "no.jsonl.gz", 0, 1, config=TRADING, full_simulations=6, fast_simulations=3,
                          full_probability=0.25, nets_dir=nets, **kw)
        assert not (tmp_path / "no.jsonl.gz").exists()
    # The native checks, below the Python ones.
    with pytest.raises(ValueError, match="Gumbel self-play"):
        selfplay_explore(0, 1, 1, 6, 3, 0.25, TRADING, None, str(nets), 6, None, LEAGUE_NETS, LEAGUE_STYLE_BOTS)
    with pytest.raises(ValueError, match="league_nets and league_styles"):
        selfplay_explore(0, 1, 1, 6, 3, 0.25, TRADING, None, str(nets), None, None, LEAGUE_NETS, None)


def policy_id(action: int, actor: int) -> int:
    """`settler_net::policy::policy_id` on engine ids: a steal (224..228) or a trade confirmation
    (660..664) names its player relative to the actor."""
    for base in (224, 660):
        if base <= action < base + 4:
            return base + (action - base - actor) % 4
    return action


def test_a_batch_equals_the_encoder_on_the_replayed_state(tmp_path):
    games = explore_games(tmp_path, 2)
    ts = TrainingSet(games)
    assert ts.games == 2 and ts.decisions == sum(len(g["decisions"]) for g in games)
    assert 0 < ts.full_decisions < ts.decisions
    checked = moved = 0
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
            slots = [policy_id(a, d["actor"]) for a in game.legal_actions()]
            assert [i for i in range(A) if mask[i]] == sorted(slots)
            moved += slots != game.legal_actions()
            policy = floats(b["policy"])
            total = sum(d["visits"])
            for a, v in zip(d["legal"], d["visits"]):
                assert policy[policy_id(a, d["actor"])] == pytest.approx(v / total, rel=1e-6)
            assert list(b["full"]) == [int(d["full_search"])]
            checked += 1
    assert checked > 40
    assert moved > 0, "a steal by a seat other than 0 sits at its seat-relative slot"
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
    write_random_net(str(path), 2, 8, 1)
    public, hidden = encode(Game(1, CONFIG))
    out = net_forward(str(path), public * 3, hidden * 3)
    assert (len(out["policy"]), len(out["value"]), len(out["aux"])) == (3 * A * 4, 3 * 16, 3 * 16)
    with pytest.raises(ValueError, match="for some n"):
        net_forward(str(path), public, hidden * 2)
    with pytest.raises(ValueError, match="cannot read"):
        net_forward(str(tmp_path / "none.bin"), public, hidden)


def test_resolve_net_checks_the_name_and_the_file(tmp_path):
    write_random_net(str(tmp_path / "tiny.bin"), 1, 8, 1)
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
    write_random_net(str(nets / "tiny.bin"), 1, 8, 1)
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


def test_a_gumbel_record_trains_on_its_improved_policy(tmp_path):
    games = explore_games(tmp_path, 2, gumbel=6)
    ts = TrainingSet(games)
    assert ts.policy_decisions == ts.decisions == ts.full_decisions
    checked = 0
    for gi, g in enumerate(games):
        for di, d in enumerate(g["decisions"][:30]):
            b = ts.position(gi, di)
            policy = floats(b["policy"])
            for a, t in zip(d["legal"], d["target"]):
                assert policy[policy_id(a, d["actor"])] == pytest.approx(t, abs=1e-7)
            assert sum(policy) == pytest.approx(1.0, abs=1e-4)
            assert list(b["full"]) == [1]
            checked += 1
    assert checked > 30


@pytest.mark.parametrize("edit, message", [
    (lambda t: t.pop(), "target entries"),
    (lambda t: t.append(0.0), "target entries"),
    (lambda t: t.__setitem__(0, t[0] + 0.5), "sums to"),
    (lambda t: t.__setitem__(0, -1.0), "target entry 0"),
    (lambda t: t.__setitem__(0, "x"), "game 0"),
], ids=["short", "long", "sum", "negative", "not-a-number"])
def test_a_malformed_target_is_a_value_error_naming_the_decision(tmp_path, edit, message):
    (g,) = explore_games(tmp_path, 1, gumbel=6)
    bad = json.loads(json.dumps(g))
    edit(bad["decisions"][0]["target"])
    with pytest.raises(ValueError, match=message):
        TrainingSet([bad])


def test_a_temp_name_is_never_a_record_file(tmp_path):
    """F3: a half-written record is `.NAME.tmp-HOST-PID`: hidden, and not matched by `*.jsonl.gz`."""
    final = tmp_path / "20261002-selfplay-ismcts-f6-q3.jsonl.gz"
    tmp = files.temp_path(final)
    assert tmp.parent == tmp_path and tmp.name.startswith(".") and str(tmp).endswith(f".tmp-{files.host()}-{__import__('os').getpid()}")
    tmp.write_bytes(b"half a file")
    assert list(tmp_path.glob("*.jsonl.gz")) == []
    assert files.remove_dead_temps(tmp_path, final.name) == [tmp]
    assert list(tmp_path.iterdir()) == []


def test_file_helpers_replace_atomically(tmp_path):
    p = tmp_path / "x.txt"
    files.atomic_write_text(p, "one")
    files.atomic_write_bytes(p, b"two")
    assert p.read_bytes() == b"two"
    files.atomic_copy(p, tmp_path / "y.txt")
    assert files.sha256_file(tmp_path / "y.txt") == hashlib.sha256(b"two").hexdigest()
    files.temp_path(p).write_text("left over")  # this process's: no write of it is in progress
    (tmp_path / ".x.txt.tmp-1").write_text("hostless: kept")
    files.temp_path(tmp_path / "other").write_text("another name")
    files.remove_dead_temps(tmp_path, "x.txt")
    assert sorted(q.name for q in tmp_path.iterdir()) == [files.temp_path("other").name, ".x.txt.tmp-1", "x.txt",
                                                          "y.txt"]


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


def test_an_interrupt_while_reading_records_is_not_a_value_error(tmp_path):
    """Ctrl-C during `extend(records.read(path))` reaches the caller's handler unchanged."""
    games = explore_games(tmp_path, 2)

    def interrupted():
        yield games[0]
        raise KeyboardInterrupt

    ts = TrainingSet()
    with pytest.raises(KeyboardInterrupt):
        ts.extend(interrupted())
    assert ts.games == 0
    with pytest.raises(ValueError, match="game 0: .*missing") as e:
        ts.extend([{"seed": 0}])
    assert isinstance(e.value.__cause__, ValueError), "the original error is the cause"


def test_a_truncated_record_file_is_an_error_and_adds_nothing(tmp_path):
    explore_games(tmp_path, 3)
    whole = (tmp_path / "x.jsonl.gz").read_bytes()
    cut = tmp_path / "cut.jsonl.gz"
    cut.write_bytes(whole[: 2 * len(whole) // 3])
    ts = TrainingSet()
    with pytest.raises(ValueError, match="EOFError") as e:
        ts.extend(records.read(cut))
    assert isinstance(e.value.__cause__, EOFError)
    assert ts.games == 0
