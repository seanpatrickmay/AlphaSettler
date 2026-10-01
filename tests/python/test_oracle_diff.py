import copy
import json
import os
import subprocess
import sys

import pytest

pytest.importorskip("catanatron")

from oracle import allowlist  # noqa: E402
from oracle.diff import GameResult, format_summary, run, run_game, summarize  # noqa: E402

from catanatron import Color, Game as CatanGame, RandomPlayer  # noqa: E402
from catanatron.models.enums import ActionType  # noqa: E402
from oracle.translate import RESOURCE_NAMES, build_road, catan_snapshot  # noqa: E402


def test_thousand_games_match_up_to_allowlisted_differences():
    results = run(range(1000))
    s = summarize(results)
    detail = "\n".join(f"seed {r.seed}: {r.mismatch}\n  " + "\n  ".join(r.trace[-5:]) for r in s.mismatches[:3])
    assert not s.mismatches, detail
    assert s.allowlisted.keys() <= allowlist.ENTRIES.keys()
    assert s.games == 1000
    assert s.steps > 500_000
    assert s.clean_games > 600  # most games need no allowlist entry at all
    assert "games" in format_summary(s)


def test_runs_are_reproducible_across_worker_counts():
    key = lambda rs: [(r.seed, r.steps, sorted(r.allowlisted.items()), r.mismatch) for r in rs]
    assert key(run(range(30), workers=2)) == key(run(range(30), workers=3))


def test_every_allowlist_entry_is_documented():
    assert allowlist.ENTRIES
    for key, entry in allowlist.ENTRIES.items():
        assert entry.id == key
        assert entry.rule.strip() and entry.reason.strip(), key


TURN_LIMIT_GAME = """
import json
import catanatron.game
catanatron.game.TURNS_LIMIT = 40
from oracle.diff import run_game
r = run_game(11)
print(json.dumps({"mismatch": r.mismatch, "final_phase": r.final_phase, "winner": r.winner}))
"""


def test_turn_limit_draws_match():
    # A child with PYTHONHASHSEED=0, like the run workers, so the game is reproducible.
    r = subprocess.run([sys.executable, "-c", TURN_LIMIT_GAME], capture_output=True, text=True,
                       env={**os.environ, "PYTHONHASHSEED": "0"})
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout)
    assert out["mismatch"] is None, out["mismatch"]
    assert out["final_phase"] == "game_over" and out["winner"] is None


def _start() -> dict:
    return catan_snapshot(CatanGame([RandomPlayer(c) for c in [Color.RED, Color.BLUE, Color.ORANGE, Color.WHITE]],
                                    seed=1))


def test_bank_shortage_classifier():
    before = _start()
    board = before["board"]
    t = next(t for t in range(19) if board["tile_resource"][t] and t != before["robber"])
    roll, r = board["tile_number"][t], RESOURCE_NAMES.index(board["tile_resource"][t])
    a, b = sorted(allowlist._TILE_NODES[t])[:2]
    before["players"][0]["cities"] = [a]          # seat 0 is owed 2
    before["bank"][r] = 1                          # the bank has 1
    theirs = copy.deepcopy(before)                 # Catanatron pays nobody
    ours = copy.deepcopy(before)                   # the rulebook pays the single claimant what is left
    ours["bank"][r], ours["players"][0]["hand"][r] = 0, 1
    diffs = ["bank", "players[0].hand"]
    assert allowlist.classify_state(diffs, ActionType.ROLL, before, ours, theirs, roll=roll) == "bank-shortage"
    assert allowlist.classify_state(diffs, ActionType.END_TURN, before, ours, theirs, roll=roll) is None
    wrong = copy.deepcopy(ours)                    # paid to the wrong seat
    wrong["players"][0]["hand"][r], wrong["players"][1]["hand"][r] = 0, 1
    assert allowlist.classify_state(["bank", "players[1].hand"], ActionType.ROLL, before, wrong, theirs, roll=roll) is None
    two = copy.deepcopy(before)                    # a second claimant: the rulebook pays nobody either
    two["players"][1]["settlements"] = [b]
    assert allowlist.classify_state(diffs, ActionType.ROLL, two, ours, theirs, roll=roll) is None


def _trail(n: int) -> list[int]:
    """Edges of an n-road trail without revisiting a node."""
    def go(v, used, nodes):
        if len(used) == n:
            return used
        for e, u in allowlist._NODE_EDGES[v]:
            if u not in nodes:
                found = go(u, used + [e], nodes | {u})
                if found:
                    return found
        return None
    return next(t for v in sorted(allowlist._NODE_EDGES) if (t := go(v, [], {v})))


def test_longest_road_classifier_rejects_a_wrong_owner():
    before = _start()
    before["players"][0]["roads"] = sorted(_trail(4))
    ours = copy.deepcopy(before)
    ours["players"][0]["roads"] = sorted(_trail(5))
    theirs = copy.deepcopy(ours)                   # Catanatron (its cache) leaves the award unowned
    ours["longest_road_owner"] = 0                 # the rulebook gives seat 0 its 5-road trail
    d = ["longest_road_owner"]
    assert allowlist.classify_state(d, ActionType.BUILD_ROAD, before, ours, theirs) == "longest-road"
    for wrong in (1, 2, 3):
        bad = {**ours, "longest_road_owner": wrong}
        assert allowlist.classify_state(d, ActionType.BUILD_ROAD, before, bad, theirs) is None, wrong
    assert allowlist.classify_state(d, ActionType.END_TURN, before, ours, theirs) is None
    # Catanatron ends the game on its award; ours must still be in the phase the action leads to.
    before = _start()
    before.update(phase="main", current=0, setup_step=8)
    before["players"][0]["cities"] = [0, 2, 4, 6]  # 8 VP
    ours = copy.deepcopy(before)                   # the rulebook: no award, play goes on in main
    theirs = copy.deepcopy(before)                 # Catanatron awards the road (stale cache): 10 VP
    theirs.update(phase="game_over", winner=0, longest_road_owner=0)
    d = ["phase", "longest_road_owner"]
    for action in (ActionType.BUILD_ROAD, ActionType.BUILD_SETTLEMENT):
        assert allowlist.classify_state(d, action, before, ours, theirs) == "longest-road"
        for wrong in ("pre_roll", "discard", "setup_settlement"):
            bad = {**ours, "phase": wrong}
            assert allowlist.classify_state(d, action, before, bad, theirs) is None, (action, wrong)
    # A Road Building road: ours places the next free road or returns to the phase it came from.
    rb = {**before, "phase": "road_building", "roads_left": 2, "return_phase": "main"}
    nxt = {**ours, "phase": "road_building", "roads_left": 1, "return_phase": "main"}
    for ok in (nxt, {**ours, "phase": "main"}):
        assert allowlist.classify_state(d, ActionType.BUILD_ROAD, rb, ok, theirs) == "longest-road"
    for bad in ({**nxt, "roads_left": 2}, {**ours, "phase": "pre_roll"}, {**nxt, "return_phase": "pre_roll"}):
        assert allowlist.classify_state(d, ActionType.BUILD_ROAD, rb, bad, theirs) is None, bad


def test_win_off_turn_classifier_checks_action_and_road_owner():
    before = _start()
    before["phase"], before["current"] = "main", 0
    before["players"][1]["cities"] = [0, 2, 4, 6]  # 8 VP; Longest Road would make 10
    ours = copy.deepcopy(before)                   # the rulebook: nobody has 5 roads, nobody wins
    theirs = copy.deepcopy(before)                 # Catanatron hands seat 1 the award after a cut
    theirs.update(phase="game_over", winner=1, longest_road_owner=1)
    d = ["phase", "longest_road_owner"]
    assert allowlist.classify_state(d, ActionType.BUILD_SETTLEMENT, before, ours, theirs) == "win-off-turn"
    assert allowlist.classify_state(d, ActionType.BUILD_ROAD, before, ours, theirs) is None
    assert allowlist.classify_state(d, ActionType.ROLL, before, ours, theirs, roll=8) is None
    wrong = {**ours, "longest_road_owner": 2}      # ours awards a road nobody has
    assert allowlist.classify_state(d, ActionType.BUILD_SETTLEMENT, before, wrong, theirs) is None
    for phase in ("pre_roll", "discard", "setup_settlement"):  # a settlement leaves ours in main
        bad = {**ours, "phase": phase}
        assert allowlist.classify_state(d, ActionType.BUILD_SETTLEMENT, before, bad, theirs) is None, phase


def test_road_from_opponent_building_needs_the_geometry():
    before = _start()
    before["phase"], before["current"] = "main", 0
    a, b = _trail(2)                               # trail x -a- v -b- u
    v = next(n for n in allowlist._EDGE_NODES[a] if n in allowlist._EDGE_NODES[b])
    before["players"][0]["roads"] = [a]
    before["players"][1]["settlements"] = [v]
    tok = build_road(b)
    assert allowlist.classify_legal(tok, before, 0, "catanatron") == "road-from-opponent-building"
    assert allowlist.classify_legal(tok, before, 0, "ours") is None
    no_road = copy.deepcopy(before)                # the actor has no road at the opponent's node
    no_road["players"][0]["roads"] = []
    assert allowlist.classify_legal(tok, no_road, 0, "catanatron") is None
    own = copy.deepcopy(before)                    # no opponent building there: ours would allow it
    own["players"][1]["settlements"] = []
    assert allowlist.classify_legal(tok, own, 0, "catanatron") is None


def test_year_of_plenty_one_card_needs_a_short_bank_and_a_playable_card():
    before = _start()
    before["phase"], before["current"] = "main", 0
    before["players"][0]["dev_hand"][3] = 1
    before["bank"] = [1, 5, 5, 5, 5]
    tok = ("year_of_plenty_one_card", 0)
    assert allowlist.classify_legal(tok, before, 0, "catanatron") == "year-of-plenty-one-card"
    full = {**before, "bank": [5] * 5}             # the bank can pay every pair
    assert allowlist.classify_legal(tok, full, 0, "catanatron") is None
    new = copy.deepcopy(before)                    # bought this turn: not playable
    new["players"][0]["dev_new"][3] = 1
    assert allowlist.classify_legal(tok, new, 0, "catanatron") is None


def test_exceptions_inside_a_game_become_mismatches(monkeypatch):
    import oracle.diff

    def boom(*_):
        raise ValueError("boom")

    monkeypatch.setattr(oracle.diff, "compare", boom)
    r = run_game(0)
    assert r.mismatch == "step 1: ValueError: boom"
    assert len(r.trace) == 1


class PanicException(BaseException):
    """Stands in for PyO3's PanicException, which derives from BaseException, not Exception."""


def test_engine_panics_become_mismatches_but_interrupts_stop_the_run(monkeypatch):
    import oracle.diff

    def panic(*_):
        raise PanicException("index out of bounds")

    monkeypatch.setattr(oracle.diff, "compare", panic)
    assert run_game(0).mismatch == "step 1: PanicException: index out of bounds"

    def interrupt(*_):
        raise KeyboardInterrupt

    monkeypatch.setattr(oracle.diff, "compare", interrupt)
    with pytest.raises(KeyboardInterrupt):
        run_game(0)


def test_run_leaves_the_parent_environment_alone(monkeypatch):
    monkeypatch.delenv("PYTHONHASHSEED", raising=False)
    run(range(2), workers=1)
    assert "PYTHONHASHSEED" not in os.environ
    monkeypatch.setenv("PYTHONHASHSEED", "7")
    run(range(1), workers=1)
    assert os.environ["PYTHONHASHSEED"] == "7"


def test_run_starts_no_more_workers_than_seeds(monkeypatch):
    import multiprocessing

    import oracle.diff

    sizes, spawn = [], multiprocessing.get_context("spawn")

    class Recording:
        def Pool(self, n):
            sizes.append(n)
            return spawn.Pool(n)

    monkeypatch.setattr(oracle.diff.mp, "get_context", lambda kind: Recording())
    assert [r.seed for r in run(range(2), workers=8)] == [0, 1]
    assert sizes == [2]


def test_format_summary_counts_games_per_entry():
    from collections import Counter

    rs = [GameResult(0, 10, Counter({"longest-road": 3})), GameResult(1, 10, Counter({"longest-road": 1})),
          GameResult(2, 10)]
    text = format_summary(summarize(rs))
    assert "3 games, 30 steps compared, 1 games identical end to end, 0 mismatches" in text
    assert "allowlisted longest-road: 4 events in 2 games (Longest road length and award)" in text


def test_cli_oracle_diff_small_run(tmp_path):
    out = tmp_path / "diff.jsonl"
    r = subprocess.run([sys.executable, "-m", "alphasettler", "oracle-diff", "--games", "8", "--workers", "2",
                        "--out", str(out)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "8 games" in r.stdout and "0 mismatches" in r.stdout
    lines = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(lines) == 8
    for rec in lines:
        assert set(rec) == {"seed", "steps", "allowlisted", "mismatch", "unimported_steps", "ended_unimported",
                            "final_phase", "winner"}
        assert rec["final_phase"] == "game_over" and rec["unimported_steps"] == 0 and rec["ended_unimported"] is False
    assert any(rec["winner"] is not None for rec in lines)


TAKE_ONE_CARD_YOP = """
import json
from catanatron.models.enums import ActionType
from catanatron.models.player import RandomPlayer
from oracle.diff import run_game

taken = []
default = RandomPlayer.decide

def decide(self, game, playable_actions):
    one = [a for a in playable_actions if a.action_type == ActionType.PLAY_YEAR_OF_PLENTY and len(a.value) == 1]
    if one:
        taken.append(repr(one[0]))
        return one[0]
    return default(self, game, playable_actions)

RandomPlayer.decide = decide
r = run_game(348)
print(json.dumps({"taken": taken, "mismatch": r.mismatch, "allowlisted": dict(r.allowlisted)}))
"""


def test_executed_one_card_year_of_plenty_is_allowlisted():
    # Seed 348 offers a one-card Year of Plenty under PYTHONHASHSEED=0 (Catanatron's move order
    # depends on the hash seed). Random players rarely take it, so this player always does.
    import json
    import os

    r = subprocess.run([sys.executable, "-c", TAKE_ONE_CARD_YOP], capture_output=True, text=True,
                       env={**os.environ, "PYTHONHASHSEED": "0"})
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout)
    assert out["taken"], "seed 348 no longer offers a one-card Year of Plenty"
    assert out["mismatch"] is None, out["mismatch"]
    assert out["allowlisted"]["year-of-plenty-one-card"] > len(out["taken"])  # offers plus executions


def test_cli_oracle_diff_refuses_existing_output_before_playing(tmp_path):
    out = tmp_path / "diff.jsonl"
    out.write_text("keep\n")
    r = subprocess.run([sys.executable, "-m", "alphasettler", "oracle-diff", "--games", "2", "--out", str(out)],
                       capture_output=True, text=True)
    assert r.returncode == 2
    assert "error:" in r.stderr and "games" not in r.stdout
    assert out.read_text() == "keep\n"


def test_cli_oracle_diff_reports_mismatches_with_exit_1(monkeypatch, capsys):
    import oracle.diff
    from alphasettler import cli

    bad = GameResult(7, 3, mismatch="step 3 after X: ['bank']", trace=[f"record {i}" for i in range(8)])
    monkeypatch.setattr(oracle.diff, "run", lambda seeds, workers=None: [bad])
    assert cli.main(["oracle-diff", "--games", "1"]) == 1
    out, err = capsys.readouterr()
    assert "1 games, 3 steps compared, 0 games identical end to end, 1 mismatches" in out
    assert "mismatch seed 7: step 3 after X: ['bank']" in err
    assert "    record 3" in err and "    record 7" in err and "record 2" not in err


# Seeds the first 50,000-game run reported as mismatches. Each traced to a documented
# difference: two players reaching 10 at once (Catanatron names the last seat), Catanatron
# ending the game on a Road Building road, and Road Building offered only through an
# opponent's settlement. They must now pass, classified, with no mismatch.
FIFTY_K_SEEDS = {
    3770: "win-off-turn", 3955: "win-off-turn", 8872: "win-off-turn", 26245: "win-off-turn",
    32447: "win-off-turn", 48564: "win-off-turn",
    10999: "longest-road", 14670: "longest-road", 15347: "longest-road", 46272: "longest-road",
    48328: "longest-road",
    33206: "road-from-opponent-building",
}


def test_fifty_k_mismatch_seeds_are_explained():
    results = {r.seed: r for r in run(list(FIFTY_K_SEEDS), workers=4)}
    assert [(seed, r.mismatch) for seed, r in results.items() if r.mismatch] == []
    for seed, kind in FIFTY_K_SEEDS.items():
        assert results[seed].allowlisted[kind] > 0, (seed, results[seed].allowlisted)
    # Seed 33206: Road Building whose only edge runs through an opponent's settlement cannot be
    # imported; Catanatron places that road, and the one step is reported as not compared.
    assert results[33206].unimported_steps == 1
    assert all(r.unimported_steps == 0 for seed, r in results.items() if seed != 33206)


def test_an_unimportable_position_after_other_entries_is_a_mismatch(monkeypatch):
    import oracle.diff as diff

    monkeypatch.setattr(diff, "compare", lambda ours, theirs: ["players[0].hand"])
    monkeypatch.setattr(diff.allowlist, "classify_state", lambda *a, **k: "bank-shortage")
    monkeypatch.setattr(diff, "_import", lambda seed, snap, cfg: (None, "cannot express this"))
    r = diff.run_game(5)
    assert r.mismatch is not None and "cannot import Catanatron's position: cannot express this" in r.mismatch
    assert r.unimported_steps == 0


def test_play_road_building_classifier_needs_no_rulebook_edge():
    snap = catan_snapshot(CatanGame([RandomPlayer(c) for c in [Color.RED, Color.BLUE, Color.ORANGE, Color.WHITE]], seed=2))
    snap.update(phase="main", setup_step=8, current=0)
    me, opp = snap["players"][0], snap["players"][1]
    me["dev_hand"][2] = 1
    # Our road a-b ends at b, where an opponent has a settlement; the only edge beyond b is b-c.
    a, b, c = next((a, b, c) for e1, (a, b) in enumerate(allowlist._EDGE_NODES)
                   for e2, (x, y) in enumerate(allowlist._EDGE_NODES)
                   if e1 != e2 and b in (x, y) for c in [y if x == b else x] if c != a)
    e_ab = allowlist._EDGE_NODES.index((a, b)) if (a, b) in allowlist._EDGE_NODES else allowlist._EDGE_NODES.index((b, a))
    me.update(roads=[e_ab], settlements=[], cities=[])
    opp.update(settlements=[b], cities=[], roads=[])
    for p in snap["players"][2:]:
        p.update(settlements=[], cities=[], roads=[])
    # a has another free edge (every node has degree >= 2), so the rulebook lets us build there:
    # a missing Road Building on our side would be a real bug, not this entry.
    assert allowlist._rulebook_free_edges(snap, 0)
    assert allowlist.classify_legal(4, snap, 0, "catanatron") is None
    blocked = copy.deepcopy(snap)
    # Occupy every edge at a other than a-b, so a has nowhere to go; only b-c (through b) remains.
    others = [e for e, (x, y) in enumerate(allowlist._EDGE_NODES) if a in (x, y) and e != e_ab]
    blocked["players"][2]["roads"] = others
    assert not allowlist._rulebook_free_edges(blocked, 0)
    assert allowlist.classify_legal(4, blocked, 0, "catanatron") == "road-from-opponent-building"
    assert allowlist.classify_legal(4, blocked, 0, "ours") is None
    no_card = copy.deepcopy(blocked)
    no_card["players"][0]["dev_hand"][2] = 0
    assert allowlist.classify_legal(4, no_card, 0, "catanatron") is None


def test_win_off_turn_when_both_games_end_names_the_rulebook_winner():
    snap = catan_snapshot(CatanGame([RandomPlayer(c) for c in [Color.RED, Color.BLUE, Color.ORANGE, Color.WHITE]], seed=1))
    snap.update(phase="game_over", current=0, setup_step=8)
    snap["players"][0]["dev_hand"][1] = 10  # seat 0 at 10 VP through victory point cards
    snap["players"][3]["dev_hand"][1] = 10
    ours, theirs = copy.deepcopy(snap), copy.deepcopy(snap)
    ours["winner"], theirs["winner"] = 0, 3
    kind = allowlist.classify_state(["winner"], ActionType.BUILD_SETTLEMENT, snap, ours, theirs)
    assert kind == "win-off-turn"
    wrong_ours = copy.deepcopy(ours)
    wrong_ours["winner"] = 3  # ours naming an off-turn winner is a bug, not this entry
    assert allowlist.classify_state(["winner"], ActionType.BUILD_SETTLEMENT, snap, wrong_ours, theirs) is None
    not_last = copy.deepcopy(theirs)
    not_last["players"][3]["dev_hand"][1] = 0
    not_last["players"][2]["dev_hand"][1] = 10
    assert allowlist.classify_state(["winner"], ActionType.BUILD_SETTLEMENT, snap, ours, not_last) is None
    assert allowlist.classify_state(["winner"], ActionType.ROLL, snap, ours, theirs) is None
    assert allowlist.classify_state(["winner"], ActionType.BUILD_ROAD, snap, ours, theirs) is None
    earlier = copy.deepcopy(theirs)
    earlier["players"][1]["dev_hand"][1] = 10  # seats 0, 1 and 3 at 10
    earlier["winner"] = 1                      # at 10, but not the last such seat
    assert allowlist.classify_state(["winner"], ActionType.BUILD_SETTLEMENT, snap, ours, earlier) is None
