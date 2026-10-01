# Differential test and Catanatron arena: results

Date: 2026-10-01. Machine: Apple M1 Pro, 10 cores, macOS (Darwin 25.5.0). Python 3.12.6,
rustc 1.98.1.

AlphaSettler: uncommitted working tree on top of b255370 (the last commit; Plan 2's and Plan 3's
remaining work was queued for commit while the author was away). Catanatron 3.3.0 at
git commit ecf931181b9a65bb4116a2153fb78c16f1438e00, installed through the `oracle` extra.

Parity config for the differential test (`oracle/diff.py`, `config()`):
`max_offers_per_turn = 0` (Catanatron's random player never offers domestic trades),
`catanatron_compat = False` (we play the rulebook; its differences are allowlisted),
`max_turns = 994` (Catanatron's `TURNS_LIMIT` 1000 minus its 6 setup turns). The Catanatron
arena uses `max_offers_per_turn = 0` with default rules.

Every number below comes from the final code, the tree these results ship with.

## Differential test

Catanatron random players play each game. At each step the two engines' legal action sets are
compared, Catanatron's action and its realized chance outcome are forced on ours, and then the
full positions are compared. After an allowlisted divergence, our engine re-imports Catanatron's
position and the comparison continues. The reason for each allowlist entry is in
[`oracle/allowlist.py`](../../oracle/allowlist.py).

| Run | Games | Steps compared | Identical end to end | Mismatches | Wall time |
|---|---|---|---|---|---|
| 1,000 (`alphasettler oracle-diff --games 1000 --workers 10`) | 1,000 | 1,082,142 | 747 | **0** | 23.7 s |
| 50,000 (`alphasettler oracle-diff --games 50000 --workers 10`) | 50,000 | 53,263,391 | 37,882 | **0** | 23:09 |

Allowlisted events (events / games they occur in):

| Entry | Rule | 1,000 | 50,000 |
|---|---|---|---|
| `longest-road` | Longest road length and award | 1,074 / 225 | 49,261 / 10,750 |
| `bank-shortage` | Production when the bank runs short | 33 / 28 | 1,948 / 1,475 |
| `road-from-opponent-building` | Building a road through an opponent's settlement | 13 / 7 | 706 / 235 |
| `win-off-turn` | Winning on another player's turn | 2 / 2 | 154 / 154 |
| `year-of-plenty-one-card` | Year of Plenty with a short bank | 4 / 1 | 74 / 30 |

In the 50,000-game run, 1 Catanatron step went uncompared: after a `road-from-opponent-building`
divergence, Catanatron was in Road Building with no rulebook edge for the player ("RoadBuilding, but
player 0 has nowhere to build a road"). This is the only tolerated reason, at most 2 steps in a row.
The 1,000-game run had none. The 50,000-game counts are identical to the first rerun after the
classifier fixes (before the last fix round), step for step.

### The first 50,000-game run

The first 50,000-game run, on code from before the last fixes, found 12 mismatches. All 12 were
gaps in the allowlist classifiers, not engine bugs. Each seed is now pinned to its entry in
`tests/python/test_oracle_diff.py` (`FIFTY_K_SEEDS`).

- 6 seeds (3770, 3955, 8872, 26245, 32447, 48564): a settlement takes two players past 10 VP
  at once, for example through a longest-road cut. Catanatron declares the last seat at 10 the
  winner. The `win-off-turn` classifier handled one player over, not both, so it now covers
  that case too.
- 5 seeds (10999, 14670, 15347, 46272, 48328): Catanatron ends the game partway through Road
  Building. Comparing `return_phase` there was a translation artifact. `compare` now checks it
  only when both phases agree.
- 1 seed (33206): Catanatron offered Road Building when its only free edge leads from an
  opponent's settlement. The `road-from-opponent-building` classifier now covers
  `PlayRoadBuilding` when our engine has no rulebook edge. Catanatron's resulting position
  cannot be expressed in our rules, so up to 2 of its steps go uncompared, with that reason
  only (`oracle/diff.py`, `MAX_UNIMPORTED`).

The rerun after these fixes gave 0 mismatches. The last fix round tightened the classifiers
further and added state invariants. The table above is a fresh run on that final code.

## Catanatron arena and native arena

GreedyBot (pip-count scoring) against three copies of each baseline. The candidate is rotated
through every seat, and 1 seed means 4 games. The null hypothesis is a 0.25 win rate. In the
Catanatron arena, `AlphaSettlerPlayer` imports Catanatron's position before each decision, and
the bot sees only its own observation. A "fallback" is a decision our side couldn't make
(Catanatron's first playable action is played instead).

| Baseline | Seeds (games) | Win rate [95% CI] | z vs 0.25 | Mean VP | Mean turns | Fallbacks | Wall time |
|---|---|---|---|---|---|---|---|
| native `random` | 2,000 (8,000) | 0.995 [0.993, 0.997] | 906.56 | 10.10 | 107.8 | n/a | 0.16 s |
| `catanatron:random` | 200 (800) | 0.996 [0.992, 1.000] | 346.42 | 10.10 | 109.8 | 0 | 2.0 s |
| `catanatron:value` | 100 (400) | 0.185 [0.147, 0.223] | -3.36 | 6.26 | 88.5 | 0 | 18.5 s |
| `catanatron:alphabeta` | 100 (400) | 0.102 [0.072, 0.133] | -9.50 | 5.89 | 85.5 | 0 | 16:12 |

Commands: `alphasettler arena --candidate greedy --baseline <baseline> --seeds <n>`. GreedyBot is
a harness sanity baseline, not a strong player, so losing to Catanatron's value-function and
AlphaBeta bots is expected. These numbers are the bar later sub-projects have to clear.
AlphaBeta searches with a wall-clock limit, so its games are not guaranteed to be reproducible.
Even so, runs before and after the last fix round gave identical summaries.

## Tests

Run on the final tree after all the runs above:

- `cargo test --workspace --release`: 186 passed, 0 failed, 0 ignored (24 test targets across engine, bots and bindings).
- `.venv/bin/maturin develop --release && .venv/bin/pytest -q`: 136 passed in 67 s. This includes the 1,000-game
  differential test, the oracle isolation tests (installing `alphasettler` never pulls in Catanatron, and only
  `oracle/` imports it) and the Catanatron arena tests.
- The golden trace (600 seeded games, `engine/tests/golden_trace.rs`) is unchanged since it was re-pinned for
  Plan 3's rule changes.

## Dependency audit

`pip-audit --skip-editable` in the oracle environment: "No known vulnerabilities found".
Catanatron itself is skipped because it is not on PyPI. It is pinned to the git commit above.
`cargo audit` on `Cargo.lock` (109 crates; the RustSec database had 1,278 advisories): no vulnerabilities.

The `oracle` extra pins Catanatron only, so its transitive dependencies are unpinned. These are
the versions of Catanatron's runtime dependency tree the runs above used (from `pip3 freeze`;
`rich` and `markdown-it-py` pull in the last three):

```
catanatron @ git+https://github.com/bcollazo/catanatron.git@ecf931181b9a65bb4116a2153fb78c16f1438e00
click==8.5.0
networkx==3.7
rich==15.0.0
markdown-it-py==4.2.0
mdurl==0.1.2
Pygments==2.21.0
```

## Spec Section 4 done criteria

- [x] **All rule unit tests and property tests pass.** See Tests above:
  `engine/tests/` (rules, golden trace, longest-road reference oracle, snapshots) and
  `engine/tests/properties.rs`, which checks `State::check_invariants` after every step of
  random games.
- [x] **The 1,000-game differential test passes with only allowlisted differences, and the
  50,000-game run has been executed once and passes.** The 1,000-game run is in the default
  suite (`tests/python/test_oracle_diff.py::test_thousand_games_match_up_to_allowlisted_differences`).
  Both runs' numbers are above.
- [x] **Performance targets in Section 2 are met.** `clone()` 11.7 ns against the < 100 ns
  target ([`docs/perf/engine-baseline.md`](../perf/engine-baseline.md)). 13,347 random games/s
  on a single core against the ≥ 1,000 target, at d140be6. The side-by-side comparison with
  Catanatron and catan-rl is in [`docs/perf/comparison.md`](../perf/comparison.md): with trades
  off, about 300× faster per step than Catanatron. Those tables were measured at d140be6, before Plan 3.
  Rechecked on the final tree with `cargo run --release -p settler-engine --example throughput 10000`:
  13,619 games/s and 66.8 ns/action on one thread (trades off). That matches d140be6's 13,347 games/s and 68.2 ns.
- [x] **`alphasettler arena` runs GreedyBot vs RandomBot natively and against Catanatron's
  AlphaBeta bot, and prints win rate with CI.** See the arena table above.
- [x] **GreedyBot beats RandomBot with z > 3 in the native arena.** z = 906.56 over 2,000 seeds.
