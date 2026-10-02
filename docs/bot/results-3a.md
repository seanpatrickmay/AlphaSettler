# Sub-project 3a (ISMCTS search bot): results

Date: 2026-10-02. Machine: Apple M1 Pro, 10 cores, macOS (Darwin 25.5.0). Python 3.12.6,
rustc 1.98.1.

AlphaSettler: 33785ba (the bot as shipped, after the final-review fixes), plus the commit that
updates this file and `docs/perf/search.md`. Neither of those changes the bot.
Catanatron 3.3.0 at git commit ecf931181b9a65bb4116a2153fb78c16f1438e00.

Spec: [`docs/superpowers/specs/2026-10-01-bot-3a-ismcts-design.md`](../superpowers/specs/2026-10-01-bot-3a-ismcts-design.md).

## Search speed

About 700k simulations/s on one core, or 1.4 ms per decision at 1,000 simulations. Three runs at
33785ba gave 701,403, 690,690 and 702,146 simulations/s. The spec estimated 50k–100k. Details are
in [`docs/perf/search.md`](../perf/search.md).

## Configuration

`ismcts` means 1,000 simulations, c_puct 1.5, batch 8, rollout 0. The belief is exact, with a cap
of 65,536 states. Worlds condition every opponent's hidden VP cards below the win. The heuristic
weights are the hand-set ones the bot was built with:

```
pub const DEFAULT_WEIGHTS: [f32; NUM_FEATURES] = [1.0, 0.08, 0.05, -0.1, 0.3, 0.05, 0.1, 0.2];
```

In feature order these are vp, production, hand, over_limit, dev_cards, road_length, knights and
settlement_spots.

## Round 0: the bot as built

Round 0 met the headline, so no tuning round was run. The plan says to stop once the headline is
met, and the spec allows tuning rounds only if it is missed. Weights, fit and rollout are therefore
unchanged from the build.

Each seed is 4 games, with the candidate rotated through every seat against three copies of the
baseline. The null hypothesis is a 0.25 win rate. The native runs use `--no-trades`. The Catanatron
runs use the arena's default config, which already has domestic trades off. Native runs have no
fallbacks, because a bot that picks an illegal move panics.

| Command | Summary line | Wall time | Fallbacks | Belief resets |
|---|---|---|---|---|
| `arena --candidate ismcts@100 --baseline greedy --seeds 200 --no-trades` | win rate 0.330 [0.308, 0.352] over 200 seeds (800 games, 264 wins, 0 draws), z = 6.99, mean VP 7.11, mean turns 102.7 | 1.7 s | n/a | 0 |
| `arena --candidate ismcts@300 --baseline greedy --seeds 200 --no-trades` | win rate 0.388 [0.360, 0.415] over 200 seeds (800 games, 310 wins, 0 draws), z = 9.87, mean VP 7.46, mean turns 101.7 | 3.4 s | n/a | 0 |
| `arena --candidate ismcts@1000 --baseline greedy --seeds 200 --no-trades` | win rate 0.443 [0.411, 0.474] over 200 seeds (800 games, 354 wins, 0 draws), z = 12.02, mean VP 7.75, mean turns 100.3 | 11.3 s | n/a | 0 |
| `arena --candidate ismcts@3000 --baseline greedy --seeds 200 --no-trades` | win rate 0.435 [0.403, 0.467] over 200 seeds (800 games, 348 wins, 0 draws), z = 11.44, mean VP 7.83, mean turns 101.1 | 34.9 s | n/a | 0 |
| `arena --candidate ismcts --baseline catanatron:value --seeds 200` | win rate 0.451 [0.417, 0.485] over 200 seeds (800 games, 361 wins, 0 draws), z = 11.55, mean VP 7.63, mean turns 85.1 | 47.3 s | 0 | 0 |
| `arena --candidate ismcts --baseline catanatron:alphabeta --seeds 200` | win rate 0.383 [0.347, 0.418] over 200 seeds (800 games, 306 wins, 0 draws), z = 7.41, mean VP 7.29, mean turns 82.6 | 20:40 | 0 | 0 |

The records are in `runs/p4/f0-*.jsonl`, which are gitignored.

### Before the final-review fixes

The same six runs at 12dfc02 came before the final review. Back then, worlds could deal an off-turn
opponent hidden VP cards worth a win. The 12dfc02 results were:

| Run | 12dfc02 | 33785ba |
|---|---|---|
| ismcts@100 vs greedy | 0.331, z = 7.25 | 0.330, z = 6.99 |
| ismcts@300 vs greedy | 0.381, z = 9.73 | 0.388, z = 9.87 |
| ismcts@1000 vs greedy | 0.426, z = 11.35 | 0.443, z = 12.02 |
| ismcts@3000 vs greedy | 0.425, z = 10.77 | 0.435, z = 11.44 |
| ismcts vs catanatron:value | 0.458, z = 12.01 | 0.451, z = 11.55 |
| ismcts vs catanatron:alphabeta | 0.389, z = 7.65 | 0.383, z = 7.41 |

These are unpaired runs, so the differences between the two columns are all within noise. Each
cell's CI is about ±0.03. Those runs also had 0 fallbacks and 0 belief resets against Catanatron
(`runs/p4/r0.log`).

## Strength curve against GreedyBot (final configuration)

| Simulations | Win rate [95% CI] | z vs 0.25 |
|---|---|---|
| 100 | 0.330 [0.308, 0.352] | 6.99 |
| 300 | 0.388 [0.360, 0.415] | 9.87 |
| 1,000 | 0.443 [0.411, 0.474] | 12.02 |
| 3,000 | 0.435 [0.403, 0.467] | 11.44 |

Win rate goes up at each step to 1,000 simulations. The two steps are about 0.06 each, with an
unpaired z of about 2.2–2.4, and no paired test was run. After that it stops rising: 3,000 is level
with 1,000. This happened in both rounds. The plateau is consistent with the hand-set heuristic
evaluator being the limit. That was not tested, since no tuning round ran, and determinization
effects or c_puct could also contribute. A learned value and prior are 3b's job.

## Headline

**Met.** At 1,000 simulations, IsmctsBot beats both of Catanatron's strongest bots by z > 3 over
200 seeds:

| Baseline | IsmctsBot | GreedyBot (for scale, [`docs/oracle/results.md`](../oracle/results.md)) |
|---|---|---|
| `catanatron:value` | 0.451, z = 11.55 | 0.185 (100 seeds) |
| `catanatron:alphabeta` | 0.383, z = 7.41 | 0.102 (100 seeds) |

AlphaBeta searches against a wall-clock limit, so its games are not guaranteed to be reproducible
and depend on machine load. All runs above were sequential on an otherwise idle machine. An earlier
20-seed reading at 5b534c8 (`runs/p4-early/`) gave 0.375, which agrees.

## Spec Section 5 done criteria

- [x] **1. All correctness tests pass, and the Catanatron arena records 0 fallbacks and 0 belief
  resets.** Both Catanatron runs above print `fallbacks: 0` and `belief resets: 0`. Each spec test
  maps to these tests:
  - Belief, exact match against a brute-force posterior: `search/tests/belief.rs`
    `hand_built_histories_match_the_exact_posterior` and
    `random_histories_match_the_exact_posterior`. They are held to 1e-9, tighter than the spec's 0.02.
  - Belief, the true hands never ruled out and every state matching `hand_counts`:
    `the_full_log_reproduces_every_hand`, and
    `every_state_matches_the_visible_counts_and_the_truth_stays_possible`.
  - Sampled worlds: `search/tests/world.rs`
    `sampled_worlds_are_valid_and_look_exactly_like_the_observation`.
  - Search mechanics:
    - Availability: `search/src/puct.rs` `exploration_grows_with_availability` and
      `children_not_allowed_in_this_world_are_skipped`.
    - Max^n backup: `search/src/puct.rs` `each_player_maximises_their_own_value` (selection), and
      `search/tests/search.rs` `batched_search_leaves_no_virtual_loss_and_backs_up_whole_value_vectors`
      (every child's 4-vector value sum equals its visits) and
      `terminal_values_are_one_hot_for_a_winner_and_even_at_the_turn_cap`.
    - Virtual loss: `search/src/puct.rs` `pending_visits_lower_the_score`, and
      `batched_search_leaves_no_virtual_loss_and_backs_up_whole_value_vectors` (after a batch-8
      search of 2,500 simulations no child keeps a pending visit and no node awaits evaluation).
    - Public chance branches: `search/tests/search.rs` `dice_branch_into_chance_outcomes`.
    - Hidden chance does not branch: `search/tests/search.rs` `hidden_chance_makes_no_chance_node`
      (an opponent's dev-card purchase is a decision node, the viewer's own a chance node) and
      `which_chance_outcomes_the_viewer_sees`.
  - Evaluator contract: `bots/tests/heuristic.rs` `the_prior_depends_only_on_what_the_actor_sees`.
  - Tactics:
    - `search/tests/search.rs` `takes_a_winning_build`.
    - `bots/tests/ismcts.rs` `builds_an_affordable_city_rather_than_ending_the_turn` and
      `never_robs_its_own_buildings_when_an_opponent_tile_is_free`.
  - Determinism:
    - `search/tests/search.rs` `the_same_seed_gives_the_same_search`.
    - `bots/tests/ismcts.rs` `arena_results_do_not_depend_on_thread_count`.
  - Catanatron event feed: `tests/python/test_oracle_events.py`
    `test_translated_events_equal_our_log_in_clean_games`.
- [x] **2. Search throughput is recorded in `docs/perf/`.** See
  [`docs/perf/search.md`](../perf/search.md).
- [x] **3. Strength curve at 100, 300, 1,000 and 3,000 simulations against GreedyBot, each z > 3.**
  The lowest is z = 6.99, at 100 simulations.
- [x] **4. Headline: `ismcts` (1,000 simulations) beats `catanatron:alphabeta` and
  `catanatron:value` with z > 3 over 200 seeds.** The results are z = 7.41 and z = 11.55.
- [x] **5. `alphasettler selfplay` writes records that load in Python, and seed plus actions
  reproduce every stored observation exactly.** Tested by these:
  - `tests/python/test_selfplay.py` `test_records_round_trip_and_replay_exactly`.
  - `tests/python/test_cli.py` `test_cli_selfplay_streams_batches_and_replays`.
  - `tests/python/test_selfplay.py` `test_replay_detects_a_tampered_record`, which shows the replay check is not vacuous.
  - `bots/tests/selfplay.rs` `every_searched_decision_is_recorded_and_replays`.

## Tests

These ran on the final tree, after all the runs above:

- `cargo test --workspace --release`: 254 passed, 0 failed, 0 ignored, across 33 test targets in
  engine, search, bots and bindings.
- `.venv/bin/maturin develop --release && .venv/bin/pytest -q`: 158 passed in 41 s.
- `cargo fmt --check` and `cargo clippy --workspace --release --all-targets`: clean.
- The golden trace (`engine/tests/golden_trace.rs`) has been unchanged since its one re-pin in this
  sub-project. That re-pin was for the per-victim `from` field on `MonopolyTaken`, which changes the
  logged events but not the states: a states-only hash stayed identical.
