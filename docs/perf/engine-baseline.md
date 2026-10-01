# Engine performance baseline

Machine: Apple M1 Pro, 10 cores
Commit: `random_game` and throughput figures measured at 255cd18 plus the then-uncommitted bench files (no engine changes since); `clone_state` and `legal_actions_midgame` re-measured on the working tree on top of 946414a that became the "rich Main-phase position" commit
Config: default rules, domestic trades disabled (`max_offers_per_turn = 0`)

| Measurement | Result | Target |
|---|---|---|
| `clone_state` | 11.654 ns | < 100 ns |
| `legal_actions_midgame` | 339.59 ns (`Phase::Main`, 8 legal actions) | — |
| `random_game` | 145.43 µs | — |
| Single-thread games/s | 6887 | ≥ 1,000 |
| Actions per game | 1100 | — |
| All-thread games/s | 51995 (10 threads) | — |

All targets met; no "Missed targets" section is needed.

Note on `legal_actions_midgame`: the position is found by playing random legal actions from seed 7
until the phase is `Phase::Main` with at least 8 legal actions including at least one build or
maritime action (the bench asserts this, trying further seeds if needed, and prints the position).
The chosen position is seed 7 at step 411: `Phase::Main`, 8 legal actions = 6 roads, 1 settlement,
0 cities, 0 maritime, 0 dev, 1 other (EndTurn). Move generation is therefore about 340 ns for this
position, compared with 3.2 ns for a `PreRoll` position and 10.7 ns for a 2-action `Main` position
that earlier versions of this bench measured. As an extra whole-game figure, `random_game` is
145.43 µs / 1100 actions, about 132 ns per generate-choose-apply step.

## Updates

- 2026-09-30, incremental longest road (final-review fix): `cargo run --release -p settler-engine
  --example throughput 10000`, same machine and config. Single-thread games/s 9686 (was 6866 on
  the commit just before the change, measured interleaved with the new build over 5 runs each:
  6822-6871 before vs 9674-9716 after); actions per game unchanged at 1100; all-thread games/s
  67796 (10 threads). The table above keeps the original baseline.

Catanatron side-by-side comparison: Plan 3.
