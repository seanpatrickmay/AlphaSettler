# Engine performance baseline

Machine: Apple M1 Pro, 10 cores
Commit: 255cd18 plus the then-uncommitted bench files for `clone_state`, `random_game` and throughput; `legal_actions_midgame` re-measured after the position fix, on the working tree on top of bd4fe70
Config: default rules, domestic trades disabled (`max_offers_per_turn = 0`)

| Measurement | Result | Target |
|---|---|---|
| `clone_state` | 11.485 ns | < 100 ns |
| `legal_actions_midgame` | 10.744 ns (`Phase::Main`, 2 legal actions) | — |
| `random_game` | 145.43 µs | — |
| Single-thread games/s | 6887 | ≥ 1,000 |
| Actions per game | 1100 | — |
| All-thread games/s | 51995 (10 threads) | — |

All targets met; no "Missed targets" section is needed.

Note on `legal_actions_midgame`: the position is built by 400 random actions from seed 7, then random
steps until the phase is `Phase::Main` with more than one legal action (the bench asserts this and
prints the position). It landed 1 step later: `Phase::Main` with 2 legal actions, so the position
is a modest one. As an extra whole-game figure, `random_game` is 145.43 µs / 1100 actions, about
132 ns per generate-choose-apply step.

Catanatron side-by-side comparison: Plan 3.
