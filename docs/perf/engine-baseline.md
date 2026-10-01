# Engine performance baseline

Machine: Apple M1 Pro, 10 cores
Commit: 255cd18
Config: default rules, domestic trades disabled (`max_offers_per_turn = 0`)

| Measurement | Result | Target |
|---|---|---|
| `clone_state` | 11.485 ns | < 100 ns |
| `legal_actions_midgame` | 3.2115 ns | — |
| `random_game` | 145.43 µs | — |
| Single-thread games/s | 6887 | ≥ 1,000 |
| Actions per game | 1100 | — |
| All-thread games/s | 51995 (10 threads) | — |

All targets met; no "Missed targets" section is needed.

Note on `legal_actions_midgame`: the position produced by 400 random actions from seed 7 is in
`Phase::PreRoll`, where the only legal actions are `Roll` (and dev-card plays), so 3.2 ns measures
a near-trivial call, not full main-phase move generation. Per-action cost across a whole game is
better read from `random_game`: 145.43 µs / 1100 actions is about 132 ns per
generate-choose-apply step.

Catanatron side-by-side comparison: Plan 3.
