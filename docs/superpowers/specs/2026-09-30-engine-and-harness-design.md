# AlphaSettler — Sub-project 1+2: Engine & Benchmark Harness

Status: approved 2026-09-30

## Goal

AlphaSettler aims to be a Settlers of Catan bot stronger than any human, judged on
4-player base-game play. Human play happens only through a clearly labeled bot account
approved by the platform (see `outreach/colonist-email.md`); there is no undisclosed
ladder play.

The project is decomposed into:

1. **Engine** — fast, rule-verified Rust simulator (this spec)
2. **Benchmark harness** — arena + statistics to measure any bot (this spec)
3. **Bot** — search (determinized MCTS), learned value/policy, trade model (later spec)
4. **Colonist interface** — only if Colonist approves a labeled bot account (later spec)

This spec ends at: a fast, rule-checked engine plus a harness that can measure bot
strength with statistical confidence.

### Decisions

| Question | Decision |
|---|---|
| Target format | 4-player base game |
| Compute | Unknown — design for one Mac first, must scale out without rewrite |
| Engine | Own engine in Rust; Catanatron (pure Python, GPL-3.0) as rules oracle + benchmark arena only |
| Rules | Standard base rules by default; Colonist-specific differences as config flags once verified |
| Domestic trade | Full offer → accept/reject → confirm protocol, capped (≤ N offers/turn, ≤ K cards per side) |

## 1. Architecture

```
AlphaSettler/
  engine/        Rust crate: rules, state, legal moves (pure Rust, no Python)
  bindings/      PyO3 crate -> Python module `alphasettler._engine` (built with maturin)
  alphasettler/  Python package: arena, baseline bots, stats, CLI
  oracle/        Separate Python package: Catanatron adapter + differential tests
  outreach/      Colonist outreach draft
```

- `engine/` is the source of truth for rules. API: `new_game(seed, config)`,
  `legal_actions()`, `apply(action)`, `clone()`, `observation(player)`.
  Actions are integers in one fixed action space, shared by search and (later) the
  network's policy head.
- `bindings/` is a thin wrapper with no logic.
- `alphasettler/` runs bot-vs-bot games natively in Rust (fast), and runs our bots
  against Catanatron's bots inside Catanatron's engine via the adapter.
- `oracle/` is the only code that imports Catanatron. Installing `alphasettler` never
  pulls in Catanatron. The engine is written from the rulebook, never ported from
  Catanatron source.
- Hidden information is a first-class concept: the engine holds true state; bots only
  ever receive `observation(player)`. The engine's interface is designed so a later
  `sample_consistent_state(observation, rng)` (bot spec) fits without API changes.

Out of scope: neural nets, MCTS, trade model, Colonist interface.

## 2. Engine internals

### State

One fixed-size `Copy` struct, target < 512 bytes; `clone()` is a memcpy.

- **Topology** (19 tiles, 54 nodes, 72 edges, adjacency, port locations): compile-time
  constant tables, not stored in state.
- **Per-game board:** tile resources, number tokens, port types — small fixed arrays.
- **Per player:** settlements `u64` bitmask, cities `u64`, roads `u128`, hand `[u8; 5]`,
  dev cards (playable vs. bought-this-turn), knights played, cached longest-road length
  and VP.
- **Shared:** bank, robber tile, dev deck order + index, turn/phase, pending
  trade/discard state, RNG streams.

### Action space

Fixed enumeration, roughly 700 actions. Compound decisions are factored into sequential
sub-decisions to keep per-step branching small:

- Robber: choose tile, then choose victim.
- Discard: one card at a time.
- Maritime trade: (give r, get r′); the engine applies the best available rate.
- Domestic trade: offer (each side ≤ K cards) → each opponent accepts/rejects →
  offerer confirms with one acceptor, or cancels.

Longest road is updated incrementally (recomputed locally on road placement or when a
settlement breaks a road), not by full-board search.

### Randomness and CRN

Independent RNG streams so outcomes don't depend on bot choices:

- Dice: derived from `(seed, turn)`.
- Dev deck: shuffled once from the seed.
- Steals: their own stream.

Two different bots playing the same seed see identical boards and rolls (common random
numbers). Chance outcomes can also be supplied explicitly (forced), used for replaying
Catanatron games and, later, for search chance nodes.

### Observation

`observation(p)`: own exact hand, opponents' card counts, public event log. Steals reveal
only that a card moved between two players, not which card.

### Config

`GameConfig`: VP to win (default 10), discard limit (default 7), friendly robber (default off),
trade caps (defaults: N = 3 offers per turn, K = 2 cards per side),
`catanatron_compat` (accept forced random discards/steals as Catanatron does).

### Performance targets

Measured with `criterion`:

- `clone()` < 100 ns
- ≥ 1,000 complete random-play games/sec on a single core (domestic trades disabled for
  random play)
- Side-by-side games/sec against Catanatron on the same machine, recorded in the repo

## 3. Correctness and benchmark harness

### Correctness (three layers)

1. **Rule unit tests (Rust, TDD).** One test per easy-to-get-wrong rule: snake-order
   setup, second settlement pays starting resources, distance rule, road connectivity,
   longest road (including breaks by an opponent's settlement), largest army, dev card
   not playable the turn it's bought, one dev card per turn, hidden VP cards, discard
   above the limit, robber blocks production, bank shortage (if the bank can't pay every
   owed player a resource, no one receives it, unless only one player is owed), port
   rates, a player wins only on their own turn.
2. **Property tests (`proptest`) on random games.** Resource conservation (bank + all
   hands = 19 per resource), piece limits (5 settlements, 4 cities, 15 roads), every
   legal action applies without panic, VP matches board state, every game terminates.
3. **Differential testing against Catanatron.** Play random-bot games in Catanatron,
   record each action plus its chance outcome (dice, stolen card, random discard).
   Replay in our engine with outcomes forced; at every step compare the legal-action set
   (mapped between encodings) and the resulting state. Replaying recorded outcomes
   sidesteps Catanatron's unseeded global `random`. The default test suite runs 1,000
   games; a 50,000-game run is available on demand. Intentional rule differences go in
   a documented allowlist, each entry with its reason.

### Benchmark harness

- **Format:** one candidate vs. three copies of a baseline. Each seed is played four
  times with the candidate rotated through every seat.
- **CRN:** every candidate plays the same seed list (identical boards and dice). In the
  Catanatron arena, CRN covers the board and only the early rolls, since steals and
  discards share Catanatron's global random stream; paired comparisons there are weaker.
- **Statistics:** the unit is a seed (four rotations). Report win rate vs. the 25% null
  with a confidence interval. Comparing two candidates uses the paired per-seed
  difference: mean, standard error, z. Also report mean VP and game length.
- **Arenas:** native Rust arena (fast; our bots vs. our bots) and Catanatron arena (vs.
  their AlphaBeta and value-function bots, via `oracle/`).
- **Baseline bots (Rust):** `RandomBot` (uniform over legal actions; domestic offers
  disabled), `GreedyBot` (builds whenever possible using simple placement scoring; never
  offers domestic trades, rejects all offers).
- **Output:** one JSON line per game in `runs/` (gitignored) plus a printed summary
  table. CLI: `alphasettler arena --candidate X --baseline Y --seeds N`.

## 4. Done criteria

This sub-project is complete when:

- All rule unit tests and property tests pass.
- 1,000-game differential test against Catanatron passes with only allowlisted
  differences; the 50,000-game run has been executed once and passes.
- Performance targets in Section 2 are met (or the measured numbers and the reason
  they're missed are recorded).
- `alphasettler arena` runs `GreedyBot` vs. `RandomBot` natively and against Catanatron's
  AlphaBeta bot, and prints win rate with CI.
- `GreedyBot` beats `RandomBot` with z > 3 in the native arena (sanity check that the
  harness detects a real strength difference).
