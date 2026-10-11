# Targets v2: denser value and auxiliary targets for the fast loop

Date: 2026-10-11. Builds on arm A (`runs/3d2/nets/A.bin`, gen-040, encoder v4) and its 16-generation
continuation L (`runs/train/L`, `runs/train/long-A.sh`).

**Why.** Training has plateaued:
- 20× more search gave H0.
- L, sixteen more generations of A's settings, gave H0 against A: 0.260 over 168 seeds, LLR −3.09. That is despite L's lower held-out loss (2.789 against A's 2.835).

The research of 2026-10-11 (`runs/research/research-{rl,catan,efficiency}.md`) kept AZ + ISMCTS. It named the training signal as the likely limit:
- The value head learns only the final winner z. That one label is shared by about 370 positions of a game, which is why value trains on a sparse subsample (`train.py`, VALUE_FRACTION).
- The aux head is a single 4-vector of final VP.

KataGo's measured gains from score, ownership and search-value targets are the closest precedent.

The user approved this sub-project ("Sounds good. Go ahead and do these") and is away. Every decision the user did not make is marked **Ruling**, with what it costs if wrong.

## 1. Diagnostic evidence

The diagnostics live in `runs/diag/`. This section is filled in when they finish; the design below does not wait on them.

- **D1, value noise floor** (`noise_floor.py`): the excess cross-entropy of A's and L's value heads over the empirical winner distribution of 48 continuations, by turn bucket.
- **D5, learning curve** (`learning-curve.sh`): held-out loss of from-scratch nets trained on 4k to 54k games.

## 2. Done criteria

1. **Tests.** Each stage has its own tests, all passing, and so does the whole existing suite: `cargo test`, `pytest`.
2. **Arm T, targets v2.** `fast-loop --targets-v2`, set up exactly as L was:
   - the same seed directories and start net, generations 41..56, trading game;
   - run directory `runs/tv2/T`.

   Then an SPRT of `net:T@100+t` (T's final best) against three `net:A@100+t`: trading game, p0 0.25, p1 0.30, alpha = beta = 0.05, batch 12, at most 400 seeds, seeds from 12,000,000,000. L is the control: same compute, same start, old targets, H0.
   - **H1:** `--targets-v2` becomes the documented setting for new fast loops.
   - **H0:** recorded. Then rule on whether arm E still runs, from what D1 says.
3. **Arm E, weight averaging.** It runs after T: T's settings plus `--ema` (Section 6), in `runs/tv2/E`. It is SPRT'd against T's best if T was H1, otherwise against A, with seeds from 13,000,000,000.
4. **Results** are recorded in `docs/bot/results-targets-v2.md`:
   - D1's and D5's tables;
   - both arms' generation tables (held-out score, gate, steps);
   - both SPRT lines;
   - every ruling taken while running.

## 3. Root search value q (records change)

**The value.** `SearchResult` gains `root_q: Option<[f32; 4]>`. It is the mean of every simulation's backed-up value at the root, in absolute seats:

`root_q = (root_value + Σ_children value_sum) / (1 + Σ_children visits)`

- The children include compound moves.
- `root_value` is the first simulation's evaluation (`Sim::root_value`, now set for PUCT too; it already is).
- It is `None` when the search returned without searching (a single searchable move, `simulations` 0).

The values are win-probability vectors, so `root_q` sums to 1, within float error.

**Ruling:** the root mean, not the chosen child's Q. It is the standard AlphaZero search value. It is defined for every search, including sampled moves and Gumbel roots. If wrong, the cost is a slightly pessimistic target for the actor, because the mean includes exploration of bad moves, and noise on full searches adds to that.

**The record.**
- Self-play's `Decision` and data's `RecordedDecision` gain `q: Option<[f32; 4]>`, written to the record dict as `"q"` (a list of 4 floats) when present.
- `train.rs` reads an optional `"q"`.
- `check_game` rejects a decision whose `q` entries are not finite or not in [−1e-4, 1 + 1e-4], or whose sum is more than 1e-3 from 1.
- Old records have no `q`, and they stay valid.

**The batch.** `Batch` gains two fields:
- `q` (n × 4, seat-relative like `outcome`, zeros when absent);
- `has_q` (n, u8).

**The target.** When `--targets-v2` is on, the value loss's target on a row with `has_q` is

`(1 − Q_MIX) · outcome + Q_MIX · q`, with `Q_MIX = 0.5`.

Rows without `q` keep `outcome`.
- **Ruling:** `Q_MIX = 0.5`. The middle of the range reported to help (KataGo; Willemsen et al. 2020). If wrong, a mis-weighted target, and the constant is one line.
- **Ruling:** no short-horizon TD heads. They would need a second pass over each game's later decisions, and the q mix already gives a low-variance, per-position target. If wrong, a missed gain we can add later.

Held-out scoring stays as it is: value cross-entropy against z on every row. Scores stay comparable with every past run.

## 4. Dense auxiliary targets (from the replayed final state)

Records replay exactly, so every target below comes from a game's final state. That covers all ~54k existing games too.

**Computing them.** `check_game` already replays each game to its end. It now also returns a per-game `Final`, which `TrainingSet` stores beside the game, about 150 bytes each:
- `node`: `[u8; 54]`, per node: 0 empty, 1 + p a settlement of seat p, 5 + p a city of seat p. Node i is bit i of `settlements` and `cities`, the encoder's node order.
- `edge`: `[u8; 72]`, per edge: 0 none, 1 + p a road of seat p. Edge i is bit i of `roads`.
- `parts`: `[[u8; 3]; 4]`, per seat: settlements, cities, victory-point cards held.
- `road`, `army`: `Option<PlayerId>`, the Longest Road and Largest Army holders.
- `turn`: the final `State::turn`.

**In the batch.** `Batch::push` writes each target seat-relative to the decision's actor (seat p becomes `rel(p, actor)`):

| field | shape | target |
|---|---|---|
| `own_node` | n × 54, u8 | class 0..8 as above, relative seats |
| `own_edge` | n × 72, u8 | class 0..4 |
| `parts` | n × 12, f32 | settlements / 5, cities / 4, VP cards / 5, per relative seat |
| `holders` | n × 2, u8 | 0 none or 1 + relative seat, for road then army |
| `turns_left` | n, f32 | (final turn − decision's turn) / TURNS_DEN, TURNS_DEN = 100 |

**The heads.** These are training only, never in the weights file. They live in a new `alphasettler/train/extras.py`:
- `Extras(d)` has
  - `node = Linear(2d, 9)` on [node || glob] per node,
  - `edge = Linear(2d, 5)` on [edge || glob] per edge,
  - `scalar = Linear(4d, 12 + 10 + 1)` on the value head's hidden layer `v` (parts, two 5-way holder logits, turns left).
- `Net` gains `trunk(public, hidden)`, which returns (node, edge, tile, glob, pooled, v). `forward` is rebuilt on it, so its outputs are bit-identical to before. Parity with Rust is unchanged.

**The loss.** It is added to `losses` when `--targets-v2` is on, on the value rows only:
- 0.25 × mean node cross-entropy;
- 0.25 × mean edge cross-entropy;
- 0.25 × MSE(parts);
- 0.1 × each holder cross-entropy;
- 0.1 × MSE(turns left).

Held-out losses report each term as `extras_*`. The selection score stays policy + value.

**Ruling:** value rows only. The 2.28-vs-0.93 memorisation result applies to any per-game label. Final ownership is just as shared across a game's positions. If wrong, the heads see about half the rows they could: a slower gain, not a wrong one.

**Ruling:** these loss weights. Ownership gets about the value loss's weight, the rest less, after KataGo's ownership weight relative to value. If wrong, mis-weighted aux. All are constants in one place.

**Persisting the heads.**
- `fast_train` writes the training-only state to `<out>.train.pt`, beside `<out>.json`: the extras' state dict, plus Section 6's when `--ema` is on, plus the sha256 of the exported `<out>`.
- At start, with `init` given, it loads `<init>.train.pt` if that exists (`torch.load(..., weights_only=True)`). A sha256 mismatch with `init` is a ValueError. A missing sidecar means fresh heads, with a printed notice.
- `fast-loop` copies the sidecar with every `nets/gen-NNN.bin` it copies, so generation k+1's heads continue from generation k's.

## 5. Settings plumbing

- `fast-loop --targets-v2` is set at set-up, recorded in `state.json` as `"targets": 2`, and kept on resume like `trades` and `compound`. State files without it mean the old targets.
- `FastConfig.targets: int = 1`, and `fast-train --targets-v2` is the flag.
- **Ruling:** `train-loop` and `train.train` do not get it. The fast loop replaced them, and `losses` keeps its old behaviour when the flag is off. If wrong, the cost is a later port.

## 6. Weight averaging and a single LR schedule (`--ema`, arm E)

Today each 10-minute generation restarts Adam and runs its own cosine down to 1e-5, a warm restart every 37 minutes. With `--ema`, also a set-up setting kept on resume (`"ema": true`), these change:
- **Adam's state carries over.** Its state dict goes in the `.train.pt` sidecar and is loaded with the raw weights.
- **The learning rate is constant** at `EMA_LR = 2.5e-4`, about the mean of today's cosine, after the usual linear warmup, which runs only when the init has no Adam state in its sidecar (the loop's first generation).
- **An exponential moving average of the weights** (decay `EMA_DECAY = 0.9998` per step, a horizon of about 5,000 steps, or 4 minutes) is what held-out evaluation scores and what is exported. The raw weights, which training continues from, go in the sidecar.

**Ruling:** arm E runs after T, not bundled with it. One change per arm keeps the two attributable. If wrong, about 12 more hours of machine time.

**Ruling:** these constants. They come from today's measured 13,180 steps per generation. If wrong, a re-run.

## 7. Not in this sub-project

- **Policy-target pruning** (KataGo's forced-playout pruning, or completed-Q targets for PUCT): it needs per-child priors and Q at the end of every full search. Its own spec, after a cheap measurement of how much of the visit mass the Dirichlet noise moves.
- **Bigger nets, a human-data prior, opening search, trading.** These are on the research report's list, in that order after this sub-project.
