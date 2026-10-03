# AlphaSettler — Sub-project 3b: a learned evaluator trained by self-play

Status: draft for review, 2026-10-02

## Goal

Replace 3a's hand-set heuristic evaluator with a value/policy network trained by self-play, and
clearly beat 3a's own bot with it. 3a's strength stopped rising from 1,000 to 3,000 simulations
(0.443 against 0.435 vs GreedyBot, `docs/bot/results-3a.md`), which points at the evaluator as the
limit. 3b keeps everything else from 3a: the engine, the exact belief, the world sampler, the
ISMCTS search, the `Evaluator` seam, self-play records, both arenas and the Catanatron feed.

Builds on `docs/superpowers/specs/2026-10-01-bot-3a-ismcts-design.md` (3a) and its results.

### Where 3b fits

| Stage | Builds | Reused by the next stage |
|---|---|---|
| 3a (done) | ISMCTS, exact belief, `Evaluator` seam, heuristic evaluator, self-play records | Everything, except the heuristic is no longer the bot's evaluator (it remains a baseline) |
| **3b (this spec)** | Feature encoder, network inference in Rust, PyTorch training, the generation loop, playout cap randomization | Network, training loop, the strongest net as 3c's evaluator |
| 3c | Domestic trade offers and responses | — |

### Decisions

| Question | Decision |
|---|---|
| Compute | Overnight runs on this Mac (Apple M1 Pro: 10 CPU cores, 16 GPU cores, 32 GB). One generation (self-play, training, gating) fits in one night. Self-play can scale out to more machines later without a redesign |
| Inference | Train in PyTorch (MPS); run inference in Rust with a hand-written MLP forward pass (no new crates). Self-play stays inside Rust, one game per thread |
| Rejected: inference server | A Python/MPS server fed by Rust workers pays off only for big nets, adds IPC and nondeterministic batching, and small-batch MPS throughput is often worse than the CPU's |
| Rejected: ONNX Runtime / CoreML | A large native dependency and an Apple-only build path, against the scale-out goal |
| Done bar | Beat 3a head-to-head (Section 5), with a bounded retry of at most 3 extra generations, then record the gap and move to 3c |
| Policy-weighted belief | Out of scope (Section 3) |

## 1. Architecture

```
net/        NEW Rust crate settler-net, depends on engine and search only
  encode.rs     Observation (+ hidden block) -> seat-relative feature vector; ENCODING_HASH
  weights.rs    weights file: load, validate, save (used by tests); shapes
  forward.rs    batched MLP forward pass (f32), the only inference code
  evaluator.rs  NetEvaluator: impl search::Evaluator
bots/       + NetBot names (`net:NAME`, `net:NAME@N`, `net:NAME@N+rD`) built on IsmctsBot with a
              NetEvaluator; self-play gains playout cap randomization and root exploration
bindings/   + encode, replay-and-sample training batches, NetBot, selfplay --net
alphasettler/
  train/    NEW Python package (needs the `train` extra): model.py (PyTorch twin of forward.rs),
              data.py (batches from the bindings), train.py (one training step loop + export),
              loop.py (the generation loop, its manifest and resume logic)
  cli.py    + `train`, `train-loop`, `selfplay --net`, `--nets-dir` on arena/compare
```

- Dependencies run one way: `bots → net → search → engine`. `net` never depends on `bots`.
- The Rust encoder is the only encoder. Python gets features by calling it through the bindings,
  so Rust inference and PyTorch training cannot see different inputs.
- PyTorch is pinned in a new optional extra, `train`. The base install and every Rust crate stay
  free of it. `pip-audit` runs after adding it.

## 2. The network

**Seat-relative view.** Before encoding, players are rotated so the actor is seat 0 and the others
follow in turn order. The value head's output is rotated back to absolute seats.

**Public features** (from `world.observation(actor)` only), 1,557 floats:

| Block | Per item |
|---|---|
| 54 nodes | building owner and kind, seat-relative (8: settlement or city × 4 seats); port kind (6); pips of adjacent tiles per resource (5) |
| 72 edges | road owner, seat-relative (4) |
| 19 tiles | resource or desert (6); pips (1); robber here (1) |
| Scalars | own hand (5); own dev cards, old and new (10); per seat: hand count, dev card count, new dev card count, dev cards played by kind (5, knights included), public VP, discards owed; longest road and largest army owner (5 each: a seat or none); bank (5); dev deck remaining; phase (one-hot); turn / turn cap; dev card played this turn; current player, seat-relative (4: differs from the actor in Discard); return phase (2: PreRoll or Main); free roads left in Road Building (1) |

Counts are scaled to roughly [0, 1] by fixed constants in the encoder: each is divided by the
engine's maximum (a resource's 19 cards, the 25-card dev deck, each dev kind's count in the deck),
or by a realistic maximum where the hard one is far out (hand total / 20, discards owed / 10).

**Hidden block** (from the world, value head only): each opponent's hand (5) and dev cards by
kind (5), seat-relative, 30 floats.

**Network** (about 1.7M multiply-adds per position):

```
public features → 512 → 512 → 512   shared trunk, ReLU
  policy head:  512 → 665 logits, masked to the legal moves, softmax → prior
  value head:   [trunk 512 ‖ hidden 30] → 256 → 4, softmax → value (win probability per seat)
  aux head:     value head's 256 → 4, each seat's final VP / 10 (training only)
```

- The prior depends only on the actor's observation by construction: the hidden block enters
  only the value head. 3a's property test (resampling hidden cards leaves the prior unchanged)
  runs against `NetEvaluator` too.
- `OfferTrade` ids are always masked; 3b never trades.

**Weights file.** Little-endian. Header: magic `ASNET\0\0\0`, format version (u32),
`ENCODING_HASH` (u64, a hash of the encoder's layout constants, a canonical description of its
block and slot order, and its version), layer count and each
layer's shape. Then f32 weights and biases in layer order. Loading fails with a clear error on a
wrong magic, version, hash or shape, or on a short file.

## 3. Search with the network

**`NetEvaluator`** implements 3a's `Evaluator`. For each leaf it encodes the public features from
`world.observation(actor)` and the hidden block from the world, runs the batch through one forward
pass, masks and softmaxes the policy over `legal`, and rotates the value back. The search loop,
belief, sampler and turn-cap handling are 3a's, unchanged. Weights are loaded once per process and
shared read-only (`Arc`) across threads.

**Leaf batch size.** 3a used 8. A larger batch uses the matrix multiply better but weakens
selection (more virtual loss). One measurement picks the default: 8, 16 and 32 compared on
simulations/s and on a paired arena match at equal simulations (Section 5).

**Bot names.** `ismcts` keeps meaning 3a's heuristic bot (a fixed baseline). The network bots are
`net:NAME` (1,000 simulations), `net:NAME@N` and `net:NAME@N+rD`, with the same limits on N and D as
`ismcts`. NAME is `[A-Za-z0-9_-]{1,64}` and resolves to `<nets-dir>/NAME.bin`, where `--nets-dir`
defaults to `runs/3b/nets`. A malformed name, a missing file or a weights file that fails
validation is an error before any game starts.

**Policy-weighted belief: out of scope.** Weighting each joint-hand state by the net's likelihood of
an opponent's actual move needs that opponent's whole observation, including their dev cards,
which the belief does not hold. It costs a forward pass per state per opponent move (about 3 ms
each), several times the self-play budget, and it can make the belief worse against opponents
that play nothing like our net (Catanatron, humans). It belongs to a later stage, once the net is
shown to model opponents.

**Unchanged from 3a:** simulation-count budgets, determinism and thread-count independence, never
trading, ¼ each at the turn cap.

## 4. Training

**Self-play** (`alphasettler selfplay --net NAME`, Rust, one game per thread, seeded):

- Playout cap randomization: each move is a full search with probability 0.25 (600 simulations) or
  a fast search otherwise (100 simulations), drawn from the game's seeded rng. Only full searches
  are policy targets (the records' `full_search` flag). Every searched position is a value target.
- Root exploration on full searches only: Dirichlet noise mixed into the root prior,
  `prior' = 0.75 · prior + 0.25 · Dir(α)` with α = 10 / (number of legal moves). Gamma variates
  come from the engine's seeded `Rng` (Marsaglia–Tsang), so no new crates.
- Move choice: during setup and the first 8 turns, sample the move in proportion to root visits;
  after that, play the most-visited move.
- Records: 3a's per-game gzipped JSONL, plus the net's name and file hash, both simulation caps and
  the exploration settings. Fast-search decisions are recorded with their visit counts and
  `full_search: false`.

**Generation 0: the warm start.** 3a's `ismcts` self-plays 100,000 games with the same playout cap
randomization (with the heuristic evaluator this takes minutes). net₀ is trained on those records
by supervision: the policy imitates the search's visit distribution and the value learns the
outcomes. net₀ should play about as well as 3a, so the first learned generation starts from a
search that already beats AlphaBeta.

**Training** (`alphasettler train`, PyTorch on MPS):

- Data: the records of the most recent 4 generations (net₀'s run counts as one). A Rust sampler in
  the bindings loads the games' compact action lists, draws uniform random (game, decision) pairs
  from the seeded training rng, replays each game to the decision (about 70 µs at 60 ns per action)
  on a thread pool, and returns the batch as raw little-endian buffers (features, hidden block,
  legal mask, visit target, full-search flag, outcome, final VP). Python wraps them with
  `torch.frombuffer`, so no numpy and no per-sample Python objects. Nothing is held in memory per
  position, so the window size is limited only by disk.
- Loss: policy cross-entropy against the normalised visit counts, on full-search positions only;
  value cross-entropy against the outcome (one-hot winner, ¼ each at the turn cap) and 0.25 × MSE
  of the aux head against final VP / 10, both on a random 1/16 of each batch's rows (at least one;
  drawn from a generator seeded by the run's seed and the step); L2 1e-4.
- Validation and early stopping: the newest generation's games whose seed is a multiple of 20 are
  held out and not trained on (older generations in the window are all training data). Every 500 steps and
  at the last, the held-out policy, value (every row) and aux losses are measured on 10 fixed
  batches of 1,024, and the checkpoint with the lowest policy + value is the one exported. The
  first warm start (20,000 games, value loss on every row) scored a held-out value cross-entropy
  of 2.28, worse than a uniform guess (1.39), because about 240 positions per game share one
  outcome and the trunk learned board → winner.
- Optimiser: Adam, batch 1,024, learning rate 1e-3 for net₀ and 3e-4 afterwards. Each generation
  trains from the previous generation's weights (net₀ from scratch) for enough steps to draw each
  of the newest generation's positions twice on average (steps = 2 × new positions / 1,024). Then
  it exports the best held-out checkpoint's weights file and checks it by running the parity
  test's positions through both the Rust and PyTorch forward passes.
- Training on MPS is not bit-reproducible. The determinism guarantee covers self-play, search and
  arenas given a weights file, not training.

**Gating** (in the native CRN arena, no trades): the new net plays 1 vs 3 copies of the current
best net at 600 simulations over 200 seeds. It is promoted if its win rate is at least 0.25.
Otherwise the best net stays. Either way its self-play data stays in the window.

**The generation loop** (`alphasettler train-loop --dir runs/3b --generations N`):
`runs/3b/gen-NNN/` holds that generation's records, weights and gating result, and
`runs/3b/manifest.json` lists the completed steps (self-play, train, gate) with their outputs'
hashes. The loop resumes at the first incomplete step after a Ctrl-C or crash. A step writes its
outputs under a temporary name and renames them when complete, so a half-written file is never
mistaken for a finished one. Self-play from other machines can drop record files into a
generation's directory before training. That is the scale-out path, and it needs no code change.

## 5. Testing and done criteria

**Correctness tests.**

- Encoder:
  - a golden feature vector for a fixed position;
  - seat rotation: rotating every seat and the actor together gives the same features;
  - the prior-invariance property test through `NetEvaluator`.
- Rust/PyTorch parity: random weights, 100 positions from random games, every output equal
  within 1e-5.
- Weights file: round trip; a wrong magic, version, encoding hash or shape, or a truncated file, is
  an error.
- Training data: a sampled batch's features equal the encoder run on the replayed state; fast
  searches contribute nothing to the policy loss (masked); a tiny dataset can be overfit (loss
  falls close to 0).
- Self-play: about 25% of decisions are full searches (within 3 standard errors over 2,000
  decisions); the same seed gives the same games, independent of thread count, with noise and
  sampling on.
- `train-loop`: with tiny settings, interrupting after any step and resuming finishes with the same
  manifest as an uninterrupted run, without redoing completed steps.
- Gating: the promotion rule, including a tie at exactly 0.25.
- Names: malformed `net:` names, a missing file and a mismatched encoding hash are errors in the
  native arena, the Catanatron arena and `compare`.

**Done criteria.**

1. All tests pass, including the Python tests that need the `train` extra (run with it installed;
   none skipped). Every arena run reports 0 fallbacks and 0 belief resets.
2. Search throughput with the net is recorded in `docs/perf/`: simulations/s, evaluations/s and
   the 8/16/32 batch sweep.
3. Warm start (recorded, not a gate): net₀ at 1,000 simulations vs three `ismcts@1000`, 200 seeds.
4. **Headline:** the best net at 1,000 simulations beats three `ismcts@1000` with z > 3 against the
   0.25 null over 200 seeds (native CRN arena, no trades), **and** against `catanatron:alphabeta`
   over 200 seeds it scores a win rate above 3a's 0.383 with z > 3 against 0.25.
5. Strength curve (recorded): the best net at 100, 300, 1,000 and 3,000 simulations vs GreedyBot, to
   show whether 3a's plateau is gone.

**If the headline is missed** after the 5 scheduled generations (net₁ to net₅, after the warm
start): at most 3 more generations, changing only training hyperparameters (learning rate, window,
steps, simulation caps). If it is still missed, record the gap and its suspected cause in `docs/bot/results-3b.md` and move on to 3c.
All other criteria remain hard requirements.

Out of scope: policy-weighted belief, domestic trading (3c), tree reuse, the Colonist interface,
inference on anything but the CPU.
