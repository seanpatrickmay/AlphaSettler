# AlphaSettler — Sub-project 3a: ISMCTS search bot

Status: draft for review, 2026-10-01

## Goal

A search bot, IsmctsBot, that clearly beats Catanatron's AlphaBeta bot. It also lays the
foundation that 3b (a learned value/policy network trained by self-play) and 3c (domestic trades)
build on, so nothing in 3a gets thrown away except one part: the hand-written heuristic evaluator,
which the network replaces.

Builds on the finished engine and benchmark harness
(`docs/superpowers/specs/2026-09-30-engine-and-harness-design.md`, `docs/oracle/results.md`).

### Where 3a fits

Sub-project 3 (the bot) is split into stages. Each stage ships a stronger bot and reuses the
previous stage:

| Stage | Builds | Reused by the next stage |
|---|---|---|
| **3a (this spec)** | Single-observer ISMCTS, card-tracking belief, `Evaluator` seam, heuristic evaluator, self-play records | Search, belief, evaluator seam, records (a supervised warm-start dataset), the bot as a fixed opponent |
| 3b | Value/policy network, self-play training loop, playout cap randomization, policy-weighted belief | Network and training loop |
| 3c | Domestic trade offers and responses | — |

### Decisions

| Question | Decision |
|---|---|
| Compute | Not known yet. Design for one Mac (M1 Pro, 10 cores) and keep a path to scale out without a rewrite |
| Iterating | Iterate only where each stage feeds the next. Otherwise build toward the best final bot directly |
| Hidden information | Single-observer ISMCTS: one tree over the bot's information sets, one sampled world per simulation |
| Search budget | A simulation count, never wall time, everywhere inside AlphaSettler. Converting a clock to simulations belongs to the later Colonist-interface spec |
| Domestic trades | Off in 3a. The bot never offers and declines every offer. 3c adds trading |
| Done bar | Hard headline target (Section 5) with a bounded retry: at most 3 tuning rounds on the heuristic evaluator, then record the gap and move to 3b |

### How other game bots set their budget (research behind the budget decision)

- Self-play training uses fixed simulation counts. AlphaZero used 800 simulations per move.
  AlphaGo Zero used 1,600, about 0.4 s per move on its hardware. KataGo's *playout cap
  randomization* gives 25% of moves a full search (600 rising to 1,000 visits), and only those are
  recorded as policy targets. The rest get a fast search (100 rising to 200), which yields more
  games for the value target at the same cost.
- Progress between versions is measured at a small fixed budget. AlphaZero rated its checkpoints in
  1 s/move tournaments and plotted strength against thinking time. Showcase matches used 1 min/move.
- Against humans, Pluribus searched 1–33 s depending on the situation, about 20 s per hand on
  average, roughly twice as fast as human professionals.

Sources: Silver et al., "Mastering Chess and Shogi by Self-Play…" (arXiv:1712.01815); Silver et al.,
"Mastering the Game of Go without Human Knowledge" (Nature 2017); Wu, "Accelerating Self-Play
Learning in Go" (arXiv:1902.10565); Brown & Sandholm, "Superhuman AI for multiplayer poker"
(Science 2019).

## 1. Architecture

```
engine/     rules unchanged; + State::reseed and Game::log (the unredacted log, for tests)
search/     NEW Rust crate, depends on engine only
  belief.rs     card tracker: event log -> particles holding all four hands
  world.rs      sampled worlds: a per-decision template with the hidden parts overwritten
  tree.rs       single-observer ISMCTS tree, keyed by the bot's information set
  puct.rs       selection and backup arithmetic; 4-player value vectors
  evaluator.rs  trait Evaluator: a batch of leaves -> (value[4], prior over legal actions)
  search.rs     the simulation loop, chance nodes, leaf batching
bots/       + heuristic.rs: 3a's evaluator (replaced by the network in 3b). It lives here, not in
              search/, because it reuses GreedyBot's scoring and bots already depends on search
            + IsmctsBot: wraps search as a Bot, registered as "ismcts"
bindings/   + IsmctsBot and self-play exposed to Python
alphasettler/ + `alphasettler selfplay` CLI and record loader
oracle/     + translate Catanatron's action records into our Events for the bot
```

- Dependencies run one way: `bots → search → engine`. In 3b, self-play needs the search without
  the arena or the baseline bots.
- **Bot interface change.** Today `Bot::act(obs, legal)` sees only the current observation, but
  card tracking needs history. `Bot` gains `fn observe(&mut self, viewer: PlayerId, events: &[Event])`, a
  no-op by default. The native arena calls it before every `act` with the events since that seat
  last acted, redacted for that seat (the same events `Game::log_for(viewer)` holds). `Bot` also
  gains `fn diagnostics(&self) -> Vec<(&'static str, u64)>` (empty by default), so arenas can
  report how often IsmctsBot had to rebuild its belief.
- **Catanatron arena.** `AlphaSettlerPlayer` keeps importing snapshots for its legal-move mapping.
  It also translates Catanatron's action records into our viewer-filtered `Event`s, so the bot plays
  inside Catanatron on exactly the information it would have natively (tested in Section 5).
- **Sampled worlds.** Once per decision, `WorldSampler` builds a template `State` with
  `State::from_snapshot` (fully validated) from the observation plus one sampled deal. Each
  simulation copies the template and overwrites only the hidden parts: opponents' hands, their dev
  cards, the deck order, and a fresh random seed (`State::reseed`) so future dice and steals are
  random rather than the real game's. Tests check every sampled world passes
  `State::check_invariants`, equals `State::from_snapshot` of its own snapshot, and shows the
  viewer exactly the observation it was sampled from.
- Chance inside the tree replays through `apply_forced`.

## 2. Belief tracker

**Resources.** Every event that moves resources is public in our engine, except a steal seen by
someone other than the thief and the victim. The public events are production, builds, dev-card
purchases, maritime and domestic trades, discards (each card is public), Monopoly and Year of
Plenty. So once the outcome of each hidden steal is known, every opponent's hand follows exactly.
The hidden resource state is one latent variable per hidden steal, with 5 possible values.

The tracker is a particle filter over those latents, with the exact prior:

- `N` particles (config, default 1,024). Each one assigns every hidden steal an outcome, which
  implies all three opponents' hands.
- At a hidden steal, each particle draws the stolen resource in proportion to the victim's hand in
  that particle. This is the engine's actual steal distribution, a uniform card from the hand.
- When a player spends or gives cards (a build, a purchase, a trade, a discard, a Monopoly loss),
  particles in which that player could not have done it are dropped. The survivors are an exact
  sample of the posterior given public information.
- If fewer than `N/8` particles survive, the tracker rebuilds: it replays the stored log from
  scratch with rejection, up to `32·N` attempts, and fills any shortfall by resampling the survivors
  (old and new) with replacement. Only when no particle survives at all does `observe` return an
  error. IsmctsBot then rebuilds its belief from the observation (cards nobody can see dealt
  uniformly) and counts a `belief_resets` diagnostic, which every arena run must report as 0.
- Invariant: every particle's hand sizes equal the observation's `hand_counts`.

**Dev cards and the deck.** The unknown pool is the 25-card deck minus the viewer's own cards and
every card played publicly. Opponents' held dev cards (only counts are visible) and the remaining
deck order are a uniform random deal from that pool. That includes hidden VP cards, so a sampled
world can let an opponent win before their public VP says so.

**Sampling one world** (once per simulation): pick a uniform random particle, then deal the dev
holdings and the deck order fresh.

**Not in 3a:** behavioural inference (for example, "they didn't play a knight when it was
obvious"). Particles are uniformly weighted in 3a. 3b adds a weight per particle: the policy
network's likelihood of the opponents' actual moves.

## 3. Search

**Tree.** Single-observer ISMCTS from the searching bot's perspective.

- A child is keyed by `(action, outcome visible to the bot)`. Public chance branches: dice (11
  outcomes, sampled by probability) and steals where the bot is thief or victim. Chance the bot
  cannot see does not branch. That covers an opponent's dev draw and a steal between two opponents;
  the sampled world fixes those outcomes.
- A simulation: sample a world (Section 2), descend using only actions legal in that world, expand
  one leaf, evaluate it, back up.
- Selection uses ISMCTS availability counts: a move's exploration term uses the number of visits in
  which it was legal, not the parent's visit count.

**Four players.** Values are 4-vectors of win probabilities. The player to act at a node picks the
move that maximises their own entry (max^n). A finished game backs up a one-hot vector for the
winner. A game that reaches `max_turns` backs up ¼ for each player.

**Selection and evaluation.** PUCT, with the prior and the value from the `Evaluator`, and `c_puct`
as a config value. There are no random rollouts. The heuristic evaluator may run a short greedy
rollout internally, behind a config switch that the arena decides (Section 5).

**Batching.** The search collects up to `B` leaves (config, default 8) using virtual loss, and
makes one `Evaluator` call per batch. 3a's heuristic doesn't need this, but the network will.

**Determinism.** Each search is single-threaded and seeded, so the same seed and budget always give
the same move. Parallelism comes from running games side by side, as in the current arena. Arena
results stay reproducible and independent of thread count.

**Budget and move choice.** The budget is `simulations` per decision. When only one move is legal,
the bot plays it without searching. In the arena the bot plays the most-visited root move. 3b adds
temperature sampling for self-play.

**Trades.** The search assumes no domestic offers. If a config allows them, IsmctsBot never offers
and declines every offer it receives.

**Not in 3a:** reusing the tree between moves (an optional speedup that changes no interface), and
parallel search within a single decision.

**Rough speed** (to be measured, not promised): a world sample, a path replay of about 10–30
actions at about 60 ns each, plus one evaluation. That should come to roughly 50k–100k simulations
per second per core, or 10–20 ms per real decision at 1,000 simulations.

## 4. Evaluator and 3b handoff

**Contract.**

```rust
pub struct Leaf<'a> { pub world: &'a State, pub actor: PlayerId, pub legal: &'a [Action] }
pub struct Eval { pub value: [f32; 4], pub prior: Vec<f32> } // prior aligned with `legal`, sums to 1
pub trait Evaluator { fn evaluate(&mut self, leaves: &[Leaf]) -> Vec<Eval>; }
```

- `value` sums to 1 and may use the whole sampled world. Averaged over worlds, that gives an
  expected value.
- `prior` may depend only on `world.observation(actor)`. Otherwise opponents inside the tree would
  play as if they could see the bot's cards. A property test enforces this: resampling other
  players' hidden cards must leave the prior unchanged.

**Heuristic evaluator (3a).**

- Value: each player gets a strength score, a weighted sum of these features:
  - VP, including sampled hidden VP;
  - production pips, weighted by resource scarcity on this board;
  - cards in hand and dev cards held;
  - distance to longest road and to largest army;
  - open legal settlement spots.

  A softmax over the four scores gives win probabilities. The weights are fit by maximum
  likelihood on recorded games (a conditional logit solved by Newton's method in Rust, run by
  `alphasettler fit-heuristic` on the self-play records below). A softmax temperature would only
  rescale the weights, so the fit absorbs it. This fit is the first use of the data pipeline.
- Prior: GreedyBot's choice (`bots/src/greedy.rs`, shared rather than copied) gets logit +2, any
  build +1, buying a dev card +0.5, everything else 0, softmaxed over the legal moves.
- Greedy rollout switch: an optional number of greedy moves played from the leaf before scoring
  it. Bot names select it: `ismcts` (1,000 simulations), `ismcts@N`, `ismcts@N+rD` (D rollout
  moves).

**Self-play records (built in 3a).** `alphasettler selfplay --games N --simulations S --out DIR`
plays IsmctsBot in all four seats and records every searched decision: the decision index, the
actor, the actor's observation, the legal moves, the root visit counts and a `full_search` flag
(always true in 3a). Records are grouped per game, one gzipped JSONL line per game holding the
seed, config, every action, the decisions and the outcome. It switches to a columnar format only
if 3b's data loading measures too slow. Storing the seed and the
observation means any feature encoding can be computed later, so 3b designs its network input
without replaying games. If 3a meets its headline target, the first network trains on a search that
already beats AlphaBeta, instead of starting from random play.

**Sketched for 3b, not built here:** the observation encoding (tile, node and edge planes plus
scalars); a 665-way policy head masked to legal moves and a 4-way value head; how inference runs on
the Mac (Rust-native, ONNX/CoreML or batched PyTorch, benchmarked as 3b's first decision); playout
cap randomization; policy-weighted belief particles.

## 5. Testing and done criteria

**Correctness tests.**

- Belief tracker:
  - exact match (within 0.02) with a brute-force posterior, on hand-built and on random small
    histories;
  - across random games, the unredacted log replayed through the tracker reproduces every hand
    exactly (so the true history is never ruled out), and every particle matches `hand_counts`.
  - No in-game calibration test: random players' choices depend on their hands, and the belief
    deliberately ignores that behavioural evidence (Section 2). An in-game calibration test would
    measure the omission, not the tracker.
- Sampled worlds: pass `check_invariants`, equal `from_snapshot` of their own snapshot, and give
  the viewer exactly the observation they were sampled from, in every phase of random games.
- Search mechanics with a mock evaluator on tiny synthetic trees: availability counts, max^n backup,
  virtual loss, branching on public chance, no branching on hidden chance.
- Evaluator contract: the prior-invariance property test (Section 4).
- Tactical positions the bot must solve:
  - take a winning build when it is available (uniform evaluator, 200 simulations);
  - build an affordable city rather than end the turn (heuristic, 1,000 simulations);
  - never put the robber on its own buildings when an opponent's tile is available (heuristic,
    1,000 simulations).
  The discard tactic first proposed here needs lookahead past three opponents' turns, which 3a's
  evaluator cannot supply. It belongs in 3b's evaluation suite.
- Determinism: same seed and budget give the same move. The native arena stays independent of
  thread count with IsmctsBot playing.
- Catanatron event feed: in lockstep differential games (the existing `oracle/diff.py` harness), the
  events translated from Catanatron's records equal our engine's viewer-filtered log, for every
  viewer and every game.

**Done criteria.**

1. All of the above pass. The Catanatron arena records **0 fallbacks** and **0 belief resets**
   for IsmctsBot.
2. Search throughput (simulations/s per core) is measured and recorded in `docs/perf/`.
3. Strength curve: IsmctsBot at 100, 300, 1,000 and 3,000 simulations against GreedyBot in the
   native CRN arena, each z > 3 against the 0.25 null.
4. **Headline:** IsmctsBot at 1,000 simulations beats `catanatron:alphabeta` with z > 3 against the
   0.25 null (1 vs 3, seats rotated, 200 seeds), and also beats `catanatron:value` with z > 3.
5. `alphasettler selfplay` writes records that load in Python, and a test confirms that seed plus
   actions reproduce every stored observation exactly.

**If the headline is missed:** run at most 3 tuning rounds, changing only the heuristic evaluator
(feature weights, the calibration fit, the rollout switch). Record each round's arena numbers in
`docs/bot/results-3a.md`. If the target is still missed, record the gap and its suspected cause
there and move on to 3b. All other criteria remain hard requirements.

Out of scope: neural networks, the training loop, domestic trading, tree reuse, the
Colonist interface.
