# Performance track P2: Gumbel root search

Date: 2026-10-08. Builds on the Accelerate forward pass (`docs/perf/net-search.md`, 285d5d7, 1cd6a37)
and the trading net `runs/3c/nets/gen-034.bin`.

The user asked for the whole performance track before 3d-1: speed first, then batching across games
(P3), with GPU offload deferred (its measured ceiling is about one CPU thread). This is P2, ahead of
P3 because its expected gain is larger (research note in the conversation of 2026-10-08: Gumbel
AlphaZero, Danihelka et al., ICLR 2022, "Policy improvement by planning with Gumbel"; DeepMind's
`mctx`; MiniZero). Decisions the user did not make are marked **Ruling**.

## 1. Why

Self-play spends most of its compute on policy targets: with playout cap randomization a quarter of
moves get 600 simulations and the rest 100, about 225 per move, and only the 600-simulation searches
train the policy. Gumbel root search gives a policy target that is a guaranteed improvement over the
prior even with few simulations (the "improved policy" from completed Q-values), and replaces
Dirichlet noise and visit sampling with one principled exploration draw. If 100 Gumbel simulations
train as well as today's mix, self-play makes about 2.25x more games per hour, each with four times
as many policy targets.

## 2. Done criteria

1. **Play strength, no training.** With no noise (`+g`, Section 4):
   - `net:gen-034@32+g` against three `net:gen-034@32`: SPRT (`alphasettler sprt`, p0 0.25, p1
     0.30, alpha = beta = 0.05, fresh seeds) in the no-trade game. Recorded either way; the paper
     predicts a gain at low budgets.
   - `net:gen-034@100+g` against three `net:gen-034@100`: the same SPRT. Recorded either way.
   - `net:gen-034@600+g` against three `net:gen-034@600`, 200 fixed fresh seeds (`arena`): z > -2.
2. **Training at equal wall time (the headline).** Two `fast-loop` runs from `gen-034`, trading game,
   the same window, training minutes and gate, each given the same wall-clock budget of 3 hours of
   generations, run one after the other on an idle machine:
   - **A (control):** today's self-play (600 / 100 simulations, p 0.25, Dirichlet noise, 8 sampled
     turns).
   - **B (Gumbel):** every move a Gumbel search of 100 simulations (Section 5), every one a policy
     target.

   Then `net:B@100+t` against three `net:A@100+t`, trading game, SPRT from fresh seeds, where A and B
   are each run's final best net. **H1: Gumbel self-play becomes the default** (`fast-loop` and
   `train-loop` gain it as an option either way). H0: recorded; the control stays.
3. **Results recorded** in `docs/bot/results-p2-gumbel.md`: criterion 1's three lines, both runs'
   generation tables (games, steps, held-out score, gate), games per hour of each, and criterion 2's
   line.

## 3. The algorithm (root only)

At a decision with root actions `A` (the searchable moves of `search_moves`, offers included for a
bot that may offer), at least two of them:

1. The first simulation evaluates the root as now; its prior `π` (over `A`, renormalised) gives
   logits `ℓ(a) = ln π(a)`, and its value `v̂` is the actor's root value.
2. **Candidates.** `g(a)` are i.i.d. Gumbel(0, 1) draws (from the search's own `Rng`) in self-play,
   and all zero in play. The `m = min(16, |A|)` actions with the largest `g(a) + ℓ(a)` are the
   candidates (ties to the lower action id).
3. **Sequential halving.** The remaining budget `n = simulations - 1` is spent over
   `ceil(log2 m)` phases. In each phase every remaining candidate gets
   `max(1, floor(n / (phases · |remaining|)))` simulations whose first step from the root is forced
   to that candidate; below the root the descent is today's PUCT. After a phase the
   `ceil(|remaining| / 2)` candidates with the largest `g + ℓ + σ(q̂)` stay. The last phase also gets
   whatever budget is left over, spread round-robin. A phase with one candidate left ends the halving
   and spends the rest on it.
4. `q̂(a)` is the mean root-actor value of child `a`'s visits, and
   `σ(q) = (c_visit + max_b N(b)) · c_scale · q` with `c_visit = 50`, `c_scale = 1.0` (the paper's
   values; our values are win probabilities in [0, 1], as the paper assumes).
5. **The move:** among the candidates still remaining, the largest `g + ℓ + σ(q̂)`.
6. **The improved policy** over all of `A`: `π' = softmax(ℓ + σ(completedQ))`, where
   `completedQ(a) = q̂(a)` for a visited `a` and `v_mix` otherwise, with
   `v_mix = (v̂ + (ΣN / Σ_{b visited} π(b)) · Σ_{b visited} π(b) q̂(b)) / (1 + ΣN)` (paper, appendix,
   eq. 33).

**ISMCTS.** The root decision is the searching player's own, so the root actions are the same in
every sampled world; a forced root step therefore always exists, and the search asserts it. Leaf
batching keeps virtual loss as now: a phase's scheduled root steps are taken in round-robin order,
`batch` at a time.

**Ruling:** root only, PUCT unchanged inside the tree. The paper's interior rule (Gumbel MuZero's
deterministic selection) matters most for tiny budgets; the root is where the policy target and the
exploration come from, and it is the smaller change.

**Ruling:** `m = 16`. The median root has 13 legal moves and 63% of roots have 16 or fewer (measured
on `runs/3c/gen-035`), so most roots consider every move.

## 4. Bots

`+g` is a bot-name suffix like `+t`, `+sK` and `+rD` (any order, at most once; `net:` and `ismcts`
names): the bot's searches use Section 3 with `g = 0`. Setup searches with `+sK` keep their factor.
Without `+g` nothing changes, so every recorded result and golden digest stays valid.

## 5. Self-play and training data

- `Exploration` gains a Gumbel mode: every move is a Gumbel search of `gumbel_simulations` (with
  `g` drawn), recorded as a full search with its improved policy. Dirichlet noise, playout cap
  randomization and visit sampling are off in this mode: the Gumbel draw is the exploration.
- A recorded decision gains an optional `target` (floats aligned with `legal`, summing to 1): the
  improved policy. The training sampler uses `target` when a decision has one and the normalised
  visits otherwise, so every existing record reads as before. A decision with a `target` is a policy
  target whatever its visits.
- `fast-loop` and `train-loop` gain `--gumbel SIMS` (the mode above for self-play; gates unchanged).
  A loop keeps the self-play mode it was set up with, as it keeps `--trades`.

## 6. Out of scope

The interior selection rule, Gumbel at chance or opponent nodes, changing the gate, P3 (batching
across games), GPU offload.

## 7. Testing

- Search: with `g = 0` and a uniform evaluator, the candidates are the top-m by prior; sequential
  halving spends exactly `simulations - 1` root visits and visits each phase's candidates equally
  (within one); the chosen move is the halving's winner on a hand-built evaluator where one move's
  value is clearly best; `π'` sums to 1, equals `softmax(ℓ)` when nothing is visited, and moves mass
  toward the higher-value move; a single legal move is returned without searching, as now; the same
  seed gives the same move and `π'`.
- Bots: `+g` names parse in any order with the others, are refused when repeated, and leave
  non-`+g` bots' digests unchanged.
- Self-play: a Gumbel-mode game records a `target` on every searched decision that sums to 1 over
  the legal moves; `belief_resets` is 0.
- Data: a record with `target` trains on it (the batch's policy row equals the target at the legal
  ids); one without is unchanged; a `target` of the wrong length or not summing to 1 within 1e-4 is
  an error naming the decision.
- Python: `fast-loop --gumbel 100` runs two tiny generations; a loop set up without `--gumbel`
  refuses it on resume.
