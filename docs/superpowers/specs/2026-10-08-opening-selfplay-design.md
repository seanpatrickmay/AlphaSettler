# Sub-project 3d-1b: the opening model in self-play

Date: 2026-10-08. Builds on 3d-1 (`docs/bot/results-3d1.md`: `+o`, the opening model
`runs/3c/nets/opening.bin`, SPRT H1 at 100 simulations, confirmed at 600) and P2's control arm
(`runs/p2/A`, best `gen-040`, three hours of today's self-play from `gen-034`).

The user asked (2026-10-08) to keep going with self-play and to do all of the queued work (3d-2,
3d-3, 3d-4). This is the first step: self-play whose setup settlements come from the opening model,
so the main net trains on games opened by the stronger opening policy. Openings are judged by win
rate only (the user: which resources an opening touches is not a goal). Decisions the user did not
make are marked **Ruling**.

## 1. Done criteria

1. **Self-play with the model.** `fast-loop` and `train-loop` gain `--opening PATH` (an `ASOPEN`
   file): self-play's setup settlements are chosen by the model (Section 2) and the gates play `+o`
   on both sides. A loop keeps the setting it was set up with, as it keeps `--trades` and `--gumbel`.
2. **The test, at equal wall time.** Arm C: a `fast-loop` from `gen-034` with exactly arm A's
   settings (`runs/p2/arm.sh`: the trading game, the same seed directories, window, training minutes,
   gate and self-play seeds; set up untimed, then 10,800 s of generations) plus
   `--opening runs/3c/nets/opening.bin`. Then `net:C@100+t+o` against three `net:A@100+t+o`
   (`alphasettler sprt --trades`, p0 0.25, p1 0.30, alpha = beta = 0.05, fresh seeds from
   4,000,000,000), where A is `gen-040` and C is arm C's best at the cut. H1: new loops use
   `--opening`. H0 or undecided: recorded; the loop without it stays the default.
3. **Results** in `docs/bot/results-3d1b.md`: arm C's generation table (as `runs/p2/tables.py`
   prints it) beside arm A's, and the SPRT line.

## 2. The choice in self-play

At a setup-settlement decision in self-play, the model scores every legal settlement as `+o` does
(`choose_settlement`: placed on a copy, the actor's own-win probability). Then:

- with probability `1 − ε` the highest-scoring spot (ties to the first in engine order), else
- a spot drawn uniformly from the `K` highest-scoring spots (all of them if fewer are legal).

`ε` = 0.25 and `K` = 5, drawn from the self-play game's own exploration stream, so a seed replays
exactly. No search runs and no decision is recorded for those picks (there is no policy target);
setup roads and every later move are searched as now.

**Ruling:** ε-greedy over the top 5 rather than sampling from the model's probabilities. The
model's win probabilities differ by a few points between good and bad spots, so a softmax over them
puts most of its mass on poor spots unless a temperature is tuned; ε-greedy keeps three picks in
four at the model's choice and still varies the draft (about two exploratory picks a game). Cost if
wrong: less or more opening variety than ideal; the SPRT measures the result either way.

**Ruling:** the opening model stays fixed for the whole loop (`runs/3c/nets/opening.bin`), copied
into the loop's nets directory at set-up so the gates' `+o` finds it beside every generation's net.
Retraining it on the loop's games is a follow-up. Cost if wrong: the model slowly goes stale as the
net improves; three hours is too short for that to matter.

## 3. Out of scope

Retraining the opening model per generation, setup roads by the model, a draft search, `+o` with
Gumbel self-play (`--gumbel` and `--opening` together are refused: Gumbel self-play stays the P2
measurement it was), 3d-2 to 3d-4.

## 4. Testing

- Rust: a self-play game with a model records no decision at any setup settlement, plays every
  setup settlement the model's top-K allows and at least one non-argmax pick over a few seeds (the
  exploration is on), is deterministic per seed and thread count, and `belief_resets` is 0; without
  a model every pinned self-play digest is unchanged.
- Python: `fast-loop --opening FILE` runs two tiny generations, its gate names carry `+o` on both
  sides, a loop set up without it refuses it on resume (and the reverse), a bad file is refused at
  set-up, and `--opening` with `--gumbel` is refused.
