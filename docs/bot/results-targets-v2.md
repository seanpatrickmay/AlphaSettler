# Targets v2: results

Date: 2026-10-12. Machine: Apple M1 Pro, macOS (25.5.0).
Commit: fa15d44 (the plan 14 merge), plus 79fb903 (`runs/tv2/arms.sh`) and the commit adding this file.

Spec: [`docs/superpowers/specs/2026-10-11-targets-v2-design.md`](../superpowers/specs/2026-10-11-targets-v2-design.md).
Plan: [`docs/superpowers/plans/2026-10-11-plan-14-targets-v2.md`](../superpowers/plans/2026-10-11-plan-14-targets-v2.md).

A = `runs/3d2/nets/A.bin` (sha256 `31909c7f9595d88bbadd179299a270013216c8e373a2ec33aa09acf171d7fbbe`), arm A's
gen-040. Both arms below and the control L start from it, with the same seed directories, generations 41..56 and
the trading game (`runs/tv2/arms.sh`, `runs/train/long-A.sh`).

## D1: value noise floor (`runs/diag/noise_floor.py`)

640 positions from 160 games of `A@100+t`, 48 continuations each. Cross-entropy (nats) against the continuation
winners. The floor is the empirical winner distribution, leave-one-out with an add-0.5 prior, an upper bound on the
true floor. Uniform is 1.386.

| positions | n | floor (LOO) | floor (in-sample) | net A | net L | excess A | excess L | CE vs z, A | CE vs z, L |
|---|---|---|---|---|---|---|---|---|---|
| turns 0-19 | 138 | 1.179 | 1.116 | 1.268 | 1.258 | +0.088 | +0.079 | 1.277 | 1.252 |
| turns 20-39 | 136 | 1.035 | 0.974 | 1.084 | 1.072 | +0.049 | +0.038 | 1.082 | 1.088 |
| turns 40-59 | 151 | 0.826 | 0.771 | 0.882 | 0.881 | +0.056 | +0.054 | 0.886 | 0.880 |
| turns 60+ | 215 | 0.670 | 0.619 | 0.777 | 0.759 | +0.107 | +0.089 | 0.748 | 0.735 |
| all | 640 | 0.894 | 0.838 | 0.973 | 0.962 | +0.079 | +0.068 | 0.966 | 0.956 |

The value head sits 0.07 to 0.08 nats above the floor, and the gap is largest after turn 60. Late-game misreads are
the biggest error, which the VP-parts and turns-left aux targets aim at.

## D5: learning curve (`runs/diag/learning-curve.sh`)

48/2 nets trained from scratch for 45 minutes each (about 102,000 steps) with the old targets, on the newest 4k to
54k self-play games: L's generations newest first, then arm A's, then 3c's. Every net is scored on one fixed
held-out set, the 400 held-out games of L's gen-053..056 (`runs/diag/curve/score.log`). T was scored on the same
set afterwards (`runs/diag/curve/score-T.log`).

| net | games | score | policy | value | aux |
|---|---|---|---|---|---|
| scratch | 4k | 2.8280 | 1.8238 | 1.0042 | 0.0387 |
| scratch | 8k | 2.7915 | 1.8199 | 0.9716 | 0.0372 |
| scratch | 16k | 2.7705 | 1.8165 | 0.9540 | 0.0364 |
| scratch | 32k | 2.7727 | 1.8176 | 0.9551 | 0.0363 |
| scratch | 54k | 2.7768 | 1.8215 | 0.9553 | 0.0365 |
| A (gen-040) | | 2.7914 | 1.8382 | 0.9532 | 0.0362 |
| L (gen-056) | | 2.7534 | 1.8103 | 0.9431 | 0.0359 |
| T (gen-056) | | 2.7687 | 1.8136 | 0.9550 | 0.0361 |

The curve is flat from 16k games on. At this net size and budget, more games of the same kind do not lower the
loss, so the targets, not the amount of data, were the limit. That matches T's gain and L's H0.

Held-out loss does not predict playing strength here. L scores best on its own games, and T scores worse on value
because it trains toward the z/q mix. But T is the one that beats A.

## Arm T: `--targets-v2` (done criterion 2)

`runs/tv2/T`. Each generation: 2000 self-play games, 10 minutes of training, a gate of 400 games against the best
net. The held-out score is policy + value cross-entropy against z on the newest games' held-out split. The arms'
own self-play makes those games, so T's scores are not comparable with L's. T also trains value toward a z/q mix,
which raises its cross-entropy against z by design.

| gen | self-play net | steps | pool games | held-out score | policy | value | gate win rate | z | promoted |
|---|---|---|---|---|---|---|---|---|---|
| 41 | start | 12278 | 22800 | 2.8495 | 1.8719 | 0.9776 | 0.260 | 0.54 | yes |
| 42 | gen-041 | 12227 | 24700 | 2.8522 | 1.8740 | 0.9782 | 0.283 | 1.58 | yes |
| 43 | gen-042 | 12294 | 26600 | 2.8867 | 1.8984 | 0.9883 | 0.250 | 0.00 | yes |
| 44 | gen-043 | 12308 | 28500 | 2.8920 | 1.8991 | 0.9929 | 0.253 | 0.13 | yes |
| 45 | gen-044 | 12306 | 30400 | 2.8893 | 1.8903 | 0.9990 | 0.258 | 0.43 | yes |
| 46 | gen-045 | 12280 | 32300 | 2.8719 | 1.8688 | 1.0031 | 0.235 | -0.82 | yes |
| 47 | gen-046 | 11975 | 34200 | 2.8630 | 1.8670 | 0.9959 | 0.225 | -1.68 | no |
| 48 | gen-046 | 12103 | 36100 | 2.8682 | 1.8647 | 1.0035 | 0.240 | -0.60 | yes |
| 49 | gen-048 | 10799 | 38000 | 2.8727 | 1.8738 | 0.9989 | 0.240 | -0.52 | yes |
| 50 | gen-049 | 11697 | 38000 | 2.8670 | 1.8764 | 0.9907 | 0.243 | -0.43 | yes |
| 51 | gen-050 | 12517 | 38000 | 2.8878 | 1.8917 | 0.9961 | 0.270 | 1.21 | yes |
| 52 | gen-051 | 12533 | 38000 | 2.8760 | 1.8781 | 0.9979 | 0.225 | -1.52 | no |
| 53 | gen-051 | 12510 | 38000 | 2.8429 | 1.8511 | 0.9918 | 0.270 | 1.13 | yes |
| 54 | gen-053 | 12516 | 38000 | 2.8838 | 1.8783 | 1.0055 | 0.263 | 0.68 | yes |
| 55 | gen-054 | 12496 | 38000 | 2.8632 | 1.8643 | 0.9989 | 0.233 | -1.02 | no |
| 56 | gen-054 | 12485 | 38000 | 2.8434 | 1.8481 | 0.9953 | 0.260 | 0.52 | yes |

T's best is gen-056, `runs/tv2/nets/T.bin` (sha256
`823595c3c762d644bac1e9515a9a8a7a7f0ee99618333e658e45577c3bcedf65`). It trained at about 20 steps/s against
about 22 for the old targets (91%), so T took about 1,000 fewer steps a generation than L.

### SPRT against A

Trading game, three `net:A@100+t` opponents, p0 0.25, p1 0.30, alpha = beta = 0.05, batch 12, at most 400 seeds,
seeds from 12,000,000,000 (`runs/tv2/sprt-T-vs-A.log`):

```
net:T@100+t vs net:A@100+t: win rate 0.288 [0.260, 0.316] 95% CI over 192 seeds (768 games, 221 wins, 0 draws), z vs 0.25 = 2.63, mean VP 6.98, mean turns 88.1
sprt: H1: a gain (win rate >= 0.3) after 192 seeds (768 games), LLR 3.12
```

The control is L, which had the same start, seed directories, generations and compute but kept the old targets
(`runs/train/sprt-L-vs-A.log`, seeds from 9,100,000,000):

```
net:L@100+t vs net:A@100+t: win rate 0.260 [0.230, 0.291] 95% CI over 168 seeds (672 games, 175 wins, 0 draws), z vs 0.25 = 0.68, mean VP 6.97, mean turns 89.2
sprt: H0: no gain worth 0.3 after 168 seeds (672 games), LLR -3.09
```

**H1.** It is the first measured gain since the plateau. Per the spec, `--targets-v2` is now the setting for new
fast loops.

## Rulings while running

- **Ruling:** D5 runs between T's SPRT and arm E, not beside T. Its trainings are timed and must not share the
  machine. If wrong, about 4 hours later for E.
- **Ruling:** T was scored on D5's held-out set by hand, during arm E's first self-play phase, outside the timed
  training. The spec's D5 lists only A and L. If wrong, a 17-second slowdown of one self-play phase.
- **Ruling:** T's held-out scores are not read as a regression. They are scored on T's own newest games and against
  z only, while T trains toward the z/q mix. If wrong, the SPRT would have caught a weaker net, and it did not.
