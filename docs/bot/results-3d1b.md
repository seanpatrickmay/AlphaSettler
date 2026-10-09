# Sub-project 3d-1b (the opening model in self-play): results

Date: 2026-10-09. Machine: Apple M1 Pro, macOS (25.5.0).
Commit: 19f1ceb (Plan 9, Task 2; arm C and its SPRT ran from a worktree of 57e416a, Plan 9's code with the plans after it, built on its own, because main had moved to encoder v4), plus the commit adding this file.

Spec: [`docs/superpowers/specs/2026-10-08-opening-selfplay-design.md`](../superpowers/specs/2026-10-08-opening-selfplay-design.md).
Plan: [`docs/superpowers/plans/2026-10-08-plan-9-opening-selfplay.md`](../superpowers/plans/2026-10-08-plan-9-opening-selfplay.md).
Builds on: [`results-3d1.md`](results-3d1.md) (the opening model and `+o`) and
[`results-p2-gumbel.md`](results-p2-gumbel.md) (arm A, the control).

## Training at equal wall time (done criteria 2 and 3)

Arm C is a `fast-loop` from gen-034 with exactly arm A's settings, run through the same script
(`bash runs/p2/arm-v3.sh C --opening runs/3c/nets/opening.bin`, `runs/p2/arm.sh` run from the v3 worktree): a fresh root, the trading game, seed
directories `runs/3b-v2/gen-002` to `gen-005` and `runs/3c/gen-030` to `gen-034`, 2,000 games a
generation, a window of 40,000 games, 10 minutes of training, a gate of 100 seeds at 100
simulations; set up untimed, then 10,800 s of generations (C cut after 10,807 s).
Arm A is P2's control (`runs/p2/A`, cut after 10,809 s), not rerun. Both arms played the same
self-play and gate seeds. In C the opening model (`runs/3c/nets/opening.bin`, sha256
9ab63da1309e35f2a87990edc85064ee9888057de84d52fb0e09ae382ced922b) picked every self-play setup settlement: its best spot with
probability 0.75, else one of its 5 best uniformly, with no search and no recorded decision; setup
roads and every later move were searched as in A (600 / 100 simulations, p 0.25, Dirichlet noise, 8
sampled turns). C's gates were `+t+o` on both sides, A's `+t`, so the two gate columns measure
different bots.

### A: best net gen-040, 6 generations finished in the budget

| gen | games | decisions | full searches | steps | held-out score | policy | value | aux | gate wins / games | z | promoted | self-play minutes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 35 | 2,000 | 651,593 | 162,376 | 13,280 | 2.5404 | 1.6150 | 0.9254 | 0.0334 | 101 / 400 | 0.13 | yes | 16.9 |
| 36 | 2,000 | 653,904 | 163,795 | 13,207 | 2.5663 | 1.6305 | 0.9358 | 0.0333 | 92 / 400 | -1.13 | no | 18.1 |
| 37 | 2,000 | 656,783 | 164,159 | 13,175 | 2.5759 | 1.6418 | 0.9341 | 0.0334 | 103 / 400 | 0.40 | yes | 17.1 |
| 38 | 2,000 | 656,109 | 164,131 | 13,229 | 2.6237 | 1.6755 | 0.9482 | 0.0335 | 103 / 400 | 0.36 | yes | 17.9 |
| 39 | 2,000 | 662,391 | 166,476 | 13,229 | 2.6852 | 1.7273 | 0.9579 | 0.0342 | 105 / 400 | 0.71 | yes | 17.6 |
| 40 | 2,000 | 674,797 | 168,467 | 13,243 | 2.6963 | 1.7344 | 0.9620 | 0.0341 | 96 / 400 | -0.56 | yes | 17.8 |

self-play: 12,000 games in 1.76 h of self-play, 6,829 games per hour

### C: best net gen-039, 5 generations finished in the budget

| gen | games | decisions | full searches | steps | held-out score | policy | value | aux | gate wins / games | z | promoted | self-play minutes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 35 | 2,000 | 610,413 | 152,185 | 12,404 | 2.5569 | 1.6246 | 0.9323 | 0.0335 | 97 / 400 | -0.45 | yes | 19.3 |
| 36 | 2,000 | 612,433 | 153,434 | 12,357 | 2.5629 | 1.6322 | 0.9307 | 0.0335 | 121 / 400 | 3.06 | yes | 17.5 |
| 37 | 2,000 | 613,464 | 152,978 | 11,568 | 2.5947 | 1.6466 | 0.9481 | 0.0336 | 103 / 400 | 0.45 | yes | 21.0 |
| 38 | 2,000 | 619,075 | 154,461 | 12,696 | 2.5997 | 1.6597 | 0.9400 | 0.0334 | 96 / 400 | -0.66 | yes | 24.0 |
| 39 | 2,000 | 620,152 | 155,205 | 12,948 | 2.6583 | 1.6987 | 0.9596 | 0.0339 | 110 / 400 | 1.45 | yes | 16.1 |

self-play: 10,000 games in 1.63 h of self-play, 6,130 games per hour

Final nets: A gen-040 (`runs/p2/nets/A.bin`), C gen-039 (`runs/p2/nets/C.bin`).

| Run | Result | Bar | Outcome |
|---|---|---|---|
| `sprt --candidate net:C@100+t+o --baseline net:A@100+t+o --trades`, seeds from 4,000,000,000 | net:C@100+t+o vs net:A@100+t+o: win rate 0.250 [0.214, 0.286] 95% CI over 96 seeds (384 games, 96 wins, 0 draws), z vs 0.25 = 0.00, mean VP 6.94, mean turns 80.6; sprt: H0: no gain worth 0.3 after 96 seeds (384 games), LLR -3.69 | H1 (p1 0.30 against p0 0.25, alpha = beta = 0.05) | H0: recorded, the loop without it stays the default |

Belief resets: 384 games, belief resets 0.

## Rulings the build made

- The model is a parameter of new self-play entry points (`play_with_opening`,
  `run_with_opening`); without a model self-play is unchanged (the 3a and P2 digests pass through
  them).
- A model pick draws one number from the game's exploration stream (explore if below 0.25) and,
  when exploring, one more (the spot among the 5 best); the ranking is a stable sort, so ties go to
  engine order and the greedy pick is `+o`'s spot. About 0.8 of picks are the best spot.
- The seat's bot still receives the game's events at a model pick; it does not decide.
- `--opening` with `--gumbel` is refused, in Rust and in Python, before anything is written.
- Records with a model carry `opening_sha256`, `opening_epsilon` and `opening_top` in their
  `selfplay` block; records without one are byte-identical to before.
- Loops keep the model by its sha256 (`state.json`, `manifest.json`); a resume without the flag
  keeps it, another file is refused, and a missing or changed `nets/opening.bin` stops the loop.
  `train-loop`'s warm start uses the model too.
- Arm C ran through `runs/p2/arm.sh`'s logic unchanged (as `runs/p2/arm-v3.sh`, which only changes the directory and the Python entry point to the v3 worktree); arm A was not rerun. The SPRT's nets directory
  is `runs/p2/nets` with `C.bin` and `opening.bin` copied in beside `A.bin`.
- Fresh seeds: 4,000,000,000.

## Next

Self-play with the opening model did not train a stronger net in three hours: C's best (gen-039)
and A's best (gen-040) are level when both play with `+o` (0.250, z 0.00). C promoted all five of
its generations (one gate at z 3.06) against A's five of six, but a gate measures a generation
against the one before it, not against the other arm. C played 6,130 self-play games an hour
against A's 6,829; the model's picks cost almost nothing, so the gap is most likely the machine (gen-038's self-play
took 24 minutes against 16 to 21 for the others). Held-out scores are not comparable across the two
arms: C's self-play records carry no policy targets at setup settlements. The opening model stays a
play-time gain (`+o`, `results-3d1.md`), and the loop without it stays the default; 3d-2's arm D
runs with arm A's settings.

## How the run went

The first arm C (`runs/p2/C-void-sleep*`) is void: the Mac slept on battery from 17:56 to 19:17, so
its cut came after 12,109 s of wall time with three generations. A second (`runs/p2/C-informal*`)
ran while the machine was shared with builds and a Catanatron match and is informal. The run
reported here started at 23:00 on an otherwise idle machine; it was on battery (low-power mode) for
210 s during gen-035's self-play and 31 s just before the cut.
