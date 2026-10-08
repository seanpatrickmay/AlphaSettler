# Performance track P2 (Gumbel root search): results

Date: 2026-10-08. Machine: Apple M1 Pro, 10 cores, macOS (Darwin 25.5.0).
Commits: criterion 1 ran on the code range 190926f..1bed714 (Plan 8, Tasks 2 to 4; none of those
commits changes how a `+g` or a plain bot plays); criterion 2 ran at 90097ed (Plan 8, Task 5),
plus the commit adding this file.

Spec: [`docs/superpowers/specs/2026-10-08-gumbel-root-design.md`](../superpowers/specs/2026-10-08-gumbel-root-design.md).
Plan: [`docs/superpowers/plans/2026-10-08-plan-8-gumbel-root.md`](../superpowers/plans/2026-10-08-plan-8-gumbel-root.md).

## Play strength, no training (done criterion 1)

`+g` searches the root by sequential halving with no Gumbel draws; the baseline is the same net
with a PUCT root. The no-trade game; `runs/3c/nets/gen-034.bin` on every seat.

| Run | Result | Bar | Outcome |
|---|---|---|---|
| `sprt --candidate net:gen-034@32+g --baseline net:gen-034@32 --no-trades`, seeds from 3,000,000,000 | win rate 0.323 [0.269, 0.376] over 72 seeds (288 games, 93 wins), z 2.67; sprt: H1 after 72 seeds, LLR 3.26 | recorded (p0 0.25, p1 0.30, alpha = beta = 0.05) | H1 |
| `sprt --candidate net:gen-034@100+g --baseline net:gen-034@100 --no-trades`, seeds from 3,000,001,000 | win rate 0.292 [0.260, 0.323] over 168 seeds (672 games, 196 wins), z 2.58; sprt: H1 after 168 seeds, LLR 3.21 | recorded (same test) | H1 |
| `arena --candidate net:gen-034@600+g --baseline net:gen-034@600 --no-trades`, 200 seeds from 3,000,002,000 | win rate 0.233 [0.205, 0.260] over 200 seeds (800 games, 186 wins), z -1.25 | z above -2 | met |

Belief resets: 0 in all three (288, 672 and 800 games).

The gain shrinks with the budget, as the paper predicts: clear at 32 and 100 simulations, none
measurable at 600 (the point estimate is below 0.25, within the bar).

## Training at equal wall time (done criterion 2)

Two `fast-loop` runs from gen-034, one after the other on an otherwise idle machine
(`runs/p2/arm.sh`, `runs/p2/both.sh`): fresh roots, the trading game, seed directories
`runs/3b-v2/gen-002` to `gen-005` and `runs/3c/gen-030` to `gen-034`, 2,000 games a generation,
a window of 40,000 games, 10 minutes of training, a gate of 100 seeds at 100 simulations (PUCT,
`+t` on both sides). A is today's self-play (600 / 100 simulations, p 0.25, Dirichlet noise, 8
sampled turns); B is `--gumbel 100` (every move a Gumbel search of 100 simulations with draws,
every one a policy target). Each arm was set up untimed, then given 10,800 s of generations; only
generations finished by the cut count (A cut after 10,809 s, B after 10,807 s). Both arms played
the same self-play seeds.

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

### B: best net gen-041, 7 generations finished in the budget

| gen | games | decisions | full searches | steps | held-out score | policy | value | aux | gate wins / games | z | promoted | self-play minutes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 35 | 2,000 | 1,053,688 | 1,053,688 | 13,231 | 2.4506 | 1.5349 | 0.9157 | 0.0337 | 98 / 400 | -0.27 | yes | 12.9 |
| 36 | 2,000 | 1,086,324 | 1,086,324 | 13,200 | 2.4170 | 1.4879 | 0.9291 | 0.0340 | 96 / 400 | -0.54 | yes | 13.1 |
| 37 | 2,000 | 1,056,968 | 1,056,968 | 13,208 | 2.3841 | 1.4557 | 0.9284 | 0.0342 | 97 / 400 | -0.40 | yes | 12.8 |
| 38 | 2,000 | 1,051,178 | 1,051,178 | 13,216 | 2.3650 | 1.4346 | 0.9304 | 0.0343 | 109 / 400 | 1.32 | yes | 12.8 |
| 39 | 2,000 | 1,036,241 | 1,036,241 | 13,242 | 2.3471 | 1.4154 | 0.9317 | 0.0344 | 108 / 400 | 1.16 | yes | 12.2 |
| 40 | 2,000 | 1,041,721 | 1,041,721 | 13,270 | 2.3411 | 1.4111 | 0.9300 | 0.0346 | 102 / 400 | 0.29 | yes | 12.3 |
| 41 | 2,000 | 1,059,561 | 1,059,561 | 13,132 | 2.3296 | 1.3879 | 0.9417 | 0.0348 | 101 / 400 | 0.15 | yes | 12.8 |

self-play: 14,000 games in 1.48 h of self-play, 9,433 games per hour

Final nets (`runs/p2/finals.log`): A gen-040, B gen-041.

| Run | Result | Bar | Outcome |
|---|---|---|---|
| `sprt --candidate net:B@100+t --baseline net:A@100+t --trades`, seeds from 3,000,003,000 | win rate 0.245 [0.209, 0.281] over 108 seeds (432 games, 106 wins), z -0.25, mean VP 6.96, mean turns 86.8; sprt: H0 after 108 seeds, LLR -4.46 | H1 (p1 0.30 against p0 0.25, alpha = beta = 0.05) | H0: the control stays |

Belief resets: 0 (432 games).

Reading the tables:

- **Held-out scores are not comparable across arms.** B's policy targets are improved policies,
  sharper than A's normalised visits, so B's held-out policy loss is measured against a different
  target. Its falling policy column says the net fits Gumbel targets better each generation, not
  that it plays better. Value and aux are on the same targets in both arms (game results and the
  auxiliary heads): value ends at 0.9620 in A and 0.9417 in B, aux at 0.0341 and 0.0348. Within an
  arm, the held-out set moves with the window, so scores drift between generations too.
- **B played 1.38x as many games per hour (9,433 against 6,829), not the spec's 2.25x.** Gumbel
  draws at every move pick trade offers far more often than A's exploration does: 77 offers a game
  in B's gen-035 against 10 in A's (85 against 13 in the last generations), and each offer brings
  four responses. B's games are 785 actions long against 427, and B searches 527 decisions a game
  against A's 326, on about the same number of turns. Self-play time tracks simulations: B's
  105M a generation (100 on each of 1.05M decisions) against A's 146M (162,376 at 600, the rest at
  100) is 0.72x, and its 12.8 minutes against A's 17.6 is 0.73x. The extra offers and responses are
  what keep B's simulations per game that high.
- **B's generations are 6.5x larger on disk** (about 110 MB against 17 MB): every decision carries
  a full search's visits and its target.
- B's gates all promoted, and the last four had positive z (1.32, 1.16, 0.29, 0.15); A rejected one
  of six. A gate measures a generation against the one before it, not against the other arm.

## Rulings the build made

- Gumbel is a root mode of a bot's search config (`+g` bots: no draws; Gumbel self-play: draws
  from the search's own stream); PUCT inside the tree; PUCT bots and PUCT self-play play exactly
  the games they played before (three digests pinned at eb1f739).
- The prior is floored at the smallest positive f32 before its log; π in v_mix and the improved
  policy is the softmax of those logits.
- Ties in the candidates, the halving and the final pick go to the lower action id. A candidate
  left unvisited because the budget ran out is ranked by v_mix, its completed Q.
- A phase's share is capped by the budget left, so that the root gets exactly simulations - 1
  visits; at tiny budgets some candidates get none.
- The last phase spends what is left round-robin from the leading candidate; v̂ is the root
  evaluation's value for the searching player.
- A decision with a target is a policy target whatever its visits; targets must sum to 1 within
  1e-4. PUCT records carry no target and are unchanged.
- Loops keep their self-play mode from set-up; files from before P2 read as PUCT self-play.
- Each arm's budget is 10,800 s of generations, its set-up untimed (spec Section 2 says "3 hours
  of generations"); the state at the cut decides its generations and its final net.
- Fresh seeds: 3,000,000,000, 3,000,001,000, 3,000,002,000 and 3,000,003,000.
- Games per hour counts self-play time only (the log's `selfplay` to `train` lines, the last
  column of each table).

## Next

Gumbel self-play at 100 simulations trained a net no stronger than today's self-play in the same
three hours, so new loops stay on PUCT self-play; `--gumbel` remains an option. The tables point
at the offers: B's games-per-hour gain was mostly spent on trade offers its draws made and their
responses, and a follow-up that draws only at non-offer roots (or keeps offers out of the
candidates) would test the method on the budget it was meant to save. At play time, `+g` is a
measured gain at 32 and 100 simulations and neutral at 600, so cheap bots (gates, the league of
3d-3) can use it. P3, batching across games, is next on the performance track.
