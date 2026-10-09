# Sub-project 3d-3 (a league of past nets and scripted styles in self-play): results

Date: 2026-10-09. Machine: Apple M1 Pro, macOS (25.5.0).
Commit: eda034c (Plan 11, Task 2; run at cb5d074, which adds only run scripts), plus the commit adding this file.

Spec: [`docs/superpowers/specs/2026-10-08-league-design.md`](../superpowers/specs/2026-10-08-league-design.md).
Plan: [`docs/superpowers/plans/2026-10-08-plan-11-league.md`](../superpowers/plans/2026-10-08-plan-11-league.md).
Builds on: [`results-3d2.md`](results-3d2.md) (encoder v4, arm D) and [`results-3d1b.md`](results-3d1b.md)
(the opening model in self-play, arm C).

## The control

3d-1b: sprt: H0: no gain worth 0.3 after 96 seeds (384 games), LLR -3.69. 3d-2: sprt: H0: no gain worth 0.3 after 48 seeds (192 games), LLR -3.09. So the control is arm A (3d-2's control, 3d-2 was not H1), without the
opening model.

## The league

With `--league`, each self-play game was a league game with probability 1/2: two learner seats and
two pool bots, one past net and one style, at seats drawn from the four; only the learner seats'
decisions were recorded. Past nets: gen-030 to gen-034 of the trading loop and arm A's promoted
generations at its cut (gen-035, gen-037, gen-038, gen-039 and gen-040: 10 nets, `league-000` to
`league-009` in that order), each upgraded to encoder v4 by `net-upgrade`, plus every net arm E
promoted, from the generation after its promotion. Styles: `ismcts@100+t` with the heuristic
weights' road_length and settlement_spots tripled (roads), dev_cards and knights tripled (army), or
production tripled (production). Every league seat searched 100 simulations.

league games: 5,903 of 12,000 (0.492; 0.5 expected, 3 standard deviations 0.014)
style seats: 5,903 of 11,806 league seats (0.500; exactly 0.5 by construction)

| pool member | league seats | share of league seats |
|---|---|---|
| `net:league-000@100+t` | 473 | 0.040 |
| `net:league-001@100+t` | 529 | 0.045 |
| `net:league-002@100+t` | 492 | 0.042 |
| `net:league-003@100+t` | 515 | 0.044 |
| `net:league-004@100+t` | 520 | 0.044 |
| `net:league-005@100+t` | 484 | 0.041 |
| `net:league-006@100+t` | 532 | 0.045 |
| `net:league-007@100+t` | 518 | 0.044 |
| `net:league-008@100+t` | 508 | 0.043 |
| `net:league-009@100+t` | 481 | 0.041 |
| `style:roads@100+t` | 1,966 | 0.167 |
| `style:army@100+t` | 1,919 | 0.163 |
| `style:production@100+t` | 2,018 | 0.171 |
| `net:gen-035@100+t` | 431 | 0.037 |
| `net:gen-037@100+t` | 231 | 0.020 |
| `net:gen-038@100+t` | 120 | 0.010 |
| `net:gen-039@100+t` | 69 | 0.006 |

## Training at equal wall time (done criterion 2)

Arm E is a `fast-loop` with arm A's settings and self-play seeds, run through the same script
(`bash runs/p2/arm.sh E --start-net runs/3d2/gen-034-v4.bin --league` and the ten pool files): a
fresh root, the trading game, seed directories `runs/3b-v2/gen-002` to `gen-005` and
`runs/3c/gen-030` to `gen-034`, 2,000 games a generation, a window of 40,000 games, 10 minutes of
training, a gate of 100 seeds at 100 simulations; set up untimed, then 10,800 s of generations (E
cut after 10,808 s), from gen-034 upgraded to v4 (`runs/3d2/gen-034-v4.bin`). The control ran at
encoder v3 from gen-034 (its set-up line is `arm.sh`'s defaults alone); E differs from it by the
league and by the v4 inputs, which 3d-2 measured as no gain (H0); the control's best plays the SPRT
upgraded, as its v3 self within 1e-5. Arm A was not rerun.

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

### E: best net gen-040, 6 generations finished in the budget

| gen | games | decisions | full searches | steps | held-out score | policy | value | aux | gate wins / games | z | promoted | self-play minutes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 35 | 2,000 | 491,774 | 122,422 | 12,857 | 2.5489 | 1.6187 | 0.9302 | 0.0335 | 112 / 400 | 1.62 | yes | 14.8 |
| 36 | 2,000 | 481,881 | 120,459 | 12,801 | 2.5441 | 1.6168 | 0.9273 | 0.0332 | 91 / 400 | -1.24 | no | 14.5 |
| 37 | 2,000 | 491,492 | 123,445 | 12,782 | 2.5795 | 1.6535 | 0.9260 | 0.0331 | 102 / 400 | 0.27 | yes | 13.9 |
| 38 | 2,000 | 497,306 | 124,079 | 12,799 | 2.5614 | 1.6262 | 0.9352 | 0.0336 | 93 / 400 | -1.00 | yes | 15.0 |
| 39 | 2,000 | 488,787 | 122,603 | 12,800 | 2.6561 | 1.7138 | 0.9423 | 0.0335 | 98 / 400 | -0.25 | yes | 15.2 |
| 40 | 2,000 | 500,512 | 125,063 | 12,808 | 2.6865 | 1.7359 | 0.9506 | 0.0336 | 97 / 400 | -0.40 | yes | 15.4 |

self-play: 12,000 games in 1.48 h of self-play, 8,108 games per hour

Held-out scores are not comparable across the arms: E's held-out games include its league games.
E records about 25% fewer decisions a generation (481,881 to 500,512 against A's 651,593 to
674,797) because a league game records two of its four seats, so its 40,000-game window holds fewer
training positions. Its higher self-play rate (8,108 games an hour against 6,829) comes from the
same place in part, since league seats search at the fast budget always, and in part from the
machine on two days (3d-2 saw D at 7,084 with no league); this run does not separate the two.

Final nets: E: gen-040 (`runs/3d3/nets/E.bin`, sha256 09db7e11122bec76...); A: its best at
the cut, gen-040, at encoder v4 (`runs/3d2/nets/A.bin`, sha256 31909c7f9595d88b...); D: 3d-2's
arm D best (`runs/3d2/nets/D.bin`, sha256 86dbad0524820da7...), for the second SPRT.

| Run | Result | Bar | Outcome |
|---|---|---|---|
| `sprt --candidate net:E@100+t --baseline net:A@100+t --trades`, seeds from 6,000,000,000 | net:E@100+t vs net:A@100+t: win rate 0.250 [0.214, 0.286] 95% CI over 96 seeds (384 games, 96 wins, 0 draws), z vs 0.25 = 0.00, mean VP 7.03, mean turns 88.0; sprt: H0: no gain worth 0.3 after 96 seeds (384 games), LLR -3.84 | H1 (p1 0.30 against p0 0.25, alpha = beta = 0.05) | H0: recorded, the loop without a league stays the default |
| `sprt --candidate net:E@100+t --baseline net:D@100+t --trades`, seeds from 6,002,000,000 | net:E@100+t vs net:D@100+t: win rate 0.244 [0.203, 0.286] 95% CI over 84 seeds (336 games, 82 wins, 0 draws), z vs 0.25 = -0.28, mean VP 6.92, mean turns 87.9; sprt: H0: no gain worth 0.3 after 84 seeds (336 games), LLR -3.50 | the same | H0: recorded |

Belief resets: E vs A 384 games, belief resets 0; E vs D 336 games, belief resets 0.

The control's best is one of the control's promoted generations, so it was in E's pool: the SPRT
opponent is a net E met in training. The robustness lines below share no such link. The second
SPRT (a ledger ruling, not in the spec) isolates the league: D and E both started from gen-034
upgraded and ran v4 for three hours; only E had the league.

## Robustness against the styles (done criterion 3)

Each net as one seat against three copies of one style, 200 seeds from 6,001,000,000 (800 games
each), the same seeds for every line; z vs 0.25:

| net | opponents | win rate | 95% CI | games | z vs 0.25 |
|---|---|---|---|---|---|
| `net:E@100+t` | `style:roads@100+t` | 0.586 | [0.549, 0.624] | 800 | 17.65 |
| `net:E@100+t` | `style:army@100+t` | 0.610 | [0.574, 0.646] | 800 | 19.86 |
| `net:E@100+t` | `style:production@100+t` | 0.559 | [0.522, 0.596] | 800 | 16.46 |
| `net:E@100+t` | the three styles (pooled, seed x style units) | 0.585 | [0.564, 0.606] | 2,400 | 31.07 |
| `net:A@100+t` | `style:roads@100+t` | 0.565 | [0.528, 0.602] | 800 | 16.78 |
| `net:A@100+t` | `style:army@100+t` | 0.595 | [0.559, 0.631] | 800 | 18.86 |
| `net:A@100+t` | `style:production@100+t` | 0.527 | [0.489, 0.566] | 800 | 14.25 |
| `net:A@100+t` | the three styles (pooled, seed x style units) | 0.562 | [0.541, 0.584] | 2,400 | 28.60 |

Belief resets: 0 for each net (both `belief resets: 0` lines of `runs/3d3/robust.log`).

Both nets win more than twice their share against every style, so the styles are far weaker than
either net, as the spec expected. Since E and A played the same seeds and styles, the two can be
compared seed by seed (`runs/3d3/robust-paired.log`):

    paired by seed: E's win rate minus A's on the same seed and style (4 games each, one per seat)
    vs style:roads: difference +0.0213, standard error 0.0181, z +1.17 over 200 seeds
    vs style:army: difference +0.0150, standard error 0.0166, z +0.90 over 200 seeds
    vs style:production: difference +0.0312, standard error 0.0189, z +1.65 over 200 seeds
    pooled (seed x style units): difference +0.0225, standard error 0.0103, z +2.18 over 600 units

This comparison was not set in advance; it is one of four lines, and its seed-by-style units share
a board three at a time, so its standard error is somewhat too small. It is a hint that the league
made E a little harder for the styles to beat (about 2 points), not a decision.

The styles in these games (style seats only, both nets' arenas pooled):

| style | style seats | style wins | win share per seat | mean VP per seat | mean turns |
|---|---|---|---|---|---|
| `style:roads@100+t` | 4,800 | 679 | 0.141 | 5.91 | 95.7 |
| `style:army@100+t` | 4,800 | 636 | 0.133 | 5.86 | 95.5 |
| `style:production@100+t` | 4,800 | 731 | 0.152 | 6.06 | 93.8 |

The three styles finish within 0.2 VP and 2 points of win share of each other: tripling a feature of
the heuristic leaf value changes their play little at 100 simulations, because the heuristic
search's prior ignores the weights and only the leaf values see them.

## Rulings the build made

- Styles are `style:STYLE[@N[+rD]]` bots (the `ismcts` suffixes but `+o`): the heuristic ISMCTS bot
  with the style's weights, a table of feature names times 3.
- The league is a parameter of new self-play entry points (`play_with_league`, `run_with_league`);
  without a league self-play is unchanged (the 3a, P2 and 3d-1b digests pass through them).
- The learner's four bots are built as before; a league seat plays a pool bot built from its name
  (seeded as the learner it replaces) with its own search, records nothing and draws nothing from
  the exploration stream.
- The draw has its own stream, so a plain game in a league run is the game a plain run plays; every
  league game has exactly one past net and one style.
- A league game's record names its seats under `league`; records without a league are
  byte-identical to before. `--league` with `--gumbel` is refused, in Rust and in Python.
- `--league` takes the set-up nets, copied to `nets/league-NNN.bin` and kept by sha256; the pool
  grows by each promoted generation from the next generation on; a resume without the flag keeps
  the pool, other files are refused, a changed copy stops the loop. The gate is unchanged.
- Robustness plays each style in turn (three arenas per net on the same seeds) and a pooled line.
  The pooled line treats each seed-and-style pair as its own unit (600 per net); the three lines on
  a seed share its board, so its interval is a little narrower than a seed-level one.
- Arm E ran through `runs/p2/arm.sh` unchanged at encoder v4 from gen-034 upgraded (3d-2's ruling:
  new loops start from the control's best upgraded, and A's chain starts at gen-034); pool nets were
  upgraded by `net-upgrade`. The SPRTs' nets directory is `runs/3d3/nets`.
- A second SPRT, E against D, from 6,002,000,000, and the paired and per-style lines above, were
  added by the controller; the spec's SPRT is E against A.
- Fresh seeds: 6,000,000,000 (SPRT), 6,001,000,000 (robustness), 6,002,000,000 (E against D).

## Next

The league did not train a stronger net in three hours: E's best scored 0.250 against A's best
and 0.244 against D's, and both SPRTs rule out a gain of 0.05 at alpha = beta = 0.05. Two costs
sit beside the league's benefit at equal wall time: E trained on about 25% fewer positions a
generation (half its games record two seats), and the styles it met are weak and alike, so a
quarter of its league seats taught it little. The paired robustness line (+0.022, z 2.18, a
post-hoc look) says the league may still have made E harder to exploit by styles; a test of that
needs opponents it did not train against. The loop without a league stays the default; prioritised
opponent sampling, exploiter agents, and recording the league seats too are the follow-ups, and
styles that differ in their prior, not just their leaf value, would make the style half of the
pool worth its seats.
