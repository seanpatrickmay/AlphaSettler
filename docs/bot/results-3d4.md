# Sub-project 3d-4 (compound build moves in search): results

Date: 2026-10-09. Machine: Apple M1 Pro, macOS (25.5.0).
Commit: a324657 (Plan 12, Tasks 1–4 and the final-review fixes; the runs at f9e6096..4acbd28, which add
only run scripts), plus the commit adding this file.

Spec: [`docs/superpowers/specs/2026-10-08-compound-moves-design.md`](../superpowers/specs/2026-10-08-compound-moves-design.md).
Plan: [`docs/superpowers/plans/2026-10-08-plan-12-compound-moves.md`](../superpowers/plans/2026-10-08-plan-12-compound-moves.md).
Builds on: [`results-3d3.md`](results-3d3.md), [`results-3d2.md`](results-3d2.md) and
[`results-3d1b.md`](results-3d1b.md) (all three H0) and [`results-p2-gumbel.md`](results-p2-gumbel.md) (arm A).

B is arm A's best at its cut, gen-040, at encoder v4 (`runs/3d2/nets/A.bin`): 3d-3, 3d-2 and 3d-1b
were all H0 (`runs/3d3/sprt-E.log`, `runs/3d2/sprt-D.log`, `runs/p2/sprt-C-vs-A.log`), so P2's arm A
is the best arm. Arm A did not use `--opening`, so neither side plays `+o`.

## Play strength, no training (done criterion 1)

| Run | Result | Bar | Outcome |
|---|---|---|---|
| `sprt --candidate net:B@100+t+c --baseline net:B@100+t --trades`, seeds from 7,000,000,000 | net:B@100+t+c vs net:B@100+t: win rate 0.237 [0.192, 0.283] 95% CI over 60 seeds (240 games, 57 wins, 0 draws), z vs 0.25 = -0.54, mean VP 6.84, mean turns 92.9; sprt: H0: no gain worth 0.3 after 60 seeds (240 games), LLR -3.50 | H1 (p1 0.30 against p0 0.25, alpha = beta = 0.05) | H0: recorded |

Belief resets: 240 games, belief resets 0.

## Compound moves in play (done criterion 3, from the criterion-1 records)

- decisions with a compound move: 3544 of 19452 searched decisions of the +c seat (0.182)
- bank trades per game: 3.85 by the +c seat, 5.91 by each seat without +c

The `+c` seat trades with the bank less because compounds replace bare bank trades (the spec's first
ruling): it never trades only to hold cards for a later turn. The Plan 12 final review logged 32
further games of the same match (seeds 7,000,000,000 to 7,000,000,007, each seat; a reviewer's driver,
not a run file here) and classified every bank trade by what the player did next:

| | `+c` seat (32 games) | plain seats (96 seat-games) |
|---|---|---|
| bank trades a game | 4.38 | 5.59 |
| followed at once by a build those trades enabled | 4.38 (all) | 3.25 |
| followed by the end of the turn with no build | 0 | about 2.0 |

So the `+c` seat makes more build-enabling bank trades than a plain seat; what it gives up is the
plain bot's trades that bring a hand of more than 7 cards down before the end of its turn (58% of
those trades started above 7). It ended its turn holding more than 7 cards 12.5% of the time against
6.6% and discarded about 12.3 cards a game against 10.3 (a small sample). That is the cost the
spec's ruling named, and a candidate reason for criterion 1's H0. In the same games a searched Main
decision cost about 2% more with `+c` (4.97 against 4.87 ms), offset by the searches plan steps skip
(78.4 against 79.4 a game); plain searches did not slow (`runs/3d4/speed-plain.log`: 26,210 against
26,266 simulations a second, 0.998, the same 596 decisions).

## Training at equal wall time (done criterion 2)

Arm F is arm A's set-up plus `--compound`, run through `runs/p2/arm.sh` unchanged
(`bash runs/p2/arm.sh F --start-net runs/3d2/gen-034-v4.bin --compound`; the second `--start-net`
replaces the script's v3 one, as for arms D and E): the same seed directories, window, training
minutes and gate, the same self-play and gate seeds; set up untimed, then 10,800 s of generations
(F cut after 10,802 s). Its self-play and both gate sides searched compound moves (`+t+c` gates); the
control's gates were `+t`, so the two gate columns measure different bots.

**Not an idle machine.** At the user's request Plan 13 (the JSettlers2 arena) was built beside arm F,
in its own worktree under `nice -n 19`: builds, JVM games and full test runs from F's start to about
22:15, then only code writing, unit tests that start no JVM, and reviews until the cut. F's training windows made 10,919 to 12,981 steps against A's 13,175 to
13,280, and its self-play took 18.9 to 26.9 minutes a generation against A's 16.9 to 18.1. Compound
self-play also records more decisions a game (695,877 to 727,873 a generation against 651,593 to
674,797). The run does not separate the two costs; F finished 4 generations to A's 6. An earlier
arm F (`runs/p2/F-void-sleep*`) was voided when the Mac slept with the lid closed inside its budget.

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

### F: best net gen-037, 4 generations finished in the budget

| gen | games | decisions | full searches | steps | held-out score | policy | value | aux | gate wins / games | z | promoted | self-play minutes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 35 | 2,000 | 695,877 | 173,711 | 12,981 | 2.5318 | 1.6067 | 0.9251 | 0.0334 | 110 / 400 | 1.23 | yes | 18.9 |
| 36 | 2,000 | 699,929 | 175,446 | 11,220 | 2.5625 | 1.6395 | 0.9230 | 0.0333 | 96 / 400 | -0.55 | yes | 24.9 |
| 37 | 2,000 | 717,241 | 179,164 | 11,104 | 2.5860 | 1.6488 | 0.9372 | 0.0333 | 102 / 400 | 0.29 | yes | 24.1 |
| 38 | 2,000 | 727,873 | 181,934 | 10,919 | 2.6000 | 1.6658 | 0.9342 | 0.0334 | 91 / 400 | -1.29 | no | 26.9 |

self-play: 8,000 games in 1.58 h of self-play, 5,058 games per hour

Final nets: F: gen-037 (`runs/p2/F/nets/gen-037.bin`, `runs/3d4/nets/F.bin`, sha256 ecce88130b2f433b...);
B: `runs/3d4/nets/B.bin` (sha256 31909c7f9595d88b..., A's gen-040 at v4).

| Run | Result | Bar | Outcome |
|---|---|---|---|
| `sprt --candidate net:F@100+t+c --baseline net:B@100+t+c --trades`, seeds from 7,001,000,000 | net:F@100+t+c vs net:B@100+t+c: win rate 0.215 [0.160, 0.271] 95% CI over 36 seeds (144 games, 31 wins, 0 draws), z vs 0.25 = -1.22, mean VP 6.69, mean turns 93.7; sprt: H0: no gain worth 0.3 after 36 seeds (144 games), LLR -3.80 | H1 (p1 0.30 against p0 0.25, alpha = beta = 0.05) | H0: recorded, loops without `--compound` stay the default |

Belief resets: 144 games, belief resets 0.

## Rulings the build made

- Compound moves at every Main decision of the searching player in its tree, the root included;
  opponents keep their plain moves. Compounds replace bare bank trades (the spec's ruling), so a
  turn whose trades enable no build within two trades cannot trade with the bank.
- The trade rule: the shortfall covered resource by resource in index order; a resource is given
  only while its cards beyond the target's cost cover its rate, the most such cards first, ties to
  the lower index; the bank must hold each card got. Targets are the engine's own build rules on a
  copy of the state with a hand that covers every cost.
- A compound is a search edge keyed 0x8000 | its build's id; its visits are recorded on its first
  trade's entry; its prior is its first trade's, split among compounds sharing that trade, and the
  move set is renormalised (uniform if nothing is left). Root noise mixes in before.
- A plan goes on only while the bot's own trade is the only event since its last step and the next
  step is legal; anything else, including no events, drops it and the bot searches. Plan steps are
  not searches and record nothing. Visit sampling draws whole compounds.
- `+c` with `+g` is an unknown bot; `--compound` with `--gumbel` is refused. Loops keep the setting
  like `--trades`; the train-loop's warm start uses compound moves too. In a league, only learner
  seats get compound moves; league seats keep their pool names.
- Match records carry per-seat `bank_trades`, `searches` and `compound_decisions`.
- Arm F ran through `runs/p2/arm.sh` unchanged with `--start-net runs/3d2/gen-034-v4.bin` (A's start
  net at encoder v4, as for arms D and E) plus `--compound`. Fresh seeds: 7,000,000,000 and
  7,001,000,000.

## Next

Compound moves did not help at either stage. With no training, `+c` scored 0.237 against the same
net without it, and the SPRT rules out a gain of 0.05; the bot made build-enabling trades as often
as before but stopped trading down large hands, and discarded more. Trained with compound moves for
three hours, F's best scored 0.215 against B (both `+c`), also H0, but F trained beside other work
and finished 4 generations to A's 6, so this criterion is weaker evidence than the others. Loops
without `--compound` stay the default. The cheapest follow-up the evidence points to is keeping the
bare bank trades beside the compounds (the spec's first ruling reversed), so the bot can still trade
a large hand down; that is a new measured change, not a fix.
