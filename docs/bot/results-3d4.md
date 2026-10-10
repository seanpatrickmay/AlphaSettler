# Sub-project 3d-4 (compound build moves in search): results

Date: 2026-10-09. Machine: Apple M1 Pro, macOS (25.5.0).
Commit: arm F and criterion 2 ran at a324657 (Plan 12, Tasks 1–4 and the final-review fixes, which
change no move, record or digest). Criterion 1 and the final review's 32-game log ran earlier, on the
Task 3 extension build with the scripts at 9427fc1 (Task 4 is Python-only); the plain-speed check
compared e2b8fb9 against 37406c6. Run scripts are at 9427fc1..4acbd28, plus the commit adding this
file.

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
further games of the same pairing on criterion 1's first 8 seeds (7,000,000,000 to 7,000,000,007,
with the `+c` seat in each position). They come from a reviewer's driver, not a run file here, and
are not replays of criterion 1's games: on those seeds `runs/3d4/sprt-c1.jsonl` has the `+c` seat at
3.66 bank trades a game. The review classified every bank trade by what the player did next:

| | `+c` seat (32 games) | plain seats (96 seat-games) |
|---|---|---|
| bank trades a game | 4.38 | 5.59 |
| followed at once by a build those trades enabled | 4.38 (all) | 3.25 |
| followed by the end of the turn with no build | 0 | about 2.0 |
| before a build already affordable / in chains of 3 | 0 | 0.16 / 0.03 |

So the `+c` seat makes more build-enabling bank trades than a plain seat. What it gives up is the
plain bot's hold trades, the roughly 2.0 a game followed by the end of the turn with no build. Of
those, 111 of 191 (58%) started from a hand of more than 7 cards, and 154 of 191 left the hand at 7
or fewer: the plain bot uses them to keep its hand out of the 7-discard. The `+c` seat ended its turn holding more than 7 cards 12.5% of the time against
6.6% and discarded about 12.3 cards a game against 10.3 (a small sample). That is the cost the
spec's ruling named, and a candidate reason for criterion 1's H0. In the same games (4 driver
processes in parallel, so indicative), a searched Main decision cost about 2% more with `+c` (4.97
against 4.87 ms), offset by the searches plan steps skip (78.4 against 79.4 a game; 0.385 against
0.387 s of search a game). Plain searches showed no detectable slowdown (`runs/3d4/speed-plain.log`:
HEAD e2b8fb9 at 26,210 against base 37406c6 at 26,266 simulations a second, the mean of 3 runs each,
0.998; single runs spread from 24,037 to 27,397; the same 596 decisions).

## Training at equal wall time (done criterion 2)

Arm F is arm A's set-up plus `--compound`, run through `runs/p2/arm.sh` unchanged
(`bash runs/p2/arm.sh F --start-net runs/3d2/gen-034-v4.bin --compound`; the second `--start-net`
replaces the script's v3 one, as for arms D and E): the same seed directories, window, training
minutes and gate, the same self-play and gate seeds; set up untimed, then 10,800 s of generations
(F cut after 10,802 s). Its self-play and both gate sides searched compound moves (`+t+c` gates); the
control's gates were `+t`, so the two gate columns measure different bots.

**Not an idle machine.** With the user's permission, Plan 13 (the JSettlers2 arena) was built beside
arm F, in its own worktree under `nice -n 19`: builds, JVM games and full test runs from F's start to
about 22:15, then only code writing, unit tests that start no JVM, and reviews until the cut. Once,
after gen-037, a Plan 13 implementer ran `pkill -f multiprocessing.spawn`; arm F was checked and was
unaffected (its self-play is Rust-threaded). F's training windows made 10,919 to 12,981 steps
against A's 13,175 to 13,280, and its self-play took 18.9 to 26.9 minutes a generation against A's
16.9 to 18.1. Compound self-play also records more decisions a game (695,877 to 727,873 a generation
against 651,593 to 674,797). The run does not separate the two costs, and contention is not all of
it: F's gen-038 (self-play 22:11 to 22:38, training to 22:48), mostly after the heavy work stopped,
was its slowest generation (26.9 minutes of self-play, 10,919 steps), and the voided run's gen-035,
launched through the idle-machine check before any Plan 13 build or test, took 19.6 minutes of
self-play against A's 16.9. F finished 4 generations to A's 6. An earlier
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

Compound moves did not help at play time, and the training test found no gain. With no training,
`+c` won 0.237 of its games against three copies of the same net without it (0.25 is parity; 95% CI
[0.192, 0.283]). The SPRT accepted H0 against p1 0.30 at alpha = beta = 0.05: a gain as large as 0.05
is unlikely, but a smaller one is not excluded. The bot made more build-enabling bank trades than the
plain seats (4.38 against 3.25 a game in 32 logged games) but none that bring a large hand down, and
it discarded more (12.3 against 10.3 cards a game, a small sample). Trained with compound moves for
10,800 s, F's best scored 0.215 against B (both `+c`), also H0. This is not the clean equal-wall-time
test criterion 2 asks for: F trained beside other work (the ledger's ruling C1 asks for an idle
machine, and contention biases this test toward H0) and finished 4 generations to A's 6, so its H0
does not show that training with compound moves cannot help. Loops without `--compound` stay the
default. The cheapest follow-up the evidence points to is keeping the bare bank trades beside the
compounds (the spec's first ruling reversed), so the bot can still trade a large hand down; that is
a new measured change, not a fix. The final review names a second candidate: a shared first trade's
prior is split among its compounds (a dozen road compounds each get 1/12), which thins exploration at
100 simulations.
