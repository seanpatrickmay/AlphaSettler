# Sub-project 3d-1 (opening diagnosis and opening model): results

Date: 2026-10-08. Machine: Apple M1 Pro, macOS (25.5.0).
Commit: ca40ba2 (Plan 7, Task 5), plus the commit adding this file.

Spec: [`docs/superpowers/specs/2026-10-08-opening-model-design.md`](../superpowers/specs/2026-10-08-opening-model-design.md).
Plan: [`docs/superpowers/plans/2026-10-08-plan-7-opening-model.md`](../superpowers/plans/2026-10-08-plan-7-opening-model.md).

## Diagnosis (done criterion 1)

`opening-diagnose` with gen-034 (`runs/3c/nets/gen-034.bin`) on the trading loop's self-play,
`runs/3c/gen-030` to `gen-034`: 10000 games (0 without a winner), 40000 openings, from runs/3c/gen-030, runs/3c/gen-031, runs/3c/gen-032, runs/3c/gen-033, runs/3c/gen-034; net gen-034.

D1, forced or chosen: weak openings where a spot touching the missing resource was legal within d pips of the chosen spot, at either pick

| group | openings | within 0 | within 1 | within 2 | within 3 | none |
|---|---|---|---|---|---|---|
| no wood and brick | 2391 | 0.982 | 0.997 | 1.000 | 1.000 | 0.000 |
| no wheat | 5618 | 0.750 | 0.917 | 0.982 | 0.997 | 0.000 |
| no ore | 9200 | 0.736 | 0.902 | 0.978 | 0.995 | 0.000 |
| weak groups | 17209 | 0.775 | 0.920 | 0.982 | 0.997 | 0.000 |

D2, misjudged: the net's win probability for each seat after the setup, from its own view, against its result (decided games)

| group | seats | predicted | actual | gap | se |
|---|---|---|---|---|---|
| no wood and brick | 2391 | 0.202 | 0.216 | -0.014 | 0.008 |
| no wheat | 5618 | 0.186 | 0.153 | +0.033 | 0.005 |
| no ore | 9200 | 0.188 | 0.197 | -0.009 | 0.004 |
| no wood or no brick | 12501 | 0.243 | 0.250 | -0.006 | 0.004 |
| no sheep | 3269 | 0.250 | 0.282 | -0.032 | 0.008 |
| all five | 7021 | 0.270 | 0.395 | -0.125 | 0.006 |
| weak groups | 17209 | 0.189 | 0.185 | +0.004 | 0.003 |

D3, search budget: the setup settlements of 200 games from runs/3c/gen-034, re-chosen from the recorded position by net:gen-034@budget (a fresh bot each)

| budget | picks | changed | weak-group openings |
|---|---|---|---|
| recorded | 1600 | - | 0.425 |
| 1000 | 1600 | 0.504 | 0.396 |
| 8000 | 1600 | 0.524 | 0.390 |

**verdict: misjudged (a spot within 1 pip in 0.920 of 17209 weak openings; the weak groups' gap +0.004 exceeds all five's -0.125 by 20.14 standard errors)**

Self-play draws its setup moves in proportion to root visits, with root noise (the first
`SAMPLE_TURNS` = 8 turns include the setup), so D1 and D2 describe the self-play policy's
openings, and D3's "changed" share counts that sampling as well as the budget. D3's re-chosen
openings and D4's below are the bot's own, most-visited choices.

The verdict's standard error treats the weak groups and all five as independent; wins within one
game are negatively correlated, so the true standard error of the difference is somewhat larger
(the Task 5 review estimates 5 to 10%). At 20 standard errors this does not change the verdict.

Reading it: the bot is not forced (in 92% of weak openings a spot touching the missing resource was
legal within 1 pip), and the value head's error is mostly on the strong side. It rates the weak
groups about right (+0.004 pooled; no wheat is overrated by 0.033), but it underrates openings that
touch all five resources by 0.125 (0.270 predicted, 0.395 actual), so it does not see why they are
better. More search does not fix it: at 1,000 and 8,000 simulations half the picks change, but the
weak-group share only falls from 0.425 to 0.396 and 0.390.

For comparison, gen-4's no-trade self-play (`runs/3b-v2/gen-005`), the same net reading it:
10600 games (0 without a winner), 42400 openings, from runs/3b-v2/gen-005; net gen-034.

D1, forced or chosen: weak openings where a spot touching the missing resource was legal within d pips of the chosen spot, at either pick

| group | openings | within 0 | within 1 | within 2 | within 3 | none |
|---|---|---|---|---|---|---|
| no wood and brick | 2594 | 0.981 | 0.998 | 1.000 | 1.000 | 0.000 |
| no wheat | 6076 | 0.767 | 0.923 | 0.985 | 0.998 | 0.000 |
| no ore | 9921 | 0.747 | 0.911 | 0.977 | 0.995 | 0.000 |
| weak groups | 18591 | 0.787 | 0.927 | 0.983 | 0.997 | 0.000 |

D2, misjudged: the net's win probability for each seat after the setup, from its own view, against its result (decided games)

| group | seats | predicted | actual | gap | se |
|---|---|---|---|---|---|
| no wood and brick | 2594 | 0.212 | 0.240 | -0.028 | 0.008 |
| no wheat | 6076 | 0.168 | 0.137 | +0.032 | 0.004 |
| no ore | 9921 | 0.193 | 0.184 | +0.009 | 0.004 |
| no wood or no brick | 12941 | 0.251 | 0.259 | -0.008 | 0.004 |
| no sheep | 3566 | 0.255 | 0.278 | -0.023 | 0.007 |
| all five | 7302 | 0.277 | 0.407 | -0.130 | 0.006 |
| weak groups | 18591 | 0.188 | 0.176 | +0.011 | 0.003 |

**verdict: misjudged (a spot within 1 pip in 0.927 of 18591 weak openings; the weak groups' gap +0.011 exceeds all five's -0.130 by 22.75 standard errors)**

## The opening model (done criterion 2)

78 → 64 → 64 → 4, trained by `opening-train` on `runs/3b-v2/gen-001` to `gen-005` and
`runs/3c/gen-030` to `gen-034`: samples: 478800 training, 25200 held out, from 63000 games (0 without a winner); best epoch 18 of 21.

| Held-out samples | model | uniform | gen-034 value head |
|---|---|---|---|
| all games (25200) | 1.2890 | 1.3863 | 1.3082 |
| trading games (4000) | 1.3292 | 1.3863 | 1.3474 |

## The test (done criterion 3)

| Run | Result | Bar | Outcome |
|---|---|---|---|
| `sprt --candidate net:gen-034@100+t+o --baseline net:gen-034@100+t --trades`, seeds from 2,000,000,000 | net:gen-034@100+t+o vs net:gen-034@100+t: win rate 0.293 [0.262, 0.324] 95% CI over 240 seeds (960 games, 281 wins, 0 draws), z vs 0.25 = 2.71, mean VP 7.22, mean turns 85.7; sprt: H1: a gain (win rate >= 0.3) after 240 seeds (960 games), LLR 3.58 | H1 (p1 0.30 against p0 0.25, alpha = beta = 0.05) | H1 |
| `arena --candidate net:gen-034@600+t+o --baseline net:gen-034@600+t --trades`, 200 seeds from 2,000,001,000 | net:gen-034@600+t+o vs net:gen-034@600+t: win rate 0.284 [0.255, 0.313] 95% CI over 200 seeds (800 games, 227 wins, 0 draws), z vs 0.25 = 2.28, mean VP 7.20, mean turns 83.2 | reported with its z | z 2.28 |

Belief resets: 0 of 960 games (SPRT) and 0 of 800 games (confirmation).

## Openings played (D4: 200 setups from seed 2,000,000,000, the trading game)

| Bot | mean pips | no wood and brick | no wheat | no ore | no wood or no brick | no sheep | all five | weak groups |
|---|---|---|---|---|---|---|---|---|
| `net:gen-034@100+t` | 20.09 | 0.052 | 0.100 | 0.211 | 0.346 | 0.090 | 0.200 | 0.364 |
| `net:gen-034@1000+t` | 19.98 | 0.045 | 0.119 | 0.204 | 0.324 | 0.091 | 0.217 | 0.367 |
| `net:gen-034@100+t+o` | 19.73 | 0.018 | 0.009 | 0.084 | 0.256 | 0.051 | 0.583 | 0.110 |

The opening model trades about a third of a pip for coverage: it opens on all five resources in
0.583 of seats against 0.200, and on a weak group in 0.110 against 0.364. Ten times the search
(`@1000`) changes neither share.

## Rulings the build made

- After the setup the engine's phase is `PreRoll`; D2 reads that position (setup done, before the
  first roll), from each seat's own view, and leaves out games with no winner.
- `net_evaluate` reads the hidden block from the game itself; right after the setup every card is
  public, so this is each seat's own view. It gives the evaluation a search leaf of the same world
  gets, equal within rounding across batch shapes (the Accelerate forward pass).
- Pips sum per settlement (a tile touched twice pays twice); "distance 2" is two steps on the node
  graph, whoever owns the edges.
- `+o` is for `net:` names only; with `+o`, `+sK` changes nothing. `+o` composes with P2's `+g` in
  any order (the model picks every setup settlement, `+g` governs every searched decision). The
  opening model is used at play time only; self-play does not use it.
- D3 drew `random.Random(0).sample` of 200 of `runs/3c/gen-034`'s games; each pick was re-chosen
  from its recorded position.
- The verdict pools the three weak groups; misjudged means their D2 gap exceeds all five's by more
  than 2 standard errors of the difference (independence assumed, see the diagnosis).
- Training ran on the CPU, seed 0, held out by seed % 20 == 0 across all ten directories.
- The confirmation's seeds start at 2,000,001,000, clear of every SPRT seed.

## Next

The opening model is a gain at 100 and at 600 simulations, so `+o` should be on wherever the bot
plays for strength, and the next loop should use it in self-play too: the main net then trains on
games that open on all five resources far more often, which is where its value head is most wrong.
Setup roads and a draft search over the model (spec Section 6) are the follow-ups for openings;
D2's underrating of all-five openings also argues for 3d-2's whole-board features in the encoder,
since two rounds of message passing may not carry a seat's resource coverage across the board
(a hypothesis, not measured here).
