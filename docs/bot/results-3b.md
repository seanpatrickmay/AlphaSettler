# Sub-project 3b (learned evaluator): results

Date: 2026-10-07. Machine: Apple M1 Pro, 10 cores, macOS (Darwin 25.5.0). Python 3.12.6, rustc
1.98.1, PyTorch 2.14.1. Catanatron 3.3.0 at ecf931181b9a65bb4116a2153fb78c16f1438e00.

The nets were trained and the headline runs played at ca82614 (network v2, the newest-share fix).
The fresh-seed rerun and the strength curve ran from the `search-sweep` worktree at bb87e71 plus
budget code, whose v2 build loads the same files; neither changes how a `net:` bot without
`+b`/`+t` suffixes plays. This file is committed after Plan 6's code.

Spec: [`docs/superpowers/specs/2026-10-02-bot-3b-network-design.md`](../superpowers/specs/2026-10-02-bot-3b-network-design.md)
(network v2 amendment 2d4742f, newest-share amendment ca82614).

## Search speed

Network v2 (width 48, 2 rounds, 247,294 parameters, 2.78M multiply-adds per position): 4,171
simulations/s on one core at `NET_BATCH` = 8. The batch sweep found no batch at least 10% faster,
so `NET_BATCH` stays 8. Details: [`docs/perf/net-search.md`](../perf/net-search.md).

## Generations (`runs/3b-v2`, `train-loop`)

Each learned generation: 10,600 self-play games with the best net (playout cap randomization,
600 / 100 simulations, p 0.25), training continued from the previous generation on a window of 4,
half of each batch from the newest generation, and a gate of 1 against 3 copies of the best net at
600 simulations over 200 seeds (promote iff 4 × wins ≥ games).

| gen | games | decisions | full searches | steps | held-out policy | value | aux | parity | gate wins / games | win rate | z | promoted | best after |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 100,000 | 24,523,371 | 6,129,841 | 45,494 | 1.4145 | 0.7329 | 0.0313 | 2.62e-06 | warm start | — | — | yes | 0 |
| 1 | 10,600 | 2,358,299 | 588,643 | 8,754 | 1.4493 | 0.8281 | 0.0315 | 2.38e-06 | 215 / 800 | 0.269 | 1.37 | yes | 1 |
| 2 | 10,600 | 2,339,811 | 586,188 | 8,676 | 1.4827 | 0.8939 | 0.0324 | 2.44e-06 | 187 / 800 | 0.234 | −1.34 | no | 1 |
| 3 | 10,600 | 2,357,150 | 588,878 | 8,741 | 1.4767 | 0.9082 | 0.0335 | 2.26e-06 | 191 / 800 | 0.239 | −0.93 | no | 1 |
| 4 | 10,600 | 2,357,216 | 588,752 | 8,743 | 1.4650 | 0.8892 | 0.0321 | 1.83e-06 | 230 / 800 | 0.288 | 2.94 | yes | 4 |
| 5 | 10,600 | 2,380,470 | 594,501 | 8,839 | 1.5140 | 0.9772 | 0.0337 | 1.67e-06 | 199 / 800 | 0.249 | −0.09 | no | 4 |

Every generation's held-out set is its own newest games, so the held-out columns do not compare
across rows. On the one set no net trained on (gen-5's held-out games, 530 games), held-out score
(policy + value) falls every generation: gen-0 2.568, gen-1 2.501, gen-4 2.461, gen-5 2.453
(`alphasettler fast-score`). The 800-game gate cannot see a one-generation gain of that size, which
is why gens 2, 3 and 5 failed it.

No extra generations were run: the headline was met after generation 5.

## Warm start (done criterion 3, recorded)

`net:gen-000 vs ismcts@1000: win rate 0.432 [0.397, 0.468] 95% CI over 200 seeds (800 games, 346
wins, 0 draws), z vs 0.25 = 10.06`, belief resets 0. These seeds (0–199) are also the boards of
gen-0 self-play games 0–199, of which 190 were training games. The fresh-seed rerun of the headline
below shows that overlap moved the result by 0.004.

## Acceptance gate (done criterion 3b)

Not met as written, and waived by the user on 2026-10-04: network v2's net₀ reached top-1
agreement 0.590 on setup settlements and 0.766 on robber moves but 0.419 on roads (bar 0.5); the
hybrid bar (net prior + heuristic value above 0.140) passed at 0.285. The table is in
[`docs/perf/net-search.md`](../perf/net-search.md).

## Headline (done criterion 4): met

`net:best` is gen-4 in every run.

| Run | Result | Bar | Wall time |
|---|---|---|---|
| vs three `ismcts@1000`, seeds 0–199 | `win rate 0.480 [0.444, 0.516] 95% CI over 200 seeds (800 games, 384 wins, 0 draws), z vs 0.25 = 12.47`, belief resets 0 | z > 3 | 6,476 s |
| vs three `ismcts@1000`, fresh seeds from 2,000,000,000 | `win rate 0.476 [0.441, 0.511] 95% CI over 200 seeds (800 games, 381 wins, 0 draws), z vs 0.25 = 12.60`, belief resets 0 | z > 3 | shared CPU, paused once |
| vs three `catanatron:alphabeta`, seeds 0–199 (Catanatron builds its own boards) | `win rate 0.585 [0.548, 0.622] 95% CI over 200 seeds (800 games, 468 wins, 0 draws), z vs 0.25 = 17.59`, fallbacks 0, belief resets 0 | win rate > 0.383, z > 3 | 15,928 s |

The native seeds 0–199 overlap gen-0's training boards; the fresh-seed run is the clean number and
agrees within 0.004. The Catanatron run's boards come from Catanatron's own generator, so they were
never trained on.

## Strength curve (done criterion 5)

`net:best` (gen-4) vs three GreedyBot, 200 fresh seeds from 2,000,000,000, beside 3a's curve:

| Simulations | 3b win rate | z vs 0.25 | 3a win rate |
|---|---|---|---|
| 100 | 0.580 | 18.19 | 0.330 |
| 300 | 0.680 | 24.87 | 0.388 |
| 1,000 | 0.671 | 21.47 | 0.443 |
| 3,000 | 0.684 | 25.45 | 0.435 |

Every run: belief resets 0. Against GreedyBot the curve flattens from 300 simulations (0.680, 0.671, 0.684), near a
ceiling against a weak opponent; against itself, more search keeps paying (`@1000` beats three
`@200` at 0.307 over 96 seeds by sequential test, `runs/search/sweep.log`), so 3a's plateau, where
search stopped helping against strong opponents, is gone.

## Done criteria checklist (spec Section 5)

1. **Tests:** the Rust workspace and the Python suite with the `train` extra pass with no skips
   (Plan 5 and 5b task reports; re-run at each later commit). Encoder golden and rotation tests:
   `net/tests/encode.rs`. Prior invariance: `net/tests/evaluator.rs`. Parity:
   `tests/python/test_train.py::test_rust_and_pytorch_forward_passes_agree_on_random_weights`.
   Weights-file errors: `net/tests/weights.rs`, `test_weights_round_trip_and_bad_files_are_errors`.
   Training data: `net/tests/data.rs`, `tests/python/test_training_data.py`,
   `test_fast_searches_add_nothing_to_the_policy_loss`, `test_a_tiny_dataset_can_be_overfit`.
   Self-play: `bots/tests/selfplay.rs`. Loop and resume: `tests/python/test_train_loop.py`. Gating
   rule: `test_train_loop.py::test_the_promotion_rule_includes_a_tie_at_a_quarter`. Names: `bots/tests/netbot.rs`,
   `tests/python/test_netbot.py`, `test_net_names_inside_catanatron`. Every arena run above: 0
   fallbacks, 0 belief resets. **Met.**
2. **Throughput recorded:** `docs/perf/net-search.md`. **Met.**
3. **Warm start recorded:** above. **Met.**
3b. **Acceptance gate:** roads at 0.419 against 0.5. **Waived by the user.**
4. **Headline:** both runs pass with wide margins. **Met.**
5. **Strength curve recorded:** above. **Met.**

## After the generations

The fast tooling (`alphasettler fast-train`, `fast-score`, `fast-loop`, `sprt`) came out of these
runs: a ten-minute fresh net reaches gen-1's held-out loss; every net trained on gens 1–5
(widths 48–96, rounds 2–3, fine-tunes) levels off near gen-4's strength; 96/3 and 64/3 nets cost
about 2.7× per search and do not beat gen-4. Sub-project 3c (domestic trading) starts from gen-4.
