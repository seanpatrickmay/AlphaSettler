# Network search: throughput, the batch sweep and the acceptance gate (sub-project 3b)

Machine: Apple M1 Pro. Commit: 9e451ed, plus the commit that adds this file.
Config: trades off, exact belief with 65,536 states, c_puct 1.5, 1,000 simulations. `speed` runs on one
core; `agree`, `hybrid`, self-play and the arena run on all ten.

## Network v2 (entity-structured graph net), net₀

- Net: `runs/3b-v2/nets/gen-000.bin`, sha256 27eadefdc8b3d0fc2a349246dc864e04225630fe2cd1ceb852b1390ee7cf1cd6,
  width 48, 2 rounds, 247,294 parameters, 2,784,912 multiply-adds per position.
- Training: 45,494 steps from scratch on the 100,000-game warm start (95,000 training games, 5,000
  held out). Best held-out losses: policy 1.414, value 0.733, aux 0.031, at step 42,000. The MLP's were
  policy 1.550, value 0.811 and aux 0.037, at step 36,000.
- Random v2 weights (Plan 5b Task B): `batch 8: 300 searched decisions over 13 games at 1000
  simulations: 3721 simulations/s, 3713 evaluations/s, 268.78 ms/decision`.

### Acceptance gate 3b

The search's most-visited move is compared with each prior's top move, over 100 fresh noisy `ismcts`
self-play games (`prior_check agree`, 6,382 full searches):

| Move type | n | heuristic top-1 | net top-1 | net = heuristic |
|---|---|---|---|---|
| buy-dev | 413 | 0.998 | 0.995 | 0.998 |
| city | 145 | 0.993 | 0.931 | 0.938 |
| discard | 900 | 0.999 | 0.922 | 0.923 |
| end-turn | 947 | 1.000 | 0.951 | 0.951 |
| maritime | 754 | 0.893 | 0.781 | 0.855 |
| play-dev | 380 | 0.734 | 0.755 | 0.916 |
| road | 599 | 0.920 | **0.419** | 0.446 |
| robber | 674 | 0.926 | **0.766** | 0.792 |
| roll | 642 | 1.000 | 0.995 | 0.995 |
| settlement | 97 | 1.000 | 0.918 | 0.918 |
| setup-road | 193 | 1.000 | 0.881 | 0.881 |
| setup-settlement | 183 | 0.885 | **0.590** | 0.601 |
| steal | 455 | 0.991 | 0.910 | 0.919 |
| ALL | 6,382 | 0.952 | 0.837 | 0.862 |

The tool prints `agreement gate (net-top1 >= 0.5 on setup-settlement, robber, road, over >= 3000
searches): FAIL over 100 games, 6382 searches`. Roads came in at 0.419. With the user's agreement
(2026-10-04), the run went ahead anyway. The agreement rows measure how well the net imitates the
heuristic search, and a mid-game road often has several near-equal choices. The strength results below
are the criterion that matters.

Each mix plays against three `ismcts@100` at 100 simulations, seeds 0..50 (`prior_check hybrid`, 200
games each, 0 draws, belief resets 0 in every line):

| Prior + value | Win rate | z vs 0.25 | Mean VP |
|---|---|---|---|
| heuristic + heuristic | 0.250 (50/200) | 0.00 | 6.74 |
| net + net | 0.445 (89/200) | 6.57 | 7.89 |
| net + heuristic | 0.285 (57/200) | 1.12 | 7.20 |
| heuristic + net | 0.215 (43/200) | −2.00 | 6.59 |

The tool prints `hybrid gate (net prior + heuristic value above 0.140): pass at 0.285`.

### Throughput (one core, `net_speed speed runs/3b-v2/nets/gen-000.bin 500 1000 B`)

| batch | simulations/s (3 runs) | mean | evaluations/s (3 runs) | ms/decision (3 runs) |
|---|---|---|---|---|
| 8 | 4,151 / 4,203 / 4,159 | 4,171 | 4,092 / 4,143 / 4,100 | 240.90 / 237.95 / 240.45 |
| 16 | 4,197 / 4,170 / 4,205 | 4,191 | 4,064 / 4,038 / 4,072 | 238.29 / 239.83 / 237.82 |
| 32 | 4,177 / 3,638 / 3,539 | 3,785 | 4,066 / 3,541 / 3,445 | 239.42 / 274.90 / 282.60 |

The paired batch match was not run. Under Plan 5 Task 8's rule, a batch qualifies only if its mean is at
least 10% above batch 8's. Batch 16 is +0.5% and batch 32 is −9.3%, so neither qualifies and the default
stays 8 whatever the paired result. `NET_BATCH` = 8.

Self-play: 200 games of `net:gen-000` (41,882 searched decisions, 10,399 full searches) on 10 threads in
406.20 s real (runs/3b-v2/selfplay-speed.log).
Self-play games per generation: 10600

Warm-start check (done criterion 3, recorded): `net:gen-000 vs ismcts@1000: win rate 0.432 [0.397,
0.468] 95% CI over 200 seeds (800 games, 346 wins, 0 draws), z vs 0.25 = 10.06, mean VP 7.93, mean turns
91.0`, belief resets 0, 32:05 wall time.

## History: the flat MLP (network v1, format 1, refused since Plan 5b)

`public 1,557 → 512 → 512 → 512`, a 665-way policy head, value on [trunk ‖ hidden 30] → 256 → 4.
Net `runs/3b/nets/gen-000.bin`, sha256 26abda91510a15624bd7e788ad10bb54c09b44301231706fc8c332096c4c482e,
trained 45,494 steps on the same 100,000 games; best held out policy 1.550, value 0.811, aux 0.037 at
step 36,000.

| batch | simulations/s (3 runs) | mean | evaluations/s (3 runs) | ms/decision (3 runs) |
|---|---|---|---|---|
| 8 | 4,767 / 4,563 / 4,554 | 4,628 | 4,742 / 4,539 / 4,530 | 209.77 / 219.17 / 219.61 |
| 16 | 4,773 / 4,529 / 4,825 | 4,709 | 4,755 / 4,512 / 4,807 | 209.51 / 220.80 / 207.25 |
| 32 | 4,259 / 4,017 / 3,935 | 4,070 | 4,230 / 3,990 / 3,909 | 234.81 / 248.92 / 254.10 |

MLP self-play: 200 games, 50,944 searched decisions (12,872 full searches) on 10 threads in 299.43 s real.
Warm-start check: `net:gen-000` vs `ismcts@1000`, win rate 0.077 [0.057, 0.098] over 200 seeds (62 wins
of 800), z vs 0.25 = −16.14, mean VP 5.03.

Why it was replaced (spec Section 2): its prior's top move matched the search's most-visited move 58%
of the time over 3,954 full searches (the heuristic prior: 95%). It matched 9% on setup settlements, 11%
on robber moves and 19% on roads. At 100 simulations vs three `ismcts@100` over 50 seeds, heuristic prior
and value scored 0.250, net prior with heuristic value 0.140, and heuristic prior with net value 0.200.
