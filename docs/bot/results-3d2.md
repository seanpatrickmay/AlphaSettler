# Sub-project 3d-2 (encoder v4: road-network and endgame-threat inputs): results

Date: 2026-10-09. Machine: Apple M1 Pro, macOS (25.5.0).
Commit: e7d9390 (Plan 10, Task 3; built and run at 3bce48e, which adds only run scripts), plus the commit adding this file. Encoder v3 build: 57e416a plan 12: 3d-4, compound build moves in search.

Spec: [`docs/superpowers/specs/2026-10-08-encoder-v4-roads-threats-design.md`](../superpowers/specs/2026-10-08-encoder-v4-roads-threats-design.md).
Plan: [`docs/superpowers/plans/2026-10-08-plan-10-encoder-v4.md`](../superpowers/plans/2026-10-08-plan-10-encoder-v4.md).
Builds on: [`results-3d1b.md`](results-3d1b.md) (arm C, the opening model in self-play) and
[`results-p2-gumbel.md`](results-p2-gumbel.md) (arm A).

## Encoder v4

Per node, the fewest roads each seat must build to reach it (0 to 3, a third each; 3 or more, or
never, is 1) and whether it is an open spot; per edge, whether it lies on each seat's longest trail;
per seat, its longest road / 15, roads to the longest-road award / 5, knights to the largest-army
award / 3, points to win / 10 (the viewer's own VP cards counted) and settlements and cities left.
`PUBLIC_LEN` 1,581 → 2,163.

## The upgrade (done criterion 2)

upgraded runs/3c/nets/gen-034.bin (encoder v3) to encoder v4: width 48, 2 rounds; 5 node, 4 edge and 24 global inputs zeroed; wrote runs/3d2/gen-034-v4.bin

    5843c12edb4615ec784f89fe039c0dfb399e57fff96ba84557ad13b755713092  runs/3c/nets/gen-034.bin
    defab19e9e1f80a94c96c16989973c5bd34a72dce1bdebc081d593155746b105  runs/3d2/gen-034-v4.bin

The v3 build's gen-034 and the v4 build's upgrade on the same 200 positions of random trading
games (runs/3d2/forward.log):

    batch policy: 133000 outputs, max relative difference 0, bit-equal 133000 of 133000
    batch value: 800 outputs, max relative difference 0, bit-equal 800 of 800
    batch aux: 800 outputs, max relative difference 0, bit-equal 800 of 800
    single policy: 133000 outputs, max relative difference 8.94e-07, bit-equal 102227 of 133000
    single value: 800 outputs, max relative difference 1.01e-06, bit-equal 196 of 800
    single aux: 800 outputs, max relative difference 1.19e-07, bit-equal 454 of 800
    worst 1.01e-06: pass (parity rule 1e-5)

## Search speed (done criterion 4)

`net_speed speed PATH 500 1000 8`: one core, batch 8, 1,000 simulations, the no-trade game against
three GreedyBots; v3 is gen-034 in the encoder v3 build, v4 the same net upgraded.

| run | v3 simulations/s | v4 simulations/s | v3 ms/decision | v4 ms/decision |
|---|---|---|---|---|
| 1 | 27,939 | 27,770 | 35.79 | 36.01 |
| 2 | 28,217 | 27,564 | 35.44 | 36.28 |
| 3 | 27,835 | 27,505 | 35.93 | 36.36 |
| mean | 27,997 | 27,613 | 35.72 | 36.22 |


v4 / v3: 0.99 in simulations/s (27,613 against 27,997; the six runs alternated builds and played the same 8 games).

## Training at equal wall time (done criterion 3)

Arm D is a `fast-loop` from gen-034 upgraded to v4, with exactly arm A's settings, run through
the same script (`bash runs/p2/arm.sh D --start-net runs/3d2/gen-034-v4.bin`):
a fresh root, the trading game, seed directories `runs/3b-v2/gen-002` to `gen-005` and
`runs/3c/gen-030` to `gen-034` (records written by encoder v2 and v3 nets, re-encoded under v4 from their actions), 2,000
games a generation, a window of 40,000 games, 10 minutes of training, a gate of 100 seeds at 100
simulations; set up untimed, then 10,800 s of generations (D cut after 10,803 s).
Arm A is the control (3d-1b was H0), not rerun. Both arms played the same
self-play and gate seeds.

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

### D: best net gen-040, 6 generations finished in the budget

| gen | games | decisions | full searches | steps | held-out score | policy | value | aux | gate wins / games | z | promoted | self-play minutes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 35 | 2,000 | 651,377 | 162,174 | 12,953 | 2.5230 | 1.6036 | 0.9194 | 0.0332 | 109 / 400 | 1.22 | yes | 17.8 |
| 36 | 2,000 | 651,720 | 163,085 | 13,002 | 2.5806 | 1.6492 | 0.9314 | 0.0335 | 101 / 400 | 0.13 | yes | 17.0 |
| 37 | 2,000 | 660,309 | 165,025 | 13,018 | 2.5892 | 1.6630 | 0.9262 | 0.0333 | 95 / 400 | -0.68 | yes | 16.5 |
| 38 | 2,000 | 657,702 | 164,469 | 12,999 | 2.5850 | 1.6549 | 0.9301 | 0.0333 | 109 / 400 | 1.24 | yes | 16.6 |
| 39 | 2,000 | 658,252 | 165,336 | 12,844 | 2.6618 | 1.7187 | 0.9431 | 0.0336 | 104 / 400 | 0.52 | yes | 17.1 |
| 40 | 2,000 | 664,437 | 165,871 | 12,950 | 2.6839 | 1.7371 | 0.9468 | 0.0336 | 103 / 400 | 0.43 | yes | 16.6 |

self-play: 12,000 games in 1.69 h of self-play, 7,084 games per hour

Final nets: 

    upgraded runs/p2/nets/A.bin (encoder v3) to encoder v4: width 48, 2 rounds; 5 node, 4 edge and 24 global inputs zeroed; wrote runs/3d2/nets/A.bin
    D: gen-040 (runs/3d2/nets/D.bin)
    86dbad0524820da7f55fdfb24954d2eb04f3c7b302ea37131fdd50430f5c160a  runs/3d2/nets/D.bin
    31909c7f9595d88bbadd179299a270013216c8e373a2ec33aa09acf171d7fbbe  runs/3d2/nets/A.bin
    defab19e9e1f80a94c96c16989973c5bd34a72dce1bdebc081d593155746b105  runs/3d2/gen-034-v4.bin.

| Run | Result | Bar | Outcome |
|---|---|---|---|
| `sprt --candidate net:D@100+t --baseline net:A@100+t --trades`, seeds from 5,000,000,000 | net:D@100+t vs net:A@100+t: win rate 0.224 [0.167, 0.281] 95% CI over 48 seeds (192 games, 43 wins, 0 draws), z vs 0.25 = -0.90, mean VP 6.88, mean turns 90.6; sprt: H0: no gain worth 0.3 after 48 seeds (192 games), LLR -3.09 | H1 (p1 0.30 against p0 0.25, alpha = beta = 0.05) | H0: recorded; new loops start from the control's best, upgraded |

Belief resets: 192 games, belief resets 0.

## Rulings the build made

- A node another seat built on at the end of a seat's road is on that seat's network (distance 0),
  but no path leaves it. A seat with nothing on the board reads 1 everywhere. Distances ignore road
  supply and cards.
- Ties between longest trails go to the engine walk's first (starts and edges in ascending edge
  order, strictly longer replaces); `longest_road` returns `longest_trail`'s length, and the golden
  trace did not move.
- Roads to the award use the holder's trail length as the encoder computes it; with no holder, a
  seat at 5 or more reads 0 to go even when tied (the spec's formula as written).
- The viewer's points include VP cards bought this turn; `vp_to_win` scales by the game's config.
  Six scales entered the layout: distance 3, road length 15, road award 5, army award 3,
  settlements 5, cities 4.
- An upgraded net equals the original within the forward pass's 1e-5, not bit for bit: Accelerate
  sums the wider rows in another order (the bit-equal counts above).
- Encoder v2 nets upgrade straight to v4 (the trade rows reset as before); `net-upgrade` always
  writes this build's encoder. Loaders name `net-upgrade` for v2 and v3 files.
- P2's net pin keeps its digest: its net is v3's random net upgraded (`Weights::random_v3`).
- Arm D ran through `runs/p2/arm.sh` unchanged, its start net replaced by a second `--start-net`;
  the control was not rerun, and its best net was upgraded for the SPRT (`runs/3d2/nets`).
- Speed and outputs were compared against a build of the commit before Task 1.
- Fresh seeds: 5,000,000,000.

## Next

The v4 inputs did not train a stronger net in three hours: D's best (gen-040) scored 0.224
against A's best upgraded (95% CI [0.167, 0.281]), so the SPRT rules out a gain of 0.05 at
alpha = beta = 0.05; the point estimate is below 0.25 and the interval includes it. Both arms finished six generations. D's training made about 2% fewer steps in each 10-minute
window (12,844 to 13,018 against A's 13,175 to 13,280), the second cost of v4 at equal wall time
beside its 1% slower search. D's higher self-play rate (7,084 games an hour against 6,829) cannot
come from v4, which searches more slowly; arm A was not rerun (it ran on 2026-10-08 from 05:55, at
90097ed), so the gap is the machine on the two days. Held-out scores are close (D's value loss ends
at 0.9468, A's at 0.9620), but each arm's held-out set is its own window, so they are not a strength
measure.
Zero-initialised inputs start the net where v3 left it (within 1e-5; bit-equal in a batch), and six generations of training
continued from that point may simply be too little to learn what the new inputs say; whether they
pay off over longer training is open. New loops start from the control's best upgraded to v4 (the
code stays at v4, which costs only the speed above); 3d-3's league arm runs with A's settings on
encoder v4.
