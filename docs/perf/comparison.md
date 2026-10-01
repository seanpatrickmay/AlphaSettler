# Engine comparison: AlphaSettler vs catan-rl vs Catanatron

Reproduce: `bench/compare/run.sh 20000`. Random players, same machine, uniform choice over
legal actions. A "step" is one legal-action generation + choice + apply. ns/step is the fair
metric; games/s depends on how long each engine's random games last.

## 2026-10-01 baseline (before optimization work)

Machine: Apple M1 Pro, 10 cores. Commits: alphasettler cc91f97, catan-rl 021279c, catanatron ecf9311.

| Engine | Config | 1 thread ns/step | 1 thread games/s | steps/game | 10 threads steps/s |
|---|---|---|---|---|---|
| AlphaSettler | domestic trades off | 93.6 | 9,725 | 1,099 | 80.7M |
| AlphaSettler | trades on (≤3 offers/turn, ≤2 cards per side) | 185.3 | 1,255 | 4,300 | 42.5M |
| catan-rl | random players, its trade menu | 87 | 3,898 | 2,940 | 71.7M |
| Catanatron | random players | 20,524 | 46 | 1,054 | — (single-threaded Python) |

Rule differences that affect these numbers:

- **Trade menus differ.** catan-rl offers 1–2 of one resource for 1 of another (≤ 40 offers);
  ours offers any bundle of ≤ 2 cards for any disjoint bundle of ≤ 2 cards (≤ 400 offers).
  Random players offer often, so this changes both steps/game and per-step cost.
- **Catanatron's random player never offers domestic trades.** It chooses its discards one
  card per action, as ours does (Catanatron 3.3; only pre-3.0 discarded at random), so its
  games are closest to our trades-off config.
- **catan-rl allows winning at 10 VP on any turn**; ours wins only on the player's own turn.

Reading: with trades off we are ~220× faster per step than Catanatron and within ~8% of
catan-rl's per-step cost; with trades on we are ~2.1× slower per step than catan-rl.

## 2026-10-01 after optimization (alphasettler d140be6)

Same machine and harness. Changes, each verified against the golden trace (600 seeded games,
every legal-action list, event and final state) and the longest-road reference oracle:
bitmask frontier for road/settlement legality, a precomputed table of trade offers per give
bundle, longest-road recompute limited to the new road's network, and a compact per-call
graph for the trail search that starts only from dead ends, branch points and blocked nodes.

| Engine | Config | 1 thread ns/step | 1 thread games/s | steps/game | 10 threads steps/s |
|---|---|---|---|---|---|
| AlphaSettler | trades off | **68.2** (was 93.6) | 13,347 | 1,099 | 115.5M |
| AlphaSettler | trades on, ≤2 cards per side (≤400 offers) | **57.7** (was 185.3) | 4,033 | 4,300 | 121.7M |
| AlphaSettler | trades on, 1 card for 1 (20 offers) | **54.2** | 4,775 | 3,866 | 141.3M |
| catan-rl | random players, its trade menu (≤40 offers) | 88 | 3,869 | 2,940 | 70.1M |
| Catanatron | random players | 20,691 | 46 | 1,052 | — |

Our two trades-on configs bracket catan-rl's menu size (20 and ≤400 offers vs its ≤40); both are
~1.5–1.6× faster per step. Trades off is ~300× faster per step than Catanatron.

Tried and not kept: `-C target-cpu=native` (no gain) and PGO (≤2%, not worth the build complexity).
