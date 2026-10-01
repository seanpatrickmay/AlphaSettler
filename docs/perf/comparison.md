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
- **Catanatron's random player never offers domestic trades** and discards are random
  (no discard decisions), so its games are closest to our trades-off config.
- **catan-rl allows winning at 10 VP on any turn**; ours wins only on the player's own turn.

Reading: with trades off we are ~220× faster per step than Catanatron and within ~8% of
catan-rl's per-step cost; with trades on we are ~2.1× slower per step than catan-rl.
