# IsmctsBot search throughput

Machine: Apple M1 Pro, 10 cores (one core used), otherwise idle
Commit: 5b534c8 plus the then-uncommitted `bots/examples/search_speed.rs`
Config: domestic trades off (`max_offers_per_turn = 0`), exact belief (cap 65,536 states), batch 8,
c_puct 1.5, heuristic evaluator with `DEFAULT_WEIGHTS` as of 5b534c8, rollout 0

IsmctsBot sits in seat 0 against three GreedyBots. Only decisions that searched are timed (a decision
with one legal move, or an answer to a trade offer, does not search). The timed span is the whole
`act` call, which includes sampling the worlds and building the tree.

Command: `cargo run --release -p settler-bots --example search_speed 2000 1000`

| Run | Output |
|---|---|
| 1 | 2027 searched decisions over 34 games at 1000 simulations: 708105 simulations/s, 1.41 ms/decision, 1111 nodes/decision |
| 2 | 2027 searched decisions over 34 games at 1000 simulations: 696050 simulations/s, 1.44 ms/decision, 1111 nodes/decision |
| 3 | 2027 searched decisions over 34 games at 1000 simulations: 699229 simulations/s, 1.43 ms/decision, 1111 nodes/decision |

That is about 700k simulations/s per core, or 1.4 ms per decision at 1,000 simulations. The spec
estimated 50k–100k simulations/s and 10–20 ms per decision, so the search is about 7 to 14 times
faster than estimated. The estimate assumed a replay of 10–30 actions at about 60 ns each, which
is only 0.6–1.8 µs, plus a world sample and an evaluation. To reach 10–20 µs per simulation, the
estimate must have budgeted roughly 10 µs for the sample and the evaluation together. The measured
1.4 µs per simulation means those two are far cheaper than that. No per-step profile was taken, so
how the 1.4 µs splits between replay, sampling and evaluation is not known. With rollout 0, each
leaf costs one heuristic evaluation.
