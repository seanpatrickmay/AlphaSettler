# Plan 5b: Network v2 (the Entity-Structured Graph Net) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace AlphaSettler's flat MLP evaluator with network v2, the entity-structured graph net of the amended spec, make the policy's player ids seat-relative, retrain net₀ on the existing 100,000-game warm start, and pass acceptance gate 3b before any learned generation runs.

**Architecture:** `net/` gains `policy.rs` (the one function mapping an `Action` and its actor to a policy slot, used by both the training sampler and `NetEvaluator`) and `graph.rs` (static adjacency built once from the engine's topology). `weights.rs` moves to format 2 (width `d` and rounds `R` in the header; version 1 refused) and `forward.rs` becomes the graph forward pass, where each per-entity linear map runs as one matrix product over every entity of its type in the batch. The PyTorch twin (`alphasettler/train/model.py`) mirrors it, reading the same adjacency through the bindings, and `train` / `train-loop` take `--width` / `--rounds`. A new example, `bots/examples/prior_check.rs`, measures the gate; the controller then retrains net₀ in a fresh `runs/3b-v2` on the existing records, runs the gate, the throughput sweep and the warm-start check, writes `docs/perf/net-search.md` and hands back to Plan 5 Task 9.

**Tech Stack:** Rust 1.80+ (edition 2021, no new external crates), PyO3 0.29.3 + maturin 1.15.0, Python 3.12 stdlib, PyTorch 2.14.1 (`train` extra only, MPS for training), pytest 9.1.1.

**Spec:** `docs/superpowers/specs/2026-10-02-bot-3b-network-design.md`, as amended 2026-10-04: Section 2 ("Network v2", "Seat-relative player ids", the weights file, the acceptance gate) and Section 5 (done criterion 3b). **Amends** `docs/superpowers/plans/2026-10-02-plan-5-network-bot.md` (Tasks 1–7 and 7b are done at `2d4742f`): this plan replaces the MLP of Plan 5 Tasks 2 and 6, does Plan 5 Task 8's measurements for network v2 (Task 8 is not rerun), and hands back to Plan 5 Task 9.

## Global Constraints

- Rust MSRV 1.80, edition 2021. The installed toolchain is newer: use nothing stabilised after 1.80 (no `is_multiple_of`, no `as_chunks`). No new external crates. `settler-net` depends only on `settler-engine` and `settler-search`; dependencies run one way, `bots → net → search → engine`.
- Commands: Rust is `~/.cargo/bin/cargo` (cargo is not on PATH). Python tools are `.venv/bin/python3`, `.venv/bin/pytest`, `.venv/bin/alphasettler`. Never `python` or `pip`.
- Rebuild the extension after any Rust change used from Python: `PATH=$HOME/.cargo/bin:$PATH .venv/bin/maturin develop --release -q`.
- Every task ends with `~/.cargo/bin/cargo fmt --all` then `~/.cargo/bin/cargo fmt --check` (no output), `~/.cargo/bin/cargo clippy --workspace --release --all-targets -- -D warnings` (exit 0), the full `~/.cargo/bin/cargo test --workspace --release` (all pass) and, after the rebuild, `.venv/bin/pytest -q -rs` (all pass, the summary lists no skipped tests).
- The engine does not change: no file under `engine/` is edited, and `engine/tests/golden_trace.rs` stays green **without re-pinning**.
- The encoder does not change: `net/src/encode.rs` and `net/tests/encode.rs` are not edited. `ENCODER_VERSION` 2, `ENCODING_HASH`, `PUBLIC_LEN` 1557 (nodes 54 × 19 from `NODE_BASE`, edges 72 × 4 from `EDGE_BASE`, tiles 19 × 8 from `TILE_BASE`, 91 global scalars from `OWN_HAND`) and `HIDDEN_LEN` 30 stay. Library code reads block offsets from those constants, never as literals.
- Determinism: same seed and weights give the same games, independent of thread count, and a forward-pass row never depends on the rest of its batch. Budgets are simulation counts, never wall time. 3b never trades: `OfferTrade` ids are always masked. Only `oracle/` imports Catanatron. `torch==2.14.1` lives only in the `train` extra, no module outside `alphasettler/train/` imports torch, and there is no numpy.
- Network v2 (spec Section 2): width `d`, `R` rounds, both in the weights header; defaults `d = 48`, `R = 2`, about 3M multiply-adds per position (2.78M as built). Encoders, one per entity type and shared across that type's entities, ReLU: node 19 → d, edge 4 → d, tile 8 → d, global 91 → d. R rounds, each with its own weights, residual (x ← x + ReLU(W·[...] + b)), in this order: tile ← [tile, mean of its nodes, global]; node ← [node, mean of its tiles, mean of its edges, mean of its neighbouring nodes, global]; edge ← [edge, sum of its two nodes, global]; global ← [global, mean of nodes, mean of edges, mean of tiles]. Policy (665 logits, masked to the legal moves, softmax): BuildSettlement(n), BuildCity(n) from a per-node linear map on [node n, global] → 2 logits; BuildRoad(e) from a per-edge map on [edge e, global] → 1; MoveRobber(t) from a per-tile map on [tile t, global] → 1; every other id from [global, mean of nodes, mean of edges, mean of tiles] → 4d → 466 logits. Value: [global, the three means, hidden 30] → 4d → 4, softmax. Aux: the value head's 4d layer → 4 (final VP / 10, training only). The hidden block enters only the value head.
- Seat-relative player ids: the policy treats `StealFrom(p)` and `ConfirmTrade(p)` as `StealFrom(rel(p, actor))` and `ConfirmTrade(rel(p, actor))`, both in the training targets and in `NetEvaluator`'s masking, through the one function `settler_net::policy::policy_id`.
- Weights file (spec Section 2): format version 2, little-endian. Header: magic `ASNET\0\0\0`, format version (u32), `ENCODING_HASH` (u64), `d` and `R` (u32 each), layer count and each layer's shape; then f32 weights and biases in layer order. Loading fails with a clear error on a wrong magic, version, hash, width, round count or shape, or on a short file; a version 1 file is refused with a message naming the flat MLP. Bounds: `1 ≤ d ≤ MAX_WIDTH = 1024`, `1 ≤ R ≤ MAX_ROUNDS = 16`, checked before anything is sized from the file.
- Rust/PyTorch parity: random weights, 100 positions from random games; the prior over legal moves and the value probabilities agree within 1e-5 absolute (raw logits and aux within 1e-5 × max(1, |PyTorch|), Plan 5's `parity_error`).
- Training keeps Plan 5 Task 7b as it is: value and aux losses on a seeded 1/16 of each batch's rows, the newest generation's seeds divisible by 20 held out, held-out evaluation every 500 steps and at the last, and export of the checkpoint with the lowest held-out policy + value.
- Acceptance gate 3b (spec Sections 2 and 5): over at least 3,000 full searches of fresh `ismcts` self-play, net₀'s top-1 agreement with the search's most-visited move is at least 0.5 on setup settlements, robber moves and roads; and its prior in the "net prior + heuristic value" hybrid at 100 simulations over 50 seeds beats the MLP's 0.140. If the gate fails, the controller stops and reports the numbers; no learned generation runs.
- Git: one commit per task with the message given in the task, run as `git -c user.email=sean.may101@gmail.com commit -m "..."`. No Co-Authored-By or AI attribution, never `--no-verify`, never push. Before every commit run the secret scan `git diff --cached | grep -nE 'sk-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{36}|password=[^[:space:]]+'` and expect no output.
- Long runs (net₀'s training, every multi-seed arena, the throughput sweep) are run by the controller in the background, never by implementer subagents. Steps marked **controller runs this** give the exact command and the pass condition.
- No README or docs beyond `docs/perf/net-search.md` (Task D) and `docs/bot/results-3b.md` (Plan 5 Task 9).

## Review Focus

1. A version 1 weights file (the flat MLP) reaching this build, for example `runs/3b/nets/gen-000.bin` through the default `--nets-dir runs/3b/nets`, or as `train --init`: refused with `format version 1, expected 2; version 1 holds the flat MLP that network v2 replaced: retrain the net` before any game is played or any record is read (Task B: the Rust and Python loader tests, `test_bad_names_missing_files_and_stale_hashes_are_errors`, `test_cli_names_a_missing_generation_and_checks_init_before_reading_records`).
2. `train-loop` pointed at the MLP's run directory (`runs/3b`, manifest format 1): refused at start, naming both formats, with nothing written, instead of reporting "nothing to do" or failing a generation later on an MLP net (Task B: `test_a_manifest_of_another_format_is_refused_before_anything_is_written`).
3. The warm-start records reused in a new run directory: hard-linked `selfplay.json` and chunks complete the new run's self-play step without playing a game, and the manifest entry equals the source run's (Task B: `test_linked_records_satisfy_a_new_runs_self_play_without_playing`; Task D relies on it).
4. A corrupt or hostile header declaring a huge width or round count (2^31, u32::MAX, MAX + 1): refused by both loaders before any allocation is sized from it, and Python builds no module first (Task B: `a_bad_file_is_an_error_naming_what_is_wrong`, `test_widths_and_rounds_are_checked_and_non_finite_weights_rejected_before_building_a_net`).
5. A width that is not a multiple of 8, where Rust's 8-lane dot product takes its remainder path: the forward pass still matches the f64 reference and the PyTorch twin (Task B: `the_forward_pass_matches_a_plain_reference` at d = 5, the parity test at width 5 with 3 rounds).

## File Structure

| File | Responsibility | Task |
|---|---|---|
| `net/src/policy.rs` | policy slots: `policy_id` (seat-relative steals and trade confirmations), the entity-head slot bases, `other_slot` | A |
| `net/src/graph.rs` | the global block's offset and width; static adjacency (`Links`, `Graph`, `graph()`) | A |
| `net/src/lib.rs` | module list | A |
| `net/src/data.rs` | training targets and mask at `policy_id` slots | A |
| `net/src/evaluator.rs` | `masked_prior` reads `policy_id` slots (new `actor` argument) | A |
| `net/src/weights.rs` | weights format 2: `Arch { width, rounds }`, the layer table, load and validate | B |
| `net/src/forward.rs` | the batched graph forward pass | B |
| `bindings/src/train.rs` | `write_random_net(path, seed, width, rounds)`, `net_adjacency()`, v2 constants | B |
| `alphasettler/train/model.py` | the PyTorch twin of network v2 and format 2 | B |
| `alphasettler/train/train.py`, `loop.py`, `alphasettler/cli.py` | `width` / `rounds` instead of `trunk` / `value`; manifest format 2 | B |
| `bots/examples/prior_check.rs` | the acceptance gate's two diagnostics, `agree` and `hybrid` | C |
| `docs/perf/net-search.md` | network v2's gate, throughput, batch sweep and warm-start numbers; the MLP's as history | D |
| `bots/src/netbot.rs` | `NET_BATCH`, only if the sweep picks another batch | D |
| `net/tests/{policy,graph,data,evaluator,weights}.rs`, `bots/tests/netbot.rs` | Rust tests | A, B |
| `tests/python/{test_training_data,test_train,test_train_loop,test_netbot,test_oracle_arena}.py` | Python tests | A, B |

Task order: the PyTorch twin, the training plumbing and the `--width` / `--rounds` flags share Task B with the Rust network because a version 1 file is refused: once Rust reads only format 2, the Python training tests (parity, export, the loop) cannot pass until the twin writes format 2, and every task must end with the whole suite green.

---

### Task A: Seat-relative policy ids and the board graph

**Files:**
- Create: `net/src/policy.rs`, `net/src/graph.rs`, `net/tests/policy.rs`, `net/tests/graph.rs`
- Modify: `net/src/lib.rs`, `net/src/data.rs` (the `Batch` field docs and one line of `Batch::push`), `net/src/evaluator.rs` (`masked_prior` and its one call)
- Test: `net/tests/data.rs`, `net/tests/evaluator.rs`, `tests/python/test_training_data.py`

**Interfaces:**
- Consumes: `settler_net::encode::{rel, NODE_BASE, NODE_WIDTH, EDGE_BASE, EDGE_WIDTH, TILE_BASE, TILE_WIDTH, OWN_HAND, PUBLIC_LEN}`; `settler_engine::topology::{topo, NUM_NODES, NUM_EDGES, NUM_TILES}` with `Topology::{node_tiles, node_edges, node_neighbors}: Vec<Vec<u8>>`, `edge_nodes: [(u8, u8); 72]`, `tile_nodes: [[u8; 6]; 19]`; `Action::encode`.
- Produces:
  - `settler_net::policy`: `SETTLEMENT_BASE` (25), `CITY_BASE` (79), `ROAD_BASE` (133), `ROBBER_BASE` (205), `ENTITY_END` (224), `OTHER_POLICY_LEN` (466), all `usize`; `pub fn other_slot(j: usize) -> usize` (the pooled head's output `j` → its slot: `j` below 25, else `j + 199`); `pub fn policy_id(a: Action, actor: PlayerId) -> usize`.
  - `settler_net::graph`: `GLOBAL_BASE: usize` (= `OWN_HAND`), `GLOBAL_WIDTH: usize` (91); `pub struct Links { pub start: Vec<usize>, pub index: Vec<u8>, pub scale: Vec<f32> }` with `count(&self) -> usize`, `of(&self, j: usize) -> &[u8]`, `lists(&self) -> Vec<Vec<usize>>`; `pub struct Graph { pub node_tiles, pub node_edges, pub node_neighbours, pub edge_nodes, pub tile_nodes: Links }` (`edge_nodes` sums, the rest average); `pub fn graph() -> &'static Graph`.
  - `settler_net::evaluator::masked_prior(logits: &[f32], legal: &[Action], actor: PlayerId) -> Vec<f32>` (the new `actor` argument; `NetEvaluator` passes the leaf's actor).
  - The training sampler's `Batch::mask` and `Batch::policy` are indexed by `policy_id` (the Python side sees the same bytes).

- [ ] **Step 1: Write the failing tests**

Create `net/tests/policy.rs`:

```rust
use settler_engine::topology::{topo, NUM_EDGES, NUM_TILES};
use settler_engine::*;
use settler_net::policy::*;

#[test]
fn entity_slots_are_the_engine_ids_and_the_pooled_head_has_the_rest() {
    assert_eq!(
        (
            Action::BuildSettlement(0).encode() as usize,
            Action::BuildCity(0).encode() as usize,
            Action::BuildRoad(0).encode() as usize,
            Action::MoveRobber(0).encode() as usize,
        ),
        (SETTLEMENT_BASE, CITY_BASE, ROAD_BASE, ROBBER_BASE)
    );
    assert_eq!(
        Action::BuildRoad(NUM_EDGES as u8 - 1).encode() as usize + 1,
        ROBBER_BASE
    );
    assert_eq!(
        Action::MoveRobber(NUM_TILES as u8 - 1).encode() as usize + 1,
        ENTITY_END
    );
    assert_eq!((ENTITY_END, OTHER_POLICY_LEN), (224, 466));
    let others: Vec<usize> = (0..OTHER_POLICY_LEN).map(other_slot).collect();
    let expected: Vec<usize> = (0..SETTLEMENT_BASE)
        .chain(ENTITY_END..ACTION_SPACE_SIZE)
        .collect();
    assert_eq!(others, expected);
}

#[test]
fn every_actor_maps_the_moves_one_to_one_onto_the_slots() {
    let moves: Vec<Action> = (0..ACTION_SPACE_SIZE as u16)
        .filter_map(Action::decode)
        .collect();
    for actor in 0..NUM_PLAYERS as PlayerId {
        let mut seen = vec![false; ACTION_SPACE_SIZE];
        for &a in &moves {
            let k = policy_id(a, actor);
            assert!(
                k < ACTION_SPACE_SIZE && !seen[k],
                "actor {actor}: {a:?} -> {k}"
            );
            seen[k] = true;
            if !matches!(a, Action::StealFrom(_) | Action::ConfirmTrade(_)) {
                assert_eq!(k, a.encode() as usize, "{a:?}");
            }
        }
    }
}

#[test]
fn steals_and_trade_confirmations_name_the_player_relative_to_the_actor() {
    for actor in 0..NUM_PLAYERS as PlayerId {
        for k in 0..NUM_PLAYERS {
            let p = ((actor as usize + k) % NUM_PLAYERS) as PlayerId;
            assert_eq!(policy_id(Action::StealFrom(p), actor), 224 + k);
            assert_eq!(policy_id(Action::ConfirmTrade(p), actor), 660 + k);
        }
    }
}

/// The same steal from every seat: a hand-built Steal phase whose only victim is the player after
/// the actor. Rotating the actor and the victim together leaves the steal's policy id unchanged.
#[test]
fn rotating_every_seat_leaves_a_steals_policy_id_unchanged() {
    let tile = 9usize;
    let mut ids = Vec::new();
    for actor in 0..NUM_PLAYERS as PlayerId {
        let victim = (actor + 1) % NUM_PLAYERS as PlayerId;
        let mut s = State::new(7, GameConfig::default());
        s.setup_step = 8;
        s.phase = Phase::Steal;
        s.return_phase = Phase::Main;
        s.current = actor;
        s.robber = tile as u8;
        s.players[victim as usize].settlements = 1u64 << topo().tile_nodes[tile][0];
        s.players[victim as usize].hand = [1, 0, 0, 0, 0];
        let legal = s.legal_actions();
        assert_eq!(legal, vec![Action::StealFrom(victim)], "actor {actor}");
        ids.push(policy_id(legal[0], actor));
    }
    assert_eq!(ids, vec![225; NUM_PLAYERS]);
}
```

Create `net/tests/graph.rs`:

```rust
use settler_engine::topology::{topo, NUM_EDGES, NUM_NODES, NUM_TILES};
use settler_net::encode::{OWN_HAND, PUBLIC_LEN};
use settler_net::graph::*;

fn lists(v: &[Vec<u8>]) -> Vec<Vec<usize>> {
    v.iter()
        .map(|l| l.iter().map(|&x| x as usize).collect())
        .collect()
}

#[test]
fn the_global_block_is_the_91_scalars_after_the_tiles() {
    assert_eq!((GLOBAL_BASE, GLOBAL_WIDTH), (OWN_HAND, 91));
    assert_eq!(GLOBAL_BASE + GLOBAL_WIDTH, PUBLIC_LEN);
}

#[test]
fn the_tables_are_the_engines_topology() {
    let (g, t) = (graph(), topo());
    assert_eq!(g.node_tiles.lists(), lists(&t.node_tiles));
    assert_eq!(g.node_edges.lists(), lists(&t.node_edges));
    assert_eq!(g.node_neighbours.lists(), lists(&t.node_neighbors));
    let ends: Vec<Vec<u8>> = t.edge_nodes.iter().map(|&(a, b)| vec![a, b]).collect();
    assert_eq!(g.edge_nodes.lists(), lists(&ends));
    let corners: Vec<Vec<u8>> = t.tile_nodes.iter().map(|c| c.to_vec()).collect();
    assert_eq!(g.tile_nodes.lists(), lists(&corners));
    assert_eq!(
        (
            g.node_tiles.count(),
            g.node_edges.count(),
            g.node_neighbours.count(),
            g.edge_nodes.count(),
            g.tile_nodes.count()
        ),
        (NUM_NODES, NUM_NODES, NUM_NODES, NUM_EDGES, NUM_TILES)
    );
}

#[test]
fn the_tables_agree_with_each_other() {
    let g = graph();
    // 19 hexes with 6 corners each; 72 edges with 2 ends each.
    assert_eq!(g.node_tiles.index.len(), 6 * NUM_TILES);
    assert_eq!(g.tile_nodes.index.len(), 6 * NUM_TILES);
    assert_eq!(g.node_edges.index.len(), 2 * NUM_EDGES);
    assert_eq!(g.node_neighbours.index.len(), 2 * NUM_EDGES);
    for n in 0..NUM_NODES {
        let me = n as u8;
        assert!((1..=3).contains(&g.node_tiles.of(n).len()), "node {n}");
        assert!((2..=3).contains(&g.node_edges.of(n).len()), "node {n}");
        for &t in g.node_tiles.of(n) {
            assert!(
                g.tile_nodes.of(t as usize).contains(&me),
                "node {n}, tile {t}"
            );
        }
        for &e in g.node_edges.of(n) {
            assert!(
                g.edge_nodes.of(e as usize).contains(&me),
                "node {n}, edge {e}"
            );
        }
        for &m in g.node_neighbours.of(n) {
            assert!(g.node_neighbours.of(m as usize).contains(&me), "symmetric");
            assert!(
                g.node_edges
                    .of(n)
                    .iter()
                    .any(|&e| g.edge_nodes.of(e as usize).contains(&m)),
                "node {m} is the far end of one of node {n}'s edges"
            );
        }
    }
    for links in [
        &g.node_tiles,
        &g.node_edges,
        &g.node_neighbours,
        &g.tile_nodes,
    ] {
        for j in 0..links.count() {
            assert_eq!(links.scale[j], 1.0 / links.of(j).len() as f32, "a mean");
        }
    }
    for e in 0..NUM_EDGES {
        let ends = g.edge_nodes.of(e);
        assert!(ends.len() == 2 && ends[0] != ends[1], "edge {e}");
        assert_eq!(g.edge_nodes.scale[e], 1.0, "a sum");
    }
}
```

In `net/tests/data.rs`, add three imports below the existing `use settler_net::encode::...` line:

```rust
use settler_net::evaluator::masked_prior;
use settler_net::policy::policy_id;
use settler_search::search_actions;
```

Replace the head of `recorded` so it delegates to a new `recorded_every` (the rest of the old body becomes `recorded_every`'s body, unchanged except the `% 5`):

```rust
/// A finished random game recorded with a decision at every 5th move: random visit counts (some
/// all zero), alternating full and fast searches.
fn recorded(seed: u64, config: GameConfig) -> RecordedGame {
    recorded_every(seed, config, 5)
}

/// `recorded` with a decision at every `every`-th move.
fn recorded_every(seed: u64, config: GameConfig, every: usize) -> RecordedGame {
    let mut s = State::new(seed, config);
```

and in that body change `if actions.len() % 5 == 0 {` to `if actions.len() % every == 0 {`.

In `a_position_equals_the_encoder_on_the_replayed_state`, the expected mask and policy move to policy slots:

```rust
                for &(a, v) in &searchable {
                    mask[policy_id(a, d.actor)] = 1;
                    policy[policy_id(a, d.actor)] = if total > 0 {
```

Append to `net/tests/data.rs`:

```rust
/// Seat-relative ids in both places (spec Section 2). With finite logits only in the slots the
/// sampler's mask marks, `masked_prior` must give every searchable move its softmax share of the
/// logit at that move's `policy_id`: the evaluator reads exactly the slots the training targets
/// fill, at every decision of three games, steals by every seat included.
#[test]
fn the_sampler_and_the_evaluator_use_the_same_policy_ids() {
    let config = GameConfig {
        max_turns: 150,
        ..no_trades()
    };
    let games: Vec<RecordedGame> = (1..4).map(|seed| recorded_every(seed, config, 1)).collect();
    let set = TrainingSet::new(games.clone()).unwrap();
    let (mut checked, mut relative_steals) = (0, 0);
    let mut legal = Vec::new();
    for (gi, g) in games.iter().enumerate() {
        let mut s = State::new(g.seed, g.config);
        for (di, d) in g.decisions.iter().enumerate() {
            assert_eq!(d.index as usize, di, "a decision at every move");
            let mut b = Batch::default();
            set.position(gi, di, &mut b);
            search_actions(&s.legal_actions(), &mut legal);
            let slots: Vec<usize> = (0..ACTION_SPACE_SIZE).filter(|&k| b.mask[k] == 1).collect();
            assert_eq!(slots.len(), legal.len(), "one slot per searchable move");
            let logit = |k: usize| k as f32 / 100.0;
            let mut logits = vec![f32::NEG_INFINITY; ACTION_SPACE_SIZE];
            for &k in &slots {
                logits[k] = logit(k);
            }
            let z: f32 = slots.iter().map(|&k| logit(k).exp()).sum();
            let prior = masked_prior(&logits, &legal, d.actor);
            for (a, p) in legal.iter().zip(&prior) {
                let k = policy_id(*a, d.actor);
                assert!(slots.contains(&k), "{a:?} by seat {}", d.actor);
                assert!(
                    (p - logit(k).exp() / z).abs() < 1e-5,
                    "{a:?} by seat {}",
                    d.actor
                );
                relative_steals +=
                    (matches!(a, Action::StealFrom(_)) && k != a.encode() as usize) as u32;
            }
            checked += 1;
            s.apply(g.actions[di]);
        }
    }
    assert!(checked > 1000, "{checked} decisions");
    assert!(
        relative_steals > 0,
        "steals by seats other than 0 were checked"
    );
}
```

In `net/tests/evaluator.rs`, pass actor `0` in the four existing `masked_prior` calls (`masked_prior(&vec![0.0; ACTION_SPACE_SIZE], &[offer], 0)`, `masked_prior(&dead, &[Action::EndTurn, offer, Action::BuyDev], 0)`, `masked_prior(&dead, &[offer, offer], 0)`, `masked_prior(&huge, &[Action::EndTurn, offer, Action::BuyDev], 0)`), and add before `values_are_rotated_back_to_absolute_seats`:

```rust
#[test]
fn a_steal_reads_the_slot_of_the_victim_relative_to_the_actor() {
    let mut logits = vec![0.0f32; ACTION_SPACE_SIZE];
    logits[225] = 10.0; // steal from the player after the actor
    for actor in 0..NUM_PLAYERS as PlayerId {
        let next = (actor + 1) % NUM_PLAYERS as PlayerId;
        let across = (actor + 2) % NUM_PLAYERS as PlayerId;
        let p = masked_prior(
            &logits,
            &[Action::StealFrom(across), Action::StealFrom(next)],
            actor,
        );
        assert!(p[1] > 0.99, "actor {actor}: {p:?}");
    }
}
```

In `tests/python/test_training_data.py`, add above `test_a_batch_equals_the_encoder_on_the_replayed_state`:

```python
def policy_id(action: int, actor: int) -> int:
    """`settler_net::policy::policy_id` on engine ids: a steal (224..228) or a trade confirmation
    (660..664) names its player relative to the actor."""
    for base in (224, 660):
        if base <= action < base + 4:
            return base + (action - base - actor) % 4
    return action
```

and replace that test with:

```python
def test_a_batch_equals_the_encoder_on_the_replayed_state(tmp_path):
    games = explore_games(tmp_path, 2)
    ts = TrainingSet(games)
    assert ts.games == 2 and ts.decisions == sum(len(g["decisions"]) for g in games)
    assert 0 < ts.full_decisions < ts.decisions
    checked = moved = 0
    for gi, g in enumerate(games):
        for di, d in enumerate(g["decisions"][:40]):
            b = ts.position(gi, di)
            game = Game(g["seed"], g["config"])
            for a in g["actions"][: d["index"]]:
                game.apply(a)
            assert game.current_actor == d["actor"]
            public, hidden = encode(game)
            assert b["n"] == 1 and b["public"] == public and b["hidden"] == hidden
            mask = list(b["mask"])
            slots = [policy_id(a, d["actor"]) for a in game.legal_actions()]
            assert [i for i in range(A) if mask[i]] == sorted(slots)
            moved += slots != game.legal_actions()
            policy = floats(b["policy"])
            total = sum(d["visits"])
            for a, v in zip(d["legal"], d["visits"]):
                assert policy[policy_id(a, d["actor"])] == pytest.approx(v / total, rel=1e-6)
            assert list(b["full"]) == [int(d["full_search"])]
            checked += 1
    assert checked > 40
    assert moved > 0, "a steal by a seat other than 0 sits at its seat-relative slot"
    batch = ts.sample(50, 3, 2)
    assert batch == ts.sample(50, 3, 1)
    assert (len(batch["public"]), len(batch["mask"]), len(batch["outcome"])) == (50 * len(public), 50 * A, 50 * 16)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `~/.cargo/bin/cargo test --release -p settler-net --tests`
Expected: FAIL to compile: `error[E0432]: unresolved import settler_net::policy` (and `settler_net::graph`), and `error[E0061]` on `masked_prior` taking 2 arguments, not 3.

Run: `.venv/bin/pytest -q tests/python/test_training_data.py::test_a_batch_equals_the_encoder_on_the_replayed_state`
Expected: FAIL on the mask assertion at a steal by a seat other than 0 (for example `[225, 226] == [224, 227]`): the installed extension still writes absolute ids.

- [ ] **Step 3: The policy slots**

Create `net/src/policy.rs`:

```rust
//! Policy ids (spec Section 2): the network's 665 policy slots. A move's slot is its engine id
//! (`Action::encode`), except that `StealFrom(p)` and `ConfirmTrade(p)` name `p` relative to the
//! actor, the way the inputs name players: "steal from the next player" has one slot whoever
//! acts. The training sampler (`data.rs`) and `NetEvaluator` both go through `policy_id`, so
//! targets and priors sit in the same slots. Slots `SETTLEMENT_BASE..ENTITY_END` come from network
//! v2's per-entity heads (settlements and cities per node, roads per edge, robber moves per
//! tile); every other slot comes from the pooled head, in slot order (`other_slot`).

use crate::encode::rel;
use settler_engine::topology::{NUM_EDGES, NUM_NODES, NUM_TILES};
use settler_engine::{Action, PlayerId, ACTION_SPACE_SIZE};

/// `BuildSettlement(n)` is slot `SETTLEMENT_BASE + n`; cities, roads and robber moves follow
/// contiguously (`net/tests/policy.rs` checks each base against `Action::encode`).
pub const SETTLEMENT_BASE: usize = 25;
pub const CITY_BASE: usize = SETTLEMENT_BASE + NUM_NODES;
pub const ROAD_BASE: usize = CITY_BASE + NUM_NODES;
pub const ROBBER_BASE: usize = ROAD_BASE + NUM_EDGES;
/// One past the last robber slot.
pub const ENTITY_END: usize = ROBBER_BASE + NUM_TILES;
/// The slots no entity head produces: 0..25 and 224..665.
pub const OTHER_POLICY_LEN: usize = ACTION_SPACE_SIZE - (ENTITY_END - SETTLEMENT_BASE);

/// The slot of the pooled head's output `j` (`j < OTHER_POLICY_LEN`).
#[inline]
pub fn other_slot(j: usize) -> usize {
    if j < SETTLEMENT_BASE {
        j
    } else {
        j + (ENTITY_END - SETTLEMENT_BASE)
    }
}

/// The policy slot of `a` played by `actor`.
#[inline]
pub fn policy_id(a: Action, actor: PlayerId) -> usize {
    let a = match a {
        Action::StealFrom(p) => Action::StealFrom(rel(p, actor) as PlayerId),
        Action::ConfirmTrade(p) => Action::ConfirmTrade(rel(p, actor) as PlayerId),
        other => other,
    };
    a.encode() as usize
}
```

- [ ] **Step 4: The board graph**

Create `net/src/graph.rs`:

```rust
//! The board as a graph for network v2 (spec Section 2): where the global scalars sit in the
//! public vector, and the static adjacency, built once from the engine's topology. The PyTorch
//! twin gets the same lists through the bindings (`net_adjacency`).

use crate::encode::{
    EDGE_BASE, EDGE_WIDTH, NODE_BASE, NODE_WIDTH, OWN_HAND, PUBLIC_LEN, TILE_BASE, TILE_WIDTH,
};
use settler_engine::topology::{topo, NUM_EDGES, NUM_NODES, NUM_TILES};
use std::sync::LazyLock;

/// The global scalars: every slot after the tile block, from the actor's own hand on (91).
pub const GLOBAL_BASE: usize = OWN_HAND;
pub const GLOBAL_WIDTH: usize = PUBLIC_LEN - GLOBAL_BASE;

// The entity blocks tile the public vector in order: nodes, edges, tiles, then the globals.
const _: () = assert!(EDGE_BASE == NODE_BASE + NUM_NODES * NODE_WIDTH);
const _: () = assert!(TILE_BASE == EDGE_BASE + NUM_EDGES * EDGE_WIDTH);
const _: () = assert!(GLOBAL_BASE == TILE_BASE + NUM_TILES * TILE_WIDTH);

/// Each entity's neighbours of one kind, flattened: entity `j`'s are `index[start[j]..start[j + 1]]`
/// in the order they are summed, and the sum is multiplied by `scale[j]` (1 / their count for a
/// mean, 1 for a sum).
#[derive(Clone, Debug, PartialEq)]
pub struct Links {
    pub start: Vec<usize>,
    pub index: Vec<u8>,
    pub scale: Vec<f32>,
}

impl Links {
    fn build(lists: &[Vec<u8>], mean: bool) -> Links {
        let mut links = Links {
            start: vec![0],
            index: Vec::new(),
            scale: Vec::new(),
        };
        for l in lists {
            assert!(!l.is_empty(), "every entity has a neighbour of each kind");
            links.index.extend_from_slice(l);
            links.start.push(links.index.len());
            links
                .scale
                .push(if mean { 1.0 / l.len() as f32 } else { 1.0 });
        }
        links
    }

    /// How many entities the table covers.
    pub fn count(&self) -> usize {
        self.scale.len()
    }

    /// Entity `j`'s neighbours.
    pub fn of(&self, j: usize) -> &[u8] {
        &self.index[self.start[j]..self.start[j + 1]]
    }

    /// Every entity's neighbours, as lists (for the bindings and tests).
    pub fn lists(&self) -> Vec<Vec<usize>> {
        (0..self.count())
            .map(|j| self.of(j).iter().map(|&k| k as usize).collect())
            .collect()
    }
}

pub struct Graph {
    /// Mean over a node's 1 to 3 tiles.
    pub node_tiles: Links,
    /// Mean over a node's 2 or 3 edges.
    pub node_edges: Links,
    /// Mean over a node's 2 or 3 neighbouring nodes.
    pub node_neighbours: Links,
    /// Sum over an edge's two nodes.
    pub edge_nodes: Links,
    /// Mean over a tile's six nodes.
    pub tile_nodes: Links,
}

static GRAPH: LazyLock<Graph> = LazyLock::new(|| {
    let t = topo();
    let edge_nodes: Vec<Vec<u8>> = t.edge_nodes.iter().map(|&(a, b)| vec![a, b]).collect();
    let tile_nodes: Vec<Vec<u8>> = t.tile_nodes.iter().map(|n| n.to_vec()).collect();
    Graph {
        node_tiles: Links::build(&t.node_tiles, true),
        node_edges: Links::build(&t.node_edges, true),
        node_neighbours: Links::build(&t.node_neighbors, true),
        edge_nodes: Links::build(&edge_nodes, false),
        tile_nodes: Links::build(&tile_nodes, true),
    }
});

pub fn graph() -> &'static Graph {
    &GRAPH
}
```

Replace `net/src/lib.rs` with:

```rust
//! The learned evaluator for AlphaSettler (spec:
//! docs/superpowers/specs/2026-10-02-bot-3b-network-design.md): the feature encoder, the board
//! graph and policy ids, the weights file, the forward pass, `NetEvaluator` and the training-data
//! sampler.

pub mod data;
pub mod encode;
pub mod evaluator;
pub mod forward;
pub mod graph;
pub mod policy;
pub mod weights;
```

- [ ] **Step 5: Targets and masking at policy slots**

In `net/src/data.rs`, add `use crate::policy::policy_id;` below the `use crate::encode::...` line; replace the docs of `Batch::mask` and `Batch::policy`:

```rust
    /// n × ACTION_SPACE_SIZE: 1 at the policy id (`policy::policy_id`, seat-relative for steals
    /// and trade confirmations) of each legal move except domestic offers.
    pub mask: Vec<u8>,
    /// n × ACTION_SPACE_SIZE: root visits normalised over the mask (uniform if none), at the same
    /// policy ids.
    pub policy: Vec<f32>,
```

and in `Batch::push` replace `let k = m0 + a.encode() as usize;` with:

```rust
            let k = m0 + policy_id(*a, actor);
```

In `net/src/evaluator.rs`, add `use crate::policy::policy_id;` below the `use crate::forward::...` line, and replace `masked_prior`'s doc and signature and its logit lookup:

```rust
/// The prior over `legal` for `actor`: a softmax of the policy logits at the moves' policy ids
/// (`policy::policy_id`: steals and trade confirmations seat-relative, as in the training
/// targets). Domestic offers always get 0 (3b never trades); if `legal` holds nothing else the
/// prior is uniform. With no finite maximum (every logit -inf, or some +inf) it is the softmax's
/// limit: uniform over the moves at the maximum, never NaN.
pub fn masked_prior(logits: &[f32], legal: &[Action], actor: PlayerId) -> Vec<f32> {
    let mut p: Vec<f32> = legal
        .iter()
        .map(|a| {
            if is_offer(a) {
                f32::NEG_INFINITY
            } else {
                logits[policy_id(*a, actor)]
            }
        })
        .collect();
```

(the rest of the function is unchanged). In `NetEvaluator::evaluate`, pass the leaf's actor:

```rust
                prior: masked_prior(
                    &self.out.policy[i * ACTION_SPACE_SIZE..(i + 1) * ACTION_SPACE_SIZE],
                    l.legal,
                    l.actor,
                ),
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `~/.cargo/bin/cargo test --release -p settler-net`
Expected: all pass; `tests/policy.rs` 4 passed, `tests/graph.rs` 3 passed, `tests/data.rs` 8 passed, `tests/evaluator.rs` 9 passed.

Run: `PATH=$HOME/.cargo/bin:$PATH .venv/bin/maturin develop --release -q && .venv/bin/pytest -q tests/python/test_training_data.py`
Expected: 17 passed.

- [ ] **Step 7: Format, lint, full checks**

Run: `~/.cargo/bin/cargo fmt --all && ~/.cargo/bin/cargo fmt --check`
Expected: no output.
Run: `~/.cargo/bin/cargo clippy --workspace --release --all-targets -- -D warnings`
Expected: exits 0.
Run: `~/.cargo/bin/cargo test --workspace --release`
Expected: all pass, `golden_trace` included.
Run: `.venv/bin/pytest -q -rs`
Expected: all pass; the summary lists no skipped tests.

- [ ] **Step 8: Commit**

```bash
git add net/src/policy.rs net/src/graph.rs net/src/lib.rs net/src/data.rs net/src/evaluator.rs net/tests/policy.rs net/tests/graph.rs net/tests/data.rs net/tests/evaluator.rs tests/python/test_training_data.py
git diff --cached | grep -nE 'sk-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{36}|password=[^[:space:]]+'
git -c user.email=sean.may101@gmail.com commit -m "net: seat-relative policy ids for steals and trade confirmations, and the board graph"
```

---

### Task B: Network v2 in Rust and PyTorch (weights format 2, the graph forward pass, the twin, `--width` / `--rounds`)

**Files:**
- Rewrite: `net/src/weights.rs`, `net/src/forward.rs`, `alphasettler/train/model.py`, `net/tests/weights.rs`
- Modify: `bindings/src/train.rs`, `alphasettler/train/train.py`, `alphasettler/train/loop.py`, `alphasettler/cli.py`
- Test: `net/tests/weights.rs`, `bots/tests/netbot.rs`, `tests/python/test_train.py`, `tests/python/test_train_loop.py`, `tests/python/test_training_data.py`, `tests/python/test_netbot.py`, `tests/python/test_oracle_arena.py`
- Unchanged, and network v2 from here on: `net/src/evaluator.rs` (`NetEvaluator` calls `Forward::run`, whose signature stays), `bots/examples/net_speed.rs` (`random` is `Weights::random(Arch::DEFAULT, 1)`), `net/tests/evaluator.rs` (its prior-invariance property test now runs on a random v2 net)

**Interfaces:**
- Consumes (Task A): `settler_net::policy::{other_slot, SETTLEMENT_BASE, CITY_BASE, ROAD_BASE, ROBBER_BASE, OTHER_POLICY_LEN}`, `settler_net::graph::{graph, Links, GLOBAL_BASE, GLOBAL_WIDTH}`.
- Produces:
  - `settler_net::weights`: `MAGIC`, `FORMAT_VERSION: u32` (2), `FIXED_HEADER_LEN: usize` (32), `MAX_WIDTH: usize` (1024), `MAX_ROUNDS: usize` (16); layer indices `NODE_ENCODER` 0, `EDGE_ENCODER` 1, `TILE_ENCODER` 2, `GLOBAL_ENCODER` 3, `ROUND_BASE` 4, `LAYERS_PER_ROUND` 4, `ROUND_TILE` 0, `ROUND_NODE` 1, `ROUND_EDGE` 2, `ROUND_GLOBAL` 3, `HEAD_LAYERS` 8, `NODE_POLICY` 0, `EDGE_POLICY` 1, `TILE_POLICY` 2, `OTHER_HIDDEN` 3, `OTHER_POLICY` 4, `VALUE_HIDDEN` 5, `VALUE` 6, `AUX` 7; `pub struct Arch { pub width: usize, pub rounds: usize }` with `Arch::DEFAULT` (48, 2), `check(&self) -> Result<(), String>`, `layer_count(&self) -> usize`, `header_len(&self) -> usize`, `round_layer(&self, r: usize, which: usize) -> usize`, `head_layer(&self, which: usize) -> usize`, `shapes(&self) -> Vec<(usize, usize)>`; `Layer` and `Weights { arch, layers }` with `random(arch, seed)`, `param_count()`, `to_bytes()`, `from_bytes(&[u8]) -> Result<Weights, String>`, `save(&Path)`, `load(&Path)` as before.
  - `settler_net::forward`: the same API as before (`Outputs`, `Forward::new`, `Forward::run(&mut self, w: &Weights, public: &[f32], hidden: &[f32], n: usize, out: &mut Outputs)`, `dot`, `softmax`).
  - `alphasettler._engine`: `write_random_net(path, seed, width=None, rounds=None)`; `net_adjacency() -> dict` with keys `node_tiles`, `node_edges`, `node_neighbours`, `edge_nodes`, `tile_nodes`, each a list of index lists; constants `NET_FORMAT_VERSION` (2), `NET_MAX_WIDTH` (1024), `NET_MAX_ROUNDS` (16), `NET_DEFAULT_WIDTH` (48), `NET_DEFAULT_ROUNDS` (2), `NET_NODE_WIDTH` (19), `NET_EDGE_WIDTH` (4), `NET_TILE_WIDTH` (8), `NET_GLOBAL_WIDTH` (91), `NET_SETTLEMENT_BASE` (25).
  - `alphasettler.train.model`: `Net(width=48, rounds=2)` with attributes `width`, `rounds`, `layers()` in file order and `forward(public, hidden) -> (policy, value, aux)`; `check_arch(width, rounds)`, `layer_count(rounds)`, `shapes(width, rounds)`, `to_bytes`, `from_bytes`, `save`, `load`, `FIXED_HEADER_LEN`, `ACTIONS`, `OTHER`.
  - `TrainConfig(width=48, rounds=2, ...)` (replacing `trunk`, `value`), whose `check()` bounds both; `train()`'s summary gains `"width"` and `"rounds"` (the trained net's, an `init` file's own); `LoopSettings(width=48, rounds=2, ...)`; `MANIFEST_FORMAT` 2, each train entry recording `width` and `rounds`; the CLI flags `--width` and `--rounds` on `train` and `train-loop` (no `--trunk`, no `--value-width`).

- [ ] **Step 1: Write the failing Rust tests**

Replace `net/tests/weights.rs` with:

```rust
mod common;

use common::*;
use settler_engine::topology::{topo, NUM_EDGES, NUM_NODES, NUM_TILES};
use settler_engine::{GameConfig, ACTION_SPACE_SIZE, NUM_PLAYERS};
use settler_net::encode::*;
use settler_net::forward::{softmax, Forward, Outputs};
use settler_net::weights::*;

const SMALL: Arch = Arch {
    width: 4,
    rounds: 1,
};

fn err(bytes: &[u8]) -> String {
    Weights::from_bytes(bytes).unwrap_err()
}

#[test]
fn the_header_and_the_layer_shapes_follow_the_arch() {
    assert_eq!(FIXED_HEADER_LEN, 32);
    assert_eq!(MAGIC, *b"ASNET\0\0\0");
    assert_eq!(FORMAT_VERSION, 2);
    assert_eq!(
        Arch::DEFAULT,
        Arch {
            width: 48,
            rounds: 2
        }
    );
    let a = Arch {
        width: 3,
        rounds: 2,
    };
    assert_eq!(a.layer_count(), 20);
    assert_eq!(a.header_len(), 32 + 8 * 20);
    #[rustfmt::skip]
    let want = vec![
        (3, 19), (3, 4), (3, 8), (3, 91),
        (3, 9), (3, 15), (3, 9), (3, 12),
        (3, 9), (3, 15), (3, 9), (3, 12),
        (2, 6), (1, 6), (1, 6), (12, 12), (466, 12), (12, 42), (4, 12), (4, 12),
    ];
    assert_eq!(a.shapes(), want);
    assert_eq!(
        (
            a.round_layer(0, ROUND_TILE),
            a.round_layer(1, ROUND_EDGE),
            a.head_layer(NODE_POLICY),
            a.head_layer(VALUE)
        ),
        (4, 10, 12, 18)
    );
    let w = Weights::random(Arch::DEFAULT, 1);
    assert_eq!(w.param_count(), 247_294);
    assert_eq!(w.to_bytes().len(), Arch::DEFAULT.header_len() + 4 * 247_294);
    let (n, public, hidden) = inputs(301);
    let mut out = Outputs::default();
    Forward::new().run(&w, &public, &hidden, n, &mut out);
    assert_eq!(
        (out.n, out.policy.len(), out.value.len(), out.aux.len()),
        (n, n * ACTION_SPACE_SIZE, n * NUM_PLAYERS, n * NUM_PLAYERS)
    );
    assert!(out.policy.iter().chain(&out.value).all(|x| x.is_finite()));
}

#[test]
fn widths_and_round_counts_are_bounded() {
    let ok = |width, rounds| Arch { width, rounds }.check();
    assert!(ok(1, 1).is_ok() && ok(MAX_WIDTH, MAX_ROUNDS).is_ok());
    assert_eq!(
        ok(0, 1).unwrap_err(),
        format!("width 0 is out of range 1..={MAX_WIDTH}")
    );
    assert!(ok(MAX_WIDTH + 1, 1).unwrap_err().contains("width"));
    assert_eq!(
        ok(4, 0).unwrap_err(),
        format!("rounds 0 is out of range 1..={MAX_ROUNDS}")
    );
    assert!(ok(4, MAX_ROUNDS + 1).unwrap_err().contains("rounds"));
    let deepest = Weights::random(
        Arch {
            width: 1,
            rounds: MAX_ROUNDS,
        },
        2,
    );
    assert_eq!(Weights::from_bytes(&deepest.to_bytes()).unwrap(), deepest);
}

#[test]
fn weights_round_trip_through_bytes_and_files() {
    let w = Weights::random(SMALL, 7);
    assert_eq!(Weights::from_bytes(&w.to_bytes()).unwrap(), w);
    let path = temp_file("round-trip.bin");
    w.save(&path).unwrap();
    assert_eq!(Weights::load(&path).unwrap(), w);
    std::fs::remove_file(&path).unwrap();
    assert_ne!(Weights::random(SMALL, 8), w);
}

#[test]
fn a_bad_file_is_an_error_naming_what_is_wrong() {
    let good = Weights::random(SMALL, 3).to_bytes();
    let patch = |at: usize, v: u32| {
        let mut b = good.clone();
        b[at..at + 4].copy_from_slice(&v.to_le_bytes());
        b
    };
    let mut magic = good.clone();
    magic[0] = b'X';
    assert!(err(&magic).contains("magic"), "{}", err(&magic));
    let v1 = err(&patch(8, 1));
    assert!(
        v1.contains("format version 1, expected 2") && v1.contains("flat MLP"),
        "{v1}"
    );
    assert_eq!(err(&patch(8, 3)), "format version 3, expected 2");
    let mut hash = good.clone();
    hash[12] ^= 1;
    let e = err(&hash);
    assert!(e.contains("encoding hash"), "{e}");
    assert!(
        e.contains(&format!("{ENCODING_HASH:#018x}")),
        "names this encoder's hash: {e}"
    );
    assert!(
        e.contains(&format!("{:#018x}", ENCODING_HASH ^ 1)),
        "names the file's hash: {e}"
    );
    for (at, v, what) in [
        (20, 0, "width 0 is out of range"),
        (20, MAX_WIDTH as u32 + 1, "width 1025 is out of range"),
        (20, u32::MAX, "width 4294967295 is out of range"),
        (24, 0, "rounds 0 is out of range"),
        (24, MAX_ROUNDS as u32 + 1, "rounds 17 is out of range"),
    ] {
        assert!(err(&patch(at, v)).contains(what), "{}", err(&patch(at, v)));
    }
    // A declared width at the bound passes the bound and fails on the shapes it implies.
    assert!(err(&patch(20, MAX_WIDTH as u32)).contains("do not fit network v2"));
    assert!(err(&patch(24, 2)).contains("16 layers, expected 20 for 2 rounds"));
    assert!(err(&patch(28, 15)).contains("15 layers, expected 16"));
    let shape = patch(FIXED_HEADER_LEN + 8, 7); // the edge encoder's rows: 4 -> 7
    assert!(err(&shape).contains("shapes"), "{}", err(&shape));
    assert!(err(&good[..good.len() - 1]).contains("truncated"));
    assert!(err(&good[..40]).contains("shorter than the 160-byte header"));
    assert!(err(&good[..20]).contains("shorter than the 32-byte header"));
    assert!(err(&[]).contains("truncated"));
    let mut long = good.clone();
    long.push(0);
    assert!(err(&long).contains("trailing"), "{}", err(&long));
    let first = SMALL.header_len();
    let mut nan = good.clone();
    nan[first..first + 4].copy_from_slice(&f32::NAN.to_le_bytes());
    assert!(err(&nan).contains("NaN"), "{}", err(&nan));
    let mut inf = good.clone();
    let last = inf.len() - 4; // the aux head's last bias
    inf[last..].copy_from_slice(&f32::INFINITY.to_le_bytes());
    assert!(err(&inf).contains("infinite"), "{}", err(&inf));
    let missing = Weights::load(&temp_file("no-such.bin")).unwrap_err();
    assert!(
        missing.contains("cannot read") && missing.contains("no-such.bin"),
        "{missing}"
    );
}

/// A layer whose weights are all zero but for `(row, col, value)` entries, with these biases.
fn sparse(rows: usize, cols: usize, entries: &[(usize, usize, f32)], b: &[f32]) -> Layer {
    let mut w = vec![0.0; rows * cols];
    for &(r, c, v) in entries {
        w[r * cols + c] = v;
    }
    assert_eq!(b.len(), rows);
    Layer {
        rows,
        cols,
        w,
        b: b.to_vec(),
    }
}

/// d = 2, R = 1, every encoder weight zero: each entity type starts from its bias, all entities of
/// a type equal, so every mean is that value and an edge's sum is twice its nodes'. Each layer
/// picks out a few input columns, so the outputs below follow by hand from the column order.
#[test]
fn a_hand_computed_tiny_network() {
    let d = 2;
    let id: Vec<(usize, usize, f32)> = (0..4 * d).map(|k| (k, k, 1.0)).collect();
    let other_b: Vec<f32> = (0..466).map(|j| j as f32 / 1000.0).collect();
    let other: Vec<(usize, usize, f32)> = (0..466).map(|j| (j, j % 8, 1.0)).collect();
    let layers = vec![
        sparse(d, 19, &[], &[1.0, 2.0]), // every node (1, 2)
        sparse(d, 4, &[], &[3.0, 4.0]),  // every edge (3, 4)
        sparse(d, 8, &[], &[5.0, 6.0]),  // every tile (5, 6)
        sparse(d, 91, &[], &[7.0, 8.0]), // global (7, 8)
        // tile + ReLU([mean node 0, global 1 - 10]) = (5, 6) + (1, 0) = (6, 6)
        sparse(d, 3 * d, &[(0, 2, 1.0), (1, 5, 1.0)], &[0.0, -10.0]),
        // node + ReLU([mean tile 0 + 0.5, mean edge 1 + 10 x mean neighbour 1]) = (1, 2) + (6.5, 24)
        sparse(
            d,
            5 * d,
            &[(0, 2, 1.0), (1, 5, 1.0), (1, 7, 10.0)],
            &[0.5, 0.0],
        ),
        // edge + ReLU([node sum 0, edge 0]) = (3, 4) + (15, 3) = (18, 7)
        sparse(d, 3 * d, &[(0, 2, 1.0), (1, 0, 1.0)], &[0.0, 0.0]),
        // global + ReLU([mean node 0 + mean edge 1, mean tile 0]) = (7, 8) + (14.5, 6) = (21.5, 14)
        sparse(
            d,
            4 * d,
            &[(0, 2, 1.0), (0, 5, 1.0), (1, 6, 1.0)],
            &[0.0, 0.0],
        ),
        // settlement = node 0, city = global 1 + 1
        sparse(2, 2 * d, &[(0, 0, 1.0), (1, 3, 1.0)], &[0.0, 1.0]),
        // road = edge 1 + global 0
        sparse(1, 2 * d, &[(0, 1, 1.0), (0, 2, 1.0)], &[0.0]),
        // robber = 0.25 - tile 0
        sparse(1, 2 * d, &[(0, 0, -1.0)], &[0.25]),
        sparse(4 * d, 4 * d, &id, &[0.0; 8]),
        // other slot j = pooled[j % 8] + j / 1000
        sparse(466, 4 * d, &other, &other_b),
        // value hidden = (pooled 0, hidden 0, hidden 29, 0, ...)
        sparse(
            4 * d,
            4 * d + HIDDEN_LEN,
            &[(0, 0, 1.0), (1, 8, 1.0), (2, 37, 1.0)],
            &[0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0],
        ),
        sparse(
            4,
            4 * d,
            &[(0, 0, 1.0), (1, 1, 1.0), (2, 2, 1.0)],
            &[0.0, 0.0, 0.0, 1.0],
        ),
        sparse(
            4,
            4 * d,
            &[(0, 0, 1.0), (1, 0, 2.0), (2, 0, 3.0), (3, 0, 4.0)],
            &[0.0; 4],
        ),
    ];
    let w = Weights {
        arch: Arch {
            width: d,
            rounds: 1,
        },
        layers,
    };
    assert_eq!(
        w.layers
            .iter()
            .map(|l| (l.rows, l.cols))
            .collect::<Vec<_>>(),
        w.arch.shapes()
    );
    let w = Weights::from_bytes(&w.to_bytes()).unwrap();
    let mut hidden = vec![0.0; HIDDEN_LEN];
    hidden[0] = 0.5;
    hidden[29] = 0.25;
    let mut out = Outputs::default();
    Forward::new().run(&w, &vec![0.0; PUBLIC_LEN], &hidden, 1, &mut out);
    let pooled = [21.5f32, 14.0, 7.5, 26.0, 18.0, 7.0, 6.0, 6.0];
    let close = |got: f32, want: f32, what: String| {
        assert!((got - want).abs() <= 1e-4, "{what}: {got} vs {want}");
    };
    for n in 0..NUM_NODES {
        close(out.policy[25 + n], 7.5, format!("settlement {n}"));
        close(out.policy[79 + n], 15.0, format!("city {n}"));
    }
    for e in 0..NUM_EDGES {
        close(out.policy[133 + e], 28.5, format!("road {e}"));
    }
    for t in 0..NUM_TILES {
        close(out.policy[205 + t], -5.75, format!("robber {t}"));
    }
    for j in 0..466 {
        let id = if j < 25 { j } else { j + 199 };
        close(
            out.policy[id],
            pooled[j % 8] + j as f32 / 1000.0,
            format!("slot {id}"),
        );
    }
    for (s, want) in [21.5, 0.5, 0.25, 1.0].into_iter().enumerate() {
        close(out.value[s], want, format!("value {s}"));
    }
    for (s, want) in [21.5, 43.0, 64.5, 86.0].into_iter().enumerate() {
        close(out.aux[s], want, format!("aux {s}"));
    }
}

/// Network v2 written the slow, obvious way, in f64, from the engine's topology (not
/// `settler_net::graph`): one position. Slot numbers are spelled out, not taken from `policy`.
fn reference(w: &Weights, public: &[f32], hidden: &[f32]) -> [Vec<f64>; 3] {
    let t = topo();
    let d = w.arch.width;
    let lin = |l: &Layer, x: &[f64]| -> Vec<f64> {
        (0..l.rows)
            .map(|o| {
                l.b[o] as f64
                    + (0..l.cols)
                        .map(|k| l.w[o * l.cols + k] as f64 * x[k])
                        .sum::<f64>()
            })
            .collect()
    };
    let relu = |v: Vec<f64>| -> Vec<f64> { v.into_iter().map(|x| x.max(0.0)).collect() };
    let plus =
        |a: &[f64], b: Vec<f64>| -> Vec<f64> { a.iter().zip(b).map(|(x, y)| x + y).collect() };
    let mean = |rows: &[Vec<f64>], idx: &[usize]| -> Vec<f64> {
        (0..d)
            .map(|c| idx.iter().map(|&i| rows[i][c]).sum::<f64>() / idx.len() as f64)
            .collect()
    };
    let ids = |v: &[u8]| -> Vec<usize> { v.iter().map(|&x| x as usize).collect() };
    let x: Vec<f64> = public.iter().map(|&v| v as f64).collect();
    let item = |base: usize, width: usize, i: usize| &x[base + i * width..base + (i + 1) * width];
    let mut node: Vec<Vec<f64>> = (0..NUM_NODES)
        .map(|n| relu(lin(&w.layers[NODE_ENCODER], item(NODE_BASE, NODE_WIDTH, n))))
        .collect();
    let mut edge: Vec<Vec<f64>> = (0..NUM_EDGES)
        .map(|e| relu(lin(&w.layers[EDGE_ENCODER], item(EDGE_BASE, EDGE_WIDTH, e))))
        .collect();
    let mut tile: Vec<Vec<f64>> = (0..NUM_TILES)
        .map(|i| relu(lin(&w.layers[TILE_ENCODER], item(TILE_BASE, TILE_WIDTH, i))))
        .collect();
    let mut global = relu(lin(&w.layers[GLOBAL_ENCODER], &x[OWN_HAND..]));
    let every = |k: usize| (0..k).collect::<Vec<usize>>();
    for r in 0..w.arch.rounds {
        let l = |which: usize| &w.layers[w.arch.round_layer(r, which)];
        tile = (0..NUM_TILES)
            .map(|i| {
                let input = [
                    tile[i].clone(),
                    mean(&node, &ids(&t.tile_nodes[i])),
                    global.clone(),
                ]
                .concat();
                plus(&tile[i], relu(lin(l(ROUND_TILE), &input)))
            })
            .collect();
        node = (0..NUM_NODES)
            .map(|n| {
                let input = [
                    node[n].clone(),
                    mean(&tile, &ids(&t.node_tiles[n])),
                    mean(&edge, &ids(&t.node_edges[n])),
                    mean(&node, &ids(&t.node_neighbors[n])),
                    global.clone(),
                ]
                .concat();
                plus(&node[n], relu(lin(l(ROUND_NODE), &input)))
            })
            .collect();
        edge = (0..NUM_EDGES)
            .map(|e| {
                let (p, q) = t.edge_nodes[e];
                let sum: Vec<f64> = (0..d)
                    .map(|c| node[p as usize][c] + node[q as usize][c])
                    .collect();
                let input = [edge[e].clone(), sum, global.clone()].concat();
                plus(&edge[e], relu(lin(l(ROUND_EDGE), &input)))
            })
            .collect();
        let input = [
            global.clone(),
            mean(&node, &every(NUM_NODES)),
            mean(&edge, &every(NUM_EDGES)),
            mean(&tile, &every(NUM_TILES)),
        ]
        .concat();
        global = plus(&global, relu(lin(l(ROUND_GLOBAL), &input)));
    }
    let head = |which: usize| &w.layers[w.arch.head_layer(which)];
    let pooled = [
        global.clone(),
        mean(&node, &every(NUM_NODES)),
        mean(&edge, &every(NUM_EDGES)),
        mean(&tile, &every(NUM_TILES)),
    ]
    .concat();
    let mut policy = vec![0.0; ACTION_SPACE_SIZE];
    for n in 0..NUM_NODES {
        let y = lin(
            head(NODE_POLICY),
            &[node[n].clone(), global.clone()].concat(),
        );
        policy[25 + n] = y[0];
        policy[79 + n] = y[1];
    }
    for e in 0..NUM_EDGES {
        policy[133 + e] = lin(
            head(EDGE_POLICY),
            &[edge[e].clone(), global.clone()].concat(),
        )[0];
    }
    for i in 0..NUM_TILES {
        policy[205 + i] = lin(
            head(TILE_POLICY),
            &[tile[i].clone(), global.clone()].concat(),
        )[0];
    }
    let other = lin(head(OTHER_POLICY), &relu(lin(head(OTHER_HIDDEN), &pooled)));
    for (j, v) in other.into_iter().enumerate() {
        policy[if j < 25 { j } else { j + 199 }] = v;
    }
    let mut input = pooled;
    input.extend(hidden.iter().map(|&v| v as f64));
    let v = relu(lin(head(VALUE_HIDDEN), &input));
    [policy, lin(head(VALUE), &v), lin(head(AUX), &v)]
}

fn inputs(n_every: usize) -> (usize, Vec<f32>, Vec<f32>) {
    let positions = random_positions(0..2, n_every, GameConfig::default());
    let n = positions.len();
    let (mut public, mut hidden) = (vec![0.0; n * PUBLIC_LEN], vec![0.0; n * HIDDEN_LEN]);
    for (i, s) in positions.iter().enumerate() {
        encode_state(
            s,
            s.current_actor(),
            &mut public[i * PUBLIC_LEN..(i + 1) * PUBLIC_LEN],
            &mut hidden[i * HIDDEN_LEN..(i + 1) * HIDDEN_LEN],
        );
    }
    (n, public, hidden)
}

/// d = 5 (not a multiple of the 8-lane dot product) and 2 rounds, on real positions.
#[test]
fn the_forward_pass_matches_a_plain_reference() {
    let w = Weights::random(
        Arch {
            width: 5,
            rounds: 2,
        },
        5,
    );
    let (n, public, hidden) = inputs(97);
    assert!(n > 20, "{n} positions");
    let mut out = Outputs::default();
    Forward::new().run(&w, &public, &hidden, n, &mut out);
    for i in 0..n {
        let [p, v, a] = reference(
            &w,
            &public[i * PUBLIC_LEN..(i + 1) * PUBLIC_LEN],
            &hidden[i * HIDDEN_LEN..(i + 1) * HIDDEN_LEN],
        );
        let got = [
            &out.policy[i * ACTION_SPACE_SIZE..(i + 1) * ACTION_SPACE_SIZE],
            &out.value[i * NUM_PLAYERS..(i + 1) * NUM_PLAYERS],
            &out.aux[i * NUM_PLAYERS..(i + 1) * NUM_PLAYERS],
        ];
        for (g, want) in got.iter().zip([p, v, a]) {
            for (x, y) in g.iter().zip(&want) {
                assert!(
                    (*x as f64 - y).abs() <= 1e-4 * y.abs().max(1.0),
                    "position {i}: {x} vs {y}"
                );
            }
        }
    }
}

#[test]
fn each_row_is_independent_of_the_rest_of_the_batch() {
    let w = Weights::random(Arch::DEFAULT, 9);
    let (n, public, hidden) = inputs(211);
    let (mut fwd, mut all, mut one) = (Forward::new(), Outputs::default(), Outputs::default());
    fwd.run(&w, &public, &hidden, n, &mut all);
    for i in 0..n {
        fwd.run(
            &w,
            &public[i * PUBLIC_LEN..(i + 1) * PUBLIC_LEN],
            &hidden[i * HIDDEN_LEN..(i + 1) * HIDDEN_LEN],
            1,
            &mut one,
        );
        assert_eq!(
            one.policy[..],
            all.policy[i * ACTION_SPACE_SIZE..(i + 1) * ACTION_SPACE_SIZE]
        );
        assert_eq!(
            one.value[..],
            all.value[i * NUM_PLAYERS..(i + 1) * NUM_PLAYERS]
        );
        assert_eq!(one.aux[..], all.aux[i * NUM_PLAYERS..(i + 1) * NUM_PLAYERS]);
    }
}

#[test]
fn softmax_sums_to_one_and_sends_minus_infinity_to_zero() {
    let mut x = [1.0f32, f32::NEG_INFINITY, 3.0, -2.0, f32::NEG_INFINITY];
    softmax(&mut x);
    assert!((x.iter().sum::<f32>() - 1.0).abs() < 1e-6, "{x:?}");
    assert_eq!((x[1], x[4]), (0.0, 0.0));
    let z = 1f32.exp() + 3f32.exp() + (-2f32).exp();
    assert!((x[2] - 3f32.exp() / z).abs() < 1e-6, "{x:?}");
    let mut big = [1000.0f32, 999.0];
    softmax(&mut big);
    assert!(
        big.iter().all(|v| v.is_finite()),
        "stable for large logits: {big:?}"
    );
    assert!((big[0] - 1.0 / (1.0 + (-1f32).exp())).abs() < 1e-6);
}
```

In `bots/tests/netbot.rs`, both `Weights::random(Arch { trunk: 16, value: 8 }, ...)` calls become `Arch { width: 8, rounds: 1 }`:

```rust
        Arch {
            width: 8,
            rounds: 1,
        },
```

and the doc comment of `nets_dir` reads `/// A fresh nets directory holding `tiny.bin` (random weights, width 8, 1 round).`

- [ ] **Step 2: Run the Rust tests to verify they fail**

Run: `~/.cargo/bin/cargo test --release -p settler-net --test weights`
Expected: FAIL to compile: `struct Arch has no field named width`, `cannot find value FIXED_HEADER_LEN`, `cannot find value NODE_ENCODER`.

- [ ] **Step 3: The weights file, format 2**

Replace `net/src/weights.rs` with:

```rust
//! The weights file, format version 2: network v2, the entity-structured graph net (spec
//! Section 2). Little-endian throughout. A file declares its width `d` (1..=`MAX_WIDTH`) and its
//! round count `R` (1..=`MAX_ROUNDS`); every other dimension is fixed by the encoder and the
//! action space. A version 1 file (the flat MLP) is refused.
//!
//! | bytes | field |
//! |---|---|
//! | 0..8 | magic `ASNET\0\0\0` |
//! | 8..12 | format version, u32 = 2 |
//! | 12..20 | `ENCODING_HASH`, u64 |
//! | 20..24 | width `d`, u32 |
//! | 24..28 | rounds `R`, u32 |
//! | 28..32 | layer count `L`, u32 = 12 + 4R |
//! | 32..32 + 8L | per layer: rows, then cols (u32 each); rows are outputs |
//! | then | per layer, in order: rows × cols weights (row-major), then rows biases (f32) |
//!
//! Layers, in file order (the PyTorch twin's `Net.layers()` returns its modules in this order):
//!
//! | index | layer | rows × cols | input, in column order |
//! |---|---|---|---|
//! | 0 | node encoder | d × 19 | a node's 19 features |
//! | 1 | edge encoder | d × 4 | an edge's 4 features |
//! | 2 | tile encoder | d × 8 | a tile's 8 features |
//! | 3 | global encoder | d × 91 | the 91 global scalars (`graph::GLOBAL_BASE..PUBLIC_LEN`) |
//! | 4 + 4r | round r, tile | d × 3d | [tile, mean of its 6 nodes, global] |
//! | 5 + 4r | round r, node | d × 5d | [node, mean of its tiles, mean of its edges, mean of its neighbouring nodes, global] |
//! | 6 + 4r | round r, edge | d × 3d | [edge, sum of its 2 nodes, global] |
//! | 7 + 4r | round r, global | d × 4d | [global, mean of the nodes, mean of the edges, mean of the tiles] |
//! | P = 4 + 4R | node policy | 2 × 2d | [node, global] -> BuildSettlement, BuildCity logits |
//! | P + 1 | edge policy | 1 × 2d | [edge, global] -> BuildRoad logit |
//! | P + 2 | tile policy | 1 × 2d | [tile, global] -> MoveRobber logit |
//! | P + 3 | other hidden | 4d × 4d | pooled = [global, mean of the nodes, mean of the edges, mean of the tiles] |
//! | P + 4 | other policy | 466 × 4d | the other hidden layer -> every other slot's logit, in slot order |
//! | P + 5 | value hidden | 4d × (4d + 30) | [pooled, hidden block] |
//! | P + 6 | value | 4 × 4d | the value hidden layer -> seat-relative win logits |
//! | P + 7 | aux | 4 × 4d | the value hidden layer -> seat-relative final VP / 10 |
//!
//! The encoders, the other hidden layer and the value hidden layer end in a ReLU. A round layer
//! updates its entities as x ← x + ReLU(W·input + b); a round runs its four layers in the order
//! listed, each reading the values the earlier ones just wrote (`forward.rs`).

use crate::encode::{EDGE_WIDTH, ENCODING_HASH, HIDDEN_LEN, NODE_WIDTH, TILE_WIDTH};
use crate::graph::GLOBAL_WIDTH;
use crate::policy::OTHER_POLICY_LEN;
use settler_engine::rng::Rng;
use settler_engine::NUM_PLAYERS;
use std::path::Path;

pub const MAGIC: [u8; 8] = *b"ASNET\0\0\0";
pub const FORMAT_VERSION: u32 = 2;
/// Magic, version, hash, width, rounds and layer count: the bytes before the layer shapes.
pub const FIXED_HEADER_LEN: usize = 8 + 4 + 8 + 4 + 4 + 4;
/// Widest `d` a file may declare: a guard against a corrupt header.
pub const MAX_WIDTH: usize = 1024;
/// Most rounds a file may declare.
pub const MAX_ROUNDS: usize = 16;

pub const NODE_ENCODER: usize = 0;
pub const EDGE_ENCODER: usize = 1;
pub const TILE_ENCODER: usize = 2;
pub const GLOBAL_ENCODER: usize = 3;
/// Round r's layers start at `ROUND_BASE + LAYERS_PER_ROUND * r` (`Arch::round_layer`).
pub const ROUND_BASE: usize = 4;
pub const LAYERS_PER_ROUND: usize = 4;
pub const ROUND_TILE: usize = 0;
pub const ROUND_NODE: usize = 1;
pub const ROUND_EDGE: usize = 2;
pub const ROUND_GLOBAL: usize = 3;
/// The heads follow the last round (`Arch::head_layer`).
pub const HEAD_LAYERS: usize = 8;
pub const NODE_POLICY: usize = 0;
pub const EDGE_POLICY: usize = 1;
pub const TILE_POLICY: usize = 2;
pub const OTHER_HIDDEN: usize = 3;
pub const OTHER_POLICY: usize = 4;
pub const VALUE_HIDDEN: usize = 5;
pub const VALUE: usize = 6;
pub const AUX: usize = 7;

/// The two free dimensions: the width `d` and the round count `R`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Arch {
    pub width: usize,
    pub rounds: usize,
}

impl Arch {
    /// Spec Section 2's defaults: about 2.8M multiply-adds per position.
    pub const DEFAULT: Arch = Arch {
        width: 48,
        rounds: 2,
    };

    /// An error naming the dimension out of range, if either is.
    pub fn check(&self) -> Result<(), String> {
        if !(1..=MAX_WIDTH).contains(&self.width) {
            return Err(format!(
                "width {} is out of range 1..={MAX_WIDTH}",
                self.width
            ));
        }
        if !(1..=MAX_ROUNDS).contains(&self.rounds) {
            return Err(format!(
                "rounds {} is out of range 1..={MAX_ROUNDS}",
                self.rounds
            ));
        }
        Ok(())
    }

    pub fn layer_count(&self) -> usize {
        ROUND_BASE + LAYERS_PER_ROUND * self.rounds + HEAD_LAYERS
    }

    pub fn header_len(&self) -> usize {
        FIXED_HEADER_LEN + 8 * self.layer_count()
    }

    /// The index of round `r`'s layer `which` (`ROUND_TILE` ..= `ROUND_GLOBAL`).
    pub fn round_layer(&self, r: usize, which: usize) -> usize {
        ROUND_BASE + LAYERS_PER_ROUND * r + which
    }

    /// The index of head layer `which` (`NODE_POLICY` ..= `AUX`).
    pub fn head_layer(&self, which: usize) -> usize {
        ROUND_BASE + LAYERS_PER_ROUND * self.rounds + which
    }

    /// (rows, cols) of each layer in file order; rows are outputs.
    pub fn shapes(&self) -> Vec<(usize, usize)> {
        let d = self.width;
        let mut s = vec![
            (d, NODE_WIDTH),
            (d, EDGE_WIDTH),
            (d, TILE_WIDTH),
            (d, GLOBAL_WIDTH),
        ];
        for _ in 0..self.rounds {
            s.extend([(d, 3 * d), (d, 5 * d), (d, 3 * d), (d, 4 * d)]);
        }
        s.extend([
            (2, 2 * d),
            (1, 2 * d),
            (1, 2 * d),
            (4 * d, 4 * d),
            (OTHER_POLICY_LEN, 4 * d),
            (4 * d, 4 * d + HIDDEN_LEN),
            (NUM_PLAYERS, 4 * d),
            (NUM_PLAYERS, 4 * d),
        ]);
        s
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Layer {
    pub rows: usize,
    pub cols: usize,
    /// `rows × cols`, row-major.
    pub w: Vec<f32>,
    pub b: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Weights {
    pub arch: Arch,
    pub layers: Vec<Layer>,
}

fn uniform(rng: &mut Rng, bound: f32) -> f32 {
    ((rng.next_u64() >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0) * bound
}

impl Weights {
    /// He-uniform weights and small biases from `seed`, for tests and measurements. Panics if
    /// `arch` is out of range.
    pub fn random(arch: Arch, seed: u64) -> Weights {
        if let Err(e) = arch.check() {
            panic!("{e}");
        }
        let mut rng = Rng::new(seed);
        let layers = arch
            .shapes()
            .into_iter()
            .map(|(rows, cols)| {
                let bound = (6.0 / cols as f32).sqrt();
                let w = (0..rows * cols).map(|_| uniform(&mut rng, bound)).collect();
                let b = (0..rows).map(|_| uniform(&mut rng, 0.1)).collect();
                Layer { rows, cols, w, b }
            })
            .collect();
        Weights { arch, layers }
    }

    pub fn param_count(&self) -> usize {
        self.layers.iter().map(|l| l.w.len() + l.b.len()).sum()
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.arch.header_len() + 4 * self.param_count());
        out.extend_from_slice(&MAGIC);
        out.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
        out.extend_from_slice(&ENCODING_HASH.to_le_bytes());
        out.extend_from_slice(&(self.arch.width as u32).to_le_bytes());
        out.extend_from_slice(&(self.arch.rounds as u32).to_le_bytes());
        out.extend_from_slice(&(self.layers.len() as u32).to_le_bytes());
        for l in &self.layers {
            out.extend_from_slice(&(l.rows as u32).to_le_bytes());
            out.extend_from_slice(&(l.cols as u32).to_le_bytes());
        }
        for l in &self.layers {
            for x in l.w.iter().chain(&l.b) {
                out.extend_from_slice(&x.to_le_bytes());
            }
        }
        out
    }

    /// Every check runs before anything is sized from the file: the width and round count are
    /// bounded first, then the layer count and shapes must be the ones they imply.
    pub fn from_bytes(bytes: &[u8]) -> Result<Weights, String> {
        if bytes.len() < FIXED_HEADER_LEN {
            return Err(format!(
                "truncated: {} bytes is shorter than the {FIXED_HEADER_LEN}-byte header",
                bytes.len()
            ));
        }
        if bytes[..8] != MAGIC {
            return Err("not an AlphaSettler weights file (wrong magic)".into());
        }
        let u32_at = |i: usize| u32::from_le_bytes(bytes[i..i + 4].try_into().unwrap());
        let version = u32_at(8);
        if version != FORMAT_VERSION {
            let why = if version == 1 {
                "; version 1 holds the flat MLP that network v2 replaced: retrain the net"
            } else {
                ""
            };
            return Err(format!(
                "format version {version}, expected {FORMAT_VERSION}{why}"
            ));
        }
        let hash = u64::from_le_bytes(bytes[12..20].try_into().unwrap());
        if hash != ENCODING_HASH {
            return Err(format!(
                "encoding hash {hash:#018x} does not match this encoder's {ENCODING_HASH:#018x}: \
                 the file was trained on a different feature layout"
            ));
        }
        let arch = Arch {
            width: u32_at(20) as usize,
            rounds: u32_at(24) as usize,
        };
        arch.check()?;
        let count = u32_at(28) as usize;
        if count != arch.layer_count() {
            return Err(format!(
                "{count} layers, expected {} for {} rounds",
                arch.layer_count(),
                arch.rounds
            ));
        }
        let header = arch.header_len();
        if bytes.len() < header {
            return Err(format!(
                "truncated: {} bytes is shorter than the {header}-byte header",
                bytes.len()
            ));
        }
        let shapes: Vec<(usize, usize)> = (0..count)
            .map(|i| {
                let at = FIXED_HEADER_LEN + 8 * i;
                (u32_at(at) as usize, u32_at(at + 4) as usize)
            })
            .collect();
        if shapes != arch.shapes() {
            return Err(format!(
                "layer shapes {shapes:?} do not fit network v2 with width {} and {} rounds",
                arch.width, arch.rounds
            ));
        }
        let floats: usize = shapes.iter().map(|&(r, c)| r * c + r).sum();
        let want = header + 4 * floats;
        if bytes.len() < want {
            return Err(format!("truncated: {} bytes, expected {want}", bytes.len()));
        }
        if bytes.len() > want {
            return Err(format!(
                "{} trailing bytes after the last layer",
                bytes.len() - want
            ));
        }
        let mut at = header;
        let mut take = |n: usize| -> Vec<f32> {
            let v = bytes[at..at + 4 * n]
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect();
            at += 4 * n;
            v
        };
        let mut layers = Vec::with_capacity(count);
        for &(rows, cols) in &shapes {
            let w = take(rows * cols);
            let b = take(rows);
            layers.push(Layer { rows, cols, w, b });
        }
        if layers
            .iter()
            .any(|l| l.w.iter().chain(&l.b).any(|x| !x.is_finite()))
        {
            return Err("a weight is NaN or infinite".into());
        }
        Ok(Weights { arch, layers })
    }

    pub fn save(&self, path: &Path) -> Result<(), String> {
        std::fs::write(path, self.to_bytes())
            .map_err(|e| format!("cannot write {}: {e}", path.display()))
    }

    pub fn load(path: &Path) -> Result<Weights, String> {
        let bytes =
            std::fs::read(path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
        Weights::from_bytes(&bytes).map_err(|e| format!("{}: {e}", path.display()))
    }
}
```

- [ ] **Step 4: The graph forward pass**

Replace `net/src/forward.rs` with:

```rust
//! The batched forward pass of network v2 (f32), the only inference code (spec Section 2; the
//! layer order is in `weights.rs`). Each per-entity map runs as one matrix product over every
//! entity of its type in the batch: its rows are the batch's nodes (or edges, or tiles), its
//! columns the layer's inputs side by side. Every row's arithmetic is the same whatever else is in
//! the batch. `net/tests/weights.rs` checks it against a hand-computed tiny network and a plain
//! f64 reference, and `tests/python/test_train.py` against the PyTorch twin
//! (`alphasettler/train/model.py`).

use crate::encode::{
    EDGE_BASE, EDGE_WIDTH, HIDDEN_LEN, NODE_BASE, NODE_WIDTH, PUBLIC_LEN, TILE_BASE, TILE_WIDTH,
};
use crate::graph::{graph, Links, GLOBAL_BASE, GLOBAL_WIDTH};
use crate::policy::{
    other_slot, CITY_BASE, OTHER_POLICY_LEN, ROAD_BASE, ROBBER_BASE, SETTLEMENT_BASE,
};
use crate::weights::{
    Layer, Weights, AUX, EDGE_ENCODER, EDGE_POLICY, GLOBAL_ENCODER, NODE_ENCODER, NODE_POLICY,
    OTHER_HIDDEN, OTHER_POLICY, ROUND_EDGE, ROUND_GLOBAL, ROUND_NODE, ROUND_TILE, TILE_ENCODER,
    TILE_POLICY, VALUE, VALUE_HIDDEN,
};
use settler_engine::topology::{NUM_EDGES, NUM_NODES, NUM_TILES};
use settler_engine::{ACTION_SPACE_SIZE, NUM_PLAYERS};

/// Raw head outputs for `n` positions, row-major: 665 policy logits, 4 seat-relative value logits
/// (before the softmax) and 4 aux outputs (final VP / 10, seat-relative) per position.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Outputs {
    pub n: usize,
    pub policy: Vec<f32>,
    pub value: Vec<f32>,
    pub aux: Vec<f32>,
}

/// Activation buffers reused across calls: the entity states (`n × count × d`), the pooled
/// readout (`n × 4d`), and scratch for aggregates, concatenated inputs and layer outputs.
#[derive(Clone, Debug, Default)]
pub struct Forward {
    node: Vec<f32>,
    edge: Vec<f32>,
    tile: Vec<f32>,
    global: Vec<f32>,
    pooled: Vec<f32>,
    a: Vec<f32>,
    b: Vec<f32>,
    c: Vec<f32>,
    cat: Vec<f32>,
    y: Vec<f32>,
}

/// Eight independent partial sums so the compiler can vectorise, combined in a fixed order: a
/// row's result never depends on the rest of the batch.
#[inline]
pub fn dot(x: &[f32], y: &[f32]) -> f32 {
    debug_assert_eq!(x.len(), y.len());
    let (xc, yc) = (x.chunks_exact(8), y.chunks_exact(8));
    let (xr, yr) = (xc.remainder(), yc.remainder());
    let mut acc = [0.0f32; 8];
    for (a, b) in xc.zip(yc) {
        for j in 0..8 {
            acc[j] += a[j] * b[j];
        }
    }
    let mut tail = 0.0;
    for (a, b) in xr.iter().zip(yr) {
        tail += a * b;
    }
    ((acc[0] + acc[1]) + (acc[2] + acc[3])) + ((acc[4] + acc[5]) + (acc[6] + acc[7])) + tail
}

/// In place, numerically stable; entries of -inf become 0. Requires at least one finite entry and
/// none of +inf: otherwise the result is NaN (`evaluator::masked_prior` handles that case itself).
pub fn softmax(x: &mut [f32]) {
    let m = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    debug_assert!(m.is_finite(), "softmax needs a finite maximum, got {m}");
    let mut total = 0.0;
    for v in x.iter_mut() {
        *v = (*v - m).exp();
        total += *v;
    }
    for v in x.iter_mut() {
        *v /= total;
    }
}

/// `y[i][o] = b[o] + w[o] · x[i]` for the `n` rows of `x`, optionally through a ReLU. Row by row,
/// so a small layer's weights stay in cache across all the rows of an entity type.
fn dense(l: &Layer, x: &[f32], n: usize, relu: bool, y: &mut Vec<f32>) {
    debug_assert_eq!(x.len(), n * l.cols);
    y.clear();
    y.resize(n * l.rows, 0.0);
    for (xi, yi) in x.chunks_exact(l.cols).zip(y.chunks_exact_mut(l.rows)) {
        for (o, out) in yi.iter_mut().enumerate() {
            let s = l.b[o] + dot(&l.w[o * l.cols..(o + 1) * l.cols], xi);
            *out = if relu { s.max(0.0) } else { s };
        }
    }
}

/// x ← x + ReLU(layer(input)) for `n` rows: a round's residual update.
fn residual(l: &Layer, input: &[f32], n: usize, y: &mut Vec<f32>, x: &mut [f32]) {
    dense(l, input, n, true, y);
    for (v, d) in x.iter_mut().zip(y.iter()) {
        *v += *d;
    }
}

/// Each position's `count` items of `width` floats at `base` of its public block, as `n × count`
/// rows.
fn block(public: &[f32], n: usize, base: usize, count: usize, width: usize, out: &mut Vec<f32>) {
    out.clear();
    for i in 0..n {
        let at = i * PUBLIC_LEN + base;
        out.extend_from_slice(&public[at..at + count * width]);
    }
}

/// For each position and each entity `j` of `links`: `scale[j]` × the sum, in link order, of the
/// `d`-float rows of `src` (`src_count` rows per position) it links to. `n × links.count()` rows.
fn aggregate(links: &Links, src: &[f32], src_count: usize, n: usize, d: usize, out: &mut Vec<f32>) {
    let count = links.count();
    out.clear();
    out.resize(n * count * d, 0.0);
    for i in 0..n {
        let s = &src[i * src_count * d..(i + 1) * src_count * d];
        for j in 0..count {
            let o = &mut out[(i * count + j) * d..(i * count + j + 1) * d];
            for &k in links.of(j) {
                let k = k as usize;
                for (v, x) in o.iter_mut().zip(&s[k * d..(k + 1) * d]) {
                    *v += *x;
                }
            }
            let scale = links.scale[j];
            for v in o.iter_mut() {
                *v *= scale;
            }
        }
    }
}

/// The mean of each position's `count` rows of `src`: `n` rows of `d`.
fn mean_rows(src: &[f32], count: usize, n: usize, d: usize, out: &mut Vec<f32>) {
    out.clear();
    out.resize(n * d, 0.0);
    for (o, rows) in out.chunks_exact_mut(d).zip(src.chunks_exact(count * d)) {
        for r in rows.chunks_exact(d) {
            for (v, x) in o.iter_mut().zip(r) {
                *v += *x;
            }
        }
        for v in o.iter_mut() {
            *v /= count as f32;
        }
    }
}

/// One input block of a layer: a row per entity, or a row per position shared by its entities.
enum Part<'a> {
    Each(&'a [f32]),
    Shared(&'a [f32]),
}

/// For each position and each of its `count` entities, the parts' `d`-float rows side by side.
fn concat(parts: &[Part<'_>], n: usize, count: usize, d: usize, out: &mut Vec<f32>) {
    out.clear();
    for i in 0..n {
        for j in 0..count {
            for p in parts {
                out.extend_from_slice(match p {
                    Part::Each(x) => &x[(i * count + j) * d..(i * count + j + 1) * d],
                    Part::Shared(x) => &x[i * d..(i + 1) * d],
                });
            }
        }
    }
}

impl Forward {
    pub fn new() -> Forward {
        Forward::default()
    }

    /// Run `n` positions: `public` holds `n × PUBLIC_LEN` floats and `hidden` `n × HIDDEN_LEN`.
    pub fn run(
        &mut self,
        w: &Weights,
        public: &[f32],
        hidden: &[f32],
        n: usize,
        out: &mut Outputs,
    ) {
        use Part::{Each, Shared};
        assert_eq!(public.len(), n * PUBLIC_LEN, "public block length");
        assert_eq!(hidden.len(), n * HIDDEN_LEN, "hidden block length");
        let (arch, g) = (w.arch, graph());
        let d = arch.width;
        let layer = |i: usize| &w.layers[i];
        let Forward {
            node,
            edge,
            tile,
            global,
            pooled,
            a,
            b,
            c,
            cat,
            y,
        } = self;

        block(public, n, NODE_BASE, NUM_NODES, NODE_WIDTH, cat);
        dense(layer(NODE_ENCODER), cat, n * NUM_NODES, true, node);
        block(public, n, EDGE_BASE, NUM_EDGES, EDGE_WIDTH, cat);
        dense(layer(EDGE_ENCODER), cat, n * NUM_EDGES, true, edge);
        block(public, n, TILE_BASE, NUM_TILES, TILE_WIDTH, cat);
        dense(layer(TILE_ENCODER), cat, n * NUM_TILES, true, tile);
        block(public, n, GLOBAL_BASE, 1, GLOBAL_WIDTH, cat);
        dense(layer(GLOBAL_ENCODER), cat, n, true, global);

        for r in 0..arch.rounds {
            // tile ← [tile, mean of its nodes, global]
            aggregate(&g.tile_nodes, node, NUM_NODES, n, d, a);
            concat(&[Each(tile), Each(a), Shared(global)], n, NUM_TILES, d, cat);
            residual(
                layer(arch.round_layer(r, ROUND_TILE)),
                cat,
                n * NUM_TILES,
                y,
                tile,
            );
            // node ← [node, mean of its tiles, mean of its edges, mean of its neighbours, global]
            aggregate(&g.node_tiles, tile, NUM_TILES, n, d, a);
            aggregate(&g.node_edges, edge, NUM_EDGES, n, d, b);
            aggregate(&g.node_neighbours, node, NUM_NODES, n, d, c);
            concat(
                &[Each(node), Each(a), Each(b), Each(c), Shared(global)],
                n,
                NUM_NODES,
                d,
                cat,
            );
            residual(
                layer(arch.round_layer(r, ROUND_NODE)),
                cat,
                n * NUM_NODES,
                y,
                node,
            );
            // edge ← [edge, sum of its two nodes, global]
            aggregate(&g.edge_nodes, node, NUM_NODES, n, d, a);
            concat(&[Each(edge), Each(a), Shared(global)], n, NUM_EDGES, d, cat);
            residual(
                layer(arch.round_layer(r, ROUND_EDGE)),
                cat,
                n * NUM_EDGES,
                y,
                edge,
            );
            // global ← [global, mean of the nodes, mean of the edges, mean of the tiles]
            mean_rows(node, NUM_NODES, n, d, a);
            mean_rows(edge, NUM_EDGES, n, d, b);
            mean_rows(tile, NUM_TILES, n, d, c);
            concat(&[Each(global), Each(a), Each(b), Each(c)], n, 1, d, cat);
            residual(layer(arch.round_layer(r, ROUND_GLOBAL)), cat, n, y, global);
        }

        // pooled = [global, mean of the nodes, mean of the edges, mean of the tiles]
        mean_rows(node, NUM_NODES, n, d, a);
        mean_rows(edge, NUM_EDGES, n, d, b);
        mean_rows(tile, NUM_TILES, n, d, c);
        concat(&[Each(global), Each(a), Each(b), Each(c)], n, 1, d, pooled);

        out.policy.clear();
        out.policy.resize(n * ACTION_SPACE_SIZE, 0.0);
        concat(&[Each(node), Shared(global)], n, NUM_NODES, d, cat);
        dense(
            layer(arch.head_layer(NODE_POLICY)),
            cat,
            n * NUM_NODES,
            false,
            y,
        );
        for (p, logits) in out
            .policy
            .chunks_exact_mut(ACTION_SPACE_SIZE)
            .zip(y.chunks_exact(2 * NUM_NODES))
        {
            for (k, pair) in logits.chunks_exact(2).enumerate() {
                p[SETTLEMENT_BASE + k] = pair[0];
                p[CITY_BASE + k] = pair[1];
            }
        }
        concat(&[Each(edge), Shared(global)], n, NUM_EDGES, d, cat);
        dense(
            layer(arch.head_layer(EDGE_POLICY)),
            cat,
            n * NUM_EDGES,
            false,
            y,
        );
        for (p, logits) in out
            .policy
            .chunks_exact_mut(ACTION_SPACE_SIZE)
            .zip(y.chunks_exact(NUM_EDGES))
        {
            p[ROAD_BASE..ROAD_BASE + NUM_EDGES].copy_from_slice(logits);
        }
        concat(&[Each(tile), Shared(global)], n, NUM_TILES, d, cat);
        dense(
            layer(arch.head_layer(TILE_POLICY)),
            cat,
            n * NUM_TILES,
            false,
            y,
        );
        for (p, logits) in out
            .policy
            .chunks_exact_mut(ACTION_SPACE_SIZE)
            .zip(y.chunks_exact(NUM_TILES))
        {
            p[ROBBER_BASE..ROBBER_BASE + NUM_TILES].copy_from_slice(logits);
        }
        dense(layer(arch.head_layer(OTHER_HIDDEN)), pooled, n, true, a);
        dense(layer(arch.head_layer(OTHER_POLICY)), a, n, false, y);
        for (p, logits) in out
            .policy
            .chunks_exact_mut(ACTION_SPACE_SIZE)
            .zip(y.chunks_exact(OTHER_POLICY_LEN))
        {
            for (j, &v) in logits.iter().enumerate() {
                p[other_slot(j)] = v;
            }
        }

        // value and aux from [pooled, hidden]
        cat.clear();
        for (p, h) in pooled
            .chunks_exact(4 * d)
            .zip(hidden.chunks_exact(HIDDEN_LEN))
        {
            cat.extend_from_slice(p);
            cat.extend_from_slice(h);
        }
        dense(layer(arch.head_layer(VALUE_HIDDEN)), cat, n, true, a);
        dense(layer(arch.head_layer(VALUE)), a, n, false, &mut out.value);
        dense(layer(arch.head_layer(AUX)), a, n, false, &mut out.aux);
        debug_assert_eq!(out.value.len(), n * NUM_PLAYERS);
        out.n = n;
    }
}
```

- [ ] **Step 5: Run the Rust tests to verify they pass**

Run: `~/.cargo/bin/cargo test --release -p settler-net && ~/.cargo/bin/cargo test --release -p settler-bots --test netbot`
Expected: all pass; `tests/weights.rs` 8 passed (the hand-computed network, the f64 reference at d = 5, batch independence, the header and error tests), `tests/evaluator.rs` 9 passed (prior invariance on a random v2 net included).

- [ ] **Step 6: Speed check with a random v2 net**

Run: `mkdir -p runs/3b-v2 && ~/.cargo/bin/cargo run --release -p settler-bots --example net_speed -- speed random 300 1000 8 2>&1 | tee runs/3b-v2/net-speed-random.log`
Expected: one line `batch 8: N searched decisions over G games at 1000 simulations: S simulations/s, E evaluations/s, M ms/decision`, about 2 minutes. A random v2 net measured 3,411 simulations/s per core when this plan was written (the trained MLP: 4,628). If S is below 1,500, stop and report the line instead of going on. Task D copies this line into `docs/perf/net-search.md`.

- [ ] **Step 7: Write the failing Python tests**

`tests/python/test_training_data.py`, `tests/python/test_netbot.py` and `tests/python/test_oracle_arena.py`: every `write_random_net(..., 1, 16, 8)` and `write_random_net(..., 2, 16, 8)` becomes `write_random_net(..., 1, 8, 1)` and `write_random_net(..., 2, 8, 1)` (width 8, 1 round; six calls). In `tests/python/test_netbot.py`, replace the last two lines of `test_bad_names_missing_files_and_stale_hashes_are_errors` (the `match="trunk"` check) with:

```python
    with pytest.raises(ValueError, match="width 0 is out of range"):
        write_random_net(str(nets / "x.bin"), 1, 0, 1)
    with pytest.raises(ValueError, match="rounds 0 is out of range"):
        write_random_net(str(nets / "x.bin"), 1, 8, 0)
    mlp = bytearray((nets / "tiny.bin").read_bytes())
    mlp[8:12] = (1).to_bytes(4, "little")  # a version 1 (flat MLP) file
    (nets / "mlp.bin").write_bytes(bytes(mlp))
    with pytest.raises(ValueError, match="format version 1, expected 2"):
        Bot("net:mlp", 0, nets_dir=str(nets))
```

In `tests/python/test_train.py`:

Replace the `_engine` and `model` imports with:

```python
from alphasettler._engine import (
    NET_DEFAULT_ROUNDS, NET_DEFAULT_WIDTH, NET_ENCODING_HASH, NET_HIDDEN_LEN, NET_MAX_ROUNDS, NET_MAX_WIDTH,
    NET_PUBLIC_LEN, encode, net_forward,
)
```

```python
from alphasettler.train.model import ACTIONS, FIXED_HEADER_LEN, Net, from_bytes, shapes, to_bytes
```

Replace `test_rust_and_pytorch_forward_passes_agree_on_random_weights` with:

```python
@pytest.mark.parametrize("width,rounds", [(NET_DEFAULT_WIDTH, NET_DEFAULT_ROUNDS), (5, 3)])
def test_rust_and_pytorch_forward_passes_agree_on_random_weights(tmp_path, width, rounds):
    """Spec parity rule (preflight F9): the prior over legal moves and the value probabilities
    agree within 1e-5 absolute; raw logits and the aux head within 1e-5 x max(1, |PyTorch|). The
    default net, and an odd width (not a multiple of Rust's 8-lane dot product) with 3 rounds."""
    torch.manual_seed(0)
    net = Net(width, rounds).eval()
    path = tmp_path / "r.bin"
    path.write_bytes(to_bytes(net))
    public, hidden, mask = random_positions(100)
    rust = net_forward(str(path), public, hidden)
    pt = torch.frombuffer(bytearray(public), dtype=torch.float32).view(100, NET_PUBLIC_LEN)
    ht = torch.frombuffer(bytearray(hidden), dtype=torch.float32).view(100, NET_HIDDEN_LEN)
    with torch.no_grad():
        outs = net(pt, ht)
    got = {key: torch.frombuffer(bytearray(rust[key]), dtype=torch.float32).view_as(ref)
           for key, ref in zip(("policy", "value", "aux"), outs)}
    for key, ref in zip(("policy", "value", "aux"), outs):
        err = ((got[key] - ref).abs() / ref.abs().clamp(min=1.0)).max().item()
        assert err <= 1e-5, (key, err)

    def prior(logits):
        return torch.softmax(logits.double().masked_fill(~mask, float("-inf")), dim=1)

    assert (prior(got["policy"]) - prior(outs[0])).abs().max().item() <= 1e-5
    value = (torch.softmax(got["value"].double(), 1) - torch.softmax(outs[1].double(), 1)).abs().max().item()
    assert value <= 1e-5
```

Replace `test_weights_round_trip_and_bad_files_are_errors`, `header` and `test_widths_are_checked_and_non_finite_weights_rejected_before_building_a_net` with:

```python
def test_the_default_net_is_width_48_with_2_rounds():
    net = Net()
    assert (net.width, net.rounds) == (NET_DEFAULT_WIDTH, NET_DEFAULT_ROUNDS) == (48, 2)
    assert [(layer.out_features, layer.in_features) for layer in net.layers()] == shapes(48, 2)
    assert sum(p.numel() for p in net.parameters()) == 247_294, "the same count as Rust's Arch::DEFAULT"
    assert len(to_bytes(net)) == FIXED_HEADER_LEN + 8 * 20 + 4 * 247_294
    assert not any("tile_nodes" in k or "edge_nodes" in k for k in net.state_dict()), "the adjacency is not a weight"


def test_weights_round_trip_and_bad_files_are_errors():
    torch.manual_seed(1)
    net = Net(8, 1)
    data = to_bytes(net)
    back = from_bytes(data)
    assert (back.width, back.rounds) == (8, 1)
    assert all(torch.equal(a, b) for a, b in zip(net.state_dict().values(), back.state_dict().values()))

    def patch(at, value):
        b = bytearray(data)
        struct.pack_into("<I", b, at, value)
        return bytes(b)

    for bad, message in [(b"X" + data[1:], "magic"), (data[:12] + bytes([data[12] ^ 1]) + data[13:], "encoding hash"),
                         (patch(8, 1), "format version 1, expected 2; version 1 holds the flat MLP"),
                         (patch(8, 3), "format version 3, expected 2$"),
                         (patch(24, 2), "16 layers, expected 20 for 2 rounds"), (patch(28, 15), "15 layers, expected 16"),
                         (patch(FIXED_HEADER_LEN + 8, 7), "shapes"), (data[:-1], "truncated"),
                         (data[:40], "shorter than the 160-byte header"), (data[:20], "shorter than the 32-byte header"),
                         (data + b"\0", "trailing")]:
        with pytest.raises(ValueError, match=message):
            from_bytes(bad)


def header(width, rounds):
    """The 32-byte fixed header of a version 2 file declaring this width and round count."""
    return b"ASNET\0\0\0" + struct.pack("<IQIII", 2, NET_ENCODING_HASH, width, rounds, (12 + 4 * rounds) % 2**32)


def test_widths_and_rounds_are_checked_and_non_finite_weights_rejected_before_building_a_net(monkeypatch):
    """Preflight F4: same rules as the Rust loader, and no torch module is built from an
    unvalidated header."""
    torch.manual_seed(4)
    good = to_bytes(Net(3, 2))
    assert good[:FIXED_HEADER_LEN] == header(3, 2)

    def no_net(*args, **kwargs):
        raise AssertionError("a Net was built before the file was validated")

    monkeypatch.setattr(model, "Net", no_net)
    for width, rounds, message in [(NET_MAX_WIDTH + 1, 1, "width 1025 is out of range"), (0, 1, "width 0 is out"),
                                   (2**31, 1, "width 2147483648 is out"), (8, 0, "rounds 0 is out of range"),
                                   (8, NET_MAX_ROUNDS + 1, "rounds 17 is out of range")]:
        with pytest.raises(ValueError, match=message):
            from_bytes(header(width, rounds))
    with pytest.raises(ValueError, match="do not fit network v2"):  # at the bound: accepted, then the shapes differ
        from_bytes(header(NET_MAX_WIDTH, 2) + good[FIXED_HEADER_LEN:])
    for x in (float("nan"), float("inf"), float("-inf")):
        bad = bytearray(good)
        struct.pack_into("<f", bad, FIXED_HEADER_LEN + 8 * 20 + 4, x)
        with pytest.raises(ValueError, match="NaN or infinite"):
            from_bytes(bytes(bad))
    with pytest.raises(ValueError, match=f"width 1025 is out of range 1..={NET_MAX_WIDTH}"):
        Net(NET_MAX_WIDTH + 1, 1)  # the class itself, not the patched module name
    with pytest.raises(ValueError, match=f"rounds 17 is out of range 1..={NET_MAX_ROUNDS}"):
        Net(8, NET_MAX_ROUNDS + 1)
```

Smaller nets in the loss tests: `Net(16, 8)` becomes `Net(8, 1)` (in `test_the_loss_is_the_spec_sum_with_l2_over_weights_only` and `test_value_and_aux_terms_use_only_the_value_rows`), `Net(32, 16)` becomes `Net(16, 1)` (in `test_fast_searches_add_nothing_to_the_policy_loss`) and `Net(64, 32)` becomes `Net(32, 2)` (in `test_a_tiny_dataset_can_be_overfit`, whose thresholds stay; if it misses them, that is a forward-pass or masking bug to find, not a bound to loosen).

`TINY_TRAIN` becomes:

```python
TINY_TRAIN = TrainConfig(lr=1e-3, steps=7, batch=16, device="cpu", width=8, rounds=1, threads=2,
                         eval_every=2, eval_batches=2, eval_batch=32)
```

Replace `test_train_exports_a_net_the_rust_side_loads_and_checks_parity` with:

```python
def test_train_exports_a_net_the_rust_side_loads_and_checks_parity(tmp_path):
    w = load_window(tmp_path, [generation(tmp_path)])
    nets = tmp_path / "nets"
    nets.mkdir()
    cfg = TrainConfig(lr=1e-3, steps=5, batch=16, device="cpu", width=8, rounds=1, threads=2)
    s = train(w, nets / "t.bin", cfg)
    assert s["steps"] == 5 and s["device"] == "cpu" and s["parity_max_error"] <= 1e-5
    assert (s["width"], s["rounds"]) == (8, 1)
    assert not [p for p in nets.iterdir() if p.name.startswith(".")], "no temp file left"
    assert Bot("net:t@5", 0, nets_dir=str(nets)).name == "net"
    train(w, nets / "u.bin", replace(cfg, threads=1))
    assert (nets / "u.bin").read_bytes() == (nets / "t.bin").read_bytes(), "CPU training is reproducible"
    s = train(w, nets / "v.bin", replace(cfg, lr=3e-4, steps=2, width=4, rounds=2), init=nets / "t.bin")
    assert s["init"] == str(nets / "t.bin") and (s["width"], s["rounds"]) == (8, 1), "the init file's shape"
    assert from_bytes((nets / "v.bin").read_bytes()).width == 8
    with pytest.raises(RuntimeError, match="disagree"):
        parity_check(Net(8, 1), nets / "t.bin", w)
    nan_net = model.load(nets / "t.bin")
    with torch.no_grad():
        nan_net.value.weight[0, 0] = float("nan")
    with pytest.raises(RuntimeError, match="disagree"):
        parity_check(nan_net, nets / "t.bin", w)
    for bad in (replace(cfg, batch=2**16 + 1), replace(cfg, lr=float("nan")), replace(cfg, lr=0.0),
                replace(cfg, eval_every=0), replace(cfg, eval_batches=0), replace(cfg, eval_batch=0),
                replace(cfg, eval_batch=2**16 + 1), replace(cfg, width=0), replace(cfg, width=NET_MAX_WIDTH + 1),
                replace(cfg, rounds=NET_MAX_ROUNDS + 1)):
        with pytest.raises(ValueError):
            train(w, nets / "x.bin", bad)
    assert not (nets / "x.bin").exists()
```

Replace `test_cli_train` and `SMALL` with:

```python
def test_cli_train(tmp_path):
    gen = generation(tmp_path)
    out = tmp_path / "nets" / "c.bin"
    r = cli("train", str(gen), "--out", str(out), "--steps", "3", "--batch", "16", "--device", "cpu",
            "--width", "8", "--rounds", "1")
    assert r.returncode == 0, r.stderr
    assert f"wrote {out}" in r.stdout and "parity" in r.stdout
    assert "1 held out" in r.stdout and "kept the best held-out score" in r.stdout
    assert r.stderr == ""
    empty = tmp_path / "gen-001"
    empty.mkdir()
    r = cli("train", str(empty), "--out", str(out))
    assert r.returncode == 2 and "no training positions" in r.stderr
    r = cli("train", str(gen), "--out", str(out), "--trunk", "16")
    assert r.returncode == 2 and "unrecognized arguments: --trunk 16" in r.stderr, "network v2 has no trunk"
    r = cli("train", str(gen), "--out", str(out), "--rounds", str(NET_MAX_ROUNDS + 1))
    assert r.returncode == 2 and "rounds 17 is out of range" in r.stderr


SMALL = ["--steps", "1", "--batch", "16", "--device", "cpu", "--width", "8", "--rounds", "1"]
```

Replace `test_cli_names_a_missing_generation_and_checks_init_before_reading_records` with (the `bad.bin` bytes shrink because the fixed header is now 32 bytes):

```python
def test_cli_names_a_missing_generation_and_checks_init_before_reading_records(tmp_path):
    out = str(tmp_path / "nets" / "c.bin")
    missing = tmp_path / "gen-404"
    r = cli("train", str(missing), "--out", out)
    assert r.returncode == 2 and "gen-404 does not exist" in r.stderr
    bad = tmp_path / "bad.bin"
    bad.write_bytes(b"not a weights file")  # shorter than the 32-byte fixed header
    # The missing directory would fail too: the --init error coming first shows the order.
    r = cli("train", str(missing), "--out", out, "--init", str(bad))
    assert r.returncode == 2 and "bad.bin: truncated" in r.stderr and "gen-404" not in r.stderr
    r = cli("train", str(missing), "--out", out, "--init", str(tmp_path / "absent.bin"))
    assert r.returncode == 2 and "absent.bin" in r.stderr and "gen-404" not in r.stderr
    mlp = bytearray(to_bytes(Net(4, 1)))
    struct.pack_into("<I", mlp, 8, 1)  # a version 1 (flat MLP) file
    (tmp_path / "mlp.bin").write_bytes(bytes(mlp))
    r = cli("train", str(missing), "--out", out, "--init", str(tmp_path / "mlp.bin"))
    assert r.returncode == 2 and "format version 1, expected 2" in r.stderr and "retrain" in r.stderr
    assert "gen-404" not in r.stderr
```

In `tests/python/test_train_loop.py`: add `from alphasettler._engine import NET_MAX_ROUNDS, NET_MAX_WIDTH` above the `alphasettler.train.loop` import; in `TINY` replace `trunk=16, value=8` with `width=8, rounds=1`; in both CLI argument lists (`test_cli_train_loop` and `test_cli_train_loop_ctrl_c_exits_130_and_the_rerun_resumes`) replace `"--trunk", "16",` / `"--value-width", "8",` with `"--width", "8",` / `"--rounds", "1",`. In `test_a_full_run_writes_every_output_and_redoes_nothing`, below the `lr` assertion add:

```python
    assert [(g["train"]["width"], g["train"]["rounds"]) for g in (g0, g1)] == [(8, 1), (8, 1)]
    assert m["format"] == 2
```

Extend the parameters of `test_bad_settings_are_an_error_before_anything_is_written` with:

```python
                                    {"warm_games": 10_000_001}, {"width": 0}, {"rounds": 0},
                                    {"width": NET_MAX_WIDTH + 1}, {"rounds": NET_MAX_ROUNDS + 1}, {"width": 2.5}])
```

(replacing the list's closing `{"warm_games": 10_000_001}])`), and add before `test_a_changed_output_of_a_completed_step_is_an_error`:

```python
def test_a_manifest_of_another_format_is_refused_before_anything_is_written(tmp_path):
    """A format 1 run directory (the flat MLP's, such as runs/3b) is not resumed by network v2."""
    (tmp_path / "manifest.json").write_text(json.dumps({"format": 1, "best": 0, "generations": []}))
    with pytest.raises(ValueError, match="has format 1, expected 2: a format 1 run trained the flat MLP"):
        Loop(tmp_path, TINY)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["manifest.json"]


def test_linked_records_satisfy_a_new_runs_self_play_without_playing(tmp_path, reference, monkeypatch):
    """Plan 5b's warm-start retrain: generation 0's plan and chunks hard-linked into a new run
    directory complete that run's self-play step without a game, with the same manifest entry."""
    import alphasettler.selfplay as selfplay_mod

    gen = tmp_path / "run" / "gen-000"
    gen.mkdir(parents=True)
    for p in (reference / "gen-000").iterdir():
        if p.name == "selfplay.json" or p.name.startswith("selfplay-local-"):
            os.link(p, gen / p.name)

    def no_games(*args, **kwargs):
        raise AssertionError("a game was played")

    monkeypatch.setattr(selfplay_mod, "write_explore", no_games)
    assert Loop(tmp_path / "run", TINY).run(stop_after=1) == 1
    assert manifest(tmp_path / "run")["generations"][0]["selfplay"] == manifest(reference)["generations"][0]["selfplay"]
```

- [ ] **Step 8: Run the Python tests to verify they fail**

Run: `.venv/bin/pytest -q tests/python/test_train.py`
Expected: FAIL at collection: `ImportError: cannot import name 'NET_DEFAULT_ROUNDS' from 'alphasettler._engine'` (the installed extension and `model.py` are still the MLP's).

- [ ] **Step 9: The bindings**

In `bindings/src/train.rs`, the module doc's first sentence ends `..., the Rust forward pass, network v2's adjacency for the PyTorch twin, net lookup and test weights.`; the `settler_net` imports become:

```rust
use settler_net::encode::{
    encode_state, EDGE_WIDTH, ENCODING_HASH, HIDDEN_LEN, NODE_WIDTH, PUBLIC_LEN, TILE_WIDTH,
};
use settler_net::forward::{Forward, Outputs};
use settler_net::graph::{graph, GLOBAL_WIDTH};
use settler_net::policy::SETTLEMENT_BASE;
use settler_net::weights::{Arch, Weights, FORMAT_VERSION, MAX_ROUNDS, MAX_WIDTH};
```

Replace `write_random_net` (with its doc comment) with:

```rust
/// Network v2's static adjacency, for the PyTorch twin: a dict of `node_tiles`, `node_edges`,
/// `node_neighbours`, `edge_nodes` and `tile_nodes`, each a list (per entity) of index lists in
/// the order the Rust forward pass sums them. `edge_nodes` is summed; the others are averaged.
#[pyfunction]
fn net_adjacency(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
    let g = graph();
    let d = PyDict::new(py);
    for (name, links) in [
        ("node_tiles", &g.node_tiles),
        ("node_edges", &g.node_edges),
        ("node_neighbours", &g.node_neighbours),
        ("edge_nodes", &g.edge_nodes),
        ("tile_nodes", &g.tile_nodes),
    ] {
        d.set_item(name, links.lists())?;
    }
    Ok(d)
}

/// Random network v2 weights (width `NET_DEFAULT_WIDTH` and `NET_DEFAULT_ROUNDS` rounds unless
/// given) saved to `path`, for tests and measurements.
#[pyfunction]
#[pyo3(signature = (path, seed, width=None, rounds=None))]
fn write_random_net(
    path: PathBuf,
    seed: &Bound<'_, PyAny>,
    width: Option<&Bound<'_, PyAny>>,
    rounds: Option<&Bound<'_, PyAny>>,
) -> PyResult<()> {
    let seed: u64 = int_arg(seed, "seed")?;
    let dim = |v: Option<&Bound<'_, PyAny>>, default: usize, what: &str| -> PyResult<usize> {
        match v {
            Some(v) => int_arg(v, what),
            None => Ok(default),
        }
    };
    let arch = Arch {
        width: dim(width, Arch::DEFAULT.width, "width")?,
        rounds: dim(rounds, Arch::DEFAULT.rounds, "rounds")?,
    };
    arch.check().map_err(value_error)?;
    Weights::random(arch, seed).save(&path).map_err(value_error)
}
```

and in `register`, add after the `write_random_net` line and after the `NET_MAX_WIDTH` line respectively:

```rust
    m.add_function(wrap_pyfunction!(net_adjacency, m)?)?;
```

```rust
    m.add("NET_MAX_ROUNDS", MAX_ROUNDS)?;
    m.add("NET_DEFAULT_WIDTH", Arch::DEFAULT.width)?;
    m.add("NET_DEFAULT_ROUNDS", Arch::DEFAULT.rounds)?;
    m.add("NET_NODE_WIDTH", NODE_WIDTH)?;
    m.add("NET_EDGE_WIDTH", EDGE_WIDTH)?;
    m.add("NET_TILE_WIDTH", TILE_WIDTH)?;
    m.add("NET_GLOBAL_WIDTH", GLOBAL_WIDTH)?;
    m.add("NET_SETTLEMENT_BASE", SETTLEMENT_BASE)?;
```

(`Vec<Vec<usize>>` becomes a list of lists of ints; a `Vec<u8>` would become `bytes`, which is why `Links::lists` widens to `usize`.)

- [ ] **Step 10: The PyTorch twin**

Replace `alphasettler/train/model.py` with:

```python
"""The PyTorch twin of net/src/forward.rs, network v2 (spec Section 2): an entity-structured graph
net over the encoder's 54 nodes, 72 edges, 19 tiles and 91 global scalars. Also the weights file
format of net/src/weights.rs, version 2: little-endian, a 32-byte fixed header (magic, format
version, ENCODING_HASH, width d, rounds R, layer count), each layer's rows and columns, then each
layer's f32 weights (row-major, rows = outputs) and biases. The layer order, byte for byte, is the
table in net/src/weights.rs; `Net.layers()` returns the modules in that order.

The adjacency comes from the Rust side (`net_adjacency`) as averaging and summing matrices, so both
forward passes read the board the same way. `from_bytes` checks a file exactly as the Rust loader
does, and checks all of it (header, length, every weight finite) before it builds a module, so a
corrupt header never sizes an allocation."""

from __future__ import annotations

import struct
import sys
from array import array
from pathlib import Path

import torch
from torch import nn

from alphasettler._engine import (
    NET_DEFAULT_ROUNDS, NET_DEFAULT_WIDTH, NET_EDGE_WIDTH, NET_ENCODING_HASH, NET_FORMAT_VERSION, NET_GLOBAL_WIDTH,
    NET_HIDDEN_LEN, NET_MAX_ROUNDS, NET_MAX_WIDTH, NET_NODE_WIDTH, NET_PUBLIC_LEN, NET_SETTLEMENT_BASE,
    NET_TILE_WIDTH, action_space_size, net_adjacency,
)

MAGIC = b"ASNET\0\0\0"
ACTIONS = action_space_size()
SEATS = 4
FIXED_HEADER_LEN = 8 + 4 + 8 + 4 + 4 + 4
ADJACENCY = net_adjacency()
NODES, EDGES, TILES = len(ADJACENCY["node_tiles"]), len(ADJACENCY["edge_nodes"]), len(ADJACENCY["tile_nodes"])
NODE_END = NODES * NET_NODE_WIDTH
EDGE_END = NODE_END + EDGES * NET_EDGE_WIDTH
TILE_END = EDGE_END + TILES * NET_TILE_WIDTH
assert TILE_END + NET_GLOBAL_WIDTH == NET_PUBLIC_LEN, "nodes, edges, tiles and the globals tile the public vector"
# The pooled head's slots: 0..NET_SETTLEMENT_BASE, then everything after the entity heads' slots.
OTHER = ACTIONS - 2 * NODES - EDGES - TILES


def check_arch(width: int, rounds: int) -> None:
    """ValueError (with the Rust loader's wording) for a width or round count out of range."""
    if not 1 <= width <= NET_MAX_WIDTH:
        raise ValueError(f"width {width} is out of range 1..={NET_MAX_WIDTH}")
    if not 1 <= rounds <= NET_MAX_ROUNDS:
        raise ValueError(f"rounds {rounds} is out of range 1..={NET_MAX_ROUNDS}")


def layer_count(rounds: int) -> int:
    return 4 + 4 * rounds + 8


def shapes(width: int, rounds: int) -> list[tuple[int, int]]:
    """(rows, cols) of each layer in file order; rows are outputs (as Rust's `Arch::shapes`)."""
    d = width
    out = [(d, NET_NODE_WIDTH), (d, NET_EDGE_WIDTH), (d, NET_TILE_WIDTH), (d, NET_GLOBAL_WIDTH)]
    out += [(d, 3 * d), (d, 5 * d), (d, 3 * d), (d, 4 * d)] * rounds
    return out + [(2, 2 * d), (1, 2 * d), (1, 2 * d), (4 * d, 4 * d), (OTHER, 4 * d),
                  (4 * d, 4 * d + NET_HIDDEN_LEN), (SEATS, 4 * d), (SEATS, 4 * d)]


def _links(lists: list[list[int]], cols: int, mean: bool) -> torch.Tensor:
    """The matrix whose row i averages (or sums) the columns `lists[i]`."""
    m = torch.zeros(len(lists), cols)
    for i, js in enumerate(lists):
        for j in js:
            m[i, j] += 1.0 / len(js) if mean else 1.0
    return m


def _spread(g: torch.Tensor, count: int) -> torch.Tensor:
    """A per-position row (batch x d) repeated for each of `count` entities (batch x count x d)."""
    return g.unsqueeze(1).expand(-1, count, -1)


class Round(nn.Module):
    """One round's four residual updates, applied in this order: tile, node, edge, global."""

    def __init__(self, d: int):
        super().__init__()
        self.tile = nn.Linear(3 * d, d)
        self.node = nn.Linear(5 * d, d)
        self.edge = nn.Linear(3 * d, d)
        self.glob = nn.Linear(4 * d, d)


class Net(nn.Module):
    """Encoders per entity type, `rounds` rounds of message passing, per-entity policy heads for
    settlements, cities, roads and robber moves, a pooled head for every other slot, and the value
    and aux heads on [pooled || hidden]. All value and aux outputs are seat-relative."""

    def __init__(self, width: int = NET_DEFAULT_WIDTH, rounds: int = NET_DEFAULT_ROUNDS):
        check_arch(width, rounds)
        super().__init__()
        d = width
        self.width, self.rounds = width, rounds
        self.node_encoder = nn.Linear(NET_NODE_WIDTH, d)
        self.edge_encoder = nn.Linear(NET_EDGE_WIDTH, d)
        self.tile_encoder = nn.Linear(NET_TILE_WIDTH, d)
        self.global_encoder = nn.Linear(NET_GLOBAL_WIDTH, d)
        self.round_layers = nn.ModuleList([Round(d) for _ in range(rounds)])
        self.node_policy = nn.Linear(2 * d, 2)
        self.edge_policy = nn.Linear(2 * d, 1)
        self.tile_policy = nn.Linear(2 * d, 1)
        self.other_hidden = nn.Linear(4 * d, 4 * d)
        self.other_policy = nn.Linear(4 * d, OTHER)
        self.value_hidden = nn.Linear(4 * d + NET_HIDDEN_LEN, 4 * d)
        self.value = nn.Linear(4 * d, SEATS)
        self.aux = nn.Linear(4 * d, SEATS)
        a = ADJACENCY
        # Not in the state dict: fixed by the board, rebuilt with every Net.
        self.register_buffer("tile_nodes", _links(a["tile_nodes"], NODES, True), persistent=False)
        self.register_buffer("node_tiles", _links(a["node_tiles"], TILES, True), persistent=False)
        self.register_buffer("node_edges", _links(a["node_edges"], EDGES, True), persistent=False)
        self.register_buffer("node_neighbours", _links(a["node_neighbours"], NODES, True), persistent=False)
        self.register_buffer("edge_nodes", _links(a["edge_nodes"], NODES, False), persistent=False)

    def layers(self) -> list[nn.Linear]:
        """In weights-file order."""
        out = [self.node_encoder, self.edge_encoder, self.tile_encoder, self.global_encoder]
        for r in self.round_layers:
            out += [r.tile, r.node, r.edge, r.glob]
        return out + [self.node_policy, self.edge_policy, self.tile_policy, self.other_hidden, self.other_policy,
                      self.value_hidden, self.value, self.aux]

    def forward(self, public: torch.Tensor, hidden: torch.Tensor):
        b = public.shape[0]
        relu = torch.relu
        node = relu(self.node_encoder(public[:, :NODE_END].reshape(b, NODES, NET_NODE_WIDTH)))
        edge = relu(self.edge_encoder(public[:, NODE_END:EDGE_END].reshape(b, EDGES, NET_EDGE_WIDTH)))
        tile = relu(self.tile_encoder(public[:, EDGE_END:TILE_END].reshape(b, TILES, NET_TILE_WIDTH)))
        glob = relu(self.global_encoder(public[:, TILE_END:]))
        for r in self.round_layers:
            tile = tile + relu(r.tile(torch.cat([tile, self.tile_nodes @ node, _spread(glob, TILES)], 2)))
            node = node + relu(r.node(torch.cat([node, self.node_tiles @ tile, self.node_edges @ edge,
                                                 self.node_neighbours @ node, _spread(glob, NODES)], 2)))
            edge = edge + relu(r.edge(torch.cat([edge, self.edge_nodes @ node, _spread(glob, EDGES)], 2)))
            glob = glob + relu(r.glob(torch.cat([glob, node.mean(1), edge.mean(1), tile.mean(1)], 1)))
        pooled = torch.cat([glob, node.mean(1), edge.mean(1), tile.mean(1)], 1)
        build = self.node_policy(torch.cat([node, _spread(glob, NODES)], 2))
        road = self.edge_policy(torch.cat([edge, _spread(glob, EDGES)], 2))[:, :, 0]
        robber = self.tile_policy(torch.cat([tile, _spread(glob, TILES)], 2))[:, :, 0]
        other = self.other_policy(relu(self.other_hidden(pooled)))
        s = NET_SETTLEMENT_BASE
        policy = torch.cat([other[:, :s], build[:, :, 0], build[:, :, 1], road, robber, other[:, s:]], 1)
        v = relu(self.value_hidden(torch.cat([pooled, hidden], 1)))
        return policy, self.value(v), self.aux(v)


def _floats(t: torch.Tensor) -> bytes:
    return array("f", t.detach().to("cpu", torch.float32).contiguous().view(-1).tolist()).tobytes()


def to_bytes(net: Net) -> bytes:
    assert sys.byteorder == "little", "the weights file is little-endian"
    layers = net.layers()
    out = [MAGIC, struct.pack("<IQIII", NET_FORMAT_VERSION, NET_ENCODING_HASH, net.width, net.rounds, len(layers))]
    out += [struct.pack("<II", layer.out_features, layer.in_features) for layer in layers]
    for layer in layers:
        out += [_floats(layer.weight), _floats(layer.bias)]
    return b"".join(out)


def from_bytes(data: bytes) -> Net:
    """The network in `data`; ValueError (with the Rust loader's wording) for a file it rejects."""
    assert sys.byteorder == "little", "the weights file is little-endian"
    if len(data) < FIXED_HEADER_LEN:
        raise ValueError(f"truncated: {len(data)} bytes is shorter than the {FIXED_HEADER_LEN}-byte header")
    if data[:8] != MAGIC:
        raise ValueError("not an AlphaSettler weights file (wrong magic)")
    version, encoding, width, rounds, count = struct.unpack_from("<IQIII", data, 8)
    if version != NET_FORMAT_VERSION:
        why = "; version 1 holds the flat MLP that network v2 replaced: retrain the net" if version == 1 else ""
        raise ValueError(f"format version {version}, expected {NET_FORMAT_VERSION}{why}")
    if encoding != NET_ENCODING_HASH:
        raise ValueError(f"encoding hash {encoding:#018x} does not match this encoder's {NET_ENCODING_HASH:#018x}: "
                         "the file was trained on a different feature layout")
    check_arch(width, rounds)
    if count != layer_count(rounds):
        raise ValueError(f"{count} layers, expected {layer_count(rounds)} for {rounds} rounds")
    header = FIXED_HEADER_LEN + 8 * count
    if len(data) < header:
        raise ValueError(f"truncated: {len(data)} bytes is shorter than the {header}-byte header")
    declared = [struct.unpack_from("<II", data, FIXED_HEADER_LEN + 8 * i) for i in range(count)]
    if declared != shapes(width, rounds):
        raise ValueError(f"layer shapes {declared} do not fit network v2 with width {width} and {rounds} rounds")
    want = header + 4 * sum(r * c + r for r, c in declared)
    if len(data) < want:
        raise ValueError(f"truncated: {len(data)} bytes, expected {want}")
    if len(data) > want:
        raise ValueError(f"{len(data) - want} trailing bytes after the last layer")
    params = torch.frombuffer(bytearray(data[header:]), dtype=torch.float32)
    if not torch.isfinite(params).all():
        raise ValueError("a weight is NaN or infinite")
    net = Net(width=width, rounds=rounds)
    at = 0
    with torch.no_grad():
        for layer in net.layers():
            for p in (layer.weight, layer.bias):
                n = p.numel()
                p.copy_(params[at : at + n].view_as(p))
                at += n
    return net


def save(net: Net, path: str | Path) -> None:
    Path(path).write_bytes(to_bytes(net))


def load(path: str | Path) -> Net:
    path = Path(path)
    try:
        return from_bytes(path.read_bytes())
    except ValueError as e:
        raise ValueError(f"{path}: {e}") from None
```

- [ ] **Step 11: Training, the loop and the CLI take width and rounds**

`alphasettler/train/train.py`: the `_engine` import becomes `from alphasettler._engine import NET_DEFAULT_ROUNDS, NET_DEFAULT_WIDTH, NET_MAX_SAMPLE, TrainingSet, net_forward`; in `TrainConfig` replace the `trunk` and `value` fields with:

```python
    width: int = NET_DEFAULT_WIDTH  # network v2's d and R, for a net trained from scratch (an `init`
    rounds: int = NET_DEFAULT_ROUNDS  # file brings its own)
```

in `TrainConfig.check`, right after the learning-rate check, add `model_mod.check_arch(self.width, self.rounds)`; in `train`, `model_mod.Net(cfg.trunk, cfg.value)` becomes `model_mod.Net(cfg.width, cfg.rounds)`; and the returned summary gains the trained net's shape:

```python
    return {"steps": steps, "lr": cfg.lr, "batch": cfg.batch, "init": None if init is None else str(init),
            "width": net.width, "rounds": net.rounds, "device": device.type, "history": history, "held_out": held_out,
            "best_step": None if best_row is None else best_row["step"], "best_held_out": best_row,
            "parity_max_error": err}
```

`alphasettler/train/loop.py`: the `_engine` import becomes `from alphasettler._engine import MAX_SEEDS, MAX_SIMULATIONS, NET_DEFAULT_ROUNDS, NET_DEFAULT_WIDTH, resolve_config`; the manifest format becomes:

```python
# 2: network v2 (train steps record the net's width and rounds). A format 1 run holds flat-MLP nets,
# which no longer load.
MANIFEST_FORMAT = 2
```

in `LoopSettings` replace `trunk: int = 512` / `value: int = 256` with `width: int = NET_DEFAULT_WIDTH` / `rounds: int = NET_DEFAULT_ROUNDS`; in `_check`, the positive-integer names become `("games", "warm_games", "chunk_games", "window", "width", "rounds", "gate_seeds", "gate_simulations", "max_turns")`, the `for name in ("trunk", "value"): ... NET_MAX_WIDTH ...` loop is deleted, and the learning-rate loop checks the shape too:

```python
    for lr in (s.warm_lr, s.lr):
        TrainConfig(lr=lr, steps_factor=s.steps_factor, batch=s.batch, threads=s.threads, width=s.width,
                    rounds=s.rounds).check()
```

in `_load` the format error explains itself:

```python
        if m.get("format") != MANIFEST_FORMAT:
            raise ValueError(f"{path} has format {m.get('format')!r}, expected {MANIFEST_FORMAT}: a format 1 run "
                             "trained the flat MLP, whose nets no longer load; start network v2 in a new directory")
```

in `_train`, the `TrainConfig(...)` call ends `width=self.s.width, rounds=self.s.rounds)` instead of `trunk=self.s.trunk, value=self.s.value)`, and the returned entry gains, after `"lr": cfg.lr,`:

```python
            "width": summary["width"],
            "rounds": summary["rounds"],
```

`alphasettler/cli.py`: the `_engine` import becomes `from alphasettler._engine import DEFAULT_NETS_DIR, NET_DEFAULT_ROUNDS, NET_DEFAULT_WIDTH, Bot, bot_names`; in `train`'s parser replace the `--trunk` and `--value-width` arguments with:

```python
    t.add_argument("--width", type=_positive_int, default=NET_DEFAULT_WIDTH,
                   help="network v2's width d, from scratch (an --init file brings its own)")
    t.add_argument("--rounds", type=_positive_int, default=NET_DEFAULT_ROUNDS,
                   help="network v2's message-passing rounds R, from scratch (an --init file brings its own)")
```

in `train-loop`'s parser replace them with:

```python
    lp.add_argument("--width", type=_positive_int, default=NET_DEFAULT_WIDTH, help="network v2's width d for net 0")
    lp.add_argument("--rounds", type=_positive_int, default=NET_DEFAULT_ROUNDS,
                    help="network v2's message-passing rounds R for net 0")
```

and in `_train` and `_train_loop` pass `width=args.width, rounds=args.rounds` where they passed `trunk=args.trunk, value=args.value_width`.

- [ ] **Step 12: Run the Python tests to verify they pass**

Run: `PATH=$HOME/.cargo/bin:$PATH .venv/bin/maturin develop --release -q && .venv/bin/pytest -q -rs tests/python/test_train.py tests/python/test_train_loop.py tests/python/test_training_data.py tests/python/test_netbot.py tests/python/test_oracle_arena.py`
Expected: 111 passed, none skipped (about 2 minutes). The parity test's worst error is near 1e-6 (2.6e-6 at width 5, 3 rounds, when this plan was written).

- [ ] **Step 13: Format, lint, full checks**

Run: `~/.cargo/bin/cargo fmt --all && ~/.cargo/bin/cargo fmt --check`
Expected: no output.
Run: `~/.cargo/bin/cargo clippy --workspace --release --all-targets -- -D warnings`
Expected: exits 0.
Run: `~/.cargo/bin/cargo test --workspace --release`
Expected: all pass, `golden_trace` included.
Run: `.venv/bin/pytest -q -rs`
Expected: all pass (255 when this plan was written); the summary lists no skipped tests.

- [ ] **Step 14: Commit**

```bash
git add net/src/weights.rs net/src/forward.rs net/tests/weights.rs bots/tests/netbot.rs bindings/src/train.rs alphasettler/train/model.py alphasettler/train/train.py alphasettler/train/loop.py alphasettler/cli.py tests/python/test_train.py tests/python/test_train_loop.py tests/python/test_training_data.py tests/python/test_netbot.py tests/python/test_oracle_arena.py
git diff --cached | grep -nE 'sk-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{36}|password=[^[:space:]]+'
git -c user.email=sean.may101@gmail.com commit -m "net: network v2, the entity-structured graph net, in Rust and PyTorch (weights format 2)"
```

(`runs/` is git-ignored: `runs/3b-v2/net-speed-random.log` stays out of the commit.)

---

### Task C: `prior_check`, the acceptance gate's diagnostics

**Files:**
- Create: `bots/examples/prior_check.rs`

**Interfaces:**
- Consumes: `settler_bots::selfplay::{run_with, Exploration}` (`run_with(seeds: Range<u64>, config, &Exploration, batch: usize, make_eval: &F, threads) -> Result<Vec<SelfPlayGame>, String>`, each `Decision` with `index`, `actor`, `legal`, `visits`, `full_search`); `settler_bots::arena::{run_match_by, GameRecord}`; `settler_bots::heuristic::{prior, HeuristicEvaluator, DEFAULT_WEIGHTS}`; `IsmctsBot::with_evaluator`; `make_bot`; `NetEvaluator::new` (Task A's seat-relative masking); `Weights::{load, random}` and `Arch::DEFAULT` (Task B); `settler_search::{Eval, Evaluator, Leaf, SearchConfig}`.
- Produces, for Task D:
  - `prior_check agree PATH GAMES`: one line per move kind (`setup-settlement`, `setup-road`, `settlement`, `city`, `road`, `robber`, `steal`, ...), then `ALL`, each `KIND: n N  heur-top1 H  net-top1 T  net==heur E`; the last line `agreement gate (net-top1 >= 0.5 on setup-settlement, robber, road, over >= 3000 searches): pass|FAIL over G games, N searches`.
  - `prior_check hybrid PATH SIMULATIONS SEEDS`: four lines `MIX @S vs three ismcts@S: win rate R (W/G, D draws), z vs 0.25 = Z, mean VP V, belief resets B`, then `hybrid gate (net prior + heuristic value above 0.140): pass|FAIL at R`.
  - `PATH` may be `random` (random v2 weights of the default shape), for smoke runs.

The two throwaway diagnostics behind the spec's numbers (the MLP's 9 / 11 / 19% agreement and its 0.140 hybrid) become this one tool: `agree` is the agreement count by move kind, `hybrid` the prior/value mixes. The hybrid evaluator lives here, not in library code. Its test is the smoke run in Steps 2–3: on a random net the heuristic rows reproduce known values exactly.

- [ ] **Step 1: Write the example**

Create `bots/examples/prior_check.rs`:

```rust
//! The network acceptance gate (spec Section 2 and done criterion 3b): how good is a net's prior?
//!
//!   prior_check agree PATH GAMES
//!       GAMES fresh `ismcts` self-play games with the warm start's exploration (playout cap
//!       randomization, root noise, opening sampling; seeds from 800,000,000, trades off) on all
//!       cores. At each full search with two or more moves: does the heuristic prior's top move,
//!       and the net prior's, equal the search's most-visited move? Reported by that move's kind,
//!       then the gate: the net at least 0.5 on setup settlements, robber moves and roads, over at
//!       least 3,000 searches.
//!   prior_check hybrid PATH SIMULATIONS SEEDS
//!       four evaluators, each at SIMULATIONS against three ismcts@SIMULATIONS over seeds
//!       0..SEEDS (all cores, trades off): heuristic prior and value (plays exactly as ismcts),
//!       net prior and value, net prior with the heuristic value, heuristic prior with the net
//!       value. The gate: "net prior + heuristic value" above the flat MLP's 0.140 (measured at
//!       100 simulations over 50 seeds).
//!
//! PATH is a weights file, or `random` for random weights of the default shape.

use settler_bots::arena::{run_match_by, GameRecord};
use settler_bots::heuristic::{self, HeuristicEvaluator, DEFAULT_WEIGHTS};
use settler_bots::ismcts::IsmctsBot;
use settler_bots::selfplay::{run_with, Exploration};
use settler_bots::{make_bot, Bot};
use settler_engine::{Action, GameConfig, Phase, State};
use settler_net::evaluator::NetEvaluator;
use settler_net::weights::{Arch, Weights};
use settler_search::{Eval, Evaluator, Leaf, SearchConfig};
use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

/// Seeds of the agreement games: far from the warm start's 0..100,000 and the gate's 10^9.
const AGREE_SEED_START: u64 = 800_000_000;
const AGREE_GATE: f64 = 0.5;
const AGREE_MIN_SEARCHES: u64 = 3_000;
const AGREE_GATE_KINDS: [&str; 3] = ["setup-settlement", "robber", "road"];
/// The flat MLP's "net prior + heuristic value" win rate at 100 simulations over 50 seeds.
const HYBRID_BAR: f64 = 0.140;

fn weights(path: &str) -> Arc<Weights> {
    Arc::new(if path == "random" {
        Weights::random(Arch::DEFAULT, 1)
    } else {
        Weights::load(Path::new(path)).unwrap_or_else(|e| panic!("{e}"))
    })
}

fn no_trades() -> GameConfig {
    GameConfig {
        max_offers_per_turn: 0,
        ..GameConfig::default()
    }
}

fn cores() -> usize {
    std::thread::available_parallelism().map_or(1, |c| c.get())
}

/// The kind of move a search chose; setup settlements and roads apart from later ones.
fn kind(a: &Action, setup: bool) -> &'static str {
    match a {
        Action::BuildSettlement(_) if setup => "setup-settlement",
        Action::BuildRoad(_) if setup => "setup-road",
        Action::BuildSettlement(_) => "settlement",
        Action::BuildCity(_) => "city",
        Action::BuildRoad(_) => "road",
        Action::Roll => "roll",
        Action::EndTurn => "end-turn",
        Action::BuyDev => "buy-dev",
        Action::PlayKnight
        | Action::PlayRoadBuilding
        | Action::PlayMonopoly(_)
        | Action::PlayYearOfPlenty(..) => "play-dev",
        Action::MoveRobber(_) => "robber",
        Action::StealFrom(_) => "steal",
        Action::Discard(_) => "discard",
        Action::MaritimeTrade { .. } => "maritime",
        _ => "other",
    }
}

/// The first index of the largest entry.
fn argmax(v: &[f32]) -> usize {
    let mut best = 0;
    for i in 1..v.len() {
        if v[i] > v[best] {
            best = i;
        }
    }
    best
}

/// [searches, heuristic agrees, net agrees, net == heuristic] by kind.
type Table = BTreeMap<&'static str, [u64; 4]>;

fn agree(path: &str, games: u64) {
    let w = weights(path);
    let cfg = no_trades();
    let make = || HeuristicEvaluator::new(DEFAULT_WEIGHTS, 0);
    let seeds = AGREE_SEED_START..AGREE_SEED_START + games;
    let played = run_with(
        seeds,
        cfg,
        &Exploration::standard(),
        SearchConfig::default().batch,
        &make,
        cores(),
    )
    .unwrap_or_else(|e| panic!("{e}"));
    let mut net = NetEvaluator::new(w);
    let mut table = Table::new();
    for g in &played {
        let mut s = State::new(g.seed, cfg);
        let mut next = 0;
        for d in &g.decisions {
            while next < d.index as usize {
                s.apply(g.actions[next]);
                next += 1;
            }
            let searchable: Vec<(Action, f32)> = d
                .legal
                .iter()
                .zip(&d.visits)
                .filter(|(a, _)| !matches!(a, Action::OfferTrade { .. }))
                .map(|(&a, &v)| (a, v as f32))
                .collect();
            if !d.full_search || searchable.len() < 2 {
                continue;
            }
            let legal: Vec<Action> = searchable.iter().map(|x| x.0).collect();
            let visits: Vec<f32> = searchable.iter().map(|x| x.1).collect();
            let target = argmax(&visits);
            let heuristic = argmax(&heuristic::prior(&s.observation(d.actor), &legal));
            let leaf = Leaf {
                world: &s,
                actor: d.actor,
                legal: &legal,
            };
            let mine = argmax(&net.evaluate(&[leaf])[0].prior);
            let setup = matches!(s.phase, Phase::SetupSettlement | Phase::SetupRoad { .. });
            let row = table.entry(kind(&legal[target], setup)).or_insert([0; 4]);
            row[0] += 1;
            row[1] += (heuristic == target) as u64;
            row[2] += (mine == target) as u64;
            row[3] += (mine == heuristic) as u64;
        }
    }
    let line = |name: &str, r: &[u64; 4]| {
        let f = |k: usize| r[k] as f64 / r[0].max(1) as f64;
        println!(
            "{name:>17}: n {:>6}  heur-top1 {:.3}  net-top1 {:.3}  net==heur {:.3}",
            r[0],
            f(1),
            f(2),
            f(3)
        );
    };
    let mut all = [0u64; 4];
    for (name, r) in &table {
        line(name, r);
        for k in 0..4 {
            all[k] += r[k];
        }
    }
    line("ALL", &all);
    let rate = |name: &str| {
        table
            .get(name)
            .map_or(0.0, |r| r[2] as f64 / r[0].max(1) as f64)
    };
    let pass =
        all[0] >= AGREE_MIN_SEARCHES && AGREE_GATE_KINDS.iter().all(|k| rate(k) >= AGREE_GATE);
    println!(
        "agreement gate (net-top1 >= {AGREE_GATE} on {}, over >= {AGREE_MIN_SEARCHES} searches): {} over {} games, {} searches",
        AGREE_GATE_KINDS.join(", "),
        if pass { "pass" } else { "FAIL" },
        played.len(),
        all[0]
    );
}

#[derive(Clone, Copy)]
enum Mix {
    Heuristic,
    Net,
    NetPriorHeuristicValue,
    HeuristicPriorNetValue,
}

const MIXES: [(&str, Mix); 4] = [
    ("heuristic prior + heuristic value", Mix::Heuristic),
    ("net prior + net value", Mix::Net),
    ("net prior + heuristic value", Mix::NetPriorHeuristicValue),
    ("heuristic prior + net value", Mix::HeuristicPriorNetValue),
];

/// Both evaluators on every leaf; the prior from one and the value from the other.
struct Hybrid {
    heuristic: HeuristicEvaluator,
    net: NetEvaluator,
    mix: Mix,
}

impl Evaluator for Hybrid {
    fn evaluate(&mut self, leaves: &[Leaf<'_>]) -> Vec<Eval> {
        let h = self.heuristic.evaluate(leaves);
        let n = self.net.evaluate(leaves);
        h.into_iter()
            .zip(n)
            .map(|(h, n)| match self.mix {
                Mix::Heuristic => h,
                Mix::Net => n,
                Mix::NetPriorHeuristicValue => Eval {
                    value: h.value,
                    prior: n.prior,
                },
                Mix::HeuristicPriorNetValue => Eval {
                    value: n.value,
                    prior: h.prior,
                },
            })
            .collect()
    }
}

/// Per-seed candidate win rates; records are sorted by (seed, seat), four per seed.
fn per_seed(records: &[GameRecord]) -> Vec<f64> {
    records
        .chunks(4)
        .map(|c| {
            c.iter()
                .filter(|r| r.winner == Some(r.candidate_seat))
                .count() as f64
                / 4.0
        })
        .collect()
}

fn mean_se(xs: &[f64]) -> (f64, f64) {
    let n = xs.len() as f64;
    let mean = xs.iter().sum::<f64>() / n;
    let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0).max(1.0);
    (mean, (var / n).sqrt())
}

fn hybrid(path: &str, simulations: u32, seeds: u64) {
    let w = weights(path);
    let baseline = format!("ismcts@{simulations}");
    assert!(make_bot(&baseline, 0).is_some(), "{baseline} is a bot");
    let mut gate = 0.0;
    for (name, mix) in MIXES {
        let records = run_match_by(0..seeds, no_trades(), cores(), &|is_candidate, seed| {
            if is_candidate {
                let eval = Hybrid {
                    heuristic: HeuristicEvaluator::new(DEFAULT_WEIGHTS, 0),
                    net: NetEvaluator::new(w.clone()),
                    mix,
                };
                let cfg = SearchConfig {
                    simulations,
                    ..SearchConfig::default()
                };
                Box::new(IsmctsBot::with_evaluator("hybrid", seed, cfg, eval)) as Box<dyn Bot>
            } else {
                make_bot(&baseline, seed).expect("checked above")
            }
        })
        .unwrap_or_else(|e| panic!("{e}"));
        let wins = records
            .iter()
            .filter(|r| r.winner == Some(r.candidate_seat))
            .count();
        let draws = records.iter().filter(|r| r.winner.is_none()).count();
        let rate = wins as f64 / records.len() as f64;
        let (_, se) = mean_se(&per_seed(&records));
        let z = if se > 0.0 { (rate - 0.25) / se } else { 0.0 };
        let vp = records
            .iter()
            .map(|r| r.vp[r.candidate_seat as usize] as f64)
            .sum::<f64>()
            / records.len() as f64;
        let resets: u64 = records.iter().map(|r| r.belief_resets).sum();
        println!(
            "{name:>33} @{simulations} vs three {baseline}: win rate {rate:.3} ({wins}/{}, {draws} draws), z vs 0.25 = {z:.2}, mean VP {vp:.2}, belief resets {resets}",
            records.len()
        );
        if let Mix::NetPriorHeuristicValue = mix {
            gate = rate;
        }
    }
    println!(
        "hybrid gate (net prior + heuristic value above {HYBRID_BAR:.3}): {} at {gate:.3}",
        if gate > HYBRID_BAR { "pass" } else { "FAIL" }
    );
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let num = |i: usize| -> u64 {
        args.get(i)
            .and_then(|a| a.parse().ok())
            .unwrap_or_else(|| panic!("argument {i} must be a number"))
    };
    match args.first().map(String::as_str) {
        Some("agree") if args.len() == 3 => agree(&args[1], num(2)),
        Some("hybrid") if args.len() == 4 => hybrid(&args[1], num(2) as u32, num(3)),
        _ => {
            eprintln!(
                "usage: prior_check agree PATH GAMES | prior_check hybrid PATH SIMULATIONS SEEDS"
            );
            std::process::exit(2)
        }
    }
}
```

- [ ] **Step 2: Smoke-run `agree`**

Run: `~/.cargo/bin/cargo run --release -p settler-bots --example prior_check -- agree random 4`
Expected (a few seconds): about a dozen kind lines and `ALL: n ...` with about 390 searches; `heur-top1` on the `ALL` line near 0.95 (0.951 when this plan was written: the heuristic prior's agreement the spec reports), `net-top1` low (a random net); the last line `agreement gate (net-top1 >= 0.5 on setup-settlement, robber, road, over >= 3000 searches): FAIL over 4 games, N searches`.

- [ ] **Step 3: Smoke-run `hybrid` and the usage line**

Run: `~/.cargo/bin/cargo run --release -p settler-bots --example prior_check -- hybrid random 10 3`
Expected: four lines, each ending `belief resets 0`; the first exactly `heuristic prior + heuristic value @10 vs three ismcts@10: win rate 0.250 (3/12, 0 draws), z vs 0.25 = 0.00, mean VP ...` (that mix plays exactly as `ismcts@10` in every seat, so each seed's four rotations replay one game and the candidate wins exactly one); then `hybrid gate (net prior + heuristic value above 0.140): FAIL at ...`.
Run: `~/.cargo/bin/cargo run --release -p settler-bots --example prior_check -- nope; echo "exit $?"`
Expected: `usage: prior_check agree PATH GAMES | prior_check hybrid PATH SIMULATIONS SEEDS` and `exit 2`.

- [ ] **Step 4: Format, lint, full checks**

Run: `~/.cargo/bin/cargo fmt --all && ~/.cargo/bin/cargo fmt --check`
Expected: no output.
Run: `~/.cargo/bin/cargo clippy --workspace --release --all-targets -- -D warnings`
Expected: exits 0.
Run: `~/.cargo/bin/cargo test --workspace --release`
Expected: all pass.
Run: `PATH=$HOME/.cargo/bin:$PATH .venv/bin/maturin develop --release -q && .venv/bin/pytest -q -rs`
Expected: all pass; the summary lists no skipped tests.

- [ ] **Step 5: Commit**

```bash
git add bots/examples/prior_check.rs
git diff --cached | grep -nE 'sk-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{36}|password=[^[:space:]]+'
git -c user.email=sean.may101@gmail.com commit -m "perf: prior_check, the acceptance gate's agreement and hybrid diagnostics"
```

---

### Task D: Retrain net₀ on the warm-start records, the acceptance gate, throughput, and the hand-back

**Files:**
- Create: `docs/perf/net-search.md`
- Modify: `bots/src/netbot.rs` (`NET_BATCH` only, and only if Step 7 picks another batch)

**Interfaces:**
- Consumes: `alphasettler train-loop` with `--width` / `--rounds` and manifest format 2 (Task B); `prior_check agree` / `hybrid` (Task C); `net_speed speed` / `paired` (Plan 5 Task 8, unchanged); `alphasettler selfplay --net` and `alphasettler arena --nets-dir` (Plan 5); the MLP's logs in `runs/3b/` (`net-speed.log`, `selfplay-speed.log`, `warm-start.log`, `manifest.json`).
- Produces: `runs/3b-v2/` (manifest format 2, `gen-000` complete, `nets/gen-000.bin` and `nets/best.bin`); the default `NET_BATCH`; `docs/perf/net-search.md` with exactly one line `Self-play games per generation: G`, which Plan 5 Task 9 reads.

Every step but Steps 7 and 10 is run by the controller in the background. A `train-loop` command resumes where it stopped: after a Ctrl-C, a crash or a reboot, rerun the same command.

- [ ] **Step 1: Build at Task C's commit (controller runs this)**

```bash
~/.cargo/bin/cargo build --release -p settler-bots --examples
PATH=$HOME/.cargo/bin:$PATH .venv/bin/maturin develop --release -q
```

Pass: both exit 0.

- [ ] **Step 2: A fresh run directory holding the warm-start records (controller runs this)**

Generation 0's self-play step is satisfied without replaying a game. `Loop._selfplay` writes `gen-000/selfplay.json` only if it is absent and otherwise requires it to equal the plan the current settings imply (bot `ismcts`, seed start 0, 100,000 games, chunks of 500, 600 / 100 simulations, p = 0.25, config `{"max_offers_per_turn": 0, "max_turns": 1000}`: exactly `runs/3b/gen-000/selfplay.json` under the default flags); it plays a chunk only if `selfplay-local-NNNNN.jsonl.gz` is missing, and then records the step from the files it reads (games, decisions, full searches, belief resets, each chunk's sha256). Hard links give the new run the same 200 chunk files and plan at no disk cost; every loop writer renames a temp into place, so neither run can change the other's files. `runs/3b/gen-000/net.bin` (the MLP) is not linked.

```bash
test ! -e runs/3b-v2/manifest.json && test ! -e runs/3b-v2/gen-000
mkdir -p runs/3b-v2/gen-000
ln runs/3b/gen-000/selfplay.json runs/3b/gen-000/selfplay-local-*.jsonl.gz runs/3b-v2/gen-000/
ls runs/3b-v2/gen-000 | wc -l
ls -a runs/3b/gen-000 | grep -c '\.tmp-' || true
```

Pass: the first command exits 0 (a new directory; Task B's speed log may already be in `runs/3b-v2/`), `201` files linked, and `0` temp files in the source. (`test_linked_records_satisfy_a_new_runs_self_play_without_playing`, Task B, is this procedure in miniature.)

- [ ] **Step 3: Retrain net₀ (controller runs this)**

```bash
.venv/bin/alphasettler train-loop --dir runs/3b-v2 --generations 0 --threads 10 2>&1 | tee -a runs/3b-v2/train-loop.log
```

The defaults are the warm start's (100,000 games, 600 / 100 simulations, p = 0.25, max turns 1000) and network v2's (`--width 48 --rounds 2`); training is from scratch at lr 1e-3 on MPS for 2 × 23,292,888 / 1,024 = 45,494 steps, with Task 7b's value subsampling, held-out split and early stopping. Expect the self-play step to take minutes (it reads 388 MB of records) and training some hours.
Pass: exit 0 and `3 steps run`; the log shows `gen-000: selfplay` followed directly by `  100000 games, 24523371 decisions, belief resets 0` with no `selfplay-local-...: seeds` line (no chunk was played); no `notice:` line (investigate any before going on). Then:

```bash
.venv/bin/python3 -c "
import json
old = json.load(open('runs/3b/manifest.json'))['generations'][0]
new = json.load(open('runs/3b-v2/manifest.json'))['generations'][0]
t = new['train']
print(old['selfplay'] == new['selfplay'], t['width'], t['rounds'], t['device'], t['steps'], t['positions'], t['validation_games'], t['parity_max_error'] <= 1e-5, new['gate']['promoted'])
print('v2 best held out:', t['best_held_out'])
print('MLP best held out:', old['train']['best_held_out'])
"
```

Pass: the first line is `True 48 2 mps 45494 23292888 5000 True True`. The two held-out lines are recorded in Step 9 (the MLP's was policy 1.550, value 0.811 at step 36,000).

- [ ] **Step 4: The acceptance gate (controller runs this)**

```bash
~/.cargo/bin/cargo run --release -p settler-bots --example prior_check -- agree runs/3b-v2/nets/gen-000.bin 100 2>&1 | tee runs/3b-v2/agree.log
~/.cargo/bin/cargo run --release -p settler-bots --example prior_check -- hybrid runs/3b-v2/nets/gen-000.bin 100 50 2>&1 | tee runs/3b-v2/hybrid.log
```

100 fresh games give about 9,800 full searches (about 98 per game), comfortably over the spec's 3,000; seeds 0..50 at 100 simulations are the ones the MLP's 0.140 was measured on.
Pass, all of: `agree.log` ends `agreement gate (...): pass over 100 games, N searches` with N ≥ 3,000, and its `ALL` line has `heur-top1` between 0.93 and 0.97 (the tool's sanity check against the spec's 95%); `hybrid.log` ends `hybrid gate (net prior + heuristic value above 0.140): pass at R`, its first line reads win rate 0.250 if it shows 0 draws, and all four lines show `belief resets 0`.
**If the gate fails, stop here.** Do not run Steps 5–11 or any generation. Report to the user: every line of `agree.log` and `hybrid.log`, beside the MLP's numbers (top-1 agreement 9% setup settlements, 11% robber moves, 19% roads, 58% overall; hybrids 0.250 heuristic, 0.140 net prior + heuristic value, 0.200 heuristic prior + net value), and Step 3's held-out lines.

- [ ] **Step 5: Throughput (controller runs this)**

On an otherwise idle machine:

```bash
for b in 8 16 32; do for i in 1 2 3; do ~/.cargo/bin/cargo run --release -p settler-bots --example net_speed -- speed runs/3b-v2/nets/gen-000.bin 500 1000 $b; done; done 2>&1 | tee runs/3b-v2/net-speed.log
```

Pass: nine `batch N:` lines (about half an hour).

- [ ] **Step 6: The paired batch match (controller runs this)**

```bash
~/.cargo/bin/cargo run --release -p settler-bots --example net_speed -- paired runs/3b-v2/nets/gen-000.bin 200 1000 ismcts@1000 2>&1 | tee runs/3b-v2/net-batch-paired.log
```

Pass: three `batch N at 1000 simulations vs three ismcts@1000:` lines, each with `belief resets 0`, and two `paired batch` lines (one to two hours).

- [ ] **Step 7: Choose the default batch**

Plan 5 Task 8's rule: a batch B in {16, 32} qualifies if the mean of its three simulations/s readings is at least 10% above batch 8's mean **and** its paired difference from batch 8 has z > −2. The default is the qualifying batch with the higher simulations/s; if neither qualifies, it stays 8. If it changes, edit the one line in `bots/src/netbot.rs`:

```rust
/// Leaves per network call, chosen by the batch sweep in docs/perf/net-search.md.
pub const NET_BATCH: usize = 16;
```

(with the chosen value), then run `~/.cargo/bin/cargo test --workspace --release && PATH=$HOME/.cargo/bin:$PATH .venv/bin/maturin develop --release -q && .venv/bin/pytest -q -rs` and expect all to pass with none skipped.

- [ ] **Step 8: Self-play speed, the generation size, and the warm-start check (controller runs this)**

```bash
/usr/bin/time -p .venv/bin/alphasettler selfplay --net gen-000 --nets-dir runs/3b-v2/nets --games 200 --seed-start 900000000 --threads 10 --out-dir runs/3b-v2/speed 2>&1 | tee runs/3b-v2/selfplay-speed.log
```

Pass: `200 games`, `belief resets: 0` and a `real R` line. The games per generation are G = max(1000, floor(21600 × 200 / R / 100) × 100): six hours of self-play at net₀'s speed, in hundreds (Plan 5 Task 8's formula).

```bash
.venv/bin/alphasettler arena --candidate net:gen-000 --baseline ismcts@1000 --seeds 200 --no-trades --nets-dir runs/3b-v2/nets --out runs/3b-v2/warm-start-vs-ismcts.jsonl 2>&1 | tee runs/3b-v2/warm-start.log
```

Pass: a summary line and `belief resets: 0` (spec done criterion 3: recorded, not a gate; the MLP scored 0.077).

- [ ] **Step 9: Write `docs/perf/net-search.md`**

Every number is copied from the logs and manifests named here; nothing is estimated. The phrase `Self-play games per generation:` appears exactly once (Plan 5 Task 9 greps it), so the MLP section does not use it. The file:

```markdown
# Network search: throughput, the batch sweep and the acceptance gate (sub-project 3b)

Machine: <`sysctl -n machdep.cpu.brand_string`>. Commit: <`git log -1 --format=%h`, or "uncommitted on top of <hash>">.
Config: trades off, exact belief with 65,536 states, c_puct 1.5, 1,000 simulations; `speed` on one core,
`paired`, `agree`, `hybrid`, self-play and the arena on all ten.

## Network v2 (entity-structured graph net), net₀

- Net: `runs/3b-v2/nets/gen-000.bin`, sha256 <the manifest's `train.net`>, width 48, 2 rounds, 247,294
  parameters, 2,784,912 multiply-adds per position.
- Training: 45,494 steps from scratch on the 100,000-game warm start (95,000 training games, 5,000 held
  out); best held out <policy, value, aux> at step <best_step>.
- Acceptance gate 3b: <every line of runs/3b-v2/agree.log>; <every line of runs/3b-v2/hybrid.log>.
- Random v2 weights (Plan 5b Task B): <the line of runs/3b-v2/net-speed-random.log>.

| batch | simulations/s (3 runs) | mean | evaluations/s (3 runs) | ms/decision (3 runs) |
|---|---|---|---|---|
| 8 | <from runs/3b-v2/net-speed.log> | | | |
| 16 | | | | |
| 32 | | | | |

Paired match at 1,000 simulations vs three `ismcts@1000`, 200 seeds: <the three summary lines and the two
paired lines of runs/3b-v2/net-batch-paired.log>. Rule (Plan 5 Task 8): <one sentence applying it>.
`NET_BATCH` = <value>.

Self-play: 200 games of `net:gen-000` on 10 threads in <R> s real (runs/3b-v2/selfplay-speed.log).
Self-play games per generation: <G>

Warm-start check (done criterion 3, recorded): <the summary line of runs/3b-v2/warm-start.log>.

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
Why it was replaced (spec Section 2): its prior's top move matched the search's most-visited move 58% of
the time over 3,954 full searches (the heuristic prior: 95%), 9% on setup settlements, 11% on robber
moves and 19% on roads; at 100 simulations vs three `ismcts@100` over 50 seeds, heuristic prior and
value 0.250, net prior with heuristic value 0.140, heuristic prior with net value 0.200.
```

Fill every `<...>` from the named source before committing; the committed file holds numbers only, no angle brackets.

- [ ] **Step 10: Commit**

Run: `~/.cargo/bin/cargo fmt --check && ~/.cargo/bin/cargo clippy --workspace --release --all-targets -- -D warnings`
Expected: no fmt output; clippy exits 0.
Run: `grep -c '<' docs/perf/net-search.md; grep -c 'Self-play games per generation:' docs/perf/net-search.md`
Expected: `0`, then `1`.

```bash
git add docs/perf/net-search.md bots/src/netbot.rs
git diff --cached | grep -nE 'sk-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{36}|password=[^[:space:]]+'
git -c user.email=sean.may101@gmail.com commit -m "perf: network v2's acceptance gate, search throughput and the leaf-batch sweep"
```

- [ ] **Step 11: Hand back to Plan 5 Task 9**

Run Plan 5 Task 9 (`docs/superpowers/plans/2026-10-02-plan-5-network-bot.md`) with network v2, reading every `runs/3b` in its commands and pass conditions as `runs/3b-v2`: `train-loop --dir runs/3b-v2`, `--nets-dir runs/3b-v2/nets` on every arena, its logs under `runs/3b-v2/`. Its `GAMES` line reads `docs/perf/net-search.md` as written. Never rely on the default `--nets-dir` (`runs/3b/nets` holds the refused MLP nets). Generation k ≥ 1 trains from generation k − 1's file, so `--width` and `--rounds` matter only for net₀ and are not passed. `docs/bot/results-3b.md` describes network v2, links this plan, and records the MLP attempt and the gate (from `docs/perf/net-search.md`).

---

## Self-review

### Spec coverage

| Spec requirement (2026-10-04 amendment unless noted) | Task |
|---|---|
| Inputs read as entities from the encoder's existing blocks (54 × 19, 72 × 4, 19 × 8, 91 scalars); hidden block 30 unchanged; encoder unchanged | A (`GLOBAL_BASE`, `GLOBAL_WIDTH`, compile-time block asserts), B (`block`, `model.py`'s `NODE_END`/`EDGE_END`/`TILE_END` assert) |
| Static adjacency from the engine's topology: a node's tiles, edges and neighbouring nodes, an edge's two nodes, a tile's nodes | A (`graph.rs`, `net/tests/graph.rs`), B (`net_adjacency` for the twin) |
| Encoders per entity type, shared, ReLU: 19/4/8/91 → d | B (layers 0–3, `forward.rs`, `model.py`) |
| R rounds with their own weights, residual, order tile → node → edge → global, with the listed inputs | B (`forward.rs` round loop; `a_hand_computed_tiny_network` pins the order and column layout; the f64 reference) |
| Policy: per-node [node, global] → 2 (settlement, city); per-edge → road; per-tile → robber; pooled [global, three means] → 4d → 466 others | A (`policy.rs` slots), B (heads, `other_slot` scatter, both twins) |
| Value [global, three means, hidden 30] → 4d → 4; aux from the value head's 4d layer → 4; hidden only in the value head | B (layers P+5..P+7), prior invariance still tested through `NetEvaluator` (`net/tests/evaluator.rs`) |
| `d`, `R` in the header; defaults 48, 2; about 3M multiply-adds | B (`Arch::DEFAULT`; 2,784,912 counted in Task D's doc; speed check Step 6) |
| Seat-relative `StealFrom` / `ConfirmTrade` in the training targets and in `NetEvaluator`'s masking | A (`policy_id` used by `data.rs` and `masked_prior`; sampler/evaluator consistency, steal rotation and Python batch tests) |
| `OfferTrade` ids always masked | A (unchanged in `data.rs` and `masked_prior`; existing tests kept) |
| Weights format 2: magic, version, hash, `d`, `R`, layer count and shapes, f32 LE; version 1 refused clearly; errors on magic, version, hash, width, rounds, shape, short file | B (`weights.rs`, `model.py`; `a_bad_file_is_an_error_naming_what_is_wrong`, `widths_and_round_counts_are_bounded`, the Python twins) |
| Rust/PyTorch parity, 100 positions, within 1e-5 | B (`test_rust_and_pytorch_forward_passes_agree_on_random_weights` at (48, 2) and (5, 3); the export-time `parity_check`) |
| Training: the 7b value subsampling, held-out split and early stopping (Section 4) | B (unchanged code paths; their tests run on network v2) |
| Acceptance gate: top-1 agreement ≥ 0.5 on setup settlements, robber moves, roads over ≥ 3,000 full searches of fresh `ismcts` self-play | C (`prior_check agree`), D Step 4 |
| Acceptance gate: "net prior + heuristic value" hybrid beats 0.140 at 100 simulations over 50 seeds | C (`prior_check hybrid`), D Step 4 |
| Done criterion 2: throughput with the net, evaluations/s, the 8/16/32 sweep in `docs/perf/` | B Step 6 (random net), D Steps 5–7 and 9 |
| Done criterion 3: warm start vs three `ismcts@1000`, 200 seeds | D Step 8 |
| Done criterion 3b before any learned generation; stop and report on failure | D Step 4, then Step 11 hands back |
| Section 3: leaf batch rerun for network v2 (simulations/s and a paired arena match) | D Steps 5–7 |
| Generation 0 warm start reused (Section 4), net₀ by supervision from scratch | D Steps 2–3 |
| Done criterion 1: all tests pass with the `train` extra, none skipped; 0 belief resets | every task's last check; D's pass conditions |

### Placeholder scan

Searched for "TBD", "TODO", "implement later", "fill in", "similar to Task", "appropriate", "handle edge cases": none. Every code step carries the code: whole files where a file is new or rewritten, the exact old and new lines where it is edited. The one template is Task D Step 9's `docs/perf/net-search.md`, whose `<...>` marks are run-time measurements, each naming the log or manifest field it is copied from; Step 10 refuses to commit while one is left (`grep -c '<'` must print `0`).

### Type consistency

- `policy_id(a: Action, actor: PlayerId) -> usize`, `other_slot(j: usize) -> usize` and the `*_BASE` / `ENTITY_END` / `OTHER_POLICY_LEN` constants are defined in Task A and used with those names and types in `data.rs`, `evaluator.rs` (A) and `forward.rs`, `weights.rs`, `bindings/src/train.rs` (B).
- `graph()` returns `&'static Graph`; `Graph`'s fields are `node_tiles`, `node_edges`, `node_neighbours` (British spelling, as in the spec; the engine's `Topology` field is `node_neighbors`), `edge_nodes`, `tile_nodes`; `Links::{count, of, lists}` and the field `scale` are used as defined. `net_adjacency()` returns the same five keys, and `model.py` registers buffers of the same names.
- `masked_prior(logits, legal, actor)` gains its third argument in Task A; every caller (`NetEvaluator::evaluate`, `net/tests/evaluator.rs`, `net/tests/data.rs`) passes it.
- `Arch { width, rounds }`, `Arch::DEFAULT`, `check`, `layer_count`, `header_len`, `round_layer`, `head_layer`, `shapes` and `FIXED_HEADER_LEN` are defined in Task B's `weights.rs` and used with those names by `forward.rs`, `net/tests/weights.rs`, `bots/tests/netbot.rs`, `bindings/src/train.rs`, `prior_check.rs` (C) and `net_speed.rs` (unchanged). The error strings (`format version {v}, expected 2`, `width {d} is out of range 1..=1024`, `rounds {r} is out of range 1..=16`, `{count} layers, expected {n} for {r} rounds`, `do not fit network v2`, `truncated: ...`) are identical in Rust and Python and are what the tests match.
- Python: `Net(width, rounds)` with `.width` / `.rounds`, `TrainConfig.width` / `.rounds`, `LoopSettings.width` / `.rounds`, the summary keys `"width"` / `"rounds"`, the manifest keys `train.width` / `train.rounds`, and the CLI's `args.width` / `args.rounds` agree across `model.py`, `train.py`, `loop.py`, `cli.py` and the tests. The binding constants used by `model.py` (`NET_NODE_WIDTH`, `NET_EDGE_WIDTH`, `NET_TILE_WIDTH`, `NET_GLOBAL_WIDTH`, `NET_SETTLEMENT_BASE`, `NET_MAX_WIDTH`, `NET_MAX_ROUNDS`, `NET_DEFAULT_WIDTH`, `NET_DEFAULT_ROUNDS`) are all registered in Task B Step 9.
- The `prior_check` output lines Task D's pass conditions read (`agreement gate (...): pass|FAIL over G games, N searches`, `hybrid gate (...): pass|FAIL at R`) are the ones Task C prints.

### Review Focus

Each of the five lines above has its test in Task B (items 1–5); item 3's test is the procedure Task D Step 2 runs. Checked and not added: steals and trade confirmations by seats other than 0 (spec-implied, tested in Task A); a batch mixing positions (batch independence, Task B); an odd number of positions (every row is independent of `n`).
