# Plan 1: Rust Engine — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A fast, fully tested Rust crate (`settler-engine`) implementing 4-player base Catan with domestic trading, hidden information, forced chance outcomes, and an event log.

**Architecture:** Board topology is generated once into a static (never stored in game state). `State` is a fixed-size `Copy` struct; `legal_actions` and `apply_with` dispatch on `Phase` into one small module per rule area (`rules/*.rs`). Every random outcome comes from a seeded stream or can be forced via `Chance`, and every state change can be reported to an `EventSink` (a zero-cost no-op in search). A thin `Game` wrapper adds legality checking, an event log, and per-player redaction.

**Tech Stack:** Rust (edition 2021, MSRV 1.80), std only at runtime; `proptest` and `criterion` as dev-dependencies.

**Spec:** `docs/superpowers/specs/2026-09-30-engine-and-harness-design.md`

This is Plan 1 of 3. Plan 2 (PyO3 bindings, baseline bots, arena, stats, CLI) and Plan 3 (Catanatron oracle, differential tests, side-by-side perf) are written after this plan lands, against the real interfaces built here.

## Global Constraints

- Rust edition 2021, `rust-version = "1.80"` (needed for `std::sync::LazyLock`).
- Engine crate has **zero runtime dependencies**. Dev-dependencies are pinned exactly: `proptest = "=1.5.0"`, `criterion = "=0.5.1"`.
- `State` is `Copy`, holds no heap data, and `size_of::<State>() <= 512`.
- Topology (19 tiles, 54 nodes, 72 edges, adjacency) lives in a `LazyLock` static, never in `State`. (The spec says "compile-time constant tables"; a once-initialized static gives the same runtime properties.)
- Action space is fixed: `ACTION_SPACE_SIZE = 665`, encoding exactly as in Task 4. Never renumber.
- `GameConfig` defaults: `vp_to_win = 10`, `discard_limit = 7`, `max_offers_per_turn = 3`, `max_trade_cards = 2` (must be 1 or 2), `max_turns = 1000`, `catanatron_compat = false`.
- Dice depend only on `(seed, turn)`. Dev deck is shuffled once from the seed. Steals and compat-mode discards use their own streams.
- The engine is written from the Catan rulebook. Never copy or port Catanatron source (GPL-3.0).
- Friendly robber is **deferred**: it's in the spec's config list, but its semantics are only added once Colonist's ranked rules are verified (spec Decisions table).
- After each task: run the task's tests, secret-scan the staged diff, commit with `sean.may101@gmail.com`, no AI attribution, never `--no-verify`, then `git push`.

## Review Focus

1. **Impossible forced chance outcomes** (forced steal of a resource the victim lacks, forced dev card not left in the deck, a `Chance` kind that doesn't match the action) must panic with a clear message, never silently corrupt state. Tests: Task 6 (`forced_steal_of_missing_resource_panics`, `mismatched_chance_panics`), Task 9 (`forced_dev_card_missing_panics`).
2. **Road Building with no legal placement or no road pieces left** must return to `Main`, not leave the game stuck in `RoadBuilding`. Test: Task 9 (`road_building_with_no_legal_road_returns_to_main`).
3. **Several players over the discard limit on a 7** discard in seat order, and the robber moves only after all of them finish. Test: Task 6 (`multiple_discarders_go_in_seat_order`).
4. **Illegal actions and actions after game over through `Game`** return `Err` and leave state unchanged. Tests: Task 12 (`illegal_action_is_rejected_without_changing_state`, `actions_after_game_over_are_rejected`).
5. **Robber on a tile with only the mover's buildings, or only empty-handed victims**, skips the Steal phase. Tests: Task 6 (`no_steal_when_only_own_buildings`, `empty_handed_players_are_not_victims`).

---

## File Structure

```
Cargo.toml                         workspace root
engine/Cargo.toml                  crate `settler-engine`, lib name `settler_engine`
engine/src/lib.rs                  module list + re-exports
engine/src/types.rs                Resource, DevCard, Hand, costs, limits, bit iterators
engine/src/topology.rs             static board graph
engine/src/rng.rs                  SplitMix64 RNG, stream mixing, dice_for
engine/src/board.rs                per-game board (terrain, numbers, ports), random generation
engine/src/config.rs               GameConfig
engine/src/action.rs               Action enum, fixed encoding, trade bundles
engine/src/events.rs               Event enum + redaction
engine/src/state.rs                State, PlayerState, Phase, PendingTrade, VP helpers
engine/src/legal.rs                legal_actions dispatcher
engine/src/apply.rs                apply / apply_with dispatcher, Chance, EventSink
engine/src/rules/mod.rs            rule modules
engine/src/rules/setup.rs          snake-order setup
engine/src/rules/roll.rs           dice, production, bank shortage, sevens, discards
engine/src/rules/robber.rs         robber movement and stealing
engine/src/rules/build.rs          roads, settlements, cities, end turn
engine/src/rules/awards.rs         longest road, largest army, win check
engine/src/rules/dev.rs            development cards
engine/src/rules/maritime.rs       bank/port trades
engine/src/rules/trade.rs          domestic trade protocol
engine/src/observation.rs          per-player Observation
engine/src/game.rs                 Game wrapper: legality check, event log
engine/src/sim.rs                  random playout helper
engine/tests/common/mod.rs         test helpers
engine/tests/*.rs                  one integration test file per task
engine/benches/engine.rs           criterion benchmarks
engine/examples/throughput.rs      games/sec measurement
docs/perf/engine-baseline.md       recorded performance numbers
```

---

### Task 1: Toolchain, workspace, and core types

**Files:**
- Create: `Cargo.toml`, `engine/Cargo.toml`, `engine/src/lib.rs`, `engine/src/types.rs`
- Test: `engine/tests/types.rs`

**Interfaces:**
- Consumes: nothing.
- Produces: `settler_engine::types::{NUM_PLAYERS, NUM_RESOURCES, PlayerId, Hand, Resource, DevCard, DEV_DECK_COUNTS, DEV_DECK_SIZE, BANK_PER_RESOURCE, MAX_SETTLEMENTS, MAX_CITIES, MAX_ROADS, ROAD_COST, SETTLEMENT_COST, CITY_COST, DEV_COST, covers, hand_sub, hand_add, hand_total, bits64, bits128}`. `Resource::{ALL, index, from_index}`, `DevCard::{ALL, index, from_index}`.

- [ ] **Step 1: Install Rust (skip if `cargo --version` works)**

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --component rustfmt,clippy
source "$HOME/.cargo/env"
cargo --version
```

Expected: prints `cargo 1.8x` or newer (must be ≥ 1.80).

- [ ] **Step 2: Create the workspace files**

`Cargo.toml`:

```toml
[workspace]
members = ["engine"]
resolver = "2"

[profile.release]
lto = "fat"
codegen-units = 1

[profile.bench]
inherits = "release"
```

`engine/Cargo.toml`:

```toml
[package]
name = "settler-engine"
version = "0.1.0"
edition = "2021"
rust-version = "1.80"

[lib]
name = "settler_engine"

# Index loops over fixed 4-player / 5-resource arrays read better than iterator chains here.
[lints.clippy]
needless_range_loop = "allow"
```

`engine/src/lib.rs`:

```rust
//! AlphaSettler's Catan rules engine.

pub mod types;
```

- [ ] **Step 3: Write the failing test**

`engine/tests/types.rs`:

```rust
use settler_engine::types::*;

#[test]
fn resource_index_roundtrip() {
    for (i, r) in Resource::ALL.iter().enumerate() {
        assert_eq!(r.index(), i);
        assert_eq!(Resource::from_index(i), *r);
    }
}

#[test]
fn dev_card_index_roundtrip() {
    for (i, c) in DevCard::ALL.iter().enumerate() {
        assert_eq!(c.index(), i);
        assert_eq!(DevCard::from_index(i), *c);
    }
}

#[test]
fn dev_deck_has_25_cards() {
    let total: usize = DEV_DECK_COUNTS.iter().map(|&c| c as usize).sum();
    assert_eq!(total, DEV_DECK_SIZE);
    assert_eq!(DEV_DECK_COUNTS[DevCard::Knight.index()], 14);
    assert_eq!(DEV_DECK_COUNTS[DevCard::VictoryPoint.index()], 5);
}

#[test]
fn costs_match_rulebook() {
    assert_eq!(ROAD_COST, [1, 1, 0, 0, 0]);
    assert_eq!(SETTLEMENT_COST, [1, 1, 1, 1, 0]);
    assert_eq!(CITY_COST, [0, 0, 0, 2, 3]);
    assert_eq!(DEV_COST, [0, 0, 1, 1, 1]);
}

#[test]
fn hand_arithmetic() {
    let mut h: Hand = [2, 1, 0, 0, 0];
    assert!(covers(&h, &ROAD_COST));
    hand_sub(&mut h, &ROAD_COST);
    assert_eq!(h, [1, 0, 0, 0, 0]);
    assert!(!covers(&h, &ROAD_COST));
    hand_add(&mut h, &ROAD_COST);
    assert_eq!(h, [2, 1, 0, 0, 0]);
    assert_eq!(hand_total(&h), 3);
}

#[test]
fn bit_iterators() {
    assert_eq!(bits64(0b1010_0001).collect::<Vec<_>>(), vec![0, 5, 7]);
    assert_eq!(bits128((1u128 << 100) | 1).collect::<Vec<_>>(), vec![0, 100]);
    assert_eq!(bits64(0).count(), 0);
}
```

- [ ] **Step 4: Run test to verify it fails**

Run: `cargo test -p settler-engine --test types`
Expected: FAIL to compile — `unresolved import` / `cannot find type Resource`.

- [ ] **Step 5: Write the implementation**

`engine/src/types.rs`:

```rust
//! Core value types shared across the engine.

pub const NUM_PLAYERS: usize = 4;
pub const NUM_RESOURCES: usize = 5;

pub type PlayerId = u8;
/// Resource counts indexed by `Resource::index()`.
pub type Hand = [u8; NUM_RESOURCES];

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(u8)]
pub enum Resource {
    Wood = 0,
    Brick = 1,
    Sheep = 2,
    Wheat = 3,
    Ore = 4,
}

impl Resource {
    pub const ALL: [Resource; NUM_RESOURCES] = [
        Resource::Wood,
        Resource::Brick,
        Resource::Sheep,
        Resource::Wheat,
        Resource::Ore,
    ];

    #[inline]
    pub fn index(self) -> usize {
        self as usize
    }

    #[inline]
    pub fn from_index(i: usize) -> Resource {
        Self::ALL[i]
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum DevCard {
    Knight = 0,
    VictoryPoint = 1,
    RoadBuilding = 2,
    YearOfPlenty = 3,
    Monopoly = 4,
}

impl DevCard {
    pub const ALL: [DevCard; 5] = [
        DevCard::Knight,
        DevCard::VictoryPoint,
        DevCard::RoadBuilding,
        DevCard::YearOfPlenty,
        DevCard::Monopoly,
    ];

    #[inline]
    pub fn index(self) -> usize {
        self as usize
    }

    #[inline]
    pub fn from_index(i: usize) -> DevCard {
        Self::ALL[i]
    }
}

/// Cards per type, indexed by `DevCard::index()`.
pub const DEV_DECK_COUNTS: [u8; 5] = [14, 5, 2, 2, 2];
pub const DEV_DECK_SIZE: usize = 25;
pub const BANK_PER_RESOURCE: u8 = 19;

pub const MAX_SETTLEMENTS: u32 = 5;
pub const MAX_CITIES: u32 = 4;
pub const MAX_ROADS: u32 = 15;

pub const ROAD_COST: Hand = [1, 1, 0, 0, 0];
pub const SETTLEMENT_COST: Hand = [1, 1, 1, 1, 0];
pub const CITY_COST: Hand = [0, 0, 0, 2, 3];
pub const DEV_COST: Hand = [0, 0, 1, 1, 1];

#[inline]
pub fn covers(hand: &Hand, cost: &Hand) -> bool {
    (0..NUM_RESOURCES).all(|i| hand[i] >= cost[i])
}

#[inline]
pub fn hand_sub(hand: &mut Hand, x: &Hand) {
    for i in 0..NUM_RESOURCES {
        hand[i] -= x[i];
    }
}

#[inline]
pub fn hand_add(hand: &mut Hand, x: &Hand) {
    for i in 0..NUM_RESOURCES {
        hand[i] += x[i];
    }
}

#[inline]
pub fn hand_total(hand: &Hand) -> u32 {
    hand.iter().map(|&c| c as u32).sum()
}

/// Indices of set bits, lowest first.
pub fn bits64(mut m: u64) -> impl Iterator<Item = u8> {
    std::iter::from_fn(move || {
        if m == 0 {
            return None;
        }
        let i = m.trailing_zeros() as u8;
        m &= m - 1;
        Some(i)
    })
}

/// Indices of set bits, lowest first.
pub fn bits128(mut m: u128) -> impl Iterator<Item = u8> {
    std::iter::from_fn(move || {
        if m == 0 {
            return None;
        }
        let i = m.trailing_zeros() as u8;
        m &= m - 1;
        Some(i)
    })
}
```

- [ ] **Step 6: Run test to verify it passes**

Run: `cargo test -p settler-engine --test types`
Expected: PASS, 6 tests.

- [ ] **Step 7: Commit**

```bash
git add Cargo.toml Cargo.lock engine
git diff --cached | grep -nE 'sk-|AKIA|ghp_|password=' && echo "SECRET FOUND - abort" || git commit -m "engine: workspace and core types"
git push
```

---

### Task 2: Board topology

**Files:**
- Create: `engine/src/topology.rs`
- Modify: `engine/src/lib.rs`
- Test: `engine/tests/topology.rs`

**Interfaces:**
- Consumes: nothing.
- Produces: `settler_engine::topology::{NUM_TILES (19), NUM_NODES (54), NUM_EDGES (72), Topology, topo() -> &'static Topology}`. `Topology` fields: `tile_coords: [(i8, i8); 19]` (axial q, r), `tile_nodes: [[u8; 6]; 19]` (clockwise from top), `tile_node_mask: [u64; 19]`, `tile_neighbors: Vec<Vec<u8>>`, `node_tiles: Vec<Vec<u8>>`, `node_neighbors: Vec<Vec<u8>>`, `node_neighbor_mask: [u64; 54]`, `node_edges: Vec<Vec<u8>>`, `node_edge_mask: [u128; 54]`, `edge_nodes: [(u8, u8); 72]` (low id first), `coastal_edges: Vec<u8>` (30 edges in cyclic order).

- [ ] **Step 1: Write the failing test**

`engine/tests/topology.rs`:

```rust
use settler_engine::topology::*;

#[test]
fn counts() {
    let t = topo();
    assert_eq!(t.tile_nodes.len(), NUM_TILES);
    assert_eq!(t.edge_nodes.len(), NUM_EDGES);
    assert_eq!(t.node_tiles.len(), NUM_NODES);
    assert_eq!((NUM_TILES, NUM_NODES, NUM_EDGES), (19, 54, 72));
}

#[test]
fn every_tile_has_six_distinct_nodes() {
    let t = topo();
    for (i, nodes) in t.tile_nodes.iter().enumerate() {
        let mut v = nodes.to_vec();
        v.sort();
        v.dedup();
        assert_eq!(v.len(), 6, "tile {i}");
        assert_eq!(t.tile_node_mask[i].count_ones(), 6, "tile {i}");
    }
}

#[test]
fn node_tile_counts_and_degrees() {
    let t = topo();
    let tile_slots: usize = t.node_tiles.iter().map(|v| v.len()).sum();
    assert_eq!(tile_slots, NUM_TILES * 6);
    let degree_sum: usize = t.node_neighbors.iter().map(|v| v.len()).sum();
    assert_eq!(degree_sum, 2 * NUM_EDGES);
    for n in 0..NUM_NODES {
        assert!((1..=3).contains(&t.node_tiles[n].len()), "node {n}");
        assert!((2..=3).contains(&t.node_neighbors[n].len()), "node {n}");
        assert_eq!(t.node_neighbors[n].len(), t.node_edges[n].len());
    }
}

#[test]
fn masks_agree_with_lists() {
    let t = topo();
    for n in 0..NUM_NODES {
        let nm: u64 = t.node_neighbors[n].iter().map(|&x| 1u64 << x).sum();
        assert_eq!(nm, t.node_neighbor_mask[n]);
        let em: u128 = t.node_edges[n].iter().map(|&e| 1u128 << e).sum();
        assert_eq!(em, t.node_edge_mask[n]);
    }
    for (e, &(a, b)) in t.edge_nodes.iter().enumerate() {
        assert!(a < b);
        assert!(t.node_edges[a as usize].contains(&(e as u8)));
        assert!(t.node_edges[b as usize].contains(&(e as u8)));
    }
}

#[test]
fn tile_neighbors() {
    let t = topo();
    assert_eq!(t.tile_coords[9], (0, 0));
    assert_eq!(t.tile_neighbors[9].len(), 6);
    assert_eq!(t.tile_coords[0], (0, -2));
    assert_eq!(t.tile_neighbors[0].len(), 3);
    for (i, ns) in t.tile_neighbors.iter().enumerate() {
        for &j in ns {
            assert!(t.tile_neighbors[j as usize].contains(&(i as u8)));
            let shared = (t.tile_node_mask[i] & t.tile_node_mask[j as usize]).count_ones();
            assert_eq!(shared, 2, "neighboring tiles {i},{j} share an edge");
        }
    }
}

#[test]
fn coastline_is_a_30_edge_cycle() {
    let t = topo();
    let c = &t.coastal_edges;
    assert_eq!(c.len(), 30);
    let mut sorted = c.clone();
    sorted.sort();
    sorted.dedup();
    assert_eq!(sorted.len(), 30);
    for i in 0..30 {
        let (a, b) = t.edge_nodes[c[i] as usize];
        let (x, y) = t.edge_nodes[c[(i + 1) % 30] as usize];
        assert!(a == x || a == y || b == x || b == y, "coastal edges {i} and {} not adjacent", i + 1);
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p settler-engine --test topology`
Expected: FAIL to compile — `could not find topology`.

- [ ] **Step 3: Write the implementation**

Add to `engine/src/lib.rs`:

```rust
pub mod topology;
```

`engine/src/topology.rs`:

```rust
//! The fixed graph of the standard 19-tile board, built once.
//!
//! Tiles use axial coordinates (q, r) in a radius-2 hexagon, ordered by r then q.
//! Pointy-top corners are placed on an integer lattice: a tile's center is
//! (2q + r, 3r) and its corners are offset by `CORNER_OFFSETS`. Nodes are numbered
//! top-to-bottom, left-to-right; edges are sorted by (low node, high node).

use std::collections::{BTreeMap, BTreeSet};
use std::sync::LazyLock;

pub const NUM_TILES: usize = 19;
pub const NUM_NODES: usize = 54;
pub const NUM_EDGES: usize = 72;

/// Clockwise from the top corner (y grows downward).
const CORNER_OFFSETS: [(i32, i32); 6] = [(0, -2), (1, -1), (1, 1), (0, 2), (-1, 1), (-1, -1)];
const AXIAL_DIRS: [(i32, i32); 6] = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)];

pub struct Topology {
    pub tile_coords: [(i8, i8); NUM_TILES],
    pub tile_nodes: [[u8; 6]; NUM_TILES],
    pub tile_node_mask: [u64; NUM_TILES],
    pub tile_neighbors: Vec<Vec<u8>>,
    pub node_tiles: Vec<Vec<u8>>,
    pub node_neighbors: Vec<Vec<u8>>,
    pub node_neighbor_mask: [u64; NUM_NODES],
    pub node_edges: Vec<Vec<u8>>,
    pub node_edge_mask: [u128; NUM_NODES],
    pub edge_nodes: [(u8, u8); NUM_EDGES],
    pub coastal_edges: Vec<u8>,
}

static TOPOLOGY: LazyLock<Topology> = LazyLock::new(Topology::build);

#[inline]
pub fn topo() -> &'static Topology {
    &TOPOLOGY
}

fn corner((q, r): (i32, i32), i: usize) -> (i32, i32) {
    let (dx, dy) = CORNER_OFFSETS[i];
    (2 * q + r + dx, 3 * r + dy)
}

impl Topology {
    fn build() -> Topology {
        let mut coords: Vec<(i32, i32)> = Vec::new();
        for r in -2i32..=2 {
            for q in -2i32..=2 {
                if (q + r).abs() <= 2 {
                    coords.push((q, r));
                }
            }
        }
        assert_eq!(coords.len(), NUM_TILES);

        let mut points: Vec<(i32, i32)> = coords
            .iter()
            .flat_map(|&c| (0..6).map(move |i| corner(c, i)))
            .collect();
        points.sort_by_key(|&(x, y)| (y, x));
        points.dedup();
        assert_eq!(points.len(), NUM_NODES);
        let node_of: BTreeMap<(i32, i32), u8> =
            points.iter().enumerate().map(|(i, &p)| (p, i as u8)).collect();

        let mut tile_coords = [(0i8, 0i8); NUM_TILES];
        let mut tile_nodes = [[0u8; 6]; NUM_TILES];
        let mut tile_node_mask = [0u64; NUM_TILES];
        for (t, &c) in coords.iter().enumerate() {
            tile_coords[t] = (c.0 as i8, c.1 as i8);
            for i in 0..6 {
                let n = node_of[&corner(c, i)];
                tile_nodes[t][i] = n;
                tile_node_mask[t] |= 1u64 << n;
            }
        }

        let mut edge_set: BTreeSet<(u8, u8)> = BTreeSet::new();
        for nodes in &tile_nodes {
            for i in 0..6 {
                let (a, b) = (nodes[i], nodes[(i + 1) % 6]);
                edge_set.insert((a.min(b), a.max(b)));
            }
        }
        assert_eq!(edge_set.len(), NUM_EDGES);
        let edge_list: Vec<(u8, u8)> = edge_set.into_iter().collect();
        let mut edge_nodes = [(0u8, 0u8); NUM_EDGES];
        edge_nodes.copy_from_slice(&edge_list);

        let mut node_tiles: Vec<Vec<u8>> = vec![Vec::new(); NUM_NODES];
        for (t, nodes) in tile_nodes.iter().enumerate() {
            for &n in nodes {
                node_tiles[n as usize].push(t as u8);
            }
        }

        let mut node_neighbors: Vec<Vec<u8>> = vec![Vec::new(); NUM_NODES];
        let mut node_edges: Vec<Vec<u8>> = vec![Vec::new(); NUM_NODES];
        let mut node_neighbor_mask = [0u64; NUM_NODES];
        let mut node_edge_mask = [0u128; NUM_NODES];
        for (e, &(a, b)) in edge_nodes.iter().enumerate() {
            node_neighbors[a as usize].push(b);
            node_neighbors[b as usize].push(a);
            node_edges[a as usize].push(e as u8);
            node_edges[b as usize].push(e as u8);
            node_neighbor_mask[a as usize] |= 1u64 << b;
            node_neighbor_mask[b as usize] |= 1u64 << a;
            node_edge_mask[a as usize] |= 1u128 << e;
            node_edge_mask[b as usize] |= 1u128 << e;
        }

        let tile_neighbors: Vec<Vec<u8>> = coords
            .iter()
            .map(|&(q, r)| {
                AXIAL_DIRS
                    .iter()
                    .filter_map(|&(dq, dr)| {
                        coords.iter().position(|&c| c == (q + dq, r + dr)).map(|i| i as u8)
                    })
                    .collect()
            })
            .collect();

        // Coastal edges border exactly one tile; walk them into a cycle.
        let mut edge_tile_count = [0u8; NUM_EDGES];
        for nodes in &tile_nodes {
            for i in 0..6 {
                let (a, b) = (nodes[i], nodes[(i + 1) % 6]);
                let e = edge_list.binary_search(&(a.min(b), a.max(b))).unwrap();
                edge_tile_count[e] += 1;
            }
        }
        let is_coastal = |e: u8| edge_tile_count[e as usize] == 1;
        let start = (0..NUM_EDGES as u8).find(|&e| is_coastal(e)).unwrap();
        let mut coastal_edges = vec![start];
        let mut prev = start;
        let mut node = edge_nodes[start as usize].1;
        loop {
            let next = *node_edges[node as usize]
                .iter()
                .find(|&&e| e != prev && is_coastal(e))
                .unwrap();
            if next == start {
                break;
            }
            coastal_edges.push(next);
            let (a, b) = edge_nodes[next as usize];
            node = if a == node { b } else { a };
            prev = next;
        }
        assert_eq!(coastal_edges.len(), 30);

        Topology {
            tile_coords,
            tile_nodes,
            tile_node_mask,
            tile_neighbors,
            node_tiles,
            node_neighbors,
            node_neighbor_mask,
            node_edges,
            node_edge_mask,
            edge_nodes,
            coastal_edges,
        }
    }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p settler-engine --test topology`
Expected: PASS, 6 tests.

- [ ] **Step 5: Commit**

```bash
git add engine
git diff --cached | grep -nE 'sk-|AKIA|ghp_|password=' && echo "SECRET FOUND - abort" || git commit -m "engine: board topology"
git push
```

---

### Task 3: RNG, game config, and board generation

**Files:**
- Create: `engine/src/rng.rs`, `engine/src/config.rs`, `engine/src/board.rs`
- Modify: `engine/src/lib.rs`
- Test: `engine/tests/board.rs`

**Interfaces:**
- Consumes: `topology::{topo, NUM_TILES, NUM_NODES}`, `types::Resource`.
- Produces:
  - `rng::Rng` (`Copy`, `PartialEq`): `Rng::new(seed: u64)`, `next_u64(&mut self) -> u64`, `below(&mut self, n: u32) -> u32`, `shuffle<T>(&mut self, xs: &mut [T])`. `rng::mix(a: u64, b: u64) -> u64`. `rng::dice_for(seed: u64, turn: u32) -> (u8, u8)`.
  - `config::GameConfig { vp_to_win: u8, discard_limit: u8, max_offers_per_turn: u8, max_trade_cards: u8, max_turns: u32, catanatron_compat: bool }`: `Copy`, `Default`, `validate(&self) -> Result<(), String>`.
  - `board::PortKind { Generic, Specific(Resource) }`. `board::Board { tile_resource: [Option<Resource>; 19], tile_number: [u8; 19], ports: [(u8, PortKind); 9], generic_port_nodes: u64, specific_port_nodes: [u64; 5] }` with `Board::new(tile_resource, tile_number, ports)`, `Board::random(rng: &mut Rng)`, `desert(&self) -> u8`, `maritime_rate(&self, buildings: u64, r: Resource) -> u8`. Constants `STANDARD_TERRAIN`, `STANDARD_NUMBERS`, `STANDARD_PORT_KINDS`, `PORT_SLOTS`.

- [ ] **Step 1: Write the failing test**

`engine/tests/board.rs`:

```rust
use settler_engine::board::*;
use settler_engine::config::GameConfig;
use settler_engine::rng::*;
use settler_engine::topology::topo;
use settler_engine::types::Resource;

#[test]
fn rng_is_deterministic_and_bounded() {
    let mut a = Rng::new(5);
    let mut b = Rng::new(5);
    for _ in 0..100 {
        assert_eq!(a.next_u64(), b.next_u64());
    }
    let mut r = Rng::new(1);
    let mut seen = [false; 6];
    for _ in 0..1000 {
        let x = r.below(6);
        assert!(x < 6);
        seen[x as usize] = true;
    }
    assert!(seen.iter().all(|&s| s));
}

#[test]
fn shuffle_is_a_permutation() {
    let mut xs: Vec<u32> = (0..50).collect();
    Rng::new(3).shuffle(&mut xs);
    let mut sorted = xs.clone();
    sorted.sort();
    assert_eq!(sorted, (0..50).collect::<Vec<_>>());
    assert_ne!(xs, sorted);
}

#[test]
fn dice_depend_only_on_seed_and_turn() {
    assert_eq!(dice_for(9, 3), dice_for(9, 3));
    let rolls: Vec<_> = (0..50).map(|t| dice_for(9, t)).collect();
    assert!(rolls.iter().all(|&(a, b)| (1..=6).contains(&a) && (1..=6).contains(&b)));
    assert!(rolls.windows(2).any(|w| w[0] != w[1]));
    assert_ne!(
        (0..20).map(|t| dice_for(1, t)).collect::<Vec<_>>(),
        (0..20).map(|t| dice_for(2, t)).collect::<Vec<_>>()
    );
}

#[test]
fn default_config_matches_spec() {
    let c = GameConfig::default();
    assert_eq!(
        (c.vp_to_win, c.discard_limit, c.max_offers_per_turn, c.max_trade_cards, c.max_turns, c.catanatron_compat),
        (10, 7, 3, 2, 1000, false)
    );
    assert!(c.validate().is_ok());
    assert!(GameConfig { max_trade_cards: 3, ..c }.validate().is_err());
    assert!(GameConfig { max_trade_cards: 0, ..c }.validate().is_err());
}

#[test]
fn standard_board_composition() {
    for seed in 0..50 {
        let b = Board::random(&mut Rng::new(seed));
        let count = |r: Option<Resource>| b.tile_resource.iter().filter(|&&x| x == r).count();
        assert_eq!(count(Some(Resource::Wood)), 4);
        assert_eq!(count(Some(Resource::Brick)), 3);
        assert_eq!(count(Some(Resource::Sheep)), 4);
        assert_eq!(count(Some(Resource::Wheat)), 4);
        assert_eq!(count(Some(Resource::Ore)), 3);
        assert_eq!(count(None), 1);
        let d = b.desert() as usize;
        assert_eq!(b.tile_number[d], 0);
        let mut nums: Vec<u8> = (0..19).filter(|&i| i != d).map(|i| b.tile_number[i]).collect();
        nums.sort();
        assert_eq!(nums, STANDARD_NUMBERS.to_vec());
    }
}

#[test]
fn no_adjacent_red_numbers() {
    let t = topo();
    for seed in 0..200 {
        let b = Board::random(&mut Rng::new(seed));
        let red = |n: u8| n == 6 || n == 8;
        for i in 0..19 {
            if red(b.tile_number[i]) {
                for &j in &t.tile_neighbors[i] {
                    assert!(!red(b.tile_number[j as usize]), "seed {seed}: tiles {i},{j}");
                }
            }
        }
    }
}

#[test]
fn ports_are_standard() {
    let t = topo();
    let b = Board::random(&mut Rng::new(11));
    let mut edges: Vec<u8> = b.ports.iter().map(|p| p.0).collect();
    edges.sort();
    edges.dedup();
    assert_eq!(edges.len(), 9);
    for &(e, _) in &b.ports {
        assert!(t.coastal_edges.contains(&e));
    }
    assert_eq!(b.ports.iter().filter(|p| p.1 == PortKind::Generic).count(), 4);
    for r in Resource::ALL {
        assert_eq!(b.ports.iter().filter(|p| p.1 == PortKind::Specific(r)).count(), 1);
    }
}

#[test]
fn maritime_rates_follow_ports() {
    let t = topo();
    let b = Board::random(&mut Rng::new(4));
    assert_eq!(b.maritime_rate(0, Resource::Wood), 4);
    let (ge, _) = *b.ports.iter().find(|p| p.1 == PortKind::Generic).unwrap();
    let gnode = 1u64 << t.edge_nodes[ge as usize].0;
    for r in Resource::ALL {
        assert_eq!(b.maritime_rate(gnode, r), 3);
    }
    let (se, kind) = *b.ports.iter().find(|p| p.1 != PortKind::Generic).unwrap();
    let PortKind::Specific(sr) = kind else { unreachable!() };
    let snode = 1u64 << t.edge_nodes[se as usize].1;
    assert_eq!(b.maritime_rate(snode, sr), 2);
    let other = Resource::ALL.into_iter().find(|&r| r != sr).unwrap();
    assert_eq!(b.maritime_rate(snode, other), 4);
}

#[test]
fn boards_are_seed_deterministic() {
    assert_eq!(Board::random(&mut Rng::new(8)), Board::random(&mut Rng::new(8)));
    let first = Board::random(&mut Rng::new(0));
    assert!((1..10).any(|s| Board::random(&mut Rng::new(s)) != first));
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p settler-engine --test board`
Expected: FAIL to compile — `could not find board` / `rng` / `config`.

- [ ] **Step 3: Write the implementation**

Add to `engine/src/lib.rs`:

```rust
pub mod board;
pub mod config;
pub mod rng;
```

`engine/src/rng.rs`:

```rust
//! Small, fast, seedable randomness (SplitMix64). Not cryptographic.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rng {
    state: u64,
}

impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng { state: seed }
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `0..n`. `n` must be > 0.
    #[inline]
    pub fn below(&mut self, n: u32) -> u32 {
        (((self.next_u64() >> 32) * n as u64) >> 32) as u32
    }

    pub fn shuffle<T>(&mut self, xs: &mut [T]) {
        for i in (1..xs.len()).rev() {
            let j = self.below(i as u32 + 1) as usize;
            xs.swap(i, j);
        }
    }
}

/// Derive an independent stream seed from `a` and a salt/index `b`.
pub fn mix(a: u64, b: u64) -> u64 {
    Rng::new(a ^ b.wrapping_mul(0x9E37_79B9_7F4A_7C15)).next_u64()
}

const DICE_SALT: u64 = 0xD1CE;

/// The dice for `turn` of the game seeded `seed`. Independent of every bot decision.
pub fn dice_for(seed: u64, turn: u32) -> (u8, u8) {
    let mut r = Rng::new(mix(seed ^ DICE_SALT, turn as u64));
    (1 + r.below(6) as u8, 1 + r.below(6) as u8)
}
```

`engine/src/config.rs`:

```rust
//! Rule variants and limits for one game.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GameConfig {
    pub vp_to_win: u8,
    /// Players holding more than this many cards discard half on a 7.
    pub discard_limit: u8,
    /// Domestic trade offers allowed per turn (0 disables domestic trading).
    pub max_offers_per_turn: u8,
    /// Max cards on each side of a domestic offer (1 or 2).
    pub max_trade_cards: u8,
    /// The game ends with no winner when this many turns have been played.
    pub max_turns: u32,
    /// Discards on a 7 are random instead of chosen, as Catanatron does.
    pub catanatron_compat: bool,
}

impl Default for GameConfig {
    fn default() -> Self {
        GameConfig {
            vp_to_win: 10,
            discard_limit: 7,
            max_offers_per_turn: 3,
            max_trade_cards: 2,
            max_turns: 1000,
            catanatron_compat: false,
        }
    }
}

impl GameConfig {
    pub fn validate(&self) -> Result<(), String> {
        if !(1..=2).contains(&self.max_trade_cards) {
            return Err(format!("max_trade_cards must be 1 or 2, got {}", self.max_trade_cards));
        }
        if self.vp_to_win < 3 {
            return Err(format!("vp_to_win must be at least 3, got {}", self.vp_to_win));
        }
        if self.max_turns == 0 {
            return Err("max_turns must be positive".into());
        }
        Ok(())
    }
}
```

`engine/src/board.rs`:

```rust
//! The per-game board: terrain, number tokens, and ports.

use crate::rng::Rng;
use crate::topology::{topo, NUM_TILES};
use crate::types::Resource;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PortKind {
    /// 3:1 any resource.
    Generic,
    /// 2:1 for one resource.
    Specific(Resource),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Board {
    /// `None` is the desert.
    pub tile_resource: [Option<Resource>; NUM_TILES],
    /// 0 on the desert.
    pub tile_number: [u8; NUM_TILES],
    /// (edge id, kind) for each of the 9 ports.
    pub ports: [(u8, PortKind); 9],
    pub generic_port_nodes: u64,
    pub specific_port_nodes: [u64; 5],
}

use Resource::*;

pub const STANDARD_TERRAIN: [Option<Resource>; NUM_TILES] = [
    Some(Wood), Some(Wood), Some(Wood), Some(Wood),
    Some(Brick), Some(Brick), Some(Brick),
    Some(Sheep), Some(Sheep), Some(Sheep), Some(Sheep),
    Some(Wheat), Some(Wheat), Some(Wheat), Some(Wheat),
    Some(Ore), Some(Ore), Some(Ore),
    None,
];

pub const STANDARD_NUMBERS: [u8; 18] = [2, 3, 3, 4, 4, 5, 5, 6, 6, 8, 8, 9, 9, 10, 10, 11, 11, 12];

pub const STANDARD_PORT_KINDS: [PortKind; 9] = [
    PortKind::Generic,
    PortKind::Generic,
    PortKind::Generic,
    PortKind::Generic,
    PortKind::Specific(Wood),
    PortKind::Specific(Brick),
    PortKind::Specific(Sheep),
    PortKind::Specific(Wheat),
    PortKind::Specific(Ore),
];

/// Positions in `Topology::coastal_edges` that hold ports (gaps of 3, 3, 4 around the coast).
pub const PORT_SLOTS: [usize; 9] = [0, 3, 6, 10, 13, 16, 20, 23, 26];

impl Board {
    pub fn new(
        tile_resource: [Option<Resource>; NUM_TILES],
        tile_number: [u8; NUM_TILES],
        ports: [(u8, PortKind); 9],
    ) -> Board {
        let t = topo();
        let mut generic = 0u64;
        let mut specific = [0u64; 5];
        for &(e, kind) in &ports {
            let (a, b) = t.edge_nodes[e as usize];
            let m = (1u64 << a) | (1u64 << b);
            match kind {
                PortKind::Generic => generic |= m,
                PortKind::Specific(r) => specific[r.index()] |= m,
            }
        }
        Board {
            tile_resource,
            tile_number,
            ports,
            generic_port_nodes: generic,
            specific_port_nodes: specific,
        }
    }

    /// Random standard board: shuffled terrain, tokens with no adjacent 6/8, shuffled port kinds.
    pub fn random(rng: &mut Rng) -> Board {
        let t = topo();
        let mut terrain = STANDARD_TERRAIN;
        rng.shuffle(&mut terrain);
        let mut numbers = STANDARD_NUMBERS;
        let tile_number = loop {
            rng.shuffle(&mut numbers);
            let mut tn = [0u8; NUM_TILES];
            let mut k = 0;
            for i in 0..NUM_TILES {
                if terrain[i].is_some() {
                    tn[i] = numbers[k];
                    k += 1;
                }
            }
            if !red_adjacent(&tn) {
                break tn;
            }
        };
        let mut kinds = STANDARD_PORT_KINDS;
        rng.shuffle(&mut kinds);
        let mut ports = [(0u8, PortKind::Generic); 9];
        for i in 0..9 {
            ports[i] = (t.coastal_edges[PORT_SLOTS[i]], kinds[i]);
        }
        Board::new(terrain, tile_number, ports)
    }

    pub fn desert(&self) -> u8 {
        self.tile_resource
            .iter()
            .position(|r| r.is_none())
            .expect("board has no desert") as u8
    }

    /// Cards of `r` needed for one bank card, given the player's building nodes.
    #[inline]
    pub fn maritime_rate(&self, buildings: u64, r: Resource) -> u8 {
        if buildings & self.specific_port_nodes[r.index()] != 0 {
            2
        } else if buildings & self.generic_port_nodes != 0 {
            3
        } else {
            4
        }
    }
}

fn red_adjacent(tile_number: &[u8; NUM_TILES]) -> bool {
    let t = topo();
    let red = |n: u8| n == 6 || n == 8;
    (0..NUM_TILES).any(|i| {
        red(tile_number[i]) && t.tile_neighbors[i].iter().any(|&j| red(tile_number[j as usize]))
    })
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p settler-engine --test board`
Expected: PASS, 9 tests.

- [ ] **Step 5: Commit**

```bash
git add engine
git diff --cached | grep -nE 'sk-|AKIA|ghp_|password=' && echo "SECRET FOUND - abort" || git commit -m "engine: rng, config, and board generation"
git push
```

---

### Task 4: Action space encoding

**Files:**
- Create: `engine/src/action.rs`
- Modify: `engine/src/lib.rs`
- Test: `engine/tests/action.rs`

**Interfaces:**
- Consumes: `types::{PlayerId, Resource, Hand}`.
- Produces: `action::Action` (variants below), `Action::encode(self) -> u16`, `Action::decode(id: u16) -> Option<Action>`, `ACTION_SPACE_SIZE = 665`, `PAIRS: [(u8, u8); 15]`, `NUM_BUNDLES = 20`, `bundle(i: u8) -> Hand`, `bundle_size(i: u8) -> u8`.

```rust
pub enum Action {
    Roll, EndTurn, BuyDev, PlayKnight, PlayRoadBuilding,
    PlayMonopoly(Resource),
    PlayYearOfPlenty(Resource, Resource), // canonical order: first <= second
    BuildSettlement(u8), BuildCity(u8), BuildRoad(u8),
    MoveRobber(u8), StealFrom(PlayerId), Discard(Resource),
    MaritimeTrade { give: Resource, get: Resource },
    OfferTrade { give: u8, get: u8 }, // bundle indices 0..20
    AcceptTrade, RejectTrade, ConfirmTrade(PlayerId), CancelTrade,
}
```

Encoding table:

| ids | action |
|---|---|
| 0–4 | Roll, EndTurn, BuyDev, PlayKnight, PlayRoadBuilding |
| 5–9 | PlayMonopoly(r) |
| 10–24 | PlayYearOfPlenty, index into `PAIRS` |
| 25–78 | BuildSettlement(node) |
| 79–132 | BuildCity(node) |
| 133–204 | BuildRoad(edge) |
| 205–223 | MoveRobber(tile) |
| 224–227 | StealFrom(player) |
| 228–232 | Discard(r) |
| 233–257 | MaritimeTrade, `give * 5 + get` (5 diagonal ids invalid) |
| 258–657 | OfferTrade, `give * 20 + get` |
| 658, 659 | AcceptTrade, RejectTrade |
| 660–663 | ConfirmTrade(player) |
| 664 | CancelTrade |

- [ ] **Step 1: Write the failing test**

`engine/tests/action.rs`:

```rust
use settler_engine::action::*;
use settler_engine::types::{hand_total, Resource};

#[test]
fn space_size() {
    assert_eq!(ACTION_SPACE_SIZE, 665);
}

#[test]
fn decode_encode_roundtrip() {
    let mut n = 0;
    for id in 0..ACTION_SPACE_SIZE as u16 {
        if let Some(a) = Action::decode(id) {
            assert_eq!(a.encode(), id, "{a:?}");
            n += 1;
        }
    }
    assert_eq!(n, 660);
    assert_eq!(Action::decode(ACTION_SPACE_SIZE as u16), None);
}

#[test]
fn known_offsets() {
    assert_eq!(Action::Roll.encode(), 0);
    assert_eq!(Action::BuildSettlement(0).encode(), 25);
    assert_eq!(Action::BuildRoad(71).encode(), 204);
    assert_eq!(Action::MoveRobber(18).encode(), 223);
    assert_eq!(Action::OfferTrade { give: 19, get: 19 }.encode(), 657);
    assert_eq!(Action::CancelTrade.encode(), 664);
}

#[test]
fn maritime_same_resource_is_not_an_action() {
    assert_eq!(Action::decode(233), None); // give Wood, get Wood
    assert_eq!(
        Action::decode(234),
        Some(Action::MaritimeTrade { give: Resource::Wood, get: Resource::Brick })
    );
}

#[test]
fn year_of_plenty_order_is_canonical() {
    use Resource::*;
    assert_eq!(
        Action::PlayYearOfPlenty(Ore, Wood).encode(),
        Action::PlayYearOfPlenty(Wood, Ore).encode()
    );
    assert_eq!(
        Action::decode(Action::PlayYearOfPlenty(Ore, Wood).encode()),
        Some(Action::PlayYearOfPlenty(Wood, Ore))
    );
}

#[test]
fn bundles() {
    assert_eq!(bundle(0), [1, 0, 0, 0, 0]);
    assert_eq!(bundle(5), [2, 0, 0, 0, 0]);
    assert_eq!(bundle(6), [1, 1, 0, 0, 0]);
    assert_eq!(bundle(19), [0, 0, 0, 0, 2]);
    let mut all = Vec::new();
    for i in 0..NUM_BUNDLES as u8 {
        assert_eq!(hand_total(&bundle(i)) as u8, bundle_size(i));
        all.push(bundle(i));
    }
    all.sort();
    all.dedup();
    assert_eq!(all.len(), NUM_BUNDLES);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p settler-engine --test action`
Expected: FAIL to compile — `could not find action`.

- [ ] **Step 3: Write the implementation**

Add to `engine/src/lib.rs`:

```rust
pub mod action;
```

`engine/src/action.rs`:

```rust
//! Every move as one integer in a fixed space (see the table in Plan 1, Task 4).

use crate::types::{Hand, PlayerId, Resource};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Action {
    Roll,
    EndTurn,
    BuyDev,
    PlayKnight,
    PlayRoadBuilding,
    PlayMonopoly(Resource),
    /// Canonical order: first <= second.
    PlayYearOfPlenty(Resource, Resource),
    BuildSettlement(u8),
    BuildCity(u8),
    BuildRoad(u8),
    MoveRobber(u8),
    StealFrom(PlayerId),
    Discard(Resource),
    MaritimeTrade { give: Resource, get: Resource },
    /// Indices into the trade bundles (see `bundle`).
    OfferTrade { give: u8, get: u8 },
    AcceptTrade,
    RejectTrade,
    ConfirmTrade(PlayerId),
    CancelTrade,
}

pub const ACTION_SPACE_SIZE: usize = 665;

/// Unordered resource pairs (i <= j): Year of Plenty choices and 2-card trade bundles.
pub const PAIRS: [(u8, u8); 15] = [
    (0, 0), (0, 1), (0, 2), (0, 3), (0, 4),
    (1, 1), (1, 2), (1, 3), (1, 4),
    (2, 2), (2, 3), (2, 4),
    (3, 3), (3, 4),
    (4, 4),
];

/// Bundles 0..5 are one card of a resource; 5..20 are the `PAIRS`.
pub const NUM_BUNDLES: usize = 20;

pub fn bundle(i: u8) -> Hand {
    let mut h = [0u8; 5];
    if i < 5 {
        h[i as usize] = 1;
    } else {
        let (a, b) = PAIRS[(i - 5) as usize];
        h[a as usize] += 1;
        h[b as usize] += 1;
    }
    h
}

#[inline]
pub fn bundle_size(i: u8) -> u8 {
    if i < 5 {
        1
    } else {
        2
    }
}

fn pair_index(a: Resource, b: Resource) -> u16 {
    let (a, b) = (a.index().min(b.index()) as u8, a.index().max(b.index()) as u8);
    PAIRS.iter().position(|&p| p == (a, b)).unwrap() as u16
}

impl Action {
    pub fn encode(self) -> u16 {
        match self {
            Action::Roll => 0,
            Action::EndTurn => 1,
            Action::BuyDev => 2,
            Action::PlayKnight => 3,
            Action::PlayRoadBuilding => 4,
            Action::PlayMonopoly(r) => 5 + r.index() as u16,
            Action::PlayYearOfPlenty(a, b) => 10 + pair_index(a, b),
            Action::BuildSettlement(n) => 25 + n as u16,
            Action::BuildCity(n) => 79 + n as u16,
            Action::BuildRoad(e) => 133 + e as u16,
            Action::MoveRobber(t) => 205 + t as u16,
            Action::StealFrom(p) => 224 + p as u16,
            Action::Discard(r) => 228 + r.index() as u16,
            Action::MaritimeTrade { give, get } => 233 + 5 * give.index() as u16 + get.index() as u16,
            Action::OfferTrade { give, get } => 258 + 20 * give as u16 + get as u16,
            Action::AcceptTrade => 658,
            Action::RejectTrade => 659,
            Action::ConfirmTrade(p) => 660 + p as u16,
            Action::CancelTrade => 664,
        }
    }

    pub fn decode(id: u16) -> Option<Action> {
        let r = |i: u16| Resource::from_index(i as usize);
        Some(match id {
            0 => Action::Roll,
            1 => Action::EndTurn,
            2 => Action::BuyDev,
            3 => Action::PlayKnight,
            4 => Action::PlayRoadBuilding,
            5..=9 => Action::PlayMonopoly(r(id - 5)),
            10..=24 => {
                let (a, b) = PAIRS[(id - 10) as usize];
                Action::PlayYearOfPlenty(r(a as u16), r(b as u16))
            }
            25..=78 => Action::BuildSettlement((id - 25) as u8),
            79..=132 => Action::BuildCity((id - 79) as u8),
            133..=204 => Action::BuildRoad((id - 133) as u8),
            205..=223 => Action::MoveRobber((id - 205) as u8),
            224..=227 => Action::StealFrom((id - 224) as u8),
            228..=232 => Action::Discard(r(id - 228)),
            233..=257 => {
                let k = id - 233;
                let (give, get) = (k / 5, k % 5);
                if give == get {
                    return None;
                }
                Action::MaritimeTrade { give: r(give), get: r(get) }
            }
            258..=657 => {
                let k = id - 258;
                Action::OfferTrade { give: (k / 20) as u8, get: (k % 20) as u8 }
            }
            658 => Action::AcceptTrade,
            659 => Action::RejectTrade,
            660..=663 => Action::ConfirmTrade((id - 660) as u8),
            664 => Action::CancelTrade,
            _ => return None,
        })
    }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p settler-engine --test action`
Expected: PASS, 6 tests.

- [ ] **Step 5: Commit**

```bash
git add engine
git diff --cached | grep -nE 'sk-|AKIA|ghp_|password=' && echo "SECRET FOUND - abort" || git commit -m "engine: fixed action space encoding"
git push
```

---

### Task 5: State, events, dispatchers, and setup phase

**Files:**
- Create: `engine/src/state.rs`, `engine/src/events.rs`, `engine/src/legal.rs`, `engine/src/apply.rs`, `engine/src/rules/mod.rs`, `engine/src/rules/setup.rs`
- Modify: `engine/src/lib.rs`
- Test: `engine/tests/common/mod.rs`, `engine/tests/setup.rs`

**Interfaces:**
- Consumes: everything from Tasks 1–4.
- Produces:
  - `state::Phase { SetupSettlement, SetupRoad { node: u8 }, PreRoll, Discard, MoveRobber, Steal, Main, RoadBuilding { roads_left: u8 }, TradeResponse, TradeConfirm, GameOver { winner: Option<PlayerId> } }`
  - `state::PlayerState { hand: Hand, dev_hand: [u8; 5], dev_new: [u8; 5], knights_played: u8, settlements: u64, cities: u64, roads: u128, longest_road_len: u8, discard_remaining: u8 }`
  - `state::Response { Pending, Accepted, Rejected }`, `state::PendingTrade { give: Hand, get: Hand, responses: [Response; 4], next_responder: PlayerId }` (`give` = what the offerer gives).
  - `state::State` fields: `config, seed, board, players: [PlayerState; 4], bank: Hand, dev_deck: [DevCard; 25], dev_deck_pos: u8, robber: u8, current: PlayerId, phase: Phase, turn: u32, setup_step: u8, dev_played_this_turn: bool, offers_this_turn: u8, trade: Option<PendingTrade>, robber_return: Phase, longest_road_owner: Option<PlayerId>, largest_army_owner: Option<PlayerId>, rng_steal: Rng, rng_misc: Rng`. Methods: `State::new(seed, config)`, `State::with_board(seed, config, board)`, `current_actor() -> PlayerId`, `is_over() -> bool`, `winner() -> Option<PlayerId>`, `occupied_nodes() -> u64`, `occupied_edges() -> u128`, `buildings(p: usize) -> u64`, `public_vp(p: usize) -> u8`, `total_vp(p: usize) -> u8`. `state::SETUP_ORDER`.
  - `events::Event` (all variants, below) and `Event::redacted_for(self, viewer: PlayerId) -> Event`.
  - `apply::Chance { Roll { dice: (u8, u8), discards: Option<[Hand; 4]> }, Steal(Resource), Dev(DevCard) }`, `apply::EventSink` trait (`fn emit(&mut self, e: Event)`), `apply::NoEvents`, `impl EventSink for Vec<Event>`. `State::apply(&mut self, a)`, `State::apply_with<S: EventSink>(&mut self, a, chance: Option<Chance>, sink: &mut S)`.
  - `legal::legal_actions(s: &State, out: &mut Vec<Action>)`, `State::legal_actions(&self) -> Vec<Action>`.
  - Test helpers in `engine/tests/common/mod.rs` (listed in Step 1).

- [ ] **Step 1: Write the test helpers**

`engine/tests/common/mod.rs`:

```rust
#![allow(dead_code)]

use settler_engine::apply::{Chance, NoEvents};
use settler_engine::topology::topo;
use settler_engine::*;

pub fn cfg() -> GameConfig {
    GameConfig::default()
}

/// A fresh game skipped past setup: no buildings, empty hands, player 0 to act in `phase`.
pub fn blank(seed: u64, phase: Phase) -> State {
    blank_with(seed, phase, cfg())
}

pub fn blank_with(seed: u64, phase: Phase, config: GameConfig) -> State {
    let mut s = State::new(seed, config);
    s.setup_step = 8;
    s.current = 0;
    s.phase = phase;
    s
}

/// Play setup by always taking the first legal action.
pub fn after_setup(seed: u64) -> State {
    let mut s = State::new(seed, cfg());
    while matches!(s.phase, Phase::SetupSettlement | Phase::SetupRoad { .. }) {
        let a = s.legal_actions()[0];
        s.apply(a);
    }
    s
}

/// Move cards from the bank to player `p` (keeps resource totals conserved).
pub fn give(s: &mut State, p: usize, h: Hand) {
    hand_sub(&mut s.bank, &h);
    hand_add(&mut s.players[p].hand, &h);
}

pub fn dice_for_sum(sum: u8) -> (u8, u8) {
    let a = (sum - 1).min(6);
    (a, sum - a)
}

/// Roll a specific total.
pub fn roll(s: &mut State, sum: u8) {
    s.apply_with(
        Action::Roll,
        Some(Chance::Roll { dice: dice_for_sum(sum), discards: None }),
        &mut NoEvents,
    );
}

pub fn mask64(xs: &[u8]) -> u64 {
    xs.iter().fold(0, |m, &x| m | (1u64 << x))
}

pub fn mask128(xs: &[u8]) -> u128 {
    xs.iter().fold(0, |m, &x| m | (1u128 << x))
}

pub fn edge_between(a: u8, b: u8) -> u8 {
    let t = topo();
    *t.node_edges[a as usize]
        .iter()
        .find(|&&e| {
            let (x, y) = t.edge_nodes[e as usize];
            (x == a && y == b) || (x == b && y == a)
        })
        .expect("nodes are not adjacent")
}

/// A simple path of `len` edges starting at `start`, avoiding nodes in `avoid`. Returns its nodes.
pub fn path_nodes(start: u8, len: usize, avoid: u64) -> Vec<u8> {
    fn go(path: &mut Vec<u8>, len: usize, avoid: u64) -> bool {
        if path.len() == len + 1 {
            return true;
        }
        let last = *path.last().unwrap();
        for &nb in &topo().node_neighbors[last as usize] {
            if !path.contains(&nb) && avoid & (1u64 << nb) == 0 {
                path.push(nb);
                if go(path, len, avoid) {
                    return true;
                }
                path.pop();
            }
        }
        false
    }
    let mut path = vec![start];
    assert!(go(&mut path, len, avoid), "no path of length {len} from {start}");
    path
}

pub fn path_edges(nodes: &[u8]) -> Vec<u8> {
    nodes.windows(2).map(|w| edge_between(w[0], w[1])).collect()
}

/// Nodes of tile `t` that touch no other tile.
pub fn exclusive_nodes(t: usize) -> Vec<u8> {
    topo().tile_nodes[t]
        .iter()
        .copied()
        .filter(|&n| topo().node_tiles[n as usize].len() == 1)
        .collect()
}

/// A non-desert tile with at least two nodes that touch only it.
pub fn lonely_tile(s: &State) -> usize {
    (0..19)
        .find(|&t| s.board.tile_resource[t].is_some() && exclusive_nodes(t).len() >= 2)
        .unwrap()
}

pub fn sorted(mut v: Vec<Action>) -> Vec<Action> {
    v.sort_by_key(|a| a.encode());
    v
}
```

- [ ] **Step 2: Write the failing test**

`engine/tests/setup.rs`:

```rust
mod common;
use common::*;
use settler_engine::topology::topo;
use settler_engine::*;

#[test]
fn state_fits_in_512_bytes() {
    let size = std::mem::size_of::<State>();
    assert!(size <= 512, "State is {size} bytes");
}

#[test]
fn new_game_starts_in_setup() {
    let s = State::new(1, cfg());
    assert_eq!(s.phase, Phase::SetupSettlement);
    assert_eq!(s.current_actor(), 0);
    assert_eq!(s.robber, s.board.desert());
    assert_eq!(s.bank, [19; 5]);
    let legal = s.legal_actions();
    assert_eq!(legal.len(), 54);
    assert!(legal.iter().all(|a| matches!(a, Action::BuildSettlement(_))));
}

#[test]
fn same_seed_same_game() {
    assert_eq!(State::new(5, cfg()), State::new(5, cfg()));
    assert_ne!(State::new(5, cfg()).dev_deck, State::new(6, cfg()).dev_deck);
}

#[test]
fn settlement_then_road_at_that_node() {
    let mut s = State::new(1, cfg());
    let n = 20u8;
    s.apply(Action::BuildSettlement(n));
    assert_eq!(s.phase, Phase::SetupRoad { node: n });
    let expected: Vec<Action> = topo().node_edges[n as usize].iter().map(|&e| Action::BuildRoad(e)).collect();
    assert_eq!(sorted(s.legal_actions()), sorted(expected));
}

#[test]
fn distance_rule_in_setup() {
    let mut s = State::new(1, cfg());
    let n = 20u8;
    s.apply(Action::BuildSettlement(n));
    let e = topo().node_edges[n as usize][0];
    s.apply(Action::BuildRoad(e));
    assert_eq!(s.current_actor(), 1);
    let legal = s.legal_actions();
    assert!(!legal.contains(&Action::BuildSettlement(n)));
    for &nb in &topo().node_neighbors[n as usize] {
        assert!(!legal.contains(&Action::BuildSettlement(nb)));
    }
    assert_eq!(legal.len(), 54 - 1 - topo().node_neighbors[n as usize].len());
}

#[test]
fn snake_order() {
    let mut s = State::new(2, cfg());
    let mut order = Vec::new();
    while matches!(s.phase, Phase::SetupSettlement | Phase::SetupRoad { .. }) {
        if s.phase == Phase::SetupSettlement {
            order.push(s.current_actor());
        }
        let a = s.legal_actions()[0];
        s.apply(a);
    }
    assert_eq!(order, vec![0, 1, 2, 3, 3, 2, 1, 0]);
    assert_eq!(s.phase, Phase::PreRoll);
    assert_eq!(s.current, 0);
    assert_eq!(s.turn, 0);
}

#[test]
fn only_second_settlement_pays_out() {
    let mut s = State::new(3, cfg());
    let mut placed = 0;
    while matches!(s.phase, Phase::SetupSettlement | Phase::SetupRoad { .. }) {
        let a = s.legal_actions()[0];
        if let Action::BuildSettlement(n) = a {
            let p = s.current as usize;
            let before = s.players[p].hand;
            let mut expected = [0u8; 5];
            if placed >= 4 {
                for &t in &topo().node_tiles[n as usize] {
                    if let Some(r) = s.board.tile_resource[t as usize] {
                        expected[r.index()] += 1;
                    }
                }
            }
            s.apply(a);
            let mut delta = s.players[p].hand;
            hand_sub(&mut delta, &before);
            assert_eq!(delta, expected, "settlement #{placed}");
            placed += 1;
        } else {
            s.apply(a);
        }
    }
    for r in 0..5 {
        let held: u32 = s.players.iter().map(|p| p.hand[r] as u32).sum();
        assert_eq!(held + s.bank[r] as u32, 19);
    }
}

#[test]
#[should_panic(expected = "illegal action")]
fn illegal_action_panics() {
    let mut s = State::new(1, cfg());
    s.apply(Action::Roll);
}
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cargo test -p settler-engine --test setup`
Expected: FAIL to compile — `cannot find type State` / `Phase`.

- [ ] **Step 4: Write the implementation**

`engine/src/lib.rs` becomes:

```rust
//! AlphaSettler's Catan rules engine.

pub mod action;
pub mod apply;
pub mod board;
pub mod config;
pub mod events;
pub mod legal;
pub mod rng;
pub mod rules;
pub mod state;
pub mod topology;
pub mod types;

pub use action::{Action, ACTION_SPACE_SIZE};
pub use apply::{Chance, EventSink, NoEvents};
pub use board::{Board, PortKind};
pub use config::GameConfig;
pub use events::Event;
pub use state::{PendingTrade, Phase, PlayerState, Response, State};
pub use types::*;
```

`engine/src/state.rs`:

```rust
//! The complete game state: a fixed-size `Copy` value.

use crate::board::Board;
use crate::config::GameConfig;
use crate::rng::{mix, Rng};
use crate::types::*;

const BOARD_SALT: u64 = 0xB0A2D;
const DECK_SALT: u64 = 0xDEC4;
const STEAL_SALT: u64 = 0x57EA1;
const MISC_SALT: u64 = 0x3155C;

/// Who places in each of the 8 setup rounds.
pub const SETUP_ORDER: [PlayerId; 8] = [0, 1, 2, 3, 3, 2, 1, 0];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    SetupSettlement,
    SetupRoad { node: u8 },
    PreRoll,
    /// Players with `discard_remaining > 0` discard one card at a time, lowest seat first.
    Discard,
    MoveRobber,
    Steal,
    Main,
    RoadBuilding { roads_left: u8 },
    /// Opponents answer the pending offer in seat order after the offerer.
    TradeResponse,
    TradeConfirm,
    GameOver { winner: Option<PlayerId> },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct PlayerState {
    pub hand: Hand,
    /// Every dev card held, including victory points and cards bought this turn.
    pub dev_hand: [u8; 5],
    /// Cards bought this turn (not yet playable).
    pub dev_new: [u8; 5],
    pub knights_played: u8,
    pub settlements: u64,
    pub cities: u64,
    pub roads: u128,
    pub longest_road_len: u8,
    pub discard_remaining: u8,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Response {
    Pending,
    Accepted,
    Rejected,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PendingTrade {
    /// What the offerer gives.
    pub give: Hand,
    /// What the offerer wants back.
    pub get: Hand,
    pub responses: [Response; NUM_PLAYERS],
    pub next_responder: PlayerId,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct State {
    pub config: GameConfig,
    pub seed: u64,
    pub board: Board,
    pub players: [PlayerState; NUM_PLAYERS],
    pub bank: Hand,
    pub dev_deck: [DevCard; DEV_DECK_SIZE],
    pub dev_deck_pos: u8,
    pub robber: u8,
    pub current: PlayerId,
    pub phase: Phase,
    pub turn: u32,
    pub setup_step: u8,
    pub dev_played_this_turn: bool,
    pub offers_this_turn: u8,
    pub trade: Option<PendingTrade>,
    /// Phase to resume after the robber is resolved (PreRoll for a pre-roll knight).
    pub robber_return: Phase,
    pub longest_road_owner: Option<PlayerId>,
    pub largest_army_owner: Option<PlayerId>,
    pub rng_steal: Rng,
    pub rng_misc: Rng,
}

impl State {
    pub fn new(seed: u64, config: GameConfig) -> State {
        let board = Board::random(&mut Rng::new(mix(seed, BOARD_SALT)));
        State::with_board(seed, config, board)
    }

    pub fn with_board(seed: u64, config: GameConfig, board: Board) -> State {
        config.validate().expect("invalid GameConfig");
        let mut dev_deck = [DevCard::Knight; DEV_DECK_SIZE];
        let mut k = 0;
        for (i, &count) in DEV_DECK_COUNTS.iter().enumerate() {
            for _ in 0..count {
                dev_deck[k] = DevCard::from_index(i);
                k += 1;
            }
        }
        Rng::new(mix(seed, DECK_SALT)).shuffle(&mut dev_deck);
        State {
            config,
            seed,
            board,
            players: [PlayerState::default(); NUM_PLAYERS],
            bank: [BANK_PER_RESOURCE; NUM_RESOURCES],
            dev_deck,
            dev_deck_pos: 0,
            robber: board.desert(),
            current: SETUP_ORDER[0],
            phase: Phase::SetupSettlement,
            turn: 0,
            setup_step: 0,
            dev_played_this_turn: false,
            offers_this_turn: 0,
            trade: None,
            robber_return: Phase::Main,
            longest_road_owner: None,
            largest_army_owner: None,
            rng_steal: Rng::new(mix(seed, STEAL_SALT)),
            rng_misc: Rng::new(mix(seed, MISC_SALT)),
        }
    }

    /// The player who must choose the next action.
    pub fn current_actor(&self) -> PlayerId {
        match self.phase {
            Phase::Discard => (0..NUM_PLAYERS)
                .find(|&p| self.players[p].discard_remaining > 0)
                .expect("Discard phase with nobody discarding") as PlayerId,
            Phase::TradeResponse => self.trade.expect("TradeResponse without a trade").next_responder,
            _ => self.current,
        }
    }

    #[inline]
    pub fn is_over(&self) -> bool {
        matches!(self.phase, Phase::GameOver { .. })
    }

    pub fn winner(&self) -> Option<PlayerId> {
        match self.phase {
            Phase::GameOver { winner } => winner,
            _ => None,
        }
    }

    #[inline]
    pub fn buildings(&self, p: usize) -> u64 {
        self.players[p].settlements | self.players[p].cities
    }

    #[inline]
    pub fn occupied_nodes(&self) -> u64 {
        (0..NUM_PLAYERS).fold(0, |m, p| m | self.buildings(p))
    }

    #[inline]
    pub fn occupied_edges(&self) -> u128 {
        self.players.iter().fold(0, |m, p| m | p.roads)
    }

    /// Victory points visible to everyone (excludes hidden VP cards).
    pub fn public_vp(&self, p: usize) -> u8 {
        let pl = &self.players[p];
        let mut vp = pl.settlements.count_ones() as u8 + 2 * pl.cities.count_ones() as u8;
        if self.longest_road_owner == Some(p as PlayerId) {
            vp += 2;
        }
        if self.largest_army_owner == Some(p as PlayerId) {
            vp += 2;
        }
        vp
    }

    pub fn total_vp(&self, p: usize) -> u8 {
        self.public_vp(p) + self.players[p].dev_hand[DevCard::VictoryPoint.index()]
    }
}
```

`engine/src/events.rs`:

```rust
//! What happened, for logs and observers. `Stole` and `BoughtDev` carry private details.

use crate::types::{DevCard, Hand, PlayerId, Resource};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Event {
    BuiltSettlement { player: PlayerId, node: u8 },
    BuiltCity { player: PlayerId, node: u8 },
    BuiltRoad { player: PlayerId, edge: u8 },
    Rolled { player: PlayerId, dice: (u8, u8) },
    Produced { player: PlayerId, resources: Hand },
    Discarded { player: PlayerId, resource: Resource },
    RobberMoved { player: PlayerId, tile: u8 },
    /// `resource` is `None` in views of players other than thief and victim.
    Stole { thief: PlayerId, victim: PlayerId, resource: Option<Resource> },
    /// `card` is `None` in views of players other than the buyer.
    BoughtDev { player: PlayerId, card: Option<DevCard> },
    PlayedDev { player: PlayerId, card: DevCard },
    MonopolyTaken { player: PlayerId, resource: Resource, amount: u8 },
    YearOfPlentyTaken { player: PlayerId, resources: Hand },
    MaritimeTraded { player: PlayerId, gave: Hand, got: Hand },
    TradeOffered { player: PlayerId, give: Hand, get: Hand },
    TradeResponded { player: PlayerId, accepted: bool },
    TradeConfirmed { offerer: PlayerId, partner: PlayerId, offerer_gave: Hand, partner_gave: Hand },
    TradeCancelled { player: PlayerId },
    TurnEnded { player: PlayerId },
    GameOver { winner: Option<PlayerId> },
}

impl Event {
    /// This event as `viewer` is allowed to see it.
    pub fn redacted_for(self, viewer: PlayerId) -> Event {
        match self {
            Event::Stole { thief, victim, .. } if viewer != thief && viewer != victim => {
                Event::Stole { thief, victim, resource: None }
            }
            Event::BoughtDev { player, .. } if viewer != player => Event::BoughtDev { player, card: None },
            e => e,
        }
    }
}
```

`engine/src/apply.rs`:

```rust
//! Applying actions. Actions must be legal (see `legal_actions`); illegal ones panic or corrupt.

use crate::action::Action;
use crate::events::Event;
use crate::rules::setup;
use crate::state::{Phase, State};
use crate::types::{DevCard, Hand, Resource, NUM_PLAYERS};

/// A random outcome supplied by the caller instead of drawn from the state's streams.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Chance {
    /// `discards` is only used in `catanatron_compat` mode on a 7.
    Roll { dice: (u8, u8), discards: Option<[Hand; NUM_PLAYERS]> },
    Steal(Resource),
    Dev(DevCard),
}

pub trait EventSink {
    fn emit(&mut self, e: Event);
}

/// Discards events; compiles away in search.
pub struct NoEvents;

impl EventSink for NoEvents {
    #[inline(always)]
    fn emit(&mut self, _e: Event) {}
}

impl EventSink for Vec<Event> {
    fn emit(&mut self, e: Event) {
        self.push(e);
    }
}

impl State {
    pub fn apply(&mut self, a: Action) {
        self.apply_with(a, None, &mut NoEvents);
    }

    pub fn apply_with<S: EventSink>(&mut self, a: Action, chance: Option<Chance>, sink: &mut S) {
        match (self.phase, a) {
            (Phase::SetupSettlement, Action::BuildSettlement(n)) => setup::apply_settlement(self, n, sink),
            (Phase::SetupRoad { node }, Action::BuildRoad(e)) => setup::apply_road(self, node, e, sink),
            (phase, a) => panic!("illegal action {a:?} in phase {phase:?} (chance {chance:?})"),
        }
    }
}
```

`engine/src/legal.rs`:

```rust
//! Legal action generation, dispatched on phase.

use crate::action::Action;
use crate::rules::setup;
use crate::state::{Phase, State};

pub fn legal_actions(s: &State, out: &mut Vec<Action>) {
    out.clear();
    match s.phase {
        Phase::SetupSettlement => setup::legal_settlement(s, out),
        Phase::SetupRoad { node } => setup::legal_road(s, node, out),
        Phase::GameOver { .. } => {}
        Phase::PreRoll
        | Phase::Discard
        | Phase::MoveRobber
        | Phase::Steal
        | Phase::Main
        | Phase::RoadBuilding { .. }
        | Phase::TradeResponse
        | Phase::TradeConfirm => {}
    }
}

impl State {
    pub fn legal_actions(&self) -> Vec<Action> {
        let mut v = Vec::new();
        legal_actions(self, &mut v);
        v
    }
}
```

`engine/src/rules/mod.rs`:

```rust
//! One module per rule area.

pub mod setup;
```

`engine/src/rules/setup.rs`:

```rust
//! Snake-order initial placement.

use crate::action::Action;
use crate::apply::EventSink;
use crate::events::Event;
use crate::state::{Phase, State, SETUP_ORDER};
use crate::topology::{topo, NUM_NODES};
use crate::types::{hand_add, hand_sub, hand_total};

pub fn legal_settlement(s: &State, out: &mut Vec<Action>) {
    let t = topo();
    let occ = s.occupied_nodes();
    for n in 0..NUM_NODES {
        if occ & (1u64 << n) == 0 && occ & t.node_neighbor_mask[n] == 0 {
            out.push(Action::BuildSettlement(n as u8));
        }
    }
}

pub fn legal_road(s: &State, node: u8, out: &mut Vec<Action>) {
    let occ = s.occupied_edges();
    for &e in &topo().node_edges[node as usize] {
        if occ & (1u128 << e) == 0 {
            out.push(Action::BuildRoad(e));
        }
    }
}

pub fn apply_settlement<S: EventSink>(s: &mut State, n: u8, sink: &mut S) {
    let p = s.current as usize;
    s.players[p].settlements |= 1u64 << n;
    sink.emit(Event::BuiltSettlement { player: s.current, node: n });
    if s.setup_step >= 4 {
        let mut got = [0u8; 5];
        for &tile in &topo().node_tiles[n as usize] {
            if let Some(r) = s.board.tile_resource[tile as usize] {
                got[r.index()] += 1;
            }
        }
        hand_add(&mut s.players[p].hand, &got);
        hand_sub(&mut s.bank, &got);
        if hand_total(&got) > 0 {
            sink.emit(Event::Produced { player: s.current, resources: got });
        }
    }
    s.phase = Phase::SetupRoad { node: n };
}

pub fn apply_road<S: EventSink>(s: &mut State, node: u8, e: u8, sink: &mut S) {
    debug_assert!(topo().node_edges[node as usize].contains(&e));
    let p = s.current as usize;
    s.players[p].roads |= 1u128 << e;
    sink.emit(Event::BuiltRoad { player: s.current, edge: e });
    s.setup_step += 1;
    if s.setup_step as usize == SETUP_ORDER.len() {
        s.current = 0;
        s.phase = Phase::PreRoll;
    } else {
        s.current = SETUP_ORDER[s.setup_step as usize];
        s.phase = Phase::SetupSettlement;
    }
}
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cargo test -p settler-engine --test setup`
Expected: PASS, 8 tests. If `state_fits_in_512_bytes` fails, print the size, then shrink `State` (e.g. store `ports` edge ids as `[u8; 9]` plus kinds as `[u8; 9]`) before moving on. Do not raise the limit.

- [ ] **Step 6: Commit**

```bash
git add engine
git diff --cached | grep -nE 'sk-|AKIA|ghp_|password=' && echo "SECRET FOUND - abort" || git commit -m "engine: state, events, dispatch, and setup phase"
git push
```

---

### Task 6: Rolling, production, sevens, discards, and the robber

**Files:**
- Create: `engine/src/rules/roll.rs`, `engine/src/rules/robber.rs`
- Modify: `engine/src/rules/mod.rs`, `engine/src/legal.rs`, `engine/src/apply.rs`
- Test: `engine/tests/roll.rs`

**Interfaces:**
- Consumes: `State`, `Phase`, `Chance`, `EventSink`, `rng::dice_for`, test helpers.
- Produces: `rules::roll::{apply_roll, legal_discard, apply_discard, pick_card, sample_cards}`, `rules::robber::{legal_move_robber, apply_move_robber, legal_steal, apply_steal, victims}`. `pick_card(hand: &Hand, rng: &mut Rng) -> Resource` (hand must be non-empty). `sample_cards(hand: &Hand, k: u8, rng: &mut Rng) -> Hand`. `victims(s: &State) -> [bool; 4]`.

- [ ] **Step 1: Write the failing test**

`engine/tests/roll.rs`:

```rust
mod common;
use common::*;
use settler_engine::apply::{Chance, NoEvents};
use settler_engine::rng::dice_for;
use settler_engine::topology::topo;
use settler_engine::*;

#[test]
fn preroll_offers_roll() {
    let s = blank(1, Phase::PreRoll);
    assert_eq!(s.legal_actions(), vec![Action::Roll]);
}

#[test]
fn unforced_roll_uses_seed_and_turn() {
    let mut s = after_setup(4);
    let mut log: Vec<Event> = Vec::new();
    s.apply_with(Action::Roll, None, &mut log);
    assert_eq!(log[0], Event::Rolled { player: 0, dice: dice_for(4, 0) });
}

#[test]
fn settlement_collects_one_city_two() {
    let mut s = blank(3, Phase::PreRoll);
    let t = lonely_tile(&s);
    let r = s.board.tile_resource[t].unwrap();
    let ex = exclusive_nodes(t);
    s.players[0].settlements = 1u64 << ex[0];
    s.players[1].cities = 1u64 << ex[1];
    roll(&mut s, s.board.tile_number[t]);
    assert_eq!(s.players[0].hand[r.index()], 1);
    assert_eq!(s.players[1].hand[r.index()], 2);
    assert_eq!(s.bank[r.index()], 16);
    assert_eq!(s.phase, Phase::Main);
}

#[test]
fn robber_blocks_production() {
    let mut s = blank(3, Phase::PreRoll);
    let t = lonely_tile(&s);
    s.players[0].settlements = 1u64 << exclusive_nodes(t)[0];
    s.robber = t as u8;
    roll(&mut s, s.board.tile_number[t]);
    assert_eq!(hand_total(&s.players[0].hand), 0);
}

#[test]
fn bank_shortage_with_two_claimants_pays_nobody() {
    let mut s = blank(3, Phase::PreRoll);
    let t = lonely_tile(&s);
    let r = s.board.tile_resource[t].unwrap().index();
    let ex = exclusive_nodes(t);
    s.players[0].settlements = 1u64 << ex[0];
    s.players[1].settlements = 1u64 << ex[1];
    s.bank[r] = 1;
    roll(&mut s, s.board.tile_number[t]);
    assert_eq!(s.players[0].hand[r], 0);
    assert_eq!(s.players[1].hand[r], 0);
    assert_eq!(s.bank[r], 1);
}

#[test]
fn bank_shortage_with_one_claimant_pays_what_is_left() {
    let mut s = blank(3, Phase::PreRoll);
    let t = lonely_tile(&s);
    let r = s.board.tile_resource[t].unwrap().index();
    s.players[0].cities = 1u64 << exclusive_nodes(t)[0];
    s.bank[r] = 1;
    roll(&mut s, s.board.tile_number[t]);
    assert_eq!(s.players[0].hand[r], 1);
    assert_eq!(s.bank[r], 0);
}

#[test]
fn seven_without_big_hands_goes_to_robber() {
    let mut s = blank(1, Phase::PreRoll);
    give(&mut s, 0, [2, 2, 2, 1, 0]); // exactly 7: no discard
    roll(&mut s, 7);
    assert_eq!(s.phase, Phase::MoveRobber);
    assert_eq!(hand_total(&s.players[0].hand), 7);
}

#[test]
fn seven_makes_big_hands_discard_half() {
    let mut s = blank(1, Phase::PreRoll);
    give(&mut s, 0, [3, 3, 3, 0, 0]);
    roll(&mut s, 7);
    assert_eq!(s.phase, Phase::Discard);
    assert_eq!(s.players[0].discard_remaining, 4);
    assert_eq!(s.current_actor(), 0);
    assert_eq!(
        sorted(s.legal_actions()),
        vec![Action::Discard(Resource::Wood), Action::Discard(Resource::Brick), Action::Discard(Resource::Sheep)]
    );
    for _ in 0..4 {
        s.apply(Action::Discard(Resource::Wood));
    }
    assert_eq!(s.players[0].hand, [0, 3, 3, 0, 0]);
    assert_eq!(s.phase, Phase::MoveRobber);
    assert_eq!(s.bank[0], 19);
}

#[test]
fn multiple_discarders_go_in_seat_order() {
    let mut s = blank(1, Phase::PreRoll);
    give(&mut s, 1, [8, 0, 0, 0, 0]);
    give(&mut s, 2, [0, 8, 0, 0, 0]);
    roll(&mut s, 7);
    assert_eq!(s.current_actor(), 1);
    for _ in 0..4 {
        assert_eq!(s.phase, Phase::Discard);
        s.apply(Action::Discard(Resource::Wood));
    }
    assert_eq!(s.current_actor(), 2);
    for _ in 0..4 {
        assert_eq!(s.phase, Phase::Discard);
        s.apply(Action::Discard(Resource::Brick));
    }
    assert_eq!(s.phase, Phase::MoveRobber);
    assert_eq!(s.current_actor(), 0);
}

#[test]
fn compat_mode_discards_randomly() {
    let c = GameConfig { catanatron_compat: true, ..cfg() };
    let mut s = blank_with(1, Phase::PreRoll, c);
    give(&mut s, 0, [3, 3, 3, 0, 0]);
    roll(&mut s, 7);
    assert_eq!(s.phase, Phase::MoveRobber);
    assert_eq!(hand_total(&s.players[0].hand), 5);
    assert_eq!(s.players[0].discard_remaining, 0);
}

#[test]
fn compat_mode_accepts_forced_discards() {
    let c = GameConfig { catanatron_compat: true, ..cfg() };
    let mut s = blank_with(1, Phase::PreRoll, c);
    give(&mut s, 0, [5, 4, 0, 0, 0]);
    let mut discards = [[0u8; 5]; 4];
    discards[0] = [4, 0, 0, 0, 0];
    s.apply_with(Action::Roll, Some(Chance::Roll { dice: (3, 4), discards: Some(discards) }), &mut NoEvents);
    assert_eq!(s.players[0].hand, [1, 4, 0, 0, 0]);
}

#[test]
fn robber_must_move() {
    let s = blank(1, Phase::MoveRobber);
    let legal = s.legal_actions();
    assert_eq!(legal.len(), 18);
    assert!(!legal.contains(&Action::MoveRobber(s.robber)));
}

#[test]
fn steal_from_victim_on_new_tile() {
    let mut s = blank(1, Phase::MoveRobber);
    let t = lonely_tile(&s) as u8;
    s.players[1].settlements = 1u64 << exclusive_nodes(t as usize)[0];
    give(&mut s, 1, [2, 0, 0, 0, 0]);
    s.apply(Action::MoveRobber(t));
    assert_eq!(s.robber, t);
    assert_eq!(s.phase, Phase::Steal);
    assert_eq!(s.legal_actions(), vec![Action::StealFrom(1)]);
    s.apply_with(Action::StealFrom(1), Some(Chance::Steal(Resource::Wood)), &mut NoEvents);
    assert_eq!(s.players[0].hand, [1, 0, 0, 0, 0]);
    assert_eq!(s.players[1].hand, [1, 0, 0, 0, 0]);
    assert_eq!(s.phase, Phase::Main);
}

#[test]
fn unforced_steal_takes_one_card() {
    let mut s = blank(1, Phase::MoveRobber);
    let t = lonely_tile(&s) as u8;
    s.players[1].settlements = 1u64 << exclusive_nodes(t as usize)[0];
    give(&mut s, 1, [1, 1, 1, 0, 0]);
    s.apply(Action::MoveRobber(t));
    s.apply(Action::StealFrom(1));
    assert_eq!(hand_total(&s.players[0].hand), 1);
    assert_eq!(hand_total(&s.players[1].hand), 2);
}

#[test]
fn empty_handed_players_are_not_victims() {
    let mut s = blank(1, Phase::MoveRobber);
    let t = lonely_tile(&s) as u8;
    s.players[1].settlements = 1u64 << exclusive_nodes(t as usize)[0];
    s.apply(Action::MoveRobber(t));
    assert_eq!(s.phase, Phase::Main);
}

#[test]
fn no_steal_when_only_own_buildings() {
    let mut s = blank(1, Phase::MoveRobber);
    let t = lonely_tile(&s) as u8;
    s.players[0].settlements = 1u64 << exclusive_nodes(t as usize)[0];
    give(&mut s, 0, [1, 0, 0, 0, 0]);
    s.apply(Action::MoveRobber(t));
    assert_eq!(s.phase, Phase::Main);
}

#[test]
#[should_panic(expected = "victim has none")]
fn forced_steal_of_missing_resource_panics() {
    let mut s = blank(1, Phase::MoveRobber);
    let t = lonely_tile(&s) as u8;
    s.players[1].settlements = 1u64 << exclusive_nodes(t as usize)[0];
    give(&mut s, 1, [1, 0, 0, 0, 0]);
    s.apply(Action::MoveRobber(t));
    s.apply_with(Action::StealFrom(1), Some(Chance::Steal(Resource::Ore)), &mut NoEvents);
}

#[test]
#[should_panic(expected = "does not match")]
fn mismatched_chance_panics() {
    let mut s = blank(1, Phase::PreRoll);
    s.apply_with(Action::Roll, Some(Chance::Steal(Resource::Wood)), &mut NoEvents);
}

#[test]
fn tile_masks_cover_building_production() {
    // Sanity check for the helper: lonely tile's exclusive nodes are on that tile only.
    let s = blank(9, Phase::PreRoll);
    let t = lonely_tile(&s);
    for n in exclusive_nodes(t) {
        assert_eq!(topo().node_tiles[n as usize], vec![t as u8]);
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p settler-engine --test roll`
Expected: FAIL — tests compile but panic with `illegal action Roll in phase PreRoll`, and `preroll_offers_roll` fails with `left: []`.

- [ ] **Step 3: Write the implementation**

`engine/src/rules/mod.rs` becomes:

```rust
//! One module per rule area.

pub mod robber;
pub mod roll;
pub mod setup;
```

`engine/src/rules/roll.rs`:

```rust
//! Dice, production (with the bank-shortage rule), sevens, and discards.

use crate::action::Action;
use crate::apply::{Chance, EventSink};
use crate::events::Event;
use crate::rng::{dice_for, Rng};
use crate::state::{Phase, State};
use crate::topology::{topo, NUM_TILES};
use crate::types::*;

pub fn apply_roll<S: EventSink>(s: &mut State, chance: Option<Chance>, sink: &mut S) {
    let ((d1, d2), forced_discards) = match chance {
        None => (dice_for(s.seed, s.turn), None),
        Some(Chance::Roll { dice, discards }) => (dice, discards),
        Some(c) => panic!("chance {c:?} does not match Roll"),
    };
    sink.emit(Event::Rolled { player: s.current, dice: (d1, d2) });
    let sum = d1 + d2;
    if sum != 7 {
        produce(s, sum, sink);
        s.phase = Phase::Main;
        return;
    }
    let limit = s.config.discard_limit as u32;
    let mut any = false;
    for p in 0..NUM_PLAYERS {
        let n = hand_total(&s.players[p].hand);
        if n > limit {
            s.players[p].discard_remaining = (n / 2) as u8;
            any = true;
        }
    }
    s.robber_return = Phase::Main;
    if any && s.config.catanatron_compat {
        random_discards(s, forced_discards, sink);
        s.phase = Phase::MoveRobber;
    } else if any {
        s.phase = Phase::Discard;
    } else {
        s.phase = Phase::MoveRobber;
    }
}

fn produce<S: EventSink>(s: &mut State, roll: u8, sink: &mut S) {
    let t = topo();
    let mut owed = [[0u8; NUM_RESOURCES]; NUM_PLAYERS];
    for tile in 0..NUM_TILES {
        if s.board.tile_number[tile] != roll || tile as u8 == s.robber {
            continue;
        }
        let Some(r) = s.board.tile_resource[tile] else { continue };
        let mask = t.tile_node_mask[tile];
        for p in 0..NUM_PLAYERS {
            let pl = &s.players[p];
            let n = (pl.settlements & mask).count_ones() + 2 * (pl.cities & mask).count_ones();
            owed[p][r.index()] += n as u8;
        }
    }
    for r in 0..NUM_RESOURCES {
        let total: u32 = (0..NUM_PLAYERS).map(|p| owed[p][r] as u32).sum();
        if total <= s.bank[r] as u32 {
            continue;
        }
        let claimants: Vec<usize> = (0..NUM_PLAYERS).filter(|&p| owed[p][r] > 0).collect();
        if claimants.len() == 1 {
            owed[claimants[0]][r] = s.bank[r];
        } else {
            for p in 0..NUM_PLAYERS {
                owed[p][r] = 0;
            }
        }
    }
    for p in 0..NUM_PLAYERS {
        if hand_total(&owed[p]) > 0 {
            hand_add(&mut s.players[p].hand, &owed[p]);
            hand_sub(&mut s.bank, &owed[p]);
            sink.emit(Event::Produced { player: p as PlayerId, resources: owed[p] });
        }
    }
}

/// One card drawn uniformly from a non-empty hand.
pub fn pick_card(hand: &Hand, rng: &mut Rng) -> Resource {
    let mut x = rng.below(hand_total(hand));
    for r in 0..NUM_RESOURCES {
        if x < hand[r] as u32 {
            return Resource::from_index(r);
        }
        x -= hand[r] as u32;
    }
    unreachable!("pick_card on an empty hand")
}

/// `k` cards drawn without replacement.
pub fn sample_cards(hand: &Hand, k: u8, rng: &mut Rng) -> Hand {
    let mut left = *hand;
    let mut out = [0u8; NUM_RESOURCES];
    for _ in 0..k {
        let r = pick_card(&left, rng).index();
        left[r] -= 1;
        out[r] += 1;
    }
    out
}

fn random_discards<S: EventSink>(s: &mut State, forced: Option<[Hand; NUM_PLAYERS]>, sink: &mut S) {
    for p in 0..NUM_PLAYERS {
        let k = s.players[p].discard_remaining;
        if k == 0 {
            continue;
        }
        let hand = s.players[p].hand;
        let discard = match forced {
            Some(f) => {
                assert_eq!(hand_total(&f[p]), k as u32, "forced discard for player {p} has wrong size");
                assert!(covers(&hand, &f[p]), "forced discard for player {p} exceeds hand");
                f[p]
            }
            None => sample_cards(&hand, k, &mut s.rng_misc),
        };
        for r in 0..NUM_RESOURCES {
            for _ in 0..discard[r] {
                sink.emit(Event::Discarded { player: p as PlayerId, resource: Resource::from_index(r) });
            }
        }
        hand_sub(&mut s.players[p].hand, &discard);
        hand_add(&mut s.bank, &discard);
        s.players[p].discard_remaining = 0;
    }
}

pub fn legal_discard(s: &State, out: &mut Vec<Action>) {
    let p = s.current_actor() as usize;
    for r in Resource::ALL {
        if s.players[p].hand[r.index()] > 0 {
            out.push(Action::Discard(r));
        }
    }
}

pub fn apply_discard<S: EventSink>(s: &mut State, r: Resource, sink: &mut S) {
    let p = s.current_actor() as usize;
    s.players[p].hand[r.index()] -= 1;
    s.bank[r.index()] += 1;
    s.players[p].discard_remaining -= 1;
    sink.emit(Event::Discarded { player: p as PlayerId, resource: r });
    if s.players.iter().all(|pl| pl.discard_remaining == 0) {
        s.phase = Phase::MoveRobber;
    }
}
```

`engine/src/rules/robber.rs`:

```rust
//! Moving the robber and stealing.

use crate::action::Action;
use crate::apply::{Chance, EventSink};
use crate::events::Event;
use crate::rules::roll::pick_card;
use crate::state::{Phase, State};
use crate::topology::{topo, NUM_TILES};
use crate::types::*;

pub fn legal_move_robber(s: &State, out: &mut Vec<Action>) {
    for t in 0..NUM_TILES as u8 {
        if t != s.robber {
            out.push(Action::MoveRobber(t));
        }
    }
}

/// Opponents with a building on the robber's tile and at least one card.
pub fn victims(s: &State) -> [bool; NUM_PLAYERS] {
    let mask = topo().tile_node_mask[s.robber as usize];
    let mut v = [false; NUM_PLAYERS];
    for p in 0..NUM_PLAYERS {
        v[p] = p != s.current as usize
            && s.buildings(p) & mask != 0
            && hand_total(&s.players[p].hand) > 0;
    }
    v
}

pub fn apply_move_robber<S: EventSink>(s: &mut State, tile: u8, sink: &mut S) {
    s.robber = tile;
    sink.emit(Event::RobberMoved { player: s.current, tile });
    s.phase = if victims(s).iter().any(|&v| v) { Phase::Steal } else { s.robber_return };
}

pub fn legal_steal(s: &State, out: &mut Vec<Action>) {
    for (p, &v) in victims(s).iter().enumerate() {
        if v {
            out.push(Action::StealFrom(p as PlayerId));
        }
    }
}

pub fn apply_steal<S: EventSink>(s: &mut State, victim: PlayerId, chance: Option<Chance>, sink: &mut S) {
    let v = victim as usize;
    let hand = s.players[v].hand;
    let r = match chance {
        None => pick_card(&hand, &mut s.rng_steal),
        Some(Chance::Steal(r)) => {
            assert!(hand[r.index()] > 0, "forced steal of {r:?} but victim has none");
            r
        }
        Some(c) => panic!("chance {c:?} does not match StealFrom"),
    };
    s.players[v].hand[r.index()] -= 1;
    s.players[s.current as usize].hand[r.index()] += 1;
    sink.emit(Event::Stole { thief: s.current, victim, resource: Some(r) });
    s.phase = s.robber_return;
}
```

In `engine/src/legal.rs`, change the imports and match:

```rust
use crate::rules::{robber, roll, setup};
```

and replace the `match s.phase { ... }` body with:

```rust
    match s.phase {
        Phase::SetupSettlement => setup::legal_settlement(s, out),
        Phase::SetupRoad { node } => setup::legal_road(s, node, out),
        Phase::PreRoll => out.push(Action::Roll),
        Phase::Discard => roll::legal_discard(s, out),
        Phase::MoveRobber => robber::legal_move_robber(s, out),
        Phase::Steal => robber::legal_steal(s, out),
        Phase::GameOver { .. } => {}
        Phase::Main
        | Phase::RoadBuilding { .. }
        | Phase::TradeResponse
        | Phase::TradeConfirm => {}
    }
```

In `engine/src/apply.rs`, change the rules import to `use crate::rules::{robber, roll, setup};` and add these arms before the final `(phase, a) => panic!` arm:

```rust
            (Phase::PreRoll, Action::Roll) => roll::apply_roll(self, chance, sink),
            (Phase::Discard, Action::Discard(r)) => roll::apply_discard(self, r, sink),
            (Phase::MoveRobber, Action::MoveRobber(t)) => robber::apply_move_robber(self, t, sink),
            (Phase::Steal, Action::StealFrom(v)) => robber::apply_steal(self, v, chance, sink),
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p settler-engine`
Expected: PASS — all of `roll` (19 tests) plus earlier suites.

- [ ] **Step 5: Commit**

```bash
git add engine
git diff --cached | grep -nE 'sk-|AKIA|ghp_|password=' && echo "SECRET FOUND - abort" || git commit -m "engine: rolling, production, sevens, robber"
git push
```

---

### Task 7: Building and ending the turn

**Files:**
- Create: `engine/src/rules/build.rs`
- Modify: `engine/src/rules/mod.rs`, `engine/src/legal.rs`, `engine/src/apply.rs`
- Test: `engine/tests/build.rs`

**Interfaces:**
- Consumes: `State`, test helpers (`blank`, `give`, `path_nodes`, `path_edges`, `edge_between`, `mask64`, `mask128`).
- Produces: `rules::build::{road_connected(s, p: usize, e: u8) -> bool, push_free_roads(s, p: usize, out), has_free_road(s, p: usize) -> bool, legal_main_builds(s, out), apply_build_road(s, e, free: bool, sink), apply_build_settlement(s, n, sink), apply_build_city(s, n, sink), apply_end_turn(s, sink)}`.

- [ ] **Step 1: Write the failing test**

`engine/tests/build.rs`:

```rust
mod common;
use common::*;
use settler_engine::topology::topo;
use settler_engine::*;

fn roads_in(legal: &[Action]) -> Vec<u8> {
    let mut v: Vec<u8> = legal.iter().filter_map(|a| if let Action::BuildRoad(e) = a { Some(*e) } else { None }).collect();
    v.sort();
    v
}

#[test]
fn only_end_turn_without_resources() {
    let s = blank(1, Phase::Main);
    assert_eq!(s.legal_actions(), vec![Action::EndTurn]);
}

#[test]
fn road_requires_connection() {
    let mut s = blank(1, Phase::Main);
    let a = 20u8;
    s.players[0].settlements = 1u64 << a;
    give(&mut s, 0, ROAD_COST);
    let mut expected = topo().node_edges[a as usize].clone();
    expected.sort();
    assert_eq!(roads_in(&s.legal_actions()), expected);
    s.apply(Action::BuildRoad(expected[0]));
    assert_eq!(s.players[0].hand, [0; 5]);
    assert_eq!(s.bank, [19; 5]);
    assert_eq!(s.players[0].roads, 1u128 << expected[0]);
}

#[test]
fn roads_extend_from_roads() {
    let mut s = blank(1, Phase::Main);
    let nodes = path_nodes(20, 1, 0);
    let (a, b) = (nodes[0], nodes[1]);
    let ab = edge_between(a, b);
    s.players[0].settlements = 1u64 << a;
    s.players[0].roads = 1u128 << ab;
    give(&mut s, 0, ROAD_COST);
    let legal = roads_in(&s.legal_actions());
    for &e in topo().node_edges[b as usize].iter().chain(&topo().node_edges[a as usize]) {
        if e != ab {
            assert!(legal.contains(&e), "edge {e}");
        }
    }
}

#[test]
fn opponent_settlement_blocks_road_through_node() {
    let mut s = blank(1, Phase::Main);
    let nodes = path_nodes(20, 1, 0);
    let (a, b) = (nodes[0], nodes[1]);
    let ab = edge_between(a, b);
    s.players[0].settlements = 1u64 << a;
    s.players[0].roads = 1u128 << ab;
    s.players[1].settlements = 1u64 << b;
    give(&mut s, 0, ROAD_COST);
    let legal = roads_in(&s.legal_actions());
    for &e in &topo().node_edges[b as usize] {
        assert!(!legal.contains(&e), "edge {e} passes through opponent's settlement");
    }
    assert!(!legal.is_empty());
}

#[test]
fn settlement_needs_own_road_and_distance() {
    let mut s = blank(1, Phase::Main);
    let nodes = path_nodes(20, 2, 0);
    s.players[0].settlements = 1u64 << nodes[0];
    s.players[0].roads = mask128(&path_edges(&nodes));
    give(&mut s, 0, SETTLEMENT_COST);
    let settlements: Vec<Action> =
        s.legal_actions().into_iter().filter(|a| matches!(a, Action::BuildSettlement(_))).collect();
    assert_eq!(settlements, vec![Action::BuildSettlement(nodes[2])]);
    s.apply(Action::BuildSettlement(nodes[2]));
    assert_eq!(s.players[0].settlements, mask64(&[nodes[0], nodes[2]]));
    assert_eq!(s.bank, [19; 5]);
    assert_eq!(s.public_vp(0), 2);
}

#[test]
fn city_upgrades_settlement() {
    let mut s = blank(1, Phase::Main);
    s.players[0].settlements = 1u64 << 20;
    give(&mut s, 0, CITY_COST);
    assert!(s.legal_actions().contains(&Action::BuildCity(20)));
    s.apply(Action::BuildCity(20));
    assert_eq!(s.players[0].settlements, 0);
    assert_eq!(s.players[0].cities, 1u64 << 20);
    assert_eq!(s.public_vp(0), 2);
    assert_eq!(s.bank, [19; 5]);
}

#[test]
fn piece_limits() {
    let mut s = blank(1, Phase::Main);
    let nodes = path_nodes(20, 2, 0);
    s.players[0].settlements = mask64(&[nodes[0], 0, 5, 40, 50]);
    s.players[0].roads = mask128(&path_edges(&nodes));
    give(&mut s, 0, SETTLEMENT_COST);
    assert!(!s.legal_actions().iter().any(|a| matches!(a, Action::BuildSettlement(_))));

    let mut s = blank(1, Phase::Main);
    s.players[0].settlements = 1u64 << topo().edge_nodes[0].0;
    s.players[0].roads = (1u128 << 15) - 1;
    give(&mut s, 0, ROAD_COST);
    assert!(!s.legal_actions().iter().any(|a| matches!(a, Action::BuildRoad(_))));

    let mut s = blank(1, Phase::Main);
    s.players[0].cities = mask64(&[0, 10, 30, 50]);
    s.players[0].settlements = 1u64 << 20;
    give(&mut s, 0, CITY_COST);
    assert!(!s.legal_actions().iter().any(|a| matches!(a, Action::BuildCity(_))));
}

#[test]
fn end_turn_advances() {
    let mut s = blank(1, Phase::Main);
    s.players[0].dev_hand[0] = 1;
    s.players[0].dev_new[0] = 1;
    s.dev_played_this_turn = true;
    s.offers_this_turn = 2;
    s.apply(Action::EndTurn);
    assert_eq!(s.current, 1);
    assert_eq!(s.turn, 1);
    assert_eq!(s.phase, Phase::PreRoll);
    assert_eq!(s.players[0].dev_new, [0; 5]);
    assert!(!s.dev_played_this_turn);
    assert_eq!(s.offers_this_turn, 0);
}

#[test]
fn max_turns_ends_game_without_winner() {
    let c = GameConfig { max_turns: 1, ..cfg() };
    let mut s = blank_with(1, Phase::Main, c);
    s.apply(Action::EndTurn);
    assert_eq!(s.phase, Phase::GameOver { winner: None });
    assert!(s.legal_actions().is_empty());
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p settler-engine --test build`
Expected: FAIL — `only_end_turn_without_resources` gets `[]`, others panic with `illegal action`.

- [ ] **Step 3: Write the implementation**

Add `pub mod build;` to `engine/src/rules/mod.rs`.

`engine/src/rules/build.rs`:

```rust
//! Roads, settlements, cities, and ending the turn.

use crate::action::Action;
use crate::apply::EventSink;
use crate::events::Event;
use crate::state::{Phase, State};
use crate::topology::{topo, NUM_EDGES, NUM_NODES};
use crate::types::*;

/// Free edge `e` touches player `p`'s building, or `p`'s road through a node no opponent holds.
pub fn road_connected(s: &State, p: usize, e: u8) -> bool {
    let t = topo();
    let pl = &s.players[p];
    let mine = pl.settlements | pl.cities;
    let theirs = s.occupied_nodes() & !mine;
    let (a, b) = t.edge_nodes[e as usize];
    for n in [a, b] {
        let bit = 1u64 << n;
        if mine & bit != 0 {
            return true;
        }
        if theirs & bit != 0 {
            continue;
        }
        if pl.roads & t.node_edge_mask[n as usize] & !(1u128 << e) != 0 {
            return true;
        }
    }
    false
}

pub fn push_free_roads(s: &State, p: usize, out: &mut Vec<Action>) {
    let occ = s.occupied_edges();
    for e in 0..NUM_EDGES as u8 {
        if occ & (1u128 << e) == 0 && road_connected(s, p, e) {
            out.push(Action::BuildRoad(e));
        }
    }
}

pub fn has_free_road(s: &State, p: usize) -> bool {
    let occ = s.occupied_edges();
    (0..NUM_EDGES as u8).any(|e| occ & (1u128 << e) == 0 && road_connected(s, p, e))
}

pub fn legal_main_builds(s: &State, out: &mut Vec<Action>) {
    let t = topo();
    let p = s.current as usize;
    let pl = &s.players[p];
    if covers(&pl.hand, &ROAD_COST) && pl.roads.count_ones() < MAX_ROADS {
        push_free_roads(s, p, out);
    }
    if covers(&pl.hand, &SETTLEMENT_COST) && pl.settlements.count_ones() < MAX_SETTLEMENTS {
        let occ = s.occupied_nodes();
        for n in 0..NUM_NODES {
            if occ & (1u64 << n) == 0
                && occ & t.node_neighbor_mask[n] == 0
                && pl.roads & t.node_edge_mask[n] != 0
            {
                out.push(Action::BuildSettlement(n as u8));
            }
        }
    }
    if covers(&pl.hand, &CITY_COST) && pl.cities.count_ones() < MAX_CITIES {
        for n in bits64(pl.settlements) {
            out.push(Action::BuildCity(n));
        }
    }
}

fn pay(s: &mut State, p: usize, cost: &Hand) {
    hand_sub(&mut s.players[p].hand, cost);
    hand_add(&mut s.bank, cost);
}

pub fn apply_build_road<S: EventSink>(s: &mut State, e: u8, free: bool, sink: &mut S) {
    let p = s.current as usize;
    if !free {
        pay(s, p, &ROAD_COST);
    }
    s.players[p].roads |= 1u128 << e;
    sink.emit(Event::BuiltRoad { player: s.current, edge: e });
}

pub fn apply_build_settlement<S: EventSink>(s: &mut State, n: u8, sink: &mut S) {
    let p = s.current as usize;
    pay(s, p, &SETTLEMENT_COST);
    s.players[p].settlements |= 1u64 << n;
    sink.emit(Event::BuiltSettlement { player: s.current, node: n });
}

pub fn apply_build_city<S: EventSink>(s: &mut State, n: u8, sink: &mut S) {
    let p = s.current as usize;
    pay(s, p, &CITY_COST);
    s.players[p].settlements &= !(1u64 << n);
    s.players[p].cities |= 1u64 << n;
    sink.emit(Event::BuiltCity { player: s.current, node: n });
}

pub fn apply_end_turn<S: EventSink>(s: &mut State, sink: &mut S) {
    let p = s.current as usize;
    s.players[p].dev_new = [0; 5];
    s.dev_played_this_turn = false;
    s.offers_this_turn = 0;
    sink.emit(Event::TurnEnded { player: s.current });
    s.current = ((p + 1) % NUM_PLAYERS) as PlayerId;
    s.turn += 1;
    s.phase = Phase::PreRoll;
    if s.turn >= s.config.max_turns {
        s.phase = Phase::GameOver { winner: None };
        sink.emit(Event::GameOver { winner: None });
    }
}
```

In `engine/src/legal.rs`, change the import to `use crate::rules::{build, robber, roll, setup};`, remove `Phase::Main` from the last (empty) arm, and add:

```rust
        Phase::Main => {
            build::legal_main_builds(s, out);
            out.push(Action::EndTurn);
        }
```

In `engine/src/apply.rs`, change the import to `use crate::rules::{build, robber, roll, setup};` and add before the panic arm:

```rust
            (Phase::Main, Action::BuildRoad(e)) => build::apply_build_road(self, e, false, sink),
            (Phase::Main, Action::BuildSettlement(n)) => build::apply_build_settlement(self, n, sink),
            (Phase::Main, Action::BuildCity(n)) => build::apply_build_city(self, n, sink),
            (Phase::Main, Action::EndTurn) => build::apply_end_turn(self, sink),
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p settler-engine`
Expected: PASS — `build` (9 tests) plus earlier suites.

- [ ] **Step 5: Commit**

```bash
git add engine
git diff --cached | grep -nE 'sk-|AKIA|ghp_|password=' && echo "SECRET FOUND - abort" || git commit -m "engine: building and end turn"
git push
```

---

### Task 8: Longest road, largest army, and winning

**Files:**
- Create: `engine/src/rules/awards.rs`
- Modify: `engine/src/rules/mod.rs`, `engine/src/rules/build.rs`, `engine/src/apply.rs`
- Test: `engine/tests/awards.rs`

**Interfaces:**
- Consumes: `State`, `build::*`, test helpers.
- Produces: `rules::awards::{longest_road(roads: u128, blocked: u64) -> u8, update_longest_road(s: &mut State), update_largest_army(s: &mut State, p: usize), check_win<S: EventSink>(s: &mut State, sink: &mut S)}`.

- [ ] **Step 1: Write the failing test**

`engine/tests/awards.rs`:

```rust
mod common;
use common::*;
use settler_engine::rules::awards::*;
use settler_engine::topology::topo;
use settler_engine::*;

#[test]
fn straight_path_length() {
    let nodes = path_nodes(0, 6, 0);
    assert_eq!(longest_road(mask128(&path_edges(&nodes)), 0), 6);
}

#[test]
fn opponent_building_splits_road() {
    let nodes = path_nodes(0, 6, 0);
    let roads = mask128(&path_edges(&nodes));
    assert_eq!(longest_road(roads, 1u64 << nodes[3]), 3);
    assert_eq!(longest_road(roads, 1u64 << nodes[1]), 5);
}

#[test]
fn ring_counts_every_edge() {
    let ring = topo().tile_nodes[9];
    let mut nodes = ring.to_vec();
    nodes.push(ring[0]);
    assert_eq!(longest_road(mask128(&path_edges(&nodes)), 0), 6);
}

fn two_roads(len0: usize, len1: usize) -> State {
    let mut s = blank(1, Phase::Main);
    let n0 = path_nodes(0, len0, 0);
    let avoid = n0.iter().fold(0u64, |m, &n| m | (1u64 << n) | topo().node_neighbor_mask[n as usize]);
    let n1 = path_nodes(53, len1, avoid);
    s.players[0].roads = mask128(&path_edges(&n0));
    s.players[1].roads = mask128(&path_edges(&n1));
    s
}

#[test]
fn award_needs_five() {
    let mut s = two_roads(4, 3);
    update_longest_road(&mut s);
    assert_eq!(s.longest_road_owner, None);
    let mut s = two_roads(5, 3);
    update_longest_road(&mut s);
    assert_eq!(s.longest_road_owner, Some(0));
    assert_eq!(s.public_vp(0), 2);
    assert_eq!(s.players[0].longest_road_len, 5);
}

#[test]
fn holder_keeps_award_on_tie() {
    let mut s = two_roads(5, 5);
    s.longest_road_owner = Some(1);
    update_longest_road(&mut s);
    assert_eq!(s.longest_road_owner, Some(1));
}

#[test]
fn longer_road_takes_award() {
    let mut s = two_roads(6, 5);
    s.longest_road_owner = Some(1);
    update_longest_road(&mut s);
    assert_eq!(s.longest_road_owner, Some(0));
}

#[test]
fn broken_holder_with_tied_challengers_leaves_award_unowned() {
    let mut s = two_roads(5, 5);
    s.longest_road_owner = Some(2);
    s.players[2].roads = 0; // holder's road was broken below everyone else
    update_longest_road(&mut s);
    assert_eq!(s.longest_road_owner, None);
}

#[test]
fn settlement_breaking_road_moves_award() {
    let mut s = two_roads(6, 5);
    update_longest_road(&mut s);
    assert_eq!(s.longest_road_owner, Some(0));
    // Player 1 reaches a middle node of player 0's road with a spur and settles there.
    // Cutting a 6-road at index 2, 3, or 4 leaves at most 4 on either side.
    let n0 = path_nodes(0, 6, 0);
    let t = topo();
    let (cut, spur) = (2..=4)
        .find_map(|i| {
            let n = n0[i];
            t.node_edges[n as usize]
                .iter()
                .copied()
                .find(|&e| s.players[0].roads & (1u128 << e) == 0)
                .map(|e| (n, e))
        })
        .expect("a middle node of the path with a third edge");
    s.current = 1;
    s.players[1].roads |= 1u128 << spur;
    give(&mut s, 1, SETTLEMENT_COST);
    s.apply(Action::BuildSettlement(cut));
    assert!(s.players[0].longest_road_len <= 4);
    assert_eq!(s.longest_road_owner, Some(1));
}

#[test]
fn largest_army() {
    let mut s = blank(1, Phase::Main);
    s.players[0].knights_played = 2;
    update_largest_army(&mut s, 0);
    assert_eq!(s.largest_army_owner, None);
    s.players[0].knights_played = 3;
    update_largest_army(&mut s, 0);
    assert_eq!(s.largest_army_owner, Some(0));
    s.players[1].knights_played = 3;
    update_largest_army(&mut s, 1);
    assert_eq!(s.largest_army_owner, Some(0));
    s.players[1].knights_played = 4;
    update_largest_army(&mut s, 1);
    assert_eq!(s.largest_army_owner, Some(1));
}

#[test]
fn building_to_target_vp_wins() {
    let c = GameConfig { vp_to_win: 3, ..cfg() };
    let mut s = blank_with(1, Phase::Main, c);
    let nodes = path_nodes(20, 2, 0);
    s.players[0].roads = mask128(&path_edges(&nodes));
    s.players[0].cities = 1u64 << 0;
    give(&mut s, 0, SETTLEMENT_COST);
    s.apply(Action::BuildSettlement(nodes[2]));
    assert_eq!(s.phase, Phase::GameOver { winner: Some(0) });
}

#[test]
fn players_win_only_on_their_own_turn() {
    let c = GameConfig { vp_to_win: 3, ..cfg() };
    let mut s = blank_with(1, Phase::Main, c);
    s.players[1].cities = 1u64 << 0;
    s.players[1].settlements = 1u64 << 30;
    s.players[0].settlements = 1u64 << 20;
    give(&mut s, 0, CITY_COST);
    s.apply(Action::BuildCity(20));
    assert_eq!(s.phase, Phase::Main);
    s.apply(Action::EndTurn);
    assert_eq!(s.phase, Phase::GameOver { winner: Some(1) });
}

#[test]
fn road_build_updates_longest_road() {
    let mut s = blank(1, Phase::Main);
    let nodes = path_nodes(0, 5, 0);
    let edges = path_edges(&nodes);
    s.players[0].settlements = 1u64 << nodes[0];
    s.players[0].roads = mask128(&edges[..4]);
    give(&mut s, 0, ROAD_COST);
    s.apply(Action::BuildRoad(edges[4]));
    assert_eq!(s.longest_road_owner, Some(0));
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p settler-engine --test awards`
Expected: FAIL to compile — `could not find awards`.

- [ ] **Step 3: Write the implementation**

Add `pub mod awards;` to `engine/src/rules/mod.rs`.

`engine/src/rules/awards.rs`:

```rust
//! Longest road, largest army, and the win check.

use crate::apply::EventSink;
use crate::events::Event;
use crate::state::{Phase, State};
use crate::topology::topo;
use crate::types::*;

/// Longest trail (no edge reused) in `roads`; trails may end at but not pass through `blocked` nodes.
pub fn longest_road(roads: u128, blocked: u64) -> u8 {
    let t = topo();
    let mut best = 0;
    for e in bits128(roads) {
        let (a, b) = t.edge_nodes[e as usize];
        let rest = roads & !(1u128 << e);
        best = best.max(extend(a, rest, blocked, 1)).max(extend(b, rest, blocked, 1));
    }
    best
}

fn extend(node: u8, roads: u128, blocked: u64, len: u8) -> u8 {
    if blocked & (1u64 << node) != 0 {
        return len;
    }
    let t = topo();
    let mut best = len;
    for e in bits128(roads & t.node_edge_mask[node as usize]) {
        let (a, b) = t.edge_nodes[e as usize];
        let next = if a == node { b } else { a };
        best = best.max(extend(next, roads & !(1u128 << e), blocked, len + 1));
    }
    best
}

/// Recompute every player's longest road and reassign the award (min 5; holder keeps ties).
pub fn update_longest_road(s: &mut State) {
    let occ = s.occupied_nodes();
    let mut lens = [0u8; NUM_PLAYERS];
    for p in 0..NUM_PLAYERS {
        let blocked = occ & !s.buildings(p);
        lens[p] = longest_road(s.players[p].roads, blocked);
        s.players[p].longest_road_len = lens[p];
    }
    let max = *lens.iter().max().unwrap();
    if max < 5 {
        s.longest_road_owner = None;
        return;
    }
    if let Some(h) = s.longest_road_owner {
        if lens[h as usize] == max {
            return;
        }
    }
    let leaders = lens.iter().filter(|&&l| l == max).count();
    s.longest_road_owner = if leaders == 1 {
        Some(lens.iter().position(|&l| l == max).unwrap() as PlayerId)
    } else {
        None
    };
}

/// After player `p` plays a knight: award at 3+, taken only by strictly more knights.
pub fn update_largest_army(s: &mut State, p: usize) {
    let k = s.players[p].knights_played;
    if k < 3 {
        return;
    }
    match s.largest_army_owner {
        None => s.largest_army_owner = Some(p as PlayerId),
        Some(h) if h as usize != p && k > s.players[h as usize].knights_played => {
            s.largest_army_owner = Some(p as PlayerId)
        }
        _ => {}
    }
}

/// The current player wins as soon as they reach the target on their own turn.
pub fn check_win<S: EventSink>(s: &mut State, sink: &mut S) {
    if s.is_over() {
        return;
    }
    let p = s.current;
    if s.total_vp(p as usize) >= s.config.vp_to_win {
        s.phase = Phase::GameOver { winner: Some(p) };
        sink.emit(Event::GameOver { winner: Some(p) });
    }
}
```

In `engine/src/rules/build.rs`, add `use crate::rules::awards;` to the imports, then add `awards::update_longest_road(s);` as the last line of both `apply_build_road` and `apply_build_settlement`.

In `engine/src/apply.rs`, change the import to `use crate::rules::{awards, build, robber, roll, setup};` and add this line after the closing brace of the `match` in `apply_with`:

```rust
        awards::check_win(self, sink);
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p settler-engine`
Expected: PASS — `awards` (12 tests) plus earlier suites.

- [ ] **Step 5: Commit**

```bash
git add engine
git diff --cached | grep -nE 'sk-|AKIA|ghp_|password=' && echo "SECRET FOUND - abort" || git commit -m "engine: longest road, largest army, win check"
git push
```

---

### Task 9: Development cards

**Files:**
- Create: `engine/src/rules/dev.rs`
- Modify: `engine/src/rules/mod.rs`, `engine/src/legal.rs`, `engine/src/apply.rs`
- Test: `engine/tests/dev.rs`

**Interfaces:**
- Consumes: `State`, `Chance::Dev`, `build::{push_free_roads, has_free_road, apply_build_road}`, `awards::update_largest_army`, test helpers.
- Produces: `rules::dev::{legal_knight, legal_main_dev, legal_road_building, apply_buy_dev, apply_play_knight, apply_play_road_building, apply_road_building_road, apply_year_of_plenty, apply_monopoly}`.

- [ ] **Step 1: Write the failing test**

`engine/tests/dev.rs`:

```rust
mod common;
use common::*;
use settler_engine::apply::{Chance, NoEvents};
use settler_engine::types::DevCard::*;
use settler_engine::*;

fn has(s: &State, a: Action) -> bool {
    s.legal_actions().contains(&a)
}

#[test]
fn buy_dev_pays_and_draws_top_card() {
    let mut s = blank(1, Phase::Main);
    give(&mut s, 0, DEV_COST);
    let top = s.dev_deck[0];
    assert!(has(&s, Action::BuyDev));
    s.apply(Action::BuyDev);
    assert_eq!(s.players[0].dev_hand[top.index()], 1);
    assert_eq!(s.players[0].dev_new[top.index()], 1);
    assert_eq!(s.dev_deck_pos, 1);
    assert_eq!(s.players[0].hand, [0; 5]);
}

#[test]
fn forced_dev_draw() {
    let mut s = blank(1, Phase::Main);
    give(&mut s, 0, DEV_COST);
    s.apply_with(Action::BuyDev, Some(Chance::Dev(Monopoly)), &mut NoEvents);
    assert_eq!(s.players[0].dev_hand[Monopoly.index()], 1);
}

#[test]
#[should_panic(expected = "not left in deck")]
fn forced_dev_card_missing_panics() {
    let mut s = blank(1, Phase::Main);
    s.dev_deck = [Knight; 25];
    give(&mut s, 0, DEV_COST);
    s.apply_with(Action::BuyDev, Some(Chance::Dev(Monopoly)), &mut NoEvents);
}

#[test]
fn empty_deck_cannot_buy() {
    let mut s = blank(1, Phase::Main);
    s.dev_deck_pos = 25;
    give(&mut s, 0, DEV_COST);
    assert!(!has(&s, Action::BuyDev));
}

#[test]
fn cannot_play_card_bought_this_turn() {
    let mut s = blank(1, Phase::Main);
    give(&mut s, 0, DEV_COST);
    s.apply_with(Action::BuyDev, Some(Chance::Dev(Knight)), &mut NoEvents);
    assert!(!has(&s, Action::PlayKnight));
    s.apply(Action::EndTurn);
    assert_eq!(s.players[0].dev_new, [0; 5]);
    s.current = 0;
    s.phase = Phase::Main;
    assert!(has(&s, Action::PlayKnight));
}

#[test]
fn one_dev_card_per_turn() {
    let mut s = blank(1, Phase::Main);
    s.players[0].dev_hand[Knight.index()] = 1;
    s.players[0].dev_hand[Monopoly.index()] = 1;
    assert!(has(&s, Action::PlayKnight));
    assert!(has(&s, Action::PlayMonopoly(Resource::Ore)));
    s.apply(Action::PlayMonopoly(Resource::Ore));
    assert!(!has(&s, Action::PlayKnight));
}

#[test]
fn knight_before_roll_returns_to_preroll() {
    let mut s = blank(1, Phase::PreRoll);
    s.players[0].dev_hand[Knight.index()] = 1;
    assert_eq!(sorted(s.legal_actions()), vec![Action::Roll, Action::PlayKnight]);
    s.apply(Action::PlayKnight);
    assert_eq!(s.phase, Phase::MoveRobber);
    assert_eq!(s.players[0].knights_played, 1);
    let tile = s.legal_actions()[0];
    s.apply(tile);
    assert_eq!(s.phase, Phase::PreRoll);
    assert_eq!(s.legal_actions(), vec![Action::Roll]);
}

#[test]
fn knight_after_roll_returns_to_main() {
    let mut s = blank(1, Phase::Main);
    s.players[0].dev_hand[Knight.index()] = 1;
    s.apply(Action::PlayKnight);
    let tile = s.legal_actions()[0];
    s.apply(tile);
    assert_eq!(s.phase, Phase::Main);
}

#[test]
fn third_knight_takes_largest_army() {
    let mut s = blank(1, Phase::Main);
    s.players[0].dev_hand[Knight.index()] = 1;
    s.players[0].knights_played = 2;
    s.apply(Action::PlayKnight);
    assert_eq!(s.largest_army_owner, Some(0));
}

#[test]
fn road_building_places_two_free_roads() {
    let mut s = blank(1, Phase::Main);
    s.players[0].settlements = 1u64 << 20;
    s.players[0].dev_hand[RoadBuilding.index()] = 1;
    s.apply(Action::PlayRoadBuilding);
    assert_eq!(s.phase, Phase::RoadBuilding { roads_left: 2 });
    let a = s.legal_actions()[0];
    s.apply(a);
    assert_eq!(s.phase, Phase::RoadBuilding { roads_left: 1 });
    let b = s.legal_actions()[0];
    s.apply(b);
    assert_eq!(s.phase, Phase::Main);
    assert_eq!(s.players[0].roads.count_ones(), 2);
    assert_eq!(s.bank, [19; 5]);
}

#[test]
fn road_building_with_no_legal_road_returns_to_main() {
    let mut s = blank(1, Phase::Main);
    s.players[0].dev_hand[RoadBuilding.index()] = 1;
    s.apply(Action::PlayRoadBuilding);
    assert_eq!(s.phase, Phase::Main);
    assert_eq!(s.players[0].dev_hand[RoadBuilding.index()], 0);

    let mut s = blank(1, Phase::Main);
    s.players[0].settlements = 1u64 << settler_engine::topology::topo().edge_nodes[0].0;
    s.players[0].roads = (1u128 << 14) - 1; // 14 roads: only one piece left
    s.players[0].dev_hand[RoadBuilding.index()] = 1;
    s.apply(Action::PlayRoadBuilding);
    assert_eq!(s.phase, Phase::RoadBuilding { roads_left: 1 });
    let a = s.legal_actions()[0];
    s.apply(a);
    assert_eq!(s.phase, Phase::Main);
}

#[test]
fn year_of_plenty() {
    let mut s = blank(1, Phase::Main);
    s.players[0].dev_hand[YearOfPlenty.index()] = 1;
    s.apply(Action::PlayYearOfPlenty(Resource::Wood, Resource::Ore));
    assert_eq!(s.players[0].hand, [1, 0, 0, 0, 1]);
    assert_eq!(s.bank, [18, 19, 19, 19, 18]);
}

#[test]
fn year_of_plenty_respects_bank() {
    let mut s = blank(1, Phase::Main);
    s.players[0].dev_hand[YearOfPlenty.index()] = 1;
    s.bank[Resource::Wheat.index()] = 1;
    assert!(has(&s, Action::PlayYearOfPlenty(Resource::Wheat, Resource::Ore)));
    assert!(!has(&s, Action::PlayYearOfPlenty(Resource::Wheat, Resource::Wheat)));
}

#[test]
fn monopoly_takes_all() {
    let mut s = blank(1, Phase::Main);
    s.players[0].dev_hand[Monopoly.index()] = 1;
    give(&mut s, 1, [3, 1, 0, 0, 0]);
    give(&mut s, 2, [2, 0, 0, 0, 0]);
    s.apply(Action::PlayMonopoly(Resource::Wood));
    assert_eq!(s.players[0].hand, [5, 0, 0, 0, 0]);
    assert_eq!(s.players[1].hand, [0, 1, 0, 0, 0]);
    assert_eq!(s.players[2].hand, [0; 5]);
}

#[test]
fn victory_point_cards_are_hidden_and_never_played() {
    let mut s = blank(1, Phase::Main);
    s.players[0].dev_hand[VictoryPoint.index()] = 2;
    assert_eq!(s.total_vp(0) - s.public_vp(0), 2);
    assert_eq!(s.legal_actions(), vec![Action::EndTurn]);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p settler-engine --test dev`
Expected: FAIL — `BuyDev`/`PlayKnight` missing from legal actions, and `illegal action` panics.

- [ ] **Step 3: Write the implementation**

Add `pub mod dev;` to `engine/src/rules/mod.rs`.

`engine/src/rules/dev.rs`:

```rust
//! Development cards: buying, and playing knight, road building, year of plenty, monopoly.

use crate::action::{Action, PAIRS};
use crate::apply::{Chance, EventSink};
use crate::events::Event;
use crate::rules::{awards, build};
use crate::state::{Phase, State};
use crate::types::*;

/// Not yet played a dev card this turn, and holds one of `card` not bought this turn.
fn playable(s: &State, card: DevCard) -> bool {
    let pl = &s.players[s.current as usize];
    !s.dev_played_this_turn && pl.dev_hand[card.index()] > pl.dev_new[card.index()]
}

pub fn legal_knight(s: &State, out: &mut Vec<Action>) {
    if playable(s, DevCard::Knight) {
        out.push(Action::PlayKnight);
    }
}

pub fn legal_main_dev(s: &State, out: &mut Vec<Action>) {
    let pl = &s.players[s.current as usize];
    if covers(&pl.hand, &DEV_COST) && (s.dev_deck_pos as usize) < DEV_DECK_SIZE {
        out.push(Action::BuyDev);
    }
    legal_knight(s, out);
    if playable(s, DevCard::RoadBuilding) {
        out.push(Action::PlayRoadBuilding);
    }
    if playable(s, DevCard::YearOfPlenty) {
        for &(a, b) in &PAIRS {
            let ok = if a == b {
                s.bank[a as usize] >= 2
            } else {
                s.bank[a as usize] >= 1 && s.bank[b as usize] >= 1
            };
            if ok {
                out.push(Action::PlayYearOfPlenty(
                    Resource::from_index(a as usize),
                    Resource::from_index(b as usize),
                ));
            }
        }
    }
    if playable(s, DevCard::Monopoly) {
        for r in Resource::ALL {
            out.push(Action::PlayMonopoly(r));
        }
    }
}

pub fn legal_road_building(s: &State, out: &mut Vec<Action>) {
    build::push_free_roads(s, s.current as usize, out);
}

fn use_card<S: EventSink>(s: &mut State, card: DevCard, sink: &mut S) {
    s.players[s.current as usize].dev_hand[card.index()] -= 1;
    s.dev_played_this_turn = true;
    sink.emit(Event::PlayedDev { player: s.current, card });
}

pub fn apply_buy_dev<S: EventSink>(s: &mut State, chance: Option<Chance>, sink: &mut S) {
    let p = s.current as usize;
    hand_sub(&mut s.players[p].hand, &DEV_COST);
    hand_add(&mut s.bank, &DEV_COST);
    let pos = s.dev_deck_pos as usize;
    match chance {
        None => {}
        Some(Chance::Dev(card)) => {
            let j = (pos..DEV_DECK_SIZE)
                .find(|&j| s.dev_deck[j] == card)
                .unwrap_or_else(|| panic!("forced dev card {card:?} not left in deck"));
            s.dev_deck.swap(pos, j);
        }
        Some(c) => panic!("chance {c:?} does not match BuyDev"),
    }
    let card = s.dev_deck[pos];
    s.dev_deck_pos += 1;
    s.players[p].dev_hand[card.index()] += 1;
    s.players[p].dev_new[card.index()] += 1;
    sink.emit(Event::BoughtDev { player: s.current, card: Some(card) });
}

pub fn apply_play_knight<S: EventSink>(s: &mut State, sink: &mut S) {
    let p = s.current as usize;
    let from_preroll = s.phase == Phase::PreRoll;
    use_card(s, DevCard::Knight, sink);
    s.players[p].knights_played += 1;
    awards::update_largest_army(s, p);
    s.robber_return = if from_preroll { Phase::PreRoll } else { Phase::Main };
    s.phase = Phase::MoveRobber;
}

fn road_building_phase(s: &State, left: u8) -> Phase {
    if left > 0 && build::has_free_road(s, s.current as usize) {
        Phase::RoadBuilding { roads_left: left }
    } else {
        Phase::Main
    }
}

pub fn apply_play_road_building<S: EventSink>(s: &mut State, sink: &mut S) {
    use_card(s, DevCard::RoadBuilding, sink);
    let pieces_left = MAX_ROADS - s.players[s.current as usize].roads.count_ones();
    s.phase = road_building_phase(s, pieces_left.min(2) as u8);
}

pub fn apply_road_building_road<S: EventSink>(s: &mut State, e: u8, roads_left: u8, sink: &mut S) {
    build::apply_build_road(s, e, true, sink);
    s.phase = road_building_phase(s, roads_left - 1);
}

pub fn apply_year_of_plenty<S: EventSink>(s: &mut State, a: Resource, b: Resource, sink: &mut S) {
    use_card(s, DevCard::YearOfPlenty, sink);
    let mut got = [0u8; NUM_RESOURCES];
    got[a.index()] += 1;
    got[b.index()] += 1;
    hand_add(&mut s.players[s.current as usize].hand, &got);
    hand_sub(&mut s.bank, &got);
    sink.emit(Event::YearOfPlentyTaken { player: s.current, resources: got });
}

pub fn apply_monopoly<S: EventSink>(s: &mut State, r: Resource, sink: &mut S) {
    use_card(s, DevCard::Monopoly, sink);
    let p = s.current as usize;
    let mut amount = 0u8;
    for q in 0..NUM_PLAYERS {
        if q != p {
            amount += s.players[q].hand[r.index()];
            s.players[q].hand[r.index()] = 0;
        }
    }
    s.players[p].hand[r.index()] += amount;
    sink.emit(Event::MonopolyTaken { player: s.current, resource: r, amount });
}
```

In `engine/src/legal.rs`, change the import to `use crate::rules::{build, dev, robber, roll, setup};` and update three arms:

```rust
        Phase::PreRoll => {
            out.push(Action::Roll);
            dev::legal_knight(s, out);
        }
```

```rust
        Phase::Main => {
            build::legal_main_builds(s, out);
            dev::legal_main_dev(s, out);
            out.push(Action::EndTurn);
        }
```

and move `Phase::RoadBuilding { .. }` out of the empty arm into:

```rust
        Phase::RoadBuilding { .. } => dev::legal_road_building(s, out),
```

In `engine/src/apply.rs`, change the import to `use crate::rules::{awards, build, dev, robber, roll, setup};` and add before the panic arm:

```rust
            (Phase::PreRoll | Phase::Main, Action::PlayKnight) => dev::apply_play_knight(self, sink),
            (Phase::Main, Action::BuyDev) => dev::apply_buy_dev(self, chance, sink),
            (Phase::Main, Action::PlayRoadBuilding) => dev::apply_play_road_building(self, sink),
            (Phase::Main, Action::PlayYearOfPlenty(a, b)) => dev::apply_year_of_plenty(self, a, b, sink),
            (Phase::Main, Action::PlayMonopoly(r)) => dev::apply_monopoly(self, r, sink),
            (Phase::RoadBuilding { roads_left }, Action::BuildRoad(e)) => {
                dev::apply_road_building_road(self, e, roads_left, sink)
            }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p settler-engine`
Expected: PASS — `dev` (15 tests) plus earlier suites.

- [ ] **Step 5: Commit**

```bash
git add engine
git diff --cached | grep -nE 'sk-|AKIA|ghp_|password=' && echo "SECRET FOUND - abort" || git commit -m "engine: development cards"
git push
```

---

### Task 10: Maritime (bank and port) trades

**Files:**
- Create: `engine/src/rules/maritime.rs`
- Modify: `engine/src/rules/mod.rs`, `engine/src/legal.rs`, `engine/src/apply.rs`
- Test: `engine/tests/maritime.rs`

**Interfaces:**
- Consumes: `Board::maritime_rate`, `State::buildings`.
- Produces: `rules::maritime::{legal(s, out), apply(s, give: Resource, get: Resource, sink)}`.

- [ ] **Step 1: Write the failing test**

`engine/tests/maritime.rs`:

```rust
mod common;
use common::*;
use settler_engine::topology::topo;
use settler_engine::*;
use Resource::*;

fn maritime(s: &State) -> Vec<Action> {
    sorted(s.legal_actions().into_iter().filter(|a| matches!(a, Action::MaritimeTrade { .. })).collect())
}

#[test]
fn default_rate_is_four() {
    let mut s = blank(1, Phase::Main);
    give(&mut s, 0, [3, 0, 0, 0, 0]);
    assert!(maritime(&s).is_empty());
    give(&mut s, 0, [1, 0, 0, 0, 0]);
    let expected: Vec<Action> =
        [Brick, Sheep, Wheat, Ore].iter().map(|&g| Action::MaritimeTrade { give: Wood, get: g }).collect();
    assert_eq!(maritime(&s), sorted(expected));
    s.apply(Action::MaritimeTrade { give: Wood, get: Ore });
    assert_eq!(s.players[0].hand, [0, 0, 0, 0, 1]);
    assert_eq!(s.bank, [19, 19, 19, 19, 18]);
}

#[test]
fn generic_port_is_three() {
    let mut s = blank(1, Phase::Main);
    let (e, _) = *s.board.ports.iter().find(|p| p.1 == PortKind::Generic).unwrap();
    s.players[0].settlements = 1u64 << topo().edge_nodes[e as usize].0;
    give(&mut s, 0, [0, 0, 3, 0, 0]);
    assert_eq!(maritime(&s).len(), 4);
    s.apply(Action::MaritimeTrade { give: Sheep, get: Wood });
    assert_eq!(s.players[0].hand, [1, 0, 0, 0, 0]);
}

#[test]
fn specific_port_is_two_for_its_resource_only() {
    let mut s = blank(1, Phase::Main);
    let (e, kind) = *s.board.ports.iter().find(|p| p.1 != PortKind::Generic).unwrap();
    let PortKind::Specific(r) = kind else { unreachable!() };
    s.players[0].cities = 1u64 << topo().edge_nodes[e as usize].1;
    let other = Resource::ALL.into_iter().find(|&x| x != r).unwrap();
    let mut h = [0u8; 5];
    h[r.index()] = 2;
    h[other.index()] = 2;
    give(&mut s, 0, h);
    let legal = maritime(&s);
    assert!(legal.iter().all(|a| matches!(a, Action::MaritimeTrade { give, .. } if *give == r)));
    assert_eq!(legal.len(), 4);
}

#[test]
fn cannot_take_from_empty_bank_pile() {
    let mut s = blank(1, Phase::Main);
    give(&mut s, 0, [4, 0, 0, 0, 0]);
    s.bank[Ore.index()] = 0;
    assert!(!maritime(&s).contains(&Action::MaritimeTrade { give: Wood, get: Ore }));
    assert_eq!(maritime(&s).len(), 3);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p settler-engine --test maritime`
Expected: FAIL — `maritime(&s)` is empty where trades are expected.

- [ ] **Step 3: Write the implementation**

Add `pub mod maritime;` to `engine/src/rules/mod.rs`.

`engine/src/rules/maritime.rs`:

```rust
//! Trades with the bank at 4:1, or 3:1 / 2:1 through ports.

use crate::action::Action;
use crate::apply::EventSink;
use crate::events::Event;
use crate::state::State;
use crate::types::*;

pub fn legal(s: &State, out: &mut Vec<Action>) {
    let p = s.current as usize;
    let buildings = s.buildings(p);
    for give in Resource::ALL {
        let rate = s.board.maritime_rate(buildings, give);
        if s.players[p].hand[give.index()] < rate {
            continue;
        }
        for get in Resource::ALL {
            if get != give && s.bank[get.index()] >= 1 {
                out.push(Action::MaritimeTrade { give, get });
            }
        }
    }
}

pub fn apply<S: EventSink>(s: &mut State, give: Resource, get: Resource, sink: &mut S) {
    let p = s.current as usize;
    let rate = s.board.maritime_rate(s.buildings(p), give);
    let mut gave = [0u8; NUM_RESOURCES];
    gave[give.index()] = rate;
    let mut got = [0u8; NUM_RESOURCES];
    got[get.index()] = 1;
    hand_sub(&mut s.players[p].hand, &gave);
    hand_add(&mut s.bank, &gave);
    hand_sub(&mut s.bank, &got);
    hand_add(&mut s.players[p].hand, &got);
    sink.emit(Event::MaritimeTraded { player: s.current, gave, got });
}
```

In `engine/src/legal.rs`, change the import to `use crate::rules::{build, dev, maritime, robber, roll, setup};` and in the `Phase::Main` arm add `maritime::legal(s, out);` after `dev::legal_main_dev(s, out);`.

In `engine/src/apply.rs`, change the import to include `maritime` and add before the panic arm:

```rust
            (Phase::Main, Action::MaritimeTrade { give, get }) => maritime::apply(self, give, get, sink),
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p settler-engine`
Expected: PASS — `maritime` (4 tests) plus earlier suites.

- [ ] **Step 5: Commit**

```bash
git add engine
git diff --cached | grep -nE 'sk-|AKIA|ghp_|password=' && echo "SECRET FOUND - abort" || git commit -m "engine: maritime trades"
git push
```

---

### Task 11: Domestic trade protocol

**Files:**
- Create: `engine/src/rules/trade.rs`
- Modify: `engine/src/rules/mod.rs`, `engine/src/legal.rs`, `engine/src/apply.rs`
- Test: `engine/tests/trade.rs`

**Interfaces:**
- Consumes: `action::{bundle, bundle_size, NUM_BUNDLES}`, `state::{PendingTrade, Response}`, `GameConfig::{max_offers_per_turn, max_trade_cards}`.
- Produces: `rules::trade::{legal_offers, apply_offer(s, give: u8, get: u8, sink), legal_response, apply_response(s, accepted: bool, sink), legal_confirm, apply_confirm(s, partner: PlayerId, sink), apply_cancel(s, sink)}`.

- [ ] **Step 1: Write the failing test**

`engine/tests/trade.rs`:

```rust
mod common;
use common::*;
use settler_engine::*;

fn offers(s: &State) -> Vec<Action> {
    s.legal_actions().into_iter().filter(|a| matches!(a, Action::OfferTrade { .. })).collect()
}

// Bundle 0 = one Wood, bundle 1 = one Brick.
const WOOD_FOR_BRICK: Action = Action::OfferTrade { give: 0, get: 1 };

#[test]
fn offers_require_holding_the_give_side() {
    let mut s = blank(1, Phase::Main);
    assert!(offers(&s).is_empty());
    give(&mut s, 0, [1, 0, 0, 0, 0]);
    let o = offers(&s);
    // give = 1 Wood; get = any bundle without Wood: 4 singles + 10 pairs.
    assert_eq!(o.len(), 14);
    assert!(o.iter().all(|a| matches!(a, Action::OfferTrade { give: 0, .. })));
}

#[test]
fn trade_size_cap() {
    let c = GameConfig { max_trade_cards: 1, ..cfg() };
    let mut s = blank_with(1, Phase::Main, c);
    give(&mut s, 0, [2, 0, 0, 0, 0]);
    assert_eq!(offers(&s).len(), 4);
}

#[test]
fn trading_disabled_with_zero_offers() {
    let c = GameConfig { max_offers_per_turn: 0, ..cfg() };
    let mut s = blank_with(1, Phase::Main, c);
    give(&mut s, 0, [1, 0, 0, 0, 0]);
    assert!(offers(&s).is_empty());
}

#[test]
fn accepted_trade_exchanges_cards() {
    let mut s = blank(1, Phase::Main);
    give(&mut s, 0, [1, 0, 0, 0, 0]);
    give(&mut s, 1, [0, 1, 0, 0, 0]);
    s.apply(WOOD_FOR_BRICK);
    assert_eq!(s.phase, Phase::TradeResponse);
    assert_eq!(s.current_actor(), 1);
    assert_eq!(sorted(s.legal_actions()), vec![Action::AcceptTrade, Action::RejectTrade]);
    s.apply(Action::AcceptTrade);
    assert_eq!(s.current_actor(), 2);
    assert_eq!(s.legal_actions(), vec![Action::RejectTrade]); // player 2 has no brick
    s.apply(Action::RejectTrade);
    s.apply(Action::RejectTrade);
    assert_eq!(s.phase, Phase::TradeConfirm);
    assert_eq!(s.current_actor(), 0);
    assert_eq!(sorted(s.legal_actions()), vec![Action::ConfirmTrade(1), Action::CancelTrade]);
    s.apply(Action::ConfirmTrade(1));
    assert_eq!(s.players[0].hand, [0, 1, 0, 0, 0]);
    assert_eq!(s.players[1].hand, [1, 0, 0, 0, 0]);
    assert_eq!(s.phase, Phase::Main);
    assert_eq!(s.trade, None);
    assert_eq!(s.offers_this_turn, 1);
}

#[test]
fn all_rejections_return_to_main() {
    let mut s = blank(1, Phase::Main);
    give(&mut s, 0, [1, 0, 0, 0, 0]);
    s.apply(WOOD_FOR_BRICK);
    for _ in 0..3 {
        s.apply(Action::RejectTrade);
    }
    assert_eq!(s.phase, Phase::Main);
    assert_eq!(s.trade, None);
    assert_eq!(s.players[0].hand, [1, 0, 0, 0, 0]);
}

#[test]
fn offerer_can_cancel_after_acceptance() {
    let mut s = blank(1, Phase::Main);
    give(&mut s, 0, [1, 0, 0, 0, 0]);
    give(&mut s, 1, [0, 1, 0, 0, 0]);
    s.apply(WOOD_FOR_BRICK);
    s.apply(Action::AcceptTrade);
    s.apply(Action::RejectTrade);
    s.apply(Action::RejectTrade);
    s.apply(Action::CancelTrade);
    assert_eq!(s.phase, Phase::Main);
    assert_eq!(s.players[0].hand, [1, 0, 0, 0, 0]);
    assert_eq!(s.players[1].hand, [0, 1, 0, 0, 0]);
}

#[test]
fn offer_cap_per_turn_resets_next_turn() {
    let c = GameConfig { max_offers_per_turn: 1, ..cfg() };
    let mut s = blank_with(1, Phase::Main, c);
    give(&mut s, 0, [1, 0, 0, 0, 0]);
    s.apply(WOOD_FOR_BRICK);
    for _ in 0..3 {
        s.apply(Action::RejectTrade);
    }
    assert!(offers(&s).is_empty());
    s.apply(Action::EndTurn);
    assert_eq!(s.offers_this_turn, 0);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p settler-engine --test trade`
Expected: FAIL — `offers(&s)` is empty, and `illegal action OfferTrade` panics.

- [ ] **Step 3: Write the implementation**

Add `pub mod trade;` to `engine/src/rules/mod.rs`.

`engine/src/rules/trade.rs`:

```rust
//! Player-to-player trades: offer, then each opponent accepts or rejects in seat order,
//! then the offerer confirms with one acceptor or cancels.

use crate::action::{bundle, bundle_size, Action, NUM_BUNDLES};
use crate::apply::EventSink;
use crate::events::Event;
use crate::state::{PendingTrade, Phase, Response, State};
use crate::types::*;

pub fn legal_offers(s: &State, out: &mut Vec<Action>) {
    if s.offers_this_turn >= s.config.max_offers_per_turn {
        return;
    }
    let cap = s.config.max_trade_cards;
    let hand = s.players[s.current as usize].hand;
    for gi in 0..NUM_BUNDLES as u8 {
        if bundle_size(gi) > cap {
            continue;
        }
        let g = bundle(gi);
        if !covers(&hand, &g) {
            continue;
        }
        for wi in 0..NUM_BUNDLES as u8 {
            if bundle_size(wi) > cap {
                continue;
            }
            let w = bundle(wi);
            if (0..NUM_RESOURCES).all(|r| g[r] == 0 || w[r] == 0) {
                out.push(Action::OfferTrade { give: gi, get: wi });
            }
        }
    }
}

pub fn apply_offer<S: EventSink>(s: &mut State, give: u8, get: u8, sink: &mut S) {
    let (give, get) = (bundle(give), bundle(get));
    s.offers_this_turn += 1;
    s.trade = Some(PendingTrade {
        give,
        get,
        responses: [Response::Pending; NUM_PLAYERS],
        next_responder: (s.current + 1) % NUM_PLAYERS as PlayerId,
    });
    s.phase = Phase::TradeResponse;
    sink.emit(Event::TradeOffered { player: s.current, give, get });
}

pub fn legal_response(s: &State, out: &mut Vec<Action>) {
    let tr = s.trade.expect("TradeResponse without a trade");
    if covers(&s.players[tr.next_responder as usize].hand, &tr.get) {
        out.push(Action::AcceptTrade);
    }
    out.push(Action::RejectTrade);
}

pub fn apply_response<S: EventSink>(s: &mut State, accepted: bool, sink: &mut S) {
    let mut tr = s.trade.expect("TradeResponse without a trade");
    let r = tr.next_responder;
    tr.responses[r as usize] = if accepted { Response::Accepted } else { Response::Rejected };
    sink.emit(Event::TradeResponded { player: r, accepted });
    let next = (r + 1) % NUM_PLAYERS as PlayerId;
    if next != s.current {
        tr.next_responder = next;
        s.trade = Some(tr);
    } else if tr.responses.contains(&Response::Accepted) {
        s.trade = Some(tr);
        s.phase = Phase::TradeConfirm;
    } else {
        s.trade = None;
        s.phase = Phase::Main;
        sink.emit(Event::TradeCancelled { player: s.current });
    }
}

pub fn legal_confirm(s: &State, out: &mut Vec<Action>) {
    let tr = s.trade.expect("TradeConfirm without a trade");
    for (p, &r) in tr.responses.iter().enumerate() {
        if r == Response::Accepted {
            out.push(Action::ConfirmTrade(p as PlayerId));
        }
    }
    out.push(Action::CancelTrade);
}

pub fn apply_confirm<S: EventSink>(s: &mut State, partner: PlayerId, sink: &mut S) {
    let tr = s.trade.take().expect("TradeConfirm without a trade");
    let (p, q) = (s.current as usize, partner as usize);
    hand_sub(&mut s.players[p].hand, &tr.give);
    hand_add(&mut s.players[q].hand, &tr.give);
    hand_sub(&mut s.players[q].hand, &tr.get);
    hand_add(&mut s.players[p].hand, &tr.get);
    s.phase = Phase::Main;
    sink.emit(Event::TradeConfirmed {
        offerer: s.current,
        partner,
        offerer_gave: tr.give,
        partner_gave: tr.get,
    });
}

pub fn apply_cancel<S: EventSink>(s: &mut State, sink: &mut S) {
    s.trade = None;
    s.phase = Phase::Main;
    sink.emit(Event::TradeCancelled { player: s.current });
}
```

In `engine/src/legal.rs`, the final match (complete; replace the whole `match`) is:

```rust
    match s.phase {
        Phase::SetupSettlement => setup::legal_settlement(s, out),
        Phase::SetupRoad { node } => setup::legal_road(s, node, out),
        Phase::PreRoll => {
            out.push(Action::Roll);
            dev::legal_knight(s, out);
        }
        Phase::Discard => roll::legal_discard(s, out),
        Phase::MoveRobber => robber::legal_move_robber(s, out),
        Phase::Steal => robber::legal_steal(s, out),
        Phase::Main => {
            build::legal_main_builds(s, out);
            dev::legal_main_dev(s, out);
            maritime::legal(s, out);
            trade::legal_offers(s, out);
            out.push(Action::EndTurn);
        }
        Phase::RoadBuilding { .. } => dev::legal_road_building(s, out),
        Phase::TradeResponse => trade::legal_response(s, out),
        Phase::TradeConfirm => trade::legal_confirm(s, out),
        Phase::GameOver { .. } => {}
    }
```

with import `use crate::rules::{build, dev, maritime, robber, roll, setup, trade};`.

In `engine/src/apply.rs`, change the import to `use crate::rules::{awards, build, dev, maritime, robber, roll, setup, trade};` and add before the panic arm:

```rust
            (Phase::Main, Action::OfferTrade { give, get }) => trade::apply_offer(self, give, get, sink),
            (Phase::TradeResponse, Action::AcceptTrade) => trade::apply_response(self, true, sink),
            (Phase::TradeResponse, Action::RejectTrade) => trade::apply_response(self, false, sink),
            (Phase::TradeConfirm, Action::ConfirmTrade(p)) => trade::apply_confirm(self, p, sink),
            (Phase::TradeConfirm, Action::CancelTrade) => trade::apply_cancel(self, sink),
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p settler-engine`
Expected: PASS — `trade` (7 tests) plus earlier suites. `only_end_turn_without_resources` in `build` still passes (empty hand ⇒ no offers).

- [ ] **Step 5: Commit**

```bash
git add engine
git diff --cached | grep -nE 'sk-|AKIA|ghp_|password=' && echo "SECRET FOUND - abort" || git commit -m "engine: domestic trade protocol"
git push
```

---

### Task 12: Observation and the Game wrapper

**Files:**
- Create: `engine/src/observation.rs`, `engine/src/game.rs`
- Modify: `engine/src/lib.rs`
- Test: `engine/tests/game.rs`

**Interfaces:**
- Consumes: `State`, `Event::redacted_for`, `legal::legal_actions`, `Chance`.
- Produces:
  - `observation::Observation { viewer, phase, current, actor, turn, board, robber, bank, dev_deck_remaining, my_hand, my_dev_cards, my_new_dev_cards, hand_counts: [u8; 4], dev_card_counts: [u8; 4], settlements: [u64; 4], cities: [u64; 4], roads: [u128; 4], knights_played: [u8; 4], public_vp: [u8; 4], longest_road_owner, largest_army_owner, trade, dev_played_this_turn }` and `State::observation(&self, viewer: PlayerId) -> Observation`.
  - `game::Game`: `Game::new(seed, config)`, `Game::from_state(state)`, `state(&self) -> &State`, `legal_actions(&self) -> Vec<Action>`, `apply(&mut self, a) -> Result<(), IllegalAction>`, `apply_forced(&mut self, a, chance: Option<Chance>) -> Result<(), IllegalAction>`, `observation(&self, viewer) -> Observation`, `log_for(&self, viewer) -> Vec<Event>`. `game::IllegalAction { action: Action, phase: Phase }`.

- [ ] **Step 1: Write the failing test**

`engine/tests/game.rs`:

```rust
mod common;
use common::*;
use settler_engine::apply::Chance;
use settler_engine::*;

#[test]
fn observation_hides_other_hands_and_dev_cards() {
    let mut s = blank(1, Phase::Main);
    give(&mut s, 0, [1, 1, 0, 0, 0]);
    give(&mut s, 1, [2, 0, 1, 0, 0]);
    s.players[1].dev_hand[DevCard::VictoryPoint.index()] = 1;
    s.players[1].cities = 1u64 << 20;
    let o = s.observation(0);
    assert_eq!(o.my_hand, [1, 1, 0, 0, 0]);
    assert_eq!(o.hand_counts, [2, 3, 0, 0]);
    assert_eq!(o.dev_card_counts, [0, 1, 0, 0]);
    assert_eq!(o.public_vp[1], 2);
    assert_eq!(o.my_dev_cards, [0; 5]);
    let o1 = s.observation(1);
    assert_eq!(o1.my_hand, [2, 0, 1, 0, 0]);
    assert_eq!(o1.my_dev_cards[DevCard::VictoryPoint.index()], 1);
}

fn steal_game() -> Game {
    let mut s = blank(1, Phase::MoveRobber);
    let t = lonely_tile(&s) as u8;
    s.players[1].settlements = 1u64 << exclusive_nodes(t as usize)[0];
    give(&mut s, 1, [0, 0, 0, 0, 1]);
    let mut g = Game::from_state(s);
    g.apply(Action::MoveRobber(t)).unwrap();
    g.apply(Action::StealFrom(1)).unwrap();
    g
}

#[test]
fn steals_are_private_to_thief_and_victim() {
    let g = steal_game();
    let stole = |v: PlayerId| *g.log_for(v).iter().rev().find(|e| matches!(e, Event::Stole { .. })).unwrap();
    let full = Event::Stole { thief: 0, victim: 1, resource: Some(Resource::Ore) };
    assert_eq!(stole(0), full);
    assert_eq!(stole(1), full);
    assert_eq!(stole(2), Event::Stole { thief: 0, victim: 1, resource: None });
}

#[test]
fn bought_dev_cards_are_private() {
    let mut s = blank(1, Phase::Main);
    give(&mut s, 0, DEV_COST);
    let mut g = Game::from_state(s);
    g.apply_forced(Action::BuyDev, Some(Chance::Dev(DevCard::Monopoly))).unwrap();
    assert_eq!(g.log_for(0).last(), Some(&Event::BoughtDev { player: 0, card: Some(DevCard::Monopoly) }));
    assert_eq!(g.log_for(3).last(), Some(&Event::BoughtDev { player: 0, card: None }));
}

#[test]
fn illegal_action_is_rejected_without_changing_state() {
    let mut g = Game::new(1, cfg());
    let before = *g.state();
    let err = g.apply(Action::Roll).unwrap_err();
    assert_eq!(err, IllegalAction { action: Action::Roll, phase: Phase::SetupSettlement });
    assert_eq!(*g.state(), before);
    assert!(g.log_for(0).is_empty());
}

#[test]
fn actions_after_game_over_are_rejected() {
    let c = GameConfig { max_turns: 1, ..cfg() };
    let mut g = Game::from_state(blank_with(1, Phase::Main, c));
    g.apply(Action::EndTurn).unwrap();
    assert!(g.state().is_over());
    assert!(g.legal_actions().is_empty());
    assert!(g.apply(Action::Roll).is_err());
    assert_eq!(g.log_for(0).last(), Some(&Event::GameOver { winner: None }));
}

#[test]
fn year_of_plenty_order_is_normalized() {
    let mut s = blank(1, Phase::Main);
    s.players[0].dev_hand[DevCard::YearOfPlenty.index()] = 1;
    let mut g = Game::from_state(s);
    g.apply(Action::PlayYearOfPlenty(Resource::Ore, Resource::Wood)).unwrap();
    assert_eq!(g.state().players[0].hand, [1, 0, 0, 0, 1]);
}

#[test]
fn full_game_log_is_replayable_from_seed() {
    let mut a = Game::new(42, cfg());
    let mut b = Game::new(42, cfg());
    for _ in 0..200 {
        if a.state().is_over() {
            break;
        }
        let act = a.legal_actions()[0];
        a.apply(act).unwrap();
        b.apply(act).unwrap();
    }
    assert_eq!(a.state(), b.state());
    assert_eq!(a.log_for(0), b.log_for(0));
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p settler-engine --test game`
Expected: FAIL to compile — `cannot find type Game` / no method `observation`.

- [ ] **Step 3: Write the implementation**

Add to `engine/src/lib.rs` module list:

```rust
pub mod game;
pub mod observation;
```

and to the re-exports:

```rust
pub use game::{Game, IllegalAction};
pub use observation::Observation;
```

`engine/src/observation.rs`:

```rust
//! What one player can see. Opponents' hands and dev cards appear only as counts.

use crate::board::Board;
use crate::state::{PendingTrade, Phase, State};
use crate::types::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Observation {
    pub viewer: PlayerId,
    pub phase: Phase,
    pub current: PlayerId,
    pub actor: PlayerId,
    pub turn: u32,
    pub board: Board,
    pub robber: u8,
    pub bank: Hand,
    pub dev_deck_remaining: u8,
    pub my_hand: Hand,
    pub my_dev_cards: [u8; 5],
    pub my_new_dev_cards: [u8; 5],
    pub hand_counts: [u8; NUM_PLAYERS],
    pub dev_card_counts: [u8; NUM_PLAYERS],
    pub settlements: [u64; NUM_PLAYERS],
    pub cities: [u64; NUM_PLAYERS],
    pub roads: [u128; NUM_PLAYERS],
    pub knights_played: [u8; NUM_PLAYERS],
    pub public_vp: [u8; NUM_PLAYERS],
    pub longest_road_owner: Option<PlayerId>,
    pub largest_army_owner: Option<PlayerId>,
    pub trade: Option<PendingTrade>,
    pub dev_played_this_turn: bool,
}

impl State {
    pub fn observation(&self, viewer: PlayerId) -> Observation {
        let me = &self.players[viewer as usize];
        let per = |f: &dyn Fn(usize) -> u8| -> [u8; NUM_PLAYERS] { std::array::from_fn(f) };
        Observation {
            viewer,
            phase: self.phase,
            current: self.current,
            actor: if self.is_over() { self.current } else { self.current_actor() },
            turn: self.turn,
            board: self.board,
            robber: self.robber,
            bank: self.bank,
            dev_deck_remaining: (DEV_DECK_SIZE - self.dev_deck_pos as usize) as u8,
            my_hand: me.hand,
            my_dev_cards: me.dev_hand,
            my_new_dev_cards: me.dev_new,
            hand_counts: per(&|p| hand_total(&self.players[p].hand) as u8),
            dev_card_counts: per(&|p| self.players[p].dev_hand.iter().sum()),
            settlements: std::array::from_fn(|p| self.players[p].settlements),
            cities: std::array::from_fn(|p| self.players[p].cities),
            roads: std::array::from_fn(|p| self.players[p].roads),
            knights_played: per(&|p| self.players[p].knights_played),
            public_vp: per(&|p| self.public_vp(p)),
            longest_road_owner: self.longest_road_owner,
            largest_army_owner: self.largest_army_owner,
            trade: self.trade,
            dev_played_this_turn: self.dev_played_this_turn,
        }
    }
}
```

`engine/src/game.rs`:

```rust
//! A game with legality checking and an event log, for arenas and bindings.
//! Search should use `State` directly.

use crate::action::Action;
use crate::apply::Chance;
use crate::config::GameConfig;
use crate::events::Event;
use crate::legal::legal_actions;
use crate::observation::Observation;
use crate::state::{Phase, State};
use crate::types::PlayerId;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IllegalAction {
    pub action: Action,
    pub phase: Phase,
}

#[derive(Clone, Debug)]
pub struct Game {
    state: State,
    log: Vec<Event>,
    buf: Vec<Action>,
}

impl Game {
    pub fn new(seed: u64, config: GameConfig) -> Game {
        Game::from_state(State::new(seed, config))
    }

    pub fn from_state(state: State) -> Game {
        Game { state, log: Vec::new(), buf: Vec::new() }
    }

    pub fn state(&self) -> &State {
        &self.state
    }

    pub fn legal_actions(&self) -> Vec<Action> {
        self.state.legal_actions()
    }

    pub fn apply(&mut self, a: Action) -> Result<(), IllegalAction> {
        self.apply_forced(a, None)
    }

    pub fn apply_forced(&mut self, a: Action, chance: Option<Chance>) -> Result<(), IllegalAction> {
        // Round-trip through the encoding to normalize Year of Plenty order.
        let a = Action::decode(a.encode()).unwrap_or(a);
        legal_actions(&self.state, &mut self.buf);
        if !self.buf.contains(&a) {
            return Err(IllegalAction { action: a, phase: self.state.phase });
        }
        self.state.apply_with(a, chance, &mut self.log);
        Ok(())
    }

    pub fn observation(&self, viewer: PlayerId) -> Observation {
        self.state.observation(viewer)
    }

    /// The full event log as `viewer` is allowed to see it.
    pub fn log_for(&self, viewer: PlayerId) -> Vec<Event> {
        self.log.iter().map(|e| e.redacted_for(viewer)).collect()
    }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p settler-engine`
Expected: PASS — `game` (7 tests) plus earlier suites.

- [ ] **Step 5: Commit**

```bash
git add engine
git diff --cached | grep -nE 'sk-|AKIA|ghp_|password=' && echo "SECRET FOUND - abort" || git commit -m "engine: observation and Game wrapper"
git push
```

---

### Task 13: Random playouts and property tests

**Files:**
- Create: `engine/src/sim.rs`, `engine/tests/properties.rs`
- Modify: `engine/src/lib.rs`, `engine/Cargo.toml`

**Interfaces:**
- Consumes: everything above.
- Produces: `sim::play_random(s: &mut State, rng: &mut Rng, buf: &mut Vec<Action>) -> u64` (plays uniformly random legal actions until game over; returns the number of actions).

- [ ] **Step 1: Add the dev-dependency**

In `engine/Cargo.toml` add:

```toml
[dev-dependencies]
proptest = "=1.5.0"
```

Add to `engine/src/lib.rs` module list: `pub mod sim;`

- [ ] **Step 2: Write the failing test**

`engine/tests/properties.rs`:

```rust
use proptest::prelude::*;
use settler_engine::legal::legal_actions;
use settler_engine::rng::Rng;
use settler_engine::sim::play_random;
use settler_engine::topology::topo;
use settler_engine::*;

fn check_invariants(s: &State) {
    for r in 0..NUM_RESOURCES {
        let held: u32 = s.players.iter().map(|p| p.hand[r] as u32).sum();
        assert_eq!(held + s.bank[r] as u32, BANK_PER_RESOURCE as u32, "resource {r} not conserved");
    }
    let t = topo();
    let mut nodes = 0u64;
    let mut edges = 0u128;
    for (i, p) in s.players.iter().enumerate() {
        assert!(p.settlements.count_ones() <= MAX_SETTLEMENTS, "player {i} settlements");
        assert!(p.cities.count_ones() <= MAX_CITIES, "player {i} cities");
        assert!(p.roads.count_ones() <= MAX_ROADS, "player {i} roads");
        assert_eq!(p.settlements & p.cities, 0);
        let b = p.settlements | p.cities;
        assert_eq!(nodes & b, 0, "two players share a node");
        nodes |= b;
        assert_eq!(edges & p.roads, 0, "two players share an edge");
        edges |= p.roads;
        for c in 0..5 {
            assert!(p.dev_new[c] <= p.dev_hand[c]);
        }
    }
    for n in bits64(nodes) {
        assert_eq!(nodes & t.node_neighbor_mask[n as usize], 0, "distance rule broken at {n}");
    }
    let held_dev: u32 = s.players.iter().flat_map(|p| p.dev_hand.iter()).map(|&c| c as u32).sum();
    assert!(held_dev <= s.dev_deck_pos as u32);
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(48))]

    #[test]
    fn random_games_keep_invariants(seed in any::<u64>(), trades in any::<bool>(), compat in any::<bool>()) {
        let cfg = GameConfig {
            max_offers_per_turn: if trades { 3 } else { 0 },
            catanatron_compat: compat,
            ..GameConfig::default()
        };
        let mut s = State::new(seed, cfg);
        let mut rng = Rng::new(seed ^ 0xABCD);
        let mut buf = Vec::new();
        let mut steps = 0u64;
        while !s.is_over() {
            legal_actions(&s, &mut buf);
            prop_assert!(!buf.is_empty(), "no legal actions in {:?}", s.phase);
            for &a in &buf {
                prop_assert_eq!(Action::decode(a.encode()), Some(a));
            }
            let a = buf[rng.below(buf.len() as u32) as usize];
            s.apply(a);
            check_invariants(&s);
            steps += 1;
            prop_assert!(steps < 2_000_000, "game did not terminate");
        }
        match s.winner() {
            Some(w) => prop_assert!(s.total_vp(w as usize) >= cfg.vp_to_win),
            None => prop_assert!(s.turn >= cfg.max_turns),
        }
    }
}

#[test]
fn play_random_finishes_games() {
    let cfg = GameConfig { max_offers_per_turn: 0, ..GameConfig::default() };
    let mut rng = Rng::new(1);
    let mut buf = Vec::new();
    for seed in 0..20 {
        let mut s = State::new(seed, cfg);
        let n = play_random(&mut s, &mut rng, &mut buf);
        assert!(s.is_over());
        assert!(n > 0);
    }
}
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cargo test -p settler-engine --test properties`
Expected: FAIL to compile — `could not find sim`.

- [ ] **Step 4: Write the implementation**

`engine/src/sim.rs`:

```rust
//! Uniformly random playouts (benchmarks, property tests, rollout baselines).

use crate::action::Action;
use crate::legal::legal_actions;
use crate::rng::Rng;
use crate::state::State;

/// Play uniformly random legal actions until the game ends. Returns the number of actions.
pub fn play_random(s: &mut State, rng: &mut Rng, buf: &mut Vec<Action>) -> u64 {
    let mut steps = 0;
    while !s.is_over() {
        legal_actions(s, buf);
        let a = buf[rng.below(buf.len() as u32) as usize];
        s.apply(a);
        steps += 1;
    }
    steps
}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cargo test -p settler-engine`
Expected: PASS — `properties` (2 tests) plus all earlier suites. If a property fails, proptest prints a minimal failing seed; fix the rule bug it exposes (write a focused unit test for it in the matching task's test file first), never weaken the invariant.

- [ ] **Step 6: Audit dependencies**

```bash
cargo install cargo-audit --locked
cargo audit
```

Expected: `0 vulnerabilities found` (warnings about unmaintained transitive crates are acceptable; record them in the commit message).

- [ ] **Step 7: Commit**

```bash
git add engine Cargo.lock
git diff --cached | grep -nE 'sk-|AKIA|ghp_|password=' && echo "SECRET FOUND - abort" || git commit -m "engine: random playouts and property tests"
git push
```

---

### Task 14: Benchmarks, throughput, and lint gate

**Files:**
- Create: `engine/benches/engine.rs`, `engine/examples/throughput.rs`, `docs/perf/engine-baseline.md`
- Modify: `engine/Cargo.toml`

**Interfaces:**
- Consumes: `sim::play_random`, `legal::legal_actions`, `State`, `GameConfig`, `Rng`.
- Produces: `cargo bench -p settler-engine` results and `cargo run --release -p settler-engine --example throughput [N]`, which prints single-core and all-core games/sec.

- [ ] **Step 1: Add criterion and the bench target**

In `engine/Cargo.toml`, add to `[dev-dependencies]`:

```toml
criterion = "=0.5.1"
```

and append:

```toml
[[bench]]
name = "engine"
harness = false
```

- [ ] **Step 2: Write the benchmarks**

`engine/benches/engine.rs`:

```rust
use criterion::{criterion_group, criterion_main, Criterion};
use settler_engine::legal::legal_actions;
use settler_engine::rng::Rng;
use settler_engine::sim::play_random;
use settler_engine::{GameConfig, State};
use std::hint::black_box;

fn bench(c: &mut Criterion) {
    let cfg = GameConfig { max_offers_per_turn: 0, ..GameConfig::default() };
    let mut rng = Rng::new(7);
    let mut buf = Vec::with_capacity(256);

    // A mid-game position from a random playout.
    let mut mid = State::new(7, cfg);
    for _ in 0..400 {
        if mid.is_over() {
            break;
        }
        legal_actions(&mid, &mut buf);
        let a = buf[rng.below(buf.len() as u32) as usize];
        mid.apply(a);
    }

    c.bench_function("clone_state", |b| b.iter(|| black_box(*black_box(&mid))));
    c.bench_function("legal_actions_midgame", |b| {
        b.iter(|| {
            legal_actions(black_box(&mid), &mut buf);
            black_box(buf.len())
        })
    });
    c.bench_function("random_game", |b| {
        let mut seed = 0u64;
        b.iter(|| {
            seed += 1;
            let mut s = State::new(seed, cfg);
            black_box(play_random(&mut s, &mut rng, &mut buf))
        })
    });
}

criterion_group!(benches, bench);
criterion_main!(benches);
```

`engine/examples/throughput.rs`:

```rust
//! Games per second on one core and on all cores. Usage: throughput [games_per_thread]

use settler_engine::rng::Rng;
use settler_engine::sim::play_random;
use settler_engine::{GameConfig, State};
use std::time::Instant;

fn run(seeds: std::ops::Range<u64>, cfg: GameConfig) -> (u64, u64) {
    let mut rng = Rng::new(seeds.start ^ 0x5EED);
    let mut buf = Vec::with_capacity(256);
    let (mut games, mut actions) = (0, 0);
    for seed in seeds {
        let mut s = State::new(seed, cfg);
        actions += play_random(&mut s, &mut rng, &mut buf);
        games += 1;
    }
    (games, actions)
}

fn main() {
    let cfg = GameConfig { max_offers_per_turn: 0, ..GameConfig::default() };
    let n: u64 = std::env::args().nth(1).and_then(|a| a.parse().ok()).unwrap_or(10_000);

    let t0 = Instant::now();
    let (g, a) = run(0..n, cfg);
    let dt = t0.elapsed().as_secs_f64();
    println!(
        "1 thread: {g} games in {dt:.2}s = {:.0} games/s, {:.2}M actions/s, {:.0} actions/game",
        g as f64 / dt,
        a as f64 / dt / 1e6,
        a as f64 / g as f64
    );

    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1) as u64;
    let t0 = Instant::now();
    let handles: Vec<_> = (0..threads)
        .map(|i| std::thread::spawn(move || run(n * (i + 1)..n * (i + 2), cfg)))
        .collect();
    let (mut g, mut a) = (0, 0);
    for h in handles {
        let (gg, aa) = h.join().unwrap();
        g += gg;
        a += aa;
    }
    let dt = t0.elapsed().as_secs_f64();
    println!(
        "{threads} threads: {g} games in {dt:.2}s = {:.0} games/s, {:.2}M actions/s",
        g as f64 / dt,
        a as f64 / dt / 1e6
    );
}
```

- [ ] **Step 3: Run benchmarks and throughput**

```bash
cargo bench -p settler-engine
cargo run --release -p settler-engine --example throughput 10000
```

Expected: criterion prints timings for `clone_state` (target < 100 ns), `legal_actions_midgame`, and `random_game`. Throughput prints single-thread games/s (target ≥ 1,000) and all-thread games/s.

- [ ] **Step 4: Record results**

`docs/perf/engine-baseline.md` — fill every value from the Step 3 output (no blanks left):

```markdown
# Engine performance baseline

Machine: <output of `sysctl -n machdep.cpu.brand_string`>, <core count> cores
Commit: <output of `git rev-parse --short HEAD`>
Config: default rules, domestic trades disabled (`max_offers_per_turn = 0`)

| Measurement | Result | Target |
|---|---|---|
| `clone_state` | <criterion median> | < 100 ns |
| `legal_actions_midgame` | <criterion median> | — |
| `random_game` | <criterion median> | — |
| Single-thread games/s | <throughput line 1> | ≥ 1,000 |
| Actions per game | <throughput line 1> | — |
| All-thread games/s | <throughput line 2> | — |

Catanatron side-by-side comparison: Plan 3.
```

If a target is missed, add a "Missed targets" section stating the measured value and the suspected hot spot (run `cargo flamegraph` or `samply` on the throughput example to find it). The spec allows recording a miss with its reason; it does not allow leaving it unexplained.

- [ ] **Step 5: Lint gate**

```bash
cargo fmt --all
cargo clippy -p settler-engine --all-targets -- -D warnings
cargo test -p settler-engine
```

Expected: clippy reports no warnings (fix any it finds), all tests pass.

- [ ] **Step 6: Commit**

```bash
git add engine docs/perf Cargo.lock
git diff --cached | grep -nE 'sk-|AKIA|ghp_|password=' && echo "SECRET FOUND - abort" || git commit -m "engine: benchmarks, throughput example, perf baseline"
git push
```

---

## Spec coverage (self-review)

| Spec requirement | Task |
|---|---|
| Fixed-size `Copy` state < 512 B, cheap clone | 5 (size test), 14 (clone bench) |
| Topology as static tables, not in state | 2 |
| Per-game board, ports, random generation | 3 |
| Fixed action space (~700), factored decisions | 4 (665 ids; robber tile→victim, one-card discards) |
| Domestic trade offer→accept/reject→confirm, caps N/K | 11 |
| Incremental longest road | 8 (per-player recompute on road/settlement events only) |
| Independent RNG streams / CRN dice | 3 (`dice_for`), 5 (stream salts), 6 (unforced roll test) |
| Forced chance outcomes | 5 (`Chance`), 6, 9 |
| `observation(p)` + public event log with redaction | 12 |
| `GameConfig` (VP, discard limit, trade caps, compat) | 3, 6 (compat discards) |
| Friendly robber flag | Deferred until Colonist rules verified (see Global Constraints) |
| Rule unit tests (setup order, 2nd settlement, distance, connectivity, longest road incl. breaks, largest army, dev timing, one dev/turn, hidden VP, discards, robber blocking, bank shortage, ports, own-turn win) | 5–11 |
| Property tests (conservation, piece limits, no panics, termination) | 13 |
| Performance targets measured with criterion | 14 |
| Differential testing vs Catanatron, Catanatron perf comparison | Plan 3 |
| Bindings, baseline bots, arena, stats, CLI | Plan 2 |
