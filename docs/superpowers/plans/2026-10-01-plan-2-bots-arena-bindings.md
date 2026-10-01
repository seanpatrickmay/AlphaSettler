# Plan 2: Baseline Bots, Native Arena, Python Bindings, Stats & CLI — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure bot strength: Rust `RandomBot` and `GreedyBot`, a native Rust arena that plays each seed four times with the candidate rotated through every seat (CRN), PyO3 bindings exposing the engine and arena to Python, and a Python package with win-rate statistics, JSONL output and the `alphasettler arena` / `alphasettler compare` CLI.

**Architecture:** A new Rust crate `bots/` (`settler-bots`) holds the `Bot` trait, both baseline bots, and the arena (threaded, deterministic, bots see only `Observation`). A new crate `bindings/` builds the Python extension `alphasettler._engine` with maturin; it only converts types. The pure-Python package `alphasettler/` holds stats, the arena runner and the CLI. Bot randomness is seeded from `(seed, seat)`, so swapping the candidate leaves every baseline's random stream unchanged.

**Tech Stack:** Rust (edition 2021), PyO3 `=0.29.3`, maturin `1.15.0`, Python 3.12 (`python3`/`pip3` only), pytest `9.1.1`.

**Spec:** `docs/superpowers/specs/2026-09-30-engine-and-harness-design.md` (Section 1 architecture; Section 3 "Benchmark harness"; Section 4 done criteria for the native arena). Plan 3 covers the Catanatron arena and differential testing.

## Global Constraints

- Engine and bots crates: zero runtime dependencies beyond each other; `rust-version = "1.80"`. The bindings crate may require a newer toolchain (PyO3); it is excluded from `default-members` so `cargo test` never needs Python.
- PyO3 pinned `=0.29.3`; Python dev deps pinned `maturin==1.15.0`, `pytest==9.1.1`. Always `python3` / `pip3`, never `python` / `pip`. The venv lives at `.venv/` (already gitignored).
- In PyO3 0.29, `[u8; N]` and `Vec<u8>` convert to Python `bytes`; every hand/count array must be widened to `Vec<u32>` before returning. Release the GIL for long Rust work with `py.detach(|| ...)`.
- Bots see only `Observation` and the legal-action list, never `State`.
- Arena format (spec): one candidate vs three copies of a baseline; each seed played 4 times with the candidate in seat 0, 1, 2, 3. Unit of analysis is the seed. Win rate is compared with the 25% null.
- Baseline bots (spec): `RandomBot` uniform over legal actions with domestic offers disabled; `GreedyBot` builds whenever possible using simple placement scoring, never offers domestic trades, rejects all offers.
- Output (spec): one JSON line per game in `runs/` (gitignored) plus a printed summary. CLI: `alphasettler arena --candidate X --baseline Y --seeds N`.
- Done criterion (spec Section 4): `GreedyBot` beats `RandomBot` with z > 3 in the native arena.
- Results must not depend on the thread count.
- Golden trace (`engine/tests/golden_trace.rs`) must stay green — this plan does not change engine behavior.
- After each task: run the task's tests, secret-scan the staged diff, commit (`sean.may101@gmail.com`, no AI attribution, never `--no-verify`), push.

## Review Focus

1. **A bot returns an action that isn't legal** → the arena fails loudly naming the bot and the phase, never applies it. Test: Task 3 `illegal_bot_action_panics_with_bot_name`.
2. **Bad inputs from Python/CLI** (unknown bot name, unknown config key, out-of-range action id or viewer, `--seeds 0`) → a clear `ValueError` / exit code 2 with a message, never a crash. Tests: Task 4 `test_bad_inputs_raise_value_error`, Task 6 `test_cli_rejects_unknown_bot_and_zero_seeds`.
3. **Thread count changes results** → identical records for 1 vs many threads. Tests: Task 3 `results_do_not_depend_on_thread_count`, Task 4 `test_run_match_is_deterministic_across_threads`.
4. **Degenerate statistics** (all wins, all losses, one seed, turn-cap draws) → finite or explicit ±inf z, no division by zero; draws count as candidate losses. Tests: Task 5 `test_zero_variance_and_single_seed`, `test_draws_count_as_losses`.
5. **Long Rust calls hold the GIL** → `run_match` releases it. Test: Task 4 `test_run_match_releases_the_gil` (a Python thread makes progress while a long match runs).

---

## File Structure

```
Cargo.toml                      workspace: members engine, bots, bindings; default-members engine, bots
bots/Cargo.toml                 crate settler-bots (lib settler_bots)
bots/src/lib.rs                 Bot trait, BOT_NAMES, make_bot
bots/src/random.rs              RandomBot
bots/src/greedy.rs              GreedyBot + scoring helpers
bots/src/arena.rs               play_game, run_match, GameRecord, bot_seed
bots/tests/common/mod.rs        test helpers
bots/tests/random.rs
bots/tests/greedy.rs
bots/tests/arena.rs
bindings/Cargo.toml             crate settler-bindings (cdylib _engine)
bindings/src/lib.rs             #[pymodule] _engine: Game, run_match, helpers
bindings/src/convert.rs         Rust ↔ Python conversions (config, observation, events, chance)
pyproject.toml                  maturin build; console script `alphasettler`
alphasettler/__init__.py
alphasettler/__main__.py
alphasettler/stats.py
alphasettler/arena.py
alphasettler/cli.py
tests/python/test_engine.py
tests/python/test_stats.py
tests/python/test_arena.py
tests/python/test_cli.py
```

---

### Task 1: Bots crate, `Bot` trait, `RandomBot`

**Files:**
- Create: `bots/Cargo.toml`, `bots/src/lib.rs`, `bots/src/random.rs`, `bots/tests/random.rs`
- Modify: `Cargo.toml` (workspace)

**Interfaces:**
- Consumes: `settler_engine::{Action, Observation, State, GameConfig}`, `settler_engine::legal::legal_actions`, `settler_engine::rng::Rng`.
- Produces: `settler_bots::Bot` trait (`fn name(&self) -> &'static str; fn act(&mut self, obs: &Observation, legal: &[Action]) -> Action;`, `Bot: Send`), `settler_bots::BOT_NAMES: &[&str]`, `settler_bots::make_bot(name: &str, seed: u64) -> Option<Box<dyn Bot>>`, `settler_bots::random::RandomBot::new(seed: u64)`.

- [ ] **Step 1: Workspace and crate manifest**

`Cargo.toml` (root) becomes:

```toml
[workspace]
members = ["engine", "bots"]
default-members = ["engine", "bots"]
resolver = "2"

[profile.release]
lto = "fat"
codegen-units = 1

[profile.bench]
inherits = "release"
```

`bots/Cargo.toml`:

```toml
[package]
name = "settler-bots"
version = "0.1.0"
edition = "2021"
rust-version = "1.80"
publish = false

[lib]
name = "settler_bots"

[dependencies]
settler-engine = { path = "../engine" }

[lints.clippy]
needless_range_loop = "allow"
```

- [ ] **Step 2: Write the failing test**

`bots/tests/random.rs`:

```rust
use settler_bots::{make_bot, Bot, BOT_NAMES};
use settler_engine::legal::legal_actions;
use settler_engine::*;

fn play(bot_seed: u64, game_seed: u64) -> (State, Vec<Action>) {
    let mut bots: Vec<Box<dyn Bot>> = (0..4).map(|i| make_bot("random", bot_seed + i).unwrap()).collect();
    let mut s = State::new(game_seed, GameConfig::default());
    let mut buf = Vec::new();
    let mut taken = Vec::new();
    while !s.is_over() {
        legal_actions(&s, &mut buf);
        let actor = s.current_actor();
        let a = bots[actor as usize].act(&s.observation(actor), &buf);
        assert!(buf.contains(&a), "{a:?} not legal");
        taken.push(a);
        s.apply(a);
    }
    (s, taken)
}

#[test]
fn random_bot_plays_legal_moves_to_the_end() {
    for seed in 0..20 {
        let (s, taken) = play(1, seed);
        assert!(s.is_over());
        assert!(!taken.is_empty());
    }
}

#[test]
fn random_bot_never_offers_trades() {
    // Default config allows 3 offers per turn, so offers are legal and must be filtered out.
    for seed in 0..20 {
        let (_, taken) = play(2, seed);
        assert!(!taken.iter().any(|a| matches!(a, Action::OfferTrade { .. })));
    }
}

#[test]
fn random_bot_is_deterministic_per_seed() {
    assert_eq!(play(3, 7).1, play(3, 7).1);
    assert_ne!(play(3, 7).1, play(4, 7).1);
}

#[test]
fn bot_registry() {
    assert!(make_bot("nope", 0).is_none());
    assert!(BOT_NAMES.contains(&"random"));
    assert_eq!(make_bot("random", 0).unwrap().name(), "random");
}
```

- [ ] **Step 3: Run test to verify it fails**

Run: `source "$HOME/.cargo/env" && cargo test -p settler-bots --test random`
Expected: FAIL to compile — `file not found for module` / unresolved `settler_bots`.

- [ ] **Step 4: Write the implementation**

`bots/src/lib.rs`:

```rust
//! Baseline bots and the native arena. Bots see only an `Observation` and the legal actions.

pub mod random;

use settler_engine::{Action, Observation};

pub trait Bot: Send {
    fn name(&self) -> &'static str;
    /// Choose one of `legal` (never empty) for `obs.viewer`, the player who must act now.
    fn act(&mut self, obs: &Observation, legal: &[Action]) -> Action;
}

pub const BOT_NAMES: &[&str] = &["random"];

/// A fresh bot by name, with its own random stream seeded from `seed`.
pub fn make_bot(name: &str, seed: u64) -> Option<Box<dyn Bot>> {
    match name {
        "random" => Some(Box::new(random::RandomBot::new(seed))),
        _ => None,
    }
}
```

`bots/src/random.rs`:

```rust
//! Uniform over legal actions, except it never proposes domestic trades.

use crate::Bot;
use settler_engine::rng::Rng;
use settler_engine::{Action, Observation};

pub struct RandomBot {
    rng: Rng,
    pool: Vec<Action>,
}

impl RandomBot {
    pub fn new(seed: u64) -> RandomBot {
        RandomBot {
            rng: Rng::new(seed),
            pool: Vec::new(),
        }
    }
}

impl Bot for RandomBot {
    fn name(&self) -> &'static str {
        "random"
    }

    fn act(&mut self, _obs: &Observation, legal: &[Action]) -> Action {
        self.pool.clear();
        self.pool
            .extend(legal.iter().copied().filter(|a| !matches!(a, Action::OfferTrade { .. })));
        let pool: &[Action] = if self.pool.is_empty() { legal } else { &self.pool };
        pool[self.rng.below(pool.len() as u32) as usize]
    }
}
```

- [ ] **Step 5: Run tests**

Run: `cargo test -p settler-bots --test random && cargo test`
Expected: 4 passed in `random`; the full default workspace suite (engine + bots) passes, golden trace included.

- [ ] **Step 6: Commit**

```bash
git add Cargo.toml Cargo.lock bots
git diff --cached | grep -nE 'sk-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{36}|password=[^[:space:]]+' && echo "SECRET FOUND - abort" || git commit -m "bots: Bot trait, registry, RandomBot"
git push
```

---

### Task 2: `GreedyBot`

**Files:**
- Create: `bots/src/greedy.rs`, `bots/tests/common/mod.rs`, `bots/tests/greedy.rs`
- Modify: `bots/src/lib.rs`

**Interfaces:**
- Consumes: `Bot`, `Observation` fields (`viewer, phase, board, robber, my_hand, settlements, cities, public_vp, hand_counts`), `settler_engine::topology::topo`, cost constants.
- Produces: `settler_bots::greedy::{GreedyBot, pips(number: u8) -> u32, node_value(obs: &Observation, node: u8) -> u32, road_value(obs: &Observation, edge: u8) -> u32}`; `make_bot("greedy", seed)`; `BOT_NAMES` = `["random", "greedy"]`.

Decision rules (all choices come from `legal`; ties go to the earliest action in `legal`, which is deterministic because `best_by` keeps the first maximum):

| Phase | Choice |
|---|---|
| SetupSettlement | settlement with max `node_value` |
| SetupRoad, RoadBuilding | road with max `road_value` |
| PreRoll | `PlayKnight` if legal and the robber sits on a tile touching own buildings, else `Roll` |
| Discard | resource it holds most of |
| MoveRobber | tile without own buildings maximizing Σ opponents' (settlement 1, city 2) × pips + 1 per opponent present; fallback first legal |
| Steal | victim with most public VP, then most cards |
| Main | city (max node value) → settlement (max node value) → knight (if robber on own tile) → road building card → Year of Plenty that covers missing cards of the target build → road with `road_value > 0` → buy dev card → maritime trade that gets a missing card of the target build and gives only surplus → `EndTurn` |
| TradeResponse | `RejectTrade` |
| TradeConfirm | `CancelTrade` |

Target build: city if it has a settlement and fewer than 4 cities; else settlement if fewer than 5 settlements; else dev card.

- [ ] **Step 1: Test helpers**

`bots/tests/common/mod.rs`:

```rust
#![allow(dead_code)]

use settler_engine::*;

/// A fresh game past setup: no buildings, empty hands, player 0 to act in `phase`.
pub fn blank(seed: u64, phase: Phase) -> State {
    let mut s = State::new(seed, GameConfig::default());
    s.setup_step = 8;
    s.current = 0;
    s.phase = phase;
    s
}

pub fn give(s: &mut State, p: usize, h: Hand) {
    hand_sub(&mut s.bank, &h);
    hand_add(&mut s.players[p].hand, &h);
}

pub fn act(bot: &mut dyn settler_bots::Bot, s: &State) -> Action {
    let legal = s.legal_actions();
    let a = bot.act(&s.observation(s.current_actor()), &legal);
    assert!(legal.contains(&a), "{a:?} not legal");
    a
}
```

- [ ] **Step 2: Write the failing test**

`bots/tests/greedy.rs`:

```rust
mod common;
use common::*;
use settler_bots::greedy::{node_value, pips, road_value, GreedyBot};
use settler_bots::{make_bot, BOT_NAMES};
use settler_engine::topology::topo;
use settler_engine::*;

#[test]
fn pip_counts() {
    assert_eq!([0, 2, 3, 6, 8, 12].map(pips), [0, 1, 2, 5, 5, 1]);
}

#[test]
fn registered() {
    assert!(BOT_NAMES.contains(&"greedy"));
    assert_eq!(make_bot("greedy", 1).unwrap().name(), "greedy");
}

#[test]
fn setup_takes_the_most_valuable_spot() {
    let s = State::new(3, GameConfig::default());
    let obs = s.observation(0);
    let best = (0..54u8).map(|n| node_value(&obs, n)).max().unwrap();
    let a = act(&mut GreedyBot::new(0), &s);
    let Action::BuildSettlement(n) = a else { panic!("{a:?}") };
    assert_eq!(node_value(&obs, n), best);
}

#[test]
fn setup_road_heads_for_the_best_reachable_spot() {
    let mut s = State::new(3, GameConfig::default());
    let a = act(&mut GreedyBot::new(0), &s);
    s.apply(a);
    let obs = s.observation(0);
    let best = s
        .legal_actions()
        .iter()
        .map(|a| match a {
            Action::BuildRoad(e) => road_value(&obs, *e),
            _ => 0,
        })
        .max()
        .unwrap();
    let Action::BuildRoad(e) = act(&mut GreedyBot::new(0), &s) else { panic!() };
    assert_eq!(road_value(&obs, e), best);
}

#[test]
fn prefers_city_over_settlement() {
    let mut s = blank(1, Phase::Main);
    let n0 = 20u8;
    let path: Vec<u8> = {
        let t = topo();
        let a = t.node_neighbors[n0 as usize][0];
        let b = *t.node_neighbors[a as usize].iter().find(|&&x| x != n0).unwrap();
        vec![n0, a, b]
    };
    s.players[0].settlements = 1u64 << n0;
    for w in path.windows(2) {
        let e = *topo().node_edges[w[0] as usize]
            .iter()
            .find(|&&e| topo().node_edges[w[1] as usize].contains(&e))
            .unwrap();
        s.players[0].roads |= 1u128 << e;
    }
    give(&mut s, 0, [1, 1, 1, 3, 3]);
    let legal = s.legal_actions();
    assert!(legal.iter().any(|a| matches!(a, Action::BuildSettlement(_))));
    assert_eq!(act(&mut GreedyBot::new(0), &s), Action::BuildCity(n0));
}

#[test]
fn ends_turn_with_nothing_useful() {
    let s = blank(1, Phase::Main);
    assert_eq!(act(&mut GreedyBot::new(0), &s), Action::EndTurn);
}

#[test]
fn rejects_every_offer() {
    let mut s = blank(1, Phase::Main);
    give(&mut s, 0, [1, 0, 0, 0, 0]);
    give(&mut s, 1, [0, 1, 0, 0, 0]);
    s.apply(Action::OfferTrade { give: 0, get: 1 });
    assert_eq!(s.current_actor(), 1);
    assert_eq!(act(&mut GreedyBot::new(0), &s), Action::RejectTrade);
}

#[test]
fn never_offers_trades() {
    let mut s = blank(1, Phase::Main);
    give(&mut s, 0, [2, 2, 0, 0, 0]);
    let a = act(&mut GreedyBot::new(0), &s);
    assert!(!matches!(a, Action::OfferTrade { .. }), "{a:?}");
}

#[test]
fn discards_its_most_plentiful_resource() {
    let mut s = blank(1, Phase::PreRoll);
    give(&mut s, 0, [1, 5, 2, 0, 1]);
    s.apply_with(
        Action::Roll,
        Some(Chance::Roll { dice: (3, 4), discards: None }),
        &mut NoEvents,
    );
    assert_eq!(s.phase, Phase::Discard);
    assert_eq!(act(&mut GreedyBot::new(0), &s), Action::Discard(Resource::Brick));
}

#[test]
fn robber_avoids_own_tiles_and_hits_opponents() {
    let mut s = blank(1, Phase::MoveRobber);
    let t = topo();
    let by_pips = |tile: usize| pips(s.board.tile_number[tile]);
    let mut tiles: Vec<usize> = (0..19).filter(|&x| x as u8 != s.robber).collect();
    tiles.sort_by_key(|&x| std::cmp::Reverse(by_pips(x)));
    let (mine, theirs) = (tiles[0], tiles[1]);
    let my_node = *t.tile_nodes[mine]
        .iter()
        .find(|&&n| t.tile_node_mask[theirs] & (1u64 << n) == 0)
        .unwrap();
    s.players[0].settlements = 1u64 << my_node;
    let opp_node = *t.tile_nodes[theirs]
        .iter()
        .find(|&&n| t.tile_node_mask[mine] & (1u64 << n) == 0)
        .unwrap();
    s.players[1].cities = 1u64 << opp_node;
    let Action::MoveRobber(tile) = act(&mut GreedyBot::new(0), &s) else { panic!() };
    assert_eq!(t.tile_node_mask[tile as usize] & s.players[0].settlements, 0);
    assert_ne!(t.tile_node_mask[tile as usize] & s.players[1].cities, 0);
}

#[test]
fn trades_surplus_with_the_bank_for_a_missing_card() {
    let mut s = blank(1, Phase::Main);
    s.players[0].settlements = 1u64 << 20; // target build: city (needs 2 wheat, 3 ore)
    give(&mut s, 0, [4, 0, 0, 0, 0]);
    assert_eq!(
        act(&mut GreedyBot::new(0), &s),
        Action::MaritimeTrade { give: Resource::Wood, get: Resource::Ore }
    );
}

#[test]
fn greedy_games_finish() {
    for seed in 0..20 {
        let mut bots: Vec<Box<dyn settler_bots::Bot>> =
            (0..4).map(|i| make_bot("greedy", i).unwrap()).collect();
        let mut s = State::new(seed, GameConfig::default());
        let mut steps = 0;
        while !s.is_over() {
            let legal = s.legal_actions();
            let actor = s.current_actor();
            let a = bots[actor as usize].act(&s.observation(actor), &legal);
            assert!(legal.contains(&a));
            s.apply(a);
            steps += 1;
            assert!(steps < 200_000);
        }
    }
}
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cargo test -p settler-bots --test greedy`
Expected: FAIL to compile — `could not find greedy in settler_bots`.

- [ ] **Step 4: Write the implementation**

In `bots/src/lib.rs`: add `pub mod greedy;`, set `pub const BOT_NAMES: &[&str] = &["random", "greedy"];`, and add the arm `"greedy" => Some(Box::new(greedy::GreedyBot::new(seed))),` to `make_bot`.

`bots/src/greedy.rs`:

```rust
//! A simple greedy baseline: builds whenever it can, using pip-count placement scoring.
//! Never offers domestic trades and rejects every offer.

use crate::Bot;
use settler_engine::topology::topo;
use settler_engine::types::*;
use settler_engine::{Action, Observation, Phase};

pub struct GreedyBot;

impl GreedyBot {
    pub fn new(_seed: u64) -> GreedyBot {
        GreedyBot
    }
}

/// Dots on a number token: 2 and 12 have 1, 6 and 8 have 5, the desert 0.
pub fn pips(number: u8) -> u32 {
    if number == 0 {
        0
    } else {
        6 - (7 - number as i32).unsigned_abs()
    }
}

/// Pips of a node's producing tiles plus 1 per distinct resource.
pub fn node_value(obs: &Observation, n: u8) -> u32 {
    let mut value = 0;
    let mut seen = [false; NUM_RESOURCES];
    for &tile in &topo().node_tiles[n as usize] {
        if let Some(r) = obs.board.tile_resource[tile as usize] {
            value += pips(obs.board.tile_number[tile as usize]);
            if !seen[r.index()] {
                seen[r.index()] = true;
                value += 1;
            }
        }
    }
    value
}

fn occupied(obs: &Observation) -> u64 {
    (0..NUM_PLAYERS).fold(0, |m, p| m | obs.settlements[p] | obs.cities[p])
}

/// A free node where the distance rule allows a settlement.
fn spot_ok(obs: &Observation, n: u8) -> bool {
    let occ = occupied(obs);
    occ & (1u64 << n) == 0 && occ & topo().node_neighbor_mask[n as usize] == 0
}

/// The best settlement spot a road on `e` brings within reach (its endpoints and their neighbours).
pub fn road_value(obs: &Observation, e: u8) -> u32 {
    let t = topo();
    let (a, b) = t.edge_nodes[e as usize];
    let mut best = 0;
    for x in [a, b] {
        if spot_ok(obs, x) {
            best = best.max(node_value(obs, x));
        }
        for &y in &t.node_neighbors[x as usize] {
            if spot_ok(obs, y) {
                best = best.max(node_value(obs, y));
            }
        }
    }
    best
}

/// The legal action with the highest score (`None` = not a candidate); first wins ties.
fn best_by(legal: &[Action], score: impl Fn(Action) -> Option<u32>) -> Option<Action> {
    let mut best: Option<(u32, Action)> = None;
    for &a in legal {
        if let Some(v) = score(a) {
            if best.map_or(true, |(bv, _)| v > bv) {
                best = Some((v, a));
            }
        }
    }
    best.map(|(_, a)| a)
}

fn my_buildings(obs: &Observation) -> u64 {
    let me = obs.viewer as usize;
    obs.settlements[me] | obs.cities[me]
}

fn robber_on_me(obs: &Observation) -> bool {
    topo().tile_node_mask[obs.robber as usize] & my_buildings(obs) != 0
}

/// Cost of the build the bot is saving for.
fn target_cost(obs: &Observation) -> Hand {
    let me = obs.viewer as usize;
    if obs.settlements[me] != 0 && obs.cities[me].count_ones() < MAX_CITIES {
        CITY_COST
    } else if obs.settlements[me].count_ones() < MAX_SETTLEMENTS {
        SETTLEMENT_COST
    } else {
        DEV_COST
    }
}

fn deficit(obs: &Observation) -> Hand {
    let cost = target_cost(obs);
    std::array::from_fn(|r| cost[r].saturating_sub(obs.my_hand[r]))
}

fn year_of_plenty_choice(obs: &Observation, legal: &[Action]) -> Option<Action> {
    let need = deficit(obs);
    best_by(legal, |a| match a {
        Action::PlayYearOfPlenty(x, y) => {
            let mut left = need;
            let mut covered = 0;
            for r in [x, y] {
                if left[r.index()] > 0 {
                    left[r.index()] -= 1;
                    covered += 1;
                }
            }
            (covered > 0).then_some(covered)
        }
        _ => None,
    })
}

fn maritime_choice(obs: &Observation, legal: &[Action]) -> Option<Action> {
    let cost = target_cost(obs);
    let need = deficit(obs);
    let buildings = my_buildings(obs);
    best_by(legal, |a| match a {
        Action::MaritimeTrade { give, get } => {
            let rate = obs.board.maritime_rate(buildings, give);
            let surplus = obs.my_hand[give.index()] >= rate + cost[give.index()];
            (need[get.index()] > 0 && surplus).then_some(need[get.index()] as u32)
        }
        _ => None,
    })
}

fn main_choice(obs: &Observation, legal: &[Action]) -> Option<Action> {
    let city = best_by(legal, |a| match a {
        Action::BuildCity(n) => Some(node_value(obs, n)),
        _ => None,
    });
    if city.is_some() {
        return city;
    }
    let settlement = best_by(legal, |a| match a {
        Action::BuildSettlement(n) => Some(node_value(obs, n)),
        _ => None,
    });
    if settlement.is_some() {
        return settlement;
    }
    if legal.contains(&Action::PlayKnight) && robber_on_me(obs) {
        return Some(Action::PlayKnight);
    }
    if legal.contains(&Action::PlayRoadBuilding) {
        return Some(Action::PlayRoadBuilding);
    }
    if let Some(y) = year_of_plenty_choice(obs, legal) {
        return Some(y);
    }
    let road = best_by(legal, |a| match a {
        Action::BuildRoad(e) => Some(road_value(obs, e)).filter(|&v| v > 0),
        _ => None,
    });
    if road.is_some() {
        return road;
    }
    if legal.contains(&Action::BuyDev) {
        return Some(Action::BuyDev);
    }
    if let Some(m) = maritime_choice(obs, legal) {
        return Some(m);
    }
    Some(Action::EndTurn)
}

fn robber_choice(obs: &Observation, legal: &[Action]) -> Option<Action> {
    let t = topo();
    let mine = my_buildings(obs);
    best_by(legal, |a| match a {
        Action::MoveRobber(tile) => {
            let mask = t.tile_node_mask[tile as usize];
            if mask & mine != 0 {
                return None;
            }
            let p = pips(obs.board.tile_number[tile as usize]);
            let mut score = 0;
            for q in 0..NUM_PLAYERS {
                if q == obs.viewer as usize {
                    continue;
                }
                let weight = (obs.settlements[q] & mask).count_ones() + 2 * (obs.cities[q] & mask).count_ones();
                if weight > 0 {
                    score += weight * p + 1;
                }
            }
            Some(score)
        }
        _ => None,
    })
}

impl Bot for GreedyBot {
    fn name(&self) -> &'static str {
        "greedy"
    }

    fn act(&mut self, obs: &Observation, legal: &[Action]) -> Action {
        let pick = match obs.phase {
            Phase::SetupSettlement => best_by(legal, |a| match a {
                Action::BuildSettlement(n) => Some(node_value(obs, n)),
                _ => None,
            }),
            Phase::SetupRoad { .. } | Phase::RoadBuilding { .. } => best_by(legal, |a| match a {
                Action::BuildRoad(e) => Some(road_value(obs, e)),
                _ => None,
            }),
            Phase::PreRoll => {
                if legal.contains(&Action::PlayKnight) && robber_on_me(obs) {
                    Some(Action::PlayKnight)
                } else {
                    Some(Action::Roll)
                }
            }
            Phase::Discard => best_by(legal, |a| match a {
                Action::Discard(r) => Some(obs.my_hand[r.index()] as u32),
                _ => None,
            }),
            Phase::MoveRobber => robber_choice(obs, legal),
            Phase::Steal => best_by(legal, |a| match a {
                Action::StealFrom(p) => {
                    Some(obs.public_vp[p as usize] as u32 * 100 + obs.hand_counts[p as usize] as u32)
                }
                _ => None,
            }),
            Phase::Main => main_choice(obs, legal),
            Phase::TradeResponse => Some(Action::RejectTrade),
            Phase::TradeConfirm => Some(Action::CancelTrade),
            Phase::GameOver { .. } => None,
        };
        pick.filter(|a| legal.contains(a)).unwrap_or(legal[0])
    }
}
```

- [ ] **Step 5: Run tests**

Run: `cargo test -p settler-bots --test greedy && cargo test`
Expected: 12 passed in `greedy`; full default suite passes.

- [ ] **Step 6: Commit**

```bash
git add bots
git diff --cached | grep -nE 'sk-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{36}|password=[^[:space:]]+' && echo "SECRET FOUND - abort" || git commit -m "bots: GreedyBot with pip-count placement scoring"
git push
```

---

### Task 3: Native arena

**Files:**
- Create: `bots/src/arena.rs`, `bots/tests/arena.rs`
- Modify: `bots/src/lib.rs`

**Interfaces:**
- Consumes: `make_bot`, `BOT_NAMES`, `Bot`, `State`, `legal_actions`, `rng::mix`, `GameConfig::validate`.
- Produces: `settler_bots::arena::{GameRecord { seed: u64, candidate_seat: u8, winner: Option<u8>, vp: [u8; 4], turns: u32, actions: u32 }, bot_seed(seed: u64, seat: usize) -> u64, play_game(seed: u64, config: GameConfig, bots: &mut [Box<dyn Bot>]) -> (Option<u8>, [u8; 4], u32, u32), run_match(candidate: &str, baseline: &str, seeds: std::ops::Range<u64>, config: GameConfig, threads: usize) -> Result<Vec<GameRecord>, String>}`. Records are sorted by `(seed, candidate_seat)`; `vp` is final total VP including hidden VP cards.

- [ ] **Step 1: Write the failing test**

`bots/tests/arena.rs`:

```rust
use settler_bots::arena::{bot_seed, play_game, run_match, GameRecord};
use settler_bots::{make_bot, Bot};
use settler_engine::{Action, GameConfig, Observation};

fn per_seed_rates(records: &[GameRecord]) -> Vec<f64> {
    records
        .chunks(4)
        .map(|g| g.iter().filter(|r| r.winner == Some(r.candidate_seat)).count() as f64 / 4.0)
        .collect()
}

fn z_vs_quarter(rates: &[f64]) -> f64 {
    let n = rates.len() as f64;
    let mean = rates.iter().sum::<f64>() / n;
    let var = rates.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
    (mean - 0.25) / (var / n).sqrt()
}

#[test]
fn every_seed_is_played_in_all_four_seats() {
    let recs = run_match("random", "random", 10..15, GameConfig::default(), 2).unwrap();
    assert_eq!(recs.len(), 20);
    for (i, r) in recs.iter().enumerate() {
        assert_eq!(r.seed, 10 + (i / 4) as u64);
        assert_eq!(r.candidate_seat, (i % 4) as u8);
        assert!(r.turns > 0 && r.actions > 0);
        if let Some(w) = r.winner {
            assert!(r.vp[w as usize] >= 10);
        }
    }
}

#[test]
fn results_do_not_depend_on_thread_count() {
    let a = run_match("greedy", "random", 0..12, GameConfig::default(), 1).unwrap();
    let b = run_match("greedy", "random", 0..12, GameConfig::default(), 5).unwrap();
    assert_eq!(a, b);
}

#[test]
fn bot_streams_depend_only_on_seed_and_seat() {
    // CRN: a seat's bot stream is a function of (seed, seat) alone, so a baseline in seat 2
    // makes the same random choices whichever candidate sits elsewhere.
    assert_eq!(bot_seed(9, 2), bot_seed(9, 2));
    let seeds: std::collections::HashSet<u64> =
        (0..50).flat_map(|seed| (0..4).map(move |seat| bot_seed(seed, seat))).collect();
    assert_eq!(seeds.len(), 200);
    let a = run_match("greedy", "random", 3..6, GameConfig::default(), 1).unwrap();
    let b = run_match("greedy", "random", 3..6, GameConfig::default(), 1).unwrap();
    assert_eq!(a, b);
}

#[test]
fn unknown_bot_and_bad_config_are_errors() {
    let e = run_match("nope", "random", 0..1, GameConfig::default(), 1).unwrap_err();
    assert!(e.contains("unknown bot"), "{e}");
    let bad = GameConfig { max_trade_cards: 3, ..GameConfig::default() };
    assert!(run_match("random", "random", 0..1, bad, 1).is_err());
    assert!(run_match("random", "random", 5..5, GameConfig::default(), 1).unwrap().is_empty());
}

struct Broken;
impl Bot for Broken {
    fn name(&self) -> &'static str {
        "broken"
    }
    fn act(&mut self, _obs: &Observation, _legal: &[Action]) -> Action {
        Action::BuildCity(53)
    }
}

#[test]
#[should_panic(expected = "bot broken chose illegal action")]
fn illegal_bot_action_panics_with_bot_name() {
    let mut bots: Vec<Box<dyn Bot>> = vec![Box::new(Broken), make_bot("random", 1).unwrap(), make_bot("random", 2).unwrap(), make_bot("random", 3).unwrap()];
    play_game(0, GameConfig::default(), &mut bots);
}

#[test]
fn greedy_beats_random_decisively() {
    let recs = run_match("greedy", "random", 0..200, GameConfig::default(), 8).unwrap();
    let z = z_vs_quarter(&per_seed_rates(&recs));
    assert!(z > 3.0, "z = {z}");
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p settler-bots --test arena`
Expected: FAIL to compile — `could not find arena in settler_bots`.

- [ ] **Step 3: Write the implementation**

Add `pub mod arena;` to `bots/src/lib.rs`.

`bots/src/arena.rs`:

```rust
//! Native arena: one candidate against three copies of a baseline, each seed played four
//! times with the candidate rotated through every seat.

use crate::{make_bot, Bot, BOT_NAMES};
use settler_engine::legal::legal_actions;
use settler_engine::rng::mix;
use settler_engine::{GameConfig, State, NUM_PLAYERS};
use std::ops::Range;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GameRecord {
    pub seed: u64,
    pub candidate_seat: u8,
    pub winner: Option<u8>,
    /// Final victory points including hidden VP cards.
    pub vp: [u8; NUM_PLAYERS],
    pub turns: u32,
    pub actions: u32,
}

const BOT_SALT: u64 = 0xB075;

/// A bot's random stream depends only on (seed, seat), so swapping the candidate leaves every
/// baseline's stream unchanged (common random numbers).
pub fn bot_seed(seed: u64, seat: usize) -> u64 {
    mix(mix(seed, BOT_SALT), seat as u64)
}

/// Play one game. Returns (winner, final VP, turns, actions). Panics if a bot picks an illegal action.
pub fn play_game(
    seed: u64,
    config: GameConfig,
    bots: &mut [Box<dyn Bot>],
) -> (Option<u8>, [u8; NUM_PLAYERS], u32, u32) {
    assert_eq!(bots.len(), NUM_PLAYERS, "a game needs exactly 4 bots");
    let mut s = State::new(seed, config);
    let mut buf = Vec::with_capacity(512);
    let mut actions = 0u32;
    while !s.is_over() {
        legal_actions(&s, &mut buf);
        let actor = s.current_actor() as usize;
        let a = bots[actor].act(&s.observation(actor as u8), &buf);
        assert!(
            buf.contains(&a),
            "bot {} chose illegal action {a:?} in phase {:?}",
            bots[actor].name(),
            s.phase
        );
        s.apply(a);
        actions += 1;
    }
    (s.winner(), std::array::from_fn(|p| s.total_vp(p)), s.turn, actions)
}

/// Every seed in `seeds`, four games each (candidate in seat 0..4), spread over `threads`.
/// Records are sorted by (seed, candidate seat) and independent of `threads`.
pub fn run_match(
    candidate: &str,
    baseline: &str,
    seeds: Range<u64>,
    config: GameConfig,
    threads: usize,
) -> Result<Vec<GameRecord>, String> {
    for name in [candidate, baseline] {
        if make_bot(name, 0).is_none() {
            return Err(format!("unknown bot {name:?}; known bots: {BOT_NAMES:?}"));
        }
    }
    config.validate()?;
    let seeds: Vec<u64> = seeds.collect();
    if seeds.is_empty() {
        return Ok(Vec::new());
    }
    let threads = threads.clamp(1, seeds.len());
    let chunk = seeds.len().div_ceil(threads);
    let mut records: Vec<GameRecord> = std::thread::scope(|scope| {
        let workers: Vec<_> = seeds
            .chunks(chunk)
            .map(|part| {
                scope.spawn(move || {
                    let mut out = Vec::with_capacity(part.len() * NUM_PLAYERS);
                    for &seed in part {
                        for cand in 0..NUM_PLAYERS {
                            let mut bots: Vec<Box<dyn Bot>> = (0..NUM_PLAYERS)
                                .map(|seat| {
                                    let name = if seat == cand { candidate } else { baseline };
                                    make_bot(name, bot_seed(seed, seat)).expect("validated above")
                                })
                                .collect();
                            let (winner, vp, turns, actions) = play_game(seed, config, &mut bots);
                            out.push(GameRecord {
                                seed,
                                candidate_seat: cand as u8,
                                winner,
                                vp,
                                turns,
                                actions,
                            });
                        }
                    }
                    out
                })
            })
            .collect();
        workers
            .into_iter()
            .flat_map(|w| w.join().expect("arena worker panicked"))
            .collect()
    });
    records.sort_by_key(|r| (r.seed, r.candidate_seat));
    Ok(records)
}
```

- [ ] **Step 4: Run tests**

Run: `cargo test --release -p settler-bots --test arena && cargo test`
Expected: 6 passed in `arena` (release so the 200-seed match is quick); full default suite passes.

- [ ] **Step 5: Commit**

```bash
git add bots
git diff --cached | grep -nE 'sk-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{36}|password=[^[:space:]]+' && echo "SECRET FOUND - abort" || git commit -m "bots: native CRN arena with seat rotation"
git push
```

---

### Task 4: PyO3 bindings and Python build

**Files:**
- Create: `bindings/Cargo.toml`, `bindings/src/lib.rs`, `bindings/src/convert.rs`, `pyproject.toml`, `alphasettler/__init__.py`, `tests/python/test_engine.py`
- Modify: `Cargo.toml` (add `bindings` to `members` only)

**Interfaces:**
- Consumes: `settler_engine::{Game, GameConfig, Action, ACTION_SPACE_SIZE, Chance, ApplyError, Event, Observation, Phase, Resource, DevCard, PortKind, Response, NUM_PLAYERS}`, `settler_bots::{BOT_NAMES, arena::run_match}`.
- Produces (Python module `alphasettler._engine`, re-exported from `alphasettler`):
  - `Game(seed: int, config: dict | None = None)`; methods `legal_actions() -> list[int]`, `apply(action: int) -> None`, `apply_forced(action: int, chance: dict) -> None`, `observation(viewer: int) -> dict`, `log(viewer: int) -> list[dict]`, `final_vp() -> list[int]` (only once over); properties `seed`, `is_over`, `winner` (`int | None`), `current_actor`, `turn`, `phase` (snake_case str).
  - `run_match(candidate: str, baseline: str, seed_start: int, seeds: int, threads: int, config: dict | None = None) -> list[dict]` with keys `seed, candidate_seat, winner, vp (list[int]), turns, actions`.
  - `action_space_size() -> int`, `describe_action(action: int) -> str`, `bot_names() -> list[str]`.
  - Config dict keys: `vp_to_win, discard_limit, max_offers_per_turn, max_trade_cards, max_turns, catanatron_compat`; unknown keys and invalid configs raise `ValueError`.
  - Chance dict: `{"roll": [d1, d2]}` (optional `"discards": [[5 ints] × 4]`), `{"steal": "<resource>"}`, or `{"dev": "<card>"}`; resource names `wood brick sheep wheat ore`, card names `knight victory_point road_building year_of_plenty monopoly`.
  - Every refusal (bad id, illegal action, impossible chance, bad viewer, final_vp before the end) raises `ValueError`.

- [ ] **Step 1: Crates and build config**

Root `Cargo.toml`: change `members` to `["engine", "bots", "bindings"]` (keep `default-members = ["engine", "bots"]`).

`bindings/Cargo.toml`:

```toml
[package]
name = "settler-bindings"
version = "0.1.0"
edition = "2021"
publish = false

[lib]
name = "_engine"
crate-type = ["cdylib"]

[dependencies]
pyo3 = "=0.29.3"
settler-engine = { path = "../engine" }
settler-bots = { path = "../bots" }
```

`pyproject.toml`:

```toml
[build-system]
requires = ["maturin>=1.15,<2"]
build-backend = "maturin"

[project]
name = "alphasettler"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = []

[project.optional-dependencies]
dev = ["maturin==1.15.0", "pytest==9.1.1"]

[project.scripts]
alphasettler = "alphasettler.cli:main"

[tool.maturin]
manifest-path = "bindings/Cargo.toml"
module-name = "alphasettler._engine"
python-source = "."

[tool.pytest.ini_options]
testpaths = ["tests/python"]
```

`alphasettler/__init__.py`:

```python
"""AlphaSettler: a Catan engine, baseline bots and a benchmark arena."""

from alphasettler._engine import (
    Game,
    action_space_size,
    bot_names,
    describe_action,
    run_match,
)

__all__ = ["Game", "action_space_size", "bot_names", "describe_action", "run_match"]
```

Dev environment (one time):

```bash
python3 -m venv .venv
.venv/bin/pip3 install -q maturin==1.15.0 pytest==9.1.1
```

- [ ] **Step 2: Write the failing test**

`tests/python/test_engine.py`:

```python
import random
import threading
import time

import pytest

from alphasettler import Game, action_space_size, bot_names, describe_action, run_match


def play_out(g: Game, seed: int = 0) -> None:
    rng = random.Random(seed)
    while not g.is_over:
        g.apply(rng.choice(g.legal_actions()))


def test_new_game_is_in_setup():
    g = Game(1)
    assert g.phase == "setup_settlement"
    assert g.current_actor == 0
    assert len(g.legal_actions()) == 54
    assert g.seed == 1


def test_actions_and_helpers():
    assert action_space_size() == 665
    assert describe_action(0) == "Roll"
    assert describe_action(25) == "BuildSettlement(0)"
    assert set(bot_names()) >= {"random", "greedy"}


def test_illegal_actions_raise_and_change_nothing():
    g = Game(1)
    before = g.legal_actions()
    with pytest.raises(ValueError, match="Illegal"):
        g.apply(0)  # Roll during setup
    with pytest.raises(ValueError):
        g.apply(665)
    with pytest.raises(ValueError):
        g.apply(233)  # undecodable: maritime give == get
    assert g.legal_actions() == before
    assert g.log(0) == []


def test_bad_inputs_raise_value_error():
    with pytest.raises(ValueError, match="unknown config key"):
        Game(1, {"nope": 1})
    with pytest.raises(ValueError):
        Game(1, {"max_trade_cards": 3})
    g = Game(1)
    with pytest.raises(ValueError):
        g.observation(4)
    with pytest.raises(ValueError):
        g.final_vp()
    with pytest.raises(ValueError, match="unknown bot"):
        run_match("nope", "random", 0, 1, 1)


def test_full_game_and_final_vp():
    g = Game(3, {"max_offers_per_turn": 0})
    play_out(g)
    assert g.phase == "game_over"
    vp = g.final_vp()
    assert len(vp) == 4
    if g.winner is not None:
        assert vp[g.winner] >= 10


def test_observation_shape_and_privacy():
    g = Game(5)
    play_out_steps = 0
    rng = random.Random(1)
    while play_out_steps < 400 and not g.is_over:
        g.apply(rng.choice(g.legal_actions()))
        play_out_steps += 1
    o = g.observation(0)
    for key in ["phase", "my_hand", "hand_counts", "settlements", "roads", "public_vp", "board", "bank", "config"]:
        assert key in o
    assert isinstance(o["my_hand"], list) and len(o["my_hand"]) == 5
    assert isinstance(o["hand_counts"], list) and len(o["hand_counts"]) == 4
    assert sum(o["my_hand"]) == o["hand_counts"][0]
    assert "hand" not in o  # no opponent hands
    assert len(o["board"]["tile_resource"]) == 19
    assert o["config"]["vp_to_win"] == 10


def test_event_log_types():
    g = Game(1)
    g.apply(g.legal_actions()[0])
    log = g.log(0)
    assert log[0]["type"] == "built_settlement"
    assert log[0]["player"] == 0


def test_forced_roll_and_impossible_chance():
    g = Game(2)
    while g.phase in ("setup_settlement", "setup_road"):
        g.apply(g.legal_actions()[0])
    assert g.phase == "pre_roll"
    with pytest.raises(ValueError, match="ImpossibleChance"):
        g.apply_forced(0, {"roll": [0, 7]})
    g.apply_forced(0, {"roll": [3, 4]})
    assert g.phase in ("move_robber", "discard")
    rolled = [e for e in g.log(1) if e["type"] == "rolled"]
    assert rolled[-1]["dice"] == [3, 4]


def test_run_match_records():
    recs = run_match("greedy", "random", 0, 3, 2)
    assert len(recs) == 12
    assert [r["candidate_seat"] for r in recs[:4]] == [0, 1, 2, 3]
    assert set(recs[0]) == {"seed", "candidate_seat", "winner", "vp", "turns", "actions"}
    assert isinstance(recs[0]["vp"], list) and len(recs[0]["vp"]) == 4


def test_run_match_is_deterministic_across_threads():
    assert run_match("greedy", "random", 10, 8, 1) == run_match("greedy", "random", 10, 8, 4)


def test_run_match_releases_the_gil():
    ticks = []
    done = threading.Event()

    def ticker():
        while not done.is_set():
            ticks.append(time.perf_counter())
            time.sleep(0.001)

    t = threading.Thread(target=ticker)
    t.start()
    start = time.perf_counter()
    run_match("greedy", "random", 0, 300, 1)
    elapsed = time.perf_counter() - start
    done.set()
    t.join()
    during = [x for x in ticks if start < x < start + elapsed]
    assert elapsed > 0.05, "match too short to observe"
    assert len(during) >= 5, f"python thread starved: {len(during)} ticks in {elapsed:.2f}s"
```

- [ ] **Step 3: Run test to verify it fails**

Run: `.venv/bin/python3 -m pytest tests/python/test_engine.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'alphasettler._engine'`.

- [ ] **Step 4: Write the bindings**

`bindings/src/convert.rs`:

```rust
//! Conversions between engine types and plain Python values.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use settler_engine::*;

pub fn value_error(msg: impl Into<String>) -> PyErr {
    PyValueError::new_err(msg.into())
}

pub fn parse_config(config: Option<&Bound<'_, PyDict>>) -> PyResult<GameConfig> {
    let mut c = GameConfig::default();
    if let Some(d) = config {
        for (k, v) in d.iter() {
            let key: String = k.extract()?;
            match key.as_str() {
                "vp_to_win" => c.vp_to_win = v.extract()?,
                "discard_limit" => c.discard_limit = v.extract()?,
                "max_offers_per_turn" => c.max_offers_per_turn = v.extract()?,
                "max_trade_cards" => c.max_trade_cards = v.extract()?,
                "max_turns" => c.max_turns = v.extract()?,
                "catanatron_compat" => c.catanatron_compat = v.extract()?,
                other => return Err(value_error(format!("unknown config key {other:?}"))),
            }
        }
    }
    c.validate().map_err(value_error)?;
    Ok(c)
}

pub fn config_dict<'py>(py: Python<'py>, c: &GameConfig) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("vp_to_win", c.vp_to_win)?;
    d.set_item("discard_limit", c.discard_limit)?;
    d.set_item("max_offers_per_turn", c.max_offers_per_turn)?;
    d.set_item("max_trade_cards", c.max_trade_cards)?;
    d.set_item("max_turns", c.max_turns)?;
    d.set_item("catanatron_compat", c.catanatron_compat)?;
    Ok(d)
}

pub fn action_from_id(id: u32) -> PyResult<Action> {
    if id >= ACTION_SPACE_SIZE as u32 {
        return Err(value_error(format!("action id {id} out of range 0..{ACTION_SPACE_SIZE}")));
    }
    Action::decode(id as u16).ok_or_else(|| value_error(format!("action id {id} is not a valid action")))
}

pub fn counts(xs: &[u8]) -> Vec<u32> {
    xs.iter().map(|&x| x as u32).collect()
}

pub fn node_list(mask: u64) -> Vec<u32> {
    bits64(mask).map(|n| n as u32).collect()
}

pub fn edge_list(mask: u128) -> Vec<u32> {
    bits128(mask).map(|e| e as u32).collect()
}

pub fn phase_name(p: Phase) -> &'static str {
    match p {
        Phase::SetupSettlement => "setup_settlement",
        Phase::SetupRoad { .. } => "setup_road",
        Phase::PreRoll => "pre_roll",
        Phase::Discard => "discard",
        Phase::MoveRobber => "move_robber",
        Phase::Steal => "steal",
        Phase::Main => "main",
        Phase::RoadBuilding { .. } => "road_building",
        Phase::TradeResponse => "trade_response",
        Phase::TradeConfirm => "trade_confirm",
        Phase::GameOver { .. } => "game_over",
    }
}

const RESOURCE_NAMES: [&str; 5] = ["wood", "brick", "sheep", "wheat", "ore"];
const DEV_NAMES: [&str; 5] = ["knight", "victory_point", "road_building", "year_of_plenty", "monopoly"];

pub fn resource_name(r: Resource) -> &'static str {
    RESOURCE_NAMES[r.index()]
}

pub fn dev_name(c: DevCard) -> &'static str {
    DEV_NAMES[c.index()]
}

fn parse_resource(name: &str) -> PyResult<Resource> {
    RESOURCE_NAMES
        .iter()
        .position(|&n| n == name)
        .map(Resource::from_index)
        .ok_or_else(|| value_error(format!("unknown resource {name:?}")))
}

fn parse_dev(name: &str) -> PyResult<DevCard> {
    DEV_NAMES
        .iter()
        .position(|&n| n == name)
        .map(DevCard::from_index)
        .ok_or_else(|| value_error(format!("unknown dev card {name:?}")))
}

pub fn parse_chance(d: &Bound<'_, PyDict>) -> PyResult<Chance> {
    if let Some(roll) = d.get_item("roll")? {
        let dice: (u8, u8) = roll.extract()?;
        let discards = match d.get_item("discards")? {
            None => None,
            Some(v) if v.is_none() => None,
            Some(v) => Some(v.extract::<[[u8; 5]; NUM_PLAYERS]>()?),
        };
        return Ok(Chance::Roll { dice, discards });
    }
    if let Some(r) = d.get_item("steal")? {
        return Ok(Chance::Steal(parse_resource(&r.extract::<String>()?)?));
    }
    if let Some(c) = d.get_item("dev")? {
        return Ok(Chance::Dev(parse_dev(&c.extract::<String>()?)?));
    }
    Err(value_error("chance must have one of the keys 'roll', 'steal', 'dev'"))
}

fn trade_dict<'py>(py: Python<'py>, t: &PendingTrade) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("give", counts(&t.give))?;
    d.set_item("get", counts(&t.get))?;
    let responses: Vec<&str> = t
        .responses
        .iter()
        .map(|r| match r {
            Response::Pending => "pending",
            Response::Accepted => "accepted",
            Response::Rejected => "rejected",
        })
        .collect();
    d.set_item("responses", responses)?;
    d.set_item("next_responder", t.next_responder)?;
    Ok(d)
}

fn board_dict<'py>(py: Python<'py>, b: &Board) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    let tiles: Vec<Option<&str>> = b.tile_resource.iter().map(|r| r.map(resource_name)).collect();
    d.set_item("tile_resource", tiles)?;
    d.set_item("tile_number", counts(&b.tile_number))?;
    let ports = PyList::empty(py);
    for &(edge, kind) in &b.ports {
        let k = match kind {
            PortKind::Generic => "generic",
            PortKind::Specific(r) => resource_name(r),
        };
        ports.append((edge as u32, k))?;
    }
    d.set_item("ports", ports)?;
    Ok(d)
}

pub fn observation_dict<'py>(py: Python<'py>, o: &Observation) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("viewer", o.viewer)?;
    d.set_item("config", config_dict(py, &o.config)?)?;
    d.set_item("phase", phase_name(o.phase))?;
    d.set_item("current", o.current)?;
    d.set_item("actor", o.actor)?;
    d.set_item("turn", o.turn)?;
    d.set_item("board", board_dict(py, &o.board)?)?;
    d.set_item("robber", o.robber)?;
    d.set_item("bank", counts(&o.bank))?;
    d.set_item("dev_deck_remaining", o.dev_deck_remaining)?;
    d.set_item("my_hand", counts(&o.my_hand))?;
    d.set_item("my_dev_cards", counts(&o.my_dev_cards))?;
    d.set_item("my_new_dev_cards", counts(&o.my_new_dev_cards))?;
    d.set_item("hand_counts", counts(&o.hand_counts))?;
    d.set_item("dev_card_counts", counts(&o.dev_card_counts))?;
    d.set_item("new_dev_card_counts", counts(&o.new_dev_card_counts))?;
    let played: Vec<Vec<u32>> = o.dev_cards_played.iter().map(|p| counts(p)).collect();
    d.set_item("dev_cards_played", played)?;
    let settlements: Vec<Vec<u32>> = o.settlements.iter().map(|&m| node_list(m)).collect();
    d.set_item("settlements", settlements)?;
    let cities: Vec<Vec<u32>> = o.cities.iter().map(|&m| node_list(m)).collect();
    d.set_item("cities", cities)?;
    let roads: Vec<Vec<u32>> = o.roads.iter().map(|&m| edge_list(m)).collect();
    d.set_item("roads", roads)?;
    d.set_item("knights_played", counts(&o.knights_played))?;
    d.set_item("public_vp", counts(&o.public_vp))?;
    d.set_item("longest_road_owner", o.longest_road_owner)?;
    d.set_item("largest_army_owner", o.largest_army_owner)?;
    match &o.trade {
        Some(t) => d.set_item("trade", trade_dict(py, t)?)?,
        None => d.set_item("trade", py.None())?,
    }
    d.set_item("dev_played_this_turn", o.dev_played_this_turn)?;
    d.set_item("offers_this_turn", o.offers_this_turn)?;
    d.set_item("discard_remaining", counts(&o.discard_remaining))?;
    d.set_item("robber_return", phase_name(o.robber_return))?;
    Ok(d)
}

pub fn event_dict<'py>(py: Python<'py>, e: &Event) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    match *e {
        Event::BuiltSettlement { player, node } => {
            d.set_item("type", "built_settlement")?;
            d.set_item("player", player)?;
            d.set_item("node", node)?;
        }
        Event::BuiltCity { player, node } => {
            d.set_item("type", "built_city")?;
            d.set_item("player", player)?;
            d.set_item("node", node)?;
        }
        Event::BuiltRoad { player, edge } => {
            d.set_item("type", "built_road")?;
            d.set_item("player", player)?;
            d.set_item("edge", edge)?;
        }
        Event::Rolled { player, dice } => {
            d.set_item("type", "rolled")?;
            d.set_item("player", player)?;
            d.set_item("dice", vec![dice.0 as u32, dice.1 as u32])?;
        }
        Event::Produced { player, resources } => {
            d.set_item("type", "produced")?;
            d.set_item("player", player)?;
            d.set_item("resources", counts(&resources))?;
        }
        Event::Discarded { player, resource } => {
            d.set_item("type", "discarded")?;
            d.set_item("player", player)?;
            d.set_item("resource", resource_name(resource))?;
        }
        Event::RobberMoved { player, tile } => {
            d.set_item("type", "robber_moved")?;
            d.set_item("player", player)?;
            d.set_item("tile", tile)?;
        }
        Event::Stole { thief, victim, resource } => {
            d.set_item("type", "stole")?;
            d.set_item("thief", thief)?;
            d.set_item("victim", victim)?;
            d.set_item("resource", resource.map(resource_name))?;
        }
        Event::BoughtDev { player, card } => {
            d.set_item("type", "bought_dev")?;
            d.set_item("player", player)?;
            d.set_item("card", card.map(dev_name))?;
        }
        Event::PlayedDev { player, card } => {
            d.set_item("type", "played_dev")?;
            d.set_item("player", player)?;
            d.set_item("card", dev_name(card))?;
        }
        Event::MonopolyTaken { player, resource, amount } => {
            d.set_item("type", "monopoly_taken")?;
            d.set_item("player", player)?;
            d.set_item("resource", resource_name(resource))?;
            d.set_item("amount", amount)?;
        }
        Event::YearOfPlentyTaken { player, resources } => {
            d.set_item("type", "year_of_plenty_taken")?;
            d.set_item("player", player)?;
            d.set_item("resources", counts(&resources))?;
        }
        Event::MaritimeTraded { player, gave, got } => {
            d.set_item("type", "maritime_traded")?;
            d.set_item("player", player)?;
            d.set_item("gave", counts(&gave))?;
            d.set_item("got", counts(&got))?;
        }
        Event::TradeOffered { player, give, get } => {
            d.set_item("type", "trade_offered")?;
            d.set_item("player", player)?;
            d.set_item("give", counts(&give))?;
            d.set_item("get", counts(&get))?;
        }
        Event::TradeResponded { player, accepted } => {
            d.set_item("type", "trade_responded")?;
            d.set_item("player", player)?;
            d.set_item("accepted", accepted)?;
        }
        Event::TradeConfirmed { offerer, partner, offerer_gave, partner_gave } => {
            d.set_item("type", "trade_confirmed")?;
            d.set_item("offerer", offerer)?;
            d.set_item("partner", partner)?;
            d.set_item("offerer_gave", counts(&offerer_gave))?;
            d.set_item("partner_gave", counts(&partner_gave))?;
        }
        Event::TradeCancelled { player } => {
            d.set_item("type", "trade_cancelled")?;
            d.set_item("player", player)?;
        }
        Event::TurnEnded { player } => {
            d.set_item("type", "turn_ended")?;
            d.set_item("player", player)?;
        }
        Event::GameOver { winner } => {
            d.set_item("type", "game_over")?;
            d.set_item("winner", winner)?;
        }
    }
    Ok(d)
}
```

`bindings/src/lib.rs`:

```rust
//! Python bindings: the extension module `alphasettler._engine`. Conversions only; rules live
//! in `settler-engine`, bots and the arena in `settler-bots`.

mod convert;

use convert::*;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use settler_engine::{ACTION_SPACE_SIZE, NUM_PLAYERS};

#[pyclass(name = "Game", module = "alphasettler._engine")]
struct PyGame {
    inner: settler_engine::Game,
}

#[pymethods]
impl PyGame {
    #[new]
    #[pyo3(signature = (seed, config=None))]
    fn new(seed: u64, config: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        Ok(PyGame {
            inner: settler_engine::Game::new(seed, parse_config(config)?),
        })
    }

    fn legal_actions(&self) -> Vec<u32> {
        self.inner.legal_actions().iter().map(|a| a.encode() as u32).collect()
    }

    fn apply(&mut self, action: u32) -> PyResult<()> {
        let a = action_from_id(action)?;
        self.inner.apply(a).map_err(|e| value_error(format!("{e:?}")))
    }

    fn apply_forced(&mut self, action: u32, chance: &Bound<'_, PyDict>) -> PyResult<()> {
        let a = action_from_id(action)?;
        let c = parse_chance(chance)?;
        self.inner.apply_forced(a, Some(c)).map_err(|e| value_error(format!("{e:?}")))
    }

    fn observation<'py>(&self, py: Python<'py>, viewer: u8) -> PyResult<Bound<'py, PyDict>> {
        if viewer as usize >= NUM_PLAYERS {
            return Err(value_error(format!("viewer {viewer} out of range 0..{NUM_PLAYERS}")));
        }
        observation_dict(py, &self.inner.observation(viewer))
    }

    fn log<'py>(&self, py: Python<'py>, viewer: u8) -> PyResult<Bound<'py, PyList>> {
        if viewer as usize >= NUM_PLAYERS {
            return Err(value_error(format!("viewer {viewer} out of range 0..{NUM_PLAYERS}")));
        }
        let out = PyList::empty(py);
        for e in self.inner.log_for(viewer) {
            out.append(event_dict(py, &e)?)?;
        }
        Ok(out)
    }

    /// Final victory points (including hidden VP cards); only available once the game is over.
    fn final_vp(&self) -> PyResult<Vec<u32>> {
        let s = self.inner.state();
        if !s.is_over() {
            return Err(value_error("final_vp is only available once the game is over"));
        }
        Ok((0..NUM_PLAYERS).map(|p| s.total_vp(p) as u32).collect())
    }

    #[getter]
    fn seed(&self) -> u64 {
        self.inner.state().seed
    }

    #[getter]
    fn is_over(&self) -> bool {
        self.inner.state().is_over()
    }

    #[getter]
    fn winner(&self) -> Option<u8> {
        self.inner.state().winner()
    }

    #[getter]
    fn current_actor(&self) -> u8 {
        let s = self.inner.state();
        if s.is_over() { s.current } else { s.current_actor() }
    }

    #[getter]
    fn turn(&self) -> u32 {
        self.inner.state().turn
    }

    #[getter]
    fn phase(&self) -> &'static str {
        phase_name(self.inner.state().phase)
    }
}

#[pyfunction]
fn action_space_size() -> usize {
    ACTION_SPACE_SIZE
}

#[pyfunction]
fn describe_action(action: u32) -> PyResult<String> {
    Ok(format!("{:?}", action_from_id(action)?))
}

#[pyfunction]
fn bot_names() -> Vec<&'static str> {
    settler_bots::BOT_NAMES.to_vec()
}

/// Native arena: `seeds` seeds starting at `seed_start`, each played four times with the
/// candidate rotated through every seat. Releases the GIL while games run.
#[pyfunction]
#[pyo3(signature = (candidate, baseline, seed_start, seeds, threads, config=None))]
fn run_match<'py>(
    py: Python<'py>,
    candidate: &str,
    baseline: &str,
    seed_start: u64,
    seeds: u64,
    threads: usize,
    config: Option<&Bound<'py, PyDict>>,
) -> PyResult<Bound<'py, PyList>> {
    let config = parse_config(config)?;
    let (candidate, baseline) = (candidate.to_owned(), baseline.to_owned());
    let records = py
        .detach(|| {
            settler_bots::arena::run_match(&candidate, &baseline, seed_start..seed_start + seeds, config, threads)
        })
        .map_err(value_error)?;
    let out = PyList::empty(py);
    for r in records {
        let d = PyDict::new(py);
        d.set_item("seed", r.seed)?;
        d.set_item("candidate_seat", r.candidate_seat)?;
        d.set_item("winner", r.winner)?;
        d.set_item("vp", counts(&r.vp))?;
        d.set_item("turns", r.turns)?;
        d.set_item("actions", r.actions)?;
        out.append(d)?;
    }
    Ok(out)
}

#[pymodule]
fn _engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyGame>()?;
    m.add_function(wrap_pyfunction!(action_space_size, m)?)?;
    m.add_function(wrap_pyfunction!(describe_action, m)?)?;
    m.add_function(wrap_pyfunction!(bot_names, m)?)?;
    m.add_function(wrap_pyfunction!(run_match, m)?)?;
    Ok(())
}
```

- [ ] **Step 5: Build and run the tests**

```bash
source "$HOME/.cargo/env"
VIRTUAL_ENV=$PWD/.venv .venv/bin/maturin develop --release
.venv/bin/python3 -m pytest tests/python/test_engine.py -q
cargo test
cargo clippy -p settler-bindings -- -D warnings
```

Expected: 11 passed; default Rust suite passes; clippy clean.

- [ ] **Step 6: Audit Python tooling**

```bash
.venv/bin/pip3 install -q pip-audit
.venv/bin/pip-audit
```

Expected: `No known vulnerabilities found` (record anything else in the commit message).

- [ ] **Step 7: Commit**

```bash
git add Cargo.toml Cargo.lock bindings pyproject.toml alphasettler tests/python
git diff --cached | grep -nE 'sk-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{36}|password=[^[:space:]]+' && echo "SECRET FOUND - abort" || git commit -m "bindings: PyO3 module alphasettler._engine (Game, run_match, helpers)"
git push
```

---

### Task 5: Win-rate statistics

**Files:**
- Create: `alphasettler/stats.py`, `tests/python/test_stats.py`

**Interfaces:**
- Consumes: record dicts from `run_match` (`seed, candidate_seat, winner, vp, turns, actions`).
- Produces: `alphasettler.stats.{candidate_won(record) -> bool, per_seed_win_rates(records) -> dict[int, float], Summary, summarize(records, null=0.25, z_crit=1.96) -> Summary, Paired, paired(a_records, b_records) -> Paired}`. `Summary` fields: `seeds, games, wins, win_rate, ci_low, ci_high, z_vs_null, null, mean_vp, mean_turns`. `Paired` fields: `seeds, mean_diff, se, z`.

- [ ] **Step 1: Write the failing test**

`tests/python/test_stats.py`:

```python
import math

import pytest

from alphasettler.stats import candidate_won, paired, per_seed_win_rates, summarize


def rec(seed, seat, winner, vp=None, turns=100):
    return {"seed": seed, "candidate_seat": seat, "winner": winner,
            "vp": vp or [5, 5, 5, 5], "turns": turns, "actions": 1000}


def four(seed, wins):
    """Four rotated games for `seed`; the candidate wins in the first `wins` seats."""
    return [rec(seed, s, s if s < wins else (s + 1) % 4) for s in range(4)]


def test_candidate_won():
    assert candidate_won(rec(0, 2, 2))
    assert not candidate_won(rec(0, 2, 1))


def test_draws_count_as_losses():
    assert not candidate_won(rec(0, 0, None))
    s = summarize([rec(0, seat, None) for seat in range(4)] + four(1, 2))
    assert s.wins == 2
    assert s.win_rate == pytest.approx(0.25)


def test_per_seed_rates():
    rates = per_seed_win_rates(four(0, 2) + four(1, 0))
    assert rates == {0: 0.5, 1: 0.0}


def test_summary_numbers():
    s = summarize(four(0, 2) + four(1, 0))
    assert (s.seeds, s.games, s.wins) == (2, 8, 2)
    assert s.win_rate == pytest.approx(0.25)
    # per-seed rates 0.5 and 0.0: sample sd = 0.3536, se = 0.25
    assert s.z_vs_null == pytest.approx(0.0)
    assert s.ci_low == pytest.approx(0.0)
    assert s.ci_high == pytest.approx(0.25 + 1.96 * 0.25)
    assert s.mean_vp == pytest.approx(5.0)
    assert s.mean_turns == pytest.approx(100.0)


def test_z_against_null():
    records = four(0, 2) + four(1, 1) + four(2, 2) + four(3, 1)
    s = summarize(records)
    # rates 0.5, 0.25, 0.5, 0.25: mean 0.375, sd 0.1443, se 0.07217, z = 1.732
    assert s.win_rate == pytest.approx(0.375)
    assert s.z_vs_null == pytest.approx(0.125 / (0.1443375673 / 2), rel=1e-6)


def test_zero_variance_and_single_seed():
    all_wins = summarize(four(0, 4) + four(1, 4))
    assert all_wins.win_rate == 1.0
    assert all_wins.z_vs_null == math.inf
    assert all_wins.ci_high == 1.0
    all_losses = summarize(four(0, 0) + four(1, 0))
    assert all_losses.z_vs_null == -math.inf
    at_null = summarize(four(0, 1) + four(1, 1))
    assert at_null.z_vs_null == 0.0
    one = summarize(four(7, 3))
    assert one.seeds == 1 and one.win_rate == 0.75 and one.z_vs_null == math.inf
    with pytest.raises(ValueError):
        summarize([])


def test_paired():
    a = four(0, 3) + four(1, 2)
    b = four(0, 1) + four(1, 1)
    p = paired(a, b)
    # diffs 0.5 and 0.25: mean 0.375, sd 0.1768, se 0.125, z 3.0
    assert p.seeds == 2
    assert p.mean_diff == pytest.approx(0.375)
    assert p.se == pytest.approx(0.125)
    assert p.z == pytest.approx(3.0)
    with pytest.raises(ValueError, match="same seeds"):
        paired(four(0, 1), four(1, 1))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python3 -m pytest tests/python/test_stats.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'alphasettler.stats'`.

- [ ] **Step 3: Write the implementation**

`alphasettler/stats.py`:

```python
"""Win-rate statistics with the seed as the unit of analysis.

Each seed is played four times with the candidate rotated through every seat, so a seed's
win rate is in {0, 0.25, 0.5, 0.75, 1}. Comparisons use per-seed rates, which also makes
two candidates evaluated on the same seeds directly paired (common random numbers).
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

Record = Mapping[str, object]


def candidate_won(r: Record) -> bool:
    """A draw (no winner at the turn limit) counts as a loss."""
    return r["winner"] is not None and r["winner"] == r["candidate_seat"]


def per_seed_win_rates(records: Iterable[Record]) -> dict[int, float]:
    games: dict[int, list[float]] = defaultdict(list)
    for r in records:
        games[int(r["seed"])].append(1.0 if candidate_won(r) else 0.0)
    return {seed: sum(v) / len(v) for seed, v in games.items()}


def _mean_se(xs: list[float]) -> tuple[float, float]:
    if not xs:
        raise ValueError("no seeds to summarize")
    n = len(xs)
    mean = sum(xs) / n
    if n == 1:
        return mean, 0.0
    var = sum((x - mean) ** 2 for x in xs) / (n - 1)
    return mean, math.sqrt(var / n)


def _z(effect: float, se: float) -> float:
    if se > 0:
        return effect / se
    if effect == 0:
        return 0.0
    return math.copysign(math.inf, effect)


@dataclass(frozen=True)
class Summary:
    seeds: int
    games: int
    wins: int
    win_rate: float
    ci_low: float
    ci_high: float
    z_vs_null: float
    null: float
    mean_vp: float
    mean_turns: float


def summarize(records: Iterable[Record], null: float = 0.25, z_crit: float = 1.96) -> Summary:
    records = list(records)
    rates = per_seed_win_rates(records)
    mean, se = _mean_se(list(rates.values()))
    vp = [r["vp"][r["candidate_seat"]] for r in records]
    turns = [r["turns"] for r in records]
    return Summary(
        seeds=len(rates),
        games=len(records),
        wins=sum(candidate_won(r) for r in records),
        win_rate=mean,
        ci_low=max(0.0, mean - z_crit * se),
        ci_high=min(1.0, mean + z_crit * se),
        z_vs_null=_z(mean - null, se),
        null=null,
        mean_vp=sum(vp) / len(vp),
        mean_turns=sum(turns) / len(turns),
    )


@dataclass(frozen=True)
class Paired:
    seeds: int
    mean_diff: float
    se: float
    z: float


def paired(a: Iterable[Record], b: Iterable[Record]) -> Paired:
    """Per-seed difference in candidate win rate, a minus b, over identical seed sets."""
    ra, rb = per_seed_win_rates(a), per_seed_win_rates(b)
    if ra.keys() != rb.keys():
        raise ValueError("paired comparison needs the same seeds on both sides")
    diffs = [ra[s] - rb[s] for s in sorted(ra)]
    mean, se = _mean_se(diffs)
    return Paired(seeds=len(diffs), mean_diff=mean, se=se, z=_z(mean, se))
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python3 -m pytest tests/python -q`
Expected: all pass (8 new in `test_stats.py`).

- [ ] **Step 5: Commit**

```bash
git add alphasettler/stats.py tests/python/test_stats.py
git diff --cached | grep -nE 'sk-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{36}|password=[^[:space:]]+' && echo "SECRET FOUND - abort" || git commit -m "alphasettler: per-seed win-rate statistics and paired comparison"
git push
```

---

### Task 6: Arena runner, JSONL output, and CLI

**Files:**
- Create: `alphasettler/arena.py`, `alphasettler/cli.py`, `alphasettler/__main__.py`, `tests/python/test_arena.py`, `tests/python/test_cli.py`

**Interfaces:**
- Consumes: `alphasettler._engine.run_match`, `alphasettler.stats.{summarize, paired, Summary}`.
- Produces: `alphasettler.arena.{run(candidate, baseline, seeds, seed_start=0, threads=None, config=None, out=None) -> tuple[list[dict], Summary], write_jsonl(path, candidate, baseline, records) -> Path, format_summary(label: str, s: Summary) -> str}`; `alphasettler.cli.main(argv: list[str] | None = None) -> int`; console script `alphasettler`; `python3 -m alphasettler`.

CLI:

```
alphasettler arena --candidate X --baseline Y --seeds N [--seed-start S] [--threads T] [--out PATH] [--no-trades]
alphasettler compare --a X --b Y --baseline Z --seeds N [--seed-start S] [--threads T] [--out-dir DIR] [--no-trades]
```

Default `--out` is `runs/<YYYYmmdd-HHMMSS>-<candidate>-vs-<baseline>.jsonl`; default `--out-dir` is `runs/`. Errors (`ValueError`, non-positive seeds) print `error: <message>` to stderr and exit 2.

- [ ] **Step 1: Write the failing tests**

`tests/python/test_arena.py`:

```python
import json

from alphasettler.arena import format_summary, run


def test_run_writes_jsonl_and_summarizes(tmp_path):
    out = tmp_path / "x" / "games.jsonl"
    records, summary = run("greedy", "random", seeds=5, seed_start=100, threads=2, out=out)
    assert len(records) == 20
    lines = out.read_text().splitlines()
    assert len(lines) == 20
    first = json.loads(lines[0])
    assert first["candidate"] == "greedy" and first["baseline"] == "random"
    assert first["seed"] == 100 and first["candidate_seat"] == 0
    assert summary.seeds == 5 and summary.games == 20
    text = format_summary("greedy vs random", summary)
    assert "win rate" in text and "z vs 0.25" in text


def test_greedy_beats_random_with_z_above_3():
    _, summary = run("greedy", "random", seeds=300)
    assert summary.z_vs_null > 3, summary


def test_random_vs_random_is_near_the_null():
    _, summary = run("random", "random", seeds=300)
    assert abs(summary.z_vs_null) < 4, summary
```

`tests/python/test_cli.py`:

```python
import subprocess
import sys


def cli(*args, cwd=None):
    return subprocess.run([sys.executable, "-m", "alphasettler", *args], capture_output=True, text=True, cwd=cwd)


def test_cli_arena(tmp_path):
    out = tmp_path / "g.jsonl"
    r = cli("arena", "--candidate", "greedy", "--baseline", "random", "--seeds", "4", "--out", str(out))
    assert r.returncode == 0, r.stderr
    assert "greedy vs random: win rate" in r.stdout
    assert f"wrote {out}" in r.stdout
    assert len(out.read_text().splitlines()) == 16


def test_cli_arena_default_output_goes_to_runs(tmp_path):
    r = cli("arena", "--candidate", "random", "--baseline", "random", "--seeds", "2", "--no-trades", cwd=tmp_path)
    assert r.returncode == 0, r.stderr
    files = list((tmp_path / "runs").glob("*-random-vs-random.jsonl"))
    assert len(files) == 1


def test_cli_compare(tmp_path):
    r = cli("compare", "--a", "greedy", "--b", "random", "--baseline", "random", "--seeds", "6",
            "--out-dir", str(tmp_path))
    assert r.returncode == 0, r.stderr
    assert "greedy vs random: win rate" in r.stdout
    assert "random vs random: win rate" in r.stdout
    assert "paired greedy - random:" in r.stdout
    assert len(list(tmp_path.glob("*.jsonl"))) == 2


def test_cli_rejects_unknown_bot_and_zero_seeds():
    r = cli("arena", "--candidate", "nope", "--baseline", "random", "--seeds", "2")
    assert r.returncode == 2
    assert "unknown bot" in r.stderr
    r = cli("arena", "--candidate", "random", "--baseline", "random", "--seeds", "0")
    assert r.returncode == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python3 -m pytest tests/python/test_arena.py tests/python/test_cli.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'alphasettler.arena'` / `No module named alphasettler.__main__`.

- [ ] **Step 3: Write the implementation**

`alphasettler/arena.py`:

```python
"""Candidate-vs-baseline matches in the native Rust arena, with per-game JSONL output."""

from __future__ import annotations

import json
import os
from pathlib import Path

from alphasettler._engine import run_match
from alphasettler.stats import Summary, summarize


def write_jsonl(path: str | Path, candidate: str, baseline: str, records: list[dict]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps({"candidate": candidate, "baseline": baseline, **r}) + "\n")
    return path


def run(
    candidate: str,
    baseline: str,
    seeds: int,
    seed_start: int = 0,
    threads: int | None = None,
    config: dict | None = None,
    out: str | Path | None = None,
) -> tuple[list[dict], Summary]:
    if seeds <= 0:
        raise ValueError("seeds must be positive")
    records = run_match(candidate, baseline, seed_start, seeds, threads or os.cpu_count() or 1, config)
    if out is not None:
        write_jsonl(out, candidate, baseline, records)
    return records, summarize(records)


def format_summary(label: str, s: Summary) -> str:
    return (
        f"{label}: win rate {s.win_rate:.3f} [{s.ci_low:.3f}, {s.ci_high:.3f}] 95% CI over {s.seeds} seeds "
        f"({s.games} games, {s.wins} wins), z vs {s.null:.2f} = {s.z_vs_null:.2f}, "
        f"mean VP {s.mean_vp:.2f}, mean turns {s.mean_turns:.1f}"
    )
```

`alphasettler/cli.py`:

```python
"""Command line: `alphasettler arena ...` and `alphasettler compare ...`."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

from alphasettler.arena import format_summary, run
from alphasettler.stats import paired


def _positive_int(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def _stamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="alphasettler", description="AlphaSettler benchmark arena")
    sub = p.add_subparsers(dest="command", required=True)

    a = sub.add_parser("arena", help="one candidate vs three copies of a baseline, every seat rotated")
    a.add_argument("--candidate", required=True)
    a.add_argument("--baseline", required=True)
    a.add_argument("--seeds", type=_positive_int, required=True)
    a.add_argument("--seed-start", type=int, default=0)
    a.add_argument("--threads", type=_positive_int, default=None)
    a.add_argument("--out", default=None, help="JSONL path (default: runs/<time>-<candidate>-vs-<baseline>.jsonl)")
    a.add_argument("--no-trades", action="store_true", help="disable domestic trade offers")

    c = sub.add_parser("compare", help="two candidates against the same baseline on the same seeds")
    c.add_argument("--a", required=True)
    c.add_argument("--b", required=True)
    c.add_argument("--baseline", required=True)
    c.add_argument("--seeds", type=_positive_int, required=True)
    c.add_argument("--seed-start", type=int, default=0)
    c.add_argument("--threads", type=_positive_int, default=None)
    c.add_argument("--out-dir", default="runs")
    c.add_argument("--no-trades", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = {"max_offers_per_turn": 0} if args.no_trades else None
    try:
        if args.command == "arena":
            out = Path(args.out or f"runs/{_stamp()}-{args.candidate}-vs-{args.baseline}.jsonl")
            _, summary = run(args.candidate, args.baseline, args.seeds, args.seed_start, args.threads, config, out)
            print(format_summary(f"{args.candidate} vs {args.baseline}", summary))
            print(f"wrote {out}")
        else:
            stamp = _stamp()
            results = {}
            for name in (args.a, args.b):
                out = Path(args.out_dir) / f"{stamp}-{name}-vs-{args.baseline}.jsonl"
                records, summary = run(name, args.baseline, args.seeds, args.seed_start, args.threads, config, out)
                results[name] = records
                print(format_summary(f"{name} vs {args.baseline}", summary))
                print(f"wrote {out}")
            pr = paired(results[args.a], results[args.b])
            print(
                f"paired {args.a} - {args.b}: mean diff {pr.mean_diff:+.3f} ± {pr.se:.3f} "
                f"(z = {pr.z:.2f}) over {pr.seeds} seeds"
            )
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

`alphasettler/__main__.py`:

```python
import sys

from alphasettler.cli import main

sys.exit(main())
```

Note: `compare` with `--a` and `--b` equal would write the same file twice; it is still correct (paired diff 0). Argparse errors (e.g. `--seeds 0`) already exit with code 2.

- [ ] **Step 4: Run tests**

```bash
VIRTUAL_ENV=$PWD/.venv .venv/bin/maturin develop --release
.venv/bin/python3 -m pytest tests/python -q
.venv/bin/alphasettler arena --candidate greedy --baseline random --seeds 500 --out runs/plan2-check.jsonl
```

Expected: all Python tests pass; the CLI prints a greedy win rate well above 0.25 with z > 3.

- [ ] **Step 5: Commit**

```bash
git add alphasettler tests/python
git diff --cached | grep -nE 'sk-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{36}|password=[^[:space:]]+' && echo "SECRET FOUND - abort" || git commit -m "alphasettler: arena runner, JSONL output, arena/compare CLI"
git push
```

---

## Spec coverage (self-review)

| Spec requirement | Task |
|---|---|
| `bindings/` thin PyO3 layer → `alphasettler._engine` via maturin | 4 |
| `alphasettler/` package: arena, baseline bots access, stats, CLI | 4, 5, 6 |
| Format: candidate vs 3 baselines, 4 seat rotations per seed | 3 |
| CRN: same seeds → identical boards/dice; baseline streams independent of candidate | 3 (`bot_seed`), engine dice from (seed, turn) |
| Statistics: per-seed unit, win rate vs 25% null with CI, paired z, mean VP, game length | 5 |
| Native Rust arena | 3 |
| `RandomBot`, `GreedyBot` in Rust with stated trade behavior | 1, 2 |
| JSONL per game in `runs/` + printed summary; `alphasettler arena --candidate X --baseline Y --seeds N` | 6 |
| Done: GreedyBot beats RandomBot with z > 3 natively | 3 (Rust test), 6 (Python test + CLI check) |
| Catanatron arena (vs AlphaBeta / value bots via `oracle/`) | Plan 3 |
