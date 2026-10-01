# Plan 4: ISMCTS Search Bot (Sub-project 3a) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** IsmctsBot — a single-observer ISMCTS bot with a card-tracking belief and a pluggable evaluator — that clearly beats Catanatron's AlphaBeta bot, plus the self-play records and evaluator seam 3b builds on.

**Architecture:** A new Rust crate `search/` (depends on `engine/` only) holds the belief tracker, world sampler, tree, PUCT arithmetic, `Evaluator` trait and simulation loop. `bots/` gains the heuristic evaluator (it reuses GreedyBot's scoring), `IsmctsBot`, a self-play driver and a weight fit; the `Bot` trait gains `observe`/`diagnostics` so bots see the event log. `bindings/`, `alphasettler/` and `oracle/` expose self-play, records and an event feed for bots playing inside Catanatron.

**Tech Stack:** Rust 1.80+ (edition 2021, no new external crates), PyO3 0.29.3 + maturin 1.15.0, Python 3.12 stdlib (gzip, json), pytest 9.1.1, Catanatron 3.3.0 (oracle extra only).

**Spec:** `docs/superpowers/specs/2026-10-01-bot-3a-ismcts-design.md` (Sections 1–5). Earlier spec for the engine and harness: `docs/superpowers/specs/2026-09-30-engine-and-harness-design.md`.

## Global Constraints

- Rust MSRV 1.80, edition 2021. The new `settler-search` crate has **no dependencies** except `settler-engine` (path). No new crates anywhere.
- Commands: Rust is `~/.cargo/bin/cargo` (cargo is not on PATH in this shell); Python is `.venv/bin/python3` / `.venv/bin/pytest` / `.venv/bin/maturin`. Never `python`/`pip`.
- Rebuild the extension after any Rust change used from Python: `PATH=$HOME/.cargo/bin:$PATH .venv/bin/maturin develop --release -q`.
- Full checks: `~/.cargo/bin/cargo test --workspace --release` and `.venv/bin/pytest -q`.
- The engine's behaviour must not change: `engine/tests/golden_trace.rs` stays green **without re-pinning**.
- Determinism: every search, game and arena result depends only on seeds; never on thread count, wall time or hash order.
- The search budget is a simulation count, never wall time.
- 3a never trades: IsmctsBot never offers, declines every offer, and the search ignores `OfferTrade` actions.
- Only `oracle/` imports Catanatron.
- Bot names: `ismcts` (1,000 simulations), `ismcts@N` (1 ≤ N ≤ 1,000,000), `ismcts@N+rD` (1 ≤ D ≤ 200 greedy rollout moves). No `/` in names (they become file names).
- Commits: one per task, message given in the task, no Co-Authored-By or AI attribution. Before each commit run the secret scan: `git diff --cached | grep -nE 'sk-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{36}|password=[^[:space:]]+'` (expect no output).
- No README or docs beyond those this plan names.

## Review Focus

1. A domestic offer reaching IsmctsBot (any config with `max_offers_per_turn > 0`): it must answer `RejectTrade` / `CancelTrade` immediately, never search a trade phase (Task 5).
2. A bot fed a contradictory or foreign history (a reused Python `Bot`, a bad event feed): it must count a `belief_resets`, rebuild from the observation and still return a legal move — never panic (Tasks 5 and 6).
3. Malformed bot names (`ismcts@0`, `ismcts@`, `ismcts@abc`, `ismcts@+5`, `ismcts@1000001`, `ismcts@300+r0`, `ismcts@300+r201`): rejected as "unknown bot" by `make_bot`, `Bot(...)`, `alphasettler arena` and `alphasettler compare` (Tasks 5 and 6).
4. The turn cap reached inside the tree (a draw): it backs up ¼ each and the search still returns a legal move (Task 4).
5. Off-turn decisions — discarding on an opponent's 7, a sampled world whose current player is an opponent with hidden VP cards: worlds stay valid and never deal the current opponent a win (Task 3).

---

### Task 1: Bots see the event log

**Files:**
- Modify: `engine/src/game.rs` (add `Game::log`)
- Modify: `bots/src/lib.rs` (trait methods)
- Modify: `bots/src/arena.rs` (EventFeed, split_seeds, play_game, run_match)
- Test: `bots/tests/arena.rs` (append), `engine/tests/game.rs` (append)

**Interfaces:**
- Produces: `Game::log(&self) -> &[Event]`; `Bot::observe(&mut self, viewer: PlayerId, events: &[Event])` (default no-op); `Bot::diagnostics(&self) -> Vec<(&'static str, u64)>` (default empty); `settler_bots::arena::EventFeed` with `push(&mut self, events: &[Event])` and `deliver(&mut self, seat: usize, bot: &mut dyn Bot)`; `settler_bots::arena::split_seeds(seeds: Range<u64>, threads: usize) -> Vec<Range<u64>>`.

- [ ] **Step 1: Write the failing tests**

Append to `engine/tests/game.rs`:

```rust
#[test]
fn the_full_log_is_unredacted_and_log_for_redacts_it() {
    let mut g = Game::new(5, GameConfig::default());
    let mut rng = settler_engine::rng::Rng::new(5);
    while !g.state().is_over() && g.log().len() < 400 {
        let legal = g.legal_actions();
        g.apply(legal[rng.below(legal.len() as u32) as usize]).unwrap();
    }
    for viewer in 0..4u8 {
        let redacted: Vec<Event> = g.log().iter().map(|e| e.redacted_for(viewer)).collect();
        assert_eq!(redacted, g.log_for(viewer));
    }
    assert!(g.log().iter().all(|e| !matches!(e, Event::Stole { resource: None, .. })));
}
```

(If `engine/tests/game.rs` does not already `use settler_engine::*;`, add `use settler_engine::{Event, Game, GameConfig};` at the top.)

Append to `bots/tests/arena.rs`:

```rust
use settler_bots::greedy::GreedyBot;
use settler_engine::{Event, Game, PlayerId};
use std::sync::{Arc, Mutex};

/// GreedyBot that records the events it is shown and appends its moves to a shared list.
struct Recorder {
    inner: GreedyBot,
    viewer: Option<PlayerId>,
    seen: Arc<Mutex<Vec<Event>>>,
    moves: Arc<Mutex<Vec<Action>>>,
}

impl Bot for Recorder {
    fn name(&self) -> &'static str {
        "recorder"
    }

    fn observe(&mut self, viewer: PlayerId, events: &[Event]) {
        assert!(self.viewer.map_or(true, |v| v == viewer), "a seat's viewer never changes");
        self.viewer = Some(viewer);
        self.seen.lock().unwrap().extend_from_slice(events);
    }

    fn act(&mut self, obs: &Observation, legal: &[Action]) -> Action {
        let a = self.inner.act(obs, legal);
        self.moves.lock().unwrap().push(a);
        a
    }
}

#[test]
fn each_bot_is_shown_its_own_redacted_log_before_it_moves() {
    for seed in 0..5 {
        let moves = Arc::new(Mutex::new(Vec::new()));
        let seen: Vec<Arc<Mutex<Vec<Event>>>> = (0..4).map(|_| Arc::new(Mutex::new(Vec::new()))).collect();
        let mut bots: Vec<Box<dyn Bot>> = (0..4)
            .map(|p| {
                Box::new(Recorder {
                    inner: GreedyBot::new(0),
                    viewer: None,
                    seen: seen[p].clone(),
                    moves: moves.clone(),
                }) as Box<dyn Bot>
            })
            .collect();
        play_game(seed, GameConfig::default(), &mut bots);

        // Replay the same moves with the logging Game and note each seat's log length at its last move.
        let mut g = Game::new(seed, GameConfig::default());
        let mut at_last_move = [0usize; 4];
        for &a in moves.lock().unwrap().iter() {
            let actor = g.state().current_actor() as usize;
            at_last_move[actor] = g.log().len();
            g.apply(a).unwrap();
        }
        assert!(g.state().is_over());
        for p in 0..4 {
            let shown = seen[p].lock().unwrap();
            assert_eq!(shown.len(), at_last_move[p], "seat {p} saw every event up to its last move");
            assert_eq!(&shown[..], &g.log_for(p as u8)[..shown.len()], "seat {p} saw its redacted log");
        }
    }
}

#[test]
fn split_seeds_covers_the_range_in_order() {
    use settler_bots::arena::split_seeds;
    let parts = split_seeds(10..33, 4);
    assert_eq!(parts.first().unwrap().start, 10);
    assert_eq!(parts.last().unwrap().end, 33);
    assert!(parts.windows(2).all(|w| w[0].end == w[1].start));
    assert!(split_seeds(5..5, 3).is_empty());
    assert_eq!(split_seeds(0..3, 100).len(), 3);
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `~/.cargo/bin/cargo test --release -p settler-engine --test game the_full_log -- --nocapture; ~/.cargo/bin/cargo test --release -p settler-bots --test arena`
Expected: compile errors — `no method named log`, `observe is not a member of trait Bot`, `cannot find function split_seeds`.

- [ ] **Step 3: Implement**

In `engine/src/game.rs`, inside `impl Game`, after `log_for`:

```rust
    /// The full event log with nothing redacted, for tests and tools. Bots get `log_for`.
    pub fn log(&self) -> &[Event] {
        &self.log
    }
```

Replace `bots/src/lib.rs` from `use settler_engine::{Action, Observation};` through the end of the `Bot` trait with:

```rust
use settler_engine::{Action, Event, Observation, PlayerId};

pub trait Bot: Send {
    fn name(&self) -> &'static str;

    /// Events since this bot last moved, as `viewer` (its seat) may see them, in log order. The
    /// arena calls this before every `act`; bots that keep no history ignore it.
    fn observe(&mut self, _viewer: PlayerId, _events: &[Event]) {}

    /// Choose one of `legal` (never empty) for `obs.viewer`, the player who must act now.
    fn act(&mut self, obs: &Observation, legal: &[Action]) -> Action;

    /// Named counters for reports and tests, e.g. how often a belief had to be rebuilt.
    fn diagnostics(&self) -> Vec<(&'static str, u64)> {
        Vec::new()
    }
}
```

In `bots/src/arena.rs`, change the imports to:

```rust
use crate::{make_bot, Bot, BOT_NAMES};
use settler_engine::legal::legal_actions;
use settler_engine::rng::mix;
use settler_engine::{Event, GameConfig, PlayerId, State, NUM_PLAYERS};
use std::ops::Range;
```

Add after `bot_seed`:

```rust
/// Events each seat has not been shown yet, already redacted for that seat.
#[derive(Default)]
pub struct EventFeed {
    pending: [Vec<Event>; NUM_PLAYERS],
}

impl EventFeed {
    pub fn push(&mut self, events: &[Event]) {
        for (p, q) in self.pending.iter_mut().enumerate() {
            q.extend(events.iter().map(|e| e.redacted_for(p as PlayerId)));
        }
    }

    /// Show the bot in `seat` everything since it last moved.
    pub fn deliver(&mut self, seat: usize, bot: &mut dyn Bot) {
        bot.observe(seat as PlayerId, &self.pending[seat]);
        self.pending[seat].clear();
    }
}

/// Contiguous parts of `seeds` for up to `threads` workers (capped at the seed count and at 4x
/// the available cores), split by offset in u64 so no bound passes `seeds.end` (a range can end
/// at u64::MAX). Empty for an empty range.
pub fn split_seeds(seeds: Range<u64>, threads: usize) -> Vec<Range<u64>> {
    let n = seeds.end.saturating_sub(seeds.start);
    if n == 0 {
        return Vec::new();
    }
    let cores = std::thread::available_parallelism().map_or(1, |c| c.get());
    let threads = threads.clamp(1, (n.min(usize::MAX as u64) as usize).min(cores * 4));
    let chunk = n.div_ceil(threads as u64);
    (0..threads as u64)
        .map(|i| i * chunk)
        .take_while(|&off| off < n)
        .map(|off| seeds.start + off..seeds.start + off + chunk.min(n - off))
        .collect()
}
```

Replace the body of `play_game` with:

```rust
    assert_eq!(bots.len(), NUM_PLAYERS, "a game needs exactly 4 bots");
    let mut s = State::new(seed, config);
    let mut buf = Vec::with_capacity(512);
    let mut feed = EventFeed::default();
    let mut step = Vec::with_capacity(16);
    let mut actions = 0u32;
    while !s.is_over() {
        legal_actions(&s, &mut buf);
        let actor = s.current_actor() as usize;
        feed.deliver(actor, bots[actor].as_mut());
        let a = bots[actor].act(&s.observation(actor as u8), &buf);
        assert!(
            buf.contains(&a),
            "bot {} chose illegal action {a:?} in phase {:?}",
            bots[actor].name(),
            s.phase
        );
        step.clear();
        s.apply_with(a, None, &mut step);
        feed.push(&step);
        actions += 1;
    }
    (
        s.winner(),
        std::array::from_fn(|p| s.total_vp(p)),
        s.turn,
        actions,
    )
```

In `run_match`, replace everything from `let cores = ...` through the `let parts: Vec<Range<u64>> = ... .collect();` statement with:

```rust
    let parts = split_seeds(seeds, threads);
```

(keep the `n == 0` early return above it, and leave the rest of `run_match` unchanged).

- [ ] **Step 4: Run the tests to verify they pass**

Run: `~/.cargo/bin/cargo test --workspace --release`
Expected: all pass, including `golden_trace` (unchanged) and the existing arena tests.

- [ ] **Step 5: Commit**

```bash
git add engine/src/game.rs engine/tests/game.rs bots/src/lib.rs bots/src/arena.rs bots/tests/arena.rs
git commit -m "bots: Bot::observe and diagnostics; arena feeds each seat its redacted event log"
```

---

### Task 2: The `search` crate and the belief tracker

**Files:**
- Modify: `Cargo.toml` (workspace members)
- Create: `search/Cargo.toml`, `search/src/lib.rs`, `search/src/belief.rs`
- Test: `search/tests/common/mod.rs`, `search/tests/belief.rs`

**Interfaces:**
- Consumes: `Game::log()` (Task 1).
- Produces (`settler_search::belief`, re-exported at the crate root):
  - `HandTracker { pub hands: [Hand; 4], .. }` with `new()`, `resume(hands, obs: &Observation)`, `step(&mut self, e: &Event, rng: &mut Rng) -> bool`.
  - `Belief` with `new(viewer, particles, seed)`, `from_observation(obs, particles, seed)`, `observe(&mut self, events: &[Event]) -> Result<(), String>`, `check(&self, obs) -> Result<(), String>`, `viewer()`, `particles() -> &[HandTracker]`, `rebuilds() -> u64`.
  - `DEFAULT_PARTICLES: usize = 1024`.

- [ ] **Step 1: Create the crate skeleton**

In the root `Cargo.toml` set:

```toml
members = ["engine", "bots", "bindings", "search"]
default-members = ["engine", "bots", "search"]
```

`search/Cargo.toml`:

```toml
[package]
name = "settler-search"
version = "0.1.0"
edition = "2021"
rust-version = "1.80"
publish = false

[lib]
name = "settler_search"

[dependencies]
settler-engine = { path = "../engine" }

[lints.clippy]
needless_range_loop = "allow"
```

`search/src/lib.rs`:

```rust
//! Information-set Monte Carlo tree search for AlphaSettler
//! (spec: docs/superpowers/specs/2026-10-01-bot-3a-ismcts-design.md).

pub mod belief;

pub use belief::{Belief, HandTracker, DEFAULT_PARTICLES};
```

`search/tests/common/mod.rs`:

```rust
#![allow(dead_code)]

use settler_engine::legal::legal_actions;
use settler_engine::rng::Rng;
use settler_engine::*;

pub fn no_trades() -> GameConfig {
    GameConfig { max_offers_per_turn: 0, ..GameConfig::default() }
}

/// Plays uniformly random legal moves from game `seed`, calling `f(&game)` after every move.
pub fn random_game(seed: u64, config: GameConfig, mut f: impl FnMut(&Game)) {
    let mut g = Game::new(seed, config);
    let mut rng = Rng::new(seed ^ 0x5EED);
    while !g.state().is_over() {
        let legal = g.legal_actions();
        g.apply(legal[rng.below(legal.len() as u32) as usize]).unwrap();
        f(&g);
    }
}

/// The position right after setup in game `seed` (player 0 to roll), with random placements.
pub fn after_setup(seed: u64, config: GameConfig) -> State {
    let mut s = State::new(seed, config);
    let mut rng = Rng::new(seed);
    let mut buf = Vec::new();
    while matches!(s.phase, Phase::SetupSettlement | Phase::SetupRoad { .. }) {
        legal_actions(&s, &mut buf);
        s.apply(buf[rng.below(buf.len() as u32) as usize]);
    }
    s
}

/// `s` with `f` applied to its snapshot, rebuilt (and validated) under `config`.
pub fn edit(s: &State, config: GameConfig, f: impl FnOnce(&mut Snapshot)) -> State {
    let mut snap = s.snapshot();
    f(&mut snap);
    State::from_snapshot(s.seed, config, &snap).expect("the edited position is valid")
}

/// Move `n` cards of `card` from the deck into player `p`'s hand (bought on an earlier turn).
pub fn give_dev(snap: &mut Snapshot, p: usize, card: DevCard, n: u8) {
    for _ in 0..n {
        let i = snap.dev_deck.iter().position(|&c| c == card).expect("the card is left in the deck");
        snap.dev_deck.remove(i);
        snap.players[p].dev_hand[card.index()] += 1;
    }
}

/// Move `cards` from the bank to player `p`.
pub fn give_cards(snap: &mut Snapshot, p: usize, cards: Hand) {
    for r in 0..NUM_RESOURCES {
        snap.bank[r] -= cards[r];
        snap.players[p].hand[r] += cards[r];
    }
}

/// The 16 free setup placements (snake order) as events, so later builds are paid.
pub fn setup_events() -> Vec<Event> {
    let mut out = Vec::new();
    for (i, &p) in settler_engine::state::SETUP_ORDER.iter().enumerate() {
        out.push(Event::BuiltSettlement { player: p, node: i as u8 });
        out.push(Event::BuiltRoad { player: p, edge: i as u8 });
    }
    out
}
```

- [ ] **Step 2: Write the failing tests**

`search/tests/belief.rs`:

```rust
mod common;

use common::*;
use settler_engine::rng::Rng;
use settler_engine::*;
use settler_search::{Belief, HandTracker};
use std::collections::HashMap;

#[test]
fn the_full_log_reproduces_every_hand() {
    let configs = (0..40).map(|s| (s, no_trades())).chain((40..48).map(|s| (s, GameConfig::default())));
    for (seed, config) in configs {
        let mut t = HandTracker::new();
        let mut rng = Rng::new(0);
        let mut seen = 0;
        random_game(seed, config, |g| {
            for e in &g.log()[seen..] {
                assert!(t.step(e, &mut rng), "seed {seed}: {e:?} rejected on the true history");
            }
            seen = g.log().len();
            for p in 0..NUM_PLAYERS {
                assert_eq!(t.hands[p], g.state().players[p].hand, "seed {seed} player {p}");
            }
        });
    }
}

#[test]
fn every_particle_matches_the_visible_counts() {
    for seed in 0..12 {
        let mut beliefs: Vec<Belief> = (0..4).map(|p| Belief::new(p, 64, seed)).collect();
        let mut seen = [0usize; 4];
        random_game(seed, no_trades(), |g| {
            for p in 0..4u8 {
                let log = g.log_for(p);
                beliefs[p as usize].observe(&log[seen[p as usize]..]).unwrap();
                seen[p as usize] = log.len();
                beliefs[p as usize].check(&g.observation(p)).unwrap();
            }
        });
    }
}

/// Exact posterior over all four hands: branch on each hidden steal, weighted by the victim's hand.
fn exact(log: &[Event]) -> HashMap<[Hand; 4], f64> {
    let mut states = vec![(HandTracker::new(), 1.0f64)];
    let mut rng = Rng::new(0);
    for e in log {
        let mut next = Vec::new();
        for (t, w) in states {
            if let Event::Stole { thief, victim, resource: None } = *e {
                let v = t.hands[victim as usize];
                let total = hand_total(&v) as f64;
                for r in Resource::ALL {
                    if v[r.index()] == 0 {
                        continue;
                    }
                    let mut u = t;
                    if u.step(&Event::Stole { thief, victim, resource: Some(r) }, &mut rng) {
                        next.push((u, w * v[r.index()] as f64 / total));
                    }
                }
            } else {
                let mut u = t;
                if u.step(e, &mut rng) {
                    next.push((u, w));
                }
            }
        }
        states = next;
    }
    let z: f64 = states.iter().map(|s| s.1).sum();
    let mut out = HashMap::new();
    for (t, w) in states {
        *out.entry(t.hands).or_insert(0.0) += w / z;
    }
    out
}

fn assert_matches_exact(log: &[Event], seed: u64) {
    let truth = exact(log);
    let mut b = Belief::new(0, 20_000, seed);
    b.observe(log).unwrap();
    let mut freq: HashMap<[Hand; 4], f64> = HashMap::new();
    for t in b.particles() {
        *freq.entry(t.hands).or_insert(0.0) += 1.0 / b.particles().len() as f64;
    }
    for (hands, p) in &truth {
        let f = freq.get(hands).copied().unwrap_or(0.0);
        assert!((f - p).abs() <= 0.02, "seed {seed}: {hands:?} exact {p:.3} tracked {f:.3}");
    }
    assert!(freq.keys().all(|h| truth.contains_key(h)), "seed {seed}: a particle the exact posterior rules out");
}

#[test]
fn hand_built_histories_match_the_exact_posterior() {
    let mut log = setup_events();
    log.extend([
        Event::Produced { player: 1, resources: [1, 0, 0, 0, 2] },
        Event::Produced { player: 3, resources: [0, 2, 0, 1, 0] },
        Event::Stole { thief: 2, victim: 1, resource: None },
        Event::Stole { thief: 2, victim: 3, resource: None },
        Event::Discarded { player: 2, resource: Resource::Ore },
    ]);
    assert_matches_exact(&log, 1);
}

/// A random small history that is feasible for some true hands: production, hidden steals between
/// players 1-3, discards and paid roads of cards the (simulated) true holder has.
fn random_history(seed: u64) -> Vec<Event> {
    let mut rng = Rng::new(seed);
    let mut log = setup_events();
    let mut truth = HandTracker::new();
    for e in &log {
        assert!(truth.step(e, &mut rng));
    }
    let mut hidden = 0;
    while log.len() < 30 {
        let p = 1 + rng.below(3) as u8;
        let e = match rng.below(4) {
            0 => {
                let mut h = [0u8; 5];
                h[rng.below(5) as usize] += 1 + rng.below(2) as u8;
                Event::Produced { player: p, resources: h }
            }
            1 if hidden < 4 => {
                let q = 1 + (p as u32 + rng.below(2)) % 3;
                if hand_total(&truth.hands[q as usize]) == 0 {
                    continue;
                }
                hidden += 1;
                let r = settler_engine::rules::roll::pick_card(&truth.hands[q as usize], &mut rng);
                let mut t = truth;
                assert!(t.step(&Event::Stole { thief: p, victim: q as u8, resource: Some(r) }, &mut rng));
                truth = t;
                log.push(Event::Stole { thief: p, victim: q as u8, resource: None });
                continue;
            }
            2 => {
                let h = truth.hands[p as usize];
                let Some(r) = Resource::ALL.into_iter().find(|r| h[r.index()] > 0) else { continue };
                Event::Discarded { player: p, resource: r }
            }
            _ => {
                if !covers(&truth.hands[p as usize], &ROAD_COST) {
                    continue;
                }
                Event::BuiltRoad { player: p, edge: 40 }
            }
        };
        assert!(truth.step(&e, &mut rng));
        log.push(e);
    }
    log
}

#[test]
fn random_histories_match_the_exact_posterior() {
    for seed in 0..25 {
        assert_matches_exact(&random_history(seed), seed);
    }
}

#[test]
fn a_rare_survivor_set_is_rebuilt_from_the_log() {
    let mut log = setup_events();
    log.extend([
        Event::Produced { player: 1, resources: [1, 0, 0, 0, 19] },
        Event::Produced { player: 2, resources: [0, 1, 0, 0, 0] },
        Event::Stole { thief: 2, victim: 1, resource: None },
        // Only the 1-in-20 particles where player 2 stole the wood can pay for this road.
        Event::BuiltRoad { player: 2, edge: 30 },
    ]);
    let mut b = Belief::new(0, 1024, 3);
    b.observe(&log).unwrap();
    assert_eq!(b.rebuilds(), 1);
    assert_eq!(b.particles().len(), 1024);
    for t in b.particles() {
        assert_eq!(t.hands[1], [0, 0, 0, 0, 19]);
        assert_eq!(t.hands[2], [0; 5]);
    }
}

#[test]
fn an_impossible_history_is_an_error() {
    let mut log = setup_events();
    log.extend([
        Event::Produced { player: 1, resources: [1, 0, 0, 0, 0] },
        Event::Discarded { player: 1, resource: Resource::Ore },
    ]);
    assert!(Belief::new(0, 64, 0).observe(&log).is_err());
}

#[test]
fn a_belief_from_an_observation_deals_the_unseen_cards() {
    let mut n = 0;
    random_game(9, no_trades(), |g| {
        n += 1;
        if n % 50 != 0 {
            return;
        }
        let obs = g.observation(2);
        let b = Belief::from_observation(&obs, 32, n);
        b.check(&obs).unwrap();
        for t in b.particles() {
            for r in 0..NUM_RESOURCES {
                let held: u8 = (0..NUM_PLAYERS).map(|p| t.hands[p][r]).sum();
                assert_eq!(held + obs.bank[r], BANK_PER_RESOURCE);
            }
        }
    });
}

#[test]
fn road_building_roads_are_free() {
    let mut log = setup_events();
    log.extend([
        Event::PlayedDev { player: 1, card: DevCard::RoadBuilding },
        Event::BuiltRoad { player: 1, edge: 50 },
        Event::BuiltRoad { player: 1, edge: 51 },
    ]);
    let mut t = HandTracker::new();
    let mut rng = Rng::new(0);
    assert!(log.iter().all(|e| t.step(e, &mut rng)), "two free roads after Road Building");
    assert!(!t.step(&Event::BuiltRoad { player: 1, edge: 52 }, &mut rng), "a third road is paid");
}
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `~/.cargo/bin/cargo test --release -p settler-search --test belief`
Expected: compile error — `unresolved module belief` / `cannot find type Belief`.

- [ ] **Step 4: Implement `search/src/belief.rs`**

```rust
//! What one player can infer about everyone's resource cards from the events they have seen.
//!
//! Every resource movement is public except a steal the viewer neither made nor suffered. Given
//! the resource taken in each such steal, every hand follows exactly, so the belief is a set of
//! particles, each holding all four hands, advanced event by event. A hidden steal draws the
//! resource from the victim's hand in that particle (the engine's own distribution), and a
//! particle in which a player could not have paid for what they did is dropped. Behavioural
//! evidence ("they would have built if they could") is ignored in 3a (spec Section 2).

use settler_engine::rng::Rng;
use settler_engine::rules::roll::{pick_card, sample_cards};
use settler_engine::types::*;
use settler_engine::{Event, Observation, Phase};

pub const DEFAULT_PARTICLES: usize = 1024;
/// Replay attempts per wanted particle when the particle set must be rebuilt from the log.
const REPLAYS_PER_PARTICLE: usize = 32;
/// Settlements and roads placed during setup (two of each per player), which cost nothing.
const SETUP_PIECES: u8 = 8;

/// All four hands, advanced by public events; also the exact tracker for a fully known log.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct HandTracker {
    pub hands: [Hand; NUM_PLAYERS],
    setup_settlements: u8,
    setup_roads: u8,
    /// Free Road Building roads still available to `free_road_player`.
    free_roads: u8,
    free_road_player: PlayerId,
}

impl HandTracker {
    /// Hands at the start of a game: all empty, setup not begun.
    pub fn new() -> HandTracker {
        HandTracker::default()
    }

    /// Known `hands` at the position `obs` shows, with the free-build bookkeeping it implies.
    pub fn resume(hands: [Hand; NUM_PLAYERS], obs: &Observation) -> HandTracker {
        let placed = |count: &dyn Fn(usize) -> u32| {
            (0..NUM_PLAYERS).map(count).sum::<u32>().min(SETUP_PIECES as u32) as u8
        };
        let (free_roads, free_road_player) = match obs.phase {
            Phase::RoadBuilding { roads_left } => (roads_left, obs.current),
            _ => (0, 0),
        };
        HandTracker {
            hands,
            setup_settlements: placed(&|p| (obs.settlements[p] | obs.cities[p]).count_ones()),
            setup_roads: placed(&|p| obs.roads[p].count_ones()),
            free_roads,
            free_road_player,
        }
    }

    /// Advance by one event. `rng` draws the resource of a steal whose resource is hidden.
    /// Returns false (leaving the tracker unspecified) when these hands make `e` impossible.
    pub fn step(&mut self, e: &Event, rng: &mut Rng) -> bool {
        if let Some((p, cost)) = self.cost(e) {
            if !covers(&self.hands[p], &cost) {
                return false;
            }
            hand_sub(&mut self.hands[p], &cost);
        }
        let h = &mut self.hands;
        match *e {
            Event::Produced { player, resources } | Event::YearOfPlentyTaken { player, resources } => {
                hand_add(&mut h[player as usize], &resources);
            }
            Event::Discarded { player, resource } => {
                let c = &mut h[player as usize][resource.index()];
                if *c == 0 {
                    return false;
                }
                *c -= 1;
            }
            Event::Stole { thief, victim, resource } => {
                let v = victim as usize;
                let r = match resource {
                    Some(r) if h[v][r.index()] > 0 => r,
                    Some(_) => return false,
                    None if hand_total(&h[v]) > 0 => pick_card(&h[v], rng),
                    None => return false,
                };
                h[v][r.index()] -= 1;
                h[thief as usize][r.index()] += 1;
            }
            Event::MonopolyTaken { player, resource, amount } => {
                let (p, r) = (player as usize, resource.index());
                let taken: u32 = (0..NUM_PLAYERS).filter(|&q| q != p).map(|q| h[q][r] as u32).sum();
                if taken != amount as u32 {
                    return false;
                }
                for q in 0..NUM_PLAYERS {
                    if q != p {
                        h[q][r] = 0;
                    }
                }
                h[p][r] += amount;
            }
            Event::MaritimeTraded { player, gave, got } => {
                let mine = &mut h[player as usize];
                if !covers(mine, &gave) {
                    return false;
                }
                hand_sub(mine, &gave);
                hand_add(mine, &got);
            }
            Event::TradeConfirmed { offerer, partner, offerer_gave, partner_gave } => {
                let (o, q) = (offerer as usize, partner as usize);
                if !covers(&h[o], &offerer_gave) || !covers(&h[q], &partner_gave) {
                    return false;
                }
                hand_sub(&mut h[o], &offerer_gave);
                hand_add(&mut h[q], &offerer_gave);
                hand_sub(&mut h[q], &partner_gave);
                hand_add(&mut h[o], &partner_gave);
            }
            _ => {}
        }
        true
    }

    /// Who paid what for `e`, updating which builds are free. A Road Building window closes at
    /// the first event that is not one of its roads.
    fn cost(&mut self, e: &Event) -> Option<(usize, Hand)> {
        let window = std::mem::take(&mut self.free_roads);
        match *e {
            Event::BuiltSettlement { player, .. } => {
                if self.setup_settlements < SETUP_PIECES {
                    self.setup_settlements += 1;
                    None
                } else {
                    Some((player as usize, SETTLEMENT_COST))
                }
            }
            Event::BuiltRoad { player, .. } => {
                if self.setup_roads < SETUP_PIECES {
                    self.setup_roads += 1;
                    None
                } else if window > 0 && player == self.free_road_player {
                    self.free_roads = window - 1;
                    None
                } else {
                    Some((player as usize, ROAD_COST))
                }
            }
            Event::BuiltCity { player, .. } => Some((player as usize, CITY_COST)),
            Event::BoughtDev { player, .. } => Some((player as usize, DEV_COST)),
            Event::PlayedDev { player, card: DevCard::RoadBuilding } => {
                self.free_roads = 2;
                self.free_road_player = player;
                None
            }
            _ => None,
        }
    }
}

/// Opponents' hands dealt uniformly from the cards neither the bank nor the viewer holds, each
/// opponent getting their visible count.
fn deal_hidden_hands(obs: &Observation, rng: &mut Rng) -> [Hand; NUM_PLAYERS] {
    let me = obs.viewer as usize;
    let mut pool: Hand = std::array::from_fn(|r| {
        BANK_PER_RESOURCE.saturating_sub(obs.bank[r]).saturating_sub(obs.my_hand[r])
    });
    let mut hands = [[0u8; NUM_RESOURCES]; NUM_PLAYERS];
    hands[me] = obs.my_hand;
    for q in (0..NUM_PLAYERS).filter(|&q| q != me) {
        let k = (obs.hand_counts[q] as u32).min(hand_total(&pool)) as u8;
        hands[q] = sample_cards(&pool, k, rng);
        hand_sub(&mut pool, &hands[q]);
    }
    hands
}

/// One player's belief about every hand.
#[derive(Clone, Debug)]
pub struct Belief {
    viewer: PlayerId,
    target: usize,
    particles: Vec<HandTracker>,
    /// Where replays start when the particles are rebuilt; `log` holds every event since.
    origin: Vec<HandTracker>,
    log: Vec<Event>,
    rng: Rng,
    rebuilds: u64,
}

impl Belief {
    /// The belief at the start of a game: every hand empty.
    pub fn new(viewer: PlayerId, particles: usize, seed: u64) -> Belief {
        assert!(particles > 0, "a belief needs at least one particle");
        Belief {
            viewer,
            target: particles,
            particles: vec![HandTracker::new(); particles],
            origin: vec![HandTracker::new()],
            log: Vec::new(),
            rng: Rng::new(seed),
            rebuilds: 0,
        }
    }

    /// A belief for a viewer who has not seen the history: the cards nobody can see are dealt
    /// uniformly (`deal_hidden_hands`). Later events refine it as usual.
    pub fn from_observation(obs: &Observation, particles: usize, seed: u64) -> Belief {
        assert!(particles > 0, "a belief needs at least one particle");
        let mut rng = Rng::new(seed);
        let origin: Vec<HandTracker> = (0..particles)
            .map(|_| HandTracker::resume(deal_hidden_hands(obs, &mut rng), obs))
            .collect();
        Belief {
            viewer: obs.viewer,
            target: particles,
            particles: origin.clone(),
            origin,
            log: Vec::new(),
            rng,
            rebuilds: 0,
        }
    }

    pub fn viewer(&self) -> PlayerId {
        self.viewer
    }

    pub fn particles(&self) -> &[HandTracker] {
        &self.particles
    }

    /// How many times the particle set had to be rebuilt from the log.
    pub fn rebuilds(&self) -> u64 {
        self.rebuilds
    }

    /// Advance by `events` as the viewer saw them, in log order. Err only when no hands at all
    /// are consistent with the history (an inconsistent feed, or a tracker bug).
    pub fn observe(&mut self, events: &[Event]) -> Result<(), String> {
        for e in events {
            self.log.push(*e);
            let rng = &mut self.rng;
            self.particles.retain_mut(|t| t.step(e, rng));
            if self.particles.len() * 8 < self.target {
                self.rebuild()?;
            }
        }
        Ok(())
    }

    /// Refill to `target` particles: replay the log from random origins (up to 32 attempts per
    /// wanted particle), keep replays that survive every event, and fill any shortfall by
    /// resampling all survivors (old and new) with replacement.
    fn rebuild(&mut self) -> Result<(), String> {
        self.rebuilds += 1;
        let mut fresh = Vec::with_capacity(self.target);
        for _ in 0..self.target * REPLAYS_PER_PARTICLE {
            if fresh.len() == self.target {
                break;
            }
            let mut t = self.origin[self.rng.below(self.origin.len() as u32) as usize];
            let rng = &mut self.rng;
            if self.log.iter().all(|e| t.step(e, rng)) {
                fresh.push(t);
            }
        }
        fresh.extend_from_slice(&self.particles);
        if fresh.is_empty() {
            return Err(format!(
                "player {}'s belief: no hands are consistent with the {} events seen",
                self.viewer,
                self.log.len()
            ));
        }
        while fresh.len() < self.target {
            let t = fresh[self.rng.below(fresh.len() as u32) as usize];
            fresh.push(t);
        }
        fresh.truncate(self.target);
        self.particles = fresh;
        Ok(())
    }

    /// Err if some particle disagrees with `obs`: the viewer's own hand or anyone's card count.
    pub fn check(&self, obs: &Observation) -> Result<(), String> {
        if obs.viewer != self.viewer {
            return Err(format!("belief of player {} checked against player {}'s view", self.viewer, obs.viewer));
        }
        for t in &self.particles {
            if t.hands[self.viewer as usize] != obs.my_hand {
                return Err(format!(
                    "belief holds {:?} for player {}, who holds {:?}",
                    t.hands[self.viewer as usize], self.viewer, obs.my_hand
                ));
            }
            for q in 0..NUM_PLAYERS {
                if hand_total(&t.hands[q]) != obs.hand_counts[q] as u32 {
                    return Err(format!(
                        "belief gives player {q} {} cards, they hold {}",
                        hand_total(&t.hands[q]),
                        obs.hand_counts[q]
                    ));
                }
            }
        }
        Ok(())
    }
}
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `~/.cargo/bin/cargo test --release -p settler-search && ~/.cargo/bin/cargo clippy --release -p settler-search --all-targets -- -D warnings`
Expected: all `belief` tests pass; clippy clean.

- [ ] **Step 6: Commit**

```bash
git add Cargo.toml Cargo.lock search/
git commit -m "search: crate skeleton and particle-filter belief over hidden steals"
```

---

### Task 3: Sampled worlds

**Files:**
- Modify: `engine/src/state.rs` (add `State::reseed`)
- Create: `search/src/world.rs`
- Modify: `search/src/lib.rs`
- Test: `engine/tests/snapshot.rs` (append), `search/tests/world.rs`

**Interfaces:**
- Consumes: `Belief::{check, particles, viewer}` (Task 2).
- Produces: `State::reseed(&mut self, seed: u64)`; `settler_search::WorldSampler` with `new(obs: &Observation, belief: &Belief) -> Result<WorldSampler, String>` and `sample(&self, belief: &Belief, rng: &mut Rng) -> State`.

- [ ] **Step 1: Write the failing tests**

Append to `engine/tests/snapshot.rs`:

```rust
#[test]
fn reseed_changes_only_the_random_streams() {
    let fresh = State::new(7, GameConfig::default());
    let mut same = fresh;
    same.reseed(7);
    assert_eq!(same, fresh);
    let mut other = fresh;
    other.reseed(8);
    assert_eq!(other.seed, 8);
    assert_ne!(other.rng_steal, fresh.rng_steal);
    assert_eq!(other.snapshot(), fresh.snapshot());
}
```

`search/tests/world.rs`:

```rust
mod common;

use common::*;
use settler_engine::rng::Rng;
use settler_engine::*;
use settler_search::{Belief, WorldSampler};

fn trade_phase(p: Phase) -> bool {
    matches!(p, Phase::TradeResponse | Phase::TradeConfirm)
}

#[test]
fn sampled_worlds_are_valid_and_look_exactly_like_the_observation() {
    let games = (0..24).map(|s| (s, no_trades())).chain((24..30).map(|s| (s, GameConfig::default())));
    let mut checked = 0;
    for (seed, config) in games {
        let mut beliefs: Vec<Belief> = (0..4).map(|p| Belief::new(p, 32, seed)).collect();
        let mut seen = [0usize; 4];
        let mut rng = Rng::new(seed);
        let mut n = 0;
        random_game(seed, config, |g| {
            n += 1;
            for p in 0..4u8 {
                let log = g.log_for(p);
                beliefs[p as usize].observe(&log[seen[p as usize]..]).unwrap();
                seen[p as usize] = log.len();
            }
            if n % 5 != 0 || g.state().is_over() || trade_phase(g.state().phase) {
                return;
            }
            for p in 0..4u8 {
                let obs = g.observation(p);
                let sampler = WorldSampler::new(&obs, &beliefs[p as usize]).unwrap();
                for _ in 0..3 {
                    let w = sampler.sample(&beliefs[p as usize], &mut rng);
                    w.check_invariants().unwrap();
                    assert_eq!(w.observation(p), obs, "seed {seed} step {n} viewer {p}");
                    assert_eq!(State::from_snapshot(w.seed, w.config, &w.snapshot()).unwrap(), w);
                    checked += 1;
                }
            }
        });
    }
    assert!(checked > 10_000, "checked {checked} worlds");
}

#[test]
fn worlds_use_fresh_randomness() {
    let s = after_setup(3, no_trades());
    let obs = s.observation(0);
    let b = Belief::from_observation(&obs, 16, 1);
    let sampler = WorldSampler::new(&obs, &b).unwrap();
    let mut rng = Rng::new(0);
    let seeds: std::collections::HashSet<u64> = (0..20).map(|_| sampler.sample(&b, &mut rng).seed).collect();
    assert_eq!(seeds.len(), 20);
    assert!(!seeds.contains(&s.seed));
}

#[test]
fn hidden_dev_cards_are_dealt_uniformly() {
    // Player 1 holds one dev card the viewer cannot see; with the rest of the deck it is one of
    // the 25 cards minus the viewer's own, so a knight shows up 14/25 of the time.
    let base = after_setup(4, no_trades());
    let s = edit(&base, no_trades(), |snap| give_dev(snap, 1, DevCard::Monopoly, 1));
    let obs = s.observation(0);
    let b = Belief::from_observation(&obs, 16, 2);
    let sampler = WorldSampler::new(&obs, &b).unwrap();
    let mut rng = Rng::new(5);
    let n = 5000;
    let knights = (0..n).filter(|_| sampler.sample(&b, &mut rng).players[1].dev_hand[DevCard::Knight.index()] == 1).count();
    let rate = knights as f64 / n as f64;
    assert!((rate - 14.0 / 25.0).abs() < 0.03, "knight rate {rate}");
}

#[test]
fn the_opponent_on_turn_is_never_dealt_a_win() {
    // With 3 VP to win, player 1 (on turn, 2 public VP) would already have won holding a VP card.
    let cfg = GameConfig { vp_to_win: 3, ..no_trades() };
    let base = after_setup(6, cfg);
    let s = edit(&base, cfg, |snap| {
        give_dev(snap, 1, DevCard::Knight, 1);
        give_dev(snap, 1, DevCard::Monopoly, 1);
        snap.current = 1;
        snap.phase = Phase::PreRoll;
    });
    let obs = s.observation(0);
    let b = Belief::from_observation(&obs, 16, 3);
    let sampler = WorldSampler::new(&obs, &b).unwrap();
    let mut rng = Rng::new(7);
    let mut vp_in_deck = 0;
    for _ in 0..2000 {
        let w = sampler.sample(&b, &mut rng);
        assert_eq!(w.players[1].dev_hand[DevCard::VictoryPoint.index()], 0);
        w.check_invariants().unwrap();
        vp_in_deck += w.dev_deck[w.dev_deck_pos as usize..].contains(&DevCard::VictoryPoint) as u32;
    }
    assert_eq!(vp_in_deck, 2000, "the VP cards are all in the deck");
}

#[test]
fn a_pending_trade_cannot_be_sampled() {
    let mut found = false;
    random_game(1, GameConfig::default(), |g| {
        if found || g.state().phase != Phase::TradeResponse {
            return;
        }
        found = true;
        let actor = g.state().current_actor();
        let obs = g.observation(actor);
        let b = Belief::from_observation(&obs, 8, 0);
        assert!(WorldSampler::new(&obs, &b).is_err());
    });
    assert!(found, "a random game with trades reaches TradeResponse");
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `~/.cargo/bin/cargo test --release -p settler-engine --test snapshot reseed; ~/.cargo/bin/cargo test --release -p settler-search --test world`
Expected: compile errors — `no method named reseed`, `no WorldSampler in the root`.

- [ ] **Step 3: Implement**

In `engine/src/state.rs`, inside `impl State` after `with_board`:

```rust
    /// Replace the game seed and the steal and misc streams with the ones `State::new(seed, ..)`
    /// would derive. Future dice follow the new seed; the board, hands, deck order and everything
    /// else stay as they are. Search uses this so a sampled world's future is not the real game's.
    pub fn reseed(&mut self, seed: u64) {
        self.seed = seed;
        self.rng_steal = Rng::new(mix(seed, STEAL_SALT));
        self.rng_misc = Rng::new(mix(seed, MISC_SALT));
    }
```

`search/src/world.rs`:

```rust
//! Sampled worlds: complete states consistent with everything one player has seen.
//!
//! Built once per decision: a template `State` (validated by `State::from_snapshot`) carries
//! everything public plus the viewer's own cards. Each sample copies it and overwrites only what
//! the viewer cannot see: opponents' hands (a belief particle), their dev cards and the deck
//! order (a uniform deal of the unseen cards), and the random seed.

use crate::belief::Belief;
use settler_engine::rng::Rng;
use settler_engine::types::*;
use settler_engine::{Observation, Phase, PlayerSnapshot, Snapshot, State};

/// Deals rejected before giving up because every deal hands the player on turn a win.
const MAX_DEALS: u32 = 10_000;

pub struct WorldSampler {
    template: State,
    viewer: PlayerId,
    /// Dev cards the viewer cannot place (in opponents' hands or the deck), in kind order.
    unseen: [DevCard; DEV_DECK_SIZE],
    unseen_len: usize,
    /// Per player: dev cards held (0 for the viewer, whose cards are known) and how many are new.
    held: [u8; NUM_PLAYERS],
    new: [u8; NUM_PLAYERS],
}

struct Deal {
    dev_hand: [[u8; 5]; NUM_PLAYERS],
    dev_new: [[u8; 5]; NUM_PLAYERS],
    deck: [DevCard; DEV_DECK_SIZE],
    deck_len: usize,
}

/// Setup placements done so far: each setup step ends with a road, so during setup it is the
/// number of roads on the board; afterwards it is 8.
fn setup_step(obs: &Observation) -> u8 {
    match obs.phase {
        Phase::SetupSettlement | Phase::SetupRoad { .. } => {
            obs.roads.iter().map(|r| r.count_ones()).sum::<u32>() as u8
        }
        _ => 8,
    }
}

impl WorldSampler {
    pub fn new(obs: &Observation, belief: &Belief) -> Result<WorldSampler, String> {
        belief.check(obs)?;
        if matches!(obs.phase, Phase::TradeResponse | Phase::TradeConfirm | Phase::GameOver { .. }) {
            return Err(format!("no worlds to sample in phase {:?}", obs.phase));
        }
        let me = obs.viewer as usize;
        let mut unseen_counts = DEV_DECK_COUNTS;
        for k in 0..5 {
            let played: u8 = (0..NUM_PLAYERS).map(|p| obs.dev_cards_played[p][k]).sum();
            unseen_counts[k] = unseen_counts[k]
                .checked_sub(played + obs.my_dev_cards[k])
                .ok_or_else(|| format!("more {:?} cards seen than the deck has", DevCard::from_index(k)))?;
        }
        let held: [u8; NUM_PLAYERS] = std::array::from_fn(|p| if p == me { 0 } else { obs.dev_card_counts[p] });
        let new: [u8; NUM_PLAYERS] = std::array::from_fn(|p| if p == me { 0 } else { obs.new_dev_card_counts[p] });
        let wanted = held.iter().map(|&c| c as u32).sum::<u32>() + obs.dev_deck_remaining as u32;
        let have: u32 = unseen_counts.iter().map(|&c| c as u32).sum();
        if wanted != have {
            return Err(format!("{have} unseen dev cards for {wanted} hidden places"));
        }
        let mut unseen = [DevCard::Knight; DEV_DECK_SIZE];
        let mut unseen_len = 0;
        for (k, &n) in unseen_counts.iter().enumerate() {
            for _ in 0..n {
                unseen[unseen_len] = DevCard::from_index(k);
                unseen_len += 1;
            }
        }
        let mut sampler = WorldSampler { template: State::new(0, obs.config), viewer: obs.viewer, unseen, unseen_len, held, new };
        // The template's hidden parts come from one sample; `sample` overwrites them each time.
        let mut rng = Rng::new(0x7E4D);
        let deal = sampler.deal(obs.current, obs.public_vp[obs.current as usize], obs.config.vp_to_win, &mut rng)?;
        let hands = belief.particles()[0].hands;
        let snap = Snapshot {
            board: obs.board,
            robber: obs.robber,
            bank: obs.bank,
            dev_deck: deal.deck[..deal.deck_len].to_vec(),
            players: std::array::from_fn(|p| PlayerSnapshot {
                hand: hands[p],
                dev_hand: if p == me { obs.my_dev_cards } else { deal.dev_hand[p] },
                dev_new: if p == me { obs.my_new_dev_cards } else { deal.dev_new[p] },
                dev_played: obs.dev_cards_played[p],
                knights_played: obs.knights_played[p],
                settlements: obs.settlements[p],
                cities: obs.cities[p],
                roads: obs.roads[p],
                discard_remaining: obs.discard_remaining[p],
            }),
            current: obs.current,
            phase: obs.phase,
            return_phase: obs.return_phase,
            turn: obs.turn,
            setup_step: setup_step(obs),
            dev_played_this_turn: obs.dev_played_this_turn,
            offers_this_turn: obs.offers_this_turn,
            longest_road_owner: obs.longest_road_owner,
            largest_army_owner: obs.largest_army_owner,
        };
        sampler.template = State::from_snapshot(0, obs.config, &snap)?;
        Ok(sampler)
    }

    /// A uniform deal of the unseen dev cards to opponents (new cards first) and the deck,
    /// rejecting deals that would give `current` (an opponent on turn) enough VP to have won.
    fn deal(&self, current: PlayerId, current_public_vp: u8, vp_to_win: u8, rng: &mut Rng) -> Result<Deal, String> {
        for _ in 0..MAX_DEALS {
            let mut cards = self.unseen;
            rng.shuffle(&mut cards[..self.unseen_len]);
            let mut d = Deal {
                dev_hand: [[0; 5]; NUM_PLAYERS],
                dev_new: [[0; 5]; NUM_PLAYERS],
                deck: [DevCard::Knight; DEV_DECK_SIZE],
                deck_len: 0,
            };
            let mut next = 0;
            for q in 0..NUM_PLAYERS {
                for i in 0..self.held[q] {
                    let c = cards[next].index();
                    next += 1;
                    d.dev_hand[q][c] += 1;
                    if i < self.new[q] {
                        d.dev_new[q][c] += 1;
                    }
                }
            }
            d.deck_len = self.unseen_len - next;
            d.deck[..d.deck_len].copy_from_slice(&cards[next..self.unseen_len]);
            let hidden_vp = d.dev_hand[current as usize][DevCard::VictoryPoint.index()];
            if current == self.viewer || current_public_vp + hidden_vp < vp_to_win {
                return Ok(d);
            }
        }
        Err(format!("no dev-card deal keeps player {current} below {vp_to_win} VP"))
    }

    /// One world: a uniformly chosen particle's hands, a fresh deal of the unseen dev cards, and
    /// a fresh random seed.
    pub fn sample(&self, belief: &Belief, rng: &mut Rng) -> State {
        let mut w = self.template;
        let particles = belief.particles();
        let hands = particles[rng.below(particles.len() as u32) as usize].hands;
        let current = w.current;
        let deal = self
            .deal(current, w.public_vp(current as usize), w.config.vp_to_win, rng)
            .expect("the template's own deal passed this check");
        for q in 0..NUM_PLAYERS {
            if q != self.viewer as usize {
                w.players[q].hand = hands[q];
                w.players[q].dev_hand = deal.dev_hand[q];
                w.players[q].dev_new = deal.dev_new[q];
            }
        }
        let rest = &deal.deck[..deal.deck_len];
        let pos = DEV_DECK_SIZE - rest.len();
        let mut drawn = DEV_DECK_COUNTS;
        for c in rest {
            drawn[c.index()] -= 1;
        }
        let mut i = 0;
        for (k, &n) in drawn.iter().enumerate() {
            for _ in 0..n {
                w.dev_deck[i] = DevCard::from_index(k);
                i += 1;
            }
        }
        w.dev_deck[pos..].copy_from_slice(rest);
        w.dev_deck_pos = pos as u8;
        w.reseed(rng.next_u64());
        w
    }
}
```

In `search/src/lib.rs` add `pub mod world;` and `pub use world::WorldSampler;`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `~/.cargo/bin/cargo test --workspace --release && ~/.cargo/bin/cargo clippy --release -p settler-search --all-targets -- -D warnings`
Expected: all pass (golden trace unchanged).

- [ ] **Step 5: Commit**

```bash
git add engine/src/state.rs engine/tests/snapshot.rs search/
git commit -m "search: sampled worlds from a validated template; engine State::reseed"
```

---

### Task 4: The search

**Files:**
- Create: `search/src/evaluator.rs`, `search/src/tree.rs`, `search/src/puct.rs`, `search/src/search.rs`
- Modify: `search/src/lib.rs`
- Test: unit tests in `search/src/puct.rs`; `search/tests/search.rs`

**Interfaces:**
- Consumes: `Belief`, `WorldSampler` (Tasks 2–3).
- Produces:
  - `settler_search::{Leaf, Eval, Evaluator, UniformEvaluator}` — `Leaf<'a> { pub world: &'a State, pub actor: PlayerId, pub legal: &'a [Action] }`, `Eval { pub value: [f32; 4], pub prior: Vec<f32> }`, `trait Evaluator { fn evaluate(&mut self, leaves: &[Leaf<'_>]) -> Vec<Eval>; }`.
  - `settler_search::{SearchConfig, SearchResult, search, search_tree, search_actions, terminal_value, visible_chance}` — `SearchConfig { simulations: u32, c_puct: f32, batch: usize }` (Default 1000 / 1.5 / 8); `SearchResult { action: Action, visits: Vec<u32> /* aligned with the legal slice passed in */, simulations: u32, nodes: usize }`; `search<E: Evaluator>(obs: &Observation, legal: &[Action], belief: &Belief, eval: &mut E, cfg: &SearchConfig, rng: &mut Rng) -> Result<SearchResult, String>`; `search_tree` (same arguments) `-> Result<(SearchResult, Tree), String>`; `search_actions(legal: &[Action], out: &mut Vec<Action>)`; `terminal_value(s: &State) -> [f32; 4]`; `visible_chance(s: &State, a: Action, viewer: PlayerId) -> bool`.
  - `settler_search::tree::{Tree, Node, NodeKind, Child, ROOT, UNEXPANDED}`.

- [ ] **Step 1: Write the failing tests**

`search/tests/search.rs`:

```rust
mod common;

use common::*;
use settler_engine::rng::Rng;
use settler_engine::*;
use settler_search::tree::{NodeKind, ROOT};
use settler_search::{search, search_tree, visible_chance, Belief, SearchConfig, UniformEvaluator};

fn cfg(simulations: u32) -> SearchConfig {
    SearchConfig { simulations, ..SearchConfig::default() }
}

/// Player 0 in Main with city resources and `vp_to_win` set so a city wins on the spot.
fn city_wins(seed: u64) -> State {
    let config = GameConfig { vp_to_win: 3, ..no_trades() };
    let base = after_setup(seed, config);
    edit(&base, config, |snap| {
        give_cards(snap, 0, CITY_COST);
        snap.phase = Phase::Main;
    })
}

#[test]
fn takes_a_winning_build() {
    for seed in 0..5 {
        let s = city_wins(seed);
        let obs = s.observation(0);
        let legal = s.legal_actions();
        let b = Belief::from_observation(&obs, 64, seed);
        let r = search(&obs, &legal, &b, &mut UniformEvaluator, &cfg(200), &mut Rng::new(seed)).unwrap();
        let mut after = s;
        after.apply(r.action);
        assert_eq!(after.winner(), Some(0), "seed {seed}: {:?} does not win", r.action);
    }
}

#[test]
fn the_same_seed_gives_the_same_search() {
    let s = city_wins(1);
    let obs = s.observation(0);
    let legal = s.legal_actions();
    let b = Belief::from_observation(&obs, 64, 1);
    let run = |seed| search(&obs, &legal, &b, &mut UniformEvaluator, &cfg(300), &mut Rng::new(seed)).unwrap();
    assert_eq!(run(9), run(9));
}

#[test]
fn root_visits_count_every_simulation_but_the_first() {
    let s = city_wins(2);
    let obs = s.observation(0);
    let legal = s.legal_actions();
    let b = Belief::from_observation(&obs, 64, 2);
    let r = search(&obs, &legal, &b, &mut UniformEvaluator, &cfg(250), &mut Rng::new(0)).unwrap();
    assert_eq!(r.visits.len(), legal.len());
    assert_eq!(r.simulations, 250);
    assert_eq!(r.visits.iter().sum::<u32>(), 249);
}

#[test]
fn a_single_legal_move_is_played_without_searching() {
    let s = after_setup(3, no_trades()); // PreRoll, no dev cards: only Roll
    let obs = s.observation(0);
    let legal = s.legal_actions();
    assert_eq!(legal, vec![Action::Roll]);
    let b = Belief::from_observation(&obs, 8, 0);
    let r = search(&obs, &legal, &b, &mut UniformEvaluator, &cfg(500), &mut Rng::new(0)).unwrap();
    assert_eq!((r.action, r.simulations), (Action::Roll, 0));
}

/// Player 0 before rolling, holding a knight: Roll and PlayKnight are both legal.
fn knight_before_roll(seed: u64, config: GameConfig) -> State {
    let base = after_setup(seed, config);
    edit(&base, config, |snap| give_dev(snap, 0, DevCard::Knight, 1))
}

#[test]
fn dice_branch_into_chance_outcomes() {
    let s = knight_before_roll(4, no_trades());
    let obs = s.observation(0);
    let legal = s.legal_actions();
    assert!(legal.contains(&Action::PlayKnight) && legal.contains(&Action::Roll));
    let b = Belief::from_observation(&obs, 64, 4);
    let (_, tree) = search_tree(&obs, &legal, &b, &mut UniformEvaluator, &cfg(400), &mut Rng::new(1)).unwrap();
    let roll = tree.nodes[ROOT as usize].children.iter().find(|c| c.key == Action::Roll.encode()).unwrap();
    let chance = &tree.nodes[roll.node as usize];
    assert_eq!(chance.kind, NodeKind::Chance { action: Action::Roll });
    assert!(chance.children.len() >= 6, "{} dice outcomes seen", chance.children.len());
    for c in &chance.children {
        assert!((2..=12).contains(&c.key));
        assert_eq!(tree.nodes[c.node as usize].kind, NodeKind::Decision);
    }
}

#[test]
fn which_chance_outcomes_the_viewer_sees() {
    let s = after_setup(5, no_trades());
    assert!(visible_chance(&s, Action::Roll, 2));
    let mut buying = s;
    buying.current = 1;
    assert!(visible_chance(&buying, Action::BuyDev, 1));
    assert!(!visible_chance(&buying, Action::BuyDev, 0));
    assert!(visible_chance(&buying, Action::StealFrom(0), 0), "the victim sees what was taken");
    assert!(visible_chance(&buying, Action::StealFrom(2), 1), "the thief sees what was taken");
    assert!(!visible_chance(&buying, Action::StealFrom(2), 0), "a bystander does not");
}

#[test]
fn reaching_the_turn_cap_inside_the_tree_is_a_draw_not_a_crash() {
    let config = GameConfig { max_turns: 1, ..no_trades() };
    let s = knight_before_roll(6, config);
    let obs = s.observation(0);
    let legal = s.legal_actions();
    let b = Belief::from_observation(&obs, 32, 6);
    let r = search(&obs, &legal, &b, &mut UniformEvaluator, &cfg(300), &mut Rng::new(2)).unwrap();
    assert!(legal.contains(&r.action));
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `~/.cargo/bin/cargo test --release -p settler-search --test search`
Expected: compile errors — `cannot find function search`, `unresolved module tree`.

- [ ] **Step 3: Implement**

`search/src/evaluator.rs`:

```rust
//! The seam between the search and whatever scores positions: a hand-written heuristic in 3a, a
//! network in 3b (spec Section 4).

use settler_engine::{Action, PlayerId, State, NUM_PLAYERS};

/// A position to evaluate: a sampled world, the player to act, and their legal moves.
pub struct Leaf<'a> {
    pub world: &'a State,
    pub actor: PlayerId,
    pub legal: &'a [Action],
}

#[derive(Clone, Debug, PartialEq)]
pub struct Eval {
    /// Win probability per player, summing to 1. May use the whole sampled world.
    pub value: [f32; NUM_PLAYERS],
    /// One probability per entry of `legal`, summing to 1. Must depend only on
    /// `world.observation(actor)`, so opponents in the tree do not play as if they saw hidden cards.
    pub prior: Vec<f32>,
}

pub trait Evaluator {
    /// One `Eval` per leaf, in order.
    fn evaluate(&mut self, leaves: &[Leaf<'_>]) -> Vec<Eval>;
}

/// Equal values and a uniform prior: the search's own baseline and a test double.
pub struct UniformEvaluator;

impl Evaluator for UniformEvaluator {
    fn evaluate(&mut self, leaves: &[Leaf<'_>]) -> Vec<Eval> {
        leaves
            .iter()
            .map(|l| Eval {
                value: [1.0 / NUM_PLAYERS as f32; NUM_PLAYERS],
                prior: vec![1.0 / l.legal.len() as f32; l.legal.len()],
            })
            .collect()
    }
}
```

`search/src/tree.rs`:

```rust
//! The search tree: nodes in one arena; a decision node's children are keyed by action id, a
//! chance node's by the outcome the searching player sees (dice sum, resource or dev card index).

use settler_engine::{Action, NUM_PLAYERS};

pub const ROOT: u32 = 0;
pub const UNEXPANDED: u32 = u32::MAX;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NodeKind {
    /// The player to act (read from each world: it is public) chooses an action.
    Decision,
    /// `action` was chosen; its random outcome is visible to the searching player.
    Chance { action: Action },
}

#[derive(Clone, Debug, PartialEq)]
pub struct Child {
    pub key: u16,
    /// Index of the child node, or `UNEXPANDED` until a simulation first goes there.
    pub node: u32,
    pub prior: f32,
    pub visits: u32,
    /// Simulations in which this action was legal at its parent (ISMCTS availability).
    pub available: u32,
    /// Simulations through this child waiting for their leaf evaluation (virtual loss).
    pub pending: u32,
    pub value_sum: [f32; NUM_PLAYERS],
}

impl Child {
    pub fn new(key: u16, prior: f32) -> Child {
        Child { key, node: UNEXPANDED, prior, visits: 0, available: 0, pending: 0, value_sum: [0.0; NUM_PLAYERS] }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Node {
    pub kind: NodeKind,
    pub children: Vec<Child>,
    /// A decision node is expanded once evaluated; chance nodes are created expanded.
    pub expanded: bool,
    /// A simulation in the current batch is waiting to evaluate this node.
    pub awaiting_eval: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Tree {
    pub nodes: Vec<Node>,
}

impl Default for Tree {
    fn default() -> Self {
        Tree::new()
    }
}

impl Tree {
    /// A tree holding only the unexpanded root decision node.
    pub fn new() -> Tree {
        let mut t = Tree { nodes: Vec::new() };
        t.add(NodeKind::Decision);
        t
    }

    pub fn add(&mut self, kind: NodeKind) -> u32 {
        self.nodes.push(Node {
            kind,
            children: Vec::new(),
            expanded: matches!(kind, NodeKind::Chance { .. }),
            awaiting_eval: false,
        });
        (self.nodes.len() - 1) as u32
    }

    /// Index (in `node`'s children) of the child keyed `key`, added with `prior` if missing.
    pub fn child(&mut self, node: u32, key: u16, prior: f32) -> usize {
        let children = &mut self.nodes[node as usize].children;
        match children.iter().position(|c| c.key == key) {
            Some(i) => i,
            None => {
                children.push(Child::new(key, prior));
                children.len() - 1
            }
        }
    }

    /// The node under `parent`'s child `i`, created as `kind` on first visit.
    pub fn descend_to(&mut self, parent: u32, i: usize, kind: NodeKind) -> u32 {
        let existing = self.nodes[parent as usize].children[i].node;
        if existing != UNEXPANDED {
            return existing;
        }
        let n = self.add(kind);
        self.nodes[parent as usize].children[i].node = n;
        n
    }
}
```

`search/src/puct.rs`:

```rust
//! Selection arithmetic, kept apart from the tree walk so it can be tested on synthetic children.

use crate::tree::Child;

/// Value assumed for an unvisited child when nothing at the node has been visited yet.
pub const DEFAULT_FPU: f32 = 0.25;

/// PUCT score of `c` for `actor`. Exploration uses the child's availability count instead of
/// the parent's visit count, as ISMCTS does when legal moves differ between worlds. Pending
/// (virtual-loss) visits count as visits worth nothing.
pub fn score(c: &Child, actor: usize, c_puct: f32, fpu: f32) -> f32 {
    let n = c.visits + c.pending;
    let q = if n == 0 { fpu } else { c.value_sum[actor] / n as f32 };
    q + c_puct * c.prior * (c.available.max(1) as f32).sqrt() / (1.0 + n as f32)
}

/// First-play urgency: the mean value for `actor` over the node's visited children.
pub fn fpu(children: &[Child], actor: usize) -> f32 {
    let (sum, n) = children.iter().fold((0.0, 0u32), |(s, n), c| (s + c.value_sum[actor], n + c.visits));
    if n == 0 {
        DEFAULT_FPU
    } else {
        sum / n as f32
    }
}

/// The best child for `actor` among those `allowed` (max^n: each player maximises their own
/// value). Ties go to the lower index.
pub fn select(children: &[Child], allowed: impl Fn(&Child) -> bool, actor: usize, c_puct: f32) -> Option<usize> {
    let f = fpu(children, actor);
    let mut best: Option<(usize, f32)> = None;
    for (i, c) in children.iter().enumerate() {
        if !allowed(c) {
            continue;
        }
        let s = score(c, actor, c_puct, f);
        if best.map_or(true, |(_, b)| s > b) {
            best = Some((i, s));
        }
    }
    best.map(|(i, _)| i)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn child(key: u16, visits: u32, available: u32, value_sum: [f32; 4]) -> Child {
        Child { visits, available, value_sum, ..Child::new(key, 0.5) }
    }

    #[test]
    fn each_player_maximises_their_own_value() {
        let kids = [child(0, 10, 10, [0.0, 0.0, 9.0, 0.0]), child(1, 10, 10, [9.0, 0.0, 0.0, 0.0])];
        assert_eq!(select(&kids, |_| true, 2, 0.0), Some(0));
        assert_eq!(select(&kids, |_| true, 0, 0.0), Some(1));
    }

    #[test]
    fn children_not_allowed_in_this_world_are_skipped() {
        let kids = [child(0, 10, 10, [9.0; 4]), child(1, 10, 10, [1.0; 4])];
        assert_eq!(select(&kids, |c| c.key == 1, 0, 1.0), Some(1));
        assert_eq!(select(&kids, |_| false, 0, 1.0), None);
    }

    #[test]
    fn exploration_grows_with_availability() {
        let kids = [child(0, 0, 1, [0.0; 4]), child(1, 0, 100, [0.0; 4])];
        assert_eq!(select(&kids, |_| true, 0, 1.0), Some(1));
    }

    #[test]
    fn pending_visits_lower_the_score() {
        let mut kids = [child(0, 4, 8, [2.0; 4]), child(1, 4, 8, [2.0; 4])];
        kids[0].pending = 3;
        assert_eq!(select(&kids, |_| true, 0, 1.0), Some(1));
    }

    #[test]
    fn ties_go_to_the_lower_index() {
        let kids = [child(0, 3, 3, [1.0; 4]), child(1, 3, 3, [1.0; 4])];
        assert_eq!(select(&kids, |_| true, 0, 1.0), Some(0));
    }
}
```

`search/src/search.rs`:

```rust
//! The simulation loop: single-observer ISMCTS with PUCT, 4-player value vectors, chance nodes
//! for outcomes the searching player sees, and leaf batching with virtual loss (spec Section 3).

use crate::belief::Belief;
use crate::evaluator::{Evaluator, Leaf};
use crate::puct;
use crate::tree::{Child, NodeKind, Tree, ROOT};
use crate::world::WorldSampler;
use settler_engine::legal::legal_actions;
use settler_engine::rng::Rng;
use settler_engine::rules::roll::pick_card;
use settler_engine::{Action, Chance, NoEvents, Observation, PlayerId, State, NUM_PLAYERS};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SearchConfig {
    /// Simulations per decision: the budget (never wall time).
    pub simulations: u32,
    pub c_puct: f32,
    /// Leaves per `Evaluator` call.
    pub batch: usize,
}

impl Default for SearchConfig {
    fn default() -> Self {
        SearchConfig { simulations: 1000, c_puct: 1.5, batch: 8 }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct SearchResult {
    pub action: Action,
    /// Root visit counts aligned with the `legal` slice given to `search` (0 for unsearched
    /// actions). They sum to `simulations - 1`: the first simulation evaluates the root.
    pub visits: Vec<u32>,
    pub simulations: u32,
    pub nodes: usize,
}

/// The actions the search considers: everything legal except domestic offers (3a never trades).
pub fn search_actions(legal: &[Action], out: &mut Vec<Action>) {
    out.clear();
    out.extend(legal.iter().copied().filter(|a| !matches!(a, Action::OfferTrade { .. })));
}

/// A finished game's value: 1 for the winner, ¼ each when the turn cap ended it.
pub fn terminal_value(s: &State) -> [f32; NUM_PLAYERS] {
    match s.winner() {
        Some(w) => {
            let mut v = [0.0; NUM_PLAYERS];
            v[w as usize] = 1.0;
            v
        }
        None => [1.0 / NUM_PLAYERS as f32; NUM_PLAYERS],
    }
}

/// Whether `viewer` sees the random outcome of `a` in `s`: every roll, their own dev-card draws,
/// and steals they make or suffer.
pub fn visible_chance(s: &State, a: Action, viewer: PlayerId) -> bool {
    match a {
        Action::Roll => true,
        Action::BuyDev => s.current == viewer,
        Action::StealFrom(v) => s.current == viewer || v == viewer,
        _ => false,
    }
}

/// Apply `a` with a sampled outcome and return the outcome code the viewer sees.
fn apply_visible(s: &mut State, a: Action, rng: &mut Rng) -> u16 {
    match a {
        Action::Roll => {
            let dice = (1 + rng.below(6) as u8, 1 + rng.below(6) as u8);
            s.apply_with(a, Some(Chance::Roll { dice, discards: None }), &mut NoEvents);
            (dice.0 + dice.1) as u16
        }
        Action::StealFrom(v) => {
            let r = pick_card(&s.players[v as usize].hand, rng);
            s.apply_with(a, Some(Chance::Steal(r)), &mut NoEvents);
            r.index() as u16
        }
        Action::BuyDev => {
            let card = s.dev_deck[s.dev_deck_pos as usize];
            s.apply(a);
            card.index() as u16
        }
        _ => unreachable!("{a:?} has no visible chance outcome"),
    }
}

/// The 665 action ids as a bitset.
#[derive(Clone, Copy, Default)]
struct ActionSet([u64; 11]);

impl ActionSet {
    fn of(actions: &[Action]) -> ActionSet {
        let mut s = ActionSet::default();
        for a in actions {
            let k = a.encode() as usize;
            s.0[k / 64] |= 1 << (k % 64);
        }
        s
    }

    fn contains(&self, key: u16) -> bool {
        (self.0[key as usize / 64] >> (key % 64)) & 1 == 1
    }

    fn remove(&mut self, key: u16) {
        self.0[key as usize / 64] &= !(1u64 << (key % 64));
    }
}

type Path = Vec<(u32, usize)>;

struct Pending {
    path: Path,
    node: u32,
    world: State,
    legal: Vec<Action>,
}

enum Descent {
    Terminal { path: Path, value: [f32; NUM_PLAYERS] },
    Leaf(Pending),
    /// Reached a node another simulation in this batch is already evaluating.
    Busy,
}

fn descend(
    tree: &mut Tree,
    mut world: State,
    viewer: PlayerId,
    c_puct: f32,
    rng: &mut Rng,
    buf: &mut Vec<Action>,
    legal: &mut Vec<Action>,
) -> Descent {
    let mut node = ROOT;
    let mut path = Path::new();
    loop {
        if world.is_over() {
            return Descent::Terminal { value: terminal_value(&world), path };
        }
        if let NodeKind::Chance { action } = tree.nodes[node as usize].kind {
            let key = apply_visible(&mut world, action, rng);
            let i = tree.child(node, key, 0.0);
            path.push((node, i));
            node = tree.descend_to(node, i, NodeKind::Decision);
            continue;
        }
        legal_actions(&world, buf);
        search_actions(buf, legal);
        let n = &mut tree.nodes[node as usize];
        if !n.expanded {
            if n.awaiting_eval {
                return Descent::Busy;
            }
            n.awaiting_eval = true;
            return Descent::Leaf(Pending { path, node, world, legal: legal.clone() });
        }
        let allowed = ActionSet::of(legal);
        let mut unseen = allowed;
        for c in n.children.iter_mut() {
            if allowed.contains(c.key) {
                c.available += 1;
                unseen.remove(c.key);
            }
        }
        // Actions this world allows that earlier worlds did not: uniform prior.
        let uniform = 1.0 / legal.len() as f32;
        for a in legal.iter() {
            if unseen.contains(a.encode()) {
                n.children.push(Child { available: 1, ..Child::new(a.encode(), uniform) });
            }
        }
        let actor = world.current_actor() as usize;
        let i = puct::select(&n.children, |c| allowed.contains(c.key), actor, c_puct)
            .expect("a live decision node has a legal action");
        let a = Action::decode(n.children[i].key).expect("children are keyed by valid action ids");
        path.push((node, i));
        if visible_chance(&world, a, viewer) {
            node = tree.descend_to(node, i, NodeKind::Chance { action: a });
        } else {
            world.apply(a);
            node = tree.descend_to(node, i, NodeKind::Decision);
        }
    }
}

fn set_pending(tree: &mut Tree, path: &[(u32, usize)], on: bool) {
    for &(n, i) in path {
        let c = &mut tree.nodes[n as usize].children[i];
        if on {
            c.pending += 1;
        } else {
            c.pending -= 1;
        }
    }
}

fn backup(tree: &mut Tree, path: &[(u32, usize)], value: &[f32; NUM_PLAYERS]) {
    for &(n, i) in path {
        let c = &mut tree.nodes[n as usize].children[i];
        c.visits += 1;
        for p in 0..NUM_PLAYERS {
            c.value_sum[p] += value[p];
        }
    }
}

fn expand(tree: &mut Tree, node: u32, legal: &[Action], prior: &[f32]) {
    assert_eq!(prior.len(), legal.len(), "the evaluator's prior must align with the legal actions");
    let n = &mut tree.nodes[node as usize];
    n.expanded = true;
    n.awaiting_eval = false;
    n.children.extend(legal.iter().zip(prior).map(|(a, &p)| Child { available: 1, ..Child::new(a.encode(), p) }));
}

/// Search `obs.viewer`'s decision among `legal` (see `search`), also returning the tree.
pub fn search_tree<E: Evaluator>(
    obs: &Observation,
    legal: &[Action],
    belief: &Belief,
    eval: &mut E,
    cfg: &SearchConfig,
    rng: &mut Rng,
) -> Result<(SearchResult, Tree), String> {
    assert!(cfg.simulations > 0 && cfg.batch > 0, "simulations and batch must be positive");
    let mut root_actions = Vec::new();
    search_actions(legal, &mut root_actions);
    match root_actions.len() {
        0 => return Err("nothing to search: no legal action".into()),
        1 => {
            let r = SearchResult { action: root_actions[0], visits: vec![0; legal.len()], simulations: 0, nodes: 0 };
            return Ok((r, Tree::new()));
        }
        _ => {}
    }
    let sampler = WorldSampler::new(obs, belief)?;
    let mut tree = Tree::new();
    let (mut buf, mut scratch) = (Vec::new(), Vec::new());
    let mut batch: Vec<Pending> = Vec::with_capacity(cfg.batch);
    let mut done = 0u32;
    while done < cfg.simulations {
        batch.clear();
        while batch.len() < cfg.batch && done + (batch.len() as u32) < cfg.simulations {
            let world = sampler.sample(belief, rng);
            match descend(&mut tree, world, obs.viewer, cfg.c_puct, rng, &mut buf, &mut scratch) {
                Descent::Terminal { path, value } => {
                    backup(&mut tree, &path, &value);
                    done += 1;
                }
                Descent::Leaf(p) => {
                    set_pending(&mut tree, &p.path, true);
                    batch.push(p);
                }
                Descent::Busy => break,
            }
        }
        if batch.is_empty() {
            continue;
        }
        let leaves: Vec<Leaf<'_>> = batch
            .iter()
            .map(|p| Leaf { world: &p.world, actor: p.world.current_actor(), legal: &p.legal })
            .collect();
        let evals = eval.evaluate(&leaves);
        assert_eq!(evals.len(), batch.len(), "the evaluator must return one evaluation per leaf");
        for (p, e) in batch.iter().zip(&evals) {
            expand(&mut tree, p.node, &p.legal, &e.prior);
            set_pending(&mut tree, &p.path, false);
            backup(&mut tree, &p.path, &e.value);
            done += 1;
        }
    }
    let root = &tree.nodes[ROOT as usize];
    let best = root
        .children
        .iter()
        .max_by(|a, b| a.visits.cmp(&b.visits).then(a.prior.total_cmp(&b.prior)).then(b.key.cmp(&a.key)))
        .expect("the first simulation expands the root");
    let action = Action::decode(best.key).expect("children are keyed by valid action ids");
    let visits = legal
        .iter()
        .map(|a| root.children.iter().find(|c| c.key == a.encode()).map_or(0, |c| c.visits))
        .collect();
    let nodes = tree.nodes.len();
    Ok((SearchResult { action, visits, simulations: done, nodes }, tree))
}

/// The most-visited move for `obs.viewer` among `legal` after `cfg.simulations` simulations, each
/// in a world sampled from `belief`. A single searchable move is returned without searching.
pub fn search<E: Evaluator>(
    obs: &Observation,
    legal: &[Action],
    belief: &Belief,
    eval: &mut E,
    cfg: &SearchConfig,
    rng: &mut Rng,
) -> Result<SearchResult, String> {
    search_tree(obs, legal, belief, eval, cfg, rng).map(|(r, _)| r)
}
```

Replace `search/src/lib.rs` with:

```rust
//! Information-set Monte Carlo tree search for AlphaSettler
//! (spec: docs/superpowers/specs/2026-10-01-bot-3a-ismcts-design.md).

pub mod belief;
pub mod evaluator;
pub mod puct;
pub mod search;
pub mod tree;
pub mod world;

pub use belief::{Belief, HandTracker, DEFAULT_PARTICLES};
pub use evaluator::{Eval, Evaluator, Leaf, UniformEvaluator};
pub use search::{search, search_actions, search_tree, terminal_value, visible_chance, SearchConfig, SearchResult};
pub use world::WorldSampler;
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `~/.cargo/bin/cargo test --release -p settler-search && ~/.cargo/bin/cargo clippy --release -p settler-search --all-targets -- -D warnings`
Expected: all pass, clippy clean.

- [ ] **Step 5: Commit**

```bash
git add search/
git commit -m "search: single-observer ISMCTS with PUCT, chance nodes and leaf batching"
```

---

### Task 5: Heuristic evaluator and IsmctsBot

**Files:**
- Modify: `bots/Cargo.toml` (depend on settler-search), `bots/src/lib.rs` (modules, registry), `bots/src/greedy.rs` (extract `choose`)
- Create: `bots/src/heuristic.rs`, `bots/src/ismcts.rs`
- Test: `bots/tests/ismcts.rs`, `bots/tests/heuristic.rs`

**Interfaces:**
- Consumes: everything in `settler_search` (Task 4); `EventFeed`, `play_game`, `run_match` (Task 1).
- Produces:
  - `settler_bots::greedy::choose(obs: &Observation, legal: &[Action]) -> Action`.
  - `settler_bots::heuristic::{NUM_FEATURES, FEATURE_NAMES, DEFAULT_WEIGHTS, features, value, prior, HeuristicEvaluator}` — `features(s: &State, p: usize) -> [f32; NUM_FEATURES]`; `value(s: &State, w: &[f32; NUM_FEATURES]) -> [f32; 4]`; `prior(obs: &Observation, legal: &[Action]) -> Vec<f32>`; `HeuristicEvaluator::new(weights, rollout: u32)`.
  - `settler_bots::ismcts::{IsmctsBot, DEFAULT_SIMULATIONS, MAX_SIMULATIONS, MAX_ROLLOUT, parse_name}` — `IsmctsBot::new(seed: u64, simulations: u32, rollout: u32)`; `take_last_search(&mut self) -> Option<SearchResult>`; `parse_name(&str) -> Option<(u32, u32)>`.
  - `make_bot` accepts the `ismcts` names (Global Constraints); `BOT_NAMES` = `["random", "greedy", "ismcts"]`.

- [ ] **Step 1: Write the failing tests**

`bots/tests/heuristic.rs`:

```rust
use settler_bots::greedy::GreedyBot;
use settler_bots::heuristic::{features, prior, value, HeuristicEvaluator, DEFAULT_WEIGHTS, NUM_FEATURES};
use settler_bots::Bot;
use settler_engine::legal::legal_actions;
use settler_engine::*;
use settler_search::{search_actions, Belief, Evaluator, Leaf, WorldSampler};

/// Positions from greedy self-play, every 9th move, skipping trade phases.
fn positions(seeds: std::ops::Range<u64>) -> Vec<State> {
    let cfg = GameConfig { max_offers_per_turn: 0, ..GameConfig::default() };
    let mut out = Vec::new();
    for seed in seeds {
        let mut s = State::new(seed, cfg);
        let mut bots: Vec<GreedyBot> = (0..4).map(GreedyBot::new).collect();
        let mut buf = Vec::new();
        let mut n = 0;
        while !s.is_over() {
            legal_actions(&s, &mut buf);
            let actor = s.current_actor() as usize;
            if n % 9 == 0 {
                out.push(s);
            }
            let a = bots[actor].act(&s.observation(actor as u8), &buf);
            s.apply(a);
            n += 1;
        }
    }
    out
}

#[test]
fn values_are_probabilities_and_favour_the_leader() {
    for s in positions(0..6) {
        let v = value(&s, &DEFAULT_WEIGHTS);
        assert!((v.iter().sum::<f32>() - 1.0).abs() < 1e-4);
        assert!(v.iter().all(|&x| x > 0.0));
        let scores: Vec<f32> = (0..4).map(|p| features(&s, p).iter().zip(&DEFAULT_WEIGHTS).map(|(f, w)| f * w).sum()).collect();
        let best = (0..4).max_by(|&a, &b| scores[a].total_cmp(&scores[b])).unwrap();
        assert!((0..4).all(|p| v[best] >= v[p]));
    }
    assert_eq!(NUM_FEATURES, DEFAULT_WEIGHTS.len());
}

#[test]
fn the_prior_is_a_distribution_over_legal_moves() {
    for s in positions(0..3) {
        let actor = s.current_actor();
        let mut legal = Vec::new();
        search_actions(&s.legal_actions(), &mut legal);
        let p = prior(&s.observation(actor), &legal);
        assert_eq!(p.len(), legal.len());
        assert!((p.iter().sum::<f32>() - 1.0).abs() < 1e-4);
    }
}

#[test]
fn the_prior_depends_only_on_what_the_actor_sees() {
    let mut eval = HeuristicEvaluator::new(DEFAULT_WEIGHTS, 0);
    let mut rng = settler_engine::rng::Rng::new(1);
    let mut checked = 0;
    for s in positions(10..16) {
        if matches!(s.phase, Phase::TradeResponse | Phase::TradeConfirm) {
            continue;
        }
        let actor = s.current_actor();
        let obs = s.observation(actor);
        let belief = Belief::from_observation(&obs, 32, checked);
        let sampler = WorldSampler::new(&obs, &belief).unwrap();
        let (w1, w2) = (sampler.sample(&belief, &mut rng), sampler.sample(&belief, &mut rng));
        let mut legal = Vec::new();
        search_actions(&s.legal_actions(), &mut legal);
        let e = eval.evaluate(&[Leaf { world: &w1, actor, legal: &legal }, Leaf { world: &w2, actor, legal: &legal }]);
        assert_eq!(e[0].prior, e[1].prior);
        checked += 1;
    }
    assert!(checked > 100);
}
```

`bots/tests/ismcts.rs`:

```rust
mod common;

use settler_bots::arena::{play_game, run_match};
use settler_bots::ismcts::{parse_name, IsmctsBot};
use settler_bots::{make_bot, Bot, BOT_NAMES};
use settler_engine::rng::Rng;
use settler_engine::*;

fn no_trades() -> GameConfig {
    GameConfig { max_offers_per_turn: 0, ..GameConfig::default() }
}

#[test]
fn names_parse_and_bad_ones_are_rejected() {
    assert!(BOT_NAMES.contains(&"ismcts"));
    assert_eq!(parse_name("ismcts"), Some((1000, 0)));
    assert_eq!(parse_name("ismcts@1"), Some((1, 0)));
    assert_eq!(parse_name("ismcts@1000000"), Some((1_000_000, 0)));
    assert_eq!(parse_name("ismcts@300+r8"), Some((300, 8)));
    for bad in ["ismcts@0", "ismcts@", "ismcts@abc", "ismcts@+5", "ismcts@1000001", "ismcts@300+r0",
                "ismcts@300+r201", "ismcts@300+r", "ismctsx", "ismcts@300/r8", "ismcts+r8"] {
        assert_eq!(parse_name(bad), None, "{bad}");
        assert!(make_bot(bad, 0).is_none(), "{bad}");
    }
    assert_eq!(make_bot("ismcts@50", 0).unwrap().name(), "ismcts");
}

#[test]
fn plays_whole_games_without_belief_resets() {
    for seed in 0..2u64 {
        let mut bots: Vec<Box<dyn Bot>> =
            (0..4u64).map(|p| Box::new(IsmctsBot::new(seed * 4 + p, 20, 0)) as Box<dyn Bot>).collect();
        let (_, _, turns, _) = play_game(seed, no_trades(), &mut bots);
        assert!(turns > 0);
        for b in &bots {
            let d: std::collections::HashMap<_, _> = b.diagnostics().into_iter().collect();
            assert_eq!(d["belief_resets"], 0, "seed {seed}");
            assert!(d["searches"] > 0);
        }
    }
}

#[test]
fn arena_results_do_not_depend_on_thread_count() {
    let a = run_match("ismcts@20", "greedy", 0..3, no_trades(), 1).unwrap();
    let b = run_match("ismcts@20", "greedy", 0..3, no_trades(), 3).unwrap();
    assert_eq!(a, b);
}

#[test]
fn beats_random() {
    let records = run_match("ismcts@100", "random", 0..10, no_trades(), 10).unwrap();
    let wins = records.iter().filter(|r| r.winner == Some(r.candidate_seat)).count();
    assert!(wins as f64 / records.len() as f64 > 0.8, "{wins}/{}", records.len());
}

#[test]
fn answers_offers_without_searching() {
    let mut g = Game::new(2, GameConfig::default());
    let mut rng = Rng::new(2);
    let (mut rejected, mut cancelled) = (false, false);
    while !g.state().is_over() && !(rejected && cancelled) {
        let s = *g.state();
        let legal = g.legal_actions();
        let actor = s.current_actor();
        let mut bot = IsmctsBot::new(0, 1_000_000, 0); // a search this size would take minutes
        if s.phase == Phase::TradeResponse {
            assert_eq!(bot.act(&s.observation(actor), &legal), Action::RejectTrade);
            rejected = true;
        }
        if s.phase == Phase::TradeConfirm {
            assert_eq!(bot.act(&s.observation(actor), &legal), Action::CancelTrade);
            cancelled = true;
        }
        let pick = if s.phase == Phase::TradeResponse && legal.contains(&Action::AcceptTrade) {
            Action::AcceptTrade
        } else {
            legal[rng.below(legal.len() as u32) as usize]
        };
        g.apply(pick).unwrap();
    }
    assert!(rejected && cancelled, "random play with trades reaches both trade phases");
}

#[test]
fn a_contradictory_history_is_counted_and_recovered_from() {
    // City cards, so the decision is searched (a single legal move would skip the belief).
    let s = main_with(3, CITY_COST, no_trades());
    let mut bot = IsmctsBot::new(1, 50, 0);
    bot.observe(0, &[
        Event::Produced { player: 1, resources: [1, 0, 0, 0, 0] },
        Event::Discarded { player: 1, resource: Resource::Ore },
    ]);
    let legal = s.legal_actions();
    let a = bot.act(&s.observation(0), &legal);
    assert!(legal.contains(&a));
    let d: std::collections::HashMap<_, _> = bot.diagnostics().into_iter().collect();
    assert_eq!(d["belief_resets"], 1);
}

/// Player 0 in Main right after setup with `cards` added.
fn main_with(seed: u64, cards: Hand, config: GameConfig) -> State {
    let mut s = State::new(seed, config);
    let mut rng = Rng::new(seed);
    let mut buf = Vec::new();
    while matches!(s.phase, Phase::SetupSettlement | Phase::SetupRoad { .. }) {
        settler_engine::legal::legal_actions(&s, &mut buf);
        s.apply(buf[rng.below(buf.len() as u32) as usize]);
    }
    let mut snap = s.snapshot();
    for r in 0..NUM_RESOURCES {
        snap.bank[r] -= cards[r];
        snap.players[0].hand[r] += cards[r];
    }
    snap.phase = Phase::Main;
    State::from_snapshot(seed, config, &snap).unwrap()
}

fn decide(bot: &mut IsmctsBot, s: &State) -> Action {
    let obs = s.observation(0);
    let legal = s.legal_actions();
    let a = bot.act(&obs, &legal);
    assert!(legal.contains(&a));
    a
}

#[test]
fn builds_an_affordable_city_rather_than_ending_the_turn() {
    for seed in 0..4 {
        let s = main_with(seed, CITY_COST, no_trades());
        let mut bot = IsmctsBot::new(seed, 1000, 0);
        assert!(matches!(decide(&mut bot, &s), Action::BuildCity(_)), "seed {seed}");
    }
}

#[test]
fn never_robs_its_own_buildings_when_an_opponent_tile_is_free() {
    use settler_engine::topology::topo;
    for seed in 0..4 {
        let mut s = main_with(seed, [0; 5], no_trades());
        let mut snap = s.snapshot();
        let i = snap.dev_deck.iter().position(|&c| c == DevCard::Knight).unwrap();
        snap.dev_deck.remove(i);
        snap.players[0].dev_hand[DevCard::Knight.index()] = 1;
        s = State::from_snapshot(seed, no_trades(), &snap).unwrap();
        s.apply(Action::PlayKnight);
        assert_eq!(s.phase, Phase::MoveRobber);
        let mine = s.buildings(0);
        let theirs = (1..4).fold(0, |m, p| m | s.buildings(p));
        let touches = |t: u8, m: u64| topo().tile_node_mask[t as usize] & m != 0;
        assert!((0..19).any(|t| t != s.robber && touches(t, theirs) && !touches(t, mine)));
        let mut bot = IsmctsBot::new(seed, 1000, 0);
        let Action::MoveRobber(t) = decide(&mut bot, &s) else { panic!("not a robber move") };
        assert!(!touches(t, mine), "seed {seed}: robbed own tile {t}");
    }
}

#[test]
fn search_results_are_kept_for_self_play() {
    let s = main_with(5, CITY_COST, no_trades());
    let mut bot = IsmctsBot::new(5, 40, 0);
    decide(&mut bot, &s);
    let r = bot.take_last_search().expect("the decision was searched");
    assert_eq!(r.simulations, 40);
    assert!(bot.take_last_search().is_none());
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `~/.cargo/bin/cargo test --release -p settler-bots --test ismcts --test heuristic`
Expected: compile errors — `unresolved module heuristic`, `unresolved module ismcts`.

- [ ] **Step 3: Implement**

`bots/Cargo.toml` `[dependencies]` gains:

```toml
settler-search = { path = "../search" }
```

In `bots/src/greedy.rs`, move the body of `GreedyBot::act` into a public function and make `act` call it:

```rust
/// GreedyBot's move for `obs.viewer` among `legal` (non-empty). Also the heuristic evaluator's prior.
pub fn choose(obs: &Observation, legal: &[Action]) -> Action {
    let pick = match obs.phase {
        // ... the existing `match obs.phase { ... }` from GreedyBot::act, unchanged ...
    };
    pick.filter(|a| legal.contains(a)).unwrap_or(legal[0])
}

impl Bot for GreedyBot {
    fn name(&self) -> &'static str {
        "greedy"
    }

    fn act(&mut self, obs: &Observation, legal: &[Action]) -> Action {
        choose(obs, legal)
    }
}
```

(The match arms move verbatim; GreedyBot's behaviour is unchanged, which the existing greedy tests confirm.)

`bots/src/heuristic.rs`:

```rust
//! The hand-written evaluator 3a searches with; 3b replaces it with a network (spec Section 4).

use crate::greedy::{self, pips};
use settler_engine::legal::legal_actions;
use settler_engine::topology::{topo, NUM_TILES};
use settler_engine::types::*;
use settler_engine::{Action, Observation, State};
use settler_search::{search_actions, terminal_value, Eval, Evaluator, Leaf};

pub const NUM_FEATURES: usize = 8;
pub const FEATURE_NAMES: [&str; NUM_FEATURES] =
    ["vp", "production", "hand", "over_limit", "dev_cards", "road_length", "knights", "settlement_spots"];
/// Weights of the per-player strength score. Hand-set until `alphasettler fit-heuristic` refits
/// them (docs/bot/results-3a.md records each fit).
pub const DEFAULT_WEIGHTS: [f32; NUM_FEATURES] = [1.0, 0.08, 0.05, -0.1, 0.3, 0.05, 0.1, 0.2];

/// Per resource: the board's mean pips per resource over this resource's pips (scarce resources
/// weigh more). 0 for a resource with no tiles.
fn scarcity(s: &State) -> [f32; NUM_RESOURCES] {
    let mut total = [0u32; NUM_RESOURCES];
    for tile in 0..NUM_TILES {
        if let Some(r) = s.board.tile_resource[tile] {
            total[r.index()] += pips(s.board.tile_number[tile]);
        }
    }
    let mean = total.iter().sum::<u32>() as f32 / NUM_RESOURCES as f32;
    total.map(|t| if t == 0 { 0.0 } else { mean / t as f32 })
}

/// Free nodes at the end of player `p`'s roads where the distance rule allows a settlement.
fn settlement_spots(s: &State, p: usize) -> u32 {
    let t = topo();
    let occupied = s.occupied_nodes();
    let mut reach = 0u64;
    for e in bits128(s.players[p].roads) {
        let (a, b) = t.edge_nodes[e as usize];
        reach |= (1u64 << a) | (1u64 << b);
    }
    bits64(reach)
        .filter(|&n| occupied & (1u64 << n) == 0 && occupied & t.node_neighbor_mask[n as usize] == 0)
        .count() as u32
}

/// Player `p`'s features in `s` (which may be a sampled world), in `FEATURE_NAMES` order.
pub fn features(s: &State, p: usize) -> [f32; NUM_FEATURES] {
    let t = topo();
    let pl = &s.players[p];
    let scarce = scarcity(s);
    let mut production = 0.0;
    for tile in 0..NUM_TILES {
        if tile as u8 == s.robber {
            continue;
        }
        let Some(r) = s.board.tile_resource[tile] else { continue };
        let mask = t.tile_node_mask[tile];
        let weight = (pl.settlements & mask).count_ones() + 2 * (pl.cities & mask).count_ones();
        production += (weight * pips(s.board.tile_number[tile])) as f32 * scarce[r.index()];
    }
    let cards = hand_total(&pl.hand) as f32;
    let dev = pl.dev_hand.iter().sum::<u8>() - pl.dev_hand[DevCard::VictoryPoint.index()];
    [
        s.total_vp(p) as f32,
        production,
        cards.min(7.0),
        (cards - 7.0).max(0.0),
        dev as f32,
        pl.longest_road_len as f32,
        pl.knights_played as f32,
        settlement_spots(s, p) as f32,
    ]
}

fn softmax(x: &mut [f32]) {
    let m = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut total = 0.0;
    for v in x.iter_mut() {
        *v = (*v - m).exp();
        total += *v;
    }
    for v in x.iter_mut() {
        *v /= total;
    }
}

/// Win probabilities: a softmax over the four players' weighted feature scores.
pub fn value(s: &State, w: &[f32; NUM_FEATURES]) -> [f32; NUM_PLAYERS] {
    let mut v: [f32; NUM_PLAYERS] =
        std::array::from_fn(|p| features(s, p).iter().zip(w).map(|(f, w)| f * w).sum());
    softmax(&mut v);
    v
}

/// Logits: GreedyBot's choice +2, a city or settlement +1, buying a dev card +0.5, everything
/// else 0; softmaxed over `legal`. Reads only the actor's observation.
pub fn prior(obs: &Observation, legal: &[Action]) -> Vec<f32> {
    let pick = greedy::choose(obs, legal);
    let mut logits: Vec<f32> = legal
        .iter()
        .map(|&a| {
            let base = match a {
                Action::BuildCity(_) | Action::BuildSettlement(_) => 1.0,
                Action::BuyDev => 0.5,
                _ => 0.0,
            };
            if a == pick {
                base + 2.0
            } else {
                base
            }
        })
        .collect();
    softmax(&mut logits);
    logits
}

pub struct HeuristicEvaluator {
    pub weights: [f32; NUM_FEATURES],
    /// Greedy moves played from each leaf (in its own sampled world) before scoring it; 0 = none.
    pub rollout: u32,
    buf: Vec<Action>,
    moves: Vec<Action>,
}

impl HeuristicEvaluator {
    pub fn new(weights: [f32; NUM_FEATURES], rollout: u32) -> HeuristicEvaluator {
        HeuristicEvaluator { weights, rollout, buf: Vec::new(), moves: Vec::new() }
    }

    fn leaf_value(&mut self, world: &State) -> [f32; NUM_PLAYERS] {
        if self.rollout == 0 {
            return value(world, &self.weights);
        }
        let mut s = *world;
        for _ in 0..self.rollout {
            if s.is_over() {
                return terminal_value(&s);
            }
            legal_actions(&s, &mut self.buf);
            search_actions(&self.buf, &mut self.moves);
            let actor = s.current_actor();
            let a = greedy::choose(&s.observation(actor), &self.moves);
            s.apply(a);
        }
        if s.is_over() {
            terminal_value(&s)
        } else {
            value(&s, &self.weights)
        }
    }
}

impl Evaluator for HeuristicEvaluator {
    fn evaluate(&mut self, leaves: &[Leaf<'_>]) -> Vec<Eval> {
        let mut out = Vec::with_capacity(leaves.len());
        for l in leaves {
            let prior = prior(&l.world.observation(l.actor), l.legal);
            let value = self.leaf_value(l.world);
            out.push(Eval { value, prior });
        }
        out
    }
}
```

`bots/src/ismcts.rs`:

```rust
//! IsmctsBot: tracks hidden cards from the event log and searches each real decision with
//! single-observer ISMCTS (spec Sections 2-3). Never trades.

use crate::greedy;
use crate::heuristic::{HeuristicEvaluator, DEFAULT_WEIGHTS};
use crate::Bot;
use settler_engine::rng::{mix, Rng};
use settler_engine::{Action, Event, Observation, Phase, PlayerId};
use settler_search::{search, search_actions, Belief, SearchConfig, SearchResult, DEFAULT_PARTICLES};

pub const DEFAULT_SIMULATIONS: u32 = 1000;
pub const MAX_SIMULATIONS: u32 = 1_000_000;
pub const MAX_ROLLOUT: u32 = 200;
const BELIEF_SALT: u64 = 0xBE11EF;

/// `ismcts` -> (1000, 0); `ismcts@N` -> (N, 0); `ismcts@N+rD` -> (N, D). Digits only, no sign.
pub fn parse_name(name: &str) -> Option<(u32, u32)> {
    let rest = name.strip_prefix("ismcts")?;
    if rest.is_empty() {
        return Some((DEFAULT_SIMULATIONS, 0));
    }
    let rest = rest.strip_prefix('@')?;
    let number = |s: &str| -> Option<u32> {
        if s.is_empty() || !s.bytes().all(|b| b.is_ascii_digit()) {
            return None;
        }
        s.parse().ok()
    };
    let (sims, rollout) = match rest.split_once("+r") {
        Some((s, r)) => (number(s)?, number(r)?),
        None => (number(rest)?, 0),
    };
    let sims_ok = (1..=MAX_SIMULATIONS).contains(&sims);
    let rollout_ok = rest.contains("+r") == (rollout > 0) && rollout <= MAX_ROLLOUT;
    (sims_ok && rollout_ok).then_some((sims, rollout))
}

pub struct IsmctsBot {
    seed: u64,
    rng: Rng,
    cfg: SearchConfig,
    eval: HeuristicEvaluator,
    belief: Option<Belief>,
    last: Option<SearchResult>,
    resets: u64,
    searches: u64,
    buf: Vec<Action>,
}

impl IsmctsBot {
    pub fn new(seed: u64, simulations: u32, rollout: u32) -> IsmctsBot {
        IsmctsBot {
            seed,
            rng: Rng::new(seed),
            cfg: SearchConfig { simulations, ..SearchConfig::default() },
            eval: HeuristicEvaluator::new(DEFAULT_WEIGHTS, rollout),
            belief: None,
            last: None,
            resets: 0,
            searches: 0,
            buf: Vec::new(),
        }
    }

    /// The last decision's search, if it searched; taking it clears it.
    pub fn take_last_search(&mut self) -> Option<SearchResult> {
        self.last.take()
    }
}

impl Bot for IsmctsBot {
    fn name(&self) -> &'static str {
        "ismcts"
    }

    fn observe(&mut self, viewer: PlayerId, events: &[Event]) {
        if self.belief.as_ref().map_or(true, |b| b.viewer() != viewer) {
            self.belief = Some(Belief::new(viewer, DEFAULT_PARTICLES, mix(self.seed, BELIEF_SALT)));
        }
        let ok = self.belief.as_mut().expect("set above").observe(events).is_ok();
        if !ok {
            self.belief = None; // rebuilt from the next observation; `act` counts the reset
        }
    }

    fn act(&mut self, obs: &Observation, legal: &[Action]) -> Action {
        self.last = None;
        match obs.phase {
            Phase::TradeResponse if legal.contains(&Action::RejectTrade) => return Action::RejectTrade,
            Phase::TradeConfirm if legal.contains(&Action::CancelTrade) => return Action::CancelTrade,
            _ => {}
        }
        search_actions(legal, &mut self.buf);
        if self.buf.len() == 1 {
            return self.buf[0];
        }
        let usable = self.belief.as_ref().is_some_and(|b| b.viewer() == obs.viewer && b.check(obs).is_ok());
        if !usable {
            self.resets += 1;
            self.belief = Some(Belief::from_observation(obs, DEFAULT_PARTICLES, self.rng.next_u64()));
        }
        let belief = self.belief.as_ref().expect("set above");
        match search(obs, legal, belief, &mut self.eval, &self.cfg, &mut self.rng) {
            Ok(r) => {
                self.searches += 1;
                let a = r.action;
                self.last = Some(r);
                a
            }
            Err(_) => {
                self.resets += 1;
                greedy::choose(obs, legal)
            }
        }
    }

    fn diagnostics(&self) -> Vec<(&'static str, u64)> {
        vec![
            ("belief_resets", self.resets),
            ("belief_rebuilds", self.belief.as_ref().map_or(0, |b| b.rebuilds())),
            ("searches", self.searches),
        ]
    }
}
```

In `bots/src/lib.rs`: add `pub mod heuristic;` and `pub mod ismcts;` (`selfplay` comes in Task 6), and replace the registry with:

```rust
pub const BOT_NAMES: &[&str] = &["random", "greedy", "ismcts"];

/// A fresh bot by name, with its own random stream seeded from `seed`. Besides `BOT_NAMES`,
/// `ismcts@N` (N simulations) and `ismcts@N+rD` (plus D greedy rollout moves per leaf).
pub fn make_bot(name: &str, seed: u64) -> Option<Box<dyn Bot>> {
    match name {
        "random" => Some(Box::new(random::RandomBot::new(seed))),
        "greedy" => Some(Box::new(greedy::GreedyBot::new(seed))),
        _ => {
            let (sims, rollout) = ismcts::parse_name(name)?;
            Some(Box::new(ismcts::IsmctsBot::new(seed, sims, rollout)))
        }
    }
}
```

In `bots/src/arena.rs` change the unknown-bot error in `run_match` to:

```rust
            return Err(format!(
                "unknown bot {name:?}; known bots: {BOT_NAMES:?} (ismcts@N for N simulations, ismcts@N+rD to add D greedy rollout moves)"
            ));
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `~/.cargo/bin/cargo test --workspace --release && ~/.cargo/bin/cargo clippy --release --workspace --all-targets -- -D warnings`
Expected: all pass. If `beats_random`, `builds_an_affordable_city...` or `never_robs...` fail, that is a search or evaluator bug to fix, not a threshold to relax.

- [ ] **Step 5: Commit**

```bash
git add bots/
git commit -m "bots: heuristic evaluator and IsmctsBot (ismcts, ismcts@N, ismcts@N+rD)"
```

---

### Task 6: Self-play records, the weight fit, bindings and CLI

**Files:**
- Create: `bots/src/selfplay.rs`, `alphasettler/records.py`, `bots/tests/selfplay.rs`, `tests/python/test_selfplay.py`
- Modify: `bots/src/lib.rs` (`pub mod selfplay;`), `bots/src/heuristic.rs` (fit), `bindings/src/lib.rs`, `bindings/src/convert.rs` (`parse_event`), `alphasettler/cli.py`, `tests/python/test_cli.py` (append)

**Interfaces:**
- Consumes: `IsmctsBot::{new, take_last_search}`, `EventFeed`, `split_seeds`, `bot_seed`, `heuristic::features` (Tasks 1, 5).
- Produces:
  - `settler_bots::selfplay::{Decision, SelfPlayGame, play, run}` — `play(seed, config, simulations, rollout) -> SelfPlayGame`; `run(seeds: Range<u64>, config, simulations, rollout, threads) -> Result<Vec<SelfPlayGame>, String>`.
  - `settler_bots::heuristic::{Sample, ReplayGame, samples_from_games, log_likelihood, fit}`.
  - Python `_engine`: `Bot.observe(viewer, events)`, `Bot.diagnostics() -> dict`, `selfplay(seed_start, games, simulations, threads, config=None, rollout=0) -> list[dict]`, `fit_heuristic(games, iterations=25, l2=0.001) -> dict`.
  - Python `alphasettler.records`: `write(path, games, config)`, `read(path)`, `replay_mismatches(game) -> list[str]`.
  - CLI: `alphasettler selfplay --games N --simulations S [--rollout D] [--seed-start K] [--threads T] [--out-dir runs] [--trades]`; `alphasettler fit-heuristic RECORDS... [--iterations 25] [--l2 0.001]`.

- [ ] **Step 1: Write the failing Rust tests**

`bots/tests/selfplay.rs`:

```rust
use settler_bots::heuristic::{fit, log_likelihood, samples_from_games, ReplayGame, Sample, DEFAULT_WEIGHTS, NUM_FEATURES};
use settler_bots::selfplay::{play, run};
use settler_engine::rng::Rng;
use settler_engine::*;

fn no_trades() -> GameConfig {
    GameConfig { max_offers_per_turn: 0, ..GameConfig::default() }
}

#[test]
fn every_searched_decision_is_recorded_and_replays() {
    let g = play(3, no_trades(), 10, 0);
    assert!(!g.decisions.is_empty());
    let mut s = State::new(3, no_trades());
    let mut decisions = g.decisions.iter().peekable();
    for (i, &a) in g.actions.iter().enumerate() {
        if let Some(d) = decisions.next_if(|d| d.index as usize == i) {
            assert_eq!(d.actor, s.current_actor());
            assert_eq!(d.legal, s.legal_actions());
            assert!(d.legal.contains(&a));
            assert_eq!(d.visits.len(), d.legal.len());
            assert_eq!(d.visits.iter().sum::<u32>(), 9);
            assert!(d.full_search);
        }
        s.apply(a);
    }
    assert!(decisions.next().is_none());
    assert_eq!(s.winner(), g.winner);
    assert_eq!(s.turn, g.turns);
}

#[test]
fn self_play_does_not_depend_on_thread_count() {
    assert_eq!(run(0..4, no_trades(), 10, 0, 1).unwrap(), run(0..4, no_trades(), 10, 0, 3).unwrap());
}

#[test]
fn the_fit_recovers_known_weights() {
    let truth = [0.8f32, -0.5, 0.3, 0.0, 0.6, -0.2, 0.1, 0.4];
    let mut rng = Rng::new(11);
    let mut uniform = || (rng.next_u64() >> 40) as f32 / (1u64 << 24) as f32;
    let samples: Vec<Sample> = (0..6000)
        .map(|_| {
            let features: [[f32; NUM_FEATURES]; 4] = std::array::from_fn(|_| std::array::from_fn(|_| 3.0 * uniform()));
            let scores: Vec<f32> = features.iter().map(|f| f.iter().zip(&truth).map(|(x, w)| x * w).sum()).collect();
            let m = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let e: Vec<f32> = scores.iter().map(|s| (s - m).exp()).collect();
            let total: f32 = e.iter().sum();
            let (mut u, mut winner) = (uniform() * total, 3);
            for (p, &x) in e.iter().enumerate() {
                if u < x {
                    winner = p;
                    break;
                }
                u -= x;
            }
            Sample { features, winner }
        })
        .collect();
    let w = fit(&samples, [0.0; NUM_FEATURES], 25, 0.0);
    for i in 0..NUM_FEATURES {
        assert!((w[i] - truth[i]).abs() < 0.15, "weight {i}: {} vs {}", w[i], truth[i]);
    }
    assert!(log_likelihood(&samples, &w) >= log_likelihood(&samples, &[0.0; NUM_FEATURES]));
}

#[test]
fn fitting_on_real_games_does_not_lower_the_likelihood() {
    let games: Vec<ReplayGame> = run(0..6, no_trades(), 10, 0, 6)
        .unwrap()
        .into_iter()
        .map(|g| ReplayGame {
            seed: g.seed,
            config: no_trades(),
            at: g.decisions.iter().map(|d| d.index).collect(),
            actions: g.actions,
            winner: g.winner,
        })
        .collect();
    let samples = samples_from_games(&games).unwrap();
    assert!(samples.len() > 100);
    let w = fit(&samples, DEFAULT_WEIGHTS, 25, 1e-3);
    assert!(log_likelihood(&samples, &w) >= log_likelihood(&samples, &DEFAULT_WEIGHTS));
}
```

- [ ] **Step 2: Run them to verify they fail**

Run: `~/.cargo/bin/cargo test --release -p settler-bots --test selfplay`
Expected: compile errors — `unresolved module selfplay`, `cannot find fit`.

- [ ] **Step 3: Implement the Rust side**

`bots/src/selfplay.rs`:

```rust
//! Self-play: IsmctsBot in all four seats, recording every searched decision (spec Section 4).

use crate::arena::{bot_seed, split_seeds, EventFeed, MAX_SEEDS};
use crate::ismcts::IsmctsBot;
use crate::Bot;
use settler_engine::legal::legal_actions;
use settler_engine::{Action, GameConfig, PlayerId, State, NUM_PLAYERS};
use std::ops::Range;

#[derive(Clone, Debug, PartialEq)]
pub struct Decision {
    /// Index into the game's `actions` of the move this search chose.
    pub index: u32,
    pub actor: PlayerId,
    /// Every legal action (domestic offers included), in engine order.
    pub legal: Vec<Action>,
    /// Root visits aligned with `legal`.
    pub visits: Vec<u32>,
    /// Always true in 3a; 3b's playout cap randomization records fast searches as false.
    pub full_search: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SelfPlayGame {
    pub seed: u64,
    pub actions: Vec<Action>,
    pub decisions: Vec<Decision>,
    pub winner: Option<PlayerId>,
    pub vp: [u8; NUM_PLAYERS],
    pub turns: u32,
}

pub fn play(seed: u64, config: GameConfig, simulations: u32, rollout: u32) -> SelfPlayGame {
    let mut bots: Vec<IsmctsBot> =
        (0..NUM_PLAYERS).map(|p| IsmctsBot::new(bot_seed(seed, p), simulations, rollout)).collect();
    let mut s = State::new(seed, config);
    let (mut feed, mut buf, mut step) = (EventFeed::default(), Vec::new(), Vec::new());
    let (mut actions, mut decisions) = (Vec::new(), Vec::new());
    while !s.is_over() {
        legal_actions(&s, &mut buf);
        let actor = s.current_actor() as usize;
        feed.deliver(actor, &mut bots[actor]);
        let a = bots[actor].act(&s.observation(actor as PlayerId), &buf);
        assert!(buf.contains(&a), "ismcts chose illegal action {a:?} in phase {:?}", s.phase);
        if let Some(r) = bots[actor].take_last_search() {
            decisions.push(Decision {
                index: actions.len() as u32,
                actor: actor as PlayerId,
                legal: buf.clone(),
                visits: r.visits,
                full_search: true,
            });
        }
        actions.push(a);
        step.clear();
        s.apply_with(a, None, &mut step);
        feed.push(&step);
    }
    SelfPlayGame { seed, actions, decisions, winner: s.winner(), vp: std::array::from_fn(|p| s.total_vp(p)), turns: s.turn }
}

/// Every seed in `seeds`, spread over `threads`; sorted by seed and independent of `threads`.
pub fn run(seeds: Range<u64>, config: GameConfig, simulations: u32, rollout: u32, threads: usize) -> Result<Vec<SelfPlayGame>, String> {
    config.validate()?;
    let n = seeds.end.saturating_sub(seeds.start);
    if n > MAX_SEEDS {
        return Err(format!("at most {MAX_SEEDS} games per call, got {n}"));
    }
    let mut games: Vec<SelfPlayGame> = std::thread::scope(|scope| {
        let workers: Vec<_> = split_seeds(seeds, threads)
            .into_iter()
            .map(|part| scope.spawn(move || part.map(|seed| play(seed, config, simulations, rollout)).collect::<Vec<_>>()))
            .collect();
        workers
            .into_iter()
            .flat_map(|w| w.join().unwrap_or_else(|panic| std::panic::resume_unwind(panic)))
            .collect()
    });
    games.sort_by_key(|g| g.seed);
    Ok(games)
}
```

Add `pub mod selfplay;` to `bots/src/lib.rs`.

Append to `bots/src/heuristic.rs`:

```rust
/// One position for the weight fit: every player's features, and who went on to win.
#[derive(Clone, Debug, PartialEq)]
pub struct Sample {
    pub features: [[f32; NUM_FEATURES]; NUM_PLAYERS],
    pub winner: usize,
}

/// A recorded game to replay for samples: its moves and the move indices to sample at.
#[derive(Clone, Debug, PartialEq)]
pub struct ReplayGame {
    pub seed: u64,
    pub config: settler_engine::GameConfig,
    pub actions: Vec<Action>,
    pub winner: Option<PlayerId>,
    pub at: Vec<u32>,
}

/// The true position before each sampled move, labelled with the winner. Draws are skipped.
pub fn samples_from_games(games: &[ReplayGame]) -> Result<Vec<Sample>, String> {
    let mut out = Vec::new();
    let mut buf = Vec::new();
    for g in games {
        let Some(winner) = g.winner else { continue };
        let mut s = State::new(g.seed, g.config);
        let mut at = g.at.iter().peekable();
        for (i, &a) in g.actions.iter().enumerate() {
            if at.next_if(|&&k| k as usize == i).is_some() {
                out.push(Sample { features: std::array::from_fn(|p| features(&s, p)), winner: winner as usize });
            }
            legal_actions(&s, &mut buf);
            if !buf.contains(&a) {
                return Err(format!("game {}: move {i} ({a:?}) is illegal on replay", g.seed));
            }
            s.apply(a);
        }
        if s.winner() != g.winner {
            return Err(format!("game {}: the replay ends with winner {:?}, the record says {:?}", g.seed, s.winner(), g.winner));
        }
    }
    Ok(out)
}

fn scores(w: &[f64; NUM_FEATURES], f: &[[f32; NUM_FEATURES]; NUM_PLAYERS]) -> [f64; NUM_PLAYERS] {
    std::array::from_fn(|p| (0..NUM_FEATURES).map(|i| w[i] * f[p][i] as f64).sum())
}

fn probabilities(z: [f64; NUM_PLAYERS]) -> [f64; NUM_PLAYERS] {
    let m = z.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let e = z.map(|x| (x - m).exp());
    let total: f64 = e.iter().sum();
    e.map(|x| x / total)
}

fn mean_ll(samples: &[Sample], w: &[f64; NUM_FEATURES]) -> f64 {
    let total: f64 = samples.iter().map(|s| probabilities(scores(w, &s.features))[s.winner].ln()).sum();
    total / samples.len().max(1) as f64
}

/// Mean log-likelihood of the winners under `weights`.
pub fn log_likelihood(samples: &[Sample], weights: &[f32; NUM_FEATURES]) -> f64 {
    mean_ll(samples, &weights.map(|x| x as f64))
}

/// Solve `a x = b` by Gaussian elimination with partial pivoting; None if singular.
fn solve(mut a: [[f64; NUM_FEATURES]; NUM_FEATURES], mut b: [f64; NUM_FEATURES]) -> Option<[f64; NUM_FEATURES]> {
    let n = NUM_FEATURES;
    for col in 0..n {
        let pivot = (col..n).max_by(|&i, &j| a[i][col].abs().total_cmp(&a[j][col].abs()))?;
        if a[pivot][col].abs() < 1e-12 {
            return None;
        }
        a.swap(col, pivot);
        b.swap(col, pivot);
        for row in col + 1..n {
            let f = a[row][col] / a[col][col];
            for k in col..n {
                a[row][k] -= f * a[col][k];
            }
            b[row] -= f * b[col];
        }
    }
    let mut x = [0.0; NUM_FEATURES];
    for row in (0..n).rev() {
        let rest: f64 = (row + 1..n).map(|k| a[row][k] * x[k]).sum();
        x[row] = (b[row] - rest) / a[row][row];
    }
    Some(x)
}

/// Maximum-likelihood weights of the conditional logit P(winner) = softmax(w · features),
/// maximising mean log-likelihood minus `l2`·|w|², by Newton's method with step halving.
pub fn fit(samples: &[Sample], start: [f32; NUM_FEATURES], iterations: u32, l2: f64) -> [f32; NUM_FEATURES] {
    let n = samples.len().max(1) as f64;
    let objective = |w: &[f64; NUM_FEATURES]| mean_ll(samples, w) - l2 * w.iter().map(|x| x * x).sum::<f64>();
    let mut w = start.map(|x| x as f64);
    for _ in 0..iterations {
        let mut g = [0.0f64; NUM_FEATURES];
        let mut h = [[0.0f64; NUM_FEATURES]; NUM_FEATURES];
        for s in samples {
            let pi = probabilities(scores(&w, &s.features));
            let f = s.features.map(|r| r.map(|x| x as f64));
            let mut mean = [0.0f64; NUM_FEATURES];
            for p in 0..NUM_PLAYERS {
                for i in 0..NUM_FEATURES {
                    mean[i] += pi[p] * f[p][i];
                }
            }
            for i in 0..NUM_FEATURES {
                g[i] += f[s.winner][i] - mean[i];
                for j in 0..NUM_FEATURES {
                    let second: f64 = (0..NUM_PLAYERS).map(|p| pi[p] * f[p][i] * f[p][j]).sum();
                    h[i][j] -= second - mean[i] * mean[j];
                }
            }
        }
        for i in 0..NUM_FEATURES {
            g[i] = g[i] / n - 2.0 * l2 * w[i];
            for j in 0..NUM_FEATURES {
                h[i][j] /= n;
            }
            h[i][i] -= 2.0 * l2;
        }
        let Some(step) = solve(h, g) else { break };
        let base = objective(&w);
        let mut t = 1.0;
        let next = loop {
            let cand: [f64; NUM_FEATURES] = std::array::from_fn(|i| w[i] - t * step[i]);
            if objective(&cand) >= base || t < 1e-6 {
                break cand;
            }
            t /= 2.0;
        };
        let moved: f64 = (0..NUM_FEATURES).map(|i| (next[i] - w[i]).abs()).sum();
        w = next;
        if moved < 1e-9 {
            break;
        }
    }
    w.map(|x| x as f32)
}
```

Also add `use settler_engine::PlayerId;` to the heuristic imports if not already reachable through `types::*` (it is: `PlayerId` lives in `types`).

- [ ] **Step 4: Run the Rust tests to verify they pass**

Run: `~/.cargo/bin/cargo test --workspace --release`
Expected: all pass.

- [ ] **Step 5: Write the failing Python tests**

`tests/python/test_selfplay.py`:

```python
import gzip
import json

import pytest

from alphasettler import Bot, Game, records
from alphasettler._engine import fit_heuristic, selfplay

CONFIG = {"max_offers_per_turn": 0}


def test_selfplay_games_have_searched_decisions():
    games = selfplay(0, 2, 10, 2, CONFIG)
    assert [g["seed"] for g in games] == [0, 1]
    for g in games:
        assert g["decisions"]
        for d in g["decisions"]:
            assert g["actions"][d["index"]] in d["legal"]
            assert len(d["visits"]) == len(d["legal"]) and sum(d["visits"]) == 9
            assert d["full_search"] is True
            assert d["observation"]["viewer"] == d["actor"]


def test_records_round_trip_and_replay_exactly(tmp_path):
    path = tmp_path / "sp.jsonl.gz"
    games = selfplay(5, 2, 10, 2, CONFIG)
    records.write(path, games, CONFIG)
    back = list(records.read(path))
    assert [g["seed"] for g in back] == [5, 6]
    assert all(g["config"] == CONFIG for g in back)
    for g in back:
        assert records.replay_mismatches(g) == []
    with pytest.raises(FileExistsError):
        records.write(path, games, CONFIG)


def test_replay_detects_a_tampered_record(tmp_path):
    g = json.loads(json.dumps({"config": CONFIG, **selfplay(7, 1, 10, 1, CONFIG)[0]}))
    g["decisions"][0]["observation"]["my_hand"][0] += 1
    assert records.replay_mismatches(g)


def test_fit_heuristic_reports_weights_and_likelihoods():
    games = [{"config": CONFIG, **g} for g in selfplay(0, 4, 10, 4, CONFIG)]
    r = fit_heuristic(games)
    assert len(r["weights"]) == 8 and r["samples"] > 0
    assert r["log_likelihood_after"] >= r["log_likelihood_before"]


def test_selfplay_arguments_are_checked():
    for bad in [dict(simulations=0), dict(simulations=1_000_001), dict(rollout=201), dict(games=0)]:
        args = {"seed_start": 0, "games": 1, "simulations": 5, "threads": 1, "config": CONFIG, "rollout": 0, **bad}
        with pytest.raises(ValueError):
            selfplay(**args)


def test_ismcts_bot_from_python_observes_and_reports():
    g = Game(3, CONFIG)
    bot = Bot("ismcts@10", 1)
    seen = 0
    while not g.is_over and g.turn < 3:
        actor = g.current_actor
        if actor == 0:
            log = g.log(0)
            bot.observe(0, log[seen:])
            seen = len(log)
            a = bot.act(g)
        else:
            a = g.legal_actions()[0]
        g.apply(a)
    d = bot.diagnostics()
    assert d["belief_resets"] == 0 and d["searches"] > 0


def test_a_reused_bot_survives_a_new_game():
    bot = Bot("ismcts@10", 2)
    for seed in (1, 2):
        g = Game(seed, CONFIG)
        for _ in range(60):
            if g.is_over:
                break
            a = bot.act(g)
            assert a in g.legal_actions()
            g.apply(a)
    assert bot.diagnostics()["belief_resets"] >= 1


def test_bad_ismcts_names():
    for bad in ["ismcts@0", "ismcts@abc", "ismcts@300+r0"]:
        with pytest.raises(ValueError, match="unknown bot"):
            Bot(bad, 0)
```

Append to `tests/python/test_cli.py`:

```python
def test_cli_selfplay_and_fit(tmp_path):
    r = cli("selfplay", "--games", "3", "--simulations", "10", "--out-dir", str(tmp_path))
    assert r.returncode == 0, r.stderr
    files = list(tmp_path.glob("*-selfplay-s10.jsonl.gz"))
    assert len(files) == 1
    assert "3 games" in r.stdout
    f = cli("fit-heuristic", str(files[0]))
    assert f.returncode == 0, f.stderr
    assert "pub const DEFAULT_WEIGHTS: [f32; NUM_FEATURES] = [" in f.stdout
    assert "log-likelihood per sample" in f.stdout


def test_cli_compare_accepts_ismcts_names(tmp_path):
    r = cli("compare", "--a", "ismcts@5", "--b", "greedy", "--baseline", "random", "--seeds", "1",
            "--no-trades", "--out-dir", str(tmp_path))
    assert r.returncode == 0, r.stderr
    bad = cli("compare", "--a", "ismcts@0", "--b", "greedy", "--baseline", "random", "--seeds", "1")
    assert bad.returncode == 2 and "unknown bot" in bad.stderr
```

- [ ] **Step 6: Run them to verify they fail**

Run: `PATH=$HOME/.cargo/bin:$PATH .venv/bin/maturin develop --release -q && .venv/bin/pytest -q tests/python/test_selfplay.py tests/python/test_cli.py`
Expected: FAIL — `cannot import name 'fit_heuristic'`, `cannot import name 'records'`, `invalid choice: 'selfplay'`.

- [ ] **Step 7: Implement the bindings**

In `bindings/src/convert.rs` add (it uses the existing private helpers `item`, `index_arg`, `small_array`, `parse_resource`, `parse_dev`, `opt`, `opt_player`):

```rust
/// An event dict in the shape `event_dict` writes, e.g. from another engine's translated log.
pub fn parse_event(v: &Bound<'_, PyAny>) -> PyResult<Event> {
    const E: &str = "event";
    let d = v.cast::<PyDict>()?;
    let get = |k: &str| item(d, k, E);
    let player = |k: &str| -> PyResult<PlayerId> { Ok(index_arg(&get(k)?, NUM_PLAYERS, k)? as PlayerId) };
    let hand = |k: &str| -> PyResult<Hand> { small_array(&get(k)?, k) };
    let resource = |k: &str| -> PyResult<Resource> { parse_resource(&get(k)?.extract::<String>()?) };
    let kind: String = get("type")?.extract()?;
    Ok(match kind.as_str() {
        "built_settlement" => Event::BuiltSettlement { player: player("player")?, node: index_arg(&get("node")?, NUM_NODES, "node")? as u8 },
        "built_city" => Event::BuiltCity { player: player("player")?, node: index_arg(&get("node")?, NUM_NODES, "node")? as u8 },
        "built_road" => Event::BuiltRoad { player: player("player")?, edge: index_arg(&get("edge")?, NUM_EDGES, "edge")? as u8 },
        "rolled" => {
            let dice = small_dice(&get("dice")?)?;
            Event::Rolled { player: player("player")?, dice }
        }
        "produced" => Event::Produced { player: player("player")?, resources: hand("resources")? },
        "discarded" => Event::Discarded { player: player("player")?, resource: resource("resource")? },
        "robber_moved" => Event::RobberMoved { player: player("player")?, tile: index_arg(&get("tile")?, NUM_TILES, "tile")? as u8 },
        "stole" => Event::Stole {
            thief: player("thief")?,
            victim: player("victim")?,
            resource: opt(&get("resource")?, |r| parse_resource(&r.extract::<String>()?))?,
        },
        "bought_dev" => Event::BoughtDev {
            player: player("player")?,
            card: opt(&get("card")?, |c| parse_dev(&c.extract::<String>()?))?,
        },
        "played_dev" => Event::PlayedDev { player: player("player")?, card: parse_dev(&get("card")?.extract::<String>()?)? },
        "monopoly_taken" => Event::MonopolyTaken {
            player: player("player")?,
            resource: resource("resource")?,
            amount: int_arg::<u8>(&get("amount")?, "amount")?,
        },
        "year_of_plenty_taken" => Event::YearOfPlentyTaken { player: player("player")?, resources: hand("resources")? },
        "maritime_traded" => Event::MaritimeTraded { player: player("player")?, gave: hand("gave")?, got: hand("got")? },
        "trade_offered" => Event::TradeOffered { player: player("player")?, give: hand("give")?, get: hand("get")? },
        "trade_responded" => Event::TradeResponded { player: player("player")?, accepted: get("accepted")?.extract::<bool>()? },
        "trade_confirmed" => Event::TradeConfirmed {
            offerer: player("offerer")?,
            partner: player("partner")?,
            offerer_gave: hand("offerer_gave")?,
            partner_gave: hand("partner_gave")?,
        },
        "trade_cancelled" => Event::TradeCancelled { player: player("player")? },
        "turn_ended" => Event::TurnEnded { player: player("player")? },
        "game_over" => Event::GameOver { winner: opt_player(&get("winner")?, "winner")? },
        other => return Err(value_error(format!("unknown event type {other:?}"))),
    })
}

fn small_dice(v: &Bound<'_, PyAny>) -> PyResult<(u8, u8)> {
    let xs = seq_arg(v, 2, "dice")?;
    let die = |x: &Bound<'_, PyAny>| -> PyResult<u8> {
        let d = int_arg::<u8>(x, "dice")?;
        if (1..=6).contains(&d) { Ok(d) } else { Err(value_error(format!("a die shows 1-6, got {d}"))) }
    };
    Ok((die(&xs[0])?, die(&xs[1])?))
}
```

(Import `NUM_NODES`, `NUM_EDGES`, `NUM_TILES` from `settler_engine::topology` and `Event` from `settler_engine` at the top of `convert.rs` if they are not already imported.)

In `bindings/src/lib.rs`, add to `impl PyBot`:

```rust
    /// Show the bot new events as `viewer` saw them (dicts shaped like `Game.log` entries).
    fn observe(&self, viewer: &Bound<'_, PyAny>, events: &Bound<'_, PyList>) -> PyResult<()> {
        let viewer = viewer_arg(viewer)?;
        let events = events.iter().map(|e| parse_event(&e)).collect::<PyResult<Vec<_>>>()?;
        let mut bot = self.inner.lock().map_err(|_| value_error("bot state is poisoned"))?;
        bot.observe(viewer, &events);
        Ok(())
    }

    /// The bot's named counters (e.g. `belief_resets` for ismcts); empty for bots without any.
    fn diagnostics<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let bot = self.inner.lock().map_err(|_| value_error("bot state is poisoned"))?;
        let d = PyDict::new(py);
        for (k, v) in bot.diagnostics() {
            d.set_item(k, v)?;
        }
        Ok(d)
    }
```

and add these functions (register both in `_engine` with `m.add_function(wrap_pyfunction!(selfplay, m)?)?;` and the same for `fit_heuristic`):

```rust
/// IsmctsBot self-play: `games` games from `seed_start`, every searched decision recorded with
/// the actor's observation. Releases the GIL while games run.
#[pyfunction]
#[pyo3(signature = (seed_start, games, simulations, threads, config=None, rollout=0))]
fn selfplay<'py>(
    py: Python<'py>,
    seed_start: &Bound<'py, PyAny>,
    games: &Bound<'py, PyAny>,
    simulations: &Bound<'py, PyAny>,
    threads: &Bound<'py, PyAny>,
    config: Option<&Bound<'py, PyDict>>,
    rollout: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyList>> {
    use settler_bots::ismcts::{MAX_ROLLOUT, MAX_SIMULATIONS};
    let seed_start: u64 = int_arg(seed_start, "seed_start")?;
    let games: u64 = int_arg(games, "games")?;
    let simulations: u32 = int_arg(simulations, "simulations")?;
    let threads: usize = int_arg(threads, "threads")?;
    let rollout: u32 = int_arg(rollout, "rollout")?;
    if games == 0 {
        return Err(value_error("games must be positive"));
    }
    if !(1..=MAX_SIMULATIONS).contains(&simulations) {
        return Err(value_error(format!("simulations must be 1..={MAX_SIMULATIONS}, got {simulations}")));
    }
    if rollout > MAX_ROLLOUT {
        return Err(value_error(format!("rollout must be 0..={MAX_ROLLOUT}, got {rollout}")));
    }
    let end = seed_start
        .checked_add(games)
        .ok_or_else(|| value_error("seed_start + games exceeds the u64 seed range"))?;
    let config = parse_config(config)?;
    let played = py
        .detach(|| settler_bots::selfplay::run(seed_start..end, config, simulations, rollout, threads))
        .map_err(value_error)?;
    let out = PyList::empty(py);
    for g in played {
        let d = PyDict::new(py);
        d.set_item("seed", g.seed)?;
        d.set_item("actions", g.actions.iter().map(|a| a.encode() as u32).collect::<Vec<_>>())?;
        d.set_item("winner", g.winner)?;
        d.set_item("vp", counts(&g.vp))?;
        d.set_item("turns", g.turns)?;
        // Replay to attach each decision's observation.
        let mut s = settler_engine::State::new(g.seed, config);
        let decisions = PyList::empty(py);
        let mut next = g.decisions.iter().peekable();
        for (i, &a) in g.actions.iter().enumerate() {
            if let Some(dec) = next.next_if(|dec| dec.index as usize == i) {
                let e = PyDict::new(py);
                e.set_item("index", dec.index)?;
                e.set_item("actor", dec.actor)?;
                e.set_item("legal", dec.legal.iter().map(|a| a.encode() as u32).collect::<Vec<_>>())?;
                e.set_item("visits", dec.visits.clone())?;
                e.set_item("full_search", dec.full_search)?;
                e.set_item("observation", observation_dict(py, &s.observation(dec.actor))?)?;
                decisions.append(e)?;
            }
            s.apply(a);
        }
        d.set_item("decisions", decisions)?;
        out.append(d)?;
    }
    Ok(out)
}

/// Fit the heuristic's weights on recorded games (dicts with `seed`, `config`, `actions`,
/// `winner` and `decisions[].index`). Returns the weights, the sample count and the mean
/// log-likelihood under the default and the fitted weights.
#[pyfunction]
#[pyo3(signature = (games, iterations=25, l2=0.001))]
fn fit_heuristic<'py>(py: Python<'py>, games: &Bound<'py, PyList>, iterations: u32, l2: f64) -> PyResult<Bound<'py, PyDict>> {
    use settler_bots::heuristic::{fit, log_likelihood, samples_from_games, ReplayGame, DEFAULT_WEIGHTS};
    let mut replay = Vec::new();
    for g in games.iter() {
        let g = g.cast::<PyDict>()?;
        let get = |k: &str| g.get_item(k)?.ok_or_else(|| value_error(format!("game record is missing {k:?}")));
        let config_obj = get("config")?;
        let config = parse_config(Some(config_obj.cast::<PyDict>()?))?;
        let actions = get("actions")?
            .extract::<Vec<Bound<'_, PyAny>>>()?
            .iter()
            .map(action_from_id)
            .collect::<PyResult<Vec<_>>>()?;
        let at = get("decisions")?
            .extract::<Vec<Bound<'_, PyDict>>>()?
            .iter()
            .map(|d| d.get_item("index")?.ok_or_else(|| value_error("decision is missing \"index\""))?.extract::<u32>())
            .collect::<PyResult<Vec<_>>>()?;
        let winner = get("winner")?.extract::<Option<u8>>()?;
        replay.push(ReplayGame { seed: int_arg(&get("seed")?, "seed")?, config, actions, winner, at });
    }
    let (samples, weights) = py
        .detach(|| {
            let samples = samples_from_games(&replay)?;
            let weights = fit(&samples, DEFAULT_WEIGHTS, iterations, l2);
            Ok::<_, String>((samples, weights))
        })
        .map_err(value_error)?;
    let d = PyDict::new(py);
    d.set_item("weights", weights.to_vec())?;
    d.set_item("samples", samples.len())?;
    d.set_item("log_likelihood_before", log_likelihood(&samples, &DEFAULT_WEIGHTS))?;
    d.set_item("log_likelihood_after", log_likelihood(&samples, &weights))?;
    Ok(d)
}
```

- [ ] **Step 8: Implement the Python side**

`alphasettler/records.py`:

```python
"""Self-play records: one gzipped JSONL line per game (spec Section 4)."""

from __future__ import annotations

import gzip
import json
from collections.abc import Iterator
from pathlib import Path

from alphasettler._engine import Game


def write(path: str | Path, games: list[dict], config: dict) -> None:
    """Write `games` (as returned by `selfplay`) to a new file; an existing file is an error."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "xt") as f:
        for g in games:
            f.write(json.dumps({"config": config, **g}) + "\n")


def read(path: str | Path) -> Iterator[dict]:
    with gzip.open(path, "rt") as f:
        for line in f:
            yield json.loads(line)


def _plain(x):
    """As it would read back from JSON (tuples become lists)."""
    return json.loads(json.dumps(x))


def replay_mismatches(game: dict) -> list[str]:
    """Replay `game` from its seed and actions; describe every stored decision that differs."""
    g = Game(game["seed"], game["config"])
    decisions = {d["index"]: d for d in game["decisions"]}
    out = []
    for i, a in enumerate(game["actions"]):
        d = decisions.get(i)
        if d is not None:
            if g.current_actor != d["actor"]:
                out.append(f"move {i}: actor {g.current_actor}, record says {d['actor']}")
            if g.legal_actions() != d["legal"]:
                out.append(f"move {i}: legal actions differ")
            if _plain(g.observation(d["actor"])) != d["observation"]:
                out.append(f"move {i}: observation differs")
        g.apply(a)
    if g.winner != game["winner"]:
        out.append(f"winner {g.winner}, record says {game['winner']}")
    return out
```

In `alphasettler/__init__.py` add `from alphasettler import records  # noqa: F401` after the existing imports (and `"records"` to `__all__`).

In `alphasettler/cli.py`:

1. Add parsers in `_parser()` before `return p`:

```python
    s = sub.add_parser("selfplay", help="IsmctsBot self-play records for training and weight fits")
    s.add_argument("--games", type=_positive_int, required=True)
    s.add_argument("--simulations", type=_positive_int, required=True)
    s.add_argument("--rollout", type=int, default=0)
    s.add_argument("--seed-start", type=int, default=0)
    s.add_argument("--threads", type=_positive_int, default=None)
    s.add_argument("--out-dir", default="runs")
    s.add_argument("--trades", action="store_true", help="allow domestic offers (ismcts never makes any)")

    f = sub.add_parser("fit-heuristic", help="fit the heuristic evaluator's weights on self-play records")
    f.add_argument("records", nargs="+")
    f.add_argument("--iterations", type=_positive_int, default=25)
    f.add_argument("--l2", type=float, default=0.001)
```

2. Add handlers:

```python
SELFPLAY_BATCH = 100  # games per native call; Ctrl-C is handled between calls


def _selfplay(args) -> int:
    from alphasettler import records
    from alphasettler._engine import selfplay

    config = {} if args.trades else {"max_offers_per_turn": 0}
    if args.seed_start < 0 or args.seed_start + args.games > SEED_LIMIT:
        raise ValueError(f"seeds {args.seed_start}..{args.seed_start + args.games} are outside the u64 seed range")
    out = Path(args.out_dir) / f"{_stamp()}-selfplay-s{args.simulations}.jsonl.gz"
    threads = args.threads or os.cpu_count() or 1
    games = []
    end = args.seed_start + args.games
    for lo in range(args.seed_start, end, SELFPLAY_BATCH):
        games.extend(selfplay(lo, min(SELFPLAY_BATCH, end - lo), args.simulations, threads, config, args.rollout))
    records.write(out, games, config)
    decisions = sum(len(g["decisions"]) for g in games)
    print(f"{len(games)} games, {decisions} searched decisions, wrote {out}")
    return 0


def _fit_heuristic(args) -> int:
    from alphasettler import records
    from alphasettler._engine import fit_heuristic

    games = []
    for path in args.records:
        for g in records.read(path):
            g["decisions"] = [{"index": d["index"]} for d in g["decisions"]]  # observations are not needed
            games.append(g)
    r = fit_heuristic(games, args.iterations, args.l2)
    print(f"samples: {r['samples']} from {len(games)} games")
    print(f"log-likelihood per sample: default {r['log_likelihood_before']:.4f}, fitted {r['log_likelihood_after']:.4f}")
    weights = ", ".join(f"{w:.4f}" for w in r["weights"])
    print(f"pub const DEFAULT_WEIGHTS: [f32; NUM_FEATURES] = [{weights}];")
    return 0
```

(Import `os` at the top of `cli.py` if it is not already imported.)

3. In `main`, dispatch the new commands next to `oracle-diff`:

```python
        elif args.command == "selfplay":
            return _selfplay(args)
        elif args.command == "fit-heuristic":
            return _fit_heuristic(args)
```

4. In the `compare` branch, delete the `known = bot_names()` loop that checks names against the list (the existing `check(...)` calls validate every name, including `ismcts@N`), and drop the now-unused `bot_names` import if nothing else uses it.

5. Validate `--rollout` the same way the bindings do: the native call raises `ValueError`, which `main` already turns into exit code 2.

- [ ] **Step 9: Run everything**

Run: `~/.cargo/bin/cargo test --workspace --release && PATH=$HOME/.cargo/bin:$PATH .venv/bin/maturin develop --release -q && .venv/bin/pytest -q`
Expected: all pass.

- [ ] **Step 10: Commit**

```bash
git add bots/ bindings/ alphasettler/ tests/python/test_selfplay.py tests/python/test_cli.py
git commit -m "selfplay: IsmctsBot self-play records, heuristic weight fit, bindings and CLI"
```

---

### Task 7: IsmctsBot inside Catanatron (event feed)

**Files:**
- Create: `oracle/events.py`, `tests/python/test_oracle_events.py`
- Modify: `oracle/arena.py` (play loop, observe, diagnostics), `alphasettler/cli.py` (`_catanatron_arena` prints belief resets)

**Interfaces:**
- Consumes: `Bot.observe`, `Bot.diagnostics` (Task 6); `oracle.translate.{catan_snapshot, seat, our_edge, our_steps, RESOURCE_NAMES, DEV_NAMES, CAT_DEV, CUBE_TO_OUR_TILE, _res}`; `oracle.tables.CAT_NODE_TO_OUR_NODE`; `oracle.diff.{run_game, config}`.
- Produces: `oracle.events.record_events(before: dict, after: dict, record, state) -> list[dict]`, `oracle.events.redact(event: dict, viewer: int) -> dict`; `AlphaSettlerPlayer.observe(state, events)`; arena records gain `"belief_resets"`.

- [ ] **Step 1: Write the failing tests**

`tests/python/test_oracle_events.py`:

```python
import pytest

pytest.importorskip("catanatron")

from catanatron import Color, Game as CatanGame, RandomPlayer  # noqa: E402

from alphasettler import Game  # noqa: E402
from oracle.arena import run_match  # noqa: E402
from oracle.diff import config, run_game  # noqa: E402
from oracle.events import record_events, redact  # noqa: E402
from oracle.translate import catan_snapshot, our_steps  # noqa: E402

COLORS = [Color.RED, Color.BLUE, Color.ORANGE, Color.WHITE]


def _clean_seeds(n):
    """Seeds whose lockstep game needs no allowlist entry at all, so both logs must agree."""
    out = []
    seed = 0
    while len(out) < n:
        r = run_game(seed)
        if not r.allowlisted and r.mismatch is None:
            out.append(seed)
        seed += 1
    return out


def test_translated_events_equal_our_log_in_clean_games():
    # Classification and replay run in this process, under the same hash seed, so they agree.
    for seed in _clean_seeds(12):
        cat = CatanGame([RandomPlayer(c) for c in COLORS], seed=seed)
        ours = Game.from_snapshot(seed, catan_snapshot(cat), config())
        by_color = {p.color: p for p in cat.state.players}
        translated = []
        before = catan_snapshot(cat)
        while before["phase"] != "game_over":
            st = cat.state
            record = cat.execute(by_color[st.current_color()].decide(cat, cat.playable_actions))
            for a, chance in our_steps(cat.state, record):
                if chance is None:
                    ours.apply(a)
                else:
                    ours.apply_forced(a, chance)
            after = catan_snapshot(cat)
            translated += record_events(before, after, record, cat.state)
            before = after
        for viewer in range(4):
            assert ours.log(viewer) == [redact(e, viewer) for e in translated], f"seed {seed} viewer {viewer}"


def test_redaction():
    stole = {"type": "stole", "thief": 1, "victim": 2, "resource": "ore"}
    assert redact(stole, 1) == stole and redact(stole, 2) == stole
    assert redact(stole, 0)["resource"] is None
    bought = {"type": "bought_dev", "player": 3, "card": "knight"}
    assert redact(bought, 3) == bought and redact(bought, 0)["card"] is None


def test_ismcts_plays_inside_catanatron_without_fallbacks_or_resets():
    records = run_match("ismcts@10", "random", seeds=1, workers=4)
    assert len(records) == 4
    for r in records:
        assert r["fallbacks"] == 0, r
        assert r["belief_resets"] == 0, r
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/bin/pytest -q tests/python/test_oracle_events.py`
Expected: FAIL — `No module named 'oracle.events'`.

(If `_clean_seeds` cannot find 12 clean seeds among the first few hundred, that contradicts the 1,000-game run in `docs/oracle/results.md`, where 747 of 1,000 games were identical end to end. Investigate rather than lower the count.)

- [ ] **Step 3: Implement `oracle/events.py`**

```python
"""Our engine's events for each executed Catanatron action, so an AlphaSettler bot playing inside
Catanatron tracks hidden cards from the same information it would have natively (spec Section 1).
Events are read off the action record and the positions before and after it, so they follow
Catanatron's rules where those differ from ours (for example production when the bank runs short).
The shape is that of `alphasettler.Game.log` entries."""

from __future__ import annotations

from catanatron.models.enums import ActionType

from oracle.tables import CAT_NODE_TO_OUR_NODE
from oracle.translate import (
    CAT_DEV, CUBE_TO_OUR_TILE, DEV_NAMES, RESOURCE_NAMES, Untranslatable, _res, our_edge, seat,
)


def _gain(before: dict, after: dict, p: int) -> list[int]:
    return [a - b for a, b in zip(after["players"][p]["hand"], before["players"][p]["hand"])]


def _played(p: int, card: str) -> dict:
    return {"type": "played_dev", "player": p, "card": card}


def record_events(before: dict, after: dict, record, state) -> list[dict]:
    """Events for one executed action. `before`/`after` are `catan_snapshot`s around it."""
    a = record.action
    t, v, p = a.action_type, a.value, seat(state, a.color)
    ev: list[dict] = []
    if t == ActionType.ROLL:
        ev.append({"type": "rolled", "player": p, "dice": list(record.result)})
        if sum(record.result) != 7:
            for q in range(4):
                g = _gain(before, after, q)
                if any(g):
                    ev.append({"type": "produced", "player": q, "resources": g})
    elif t == ActionType.BUILD_SETTLEMENT:
        ev.append({"type": "built_settlement", "player": p, "node": CAT_NODE_TO_OUR_NODE[v]})
        g = _gain(before, after, p)
        if before["phase"] == "setup_settlement" and any(g):
            ev.append({"type": "produced", "player": p, "resources": g})
    elif t == ActionType.BUILD_ROAD:
        ev.append({"type": "built_road", "player": p, "edge": our_edge(v)})
    elif t == ActionType.BUILD_CITY:
        ev.append({"type": "built_city", "player": p, "node": CAT_NODE_TO_OUR_NODE[v]})
    elif t == ActionType.BUY_DEVELOPMENT_CARD:
        ev.append({"type": "bought_dev", "player": p, "card": DEV_NAMES[CAT_DEV.index(record.result)]})
    elif t == ActionType.PLAY_KNIGHT_CARD:
        ev.append(_played(p, "knight"))
    elif t == ActionType.PLAY_ROAD_BUILDING:
        ev.append(_played(p, "road_building"))
    elif t == ActionType.PLAY_YEAR_OF_PLENTY:
        ev.append(_played(p, "year_of_plenty"))
        ev.append({"type": "year_of_plenty_taken", "player": p, "resources": _gain(before, after, p)})
    elif t == ActionType.PLAY_MONOPOLY:
        r = _res(v)
        ev.append(_played(p, "monopoly"))
        ev.append({"type": "monopoly_taken", "player": p, "resource": RESOURCE_NAMES[r],
                   "amount": _gain(before, after, p)[r]})
    elif t == ActionType.MOVE_ROBBER:
        cube, victim = v[0], v[1]
        ev.append({"type": "robber_moved", "player": p, "tile": CUBE_TO_OUR_TILE[cube]})
        if victim is not None and record.result is not None:
            ev.append({"type": "stole", "thief": p, "victim": seat(state, victim),
                       "resource": RESOURCE_NAMES[_res(record.result)]})
    elif t == ActionType.MARITIME_TRADE:
        gave = [0] * 5
        for r in v[:4]:
            if r is not None:
                gave[_res(r)] += 1
        got = [0] * 5
        got[_res(v[4])] = 1
        ev.append({"type": "maritime_traded", "player": p, "gave": gave, "got": got})
    elif t == ActionType.DISCARD_RESOURCE:
        ev.append({"type": "discarded", "player": p, "resource": RESOURCE_NAMES[_res(v)]})
    elif t == ActionType.END_TURN:
        ev.append({"type": "turn_ended", "player": p})
    else:
        raise Untranslatable("domestic-trade", f"{t} has no event translation")
    if after["phase"] == "game_over" and before["phase"] != "game_over":
        ev.append({"type": "game_over", "winner": after["winner"]})
    return ev


def redact(event: dict, viewer: int) -> dict:
    """`event` as `viewer` may see it (the engine's `Event::redacted_for`)."""
    if event["type"] == "stole" and viewer not in (event["thief"], event["victim"]):
        return {**event, "resource": None}
    if event["type"] == "bought_dev" and viewer != event["player"]:
        return {**event, "card": None}
    return event
```

- [ ] **Step 4: Feed the candidate in `oracle/arena.py`**

1. Imports: add `import catanatron.game as catan_game` and `from oracle.events import record_events, redact`, and `catan_snapshot` to the `oracle.translate` import list.

2. Add to `AlphaSettlerPlayer`:

```python
    def observe(self, state, events: list[dict]) -> None:
        """Show the bot the events of one executed action, redacted for its seat."""
        viewer = seat(state, self.color)
        self.bot.observe(viewer, [redact(e, viewer) for e in events])
```

3. In `play_game`, replace `game.play()` with an explicit loop. First read Catanatron's own `Game.play` (`.venv/lib/python3.12/site-packages/catanatron/game.py`) and keep its loop condition exactly, so games with bots that ignore events (greedy) are unchanged:

```python
    me = players[candidate_seat]
    before = catan_snapshot(game)
    while game.winning_color() is None and game.state.num_turns < catan_game.TURNS_LIMIT:
        game.play_tick()
        after = catan_snapshot(game)
        me.observe(game.state, record_events(before, after, game.state.action_records[-1], game.state))
        before = after
```

4. Add to the returned record: `"belief_resets": me.bot.diagnostics().get("belief_resets", 0),`.

5. In `alphasettler/cli.py` `_catanatron_arena`, after the `fallbacks:` line add:

```python
    print(f"belief resets: {sum(r['belief_resets'] for r in records)}")
```

and in `tests/python/test_oracle_arena.py`, `test_records_have_the_native_shape`, add `"belief_resets"` to the set of expected keys if that test lists the keys exactly.

- [ ] **Step 5: Run everything**

Run: `.venv/bin/pytest -q && ~/.cargo/bin/cargo test --workspace --release`
Expected: all pass.

- [ ] **Step 6: Check greedy results are unchanged by the explicit loop**

Run: `.venv/bin/alphasettler arena --candidate greedy --baseline catanatron:random --seeds 200 --out runs/check-greedy-vs-catanatron-random.jsonl` (delete the file afterwards; `runs/` is gitignored)
Expected: `win rate 0.996 [0.992, 1.000]`, `fallbacks: 0`, `belief resets: 0` (identical to `docs/oracle/results.md`).

- [ ] **Step 7: Commit**

```bash
git add oracle/ alphasettler/cli.py tests/python/test_oracle_events.py tests/python/test_oracle_arena.py
git commit -m "oracle: translate Catanatron actions into events for bots playing inside Catanatron"
```

---

### Task 8: Measurements, tuning rounds and the results record

**Files:**
- Create: `bots/examples/search_speed.rs`, `docs/perf/search.md`, `docs/bot/results-3a.md`
- Modify (only during tuning rounds): `bots/src/heuristic.rs` (`DEFAULT_WEIGHTS`)

**Interfaces:**
- Consumes: every CLI from Tasks 6–7; `IsmctsBot`, `GreedyBot`, `play_game`.

- [ ] **Step 1: Write the speed example**

`bots/examples/search_speed.rs`:

```rust
//! IsmctsBot search throughput on one core: IsmctsBot in seat 0 against three GreedyBots, timing
//! only decisions that searched. Usage: search_speed [decisions] [simulations]

use settler_bots::arena::play_game;
use settler_bots::greedy::GreedyBot;
use settler_bots::ismcts::IsmctsBot;
use settler_bots::Bot;
use settler_engine::{Action, Event, GameConfig, Observation, PlayerId};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

#[derive(Default)]
struct Totals {
    decisions: u64,
    simulations: u64,
    nodes: u64,
    time: Duration,
}

struct Timed {
    inner: IsmctsBot,
    totals: Arc<Mutex<Totals>>,
}

impl Bot for Timed {
    fn name(&self) -> &'static str {
        "timed-ismcts"
    }

    fn observe(&mut self, viewer: PlayerId, events: &[Event]) {
        self.inner.observe(viewer, events);
    }

    fn act(&mut self, obs: &Observation, legal: &[Action]) -> Action {
        let start = Instant::now();
        let a = self.inner.act(obs, legal);
        let elapsed = start.elapsed();
        if let Some(r) = self.inner.take_last_search() {
            let mut t = self.totals.lock().unwrap();
            t.decisions += 1;
            t.simulations += r.simulations as u64;
            t.nodes += r.nodes as u64;
            t.time += elapsed;
        }
        a
    }
}

fn main() {
    let args: Vec<u64> = std::env::args().skip(1).map(|a| a.parse().expect("a number")).collect();
    let decisions = args.first().copied().unwrap_or(2000);
    let simulations = args.get(1).copied().unwrap_or(1000) as u32;
    let config = GameConfig { max_offers_per_turn: 0, ..GameConfig::default() };
    let totals = Arc::new(Mutex::new(Totals::default()));
    let mut seed = 0;
    while totals.lock().unwrap().decisions < decisions {
        let mut bots: Vec<Box<dyn Bot>> = vec![Box::new(Timed { inner: IsmctsBot::new(seed, simulations, 0), totals: totals.clone() })];
        bots.extend((1..4).map(|p| Box::new(GreedyBot::new(p)) as Box<dyn Bot>));
        play_game(seed, config, &mut bots);
        seed += 1;
    }
    let t = totals.lock().unwrap();
    let secs = t.time.as_secs_f64();
    println!(
        "{} searched decisions over {seed} games at {simulations} simulations: {:.0} simulations/s, {:.2} ms/decision, {:.0} nodes/decision",
        t.decisions,
        t.simulations as f64 / secs,
        1000.0 * secs / t.decisions as f64,
        t.nodes as f64 / t.decisions as f64
    );
}
```

Run: `~/.cargo/bin/cargo run --release -p settler-bots --example search_speed 2000 1000`
Expected: one line with simulations/s. Run it 3 times on an otherwise idle machine and record all three lines in `docs/perf/search.md` (machine: `sysctl -n machdep.cpu.brand_string`; commit: `git log -1 --format=%h`; config: trades off, 1,024 particles, batch 8, c_puct 1.5). Compare with the spec's rough estimate (50k–100k simulations/s) in one sentence; a miss is recorded with its measured profile reason, not hidden.

- [ ] **Step 2: Round 0 — the bot as built**

Run each command; keep the stdout:

```bash
for n in 100 300 1000 3000; do .venv/bin/alphasettler arena --candidate ismcts@$n --baseline greedy --seeds 200 --no-trades; done
.venv/bin/alphasettler arena --candidate ismcts --baseline catanatron:value --seeds 200
.venv/bin/alphasettler arena --candidate ismcts --baseline catanatron:alphabeta --seeds 200
```

Every Catanatron run must print `fallbacks: 0` and `belief resets: 0`; anything else is a bug to fix before going on. The headline is met when both Catanatron runs show z > 3. If it is met, skip to Step 6.

- [ ] **Step 3: Round 1 — fit the weights**

```bash
.venv/bin/alphasettler selfplay --games 400 --simulations 300
.venv/bin/alphasettler fit-heuristic runs/*-selfplay-s300.jsonl.gz
```

Paste the printed `pub const DEFAULT_WEIGHTS ...` line into `bots/src/heuristic.rs`. Rebuild (`maturin develop --release -q`), run `cargo test --workspace --release` and `pytest -q` (all must pass), then rerun the ismcts@300-vs-greedy line and both Catanatron runs from Step 2. Stop if the headline is met.

- [ ] **Step 4: Round 2 — the rollout switch**

```bash
for d in 8 32; do .venv/bin/alphasettler arena --candidate "ismcts@300+r$d" --baseline greedy --seeds 200 --no-trades; done
```

Compare with Round 1's `ismcts@300` line using `alphasettler compare --a ismcts@300+r<best> --b ismcts@300 --baseline greedy --seeds 200 --no-trades` (paired). If a rollout depth is better with paired z > 2, rerun both Catanatron runs with `ismcts@1000+r<that depth>`; otherwise keep rollout 0 and record that. Stop if the headline is met.

- [ ] **Step 5: Round 3 — refit on the improved bot**

Self-play 400 games with the current best name's settings (`--simulations 300`, plus `--rollout D` if Round 2 chose one), refit starting from the current weights (the fit starts from `DEFAULT_WEIGHTS`, which now holds Round 1's fit), paste, rebuild, retest, and rerun both Catanatron runs. This is the last tuning round. The spec allows at most 3.

- [ ] **Step 6: Write `docs/bot/results-3a.md`**

Record, with numbers copied from the runs (no placeholders):
- date, machine, `git log -1 --format=%h` (or "uncommitted working tree on top of <last commit>" if commits are queued), Catanatron 3.3.0 at ecf931181b9a65bb4116a2153fb78c16f1438e00;
- search speed (link `docs/perf/search.md`);
- for each round run: what changed (weights as the literal array, rollout depth), and every arena summary line with wall time, fallbacks and belief resets;
- the strength curve table (100/300/1,000/3,000 vs greedy) from the final configuration;
- the headline result against `catanatron:value` and `catanatron:alphabeta`, and whether it is met. If it is not met after round 3: the gap (win rate and z for each) and the suspected cause, read from the runs (for example, which phases' decisions differ most from AlphaBeta's);
- a checklist of the spec's Section 5 done criteria, each with its evidence (test names, commands, this file's numbers).

- [ ] **Step 7: Final checks and commit**

Run: `~/.cargo/bin/cargo test --workspace --release && PATH=$HOME/.cargo/bin:$PATH .venv/bin/maturin develop --release -q && .venv/bin/pytest -q`
Expected: all pass. Then:

```bash
git add bots/examples/search_speed.rs bots/src/heuristic.rs docs/perf/search.md docs/bot/results-3a.md
git commit -m "docs: 3a search speed, tuning rounds and results against Catanatron"
```

---

## Spec coverage (self-review)

| Spec requirement | Task |
|---|---|
| `search/` crate depending on engine only; bots → search → engine | 2, 5 |
| `Bot::observe` / `diagnostics`; arena feeds redacted events before every `act` | 1 |
| Belief: particles over hidden steals with the exact prior; spends prune; rebuild policy; `hand_counts` invariant | 2 |
| Dev-card deal and deck order uniform over the unseen pool; current opponent never dealt a win | 3 |
| Fast sampled worlds, validated template, `reseed`, equal to `from_snapshot`, observation-identical | 3 |
| Single-observer ISMCTS, children keyed by action + visible outcome, availability counts | 4 |
| 4-player values, max^n, ¼ on the turn cap | 4 |
| PUCT with evaluator prior and value; leaf batching with virtual loss; determinism | 4 |
| Simulation budget; no search for a single move; most-visited move | 4 |
| No trades: never offers, declines offers, search ignores offers | 4, 5 |
| Evaluator contract; prior reads only the actor's observation (property test) | 4, 5 |
| Heuristic value features and softmax; GreedyBot prior; rollout switch | 5 |
| Bot names `ismcts`, `ismcts@N`, `ismcts@N+rD` | 5, 6 |
| Self-play records (per game, gzipped JSONL) and exact replay test | 6 |
| Weight fit by maximum likelihood (Newton, in Rust) via `fit-heuristic` | 6 |
| Catanatron event feed, equal to our log in clean lockstep games; 0 fallbacks, 0 belief resets | 7 |
| Tactical tests (winning build, city over end turn, robber off own tiles) | 4, 5 |
| Search throughput recorded in `docs/perf/` | 8 |
| Strength curve vs greedy; headline vs AlphaBeta and value; ≤ 3 tuning rounds; `docs/bot/results-3a.md` | 8 |
