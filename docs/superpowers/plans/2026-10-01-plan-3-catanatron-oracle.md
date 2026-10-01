# Plan 3: Catanatron Oracle — Differential Testing and the Catanatron Arena — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Check our engine against Catanatron and benchmark against it. A differential test plays Catanatron random-bot games and compares Catanatron's legal actions and resulting position with ours at every step. The Catanatron arena plays our Rust bots inside Catanatron against its RandomPlayer, ValueFunction and AlphaBeta bots, rotating seats as the native arena does.

**Architecture:** Five pieces.
- **Engine rule fixes:** two fixes found by research. Every dev card may be played before rolling, and Road Building needs a placeable road.
- **Position snapshots:** a plain-data snapshot of a full position, exported and imported by the engine (`State::snapshot` / `State::from_snapshot`, validated by `State::check_invariants`) and exposed in Python as dicts.
- **`oracle/`:** a top-level Python package, the only code that imports Catanatron. It translates boards, actions, legal sets and full positions between the engines.
- **Differential runner:** steps Catanatron and our engine in lockstep. After an intended (allowlisted) rule difference, it re-imports Catanatron's position and continues, so one difference doesn't end the comparison.
- **Catanatron arena:** imports Catanatron's position into our engine at each of our bot's decisions and maps the bot's choice back.

**Tech Stack:** Rust (engine, PyO3 `=0.29.3` bindings), Python 3.12 (`python3`/`pip3` only), Catanatron 3.3.0 pinned to git commit `ecf931181b9a65bb4116a2153fb78c16f1438e00` (GPL-3.0, runtime import only), pytest `9.1.1`.

**Spec:** `docs/superpowers/specs/2026-09-30-engine-and-harness-design.md` (Section 1: `oracle/` is a separate package and the only code importing Catanatron; Section 3: differential testing and the Catanatron arena; Section 4: done criteria).

**Research:** `.superpowers/sdd/catanatron-reference.md` (gitignored scratch, 2026-10-01). It holds the verified Catanatron internals this plan relies on: coordinate and port mapping tables, the `ActionRecord` forcing mechanism, the `PYTHONHASHSEED` dependence, the rule-difference table, and a 1000-game prototype replay (757 games identical end to end, every divergence explained). **Implementers must read the sections of it their task names.**

## Rulings this plan encodes

These were decided while writing the plan (the spec says "standard rules"; the rulebook is the authority):

1. **Dev cards before rolling.** The rulebook allows playing a development card at any time on your turn, including before rolling. Our engine allowed only the Knight. Fixed in Task 1. Catanatron agrees, so this removes the largest divergence class in the prototype replay (186 of 200 games).
2. **Road Building needs a placeable road.** We offered the card when no road could be placed, which spends it for nothing. It now needs a road piece left and a legal edge. Fixed in Task 1; Catanatron agrees.
3. **We keep the rulebook where Catanatron differs.** Each difference below is a documented allowlist entry:
   - **longest road:** a trail may end at an opponent's building, and the award needs 5 roads;
   - **bank shortage:** a single claimant gets what's left;
   - **Year of Plenty:** two cards only;
   - **building through an opponent's settlement:** forbidden;
   - **winning:** only on your own turn.
4. **`catanatron_compat`** models pre-3.0 Catanatron, which used random discards. Catanatron 3.3 lets players choose, exactly like our default. The flag stays, with its docs corrected, and the oracle runs with it **off**.
5. **Lockstep instead of record-then-replay.** Recording a game and replaying it with forced chance outcomes happens step by step in one process (Catanatron's realized outcomes come from its `ActionRecord`s). This is equivalent to the spec's record/replay and needs no file format. A failing game's last steps are kept in the result for debugging.
6. **Resync after an allowlisted divergence.** When the positions differ for an allowlisted reason, our engine re-imports Catanatron's position (`Game.from_snapshot`) and the comparison continues. That keeps step coverage near 100% instead of stopping about 24% of games early.

## Global Constraints

- Always `python3` / `pip3`, never `python` / `pip`. The venv is `.venv/`. Rebuild the extension with `source "$HOME/.cargo/env" && VIRTUAL_ENV=$PWD/.venv .venv/bin/maturin develop --release`.
- **Catanatron isolation:**
  - Catanatron is imported only inside `oracle/` (the spec's `oracle/` package). `oracle/__init__.py` itself must not import Catanatron, so callers can read its constants when Catanatron is missing.
  - Never copy, port or vendor Catanatron source into this repo (GPL-3.0).
  - Install Catanatron only as the optional `oracle` extra, pinned to the git commit above. Installing `alphasettler` must never pull it in.
- **Hash seed:** Catanatron's move order depends on `PYTHONHASHSEED`. Every multi-game run executes in `spawn` worker processes with `PYTHONHASHSEED=0` in their environment, so results are reproducible.
- **Golden trace:** Task 1 changes the rules, so it re-pins the three golden-trace constants in `engine/tests/golden_trace.rs` and states why. No later task may change them.
- **MSRV:** the engine keeps `rust-version = "1.80"` and zero runtime dependencies.
- **Parity config:** our engine runs with `{"max_offers_per_turn": 0, "catanatron_compat": False, "max_turns": TURNS_LIMIT - 6}`. Catanatron's turn counter includes 6 setup advances (`catanatron.game.TURNS_LIMIT` is 1000).
- **Seats:** Catanatron seat `i` (index in `state.colors`) is our player `i`.
- **Per-task workflow:** after each task, run its tests, the full `cargo test`, and `.venv/bin/python3 -m pytest tests/python -q`. Then secret-scan and commit with the `sean.may101@gmail.com` identity, no AI attribution, never `--no-verify`.
  - **While the user is away, commits are queued and not run:** no git writes at all.

## Review Focus

1. **A malformed or inconsistent snapshot** (wrong list lengths, ids out of range, overlapping buildings, resource or dev-card totals that don't add up, a pending trade, unknown keys) raises a `ValueError` naming the problem. It never panics across FFI or builds an impossible state. Tests: Task 2 `rejects_inconsistent_snapshots`, Task 3 `test_bad_snapshots_raise_value_error`.
2. **Catanatron not installed:** `oracle` tests skip, and `alphasettler oracle-diff` / `arena --baseline catanatron:*` print an install hint and exit 2, with no ImportError traceback. Tests: Task 5 `test_cli_without_catanatron_explains_how_to_install`, Task 6 (same pattern).
3. **Hash-seed nondeterminism:** two runs of the same seeds give identical per-game results, whatever the worker count. Test: Task 5 `test_runs_are_reproducible_across_worker_counts`.
4. **Our bot picks something Catanatron doesn't offer, or a position won't import:** the arena falls back to a legal Catanatron action and counts it, never crashing. Tests: Task 6 `test_fallback_is_counted_not_fatal`, `test_greedy_beats_catanatron_random` (asserts zero fallbacks in normal play).
5. **Turn-limit draws:** both engines end in a draw at the same turn, and the comparison passes. Test: Task 5 `test_turn_limit_draws_match`.

---

## File Structure

```
engine/src/legal.rs              PreRoll offers every playable dev card
engine/src/rules/dev.rs          legal_dev_plays; Road Building playability; return_phase
engine/src/apply.rs              PreRoll dispatch for RB / YoP / Monopoly
engine/src/state.rs              robber_return -> return_phase
engine/src/observation.rs        robber_return -> return_phase
engine/src/rules/robber.rs       return_phase
engine/src/rules/roll.rs         return_phase
engine/src/config.rs             catanatron_compat doc
engine/src/board.rs              Board::validate
engine/src/invariants.rs         State::check_invariants
engine/src/snapshot.rs           Snapshot, PlayerSnapshot, State::{snapshot, from_snapshot}
engine/src/game.rs               Game::from_snapshot
engine/src/lib.rs                modules + re-exports
engine/tests/dev.rs              pre-roll dev card tests
engine/tests/game.rs             renamed return_phase test
engine/tests/snapshot.rs         round trip + rejection tests
engine/tests/properties.rs       uses State::check_invariants
engine/tests/golden_trace.rs     re-pinned constants (Task 1 only)
bindings/src/convert.rs          return_phase key; snapshot dict <-> Snapshot
bindings/src/lib.rs              Game.from_snapshot / snapshot / copy; Bot class
alphasettler/__init__.py         export Bot
alphasettler/cli.py              oracle-diff subcommand; catanatron:* arena baselines
pyproject.toml                   oracle extra; maturin python-packages = ["oracle"]
oracle/__init__.py               requirement constant only (no Catanatron import)
oracle/tables.py                 verified id mapping tables
oracle/translate.py              board/action/legal/position translation + compare
oracle/allowlist.py              documented intended differences + classifiers
oracle/diff.py                   lockstep differential runner
oracle/arena.py                  our bots inside Catanatron
tests/python/test_snapshot.py
tests/python/test_oracle_translate.py
tests/python/test_oracle_diff.py
tests/python/test_oracle_arena.py
tests/python/test_oracle_isolation.py
docs/oracle/results.md           recorded results of the 1k / 50k runs and arena runs (Task 7)
```

---

### Task 1: Engine — dev cards before rolling; Road Building needs a placeable road

**Files:**
- Modify: `engine/src/legal.rs`, `engine/src/rules/dev.rs`, `engine/src/apply.rs`, `engine/src/state.rs`, `engine/src/observation.rs`, `engine/src/rules/robber.rs`, `engine/src/rules/roll.rs`, `engine/src/config.rs`, `bindings/src/convert.rs`
- Test: `engine/tests/dev.rs`, `engine/tests/game.rs`, `engine/tests/golden_trace.rs`

**Interfaces:**
- Produces: `State.return_phase: Phase` (renamed from `robber_return`; the phase resumed after the robber is resolved **or Road Building ends**), `Observation.return_phase`, the Python observation key `"return_phase"` (was `"robber_return"`), and `settler_engine::rules::dev::legal_dev_plays(s: &State, out: &mut Vec<Action>)`.
- PreRoll legal actions become: `Roll`, then the same dev plays Main offers, in the same order (knight, road building, year of plenty pairs, monopoly). `BuyDev` stays Main-only.
- `PlayRoadBuilding` is legal only when the player has a road piece left (`roads.count_ones() < MAX_ROADS`) **and** `build::has_free_road(s, p)`.

- [ ] **Step 1: Write the failing tests**

Add to `engine/tests/common/mod.rs`:

```rust
/// Give player `p` a dev card by drawing it from the deck (keeps the deck consistent).
pub fn deal_dev(s: &mut State, p: usize, card: DevCard) {
    let pos = s.dev_deck_pos as usize;
    let j = (pos..DEV_DECK_SIZE)
        .find(|&j| s.dev_deck[j] == card)
        .expect("card left in the deck");
    s.dev_deck.swap(pos, j);
    s.dev_deck_pos += 1;
    s.players[p].dev_hand[card.index()] += 1;
}
```

Append to `engine/tests/dev.rs` (add `use common::deal_dev;` / adjust imports to the file's existing style):

```rust
#[test]
fn pre_roll_offers_every_held_dev_card() {
    let mut s = after_setup(1);
    assert_eq!(s.phase, Phase::PreRoll);
    let p = s.current as usize;
    for c in [DevCard::Knight, DevCard::RoadBuilding, DevCard::YearOfPlenty, DevCard::Monopoly] {
        deal_dev(&mut s, p, c);
    }
    let legal = s.legal_actions();
    assert_eq!(legal[0], Action::Roll);
    for a in [
        Action::PlayKnight,
        Action::PlayRoadBuilding,
        Action::PlayYearOfPlenty(Resource::Wood, Resource::Ore),
        Action::PlayMonopoly(Resource::Sheep),
    ] {
        assert!(legal.contains(&a), "{a:?} missing from {legal:?}");
    }
    assert!(!legal.contains(&Action::BuyDev));
}

#[test]
fn year_of_plenty_before_rolling_stays_pre_roll_and_uses_the_turns_card() {
    let mut s = after_setup(2);
    let p = s.current as usize;
    deal_dev(&mut s, p, DevCard::YearOfPlenty);
    deal_dev(&mut s, p, DevCard::Monopoly);
    let before = s.players[p].hand;
    s.apply(Action::PlayYearOfPlenty(Resource::Wood, Resource::Brick));
    assert_eq!(s.phase, Phase::PreRoll);
    assert_eq!(s.players[p].hand[0], before[0] + 1);
    assert_eq!(s.players[p].hand[1], before[1] + 1);
    assert_eq!(s.legal_actions(), vec![Action::Roll]);
}

#[test]
fn monopoly_before_rolling() {
    let mut s = after_setup(3);
    let p = s.current as usize;
    let q = (p + 1) % NUM_PLAYERS;
    deal_dev(&mut s, p, DevCard::Monopoly);
    give(&mut s, q, [0, 0, 3, 0, 0]);
    let mine = s.players[p].hand[2];
    s.apply(Action::PlayMonopoly(Resource::Sheep));
    assert_eq!(s.phase, Phase::PreRoll);
    assert_eq!(s.players[q].hand[2], 0);
    assert!(s.players[p].hand[2] >= mine + 3);
}

#[test]
fn road_building_before_rolling_returns_to_pre_roll() {
    let mut s = after_setup(4);
    let p = s.current as usize;
    deal_dev(&mut s, p, DevCard::RoadBuilding);
    let roads = s.players[p].roads.count_ones();
    s.apply(Action::PlayRoadBuilding);
    assert_eq!(s.phase, Phase::RoadBuilding { roads_left: 2 });
    for _ in 0..2 {
        let a = s.legal_actions()[0];
        assert!(matches!(a, Action::BuildRoad(_)));
        s.apply(a);
    }
    assert_eq!(s.players[p].roads.count_ones(), roads + 2);
    assert_eq!(s.phase, Phase::PreRoll);
    assert_eq!(s.legal_actions(), vec![Action::Roll]);
}

#[test]
fn road_building_needs_a_placeable_road() {
    // No roads or buildings: nowhere to place a free road.
    let mut s = blank(1, Phase::Main);
    deal_dev(&mut s, 0, DevCard::RoadBuilding);
    assert!(!s.legal_actions().contains(&Action::PlayRoadBuilding));
}

#[test]
fn road_building_needs_a_road_piece() {
    let mut s = blank(1, Phase::Main);
    let nodes = path_nodes(0, 16, 0);
    s.players[0].roads = mask128(&path_edges(&nodes));
    assert_eq!(s.players[0].roads.count_ones(), MAX_ROADS);
    deal_dev(&mut s, 0, DevCard::RoadBuilding);
    assert!(!s.legal_actions().contains(&Action::PlayRoadBuilding));
}
```

If `path_nodes(0, 16, 0)` cannot find a 16-node simple path from node 0, use any start node it succeeds from. The assertion that 15 roads were placed is what matters.

In `engine/tests/game.rs`, rename `observation_shows_robber_return_after_knights` to `observation_shows_return_phase_after_knights` and `robber_return` to `return_phase` inside it.

- [ ] **Step 2: Run tests to verify they fail**

Run: `source "$HOME/.cargo/env" && cargo test -p settler-engine --test dev`
Expected: FAIL (compile error on `return_phase` in game.rs is fine to see later; in dev.rs the four pre-roll / road-building tests fail their assertions, e.g. `PlayRoadBuilding missing from [Roll, PlayKnight]`).

- [ ] **Step 3: Implement**

`engine/src/state.rs`: rename the field `robber_return` to `return_phase`. Its doc becomes:

```rust
    /// Phase resumed after the robber is resolved or Road Building ends: PreRoll when the card
    /// was played before rolling, otherwise Main.
    pub return_phase: Phase,
```

Make the same rename in `engine/src/observation.rs`, keeping the doc wording, and at every use in `engine/src/rules/robber.rs` and `engine/src/rules/roll.rs`. Use `grep -rn robber_return engine bindings` to find them all.

`engine/src/legal.rs`, PreRoll arm:

```rust
        Phase::PreRoll => {
            out.push(Action::Roll);
            dev::legal_dev_plays(s, out);
        }
```

`engine/src/rules/dev.rs`: replace `legal_knight` and `legal_main_dev` with:

```rust
pub fn legal_main_dev(s: &State, out: &mut Vec<Action>) {
    let pl = &s.players[s.current as usize];
    if covers(&pl.hand, &DEV_COST) && (s.dev_deck_pos as usize) < DEV_DECK_SIZE {
        out.push(Action::BuyDev);
    }
    legal_dev_plays(s, out);
}

/// Dev cards the current player may play now, before or after rolling: one per turn, never a
/// card bought this turn, and Road Building only when a free road can be placed.
pub fn legal_dev_plays(s: &State, out: &mut Vec<Action>) {
    if playable(s, DevCard::Knight) {
        out.push(Action::PlayKnight);
    }
    if playable(s, DevCard::RoadBuilding) && can_place_free_road(s) {
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

fn can_place_free_road(s: &State) -> bool {
    let p = s.current as usize;
    s.players[p].roads.count_ones() < MAX_ROADS && build::has_free_road(s, p)
}
```

In the same file, the knight sets `s.return_phase` (renamed), and Road Building returns to it:

```rust
fn road_building_phase(s: &State, left: u8) -> Phase {
    if left > 0 && build::has_free_road(s, s.current as usize) {
        Phase::RoadBuilding { roads_left: left }
    } else {
        s.return_phase
    }
}

pub fn apply_play_road_building<S: EventSink>(s: &mut State, sink: &mut S) {
    s.return_phase = if s.phase == Phase::PreRoll {
        Phase::PreRoll
    } else {
        Phase::Main
    };
    use_card(s, DevCard::RoadBuilding, sink);
    let pieces_left = MAX_ROADS - s.players[s.current as usize].roads.count_ones();
    s.phase = road_building_phase(s, pieces_left.min(2) as u8);
}
```

`engine/src/apply.rs`: widen the three dispatch arms so they also accept PreRoll:

```rust
            (Phase::PreRoll | Phase::Main, Action::PlayRoadBuilding) => {
                dev::apply_play_road_building(self, sink)
            }
            (Phase::PreRoll | Phase::Main, Action::PlayYearOfPlenty(a, b)) => {
                dev::apply_year_of_plenty(self, a, b, sink)
            }
            (Phase::PreRoll | Phase::Main, Action::PlayMonopoly(r)) => dev::apply_monopoly(self, r, sink),
```

Keep each arm's existing body. The match only changes its phase pattern.

`engine/src/config.rs`: replace the doc on `catanatron_compat`:

```rust
    /// Discards on a 7 are random instead of chosen, as Catanatron did before version 3.
    /// Catanatron 3.3 lets each player choose, like the default; leave this off against it.
```

`bindings/src/convert.rs`: in `observation_dict`, write `d.set_item("return_phase", phase_name(o.return_phase))?;` instead of the `robber_return` line.

Existing tests that encode the old rules will now fail and must be updated to the new rule; list each one in the report:
- PreRoll legal sets equal to `[Roll, PlayKnight]`.
- `road_building_with_no_legal_road_returns_to_main` (`engine/tests/dev.rs:127`) should now assert that `PlayRoadBuilding` is not legal there.
- Rename it to `road_building_is_not_offered_without_a_legal_road`.

- [ ] **Step 4: Re-pin the golden trace**

The legal-action lists change (PreRoll now lists more dev plays, and Road Building is offered less often), so all three hashes in `engine/tests/golden_trace.rs` change. Run `cargo test --release -p settler-engine --test golden_trace`. Copy each actual value it reports into `TRADES_OFF`, `TRADES_ON` and `COMPAT`. If the test doesn't print the actual value, add it to the assertion message.

Add one line above the constants:

```rust
// Re-pinned 2026-10-01 for two rule fixes: every dev card may be played before rolling, and
// Road Building needs a placeable road (Plan 3, Task 1).
```

- [ ] **Step 5: Run everything**

```bash
source "$HOME/.cargo/env"
cargo test
cargo test --release -p settler-bots --test arena
cargo clippy --workspace --all-targets -- -D warnings
VIRTUAL_ENV=$PWD/.venv .venv/bin/maturin develop --release
.venv/bin/python3 -m pytest tests/python -q
```

Expected: everything passes. That includes the property tests (every legal action applies, invariants hold) and the arena's greedy-beats-random check.

- [ ] **Step 6: Commit (queued while the user is away)**

```bash
git add engine bindings/src/convert.rs
git commit -m "engine: dev cards playable before rolling; Road Building needs a placeable road"
```

---

### Task 2: Engine — board validation, state invariants, snapshots

**Files:**
- Create: `engine/src/invariants.rs`, `engine/src/snapshot.rs`, `engine/tests/snapshot.rs`
- Modify: `engine/src/board.rs`, `engine/src/game.rs`, `engine/src/lib.rs`, `engine/tests/properties.rs`

**Interfaces:**
- Produces:
  - `Board::validate(&self) -> Result<(), String>`
  - `State::check_invariants(&self) -> Result<(), String>`
  - `settler_engine::{Snapshot, PlayerSnapshot}`
  - `State::snapshot(&self) -> Snapshot`
  - `State::from_snapshot(seed: u64, config: GameConfig, snap: &Snapshot) -> Result<State, String>`
  - `Game::from_snapshot(seed: u64, config: GameConfig, snap: &Snapshot) -> Result<Game, String>`
- `Snapshot.dev_deck` lists the cards left in the deck, next draw first.
- Pending domestic trades are not representable: `from_snapshot` rejects the `TradeResponse` and `TradeConfirm` phases.
- `check_invariants` checks structure only (conservation, pieces, deck, phase consistency, cached road lengths). It does **not** check award thresholds, because an imported position may follow another engine's award rules. The property test keeps its own award checks.

- [ ] **Step 1: Write the failing tests**

`engine/tests/snapshot.rs`:

```rust
mod common;

use settler_engine::rng::Rng;
use settler_engine::*;

/// Positions from random games without domestic trades, sampled every 23 actions plus the end.
fn sample_states() -> Vec<State> {
    let cfg = GameConfig {
        max_offers_per_turn: 0,
        ..GameConfig::default()
    };
    let mut out = Vec::new();
    for seed in 0..30 {
        let mut s = State::new(seed, cfg);
        let mut rng = Rng::new(seed ^ 0x5A);
        let mut step = 0;
        while !s.is_over() {
            if step % 23 == 0 {
                out.push(s);
            }
            let legal = s.legal_actions();
            s.apply(legal[rng.below(legal.len() as u32) as usize]);
            step += 1;
        }
        out.push(s);
    }
    out
}

#[test]
fn sampled_states_satisfy_invariants() {
    for s in sample_states() {
        s.check_invariants().unwrap_or_else(|e| panic!("seed {}: {e}", s.seed));
    }
}

#[test]
fn snapshot_round_trips() {
    for s in sample_states() {
        let snap = s.snapshot();
        let t = State::from_snapshot(s.seed, s.config, &snap)
            .unwrap_or_else(|e| panic!("seed {}: {e}", s.seed));
        assert_eq!(t.snapshot(), snap);
        assert_eq!(t.legal_actions(), s.legal_actions());
        assert_eq!(t.observation(1), s.observation(1));
    }
}

#[test]
fn game_from_snapshot() {
    let s = common::after_setup(5);
    let g = Game::from_snapshot(5, s.config, &s.snapshot()).unwrap();
    assert_eq!(g.legal_actions(), s.legal_actions());
}

#[test]
fn rejects_inconsistent_snapshots() {
    let base = common::after_setup(3).snapshot();
    let cfg = GameConfig::default();
    let expect_err = |change: &dyn Fn(&mut Snapshot), needle: &str| {
        let mut snap = base.clone();
        change(&mut snap);
        let e = State::from_snapshot(3, cfg, &snap).expect_err(needle);
        assert!(e.contains(needle), "expected {needle:?} in {e:?}");
    };
    expect_err(&|s| s.bank[0] += 1, "resource 0");
    expect_err(&|s| s.players[1].settlements |= s.players[0].settlements, "share a node");
    expect_err(&|s| s.dev_deck.iter_mut().for_each(|c| *c = DevCard::Monopoly), "Monopoly");
    expect_err(&|s| s.dev_deck.push(DevCard::Knight), "dev_deck");
    expect_err(
        &|s| {
            let d = s.board.desert() as usize;
            s.board.tile_resource[d] = Some(Resource::Wood);
        },
        "desert",
    );
    expect_err(&|s| s.phase = Phase::TradeResponse, "pending domestic trade");
    expect_err(&|s| s.players[2].discard_remaining = 3, "Discard");
    expect_err(&|s| s.players[0].roads = u128::MAX, "roads");
    expect_err(&|s| s.players[0].settlements = u64::MAX, "settlements");
    expect_err(&|s| s.robber = 19, "robber");
    expect_err(&|s| s.current = 4, "current");
    expect_err(&|s| s.return_phase = Phase::Discard, "return_phase");
}

#[test]
fn board_validation() {
    let b = State::new(1, GameConfig::default()).board;
    assert!(b.validate().is_ok());
    let mut bad = b;
    bad.tile_number[(b.desert() as usize + 1) % 19] = 7;
    assert!(bad.validate().unwrap_err().contains("number 7"));
    let mut bad = b;
    bad.ports[1].0 = bad.ports[0].0;
    assert!(bad.validate().unwrap_err().contains("share edge"));
    let mut bad = b;
    bad.generic_port_nodes ^= 1;
    assert!(bad.validate().unwrap_err().contains("port node masks"));
}
```

In `engine/tests/properties.rs`, replace the body of `check_invariants`:

```rust
fn check_invariants(s: &State) {
    s.check_invariants().unwrap_or_else(|e| panic!("{e}"));
    check_vp(s);
}
```

Delete the now-unused imports (`longest_road` and `topo` if nothing else uses them).

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p settler-engine --test snapshot`
Expected: FAIL to compile: `no method named check_invariants` / `cannot find type Snapshot`.

- [ ] **Step 3: Implement**

`engine/src/board.rs`, inside `impl Board`:

```rust
    /// Checks a board built outside `Board::random`, e.g. imported from another engine.
    pub fn validate(&self) -> Result<(), String> {
        let deserts = self.tile_resource.iter().filter(|r| r.is_none()).count();
        if deserts != 1 {
            return Err(format!("board must have exactly one desert, found {deserts}"));
        }
        for t in 0..NUM_TILES {
            let n = self.tile_number[t];
            match self.tile_resource[t] {
                None if n != 0 => return Err(format!("desert tile {t} has number {n}")),
                Some(_) if !(2..=12).contains(&n) || n == 7 => {
                    return Err(format!("tile {t} has number {n}"))
                }
                _ => {}
            }
        }
        let coastal = &topo().coastal_edges;
        for (i, &(e, _)) in self.ports.iter().enumerate() {
            if !coastal.contains(&e) {
                return Err(format!("port {i} is on edge {e}, which is not on the coast"));
            }
            if self.ports[..i].iter().any(|&(f, _)| f == e) {
                return Err(format!("two ports share edge {e}"));
            }
        }
        if *self != Board::new(self.tile_resource, self.tile_number, self.ports) {
            return Err("port node masks do not match the ports".into());
        }
        Ok(())
    }
```

Note: `"tile {t} has number {n}"` with `n = 7` contains `"number 7"`, which the test relies on.

`engine/src/invariants.rs`:

```rust
//! Structural consistency checks, for states built outside normal play (snapshots, tests).

use crate::rules::awards::longest_road;
use crate::state::{Phase, State};
use crate::topology::{topo, NUM_EDGES, NUM_NODES, NUM_TILES};
use crate::types::*;

impl State {
    /// Every structural invariant normal play maintains. Award ownership is not checked: an
    /// imported position may follow another engine's award rules.
    pub fn check_invariants(&self) -> Result<(), String> {
        self.config.validate()?;
        self.board.validate()?;
        for r in 0..NUM_RESOURCES {
            let held: u32 = self.players.iter().map(|p| p.hand[r] as u32).sum();
            let total = held + self.bank[r] as u32;
            if total != BANK_PER_RESOURCE as u32 {
                return Err(format!(
                    "resource {r}: hands and bank hold {total} cards, expected {BANK_PER_RESOURCE}"
                ));
            }
        }
        let t = topo();
        let mut nodes = 0u64;
        let mut edges = 0u128;
        for (i, p) in self.players.iter().enumerate() {
            if (p.settlements | p.cities) >> NUM_NODES != 0 {
                return Err(format!("player {i} settlements/cities use node ids beyond {}", NUM_NODES - 1));
            }
            if p.roads >> NUM_EDGES != 0 {
                return Err(format!("player {i} roads use edge ids beyond {}", NUM_EDGES - 1));
            }
            if p.settlements.count_ones() > MAX_SETTLEMENTS
                || p.cities.count_ones() > MAX_CITIES
                || p.roads.count_ones() > MAX_ROADS
            {
                return Err(format!("player {i} has more pieces than the game allows"));
            }
            if p.settlements & p.cities != 0 {
                return Err(format!("player {i} has a settlement and a city on one node"));
            }
            let b = p.settlements | p.cities;
            if nodes & b != 0 {
                return Err("two players share a node".into());
            }
            nodes |= b;
            if edges & p.roads != 0 {
                return Err("two players share an edge".into());
            }
            edges |= p.roads;
            for c in 0..5 {
                if p.dev_new[c] > p.dev_hand[c] {
                    return Err(format!("player {i} bought more {:?} cards than they hold", DevCard::from_index(c)));
                }
            }
            if p.knights_played != p.dev_played[DevCard::Knight.index()] {
                return Err(format!("player {i}: knights_played disagrees with dev_played"));
            }
            if p.dev_played[DevCard::VictoryPoint.index()] != 0 {
                return Err(format!("player {i}: victory point cards are never played"));
            }
        }
        for n in bits64(nodes) {
            if nodes & t.node_neighbor_mask[n as usize] != 0 {
                return Err(format!("distance rule broken at node {n}"));
            }
        }
        let left = &self.dev_deck[self.dev_deck_pos as usize..];
        for (k, &count) in DEV_DECK_COUNTS.iter().enumerate() {
            let out: u32 = self
                .players
                .iter()
                .map(|p| p.dev_hand[k] as u32 + p.dev_played[k] as u32)
                .sum();
            let in_deck = left.iter().filter(|c| c.index() == k).count() as u32;
            if out + in_deck != count as u32 {
                return Err(format!(
                    "{:?}: {out} held or played + {in_deck} in the deck, expected {count}",
                    DevCard::from_index(k)
                ));
            }
        }
        if self.robber as usize >= NUM_TILES {
            return Err(format!("robber on tile {}, beyond {}", self.robber, NUM_TILES - 1));
        }
        if self.current as usize >= NUM_PLAYERS {
            return Err(format!("current player {} out of range", self.current));
        }
        for owner in [self.longest_road_owner, self.largest_army_owner].into_iter().flatten() {
            if owner as usize >= NUM_PLAYERS {
                return Err(format!("award owner {owner} out of range"));
            }
        }
        let owing = self.players.iter().any(|p| p.discard_remaining > 0);
        if owing != (self.phase == Phase::Discard) {
            return Err("discard_remaining must be set exactly in the Discard phase".into());
        }
        let setup = matches!(self.phase, Phase::SetupSettlement | Phase::SetupRoad { .. });
        if setup != (self.setup_step < 8) {
            return Err(format!("setup_step {} does not match phase {:?}", self.setup_step, self.phase));
        }
        if !matches!(self.return_phase, Phase::PreRoll | Phase::Main) {
            return Err(format!("return_phase must be PreRoll or Main, got {:?}", self.return_phase));
        }
        let trading = matches!(self.phase, Phase::TradeResponse | Phase::TradeConfirm);
        if trading != self.trade.is_some() {
            return Err("a pending trade must exist exactly in the trade phases".into());
        }
        if let Phase::SetupRoad { node } = self.phase {
            if node as usize >= NUM_NODES
                || self.players[self.current as usize].settlements & (1u64 << node) == 0
            {
                return Err(format!("SetupRoad at node {node}, which has no settlement of the current player"));
            }
        }
        if let Phase::RoadBuilding { roads_left } = self.phase {
            if !(1..=2).contains(&roads_left) {
                return Err(format!("RoadBuilding with {roads_left} roads left"));
            }
        }
        let occupied = self.occupied_nodes();
        for p in 0..NUM_PLAYERS {
            let len = longest_road(self.players[p].roads, occupied & !self.buildings(p));
            if self.players[p].longest_road_len != len {
                return Err(format!("cached longest road of player {p} is stale"));
            }
        }
        Ok(())
    }
}
```

If `NUM_NODES` / `NUM_EDGES` / `NUM_TILES` are not exported from `topology`, use the constants where they are defined. In `catanatron_compat` mode, check that `random_discards` leaves every `discard_remaining` at 0 (it does: `engine/src/rules/roll.rs`). Otherwise the Discard-phase check would fail in the property test.

`engine/src/snapshot.rs`:

```rust
//! Plain-data snapshots of a full position: exported for comparison with another engine and
//! imported to continue from another engine's position (differential testing, playing inside
//! Catanatron).

use crate::board::Board;
use crate::config::GameConfig;
use crate::rules::awards::longest_road;
use crate::state::{Phase, State};
use crate::types::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct PlayerSnapshot {
    pub hand: Hand,
    pub dev_hand: [u8; 5],
    pub dev_new: [u8; 5],
    pub dev_played: [u8; 5],
    pub knights_played: u8,
    pub settlements: u64,
    pub cities: u64,
    pub roads: u128,
    pub discard_remaining: u8,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Snapshot {
    pub board: Board,
    pub robber: u8,
    pub bank: Hand,
    /// Cards left in the deck, next draw first.
    pub dev_deck: Vec<DevCard>,
    pub players: [PlayerSnapshot; NUM_PLAYERS],
    pub current: PlayerId,
    pub phase: Phase,
    pub return_phase: Phase,
    pub turn: u32,
    pub setup_step: u8,
    pub dev_played_this_turn: bool,
    pub longest_road_owner: Option<PlayerId>,
    pub largest_army_owner: Option<PlayerId>,
}

impl State {
    pub fn snapshot(&self) -> Snapshot {
        Snapshot {
            board: self.board,
            robber: self.robber,
            bank: self.bank,
            dev_deck: self.dev_deck[self.dev_deck_pos as usize..].to_vec(),
            players: std::array::from_fn(|p| {
                let pl = &self.players[p];
                PlayerSnapshot {
                    hand: pl.hand,
                    dev_hand: pl.dev_hand,
                    dev_new: pl.dev_new,
                    dev_played: pl.dev_played,
                    knights_played: pl.knights_played,
                    settlements: pl.settlements,
                    cities: pl.cities,
                    roads: pl.roads,
                    discard_remaining: pl.discard_remaining,
                }
            }),
            current: self.current,
            phase: self.phase,
            return_phase: self.return_phase,
            turn: self.turn,
            setup_step: self.setup_step,
            dev_played_this_turn: self.dev_played_this_turn,
            longest_road_owner: self.longest_road_owner,
            largest_army_owner: self.largest_army_owner,
        }
    }

    /// A state at exactly this position. Randomness the snapshot does not fix (dice, steals)
    /// comes from `seed`. Pending domestic trades are not representable.
    pub fn from_snapshot(seed: u64, config: GameConfig, snap: &Snapshot) -> Result<State, String> {
        config.validate()?;
        snap.board.validate()?;
        if matches!(snap.phase, Phase::TradeResponse | Phase::TradeConfirm) {
            return Err("snapshots with a pending domestic trade are not supported".into());
        }
        if snap.dev_deck.len() > DEV_DECK_SIZE {
            return Err(format!(
                "dev_deck has {} cards, the deck has only {DEV_DECK_SIZE}",
                snap.dev_deck.len()
            ));
        }
        // Cards already drawn first (in kind order), then the remaining cards in draw order.
        let mut drawn = DEV_DECK_COUNTS.map(|c| c as i32);
        for c in &snap.dev_deck {
            drawn[c.index()] -= 1;
        }
        if let Some(k) = (0..5).find(|&k| drawn[k] < 0) {
            return Err(format!(
                "dev_deck holds more {:?} cards than the deck has",
                DevCard::from_index(k)
            ));
        }
        let pos = DEV_DECK_SIZE - snap.dev_deck.len();
        let mut deck = [DevCard::Knight; DEV_DECK_SIZE];
        let mut i = 0;
        for (k, &n) in drawn.iter().enumerate() {
            for _ in 0..n {
                deck[i] = DevCard::from_index(k);
                i += 1;
            }
        }
        deck[pos..].copy_from_slice(&snap.dev_deck);

        let mut s = State::with_board(seed, config, snap.board);
        s.dev_deck = deck;
        s.dev_deck_pos = pos as u8;
        for (p, ps) in snap.players.iter().enumerate() {
            if ps.roads.count_ones() > MAX_ROADS {
                return Err(format!("player {p} has {} roads", ps.roads.count_ones()));
            }
            let pl = &mut s.players[p];
            pl.hand = ps.hand;
            pl.dev_hand = ps.dev_hand;
            pl.dev_new = ps.dev_new;
            pl.dev_played = ps.dev_played;
            pl.knights_played = ps.knights_played;
            pl.settlements = ps.settlements;
            pl.cities = ps.cities;
            pl.roads = ps.roads;
            pl.discard_remaining = ps.discard_remaining;
        }
        s.bank = snap.bank;
        s.robber = snap.robber;
        s.current = snap.current;
        s.phase = snap.phase;
        s.return_phase = snap.return_phase;
        s.turn = snap.turn;
        s.setup_step = snap.setup_step;
        s.dev_played_this_turn = snap.dev_played_this_turn;
        s.offers_this_turn = 0;
        s.trade = None;
        s.longest_road_owner = snap.longest_road_owner;
        s.largest_army_owner = snap.largest_army_owner;
        let occupied = s.occupied_nodes();
        for p in 0..NUM_PLAYERS {
            if s.players[p].roads >> crate::topology::NUM_EDGES != 0 {
                return Err(format!("player {p} roads use edge ids beyond {}", crate::topology::NUM_EDGES - 1));
            }
            s.players[p].longest_road_len = longest_road(s.players[p].roads, occupied & !s.buildings(p));
        }
        s.check_invariants()?;
        Ok(s)
    }
}
```

The road-count and edge-range checks run before `longest_road` because it asserts at most 16 roads.

`engine/src/game.rs`, inside `impl Game`:

```rust
    /// A game continuing from `snap` (see `State::from_snapshot`), with an empty event log.
    pub fn from_snapshot(seed: u64, config: GameConfig, snap: &Snapshot) -> Result<Game, String> {
        Ok(Game::from_state(State::from_snapshot(seed, config, snap)?))
    }
```

(import `crate::snapshot::Snapshot`).

`engine/src/lib.rs`: add `pub mod invariants;` and `pub mod snapshot;`, plus `pub use snapshot::{PlayerSnapshot, Snapshot};`.

If `Phase` does not derive `Hash`, that is fine. `Snapshot` derives only `Clone, Debug, PartialEq, Eq`. `Observation` must derive `PartialEq` for the round-trip test; it already does.

- [ ] **Step 4: Run tests**

Run: `cargo test -p settler-engine --test snapshot && cargo test && cargo clippy --workspace --all-targets -- -D warnings`
Expected: the 5 snapshot tests pass, the full suite passes (the property tests now call `State::check_invariants`), and clippy is clean.

- [ ] **Step 5: Commit (queued while the user is away)**

```bash
git add engine
git commit -m "engine: board validation, state invariants, position snapshots"
```

---

### Task 3: Bindings — snapshots, copies, and bots in Python

**Files:**
- Modify: `bindings/src/convert.rs`, `bindings/src/lib.rs`, `alphasettler/__init__.py`
- Create: `tests/python/test_snapshot.py`

**Interfaces:**
- Consumes: `settler_engine::{Snapshot, PlayerSnapshot, Game::from_snapshot, State::snapshot}`, `settler_bots::{make_bot, BOT_NAMES, Bot}`, and the existing convert.rs helpers (`int_arg`, `index_arg`, `seq_arg`, `value_error`, `parse_resource`, `parse_dev`, `counts`, `node_list`, `edge_list`, `board_dict`, `resource_name`, `dev_name`, `phase_name`, `parse_config`).
- Produces (Python):
  - `Game.from_snapshot(seed: int, snapshot: dict, config: dict | None = None) -> Game` (staticmethod)
  - `Game.snapshot() -> dict`
  - `Game.copy() -> Game` (also `copy.copy(game)`)
  - `Bot(name: str, seed: int)`, with property `name` and method `act(game: Game) -> int`, which returns the bot's chosen action id for whoever must act now. It raises `ValueError` once the game is over.
  - `alphasettler.Bot` is exported.
- Snapshot dict schema. It is exact: every key is required, unknown keys are rejected, and `board` uses the observation's board dict format:

```python
{
  "board": {"tile_resource": [str | None] * 19, "tile_number": [int] * 19,
            "ports": [(edge: int, kind: "generic" | resource name)] * 9},
  "robber": int, "bank": [int] * 5,
  "dev_deck": [card name, ...],          # cards left, next draw first
  "players": [{"hand": [5 ints], "dev_hand": [5], "dev_new": [5], "dev_played": [5],
               "knights_played": int, "settlements": [node ids], "cities": [node ids],
               "roads": [edge ids], "discard_remaining": int}] * 4,
  "current": int, "phase": phase name,
  "setup_road_node": int | None,         # used when phase == "setup_road"
  "roads_left": int | None,              # used when phase == "road_building"
  "winner": int | None,                  # used when phase == "game_over"
  "return_phase": "pre_roll" | "main",
  "turn": int, "setup_step": int, "dev_played_this_turn": bool, "offers_this_turn": int,
  "longest_road_owner": int | None, "largest_army_owner": int | None,
}
```

Dev arrays are indexed in the order `knight, victory_point, road_building, year_of_plenty, monopoly`. Resource arrays use the order `wood, brick, sheep, wheat, ore`.

- [ ] **Step 1: Write the failing tests**

`tests/python/test_snapshot.py`:

```python
import copy
import random

import pytest

from alphasettler import Bot, Game

NO_TRADES = {"max_offers_per_turn": 0}


def play(g, steps, seed=0):
    rng = random.Random(seed)
    for _ in range(steps):
        if g.is_over:
            return
        g.apply(rng.choice(g.legal_actions()))


def test_snapshot_shape_at_start():
    snap = Game(1).snapshot()
    assert snap["phase"] == "setup_settlement" and snap["setup_step"] == 0
    assert len(snap["players"]) == 4 and snap["players"][0]["hand"] == [0] * 5
    assert len(snap["dev_deck"]) == 25 and snap["bank"] == [19] * 5
    assert set(snap["board"]) == {"tile_resource", "tile_number", "ports"}
    assert snap["winner"] is None and snap["return_phase"] == "main"


def test_snapshot_round_trip_mid_game():
    for seed in range(5):
        g = Game(seed, NO_TRADES)
        play(g, 150 + 97 * seed, seed)
        snap = g.snapshot()
        h = Game.from_snapshot(seed, snap, NO_TRADES)
        assert h.snapshot() == snap
        assert h.legal_actions() == g.legal_actions()
        assert h.observation(2) == g.observation(2)


def test_observation_has_return_phase():
    assert Game(1).observation(0)["return_phase"] == "main"


def test_copy_is_independent():
    g = Game(3)
    h = g.copy()
    h.apply(h.legal_actions()[0])
    assert g.snapshot() != h.snapshot()
    assert g.log(0) == []
    assert copy.copy(g).snapshot() == g.snapshot()


@pytest.mark.parametrize(
    "mutate, needle",
    [
        (lambda s: s["bank"].__setitem__(0, 20), "resource 0"),
        (lambda s: s.__setitem__("phase", "trade_response"), "pending domestic trade"),
        (lambda s: s.__setitem__("phase", "nonsense"), "phase"),
        (lambda s: s["players"][0]["roads"].append(999), "edge"),
        (lambda s: s["players"][0]["settlements"].extend([3, 3]), "twice"),
        (lambda s: s.__setitem__("extra", 1), "unknown snapshot key"),
        (lambda s: s["players"][0].__setitem__("extra", 1), "unknown"),
        (lambda s: s.pop("robber"), "robber"),
        (lambda s: s["players"].pop(), "players"),
        (lambda s: s["dev_deck"].__setitem__(0, "joker"), "dev card"),
        (lambda s: s["board"]["tile_resource"].__setitem__(0, "gold"), "resource"),
        (lambda s: s.__setitem__("robber", -1), "robber"),
    ],
)
def test_bad_snapshots_raise_value_error(mutate, needle):
    snap = Game(2).snapshot()
    mutate(snap)
    with pytest.raises(ValueError, match=needle):
        Game.from_snapshot(2, snap)


def test_bot_plays_a_full_game():
    g = Game(5, NO_TRADES)
    bots = [Bot("greedy", i) for i in range(4)]
    while not g.is_over:
        a = bots[g.current_actor].act(g)
        assert a in g.legal_actions()
        g.apply(a)
    assert g.phase == "game_over"
    with pytest.raises(ValueError, match="over"):
        bots[0].act(g)


def test_bot_names_and_errors():
    assert Bot("random", 1).name == "random"
    with pytest.raises(ValueError, match="unknown bot"):
        Bot("nope", 0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python3 -m pytest tests/python/test_snapshot.py -q`
Expected: FAIL: `ImportError: cannot import name 'Bot'`.

- [ ] **Step 3: Implement**

In `bindings/src/convert.rs`, add the conversion pair below. It follows the file's existing helpers: integers go through `int_arg` / `index_arg`, so out-of-range values raise `ValueError`, while wrong types stay `TypeError`.

```rust
const SNAPSHOT_KEYS: [&str; 16] = [
    "board", "robber", "bank", "dev_deck", "players", "current", "phase", "setup_road_node",
    "roads_left", "winner", "return_phase", "turn", "setup_step", "dev_played_this_turn",
    "longest_road_owner", "largest_army_owner",
];
const PLAYER_KEYS: [&str; 9] = [
    "hand", "dev_hand", "dev_new", "dev_played", "knights_played", "settlements", "cities",
    "roads", "discard_remaining",
];
const BOARD_KEYS: [&str; 3] = ["tile_resource", "tile_number", "ports"];
```

Helpers (all private to convert.rs):
- `fn check_keys(d: &Bound<PyDict>, allowed: &[&str], what: &str) -> PyResult<()>` returns `value_error(format!("unknown {what} key {k:?}"))` for the first key not in `allowed`. A top-level `what` of `"snapshot"` produces the message `unknown snapshot key "extra"`.
- `fn item<'py>(d: &Bound<'py, PyDict>, key: &str, what: &str) -> PyResult<Bound<'py, PyAny>>` returns `value_error(format!("{what} is missing {key:?}"))` when absent.
- `fn opt_player(v) -> PyResult<Option<PlayerId>>` maps `None` to `None`, and otherwise calls `index_arg(v, NUM_PLAYERS, what)`.
- `fn id_mask64(v, end, what) -> PyResult<u64>` takes a list of ints, each checked with `index_arg(.., end, what)`. A repeated id is `value_error(format!("{what} lists node {n} twice"))`.
- `fn id_mask128(v, end, what) -> PyResult<u128>`: same as `id_mask64`, but for edges (`"... lists edge {e} twice"`). Make the out-of-range message contain the word `edge`, e.g. `what = "player 0 roads (edge ids)"`.
- `fn small_array(v, what) -> PyResult<[u8; 5]>` uses `seq_arg(v, 5, what)` with `int_arg::<u8>` on each item.

Parsing:

```rust
fn parse_board(v: &Bound<'_, PyAny>) -> PyResult<Board> {
    let d = v.downcast::<PyDict>()?;
    check_keys(d, &BOARD_KEYS, "board")?;
    let res = seq_arg(&item(d, "tile_resource", "board")?, NUM_TILES, "tile_resource")?;
    let mut tile_resource = [None; NUM_TILES];
    for (t, r) in res.iter().enumerate() {
        tile_resource[t] = if r.is_none() { None } else { Some(parse_resource(&r.extract::<String>()?)?) };
    }
    let nums = seq_arg(&item(d, "tile_number", "board")?, NUM_TILES, "tile_number")?;
    let mut tile_number = [0u8; NUM_TILES];
    for (t, n) in nums.iter().enumerate() {
        tile_number[t] = int_arg::<u8>(n, "tile_number")?;
    }
    let ports_in = seq_arg(&item(d, "ports", "board")?, 9, "ports")?;
    let mut ports = [(0u8, PortKind::Generic); 9];
    for (i, p) in ports_in.iter().enumerate() {
        let pair = seq_arg(p, 2, "port")?;
        let edge = index_arg(&pair[0], NUM_EDGES, "port edge")? as u8;
        let kind: String = pair[1].extract()?;
        ports[i] = (edge, if kind == "generic" { PortKind::Generic } else { PortKind::Specific(parse_resource(&kind)?) });
    }
    Ok(Board::new(tile_resource, tile_number, ports))
}

fn parse_phase(d: &Bound<'_, PyDict>) -> PyResult<Phase> {
    let name: String = item(d, "phase", "snapshot")?.extract()?;
    Ok(match name.as_str() {
        "setup_settlement" => Phase::SetupSettlement,
        "setup_road" => Phase::SetupRoad {
            node: index_arg(&item(d, "setup_road_node", "snapshot")?, NUM_NODES, "setup_road_node")? as u8,
        },
        "pre_roll" => Phase::PreRoll,
        "discard" => Phase::Discard,
        "move_robber" => Phase::MoveRobber,
        "steal" => Phase::Steal,
        "main" => Phase::Main,
        "road_building" => Phase::RoadBuilding {
            roads_left: int_arg::<u8>(&item(d, "roads_left", "snapshot")?, "roads_left")?,
        },
        "game_over" => Phase::GameOver { winner: opt_player(&item(d, "winner", "snapshot")?, "winner")? },
        "trade_response" | "trade_confirm" => {
            return Err(value_error("snapshots with a pending domestic trade are not supported"))
        }
        other => return Err(value_error(format!("unknown phase {other:?}"))),
    })
}

pub fn parse_snapshot(d: &Bound<'_, PyDict>) -> PyResult<Snapshot> { /* see below */ }
```

`parse_snapshot` does the following:
- calls `check_keys(d, &SNAPSHOT_KEYS, "snapshot")`;
- parses the board with `parse_board`;
- reads `robber` with `index_arg(.., NUM_TILES, "robber")`;
- reads `bank` with `small_array`;
- reads `dev_deck` as a list of names via `parse_dev`. An unknown name gives `parse_dev`'s message, which must contain "dev card";
- reads `players` with `seq_arg(.., NUM_PLAYERS, "players")`. Each entry gets `check_keys(.., &PLAYER_KEYS, &format!("player {p}"))` and these fields:
  - `small_array` for `hand`, `dev_hand`, `dev_new`, `dev_played`;
  - `int_arg::<u8>` for `knights_played` and `discard_remaining`;
  - `id_mask64(.., NUM_NODES, ..)` for `settlements` and `cities`;
  - `id_mask128(.., NUM_EDGES, ..)` for `roads`;
- reads `current` with `index_arg(.., NUM_PLAYERS, "current")`;
- reads `phase` with `parse_phase`;
- reads `return_phase`: `"pre_roll"` maps to `PreRoll`, `"main"` to `Main`, and anything else is `ValueError("return_phase must be 'pre_roll' or 'main'")`;
- reads `turn` with `int_arg::<u32>`, `setup_step` with `int_arg::<u8>`, and `dev_played_this_turn` with `extract::<bool>()`;
- reads `longest_road_owner` and `largest_army_owner` with `opt_player`.

If `parse_resource` / `parse_dev`'s messages do not contain `"resource"` / `"dev card"`, adjust them so they do. The current text is `unknown resource "gold"` / `unknown dev card "joker"`.

Export:

```rust
pub fn snapshot_dict<'py>(py: Python<'py>, s: &Snapshot) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("board", board_dict(py, &s.board)?)?;
    d.set_item("robber", s.robber)?;
    d.set_item("bank", counts(&s.bank))?;
    let deck: Vec<&str> = s.dev_deck.iter().map(|&c| dev_name(c)).collect();
    d.set_item("dev_deck", deck)?;
    let players = PyList::empty(py);
    for p in &s.players {
        let pd = PyDict::new(py);
        pd.set_item("hand", counts(&p.hand))?;
        pd.set_item("dev_hand", counts(&p.dev_hand))?;
        pd.set_item("dev_new", counts(&p.dev_new))?;
        pd.set_item("dev_played", counts(&p.dev_played))?;
        pd.set_item("knights_played", p.knights_played)?;
        pd.set_item("settlements", node_list(p.settlements))?;
        pd.set_item("cities", node_list(p.cities))?;
        pd.set_item("roads", edge_list(p.roads))?;
        pd.set_item("discard_remaining", p.discard_remaining)?;
        players.append(pd)?;
    }
    d.set_item("players", players)?;
    d.set_item("current", s.current)?;
    d.set_item("phase", phase_name(s.phase))?;
    d.set_item("setup_road_node", match s.phase { Phase::SetupRoad { node } => Some(node), _ => None })?;
    d.set_item("roads_left", match s.phase { Phase::RoadBuilding { roads_left } => Some(roads_left), _ => None })?;
    d.set_item("winner", match s.phase { Phase::GameOver { winner } => winner, _ => None })?;
    d.set_item("return_phase", phase_name(s.return_phase))?;
    d.set_item("turn", s.turn)?;
    d.set_item("setup_step", s.setup_step)?;
    d.set_item("dev_played_this_turn", s.dev_played_this_turn)?;
    d.set_item("longest_road_owner", s.longest_road_owner)?;
    d.set_item("largest_army_owner", s.largest_army_owner)?;
    Ok(d)
}
```

`bindings/src/lib.rs`, inside `#[pymethods] impl PyGame`. Take `seed` the way `PyGame::new` does after Plan 2's fix (a `&Bound<PyAny>` through `int_arg::<u64>`):

```rust
    #[staticmethod]
    #[pyo3(signature = (seed, snapshot, config=None))]
    fn from_snapshot(
        seed: &Bound<'_, PyAny>,
        snapshot: &Bound<'_, PyDict>,
        config: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let seed = int_arg::<u64>(seed, "seed")?;
        let config = parse_config(config)?;
        let snap = parse_snapshot(snapshot)?;
        let inner = settler_engine::Game::from_snapshot(seed, config, &snap).map_err(value_error)?;
        Ok(PyGame { inner })
    }

    fn snapshot<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        snapshot_dict(py, &self.inner.state().snapshot())
    }

    fn copy(&self) -> Self {
        PyGame { inner: self.inner.clone() }
    }

    fn __copy__(&self) -> Self {
        self.copy()
    }
```

The bot class:

```rust
/// A native bot. Not thread-shared in practice; the mutex only satisfies PyO3's Sync bound.
#[pyclass(name = "Bot", module = "alphasettler._engine")]
struct PyBot {
    name: &'static str,
    inner: std::sync::Mutex<Box<dyn settler_bots::Bot>>,
}

#[pymethods]
impl PyBot {
    #[new]
    fn new(name: &str, seed: &Bound<'_, PyAny>) -> PyResult<Self> {
        let seed = int_arg::<u64>(seed, "seed")?;
        let bot = settler_bots::make_bot(name, seed).ok_or_else(|| {
            value_error(format!("unknown bot {name:?}; known bots: {:?}", settler_bots::BOT_NAMES))
        })?;
        Ok(PyBot { name: bot.name(), inner: std::sync::Mutex::new(bot) })
    }

    #[getter]
    fn name(&self) -> &'static str {
        self.name
    }

    /// The bot's choice for whoever must act in `game` now, as an action id.
    fn act(&self, game: PyRef<'_, PyGame>) -> PyResult<u32> {
        let s = game.inner.state();
        if s.is_over() {
            return Err(value_error("the game is over"));
        }
        let legal = game.inner.legal_actions();
        let obs = s.observation(s.current_actor());
        let mut bot = self.inner.lock().map_err(|_| value_error("bot state is poisoned"))?;
        Ok(bot.act(&obs, &legal).encode() as u32)
    }
}
```

Register it with `m.add_class::<PyBot>()?;`. In `alphasettler/__init__.py`, import and export `Bot` next to `Game`.

- [ ] **Step 4: Build and run**

```bash
source "$HOME/.cargo/env"
VIRTUAL_ENV=$PWD/.venv .venv/bin/maturin develop --release
.venv/bin/python3 -m pytest tests/python -q
cargo clippy -p settler-bindings -- -D warnings
```

Expected: all Python tests pass (the new file has 23 test cases including the parametrized ones), and clippy is clean.

- [ ] **Step 5: Commit (queued while the user is away)**

```bash
git add bindings alphasettler/__init__.py tests/python/test_snapshot.py
git commit -m "bindings: position snapshots, game copies, native bots in Python"
```

---

### Task 4: `oracle/` — install Catanatron, mapping tables, translation

**Files:**
- Modify: `pyproject.toml`
- Create: `oracle/__init__.py`, `oracle/tables.py`, `oracle/translate.py`, `tests/python/test_oracle_translate.py`, `tests/python/test_oracle_isolation.py`

**Read first:** `.superpowers/sdd/catanatron-reference.md` §1 (mapping), §2.2–2.6 (state fields, seating, randomness, turns, discards), and §3 (action shapes and forcing).

**Interfaces:**
- Consumes:
  - `alphasettler.Game` (`from_snapshot`, `snapshot`, `copy`, `legal_actions`, `phase`)
  - `alphasettler.describe_action`
  - Catanatron: `catanatron.Game`, `catanatron.models.enums.{ActionPrompt, ActionType, RESOURCES, SETTLEMENT, CITY}`, `catanatron.game` module (`TURNS_LIMIT`)
- Produces (`oracle.translate`):
  - constants `RESOURCE_NAMES`, `DEV_NAMES`, `CAT_DEV`
  - action-id helpers `ROLL, END_TURN, BUY_DEV, PLAY_KNIGHT, PLAY_ROAD_BUILDING, play_monopoly(r), play_year_of_plenty(a, b), build_settlement(n), build_city(n), build_road(e), move_robber(t), steal_from(p), discard(r), maritime(give, get)`
  - `class Untranslatable(Exception)`, with attribute `kind: str`
  - `board_dict(catan_map) -> dict`
  - `seat(state, color) -> int`
  - `action_token(state, action)`
  - `catan_tokens(state, playable) -> set`
  - `our_tokens(game) -> set`
  - `our_steps(state, record) -> list[tuple[int, dict | None]]`
  - `catan_snapshot(game) -> dict` (the Task 3 schema)
  - `compare(ours: dict, theirs: dict) -> list[str]`
  - `edge_nodes(e) -> tuple[int, int]` (our node ids)
- Produces (`oracle`): `CATANATRON_REQUIREMENT: str`.

Tokens:
- An **int** is one of our action ids.
- A robber move is `("robber", our_tile, victim_seat_or_None)`. Catanatron's MOVE_ROBBER combines moving and stealing, so our MoveRobber is expanded per steal choice.
- Catanatron's one-card Year of Plenty is `("year_of_plenty_one_card", resource_index)`, which has no equivalent in our engine.

- [ ] **Step 1: Install and pin**

`pyproject.toml`:

```toml
[project.optional-dependencies]
dev = ["maturin==1.15.0", "pytest==9.1.1"]
oracle = ["catanatron @ git+https://github.com/bcollazo/catanatron.git@ecf931181b9a65bb4116a2153fb78c16f1438e00"]
```

and in `[tool.maturin]` add `python-packages = ["oracle"]`.

```bash
.venv/bin/pip3 install "catanatron @ git+https://github.com/bcollazo/catanatron.git@ecf931181b9a65bb4116a2153fb78c16f1438e00"
.venv/bin/pip3 show catanatron | head -3        # Version: 3.3.0
.venv/bin/pip-audit
VIRTUAL_ENV=$PWD/.venv .venv/bin/maturin develop --release
```

Record the installed versions of catanatron, networkx, click and rich, plus the pip-audit output, in the report.

- [ ] **Step 2: Write the failing tests**

`tests/python/test_oracle_translate.py`:

```python
import random

import pytest

catanatron = pytest.importorskip("catanatron")

from catanatron import Color, Game as CatanGame, RandomPlayer  # noqa: E402
from catanatron.models.enums import ActionRecord  # noqa: E402

from alphasettler import Game, action_space_size, describe_action  # noqa: E402
from oracle import tables  # noqa: E402
from oracle.translate import (  # noqa: E402
    BUY_DEV, END_TURN, PLAY_KNIGHT, PLAY_ROAD_BUILDING, ROLL, build_city, build_road, build_settlement,
    catan_snapshot, catan_tokens, compare, discard, maritime, move_robber, our_tokens,
    play_monopoly, play_year_of_plenty, steal_from,
)

COLORS = [Color.RED, Color.BLUE, Color.ORANGE, Color.WHITE]
CONFIG = {"max_offers_per_turn": 0, "catanatron_compat": False, "max_turns": 994}


def new_catan(seed):
    return CatanGame([RandomPlayer(c) for c in COLORS], seed=seed)


def test_python_action_ids_match_the_engine():
    expected = {}
    expected.update({ROLL: "Roll", END_TURN: "EndTurn", BUY_DEV: "BuyDev",
                     PLAY_KNIGHT: "PlayKnight", PLAY_ROAD_BUILDING: "PlayRoadBuilding"})
    names = ["Wood", "Brick", "Sheep", "Wheat", "Ore"]
    for r in range(5):
        expected[play_monopoly(r)] = f"PlayMonopoly({names[r]})"
        expected[discard(r)] = f"Discard({names[r]})"
        for g in range(5):
            if g != r:
                expected[maritime(r, g)] = f"MaritimeTrade {{ give: {names[r]}, get: {names[g]} }}"
        for b in range(r, 5):
            expected[play_year_of_plenty(r, b)] = f"PlayYearOfPlenty({names[r]}, {names[b]})"
    for n in range(54):
        expected[build_settlement(n)] = f"BuildSettlement({n})"
        expected[build_city(n)] = f"BuildCity({n})"
    for e in range(72):
        expected[build_road(e)] = f"BuildRoad({e})"
    for t in range(19):
        expected[move_robber(t)] = f"MoveRobber({t})"
    for p in range(4):
        expected[steal_from(p)] = f"StealFrom({p})"
    for a, text in expected.items():
        assert 0 <= a < action_space_size()
        assert describe_action(a) == text


def test_tables_are_bijections():
    assert sorted(tables.CAT_NODE_TO_OUR_NODE) == list(range(54))
    assert [tables.CAT_NODE_TO_OUR_NODE[c] for c in tables.OUR_NODE_TO_CAT_NODE] == list(range(54))
    assert len(set(tables.OUR_EDGE_TO_CAT_EDGE)) == 72
    assert len(set(tables.OUR_TILE_TO_CAT_CUBE)) == 19
    assert len(set(tables.CAT_PORT_ID_TO_OUR_EDGE)) == 9


@pytest.mark.parametrize("seed", range(10))
def test_initial_position_imports_and_matches(seed):
    cat = new_catan(seed)
    theirs = catan_snapshot(cat)
    ours = Game.from_snapshot(seed, theirs, CONFIG)
    assert compare(ours.snapshot(), theirs) == []
    assert our_tokens(ours) == catan_tokens(cat.state, cat.playable_actions)


def test_forced_outcomes_translate():
    # Play a Catanatron game to completion; every executed record must translate and the
    # translated position must import cleanly at every step.
    cat = new_catan(7)
    while cat.winning_color() is None and cat.state.num_turns < 1000:
        cat.play_tick()
        Game.from_snapshot(7, catan_snapshot(cat), CONFIG)
    assert any(r.action.action_type.name == "ROLL" for r in cat.state.action_records)
```

`tests/python/test_oracle_isolation.py`:

```python
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_only_oracle_imports_catanatron():
    offenders = []
    for path in [*ROOT.glob("alphasettler/**/*.py"), *ROOT.glob("tests/python/*.py")]:
        if path.name.startswith("test_oracle"):
            continue
        text = path.read_text()
        if "import catanatron" in text or "from catanatron" in text:
            offenders.append(str(path.relative_to(ROOT)))
    assert offenders == []


def test_oracle_package_init_does_not_import_catanatron():
    text = (ROOT / "oracle" / "__init__.py").read_text()
    assert "catanatron import" not in text and "import catanatron" not in text
```

(`bench/compare/catanatron_bench.py` is a standalone benchmark that runs in its own venv. It is not part of the package and stays outside this check.)

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv/bin/python3 -m pytest tests/python/test_oracle_translate.py tests/python/test_oracle_isolation.py -q`
Expected: FAIL: `ModuleNotFoundError: No module named 'oracle'`.

- [ ] **Step 4: Implement**

`oracle/__init__.py`:

```python
"""Catanatron adapter: the only code that imports Catanatron (GPL-3.0, imported at runtime,
never copied). Submodules import it; this module does not, so callers can read the install
requirement when Catanatron is missing."""

CATANATRON_REQUIREMENT = (
    "catanatron @ git+https://github.com/bcollazo/catanatron.git@ecf931181b9a65bb4116a2153fb78c16f1438e00"
)
```

`oracle/tables.py` copies the tables from the reference §1.6 **verbatim**: `OUR_TILE_TO_CAT_CUBE`, `CAT_LAND_TILE_ID_TO_OUR_TILE`, `CAT_NODE_TO_OUR_NODE`, `OUR_NODE_TO_CAT_NODE`, `OUR_EDGE_TO_CAT_EDGE`, `CAT_EDGE_TO_OUR_EDGE` (as the dict comprehension shown there) and `CAT_PORT_ID_TO_OUR_EDGE`. Add a module docstring: "Id mappings between Catanatron 3.3 (commit ecf9311) and AlphaSettler, derived by geometry and verified by probe (catanatron-reference §1.3): tiles, nodes, edges and ports are bijections."

`oracle/translate.py`:

```python
"""Translation between Catanatron 3.3 and AlphaSettler: boards, seats, actions, legal sets and
full positions. Catanatron seat i (index in ``state.colors``) is our player i."""

from __future__ import annotations

from collections import Counter

import catanatron.game as catan_game
from catanatron.models.enums import CITY, RESOURCES, SETTLEMENT, ActionPrompt, ActionType

from oracle.tables import (
    CAT_EDGE_TO_OUR_EDGE, CAT_NODE_TO_OUR_NODE, CAT_PORT_ID_TO_OUR_EDGE, OUR_EDGE_TO_CAT_EDGE,
    OUR_TILE_TO_CAT_CUBE,
)

RESOURCE_NAMES = ["wood", "brick", "sheep", "wheat", "ore"]  # same order as Catanatron RESOURCES
DEV_NAMES = ["knight", "victory_point", "road_building", "year_of_plenty", "monopoly"]
CAT_DEV = ["KNIGHT", "VICTORY_POINT", "ROAD_BUILDING", "YEAR_OF_PLENTY", "MONOPOLY"]
CUBE_TO_OUR_TILE = {cube: t for t, cube in enumerate(OUR_TILE_TO_CAT_CUBE)}
SETUP_TURNS = 6  # Catanatron's num_turns counts 6 advances during setup

# Our action ids (engine/src/action.rs).
ROLL, END_TURN, BUY_DEV, PLAY_KNIGHT, PLAY_ROAD_BUILDING = 0, 1, 2, 3, 4
PAIRS = [(a, b) for a in range(5) for b in range(a, 5)]


def play_monopoly(r): return 5 + r
def play_year_of_plenty(a, b): return 10 + PAIRS.index((min(a, b), max(a, b)))
def build_settlement(n): return 25 + n
def build_city(n): return 79 + n
def build_road(e): return 133 + e
def move_robber(t): return 205 + t
def steal_from(p): return 224 + p
def discard(r): return 228 + r
def maritime(give, get): return 233 + 5 * give + get


class Untranslatable(Exception):
    """A Catanatron action with no AlphaSettler equivalent; ``kind`` names the allowlist entry."""

    def __init__(self, kind: str, detail: str):
        super().__init__(f"{kind}: {detail}")
        self.kind = kind


def _res(name: str) -> int:
    return RESOURCES.index(name)


def seat(state, color) -> int:
    return state.color_to_index[color]


def our_edge(edge) -> int:
    return CAT_EDGE_TO_OUR_EDGE[tuple(sorted(edge))]


def edge_nodes(e: int) -> tuple[int, int]:
    a, b = OUR_EDGE_TO_CAT_EDGE[e]
    return CAT_NODE_TO_OUR_NODE[a], CAT_NODE_TO_OUR_NODE[b]


def board_dict(catan_map) -> dict:
    tile_resource = [None] * 19
    tile_number = [0] * 19
    for cube, tile in catan_map.land_tiles.items():
        t = CUBE_TO_OUR_TILE[cube]
        tile_resource[t] = None if tile.resource is None else RESOURCE_NAMES[_res(tile.resource)]
        tile_number[t] = tile.number or 0
    ports = [
        (CAT_PORT_ID_TO_OUR_EDGE[p.id], "generic" if p.resource is None else RESOURCE_NAMES[_res(p.resource)])
        for p in sorted(catan_map.ports_by_id.values(), key=lambda p: p.id)
    ]
    return {"tile_resource": tile_resource, "tile_number": tile_number, "ports": ports}


def action_token(state, action):
    """The comparable token for one Catanatron action (see module docstring of oracle.diff)."""
    t, v = action.action_type, action.value
    simple = {
        ActionType.ROLL: ROLL, ActionType.END_TURN: END_TURN, ActionType.BUY_DEVELOPMENT_CARD: BUY_DEV,
        ActionType.PLAY_KNIGHT_CARD: PLAY_KNIGHT, ActionType.PLAY_ROAD_BUILDING: PLAY_ROAD_BUILDING,
    }
    if t in simple:
        return simple[t]
    if t == ActionType.PLAY_MONOPOLY:
        return play_monopoly(_res(v))
    if t == ActionType.PLAY_YEAR_OF_PLENTY:
        if len(v) == 1:
            return ("year_of_plenty_one_card", _res(v[0]))
        return play_year_of_plenty(_res(v[0]), _res(v[1]))
    if t == ActionType.BUILD_SETTLEMENT:
        return build_settlement(CAT_NODE_TO_OUR_NODE[v])
    if t == ActionType.BUILD_CITY:
        return build_city(CAT_NODE_TO_OUR_NODE[v])
    if t == ActionType.BUILD_ROAD:
        return build_road(our_edge(v))
    if t == ActionType.MOVE_ROBBER:
        cube, victim = v[0], v[1]
        return ("robber", CUBE_TO_OUR_TILE[cube], None if victim is None else seat(state, victim))
    if t == ActionType.DISCARD_RESOURCE:
        return discard(_res(v))
    if t == ActionType.MARITIME_TRADE:
        return maritime(_res(v[0]), _res(v[4]))
    raise Untranslatable("domestic-trade", f"{t} is outside the differential test")


def catan_tokens(state, playable) -> set:
    return {action_token(state, a) for a in playable}


def our_tokens(game) -> set:
    """Our legal actions as tokens; each robber move expands into one token per steal choice."""
    if game.phase != "move_robber":
        return set(game.legal_actions())
    out = set()
    for a in game.legal_actions():
        tile = a - move_robber(0)
        after = game.copy()
        after.apply(a)
        if after.phase == "steal":
            out.update(("robber", tile, v - steal_from(0)) for v in after.legal_actions())
        else:
            out.add(("robber", tile, None))
    return out


def our_steps(state, record) -> list[tuple[int, dict | None]]:
    """Our actions, with forced chance outcomes, equivalent to one executed Catanatron action."""
    action, result = record.action, record.result
    t = action.action_type
    if t == ActionType.ROLL:
        return [(ROLL, {"roll": list(result)})]
    if t == ActionType.BUY_DEVELOPMENT_CARD:
        return [(BUY_DEV, {"dev": DEV_NAMES[CAT_DEV.index(result)]})]
    token = action_token(state, action)
    if t == ActionType.MOVE_ROBBER:
        _, tile, victim = token
        steps = [(move_robber(tile), None)]
        if victim is not None:
            steps.append((steal_from(victim), {"steal": RESOURCE_NAMES[_res(result)]}))
        return steps
    if isinstance(token, tuple):
        raise Untranslatable(token[0], repr(action))
    return [(token, None)]


def _phase(game) -> tuple[str, dict]:
    st = game.state
    winner = game.winning_color()
    if winner is not None:
        return "game_over", {"winner": seat(st, winner)}
    if st.num_turns >= catan_game.TURNS_LIMIT:
        return "game_over", {"winner": None}
    prompt = st.current_prompt
    if prompt == ActionPrompt.BUILD_INITIAL_SETTLEMENT:
        return "setup_settlement", {}
    if prompt == ActionPrompt.BUILD_INITIAL_ROAD:
        node = st.action_records[-1].action.value  # the settlement just placed
        return "setup_road", {"setup_road_node": CAT_NODE_TO_OUR_NODE[node]}
    if prompt == ActionPrompt.DISCARD:
        return "discard", {}
    if prompt == ActionPrompt.MOVE_ROBBER:
        return "move_robber", {}
    if prompt == ActionPrompt.PLAY_TURN:
        if st.is_road_building:
            return "road_building", {"roads_left": st.free_roads_available}
        rolled = st.player_state[f"P{st.current_turn_index}_HAS_ROLLED"]
        return ("main" if rolled else "pre_roll"), {}
    raise Untranslatable("domestic-trade", f"prompt {prompt}")


def catan_snapshot(game) -> dict:
    """A Catanatron position in the snapshot schema of ``alphasettler.Game.snapshot``."""
    st = game.state
    board = st.board
    ps = st.player_state
    cur = st.current_turn_index
    players = []
    for i, color in enumerate(st.colors):
        key = f"P{i}_"
        dev_hand = [ps[f"{key}{d}_IN_HAND"] for d in CAT_DEV]
        # Catanatron tracks "held at the start of the turn" per kind; a card not held then was
        # bought this turn. Victory points have no flag and are never played.
        owned = [ps.get(f"{key}{d}_OWNED_AT_START", True) for d in CAT_DEV]
        players.append({
            "hand": [ps[f"{key}{r}_IN_HAND"] for r in RESOURCES],
            "dev_hand": dev_hand,
            "dev_new": [0 if owned[k] else dev_hand[k] for k in range(5)],
            "dev_played": [ps[f"{key}PLAYED_{d}"] for d in CAT_DEV],
            "knights_played": ps[f"{key}PLAYED_KNIGHT"],
            "settlements": sorted(CAT_NODE_TO_OUR_NODE[n] for n, (c, kind) in board.buildings.items()
                                  if c == color and kind == SETTLEMENT),
            "cities": sorted(CAT_NODE_TO_OUR_NODE[n] for n, (c, kind) in board.buildings.items()
                             if c == color and kind == CITY),
            "roads": sorted({our_edge(e) for e, c in board.roads.items() if c == color}),
            "discard_remaining": st.discard_counts[i] if st.is_discarding else 0,
        })
    phase, extra = _phase(game)
    rolled = ps[f"P{cur}_HAS_ROLLED"]
    snap = {
        "board": board_dict(board.map),
        "robber": CUBE_TO_OUR_TILE[board.robber_coordinate],
        "bank": list(st.resource_freqdeck),
        "dev_deck": [DEV_NAMES[CAT_DEV.index(c)] for c in reversed(st.development_listdeck)],
        "players": players,
        "current": cur,
        "phase": phase,
        "setup_road_node": None,
        "roads_left": None,
        "winner": None,
        "return_phase": "pre_roll" if phase in ("move_robber", "road_building") and not rolled else "main",
        "turn": 0 if st.is_initial_build_phase else st.num_turns - SETUP_TURNS,
        "setup_step": len(board.roads) // 2 if st.is_initial_build_phase else 8,
        "dev_played_this_turn": ps[f"P{cur}_HAS_PLAYED_DEVELOPMENT_CARD_IN_TURN"],
        "offers_this_turn": 0,  # Catanatron's bots never offer domestic trades
        "longest_road_owner": None if board.road_color is None else seat(st, board.road_color),
        "largest_army_owner": next((i for i in range(len(st.colors)) if ps[f"P{i}_HAS_ARMY"]), None),
    }
    snap.update(extra)
    return snap


def _playable(p: dict) -> list[bool]:
    return [p["dev_hand"][k] > p["dev_new"][k] for k in (0, 2, 3, 4)]


def compare(ours: dict, theirs: dict) -> list[str]:
    """Paths where two snapshots disagree. Fields the engines represent differently are compared
    by meaning: the deck as a multiset (draw order beyond the next forced card is irrelevant),
    bought-this-turn cards by playability, and the return phase only where it is used."""
    diffs = [k for k in ("board", "robber", "bank", "current", "phase", "turn", "setup_step",
                         "longest_road_owner", "largest_army_owner", "dev_played_this_turn")
             if ours[k] != theirs[k]]
    used = {"setup_road": "setup_road_node", "road_building": "roads_left", "game_over": "winner"}
    if ours["phase"] == theirs["phase"] and ours["phase"] in used:
        k = used[ours["phase"]]
        if ours[k] != theirs[k]:
            diffs.append(k)
    if ours["phase"] in ("move_robber", "road_building", "steal") and ours["return_phase"] != theirs["return_phase"]:
        diffs.append("return_phase")
    if Counter(ours["dev_deck"]) != Counter(theirs["dev_deck"]):
        diffs.append("dev_deck")
    for i, (a, b) in enumerate(zip(ours["players"], theirs["players"])):
        for k in ("hand", "dev_hand", "dev_played", "knights_played", "settlements", "cities",
                  "roads", "discard_remaining"):
            if a[k] != b[k]:
                diffs.append(f"players[{i}].{k}")
        if _playable(a) != _playable(b):
            diffs.append(f"players[{i}].playable_dev")
    return diffs
```

Verify these Catanatron details against the reference doc and the installed source. Adjust the code if a name differs, and record each adjustment in the report:
- `state.discard_counts`, `free_roads_available`, `is_road_building` and `current_turn_index` during setup;
- the `_OWNED_AT_START` keys (booleans);
- `board.map`, which is the `CatanMap` (`board_dict` needs `land_tiles` and `ports_by_id`).

`test_initial_position_imports_and_matches` and `test_forced_outcomes_translate` exercise all of them.

- [ ] **Step 5: Run tests**

Run: `.venv/bin/python3 -m pytest tests/python -q`
Expected: everything passes, including 13 new oracle tests (with parametrization).

- [ ] **Step 6: Commit (queued while the user is away)**

```bash
git add pyproject.toml oracle tests/python/test_oracle_translate.py tests/python/test_oracle_isolation.py
git commit -m "oracle: Catanatron mapping tables and board/action/position translation"
```

---

### Task 5: Differential runner, allowlist, and `alphasettler oracle-diff`

**Files:**
- Create: `oracle/allowlist.py`, `oracle/diff.py`, `tests/python/test_oracle_diff.py`
- Modify: `alphasettler/cli.py`, `tests/python/test_cli.py`

**Read first:** reference §4 (rule-difference table) and §7 (prototype replay and its findings).

**Interfaces:**
- Consumes: `oracle.translate.*`; `alphasettler.Game`.
- Produces:
  - `oracle.allowlist`:
    - `Entry(id, rule, reason)`, a frozen dataclass
    - `ENTRIES: dict[str, Entry]`
    - `classify_legal(token, before: dict, actor: int, only_in: str) -> str | None`, where `only_in` is `"catanatron"` or `"ours"`
    - `classify_state(diffs: list[str], action_type, before: dict, ours: dict, theirs: dict) -> str | None`
  - `oracle.diff`:
    - `GameResult(seed, steps, allowlisted: Counter, mismatch: str | None, trace: list[str], final_phase: str | None, winner: int | None)`
    - `config() -> dict`
    - `run_game(seed: int) -> GameResult`
    - `run(seeds, workers: int | None = None) -> list[GameResult]`
    - `Summary(games, steps, clean_games, allowlisted: Counter, mismatches: list[GameResult])`
    - `summarize(results) -> Summary`
    - `format_summary(s: Summary) -> str`
- CLI: `alphasettler oracle-diff --games N [--seed-start S] [--workers W] [--out PATH]`.
  - Exit 0 when there are no mismatches.
  - Exit 1 when there are mismatches; print up to 5, with seed, step, detail and the last trace lines.
  - Exit 2 when Catanatron is missing, printing `error: the Catanatron oracle is not installed; install it with: .venv/bin/pip3 install "<CATANATRON_REQUIREMENT>"`.
  - `--out` writes one JSON line per game (seed, steps, allowlisted, mismatch).

- [ ] **Step 1: Write the failing tests**

`tests/python/test_oracle_diff.py`:

```python
import subprocess
import sys
from collections import Counter

import pytest

pytest.importorskip("catanatron")

from oracle import allowlist  # noqa: E402
from oracle.diff import format_summary, run, run_game, summarize  # noqa: E402


def test_thousand_games_match_up_to_allowlisted_differences():
    results = run(range(1000))
    s = summarize(results)
    detail = "\n".join(f"seed {r.seed}: {r.mismatch}\n  " + "\n  ".join(r.trace[-5:]) for r in s.mismatches[:3])
    assert not s.mismatches, detail
    assert s.games == 1000
    assert s.steps > 500_000
    assert s.clean_games > 600  # most games need no allowlist entry at all
    assert "games" in format_summary(s)


def test_runs_are_reproducible_across_worker_counts():
    key = lambda rs: [(r.seed, r.steps, sorted(r.allowlisted.items()), r.mismatch) for r in rs]
    assert key(run(range(30), workers=2)) == key(run(range(30), workers=3))


def test_every_allowlisted_kind_is_documented():
    s = summarize(run(range(200)))
    for kind in s.allowlisted:
        entry = allowlist.ENTRIES[kind]
        assert entry.reason and entry.rule


def test_turn_limit_draws_match(monkeypatch):
    import catanatron.game

    monkeypatch.setattr(catanatron.game, "TURNS_LIMIT", 40)
    r = run_game(11)
    assert r.mismatch is None, r.mismatch
    assert r.final_phase == "game_over" and r.winner is None


def test_bank_shortage_classifier():
    before = {"bank": [0, 0, 0, 0, 1], "players": [{"hand": [0] * 5} for _ in range(4)]}
    ours = {"bank": [0, 0, 0, 0, 0], "players": [{"hand": [0, 0, 0, 0, 1]}] + [{"hand": [0] * 5}] * 3}
    theirs = {"bank": [0, 0, 0, 0, 1], "players": [{"hand": [0] * 5} for _ in range(4)]}
    from catanatron.models.enums import ActionType

    kind = allowlist.classify_state(["bank", "players[0].hand"], ActionType.ROLL, before, ours, theirs)
    assert kind == "bank-shortage"
    assert allowlist.classify_state(["bank", "players[0].hand"], ActionType.END_TURN, before, ours, theirs) is None


def test_cli_oracle_diff_small_run(tmp_path):
    out = tmp_path / "diff.jsonl"
    r = subprocess.run([sys.executable, "-m", "alphasettler", "oracle-diff", "--games", "8", "--workers", "2",
                        "--out", str(out)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "8 games" in r.stdout and "0 mismatches" in r.stdout
    assert len(out.read_text().splitlines()) == 8
```

Append to `tests/python/test_cli.py`:

```python
def test_cli_without_catanatron_explains_how_to_install(tmp_path):
    # Hide Catanatron from a child interpreter by shadowing it with a package that fails to import.
    fake = tmp_path / "catanatron"
    fake.mkdir()
    (fake / "__init__.py").write_text("raise ImportError('hidden for this test')\n")
    env = {**__import__("os").environ, "PYTHONPATH": str(tmp_path)}
    r = subprocess.run([sys.executable, "-m", "alphasettler", "oracle-diff", "--games", "1"],
                       capture_output=True, text=True, env=env)
    assert r.returncode == 2
    assert "not installed" in r.stderr and "pip3 install" in r.stderr
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python3 -m pytest tests/python/test_oracle_diff.py tests/python/test_cli.py -q`
Expected: FAIL: `ModuleNotFoundError: No module named 'oracle.diff'`, and the CLI test fails on the unknown `oracle-diff` subcommand.

- [ ] **Step 3: Implement**

`oracle/allowlist.py`:

```python
"""Intended rule differences between AlphaSettler (which follows the Catan rulebook) and
Catanatron 3.3. Each entry says what differs and why we keep our behavior; the classifiers
recognise a divergence as one of them. Anything they do not recognise is a mismatch."""

from __future__ import annotations

from dataclasses import dataclass

from catanatron.models.enums import ActionType

from oracle.translate import build_road, edge_nodes


@dataclass(frozen=True)
class Entry:
    id: str
    rule: str
    reason: str


ENTRIES = {e.id: e for e in [
    Entry("longest-road", "Longest road length and award",
          "Catanatron does not count a road segment that ends at an opponent's building, does not "
          "recompute lengths in every case after a cut, and after a cut gives the award to the "
          "longest road even below 5. The rulebook lets a trail end at an opponent's building "
          "and requires 5 roads for the award; we follow it."),
    Entry("bank-shortage", "Production when the bank runs short",
          "When the bank cannot pay every claimant of a resource, Catanatron pays no one, even a "
          "single claimant. The rulebook pays a single claimant what the bank has left."),
    Entry("year-of-plenty-one-card", "Year of Plenty with a short bank",
          "Catanatron offers taking one card when the bank lacks two; we offer only pairs the "
          "bank can pay."),
    Entry("road-from-opponent-building", "Building a road through an opponent's settlement",
          "After a cut, Catanatron can let a road continue from an opponent's settlement. The "
          "rulebook forbids building through it."),
    Entry("win-off-turn", "Winning on another player's turn",
          "Catanatron declares any player at 10 VP the winner immediately; the rulebook only lets "
          "a player win on their own turn."),
]}


def _opponent_building_nodes(snap: dict, actor: int) -> set[int]:
    return {n for i, p in enumerate(snap["players"]) if i != actor for n in p["settlements"] + p["cities"]}


def classify_legal(token, before: dict, actor: int, only_in: str) -> str | None:
    """Which entry explains a token in only one engine's legal set (None: unexplained).
    ``only_in`` is "catanatron" or "ours"; every entry so far is a Catanatron-only action."""
    if only_in != "catanatron":
        return None
    if isinstance(token, tuple) and token[0] == "year_of_plenty_one_card":
        return "year-of-plenty-one-card"
    if isinstance(token, int) and build_road(0) <= token < build_road(72):
        if set(edge_nodes(token - build_road(0))) & _opponent_building_nodes(before, actor):
            return "road-from-opponent-building"
    return None


def classify_state(diffs, action_type, before: dict, ours: dict, theirs: dict) -> str | None:
    """Which entry explains the positions differing after one action (None: unexplained)."""
    d = set(diffs)
    if (theirs.get("phase") == "game_over" and ours.get("phase") != "game_over"
            and theirs.get("winner") != theirs.get("current") and d <= {"phase", "winner", "longest_road_owner"}):
        return "win-off-turn"
    if ("longest_road_owner" in d and d <= {"longest_road_owner", "phase", "winner"}
            and action_type in (ActionType.BUILD_ROAD, ActionType.BUILD_SETTLEMENT)):
        return "longest-road"
    if action_type == ActionType.ROLL and d and all(k == "bank" or k.endswith(".hand") for k in d):
        for r in range(5):
            if ours["bank"][r] == theirs["bank"][r]:
                continue
            gainers = [i for i, p in enumerate(ours["players"]) if p["hand"][r] > before["players"][i]["hand"][r]]
            paid_nobody = theirs["bank"][r] == before["bank"][r]
            if not (ours["bank"][r] == 0 and paid_nobody and len(gainers) == 1):
                return None
        return "bank-shortage"
    return None
```

`oracle/diff.py`:

```python
"""Lockstep differential test against Catanatron.

Each game is played by Catanatron random players. Before every action, Catanatron's legal
actions are compared with ours (as tokens; see oracle.translate). The action is executed in
Catanatron, and its realized chance outcome is forced on ours. Then the full positions are
compared. A divergence an allowlist entry explains is counted, and our engine re-imports
Catanatron's position so the comparison continues. Anything else ends the game as a mismatch.
Runs use spawn workers with PYTHONHASHSEED=0, because Catanatron's move order depends on the
hash seed.
"""

from __future__ import annotations

import multiprocessing as mp
import os
from collections import Counter
from dataclasses import dataclass, field

import catanatron.game as catan_game
from catanatron import Color, Game as CatanGame, RandomPlayer

from alphasettler import Game
from oracle import allowlist
from oracle.translate import (
    SETUP_TURNS, Untranslatable, catan_snapshot, catan_tokens, compare, our_steps, our_tokens, seat,
)

COLORS = [Color.RED, Color.BLUE, Color.ORANGE, Color.WHITE]
TRACE_LEN = 20


@dataclass
class GameResult:
    seed: int
    steps: int = 0
    allowlisted: Counter = field(default_factory=Counter)
    mismatch: str | None = None
    trace: list[str] = field(default_factory=list)
    final_phase: str | None = None
    winner: int | None = None


def config() -> dict:
    return {"max_offers_per_turn": 0, "catanatron_compat": False,
            "max_turns": catan_game.TURNS_LIMIT - SETUP_TURNS}


def run_game(seed: int) -> GameResult:
    res = GameResult(seed)
    cfg = config()
    cat = CatanGame([RandomPlayer(c) for c in COLORS], seed=seed)
    by_color = {p.color: p for p in cat.state.players}
    ours = Game.from_snapshot(seed, catan_snapshot(cat), cfg)
    theirs = catan_snapshot(cat)
    while theirs["phase"] != "game_over":
        st = cat.state
        actor = seat(st, st.current_color())
        if ours.current_actor != actor:
            res.mismatch = f"step {res.steps}: actor is {ours.current_actor} in ours, {actor} in Catanatron"
            return res
        before = ours.snapshot()
        mine, catans = our_tokens(ours), catan_tokens(st, cat.playable_actions)
        for token in sorted(mine ^ catans, key=repr):
            side = "catanatron" if token in catans else "ours"
            kind = allowlist.classify_legal(token, before, actor, side)
            if kind is None:
                res.mismatch = f"step {res.steps}: {token!r} is legal only in {side}"
                return res
            res.allowlisted[kind] += 1
        record = cat.execute(by_color[st.current_color()].decide(cat, cat.playable_actions))
        res.steps += 1
        res.trace = (res.trace + [repr(record)])[-TRACE_LEN:]
        theirs = catan_snapshot(cat)
        try:
            for a, chance in our_steps(cat.state, record):
                if a not in ours.legal_actions():
                    kind = allowlist.classify_legal(a, before, actor, "catanatron")
                    raise Untranslatable(kind or "illegal-in-ours", f"action {a}")
                if chance is None:
                    ours.apply(a)
                else:
                    ours.apply_forced(a, chance)
        except Untranslatable as e:
            if e.kind not in allowlist.ENTRIES:
                res.mismatch = f"step {res.steps}: {e}"
                return res
            res.allowlisted[e.kind] += 1
            ours = Game.from_snapshot(seed, theirs, cfg)
            continue
        now = ours.snapshot()
        diffs = compare(now, theirs)
        if diffs:
            kind = allowlist.classify_state(diffs, record.action.action_type, before, now, theirs)
            if kind is None:
                res.mismatch = f"step {res.steps} after {record.action}: {diffs}"
                return res
            res.allowlisted[kind] += 1
            ours = Game.from_snapshot(seed, theirs, cfg)
    res.final_phase, res.winner = theirs["phase"], theirs["winner"]
    return res


def run(seeds, workers: int | None = None) -> list[GameResult]:
    os.environ["PYTHONHASHSEED"] = "0"  # read by the spawned workers at startup
    seeds = list(seeds)
    with mp.get_context("spawn").Pool(workers or os.cpu_count() or 1) as pool:
        return pool.map(run_game, seeds, chunksize=max(1, len(seeds) // ((workers or os.cpu_count() or 1) * 8)))


@dataclass
class Summary:
    games: int
    steps: int
    clean_games: int
    allowlisted: Counter
    mismatches: list[GameResult]


def summarize(results) -> Summary:
    results = list(results)
    total = Counter()
    for r in results:
        total.update(r.allowlisted)
    return Summary(
        games=len(results),
        steps=sum(r.steps for r in results),
        clean_games=sum(1 for r in results if not r.allowlisted and r.mismatch is None),
        allowlisted=total,
        mismatches=[r for r in results if r.mismatch is not None],
    )


def format_summary(s: Summary) -> str:
    lines = [f"{s.games} games, {s.steps} steps compared, {s.clean_games} games identical end to end, "
             f"{len(s.mismatches)} mismatches"]
    for kind, n in sorted(s.allowlisted.items()):
        lines.append(f"  allowlisted {kind}: {n} ({allowlist.ENTRIES[kind].rule})")
    return "\n".join(lines)
```

`alphasettler/cli.py`: add the subcommand.

```python
    o = sub.add_parser("oracle-diff", help="differential test against Catanatron (needs the oracle extra)")
    o.add_argument("--games", type=_positive_int, required=True)
    o.add_argument("--seed-start", type=int, default=0)
    o.add_argument("--workers", type=_positive_int, default=None)
    o.add_argument("--out", default=None, help="JSONL path, one line per game")
```

The handler in `main()`:

```python
        elif args.command == "oracle-diff":
            return _oracle_diff(args)
```

with:

```python
def _oracle_missing() -> int:
    from oracle import CATANATRON_REQUIREMENT

    print("error: the Catanatron oracle is not installed; install it with: "
          f'.venv/bin/pip3 install "{CATANATRON_REQUIREMENT}"', file=sys.stderr)
    return 2


def _oracle_diff(args) -> int:
    try:
        from oracle.diff import format_summary, run, summarize
    except ImportError:
        return _oracle_missing()
    results = run(range(args.seed_start, args.seed_start + args.games), args.workers)
    s = summarize(results)
    print(format_summary(s))
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w") as f:
            for r in results:
                f.write(json.dumps({"seed": r.seed, "steps": r.steps, "allowlisted": dict(r.allowlisted),
                                    "mismatch": r.mismatch}) + "\n")
        print(f"wrote {out}")
    for r in s.mismatches[:5]:
        print(f"mismatch seed {r.seed}: {r.mismatch}", file=sys.stderr)
        for line in r.trace[-5:]:
            print(f"    {line}", file=sys.stderr)
    return 1 if s.mismatches else 0
```

(`import json` at the top of cli.py). Keep the existing `except ValueError` exit-2 handling around every command.

- [ ] **Step 4: Run tests**

```bash
VIRTUAL_ENV=$PWD/.venv .venv/bin/maturin develop --release
.venv/bin/python3 -m pytest tests/python -q
```

Expected: all pass. Record the 1000-game test's wall time and the `format_summary` output in the report: run `.venv/bin/alphasettler oracle-diff --games 1000` once.

**If mismatches appear,** each is either:
- **an engine bug:** report BLOCKED with the seed and trace; it must be fixed in the engine, not allowlisted;
- **a Catanatron rule difference missing from `ENTRIES`:** the same report. The controller decides whether a new, documented entry is justified;
- **a translation bug in `oracle/`:** fix it.

Never widen a classifier just to make a mismatch disappear.

- [ ] **Step 5: Commit (queued while the user is away)**

```bash
git add oracle alphasettler/cli.py tests/python/test_oracle_diff.py tests/python/test_cli.py
git commit -m "oracle: lockstep differential test against Catanatron, allowlist, oracle-diff CLI"
```

---

### Task 6: Catanatron arena

**Files:**
- Create: `oracle/arena.py`, `tests/python/test_oracle_arena.py`
- Modify: `alphasettler/cli.py`

**Read first:** reference §2.3 (forcing the seating), §5 (players, custom `Player`, AlphaBeta quirks).

**Interfaces:**
- Consumes:
  - `alphasettler.{Bot, Game}`
  - `oracle.translate.{action_token, catan_snapshot, seat, move_robber, steal_from, SETUP_TURNS}`
  - `alphasettler.arena.write_jsonl`, `alphasettler.stats.summarize`, `alphasettler.arena.format_summary`
- Produces:
  - `oracle.arena.BASELINES: dict[str, type]`, with keys `random`, `weighted`, `value`, `alphabeta`
  - `AlphaSettlerPlayer(color, bot_name, seed)`, which has a `fallbacks: int` counter
  - `play_game(seed, candidate, baseline, candidate_seat) -> dict`
  - `run_match(candidate, baseline, seeds, seed_start=0, workers=None) -> list[dict]`
- Records use the native shape plus `"fallbacks"`: `seed, candidate_seat, winner, vp (ACTUAL_VICTORY_POINTS by seat), turns (num_turns - 6), actions (len(action_records)), fallbacks`.
- CLI: `alphasettler arena --candidate greedy --baseline catanatron:alphabeta --seeds N [--threads T]` runs this arena. `--threads` is the worker-process count. It writes the same JSONL and summary as the native arena, plus a `fallbacks: N` line. A missing Catanatron install exits 2 with the install hint, and an unknown Catanatron bot name exits 2 too.

- [ ] **Step 1: Write the failing tests**

`tests/python/test_oracle_arena.py`:

```python
import subprocess
import sys

import pytest

pytest.importorskip("catanatron")

from catanatron import Color  # noqa: E402

from alphasettler.stats import summarize  # noqa: E402
from oracle.arena import AlphaSettlerPlayer, run_match  # noqa: E402


def test_records_have_the_native_shape():
    recs = run_match("random", "random", seeds=2, workers=2)
    assert len(recs) == 8
    assert [r["candidate_seat"] for r in recs[:4]] == [0, 1, 2, 3]
    assert {"seed", "candidate_seat", "winner", "vp", "turns", "actions", "fallbacks"} <= set(recs[0])
    for r in recs:
        if r["winner"] is not None:
            assert r["vp"][r["winner"]] >= 10


def test_greedy_beats_catanatron_random():
    recs = run_match("greedy", "random", seeds=40)
    s = summarize(recs)
    assert s.z_vs_null > 3, s
    assert sum(r["fallbacks"] for r in recs) == 0


def test_alphabeta_smoke():
    recs = run_match("greedy", "alphabeta", seeds=1, workers=4)
    assert len(recs) == 4 and all(r["turns"] > 0 for r in recs)


def test_unknown_names():
    with pytest.raises(ValueError, match="unknown Catanatron bot"):
        run_match("greedy", "nope", seeds=1)
    with pytest.raises(ValueError, match="unknown bot"):
        run_match("nope", "random", seeds=1)


def test_fallback_is_counted_not_fatal():
    from catanatron import Game as CatanGame, RandomPlayer

    class EndTurnBot:
        def act(self, game):
            return 1  # EndTurn: never playable during setup

    p = AlphaSettlerPlayer(Color.RED, "random", 1)
    p.bot = EndTurnBot()
    game = CatanGame([p] + [RandomPlayer(c) for c in (Color.BLUE, Color.ORANGE, Color.WHITE)], seed=3)
    choice = p.decide(game, game.playable_actions)
    assert choice == game.playable_actions[0] and p.fallbacks == 1


def test_cli_catanatron_baseline(tmp_path):
    out = tmp_path / "c.jsonl"
    r = subprocess.run([sys.executable, "-m", "alphasettler", "arena", "--candidate", "greedy",
                        "--baseline", "catanatron:random", "--seeds", "2", "--out", str(out)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "greedy vs catanatron:random: win rate" in r.stdout and "fallbacks: 0" in r.stdout
    assert len(out.read_text().splitlines()) == 8
    r = subprocess.run([sys.executable, "-m", "alphasettler", "arena", "--candidate", "greedy",
                        "--baseline", "catanatron:nope", "--seeds", "1"], capture_output=True, text=True)
    assert r.returncode == 2 and "unknown Catanatron bot" in r.stderr
```

`test_fallback_is_counted_not_fatal` swaps in a stub bot that always picks EndTurn, which is never playable during setup. That pins the fallback path without depending on any rule difference.

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python3 -m pytest tests/python/test_oracle_arena.py -q`
Expected: FAIL: `ModuleNotFoundError: No module named 'oracle.arena'`.

- [ ] **Step 3: Implement**

`oracle/arena.py`:

```python
"""AlphaSettler bots playing inside Catanatron against Catanatron's bots.

The candidate is rotated through every seat, as in the native arena. The board depends only
on the seed (seating is forced after Catanatron draws it, so every rotation sees the same
map). Dice, steals and Catanatron's bots share one random stream, so common random numbers
hold only for the board and the early rolls (spec, Section 3). AlphaBeta has a 20 s
wall-clock search limit, so its games are not exactly reproducible.
"""

from __future__ import annotations

import multiprocessing as mp
import os

from catanatron import Color, Game as CatanGame, Player, RandomPlayer
from catanatron.models.actions import generate_playable_actions
from catanatron.players import AlphaBetaPlayer, ValueFunctionPlayer, WeightedRandomPlayer

from alphasettler import Bot, Game
from oracle.translate import (
    SETUP_TURNS, Untranslatable, action_token, catan_snapshot, move_robber, seat, steal_from,
)

COLORS = [Color.RED, Color.BLUE, Color.ORANGE, Color.WHITE]
CONFIG = {"max_offers_per_turn": 0}
BASELINES = {
    "random": RandomPlayer,
    "weighted": WeightedRandomPlayer,
    "value": ValueFunctionPlayer,
    "alphabeta": AlphaBetaPlayer,
}


class AlphaSettlerPlayer(Player):
    """Plays an AlphaSettler bot. Each decision imports Catanatron's position into our engine,
    asks the bot (which sees only its own observation), and maps the choice back. A choice
    Catanatron does not offer, or a position that will not import, falls back to Catanatron's
    first playable action and is counted in ``fallbacks``."""

    def __init__(self, color, bot_name: str, seed: int):
        super().__init__(color)
        self.bot = Bot(bot_name, seed)
        self.seed = seed
        self.fallbacks = 0

    def decide(self, game, playable_actions):
        by_token = {}
        for a in playable_actions:
            try:
                by_token[action_token(game.state, a)] = a
            except Untranslatable:  # an action we cannot express is simply not choosable
                pass
        try:
            ours = Game.from_snapshot(self.seed, catan_snapshot(game), CONFIG)
            a = self.bot.act(ours)
            if ours.phase == "move_robber":
                after = ours.copy()
                after.apply(a)
                victim = self.bot.act(after) - steal_from(0) if after.phase == "steal" else None
                token = ("robber", a - move_robber(0), victim)
            else:
                token = a
        except ValueError:
            token = None
        choice = by_token.get(token)
        if choice is None:
            self.fallbacks += 1
            return playable_actions[0]
        return choice


def _force_seating(game, players) -> None:
    st = game.state
    st.players = list(players)
    st.colors = tuple(p.color for p in players)
    st.color_to_index = {c: i for i, c in enumerate(st.colors)}
    game.playable_actions = generate_playable_actions(st)


def play_game(seed: int, candidate: str, baseline: str, candidate_seat: int) -> dict:
    players = [
        AlphaSettlerPlayer(COLORS[i], candidate, seed * 4 + i) if i == candidate_seat else BASELINES[baseline](COLORS[i])
        for i in range(4)
    ]
    game = CatanGame(players, seed=seed)
    _force_seating(game, players)
    game.play()
    st = game.state
    winner = game.winning_color()
    return {
        "seed": seed,
        "candidate_seat": candidate_seat,
        "winner": None if winner is None else seat(st, winner),
        "vp": [st.player_state[f"P{i}_ACTUAL_VICTORY_POINTS"] for i in range(4)],
        "turns": st.num_turns - SETUP_TURNS,
        "actions": len(st.action_records),
        "fallbacks": players[candidate_seat].fallbacks,
    }


def run_match(candidate: str, baseline: str, seeds: int, seed_start: int = 0, workers: int | None = None) -> list[dict]:
    if baseline not in BASELINES:
        raise ValueError(f"unknown Catanatron bot {baseline!r}; known: {sorted(BASELINES)}")
    Bot(candidate, 0)  # raises ValueError("unknown bot ...") for a bad name
    if seeds <= 0:
        raise ValueError("seeds must be positive")
    os.environ["PYTHONHASHSEED"] = "0"
    jobs = [(s, candidate, baseline, k) for s in range(seed_start, seed_start + seeds) for k in range(4)]
    with mp.get_context("spawn").Pool(workers or os.cpu_count() or 1) as pool:
        records = pool.starmap(play_game, jobs)
    return sorted(records, key=lambda r: (r["seed"], r["candidate_seat"]))
```

`alphasettler/cli.py`, arena branch: if `args.baseline.startswith("catanatron:")`, do the following.
- Import `oracle.arena` lazily. On `ImportError`, `return _oracle_missing()`.
- Call `records = oracle.arena.run_match(args.candidate, args.baseline.split(":", 1)[1], args.seeds, args.seed_start, args.threads)`.
- Write the JSONL with `alphasettler.arena.write_jsonl(out, args.candidate, args.baseline, records)`.
- Print `format_summary(f"{args.candidate} vs {args.baseline}", summarize(records))`, then `fallbacks: {sum(r['fallbacks'] for r in records)}`, then `wrote {out}`.
- `--no-trades` is meaningless here; Catanatron bots never trade.

The default output filename replaces `:` with `-`: `runs/<time>-greedy-vs-catanatron-alphabeta.jsonl`. Keep the native branch unchanged.

- [ ] **Step 4: Run tests**

```bash
.venv/bin/python3 -m pytest tests/python -q
.venv/bin/alphasettler arena --candidate greedy --baseline catanatron:random --seeds 50
.venv/bin/alphasettler arena --candidate greedy --baseline catanatron:value --seeds 25
.venv/bin/alphasettler arena --candidate greedy --baseline catanatron:alphabeta --seeds 25
```

Expected: all tests pass. Paste the three summaries, with wall times, into the report.

If `test_greedy_beats_catanatron_random` sees fallbacks, find out which tokens our bot produced that Catanatron did not offer, and report them. Do not relax the assertion without a ruling.

- [ ] **Step 5: Commit (queued while the user is away)**

```bash
git add oracle/arena.py alphasettler/cli.py tests/python/test_oracle_arena.py
git commit -m "oracle: Catanatron arena (our bots vs Catanatron's bots, seat rotation)"
```

---

### Task 7: The 50,000-game run and the results record

**Files:**
- Create: `docs/oracle/results.md`

**Interfaces:**
- Consumes: the `alphasettler oracle-diff` and `alphasettler arena` CLIs.

- [ ] **Step 1: Run the long differential test**

```bash
mkdir -p runs
time .venv/bin/alphasettler oracle-diff --games 50000 --workers "$(sysctl -n hw.ncpu)" --out runs/oracle-diff-50k.jsonl
```

Expected: exit 0, `0 mismatches`.

If any mismatch appears, stop and report BLOCKED with the first five seeds and their traces. Rerun a failing seed alone with `.venv/bin/python3 -c "from oracle.diff import run; print(run([SEED], 1)[0])"`.

- [ ] **Step 2: Arena results**

```bash
.venv/bin/alphasettler arena --candidate greedy --baseline random --seeds 2000
.venv/bin/alphasettler arena --candidate greedy --baseline catanatron:random --seeds 200
.venv/bin/alphasettler arena --candidate greedy --baseline catanatron:value --seeds 100
.venv/bin/alphasettler arena --candidate greedy --baseline catanatron:alphabeta --seeds 100
```

- [ ] **Step 3: Write `docs/oracle/results.md`**

Record the following, with the numbers copied from the runs above (no placeholders):
- date and machine (`sysctl -n machdep.cpu.brand_string`);
- the AlphaSettler commit as `git log -1 --format=%h`. If commits are still queued because the user is away, write "uncommitted working tree on top of <last commit>";
- the Catanatron commit and version;
- the parity config;
- for the 1000-game and 50,000-game runs: games, steps compared, games identical end to end, mismatches (0), and allowlisted counts per entry with each entry's rule. Link `oracle/allowlist.py` for the reasons;
- the four arena summaries with wall time and fallbacks;
- a checklist of the spec's Section 4 done criteria, each with its evidence (test name, command, or this file's numbers):
  - rule and property tests;
  - the 1k and 50k differential runs;
  - performance targets: link `docs/perf/comparison.md`;
  - the arena native and against AlphaBeta;
  - greedy over random with z > 3.

- [ ] **Step 4: Commit (queued while the user is away)**

```bash
git add docs/oracle/results.md
git commit -m "docs: differential and Catanatron-arena results; spec done-criteria checklist"
```

---

## Spec coverage (self-review)

| Spec requirement | Task |
|---|---|
| `oracle/` separate package, the only code importing Catanatron; installing alphasettler never pulls it in | 4 (extra + isolation tests) |
| Differential: Catanatron random games; each action + chance outcome forced on ours; compare legal sets and state every step | 4 (translation), 5 (lockstep runner) |
| Sidestep Catanatron randomness | 5: realized outcomes from `ActionRecord`s are forced, and PYTHONHASHSEED is pinned |
| Allowlist of intentional differences, each with its reason | 5 (`oracle/allowlist.py`) |
| 1,000 games in the default suite; 50,000 on demand, run once | 5 (test), 7 (run + record) |
| Catanatron arena vs AlphaBeta and value-function bots, via `oracle/` | 6 |
| `alphasettler arena` prints win rate with CI natively and against AlphaBeta | Plan 2 Task 6, 6 |
| Done-criteria evidence | 7 |
| Standard rules (rulebook) | 1 (pre-roll dev cards, Road Building playability) |
