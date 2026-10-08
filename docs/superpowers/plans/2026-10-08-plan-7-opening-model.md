# Plan 7: Opening Diagnosis and Opening Model (Sub-project 3d-1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Find out whether `gen-034` is forced into its weak openings, misjudges them or misdecides them, and test whether a separate opening model that picks setup settlements (`+o`) beats `net:gen-034@100+t` in the trading game.

**Architecture:** Rust gains the 78 opening features (`bots/src/opening.rs`), the `ASOPEN` model file, its MLP forward pass and the `+o` choice rule (`bots/src/opening_model.rs`, wired into `IsmctsBot`, the bot-name parser and the `Roster`), and `settler_net::evaluator::position_value`. The bindings expose the features, the model's forward pass and `net_evaluate`, and `Bot.act` releases the GIL. Python adds the setup replay and the model writer (`alphasettler/opening.py`), training on PyTorch (`alphasettler/train/opening.py`, `alphasettler opening-train`) and the diagnosis D1–D4 (`alphasettler/diagnose.py`, `alphasettler opening-diagnose`). The controller then runs the diagnosis, trains the model, runs the SPRT and writes `docs/bot/results-3d1.md`.

**Tech Stack:** Rust 1.80+ (edition 2021, no new crates), PyO3 0.29.3 + maturin 1.15.0, Python 3.12 stdlib, PyTorch 2.14.1 (`train` extra only), pytest 9.1.1.

**Spec:** `docs/superpowers/specs/2026-10-08-opening-model-design.md` (binding). Builds on Plan 6 (`docs/superpowers/plans/2026-10-07-plan-6-trading.md`: `+t`, the trading game) and commit `ca896ee` (`sprt --trades`).

## Global Constraints

- Python is `.venv/bin/python3`, tests `.venv/bin/pytest`, the CLI `.venv/bin/alphasettler`. Never `python`, `pip`, `python3` outside the venv, or `pip3` outside the venv.
- Rust is `PATH=$HOME/.cargo/bin:$PATH cargo ...` (cargo is not on PATH; `~/.cargo/bin/cargo` is the same binary). MSRV 1.80 (`rust-version = "1.80"`): use nothing stabilised after 1.80 (no `Option::is_none_or`, no `is_multiple_of`). No new external crates and no new Python dependencies. No numpy anywhere.
- Rebuild the extension after any Rust change used from Python. Another measurement holds the extension: **never run `maturin develop`**. Build a wheel and rename the new library over the old one (an atomic rename, so running processes keep their mapped copy):
  ```bash
  rm -rf target/wheels-3d1 && PATH=$HOME/.cargo/bin:$PATH .venv/bin/maturin build --release -q -o target/wheels-3d1 \
    && unzip -q -o target/wheels-3d1/alphasettler-*.whl 'alphasettler/_engine*.so' -d target/wheels-3d1/x \
    && cp target/wheels-3d1/x/alphasettler/_engine.cpython-312-darwin.so alphasettler/._engine.new.so \
    && mv -f alphasettler/._engine.new.so alphasettler/_engine.cpython-312-darwin.so
  ```
  "Rebuild the extension" in any step below means exactly this command.
- Every task ends with `PATH=$HOME/.cargo/bin:$PATH cargo fmt --all` then `PATH=$HOME/.cargo/bin:$PATH cargo fmt --check` (no output), `PATH=$HOME/.cargo/bin:$PATH cargo clippy --workspace --release --all-targets -- -D warnings` (exit 0), the full `PATH=$HOME/.cargo/bin:$PATH cargo test --workspace --release` (all pass) and, after the rebuild, `.venv/bin/pytest -q -rs` (all pass, the summary lists no skipped tests).
- Nothing under `engine/` or `search/` is edited. `engine/tests/golden_trace.rs` passes unchanged; 3a's digests (`the_generic_refactor_leaves_3a_ismcts_games_unchanged`, `fixed_exploration_is_3a_self_play`) are never re-pinned.
- Opening features (spec 4.2): `OPENING_FEATURES` = 78, seat-relative to the seat `p` that just picked (r0 = `p`, then the others in turn order); computed in Rust (`bots/src/opening.rs`) by one function used by both training (through the binding) and play. `OPENING_HASH` hashes the feature list's names and order.
- Opening model (spec 4.3): an MLP, 78 inputs, two hidden layers of 64 (ReLU), 4 outputs, softmax over seats (seat-relative). File `<nets_dir>/opening.bin`: magic `ASOPEN\0\0`, format version 1, `OPENING_HASH`, layer count, each layer's rows and columns, then each layer's f32 weights (row-major, rows = outputs) and biases, little-endian. The Rust loader checks magic, version, hash, shapes (78 in, 4 out, each layer's inputs equal to the previous outputs), exact length and that every float is finite, and refuses anything else with an error naming the file.
- Training (spec 4.4): eight samples a game (after each settlement, before its road: the features for the seat that placed it; target = the winner relative to that seat; a game with no winner adds nothing). Data `runs/3b-v2/gen-001` .. `gen-005` and `runs/3c/gen-030` .. `gen-034`, never `runs/3b-v2/gen-000`. Held out: seed % 20 == 0. Cross-entropy, Adam, lr 1e-3, batch 1,024, until the held-out loss has not improved for 3 epochs; the best held-out epoch is written. Report: held-out cross-entropy overall and on the trading games only, beside a uniform guess (ln 4 = 1.386) and `gen-034`'s value head on the same positions.
- `+o` (spec 4.5): a `net:` name suffix like `+t`, `+sK`, `+rD` (any order, at most once); `net:NAME@N+o` loads `<nets_dir>/opening.bin` beside `NAME.bin`. At a setup-settlement decision the bot scores each legal settlement with the model from its own seat and plays the highest own-win probability, ties to the first in engine order, with no search. Every other decision is unchanged. Self-play does not use `+o` (`bots/src/selfplay.rs` is not edited).
- The trading game is exactly `{"max_offers_per_turn": 2, "max_trade_cards": 1}` (`TRADES` in `alphasettler/arena.py`, `trading()` in Rust tests).
- The test (spec Section 2.3): `net:gen-034@100+t+o` against three `net:gen-034@100+t`, `alphasettler sprt --trades` with p0 0.25, p1 0.30, alpha = beta = 0.05, fresh seeds from 2,000,000,000. If H1, a fixed 200-seed run at 600 simulations (`arena --trades`) on seeds no SPRT used, reported with its z.
- Determinism: same seed and weights give the same games and tables, independent of thread count. Budgets are simulation counts, never wall time.
- No module outside `alphasettler/train/` imports torch. `alphasettler/opening.py` and `alphasettler/diagnose.py` use the standard library and the extension only.
- Git: one commit per task with the message given in the task, run as `git -c user.email=sean.may101@gmail.com commit -m "..."`. No Co-Authored-By or AI attribution, never `--no-verify`, never push. Before every commit run `git diff --cached | grep -nE 'sk-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{36}|password=[^[:space:]]+'` and expect no output.
- Testing: integration tests over mocks (real games, real record files, real model files read by the real Rust loader); test behaviour, not implementation; independent tests (each builds its own temp files); descriptive names; pytest fixtures and `parametrize` where cases repeat.
- Long runs (the diagnosis on the real runs, training on 63,000 games, the SPRT, the confirmation, D4 on the real net) are run by the controller in the background, never by implementer subagents. Steps marked **controller runs this** give the exact command and the pass condition.
- No README or docs beyond `docs/bot/results-3d1.md` (Task 6).

## Review Focus

1. A `+o` bot whose nets directory has no `opening.bin`, or one written for other features (hash), cut short, with a NaN, or of another version. Expected: refused when the bot is built (`Roster::new`, so `arena`, `sprt`, `Bot(...)` and `opening-diagnose --bot`), before any game, with an error naming the file; a name without `+o` never reads the file. Pinned by Task 2's `a_missing_or_bad_opening_model_is_refused_before_any_game` and Task 3's `test_a_plus_o_bot_needs_a_valid_opening_model`.
2. Bot names with `+o` repeated, malformed, misplaced or on a bot that takes no model (`+o+o`, `+o5`, `+O`, `+oo`, `net:a+o@300`, `ismcts@100+o`, `greedy+o`). Expected: refused as unknown bots before any file is read; equivalent orders parse to the same bot. Pinned by Task 2's `the_opening_suffix_parses_in_any_order_and_only_once` and the shared `tests/python/net_names.py`.
3. Records the setup replay cannot use: fewer than 16 actions, a road where a settlement belongs, an illegal setup move, no winner. Expected: no-winner games add no training sample and no D2 row and are counted; anything else is a ValueError naming the file and the game's seed before any training or table. Pinned by Task 4's `test_replay_setup_refuses_records_that_do_not_replay_a_setup` and `test_samples_are_eight_per_decided_game_with_the_winner_relative_to_the_seat`.
4. A diagnosis whose input leaves a group empty or with one seat (a tiny directory, the comparison directory) or holds no decided game. Expected: the row prints `-` (no ZeroDivisionError), a one-seat group's standard error is `inf`, and the verdict reads `undecided` instead of crashing. Pinned by Task 5's `test_tables_print_empty_groups_as_dashes` and `test_d1_shares_are_cumulative_and_count_openings_with_no_alternative`.
5. A model that scores several spots exactly the same (a zero model), and features asked for outside the setup or for a bad seat. Expected: the first legal settlement in engine order; a ValueError naming the phase or the seat. Pinned by Task 2's `a_model_that_ties_picks_the_first_legal_spot` and Task 3's `test_opening_features_refuse_positions_outside_the_setup_and_bad_seats`.

## File Structure

| File | Responsibility | Task |
|---|---|---|
| `bots/src/opening.rs` (new) | the 78 features, block offsets, `OPENING_LAYOUT`, `OPENING_HASH`, `legal_spots`, `spot_pips` | 1 |
| `bots/src/opening_model.rs` (new) | `ASOPEN` file (read, write, checks), MLP forward, `choose_settlement` | 2 |
| `bots/src/ismcts.rs` | `Flags`, `split_suffixes` (`+o`), `split_trades` refuses `+o`; `IsmctsBot::with_opening`, the `+o` path in `act` | 2 |
| `bots/src/netbot.rs` | `NetName` struct from `parse_net_name` (replaces the 5-tuple) | 2 |
| `bots/src/lib.rs` | the two modules, `Roster` loads `opening.bin` for `+o`, `unknown_bot` text | 1, 2 |
| `alphasettler/cli.py` | `_check_bot_name` text (2); `opening-train` (4); `opening-diagnose` (5) | 2, 4, 5 |
| `net/src/evaluator.rs` | `position_value(weights, state, viewer)` | 3 |
| `bindings/src/opening.rs` (new), `bindings/src/lib.rs`, `bindings/src/convert.rs` | `opening_features`, `opening_forward`, `net_evaluate`, constants; `Bot.act` releases the GIL; `index_arg` made public | 3 |
| `alphasettler/records.py`, `alphasettler/train/data.py` | `record_files` moves to `records` (torch-free), `data` re-exports it | 4 |
| `alphasettler/opening.py` (new) | `replay_setup`, `Pick`, `Setup`, `model_bytes`, `write_model` | 4 |
| `alphasettler/train/opening.py` (new) | `OpeningNet`, `Samples`, `load_samples`, `fit`, `report`, `train_opening`, `format_report` | 4 |
| `alphasettler/diagnose.py` (new) | groups and pips, `Diagnosis` (D1, D2), `audit` (D3), `play_setups` (D4), tables, `verdict`, `diagnose` | 5 |
| `docs/bot/results-3d1.md` (new) | the measurements | 6 |
| tests: `bots/tests/opening.rs` (new, 1), `bots/tests/opening_model.rs` (new, 2), `bots/tests/netbot.rs` (2), `net/tests/evaluator.rs` (3), `tests/python/net_names.py` (2), `tests/python/test_opening_bindings.py` (new, 3), `tests/python/opening_games.py` (new, 4), `tests/python/test_opening_train.py` (new, 4), `tests/python/test_opening_diagnose.py` (new, 5) | | 1–5 |

## Where the code forces a choice the spec does not make

Each is a **Ruling** (with its reason) the results file repeats, so it can be overridden.

- **After the setup the phase is `PreRoll`, not `Main`** (`engine/src/rules/setup.rs`: the 16th action sets `PreRoll`). D2 reads the position right after the 16th setup action: setup done, before the first roll, which is what the spec's "first `main`-phase position" describes.
- **`net_evaluate` reads the hidden block from the game itself** (`encode_state(state, viewer)`, exactly what a search leaf whose world is the real game gets). After the setup every card is public (setup production is announced), so this is the viewer's own view at every position the spec uses it on. `net` is the bare NAME (`gen-034`), as in `net:NAME`; a bad name or file raises the same errors `Bot(...)` raises for it (`bad net name`, `cannot read`, `encoding hash`).
- **D2 reads each seat's own view** (`net_evaluate(after_setup, seat)[seat]`), the number that seat's search starts from, and skips games with no winner (four predictions summing to 1 against four losses would bias every group).
- **Pips sum per settlement**: a tile two of a seat's settlements touch counts twice, because it pays twice. Same in the Rust features, D1, D4 and the groups' "mean pips".
- **"Road distance 2" is two steps along the node graph**, whoever owns the edges in between (no road is built yet at most picks). Nodes one step away are never legal, so "at distance 2" is the same as "within 2".
- **Feature order**: the spec's table order, each per-seat block holding r0 r1 r2 r3, the port block per seat as 3:1 then 2:1 wood brick sheep wheat ore. `OPENING_LAYOUT` spells it out and is what `OPENING_HASH` hashes.
- **`opening_features` refuses positions outside the setup** (`setup_settlement` or `setup_road`): later, "settlements placed / 2" would exceed 1 and the draft block means nothing.
- **`+o` is for `net:` names only.** `ismcts…+o` is an unknown bot: `make_bot` has no nets directory to read `opening.bin` from. `+sK` with `+o` parses but changes nothing, since `+o` decides every setup settlement without a search.
- **`parse_net_name` returns a `NetName` struct** instead of growing its tuple to six fields. Every existing name parses as before (the existing test is rewritten field for field).
- **Self-play samples its setup moves** (moves in proportion to visits with root noise for the first `SAMPLE_TURNS` = 8 turns, and setup is turn 0). So D1 and D2 describe the self-play policy's openings, and D3's "changed" share includes that sampling as well as the budget; D3's re-chosen openings and D4's are the bot's own most-visited choices. The results file says so beside the tables.
- **D3 draws from the last `--games` directory**: `random.Random(0).sample` of its games sorted by seed. A seat's re-chosen opening is its two re-chosen spots, each chosen from the recorded position at that pick (the earlier picks as recorded). `--audit-budgets` (default `1000 8000`) exists so the tests can run D3 with tiny budgets.
- **The verdict pools the three weak groups**: *forced* if the pooled share "within 1" is under 1/3; else *misjudged* if the pooled weak gap minus all five's gap exceeds 2 × sqrt(se_weak² + se_all_five²); else *misdecided*. With no weak opening or no decided all-five seat it reads *undecided* (only possible on toy inputs).
- **The diagnosis runs twice**: on `runs/3c/gen-030` .. `gen-034` with D3 on `gen-034` (the verdict line: the trading game is where 3d is built) and, for comparison, on `runs/3b-v2/gen-005` with `--audit 0`.
- **Training runs on the CPU** (a 10,000-parameter MLP: MPS launch overhead exceeds the work), seed 0, at most 100 epochs as a guard. The held-out rule seed % 20 == 0 applies across every directory. `--out` must not exist (as `net-upgrade`): the measured model is never overwritten. A summary goes to `OUT.json`, as `fast-train` does. The compare net's probability of the winner is floored at 1e-12 before the log.
- **`Bot.act` releases the GIL** while the bot decides, so D3 and D4 can use threads. Behaviour is unchanged (pinned by a thread-independence test).
- **D4 on "`@1000`"** is `net:gen-034@1000+t` (`+t` does nothing during the setup). D4 seeds start at 2,000,000,000 (setups only, no game is played on).
- **The SPRT's default `--max-seeds` 400 stays**; an undecided SPRT is recorded as undecided with its win rate and z, and no confirmation runs. The confirmation's fresh seeds start at 2,000,001,000, clear of any SPRT seed.
- **The loader bounds the header before sizing anything**: 1..=8 layers, every dimension 1..=4096. Hidden widths are otherwise free (tests use small models); `opening-train` writes 78 → 64 → 64 → 4.
- **Task boundaries:** the `+o` parser, the loader and the choice rule are one task (2): none is testable without the others through a bot. `position_value` goes with the bindings (3), its only caller. Training (4) precedes the diagnosis (5) so that Task 5 can reuse Task 4's `replay_setup` and the fixed test game.

---

### Task 1: The opening features

**Files:**
- Create: `bots/src/opening.rs`
- Modify: `bots/src/lib.rs` (one `pub mod` line)
- Test: `bots/tests/opening.rs` (new)

**Interfaces:**
- Consumes: `settler_engine::{Board, State, PlayerId, NUM_PLAYERS, NUM_RESOURCES, bits64}`, `settler_engine::topology::{topo, NUM_NODES, NUM_TILES}` (`topo().node_tiles`, `node_neighbors`, `node_neighbor_mask`), `Board::{tile_resource, tile_number, generic_port_nodes, specific_port_nodes}`; `settler_net::encode::{abs, layout_hash, pips}`.
- Produces (all in `settler_bots::opening`):
  - `pub const OPENING_FEATURES: usize = 78;`
  - block offsets `SEAT_PIPS = 0` (4 × 5), `SEAT_TOTAL = 20`, `SEAT_RESOURCES = 24`, `SEAT_NUMBERS = 28`, `SEAT_PORTS = 32` (4 × 6: 3:1, then 2:1 wood brick sheep wheat ore), `SEAT_PLACED = 56`, `SEAT_REACH = 60`, `PLACED_ALL = 64`, `PICK_LEFT = 65`, `BEST_RESOURCE = 66` (5), `BEST_SPOT = 71`, `RICH_SPOTS = 72`, `BOARD_RESOURCE = 73` (5); a seat block of width `w` holds r0 at `BASE`, r1 at `BASE + w`, and so on.
  - `pub const OPENING_LAYOUT: &str` and `pub const OPENING_HASH: u64 = layout_hash(&[OPENING_FEATURES as u32], OPENING_LAYOUT);`
  - `pub const PORT_KINDS: usize = 6;`, `pub const RICH_PIPS: u32 = 10;`
  - `pub fn opening_feature_names() -> Vec<String>` (`block[i].token`, vector order)
  - `pub fn spot_pips(board: &Board, n: usize) -> [u32; NUM_RESOURCES]`
  - `pub fn legal_spots(settlements: &[u64; NUM_PLAYERS]) -> u64`
  - `pub fn opening_features(board: &Board, settlements: &[u64; NUM_PLAYERS], p: PlayerId) -> [f32; OPENING_FEATURES]`
  - `pub fn state_features(s: &State, p: PlayerId) -> [f32; OPENING_FEATURES]`

- [ ] **Step 1: Write the failing tests**

Create `bots/tests/opening.rs`:

```rust
//! The opening features (spec 3d-1 Section 4.2).

use settler_bots::opening::*;
use settler_engine::rng::Rng;
use settler_engine::*;
use settler_net::encode::layout_hash;

/// Seed 0's board, five settlements in: seat 0 on node 9, seat 1 on 12, seat 2 on 18, seat 3 on
/// 41 and then 26 (its second pick, right after its first); seat 3 is to place its road. What the
/// nodes touch (resource number: pips): 9 wheat 10:3, wheat 8:5, brick 5:4; 12 ore 10:3,
/// brick 9:4, sheep 8:5; 18 sheep 8:5, brick 5:4, ore 11:2; 41 wood 2:1, wheat 9:4, sheep 6:5;
/// 26 sheep 4:3, and it holds the brick 2:1 port.
fn five_placed() -> Game {
    let mut g = Game::new(0, GameConfig::default());
    for (k, n) in [9u8, 12, 18, 41, 26].into_iter().enumerate() {
        g.apply(Action::BuildSettlement(n)).unwrap();
        if k < 4 {
            let road = g.legal_actions()[0];
            g.apply(road).unwrap();
        }
    }
    assert_eq!(
        (g.state().phase, g.state().current),
        (Phase::SetupRoad { node: 26 }, 3)
    );
    g
}

#[test]
fn features_of_a_hand_built_setup_match_hand_counted_values() {
    let g = five_placed();
    let f = state_features(g.state(), 3);
    // Seat-relative: r0 = seat 3, r1 = seat 0, r2 = seat 1, r3 = seat 2.
    let mut want: Vec<f32> = Vec::new();
    // Pips per resource (wood brick sheep wheat ore) / 15. Seat 3: node 41 (wood 1, wheat 4,
    // sheep 5) and node 26 (sheep 3).
    want.extend([1.0, 0.0, 8.0, 4.0, 0.0].map(|x: f32| x / 15.0));
    want.extend([0.0, 4.0, 0.0, 8.0, 0.0].map(|x: f32| x / 15.0));
    want.extend([0.0, 4.0, 5.0, 0.0, 3.0].map(|x: f32| x / 15.0));
    want.extend([0.0, 4.0, 5.0, 0.0, 2.0].map(|x: f32| x / 15.0));
    // Total pips / 30.
    want.extend([13.0, 12.0, 12.0, 11.0].map(|x: f32| x / 30.0));
    // Resources touched / 5.
    want.extend([3.0, 2.0, 3.0, 3.0].map(|x: f32| x / 5.0));
    // Distinct numbers / 6: seat 3 {2, 9, 6, 4}; seat 0 {10, 8, 5}; seat 1 {10, 9, 8}; seat 2
    // {8, 5, 11}.
    want.extend([4.0, 3.0, 3.0, 3.0].map(|x: f32| x / 6.0));
    // Ports (3:1, then 2:1 wood brick sheep wheat ore): only seat 3's node 26, the brick 2:1.
    want.extend([0.0, 0.0, 1.0, 0.0, 0.0, 0.0]);
    want.extend([0.0; 18]);
    // Settlements placed / 2.
    want.extend([1.0, 0.5, 0.5, 0.5]);
    // The richest legal spot two steps from the seat's settlements, pips / 15: seat 3 node 31
    // (brick 5, sheep 4, wheat 9: 11); seat 0 node 19 (brick 5, wood 11, brick 5: 10); seat 1
    // node 22 (brick 9, wood 4, wood 12: 8); seat 2 node 19 (10).
    want.extend([11.0, 10.0, 8.0, 10.0].map(|x: f32| x / 15.0));
    // Settlements placed in all / 8; seat 3 has made both its picks.
    want.extend([5.0 / 8.0, 0.0]);
    // Most pips of each resource on a legal spot / 5: wood node 22 (4), brick node 19 (8), sheep
    // node 29 (5), wheat node 2 (5), ore node 0 (3).
    want.extend([4.0, 8.0, 5.0, 5.0, 3.0].map(|x: f32| x / 5.0));
    // The richest legal spot (node 31, 11) / 15; legal spots with at least 10 pips (19, 31) / 10.
    want.extend([11.0 / 15.0, 2.0 / 10.0]);
    // The board's pips per resource / 25.
    want.extend([7.0, 12.0, 18.0, 14.0, 7.0].map(|x: f32| x / 25.0));
    assert_eq!(want.len(), OPENING_FEATURES);
    assert_eq!(f.to_vec(), want);
}

/// Every per-seat block: (offset, width).
const SEAT_BLOCKS: [(usize, usize); 7] = [
    (SEAT_PIPS, 5),
    (SEAT_TOTAL, 1),
    (SEAT_RESOURCES, 1),
    (SEAT_NUMBERS, 1),
    (SEAT_PORTS, 6),
    (SEAT_PLACED, 1),
    (SEAT_REACH, 1),
];

#[test]
fn every_perspective_sees_the_same_seats_rotated() {
    let g = five_placed();
    let f: Vec<[f32; OPENING_FEATURES]> =
        (0..4u8).map(|p| state_features(g.state(), p)).collect();
    assert_eq!(
        SEAT_BLOCKS.iter().map(|&(_, w)| 4 * w).sum::<usize>(),
        PLACED_ALL
    );
    for p in 0..4 {
        for q in 0..4 {
            for &(base, w) in &SEAT_BLOCKS {
                for r in 0..4 {
                    // Absolute seat p + r is relative seat p + r - q from q's side.
                    let rq = (p + r + 4 - q) % 4;
                    assert_eq!(
                        f[p][base + w * r..base + w * (r + 1)],
                        f[q][base + w * rq..base + w * (rq + 1)],
                        "block {base}: r{r} from seat {p}, r{rq} from seat {q}"
                    );
                }
            }
            assert_eq!(f[p][PLACED_ALL], f[q][PLACED_ALL]);
            assert_eq!(f[p][BEST_RESOURCE..], f[q][BEST_RESOURCE..]);
        }
    }
    // Only seat 3 has placed both its settlements.
    assert_eq!([0, 1, 2, 3].map(|p| f[p][PICK_LEFT]), [1.0, 1.0, 1.0, 0.0]);
}

#[test]
fn legal_spots_are_the_engines_legal_setup_settlements() {
    let mut checked = 0;
    for seed in 0..6 {
        let mut g = Game::new(seed, GameConfig::default());
        let mut rng = Rng::new(seed);
        while matches!(
            g.state().phase,
            Phase::SetupSettlement | Phase::SetupRoad { .. }
        ) {
            let legal = g.legal_actions();
            if g.state().phase == Phase::SetupSettlement {
                let engine = legal.iter().fold(0u64, |m, a| match a {
                    Action::BuildSettlement(n) => m | (1u64 << *n),
                    _ => m,
                });
                let placed = std::array::from_fn(|q| g.state().players[q].settlements);
                assert_eq!(legal_spots(&placed), engine, "seed {seed}");
                checked += 1;
            }
            g.apply(legal[rng.below(legal.len() as u32) as usize]).unwrap();
        }
    }
    assert_eq!(checked, 6 * 8);
}

#[test]
fn the_layout_names_every_feature_once_in_vector_order() {
    let names = opening_feature_names();
    assert_eq!(names.len(), OPENING_FEATURES);
    let unique: std::collections::HashSet<&String> = names.iter().collect();
    assert_eq!(unique.len(), OPENING_FEATURES);
    for (at, name) in [
        (SEAT_PIPS + 5 * 2 + 3, "seat_pips[2].wheat/15"),
        (SEAT_TOTAL + 1, "seat_total[1].pips/30"),
        (SEAT_RESOURCES, "seat_resources[0].touched/5"),
        (SEAT_NUMBERS + 3, "seat_numbers[3].distinct/6"),
        (SEAT_PORTS + 6 + 2, "seat_ports[1].brick"),
        (SEAT_PLACED + 2, "seat_placed[2].settlements/2"),
        (SEAT_REACH + 3, "seat_reach[3].best_legal_at_distance_2/15"),
        (PLACED_ALL, "placed_all[0].settlements/8"),
        (PICK_LEFT, "pick_left[0].p"),
        (BEST_RESOURCE + 4, "best_legal[0].ore/5"),
        (BEST_SPOT, "best_legal_spot[0].pips/15"),
        (RICH_SPOTS, "rich_legal_spots[0].at_least_10/10"),
        (BOARD_RESOURCE, "board[0].wood/25"),
        (OPENING_FEATURES - 1, "board[0].ore/25"),
    ] {
        assert_eq!(names[at], name, "feature {at}");
    }
}

#[test]
fn the_hash_changes_when_a_feature_is_renamed_or_moved() {
    assert_eq!(
        OPENING_HASH,
        layout_hash(&[OPENING_FEATURES as u32], OPENING_LAYOUT)
    );
    let renamed = OPENING_LAYOUT.replace("pick_left x1: p", "pick_left x1: own");
    let moved = OPENING_LAYOUT.replace("wood/5 brick/5", "brick/5 wood/5");
    let block_moved = OPENING_LAYOUT.replace(
        "seat_placed x4: settlements/2\nseat_reach x4: best_legal_at_distance_2/15",
        "seat_reach x4: best_legal_at_distance_2/15\nseat_placed x4: settlements/2",
    );
    for other in [renamed, moved, block_moved] {
        assert_ne!(other, OPENING_LAYOUT, "the edit must apply");
        assert_ne!(
            layout_hash(&[OPENING_FEATURES as u32], &other),
            OPENING_HASH
        );
    }
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PATH=$HOME/.cargo/bin:$PATH cargo test --release -p settler-bots --test opening`
Expected: FAIL to compile with `unresolved import settler_bots::opening`.

- [ ] **Step 3: Write the implementation**

In `bots/src/lib.rs`, after `pub mod netbot;` add:

```rust
pub mod opening;
```

Create `bots/src/opening.rs`:

```rust
//! The opening model's inputs (spec 3d-1 Section 4.2): 78 hand-built whole-board features of a
//! setup position for one seat `p`, seat-relative (r0 is `p`, then the others in turn order, as
//! the net's inputs name them). One function serves training (through the bindings) and play
//! (`opening_model::choose_settlement`), so the two cannot disagree.
//!
//! A spot's pips are the dice dots (`settler_net::encode::pips`) of the tiles it touches; a seat's
//! pips sum over its settlements, so a tile two of them touch counts twice (it pays twice).
//! "Legal" is the distance rule against every settlement placed. "Distance 2" is two steps along
//! the node graph, whoever owns the edges; nodes one step away are never legal.

use settler_engine::topology::{topo, NUM_NODES, NUM_TILES};
use settler_engine::{bits64, Board, PlayerId, State, NUM_PLAYERS, NUM_RESOURCES};
use settler_net::encode::{abs, layout_hash, pips};

pub const OPENING_FEATURES: usize = 78;

/// Block offsets in vector order. A per-seat block of width `w` holds r0 at the offset, r1 at
/// offset + w, and so on.
pub const SEAT_PIPS: usize = 0;
pub const SEAT_TOTAL: usize = SEAT_PIPS + NUM_PLAYERS * NUM_RESOURCES;
pub const SEAT_RESOURCES: usize = SEAT_TOTAL + NUM_PLAYERS;
pub const SEAT_NUMBERS: usize = SEAT_RESOURCES + NUM_PLAYERS;
/// Per seat: 3:1, then the 2:1 ports for wood brick sheep wheat ore.
pub const SEAT_PORTS: usize = SEAT_NUMBERS + NUM_PLAYERS;
pub const PORT_KINDS: usize = 1 + NUM_RESOURCES;
pub const SEAT_PLACED: usize = SEAT_PORTS + NUM_PLAYERS * PORT_KINDS;
pub const SEAT_REACH: usize = SEAT_PLACED + NUM_PLAYERS;
pub const PLACED_ALL: usize = SEAT_REACH + NUM_PLAYERS;
pub const PICK_LEFT: usize = PLACED_ALL + 1;
pub const BEST_RESOURCE: usize = PICK_LEFT + 1;
pub const BEST_SPOT: usize = BEST_RESOURCE + NUM_RESOURCES;
pub const RICH_SPOTS: usize = BEST_SPOT + 1;
pub const BOARD_RESOURCE: usize = RICH_SPOTS + 1;
const _: () = assert!(BOARD_RESOURCE + NUM_RESOURCES == OPENING_FEATURES);

/// Pips a legal spot needs to count as rich.
pub const RICH_PIPS: u32 = 10;

/// The feature list as canonical text: each `NAME xN: tokens` line is a block of N items (N = 4:
/// one per relative seat) with one token per feature (`name/scale` for a count over that scale).
/// `OPENING_HASH` hashes it, so renaming, reordering or rescaling a feature changes the hash and
/// a model trained on the old list is refused. Keep it in step with `opening_features`.
pub const OPENING_LAYOUT: &str = "\
settler-bots opening features v1
seats relative to p: r0 is p, then r1 r2 r3 in turn order; a seat block holds r0 r1 r2 r3
resources: wood brick sheep wheat ore
pips: dice dots of the tiles a spot touches, summed over a seat's settlements
legal: the distance rule against every settlement placed; distance 2 is two steps on the node graph
seat_pips x4: wood/15 brick/15 sheep/15 wheat/15 ore/15
seat_total x4: pips/30
seat_resources x4: touched/5
seat_numbers x4: distinct/6
seat_ports x4: generic wood brick sheep wheat ore
seat_placed x4: settlements/2
seat_reach x4: best_legal_at_distance_2/15
placed_all x1: settlements/8
pick_left x1: p
best_legal x1: wood/5 brick/5 sheep/5 wheat/5 ore/5
best_legal_spot x1: pips/15
rich_legal_spots x1: at_least_10/10
board x1: wood/25 brick/25 sheep/25 wheat/25 ore/25
";

/// Written into every opening model file; a file with another value was trained on other features.
pub const OPENING_HASH: u64 = layout_hash(&[OPENING_FEATURES as u32], OPENING_LAYOUT);

/// Every feature's name in vector order: `block[i].token`.
pub fn opening_feature_names() -> Vec<String> {
    let mut names = Vec::new();
    for line in OPENING_LAYOUT.lines() {
        let Some((head, tokens)) = line.split_once(": ") else {
            continue;
        };
        let Some((block, count)) = head.split_once(" x") else {
            continue;
        };
        let Ok(count) = count.parse::<usize>() else {
            continue;
        };
        if block.contains(' ') {
            continue;
        }
        for i in 0..count {
            for token in tokens.split(' ') {
                names.push(format!("{block}[{i}].{token}"));
            }
        }
    }
    names
}

/// Node `n`'s pips of each resource.
pub fn spot_pips(board: &Board, n: usize) -> [u32; NUM_RESOURCES] {
    let mut out = [0; NUM_RESOURCES];
    for &tile in &topo().node_tiles[n] {
        if let Some(r) = board.tile_resource[tile as usize] {
            out[r.index()] += pips(board.tile_number[tile as usize]);
        }
    }
    out
}

/// Nodes where a settlement may still go: empty, with no settlement on a neighbour.
pub fn legal_spots(settlements: &[u64; NUM_PLAYERS]) -> u64 {
    let t = topo();
    let occupied = settlements.iter().fold(0u64, |m, s| m | s);
    let blocked = bits64(occupied).fold(occupied, |m, n| m | t.node_neighbor_mask[n as usize]);
    ((1u64 << NUM_NODES) - 1) & !blocked
}

fn scaled(x: u32, den: u32) -> f32 {
    x as f32 / den as f32
}

fn flag(b: bool) -> f32 {
    b as u8 as f32
}

/// The 78 opening features for seat `p` (spec 3d-1 Section 4.2) of a position whose settlements
/// are `settlements` (by absolute seat). `p` is the seat that just picked, or would pick.
pub fn opening_features(
    board: &Board,
    settlements: &[u64; NUM_PLAYERS],
    p: PlayerId,
) -> [f32; OPENING_FEATURES] {
    let t = topo();
    let mut out = [0.0; OPENING_FEATURES];
    let legal = legal_spots(settlements);
    let spots: Vec<[u32; NUM_RESOURCES]> = (0..NUM_NODES).map(|n| spot_pips(board, n)).collect();
    let total = |n: usize| spots[n].iter().sum::<u32>();
    for r in 0..NUM_PLAYERS {
        let mine = settlements[abs(r, p)];
        let mut res = [0u32; NUM_RESOURCES];
        let mut touched = [false; NUM_RESOURCES];
        let mut numbers = 0u16;
        let mut two_steps = 0u64;
        for n in bits64(mine) {
            let n = n as usize;
            for &tile in &t.node_tiles[n] {
                if let Some(x) = board.tile_resource[tile as usize] {
                    let number = board.tile_number[tile as usize];
                    res[x.index()] += pips(number);
                    touched[x.index()] = true;
                    numbers |= 1u16 << number;
                }
            }
            for &m in &t.node_neighbors[n] {
                two_steps |= t.node_neighbor_mask[m as usize];
            }
        }
        for k in 0..NUM_RESOURCES {
            out[SEAT_PIPS + NUM_RESOURCES * r + k] = scaled(res[k], 15);
        }
        out[SEAT_TOTAL + r] = scaled(res.iter().sum(), 30);
        out[SEAT_RESOURCES + r] = scaled(touched.iter().filter(|&&x| x).count() as u32, 5);
        out[SEAT_NUMBERS + r] = scaled(numbers.count_ones(), 6);
        out[SEAT_PORTS + PORT_KINDS * r] = flag(mine & board.generic_port_nodes != 0);
        for k in 0..NUM_RESOURCES {
            out[SEAT_PORTS + PORT_KINDS * r + 1 + k] =
                flag(mine & board.specific_port_nodes[k] != 0);
        }
        out[SEAT_PLACED + r] = scaled(mine.count_ones(), 2);
        let reach = bits64(two_steps & legal)
            .map(|n| total(n as usize))
            .max()
            .unwrap_or(0);
        out[SEAT_REACH + r] = scaled(reach, 15);
    }
    let placed: u32 = settlements.iter().map(|s| s.count_ones()).sum();
    out[PLACED_ALL] = scaled(placed, 8);
    out[PICK_LEFT] = flag(settlements[p as usize].count_ones() < 2);
    let (mut best, mut best_spot, mut rich) = ([0u32; NUM_RESOURCES], 0u32, 0u32);
    for n in bits64(legal) {
        let n = n as usize;
        for k in 0..NUM_RESOURCES {
            best[k] = best[k].max(spots[n][k]);
        }
        best_spot = best_spot.max(total(n));
        rich += (total(n) >= RICH_PIPS) as u32;
    }
    for k in 0..NUM_RESOURCES {
        out[BEST_RESOURCE + k] = scaled(best[k], 5);
    }
    out[BEST_SPOT] = scaled(best_spot, 15);
    out[RICH_SPOTS] = scaled(rich, 10);
    let mut board_res = [0u32; NUM_RESOURCES];
    for tile in 0..NUM_TILES {
        if let Some(x) = board.tile_resource[tile] {
            board_res[x.index()] += pips(board.tile_number[tile]);
        }
    }
    for k in 0..NUM_RESOURCES {
        out[BOARD_RESOURCE + k] = scaled(board_res[k], 25);
    }
    out
}

/// `opening_features` of a game state (its board and every seat's settlements).
pub fn state_features(s: &State, p: PlayerId) -> [f32; OPENING_FEATURES] {
    let settlements = std::array::from_fn(|q| s.players[q].settlements);
    opening_features(&s.board, &settlements, p)
}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `PATH=$HOME/.cargo/bin:$PATH cargo test --release -p settler-bots --test opening`
Expected: PASS, 5 tests.

- [ ] **Step 5: Format, lint and the full suite**

Run, in order: `PATH=$HOME/.cargo/bin:$PATH cargo fmt --all`, `PATH=$HOME/.cargo/bin:$PATH cargo fmt --check` (no output), `PATH=$HOME/.cargo/bin:$PATH cargo clippy --workspace --release --all-targets -- -D warnings` (exit 0), `PATH=$HOME/.cargo/bin:$PATH cargo test --workspace --release` (all pass). No Python-visible change: no rebuild in this task.

- [ ] **Step 6: Commit**

```bash
git add bots/src/opening.rs bots/src/lib.rs bots/tests/opening.rs
git diff --cached | grep -nE 'sk-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{36}|password=[^[:space:]]+'
git -c user.email=sean.may101@gmail.com commit -m "bots: the 78 opening features of a setup position, seat-relative, and their layout hash"
```

---

### Task 2: The opening model file, the `+o` suffix and the choice rule

**Files:**
- Create: `bots/src/opening_model.rs`
- Modify: `bots/src/lib.rs` (module, `Roster`, `unknown_bot`), `bots/src/ismcts.rs` (`Flags`, `split_suffixes`, `split_trades`, `with_opening`, `act`), `bots/src/netbot.rs` (`NetName`), `alphasettler/cli.py` (`_check_bot_name` text), `tests/python/net_names.py`
- Test: `bots/tests/opening_model.rs` (new), `bots/tests/netbot.rs`

**Interfaces:**
- Consumes: Task 1's `settler_bots::opening::{opening_features, OPENING_FEATURES, OPENING_HASH, SEAT_PIPS, PICK_LEFT, state_features}`; `settler_net::forward::{dot, softmax}`; `netbot::{load_net, net_bot}`.
- Produces:
  - `settler_bots::opening_model::{OPENING_MAGIC: [u8; 8] = *b"ASOPEN\0\0", OPENING_FORMAT_VERSION: u32 = 1, OPENING_FILE: &str = "opening.bin", MAX_OPENING_LAYERS: usize = 8, MAX_OPENING_WIDTH: usize = 4096, OPENING_HEADER_LEN: usize = 24}`
  - `pub struct OpeningLayer { pub rows: usize, pub cols: usize, pub w: Vec<f32>, pub b: Vec<f32> }`, `pub struct OpeningModel { pub layers: Vec<OpeningLayer> }` (both `Clone, Debug, PartialEq`)
  - `OpeningModel::to_bytes(&self) -> Vec<u8>`, `OpeningModel::from_bytes(&[u8]) -> Result<OpeningModel, String>`, `OpeningModel::load(&Path) -> Result<OpeningModel, String>` (errors name the file), `OpeningModel::save(&self, &Path) -> Result<(), String>`, `OpeningModel::probabilities(&self, &[f32; OPENING_FEATURES]) -> [f32; NUM_PLAYERS]` (seat-relative: entry 0 is the features' seat)
  - `pub fn choose_settlement(model: &OpeningModel, obs: &Observation, legal: &[Action]) -> Action`
  - `settler_bots::ismcts::{Flags { pub offers: bool, pub opening: bool }, split_suffixes(&str) -> Option<(String, Flags)>}`; `split_trades(&str) -> Option<(String, bool)>` keeps its signature and returns `None` for a name with `+o`; `IsmctsBot::with_opening(self, Arc<OpeningModel>) -> IsmctsBot<E>`
  - `settler_bots::netbot::NetName<'a> { pub net: &'a str, pub simulations: u32, pub rollout: u32, pub setup_factor: u32, pub offers: bool, pub opening: bool }` (`Clone, Copy, Debug, PartialEq, Eq`); `parse_net_name(&str) -> Option<NetName<'_>>`
  - `Roster::new` loads `<nets_dir>/opening.bin` once when any name has `+o`; `Roster::make` gives `+o` bots that model.

- [ ] **Step 1: Write the failing tests**

Create `bots/tests/opening_model.rs`:

```rust
//! The opening model file, its forward pass, the `+o` name suffix and `+o` bots (spec 3d-1
//! Sections 4.3 and 4.5).

mod common;

use common::trading;
use settler_bots::arena::run_match_in;
use settler_bots::netbot::{parse_net_name, NetName};
use settler_bots::opening::*;
use settler_bots::opening_model::*;
use settler_bots::{make_bot, Bot, Roster};
use settler_engine::rng::Rng;
use settler_engine::topology::topo;
use settler_engine::*;
use settler_net::encode::pips;
use settler_net::weights::{Arch, Weights};
use std::path::PathBuf;

fn temp_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "settler-bots-opening-{}-{tag}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// A nets directory holding `tiny.bin` (random weights, width 8, 1 round) and `opening.bin` if
/// `opening` is given (and no `opening.bin` otherwise).
fn nets_dir(tag: &str, opening: Option<&OpeningModel>) -> PathBuf {
    let dir = temp_dir(tag);
    Weights::random(
        Arch {
            width: 8,
            rounds: 1,
        },
        1,
    )
    .save(&dir.join("tiny.bin"))
    .unwrap();
    let path = dir.join(OPENING_FILE);
    match opening {
        Some(m) => m.save(&path).unwrap(),
        None => {
            let _ = std::fs::remove_file(&path);
        }
    }
    dir
}

/// A model with the given hidden widths between the 78 features and the 4 seats, and small
/// weights drawn from `seed`.
fn model(hidden: &[usize], seed: u64) -> OpeningModel {
    let mut rng = Rng::new(seed);
    let mut dims = vec![OPENING_FEATURES];
    dims.extend_from_slice(hidden);
    dims.push(NUM_PLAYERS);
    let mut layers = Vec::new();
    for d in dims.windows(2) {
        let (cols, rows) = (d[0], d[1]);
        let mut x = || (rng.next_u64() >> 40) as f32 / (1u64 << 24) as f32 * 0.2 - 0.1;
        let w = (0..rows * cols).map(|_| x()).collect();
        let b = (0..rows).map(|_| x()).collect();
        layers.push(OpeningLayer { rows, cols, w, b });
    }
    OpeningModel { layers }
}

/// A one-layer model whose own-seat logit is `weight` times feature `k`; every other weight is 0.
fn one_feature(k: usize, weight: f32) -> OpeningModel {
    let mut w = vec![0.0; NUM_PLAYERS * OPENING_FEATURES];
    w[k] = weight;
    OpeningModel {
        layers: vec![OpeningLayer {
            rows: NUM_PLAYERS,
            cols: OPENING_FEATURES,
            w,
            b: vec![0.0; NUM_PLAYERS],
        }],
    }
}

fn set_u32(b: &mut [u8], at: usize, v: u32) {
    b[at..at + 4].copy_from_slice(&v.to_le_bytes());
}

fn searches(bot: &dyn Bot) -> u64 {
    bot.diagnostics()
        .into_iter()
        .find(|(k, _)| *k == "searches")
        .map(|(_, v)| v)
        .unwrap()
}

/// The first legal settlement with the most wheat pips, counted from the board directly.
fn most_wheat(board: &Board, legal: &[Action]) -> Action {
    let wheat = |a: Action| match a {
        Action::BuildSettlement(n) => topo().node_tiles[n as usize]
            .iter()
            .filter(|&&t| board.tile_resource[t as usize] == Some(Resource::Wheat))
            .map(|&t| pips(board.tile_number[t as usize]))
            .sum::<u32>(),
        _ => 0,
    };
    let mut best = legal[0];
    for &a in legal {
        if wheat(a) > wheat(best) {
            best = a;
        }
    }
    best
}

#[test]
fn a_model_round_trips_through_its_file_and_gives_probabilities() {
    let m = model(&[16, 8], 1);
    let bytes = m.to_bytes();
    assert_eq!(&bytes[..8], b"ASOPEN\0\0");
    assert_eq!(OpeningModel::from_bytes(&bytes).unwrap(), m);
    let path = temp_dir("roundtrip").join(OPENING_FILE);
    m.save(&path).unwrap();
    assert_eq!(OpeningModel::load(&path).unwrap(), m);
    let g = Game::new(5, GameConfig::default());
    let p = m.probabilities(&state_features(g.state(), 0));
    assert!((p.iter().sum::<f32>() - 1.0).abs() < 1e-6, "{p:?}");
    assert!(p.iter().all(|&x| x > 0.0), "{p:?}");
}

#[test]
fn the_forward_pass_is_relu_between_layers_and_a_softmax_at_the_end() {
    // Layer 0: unit 0 = 2 x PICK_LEFT, unit 1 = -2 x PICK_LEFT (cut to 0 by the ReLU).
    // Layer 1: own-seat logit = unit 0 + unit 1; the other seats' logits are 0.
    let mut w0 = vec![0.0; 2 * OPENING_FEATURES];
    w0[PICK_LEFT] = 2.0;
    w0[OPENING_FEATURES + PICK_LEFT] = -2.0;
    let mut w1 = vec![0.0; NUM_PLAYERS * 2];
    w1[0] = 1.0;
    w1[1] = 1.0;
    let m = OpeningModel {
        layers: vec![
            OpeningLayer {
                rows: 2,
                cols: OPENING_FEATURES,
                w: w0,
                b: vec![0.0; 2],
            },
            OpeningLayer {
                rows: NUM_PLAYERS,
                cols: 2,
                w: w1,
                b: vec![0.0; NUM_PLAYERS],
            },
        ],
    };
    let mut x = [0.0; OPENING_FEATURES];
    x[PICK_LEFT] = 1.0;
    // Logits [2, 0, 0, 0] (without the ReLU they would be all 0): e^2 / (e^2 + 3) for the seat.
    let e2 = 2.0f32.exp();
    let want = [e2, 1.0, 1.0, 1.0].map(|v| v / (e2 + 3.0));
    let got = m.probabilities(&x);
    for k in 0..NUM_PLAYERS {
        assert!((got[k] - want[k]).abs() < 1e-6, "{got:?} vs {want:?}");
    }
}

#[test]
fn bad_model_files_are_refused_with_the_reason() {
    // Layer 0 is 8 x 78 (rows at byte 24, cols at 28), layer 1 is 4 x 8 (rows at 32, cols at 36).
    let good = model(&[8], 2).to_bytes();
    let edit = |f: &dyn Fn(&mut Vec<u8>)| -> Vec<u8> {
        let mut b = good.clone();
        f(&mut b);
        b
    };
    let nan = |x: f32| {
        move |b: &mut Vec<u8>| {
            let n = b.len();
            b[n - 4..].copy_from_slice(&x.to_le_bytes());
        }
    };
    let cases: Vec<(Vec<u8>, &str)> = vec![
        (good[..10].to_vec(), "shorter than the 24-byte header"),
        (edit(&|b: &mut Vec<u8>| b[0] ^= 1), "wrong magic"),
        (
            edit(&|b: &mut Vec<u8>| set_u32(b, 8, 2)),
            "format version 2, expected 1",
        ),
        (edit(&|b: &mut Vec<u8>| b[12] ^= 1), "opening feature hash"),
        (
            edit(&|b: &mut Vec<u8>| set_u32(b, 20, 0)),
            "0 layers, expected 1..=8",
        ),
        (
            edit(&|b: &mut Vec<u8>| set_u32(b, 20, 9)),
            "9 layers, expected 1..=8",
        ),
        (
            edit(&|b: &mut Vec<u8>| set_u32(b, 24, 0)),
            "each dimension must be 1..=4096",
        ),
        (
            edit(&|b: &mut Vec<u8>| set_u32(b, 28, 77)),
            "layer 0 takes 77 inputs",
        ),
        (
            edit(&|b: &mut Vec<u8>| set_u32(b, 36, 9)),
            "layer 1 takes 9 inputs, but layer 0 gives 8",
        ),
        (
            edit(&|b: &mut Vec<u8>| set_u32(b, 32, 3)),
            "the last layer gives 3 outputs",
        ),
        (
            edit(&|b: &mut Vec<u8>| {
                b.pop();
            }),
            "truncated",
        ),
        (
            edit(&|b: &mut Vec<u8>| b.push(0)),
            "1 trailing bytes after the last layer",
        ),
        (edit(&nan(f32::NAN)), "NaN or infinite"),
        (edit(&nan(f32::INFINITY)), "NaN or infinite"),
    ];
    for (bytes, want) in &cases {
        let e = OpeningModel::from_bytes(bytes).unwrap_err();
        assert!(e.contains(want), "{want}: {e}");
    }
    let dir = temp_dir("bad");
    let path = dir.join(OPENING_FILE);
    std::fs::write(&path, &cases[1].0).unwrap();
    let e = OpeningModel::load(&path).unwrap_err();
    assert!(
        e.contains(&path.display().to_string()) && e.contains("wrong magic"),
        "{e}"
    );
    let e = OpeningModel::load(&dir.join("missing.bin")).unwrap_err();
    assert!(
        e.contains("cannot read opening model") && e.contains("missing.bin"),
        "{e}"
    );
}

#[test]
fn the_opening_suffix_parses_in_any_order_and_only_once() {
    let base = NetName {
        net: "g",
        simulations: 100,
        rollout: 0,
        setup_factor: 1,
        offers: false,
        opening: true,
    };
    assert_eq!(parse_net_name("net:g@100+o"), Some(base));
    assert_eq!(
        parse_net_name("net:g+o"),
        Some(NetName {
            simulations: 1000,
            ..base
        })
    );
    let full = NetName {
        rollout: 2,
        setup_factor: 3,
        offers: true,
        ..base
    };
    for name in [
        "net:g@100+o+t+s3+r2",
        "net:g@100+t+o+r2+s3",
        "net:g@100+s3+r2+t+o",
        "net:g@100+r2+o+s3+t",
    ] {
        assert_eq!(parse_net_name(name), Some(full), "{name}");
    }
    let dir = nets_dir("names", None);
    for bad in [
        "net:g@100+o+o",
        "net:g@100+o5",
        "net:g@100+O",
        "net:g@100+oo",
        "net:g+o@100",
        "net:g@100+t+o+t",
        "ismcts@100+o",
        "ismcts+o",
        "greedy+o",
        "random+o",
    ] {
        assert_eq!(parse_net_name(bad), None, "{bad}");
        assert!(make_bot(bad, 0).is_none(), "{bad}");
        let e = Roster::new(&[bad], &dir).err().unwrap();
        assert!(e.starts_with("unknown bot"), "{bad}: {e}");
    }
}

#[test]
fn a_missing_or_bad_opening_model_is_refused_before_any_game() {
    let dir = nets_dir("refused", None);
    let e = Roster::new(&["net:tiny@10+o"], &dir).err().unwrap();
    assert!(
        e.contains("cannot read opening model") && e.contains(OPENING_FILE),
        "{e}"
    );
    let e = run_match_in(
        "net:tiny@10+o",
        "net:tiny@10",
        0..1,
        trading(),
        1,
        &dir,
    )
    .unwrap_err();
    assert!(e.contains(OPENING_FILE), "{e}");
    let mut stale = model(&[8], 3).to_bytes();
    stale[12] ^= 1;
    std::fs::write(dir.join(OPENING_FILE), &stale).unwrap();
    let e = Roster::new(&["net:tiny@10+o"], &dir).err().unwrap();
    assert!(
        e.contains("opening feature hash") && e.contains(OPENING_FILE),
        "{e}"
    );
    // A name without +o never reads the file.
    assert!(Roster::new(&["net:tiny@10+t"], &dir).is_ok());
}

#[test]
fn a_plus_o_bot_plays_the_spot_its_model_file_scores_highest_and_searches_everything_else() {
    // Own-seat logit = 10 x r0's wheat pips / 15: this model wants wheat.
    let dir = nets_dir("choice", Some(&one_feature(SEAT_PIPS + 3, 10.0)));
    let roster = Roster::new(&["net:tiny@10+o"], &dir).unwrap();
    for seed in 0..4 {
        let mut g = Game::new(seed, GameConfig::default());
        let mut bot = roster.make("net:tiny@10+o", seed).unwrap();
        let mut picks = 0;
        while matches!(
            g.state().phase,
            Phase::SetupSettlement | Phase::SetupRoad { .. }
        ) {
            let s = *g.state();
            let legal = g.legal_actions();
            let before = searches(&*bot);
            let a = bot.act(&g.observation(s.current_actor()), &legal);
            if s.phase == Phase::SetupSettlement {
                assert_eq!(a, most_wheat(&s.board, &legal), "seed {seed}, pick {picks}");
                assert_eq!(searches(&*bot), before, "the model decides without a search");
                picks += 1;
            } else if legal.len() > 1 {
                assert_eq!(searches(&*bot), before + 1, "setup roads are searched");
            }
            g.apply(a).unwrap();
        }
        assert_eq!(picks, 8);
    }
}

#[test]
fn a_model_that_ties_picks_the_first_legal_spot() {
    let zero = one_feature(0, 0.0);
    let g = Game::new(9, GameConfig::default());
    let legal = g.legal_actions();
    assert_eq!(
        choose_settlement(&zero, &g.observation(0), &legal),
        legal[0]
    );
}

#[test]
fn a_spot_scored_before_it_is_placed_has_the_features_training_sees_after() {
    let mut g = Game::new(4, GameConfig::default());
    let mut checked = 0;
    while matches!(
        g.state().phase,
        Phase::SetupSettlement | Phase::SetupRoad { .. }
    ) {
        let legal = g.legal_actions();
        if g.state().phase == Phase::SetupSettlement {
            let p = g.state().current;
            let obs = g.observation(p);
            for &a in legal.iter().step_by(5) {
                let Action::BuildSettlement(n) = a else {
                    unreachable!("only settlements are legal here")
                };
                let mut placed = obs.settlements;
                placed[p as usize] |= 1u64 << n;
                let mut after = g.clone();
                after.apply(a).unwrap();
                assert_eq!(
                    opening_features(&obs.board, &placed, p),
                    state_features(after.state(), p)
                );
                checked += 1;
            }
        }
        g.apply(legal[legal.len() / 2]).unwrap();
    }
    assert!(checked > 40, "{checked}");
}

#[test]
fn plus_o_bots_play_whole_trading_games_independent_of_thread_count() {
    let dir = nets_dir("arena", Some(&model(&[8], 4)));
    let one = run_match_in("net:tiny@10+t+o", "net:tiny@10+t", 0..1, trading(), 1, &dir).unwrap();
    let three =
        run_match_in("net:tiny@10+t+o", "net:tiny@10+t", 0..1, trading(), 3, &dir).unwrap();
    assert_eq!(one, three);
    assert_eq!(one.len(), 4);
    assert!(one.iter().all(|r| r.belief_resets == 0 && r.turns > 0));
}
```

In `bots/tests/netbot.rs`, change the netbot import to

```rust
use settler_bots::netbot::{
    load_net, net_bot, net_bot_with, parse_net_name, valid_net_name, NetName, NET_BATCH,
};
```

and replace everything in `net_names_parse_and_bad_ones_are_rejected_without_touching_the_disk` from its first line through `assert_eq!(parse_net_name(&longest).map(|p| p.0.len()), Some(64));` with:

```rust
    let name = |net: &'static str, simulations: u32, rollout: u32, setup_factor: u32, offers: bool| {
        NetName {
            net,
            simulations,
            rollout,
            setup_factor,
            offers,
            opening: false,
        }
    };
    assert_eq!(parse_net_name("net:a"), Some(name("a", 1000, 0, 1, false)));
    assert_eq!(
        parse_net_name("net:gen-001@600"),
        Some(name("gen-001", 600, 0, 1, false))
    );
    assert_eq!(
        parse_net_name("net:gen-001@600+s10"),
        Some(name("gen-001", 600, 0, 10, false))
    );
    assert_eq!(
        parse_net_name("net:x_Y-9@300+r8"),
        Some(name("x_Y-9", 300, 8, 1, false))
    );
    assert_eq!(
        parse_net_name("net:gen-004@600+t"),
        Some(name("gen-004", 600, 0, 1, true))
    );
    assert_eq!(
        parse_net_name("net:gen-004@600+s4+t+r2"),
        Some(name("gen-004", 600, 2, 4, true))
    );
    let longest = format!("net:{}", "a".repeat(64));
    assert_eq!(parse_net_name(&longest).map(|p| p.net.len()), Some(64));
```

and, in the same test's `bad` list, after `"net:a@300+T",` add:

```rust
        "net:a@300+o+o",
        "net:a+o@300",
        "net:a@300+O",
```

In `tests/python/net_names.py`, append to `BAD_NET_NAMES` (before the closing `]`):

```python
    "net:a@300+o+o",
    "net:a+o@300",
    "net:a@300+O",
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PATH=$HOME/.cargo/bin:$PATH cargo test --release -p settler-bots --test opening_model --test netbot`
Expected: FAIL to compile (`unresolved import settler_bots::opening_model`, `no NetName in netbot`).

- [ ] **Step 3: Write the implementation**

Create `bots/src/opening_model.rs`:

```rust
//! The opening model (spec 3d-1 Sections 4.3 and 4.5): an MLP over the 78 opening features that
//! gives each seat's win probability, its file `<nets_dir>/opening.bin`, and the `+o` choice of a
//! setup settlement.
//!
//! The file, little-endian:
//!
//! | bytes | field |
//! |---|---|
//! | 0..8 | magic `ASOPEN\0\0` |
//! | 8..12 | format version, u32 = 1 |
//! | 12..20 | `OPENING_HASH`, u64 |
//! | 20..24 | layer count L, u32 (1..=`MAX_OPENING_LAYERS`) |
//! | 24..24 + 8L | per layer: rows, then cols (u32 each); rows are outputs |
//! | then | per layer, in order: rows × cols weights (row-major), then rows biases (f32) |
//!
//! Layer 0 takes the 78 features, each later layer the previous one's outputs, and the last gives
//! 4 seat-relative win logits (entry 0 is the seat the features were computed for), softmaxed.
//! Every layer but the last ends in a ReLU. `alphasettler opening-train` writes 78 → 64 → 64 → 4.

use crate::opening::{opening_features, OPENING_FEATURES, OPENING_HASH};
use settler_engine::{Action, Observation, NUM_PLAYERS};
use settler_net::forward::{dot, softmax};
use std::path::Path;

pub const OPENING_MAGIC: [u8; 8] = *b"ASOPEN\0\0";
pub const OPENING_FORMAT_VERSION: u32 = 1;
/// The model's file name inside a nets directory.
pub const OPENING_FILE: &str = "opening.bin";
/// Bounds checked before anything is sized from a file: a guard against a corrupt header.
pub const MAX_OPENING_LAYERS: usize = 8;
pub const MAX_OPENING_WIDTH: usize = 4096;
/// Magic, version, hash and layer count: the bytes before the layer shapes.
pub const OPENING_HEADER_LEN: usize = 8 + 4 + 8 + 4;

#[derive(Clone, Debug, PartialEq)]
pub struct OpeningLayer {
    pub rows: usize,
    pub cols: usize,
    /// `rows × cols`, row-major.
    pub w: Vec<f32>,
    pub b: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct OpeningModel {
    pub layers: Vec<OpeningLayer>,
}

impl OpeningModel {
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&OPENING_MAGIC);
        out.extend_from_slice(&OPENING_FORMAT_VERSION.to_le_bytes());
        out.extend_from_slice(&OPENING_HASH.to_le_bytes());
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

    /// Every check runs before anything is sized from the file: the layer count and every
    /// dimension are bounded first, then the shapes must chain from 78 inputs to 4 outputs.
    pub fn from_bytes(bytes: &[u8]) -> Result<OpeningModel, String> {
        if bytes.len() < OPENING_HEADER_LEN {
            return Err(format!(
                "truncated: {} bytes is shorter than the {OPENING_HEADER_LEN}-byte header",
                bytes.len()
            ));
        }
        if bytes[..8] != OPENING_MAGIC {
            return Err("not an AlphaSettler opening model (wrong magic)".into());
        }
        let u32_at = |i: usize| u32::from_le_bytes(bytes[i..i + 4].try_into().unwrap());
        let version = u32_at(8);
        if version != OPENING_FORMAT_VERSION {
            return Err(format!(
                "format version {version}, expected {OPENING_FORMAT_VERSION}"
            ));
        }
        let hash = u64::from_le_bytes(bytes[12..20].try_into().unwrap());
        if hash != OPENING_HASH {
            return Err(format!(
                "opening feature hash {hash:#018x} does not match this build's \
                 {OPENING_HASH:#018x}: the model was trained on different features"
            ));
        }
        let count = u32_at(20) as usize;
        if !(1..=MAX_OPENING_LAYERS).contains(&count) {
            return Err(format!(
                "{count} layers, expected 1..={MAX_OPENING_LAYERS}"
            ));
        }
        let header = OPENING_HEADER_LEN + 8 * count;
        if bytes.len() < header {
            return Err(format!(
                "truncated: {} bytes is shorter than the {header}-byte header",
                bytes.len()
            ));
        }
        let shapes: Vec<(usize, usize)> = (0..count)
            .map(|i| {
                let at = OPENING_HEADER_LEN + 8 * i;
                (u32_at(at) as usize, u32_at(at + 4) as usize)
            })
            .collect();
        let mut inputs = OPENING_FEATURES;
        for (i, &(rows, cols)) in shapes.iter().enumerate() {
            let dim_ok = |d: usize| (1..=MAX_OPENING_WIDTH).contains(&d);
            if !dim_ok(rows) || !dim_ok(cols) {
                return Err(format!(
                    "layer {i} is {rows} x {cols}: each dimension must be 1..={MAX_OPENING_WIDTH}"
                ));
            }
            if cols != inputs {
                return Err(if i == 0 {
                    format!("layer 0 takes {cols} inputs, expected the {OPENING_FEATURES} opening features")
                } else {
                    format!("layer {i} takes {cols} inputs, but layer {} gives {inputs}", i - 1)
                });
            }
            inputs = rows;
        }
        if inputs != NUM_PLAYERS {
            return Err(format!(
                "the last layer gives {inputs} outputs, expected {NUM_PLAYERS} (one per seat)"
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
            layers.push(OpeningLayer { rows, cols, w, b });
        }
        if layers
            .iter()
            .any(|l| l.w.iter().chain(&l.b).any(|x| !x.is_finite()))
        {
            return Err("a weight is NaN or infinite".into());
        }
        Ok(OpeningModel { layers })
    }

    pub fn load(path: &Path) -> Result<OpeningModel, String> {
        let bytes = std::fs::read(path)
            .map_err(|e| format!("cannot read opening model {}: {e}", path.display()))?;
        OpeningModel::from_bytes(&bytes)
            .map_err(|e| format!("opening model {}: {e}", path.display()))
    }

    pub fn save(&self, path: &Path) -> Result<(), String> {
        std::fs::write(path, self.to_bytes())
            .map_err(|e| format!("cannot write {}: {e}", path.display()))
    }

    /// Win probabilities, seat-relative to the seat `features` were computed for (entry 0).
    /// The shapes are the loader's: a model built by hand must chain from 78 inputs to 4 outputs.
    pub fn probabilities(&self, features: &[f32; OPENING_FEATURES]) -> [f32; NUM_PLAYERS] {
        let mut x = features.to_vec();
        let last = self.layers.len() - 1;
        for (i, l) in self.layers.iter().enumerate() {
            let mut y: Vec<f32> = (0..l.rows)
                .map(|r| dot(&l.w[r * l.cols..(r + 1) * l.cols], &x) + l.b[r])
                .collect();
            if i < last {
                y.iter_mut().for_each(|v| *v = v.max(0.0));
            }
            x = y;
        }
        let mut p = [x[0], x[1], x[2], x[3]];
        softmax(&mut p);
        p
    }
}

/// The setup settlement a `+o` bot plays (spec 3d-1 Section 4.5): each legal settlement placed on
/// a copy of the position, scored by the model from the viewer's seat, the highest own-win
/// probability played; ties go to the first in `legal` (engine order). No search runs.
pub fn choose_settlement(model: &OpeningModel, obs: &Observation, legal: &[Action]) -> Action {
    let mut best: Option<(Action, f32)> = None;
    for &a in legal {
        let Action::BuildSettlement(n) = a else {
            continue;
        };
        let mut placed = obs.settlements;
        placed[obs.viewer as usize] |= 1u64 << n;
        let p = model.probabilities(&opening_features(&obs.board, &placed, obs.viewer))[0];
        let better = match best {
            None => true,
            Some((_, b)) => p > b,
        };
        if better {
            best = Some((a, p));
        }
    }
    best.map_or(legal[0], |(a, _)| a)
}
```

In `bots/src/lib.rs`:

1. After `pub mod opening;` add `pub mod opening_model;`.
2. Add `use opening_model::{OpeningModel, OPENING_FILE};` below the existing `use` lines.
3. Replace the `make_bot` doc comment's last sentence `` `net:` names need their weights loaded first: use a `Roster`. `` with `` `net:` names (and `+o`, which only they take) need files loaded first: use a `Roster`. ``
4. Replace `unknown_bot` with:

```rust
/// The error for a name no bot answers to.
pub fn unknown_bot(name: &str) -> String {
    format!(
        "unknown bot {name:?}; known bots: {BOT_NAMES:?} (ismcts@N for N simulations, ismcts@N+rD \
         to add D greedy rollout moves, net:NAME[@N[+rD]] for <nets-dir>/NAME.bin; +sK on either \
         searches setup settlements with K times the simulations, +t lets either make domestic \
         offers, +o lets a net: bot pick its setup settlements with <nets-dir>/{OPENING_FILE})"
    )
}
```

5. Replace the whole `Roster` struct and `impl Roster` with:

```rust
/// Bot names checked, and their networks (and the opening model, for `+o`) loaded once and shared
/// read-only across threads, so a match builds bots per game without touching the disk.
pub struct Roster {
    nets: Vec<(String, Arc<Weights>)>,
    /// `<nets_dir>/opening.bin`, loaded only if some name has `+o`.
    opening: Option<Arc<OpeningModel>>,
}

impl Roster {
    /// Every name must be a known bot, and each `net:` name's file in `nets_dir` must load and
    /// validate, as must `<nets_dir>/opening.bin` if a name has `+o`. The error names the first
    /// bad bot or file. A malformed name is rejected before any file is read.
    pub fn new(names: &[&str], nets_dir: &Path) -> Result<Roster, String> {
        let mut nets: Vec<(String, Arc<Weights>)> = Vec::new();
        let mut opening = None;
        for &name in names {
            if let Some(n) = netbot::parse_net_name(name) {
                if !nets.iter().any(|(net, _)| net == n.net) {
                    nets.push((n.net.to_owned(), netbot::load_net(nets_dir, n.net)?));
                }
                if n.opening && opening.is_none() {
                    let model = OpeningModel::load(&nets_dir.join(OPENING_FILE))?;
                    opening = Some(Arc::new(model));
                }
            } else if make_bot(name, 0).is_none() {
                return Err(unknown_bot(name));
            }
        }
        Ok(Roster { nets, opening })
    }

    /// A fresh bot; `None` for an unknown name, or a `net:` name whose network (or, with `+o`,
    /// opening model) this roster did not load.
    pub fn make(&self, name: &str, seed: u64) -> Option<Box<dyn Bot>> {
        match netbot::parse_net_name(name) {
            Some(n) => {
                let (_, w) = self.nets.iter().find(|(net, _)| net == n.net)?;
                let mut bot = netbot::net_bot(seed, w.clone(), n.simulations, n.rollout)
                    .with_setup_factor(n.setup_factor)
                    .with_offers(n.offers);
                if n.opening {
                    bot = bot.with_opening(self.opening.clone()?);
                }
                Some(Box::new(bot))
            }
            None => make_bot(name, seed),
        }
    }
}
```

In `bots/src/ismcts.rs`:

1. Change the module doc's second sentence to: `It makes offers only with `with_offers` (a `+t` name), and with `with_opening` (a `net:` name's `+o`) an opening model picks its setup settlements without a search (spec 3d-1 Section 4.5).` (keep the rest of the doc).
2. Add to the imports:

```rust
use crate::opening_model::{choose_settlement, OpeningModel};
use std::sync::Arc;
```

3. Replace the whole `split_trades` function (doc comment included) with:

```rust
/// The flag suffixes of a searching bot's name.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Flags {
    /// `+t`: the bot may make domestic offers.
    pub offers: bool,
    /// `+o`: an opening model picks the setup settlements (`net:` names only).
    pub opening: bool,
}

/// A searching bot's name with its `+t` and `+o` suffixes removed and its `+rD` / `+sK` suffixes
/// put in the order `parse_name` and `split_setup` read (`+rD` before `+sK`), and which flags it
/// had. The suffixes are everything from the first `+`, in any order, each kind at most once;
/// `None` for a repeated or unknown one (`+t5`, `+T`, `+oo`, a bare `+`). Values are checked by
/// those parsers.
pub fn split_suffixes(name: &str) -> Option<(String, Flags)> {
    let Some(i) = name.find('+') else {
        return Some((name.to_owned(), Flags::default()));
    };
    let (mut rollout, mut setup, mut flags) = (None, None, Flags::default());
    for part in name[i + 1..].split('+') {
        match part.as_bytes().first() {
            Some(b'r') if rollout.is_none() => rollout = Some(part),
            Some(b's') if setup.is_none() => setup = Some(part),
            Some(b't') if part.len() == 1 && !flags.offers => flags.offers = true,
            Some(b'o') if part.len() == 1 && !flags.opening => flags.opening = true,
            _ => return None,
        }
    }
    let mut out = name[..i].to_owned();
    for part in [rollout, setup].into_iter().flatten() {
        out.push('+');
        out.push_str(part);
    }
    Some((out, flags))
}

/// `split_suffixes` for bots that take no opening model (`ismcts` names): the name and whether it
/// had `+t`; `None` if it has `+o`.
pub fn split_trades(name: &str) -> Option<(String, bool)> {
    let (canon, flags) = split_suffixes(name)?;
    (!flags.opening).then_some((canon, flags.offers))
}
```

4. In `pub struct IsmctsBot<E = HeuristicEvaluator>`, after the `setup_factor: u32,` field add:

```rust
    /// `+o`: picks setup settlements without a search.
    opening: Option<Arc<OpeningModel>>,
```

and in `with_evaluator`, after `setup_factor: 1,` add `opening: None,`.

5. After `with_offers` add:

```rust
    /// Let `model` pick every setup settlement (a `+o` name): no search runs for them; every
    /// other decision is unchanged. `act` applies it; self-play's `act_with` does not.
    pub fn with_opening(mut self, model: Arc<OpeningModel>) -> IsmctsBot<E> {
        self.opening = Some(model);
        self
    }
```

6. In `impl<E: Evaluator + Send> Bot for IsmctsBot<E>`, replace the first line of `act` (`let factor = if obs.phase == Phase::SetupSettlement {`) with:

```rust
        if let (Phase::SetupSettlement, Some(model)) = (obs.phase, self.opening.clone()) {
            // As `act_with` does for a decision it does not search.
            self.last = None;
            self.counted_in_observe = false;
            return choose_settlement(&model, obs, legal);
        }
        let factor = if obs.phase == Phase::SetupSettlement {
```

In `bots/src/netbot.rs`:

1. Replace the module doc's first sentence with: `//! Network bots (spec Section 3): `net:NAME` (1,000 simulations), `net:NAME@N` and `net:NAME@N+rD`, each optionally with `+sK`, `+t` and `+o` (an opening model picks the setup settlements, spec 3d-1 Section 4.5) in any order, with the same limits on N, D and K as `ismcts`: IsmctsBot searching with a `NetEvaluator`.` (keep the NAME sentence).
2. Replace `parse_net_name` and its doc comment with:

```rust
/// A parsed `net:` bot name.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NetName<'a> {
    pub net: &'a str,
    pub simulations: u32,
    pub rollout: u32,
    /// `+sK`: setup settlements search K times the simulations (1 without it).
    pub setup_factor: u32,
    /// `+t`: the bot may offer.
    pub offers: bool,
    /// `+o`: `<nets_dir>/opening.bin` picks the setup settlements.
    pub opening: bool,
}

/// `net:NAME` -> 1,000 simulations, no rollout; `net:NAME@N` -> N simulations; `net:NAME@N+rD`
/// -> D rollout moves. `+sK`, `+t` and `+o` set the other fields; the suffixes come in any order
/// (`ismcts::split_suffixes`).
pub fn parse_net_name(name: &str) -> Option<NetName<'_>> {
    let (canon, flags) = ismcts::split_suffixes(name)?;
    let (canon, setup_factor) = ismcts::split_setup(&canon)?;
    let rest = name.strip_prefix("net:")?;
    let end = rest.find(['@', '+']).unwrap_or(rest.len());
    let net = &rest[..end];
    if !valid_net_name(net) {
        return None;
    }
    let tail = &canon["net:".len() + net.len()..];
    let (simulations, rollout) = ismcts::parse_name(&format!("ismcts{tail}"))?;
    Some(NetName {
        net,
        simulations,
        rollout,
        setup_factor,
        offers: flags.offers,
        opening: flags.opening,
    })
}
```

In `alphasettler/cli.py`, in `_check_bot_name`, replace the message's parenthesis `(or ismcts@N, ismcts@N+rD, net:NAME[@N[+rD]], either with +sK and +t)` with `(or ismcts@N, ismcts@N+rD, net:NAME[@N[+rD]], either with +sK and +t; net: names also with +o)`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `PATH=$HOME/.cargo/bin:$PATH cargo test --release -p settler-bots --test opening_model --test netbot --test ismcts`
Expected: PASS (9 tests in `opening_model`; `netbot` and `ismcts` unchanged in count, all passing).

- [ ] **Step 5: Format, lint, the full suite, rebuild, pytest**

Run the Global Constraints sequence: `cargo fmt --all`, `cargo fmt --check`, `cargo clippy --workspace --release --all-targets -- -D warnings`, `cargo test --workspace --release`, rebuild the extension, `.venv/bin/pytest -q -rs`. Expected: all pass; `test_netbot.py::test_bad_names_missing_files_and_stale_hashes_are_errors` now also covers the three `+o` names.

- [ ] **Step 6: Commit**

```bash
git add bots/src/opening_model.rs bots/src/lib.rs bots/src/ismcts.rs bots/src/netbot.rs alphasettler/cli.py \
  bots/tests/opening_model.rs bots/tests/netbot.rs tests/python/net_names.py
git diff --cached | grep -nE 'sk-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{36}|password=[^[:space:]]+'
git -c user.email=sean.may101@gmail.com commit -m "bots: the opening model file (ASOPEN) and its forward pass; +o lets a net bot pick setup settlements with it"
```

---

### Task 3: Bindings — opening features, the model's forward pass, `net_evaluate`, `Bot.act` without the GIL

**Files:**
- Modify: `net/src/evaluator.rs` (`position_value`), `bindings/src/lib.rs` (`mod opening`, registration, `PyBot::act`), `bindings/src/convert.rs` (`index_arg` becomes `pub`)
- Create: `bindings/src/opening.rs`
- Test: `net/tests/evaluator.rs`, `tests/python/test_opening_bindings.py` (new)

**Interfaces:**
- Consumes: Task 1's `settler_bots::opening::{state_features, OPENING_FEATURES, OPENING_HASH}`; Task 2's `settler_bots::opening_model::{OpeningModel, OPENING_FILE, OPENING_FORMAT_VERSION}`; `settler_bots::netbot::{load_net, DEFAULT_NETS_DIR}`; `settler_net::encode::encode_state`, `forward::{Forward, Outputs}`, `evaluator::absolute_value`.
- Produces:
  - Rust: `settler_net::evaluator::position_value(weights: &Weights, s: &State, viewer: PlayerId) -> [f32; NUM_PLAYERS]`.
  - Python (`alphasettler._engine`): `opening_features(game, seat) -> list[float]` (78; ValueError outside `setup_settlement` / `setup_road`, or for a seat outside 0..4); `opening_forward(path, rows: list[list[float]]) -> list[list[float]]` (seat-relative probabilities per row; the Rust loader's errors); `net_evaluate(game, viewer, net, nets_dir=None) -> list[float]` (4 probabilities by absolute seat; `net` is a bare NAME); constants `OPENING_FEATURES` (78), `OPENING_HASH`, `OPENING_FORMAT_VERSION` (1), `OPENING_FILE` (`"opening.bin"`).
  - `Bot.act(game)` releases the GIL while the bot decides (same results).

- [ ] **Step 1: Write the failing tests**

In `net/tests/evaluator.rs`, change the evaluator import to

```rust
use settler_net::evaluator::{absolute_value, masked_prior, position_value, NetEvaluator};
```

and append:

```rust
#[test]
fn position_value_is_the_value_a_search_leaf_on_the_same_world_gets() {
    let w = net();
    let mut eval = NetEvaluator::new(w.clone());
    let mut checked = 0;
    for s in random_positions(0..3, 17, no_trades()) {
        let legal = searchable(&s);
        for viewer in 0..4u8 {
            let leaf = eval.evaluate(&[Leaf {
                world: &s,
                actor: viewer,
                legal: &legal,
            }]);
            let v = position_value(&w, &s, viewer);
            assert_eq!(v, leaf[0].value, "viewer {viewer}");
            assert!((v.iter().sum::<f32>() - 1.0).abs() < 1e-5, "{v:?}");
            checked += 1;
        }
    }
    assert!(checked > 40, "{checked}");
}
```

Create `tests/python/test_opening_bindings.py`:

```python
"""Opening bindings (spec 3d-1 Section 5): the opening features, the Rust forward pass of an
opening model file, net_evaluate, and bots acting from several Python threads at once."""

import math
import struct
from array import array
from concurrent.futures import ThreadPoolExecutor

import pytest

from alphasettler import Bot, Game
from alphasettler._engine import (
    OPENING_FEATURES, OPENING_FILE, OPENING_FORMAT_VERSION, OPENING_HASH, encode, net_evaluate, net_forward,
    opening_features, opening_forward, write_random_net,
)
from alphasettler.arena import TRADES

SETTLEMENT = 25  # BuildSettlement(n) is action 25 + n


def setup_game(nodes, seed=0):
    """Seed `seed`'s game with settlements on `nodes` in setup order, each but the last followed by
    its first legal road."""
    g = Game(seed, TRADES)
    for k, n in enumerate(nodes):
        g.apply(SETTLEMENT + n)
        if k < len(nodes) - 1:
            g.apply(g.legal_actions()[0])
    return g


def model_bytes(hidden=8, own=0.0, version=OPENING_FORMAT_VERSION, feature_hash=OPENING_HASH):
    """A 78 -> hidden -> 4 opening model file: every weight 0, the own seat's output bias `own`."""
    shapes = [(hidden, OPENING_FEATURES), (4, hidden)]
    out = [b"ASOPEN\0\0", struct.pack("<IQI", version, feature_hash, len(shapes))]
    out += [struct.pack("<II", r, c) for r, c in shapes]
    for r, c in shapes:
        bias = [0.0] * r
        if r == 4:
            bias[0] = own
        out += [array("f", [0.0] * (r * c)).tobytes(), array("f", bias).tobytes()]
    return b"".join(out)


@pytest.fixture
def nets(tmp_path):
    d = tmp_path / "nets"
    d.mkdir()
    write_random_net(str(d / "tiny.bin"), 1, 8, 1)
    return d


def test_the_opening_constants():
    assert (OPENING_FEATURES, OPENING_FORMAT_VERSION, OPENING_FILE) == (78, 1, "opening.bin")
    assert 0 < OPENING_HASH < 2**64


def test_opening_features_of_a_setup_position():
    # Seed 0's board; seats 0 to 3 on nodes 9, 12, 18 and 41; seat 3 to place its road.
    g = setup_game([9, 12, 18, 41])
    assert (g.phase, g.current_actor) == ("setup_road", 3)
    f = opening_features(g, 3)
    assert len(f) == OPENING_FEATURES
    # r0 is seat 3 on node 41: wood 2 (1 pip), sheep 6 (5), wheat 9 (4); per resource / 15.
    assert f[0:5] == pytest.approx([1 / 15, 0, 5 / 15, 4 / 15, 0])
    # Settlements placed in all / 8, and seat 3 still has its second pick.
    assert f[64:66] == pytest.approx([4 / 8, 1.0])
    # The board's pips per resource / 25.
    assert f[73:78] == pytest.approx([7 / 25, 12 / 25, 18 / 25, 14 / 25, 7 / 25])
    # From seat 0's side seat 3 is r3.
    assert opening_features(g, 0)[15:20] == f[0:5]


def test_opening_features_refuse_positions_outside_the_setup_and_bad_seats():
    g = setup_game([9, 12, 18, 41, 31, 10, 4, 29])
    g.apply(g.legal_actions()[0])  # the last road: the setup is over
    assert g.phase == "pre_roll"
    with pytest.raises(ValueError, match="setup position"):
        opening_features(g, 0)
    with pytest.raises(ValueError, match="seat 4 out of range"):
        opening_features(Game(0, TRADES), 4)


def test_opening_forward_reads_a_model_file_as_the_bots_do(tmp_path):
    path = tmp_path / OPENING_FILE
    path.write_bytes(model_bytes(own=math.log(3.0)))
    # Logits [ln 3, 0, 0, 0] whatever the input: the own seat gets 3 / 6.
    for row in opening_forward(str(path), [[0.0] * OPENING_FEATURES, [1.0] * OPENING_FEATURES]):
        assert row == pytest.approx([0.5, 1 / 6, 1 / 6, 1 / 6])
    with pytest.raises(ValueError, match="row 1 has 77 features"):
        opening_forward(str(path), [[0.0] * OPENING_FEATURES, [0.0] * 77])
    stale = tmp_path / "stale.bin"
    stale.write_bytes(model_bytes(feature_hash=OPENING_HASH ^ 1))
    with pytest.raises(ValueError, match=r"stale\.bin.*opening feature hash"):
        opening_forward(str(stale), [[0.0] * OPENING_FEATURES])
    with pytest.raises(ValueError, match="cannot read opening model"):
        opening_forward(str(tmp_path / "missing.bin"), [[0.0] * OPENING_FEATURES])


def test_a_plus_o_bot_needs_a_valid_opening_model(nets):
    with pytest.raises(ValueError, match="cannot read opening model"):
        Bot("net:tiny@10+o", 0, nets_dir=str(nets))
    (nets / OPENING_FILE).write_bytes(model_bytes(version=2))
    with pytest.raises(ValueError, match="format version 2"):
        Bot("net:tiny@10+o", 0, nets_dir=str(nets))
    (nets / OPENING_FILE).write_bytes(model_bytes())
    g = Game(3, TRADES)
    # A model that scores every spot the same plays the first legal one.
    assert Bot("net:tiny@10+t+o", 0, nets_dir=str(nets)).act(g) == g.legal_actions()[0]


def softmax(xs):
    m = max(xs)
    e = [math.exp(x - m) for x in xs]
    return [x / sum(e) for x in e]


def test_net_evaluate_gives_the_value_head_by_absolute_seat(nets):
    g = Game(6, TRADES)
    for k in range(40):
        legal = g.legal_actions()
        g.apply(legal[(7 * k) % len(legal)])
    for viewer in range(4):
        p = net_evaluate(g, viewer, "tiny", str(nets))
        assert len(p) == 4 and sum(p) == pytest.approx(1.0, abs=1e-6)
    # The actor's view is the Rust forward pass of `encode` (the actor's features), rotated to
    # absolute seats.
    actor = g.current_actor
    public, hidden = encode(g)
    value = array("f")
    value.frombytes(net_forward(str(nets / "tiny.bin"), public, hidden)["value"])
    rel = softmax(list(value))
    want = [rel[(q - actor) % 4] for q in range(4)]
    assert net_evaluate(g, actor, "tiny", str(nets)) == pytest.approx(want, abs=1e-6)
    with pytest.raises(ValueError, match="bad net name"):
        net_evaluate(g, 0, "../tiny", str(nets))
    with pytest.raises(ValueError, match="cannot read"):
        net_evaluate(g, 0, "missing", str(nets))
    with pytest.raises(ValueError, match="viewer 4 out of range"):
        net_evaluate(g, 4, "tiny", str(nets))


def test_bots_act_from_several_threads_as_they_do_one_at_a_time(nets):
    """Bot.act releases the GIL while it decides; the moves still depend only on the seeds."""

    def setup(seed):
        g = Game(seed, TRADES)
        bots = [Bot("net:tiny@20", seed * 4 + p, nets_dir=str(nets)) for p in range(4)]
        moves = []
        for _ in range(16):
            a = bots[g.current_actor].act(g)
            g.apply(a)
            moves.append(a)
        return moves

    one_at_a_time = [setup(s) for s in range(6)]
    with ThreadPoolExecutor(3) as ex:
        assert list(ex.map(setup, range(6))) == one_at_a_time
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PATH=$HOME/.cargo/bin:$PATH cargo test --release -p settler-net --test evaluator`
Expected: FAIL to compile (`no position_value in evaluator`).
Run: `.venv/bin/pytest -q tests/python/test_opening_bindings.py`
Expected: FAIL at collection with `ImportError: cannot import name 'OPENING_FEATURES'`.

- [ ] **Step 3: Write the implementation**

In `net/src/evaluator.rs`, after `absolute_value`, add:

```rust
/// The net's win probabilities, by absolute seat, for `s` seen by `viewer`: the public block from
/// `s.observation(viewer)`, the hidden block from `s` itself (what a search leaf whose world is
/// `s` gets), the value head rotated back to absolute seats.
pub fn position_value(weights: &Weights, s: &State, viewer: PlayerId) -> [f32; NUM_PLAYERS] {
    let (mut public, mut hidden) = (vec![0.0; PUBLIC_LEN], vec![0.0; HIDDEN_LEN]);
    encode_state(s, viewer, &mut public, &mut hidden);
    let mut out = Outputs::default();
    Forward::new().run(weights, &public, &hidden, 1, &mut out);
    absolute_value(&out.value[..NUM_PLAYERS], viewer)
}
```

In `bindings/src/convert.rs`, change `fn index_arg(` to `pub fn index_arg(`.

Create `bindings/src/opening.rs`:

```rust
//! Opening bindings (spec 3d-1 Section 5): the opening features of a setup position, the Rust
//! forward pass of an opening model file (for the PyTorch parity check, and to check a written
//! file loads), and a net's win probabilities for a position.

use crate::convert::*;
use crate::PyGame;
use pyo3::prelude::*;
use settler_bots::netbot::{load_net, DEFAULT_NETS_DIR};
use settler_bots::opening::{state_features, OPENING_FEATURES, OPENING_HASH};
use settler_bots::opening_model::{OpeningModel, OPENING_FILE, OPENING_FORMAT_VERSION};
use settler_engine::{Phase, NUM_PLAYERS};
use settler_net::evaluator::position_value;
use std::path::PathBuf;

/// The 78 opening features for `seat` of a setup position (`setup_settlement` or `setup_road`),
/// seat-relative: the function a `+o` bot scores its spots with.
#[pyfunction]
fn opening_features(game: PyRef<'_, PyGame>, seat: &Bound<'_, PyAny>) -> PyResult<Vec<f32>> {
    let seat = index_arg(seat, NUM_PLAYERS, "seat")? as u8;
    let s = game.inner.state();
    if !matches!(s.phase, Phase::SetupSettlement | Phase::SetupRoad { .. }) {
        return Err(value_error(format!(
            "opening features need a setup position (setup_settlement or setup_road), not {}",
            phase_name(s.phase)
        )));
    }
    Ok(state_features(s, seat).to_vec())
}

/// Each row's seat-relative win probabilities under the opening model at `path`, read by the
/// loader every `+o` bot uses, so its errors (naming the file) are the bots' errors.
#[pyfunction]
fn opening_forward(path: PathBuf, rows: Vec<Vec<f32>>) -> PyResult<Vec<Vec<f32>>> {
    let model = OpeningModel::load(&path).map_err(value_error)?;
    rows.iter()
        .enumerate()
        .map(|(i, row)| {
            let x: &[f32; OPENING_FEATURES] = row.as_slice().try_into().map_err(|_| {
                value_error(format!(
                    "row {i} has {} features, expected {OPENING_FEATURES}",
                    row.len()
                ))
            })?;
            Ok(model.probabilities(x).to_vec())
        })
        .collect()
}

/// `<nets_dir>/<net>.bin`'s win probabilities by absolute seat for `game` seen by `viewer`
/// (`settler_net::evaluator::position_value`). `net` is a bare NAME, as in `net:NAME`; a bad name
/// or file raises what `Bot(...)` raises for it.
#[pyfunction]
#[pyo3(signature = (game, viewer, net, nets_dir=None))]
fn net_evaluate(
    py: Python<'_>,
    game: PyRef<'_, PyGame>,
    viewer: &Bound<'_, PyAny>,
    net: &str,
    nets_dir: Option<PathBuf>,
) -> PyResult<Vec<f32>> {
    let viewer = viewer_arg(viewer)?;
    let s = *game.inner.state();
    if s.is_over() {
        return Err(value_error("the game is over"));
    }
    let dir = nets_dir.unwrap_or_else(|| PathBuf::from(DEFAULT_NETS_DIR));
    let w = load_net(&dir, net).map_err(value_error)?;
    Ok(py.detach(|| position_value(&w, &s, viewer)).to_vec())
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(opening_features, m)?)?;
    m.add_function(wrap_pyfunction!(opening_forward, m)?)?;
    m.add_function(wrap_pyfunction!(net_evaluate, m)?)?;
    m.add("OPENING_FEATURES", OPENING_FEATURES)?;
    m.add("OPENING_HASH", OPENING_HASH)?;
    m.add("OPENING_FORMAT_VERSION", OPENING_FORMAT_VERSION)?;
    m.add("OPENING_FILE", OPENING_FILE)?;
    Ok(())
}
```

In `bindings/src/lib.rs`:

1. After `mod convert;` add `mod opening;`.
2. In `fn _engine`, after `train::register(m)?;` add `opening::register(m)?;`.
3. Replace `PyBot::act` (doc comment included) with:

```rust
    /// The bot's choice for whoever must act in `game` now, as an action id. The bot decides with
    /// the GIL released (the lock is taken inside, so a second thread acting with the same bot
    /// waits without holding the GIL), so bots on several Python threads search at once.
    fn act(&self, py: Python<'_>, game: PyRef<'_, PyGame>) -> PyResult<u32> {
        let s = game.inner.state();
        if s.is_over() {
            return Err(value_error("the game is over"));
        }
        let legal = game.inner.legal_actions();
        if legal.is_empty() {
            // Bots require a non-empty `legal`; refuse here rather than panic inside one.
            return Err(value_error("no legal actions in this position"));
        }
        let obs = s.observation(s.current_actor());
        let a = py
            .detach(|| {
                let mut bot = self.inner.lock().map_err(|_| ())?;
                Ok::<_, ()>(bot.act(&obs, &legal))
            })
            .map_err(|_| value_error("bot state is poisoned"))?;
        Ok(a.encode() as u32)
    }
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `PATH=$HOME/.cargo/bin:$PATH cargo test --release -p settler-net --test evaluator`
Expected: PASS.
Rebuild the extension, then run: `.venv/bin/pytest -q tests/python/test_opening_bindings.py tests/python/test_netbot.py tests/python/test_selfplay.py`
Expected: PASS (7 tests in the new file).

- [ ] **Step 5: Format, lint, the full suite, pytest**

Run the Global Constraints sequence (fmt, fmt --check, clippy, `cargo test --workspace --release`, `.venv/bin/pytest -q -rs`). Expected: all pass, nothing skipped.

- [ ] **Step 6: Commit**

```bash
git add net/src/evaluator.rs net/tests/evaluator.rs bindings/src/opening.rs bindings/src/lib.rs bindings/src/convert.rs \
  tests/python/test_opening_bindings.py
git diff --cached | grep -nE 'sk-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{36}|password=[^[:space:]]+'
git -c user.email=sean.may101@gmail.com commit -m "bindings: opening features, the opening model's forward pass and net_evaluate; Bot.act releases the GIL"
```

---

### Task 4: `alphasettler opening-train` — setup replay, samples, the PyTorch model, the report

**Files:**
- Create: `alphasettler/opening.py`, `alphasettler/train/opening.py`, `tests/python/opening_games.py`, `tests/python/test_opening_train.py`
- Modify: `alphasettler/records.py` (`record_files`), `alphasettler/train/data.py` (re-export it), `alphasettler/cli.py` (`opening-train`)

**Interfaces:**
- Consumes: Task 3's `opening_features`, `opening_forward`, `net_evaluate`, `OPENING_FEATURES`, `OPENING_HASH`, `OPENING_FORMAT_VERSION`, `OPENING_FILE`; `alphasettler.files.{temp_path, fsync_file, atomic_write_text}`; `alphasettler.train.data.HELD_OUT_EVERY` (20); `resolve_net`.
- Produces:
  - `alphasettler.records.record_files(gen_dir) -> list[Path]` (moved; `alphasettler.train.data.record_files` stays importable).
  - `alphasettler.opening`: `SETUP_ACTIONS = 16`, `SETTLEMENT_BASE = 25`, `ROAD_BASE = 133`, `NODES = 54`, `EDGES = 72`, `MAGIC`; `Pick(seat: int, node: int, legal: list[int], before: Game | None, features: list[float] | None, value: list[float] | None)`; `Setup(seed: int, trading: bool, winner: int | None, board: dict, picks: list[Pick], after: Game)` with `Setup.nodes(seat) -> list[int]`; `replay_setup(record, *, features=False, value=None, keep_positions=False) -> Setup` (ValueError `game SEED: ...`); `model_bytes(layers) -> bytes`; `write_model(path, layers) -> None` where `layers` is a sequence of `(rows, cols, weights, biases)`.
  - `alphasettler.train.opening`: `HIDDEN = (64, 64)`, `UNIFORM = ln 4`, `PROBABILITY_FLOOR = 1e-12`; `OpeningConfig(lr=1e-3, batch=1024, patience=3, max_epochs=100, seed=0, hidden=HIDDEN)` with `.check()`; `OpeningNet(hidden)` with `.layers()`; `Samples(x, y, trading, compare)`; `load_samples(dirs, compare=None) -> (Samples, Samples, dict)`; `held_out_loss(net, samples) -> float`; `fit(train, held, cfg, log=print) -> (OpeningNet, dict)`; `report(net, held) -> dict`; `train_opening(dirs, out, cfg, compare_net=None, nets_dir=None, log=print) -> dict`; `format_report(summary, out) -> list[str]`.
  - Test helper `tests/python/opening_games.py`: `FIXED_NODES`, `FIXED_ACTIONS`, `fixed_game(seed=0, winner=3, config=TRADES) -> dict` (a record with its resolved config), `write_dir(path, games) -> Path`.
  - CLI: `alphasettler opening-train --games DIR [DIR ...] --out PATH [--compare-net NAME] [--nets-dir D] [--seed S]`.

- [ ] **Step 1: Write the test helper and the failing tests**

Create `tests/python/opening_games.py`:

```python
"""A fixed setup on seed 0's board, shared by the opening tests (records need not come from
self-play: every opening tool reads only a record's seed, config, first 16 actions and winner).

Settlements in pick order (seats 0 1 2 3 3 2 1 0): nodes 9, 12, 18, 41, 31, 10, 4, 29, each
followed by its first legal road. What they touch (resource number):
  seat 0: 9 (wheat 10, wheat 8, brick 5), 29 (wood 12, ore 11, sheep 6): all five, 20 pips
  seat 1: 12 (ore 10, brick 9, sheep 8), 4 (ore 10, wheat 10): no wood or no brick, 18 pips
  seat 2: 18 (sheep 8, brick 5, ore 11), 10 (wheat 8, wood 11): all five, 18 pips
  seat 3: 41 (wood 2, wheat 9, sheep 6), 31 (brick 5, sheep 4, wheat 9): no ore, 21 pips
The distance rule does not depend on the board, so these actions replay on any seed:
`fixed_game(seed=...)` makes as many distinct games as a test needs (only seed 0 has this board)."""

from pathlib import Path

from alphasettler import records
from alphasettler._engine import resolve_config
from alphasettler.arena import TRADES

FIXED_NODES = [9, 12, 18, 41, 31, 10, 4, 29]
FIXED_ACTIONS = [34, 141, 37, 144, 43, 153, 66, 185, 56, 170, 35, 142, 29, 134, 54, 168]


def fixed_game(seed: int = 0, winner: int | None = 3, config: dict = TRADES) -> dict:
    """A record holding only the fixed setup, played under `config` (stored resolved, as the
    record writer stores it)."""
    return {"seed": seed, "config": resolve_config(config), "actions": list(FIXED_ACTIONS), "winner": winner,
            "vp": [0, 0, 0, 0], "turns": 0, "decisions": [], "belief_resets": 0}


def write_dir(path: Path, games: list[dict]) -> Path:
    """`games` as the record file `path/selfplay-fixed.jsonl.gz` (each keeps its own config)."""
    records.write(Path(path) / "selfplay-fixed.jsonl.gz", games, {})
    return Path(path)
```

Create `tests/python/test_opening_train.py`:

```python
"""The opening model's training (spec 3d-1 Section 4.4): the setup replay, the samples, the PyTorch
model and its parity with the Rust forward pass, early stopping, and `opening-train`.

These tests need the `train` extra (PyTorch). Without it they fail at collection on purpose: they
must be run, never skipped."""

import copy
import json
import random
import re

import pytest
import torch

from alphasettler import Bot, Game
from alphasettler._engine import OPENING_FEATURES, OPENING_FILE, opening_features, opening_forward, write_random_net
from alphasettler.arena import TRADES
from alphasettler.opening import SETUP_ACTIONS, model_bytes, replay_setup, write_model
from alphasettler.selfplay import write_explore
from alphasettler.train import opening as topen
from alphasettler.train.opening import OpeningConfig, OpeningNet, Samples, fit, load_samples
from opening_games import FIXED_ACTIONS, FIXED_NODES, fixed_game, write_dir
from test_train import cli, write_games

SEATS_IN_PICK_ORDER = [0, 1, 2, 3, 3, 2, 1, 0]


def test_replay_setup_keeps_each_pick_and_the_position_after_the_setup():
    s = replay_setup(fixed_game(), features=True, keep_positions=True)
    assert [p.seat for p in s.picks] == SEATS_IN_PICK_ORDER
    assert [p.node for p in s.picks] == FIXED_NODES
    assert s.picks[0].legal == list(range(54))
    assert all(p.node in p.legal for p in s.picks)
    assert (s.after.phase, s.winner, s.trading) == ("pre_roll", 3, True)
    assert s.nodes(3) == [41, 31]
    # The features are the seat's right after its settlement, before its road.
    before = s.picks[4].before
    assert before.phase == "setup_settlement"
    g = before.copy()
    g.apply(25 + 31)
    assert s.picks[4].features == opening_features(g, 3)


@pytest.mark.parametrize("actions,message", [
    (FIXED_ACTIONS[:15], "game 0: 15 actions, fewer than the 16 of the setup"),
    ([FIXED_ACTIONS[1]] + FIXED_ACTIONS[1:], "game 0: action 0 (141) is not a setup settlement"),
    (FIXED_ACTIONS[:2] + [25 + 9] + FIXED_ACTIONS[3:], "game 0: action 2 (34): "),  # node 9 is taken
])
def test_replay_setup_refuses_records_that_do_not_replay_a_setup(actions, message):
    with pytest.raises(ValueError, match=re.escape(message)):
        replay_setup({**fixed_game(), "actions": actions})


def test_samples_are_eight_per_decided_game_with_the_winner_relative_to_the_seat(tmp_path):
    # Seed 0 is held out (0 % 20 == 0); seed 1 trains; seed 2 has no winner and adds nothing.
    d = write_dir(tmp_path / "gen", [fixed_game(0, winner=3), fixed_game(1, winner=1), fixed_game(2, winner=None)])
    train, held, counts = load_samples([d])
    assert counts == {"games": 3, "no_winner": 1, "held_out_games": 1}
    assert (len(train), len(held)) == (8, 8)
    assert held.y.tolist() == [(3 - s) % 4 for s in SEATS_IN_PICK_ORDER]
    assert train.y.tolist() == [(1 - s) % 4 for s in SEATS_IN_PICK_ORDER]
    assert held.trading.all() and held.compare is None
    assert train.x.shape == (8, OPENING_FEATURES) and train.x.dtype == torch.float32
    assert held.x[4].tolist() == replay_setup(fixed_game(), features=True).picks[4].features
    # With a compare net, each held-out sample keeps -ln(its probability of the winner): seat 3's 0.4.
    _, held, _ = load_samples([d], compare=lambda g, seat: [0.1, 0.2, 0.3, 0.4])
    assert held.compare.tolist() == pytest.approx([-torch.log(torch.tensor(0.4, dtype=torch.float64)).item()] * 8)


def test_directories_without_records_training_or_held_out_games_are_errors(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError, match="no record files"):
        load_samples([tmp_path / "empty"])
    only_held = write_dir(tmp_path / "held", [fixed_game(0, winner=0)])
    with pytest.raises(ValueError, match="no training games"):
        load_samples([only_held])
    only_train = write_dir(tmp_path / "train", [fixed_game(1, winner=0)])
    with pytest.raises(ValueError, match="no held-out games"):
        load_samples([only_train])
    bad = write_dir(tmp_path / "bad", [{**fixed_game(1), "actions": FIXED_ACTIONS[:3]}])
    with pytest.raises(ValueError, match=r"selfplay-fixed\.jsonl\.gz: game 1: 3 actions"):
        load_samples([bad])


def random_setup_features(seeds):
    """The features after every settlement of uniformly random setups."""
    rows = []
    for seed in seeds:
        rng = random.Random(seed)
        g = Game(seed, None)
        for _ in range(SETUP_ACTIONS):
            seat, settlement = g.current_actor, g.phase == "setup_settlement"
            g.apply(rng.choice(g.legal_actions()))
            if settlement:
                rows.append(opening_features(g, seat))
    return rows


@pytest.mark.parametrize("hidden", [(64, 64), (5,)])
def test_rust_and_pytorch_opening_models_agree(tmp_path, hidden):
    torch.manual_seed(1)
    net = OpeningNet(hidden)
    path = tmp_path / OPENING_FILE
    write_model(path, net.layers())
    rows = random_setup_features(range(6))
    with torch.no_grad():
        want = torch.softmax(net(torch.tensor(rows)), dim=1)
    got = torch.tensor(opening_forward(str(path), rows))
    assert (got - want).abs().max().item() < 1e-5


def test_write_model_leaves_nothing_at_the_path_when_the_rust_loader_refuses(tmp_path):
    path = tmp_path / OPENING_FILE
    with pytest.raises(ValueError, match="layer 0 takes 77 inputs"):
        write_model(path, [(4, 77, [0.0] * (4 * 77), [0.0] * 4)])
    assert list(tmp_path.iterdir()) == []
    with pytest.raises(ValueError, match="needs 312 weights and 4 biases"):
        model_bytes([(4, 78, [0.0], [0.0] * 4)])


def tiny_samples(n, seed):
    g = torch.Generator().manual_seed(seed)
    return Samples(torch.rand(n, OPENING_FEATURES, generator=g), torch.randint(0, 4, (n,), generator=g),
                   torch.zeros(n, dtype=torch.bool), None)


def test_training_stops_three_epochs_after_the_best_and_keeps_the_best(monkeypatch):
    scores = iter([1.0, 0.9, 0.95, 0.91, 0.92, 0.1])
    seen = {}

    def scripted(net, samples):
        score = next(scores)
        seen[score] = copy.deepcopy(net.state_dict())
        return score

    monkeypatch.setattr(topen, "held_out_loss", scripted)
    net, s = fit(tiny_samples(64, 0), tiny_samples(16, 1), OpeningConfig(batch=16), log=lambda _: None)
    assert (s["best_epoch"], s["epochs"], s["best_held_out"]) == (2, 5, 0.9)
    assert all(torch.equal(v, seen[0.9][k]) for k, v in net.state_dict().items())


@pytest.mark.parametrize("field,value,message", [
    ("lr", 0.0, "learning rate"), ("batch", 0, "batch"), ("patience", 0, "patience"),
    ("max_epochs", 0, "max_epochs"), ("hidden", (64, 0), "hidden"),
])
def test_bad_training_settings_are_refused(field, value, message):
    with pytest.raises(ValueError, match=message):
        OpeningConfig(**{field: value}).check()


def test_cli_opening_train_writes_a_model_the_bots_load(tmp_path):
    nets = tmp_path / "nets"
    nets.mkdir()
    write_random_net(str(nets / "tiny.bin"), 1, 8, 1)
    a, b = tmp_path / "gen-a", tmp_path / "gen-b"
    write_games(a / "selfplay-1.jsonl.gz", 1, 3)  # no-trade self-play, seeds 1-3
    write_dir(b, [fixed_game(0, winner=3)])  # the only held-out game, a trading one
    write_explore(b / "selfplay-21.jsonl.gz", 21, 2, config=TRADES, full_simulations=6, fast_simulations=3,
                  full_probability=0.5, threads=2)
    out = nets / OPENING_FILE
    args = ["opening-train", "--games", str(a), str(b), "--out", str(out), "--compare-net", "tiny",
            "--nets-dir", str(nets)]
    r = cli(*args)
    assert r.returncode == 0, r.stderr
    assert "uniform 1.3863" in r.stdout and "net:tiny" in r.stdout
    assert "held-out cross-entropy, trading games (8 samples)" in r.stdout
    assert len(opening_forward(str(out), [[0.0] * OPENING_FEATURES])[0]) == 4
    Bot("net:tiny@5+o", 0, nets_dir=str(nets))  # the bots load it
    summary = json.loads((nets / "opening.bin.json").read_text())
    assert summary["held_out"]["all"]["samples"] == 8 and summary["best_epoch"] >= 1
    again = cli(*args)
    assert again.returncode == 2 and "already exists" in again.stderr


def test_cli_opening_train_checks_the_compare_net_before_reading_games(tmp_path):
    d = write_dir(tmp_path / "gen", [fixed_game(0), fixed_game(1)])
    out = tmp_path / "opening.bin"
    r = cli("opening-train", "--games", str(d), "--out", str(out), "--compare-net", "missing",
            "--nets-dir", str(tmp_path))
    assert r.returncode == 2 and "cannot read" in r.stderr
    assert not out.exists()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest -q tests/python/test_opening_train.py`
Expected: FAIL at collection with `ModuleNotFoundError: No module named 'alphasettler.opening'`.

- [ ] **Step 3: Write the implementation**

In `alphasettler/records.py`, append:

```python
def record_files(gen_dir: str | Path) -> list[Path]:
    """Finished record files in `gen_dir`, by name; temporary (dot) files are not."""
    return sorted(p for p in Path(gen_dir).glob("*.jsonl.gz") if not p.name.startswith("."))
```

In `alphasettler/train/data.py`, replace the whole `def record_files(...)` function with:

```python
# Finished record files in a generation directory (it lives in `records`, which needs no torch).
record_files = records.record_files
```

Create `alphasettler/opening.py`:

```python
"""Openings (spec 3d-1): replaying a recorded game's setup, and the opening model file.

A game's setup is its first 16 actions: (settlement, road) for each seat in snake order.
`replay_setup` replays them from the record's seed and config and keeps, for each of the 8
settlements, the seat, the node and the nodes legal at that pick and, on request, the position
before the pick, the opening features right after it (before its road) and a net's view of that
same position.

The model file `<nets_dir>/opening.bin` (bots/src/opening_model.rs has the byte table): magic,
format version, OPENING_HASH, layer count, each layer's rows and columns, then each layer's f32
weights (row-major, rows = outputs) and biases, little-endian. `write_model` has the Rust loader
read the file back before it takes the name, so a file the bots would refuse is never left at the
path. No torch here: the diagnosis imports this module."""

from __future__ import annotations

import os
import struct
import sys
from array import array
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from alphasettler import files
from alphasettler._engine import (
    OPENING_FEATURES, OPENING_FORMAT_VERSION, OPENING_HASH, Game, opening_features, opening_forward,
)

MAGIC = b"ASOPEN\0\0"
SETUP_ACTIONS = 16
# BuildSettlement(n) is action 25 + n and BuildRoad(e) is 133 + e (engine/src/action.rs).
SETTLEMENT_BASE = 25
ROAD_BASE = 133
NODES = 54
EDGES = 72

Layer = tuple[int, int, Sequence[float], Sequence[float]]


@dataclass
class Pick:
    """One setup settlement."""
    seat: int
    node: int
    legal: list[int]  # the nodes legal at this pick, in engine order
    before: Game | None = None  # the position at this pick (keep_positions)
    features: list[float] | None = None  # opening_features for `seat` right after it, before its road
    value: list[float] | None = None  # value(game, seat) at that same position


@dataclass
class Setup:
    seed: int
    trading: bool  # the game allowed domestic offers
    winner: int | None
    board: dict  # Game.snapshot()["board"]
    picks: list[Pick]  # the 8 settlements in pick order
    after: Game  # the setup done, before the first roll

    def nodes(self, seat: int) -> list[int]:
        """`seat`'s settlements in the order it placed them."""
        return [p.node for p in self.picks if p.seat == seat]


def replay_setup(record: dict, *, features: bool = False, value: Callable[[Game, int], list[float]] | None = None,
                 keep_positions: bool = False) -> Setup:
    """The setup of `record` (a self-play record: `seed`, `config`, `actions`, `winner`). ValueError
    naming the game's seed if its first 16 actions are not a legal setup."""
    seed = record["seed"]
    actions = record["actions"]
    if len(actions) < SETUP_ACTIONS:
        raise ValueError(f"game {seed}: {len(actions)} actions, fewer than the {SETUP_ACTIONS} of the setup")
    g = Game(seed, record["config"])
    board = g.snapshot()["board"]
    picks: list[Pick] = []
    for k, a in enumerate(actions[:SETUP_ACTIONS]):
        settlement = k % 2 == 0
        base, count = (SETTLEMENT_BASE, NODES) if settlement else (ROAD_BASE, EDGES)
        if not (isinstance(a, int) and base <= a < base + count):
            kind = "settlement" if settlement else "road"
            raise ValueError(f"game {seed}: action {k} ({a!r}) is not a setup {kind}")
        seat = g.current_actor
        if settlement:
            legal = [x - SETTLEMENT_BASE for x in g.legal_actions()]
            before = g.copy() if keep_positions else None
        try:
            g.apply(a)
        except ValueError as e:
            raise ValueError(f"game {seed}: action {k} ({a}): {e}") from None
        if settlement:
            picks.append(Pick(seat, a - SETTLEMENT_BASE, legal, before,
                              opening_features(g, seat) if features else None,
                              value(g, seat) if value is not None else None))
    trading = record["config"].get("max_offers_per_turn", 0) > 0
    return Setup(seed, trading, record["winner"], board, picks, g)


def model_bytes(layers: Sequence[Layer]) -> bytes:
    """An opening model file holding `layers`: (rows, cols, rows x cols weights row-major, rows
    biases) each, in order."""
    assert sys.byteorder == "little", "the opening model file is little-endian"
    out = [MAGIC, struct.pack("<IQI", OPENING_FORMAT_VERSION, OPENING_HASH, len(layers))]
    out += [struct.pack("<II", rows, cols) for rows, cols, _, _ in layers]
    for rows, cols, w, b in layers:
        if len(w) != rows * cols or len(b) != rows:
            raise ValueError(f"a {rows} x {cols} layer needs {rows * cols} weights and {rows} biases, "
                             f"got {len(w)} and {len(b)}")
        out += [array("f", w).tobytes(), array("f", b).tobytes()]
    return b"".join(out)


def write_model(path: str | Path, layers: Sequence[Layer]) -> None:
    """Write the model to `path` through a temp file the Rust loader reads first (raising its error,
    with nothing left behind, for a file the bots would refuse)."""
    path = Path(path)
    tmp = files.temp_path(path)
    try:
        tmp.write_bytes(model_bytes(layers))
        opening_forward(str(tmp), [[0.0] * OPENING_FEATURES])
        files.fsync_file(tmp)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
```

Create `alphasettler/train/opening.py`:

```python
"""The opening model (spec 3d-1 Section 4.4): an MLP over the 78 opening features, trained on
recorded games' setups to predict the winner relative to the seat that just placed a settlement,
written in the format of bots/src/opening_model.rs (`alphasettler.opening.write_model`).

Samples: after each of a game's 8 settlements (before its road), the features for the seat that
placed it and the winner relative to that seat; a game with no winner adds nothing. Held out:
every game whose seed % HELD_OUT_EVERY == 0, in every directory. Fit: cross-entropy, Adam, an epoch
at a time on the CPU (the model has about 10,000 weights), until the held-out loss has not improved
for `patience` epochs; the best held-out epoch is kept."""

from __future__ import annotations

import copy
import json
import math
from array import array
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from alphasettler import files, records
from alphasettler._engine import OPENING_FEATURES, Game, net_evaluate
from alphasettler.opening import Setup, replay_setup, write_model
from alphasettler.train.data import HELD_OUT_EVERY

SEATS = 4
HIDDEN = (64, 64)
# A uniform guess's cross-entropy.
UNIFORM = math.log(SEATS)
# The compare net's probability of the winner is floored here before the log, so one confident
# miss costs -ln(1e-12) rather than infinity.
PROBABILITY_FLOOR = 1e-12


@dataclass(frozen=True)
class OpeningConfig:
    lr: float = 1e-3
    batch: int = 1024
    patience: int = 3
    max_epochs: int = 100
    seed: int = 0
    hidden: tuple[int, ...] = HIDDEN

    def check(self) -> None:
        if not (math.isfinite(self.lr) and self.lr > 0):
            raise ValueError(f"the learning rate must be finite and positive, got {self.lr}")
        if self.batch < 1:
            raise ValueError(f"batch must be at least 1, got {self.batch}")
        if self.patience < 1:
            raise ValueError(f"patience must be at least 1 epoch, got {self.patience}")
        if self.max_epochs < 1:
            raise ValueError(f"max_epochs must be at least 1, got {self.max_epochs}")
        if not self.hidden or any(h < 1 for h in self.hidden):
            raise ValueError(f"hidden widths must be positive, got {self.hidden}")


class OpeningNet(nn.Module):
    """78 features -> `hidden` widths (ReLU) -> 4 seat-relative win logits."""

    def __init__(self, hidden: tuple[int, ...] = HIDDEN):
        super().__init__()
        dims = [OPENING_FEATURES, *hidden, SEATS]
        self.linears = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims, dims[1:]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.linears):
            x = layer(x)
            if i < len(self.linears) - 1:
                x = torch.relu(x)
        return x

    def layers(self) -> list[tuple[int, int, list[float], list[float]]]:
        """(rows, cols, weights row-major, biases) per layer, as `write_model` takes them."""
        return [(m.out_features, m.in_features, m.weight.detach().cpu().reshape(-1).tolist(),
                 m.bias.detach().cpu().tolist()) for m in self.linears]


@dataclass
class Samples:
    x: torch.Tensor  # n x OPENING_FEATURES, float32
    y: torch.Tensor  # n, int64: the winner relative to the sample's seat
    trading: torch.Tensor  # n, bool: the game allowed domestic offers
    compare: torch.Tensor | None  # n, float64: -ln(the compare net's probability of the winner)

    def __len__(self) -> int:
        return len(self.y)


class _Builder:
    """Samples gathered a game at a time into flat arrays (half a million rows of 78 floats would
    not fit comfortably as Python lists)."""

    def __init__(self, compare: bool):
        self.x, self.y, self.trading = array("f"), array("b"), array("b")
        self.compare = array("d") if compare else None

    def add(self, setup: Setup, winner: int) -> None:
        for pick in setup.picks:
            self.x.extend(pick.features)
            self.y.append((winner - pick.seat) % SEATS)
            self.trading.append(setup.trading)
            if self.compare is not None:
                self.compare.append(-math.log(max(pick.value[winner], PROBABILITY_FLOOR)))

    def samples(self) -> Samples:
        n = len(self.y)
        x = (torch.frombuffer(bytearray(self.x.tobytes()), dtype=torch.float32).reshape(n, OPENING_FEATURES)
             if n else torch.zeros(0, OPENING_FEATURES))
        compare = None if self.compare is None else torch.tensor(list(self.compare), dtype=torch.float64)
        return Samples(x, torch.tensor(list(self.y), dtype=torch.int64),
                       torch.tensor(list(self.trading), dtype=torch.bool), compare)


def load_samples(dirs: list[Path], compare: Callable[[Game, int], list[float]] | None = None
                 ) -> tuple[Samples, Samples, dict]:
    """(training samples, held-out samples, counts) from every record file in `dirs`. With
    `compare(game, seat)` (four win probabilities by absolute seat), each held-out sample keeps
    -ln(its probability of the winner). ValueError naming the file for a record that does not replay,
    and for a directory with no records, or no training or no held-out samples at all."""
    train, held = _Builder(False), _Builder(compare is not None)
    counts = {"games": 0, "no_winner": 0, "held_out_games": 0}
    for d in dirs:
        paths = records.record_files(d)
        if not paths:
            raise ValueError(f"no record files (*.jsonl.gz) in {d}")
        for path in paths:
            for rec in records.read(path):
                counts["games"] += 1
                if rec["winner"] is None:
                    counts["no_winner"] += 1
                    continue
                is_held = rec["seed"] % HELD_OUT_EVERY == 0
                try:
                    setup = replay_setup(rec, features=True, value=compare if is_held else None)
                except ValueError as e:
                    raise ValueError(f"{path}: {e}") from None
                (held if is_held else train).add(setup, rec["winner"])
                counts["held_out_games"] += is_held
    where = ", ".join(str(d) for d in dirs)
    if len(train.y) == 0:
        raise ValueError(f"no training games (decided, seed % {HELD_OUT_EVERY} != 0) in {where}")
    if len(held.y) == 0:
        raise ValueError(f"no held-out games (decided, seed % {HELD_OUT_EVERY} == 0) in {where}")
    return train.samples(), held.samples(), counts


def held_out_loss(net: OpeningNet, s: Samples) -> float:
    """Mean cross-entropy of `net` on `s`, in float64."""
    with torch.no_grad():
        return F.cross_entropy(net(s.x).double(), s.y).item()


def fit(train: Samples, held: Samples, cfg: OpeningConfig, log: Callable[[str], None] = print
        ) -> tuple[OpeningNet, dict]:
    """Train epoch by epoch until the held-out loss has not improved for `cfg.patience` epochs (or
    `cfg.max_epochs`); return the net at its best held-out epoch and a summary."""
    cfg.check()
    torch.manual_seed(cfg.seed)
    order_rng = torch.Generator().manual_seed(cfg.seed)
    net = OpeningNet(cfg.hidden)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    best_loss, best_epoch, best_state, stale = math.inf, 0, copy.deepcopy(net.state_dict()), 0
    history = []
    for epoch in range(1, cfg.max_epochs + 1):
        net.train()
        order = torch.randperm(len(train), generator=order_rng)
        for lo in range(0, len(train), cfg.batch):
            idx = order[lo:lo + cfg.batch]
            loss = F.cross_entropy(net(train.x[idx]), train.y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        net.eval()
        h = held_out_loss(net, held)
        history.append({"epoch": epoch, "held_out": round(h, 6)})
        log(f"epoch {epoch}: held-out cross-entropy {h:.4f}")
        if h < best_loss:
            best_loss, best_epoch, best_state, stale = h, epoch, copy.deepcopy(net.state_dict()), 0
        else:
            stale += 1
            if stale >= cfg.patience:
                break
    net.load_state_dict(best_state)
    return net, {"best_epoch": best_epoch, "epochs": len(history), "best_held_out": best_loss, "history": history}


def report(net: OpeningNet, held: Samples) -> dict:
    """Held-out cross-entropy of the model, a uniform guess and the compare net (if any), on every
    held-out sample and on the trading games' alone (None where there are none)."""

    def row(mask: torch.Tensor) -> dict | None:
        n = int(mask.sum())
        if n == 0:
            return None
        with torch.no_grad():
            model = F.cross_entropy(net(held.x[mask]).double(), held.y[mask]).item()
        compare = None if held.compare is None else held.compare[mask].mean().item()
        return {"samples": n, "model": model, "uniform": UNIFORM, "compare": compare}

    return {"all": row(torch.ones(len(held), dtype=torch.bool)), "trading": row(held.trading)}


def train_opening(dirs: list[Path], out: Path, cfg: OpeningConfig, compare_net: str | None = None,
                  nets_dir: str | None = None, log: Callable[[str], None] = print) -> dict:
    """Load the samples, fit, write the model to `out` and a summary to `out`.json; return the summary."""
    compare = None if compare_net is None else (lambda g, seat: net_evaluate(g, seat, compare_net, nets_dir))
    train, held, counts = load_samples(dirs, compare)
    log(f"samples: {len(train)} training, {len(held)} held out, from {counts['games']} games "
        f"({counts['no_winner']} without a winner)")
    net, fitted = fit(train, held, cfg, log)
    write_model(out, net.layers())
    summary = {"dirs": [str(d) for d in dirs], "config": asdict(cfg), "compare_net": compare_net, **counts,
               "train_samples": len(train), "held_out_samples": len(held), **fitted, "held_out": report(net, held)}
    files.atomic_write_text(f"{out}.json", json.dumps(summary, indent=1) + "\n")
    return summary


def format_report(s: dict, out: Path) -> list[str]:
    """The lines `opening-train` prints after training."""

    def line(label: str, row: dict | None) -> str:
        if row is None:
            return f"held-out cross-entropy, {label}: none held out"
        text = (f"held-out cross-entropy, {label} ({row['samples']} samples): model {row['model']:.4f}, "
                f"uniform {row['uniform']:.4f}")
        if row["compare"] is not None:
            text += f", net:{s['compare_net']} {row['compare']:.4f}"
        return text

    return [f"best epoch {s['best_epoch']} of {s['epochs']}", line("all games", s["held_out"]["all"]),
            line("trading games", s["held_out"]["trading"]), f"wrote {out} (summary {out}.json)"]
```

In `alphasettler/cli.py`:

1. In the module docstring's first sentence add `` `opening-train` `` after `` `net-upgrade` `` in the list of commands, and append the sentence: `` `opening-train` (the train extra) fits the opening model of spec 3d-1 on recorded games' setups. ``
2. In `_parser`, after the `net-upgrade` subparser, add:

```python
    ot = sub.add_parser("opening-train", help="train the opening model on recorded games' setups (train extra)")
    ot.add_argument("--games", nargs="+", required=True, metavar="DIR", help="generation directories of records")
    ot.add_argument("--out", required=True, help="the model file to write, e.g. <nets-dir>/opening.bin (must not exist)")
    ot.add_argument("--compare-net", default=None, metavar="NAME",
                    help="also score this net's value head on the held-out positions")
    ot.add_argument("--nets-dir", default=DEFAULT_NETS_DIR, help="where --compare-net finds NAME.bin")
    ot.add_argument("--seed", type=int, default=0)
```

3. After `_net_upgrade`, add:

```python
def _opening_train(args) -> int:
    if _train_missing():
        return 2
    from alphasettler._engine import resolve_net
    from alphasettler.train.opening import OpeningConfig, format_report, train_opening

    out = Path(args.out)
    if out.exists():
        raise ValueError(f"{out} already exists; write the opening model to a new path")
    if args.compare_net is not None:
        resolve_net(args.compare_net, args.nets_dir)  # a bad net fails now, before the games are read
    cfg = OpeningConfig(seed=args.seed)
    cfg.check()
    try:
        s = train_opening([Path(d) for d in args.games], out, cfg, args.compare_net, args.nets_dir)
    except KeyboardInterrupt:
        print("interrupted: nothing written", file=sys.stderr)
        return 130
    for line in format_report(s, out):
        print(line)
    return 0
```

4. In `main`, after the `net-upgrade` branch, add:

```python
        elif args.command == "opening-train":
            return _opening_train(args)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/pytest -q tests/python/test_opening_train.py tests/python/test_train.py`
Expected: PASS (17 tests in the new file; `test_train.py` unchanged).

- [ ] **Step 5: The full suite**

Run the Global Constraints sequence (no Rust changed in this task, so `cargo` passes as before; no rebuild). Expected: `.venv/bin/pytest -q -rs` all pass, nothing skipped.

- [ ] **Step 6: Commit**

```bash
git add alphasettler/records.py alphasettler/train/data.py alphasettler/opening.py alphasettler/train/opening.py \
  alphasettler/cli.py tests/python/opening_games.py tests/python/test_opening_train.py
git diff --cached | grep -nE 'sk-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{36}|password=[^[:space:]]+'
git -c user.email=sean.may101@gmail.com commit -m "train: opening-train fits the opening model on recorded setups and reports it beside a uniform guess and a net's value head"
```

---

### Task 5: `alphasettler opening-diagnose` — D1 to D4 and the verdict

**Files:**
- Create: `alphasettler/diagnose.py`, `tests/python/test_opening_diagnose.py`
- Modify: `alphasettler/cli.py` (`opening-diagnose`)

**Interfaces:**
- Consumes: Task 4's `alphasettler.opening.{replay_setup, Setup, Pick, SETTLEMENT_BASE, SETUP_ACTIONS, model_bytes}` and `alphasettler.records.record_files`; Task 3's `net_evaluate` and `Bot.act` without the GIL; `net_adjacency()["node_tiles"]`, `resolve_config`, `resolve_net`; the test helper `opening_games.{fixed_game, write_dir}`.
- Produces (`alphasettler.diagnose`):
  - constants `RESOURCES`, `DOTS`, `GROUPS` (`"no wood and brick", "no wheat", "no ore", "no wood or no brick", "no sheep", "all five"`), `WEAK` (the first three), `MISSING`, `WITHIN = (0, 1, 2, 3)`, `AUDIT_BUDGETS = (1000, 8000)`, `AUDIT_SEED = 0`, `FORCED_SHARE = 1/3`, `FORCED_WITHIN = 1`, `MISJUDGED_SES = 2.0`
  - `node_pips(board, node) -> dict[str, int]`, `spot_pips(board, node) -> int`, `touched(board, nodes) -> set[str]`, `group(board, nodes) -> str`, `opening_pips(board, nodes) -> int`
  - `gap(setup, seat) -> int | None` (D1 for one weak opening)
  - `class Diagnosis(value)`: `.add(setup)`, `.gaps: dict[str, list[int | None]]` (weak groups), `.calibration: dict[str, list[tuple[float, bool]]]` (every group), `.games`, `.no_winner`, `.seats`
  - `calibration(pairs) -> tuple[int, float, float, float, float] | None` (n, predicted, actual, gap, se)
  - `d1_table(gaps) -> str`, `d2_table(cal) -> str`, `d3_table(rows, games, net, source) -> str`, `d4_table(bot, setups) -> str`, `verdict(gaps, cal) -> str`
  - `rechoose(name, nets_dir, setup, k) -> int`, `weak_share(setups, nodes) -> float`, `audit(setups, net, nets_dir, budgets, threads) -> list[tuple[str, int, float | None, float]]`, `play_setup(bot, seed, config, nets_dir) -> Setup`, `play_setups(bot, seed_start, seeds, config, nets_dir, threads) -> list[Setup]`
  - `diagnose(dirs, net, nets_dir, audit_games, budgets, threads) -> str`
  - CLI: `alphasettler opening-diagnose --games DIR [DIR ...] --net NAME [--nets-dir D] [--audit N] [--audit-budgets N ...] [--threads T] [--out FILE]` (D1–D3 and the verdict) and `alphasettler opening-diagnose --bot NAME --seeds N [--seed-start S] [--trades | --no-trades] [--nets-dir D] [--threads T] [--out FILE]` (D4).

- [ ] **Step 1: Write the failing tests**

Create `tests/python/test_opening_diagnose.py`:

```python
"""The opening diagnosis (spec 3d-1 Section 3): groups, D1 to D4, the verdict and `opening-diagnose`."""

import math
import subprocess
import sys

import pytest

from alphasettler import Game
from alphasettler._engine import OPENING_FILE, write_random_net
from alphasettler.arena import TRADES
from alphasettler.diagnose import (
    GROUPS, Diagnosis, calibration, d1_table, d2_table, d4_table, gap, group, opening_pips, play_setups, verdict,
)
from alphasettler.opening import model_bytes, replay_setup
from opening_games import fixed_game, write_dir

BOARD = Game(0, None).snapshot()["board"]


def cli(*args):
    return subprocess.run([sys.executable, "-m", "alphasettler", *args], capture_output=True, text=True)


@pytest.fixture
def nets(tmp_path):
    """`tiny.bin` and an `opening.bin` that scores every spot the same."""
    d = tmp_path / "nets"
    d.mkdir()
    write_random_net(str(d / "tiny.bin"), 1, 8, 1)
    (d / OPENING_FILE).write_bytes(model_bytes([(4, 78, [0.0] * (4 * 78), [0.0] * 4)]))
    return d


@pytest.mark.parametrize("nodes,want", [
    ([9, 29], "all five"),  # wheat, wheat, brick + wood, ore, sheep
    ([4, 12], "no wood or no brick"),  # ore, wheat + ore, brick, sheep
    ([31, 41], "no ore"),  # brick, sheep, wheat + wood, wheat, sheep
    ([19, 26], "no wheat"),  # brick, wood, brick + sheep: no ore either, and no wheat comes first
    ([0, 4], "no wood and brick"),  # ore + ore, wheat
    ([53, 50], "no wood and brick"),  # sheep only: the first group that applies
    ([9, 30], "no sheep"),  # wheat, wheat, brick + ore, brick, wood
])
def test_groups_are_the_first_that_applies(nodes, want):
    assert group(BOARD, nodes) == want


def test_opening_pips_sum_over_the_settlements():
    # Node 31: brick 5 (4), sheep 4 (3), wheat 9 (4); node 41: wood 2 (1), wheat 9 (4), sheep 6 (5).
    assert opening_pips(BOARD, [31, 41]) == 21


def test_d1_on_the_fixed_game_matches_a_hand_count():
    s = replay_setup(fixed_game())
    assert [group(s.board, s.nodes(p)) for p in range(4)] == [
        "all five", "no wood or no brick", "all five", "no ore"]
    # Seat 3 (no ore) took node 41 (10 pips) when the best legal ore spot was node 44 (sheep 6, ore 3,
    # wheat 3: 9), then node 31 (11) with 44 still the best: gaps 1 and 2, so within 1 at either pick.
    assert gap(s, 3) == 1
    d = Diagnosis(None)
    d.add(s)
    table = d1_table(d.gaps)
    assert "| no ore | 1 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 |" in table
    assert "| no wheat | 0 | - | - | - | - | - |" in table
    assert "| weak groups | 1 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 |" in table


def test_d1_shares_are_cumulative_and_count_openings_with_no_alternative():
    table = d1_table({"no wood and brick": [None, 3], "no wheat": [-2, 0, 1, 2], "no ore": []})
    assert "| no wood and brick | 2 | 0.000 | 0.000 | 0.000 | 0.500 | 0.500 |" in table
    assert "| no wheat | 4 | 0.500 | 0.750 | 1.000 | 1.000 | 0.000 |" in table
    assert "| no ore | 0 | - | - | - | - | - |" in table
    assert "| weak groups | 6 | 0.333 | 0.500 | 0.667 | 0.833 | 0.167 |" in table


def test_d2_reads_each_seat_s_own_view_after_the_setup_and_skips_games_without_a_winner():
    seen = []

    def value(g, seat):
        seen.append((g.phase, seat))
        return [0.1, 0.2, 0.3, 0.4]

    d = Diagnosis(value)
    d.add(replay_setup(fixed_game(0, winner=3)))
    d.add(replay_setup(fixed_game(1, winner=None)))
    assert seen == [("pre_roll", seat) for seat in range(4)]
    assert d.calibration["no ore"] == [(0.4, True)]
    assert sorted(d.calibration["all five"]) == [(0.1, False), (0.3, False)]
    assert d.calibration["no wood or no brick"] == [(0.2, False)]
    assert (d.games, d.no_winner, d.seats) == (2, 1, 8)


def test_calibration_gap_and_standard_error():
    n, predicted, actual, g, se = calibration([(0.5, True), (0.5, False), (0.3, False), (0.3, False)])
    # Gaps -0.5, 0.5, 0.3, 0.3: mean 0.15; squared deviations sum to 0.59 over n - 1 = 3.
    assert (n, predicted, actual) == (4, pytest.approx(0.4), 0.25)
    assert g == pytest.approx(0.15) and se == pytest.approx(math.sqrt(0.59 / 3 / 4))


def test_tables_print_empty_groups_as_dashes():
    d = Diagnosis(lambda g, seat: [0.25] * 4)
    text = d2_table(d.calibration)
    for g in GROUPS:
        assert f"| {g} | 0 | - | - | - | - |" in text
    assert "| weak groups | 0 | - | - | - | - |" in text
    assert verdict(d.gaps, d.calibration).startswith("verdict: undecided")
    assert calibration([(0.3, True)])[4] == math.inf


ALL_FIVE = [(0.25, False)] * 30 + [(0.25, True)] * 10


@pytest.mark.parametrize("weak_gaps,weak_pairs,word", [
    # A spot within 1 pip in 1 of 6 weak openings, under a third: forced, whatever D2 says.
    ([2, 5, None, 1, 3, 4], [(0.25, False)] * 4, "forced"),
    # Alternatives were near; weak openings predicted 0.30 but won 4 of 44 (gap +0.21 against all
    # five's 0, about 2.5 standard errors of the difference): misjudged.
    ([0, 1, 1], [(0.3, False)] * 40 + [(0.3, True)] * 4, "misjudged"),
    # Alternatives were near and the weak groups are calibrated like all five: misdecided.
    ([0, 1, 1], ALL_FIVE, "misdecided"),
])
def test_the_verdict_rule(weak_gaps, weak_pairs, word):
    gaps = {"no wood and brick": [], "no wheat": weak_gaps, "no ore": []}
    cal = {g: [] for g in GROUPS}
    cal["no wheat"] = weak_pairs
    cal["all five"] = ALL_FIVE
    assert verdict(gaps, cal).startswith(f"verdict: {word} (")


def test_d4_plays_only_the_setup_and_does_not_depend_on_threads(nets):
    one = play_setups("net:tiny@4", 2_000_000_000, 4, TRADES, str(nets), threads=1)
    three = play_setups("net:tiny@4", 2_000_000_000, 4, TRADES, str(nets), threads=3)
    assert [s.picks for s in one] == [s.picks for s in three]
    assert all(s.after.phase == "pre_roll" and len(s.picks) == 8 for s in one)
    text = d4_table("net:tiny@4", one)
    assert "D4, net:tiny@4: the openings of 4 setups (16 seats)" in text and "| weak groups |" in text
    # A +o bot whose model scores every spot the same takes the first legal spot at every pick.
    tied = play_setups("net:tiny@4+o", 2_000_000_000, 1, TRADES, str(nets), threads=1)[0]
    assert all(p.node == p.legal[0] for p in tied.picks)


def test_cli_prints_d1_to_d3_and_the_verdict_and_writes_out(tmp_path, nets):
    d = write_dir(tmp_path / "gen", [fixed_game(0), fixed_game(1, winner=0)])
    out = tmp_path / "diagnosis.md"
    r = cli("opening-diagnose", "--games", str(d), "--net", "tiny", "--nets-dir", str(nets), "--audit", "1",
            "--audit-budgets", "2", "3", "--threads", "2", "--out", str(out))
    assert r.returncode == 0, r.stderr
    for heading in ("2 games (0 without a winner)", "D1, forced or chosen", "D2, misjudged", "D3, search budget",
                    "\nverdict: "):
        assert heading in r.stdout
    # One audited game: 8 picks at each budget, after the recorded row.
    assert "| recorded | 8 | - |" in r.stdout and "| 2 | 8 |" in r.stdout and "| 3 | 8 |" in r.stdout
    assert out.read_text() == r.stdout
    r = cli("opening-diagnose", "--bot", "net:tiny@3", "--seeds", "2", "--trades", "--nets-dir", str(nets))
    assert r.returncode == 0, r.stderr
    assert "D4, net:tiny@3: the openings of 2 setups (8 seats)" in r.stdout


@pytest.mark.parametrize("args,message", [
    (["--games", "DIR"], "--games needs --net"),
    (["--games", "DIR", "--net", "missing"], "cannot read"),
    (["--games", "DIR", "--net", "tiny", "--audit", "-1"], "--audit must be 0 or more"),
    (["--bot", "net:tiny@3"], "--bot needs --seeds"),
    (["--bot", "net:tiny@3+o+o", "--seeds", "2"], "unknown bot"),
])
def test_cli_refuses_bad_arguments_before_reading_or_playing(tmp_path, nets, args, message):
    d = write_dir(tmp_path / "gen", [fixed_game()])
    r = cli("opening-diagnose", *[str(d) if a == "DIR" else a for a in args], "--nets-dir", str(nets))
    assert r.returncode == 2 and message in r.stderr, r.stderr


def test_the_diagnosis_does_not_import_torch():
    r = subprocess.run([sys.executable, "-c",
                        "import sys, alphasettler.diagnose, alphasettler.opening; print('torch' in sys.modules)"],
                       capture_output=True, text=True)
    assert r.stdout.strip() == "False", r.stderr
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest -q tests/python/test_opening_diagnose.py`
Expected: FAIL at collection with `ModuleNotFoundError: No module named 'alphasettler.diagnose'`.

- [ ] **Step 3: Write the implementation**

Create `alphasettler/diagnose.py`:

```python
"""The opening diagnosis (spec 3d-1 Section 3), printed as Markdown tables.

An opening is a seat's two setup settlements. Its group is the first of GROUPS that applies; its
pips are the dice dots of the tiles its settlements touch (a tile two of them touch counts twice).
- D1, forced or chosen: for each weak opening, at each of its two picks, the chosen spot's pips
  against the most pips of a spot legal at that pick touching the missing resource (wood or brick
  for "no wood and brick"); per group, the share of openings where such a spot was within 0..3
  pips at either pick, and the share with none at all.
- D2, misjudged: a net's win probability for each seat right after the setup (before the first
  roll), from that seat's own view, against the seat's result, per group. Games without a winner
  are left out.
- D3, search budget: the setup settlements of games drawn from one directory, re-chosen from the
  recorded position by the net at larger budgets (a fresh bot: setup positions hide nothing).
- D4: a bot's own openings over N setups (four copies of the bot; only the setup is played).
No torch here: the standard library and the extension only."""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from alphasettler import records
from alphasettler._engine import Bot, Game, net_adjacency, net_evaluate, resolve_config
from alphasettler.opening import SETTLEMENT_BASE, SETUP_ACTIONS, Setup, replay_setup

SEATS = 4
RESOURCES = ("wood", "brick", "sheep", "wheat", "ore")
# Dice dots per number token.
DOTS = {2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 8: 5, 9: 4, 10: 3, 11: 2, 12: 1}
NODE_TILES = net_adjacency()["node_tiles"]
GROUPS = ("no wood and brick", "no wheat", "no ore", "no wood or no brick", "no sheep", "all five")
WEAK = GROUPS[:3]
MISSING = {"no wood and brick": ("wood", "brick"), "no wheat": ("wheat",), "no ore": ("ore",)}
# D1's columns: a spot touching the missing resource within this many pips of the chosen one.
WITHIN = (0, 1, 2, 3)
AUDIT_BUDGETS = (1000, 8000)
AUDIT_SEED = 0
# The verdict rule (spec Section 3).
FORCED_SHARE = 1 / 3
FORCED_WITHIN = 1
MISJUDGED_SES = 2.0


def node_pips(board: dict, node: int) -> dict[str, int]:
    out = dict.fromkeys(RESOURCES, 0)
    for t in NODE_TILES[node]:
        r = board["tile_resource"][t]
        if r is not None:
            out[r] += DOTS[board["tile_number"][t]]
    return out


def spot_pips(board: dict, node: int) -> int:
    return sum(node_pips(board, node).values())


def touched(board: dict, nodes: Sequence[int]) -> set[str]:
    return {r for n in nodes for r, p in node_pips(board, n).items() if p > 0}


def group(board: dict, nodes: Sequence[int]) -> str:
    t = touched(board, nodes)
    if "wood" not in t and "brick" not in t:
        return "no wood and brick"
    if "wheat" not in t:
        return "no wheat"
    if "ore" not in t:
        return "no ore"
    if "wood" not in t or "brick" not in t:
        return "no wood or no brick"
    if "sheep" not in t:
        return "no sheep"
    return "all five"


def opening_pips(board: dict, nodes: Sequence[int]) -> int:
    return sum(spot_pips(board, n) for n in nodes)


def gap(setup: Setup, seat: int) -> int | None:
    """D1 for `seat`'s weak opening: the smallest, over its two picks, of the chosen spot's pips
    minus the most pips of a spot legal at that pick touching a missing resource; None if no such
    spot was legal at either pick."""
    missing = MISSING[group(setup.board, setup.nodes(seat))]
    gaps = []
    for p in setup.picks:
        if p.seat != seat:
            continue
        alternatives = [spot_pips(setup.board, n) for n in p.legal
                        if any(node_pips(setup.board, n)[r] > 0 for r in missing)]
        if alternatives:
            gaps.append(spot_pips(setup.board, p.node) - max(alternatives))
    return min(gaps) if gaps else None


class Diagnosis:
    """D1 and D2 gathered a game at a time, so the records are never all in memory. `value(game,
    seat)` gives four win probabilities by absolute seat; None skips D2."""

    def __init__(self, value: Callable[[Game, int], list[float]] | None):
        self.value = value
        self.gaps: dict[str, list[int | None]] = {g: [] for g in WEAK}
        self.calibration: dict[str, list[tuple[float, bool]]] = {g: [] for g in GROUPS}
        self.games = self.no_winner = self.seats = 0

    def add(self, setup: Setup) -> None:
        self.games += 1
        self.no_winner += setup.winner is None
        for seat in range(SEATS):
            self.seats += 1
            g = group(setup.board, setup.nodes(seat))
            if g in self.gaps:
                self.gaps[g].append(gap(setup, seat))
            if setup.winner is not None and self.value is not None:
                self.calibration[g].append((self.value(setup.after, seat)[seat], setup.winner == seat))


def _share(k: int, n: int) -> str:
    return f"{k / n:.3f}" if n else "-"


def _d1_row(label: str, gaps: list[int | None]) -> str:
    n = len(gaps)
    cells = [_share(sum(x is not None and x <= d for x in gaps), n) for d in WITHIN]
    cells.append(_share(sum(x is None for x in gaps), n))
    return f"| {label} | {n} | " + " | ".join(cells) + " |"


def d1_table(gaps: dict[str, list[int | None]]) -> str:
    lines = ["D1, forced or chosen: weak openings where a spot touching the missing resource was legal within "
             "d pips of the chosen spot, at either pick", "",
             "| group | openings | " + " | ".join(f"within {d}" for d in WITHIN) + " | none |",
             "|---|---|" + "---|" * (len(WITHIN) + 1)]
    lines += [_d1_row(g, gaps[g]) for g in WEAK]
    lines.append(_d1_row("weak groups", [x for g in WEAK for x in gaps[g]]))
    return "\n".join(lines)


def calibration(pairs: list[tuple[float, bool]]) -> tuple[int, float, float, float, float] | None:
    """(seats, mean predicted, actual win rate, gap = predicted - actual, the gap's standard error);
    None for no seats. One seat's standard error is inf."""
    n = len(pairs)
    if n == 0:
        return None
    predicted = sum(p for p, _ in pairs) / n
    actual = sum(w for _, w in pairs) / n
    mean = predicted - actual
    if n == 1:
        return n, predicted, actual, mean, math.inf
    var = sum((p - w - mean) ** 2 for p, w in pairs) / (n - 1)
    return n, predicted, actual, mean, math.sqrt(var / n)


def _d2_row(label: str, pairs: list[tuple[float, bool]]) -> str:
    c = calibration(pairs)
    if c is None:
        return f"| {label} | 0 | - | - | - | - |"
    n, predicted, actual, g, se = c
    return f"| {label} | {n} | {predicted:.3f} | {actual:.3f} | {g:+.3f} | {se:.3f} |"


def d2_table(cal: dict[str, list[tuple[float, bool]]]) -> str:
    lines = ["D2, misjudged: the net's win probability for each seat after the setup, from its own view, "
             "against its result (decided games)", "",
             "| group | seats | predicted | actual | gap | se |", "|---|---|---|---|---|---|"]
    lines += [_d2_row(g, cal[g]) for g in GROUPS]
    lines.append(_d2_row("weak groups", [x for g in WEAK for x in cal[g]]))
    return "\n".join(lines)


def verdict(gaps: dict[str, list[int | None]], cal: dict[str, list[tuple[float, bool]]]) -> str:
    """Spec Section 3's rule over the three weak groups pooled."""
    weak = [x for g in WEAK for x in gaps[g]]
    share = sum(x is not None and x <= FORCED_WITHIN for x in weak) / len(weak) if weak else None
    if share is not None and share < FORCED_SHARE:
        return (f"verdict: forced (a spot within {FORCED_WITHIN} pip touching the missing resource in {share:.3f} "
                f"of {len(weak)} weak openings, under 1/3)")
    w, a = calibration([x for g in WEAK for x in cal[g]]), calibration(cal["all five"])
    if share is None or w is None or a is None:
        return "verdict: undecided (no weak openings, or no decided weak or all-five seats)"
    diff, se = w[3] - a[3], math.sqrt(w[4] ** 2 + a[4] ** 2)
    ses = diff / se if se > 0 else 0.0 if diff == 0 else math.copysign(math.inf, diff)
    near = f"a spot within {FORCED_WITHIN} pip in {share:.3f} of {len(weak)} weak openings"
    if diff > MISJUDGED_SES * se:
        return (f"verdict: misjudged ({near}; the weak groups' gap {w[3]:+.3f} exceeds all five's {a[3]:+.3f} "
                f"by {ses:.2f} standard errors)")
    return (f"verdict: misdecided ({near}; the weak groups' gap {w[3]:+.3f} is {ses:.2f} standard errors from "
            f"all five's {a[3]:+.3f}, not more than {MISJUDGED_SES:g})")


def rechoose(name: str, nets_dir: str, setup: Setup, k: int) -> int:
    """The node a fresh `name` bot picks at `setup`'s pick `k` (its position kept by `replay_setup`)."""
    bot = Bot(name, (setup.seed * 8 + k) % 2**64, nets_dir)
    return bot.act(setup.picks[k].before) - SETTLEMENT_BASE


def weak_share(setups: list[Setup], nodes: list[list[int]]) -> float:
    """The share of seats whose opening is weak, with `nodes[i]` replacing setup i's picked nodes."""
    weak = seats = 0
    for s, chosen in zip(setups, nodes):
        for seat in range(SEATS):
            weak += group(s.board, [n for p, n in zip(s.picks, chosen) if p.seat == seat]) in WEAK
            seats += 1
    return weak / seats if seats else math.nan


def audit(setups: list[Setup], net: str, nets_dir: str, budgets: Sequence[int], threads: int
          ) -> list[tuple[str, int, float | None, float]]:
    """D3 rows (budget, picks, share changed, weak share of the re-chosen openings), after a
    `recorded` row for the openings as played. Each pick is re-chosen from its recorded position."""
    tasks = [(s, k) for s in setups for k in range(len(s.picks))]
    rows: list[tuple[str, int, float | None, float]] = [
        ("recorded", len(tasks), None, weak_share(setups, [[p.node for p in s.picks] for s in setups]))]
    with ThreadPoolExecutor(threads) as ex:
        for b in budgets:
            name = f"net:{net}@{b}"
            chosen = list(ex.map(lambda t: rechoose(name, nets_dir, t[0], t[1]), tasks))
            per_setup, at = [], 0
            for s in setups:
                per_setup.append(chosen[at:at + len(s.picks)])
                at += len(s.picks)
            changed = sum(c != s.picks[k].node for c, (s, k) in zip(chosen, tasks))
            rows.append((str(b), len(tasks), changed / len(tasks) if tasks else math.nan,
                         weak_share(setups, per_setup)))
    return rows


def d3_table(rows: list[tuple[str, int, float | None, float]], games: int, net: str, source: Path) -> str:
    lines = [f"D3, search budget: the setup settlements of {games} games from {source}, re-chosen from the "
             f"recorded position by net:{net}@budget (a fresh bot each)", "",
             "| budget | picks | changed | weak-group openings |", "|---|---|---|---|"]
    for budget, picks, changed, weak in rows:
        shown = "-" if changed is None else f"{changed:.3f}"
        lines.append(f"| {budget} | {picks} | {shown} | {weak:.3f} |")
    return "\n".join(lines)


def play_setup(bot: str, seed: int, config: dict | None, nets_dir: str) -> Setup:
    """Seed `seed`'s setup played by four copies of `bot`, each shown its own events."""
    g = Game(seed, config)
    bots = [Bot(bot, (seed * SEATS + p) % 2**64, nets_dir) for p in range(SEATS)]
    seen = [0] * SEATS
    actions = []
    for _ in range(SETUP_ACTIONS):
        p = g.current_actor
        log = g.log(p)
        bots[p].observe(p, log[seen[p]:])
        seen[p] = len(log)
        a = bots[p].act(g)
        g.apply(a)
        actions.append(a)
    return replay_setup({"seed": seed, "config": resolve_config(config), "actions": actions, "winner": None})


def play_setups(bot: str, seed_start: int, seeds: int, config: dict | None, nets_dir: str, threads: int
                ) -> list[Setup]:
    """D4's setups, seeds `seed_start` onwards; the result does not depend on `threads`."""
    with ThreadPoolExecutor(threads) as ex:
        return list(ex.map(lambda s: play_setup(bot, s, config, nets_dir), range(seed_start, seed_start + seeds)))


def d4_table(bot: str, setups: list[Setup]) -> str:
    counts = dict.fromkeys(GROUPS, 0)
    pips = seats = 0
    for s in setups:
        for seat in range(SEATS):
            nodes = s.nodes(seat)
            counts[group(s.board, nodes)] += 1
            pips += opening_pips(s.board, nodes)
            seats += 1
    mean = f"{pips / seats:.2f}" if seats else "-"
    lines = [f"D4, {bot}: the openings of {len(setups)} setups ({seats} seats), mean pips {mean}", "",
             "| group | seats | share |", "|---|---|---|"]
    lines += [f"| {g} | {counts[g]} | {_share(counts[g], seats)} |" for g in GROUPS]
    weak = sum(counts[g] for g in WEAK)
    lines.append(f"| weak groups | {weak} | {_share(weak, seats)} |")
    return "\n".join(lines)


def _files(dirs: Sequence[Path]) -> list[Path]:
    out = []
    for d in dirs:
        found = records.record_files(d)
        if not found:
            raise ValueError(f"no record files (*.jsonl.gz) in {d}")
        out += found
    return out


def _replay(path: Path, rec: dict, **kwargs) -> Setup:
    try:
        return replay_setup(rec, **kwargs)
    except ValueError as e:
        raise ValueError(f"{path}: {e}") from None


def diagnose(dirs: Sequence[Path], net: str, nets_dir: str, audit_games: int, budgets: Sequence[int],
             threads: int) -> str:
    """D1 and D2 over every game in `dirs`, D3 over `audit_games` games of the last directory (none
    for 0), and the verdict, as one Markdown report."""
    d = Diagnosis(lambda g, seat: net_evaluate(g, seat, net, nets_dir))
    for path in _files(dirs):
        for rec in records.read(path):
            d.add(_replay(path, rec))
    parts = [f"{d.games} games ({d.no_winner} without a winner), {d.seats} openings, from "
             f"{', '.join(str(x) for x in dirs)}; net {net}", d1_table(d.gaps), d2_table(d.calibration)]
    if audit_games > 0:
        recs = sorted(((p, r) for p in _files(dirs[-1:]) for r in records.read(p)), key=lambda pr: pr[1]["seed"])
        drawn = random.Random(AUDIT_SEED).sample(recs, min(audit_games, len(recs)))
        setups = [_replay(p, r, keep_positions=True) for p, r in drawn]
        parts.append(d3_table(audit(setups, net, nets_dir, budgets, threads), len(setups), net, dirs[-1]))
    parts.append(verdict(d.gaps, d.calibration))
    return "\n\n".join(parts)
```

In `alphasettler/cli.py`:

1. In the module docstring's first sentence add `` `opening-diagnose` `` after `` `opening-train` ``, and append: `` `opening-diagnose` prints spec 3d-1's opening diagnosis (D1-D3 and the verdict on record directories, or D4 for a bot). ``
2. In `_parser`, after the `opening-train` subparser, add:

```python
    od = sub.add_parser("opening-diagnose", help="spec 3d-1's opening diagnosis: D1-D3 on records, or D4 for a bot")
    od_source = od.add_mutually_exclusive_group(required=True)
    od_source.add_argument("--games", nargs="+", metavar="DIR", help="record directories: D1, D2, D3 and the verdict")
    od_source.add_argument("--bot", help="play setups with four copies of this bot: D4")
    od.add_argument("--net", default=None, metavar="NAME",
                    help="with --games: the net whose value head D2 reads and whose search D3 re-runs")
    od.add_argument("--nets-dir", default=DEFAULT_NETS_DIR)
    od.add_argument("--audit", type=int, default=200,
                    help="with --games: D3 re-chooses the setups of this many games of the last DIR (0 skips D3)")
    od.add_argument("--audit-budgets", type=_positive_int, nargs="+", default=[1000, 8000], metavar="N",
                    help="with --games: D3's simulation budgets")
    od.add_argument("--seeds", type=_positive_int, default=None, help="with --bot: how many setups to play")
    od.add_argument("--seed-start", type=int, default=2_000_000_000, help="with --bot: the first seed")
    od_rules = od.add_mutually_exclusive_group()
    od_rules.add_argument("--no-trades", action="store_true", help="with --bot: domestic offers off")
    od_rules.add_argument("--trades", action="store_true", help="with --bot: the trading game")
    od.add_argument("--threads", type=_positive_int, default=None)
    od.add_argument("--out", default=None, help="also write the report to this new file")
```

3. After `_opening_train`, add:

```python
def _opening_diagnose(args, config) -> int:
    from alphasettler import diagnose
    from alphasettler._engine import resolve_net

    threads = args.threads or os.cpu_count() or 1
    if args.bot is not None:
        if args.seeds is None:
            raise ValueError("--bot needs --seeds")
        if args.seed_start < 0 or args.seed_start + args.seeds > SEED_LIMIT:
            raise ValueError(f"seeds {args.seed_start}..{args.seed_start + args.seeds} are outside the u64 seed range")
        _check_bot_name(args.bot, args.nets_dir)
    else:
        if args.net is None:
            raise ValueError("--games needs --net NAME (the net whose value head D2 reads and whose search D3 re-runs)")
        if args.audit < 0:
            raise ValueError(f"--audit must be 0 or more, got {args.audit}")
        resolve_net(args.net, args.nets_dir)  # a bad net fails now, before any record is read
    with ExitStack() as stack:
        # Open the output before the work, so a bad path fails first.
        f = stack.enter_context(open_output(args.out)) if args.out else None
        if args.bot is not None:
            setups = diagnose.play_setups(args.bot, args.seed_start, args.seeds, config, args.nets_dir, threads)
            text = diagnose.d4_table(args.bot, setups)
        else:
            text = diagnose.diagnose([Path(d) for d in args.games], args.net, args.nets_dir, args.audit,
                                     args.audit_budgets, threads)
        print(text)
        if f is not None:
            f.write(text + "\n")
    return 0
```

4. In `main`, after the `opening-train` branch, add:

```python
        elif args.command == "opening-diagnose":
            return _opening_diagnose(args, config)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/pytest -q tests/python/test_opening_diagnose.py`
Expected: PASS (24 tests).

- [ ] **Step 5: The full suite**

Run the Global Constraints sequence (no Rust changed: no rebuild). Expected: `.venv/bin/pytest -q -rs` all pass, nothing skipped.

- [ ] **Step 6: Commit**

```bash
git add alphasettler/diagnose.py alphasettler/cli.py tests/python/test_opening_diagnose.py
git diff --cached | grep -nE 'sk-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{36}|password=[^[:space:]]+'
git -c user.email=sean.may101@gmail.com commit -m "diagnose: opening-diagnose prints D1-D3 and the forced/misjudged/misdecided verdict on records, and D4 for a bot"
```

---

### Task 6: The diagnosis, the opening model and the test on the real runs

**Files:**
- Create: `docs/bot/results-3d1.md`
- Run outputs (not committed; `runs/` is ignored): `runs/3d1/` (reports, logs, arena records), `runs/3c/nets/opening.bin` and `opening.bin.json`

**Interfaces:**
- Consumes: `alphasettler opening-diagnose` (Task 5), `alphasettler opening-train` (Task 4), `net:…+o` bots (Task 2), `alphasettler sprt --trades` and `arena --trades` (Plan 6 / `ca896ee`); the nets in `runs/3c/nets` (`gen-034.bin`).
- Produces: `runs/3c/nets/opening.bin`, the reports in `runs/3d1/`, `docs/bot/results-3d1.md`.

Every step but Step 8 is run by the controller in the background (Steps 2–7 are long). If a step fails, record the failure with its numbers in the results; never rerun the SPRT or the confirmation on the same seeds after changing anything.

- [ ] **Step 1: Build at Task 5's commit (controller runs this)**

```bash
git log -1 --format=%s
PATH=$HOME/.cargo/bin:$PATH cargo build --release -p settler-bots
mkdir -p runs/3d1
ls runs/3c/nets/gen-034.bin && test ! -e runs/3c/nets/opening.bin
```

Then rebuild the extension per the Global Constraints. Pass: the commit subject is Task 5's; every command exits 0 (`gen-034.bin` is there and no `opening.bin` yet).

- [ ] **Step 2: The diagnosis on the trading loop (controller runs this)**

```bash
.venv/bin/alphasettler opening-diagnose \
  --games runs/3c/gen-030 runs/3c/gen-031 runs/3c/gen-032 runs/3c/gen-033 runs/3c/gen-034 \
  --net gen-034 --nets-dir runs/3c/nets --audit 200 --out runs/3d1/diagnose-3c.md 2>&1 | tee runs/3d1/diagnose-3c.log
```

D3 is the long part: 200 games × 8 picks at 1,000 and at 8,000 simulations (about an hour of one core at 8,000; the default threads share it). Pass: exit 0; the first line reads `10000 games (...)`; the D3 table has the rows `recorded`, `1000` and `8000`, each with `1600` picks; the last line starts `verdict: forced`, `verdict: misjudged` or `verdict: misdecided`.

- [ ] **Step 3: The comparison diagnosis on 3b's no-trade games (controller runs this)**

```bash
.venv/bin/alphasettler opening-diagnose --games runs/3b-v2/gen-005 --net gen-034 --nets-dir runs/3c/nets \
  --audit 0 --out runs/3d1/diagnose-3b-gen005.md 2>&1 | tee runs/3d1/diagnose-3b-gen005.log
```

Pass: exit 0; the first line reads `10600 games (...)`; no D3 table; a verdict line.

- [ ] **Step 4: Train the opening model (controller runs this)**

```bash
.venv/bin/alphasettler opening-train \
  --games runs/3b-v2/gen-001 runs/3b-v2/gen-002 runs/3b-v2/gen-003 runs/3b-v2/gen-004 runs/3b-v2/gen-005 \
          runs/3c/gen-030 runs/3c/gen-031 runs/3c/gen-032 runs/3c/gen-033 runs/3c/gen-034 \
  --out runs/3c/nets/opening.bin --compare-net gen-034 --nets-dir runs/3c/nets 2>&1 | tee runs/3d1/train.log
```

Pass: exit 0; `samples: ... from 63000 games (...)` (spec: about 63,000 games, about 500,000 samples: 8 per decided game, about 1 in 20 held out); the two `held-out cross-entropy` lines (all games, trading games) each with `model`, `uniform 1.3863` and `net:gen-034`; `wrote runs/3c/nets/opening.bin`. Recorded either way (spec criterion 2 is not a gate).

- [ ] **Step 5: The test, by SPRT (controller runs this)**

```bash
.venv/bin/alphasettler sprt --candidate net:gen-034@100+t+o --baseline net:gen-034@100+t --trades \
  --nets-dir runs/3c/nets --out runs/3d1/sprt-o.jsonl 2>&1 | tee runs/3d1/sprt.log
```

The defaults are the spec's: p0 0.25, p1 0.30, alpha = beta = 0.05, seeds from 2,000,000,000, batches of 12 seeds, at most 400 seeds. Pass: exit 0 and a line `sprt: H1: ...`, `sprt: H0: ...` or `sprt: undecided after 400 seeds ...`; `belief resets` is not printed by `sprt`, so also run:

```bash
.venv/bin/python3 -c "
import json
rs = [json.loads(l) for l in open('runs/3d1/sprt-o.jsonl')]
print(f'{len(rs)} games, belief resets {sum(r[\"belief_resets\"] for r in rs)}')
" | tee -a runs/3d1/sprt.log
```

Pass: `belief resets 0`.

- [ ] **Step 6: The confirmation at 600 simulations, only if Step 5 printed H1 (controller runs this)**

```bash
.venv/bin/alphasettler arena --candidate net:gen-034@600+t+o --baseline net:gen-034@600+t --trades --seeds 200 \
  --seed-start 2000001000 --nets-dir runs/3c/nets --out runs/3d1/confirm-600.jsonl 2>&1 | tee runs/3d1/confirm-600.log
```

Seeds 2,000,001,000 onwards: fresh, past the 400 an SPRT can use. Pass: exit 0, `belief resets: 0`; the summary line (win rate, CI, z vs 0.25) is the reported result whatever it says. If Step 5 printed H0 or undecided, skip this step and say so in the results.

- [ ] **Step 7: D4, the openings each bot plays (controller runs this)**

```bash
.venv/bin/alphasettler opening-diagnose --bot net:gen-034@100+t --seeds 200 --trades --nets-dir runs/3c/nets \
  --out runs/3d1/d4-gen034-100t.md 2>&1 | tee runs/3d1/d4.log
.venv/bin/alphasettler opening-diagnose --bot net:gen-034@1000+t --seeds 200 --trades --nets-dir runs/3c/nets \
  --out runs/3d1/d4-gen034-1000t.md 2>&1 | tee -a runs/3d1/d4.log
.venv/bin/alphasettler opening-diagnose --bot net:gen-034@100+t+o --seeds 200 --trades --nets-dir runs/3c/nets \
  --out runs/3d1/d4-gen034-100to.md 2>&1 | tee -a runs/3d1/d4.log
```

Pass: three D4 tables, each `the openings of 200 setups (800 seats)`.

- [ ] **Step 8: Write `docs/bot/results-3d1.md`**

Every number is copied from the files named here; nothing is estimated. The file:

```markdown
# Sub-project 3d-1 (opening diagnosis and opening model): results

Date: <date of the last run>. Machine: <`sysctl -n machdep.cpu.brand_string`>, macOS (<`uname -r`>).
Commit: <`git log -1 --format=%h`> (Plan 7, Task 5), plus the commit adding this file.

Spec: [`docs/superpowers/specs/2026-10-08-opening-model-design.md`](../superpowers/specs/2026-10-08-opening-model-design.md).
Plan: [`docs/superpowers/plans/2026-10-08-plan-7-opening-model.md`](../superpowers/plans/2026-10-08-plan-7-opening-model.md).

## Diagnosis (done criterion 1)

`opening-diagnose` with gen-034 (`runs/3c/nets/gen-034.bin`) on the trading loop's self-play,
`runs/3c/gen-030` to `gen-034`: <the first line of runs/3d1/diagnose-3c.md>.

<the D1 table of runs/3d1/diagnose-3c.md>

<the D2 table of runs/3d1/diagnose-3c.md>

<the D3 table of runs/3d1/diagnose-3c.md>

**<the verdict line of runs/3d1/diagnose-3c.md>**

Self-play draws its setup moves in proportion to root visits, with root noise (the first
`SAMPLE_TURNS` = 8 turns include the setup), so D1 and D2 describe the self-play policy's
openings, and D3's "changed" share counts that sampling as well as the budget. D3's re-chosen
openings and D4's below are the bot's own, most-visited choices.

For comparison, gen-4's no-trade self-play (`runs/3b-v2/gen-005`), the same net reading it:
<the first line of runs/3d1/diagnose-3b-gen005.md>.

<the D1 and D2 tables of runs/3d1/diagnose-3b-gen005.md>

## The opening model (done criterion 2)

78 → 64 → 64 → 4, trained by `opening-train` on `runs/3b-v2/gen-001` to `gen-005` and
`runs/3c/gen-030` to `gen-034`: <the samples line of runs/3d1/train.log>; <the best-epoch line>.

| Held-out samples | model | uniform | gen-034 value head |
|---|---|---|---|
| all games (<n>) | <model> | 1.3863 | <net> |
| trading games (<n>) | <model> | 1.3863 | <net> |

## The test (done criterion 3)

| Run | Result | Bar | Outcome |
|---|---|---|---|
| `sprt --candidate net:gen-034@100+t+o --baseline net:gen-034@100+t --trades`, seeds from 2,000,000,000 | <the summary line and the sprt line of runs/3d1/sprt.log> | H1 (p1 0.30 against p0 0.25, alpha = beta = 0.05) | <H1 / H0 / undecided> |
| `arena --candidate net:gen-034@600+t+o --baseline net:gen-034@600+t --trades`, 200 seeds from 2,000,001,000 | <the summary line of runs/3d1/confirm-600.log, or "not run: the SPRT did not reach H1"> | reported with its z | <z, or "-"> |

Belief resets: <the belief resets line of runs/3d1/sprt.log and of runs/3d1/confirm-600.log>.

## Openings played (D4: 200 setups from seed 2,000,000,000, the trading game)

| Bot | mean pips | no wood and brick | no wheat | no ore | no wood or no brick | no sheep | all five | weak groups |
|---|---|---|---|---|---|---|---|---|
| `net:gen-034@100+t` | <from runs/3d1/d4-gen034-100t.md> |
| `net:gen-034@1000+t` | <from runs/3d1/d4-gen034-1000t.md> |
| `net:gen-034@100+t+o` | <from runs/3d1/d4-gen034-100to.md> |

## Rulings the build made

- After the setup the engine's phase is `PreRoll`; D2 reads that position (setup done, before the
  first roll), from each seat's own view, and leaves out games with no winner.
- `net_evaluate` reads the hidden block from the game itself; right after the setup every card is
  public, so this is each seat's own view.
- Pips sum per settlement (a tile touched twice pays twice); "distance 2" is two steps on the node
  graph, whoever owns the edges.
- `+o` is for `net:` names only; with `+o`, `+sK` changes nothing. The opening model is used at play
  time only; self-play does not use it.
- D3 drew `random.Random(0).sample` of 200 of `runs/3c/gen-034`'s games; each pick was re-chosen
  from its recorded position.
- The verdict pools the three weak groups; misjudged means their D2 gap exceeds all five's by more
  than 2 standard errors of the difference.
- Training ran on the CPU, seed 0, held out by seed % 20 == 0 across all ten directories.
- The confirmation's seeds start at 2,000,001,000, clear of every SPRT seed.

## Next

<one paragraph: if H1, whether self-play should use +o and whether a draft search or setup roads
come next (spec Section 6); if H0 or undecided, what the verdict and the D4 rows point to>
```

Fill every `<...>` from the named source before committing. The committed file holds numbers and text only, with no angle brackets.

- [ ] **Step 9: Commit**

Run: `grep -c '<' docs/bot/results-3d1.md`
Expected: `0`.

```bash
git add docs/bot/results-3d1.md
git diff --cached | grep -nE 'sk-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{36}|password=[^[:space:]]+'
git -c user.email=sean.may101@gmail.com commit -m "docs: 3d-1 results, the opening diagnosis, the opening model and its test"
```

---

## Self-review

### Spec coverage

| Spec requirement | Task |
|---|---|
| §2.1 diagnosis recorded with the Section 3 tables and one verdict line | 5 (tables, `verdict`), 6 (Steps 2–3, 8) |
| §2.2 opening model trained, held-out cross-entropy beside gen-034's value head; not a gate | 4 (`report`, `format_report`), 6 (Step 4) |
| §2.3 SPRT `net:gen-034@100+t+o` vs three `net:gen-034@100+t`, p0 0.25, p1 0.30, alpha = beta = 0.05, seeds from 2,000,000,000; 600-simulation confirmation on 200 fresh seeds if H1; H0 recorded with D4 for both bots | 6 (Steps 5–7) |
| §3 inputs: `runs/3c/gen-030` .. `gen-034`, comparison `runs/3b-v2/gen-005`; no new self-play | 6 (Steps 2–3) |
| §3 groups in order, pips of every tile touched | 5 (`group`, `opening_pips`; `test_groups_are_the_first_that_applies`) |
| §3 D1 per pick, legal alternatives touching the missing resource, within 0–3 and none | 5 (`gap`, `d1_table`; hand-checked row) |
| §3 D2 via a new binding, per group, gap with standard error, weak vs all five | 3 (`net_evaluate`), 5 (`Diagnosis`, `calibration`, `d2_table`) |
| §3 D3 on 200 games of gen-034 at 1,000 and 8,000 simulations, choice changes and weak rate | 5 (`audit`, `d3_table`), 6 (Step 2) |
| §3 D4 a bot's openings over N seeds; on `@100+t`, `@1000`, `@100+t+o` | 5 (`play_setups`, `d4_table`), 6 (Step 7) |
| §3 verdict rule | 5 (`verdict`; `test_the_verdict_rule`) |
| §4.1 settlements only; roads and later moves unchanged | 2 (`act`; the `+o` choice test checks roads are searched) |
| §4.2 78 seat-relative features in Rust, one function for training and play, `OPENING_HASH` | 1; 2 (`a_spot_scored_before_it_is_placed_has_the_features_training_sees_after`); 3 (binding) |
| §4.3 MLP 78-64-64-4, ReLU, softmax; file layout; loader checks naming the file | 2 (loader, forward, error tests), 4 (`OpeningNet`, `write_model`) |
| §4.4 samples, data directories, held-out rule, Adam lr 1e-3 batch 1,024, patience 3, best epoch, report vs uniform and gen-034; parity test | 4 |
| §4.5 `+o` suffix, any order, once; loads `opening.bin`; highest own-win probability, ties to engine order, no search; play time only | 2 |
| §5 bindings `opening_features`, `OPENING_FEATURES`, `OPENING_HASH`, `net_evaluate`; CLI `opening-diagnose` (both forms) and `opening-train` | 3, 4, 5 |
| §6 out of scope: setup roads, draft search, `+o` in self-play, retraining the net | not built; `selfplay.rs` untouched (Global Constraints) |
| §7 Rust tests (hand-counted features, rotation, hash, file errors, `+o` names, `+o` choice and search elsewhere) | 1, 2 |
| §7 Python tests (parity, `opening-train` on two tiny directories, diagnosis on a tiny directory with a hand-checked D1 row, `net_evaluate` sums to 1 and agrees with the evaluator) | 3, 4, 5 |

### Placeholder scan

Searched for "TBD", "TODO", "implement later", "fill in", "similar to Task", "appropriate", "handle edge cases": none in the code steps. The one template is Task 6 Step 8's results file, whose `<...>` marks are run-time measurements, each naming its source; Step 9 refuses to commit while one is left.

### Type consistency

- Rust: `OPENING_FEATURES`, the block offsets, `opening_features(&Board, &[u64; 4], PlayerId) -> [f32; 78]` and `state_features(&State, PlayerId)` (Task 1) are used with those names by `opening_model.rs`, the tests and `bindings/src/opening.rs`. `OpeningModel::{load, from_bytes, to_bytes, save, probabilities}`, `OpeningLayer`, `OPENING_FILE`, `OPENING_FORMAT_VERSION`, `choose_settlement` (Task 2) are the names Task 3 binds. `NetName` fields (`net`, `simulations`, `rollout`, `setup_factor`, `offers`, `opening`) match between `netbot.rs`, `Roster` and both test files. `split_suffixes -> (String, Flags)`; `split_trades` keeps `-> (String, bool)` for `make_bot` and the 3c tests. `position_value(&Weights, &State, PlayerId)` (Task 3) is what `net_evaluate` calls.
- Python: `opening_features`, `opening_forward`, `net_evaluate`, `OPENING_*` (Task 3) are imported by `alphasettler/opening.py`, `alphasettler/train/opening.py`, `alphasettler/diagnose.py` and the tests. `replay_setup(record, *, features, value, keep_positions) -> Setup`, `Pick.{seat, node, legal, before, features, value}`, `Setup.{seed, trading, winner, board, picks, after, nodes()}`, `model_bytes`, `write_model` (Task 4) are used with those names in Task 5. `fixed_game(seed, winner, config)` and `write_dir(path, games)` are the helper's signatures in both test files. `records.record_files` replaces `train.data.record_files`, which stays an alias.

### Review Focus

Each of the five lines has its test in the owning task (1: Tasks 2 and 3; 2: Task 2 and `net_names.py`; 3: Task 4; 4: Task 5; 5: Tasks 2 and 3). Checked and not added: a `+o` bot asked to act outside the setup (it searches as before: covered by the arena test in Task 2); `net_evaluate` on a finished game (refused with "the game is over", as `encode` is, and no caller passes one); two Python threads acting with the same `Bot` at once (the lock is taken without the GIL, so the second waits; no tool does it).
