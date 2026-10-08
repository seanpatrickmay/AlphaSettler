# Plan 12: Compound Build Moves in Search (Sub-project 3d-4) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the search see "up to two bank trades, then the build they make affordable" as one move (a compound move) in place of bare maritime trades, expose it as the `+c` bot suffix and as `fast-loop --compound` / `train-loop --compound`, then measure it: `net:B@100+t+c` against three `net:B@100+t` (no training), and arm F (the best arm's settings plus `--compound`, 3 hours) settled by an SPRT.

**Architecture:** A new `search/src/compound.rs` computes, from a world, the build targets the position allows (the engine's own `legal_main_builds` / `legal_main_dev` on a copy whose hand covers every cost), the trades each target gets by the spec's rule, the move keys (`COMPOUND_BIT | build id`) and the move priors. `SearchConfig.compound` makes `Sim::descend` use that move set at the searching player's own Main decisions and apply a compound edge step by step; the root's compounds come from `WorldSampler::template`. `SearchResult` gains `plan` and `compound_visits`, and folds each compound's visits onto its first trade's entry of `legal`. `IsmctsBot::with_compound` (`+c`, split off by `ismcts::split_search_flags`) plays a compound's first step and its later ones unsearched while its own trade is the only event since; `sample_move` makes visit sampling carry plans. Match records gain per-seat `bank_trades`, `searches` and `compound_decisions` (criterion 3 reads them). `Exploration.compound` gives compound self-play, threaded through the binding (`selfplay_explore(compound=)`), `alphasettler/selfplay.py`, `FastLoop` (`state.json`) and `Loop` (`manifest.json`). The controller then runs the measurements and writes `docs/bot/results-3d4.md`.

**Tech Stack:** Rust 1.80+ (edition 2021, no new crates), PyO3 0.29.3 + maturin 1.15.0, Python 3.12 stdlib, PyTorch 2.14.1 (`train` extra only), pytest 9.1.1.

**Spec:** `docs/superpowers/specs/2026-10-08-compound-moves-design.md` (binding). Builds on Plan 8 (`docs/superpowers/plans/2026-10-08-plan-8-gumbel-root.md`: `RootMode`, `Sim::run`, `split_gumbel`, how `--gumbel` is kept, `runs/p2/arm.sh`, `runs/p2/tables.py`) and Plan 9 (`docs/superpowers/plans/2026-10-08-plan-9-opening-selfplay.md`: `play_with_opening` / `run_with_opening`, `--opening`, `+o` gates, the never-re-pin list). Written against HEAD 19f1ceb.

## Global Constraints

- Python is `.venv/bin/python3`, tests `.venv/bin/pytest`, the CLI `.venv/bin/alphasettler`. Never `python`, `pip`, `python3` outside the venv, or `pip3` outside the venv.
- Rust is `PATH=$HOME/.cargo/bin:$PATH cargo ...` (cargo is not on PATH; `~/.cargo/bin/cargo` is the same binary). MSRV 1.80 (`rust-version = "1.80"`): use nothing stabilised after 1.80 (no `Option::is_none_or`, no `is_multiple_of`). No new external crates and no new Python dependencies. No numpy anywhere.
- Rebuild the extension after any Rust change used from Python. Another measurement may hold the extension: **never run `maturin develop`**. Build a wheel and rename the new library over the old one (an atomic rename, so running processes keep their mapped copy):
  ```bash
  rm -rf target/wheels-3d4 && PATH=$HOME/.cargo/bin:$PATH .venv/bin/maturin build --release -q -o target/wheels-3d4 \
    && unzip -q -o target/wheels-3d4/alphasettler-*.whl 'alphasettler/_engine*.so' -d target/wheels-3d4/x \
    && cp target/wheels-3d4/x/alphasettler/_engine.cpython-312-darwin.so alphasettler/._engine.new.so \
    && mv -f alphasettler/._engine.new.so alphasettler/_engine.cpython-312-darwin.so
  ```
  "Rebuild the extension" in any step below means exactly this command.
- Every task ends with `PATH=$HOME/.cargo/bin:$PATH cargo fmt --all` then `PATH=$HOME/.cargo/bin:$PATH cargo fmt --check` (no output), `PATH=$HOME/.cargo/bin:$PATH cargo clippy --workspace --release --all-targets -- -D warnings` (exit 0), the full `PATH=$HOME/.cargo/bin:$PATH cargo test --workspace --release` (all pass) and, after the rebuild, `.venv/bin/pytest -q -rs` (all pass, the summary lists no skipped tests).
- Nothing under `engine/` or `net/` is edited (the spec forbids action-space and policy-head changes; the compound targets read the engine's own rules on a copy of the state, so no exception is needed). Under `search/` only what Task 1 names changes; under `bots/src/` only `ismcts.rs`, `lib.rs`, `netbot.rs` (doc lines), `arena.rs` and `selfplay.rs`. `engine/tests/golden_trace.rs` passes unchanged. These digests are never re-pinned, and pass unchanged after every task: 3a's `the_generic_refactor_leaves_3a_ismcts_games_unchanged` (`bots/tests/ismcts.rs`) and `fixed_exploration_is_3a_self_play` (`bots/tests/selfplay.rs`); P2's `bots_without_the_gumbel_suffix_play_the_games_they_played_before_p2` (`bots/tests/ismcts.rs`), `net_bots_without_the_gumbel_suffix_play_the_games_they_played_before_p2` (`bots/tests/netbot.rs`) and `standard_exploration_plays_the_games_it_played_before_p2` (`bots/tests/selfplay.rs`); 3d-1b's `self_play_without_an_opening_model_plays_the_games_it_played_before_3d1b` (`bots/tests/selfplay.rs`). Every `+o` test of Plan 7 (`bots/tests/opening_model.rs`) and every Gumbel test of Plan 8 (`search/tests/gumbel.rs`) passes unchanged. Without `+c` and without `--compound`, every bot and every self-play game plays exactly as at 19f1ceb (spec Section 4: "Without +c nothing changes, so every recorded result and pinned digest stays valid"); the pins above run through every function this plan changes.
- The compound rule (spec Section 2): at a `Phase::Main` decision of the searching player (`obs.viewer`), with `+c`, the move set is every searched legal action except `MaritimeTrade`, plus one compound per build target (each road edge, settlement node and city node the position allows, and `BuyDev`) that is not affordable now and needs at most `MAX_TRADES` = 2 trades. Its trades cover the shortfall one card at a time (resources in index order, Wood to Ore), each giving, among the other resources whose cards beyond the target's cost cover their `maritime_rate`, the one with the most such cards (ties to the lower index), the bank holding the card got. Its prior is its first trade's prior split equally among the compounds sharing that trade; kept moves keep theirs; all renormalised (uniform if nothing is left). Compound key = `COMPOUND_BIT` (0x8000) | the build's action id.
- `+c` (spec Section 4) is a suffix like `+t`, `+sK`, `+rD`, `+g`, `+o`, in any order, at most once, on `ismcts` and `net:` names; `+c` with `+g` is an unknown bot (`--compound` with `--gumbel` is refused, spec Section 4). `fast-loop --compound` / `train-loop --compound` are kept from set-up like `--trades`; self-play and both gate sides use `+c`.
- The trading game is exactly `{"max_offers_per_turn": 2, "max_trade_cards": 1}` (`TRADES` in `alphasettler/arena.py`, `trading()` in Rust tests).
- The measurements (spec Section 4): B is the best arm's best net so far (Task 5 Step 2 decides it from the recorded results of 3d-1b, 3d-2 and 3d-3), with `+o` on both sides exactly when that arm used `--opening`. Criterion 1: `net:B@100+t+c[+o]` against three `net:B@100+t[+o]`, `alphasettler sprt --trades`, p0 0.25, p1 0.30, alpha = beta = 0.05, fresh seeds from 7,000,000,000. Criterion 2: arm F = the best arm's settings and self-play seeds plus `--compound`, set up untimed, then 10,800 s of generations; `net:F@100+t+c[+o]` against three `net:B@100+t+c[+o]`, the same SPRT, fresh seeds from 7,001,000,000. H1: `--compound` becomes the setting new loops are set up with.
- Determinism: same seed and weights give the same games and decisions, independent of thread count. Budgets are simulation counts, never wall time (arm F's 3 hours is the experiment's wall clock, not a search budget). The forward pass is Accelerate-based: batch results equal alone only within 1e-5, a seed still replays bit for bit, so no test compares results across different batch compositions.
- No module outside `alphasettler/train/` imports torch.
- Git: one commit per task with the message given in the task, run as `git -c user.email=sean.may101@gmail.com commit -m "..."`. No Co-Authored-By or AI attribution, never `--no-verify`, never push. Before every commit run `git diff --cached | grep -nE 'sk-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{36}|password=[^[:space:]]+'` and expect no output. Stage only the files the task names: other work may be committed in this repository concurrently.
- Testing: integration tests over mocks (real searches on real positions, real games, real record files read back through the real Rust parser, real loops through the CLI); test behaviour, not implementation; independent tests (each builds its own temp files); descriptive names; pytest fixtures and `parametrize` where cases repeat.
- Long runs (criterion 1's SPRT, arm F's 3-hour loop, criterion 2's SPRT) are run by the controller in the background, never by implementer subagents. Steps marked **controller runs this** give the exact command and the pass condition.
- No README or docs beyond `docs/bot/results-3d4.md` (Task 5).

## Review Focus

1. A two-trade target whose second trade the bank cannot fill (the bank holds one ore and the city needs two), or whose only card is gone (no ore in the bank). Expected: no compound for that target, the others unchanged; no compound edge ever meets an illegal trade in a sampled world. Pinned by Task 1's `a_target_the_bank_cannot_fill_gets_no_compound` and `compound_edges_apply_their_steps_legally_and_their_visits_land_on_their_first_trades`.
2. A Main decision whose legal bank trades enable no build (four of one resource, nothing affordable within two trades). Expected: with `+c` the move set is the end of the turn alone; it is played without a search and records nothing, so self-play never writes a decision with zero visits. Pinned by Task 1's `a_compound_root_with_one_move_is_returned_unsearched`, Task 2's `a_compound_root_whose_trades_enable_nothing_is_played_unsearched` and the visit-sum check of Task 3's `compound_self_play_plays_each_compound_through_and_records_only_its_first_step`.
3. A caller that plays something other than the bot's planned step, or never shows the bot its events (a Python caller using `Bot.act` without `Bot.observe`). Expected: the plan is dropped and the bot searches as usual; no illegal move, no extra belief reset when the events were shown. Pinned by Task 2's `a_compound_bot_drops_its_plan_when_the_game_goes_otherwise`.
4. Visit sampling (self-play's first 8 turns) landing on a trade whose visits came from compounds. Expected: the sampled move is a whole compound (first step played, plan kept), never a bare trade alone; without compounds the draw is exactly `sample_by_visits`'s. Pinned by Task 2's `a_sampled_compound_carries_its_plan` and Task 3's `compound_self_play_plays_each_compound_through_and_records_only_its_first_step` (every bank trade in a game belongs to a finished compound).
5. Resuming a loop in another setting: `--compound` on a loop set up without it, a `state.json` / `manifest.json` from before 3d-4 (no `compound` key), `--compound` with `--gumbel` at set-up. Expected: the first refused (exit 2, the message says how it was set up), the second reads as set up without, the third refused before anything is written; a resume with no flag keeps the setting. Pinned by Task 4's `test_cli_fast_loop_keeps_the_compound_setting_it_was_set_up_with`, `test_compound_moves_with_gumbel_are_refused_before_anything_is_written`, `test_a_train_loop_set_up_without_compound_moves_refuses_them` and the `{"compound": True, "gumbel": 4}` case of `test_bad_settings_are_an_error_before_anything_is_written`.

## File Structure

| File | Responsibility | Task |
|---|---|---|
| `search/src/compound.rs` (new) | `COMPOUND_BIT`, `MAX_TRADES`, `Compound`, `compound_key`, `cost`, `applies`, `targets`, `trades_for`, `compounds`, `move_keys`, `move_priors`, `find`, `move_of` | 1 |
| `search/src/search.rs` | `SearchConfig::compound`, `SearchResult::{plan, compound_visits}`; move keys in `ActionSet`, `Pending`, `Sim::descend`, `expand_compound`, `most_visited`, `search_tree` | 1 |
| `search/src/world.rs`, `search/src/tree.rs`, `search/src/lib.rs` | `WorldSampler::template`; the key doc; the module | 1 |
| `bots/src/ismcts.rs` | `split_compound`, `split_search_flags`, `sample_move`; `IsmctsBot::with_compound`, the plan (`plan_step`, `start_plan`, `is_own_trade`), diagnostics | 2 |
| `bots/src/lib.rs`, `bots/src/netbot.rs`, `alphasettler/cli.py` | `+c` in `make_bot` and `Roster`; name texts | 2 |
| `bots/src/arena.rs`, `bindings/src/lib.rs` (`run_match`) | `PlayedGame`, `play_counted`, `seat_counts`; `GameRecord::{bank_trades, searches, compound_decisions}` and the record dict keys | 2 |
| `bots/src/selfplay.rs`, `bindings/src/lib.rs` (`selfplay_explore`), `alphasettler/selfplay.py` | `Exploration.compound`, its validation; `compound=`; `COMPOUND_WITH_GUMBEL`, `check_compound`, `settings`, `write_explore` | 3 |
| `alphasettler/train/fastloop.py`, `alphasettler/train/loop.py`, `alphasettler/cli.py` | `--compound` for both loops, kept in `state.json` / `manifest.json`; `+c` gates | 4 |
| `docs/bot/results-3d4.md` (new) | the measurements | 5 |
| tests: `search/tests/compound.rs` (new, 1), `bots/tests/ismcts.rs` (2), `bots/tests/netbot.rs` (2), `bots/tests/selfplay.rs` (3), `tests/python/test_netbot.py` (2), `tests/python/test_engine.py` (2), `tests/python/test_training_data.py` (3), `tests/python/test_fast_loop.py` (4), `tests/python/test_train_loop.py` (4) | | 1–4 |

## Where the code forces a choice the spec does not make

Each is a **Ruling** (with its reason) the results file repeats, so it can be overridden.

- **Compound moves at every Main decision of the searching player in its tree, the root included;** opponents' decisions keep the plain move set. The spec's out-of-scope list names only opponents' decisions, and "inside the tree a compound edge applies its actions in order"; the tree then plays the bot's own later turns as the bot will play them.
- **The trade rule, read exactly.** The shortfall is covered resource by resource in index order (Wood, Brick, Sheep, Wheat, Ore), one card per trade. A resource may be given only while its cards beyond the target's cost cover its rate ("never giving a resource the target needs" read as never giving the cards the target's cost needs); among those, the most such cards wins, ties to the lower index. The trades run on a copy of the hand and the bank, so each is legal where it is applied; a shortfall over two cards, no givable resource, or an empty bank pile gives no compound.
- **Targets come from the engine's own rules,** `rules::build::legal_main_builds` and `rules::dev::legal_main_dev` (its `BuyDev` only), on a copy of the state whose player holds `[1, 1, 1, 2, 3]` (every cost covered). The position rules (free edges, the distance rule, pieces left, the deck) are the engine's; nothing under `engine/` changes.
- **A compound is a search edge keyed `COMPOUND_BIT | build id`,** one per target. Child keys stay `u16`; `ActionSet` grows to 1,330 bits (action ids, then compound keys). A node's compounds are recomputed in each world that reaches it; the trades agree because the player's hand and the bank are what the path made them.
- **The root's compounds are read from `WorldSampler::template`** (the viewer's own hand, the public bank and board). With `+c` at a Main root the sampler is built before the one-move check; without `+c` it is built where it was, so plain searches draw nothing new.
- **A `+c` move set of one move is returned unsearched** (`simulations` 0), as a single searchable move always was; `IsmctsBot` plays it without counting a search or recording a decision. So a turn whose bank trades enable no build cannot trade with the bank (the spec's first ruling, measured by the SPRT).
- **Priors:** as the spec; if the kept and compound priors sum to 0 or are not finite, the move set is uniform. Root noise (`Noised`) mixes into the evaluator's prior over the engine's moves before the compound priors are formed, so the Dirichlet draw is the plain search's.
- **A compound ending in `BuyDev` reaches the dev-card chance node** exactly as a plain `BuyDev` does (its trades applied first).
- **`SearchResult` gains `plan`** (the chosen move's steps after the first) **and `compound_visits`** (each root compound's steps and visits); `action` is the first step; `visits` adds each compound's visits to its first trade's entry of `legal`.
- **A plan goes on only if the events since the bot's last step are exactly that step's own `MaritimeTraded`** (its seat, cards of the given resource only, one card of the resource got) **and the next step is legal.** Anything else, including no events at all (a caller that does not feed the bot), drops the plan and the bot searches. Plan steps are not searches (`searches` does not count them; `plan_steps` does) and record nothing.
- **Visit sampling with compounds** (`sample_move`) makes `sample_by_visits`'s draw; a draw landing on a trade's entry picks among the compounds starting with it by their visits, so a sampled compound keeps its plan. Without compounds it is `sample_by_visits` exactly.
- **`+c` is stripped by `ismcts::split_compound`, then `+g` by `split_gumbel` (`split_search_flags`),** before `split_trades` / `parse_net_name`, which keep their signatures. `+c` with `+g` is an unknown bot, and `search_tree` asserts it never sees both. `+c+o` is allowed.
- **Diagnostics:** `compound` (1 for a `+c` bot), `compound_decisions` (searched decisions whose move set held a compound) and `plan_steps`.
- **Match records gain per-seat `bank_trades`, `searches` and `compound_decisions`** (spec Section 4.3 reads them "from the criterion-1 records"); `play_game` keeps its signature through `play_counted`. New records carry the keys for every match; files recorded before keep theirs and still read. The share of decisions with a compound move is `compound_decisions` over `searches` of the `+c` seat.
- **`Exploration` gains `compound: bool`** (it is `Copy`, and its literals use struct update); `validate` refuses it with Gumbel; with an opening model it is allowed (arm F may need both). The exploration stream's draws for a plan step are made as for any move (the bot just does not search), so a seed replays.
- **Records carry `"compound": true` in their `selfplay` block only when set,** so plain records stay byte-identical.
- **Loops keep `compound` like `trades`:** a bool in `state.json` / `manifest.json`, absent read as false; `--compound` on a resume of a loop set up without it is refused (exit 2); a resume without the flag keeps it. The spec's "refuses a resume without it" is read as this (a loop set up without `--compound` refuses it on resume). Gates: `fast-loop` `net:gen-NNN@SIMS+t+c[+o]`, `train-loop` `net:gen-NNN@SIMS+c[+o]`. `train-loop`'s warm start self-plays with compound moves too; its `selfplay.json` plan gains `"compound"` only when set.
- **No `--compound` for `alphasettler selfplay`**: the spec names only the loops. Rollouts (`+rD`) keep playing plain moves.
- **The runs.** B: the last of 3d-3, 3d-2, 3d-1b whose SPRT was H1 names the best arm (E, D, C), else P2's arm A; B is that arm's best at its cut. Nets for both SPRTs sit in `runs/3d4/nets`. Arm F runs through the best arm's own script when that is `runs/p2/arm.sh`, else through a copy of `runs/p2/arm.sh` made in `runs/3d4` (Task 5 Step 4). If F's best and B are the same file, criterion 2 is H0 without a match. H1 makes `--compound` the setting new loops are set up with; no CLI default changes.
- **Fresh seeds:** criterion 1 from 7,000,000,000, criterion 2 from 7,001,000,000 (at most 400 seeds each): no self-play (below 1e9), gate (1e9 + 0..200) or SPRT so far (2e9, 3e9, 4e9, 5e9, 6e9 and 6.001e9 ranges) used them.
- **Plans 10 (3d-2) and 11 (3d-3) may land before this plan is executed.** Every replace-block here quotes 19f1ceb. If an anchor has changed, keep both changes (thread `compound` beside whatever they added, e.g. a `league` setting); never re-pin a digest to absorb a conflict.
- **Task boundaries:** the search (1), the `+c` bots and match records (2), compound self-play (3), the loops (4), the runs (5). The search is its own reviewed gate: a reviewer can accept the move set and reject how a bot plays it.

---

### Task 1: Compound moves in `settler_search`

**Files:**
- Create: `search/src/compound.rs`
- Modify: `search/src/search.rs` (module doc, imports, `SearchConfig`, `SearchResult`, `ActionSet`, `Pending`, `expand_compound` (new), `Sim`, `Sim::descend`, `Sim::run`'s expand line, `most_visited`, `search_tree`, `search`'s doc), `search/src/world.rs` (`template`), `search/src/tree.rs` (module doc), `search/src/lib.rs` (module)
- Test: `search/tests/compound.rs` (new)

**Interfaces:**
- Consumes: `settler_engine::rules::{build::legal_main_builds, dev::legal_main_dev}`, `settler_engine::board::Board::maritime_rate` (via `s.board`), `State::buildings`, `crate::world::WorldSampler`, `crate::tree::{Tree, Child, NodeKind, ROOT}`.
- Produces:
  - `SearchConfig { simulations, c_puct, batch, offers, root, compound: bool }`, `SearchConfig::default().compound == false`.
  - `SearchResult { action, visits, simulations, nodes, improved, plan: Vec<Action>, compound_visits: Vec<(Vec<Action>, u32)> }`: `action` the chosen move's first step, `plan` its later steps (empty for a plain move), `compound_visits` every root compound's steps and visits in move order (empty without one), `visits` aligned with `legal` with each compound's visits on its first trade.
  - `settler_search::compound::{COMPOUND_BIT: u16 = 0x8000, MAX_TRADES: usize = 2}`; `Compound { trades: Vec<Action>, build: Action }` (`Clone, Debug, PartialEq, Eq`) with `key(&self) -> u16` and `actions(&self) -> Vec<Action>`; `compound_key(build: Action) -> u16`; `cost(build: Action) -> Hand`; `applies(s: &State, viewer: PlayerId) -> bool`; `targets(s: &State) -> Vec<Action>`; `trades_for(s: &State, build: Action) -> Option<Vec<Action>>`; `compounds(s: &State, out: &mut Vec<Compound>)`; `move_keys(plain: &[Action], compounds: &[Compound], compound_node: bool, out: &mut Vec<u16>)`; `move_priors(plain: &[Action], prior: &[f32], compounds: &[Compound]) -> Vec<f32>`; `find(compounds: &[Compound], key: u16) -> Option<&Compound>`; `move_of(key: u16, compounds: &[Compound]) -> (Action, Vec<Action>)`.
  - `WorldSampler::template(&self) -> &State`.

- [ ] **Step 1: Run the pins at the base commit**

Run: `PATH=$HOME/.cargo/bin:$PATH cargo test --release -p settler-bots --test ismcts --test netbot --test selfplay -- 3a before_p2 before_3d1b`
Expected: PASS, 6 tests. They must still pass after every later step.

- [ ] **Step 2: Write the failing tests**

Create `search/tests/compound.rs`:

```rust
//! Compound build moves in search (spec docs/superpowers/specs/2026-10-08-compound-moves-design.md
//! Sections 2 and 6): the move set, the trades each target gets, the priors and compound edges.

mod common;

use common::*;
use settler_engine::rng::Rng;
use settler_engine::topology::{topo, NUM_EDGES, NUM_NODES};
use settler_engine::*;
use settler_search::compound::{compound_key, compounds, move_priors, Compound, COMPOUND_BIT};
use settler_search::tree::{NodeKind, ROOT};
use settler_search::{
    search, search_tree, Belief, Eval, Evaluator, Leaf, RootMode, SearchConfig, SearchResult,
    UniformEvaluator,
};
use settler_engine::Resource::{Brick, Ore, Sheep, Wheat, Wood};

fn mt(give: Resource, get: Resource) -> Action {
    Action::MaritimeTrade { give, get }
}

fn with_compound(simulations: u32) -> SearchConfig {
    SearchConfig {
        simulations,
        compound: true,
        ..SearchConfig::default()
    }
}

/// Every port node of `s`'s board.
fn port_nodes(s: &State) -> u64 {
    s.board
        .specific_port_nodes
        .iter()
        .fold(s.board.generic_port_nodes, |m, &x| m | x)
}

/// The lowest node on no port: every rate there is 4.
fn off_port(s: &State) -> u8 {
    (0..NUM_NODES as u8)
        .find(|&n| (port_nodes(s) >> n) & 1 == 0)
        .expect("a node off every port")
}

/// The lowest node of the 2:1 brick port.
fn brick_port(s: &State) -> u8 {
    let m = s.board.specific_port_nodes[Brick.index()];
    assert_ne!(m, 0, "the board has a brick port");
    m.trailing_zeros() as u8
}

/// Player 0 to act in Main on game `seed`'s board with one settlement (at `at(&board)`), no roads
/// and `hand`; player 1 holds `elsewhere` (to empty the bank of something); nobody has built
/// anything else.
fn lone_settlement(seed: u64, at: fn(&State) -> u8, hand: Hand, elsewhere: Hand) -> State {
    let s = State::new(seed, no_trades());
    let node = at(&s);
    edit(&s, no_trades(), |snap| {
        snap.players[0].settlements = 1u64 << node;
        give_cards(snap, 0, hand);
        give_cards(snap, 1, elsewhere);
        snap.setup_step = 8;
        snap.current = 0;
        snap.phase = Phase::Main;
    })
}

fn node_of(s: &State) -> u8 {
    s.players[0].settlements.trailing_zeros() as u8
}

/// The settlement's edges, in id order: the position's road targets (no road is built yet).
fn edges_at(s: &State) -> Vec<u8> {
    let n = node_of(s) as usize;
    (0..NUM_EDGES as u8)
        .filter(|&e| (topo().node_edge_mask[n] >> e) & 1 == 1)
        .collect()
}

fn compounds_of(s: &State) -> Vec<Compound> {
    let mut out = Vec::new();
    compounds(s, &mut out);
    out
}

/// A road compound on every edge of the settlement, each with `trades`.
fn roads(s: &State, trades: &[Action]) -> Vec<Compound> {
    edges_at(s)
        .into_iter()
        .map(|e| Compound {
            trades: trades.to_vec(),
            build: Action::BuildRoad(e),
        })
        .collect()
}

fn dev_card(trades: &[Action]) -> Compound {
    Compound {
        trades: trades.to_vec(),
        build: Action::BuyDev,
    }
}

/// Player 0's search of `s` from `seed` (belief and search streams).
fn run<E: Evaluator>(s: &State, eval: &mut E, cfg: &SearchConfig, seed: u64) -> SearchResult {
    let obs = s.observation(0);
    let b = Belief::from_observation(&obs, 64, seed);
    search(&obs, &s.legal_actions(), &b, eval, cfg, &mut Rng::new(seed)).unwrap()
}

/// Values ¼ each; a bank trade's prior is ten times any other move's.
struct TradeHeavy;

impl Evaluator for TradeHeavy {
    fn evaluate(&mut self, leaves: &[Leaf<'_>]) -> Vec<Eval> {
        leaves
            .iter()
            .map(|l| {
                let w: Vec<f32> = l
                    .legal
                    .iter()
                    .map(|a| {
                        if matches!(a, Action::MaritimeTrade { .. }) {
                            10.0
                        } else {
                            1.0
                        }
                    })
                    .collect();
                let total: f32 = w.iter().sum();
                Eval {
                    value: [0.25; 4],
                    prior: w.iter().map(|x| x / total).collect(),
                }
            })
            .collect()
    }
}

/// Wood 12 and wheat 2 off every port: a road is short a brick (one trade of 4 wood), the dev card
/// a sheep and an ore (two trades, the sheep first), the city three ore (more than two trades: none).
/// No road is built, so no settlement spot is reachable.
#[test]
fn each_target_gets_the_trades_the_rule_picks_and_none_needing_three() {
    for seed in 0..3 {
        let s = lone_settlement(seed, off_port, [12, 0, 0, 2, 0], [0; 5]);
        let mut want = roads(&s, &[mt(Wood, Brick)]);
        want.push(dev_card(&[mt(Wood, Sheep), mt(Wood, Ore)]));
        assert_eq!(compounds_of(&s), want, "seed {seed}");
    }
}

/// Wood 8, wheat 2, ore 1: the city is short two ore, both bought with wood (8 to spare, then 4).
/// With the bank down to one ore the second trade cannot be filled: no city compound. With no ore
/// in the bank, a dev card short one ore gets none either.
#[test]
fn a_target_the_bank_cannot_fill_gets_no_compound() {
    for seed in 0..3 {
        let s = lone_settlement(seed, off_port, [8, 0, 0, 2, 1], [0; 5]);
        let mut want = roads(&s, &[mt(Wood, Brick)]);
        want.push(Compound {
            trades: vec![mt(Wood, Ore), mt(Wood, Ore)],
            build: Action::BuildCity(node_of(&s)),
        });
        want.push(dev_card(&[mt(Wood, Sheep)]));
        assert_eq!(compounds_of(&s), want, "seed {seed}");
        // 19 ore: 1 in player 0's hand, 17 in player 1's, 1 in the bank.
        let short = lone_settlement(seed, off_port, [8, 0, 0, 2, 1], [0, 0, 0, 0, 17]);
        assert_eq!(short.bank[Ore.index()], 1);
        let mut want = roads(&short, &[mt(Wood, Brick)]);
        want.push(dev_card(&[mt(Wood, Sheep)]));
        assert_eq!(compounds_of(&short), want, "seed {seed}: one ore in the bank");
        let empty = lone_settlement(seed, off_port, [4, 4, 1, 1, 0], [0, 0, 0, 0, 19]);
        assert_eq!(empty.bank[Ore.index()], 0);
        assert_eq!(compounds_of(&empty), vec![], "seed {seed}: no ore in the bank");
    }
}

/// Off every port with wood 4, brick 4, sheep 1, wheat 1, the dev card is short one ore: wood and
/// brick both have 4 to spare and the tie goes to wood (the lower index); with brick 5, brick has
/// more to spare. The road is affordable now (no compound); the city is four cards short.
#[test]
fn the_most_cards_to_spare_give_and_ties_go_to_the_lower_resource() {
    for seed in 0..3 {
        let tie = lone_settlement(seed, off_port, [4, 4, 1, 1, 0], [0; 5]);
        assert_eq!(compounds_of(&tie), vec![dev_card(&[mt(Wood, Ore)])], "seed {seed}");
        let more = lone_settlement(seed, off_port, [4, 5, 1, 1, 0], [0; 5]);
        assert_eq!(compounds_of(&more), vec![dev_card(&[mt(Brick, Ore)])], "seed {seed}");
    }
}

/// On the 2:1 brick port, two brick buy the dev card's missing ore; off every port the same hand
/// (rate 4 everywhere) buys nothing. The road needs one of the two brick, so the one left over
/// cannot pay the rate: the road gets no compound.
#[test]
fn a_port_rate_makes_a_trade_and_a_needed_card_is_never_given() {
    for seed in 0..3 {
        let port = lone_settlement(seed, brick_port, [0, 2, 1, 1, 0], [0; 5]);
        assert_eq!(compounds_of(&port), vec![dev_card(&[mt(Brick, Ore)])], "seed {seed}");
        let off = lone_settlement(seed, off_port, [0, 2, 1, 1, 0], [0; 5]);
        assert_eq!(compounds_of(&off), vec![], "seed {seed}");
    }
}

/// EndTurn 0.4, two road compounds sharing Sheep -> Wood (0.3 / 2 each), a dev card on Sheep ->
/// Brick (0.2); Sheep -> Ore starts nothing and is gone. Over the 0.9 left: 4/9, 1/6, 1/6, 2/9.
#[test]
fn a_shared_first_trade_splits_its_prior_and_the_move_set_renormalises() {
    let plain = [
        Action::EndTurn,
        mt(Sheep, Wood),
        mt(Sheep, Brick),
        mt(Sheep, Ore),
    ];
    let cs = [
        Compound {
            trades: vec![mt(Sheep, Wood)],
            build: Action::BuildRoad(1),
        },
        Compound {
            trades: vec![mt(Sheep, Wood)],
            build: Action::BuildRoad(2),
        },
        dev_card(&[mt(Sheep, Brick)]),
    ];
    let p = move_priors(&plain, &[0.4, 0.3, 0.2, 0.1], &cs);
    let want: [f32; 4] = [0.4 / 0.9, 0.15 / 0.9, 0.15 / 0.9, 0.2 / 0.9];
    assert_eq!(p.len(), want.len());
    for (got, want) in p.iter().zip(want) {
        assert!((got - want).abs() < 1e-6, "{p:?}");
    }
    // Nothing left with any prior: the move set is uniform.
    assert_eq!(
        move_priors(&[Action::EndTurn, mt(Sheep, Ore)], &[0.0, 1.0], &[]),
        vec![1.0]
    );
}

/// The +c move set of the wood-12 position: the end of the turn and the compounds, no bare trade
/// (spec Section 2). A uniform prior gives each of the 5 engine moves 1/5: each road compound 1/5
/// split k ways, the dev card's 1/5 (its first trade starts nothing else), the end of the turn
/// 1/5; renormalised over 3/5: 1/3, 1/(3k) each, 1/3. Without +c the root is the engine's moves.
#[test]
fn the_move_set_has_no_bare_trade_and_shared_first_trades_split_their_prior() {
    for seed in 0..3 {
        let s = lone_settlement(seed, off_port, [12, 0, 0, 2, 0], [0; 5]);
        let legal = s.legal_actions();
        assert_eq!(
            legal,
            vec![
                mt(Wood, Brick),
                mt(Wood, Sheep),
                mt(Wood, Wheat),
                mt(Wood, Ore),
                Action::EndTurn
            ]
        );
        let obs = s.observation(0);
        let b = Belief::from_observation(&obs, 64, seed);
        let (r, tree) = search_tree(
            &obs,
            &legal,
            &b,
            &mut UniformEvaluator,
            &with_compound(1),
            &mut Rng::new(seed),
        )
        .unwrap();
        let k = edges_at(&s).len();
        let mut want = vec![(Action::EndTurn.encode(), 1.0 / 3.0)];
        want.extend(
            edges_at(&s)
                .into_iter()
                .map(|e| (compound_key(Action::BuildRoad(e)), 1.0 / (3.0 * k as f32))),
        );
        want.push((compound_key(Action::BuyDev), 1.0 / 3.0));
        let root = &tree.nodes[ROOT as usize];
        let got: Vec<(u16, f32)> = root.children.iter().map(|c| (c.key, c.prior)).collect();
        assert_eq!(got.len(), want.len(), "seed {seed}");
        for (&(gk, gp), &(wk, wp)) in got.iter().zip(&want) {
            assert_eq!(gk, wk, "seed {seed}");
            assert!((gp - wp).abs() < 1e-6, "seed {seed}: key {gk:#x} prior {gp}, want {wp}");
        }
        assert!(root.children.iter().all(|c| c.key & COMPOUND_BIT != 0
            || !matches!(Action::decode(c.key), Some(Action::MaritimeTrade { .. }))));
        // One simulation evaluates the root alone; equal priors go to the lower key, a plain move.
        assert_eq!((r.simulations, r.action, r.plan.clone()), (1, Action::EndTurn, vec![]));
        assert_eq!(r.compound_visits.len(), k + 1);
        let cfg = SearchConfig {
            simulations: 1,
            ..SearchConfig::default()
        };
        let (plain, tree) = search_tree(
            &obs,
            &legal,
            &b,
            &mut UniformEvaluator,
            &cfg,
            &mut Rng::new(seed),
        )
        .unwrap();
        let root = &tree.nodes[ROOT as usize];
        let keys: Vec<u16> = root.children.iter().map(|c| c.key).collect();
        assert_eq!(keys, legal.iter().map(|a| a.encode()).collect::<Vec<_>>());
        assert!(root.children.iter().all(|c| (c.prior - 0.2).abs() < 1e-6));
        assert!(plain.plan.is_empty() && plain.compound_visits.is_empty());
    }
}

/// 200 simulations of the wood-12 position under a prior that favours trades: the move priors are
/// the end of the turn 1/21, each road 10/(21k), the dev card 10/21, so the first descent takes the
/// dev card's compound (two trades, then the dev-card chance node). Every compound step is asserted
/// legal in its sampled world as the search applies it. The root's visits sum to 199 and sit on
/// the end of the turn and on the two first trades.
#[test]
fn compound_edges_apply_their_steps_legally_and_their_visits_land_on_their_first_trades() {
    for seed in 0..3 {
        let s = lone_settlement(seed, off_port, [12, 0, 0, 2, 0], [0; 5]);
        let obs = s.observation(0);
        let legal = s.legal_actions();
        let b = Belief::from_observation(&obs, 64, seed);
        let (r, tree) = search_tree(
            &obs,
            &legal,
            &b,
            &mut TradeHeavy,
            &with_compound(200),
            &mut Rng::new(seed),
        )
        .unwrap();
        assert_eq!(r.simulations, 200);
        assert_eq!(r.visits.iter().sum::<u32>(), 199, "seed {seed}");
        let k = edges_at(&s).len();
        assert_eq!(r.compound_visits.len(), k + 1);
        let on = |first: Action| -> u32 {
            r.compound_visits
                .iter()
                .filter(|(m, _)| m[0] == first)
                .map(|(_, v)| v)
                .sum()
        };
        let at = |a: Action| r.visits[legal.iter().position(|&x| x == a).unwrap()];
        assert_eq!(at(mt(Wood, Brick)), on(mt(Wood, Brick)), "seed {seed}");
        assert_eq!(at(mt(Wood, Sheep)), on(mt(Wood, Sheep)), "seed {seed}");
        assert_eq!((at(mt(Wood, Wheat)), at(mt(Wood, Ore))), (0, 0), "no compound starts with them");
        assert_eq!(at(Action::EndTurn) + on(mt(Wood, Brick)) + on(mt(Wood, Sheep)), 199);
        assert_eq!(
            r.compound_visits.last().unwrap().0,
            vec![mt(Wood, Sheep), mt(Wood, Ore), Action::BuyDev]
        );
        let root = &tree.nodes[ROOT as usize];
        let dev = root
            .children
            .iter()
            .find(|c| c.key == compound_key(Action::BuyDev))
            .unwrap();
        assert!(dev.visits > 0, "seed {seed}");
        assert_eq!(
            tree.nodes[dev.node as usize].kind,
            NodeKind::Chance {
                action: Action::BuyDev
            }
        );
        assert_eq!(r, run(&s, &mut TradeHeavy, &with_compound(200), seed), "a seed replays");
    }
}

/// Wood 4 off every port: four bank trades are legal, but the road would need the wood the trade
/// gives away and nothing else is within two trades, so the +c move set is the end of the turn
/// alone: returned without simulating, as any single move is.
#[test]
fn a_compound_root_with_one_move_is_returned_unsearched() {
    for seed in 0..3 {
        let s = lone_settlement(seed, off_port, [4, 0, 0, 0, 0], [0; 5]);
        assert_eq!(s.legal_actions().len(), 5, "seed {seed}: four trades and the end of the turn");
        let r = run(&s, &mut UniformEvaluator, &with_compound(50), seed);
        assert_eq!((r.action, r.simulations, r.nodes), (Action::EndTurn, 0, 0));
        assert!(r.visits.iter().all(|&v| v == 0) && r.plan.is_empty() && r.compound_visits.is_empty());
        let plain = SearchConfig {
            simulations: 50,
            ..SearchConfig::default()
        };
        assert_eq!(run(&s, &mut UniformEvaluator, &plain, seed).simulations, 50);
    }
}

#[test]
#[should_panic(expected = "Gumbel")]
fn compound_moves_with_a_gumbel_root_are_refused() {
    let s = lone_settlement(1, off_port, [12, 0, 0, 2, 0], [0; 5]);
    let cfg = SearchConfig {
        root: RootMode::Gumbel { draw: false },
        ..with_compound(10)
    };
    run(&s, &mut UniformEvaluator, &cfg, 1);
}
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `PATH=$HOME/.cargo/bin:$PATH cargo test --release -p settler-search --test compound`
Expected: FAIL to compile (`could not find compound in settler_search`, `struct SearchConfig has no field named compound`, no field `plan` / `compound_visits`).

- [ ] **Step 4: Write `search/src/compound.rs`**

```rust
//! Compound build moves (sub-project 3d-4, docs/superpowers/specs/2026-10-08-compound-moves-design.md
//! Section 2). At the searching player's own Main decisions, with `SearchConfig::compound`, the
//! search's move set keeps every legal action but the bare maritime trades, and adds one compound
//! move for each build target that is not affordable now but is after at most `MAX_TRADES`
//! maritime trades: those trades, then the build. The engine's action space is unchanged: a
//! compound is a search edge keyed `COMPOUND_BIT | build id`, applied step by step.

use settler_engine::rules::{build, dev};
use settler_engine::{
    Action, Hand, Phase, PlayerId, Resource, State, CITY_COST, DEV_COST, NUM_RESOURCES, ROAD_COST,
    SETTLEMENT_COST,
};

/// Set in a compound move's key; the rest of the key is its build's action id (all below 665).
pub const COMPOUND_BIT: u16 = 0x8000;
/// Most maritime trades a compound makes before its build.
pub const MAX_TRADES: usize = 2;
/// A hand covering every build cost: the targets are what the position allows, whatever the hand.
const EVERY_COST: Hand = [1, 1, 1, 2, 3];

/// One or two maritime trades, then the build they make affordable.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Compound {
    /// In the order they are made.
    pub trades: Vec<Action>,
    /// A road, settlement, city or development card.
    pub build: Action,
}

impl Compound {
    pub fn key(&self) -> u16 {
        compound_key(self.build)
    }

    /// Every step: the trades, then the build.
    pub fn actions(&self) -> Vec<Action> {
        let mut v = self.trades.clone();
        v.push(self.build);
        v
    }
}

/// The key of the compound that ends in `build`: there is at most one per target.
pub fn compound_key(build: Action) -> u16 {
    COMPOUND_BIT | build.encode()
}

/// What `build` costs; panics for an action that is not a build target.
pub fn cost(build: Action) -> Hand {
    match build {
        Action::BuildRoad(_) => ROAD_COST,
        Action::BuildSettlement(_) => SETTLEMENT_COST,
        Action::BuildCity(_) => CITY_COST,
        Action::BuyDev => DEV_COST,
        _ => panic!("{build:?} is not a build target"),
    }
}

/// Whether `s` is a decision that searches compound moves: a Main decision of `viewer` (the
/// searching player). Opponents' decisions keep their plain moves (spec Section 5).
pub fn applies(s: &State, viewer: PlayerId) -> bool {
    s.phase == Phase::Main && s.current_actor() == viewer
}

/// Every build the position lets the player to move make, ignoring only the hand (spec Section 2):
/// the engine's own rules (`legal_main_builds`, and `legal_main_dev`'s `BuyDev`) on a copy whose
/// hand covers every cost. Roads, settlements and cities in the engine's order, then the dev card.
pub fn targets(s: &State) -> Vec<Action> {
    let mut rich = *s;
    rich.players[s.current as usize].hand = EVERY_COST;
    let mut out = Vec::new();
    build::legal_main_builds(&rich, &mut out);
    let mut dev_moves = Vec::new();
    dev::legal_main_dev(&rich, &mut dev_moves);
    if dev_moves.contains(&Action::BuyDev) {
        out.push(Action::BuyDev);
    }
    out
}

/// The maritime trades that make `build` affordable for the player to move (spec Section 2), or
/// `None` if it is affordable now, needs more than `MAX_TRADES` trades, or the rule finds no trade
/// the bank can fill. The shortfall is covered one card at a time, resources in index order; each
/// trade gives, among the other resources whose cards beyond the target's cost cover their rate,
/// the one with the most such cards (ties to the lower index), at the player's rate for it. The
/// trades run on a copy of the hand and the bank, so each is legal where it is made.
pub fn trades_for(s: &State, build: Action) -> Option<Vec<Action>> {
    let p = s.current as usize;
    let cost = cost(build);
    let mut hand = s.players[p].hand;
    let short: usize = (0..NUM_RESOURCES)
        .map(|r| cost[r].saturating_sub(hand[r]) as usize)
        .sum();
    if short == 0 || short > MAX_TRADES {
        return None;
    }
    let buildings = s.buildings(p);
    let mut bank = s.bank;
    let mut trades = Vec::with_capacity(short);
    for get in Resource::ALL {
        while hand[get.index()] < cost[get.index()] {
            if bank[get.index()] == 0 {
                return None;
            }
            // (resource, cards to spare, rate)
            let mut best: Option<(Resource, u8, u8)> = None;
            for give in Resource::ALL {
                if give == get {
                    continue;
                }
                let spare = hand[give.index()].saturating_sub(cost[give.index()]);
                let rate = s.board.maritime_rate(buildings, give);
                if spare >= rate && best.map_or(true, |(_, b, _)| spare > b) {
                    best = Some((give, spare, rate));
                }
            }
            let (give, _, rate) = best?;
            hand[give.index()] -= rate;
            bank[give.index()] += rate;
            hand[get.index()] += 1;
            bank[get.index()] -= 1;
            trades.push(Action::MaritimeTrade { give, get });
        }
    }
    Some(trades)
}

/// Every compound move of the player to move in `s`, in `targets` order, into `out` (cleared).
pub fn compounds(s: &State, out: &mut Vec<Compound>) {
    out.clear();
    for build in targets(s) {
        if let Some(trades) = trades_for(s, build) {
            out.push(Compound { trades, build });
        }
    }
}

fn is_trade(a: &Action) -> bool {
    matches!(a, Action::MaritimeTrade { .. })
}

/// The keys of a node's moves into `out` (cleared): at a compound node, `plain`'s moves but the
/// bare maritime trades, then the compounds; otherwise every move of `plain`.
pub fn move_keys(plain: &[Action], compounds: &[Compound], compound_node: bool, out: &mut Vec<u16>) {
    out.clear();
    if !compound_node {
        out.extend(plain.iter().map(|a| a.encode()));
        return;
    }
    out.extend(plain.iter().filter(|a| !is_trade(a)).map(|a| a.encode()));
    out.extend(compounds.iter().map(Compound::key));
}

/// The priors of a compound node's moves, aligned with `move_keys(plain, compounds, true, ..)`
/// (spec Section 2): kept moves keep their prior from `prior` (aligned with `plain`); a compound
/// gets its first trade's, split equally among the compounds starting with that trade; then all
/// are renormalised, or made uniform if nothing is left.
pub fn move_priors(plain: &[Action], prior: &[f32], compounds: &[Compound]) -> Vec<f32> {
    let mut p: Vec<f32> = plain
        .iter()
        .zip(prior)
        .filter(|(a, _)| !is_trade(a))
        .map(|(_, &x)| x)
        .collect();
    for c in compounds {
        let first = c.trades[0];
        let sharing = compounds.iter().filter(|d| d.trades[0] == first).count() as f32;
        let x = plain
            .iter()
            .position(|&a| a == first)
            .map_or(0.0, |i| prior[i]);
        p.push(x / sharing);
    }
    let total: f32 = p.iter().sum();
    if total > 0.0 && total.is_finite() {
        for x in p.iter_mut() {
            *x /= total;
        }
    } else {
        let u = 1.0 / p.len() as f32;
        p.iter_mut().for_each(|x| *x = u);
    }
    p
}

/// The compound keyed `key` among `compounds`; `None` for a plain move's key.
pub fn find(compounds: &[Compound], key: u16) -> Option<&Compound> {
    if key & COMPOUND_BIT == 0 {
        return None;
    }
    compounds.iter().find(|c| c.key() == key)
}

/// The move keyed `key`: its first step and the steps after it (none for a plain move).
pub fn move_of(key: u16, compounds: &[Compound]) -> (Action, Vec<Action>) {
    match find(compounds, key) {
        Some(c) => (
            c.trades[0],
            c.trades[1..].iter().copied().chain([c.build]).collect(),
        ),
        None => (
            Action::decode(key).expect("a move key is an action id or a compound key"),
            Vec::new(),
        ),
    }
}
```

- [ ] **Step 5: The module, the template and the key doc**

In `search/src/lib.rs`, after `pub mod belief;` add `pub mod compound;`.

In `search/src/world.rs`, after the closing `}` of `pub fn new(...)` (before `/// A uniform deal of the unseen dev cards ...`) add:

```rust
    /// The template world: everything public plus the viewer's own cards (hidden parts from one
    /// sample). 3d-4 reads the viewer's root compound moves from it: they depend only on the
    /// viewer's hand, the bank and the board.
    pub fn template(&self) -> &State {
        &self.template
    }
```

In `search/src/tree.rs`, replace the module doc's first two lines with:

```rust
//! The search tree: nodes in one arena; a decision node's children are keyed by action id (or, at
//! a compound node, `compound::COMPOUND_BIT` | a build's id), a chance node's by the outcome the
//! searching player sees (dice sum, resource or dev card index).
```

- [ ] **Step 6: Change `search/src/search.rs`**

Append to the module doc (after its P2 lines):

```rust
//! 3d-4: with `SearchConfig::compound`, the searching player's own Main decisions search compound
//! build moves (`crate::compound`) instead of bare maritime trades.
```

Replace the import lines

```rust
use crate::belief::Belief;
use crate::evaluator::{Evaluator, Leaf};
```

with

```rust
use crate::belief::Belief;
use crate::compound::{self, Compound};
use crate::evaluator::{Evaluator, Leaf};
```

and `use settler_engine::{Action, Chance, NoEvents, Observation, PlayerId, State, NUM_PLAYERS};` with

```rust
use settler_engine::{
    Action, Chance, NoEvents, Observation, Phase, PlayerId, State, ACTION_SPACE_SIZE, NUM_PLAYERS,
};
```

In `SearchConfig`, after the `root` field add:

```rust
    /// 3d-4 (`+c`): the searching player's own Main decisions search compound build moves in
    /// place of bare maritime trades. Never with a Gumbel root.
    pub compound: bool,
```

and in `Default` add `compound: false,` after `root: RootMode::Puct,`.

Replace `SearchResult` with:

```rust
#[derive(Clone, Debug, PartialEq)]
pub struct SearchResult {
    /// The chosen move's first step (the move itself unless it is a compound).
    pub action: Action,
    /// Root visit counts aligned with the `legal` slice given to `search` (0 for unsearched
    /// actions), each compound's visits added to its first trade's entry (spec 3d-4 Section 3).
    /// They sum to `simulations - 1`: the first simulation evaluates the root.
    pub visits: Vec<u32>,
    pub simulations: u32,
    pub nodes: usize,
    /// A Gumbel search's improved policy softmax(ℓ + σ(completedQ)), aligned with `legal`,
    /// summing to 1, 0 for a move the search does not consider (an offer without `offers`).
    /// `None` for PUCT and for a decision returned without searching.
    pub improved: Option<Vec<f32>>,
    /// The chosen compound's steps after `action` (empty for a plain move).
    pub plan: Vec<Action>,
    /// Every root compound move (all its steps) with its visits, in move order; empty when the
    /// root searched none.
    pub compound_visits: Vec<(Vec<Action>, u32)>,
}
```

Replace the `ActionSet` block (its doc comment through the end of its `impl`) with:

```rust
/// Move keys as a bitset: the 665 action ids, then the compound keys (`compound::COMPOUND_BIT` |
/// a build's id) above them.
#[derive(Clone, Copy, Default)]
struct ActionSet([u64; 21]);

fn slot(key: u16) -> usize {
    if key & compound::COMPOUND_BIT == 0 {
        key as usize
    } else {
        ACTION_SPACE_SIZE + (key & !compound::COMPOUND_BIT) as usize
    }
}

impl ActionSet {
    fn of(keys: &[u16]) -> ActionSet {
        let mut s = ActionSet::default();
        for &k in keys {
            let i = slot(k);
            s.0[i / 64] |= 1 << (i % 64);
        }
        s
    }

    fn contains(&self, key: u16) -> bool {
        let i = slot(key);
        (self.0[i / 64] >> (i % 64)) & 1 == 1
    }

    fn remove(&mut self, key: u16) {
        let i = slot(key);
        self.0[i / 64] &= !(1u64 << (i % 64));
    }
}
```

In `struct Pending`, after `legal: Vec<Action>,` add:

```rust
    /// A compound node's compounds (its moves are `compound::move_keys` of `legal` and these);
    /// `None` for a plain node, whose moves are `legal`.
    compounds: Option<Vec<Compound>>,
```

After `fn expand(...)` add:

```rust
/// `expand` for a compound node (spec 3d-4 Section 2): its children are the move set, with the
/// priors `compound::move_priors` forms from the evaluator's prior over the engine's moves.
fn expand_compound(
    tree: &mut Tree,
    node: u32,
    legal: &[Action],
    prior: &[f32],
    compounds: &[Compound],
) {
    assert_eq!(
        prior.len(),
        legal.len(),
        "the evaluator's prior must align with the legal actions"
    );
    let mut keys = Vec::new();
    compound::move_keys(legal, compounds, true, &mut keys);
    let priors = compound::move_priors(legal, prior, compounds);
    let n = &mut tree.nodes[node as usize];
    n.expanded = true;
    n.awaiting_eval = false;
    n.children
        .extend(keys.iter().zip(&priors).map(|(&k, &p)| Child {
            available: 1,
            ..Child::new(k, p)
        }));
}
```

In `struct Sim`, after `scratch: Vec<Action>,` add:

```rust
    /// This node's compounds (empty at a plain node) and move keys.
    compounds: Vec<Compound>,
    keys: Vec<u16>,
```

Replace `fn descend` (its doc comment and whole body) with:

```rust
    /// One descent from the root in `world`: to a terminal, to a leaf to evaluate, or `Busy`.
    /// `force` is the first step from the root (a Gumbel phase's candidate); `None`, and every
    /// step below the root, is PUCT. With `cfg.compound`, the searching player's own Main
    /// decisions search compound moves instead of bare maritime trades (spec 3d-4 Section 2); a
    /// compound edge applies its trades, then its build, each asserted legal.
    fn descend(&mut self, mut world: State, mut force: Option<u16>) -> Descent {
        let mut node = ROOT;
        let mut path = Path::new();
        loop {
            if world.is_over() {
                return Descent::Terminal {
                    value: terminal_value(&world),
                    path,
                };
            }
            if let NodeKind::Chance { action } = self.tree.nodes[node as usize].kind {
                let key = apply_visible(&mut world, action, self.rng);
                let i = self.tree.child(node, key, 0.0);
                path.push((node, i));
                node = self.tree.descend_to(node, i, NodeKind::Decision);
                continue;
            }
            legal_actions(&world, &mut self.buf);
            search_moves(&self.buf, self.cfg.offers, &mut self.scratch);
            let compound_node = self.cfg.compound && compound::applies(&world, self.obs.viewer);
            self.compounds.clear();
            if compound_node {
                compound::compounds(&world, &mut self.compounds);
            }
            compound::move_keys(&self.scratch, &self.compounds, compound_node, &mut self.keys);
            let n = &mut self.tree.nodes[node as usize];
            if !n.expanded {
                if n.awaiting_eval {
                    return Descent::Busy;
                }
                n.awaiting_eval = true;
                return Descent::Leaf(Pending {
                    path,
                    node,
                    world,
                    legal: self.scratch.clone(),
                    compounds: compound_node.then(|| self.compounds.clone()),
                });
            }
            // Availability counts every descent through here, including one that then stops at a
            // node another simulation in the batch is evaluating (`Busy`), which is not counted as
            // a simulation: at batch 8 that over-counts the root's simulations by about 0.3%.
            let allowed = ActionSet::of(&self.keys);
            let mut unseen = allowed;
            for c in n.children.iter_mut() {
                if allowed.contains(c.key) {
                    c.available += 1;
                    unseen.remove(c.key);
                }
            }
            // Moves this world allows that earlier worlds did not: uniform prior.
            let uniform = 1.0 / self.keys.len() as f32;
            for &key in self.keys.iter() {
                if unseen.contains(key) {
                    n.children.push(Child {
                        available: 1,
                        ..Child::new(key, uniform)
                    });
                }
            }
            let i = match force.take() {
                // The root decision is the searching player's own, so its moves are the same in
                // every world (spec P2 Section 3).
                Some(key) => {
                    assert!(
                        allowed.contains(key),
                        "the forced root move {:?} is not legal in a sampled world",
                        Action::decode(key)
                    );
                    n.children
                        .iter()
                        .position(|c| c.key == key)
                        .expect("an allowed move is a child")
                }
                None => {
                    let actor = world.current_actor() as usize;
                    puct::select(
                        &n.children,
                        |c| allowed.contains(c.key),
                        actor,
                        self.cfg.c_puct,
                    )
                    .expect("a live decision node has a legal action")
                }
            };
            let key = n.children[i].key;
            path.push((node, i));
            let a = match compound::find(&self.compounds, key) {
                // A compound edge (spec 3d-4 Section 2): its trades in order, then its build, each
                // a legal engine action at that point. The build goes on as a plain move does
                // (a dev card through its chance node).
                Some(c) => {
                    for &t in &c.trades {
                        legal_actions(&world, &mut self.buf);
                        assert!(
                            self.buf.contains(&t),
                            "the compound trade {t:?} (for {:?}) is not legal in a sampled world",
                            c.build
                        );
                        world.apply(t);
                    }
                    legal_actions(&world, &mut self.buf);
                    assert!(
                        self.buf.contains(&c.build),
                        "the compound build {:?} is not legal after its trades",
                        c.build
                    );
                    c.build
                }
                None => Action::decode(key).expect("children are keyed by valid action ids"),
            };
            if visible_chance(&world, a, self.obs.viewer) {
                node = self
                    .tree
                    .descend_to(node, i, NodeKind::Chance { action: a });
            } else {
                world.apply(a);
                node = self.tree.descend_to(node, i, NodeKind::Decision);
            }
        }
    }
```

In `fn run`, replace the line `expand(&mut self.tree, p.node, &p.legal, &e.prior);` with:

```rust
                match &p.compounds {
                    None => expand(&mut self.tree, p.node, &p.legal, &e.prior),
                    Some(cs) => expand_compound(&mut self.tree, p.node, &p.legal, &e.prior, cs),
                }
```

Replace `fn most_visited` (its doc comment and body) with:

```rust
/// The most-visited root move among `root_keys` (ties: the higher prior, then the lower key).
/// Root children come from each world's moves; only one the caller's move set holds is played.
fn most_visited(tree: &Tree, root_keys: &[u16]) -> Result<u16, String> {
    let searchable = ActionSet::of(root_keys);
    let best = tree.nodes[ROOT as usize]
        .children
        .iter()
        .filter(|c| searchable.contains(c.key))
        .max_by(|a, b| {
            a.visits
                .cmp(&b.visits)
                .then(a.prior.total_cmp(&b.prior))
                .then(b.key.cmp(&a.key))
        })
        .ok_or("no searched root move is among the legal actions")?;
    Ok(best.key)
}
```

Replace `pub fn search_tree` (its doc comment and whole body) with:

```rust
/// Search `obs.viewer`'s decision among `legal` (see `search`), also returning the tree.
pub fn search_tree<E: Evaluator>(
    obs: &Observation,
    legal: &[Action],
    belief: &Belief,
    eval: &mut E,
    cfg: &SearchConfig,
    rng: &mut Rng,
) -> Result<(SearchResult, Tree), String> {
    assert!(
        cfg.simulations > 0 && cfg.batch > 0,
        "simulations and batch must be positive"
    );
    assert!(
        !(cfg.compound && matches!(cfg.root, RootMode::Gumbel { .. })),
        "compound moves are not searched with a Gumbel root (spec 3d-4 Section 4)"
    );
    let mut root_actions = Vec::new();
    search_moves(legal, cfg.offers, &mut root_actions);
    if root_actions.is_empty() {
        return Err("nothing to search: no legal action".into());
    }
    // 3d-4: at the searching player's own Main decision the moves are compounds, not bare trades,
    // read from the sampler's template (the player's own hand, the public bank and board).
    let compound_root = cfg.compound && obs.phase == Phase::Main && obs.actor == obs.viewer;
    let mut root_compounds = Vec::new();
    let early = if compound_root {
        let sampler = WorldSampler::new(obs, belief)?;
        compound::compounds(sampler.template(), &mut root_compounds);
        Some(sampler)
    } else {
        None
    };
    let mut root_keys = Vec::new();
    compound::move_keys(&root_actions, &root_compounds, compound_root, &mut root_keys);
    if root_keys.len() == 1 {
        let (action, plan) = compound::move_of(root_keys[0], &root_compounds);
        let r = SearchResult {
            action,
            visits: vec![0; legal.len()],
            simulations: 0,
            nodes: 0,
            improved: None,
            plan,
            compound_visits: root_compounds.iter().map(|c| (c.actions(), 0)).collect(),
        };
        return Ok((r, Tree::new()));
    }
    let sampler = match early {
        Some(s) => s,
        None => WorldSampler::new(obs, belief)?,
    };
    let mut sim = Sim {
        obs,
        belief,
        sampler,
        eval,
        cfg,
        rng,
        tree: Tree::new(),
        buf: Vec::new(),
        scratch: Vec::new(),
        compounds: Vec::new(),
        keys: Vec::new(),
        batch: Vec::with_capacity(cfg.batch),
        done: 0,
        root_value: None,
    };
    let (action, plan, improved) = match cfg.root {
        RootMode::Puct => {
            sim.run(cfg.simulations as usize, |_| None);
            let key = most_visited(&sim.tree, &root_keys)?;
            let (action, plan) = compound::move_of(key, &root_compounds);
            (action, plan, None)
        }
        RootMode::Gumbel { draw } => {
            let (action, policy) = gumbel_root(&mut sim, &root_actions, draw);
            let improved: Vec<f32> = legal
                .iter()
                .map(|a| {
                    root_actions
                        .iter()
                        .position(|r| r == a)
                        .map_or(0.0, |i| policy[i])
                })
                .collect();
            (action, Vec::new(), Some(improved))
        }
    };
    let root = &sim.tree.nodes[ROOT as usize];
    let child_visits = |key: u16| {
        root.children
            .iter()
            .find(|c| c.key == key)
            .map_or(0, |c| c.visits)
    };
    let mut visits: Vec<u32> = legal.iter().map(|a| child_visits(a.encode())).collect();
    // spec 3d-4 Section 3: a compound's visits go to its first trade's entry of `legal`.
    let mut compound_visits = Vec::with_capacity(root_compounds.len());
    for c in &root_compounds {
        let v = child_visits(c.key());
        let i = legal
            .iter()
            .position(|&a| a == c.trades[0])
            .expect("a compound's first trade is legal at the root");
        visits[i] += v;
        compound_visits.push((c.actions(), v));
    }
    let (simulations, nodes) = (sim.done, sim.tree.nodes.len());
    Ok((
        SearchResult {
            action,
            visits,
            simulations,
            nodes,
            improved,
            plan,
            compound_visits,
        },
        sim.tree,
    ))
}
```

Replace `search`'s doc comment with:

```rust
/// The move for `obs.viewer` among `legal` after `cfg.simulations` simulations, each in a world
/// sampled from `belief`: the most-visited (PUCT root) or sequential halving's winner (Gumbel
/// root); with `cfg.compound`, a compound's first step with the rest in `plan`. A single
/// searchable move is returned without searching.
```

- [ ] **Step 7: Run the tests to verify they pass**

Run: `PATH=$HOME/.cargo/bin:$PATH cargo test --release -p settler-search`
Expected: PASS, including the 9 tests of `search/tests/compound.rs` and every test of `search/tests/search.rs`, `gumbel.rs`, `noise.rs`, `belief.rs`, `world.rs` unchanged.

Run: `PATH=$HOME/.cargo/bin:$PATH cargo test --release -p settler-bots --test ismcts --test netbot --test selfplay -- 3a before_p2 before_3d1b`
Expected: PASS, the same 6 tests as Step 1.

- [ ] **Step 8: Format, lint and the full suite**

Run the Global Constraints sequence (fmt, fmt --check, clippy, `cargo test --workspace --release`, rebuild the extension, `.venv/bin/pytest -q -rs`). Expected: all pass, nothing skipped.

- [ ] **Step 9: Commit**

```bash
git add search/src/compound.rs search/src/search.rs search/src/world.rs search/src/tree.rs search/src/lib.rs \
  search/tests/compound.rs
git diff --cached | grep -nE 'sk-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{36}|password=[^[:space:]]+'
git -c user.email=sean.may101@gmail.com commit -m "search: compound build moves (up to two bank trades and the build they enable) replace bare bank trades at the searcher's own Main decisions; plain searches unchanged"
```

---

### Task 2: The `+c` bots, their plans and the match records

**Files:**
- Modify: `bots/src/ismcts.rs` (module doc, imports, `split_gumbel`, `split_compound`, `split_search_flags`, `sample_by_visits`, `sample_move`, `is_own_trade`, `IsmctsBot` fields, `with_evaluator`, `with_compound`, `plan_step`, `start_plan`, `act_with`, `observe`, `diagnostics`), `bots/src/lib.rs` (`make_bot`, `unknown_bot`, `Roster::new`, `Roster::make`), `bots/src/netbot.rs` (module and `parse_net_name` docs), `bots/src/arena.rs` (`GameRecord`, `PlayedGame`, `play_game`, `play_counted`, `seat_counts`, `run_match_by`), `bindings/src/lib.rs` (`run_match`), `alphasettler/cli.py` (`_check_bot_name`)
- Test: `bots/tests/ismcts.rs`, `bots/tests/netbot.rs`, `tests/python/test_netbot.py`, `tests/python/test_engine.py`

**Interfaces:**
- Consumes: Task 1's `SearchConfig.compound`, `SearchResult::{plan, compound_visits}` (`visits` folded onto first trades).
- Produces:
  - `settler_bots::ismcts::{split_compound(name: &str) -> Option<(String, bool)>, split_search_flags(name: &str) -> Option<(String, bool, bool)>}` (the name without `+c` and `+g`, then `gumbel`, then `compound`; `None` for a repeat or for both), `sample_move(legal: &[Action], visits: &[u32], compounds: &[(Vec<Action>, u32)], seed: u64) -> Option<(Action, Vec<Action>)>`; `IsmctsBot::with_compound(self, compound: bool) -> IsmctsBot<E>`; diagnostics `compound`, `compound_decisions`, `plan_steps`. After a searched decision, `take_last_search()` has `action` = the move played and `plan` = its later steps.
  - `settler_bots::arena::{PlayedGame { winner, vp, turns, actions, bank_trades: [u32; NUM_PLAYERS] }, play_counted(seed, config, bots) -> PlayedGame, seat_counts(bots: &[Box<dyn Bot>], name: &str) -> [u64; NUM_PLAYERS]}`; `GameRecord` gains `bank_trades: [u32; NUM_PLAYERS]`, `searches: [u64; NUM_PLAYERS]`, `compound_decisions: [u64; NUM_PLAYERS]`.
  - Python `run_match` records gain `"bank_trades"`, `"searches"`, `"compound_decisions"` (lists of 4 ints, by seat).

- [ ] **Step 1: Write the failing Rust tests**

Append to `bots/tests/ismcts.rs`:

```rust
/// A bot's diagnostics by name.
fn diag(bot: &dyn Bot) -> std::collections::HashMap<&'static str, u64> {
    bot.diagnostics().into_iter().collect()
}

#[test]
fn the_compound_suffix_parses_in_any_order_and_only_once() {
    use settler_bots::ismcts::{split_compound, split_search_flags};
    assert_eq!(
        split_compound("ismcts@100"),
        Some(("ismcts@100".to_owned(), false))
    );
    assert_eq!(
        split_compound("ismcts@100+c"),
        Some(("ismcts@100".to_owned(), true))
    );
    assert_eq!(
        split_search_flags("net:gen-034@100+c+t+o"),
        Some(("net:gen-034@100+t+o".to_owned(), false, true))
    );
    assert_eq!(
        split_search_flags("ismcts@100+g+t"),
        Some(("ismcts@100+t".to_owned(), true, false))
    );
    assert_eq!(split_search_flags("ismcts@100+c+g"), None, "not with a Gumbel root");
    for name in [
        "ismcts@100+c",
        "ismcts@100+c+t+r5+s3",
        "ismcts@100+t+c+r5+s3",
        "ismcts@100+r5+s3+t+c",
        "ismcts+c",
    ] {
        let bot = make_bot(name, 0).unwrap_or_else(|| panic!("{name}"));
        assert_eq!(diag(bot.as_ref())["compound"], 1, "{name}");
    }
    for name in ["ismcts@100", "ismcts@100+t+s3", "ismcts@100+g"] {
        assert_eq!(diag(make_bot(name, 0).unwrap().as_ref())["compound"], 0, "{name}");
    }
    for bad in [
        "ismcts@100+c+c",
        "ismcts@100+c2",
        "ismcts@100+C",
        "ismcts@100+cc",
        "ismcts@100+c+",
        "ismcts+c@100",
        "ismcts@100+c+g",
        "ismcts@100+g+t+c",
        "greedy+c",
        "random+c",
        "+c",
    ] {
        assert!(make_bot(bad, 0).is_none(), "{bad}");
    }
}

/// Player 0 in Main right after setup holding exactly `hand` (its setup cards back in the bank).
fn holding(seed: u64, hand: Hand, config: GameConfig) -> State {
    let mut snap = main_with(seed, [0; 5], config).snapshot();
    for r in 0..NUM_RESOURCES {
        snap.bank[r] += snap.players[0].hand[r];
        snap.players[0].hand[r] = hand[r];
        snap.bank[r] -= hand[r];
    }
    State::from_snapshot(seed, config, &snap).unwrap()
}

const WOOD_FOR_ORE: Action = Action::MaritimeTrade {
    give: Resource::Wood,
    get: Resource::Ore,
};

/// 3 VP wins; player 0 has its 2 setup settlements and wood 8, wheat 2, ore 2. A city wins on the
/// spot and is one ore short: its compound trades wood for ore (wood is the only resource with
/// cards to spare beyond the city's cost), and only the city compounds start with that trade.
fn city_one_trade_away(seed: u64) -> State {
    holding(
        seed,
        [8, 0, 0, 2, 2],
        GameConfig {
            vp_to_win: 3,
            ..no_trades()
        },
    )
}

/// spec 3d-4 Section 3: the bot plays the compound's trade and keeps the city as its plan; shown
/// its own trade, it plays the city at its next decision without a search.
#[test]
fn a_compound_bot_plays_its_plan_unsearched() {
    for seed in 0..3 {
        let mut s = city_one_trade_away(seed);
        let mut bot = IsmctsBot::new(seed, 600, 0).with_compound(true);
        let first = decide(&mut bot, &s);
        assert_eq!(first, WOOD_FOR_ORE, "seed {seed}");
        let r = bot.take_last_search().expect("the trade was searched");
        assert!(
            matches!(r.plan[..], [Action::BuildCity(_)]),
            "seed {seed}: plan {:?}",
            r.plan
        );
        assert!(r
            .compound_visits
            .iter()
            .any(|(m, v)| m[..] == [first, r.plan[0]] && *v > 0));
        let before = diag(&bot);
        let mut events = Vec::new();
        s.apply_with(first, None, &mut events);
        bot.observe(0, &events);
        assert_eq!(decide(&mut bot, &s), r.plan[0], "seed {seed}");
        assert!(
            bot.take_last_search().is_none(),
            "seed {seed}: a plan step is not searched"
        );
        let after = diag(&bot);
        assert_eq!((before["searches"], after["searches"]), (1, 1));
        assert_eq!((after["plan_steps"], after["compound_decisions"]), (1, 1));
        assert_eq!(
            after["belief_resets"], before["belief_resets"],
            "seed {seed}: the belief followed the trade"
        );
    }
}

/// The plan goes on only if the game went as planned: after another trade than the plan's (a
/// caller overriding the bot, shown to it), or the planned trade with no events shown (a caller
/// that never feeds the bot), the bot searches again.
#[test]
fn a_compound_bot_drops_its_plan_when_the_game_goes_otherwise() {
    let wood_for_brick = Action::MaritimeTrade {
        give: Resource::Wood,
        get: Resource::Brick,
    };
    for seed in 0..3 {
        let mut s = city_one_trade_away(seed);
        let mut bot = IsmctsBot::new(seed, 600, 0).with_compound(true);
        assert_eq!(decide(&mut bot, &s), WOOD_FOR_ORE, "seed {seed}");
        assert!(s.legal_actions().contains(&wood_for_brick));
        let resets = diag(&bot)["belief_resets"];
        let mut events = Vec::new();
        s.apply_with(wood_for_brick, None, &mut events);
        bot.observe(0, &events);
        decide(&mut bot, &s);
        assert!(
            bot.take_last_search().is_some(),
            "seed {seed}: searched after another trade"
        );
        assert_eq!(diag(&bot)["plan_steps"], 0);
        assert_eq!(diag(&bot)["belief_resets"], resets, "seed {seed}");

        let mut s = city_one_trade_away(seed);
        let mut bot = IsmctsBot::new(seed, 600, 0).with_compound(true);
        assert_eq!(decide(&mut bot, &s), WOOD_FOR_ORE, "seed {seed}");
        s.apply(WOOD_FOR_ORE);
        decide(&mut bot, &s);
        assert!(
            bot.take_last_search().is_some(),
            "seed {seed}: searched with no events shown"
        );
        assert_eq!(diag(&bot)["plan_steps"], 0);
    }
}

/// Self-play's visit sampling (spec 3d-4 Section 3): a sampled compound comes with its plan and a
/// bare trade is never sampled alone; without compounds the draw is `sample_by_visits`'s.
#[test]
fn a_sampled_compound_carries_its_plan() {
    use settler_bots::ismcts::{sample_by_visits, sample_move};
    let legal = [Action::EndTurn, WOOD_FOR_ORE];
    let visits = [2, 6];
    let compounds = vec![
        (vec![WOOD_FOR_ORE, Action::BuildCity(3)], 4),
        (vec![WOOD_FOR_ORE, Action::BuyDev], 2),
    ];
    let mut seen = std::collections::HashSet::new();
    for seed in 0..400 {
        let (a, plan) = sample_move(&legal, &visits, &compounds, seed).unwrap();
        assert_eq!(Some(a), sample_by_visits(&legal, &visits, seed), "seed {seed}");
        match (a, plan.as_slice()) {
            (Action::EndTurn, []) => {}
            (WOOD_FOR_ORE, [Action::BuildCity(3)]) | (WOOD_FOR_ORE, [Action::BuyDev]) => {}
            other => panic!("seed {seed}: {other:?}"),
        }
        seen.insert(plan);
    }
    assert_eq!(seen.len(), 3, "every move is drawn");
    assert_eq!(sample_move(&legal, &[0, 0], &compounds, 1), None);
    let plain = [Action::EndTurn, Action::BuyDev, Action::Roll];
    for seed in 0..50 {
        assert_eq!(
            sample_move(&plain, &[1, 2, 3], &[], seed).map(|(a, _)| a),
            sample_by_visits(&plain, &[1, 2, 3], seed)
        );
    }
    // Through the bot: the sampled move is the recorded action, and a sampled trade starts a plan.
    for seed in 0..12 {
        let s = city_one_trade_away(seed);
        let mut bot = IsmctsBot::new(seed, 60, 0).with_compound(true);
        let opts = MoveOptions {
            simulations: 60,
            noise: None,
            sample: Some(seed),
        };
        let a = bot.act_with(&s.observation(0), &s.legal_actions(), &opts);
        let r = bot.take_last_search().unwrap();
        assert_eq!(r.action, a);
        assert_eq!(
            matches!(a, Action::MaritimeTrade { .. }),
            !r.plan.is_empty(),
            "seed {seed}: {a:?} then {:?}",
            r.plan
        );
    }
}

/// Wood 4 off every port: the four bank trades enable no build (a road would need the wood the
/// trade gives away), so a +c bot's move set is the end of the turn alone: played without a
/// search, nothing to record. A plain bot searches the five moves.
#[test]
fn a_compound_root_whose_trades_enable_nothing_is_played_unsearched() {
    for seed in 0..3 {
        let mut s = common::blank(seed, Phase::Main);
        let ports = s
            .board
            .specific_port_nodes
            .iter()
            .fold(s.board.generic_port_nodes, |m, &x| m | x);
        let n = (0..54u8).find(|&n| (ports >> n) & 1 == 0).unwrap();
        s.players[0].settlements = 1u64 << n;
        common::give(&mut s, 0, [4, 0, 0, 0, 0]);
        let (obs, legal) = (s.observation(0), s.legal_actions());
        let mut plain = IsmctsBot::new(seed, 30, 0);
        plain.act(&obs, &legal);
        assert!(plain.take_last_search().is_some(), "seed {seed}");
        let mut bot = IsmctsBot::new(seed, 30, 0).with_compound(true);
        assert_eq!(bot.act(&obs, &legal), Action::EndTurn, "seed {seed}");
        assert!(bot.take_last_search().is_none(), "seed {seed}");
        assert_eq!(diag(&bot)["searches"], 0);
    }
}

/// Whole games with two +c seats: every move legal (`play_game` panics otherwise), no belief
/// reset, compounds searched and played by the +c seats only.
#[test]
fn games_of_compound_bots_play_through_with_no_belief_resets() {
    let (mut decisions, mut steps) = (0, 0);
    for seed in 0..3u64 {
        let mut bots: Vec<Box<dyn Bot>> = (0..4u64)
            .map(|p| {
                let name = if p % 2 == 0 { "ismcts@60+c" } else { "ismcts@60" };
                make_bot(name, seed * 4 + p).unwrap()
            })
            .collect();
        let (_, _, turns, _) = play_game(seed, no_trades(), &mut bots);
        assert!(turns > 0);
        assert_eq!(belief_resets(bots.iter().map(|b| &**b)), 0, "seed {seed}");
        for (p, b) in bots.iter().enumerate() {
            let d = diag(b.as_ref());
            if p % 2 == 0 {
                decisions += d["compound_decisions"];
                steps += d["plan_steps"];
            } else {
                assert_eq!(
                    (d["compound"], d["compound_decisions"], d["plan_steps"]),
                    (0, 0, 0)
                );
            }
        }
    }
    assert!(
        decisions > 0 && steps > 0,
        "compound decisions {decisions}, plan steps {steps}"
    );
}

/// Counts the bank trades its bot plays.
struct BankTrades {
    bot: Box<dyn Bot>,
    n: Arc<AtomicU32>,
}

impl Bot for BankTrades {
    fn name(&self) -> &'static str {
        "bank-trades"
    }

    fn observe(&mut self, viewer: PlayerId, events: &[Event]) {
        self.bot.observe(viewer, events);
    }

    fn act(&mut self, obs: &Observation, legal: &[Action]) -> Action {
        let a = self.bot.act(obs, legal);
        if matches!(a, Action::MaritimeTrade { .. }) {
            self.n.fetch_add(1, Ordering::Relaxed);
        }
        a
    }
}

#[test]
fn a_played_game_counts_each_seats_bank_trades() {
    use settler_bots::arena::play_counted;
    let mut total = 0;
    for seed in 0..2u64 {
        let counts: Vec<Arc<AtomicU32>> = (0..4).map(|_| Arc::new(AtomicU32::new(0))).collect();
        let mut bots: Vec<Box<dyn Bot>> = (0..4)
            .map(|p| {
                Box::new(BankTrades {
                    bot: make_bot("greedy", seed * 4 + p as u64).unwrap(),
                    n: counts[p].clone(),
                }) as Box<dyn Bot>
            })
            .collect();
        let g = play_counted(seed, no_trades(), &mut bots);
        let want: Vec<u32> = counts.iter().map(|c| c.load(Ordering::Relaxed)).collect();
        assert_eq!(g.bank_trades.to_vec(), want, "seed {seed}");
        total += want.iter().sum::<u32>();
        let mut fresh: Vec<Box<dyn Bot>> = (0..4)
            .map(|p| make_bot("greedy", seed * 4 + p as u64).unwrap())
            .collect();
        assert_eq!(
            (g.winner, g.vp, g.turns, g.actions),
            play_game(seed, no_trades(), &mut fresh)
        );
    }
    assert!(total > 0, "greedy trades with the bank");
}

#[test]
fn match_records_count_each_seats_searches_and_compound_decisions() {
    let records = run_match("ismcts@30+c", "ismcts@30", 0..2, no_trades(), 2).unwrap();
    let mut candidate = 0;
    for r in &records {
        let c = r.candidate_seat as usize;
        for p in 0..4 {
            assert!(r.searches[p] > 0, "seat {p}: {r:?}");
            assert!(r.compound_decisions[p] <= r.searches[p], "seat {p}: {r:?}");
            if p != c {
                assert_eq!(r.compound_decisions[p], 0, "seat {p}: {r:?}");
            }
        }
        candidate += r.compound_decisions[c];
    }
    assert!(candidate > 0, "the +c seat searched compound moves");
}
```

Append to `bots/tests/netbot.rs`:

```rust
#[test]
fn compound_net_names_build_compound_bots_in_any_order() {
    let dir = nets_dir("compound");
    let names = ["net:tiny@16+c", "net:tiny@16+t+c", "net:tiny@16+c+s2+t"];
    let roster = Roster::new(&names, &dir).unwrap();
    for name in names {
        let bot = roster.make(name, 0).unwrap();
        let d: std::collections::HashMap<_, _> = bot.diagnostics().into_iter().collect();
        assert_eq!((bot.name(), d["compound"], d["gumbel_root"]), ("net", 1, 0), "{name}");
    }
    let plain = Roster::new(&["net:tiny@16+t"], &dir)
        .unwrap()
        .make("net:tiny@16+t", 0)
        .unwrap();
    let d: std::collections::HashMap<_, _> = plain.diagnostics().into_iter().collect();
    assert_eq!(d["compound"], 0);
    for bad in [
        "net:tiny@16+c+g",
        "net:tiny@16+g+c",
        "net:tiny@16+c+c",
        "net:tiny+c@16",
    ] {
        let e = Roster::new(&[bad], &dir)
            .err()
            .unwrap_or_else(|| panic!("{bad} was accepted"));
        assert!(e.contains("unknown bot"), "{bad}: {e}");
    }
}
```

- [ ] **Step 2: Run the Rust tests to verify they fail**

Run: `PATH=$HOME/.cargo/bin:$PATH cargo test --release -p settler-bots --test ismcts --test netbot`
Expected: FAIL to compile (`no method named with_compound`, unresolved `split_compound`, `split_search_flags`, `sample_move`, `play_counted`, no field `searches` on `GameRecord`).

- [ ] **Step 3: Write `bots/src/ismcts.rs`**

Append to the module doc:

```rust
//! 3d-4: with `with_compound` (a `+c` name) the search's move set holds compound build moves; the
//! bot plays a compound's first step and its later ones without a search while the game goes as
//! planned (`plan_step`).
```

Replace the imports

```rust
use settler_engine::rng::{mix, Rng};
use settler_engine::{Action, Event, Observation, Phase, PlayerId};
```

with

```rust
use settler_engine::rng::{mix, Rng};
use settler_engine::types::{hand_total, NUM_RESOURCES};
use settler_engine::{Action, Event, Observation, Phase, PlayerId};
```

Replace `split_gumbel` (its doc comment and body) with:

```rust
/// `name` with every `+FLAG` part removed (`flag` is the part after the `+`), and whether it had
/// one; `None` for a repeat. Every other part is kept in order.
fn split_flag(name: &str, flag: &str) -> Option<(String, bool)> {
    let Some(i) = name.find('+') else {
        return Some((name.to_owned(), false));
    };
    let (mut out, mut found) = (name[..i].to_owned(), false);
    for part in name[i + 1..].split('+') {
        if part == flag {
            if found {
                return None;
            }
            found = true;
        } else {
            out.push('+');
            out.push_str(part);
        }
    }
    Some((out, found))
}

/// A searching bot's name with its `+g` suffix (spec P2 Section 4: Gumbel root search, no
/// draws) removed, and whether it had one. `+g` may sit anywhere among the suffixes, at most once
/// (`None` for a repeat); every other part is kept in order for `split_trades` / `split_suffixes`,
/// which refuse any other `g` part (`+g2`, `+gt`) and anything malformed.
pub fn split_gumbel(name: &str) -> Option<(String, bool)> {
    split_flag(name, "g")
}

/// A searching bot's name with its `+c` suffix (spec 3d-4 Section 4: compound build moves)
/// removed, and whether it had one; anywhere among the suffixes, at most once.
pub fn split_compound(name: &str) -> Option<(String, bool)> {
    split_flag(name, "c")
}

/// `split_compound`, then `split_gumbel`: the name for `split_trades` / `parse_net_name`, whether it
/// had `+g` and whether it had `+c`. `None` for a repeat, and for both: compound moves are not
/// searched with a Gumbel root (spec 3d-4 Section 4).
pub fn split_search_flags(name: &str) -> Option<(String, bool, bool)> {
    let (rest, compound) = split_compound(name)?;
    let (rest, gumbel) = split_gumbel(&rest)?;
    (!(compound && gumbel)).then_some((rest, gumbel, compound))
}
```

Replace `sample_by_visits` (its doc comment and body) with:

```rust
/// An entry of `legal` drawn in proportion to `visits` (aligned with it); `None` if nothing was
/// visited.
pub fn sample_by_visits(legal: &[Action], visits: &[u32], seed: u64) -> Option<Action> {
    sample_move(legal, visits, &[], seed).map(|(a, _)| a)
}

/// `sample_by_visits`'s draw, with compounds (spec 3d-4 Section 3): a compound's visits sit on its
/// first trade's entry, so a draw landing there picks among the compounds starting with that trade
/// by their visits, and returns the compound's first step with its other steps. Without compounds
/// (or on any other entry) the plan is empty.
pub fn sample_move(
    legal: &[Action],
    visits: &[u32],
    compounds: &[(Vec<Action>, u32)],
    seed: u64,
) -> Option<(Action, Vec<Action>)> {
    let total: u64 = visits.iter().map(|&v| v as u64).sum();
    if total == 0 {
        return None;
    }
    let mut x = ((Rng::new(seed).next_u64() as u128 * total as u128) >> 64) as u64;
    for (a, &v) in legal.iter().zip(visits) {
        if x < v as u64 {
            for (steps, cv) in compounds.iter().filter(|(m, _)| m[0] == *a) {
                if x < *cv as u64 {
                    return Some((steps[0], steps[1..].to_vec()));
                }
                x -= *cv as u64;
            }
            return Some((*a, Vec::new()));
        }
        x -= v as u64;
    }
    unreachable!("x is below the visit total")
}

/// Whether `events` are exactly `viewer`'s own maritime trade `a`: one `MaritimeTraded` by its
/// seat giving only cards of `a`'s resource and getting one card of the other.
fn is_own_trade(events: &[Event], viewer: PlayerId, a: Action) -> bool {
    let Action::MaritimeTrade { give, get } = a else {
        return false;
    };
    match events {
        [Event::MaritimeTraded { player, gave, got }] => {
            let mut one = [0u8; NUM_RESOURCES];
            one[get.index()] = 1;
            *player == viewer
                && *got == one
                && gave[give.index()] > 0
                && hand_total(gave) == gave[give.index()] as u32
        }
        _ => false,
    }
}
```

In `pub struct IsmctsBot<E = HeuristicEvaluator>`, after `buf: Vec<Action>,` add:

```rust
    /// 3d-4: the steps of the compound under way still to play, the step last played (whose event
    /// the next decision expects) and the events seen since it.
    plan: Vec<Action>,
    planned: Option<Action>,
    since: Vec<Event>,
    plan_steps: u64,
    /// Searched decisions whose move set held a compound move.
    compound_decisions: u64,
```

In `with_evaluator`, after `buf: Vec::new(),` add:

```rust
            plan: Vec::new(),
            planned: None,
            since: Vec::new(),
            plan_steps: 0,
            compound_decisions: 0,
```

After `with_gumbel` add:

```rust
    /// Search compound build moves in place of bare maritime trades at the bot's own Main
    /// decisions (a `+c` name, spec 3d-4), and play a chosen compound's later steps unsearched
    /// while the game goes as planned. Never with `with_gumbel(true)` (the search refuses it).
    pub fn with_compound(mut self, compound: bool) -> IsmctsBot<E> {
        self.cfg.compound = compound;
        self
    }

    /// The next step of the compound under way, if the game went as planned since the bot's last
    /// step (spec 3d-4 Section 3): the events since then are exactly that step's own trade, and the
    /// next step is legal now. Otherwise the plan is dropped and `None` returned.
    fn plan_step(&mut self, obs: &Observation, legal: &[Action], since: &[Event]) -> Option<Action> {
        let next = *self.plan.first()?;
        let played = self.planned.take();
        let on_plan = played.is_some_and(|a| is_own_trade(since, obs.viewer, a))
            && legal.contains(&next);
        if !on_plan {
            self.plan.clear();
            return None;
        }
        self.plan.remove(0);
        self.planned = (!self.plan.is_empty()).then_some(next);
        Some(next)
    }

    /// Keep `plan`, the steps after `played`, as the compound under way (none for a plain move).
    fn start_plan(&mut self, played: Action, plan: Vec<Action>) {
        self.planned = (!plan.is_empty()).then_some(played);
        self.plan = plan;
    }
```

Replace `act_with` (its doc comment and whole body) with:

```rust
    /// `act` with this decision's search settings. A decision that does not search (a single
    /// searchable move, or a compound's later step) ignores `opts`. After a searched move,
    /// `take_last_search().action` is the move played and `.plan` its later steps.
    pub fn act_with(&mut self, obs: &Observation, legal: &[Action], opts: &MoveOptions) -> Action {
        self.last = None;
        let counted_in_observe = std::mem::take(&mut self.counted_in_observe);
        // 3d-4: the next step of a compound under way, unsearched, while the game went as planned.
        let since = std::mem::take(&mut self.since);
        if let Some(a) = self.plan_step(obs, legal, &since) {
            self.plan_steps += 1;
            return a;
        }
        search_moves(legal, self.cfg.offers, &mut self.buf);
        if self.buf.len() == 1 {
            return self.buf[0];
        }
        // A live belief that disagrees with `obs`, or no history for this viewer at all, is a
        // reset; a belief dropped in `observe` was counted there.
        let rebuild = match &self.tracked {
            Tracked::Live(b) if b.viewer() == obs.viewer => b.check(obs).is_err().then_some(true),
            Tracked::Dropped(v) if *v == obs.viewer => Some(false),
            _ => Some(true),
        };
        let counted = counted_in_observe || rebuild == Some(true);
        if let Some(count) = rebuild {
            self.resets += count as u64;
            let b = Belief::from_observation(obs, DEFAULT_MAX_STATES, self.rng.next_u64());
            self.set_tracked(Tracked::Live(b));
        }
        let Tracked::Live(belief) = &self.tracked else {
            unreachable!("live after the rebuild above")
        };
        let cfg = SearchConfig {
            simulations: opts.simulations,
            ..self.cfg
        };
        let result = match opts.noise {
            Some((noise, seed)) => {
                let mut noised = Noised::new(&mut self.eval, noise, seed);
                search(obs, legal, belief, &mut noised, &cfg, &mut self.rng)
            }
            None => search(obs, legal, belief, &mut self.eval, &cfg, &mut self.rng),
        };
        match result {
            Ok(mut r) => {
                if r.simulations == 0 {
                    // Only a compound move set comes back unsearched here: one move once the bare
                    // trades are gone (spec 3d-4). Like a single searchable move, it is played
                    // without counting a search and records nothing.
                    self.start_plan(r.action, r.plan);
                    return r.action;
                }
                self.searches += 1;
                self.compound_decisions += u64::from(!r.compound_visits.is_empty());
                if let Some((a, plan)) = opts
                    .sample
                    .and_then(|seed| sample_move(legal, &r.visits, &r.compound_visits, seed))
                {
                    r.action = a;
                    r.plan = plan;
                }
                let a = r.action;
                self.start_plan(a, r.plan.clone());
                self.last = Some(r);
                a
            }
            Err(_) => {
                // At most one reset per decision: not again if `observe` or the rebuild above
                // already counted one. The next searched decision rebuilds without counting.
                if !counted {
                    self.resets += 1;
                }
                self.set_tracked(Tracked::Dropped(obs.viewer));
                greedy::choose(obs, legal)
            }
        }
    }
```

In `fn observe`, insert at the top of the body (before `match &self.tracked {`):

```rust
        // 3d-4: what happened since a compound's last step decides whether its plan goes on.
        if !self.plan.is_empty() {
            self.since.extend_from_slice(events);
        }
```

In `fn diagnostics`, after the `gumbel_root` entry add:

```rust
            // 3d-4: 1 for a `+c` bot; its searched decisions whose move set held a compound move;
            // the compound steps it played without a search.
            ("compound", self.cfg.compound as u64),
            ("compound_decisions", self.compound_decisions),
            ("plan_steps", self.plan_steps),
```

- [ ] **Step 4: Write `bots/src/lib.rs` and `bots/src/netbot.rs`**

In `make_bot`, replace

```rust
            let (name, gumbel) = ismcts::split_gumbel(name)?;
            let (name, offers) = ismcts::split_trades(&name)?;
```

with

```rust
            let (name, gumbel, compound) = ismcts::split_search_flags(name)?;
            let (name, offers) = ismcts::split_trades(&name)?;
```

and `.with_gumbel(gumbel),` with `.with_gumbel(gumbel)` followed by a new line `.with_compound(compound),`. Change its doc's `+g to search the root by Gumbel sequential halving, the suffixes in any order.` to `+g to search the root by Gumbel sequential halving and +c to search compound build moves (not with +g), the suffixes in any order.`

In `unknown_bot`, replace `+g makes either search the root by Gumbel sequential halving, +o lets a net: bot` with `+g makes either search the root by Gumbel sequential halving, +c makes either search compound build moves (not with +g), +o lets a net: bot`.

In `Roster::new`, replace

```rust
            // `+g` changes how a bot searches, not which files it loads.
            let base = ismcts::split_gumbel(name).map(|(b, _)| b);
```

with

```rust
            // `+g` and `+c` change how a bot searches, not which files it loads.
            let base = ismcts::split_search_flags(name).map(|(b, _, _)| b);
```

In `Roster::make`, replace `let (base, gumbel) = ismcts::split_gumbel(name)?;` with `let (base, gumbel, compound) = ismcts::split_search_flags(name)?;` and `.with_gumbel(gumbel);` with `.with_gumbel(gumbel)` followed by `.with_compound(compound);`.

In `bots/src/netbot.rs`, change the module doc's `each optionally with `+sK`, `+t`, `+g` and `+o`` to `each optionally with `+sK`, `+t`, `+g`, `+c` and `+o``, and its `(`+g` is split off by `ismcts::split_gumbel` before `parse_net_name` reads the rest)` to `(`+g` and `+c` are split off by `ismcts::split_search_flags` before `parse_net_name` reads the rest)`; in `parse_net_name`'s doc change `A name with `+g` gives `None`: callers split it off first with `ismcts::split_gumbel`, as `Roster` does.` to `A name with `+g` or `+c` gives `None`: callers split them off first with `ismcts::split_search_flags`, as `Roster` does.`

- [ ] **Step 5: Write `bots/src/arena.rs`**

Replace `use settler_engine::{Event, GameConfig, PlayerId, State, NUM_PLAYERS};` with `use settler_engine::{Action, Event, GameConfig, PlayerId, State, NUM_PLAYERS};`.

In `GameRecord`, after `pub belief_resets: u64,` add:

```rust
    /// 3d-4: the bank (maritime) trades each seat made.
    pub bank_trades: [u32; NUM_PLAYERS],
    /// 3d-4: each seat's searched decisions and those whose move set held a compound move (the
    /// `searches` and `compound_decisions` diagnostics; 0 for a bot that reports none).
    pub searches: [u64; NUM_PLAYERS],
    pub compound_decisions: [u64; NUM_PLAYERS],
```

After `pub fn belief_resets(...)` add:

```rust
/// Each seat's diagnostic `name`, 0 for a bot that does not report it.
pub fn seat_counts(bots: &[Box<dyn Bot>], name: &str) -> [u64; NUM_PLAYERS] {
    std::array::from_fn(|p| {
        bots[p]
            .diagnostics()
            .into_iter()
            .find(|&(n, _)| n == name)
            .map_or(0, |(_, v)| v)
    })
}

/// What `play_counted` reports of one game.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlayedGame {
    pub winner: Option<u8>,
    pub vp: [u8; NUM_PLAYERS],
    pub turns: u32,
    pub actions: u32,
    /// Bank (maritime) trades by seat.
    pub bank_trades: [u32; NUM_PLAYERS],
}
```

Replace `play_game` (its doc comment and body) with:

```rust
/// Play one game. Returns (winner, final VP, turns, actions). Panics if a bot picks an illegal action.
pub fn play_game(
    seed: u64,
    config: GameConfig,
    bots: &mut [Box<dyn Bot>],
) -> (Option<u8>, [u8; NUM_PLAYERS], u32, u32) {
    let g = play_counted(seed, config, bots);
    (g.winner, g.vp, g.turns, g.actions)
}

/// `play_game`, also counting each seat's bank trades.
pub fn play_counted(seed: u64, config: GameConfig, bots: &mut [Box<dyn Bot>]) -> PlayedGame {
    assert_eq!(bots.len(), NUM_PLAYERS, "a game needs exactly 4 bots");
    let mut s = State::new(seed, config);
    let mut buf = Vec::with_capacity(512);
    let mut feed = EventFeed::default();
    let mut step = Vec::with_capacity(16);
    let mut actions = 0u32;
    let mut bank_trades = [0u32; NUM_PLAYERS];
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
        if matches!(a, Action::MaritimeTrade { .. }) {
            bank_trades[actor] += 1;
        }
        step.clear();
        s.apply_with(a, None, &mut step);
        feed.push(&step);
        actions += 1;
    }
    PlayedGame {
        winner: s.winner(),
        vp: std::array::from_fn(|p| s.total_vp(p)),
        turns: s.turn,
        actions,
        bank_trades,
    }
}
```

In `run_match_by`, replace

```rust
                            let (winner, vp, turns, actions) = play_game(seed, config, &mut bots);
                            out.push(GameRecord {
                                seed,
                                candidate_seat: cand as u8,
                                winner,
                                vp,
                                turns,
                                actions,
                                belief_resets: belief_resets(bots.iter().map(|b| b.as_ref())),
                            });
```

with

```rust
                            let g = play_counted(seed, config, &mut bots);
                            out.push(GameRecord {
                                seed,
                                candidate_seat: cand as u8,
                                winner: g.winner,
                                vp: g.vp,
                                turns: g.turns,
                                actions: g.actions,
                                belief_resets: belief_resets(bots.iter().map(|b| b.as_ref())),
                                bank_trades: g.bank_trades,
                                searches: seat_counts(&bots, "searches"),
                                compound_decisions: seat_counts(&bots, "compound_decisions"),
                            });
```

- [ ] **Step 6: Run the Rust tests to verify they pass**

Run: `PATH=$HOME/.cargo/bin:$PATH cargo test --release -p settler-bots`
Expected: PASS, including the 8 new tests in `tests/ismcts.rs`, the new one in `tests/netbot.rs`, every `+g` and `+o` test and the 6 pins.

- [ ] **Step 7: Write the failing Python tests**

In `tests/python/test_engine.py`, replace

```python
    assert set(recs[0]) == {"seed", "candidate_seat", "winner", "vp", "turns", "actions", "belief_resets"}
```

with

```python
    assert set(recs[0]) == {"seed", "candidate_seat", "winner", "vp", "turns", "actions", "belief_resets",
                            "bank_trades", "searches", "compound_decisions"}
```

and after `test_run_match_records` add:

```python
def test_match_records_count_bank_trades_searches_and_compound_decisions_per_seat():
    recs = run_match("ismcts@20+c", "ismcts@20", 0, 1, 2)
    for r in recs:
        c = r["candidate_seat"]
        assert len(r["bank_trades"]) == len(r["searches"]) == len(r["compound_decisions"]) == 4
        assert all(n > 0 for n in r["searches"])
        assert [n for p, n in enumerate(r["compound_decisions"]) if p != c] == [0, 0, 0]
    assert sum(r["compound_decisions"][r["candidate_seat"]] for r in recs) > 0
    greedy = run_match("greedy", "random", 0, 2, 2)
    assert all(r["searches"] == [0, 0, 0, 0] and r["compound_decisions"] == [0, 0, 0, 0] for r in greedy)
    assert sum(sum(r["bank_trades"]) for r in greedy) > 0, "greedy trades with the bank"
```

In `tests/python/test_netbot.py`, after `test_malformed_gumbel_names_are_unknown_bots` add:

```python
@pytest.mark.parametrize(
    "name, compound",
    [
        ("net:tiny@10", 0),
        ("net:tiny@10+t+g", 0),
        ("net:tiny@10+c", 1),
        ("net:tiny@10+t+c+s2", 1),
        ("net:tiny@10+c+t", 1),
        ("ismcts@10", 0),
        ("ismcts@10+c", 1),
    ],
)
def test_only_names_with_the_compound_suffix_search_compound_moves(nets, name, compound):
    assert Bot(name, 0, nets_dir=str(nets)).diagnostics()["compound"] == compound


@pytest.mark.parametrize("bad", ["net:tiny@10+c+c", "net:tiny+c@10", "net:tiny@10+C", "ismcts@10+cc",
                                 "ismcts@10+c+g", "net:tiny@10+g+t+c", "greedy+c"])
def test_malformed_compound_names_are_unknown_bots(nets, bad):
    with pytest.raises(ValueError, match="unknown bot"):
        Bot(bad, 0, nets_dir=str(nets))
```

- [ ] **Step 8: Write the binding and the CLI text**

In `bindings/src/lib.rs`'s `run_match`, after `d.set_item("belief_resets", r.belief_resets)?;` add:

```rust
        d.set_item("bank_trades", r.bank_trades.to_vec())?;
        d.set_item("searches", r.searches.to_vec())?;
        d.set_item("compound_decisions", r.compound_decisions.to_vec())?;
```

In `alphasettler/cli.py`'s `_check_bot_name`, replace `either with +sK, +t and +g; net: names also with +o)` with `either with +sK, +t, +g and +c (not +g with +c); net: names also with +o)`.

- [ ] **Step 9: Run the Python tests to verify they pass**

Rebuild the extension, then run: `.venv/bin/pytest -q tests/python/test_engine.py tests/python/test_netbot.py tests/python/test_arena.py`
Expected: PASS, including the new record test and the 14 new name cases.

- [ ] **Step 10: Format, lint, the full suite, rebuild, pytest**

Run the Global Constraints sequence. Expected: all pass, nothing skipped.

- [ ] **Step 11: Commit**

```bash
git add bots/src/ismcts.rs bots/src/lib.rs bots/src/netbot.rs bots/src/arena.rs bindings/src/lib.rs alphasettler/cli.py \
  bots/tests/ismcts.rs bots/tests/netbot.rs tests/python/test_engine.py tests/python/test_netbot.py
git diff --cached | grep -nE 'sk-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{36}|password=[^[:space:]]+'
git -c user.email=sean.may101@gmail.com commit -m "bots: +c names search compound build moves and play a compound's later steps unsearched; match records count bank trades, searches and compound decisions per seat"
```

---

### Task 3: Compound self-play

**Files:**
- Modify: `bots/src/selfplay.rs` (module doc, `Exploration`, `fixed`, `standard`, `gumbel`, `validate`, `play_with_opening`'s `SearchConfig`), `bindings/src/lib.rs` (`selfplay_explore`), `alphasettler/selfplay.py` (docstring, `COMPOUND_WITH_GUMBEL`, `check_compound`, `settings`, `write_explore`)
- Test: `bots/tests/selfplay.rs`, `tests/python/test_training_data.py`

**Interfaces:**
- Consumes: Task 2's `IsmctsBot` plans (a plan step returns with `take_last_search()` `None`) and folded visits.
- Produces:
  - `Exploration.compound: bool` (false in `fixed`, `standard`, `gumbel`); `validate` errs (containing `Gumbel`) for `compound` with `gumbel`.
  - `_engine.selfplay_explore(seed_start, games, threads, full_simulations, fast_simulations, full_probability, config=None, net=None, nets_dir=None, gumbel=None, opening=None, compound=False)`.
  - `alphasettler.selfplay.COMPOUND_WITH_GUMBEL: str` (starts `--compound and --gumbel cannot be used together`), `check_compound(compound: bool, gumbel: int | None = None) -> None`, `settings(..., gumbel=None, opening=None, compound=False) -> dict`, `write_explore(..., gumbel=None, opening=None, compound: bool = False) -> dict`; records' `selfplay` block has `"compound": True` only when set.

- [ ] **Step 1: Write the failing Rust tests**

Append to `bots/tests/selfplay.rs`:

```rust
fn is_build(a: Action) -> bool {
    matches!(
        a,
        Action::BuildRoad(_) | Action::BuildSettlement(_) | Action::BuildCity(_) | Action::BuyDev
    )
}

/// spec 3d-4 Section 3: with compound moves, a searched decision that plays a bank trade starts a
/// compound; its other steps (at most one more trade, then the build) follow at once, by the same
/// player, unrecorded; no bank trade is played outside one. Every decision's visits still sum to
/// its search's: a compound's sit on its first trade.
#[test]
fn compound_self_play_plays_each_compound_through_and_records_only_its_first_step() {
    let ex = Exploration {
        compound: true,
        ..small_explore()
    };
    let (mut compounds, mut on_trades) = (0, 0);
    for seed in 0..6 {
        let g = play_with(seed, trading(), &ex, 8, &heuristic);
        assert_eq!(g.belief_resets, 0, "seed {seed}");
        let mut s = State::new(seed, trading());
        let mut decisions = g.decisions.iter().peekable();
        // The compound under way: who plays it, and its trades so far.
        let mut under_way: Option<(PlayerId, u32)> = None;
        for (i, &a) in g.actions.iter().enumerate() {
            let d = decisions.next_if(|d| d.index as usize == i);
            if let Some(d) = d {
                let want = if d.full_search { 5 } else { 2 };
                assert_eq!(d.visits.iter().sum::<u32>(), want, "seed {seed} move {i}");
                on_trades += d
                    .legal
                    .iter()
                    .zip(&d.visits)
                    .any(|(x, &v)| matches!(x, Action::MaritimeTrade { .. }) && v > 0)
                    as u32;
            }
            let actor = s.current_actor();
            match under_way {
                Some((who, trades)) => {
                    assert!(d.is_none(), "seed {seed} move {i}: a later step was recorded");
                    assert_eq!(actor, who, "seed {seed} move {i}");
                    if is_build(a) {
                        under_way = None;
                    } else {
                        assert!(
                            matches!(a, Action::MaritimeTrade { .. }) && trades < 2,
                            "seed {seed} move {i}: {a:?}"
                        );
                        under_way = Some((who, trades + 1));
                    }
                }
                None if matches!(a, Action::MaritimeTrade { .. }) => {
                    assert!(d.is_some(), "seed {seed} move {i}: a bank trade outside a compound");
                    compounds += 1;
                    under_way = Some((actor, 1));
                }
                None => {}
            }
            s.apply(a);
        }
        assert!(under_way.is_none(), "seed {seed}: a compound left unfinished");
    }
    assert!(
        compounds > 0 && on_trades > 0,
        "compounds played {compounds}, decisions with compound visits {on_trades}"
    );
}

#[test]
fn compound_self_play_is_deterministic_and_thread_independent() {
    let ex = Exploration {
        compound: true,
        ..small_explore()
    };
    let a = run_with(0..4, trading(), &ex, 8, &heuristic, 1).unwrap();
    assert_eq!(a, run_with(0..4, trading(), &ex, 8, &heuristic, 3).unwrap());
    let plain = run_with(0..4, trading(), &small_explore(), 8, &heuristic, 1).unwrap();
    assert_ne!(a, plain, "compound moves change the games");
    // With an opening model too (arm F may need both).
    let model = opening_model(7);
    let both = run_with_opening(0..2, trading(), &ex, 8, &heuristic, 2, Some(&model)).unwrap();
    assert!(both.iter().all(|g| g.belief_resets == 0));
}

#[test]
fn compound_self_play_refuses_gumbel_and_is_off_by_default() {
    let ex = Exploration {
        compound: true,
        ..Exploration::gumbel(8)
    };
    let e = run_with(0..1, no_trades(), &ex, 8, &heuristic, 1).unwrap_err();
    assert!(e.contains("Gumbel"), "{e}");
    for ex in [
        Exploration::fixed(10),
        Exploration::standard(),
        Exploration::gumbel(8),
    ] {
        assert!(!ex.compound);
    }
    // Off, self-play is the game pinned before P2 (standard_exploration_plays_the_games_it_played_before_p2).
    let off = Exploration {
        compound: false,
        ..small_explore()
    };
    assert_eq!(
        digest(&play_with(2, trading(), &off, 8, &heuristic)),
        0x94f447a9c3baf5b2
    );
}
```

- [ ] **Step 2: Run the Rust tests to verify they fail**

Run: `PATH=$HOME/.cargo/bin:$PATH cargo test --release -p settler-bots --test selfplay`
Expected: FAIL to compile (`struct Exploration has no field named compound`).

- [ ] **Step 3: Write the Rust implementation**

In `bots/src/selfplay.rs`, append to the module doc:

```rust
//! 3d-4: with `Exploration::compound`, every seat searches compound build moves (`+c`); a
//! compound's later steps are played unsearched and record no decision, and its visits sit on its
//! first trade's entry of the decision's `visits`.
```

In `Exploration`, after the `gumbel` field add:

```rust
    /// 3d-4: every seat searches compound build moves (`+c`). Not with `gumbel` (`validate`).
    pub compound: bool,
```

In `fixed`, `standard` and `gumbel`, add `compound: false,` after `gumbel: ...,`.

In `validate`, before the final `Ok(())` add:

```rust
        if self.compound && self.gumbel {
            return Err(
                "compound moves are not searched with Gumbel self-play: the Gumbel root does not \
                 take them (spec 3d-4 Section 4)"
                    .into(),
            );
        }
```

In `play_with_opening`'s `let cfg = SearchConfig { ... }`, after the `root: ...,` field add:

```rust
        // 3d-4: compound build moves for every seat, the bot playing a compound through.
        compound: ex.compound,
```

- [ ] **Step 4: Run the Rust tests to verify they pass**

Run: `PATH=$HOME/.cargo/bin:$PATH cargo test --release -p settler-bots --test selfplay`
Expected: PASS, including the 3 new tests, `fixed_exploration_is_3a_self_play`, `standard_exploration_plays_the_games_it_played_before_p2` and `self_play_without_an_opening_model_plays_the_games_it_played_before_3d1b`.

- [ ] **Step 5: Write the failing Python tests**

In `tests/python/test_training_data.py`, after `test_opening_self_play_refuses_gumbel_and_bad_model_files` add:

```python
def test_compound_self_play_records_say_so_and_put_compound_visits_on_first_trades(tmp_path):
    games = explore_games(tmp_path, 3, config=TRADING, compound=True)
    plain = explore_games(tmp_path, 3, config=TRADING, name="plain.jsonl.gz")
    ts = TrainingSet(games)
    on_trades = 0
    for gi, (g, p) in enumerate(zip(games, plain)):
        assert g["selfplay"] == {**p["selfplay"], "compound": True}
        assert g["belief_resets"] == 0 and records.replay_mismatches(g) == []
        for di, d in enumerate(g["decisions"]):
            trades = [(a, v) for a, v in zip(d["legal"], d["visits"]) if 233 <= a <= 257 and v > 0]
            if not trades:
                continue
            total = sum(d["visits"])
            policy = floats(ts.position(gi, di)["policy"])
            for a, v in trades:  # a bank trade's slot is its action id
                assert policy[policy_id(a, d["actor"])] == pytest.approx(v / total, rel=1e-6)
            on_trades += 1
    assert on_trades > 0, "no compound was visited"
    assert any(g["actions"] != p["actions"] for g, p in zip(games, plain)), "compound moves change the games"


def test_compound_self_play_refuses_gumbel(tmp_path):
    with pytest.raises(ValueError, match="--compound and --gumbel"):
        write_explore(tmp_path / "no.jsonl.gz", 0, 1, config=CONFIG, full_simulations=6, fast_simulations=3,
                      full_probability=0.25, gumbel=6, compound=True)
    assert not (tmp_path / "no.jsonl.gz").exists()
    with pytest.raises(ValueError, match="Gumbel"):  # the native check, below the Python one
        selfplay_explore(0, 1, 1, 6, 3, 0.25, CONFIG, None, None, 6, None, True)
```

- [ ] **Step 6: Write the binding**

In `bindings/src/lib.rs`'s `selfplay_explore`, change the doc comment's `refused with `gumbel`. With `net`,` to `refused with `gumbel`. With `compound`, every seat searches compound build moves (spec 3d-4); refused with `gumbel`. With `net`,`, the signature attribute to

```rust
#[pyo3(signature = (seed_start, games, threads, full_simulations, fast_simulations, full_probability, config=None, net=None, nets_dir=None, gumbel=None, opening=None, compound=false))]
```

add the parameter `compound: bool,` after `opening: Option<PathBuf>,`, and replace `ex.validate().map_err(value_error)?;` with

```rust
    let ex = Exploration { compound, ..ex };
    ex.validate().map_err(value_error)?;
```

- [ ] **Step 7: Write the Python implementation**

In `alphasettler/selfplay.py`, replace the docstring's last line `` `opening_top`."""`` with:

```python
`opening_top`. With `compound` (3d-4), every seat searches compound build moves; the block then also
holds `"compound": true`."""
```

After `OPENING_WITH_GUMBEL = (...)` add:

```python
COMPOUND_WITH_GUMBEL = ("--compound and --gumbel cannot be used together: compound moves are not searched with a "
                        "Gumbel root (spec 3d-4 Section 4)")
```

After `check_opening` add:

```python
def check_compound(compound: bool, gumbel: int | None = None) -> None:
    """ValueError if compound moves are asked for with Gumbel self-play."""
    if compound and gumbel is not None:
        raise ValueError(COMPOUND_WITH_GUMBEL)
```

In `settings`, change the signature to

```python
def settings(net: str | None, nets_dir: str | Path, full_simulations: int, fast_simulations: int,
             full_probability: float, gumbel: int | None = None, opening: str | Path | None = None,
             compound: bool = False) -> dict:
```

add `check_compound(compound, gumbel)` as the body's first line (before `opening_sha256 = ...`), and after the `if opening_sha256 is not None: ...` lines add:

```python
    if compound:  # only then: records without compound moves are unchanged
        block["compound"] = True
```

In `write_explore`, change the signature's last line to `counts: dict | None = None, gumbel: int | None = None, opening: str | Path | None = None, compound: bool = False) -> dict:`, the `meta = ...` line to `meta = settings(net, nets_dir, full_simulations, fast_simulations, full_probability, gumbel, opening, compound)`, and the native call to:

```python
            batch = selfplay_explore(lo, min(BATCH_GAMES, end - lo), threads, full_simulations, fast_simulations,
                                     full_probability, config, net, str(nets_dir), gumbel,
                                     None if opening is None else str(opening), compound)
```

- [ ] **Step 8: Run the Python tests to verify they pass**

Rebuild the extension, then run: `.venv/bin/pytest -q tests/python/test_training_data.py`
Expected: PASS, including the 2 new tests; `test_explore_records_carry_their_settings_and_no_observations` and `test_record_files_are_byte_identical_for_the_same_games` pass unchanged.

- [ ] **Step 9: Format, lint, the full suite, rebuild, pytest**

Run the Global Constraints sequence. Expected: all pass, nothing skipped.

- [ ] **Step 10: Commit**

```bash
git add bots/src/selfplay.rs bindings/src/lib.rs alphasettler/selfplay.py bots/tests/selfplay.rs \
  tests/python/test_training_data.py
git diff --cached | grep -nE 'sk-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{36}|password=[^[:space:]]+'
git -c user.email=sean.may101@gmail.com commit -m "selfplay: compound self-play, a compound's later steps unrecorded and its visits on its first trade; refused with Gumbel"
```

---

### Task 4: `fast-loop --compound` and `train-loop --compound`

**Files:**
- Modify: `alphasettler/train/fastloop.py` (docstring, `FastLoop.__init__`, `init`, `_generation`, `_selfplay`, `_gate`), `alphasettler/train/loop.py` (docstring, `LoopSettings`, `_check`, `Loop.__init__`, `_compound` (new), `_selfplay`, `_gate`), `alphasettler/cli.py` (both parsers, `_fast_loop`, `_train_loop`)
- Test: `tests/python/test_fast_loop.py`, `tests/python/test_train_loop.py`

**Interfaces:**
- Consumes: Task 3's `selfplay.check_compound(compound, gumbel=None)`, `COMPOUND_WITH_GUMBEL`, `write_explore(..., compound=)`, the record key `selfplay.compound`; Task 2's `+c` names.
- Produces:
  - `FastLoop.init(root, seed_dirs, start_net, first, trades=False, gumbel=None, opening=None, compound=False)` writes `"compound"` (bool) into `state.json`; `FastLoop(...).compound` (False for a state file without the key); each generation's `selfplay` entry has `"compound"`; gates `net:gen-NNN@SIMS+t+c[+o]` / `net:BEST@SIMS+t+c[+o]` (`+c[+o]` without trades).
  - `LoopSettings.compound: bool = False`; `Loop(...).compound`; `manifest.json` key `"compound"`; generation plans (`selfplay.json`) have `"compound"` only when set; gates `net:gen-NNN@SIMS+c[+o]`.
  - CLI: `fast-loop --compound` (first run only; on a resume of a loop set up without it: `error: <root> was set up without --compound, and a loop keeps the setting it was set up with`, exit 2) and `train-loop --compound`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/python/test_fast_loop.py`:

```python
def test_cli_fast_loop_with_compound_moves_self_plays_and_gates_with_them(tmp_path):
    seed = generation(tmp_path, "seed-gen", seed_start=0)
    save(Net(8, 1), tmp_path / "start.bin")
    root = tmp_path / "loop"
    r = cli("fast-loop", "--dir", str(root), "--generations", "2", "--first", "1", "--trades", "--compound",
            "--seed-dirs", str(seed), "--start-net", str(tmp_path / "start.bin"), "--games", "2",
            "--window-games", "3", "--minutes", "0.01", "--gate-seeds", "1", "--gate-simulations", "6",
            "--device", "cpu", "--threads", "2")
    assert r.returncode == 0, r.stderr
    state = json.loads((root / STATE).read_text())
    assert state["compound"] is True and [g["gen"] for g in state["generations"]] == [1, 2]
    for g in state["generations"]:
        assert g["selfplay"]["compound"] is True and g["selfplay"]["belief_resets"] == 0
        assert g["gate"]["candidate"].endswith("@6+t+c") and g["gate"]["baseline"].endswith("@6+t+c"), g["gate"]
        assert g["gate"]["belief_resets"] == 0
    for k in (1, 2):
        paths = sorted((root / gen_name(k)).glob("selfplay-local-*.jsonl.gz"))
        assert paths
        for path in paths:
            for game in records.read(path):
                assert game["selfplay"]["compound"] is True


def test_cli_fast_loop_keeps_the_compound_setting_it_was_set_up_with(tmp_path):
    root, seed = setup(tmp_path)  # set up without --compound
    assert FastLoop(root, FastLoopSettings(generations=0, **TINY)).compound is False
    r = cli("fast-loop", "--dir", str(root), "--generations", "0", "--compound")
    assert r.returncode == 2 and "without --compound" in r.stderr, r.stderr
    on = tmp_path / "on"
    FastLoop.init(on, [seed], tmp_path / "start.bin", first=1, compound=True)
    r = cli("fast-loop", "--dir", str(on), "--generations", "0")
    assert r.returncode == 0, r.stderr
    assert json.loads((on / STATE).read_text())["compound"] is True, "a resume without the flag keeps it"
    r = cli("fast-loop", "--dir", str(on), "--generations", "0", "--compound")
    assert r.returncode == 0, "the same setting repeats"
    # A state file from before 3d-4 has no "compound" key: it reads as set up without.
    state = json.loads((root / STATE).read_text())
    del state["compound"]
    (root / STATE).write_text(json.dumps(state))
    assert FastLoop(root, FastLoopSettings(generations=0, **TINY)).compound is False


def test_compound_moves_with_gumbel_are_refused_before_anything_is_written(tmp_path):
    seed = generation(tmp_path, "seed-gen", seed_start=0)
    save(Net(8, 1), tmp_path / "start.bin")
    with pytest.raises(ValueError, match="--compound and --gumbel"):
        FastLoop.init(tmp_path / "bad", [seed], tmp_path / "start.bin", first=1, gumbel=100, compound=True)
    assert not (tmp_path / "bad").exists()
    r = cli("fast-loop", "--dir", str(tmp_path / "cli"), "--generations", "0", "--first", "1", "--seed-dirs", str(seed),
            "--start-net", str(tmp_path / "start.bin"), "--compound", "--gumbel", "100")
    assert r.returncode == 2 and "--compound and --gumbel" in r.stderr
    assert not (tmp_path / "cli").exists()
```

In `tests/python/test_train_loop.py`, add `{"compound": True, "gumbel": 4}` to the `parametrize("change", [...])` list of `test_bad_settings_are_an_error_before_anything_is_written` (after `{"opening": "missing-opening.bin"}`), and add at the end:

```python
def test_a_compound_train_loop_self_plays_and_gates_with_compound_moves(tmp_path):
    d = tmp_path / "c"
    s = replace(TINY, compound=True)
    assert Loop(d, s).run() == 6
    m = manifest(d)
    assert m["compound"] is True
    g0, g1 = m["generations"]
    assert g0["selfplay"]["compound"] is True and g1["selfplay"]["compound"] is True
    assert g1["gate"]["candidate"] == "net:gen-001@4+c" and g1["gate"]["baseline"].endswith("@4+c")
    for k in (0, 1):
        paths = sorted((d / f"gen-00{k}").glob("selfplay-local-*.jsonl.gz"))
        assert paths
        for path in paths:
            for g in records.read(path):
                assert g["selfplay"]["compound"] is True and g["belief_resets"] == 0
    assert Loop(d, replace(s, compound=False)).compound is True, "a resume without the flag keeps it"
    assert Loop(d, replace(s, compound=False)).run() == 0


def test_a_train_loop_set_up_without_compound_moves_refuses_them(tmp_path, reference):
    assert manifest(reference)["compound"] is False
    assert all("+c" not in manifest(reference)["generations"][1]["gate"][side] for side in ("candidate", "baseline"))
    with pytest.raises(ValueError, match="without --compound"):
        Loop(reference, replace(TINY, compound=True))
    r = cli("train-loop", "--dir", str(reference), "--generations", "1", "--compound")
    assert r.returncode == 2 and "without --compound" in r.stderr
    # A run set up before 3d-4 has no "compound" key: it reads as set up without.
    old = tmp_path / "before-3d4"
    shutil.copytree(reference, old)
    m = manifest(old)
    del m["compound"]
    (old / "manifest.json").write_text(json.dumps(m))
    with pytest.raises(ValueError, match="without --compound"):
        Loop(old, replace(TINY, compound=True))
    assert Loop(old, TINY).compound is False
    assert Loop(old, TINY).run() == 0, "resuming without the flag is accepted, nothing redone"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest -q tests/python/test_fast_loop.py tests/python/test_train_loop.py -k "compound"`
Expected: FAIL (`unexpected keyword argument 'compound'`, `unrecognized arguments: --compound`, `KeyError: 'compound'`).

- [ ] **Step 3: Write the fast-loop implementation**

In `alphasettler/train/fastloop.py`, append to the module docstring (before its closing `"""`):

```python
With `compound` (set when the loop is set up: `init(..., compound=True)`, `fast-loop --compound`;
spec 3d-4), self-play searches compound build moves and both gate sides are `+c`. A resumed loop
keeps it; it cannot be combined with `gumbel`.
```

In `FastLoop.__init__`, after the `self.opening = ...` line add:

```python
        self.compound = bool(self.state.get("compound", False))  # state files from before 3d-4 have none
```

In `init`, change the signature to

```python
    def init(root: str | Path, seed_dirs: list[str | Path], start_net: str | Path, first: int,
             trades: bool = False, gumbel: int | None = None, opening: str | Path | None = None,
             compound: bool = False) -> None:
```

append to its docstring's last sentence ` `compound` searches compound build moves in self-play and gates.` (before the closing `"""`), add `selfplay_mod.check_compound(compound, gumbel)` after the `if gumbel is not None: selfplay_mod.check_gumbel(gumbel)` lines, and in the `state = {...}` literal add `"compound": compound, ` after `"opening": opening_sha256, `.

In `_generation`, after `mode += " (setup settlements from the opening model)"` (and at the same indentation as its `if`) add:

```python
        if self.compound:
            mode += " (compound build moves)"
```

In `_selfplay`, add `compound=self.compound` to the `selfplay_mod.write_explore(...)` call (after `opening=self._opening_file()`), and replace `return {"net": best, "seed_start": start, "gumbel": self.gumbel, "opening": self.opening, **totals}` with:

```python
        return {"net": best, "seed_start": start, "gumbel": self.gumbel, "opening": self.opening,
                "compound": self.compound, **totals}
```

In `_gate`, replace

```python
        suffix = ("+t" if self.trades else "") + ("" if self.opening is None else "+o")
```

with

```python
        suffix = ("+t" if self.trades else "") + ("+c" if self.compound else "") + ("" if self.opening is None else "+o")
```

- [ ] **Step 4: Write the train-loop implementation**

In `alphasettler/train/loop.py`, replace the module docstring's last line `settlement (generation 0's too), and both gate sides are `+o`. The manifest holds its sha256."""` with:

```python
settlement (generation 0's too), and both gate sides are `+o`. The manifest holds its sha256.

With `compound` (`--compound`, set when the run starts and kept on every resume; spec 3d-4), self-play
(generation 0's too) searches compound build moves and both gate sides are `+c`."""
```

In `LoopSettings`, after `opening: str | None = None ...` add:

```python
    compound: bool = False  # compound build moves in self-play and gates, kept from the run's first start
```

In `_check`, after the `if s.opening is not None:` block add:

```python
    selfplay_mod.check_compound(s.compound, s.gumbel)
```

In `Loop.__init__`, after `self.opening = self._opening_model(settings.opening)` add `self.compound = self._compound(settings.compound)`, and after `_place_opening` add:

```python
    def _compound(self, asked: bool) -> bool:
        """Whether self-play and gates search compound build moves (`--compound`). Set when the run
        starts and kept on every resume, as the fast loop keeps `--trades`; a manifest from before
        3d-4 has none (without). Asking for it of a run set up without is a ValueError."""
        if not (self.root / MANIFEST).exists():
            self.manifest["compound"] = asked
            return asked
        stored = bool(self.manifest.get("compound", False))
        if asked and not stored:
            raise ValueError(f"{self.root} was set up without --compound, and a loop keeps the setting it was "
                             f"set up with")
        self.manifest["compound"] = stored
        return stored
```

In `_selfplay`, after the `if self.opening is not None: ...` plan lines add:

```python
        if self.compound:  # only then: plans written before 3d-4 still match
            plan["compound"] = True
```

and add `compound=self.compound` to its `selfplay_mod.write_explore(...)` call (after `opening=...`).

In `_gate`, replace

```python
        suffix = "" if self.opening is None else "+o"
```

with

```python
        suffix = ("+c" if self.compound else "") + ("" if self.opening is None else "+o")
```

- [ ] **Step 5: Write the CLI**

In `alphasettler/cli.py`, after the fast-loop parser's `--opening` argument add:

```python
    fl.add_argument("--compound", action="store_true",
                    help="first run only: self-play and both gate sides search compound build moves (+c)")
```

after the train-loop parser's `--opening` argument add:

```python
    lp.add_argument("--compound", action="store_true",
                    help="first run only: self-play and both gate sides search compound build moves (+c)")
```

In `_fast_loop`, change the `FastLoop.init(...)` call to

```python
        FastLoop.init(root, args.seed_dirs, args.start_net, args.first, trades=args.trades,
                      gumbel=args.gumbel, opening=args.opening, compound=args.compound)
```

and after the `if args.trades and not loop.trades:` block add:

```python
    if args.compound and not loop.compound:
        print(f"error: {root} was set up without --compound, and a loop keeps the setting it was set up with",
              file=sys.stderr)
        return 2
```

In `_train_loop`, change `gumbel=args.gumbel, opening=args.opening,` in the `LoopSettings(...)` call to `gumbel=args.gumbel, opening=args.opening, compound=args.compound,`.

- [ ] **Step 6: Run the tests to verify they pass**

Run: `.venv/bin/pytest -q tests/python/test_fast_loop.py tests/python/test_train_loop.py`
Expected: PASS, including the 3 new fast-loop tests, the 2 new train-loop tests and the new `{"compound": True, "gumbel": 4}` case; every existing loop test passes unchanged.

- [ ] **Step 7: Format, lint, the full suite, pytest**

Run the Global Constraints sequence. Expected: all pass, nothing skipped.

- [ ] **Step 8: Commit**

```bash
git add alphasettler/train/fastloop.py alphasettler/train/loop.py alphasettler/cli.py \
  tests/python/test_fast_loop.py tests/python/test_train_loop.py
git diff --cached | grep -nE 'sk-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{36}|password=[^[:space:]]+'
git -c user.email=sean.may101@gmail.com commit -m "loops: fast-loop and train-loop --compound, kept from set-up; self-play and both gate sides +c; refused with --gumbel"
```

---

### Task 5: The measurements and `docs/bot/results-3d4.md`

**Files:**
- Create: `docs/bot/results-3d4.md`
- Run outputs (not committed; `runs/` is ignored): `runs/3d4/arm.env`, `runs/3d4/best-arm.log`, `runs/3d4/nets/{B,F}.bin` (and `opening.bin` with `+o`), `runs/3d4/sprt-c1.{jsonl,log}`, `runs/3d4/sprt-c2.{jsonl,log}`, `runs/3d4/finals.log`, `runs/3d4/tables.md`; arm F's root, log, `.out` and state at the cut beside the best arm's (`runs/p2/F*`), or under `runs/3d4/` with the copied script (Step 4)

**Interfaces:**
- Consumes: `+c` names and the record keys `bank_trades`, `searches`, `compound_decisions` (Task 2); `fast-loop --compound` (Task 4); `alphasettler sprt --trades`; `runs/p2/arm.sh` and `runs/p2/tables.py` (P2, unchanged); the results files `docs/bot/results-3d1b.md`, `results-3d2.md`, `results-3d3.md` and the runs they name.
- Produces: the run outputs above and `docs/bot/results-3d4.md`.

Every step but Step 8 is run by the controller in the background, on an otherwise idle machine. If a step fails, record the failure with its numbers in the results; never rerun a match on the same seeds after changing anything. Each step runs in a fresh shell: it starts by loading `runs/3d4/arm.env`.

- [ ] **Step 1: Build at Task 4's commit (controller runs this)**

```bash
git log -1 --format=%s
PATH=$HOME/.cargo/bin:$PATH cargo build --release -p settler-bots
ls runs/p2/arm.sh runs/p2/tables.py docs/bot/results-3d1b.md > /dev/null
test ! -e runs/3d4 && test ! -e runs/p2/F
pgrep -fl "alphasettler|settler" || echo idle
```

Then rebuild the extension per the Global Constraints. Pass: the commit subject is Task 4's (or a later commit that leaves Tasks 1–4's files as they committed them); every command exits 0; the last line is `idle`.

- [ ] **Step 2: Decide B (controller runs this)**

The best arm so far is the last of these whose SPRT (the row of its results file's table) is H1: E (`docs/bot/results-3d3.md`), D (`docs/bot/results-3d2.md`), C (`docs/bot/results-3d1b.md`); if none is H1 (or a file does not exist because its sub-project was not run), it is P2's arm A. Each later arm was set up with the then-best arm's settings, so the last H1 beat every arm before it.

```bash
grep -n "sprt --candidate" docs/bot/results-3d1b.md docs/bot/results-3d2.md docs/bot/results-3d3.md
mkdir -p runs/3d4/nets
```

Write `runs/3d4/arm.env` with exactly two lines, `ARM=<the letter>` and `RUNS=<the directory holding that arm's root, its .log and its -state-at-cut.json: runs/p2 for A and C; for D and E, the directory its results file names>`; then:

```bash
set -a; . runs/3d4/arm.env; set +a
test -d "$RUNS/$ARM" && test -f "$RUNS/$ARM.log" && test -f "$RUNS/$ARM-state-at-cut.json"
best=$(.venv/bin/python3 -c "import json; print(json.load(open('$RUNS/$ARM-state-at-cut.json'))['best'])")
O=$(.venv/bin/python3 -c "import json; print('' if json.load(open('$RUNS/$ARM-state-at-cut.json')).get('opening') is None else '+o')")
cp "$RUNS/$ARM/nets/$best.bin" runs/3d4/nets/B.bin
if [ -n "$O" ]; then cp "$RUNS/$ARM/nets/opening.bin" runs/3d4/nets/opening.bin; fi
ARGS=$(head -1 "$RUNS/$ARM.log" | sed -E "s/^[0-9]+ setup $ARM ?//")
printf 'O=%s\nARGS="%s"\n' "$O" "$ARGS" >> runs/3d4/arm.env
.venv/bin/python3 -c "
import json
st = json.load(open('$RUNS/$ARM-state-at-cut.json'))
assert not st.get('compound') and st.get('gumbel') is None, 'the best arm used neither --compound nor --gumbel'
print('best arm $ARM ($RUNS/$ARM): B is', st['best'], '| opening', st.get('opening'), '| set-up arguments: $ARGS')
" | tee runs/3d4/best-arm.log
shasum -a 256 runs/3d4/nets/*.bin | tee -a runs/3d4/best-arm.log
```

Pass: every command exits 0; `best-arm.log` names the arm, B and the set-up arguments (the text after `setup <ARM>` on the first line of its log; empty for A); `O` is `+o` exactly when the arm used `--opening`.

- [ ] **Step 3: Criterion 1, play strength without training (controller runs this)**

```bash
set -a; . runs/3d4/arm.env; set +a
pgrep -fl "alphasettler|settler" || echo idle
.venv/bin/alphasettler sprt --candidate "net:B@100+t+c$O" --baseline "net:B@100+t$O" --trades \
  --seed-start 7000000000 --nets-dir runs/3d4/nets --out runs/3d4/sprt-c1.jsonl 2>&1 | tee runs/3d4/sprt-c1.log
.venv/bin/python3 -c "
import json
rs = [json.loads(l) for l in open('runs/3d4/sprt-c1.jsonl')]
n = len(rs)
seat = [r['candidate_seat'] for r in rs]
searched = sum(r['searches'][c] for r, c in zip(rs, seat))
compound = sum(r['compound_decisions'][c] for r, c in zip(rs, seat))
with_c = sum(r['bank_trades'][c] for r, c in zip(rs, seat)) / n
without = sum(sum(r['bank_trades']) - r['bank_trades'][c] for r, c in zip(rs, seat)) / (3 * n)
print(f'{n} games, belief resets {sum(r[\"belief_resets\"] for r in rs)}')
print(f'decisions with a compound move: {compound} of {searched} searched decisions of the +c seat ({compound / searched:.3f})')
print(f'bank trades per game: {with_c:.2f} by the +c seat, {without:.2f} by each seat without +c')
" | tee -a runs/3d4/sprt-c1.log
```

The defaults are the spec's: p0 0.25, p1 0.30, alpha = beta = 0.05, batches of 12 seeds, at most 400 seeds (7,000,000,000 .. 7,000,000,399). Pass: exit 0, a line starting `sprt: ` (H1, H0 or undecided), `belief resets 0`, and the two criterion-3 lines. Recorded either way.

- [ ] **Step 4: Arm F, the 3-hour loop (controller runs this)**

Arm F is the best arm's set-up (its script's hard-coded settings and its own extra arguments, the same self-play and gate seeds) plus `--compound`. Start it detached (about 3 hours plus the set-up), with the machine otherwise idle.

If `RUNS` is `runs/p2` (the best arm is A or C, which ran through `runs/p2/arm.sh`), F runs through that script unchanged:

```bash
set -a; . runs/3d4/arm.env; set +a
pgrep -fl "alphasettler|settler" || echo idle
echo "F_RUNS=runs/p2" >> runs/3d4/arm.env
nohup bash runs/p2/arm.sh F $ARGS --compound >> runs/p2/F.out 2>&1 &
```

Otherwise (D or E: its start net, upgraded for encoder v4, or its options differ from the ones `runs/p2/arm.sh` hard-codes), the controller makes a copy of `runs/p2/arm.sh` in `runs/3d4` whose outputs land in `runs/3d4` and whose start net is a byte copy of the best arm's own (`<arm root>/nets/start.bin`, what `fast-loop` copied at its set-up); the arm's other options come in through `ARGS`, after the hard-coded ones, so they win as they did for the arm:

```bash
set -a; . runs/3d4/arm.env; set +a
pgrep -fl "alphasettler|settler" || echo idle
sed -e "s#runs/p2/#runs/3d4/#g" -e "s#--start-net runs/3c/nets/gen-034.bin#--start-net $RUNS/$ARM/nets/start.bin#" \
  runs/p2/arm.sh > runs/3d4/arm.sh
diff runs/p2/arm.sh runs/3d4/arm.sh   # only the comment, root, log, state-at-cut and --start-net lines differ
echo "F_RUNS=runs/3d4" >> runs/3d4/arm.env
nohup bash runs/3d4/arm.sh F $ARGS --compound >> runs/3d4/F.out 2>&1 &
```

Poll `$F_RUNS/F.log` until its last line holds `cut F after`, `loop exited early` or `setup failed`. Then:

```bash
set -a; . runs/3d4/arm.env; set +a
tail -1 "$F_RUNS/F.log"
.venv/bin/python3 -c "
import json
a = json.load(open('$RUNS/$ARM-state-at-cut.json'))
f = json.load(open('$F_RUNS/F-state-at-cut.json'))
for key in a.keys() - {'generations', 'best', 'latest', 'compound'}:
    assert a[key] == f.get(key), key
assert not a.get('compound') and f['compound'] is True
for ga, gf in zip(a['generations'], f['generations']):
    assert ga['gen'] == gf['gen'] and ga['selfplay']['seed_start'] == gf['selfplay']['seed_start'], gf['gen']
    assert ga['selfplay']['games'] == gf['selfplay']['games'] and ga['gate']['seeds'] == gf['gate']['seeds'], gf['gen']
suffix = '+t+c' + ('' if f.get('opening') is None else '+o')
for g in f['generations']:
    assert g['selfplay']['compound'] is True and g['selfplay']['belief_resets'] == 0 and g['gate']['belief_resets'] == 0, g['gen']
    assert g['gate']['candidate'].endswith('@100' + suffix) and g['gate']['baseline'].endswith('@100' + suffix), g['gate']
print(f'F matches $ARM: {len(f[\"generations\"])} generations, the same set-up, seeds and gate; {suffix} gates; no belief resets')
" | tee -a runs/3d4/finals.log
cmp "$RUNS/$ARM/nets/start.bin" "$F_RUNS/F/nets/start.bin" && echo "F started from $ARM's start net" | tee -a runs/3d4/finals.log
```

Pass: `$F_RUNS/F.log` ends with `cut F after 108NN s` (10,800 to 10,810); the check prints `F matches ...`; the `cmp` line prints. If the arm's state holds a key that grows with its own promotions (3d-3's pool, if its state keeps one), the check's assertion on that key fails by construction: the controller compares that key's set-up entries by hand, removes the key from the comparison, reruns the check, and says so in `finals.log`.

- [ ] **Step 5: The tables and the final nets (controller runs this)**

```bash
set -a; . runs/3d4/arm.env; set +a
if [ "$F_RUNS" = runs/p2 ] && [ "$RUNS" = runs/p2 ]; then
  .venv/bin/python3 runs/p2/tables.py "$ARM" F | tee runs/3d4/tables.md
else
  sed "s#runs/p2/#runs/3d4/#g" runs/p2/tables.py > runs/3d4/tables.py
  cp "$RUNS/$ARM.log" "$RUNS/$ARM-state-at-cut.json" runs/3d4/
  .venv/bin/python3 runs/3d4/tables.py "$ARM" F | tee runs/3d4/tables.md
fi
fbest=$(.venv/bin/python3 -c "import json; print(json.load(open('$F_RUNS/F-state-at-cut.json'))['best'])")
cp "$F_RUNS/F/nets/$fbest.bin" runs/3d4/nets/F.bin
echo "F: $fbest ($F_RUNS/F/nets/$fbest.bin, runs/3d4/nets/F.bin)" | tee -a runs/3d4/finals.log
shasum -a 256 runs/3d4/nets/B.bin runs/3d4/nets/F.bin | tee -a runs/3d4/finals.log
cmp -s runs/3d4/nets/B.bin runs/3d4/nets/F.bin && echo "F's best is B: criterion 2 is H0 without a match" | tee -a runs/3d4/finals.log || true
```

Pass: `tables.md` has both tables and both games-per-hour lines; `finals.log` names F's net and both hashes.

- [ ] **Step 6: Criterion 2's SPRT (controller runs this)**

Skip this step if `finals.log` says F's best is B (recorded as H0). Otherwise:

```bash
set -a; . runs/3d4/arm.env; set +a
pgrep -fl "alphasettler|settler" || echo idle
.venv/bin/alphasettler sprt --candidate "net:F@100+t+c$O" --baseline "net:B@100+t+c$O" --trades \
  --seed-start 7001000000 --nets-dir runs/3d4/nets --out runs/3d4/sprt-c2.jsonl 2>&1 | tee runs/3d4/sprt-c2.log
.venv/bin/python3 -c "
import json
rs = [json.loads(l) for l in open('runs/3d4/sprt-c2.jsonl')]
print(f'{len(rs)} games, belief resets {sum(r[\"belief_resets\"] for r in rs)}')
" | tee -a runs/3d4/sprt-c2.log
```

Both sides are `+c`, so the test is the training, not the search. Pass: exit 0, a line starting `sprt: `, `belief resets 0`. H1: `--compound` becomes the setting new loops are set up with; H0 or undecided: recorded, and loops without it stay the default.

- [ ] **Step 7: Check the SPRT lines against the records (controller runs this)**

```bash
grep -E '^net:B@100\+t\+c(\+o)? vs net:B@100\+t(\+o)?: |^sprt: ' runs/3d4/sprt-c1.log
grep -E '^net:F@100\+t\+c(\+o)? vs net:B@100\+t\+c(\+o)?: |^sprt: ' runs/3d4/sprt-c2.log
```

Pass: two lines from each log (the summary and the verdict; none from `sprt-c2.log` when Step 6 was skipped): these are what Step 8 copies.

- [ ] **Step 8: Write `docs/bot/results-3d4.md`**

Every number is copied from the files named here; nothing is estimated. The file:

```markdown
# Sub-project 3d-4 (compound build moves in search): results

Date: <date of the last SPRT>. Machine: <`sysctl -n machdep.cpu.brand_string`>, macOS (<`uname -r`>).
Commit: <`git log -1 --format=%h`> (Plan 12, Task 4), plus the commit adding this file.

Spec: [`docs/superpowers/specs/2026-10-08-compound-moves-design.md`](../superpowers/specs/2026-10-08-compound-moves-design.md).
Plan: [`docs/superpowers/plans/2026-10-08-plan-12-compound-moves.md`](../superpowers/plans/2026-10-08-plan-12-compound-moves.md).
Builds on: <the best arm's results file, e.g. results-3d1b.md> (the best arm so far and its best net B).

B is <the best-arm line of runs/3d4/best-arm.log>: the last of 3d-3, 3d-2 and 3d-1b whose SPRT was
H1 named the arm (<which, and the outcome rows it was read from>), else P2's arm A. Both sides play
`+o` exactly when that arm used `--opening` (<yes / no>).

## Play strength, no training (done criterion 1)

| Run | Result | Bar | Outcome |
|---|---|---|---|
| `sprt --candidate net:B@100+t+c<+o> --baseline net:B@100+t<+o> --trades`, seeds from 7,000,000,000 | <the summary line and the sprt line of runs/3d4/sprt-c1.log> | H1 (p1 0.30 against p0 0.25, alpha = beta = 0.05) | <H1 / H0 / undecided: recorded either way> |

Belief resets: <the belief resets line of runs/3d4/sprt-c1.log>.

## Compound moves in play (done criterion 3, from the criterion-1 records)

- <the "decisions with a compound move" line of runs/3d4/sprt-c1.log>
- <the "bank trades per game" line of runs/3d4/sprt-c1.log>

## Training at equal wall time (done criterion 2)

Arm F is the best arm's set-up (<the script it ran through and the set-up arguments from
runs/3d4/best-arm.log>) plus `--compound`: the same seed directories, start net, window, training
minutes and gate, the same self-play and gate seeds; set up untimed, then 10,800 s of generations
(F cut after <N from F.log> s). Its self-play and both gate sides searched compound moves (`+t+c<+o>`
gates); the control's gates were `+t<+o>`, so the two gate columns measure different bots.

<the control's table and its self-play line, from runs/3d4/tables.md>

<the F table and its self-play line, from runs/3d4/tables.md>

Final nets: <the F line and the hashes of runs/3d4/finals.log>.

| Run | Result | Bar | Outcome |
|---|---|---|---|
| `sprt --candidate net:F@100+t+c<+o> --baseline net:B@100+t+c<+o> --trades`, seeds from 7,001,000,000 | <the summary line and the sprt line of runs/3d4/sprt-c2.log, or "not run: F's best is B"> | H1 (p1 0.30 against p0 0.25, alpha = beta = 0.05) | <H1: --compound becomes the setting new loops are set up with / H0: recorded, loops without it stay the default / undecided: recorded, loops without it stay the default> |

Belief resets: <the belief resets line of runs/3d4/sprt-c2.log>.

## Rulings the build made

- Compound moves at every Main decision of the searching player in its tree, the root included;
  opponents keep their plain moves. Compounds replace bare bank trades (the spec's ruling), so a
  turn whose trades enable no build within two trades cannot trade with the bank.
- The trade rule: the shortfall covered resource by resource in index order; a resource is given
  only while its cards beyond the target's cost cover its rate, the most such cards first, ties to
  the lower index; the bank must hold each card got. Targets are the engine's own build rules on a
  copy of the state with a hand that covers every cost.
- A compound is a search edge keyed 0x8000 | its build's id; its visits are recorded on its first
  trade's entry; its prior is its first trade's, split among compounds sharing that trade, and the
  move set is renormalised (uniform if nothing is left). Root noise mixes in before.
- A plan goes on only while the bot's own trade is the only event since its last step and the next
  step is legal; anything else, including no events, drops it and the bot searches. Plan steps are
  not searches and record nothing. Visit sampling draws whole compounds.
- `+c` with `+g` is an unknown bot; `--compound` with `--gumbel` is refused. Loops keep the setting
  like `--trades`; the train-loop's warm start uses compound moves too.
- Match records carry per-seat `bank_trades`, `searches` and `compound_decisions`.
- Arm F <ran through runs/p2/arm.sh unchanged / ran through runs/3d4/arm.sh, a copy of runs/p2/arm.sh
  with its outputs in runs/3d4 and the best arm's own start net>. Fresh seeds: 7,000,000,000 and
  7,001,000,000.

## Next

<one paragraph: criterion 1's verdict says whether compound moves help a trained net at play time;
criterion 2's whether training with them helps. If criterion 2 is H1, new loops are set up with
--compound. If criterion 1 is H1 and criterion 2 is not, what the two tables (held-out scores, gate
results, games per hour) point to. If both are H0, whether the share of compound decisions and the
bank trades per game suggest the spec's first ruling (no bare trades) cost more than it gained>
```

Fill every `<...>` from the named source before committing. The committed file holds numbers and text only, with no angle brackets.

- [ ] **Step 9: Commit**

Run: `grep -c '<' docs/bot/results-3d4.md`
Expected: `0`.

```bash
git add docs/bot/results-3d4.md
git diff --cached | grep -nE 'sk-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{36}|password=[^[:space:]]+'
git -c user.email=sean.may101@gmail.com commit -m "docs: 3d-4 results, compound build moves at play time and in training at equal wall time"
```

---

## Self-review

### Spec coverage

| Spec requirement | Task |
|---|---|
| §1 the search sees a trade and the build it enables on one edge | 1 (`compound.rs`, `Sim::descend`) |
| §2 at the player to move's Main decision with `+c`: every legal action but bare trades, plus one compound per target not affordable now but after at most two trades; targets every legal-by-position road, settlement, city and dev card | 1 (`targets`, `compounds`, `move_keys`; `each_target_gets_the_trades_the_rule_picks_and_none_needing_three`, `the_move_set_has_no_bare_trade_and_shared_first_trades_split_their_prior`) |
| §2 the trade rule: one card at a time, most to spare after the cost, ties lower index, best rate, never a needed resource; more than two trades or a bank that cannot fill: none | 1 (`trades_for`; the bank, tie, port and three-trade tests) |
| §2 priors: first trade's prior split among compounds sharing it, kept moves as now, renormalised | 1 (`move_priors`; `a_shared_first_trade_splits_its_prior_and_the_move_set_renormalises` and the root-prior test) |
| §2 a compound edge applies its actions in order, each legal, asserted | 1 (`descend`'s asserts; `compound_edges_apply_their_steps_legally_and_their_visits_land_on_their_first_trades`) |
| §2 rulings: compounds replace bare trades; prior from the first trade | 1 (move set, priors); the Rulings section |
| §3 the bot plays the first action and keeps the rest as a plan; plays it unsearched while every event since is its own planned action and the next is legal; else drops it | 2 (`plan_step`, `is_own_trade`; `a_compound_bot_plays_its_plan_unsearched`, `a_compound_bot_drops_its_plan_when_the_game_goes_otherwise`) |
| §3 self-play records a searched decision as now, compound visits on the first trade's policy id; plan steps record nothing | 1 (visits folded), 3 (`compound_self_play_plays_each_compound_through_and_records_only_its_first_step`, `test_compound_self_play_records_say_so_and_put_compound_visits_on_first_trades`) |
| §4 `+c` suffix, any order, once, `ismcts` and `net:`; without it nothing changes | 2 (names tests; the never-re-pin list); 3 (`compound_self_play_refuses_gumbel_and_is_off_by_default`) |
| §4 `fast-loop` / `train-loop --compound`, kept from set-up, self-play and both gate sides `+c`; refused with `--gumbel` | 4 (all five tests and the bad-settings case) |
| §4.1 criterion 1: `net:B@100+t+c` vs three `net:B@100+t`, `+o` per the arm, SPRT, seeds from 7,000,000,000, recorded either way | 5 (Steps 2–3) |
| §4.2 criterion 2: arm F = best arm + `--compound`, same seeds, 10,800 s after untimed set-up; `net:F@100+t+c[+o]` vs three best `@100+t+c[+o]`, seeds from 7,001,000,000; H1 makes `--compound` the default | 5 (Steps 4–6) |
| §4.3 results: share of decisions with a compound, bank trades per game with and without `+c` (from criterion-1 records), F's table beside its control's, both SPRT lines | 2 (record counters), 5 (Steps 3, 5, 8) |
| §5 out of scope: no >2 trades, no two builds, no domestic offers in compounds, no opponents' compounds, no engine or policy-head change | 1 (`MAX_TRADES`, one build, maritime only, `applies` checks the viewer); nothing under `engine/` or `net/` edited |
| §6 testing: search (targets, trades incl. port, tie, bank shortage, three-trade refusal; no bare trade; priors sum to 1, split; legal compound edge; without `+c` unchanged) | 1; the pins |
| §6 bots (names; plan unsearched; drop on unexpected event; `belief_resets` 0) | 2 |
| §6 self-play and data (visits on first trade's policy id; plan steps record nothing; `fast-loop --compound` two tiny generations and the resume rule) | 3, 4 |

### Placeholder scan

Searched for "TBD", "TODO", "implement later", "fill in", "similar to Task", "appropriate", "handle edge cases": none in the code steps. The only templates are Task 5's run-time branches (Step 2's `arm.env`, whose two values the step's rule decides from the recorded results files) and Step 8's results file, whose `<...>` marks each name their source; Step 9 refuses to commit while one is left. The one digest the new tests assert (`0x94f447a9c3baf5b2` in Task 3) is the existing pinned value of `standard_exploration_plays_the_games_it_played_before_p2`. Every other expected value was derived by hand from the code at 19f1ceb (rates, spares, bank counts, priors 1/3 and 1/(3k), 10/21, the one-move root).

### Type consistency

- Rust: `Compound { trades: Vec<Action>, build: Action }`, `compound_key`, `compounds(&State, &mut Vec<Compound>)`, `move_keys(&[Action], &[Compound], bool, &mut Vec<u16>)`, `move_priors(&[Action], &[f32], &[Compound]) -> Vec<f32>`, `find`, `move_of` (Task 1) are what `search.rs` and the tests call. `SearchConfig.compound`, `SearchResult.plan: Vec<Action>` and `SearchResult.compound_visits: Vec<(Vec<Action>, u32)>` (Task 1) are what `IsmctsBot::act_with`, `sample_move(&[Action], &[u32], &[(Vec<Action>, u32)], u64)` and the bots tests use (Task 2). `split_search_flags -> Option<(String, bool, bool)>` is (rest, gumbel, compound) in `make_bot`, `Roster::new`, `Roster::make` and the tests. `GameRecord::{bank_trades: [u32; 4], searches: [u64; 4], compound_decisions: [u64; 4]}`, `PlayedGame`, `play_counted`, `seat_counts` (Task 2) are what `run_match_by`, the binding and the tests read. `Exploration.compound` (Task 3) is what `play_with_opening`, the binding and the tests set.
- Python: record keys `bank_trades`, `searches`, `compound_decisions` (Task 2) are what Task 5 Step 3 reads. `check_compound(compound, gumbel=None)`, `COMPOUND_WITH_GUMBEL`, `settings(..., compound=False)`, `write_explore(..., compound=False)` and the block key `"compound"` (Task 3) are what `FastLoop.init`, `FastLoop._selfplay`, `loop._check` and `Loop._selfplay` (Task 4) call and the tests assert. `FastLoop.compound`, `Loop.compound`, `LoopSettings.compound` and the `"compound"` keys (`state.json`, each generation's `selfplay` entry, `manifest.json`, `selfplay.json` plans) are used with those names by the CLI, Task 5's checks and the tests.

### Review Focus

Each of the five lines has its test in the owning task: 1 and 2 in Task 1 (bank shortages, the one-move root) with Task 2 and 3 checks for 2; 3 and 4 in Task 2 (plan drops, sampled compounds) with Task 3's whole-game check for 4; 5 in Task 4 (resume rules and refusals).
