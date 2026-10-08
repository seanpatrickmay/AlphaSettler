# Plan 8: Gumbel Root Search (Performance Track P2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Gumbel root search (sequential halving at the root, the completed-Q improved policy as the policy target) to the ISMCTS search, expose it as the `+g` bot suffix and as a self-play mode of `fast-loop` / `train-loop` (`--gumbel SIMS`), then measure it: `+g` against plain at 32, 100 and 600 simulations, and two 3-hour `fast-loop` runs from `gen-034` (control A, Gumbel B) settled by an SPRT.

**Architecture:** `settler_search` gains `RootMode` in `SearchConfig` (`Puct` as today, or `Gumbel { draw }`), the arithmetic in a new `search/src/gumbel.rs` (logits, candidates, phases, σ, v_mix, completedQ, the improved policy), and a search loop refactored into a `Sim` whose `run(count, step)` can force each simulation's root step; PUCT calls it once with no forced steps, so its games do not change. `SearchResult` carries `improved`. `IsmctsBot::with_gumbel` and `ismcts::split_gumbel` give `+g`; `Exploration::gumbel(n)` gives Gumbel self-play, whose `Decision`s and record dicts carry `target`; the training sampler (`net/src/data.rs`) uses `target` when a decision has one. Python threads `gumbel` through `selfplay.write_explore`, `FastLoop` (state.json) and `Loop` (manifest.json). The controller then runs the measurements and writes `docs/bot/results-p2-gumbel.md`.

**Tech Stack:** Rust 1.80+ (edition 2021, no new crates), PyO3 0.29.3 + maturin 1.15.0, Python 3.12 stdlib, PyTorch 2.14.1 (`train` extra only), pytest 9.1.1.

**Spec:** `docs/superpowers/specs/2026-10-08-gumbel-root-design.md` (binding). Builds on the Accelerate forward pass (`docs/perf/net-search.md`, 285d5d7, 1cd6a37), the trading game (Plan 6) and `runs/3c/nets/gen-034.bin`. Plan 7 (`docs/superpowers/plans/2026-10-08-plan-7-opening-model.md`) is not implemented; nothing here depends on it (see the `+g` ruling for how the two fit together).

## Global Constraints

- Python is `.venv/bin/python3`, tests `.venv/bin/pytest`, the CLI `.venv/bin/alphasettler`. Never `python`, `pip`, `python3` outside the venv, or `pip3` outside the venv.
- Rust is `PATH=$HOME/.cargo/bin:$PATH cargo ...` (cargo is not on PATH; `~/.cargo/bin/cargo` is the same binary). MSRV 1.80 (`rust-version = "1.80"`): use nothing stabilised after 1.80 (no `Option::is_none_or`, no `is_multiple_of`). No new external crates and no new Python dependencies. No numpy anywhere.
- Rebuild the extension after any Rust change used from Python. Another measurement may hold the extension: **never run `maturin develop`**. Build a wheel and rename the new library over the old one (an atomic rename, so running processes keep their mapped copy):
  ```bash
  rm -rf target/wheels-p2 && PATH=$HOME/.cargo/bin:$PATH .venv/bin/maturin build --release -q -o target/wheels-p2 \
    && unzip -q -o target/wheels-p2/alphasettler-*.whl 'alphasettler/_engine*.so' -d target/wheels-p2/x \
    && cp target/wheels-p2/x/alphasettler/_engine.cpython-312-darwin.so alphasettler/._engine.new.so \
    && mv -f alphasettler/._engine.new.so alphasettler/_engine.cpython-312-darwin.so
  ```
  "Rebuild the extension" in any step below means exactly this command.
- Every task ends with `PATH=$HOME/.cargo/bin:$PATH cargo fmt --all` then `PATH=$HOME/.cargo/bin:$PATH cargo fmt --check` (no output), `PATH=$HOME/.cargo/bin:$PATH cargo clippy --workspace --release --all-targets -- -D warnings` (exit 0), the full `PATH=$HOME/.cargo/bin:$PATH cargo test --workspace --release` (all pass) and, after the rebuild, `.venv/bin/pytest -q -rs` (all pass, the summary lists no skipped tests).
- Nothing under `engine/` is edited; `engine/tests/golden_trace.rs` passes unchanged. Under `search/` only what Task 1 names changes. 3a's digests (`the_generic_refactor_leaves_3a_ismcts_games_unchanged`, `fixed_exploration_is_3a_self_play`) and the three P2 pins Task 1 adds are never re-pinned: without `+g` and without `--gumbel`, every bot and every self-play game plays exactly as at eb1f739 (spec Section 4: "every recorded result and golden digest stays valid").
- The algorithm is spec Section 3, root only: m = min(16, |A|) candidates (`MAX_CANDIDATES` = 16) by g + ℓ, ties to the lower action id; n = simulations − 1 over ceil(log2 m) phases; each phase max(1, floor(n / (phases · |remaining|))) per remaining candidate, root step forced, PUCT below; keep ceil(|remaining| / 2) by g + ℓ + σ(q̂); the last phase (or one candidate left) spends the rest round-robin; σ(q) = (c_visit + max_b N(b)) · c_scale · q with `C_VISIT` = 50, `C_SCALE` = 1.0; the move is the best remaining by g + ℓ + σ(q̂); π' = softmax(ℓ + σ(completedQ)) over all of A with completedQ = q̂ if visited, else v_mix = (v̂ + (ΣN / Σ_vis π) · Σ_vis π q̂) / (1 + ΣN). g ~ Gumbel(0, 1) from the search's own `Rng` in self-play, g = 0 in play. Leaf batching keeps virtual loss; a phase's root steps go round-robin, `batch` at a time.
- `+g` (spec Section 4) is a suffix like `+t`, `+sK`, `+rD`, in any order, at most once, on `ismcts` and `net:` names; the bot searches with g = 0; `+sK` keeps its factor.
- Gumbel self-play (spec Section 5): every move one Gumbel search of `--gumbel SIMS` with g drawn, recorded as a full search with its improved policy as `target`; no Dirichlet noise, no playout cap randomization, no visit sampling. Gates are unchanged (PUCT bots).
- The trading game is exactly `{"max_offers_per_turn": 2, "max_trade_cards": 1}` (`TRADES` in `alphasettler/arena.py`, `trading()` in Rust tests).
- Determinism: same seed and weights give the same games, decisions and targets, independent of thread count. Budgets are simulation counts, never wall time (the criterion-2 runs' 3-hour budget is the experiment's wall clock, not a search budget). The forward pass is Accelerate-based: batch results equal alone only within 1e-5, a seed still replays bit for bit, so no test compares results across different batch compositions.
- No module outside `alphasettler/train/` imports torch.
- Git: one commit per task with the message given in the task, run as `git -c user.email=sean.may101@gmail.com commit -m "..."`. No Co-Authored-By or AI attribution, never `--no-verify`, never push. Before every commit run `git diff --cached | grep -nE 'sk-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{36}|password=[^[:space:]]+'` and expect no output.
- Testing: integration tests over mocks (real searches on real positions, real games, real record files read back through the real Rust parser, real loops through the CLI); test behaviour, not implementation; independent tests (each builds its own temp files); descriptive names; pytest fixtures and `parametrize` where cases repeat.
- Long runs (criterion 1's three matches, the two 3-hour loops, the final SPRT) are run by the controller in the background, never by implementer subagents. Steps marked **controller runs this** give the exact command and the pass condition.
- No README or docs beyond `docs/bot/results-p2-gumbel.md` (Task 6).

## Review Focus

1. Budgets below the candidate count: a `+g` bot at `@2` to `@17`, or Gumbel self-play at 2 simulations, on a root with 30 moves (n < m: the budget runs out inside phase 0). Expected: exactly `simulations - 1` root visits, a move from the root actions, π' finite and summing to 1, no panic or underflow. Pinned by Task 1's `tiny_budgets_spend_exactly_simulations_minus_one_root_visits`.
2. A leaf batch larger than the candidates left (the last phase has 2 candidates while `NET_BATCH` is 8): the third forced step meets the first one's pending leaf (`Busy`). Expected: the batch ends there and the step is retried in the next batch, never spinning, skipping or double counting; no virtual loss is left. Pinned by Task 1's `batches_larger_than_the_candidates_leave_no_virtual_loss`.
3. A root prior with exact zeros (a confident net's softmax underflows; a hand-written evaluator). Expected: finite logits (ln of the floor), π' finite and summing to 1, a zero-prior move still a candidate when the root has 16 or fewer moves. Pinned by Task 1's `a_zero_prior_has_a_finite_logit` and `a_prior_with_zeros_still_gives_a_proper_improved_policy`.
4. Record files whose `target` someone else wrote: wrong length, NaN, negative, not summing to 1, not a list of numbers; and a target on a decision with no visits and `full_search` false. Expected: a ValueError naming the game and the move before anything trains; the zero-visit decision is a policy target. Pinned by Task 4's `a_bad_target_is_an_error_naming_the_decision`, `a_decision_with_a_target_trains_on_it_whatever_its_visits` and `test_a_malformed_target_is_a_value_error_naming_the_decision`.
5. Resuming a loop in another self-play mode: set up without `--gumbel` and resumed with it, set up with 100 and resumed with 50, resumed with no flag, or a `state.json` / `manifest.json` written before P2 (no `gumbel` key). Expected: the first two refused (exit 2, the message says how it was set up), the third keeps 100, the last reads as set up without. Pinned by Task 5's `test_cli_fast_loop_keeps_the_self_play_mode_it_was_set_up_with`, `test_a_loop_set_up_before_gumbel_existed_reads_as_set_up_without_it`, `test_a_gumbel_train_loop_records_its_mode_and_improved_policies` and `test_a_train_loop_set_up_without_gumbel_refuses_it`.

## File Structure

| File | Responsibility | Task |
|---|---|---|
| `search/src/gumbel.rs` (new) | the Section 3 arithmetic on plain numbers: `gumbel`, `logits`, `softmax`, `candidates`, `phases`, `per_candidate`, `sigma`, `v_mix`, `completed_q`, `ranked`, `improved_policy`; constants | 1 |
| `search/src/search.rs` | `RootMode`, `SearchConfig::root`, `SearchResult::improved`; the loop as `Sim::{descend, run}`; `most_visited`, `gumbel_root` | 1 |
| `search/src/noise.rs`, `search/src/lib.rs` | `open_unit` visible in the crate; module and re-exports | 1 |
| `bots/src/ismcts.rs` | `split_gumbel`, `IsmctsBot::with_gumbel` | 2 |
| `bots/src/lib.rs`, `bots/src/netbot.rs`, `alphasettler/cli.py` | `+g` in `make_bot` and `Roster`; name texts | 2 |
| `bots/src/selfplay.rs` | `Exploration::gumbel`, `Exploration.gumbel`, validation, `Decision.target` | 3 |
| `bindings/src/lib.rs`, `alphasettler/selfplay.py` | `target` in record dicts; `selfplay_explore(gumbel=)`; `check_gumbel`, `settings`, `write_explore(gumbel=)` | 3 |
| `net/src/data.rs`, `bindings/src/train.rs` | `RecordedDecision.target`, its checks, the sampler's policy row and policy-target rule; parsing | 4 |
| `alphasettler/train/fastloop.py`, `alphasettler/train/loop.py`, `alphasettler/cli.py` | `--gumbel SIMS` for both loops, kept in `state.json` / `manifest.json` | 5 |
| `docs/bot/results-p2-gumbel.md` (new) | the measurements | 6 |
| tests: `search/tests/gumbel.rs` (new, 1), `bots/tests/ismcts.rs` (1, 2), `bots/tests/netbot.rs` (1, 2), `bots/tests/selfplay.rs` (1, 3), `net/tests/data.rs` (4), `tests/python/net_names.py` (2), `tests/python/test_netbot.py` (2), `tests/python/test_training_data.py` (3, 4), `tests/python/test_fast_loop.py` (5), `tests/python/test_train_loop.py` (5) | | 1–5 |

## Where the code forces a choice the spec does not make

Each is a **Ruling** (with its reason) the results file repeats, so it can be overridden.

- **Gumbel is a root mode of the bot's `SearchConfig`, not a per-move option.** `SearchConfig.root: RootMode` (`Puct`, `Gumbel { draw }`); `+g` bots set `Gumbel { draw: false }`, Gumbel self-play builds its four bots with `Gumbel { draw: true }`. `MoveOptions` is unchanged: no bot mixes modes move by move, and its literals stay as they are.
- **The search loop becomes `Sim::run(count, step)`.** PUCT calls it once with `|_| None`, which draws from the `Rng` in exactly the old order (a `Busy` descent still ends the batch and the simulation is retried in a new world), so every PUCT game is unchanged; Task 1 pins three new digests before touching the loop. `descend` becomes a `Sim` method (the forced step would have made it an eight-argument function, which clippy refuses).
- **π is softmax(ℓ) with ℓ = ln max(π_root / Σ, `f32::MIN_POSITIVE`)**, the root prior renormalised over A (uniform if it sums to nothing or is not finite). A zero prior then has a finite logit (about −87.3) and v_mix's Σ_vis π is never 0; the floor moves no probability by more than 1e-38.
- **Ties and unvisited candidates.** Halving and the final pick break ties to the lower action id, as the candidates do. If the budget runs out before a phase (n < the phase's share), its unvisited candidates are ranked with completedQ (v_mix), as mctx does; the spec's q̂ is undefined there.
- **Round-robin order** within a phase is the current ranking (best first); the last phase's `left` simulations go round-robin from the front, so the leading candidate gets the odd one. "A phase with one candidate left" cannot arise with ceil halving over ceil(log2 m) phases; the guard stays because the spec states it.
- **v̂ is the root evaluation's value for `obs.viewer`** (the root decision is the searching player's own). The root's backup path is empty, so `Sim` keeps that evaluation (`root_value`).
- **The Gumbel draws** are one per root action, in root-action (engine) order, right after the root's evaluation, from the search's `Rng`.
- **The forced-step assertion panics naming the move**, at the first world where a root action is not legal (spec: "the search asserts it").
- **`+g` is stripped by a new `ismcts::split_gumbel` before `split_trades` and `parse_net_name`**, which keep their signatures (`parse_net_name` still returns its 5-tuple; `make_bot` and `Roster` call `split_gumbel` first). Plan 7's later `+o` (its `split_suffixes`, `Flags` and `NetName`) then runs on the stripped name unchanged: whichever plan lands second keeps `split_gumbel` as the first step of `make_bot`, `Roster::new` and `Roster::make`, and adds `with_gumbel(gumbel)` to the bot it builds.
- **`Exploration` gains `gumbel: bool` and `Exploration::gumbel(n)`** (n full searches, p 1, no noise, no sampled turns). `validate` refuses `gumbel` with noise, sampled turns, p below 1 or fast ≠ full simulations: the spec turns those off in this mode.
- **A decision with a `target` is a policy target whatever its visits and its `full_search` flag**; `full_decisions` still counts the flag. A target must have one entry per legal move, each finite and non-negative, summing to 1 within `TARGET_TOLERANCE` = 1e-4.
- **Records carry `target` only on Gumbel decisions**, so PUCT records stay byte-identical. A Gumbel record's `selfplay` block is `{bot, net, net_sha256, gumbel_simulations, batch}`: the PUCT block's exploration keys do not apply.
- **`selfplay_explore(..., gumbel=N)` ignores its three PUCT arguments**, which stay positional so every existing caller is unchanged.
- **Loops keep their mode** in `state.json` (`fast-loop`) and `manifest.json` (`train-loop`) under `"gumbel"` (null or SIMS), set when the loop is set up. A resume without `--gumbel` keeps it; `--gumbel` with another value, or on a loop set up without it, is refused (exit 2). Files from before P2 have no key and read as "without". `train-loop`'s per-generation `selfplay.json` plan gains `"gumbel"` only when set, so a run started before P2 still matches its plans. The fast-loop's per-generation `selfplay` entry records `"gumbel"`.
- **No `--gumbel` for `alphasettler selfplay`**: the spec names only the loops.
- **The criterion-2 budget** (spec: "the same wall-clock budget of 3 hours of generations"): a wrapper starts the arm's `fast-loop` (set-up included), copies `state.json` at the first 10-second check at or past 10,800 s, then stops the loop with SIGTERM. The arm's generations and final net are that copy's: a generation not finished by the cut does not count. Same rule and budget for both arms. (SIGINT cannot be used: bash starts background jobs with SIGINT ignored, and Python then installs no KeyboardInterrupt handler.)
- **Both arms are set up alike**: fresh roots `runs/p2/A` and `runs/p2/B`; seed directories `runs/3b-v2/gen-002` .. `gen-005` (3c's, from `runs/3c/loop.sh`) and `runs/3c/gen-030` .. `gen-034`; start net `runs/3c/nets/gen-034.bin`; `--first 35`, `--trades`, `--generations 99` (generation 99's seeds stay below the gate's 1,000,000,000); every other setting the default 3c used (2,000 games, window 40,000 games, 10 training minutes, a gate of 100 seeds at 100 simulations). Both play the same self-play seeds (generation k from k × 10,000,000). B adds `--gumbel 100`. A runs first, then B, with nothing else running.
- **Fresh seeds**: criterion 1's SPRTs from 3,000,000,000 and 3,000,001,000, its 600-simulation arena from 3,000,002,000, criterion 2's SPRT from 3,000,003,000. No self-play (below 1e9), gate (1e9 + 0..200) or SPRT run so far (the default 2e9) used them; each SPRT uses at most 400 seeds.
- **Games per hour** is self-play games of the generations finished in the budget over the hours their self-play took (timestamped log, from `gen-NNN: selfplay` to `gen-NNN: train`); the results also give the generations finished in 3 hours.
- **If both arms' final nets are the same file** (neither loop promoted a net), criterion 2's SPRT is not run and is recorded as H0: the two bots are identical.
- **H1 makes Gumbel the mode the next loop is set up with** (`--gumbel 100`); no CLI default changes in this plan, since a resumed loop keeps its mode anyway.
- **Task boundaries:** the search (1), the `+g` names (2), self-play records (3), training data (4), the loops (5), the runs (6). Records (3) and training data (4) are separate gates: a reviewer can accept what self-play writes and still reject how the sampler reads it.

---

### Task 1: Gumbel root search in `settler_search`

**Files:**
- Create: `search/src/gumbel.rs`
- Modify: `search/src/search.rs` (whole file below), `search/src/noise.rs` (`open_unit` visibility), `search/src/lib.rs` (module, re-export)
- Test: `search/tests/gumbel.rs` (new); `bots/tests/ismcts.rs`, `bots/tests/netbot.rs`, `bots/tests/selfplay.rs` (one pin each)

**Interfaces:**
- Consumes: `settler_engine::rng::Rng`; `crate::noise::open_unit`; `crate::puct::select`; `crate::world::WorldSampler::{new, sample}`; `crate::tree::{Tree, Child, NodeKind, ROOT}`.
- Produces:
  - `settler_search::RootMode` = `enum RootMode { Puct, Gumbel { draw: bool } }` (`Clone, Copy, Debug, PartialEq, Eq`)
  - `SearchConfig { simulations, c_puct, batch, offers, root: RootMode }`, `SearchConfig::default().root == RootMode::Puct`
  - `SearchResult { action, visits, simulations, nodes, improved: Option<Vec<f32>> }` (`improved` aligned with `legal`, `Some` only for a Gumbel search that searched)
  - `settler_search::gumbel::{MAX_CANDIDATES: usize = 16, C_VISIT: f32 = 50.0, C_SCALE: f32 = 1.0, MIN_PRIOR: f32 = f32::MIN_POSITIVE}` and `gumbel(&mut Rng) -> f32`, `logits(&[f32]) -> Vec<f32>`, `softmax(&[f32]) -> Vec<f32>`, `candidates(base: &[f32], keys: &[u16]) -> Vec<usize>`, `phases(m: usize) -> u32`, `per_candidate(n: u32, phases: u32, remaining: usize) -> u32`, `sigma(q: f32, max_visits: u32) -> f32`, `v_mix(logits, visits: &[u32], q: &[f32], v_hat: f32) -> f32`, `completed_q(logits, visits, q, v_hat) -> Vec<f32>`, `ranked(of: &[usize], base, keys, cq: &[f32], max_visits: u32) -> Vec<usize>`, `improved_policy(logits, cq, max_visits) -> Vec<f32>`

- [ ] **Step 1: Pin today's games before touching the loop**

These three tests pass at the base commit (the digests were computed at eb1f739); they must still pass after every later step.

In `bots/tests/ismcts.rs`, after `the_generic_refactor_leaves_3a_ismcts_games_unchanged`, add:

```rust
/// The digest of a trading-game match of `ismcts@200+t+s2`, computed at eb1f739 before the
/// Gumbel root (P2): bots without `+g` must play the same games (spec P2 Section 4).
#[test]
fn bots_without_the_gumbel_suffix_play_the_games_they_played_before_p2() {
    let records = run_match("ismcts@200+t+s2", "greedy", 0..2, trading(), 1).unwrap();
    assert_eq!(digest(&records), 0x840a576a3ed2426e);
}
```

In `bots/tests/netbot.rs`, change the import line `use settler_bots::arena::{belief_resets, play_game, run_match_in};` to `use settler_bots::arena::{belief_resets, play_game, run_match_in, GameRecord};`, and after `a_net_bot_plays_whole_games_independent_of_thread_count` add:

```rust
/// FNV-1a over (winner, final VP, turns, actions) of each record, as in tests/ismcts.rs.
fn record_digest(records: &[GameRecord]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let mut eat = |bytes: &[u8]| {
        for &b in bytes {
            h ^= b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    };
    for r in records {
        eat(&[r.winner.unwrap_or(u8::MAX)]);
        eat(&r.vp);
        eat(&r.turns.to_le_bytes());
        eat(&r.actions.to_le_bytes());
    }
    h
}

/// The digest of a trading-game match of `net:tiny@16+t` (random weights, seed 1), computed at
/// eb1f739 before the Gumbel root (P2): net bots without `+g` must play the same games.
#[test]
fn net_bots_without_the_gumbel_suffix_play_the_games_they_played_before_p2() {
    let dir = nets_dir("p2-pin");
    let records = run_match_in("net:tiny@16+t", "greedy", 0..1, trading(), 1, &dir).unwrap();
    assert_eq!(record_digest(&records), 0xf4d5c47b40f849a3);
}
```

In `bots/tests/selfplay.rs`, after `fixed_exploration_is_3a_self_play`, add:

```rust
/// The digest of a trading-game self-play game under the 3b exploration at 6 / 3 simulations,
/// computed at eb1f739 before the Gumbel root (P2): PUCT self-play must not change.
#[test]
fn standard_exploration_plays_the_games_it_played_before_p2() {
    assert_eq!(
        digest(&play_with(2, trading(), &small_explore(), 8, &heuristic)),
        0x94f447a9c3baf5b2
    );
}
```

Run: `PATH=$HOME/.cargo/bin:$PATH cargo test --release -p settler-bots --test ismcts --test netbot --test selfplay -- before_p2`
Expected: PASS, 3 tests.

- [ ] **Step 2: Write the failing Gumbel tests**

Create `search/tests/gumbel.rs`:

```rust
//! Gumbel root search (spec docs/superpowers/specs/2026-10-08-gumbel-root-design.md Sections 3
//! and 7): candidates, sequential halving's budget, the chosen move and the improved policy.

mod common;

use common::*;
use settler_engine::rng::Rng;
use settler_engine::*;
use settler_search::gumbel::MAX_CANDIDATES;
use settler_search::tree::ROOT;
use settler_search::{
    search, search_moves, search_tree, Belief, Eval, Evaluator, Leaf, RootMode, SearchConfig,
    SearchResult, UniformEvaluator,
};
use std::collections::HashSet;

fn gumbel(simulations: u32, draw: bool) -> SearchConfig {
    SearchConfig {
        simulations,
        root: RootMode::Gumbel { draw },
        ..SearchConfig::default()
    }
}

/// Player 0 in Main after setup holding four of every resource: about 30 searchable moves
/// (roads, settlements, a dev card, 20 bank trades, end turn).
fn rich(seed: u64) -> State {
    edit(&after_setup(seed, no_trades()), no_trades(), |snap| {
        give_cards(snap, 0, [4, 4, 4, 4, 4]);
        snap.phase = Phase::Main;
    })
}

/// Player 0 in Main with city resources and `vp_to_win` 3: either city wins on the spot.
fn city_wins(seed: u64) -> State {
    let config = GameConfig {
        vp_to_win: 3,
        ..no_trades()
    };
    let base = after_setup(seed, config);
    edit(&base, config, |snap| {
        give_cards(snap, 0, CITY_COST);
        snap.phase = Phase::Main;
    })
}

/// Values ¼ each; the prior grows with a move's place in `legal`, (i + 1) / Σ, so the last moves
/// are the likeliest.
struct Graded;

impl Evaluator for Graded {
    fn evaluate(&mut self, leaves: &[Leaf<'_>]) -> Vec<Eval> {
        leaves
            .iter()
            .map(|l| {
                let n = l.legal.len();
                let total = (n * (n + 1) / 2) as f32;
                Eval {
                    value: [0.25; 4],
                    prior: (1..=n).map(|i| i as f32 / total).collect(),
                }
            })
            .collect()
    }
}

/// Values ¼ each; all the prior on the first legal move, exactly none on the rest.
struct FirstOnly;

impl Evaluator for FirstOnly {
    fn evaluate(&mut self, leaves: &[Leaf<'_>]) -> Vec<Eval> {
        leaves
            .iter()
            .map(|l| {
                let mut prior = vec![0.0; l.legal.len()];
                prior[0] = 1.0;
                Eval {
                    value: [0.25; 4],
                    prior,
                }
            })
            .collect()
    }
}

/// Player 0's search of `s` from `seed` (belief and search streams).
fn run<E: Evaluator>(s: &State, eval: &mut E, cfg: &SearchConfig, seed: u64) -> SearchResult {
    let obs = s.observation(0);
    let b = Belief::from_observation(&obs, 64, seed);
    search(&obs, &s.legal_actions(), &b, eval, cfg, &mut Rng::new(seed)).unwrap()
}

fn root_moves(s: &State) -> Vec<Action> {
    let mut moves = Vec::new();
    search_moves(&s.legal_actions(), false, &mut moves);
    moves
}

fn sums_to_one(p: &[f32]) -> bool {
    p.iter().all(|x| x.is_finite() && *x >= 0.0)
        && (p.iter().map(|&x| x as f64).sum::<f64>() - 1.0).abs() < 1e-5
}

#[test]
fn sequential_halving_spends_the_budget_phase_by_phase_on_the_lowest_ids_when_all_else_ties() {
    let s = rich(1);
    let legal = s.legal_actions();
    assert!(legal.len() > MAX_CANDIDATES, "{} moves", legal.len());
    let r = run(&s, &mut UniformEvaluator, &gumbel(100, false), 1);
    assert_eq!(r.simulations, 100);
    // n = 99 over 4 phases: 16 candidates x 1, 8 x 3, 4 x 6, then the last 2 share the 35 left
    // (18 and 17). A uniform prior and equal values tie everything, so the candidates and every
    // halving keep the lowest action ids.
    let mut by_id: Vec<usize> = (0..legal.len()).collect();
    by_id.sort_by_key(|&i| legal[i].encode());
    let want_by_rank = [28, 27, 10, 10, 4, 4, 4, 4, 1, 1, 1, 1, 1, 1, 1, 1];
    for (rank, &i) in by_id.iter().enumerate() {
        let want = want_by_rank.get(rank).copied().unwrap_or(0);
        assert_eq!(r.visits[i], want, "rank {rank}: {:?}", legal[i]);
    }
    assert_eq!(r.visits.iter().sum::<u32>(), 99);
    assert_eq!(r.action, legal[by_id[0]], "the halving's winner");
}

#[test]
fn without_draws_the_candidates_are_the_sixteen_likeliest_moves() {
    let s = rich(2);
    let n = s.legal_actions().len();
    assert!(n > MAX_CANDIDATES, "{n} moves");
    let r = run(&s, &mut Graded, &gumbel(64, false), 2);
    let visited: Vec<bool> = r.visits.iter().map(|&v| v > 0).collect();
    let want: Vec<bool> = (0..n).map(|i| i >= n - MAX_CANDIDATES).collect();
    assert_eq!(visited, want);
    assert_eq!(r.visits.iter().sum::<u32>(), 63);
}

#[test]
fn with_nothing_visited_the_improved_policy_is_the_prior() {
    let s = rich(3);
    let legal = s.legal_actions();
    let n = legal.len();
    let r = run(&s, &mut Graded, &gumbel(1, false), 3);
    assert_eq!((r.simulations, r.visits.iter().sum::<u32>()), (1, 0));
    let improved = r.improved.unwrap();
    let total = (n * (n + 1) / 2) as f32;
    for (i, p) in improved.iter().enumerate() {
        assert!(
            (p - (i + 1) as f32 / total).abs() < 1e-6,
            "move {i}: {p}"
        );
    }
    assert_eq!(r.action, legal[n - 1], "g = 0: the likeliest move");
}

#[test]
fn a_move_that_wins_on_the_spot_is_chosen_and_draws_the_improved_policy() {
    for seed in 0..5 {
        let s = city_wins(seed);
        let legal = s.legal_actions();
        let r = run(&s, &mut UniformEvaluator, &gumbel(50, false), seed);
        let mut after = s;
        after.apply(r.action);
        assert_eq!(after.winner(), Some(0), "seed {seed}: {:?}", r.action);
        let improved = r.improved.unwrap();
        assert!(sums_to_one(&improved), "seed {seed}: {improved:?}");
        // The prior gives the two cities 2 of at least 3 moves; their value moves nearly all of
        // the improved policy onto them.
        let cities: f32 = legal
            .iter()
            .zip(&improved)
            .filter(|(a, _)| matches!(a, Action::BuildCity(_)))
            .map(|(_, p)| p)
            .sum();
        assert!(cities > 0.9, "seed {seed}: {cities}");
    }
}

#[test]
fn the_improved_policy_covers_the_legal_moves_and_is_zero_off_the_searched_ones() {
    let config = GameConfig::default();
    let s = edit(&after_setup(7, config), config, |snap| {
        give_cards(snap, 0, [1, 1, 1, 1, 1]);
        snap.phase = Phase::Main;
    });
    let legal = s.legal_actions();
    let is_offer = |a: &Action| matches!(a, Action::OfferTrade { .. });
    assert!(legal.iter().any(is_offer), "offers are legal here");
    let r = run(&s, &mut UniformEvaluator, &gumbel(60, false), 4);
    let improved = r.improved.unwrap();
    assert_eq!(improved.len(), legal.len());
    assert!(sums_to_one(&improved));
    for (a, (&p, &v)) in legal.iter().zip(improved.iter().zip(&r.visits)) {
        if is_offer(a) {
            assert_eq!((p, v), (0.0, 0), "{a:?}");
        }
    }
}

#[test]
fn tiny_budgets_spend_exactly_simulations_minus_one_root_visits() {
    for s in [rich(5), city_wins(1)] {
        let moves = root_moves(&s);
        for simulations in [2, 3, 5, 17, 40] {
            for draw in [false, true] {
                let r = run(
                    &s,
                    &mut UniformEvaluator,
                    &gumbel(simulations, draw),
                    simulations as u64,
                );
                let what = format!("{} moves, {simulations} simulations, draw {draw}", moves.len());
                assert_eq!(r.simulations, simulations, "{what}");
                assert_eq!(r.visits.iter().sum::<u32>(), simulations - 1, "{what}");
                assert!(sums_to_one(r.improved.as_ref().unwrap()), "{what}");
                assert!(moves.contains(&r.action), "{what}");
            }
        }
    }
}

#[test]
fn batches_larger_than_the_candidates_leave_no_virtual_loss() {
    for (s, batch) in [(city_wins(3), 8), (rich(6), 16)] {
        let obs = s.observation(0);
        let legal = s.legal_actions();
        let b = Belief::from_observation(&obs, 64, 6);
        let cfg = SearchConfig {
            batch,
            ..gumbel(300, true)
        };
        let (r, tree) = search_tree(
            &obs,
            &legal,
            &b,
            &mut UniformEvaluator,
            &cfg,
            &mut Rng::new(6),
        )
        .unwrap();
        assert_eq!((r.simulations, r.visits.iter().sum::<u32>()), (300, 299));
        for (i, n) in tree.nodes.iter().enumerate() {
            assert!(!n.awaiting_eval, "node {i} still awaits an evaluation");
            assert!(
                n.children.iter().all(|c| c.pending == 0),
                "node {i} keeps a virtual loss"
            );
        }
        let root: u32 = tree.nodes[ROOT as usize]
            .children
            .iter()
            .map(|c| c.visits)
            .sum();
        assert_eq!(root, 299);
    }
}

#[test]
fn a_prior_with_zeros_still_gives_a_proper_improved_policy() {
    let s = rich(7);
    let legal = s.legal_actions();
    let r = run(&s, &mut FirstOnly, &gumbel(40, false), 7);
    let improved = r.improved.unwrap();
    assert!(sums_to_one(&improved), "{improved:?}");
    assert!(r.visits[0] > 0, "the one likely move is a candidate");
    assert!(legal.contains(&r.action));
}

#[test]
fn the_same_seed_gives_the_same_gumbel_search_and_draws_change_the_candidates() {
    let s = rich(8);
    let drawn = gumbel(64, true);
    let a = run(&s, &mut UniformEvaluator, &drawn, 11);
    assert_eq!(a, run(&s, &mut UniformEvaluator, &drawn, 11));
    let visited = |r: &SearchResult| r.visits.iter().map(|&v| v > 0).collect::<Vec<bool>>();
    let with_draws: HashSet<Vec<bool>> = (0..6)
        .map(|seed| visited(&run(&s, &mut UniformEvaluator, &drawn, seed)))
        .collect();
    assert!(with_draws.len() > 1, "the draws never changed the candidates");
    let plain = gumbel(64, false);
    let without: HashSet<Vec<bool>> = (0..6)
        .map(|seed| visited(&run(&s, &mut UniformEvaluator, &plain, seed)))
        .collect();
    assert_eq!(without.len(), 1, "without draws the prior alone picks them");
}

#[test]
fn a_single_searchable_move_is_returned_without_searching() {
    let s = after_setup(3, no_trades()); // PreRoll, no dev cards: only Roll
    assert_eq!(s.legal_actions(), vec![Action::Roll]);
    let r = run(&s, &mut UniformEvaluator, &gumbel(500, true), 0);
    assert_eq!(
        (r.action, r.simulations, r.improved),
        (Action::Roll, 0, None)
    );
}

#[test]
fn the_default_root_is_puct_and_reports_no_improved_policy() {
    assert_eq!(SearchConfig::default().root, RootMode::Puct);
    let cfg = SearchConfig {
        simulations: 50,
        ..SearchConfig::default()
    };
    assert_eq!(run(&rich(1), &mut UniformEvaluator, &cfg, 1).improved, None);
}
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `PATH=$HOME/.cargo/bin:$PATH cargo test --release -p settler-search --test gumbel`
Expected: FAIL to compile with `unresolved import settler_search::gumbel` (and `RootMode`).

- [ ] **Step 4: Write `search/src/gumbel.rs`**

In `search/src/noise.rs` change `fn open_unit(rng: &mut Rng) -> f64 {` to `pub(crate) fn open_unit(rng: &mut Rng) -> f64 {`.

Create `search/src/gumbel.rs`:

```rust
//! Gumbel root search arithmetic (spec docs/superpowers/specs/2026-10-08-gumbel-root-design.md
//! Section 3; Danihelka et al., ICLR 2022, Algorithm 2 and appendix eq. 33), kept apart from the
//! tree walk like `puct` so each rule can be tested on plain numbers. Every slice is aligned with
//! the root actions (the searchable moves of `search_moves`, in engine order); `keys` are their
//! action ids, which break ties (the lower first). Values are the root actor's win probabilities
//! in [0, 1], as the paper assumes.

use crate::noise::open_unit;
use settler_engine::rng::Rng;

/// Candidates sequential halving starts from: m = min(MAX_CANDIDATES, |A|).
pub const MAX_CANDIDATES: usize = 16;
/// σ(q) = (C_VISIT + max_b N(b)) · C_SCALE · q, the paper's values.
pub const C_VISIT: f32 = 50.0;
pub const C_SCALE: f32 = 1.0;
/// Root prior probabilities are floored here before their log: a zero prior gets a finite logit
/// (about -87.3).
pub const MIN_PRIOR: f32 = f32::MIN_POSITIVE;

/// A Gumbel(0, 1) draw: -ln(-ln U), U uniform in (0, 1).
pub fn gumbel(rng: &mut Rng) -> f32 {
    (-(-open_unit(rng).ln()).ln()) as f32
}

/// ℓ = ln π: the root prior renormalised over the root actions (uniform if it sums to nothing or
/// is not finite), each probability floored at `MIN_PRIOR`.
pub fn logits(prior: &[f32]) -> Vec<f32> {
    let total: f32 = prior.iter().sum();
    let uniform = 1.0 / prior.len() as f32;
    prior
        .iter()
        .map(|&p| {
            let p = if total > 0.0 && total.is_finite() {
                p / total
            } else {
                uniform
            };
            p.max(MIN_PRIOR).ln()
        })
        .collect()
}

fn softmax64(x: &[f32]) -> Vec<f64> {
    let top = x.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let e: Vec<f64> = x.iter().map(|&v| (v as f64 - top).exp()).collect();
    let total: f64 = e.iter().sum();
    e.into_iter().map(|v| v / total).collect()
}

/// softmax(x), computed in f64.
pub fn softmax(x: &[f32]) -> Vec<f32> {
    softmax64(x).into_iter().map(|p| p as f32).collect()
}

/// `of` (indices into the root actions) ordered by `score`, the largest first, ties to the lower
/// action id.
fn rank(of: &[usize], score: impl Fn(usize) -> f32, keys: &[u16]) -> Vec<usize> {
    let mut out = of.to_vec();
    out.sort_by(|&a, &b| score(b).total_cmp(&score(a)).then(keys[a].cmp(&keys[b])));
    out
}

/// The candidates: the min(MAX_CANDIDATES, |A|) root actions with the largest g + ℓ (`base`),
/// best first.
pub fn candidates(base: &[f32], keys: &[u16]) -> Vec<usize> {
    let all: Vec<usize> = (0..base.len()).collect();
    let mut c = rank(&all, |i| base[i], keys);
    c.truncate(MAX_CANDIDATES);
    c
}

/// ceil(log2 m) halving phases for m candidates (0 for one).
pub fn phases(m: usize) -> u32 {
    if m <= 1 {
        0
    } else {
        usize::BITS - (m - 1).leading_zeros()
    }
}

/// Simulations each remaining candidate gets in a phase: max(1, floor(n / (phases · remaining))).
/// `phases` and `remaining` must be positive.
pub fn per_candidate(n: u32, phases: u32, remaining: usize) -> u32 {
    (n / (phases * remaining as u32)).max(1)
}

/// σ(q) at a root whose most-visited action has `max_visits` visits.
pub fn sigma(q: f32, max_visits: u32) -> f32 {
    (C_VISIT + max_visits as f32) * C_SCALE * q
}

/// v_mix (appendix eq. 33): the root value `v_hat` mixed with the prior-weighted mean q̂ of the
/// visited actions, π = softmax(ℓ). With nothing visited it is `v_hat`.
pub fn v_mix(logits: &[f32], visits: &[u32], q: &[f32], v_hat: f32) -> f32 {
    let total: u32 = visits.iter().sum();
    if total == 0 {
        return v_hat;
    }
    let pi = softmax64(logits);
    let (mut pi_sum, mut pi_q) = (0.0f64, 0.0f64);
    for ((&p, &n), &q) in pi.iter().zip(visits).zip(q) {
        if n > 0 {
            pi_sum += p;
            pi_q += p * q as f64;
        }
    }
    ((v_hat as f64 + total as f64 / pi_sum * pi_q) / (1.0 + total as f64)) as f32
}

/// completedQ: q̂ for a visited action, v_mix for the rest.
pub fn completed_q(logits: &[f32], visits: &[u32], q: &[f32], v_hat: f32) -> Vec<f32> {
    let mix = v_mix(logits, visits, q, v_hat);
    visits
        .iter()
        .zip(q)
        .map(|(&n, &q)| if n > 0 { q } else { mix })
        .collect()
}

/// `of` ordered by g + ℓ + σ(completedQ), best first, ties to the lower action id: sequential
/// halving keeps the front half, and the move is the first.
pub fn ranked(of: &[usize], base: &[f32], keys: &[u16], cq: &[f32], max_visits: u32) -> Vec<usize> {
    rank(of, |i| base[i] + sigma(cq[i], max_visits), keys)
}

/// The improved policy π' = softmax(ℓ + σ(completedQ)) over every root action.
pub fn improved_policy(logits: &[f32], cq: &[f32], max_visits: u32) -> Vec<f32> {
    let x: Vec<f32> = logits
        .iter()
        .zip(cq)
        .map(|(&l, &q)| l + sigma(q, max_visits))
        .collect();
    softmax(&x)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn halving_takes_ceil_log2_phases() {
        assert_eq!(
            [1, 2, 3, 4, 5, 8, 9, 13, 16].map(phases),
            [0, 1, 2, 2, 3, 3, 4, 4, 4]
        );
    }

    #[test]
    fn every_phase_gives_each_candidate_at_least_one_simulation() {
        assert_eq!(per_candidate(99, 4, 16), 1);
        assert_eq!(per_candidate(99, 4, 8), 3);
        assert_eq!(per_candidate(10, 4, 16), 1);
        assert_eq!(per_candidate(599, 4, 4), 37);
    }

    #[test]
    fn candidates_are_the_largest_scores_ties_to_the_lower_id() {
        assert_eq!(
            candidates(&[0.5, 2.0, 2.0, -1.0], &[40, 30, 10, 20]),
            vec![2, 1, 0, 3]
        );
        let many: Vec<f32> = (0..20).map(|i| i as f32).collect();
        let ids: Vec<u16> = (0..20).collect();
        assert_eq!(
            candidates(&many, &ids),
            (4..20).rev().collect::<Vec<usize>>()
        );
    }

    #[test]
    fn v_mix_is_the_root_value_until_something_is_visited() {
        let l = logits(&[0.5, 0.3, 0.2]);
        assert_eq!(v_mix(&l, &[0, 0, 0], &[0.0; 3], 0.4), 0.4);
        // (0.4 + 3 / 0.8 · (0.5 · 0.6 + 0.3 · 0.3)) / 4 = 0.465625
        let m = v_mix(&l, &[2, 1, 0], &[0.6, 0.3, 0.0], 0.4);
        assert!((m - 0.465625).abs() < 1e-6, "{m}");
        let cq = completed_q(&l, &[2, 1, 0], &[0.6, 0.3, 0.0], 0.4);
        assert_eq!(cq[..2], [0.6, 0.3]);
        assert!((cq[2] - 0.465625).abs() < 1e-6);
    }

    #[test]
    fn the_improved_policy_is_the_prior_with_nothing_visited_and_favours_value() {
        let prior = [0.5, 0.3, 0.2];
        let l = logits(&prior);
        let cq = completed_q(&l, &[0; 3], &[0.0; 3], 0.4);
        for (p, want) in improved_policy(&l, &cq, 0).iter().zip(prior) {
            assert!((p - want).abs() < 1e-6, "{p} vs {want}");
        }
        let cq = completed_q(&l, &[2, 1, 0], &[0.6, 0.3, 0.0], 0.4);
        let p = improved_policy(&l, &cq, 2);
        assert!((p.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert!(p[0] > prior[0] && p[1] < prior[1], "{p:?}");
    }

    #[test]
    fn a_zero_prior_has_a_finite_logit() {
        let l = logits(&[1.0, 0.0, 0.0]);
        assert!(l.iter().all(|x| x.is_finite()), "{l:?}");
        assert_eq!(l[0], 0.0);
        assert_eq!(
            logits(&[0.0, 0.0]),
            logits(&[0.5, 0.5]),
            "nothing to renormalise: uniform"
        );
        let p = improved_policy(&l, &[0.25; 3], 0);
        assert!((p.iter().sum::<f32>() - 1.0).abs() < 1e-6 && p.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn gumbel_draws_have_the_standard_mean_and_variance() {
        let mut rng = Rng::new(3);
        let n = 40_000;
        let xs: Vec<f64> = (0..n).map(|_| gumbel(&mut rng) as f64).collect();
        assert!(xs.iter().all(|x| x.is_finite()));
        let mean = xs.iter().sum::<f64>() / n as f64;
        let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
        // Gumbel(0, 1): mean the Euler–Mascheroni constant 0.5772, variance π²/6 = 1.6449.
        assert!((mean - 0.5772).abs() < 0.03, "{mean}");
        assert!((var - 1.6449).abs() < 0.08, "{var}");
    }
}
```

In `search/src/lib.rs`, after `pub mod evaluator;` add `pub mod gumbel;`, and change the `pub use search::{...}` block to:

```rust
pub use search::{
    search, search_actions, search_moves, search_tree, terminal_value, visible_chance, RootMode,
    SearchConfig, SearchResult,
};
```

- [ ] **Step 5: Rewrite `search/src/search.rs`**

Replace the whole file with:

```rust
//! The simulation loop: single-observer ISMCTS with PUCT, 4-player value vectors, chance nodes
//! for outcomes the searching player sees, and leaf batching with virtual loss (spec Section 3).
//! P2: the root may instead be searched by Gumbel sequential halving (`RootMode::Gumbel`,
//! docs/superpowers/specs/2026-10-08-gumbel-root-design.md Section 3); below the root it is PUCT.

use crate::belief::Belief;
use crate::evaluator::{Evaluator, Leaf};
use crate::gumbel;
use crate::puct;
use crate::tree::{Child, NodeKind, Tree, ROOT};
use crate::world::WorldSampler;
use settler_engine::legal::legal_actions;
use settler_engine::rng::Rng;
use settler_engine::rules::roll::pick_card;
use settler_engine::{Action, Chance, NoEvents, Observation, PlayerId, State, NUM_PLAYERS};

/// How the root's simulations are spent and its move chosen.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RootMode {
    /// PUCT at the root as below it; the most-visited move is played (3a to 3c).
    Puct,
    /// Gumbel sequential halving over the root's candidates (spec P2 Section 3). `draw`: the
    /// Gumbel noise comes from the search's `Rng` (self-play); off, it is zero (play).
    Gumbel { draw: bool },
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SearchConfig {
    /// Simulations per decision: the budget (never wall time).
    pub simulations: u32,
    pub c_puct: f32,
    /// Leaves per `Evaluator` call.
    pub batch: usize,
    /// Domestic offers are searched moves for every seat in the tree: a bot that may offer
    /// assumes its opponents may too. Off, no seat offers in the tree (3a and 3b never offered).
    pub offers: bool,
    /// The root's rule: PUCT unless the bot asks for Gumbel (`+g`, Gumbel self-play).
    pub root: RootMode,
}

impl Default for SearchConfig {
    fn default() -> Self {
        SearchConfig {
            simulations: 1000,
            c_puct: 1.5,
            batch: 8,
            offers: false,
            root: RootMode::Puct,
        }
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
    /// A Gumbel search's improved policy softmax(ℓ + σ(completedQ)), aligned with `legal`,
    /// summing to 1, 0 for a move the search does not consider (an offer without `offers`).
    /// `None` for PUCT and for a decision returned without searching.
    pub improved: Option<Vec<f32>>,
}

/// The actions the search considers: everything legal, with domestic offers only when `offers`.
pub fn search_moves(legal: &[Action], offers: bool, out: &mut Vec<Action>) {
    out.clear();
    out.extend(
        legal
            .iter()
            .copied()
            .filter(|a| offers || !matches!(a, Action::OfferTrade { .. })),
    );
}

/// `search_moves` without offers: what a bot that never offers searches, and what rollouts play.
pub fn search_actions(legal: &[Action], out: &mut Vec<Action>) {
    search_moves(legal, false, out);
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
            s.apply_with(
                a,
                Some(Chance::Roll {
                    dice,
                    discards: None,
                }),
                &mut NoEvents,
            );
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

// A Descent is moved straight into the batch, so boxing the large variant would only add an allocation.
#[allow(clippy::large_enum_variant)]
enum Descent {
    Terminal {
        path: Path,
        value: [f32; NUM_PLAYERS],
    },
    Leaf(Pending),
    /// Reached a node another simulation in this batch is already evaluating.
    Busy,
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
    assert_eq!(
        prior.len(),
        legal.len(),
        "the evaluator's prior must align with the legal actions"
    );
    let n = &mut tree.nodes[node as usize];
    n.expanded = true;
    n.awaiting_eval = false;
    n.children
        .extend(legal.iter().zip(prior).map(|(a, &p)| Child {
            available: 1,
            ..Child::new(a.encode(), p)
        }));
}

/// One search's working state: the tree, the world sampler and the leaf batch.
struct Sim<'a, E> {
    obs: &'a Observation,
    belief: &'a Belief,
    sampler: WorldSampler,
    eval: &'a mut E,
    cfg: &'a SearchConfig,
    rng: &'a mut Rng,
    tree: Tree,
    buf: Vec<Action>,
    scratch: Vec<Action>,
    batch: Vec<Pending>,
    /// Simulations completed (each backed up).
    done: u32,
    /// The root's evaluation: its backup path is empty, so the tree does not keep it.
    root_value: Option<[f32; NUM_PLAYERS]>,
}

impl<E: Evaluator> Sim<'_, E> {
    /// One descent from the root in `world`: to a terminal, to a leaf to evaluate, or `Busy`.
    /// `force` is the first step from the root (a Gumbel phase's candidate); `None`, and every
    /// step below the root, is PUCT.
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
            let legal = &self.scratch;
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
                    legal: legal.clone(),
                });
            }
            // Availability counts every descent through here, including one that then stops at a
            // node another simulation in the batch is evaluating (`Busy`), which is not counted as
            // a simulation: at batch 8 that over-counts the root's simulations by about 0.3%.
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
                    n.children.push(Child {
                        available: 1,
                        ..Child::new(a.encode(), uniform)
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
            let a =
                Action::decode(n.children[i].key).expect("children are keyed by valid action ids");
            path.push((node, i));
            if visible_chance(&world, a, self.obs.viewer) {
                node = self.tree.descend_to(node, i, NodeKind::Chance { action: a });
            } else {
                world.apply(a);
                node = self.tree.descend_to(node, i, NodeKind::Decision);
            }
        }
    }

    /// `count` more simulations, each in a world sampled from the belief, at most `cfg.batch`
    /// leaves per evaluator call. Simulation `k` of this call takes its first step from the root
    /// to the child keyed `step(k)` (`None`: PUCT). A simulation that meets a node another one in
    /// the batch is evaluating (`Busy`) ends the batch and is retried, in a new world, in the next.
    fn run(&mut self, count: usize, step: impl Fn(usize) -> Option<u16>) {
        let mut next = 0;
        while next < count {
            self.batch.clear();
            while self.batch.len() < self.cfg.batch && next < count {
                let world = self.sampler.sample(self.belief, self.rng);
                match self.descend(world, step(next)) {
                    Descent::Terminal { path, value } => {
                        backup(&mut self.tree, &path, &value);
                        self.done += 1;
                        next += 1;
                    }
                    Descent::Leaf(p) => {
                        set_pending(&mut self.tree, &p.path, true);
                        self.batch.push(p);
                        next += 1;
                    }
                    Descent::Busy => break,
                }
            }
            if self.batch.is_empty() {
                continue;
            }
            let leaves: Vec<Leaf<'_>> = self
                .batch
                .iter()
                .map(|p| Leaf {
                    world: &p.world,
                    actor: p.world.current_actor(),
                    legal: &p.legal,
                })
                .collect();
            let evals = self.eval.evaluate(&leaves);
            assert_eq!(
                evals.len(),
                self.batch.len(),
                "the evaluator must return one evaluation per leaf"
            );
            for (p, e) in self.batch.iter().zip(&evals) {
                if p.node == ROOT {
                    self.root_value = Some(e.value);
                }
                expand(&mut self.tree, p.node, &p.legal, &e.prior);
                set_pending(&mut self.tree, &p.path, false);
                backup(&mut self.tree, &p.path, &e.value);
                self.done += 1;
            }
        }
    }
}

/// The most-visited root move among `root_actions` (ties: the higher prior, then the lower id).
/// Root children come from each world's legal moves; only one the caller listed is played.
fn most_visited(tree: &Tree, root_actions: &[Action]) -> Result<Action, String> {
    let searchable = ActionSet::of(root_actions);
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
    Ok(Action::decode(best.key).expect("children are keyed by valid action ids"))
}

/// Visits and mean `actor` value of the root children at `at` (indices into the root's
/// children); the mean is 0 for an unvisited child.
fn root_stats(tree: &Tree, at: &[usize], actor: usize) -> (Vec<u32>, Vec<f32>) {
    let children = &tree.nodes[ROOT as usize].children;
    at.iter()
        .map(|&i| {
            let c = &children[i];
            let q = if c.visits > 0 {
                c.value_sum[actor] / c.visits as f32
            } else {
                0.0
            };
            (c.visits, q)
        })
        .unzip()
}

fn max_visits(visits: &[u32]) -> u32 {
    visits.iter().copied().max().unwrap_or(0)
}

/// Gumbel root search (spec P2 Section 3): the first simulation evaluates the root, then
/// sequential halving over the candidates spends the other `simulations - 1`, each forced to its
/// candidate at the root and PUCT below. Returns the move and the improved policy over
/// `root_actions` (aligned with it).
fn gumbel_root<E: Evaluator>(
    sim: &mut Sim<'_, E>,
    root_actions: &[Action],
    draw: bool,
) -> (Action, Vec<f32>) {
    // A single simulation always ends at the unexpanded root: it evaluates the root alone.
    sim.run(1, |_| None);
    let actor = sim.obs.viewer as usize;
    let v_hat = sim.root_value.expect("the first simulation evaluates the root")[actor];
    let keys: Vec<u16> = root_actions.iter().map(|a| a.encode()).collect();
    let (at, prior): (Vec<usize>, Vec<f32>) = {
        let children = &sim.tree.nodes[ROOT as usize].children;
        keys.iter()
            .map(|&k| {
                let i = children
                    .iter()
                    .position(|c| c.key == k)
                    .unwrap_or_else(|| {
                        panic!(
                            "the root move {:?} is not legal in the first sampled world",
                            Action::decode(k)
                        )
                    });
                (i, children[i].prior)
            })
            .unzip()
    };
    let logits = gumbel::logits(&prior);
    // One draw per root action, in root-action order, right after the root's evaluation.
    let base: Vec<f32> = logits
        .iter()
        .map(|&l| if draw { l + gumbel::gumbel(sim.rng) } else { l })
        .collect();
    let mut remaining = gumbel::candidates(&base, &keys);
    let n = sim.cfg.simulations - 1;
    let phases = gumbel::phases(remaining.len());
    let mut left = n;
    for phase in 0..phases {
        // The last phase, or a phase with one candidate left, spends the rest round-robin.
        let last = phase + 1 == phases || remaining.len() == 1;
        let count = if last {
            left
        } else {
            (gumbel::per_candidate(n, phases, remaining.len()) * remaining.len() as u32).min(left)
        };
        let order = &remaining;
        sim.run(count as usize, |k| Some(keys[order[k % order.len()]]));
        left -= count;
        if last {
            break;
        }
        let (visits, q) = root_stats(&sim.tree, &at, actor);
        let cq = gumbel::completed_q(&logits, &visits, &q, v_hat);
        let mut kept = gumbel::ranked(&remaining, &base, &keys, &cq, max_visits(&visits));
        kept.truncate(remaining.len().div_ceil(2));
        remaining = kept;
    }
    let (visits, q) = root_stats(&sim.tree, &at, actor);
    let cq = gumbel::completed_q(&logits, &visits, &q, v_hat);
    let most = max_visits(&visits);
    let best = gumbel::ranked(&remaining, &base, &keys, &cq, most)[0];
    (
        root_actions[best],
        gumbel::improved_policy(&logits, &cq, most),
    )
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
    assert!(
        cfg.simulations > 0 && cfg.batch > 0,
        "simulations and batch must be positive"
    );
    let mut root_actions = Vec::new();
    search_moves(legal, cfg.offers, &mut root_actions);
    match root_actions.len() {
        0 => return Err("nothing to search: no legal action".into()),
        1 => {
            let r = SearchResult {
                action: root_actions[0],
                visits: vec![0; legal.len()],
                simulations: 0,
                nodes: 0,
                improved: None,
            };
            return Ok((r, Tree::new()));
        }
        _ => {}
    }
    let mut sim = Sim {
        obs,
        belief,
        sampler: WorldSampler::new(obs, belief)?,
        eval,
        cfg,
        rng,
        tree: Tree::new(),
        buf: Vec::new(),
        scratch: Vec::new(),
        batch: Vec::with_capacity(cfg.batch),
        done: 0,
        root_value: None,
    };
    let (action, improved) = match cfg.root {
        RootMode::Puct => {
            sim.run(cfg.simulations as usize, |_| None);
            (most_visited(&sim.tree, &root_actions)?, None)
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
            (action, Some(improved))
        }
    };
    let root = &sim.tree.nodes[ROOT as usize];
    let visits = legal
        .iter()
        .map(|a| {
            root.children
                .iter()
                .find(|c| c.key == a.encode())
                .map_or(0, |c| c.visits)
        })
        .collect();
    let (simulations, nodes) = (sim.done, sim.tree.nodes.len());
    Ok((
        SearchResult {
            action,
            visits,
            simulations,
            nodes,
            improved,
        },
        sim.tree,
    ))
}

/// The move for `obs.viewer` among `legal` after `cfg.simulations` simulations, each in a world
/// sampled from `belief`: the most-visited (PUCT root) or sequential halving's winner (Gumbel
/// root). A single searchable move is returned without searching.
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

- [ ] **Step 6: Run the tests to verify they pass**

Run: `PATH=$HOME/.cargo/bin:$PATH cargo test --release -p settler-search`
Expected: PASS: 11 tests in `gumbel`, the 7 unit tests in `gumbel::tests`, every existing search test.

Run: `PATH=$HOME/.cargo/bin:$PATH cargo test --release -p settler-bots --test ismcts --test netbot --test selfplay`
Expected: PASS, including the three `before_p2` pins and `the_generic_refactor_leaves_3a_ismcts_games_unchanged` and `fixed_exploration_is_3a_self_play`: PUCT plays exactly as before.

- [ ] **Step 7: Format, lint and the full suite**

Run, in order: `PATH=$HOME/.cargo/bin:$PATH cargo fmt --all`, `PATH=$HOME/.cargo/bin:$PATH cargo fmt --check` (no output), `PATH=$HOME/.cargo/bin:$PATH cargo clippy --workspace --release --all-targets -- -D warnings` (exit 0), `PATH=$HOME/.cargo/bin:$PATH cargo test --workspace --release` (all pass). No Python-visible change: no rebuild in this task.

- [ ] **Step 8: Commit**

```bash
git add search/src/gumbel.rs search/src/search.rs search/src/noise.rs search/src/lib.rs search/tests/gumbel.rs \
  bots/tests/ismcts.rs bots/tests/netbot.rs bots/tests/selfplay.rs
git diff --cached | grep -nE 'sk-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{36}|password=[^[:space:]]+'
git -c user.email=sean.may101@gmail.com commit -m "search: Gumbel root search (sequential halving, completed-Q improved policy) as a root mode; PUCT games unchanged"
```

---

### Task 2: The `+g` suffix

**Files:**
- Modify: `bots/src/ismcts.rs` (`split_gumbel`, `with_gumbel`, the `RootMode` import), `bots/src/lib.rs` (`make_bot`, `Roster::new`, `Roster::make`, `unknown_bot`), `bots/src/netbot.rs` (module doc), `alphasettler/cli.py` (`_check_bot_name` text), `tests/python/net_names.py`
- Test: `bots/tests/ismcts.rs`, `bots/tests/netbot.rs`, `tests/python/test_netbot.py`

**Interfaces:**
- Consumes: Task 1's `settler_search::RootMode`, `SearchResult::improved`, `SearchConfig::root`.
- Produces:
  - `settler_bots::ismcts::split_gumbel(name: &str) -> Option<(String, bool)>`: the name without `+g` and whether it had one; `None` for a repeated `+g`. Any other `g` part (`+g2`, `+gt`, `+G`) is left in for `split_trades`, which refuses it.
  - `IsmctsBot::with_gumbel(self, gumbel: bool) -> IsmctsBot<E>`: `true` sets `cfg.root = RootMode::Gumbel { draw: false }`, `false` sets `RootMode::Puct`.
  - `make_bot`, `Roster::new` and `Roster::make` accept `+g` on `ismcts` and `net:` names. `split_trades`, `split_setup`, `parse_name` and `parse_net_name` keep their signatures and behaviour.

- [ ] **Step 1: Write the failing tests**

In `bots/tests/ismcts.rs`, after `the_offer_suffix_combines_with_the_others_in_any_order`, add:

```rust
#[test]
fn the_gumbel_suffix_parses_in_any_order_and_only_once() {
    use settler_bots::ismcts::{split_gumbel, split_trades};
    assert_eq!(
        split_gumbel("ismcts@100"),
        Some(("ismcts@100".to_owned(), false))
    );
    assert_eq!(
        split_gumbel("ismcts@100+g"),
        Some(("ismcts@100".to_owned(), true))
    );
    assert_eq!(
        split_gumbel("net:gen-034@32+g"),
        Some(("net:gen-034@32".to_owned(), true))
    );
    for name in [
        "ismcts@100+g+t+r5+s3",
        "ismcts@100+t+g+r5+s3",
        "ismcts@100+r5+s3+t+g",
        "ismcts@100+s3+g+r5+t",
    ] {
        let (rest, gumbel) = split_gumbel(name).unwrap();
        assert!(gumbel, "{name}");
        assert_eq!(
            split_trades(&rest),
            Some(("ismcts@100+r5+s3".to_owned(), true)),
            "{name}"
        );
        assert!(make_bot(name, 0).is_some(), "{name}");
    }
    assert_eq!(split_gumbel("ismcts@100+g+g"), None);
    for bad in [
        "ismcts@100+g+g",
        "ismcts@100+g2",
        "ismcts@100+G",
        "ismcts@100+gt",
        "ismcts@100+g+",
        "ismcts+g@100",
        "greedy+g",
        "random+g",
        "+g",
    ] {
        assert!(make_bot(bad, 0).is_none(), "{bad}");
    }
    assert!(make_bot("ismcts+g", 0).is_some());
}

#[test]
fn a_gumbel_name_builds_the_gumbel_bot_and_plain_names_stay_puct() {
    use settler_search::RootMode;
    assert_eq!(IsmctsBot::new(0, 10, 0).config().root, RootMode::Puct);
    assert_eq!(
        IsmctsBot::new(0, 10, 0).with_gumbel(true).config().root,
        RootMode::Gumbel { draw: false }
    );
    let mut differ = 0;
    for seed in 0..12 {
        let s = main_with(seed, [3, 3, 3, 3, 3], no_trades());
        let (obs, legal) = (s.observation(0), s.legal_actions());
        let named = make_bot("ismcts@40+g", 7).unwrap().act(&obs, &legal);
        let direct = IsmctsBot::new(7, 40, 0)
            .with_gumbel(true)
            .act(&obs, &legal);
        let plain_named = make_bot("ismcts@40", 7).unwrap().act(&obs, &legal);
        let puct = IsmctsBot::new(7, 40, 0).act(&obs, &legal);
        assert_eq!(named, direct, "seed {seed}");
        assert_eq!(plain_named, puct, "seed {seed}");
        differ += (direct != puct) as u32;
    }
    assert!(
        differ > 0,
        "the Gumbel root chose as PUCT did on every position"
    );
}

#[test]
fn a_gumbel_bot_keeps_its_setup_factor() {
    let mut g = Game::new(3, no_trades());
    let mut bot = IsmctsBot::new(3, 20, 0)
        .with_setup_factor(5)
        .with_gumbel(true);
    assert_eq!(g.observation(0).phase, Phase::SetupSettlement);
    act_and_apply(&mut bot, &mut g);
    let r = bot.take_last_search().unwrap();
    assert_eq!(r.simulations, 100);
    assert!(
        r.improved.is_some(),
        "the setup settlement was a Gumbel search"
    );
}
```

In `bots/tests/netbot.rs`, in `net_names_parse_and_bad_ones_are_rejected_without_touching_the_disk`, add these entries at the end of the `for bad in [...]` list (after `"net:a@300+T",`):

```rust
        "net:a@300+g+g",
        "net:a+g@300",
        "net:a@300+g2",
        "net:a@300+G",
```

and after `net_bots_search_with_the_given_leaf_batch` add:

```rust
#[test]
fn gumbel_net_names_build_the_gumbel_bot_in_any_order() {
    let dir = nets_dir("gumbel");
    let names = ["net:tiny@16+g", "net:tiny@16+t+g", "net:tiny@16+g+s2+t"];
    let roster = Roster::new(&names, &dir).unwrap();
    for name in names {
        assert_eq!(roster.make(name, 0).unwrap().name(), "net", "{name}");
    }
    let w = load_net(&dir, "tiny").unwrap();
    let mut s = State::new(5, trading());
    let mut rng = settler_engine::rng::Rng::new(5);
    let mut compared = 0;
    for i in 0..400 {
        if s.is_over() {
            break;
        }
        let legal = s.legal_actions();
        if i % 9 == 0 && legal.len() > 1 {
            let obs = s.observation(s.current_actor());
            let named = roster.make("net:tiny@16+t+g", 3).unwrap().act(&obs, &legal);
            let direct = net_bot(3, w.clone(), 16, 0)
                .with_offers(true)
                .with_gumbel(true)
                .act(&obs, &legal);
            assert_eq!(named, direct, "move {i}");
            compared += 1;
        }
        s.apply(legal[rng.below(legal.len() as u32) as usize]);
    }
    assert!(compared >= 10, "only {compared} positions compared");
}
```

In `tests/python/net_names.py`, add to `BAD_NET_NAMES` (after `"net:a+t@300",`):

```python
    "net:a@300+g+g",
    "net:a+g@300",
    "net:a@300+g2",
```

In `tests/python/test_netbot.py`, after `test_offer_suffixes_build_net_bots_in_any_order`, add:

```python
@pytest.mark.parametrize("name", ["net:tiny@10+g", "net:tiny@10+t+g+s2", "net:tiny@10+g+r2", "ismcts@10+g"])
def test_the_gumbel_suffix_builds_bots_that_play_legal_moves(nets, name):
    g = Game(4, CONFIG)
    bot = Bot(name, 1, nets_dir=str(nets))
    for _ in range(40):
        if g.is_over:
            break
        a = bot.act(g)
        assert a in g.legal_actions()
        g.apply(a)


@pytest.mark.parametrize("bad", ["net:tiny@10+g+g", "net:tiny+g@10", "net:tiny@10+G", "ismcts@10+gg", "greedy+g"])
def test_malformed_gumbel_names_are_unknown_bots(nets, bad):
    with pytest.raises(ValueError, match="unknown bot"):
        Bot(bad, 0, nets_dir=str(nets))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PATH=$HOME/.cargo/bin:$PATH cargo test --release -p settler-bots --test ismcts --test netbot`
Expected: FAIL to compile with `no function or associated item named split_gumbel` / `no method named with_gumbel`.

- [ ] **Step 3: Write the implementation**

In `bots/src/ismcts.rs`, change the `use settler_search::{...}` block to:

```rust
use settler_search::{
    search, search_moves, Belief, Evaluator, Noised, RootMode, RootNoise, SearchConfig,
    SearchResult, DEFAULT_MAX_STATES,
};
```

and after `split_trades` add:

```rust
/// A searching bot's name with its `+g` suffix (spec P2 Section 4: Gumbel root search, no
/// draws) removed, and whether it had one. `+g` may sit anywhere among the suffixes, at most once
/// (`None` for a repeat); every other part is kept in order for `split_trades`, which refuses any
/// other `g` part (`+g2`, `+gt`) and anything malformed.
pub fn split_gumbel(name: &str) -> Option<(String, bool)> {
    let Some(i) = name.find('+') else {
        return Some((name.to_owned(), false));
    };
    let (mut out, mut gumbel) = (name[..i].to_owned(), false);
    for part in name[i + 1..].split('+') {
        if part == "g" {
            if gumbel {
                return None;
            }
            gumbel = true;
        } else {
            out.push('+');
            out.push_str(part);
        }
    }
    Some((out, gumbel))
}
```

and after `with_offers` add:

```rust
    /// Search the root by Gumbel sequential halving with no Gumbel draws (a `+g` name, spec P2
    /// Section 4); below the root the search is PUCT as before. Off, the root is PUCT.
    pub fn with_gumbel(mut self, gumbel: bool) -> IsmctsBot<E> {
        self.cfg.root = if gumbel {
            RootMode::Gumbel { draw: false }
        } else {
            RootMode::Puct
        };
        self
    }
```

In `bots/src/lib.rs`, replace the doc comment and body of `make_bot` with:

```rust
/// A fresh bot by name, with its own random stream seeded from `seed`. Besides `BOT_NAMES`,
/// `ismcts@N` (N simulations) and `ismcts@N+rD` (plus D greedy rollout moves per leaf), either
/// with `+sK` to search setup settlements with K times the simulations, `+t` to make domestic
/// offers and `+g` to search the root by Gumbel sequential halving, the suffixes in any order.
/// `net:` names need their weights loaded first: use a `Roster`.
pub fn make_bot(name: &str, seed: u64) -> Option<Box<dyn Bot>> {
    match name {
        "random" => Some(Box::new(random::RandomBot::new(seed))),
        "greedy" => Some(Box::new(greedy::GreedyBot::new(seed))),
        _ => {
            let (name, gumbel) = ismcts::split_gumbel(name)?;
            let (name, offers) = ismcts::split_trades(&name)?;
            let (base, factor) = ismcts::split_setup(&name)?;
            let (sims, rollout) = ismcts::parse_name(base)?;
            Some(Box::new(
                ismcts::IsmctsBot::new(seed, sims, rollout)
                    .with_setup_factor(factor)
                    .with_offers(offers)
                    .with_gumbel(gumbel),
            ))
        }
    }
}
```

Replace `unknown_bot`'s `format!` with:

```rust
    format!(
        "unknown bot {name:?}; known bots: {BOT_NAMES:?} (ismcts@N for N simulations, ismcts@N+rD \
         to add D greedy rollout moves, net:NAME[@N[+rD]] for <nets-dir>/NAME.bin; +sK on either \
         searches setup settlements with K times the simulations, +t lets either make domestic \
         offers, +g makes either search the root by Gumbel sequential halving)"
    )
```

Replace the bodies of `Roster::new` and `Roster::make` with:

```rust
    pub fn new(names: &[&str], nets_dir: &Path) -> Result<Roster, String> {
        let mut nets: Vec<(String, Arc<Weights>)> = Vec::new();
        for &name in names {
            // `+g` changes how a bot searches, not which net it loads.
            let base = ismcts::split_gumbel(name).map(|(b, _)| b);
            if let Some((net, ..)) = base.as_deref().and_then(netbot::parse_net_name) {
                if !nets.iter().any(|(n, _)| n == net) {
                    nets.push((net.to_owned(), netbot::load_net(nets_dir, net)?));
                }
            } else if make_bot(name, 0).is_none() {
                return Err(unknown_bot(name));
            }
        }
        Ok(Roster { nets })
    }

    /// A fresh bot; `None` for an unknown name or a `net:` name whose network this roster did not
    /// load.
    pub fn make(&self, name: &str, seed: u64) -> Option<Box<dyn Bot>> {
        let (base, gumbel) = ismcts::split_gumbel(name)?;
        match netbot::parse_net_name(&base) {
            Some((net, simulations, rollout, factor, offers)) => {
                let (_, w) = self.nets.iter().find(|(n, _)| n == net)?;
                Some(Box::new(
                    netbot::net_bot(seed, w.clone(), simulations, rollout)
                        .with_setup_factor(factor)
                        .with_offers(offers)
                        .with_gumbel(gumbel),
                ))
            }
            None => make_bot(name, seed),
        }
    }
```

(Keep `Roster::new`'s existing doc comment above it.)

In `bots/src/netbot.rs`, replace the module doc's first two lines with:

```rust
//! Network bots (spec Section 3): `net:NAME` (1,000 simulations), `net:NAME@N` and
//! `net:NAME@N+rD`, each optionally with `+sK`, `+t` and `+g` in any order (`+g` is split off by
//! `ismcts::split_gumbel` before `parse_net_name` reads the rest), with the same limits on N,
```

(the lines after, from `//! D and K as ismcts: ...`, stay).

In `alphasettler/cli.py`, in `_check_bot_name`, replace `either with +sK and +t)` with `either with +sK, +t and +g)`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `PATH=$HOME/.cargo/bin:$PATH cargo test --release -p settler-bots --test ismcts --test netbot`
Expected: PASS, including the 3 new tests in `ismcts`, `gumbel_net_names_build_the_gumbel_bot_in_any_order`, the extended bad-name list and both `before_p2` pins.

- [ ] **Step 5: Format, lint, the full suite, rebuild, pytest**

Run the Global Constraints sequence: `cargo fmt --all`, `cargo fmt --check`, `cargo clippy --workspace --release --all-targets -- -D warnings`, `cargo test --workspace --release`, rebuild the extension, `.venv/bin/pytest -q -rs`. Expected: all pass; the 9 new parametrized cases in `test_netbot.py` pass, and every test using `BAD_NET_NAMES` refuses the three new names.

- [ ] **Step 6: Commit**

```bash
git add bots/src/ismcts.rs bots/src/lib.rs bots/src/netbot.rs alphasettler/cli.py bots/tests/ismcts.rs bots/tests/netbot.rs \
  tests/python/net_names.py tests/python/test_netbot.py
git diff --cached | grep -nE 'sk-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{36}|password=[^[:space:]]+'
git -c user.email=sean.may101@gmail.com commit -m "bots: +g names search the root by Gumbel sequential halving (no draws), in any order with +t, +sK and +rD"
```

---

### Task 3: Gumbel self-play records the improved policy

**Files:**
- Modify: `bots/src/selfplay.rs` (`Exploration`, `play_with`, `Decision`), `bindings/src/lib.rs` (`game_dict`, `selfplay_explore`), `alphasettler/selfplay.py` (`check_gumbel`, `settings`, `write_explore`)
- Test: `bots/tests/selfplay.rs`, `tests/python/test_training_data.py`

**Interfaces:**
- Consumes: Task 1's `RootMode::Gumbel { draw: true }`, `SearchResult::improved`.
- Produces:
  - `settler_bots::selfplay::Exploration` gains `pub gumbel: bool`; `Exploration::gumbel(simulations: u32) -> Exploration` (full = fast = `simulations`, p 1.0, no noise, `sample_turns` 0, `gumbel` true); `fixed` and `standard` set `gumbel: false`; `validate` refuses `gumbel` with noise, sampled turns, p ≠ 1 or fast ≠ full.
  - `Decision` gains `pub target: Option<Vec<f32>>` (`Some` exactly for Gumbel self-play's searched decisions).
  - Record dicts: a decision with a target has `"target": [float, ...]` (aligned with `"legal"`); others have no `target` key.
  - `_engine.selfplay_explore(seed_start, games, threads, full_simulations, fast_simulations, full_probability, config=None, net=None, nets_dir=None, gumbel=None)`.
  - `alphasettler.selfplay.check_gumbel(simulations: int) -> None` (ValueError `gumbel simulations must be 2..=1000000, got ...`), `settings(net, nets_dir, full_simulations, fast_simulations, full_probability, gumbel=None) -> dict`, `write_explore(..., gumbel: int | None = None)`.

- [ ] **Step 1: Write the failing tests**

In `bots/tests/selfplay.rs`, after `standard_exploration_plays_the_games_it_played_before_p2`, add:

```rust
#[test]
fn gumbel_self_play_records_an_improved_policy_on_every_searched_decision() {
    let ex = Exploration::gumbel(8);
    assert!(ex.validate().is_ok());
    for (seed, config) in [(0, no_trades()), (1, no_trades()), (2, trading())] {
        let g = play_with(seed, config, &ex, 8, &heuristic);
        assert_eq!(g.belief_resets, 0, "seed {seed}");
        assert!(!g.decisions.is_empty());
        let mut s = State::new(seed, config);
        let mut decisions = g.decisions.iter().peekable();
        for (i, &a) in g.actions.iter().enumerate() {
            if let Some(d) = decisions.next_if(|d| d.index as usize == i) {
                assert_eq!(d.legal, s.legal_actions());
                assert!(d.full_search, "seed {seed} move {i}");
                assert_eq!(d.visits.iter().sum::<u32>(), 7, "seed {seed} move {i}");
                let t = d
                    .target
                    .as_ref()
                    .expect("a Gumbel search records its improved policy");
                assert_eq!(t.len(), d.legal.len());
                assert!(t.iter().all(|x| x.is_finite() && *x >= 0.0));
                let total: f64 = t.iter().map(|&x| x as f64).sum();
                assert!((total - 1.0).abs() < 1e-4, "seed {seed} move {i}: {total}");
            }
            s.apply(a);
        }
    }
}

#[test]
fn gumbel_self_play_is_deterministic_and_thread_independent() {
    let ex = Exploration::gumbel(6);
    let a = run_with(0..4, no_trades(), &ex, 8, &heuristic, 1).unwrap();
    assert_eq!(a, run_with(0..4, no_trades(), &ex, 8, &heuristic, 3).unwrap());
    assert_ne!(
        a[0].actions,
        play_with(0, no_trades(), &Exploration::fixed(6), 8, &heuristic).actions,
        "the Gumbel root plays other games"
    );
}

#[test]
fn gumbel_self_play_refuses_noise_sampling_and_fast_searches() {
    let ex = Exploration::gumbel(100);
    assert_eq!(
        (
            ex.full_simulations,
            ex.fast_simulations,
            ex.full_probability,
            ex.noise,
            ex.sample_turns,
            ex.gumbel
        ),
        (100, 100, 1.0, None, 0, true)
    );
    assert!(!Exploration::standard().gumbel && !Exploration::fixed(5).gumbel);
    for bad in [
        Exploration {
            noise: Some(NOISE),
            ..ex
        },
        Exploration {
            sample_turns: 1,
            ..ex
        },
        Exploration {
            full_probability: 0.25,
            ..ex
        },
        Exploration {
            fast_simulations: 50,
            ..ex
        },
        Exploration {
            gumbel: true,
            ..Exploration::standard()
        },
        Exploration::gumbel(1),
    ] {
        assert!(bad.validate().is_err(), "{bad:?}");
        assert!(run_with(0..1, no_trades(), &bad, 8, &heuristic, 1).is_err());
    }
}

#[test]
fn puct_self_play_records_no_target() {
    let g = play_with(2, trading(), &small_explore(), 8, &heuristic);
    assert!(g.decisions.iter().all(|d| d.target.is_none()));
}
```

In `tests/python/test_training_data.py`, after `CONFIG = {"max_offers_per_turn": 0}` add `TRADING = {"max_offers_per_turn": 2, "max_trade_cards": 1}`, and after `test_record_files_are_byte_identical_for_the_same_games` add:

```python
@pytest.mark.parametrize("config", [CONFIG, TRADING], ids=["no-trades", "trading"])
def test_gumbel_self_play_records_an_improved_policy_on_every_decision(tmp_path, config):
    games = explore_games(tmp_path, 2, config=config, gumbel=6)
    for g in games:
        assert g["selfplay"] == {"bot": "ismcts", "net": None, "net_sha256": None, "gumbel_simulations": 6,
                                 "batch": SEARCH_BATCH}
        assert g["belief_resets"] == 0 and records.replay_mismatches(g) == []
        assert g["decisions"]
        for d in g["decisions"]:
            assert d["full_search"] is True and sum(d["visits"]) == 5
            assert len(d["target"]) == len(d["legal"]) and min(d["target"]) >= 0
            assert sum(d["target"]) == pytest.approx(1.0, abs=1e-4)


def test_plain_self_play_records_carry_no_target(tmp_path):
    assert all("target" not in d for g in explore_games(tmp_path, 2) for d in g["decisions"])


def test_gumbel_self_play_needs_at_least_two_simulations(tmp_path):
    with pytest.raises(ValueError, match="gumbel simulations"):
        write_explore(tmp_path / "no.jsonl.gz", 0, 1, config=CONFIG, full_simulations=6, fast_simulations=3,
                      full_probability=0.25, gumbel=1)
    assert not (tmp_path / "no.jsonl.gz").exists()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PATH=$HOME/.cargo/bin:$PATH cargo test --release -p settler-bots --test selfplay`
Expected: FAIL to compile with `no function or associated item named gumbel` / `no field target`.

- [ ] **Step 3: Write the Rust implementation**

In `bots/src/selfplay.rs`, change `use settler_search::{unit, Evaluator, RootNoise, SearchConfig};` to `use settler_search::{unit, Evaluator, RootMode, RootNoise, SearchConfig};`, and update the module doc's last paragraph by appending:

```rust
//! P2: Gumbel self-play (`Exploration::gumbel`) searches every move with the Gumbel root and
//! records its improved policy as the decision's `target`.
```

In `Decision`, after `full_search`, add:

```rust
    /// Gumbel self-play: the search's improved policy, aligned with `legal` and summing to 1
    /// (spec P2 Section 5); `None` otherwise.
    pub target: Option<Vec<f32>>,
```

In `Exploration`, after `sample_turns`, add:

```rust
    /// Gumbel root search on every move (spec P2 Section 5), with Gumbel draws, each a full
    /// search of `full_simulations` recorded with its improved policy. Noise, playout cap
    /// randomization and visit sampling must be off: the draw is the exploration (`validate`).
    pub gumbel: bool,
```

Add `gumbel: false,` to the struct literals in `fixed` and `standard`, and after `standard` add:

```rust
    /// P2: every move a Gumbel search of `simulations` with Gumbel draws; every search full, no
    /// noise, no sampled turns.
    pub fn gumbel(simulations: u32) -> Exploration {
        Exploration {
            full_simulations: simulations,
            fast_simulations: simulations,
            full_probability: 1.0,
            noise: None,
            sample_turns: 0,
            gumbel: true,
        }
    }
```

In `validate`, before the final `Ok(())`, add:

```rust
        if self.gumbel
            && (self.noise.is_some()
                || self.sample_turns > 0
                || self.full_probability != 1.0
                || self.fast_simulations != self.full_simulations)
        {
            return Err("Gumbel self-play searches every move in full, with no root noise and no \
                        visit sampling: the Gumbel draw is the exploration"
                .into());
        }
```

In `play_with`, replace the `let cfg = SearchConfig { ... };` with:

```rust
    // Every seat searches offers exactly when the game allows them (spec 3c Section 5).
    let cfg = SearchConfig {
        simulations: ex.full_simulations,
        batch,
        offers: config.max_offers_per_turn > 0,
        // Gumbel self-play (spec P2 Section 5): every search draws its own Gumbel noise.
        root: if ex.gumbel {
            RootMode::Gumbel { draw: true }
        } else {
            RootMode::Puct
        },
        ..SearchConfig::default()
    };
```

and in the `decisions.push(Decision { ... })`, after `full_search: full,` add `target: r.improved,`.

In `bindings/src/lib.rs`, in `game_dict`, after `e.set_item("full_search", dec.full_search)?;` add:

```rust
        if let Some(t) = &dec.target {
            e.set_item("target", t.clone())?;
        }
```

Replace `selfplay_explore`'s doc comment, signature attribute, parameter list and `let ex = ...;` with:

```rust
/// 3b self-play (spec Section 4): playout cap randomization (`full_simulations` with
/// probability `full_probability`, else `fast_simulations`), root noise on full searches and
/// opening sampling. With `gumbel` (a simulation count), every move is instead a Gumbel root
/// search of that many simulations, recorded with its improved policy as `target` (spec P2
/// Section 5); the three PUCT arguments are then not used. With `net`, every seat is the network
/// `<nets_dir>/<net>.bin` (loaded once); otherwise the heuristic. Decisions carry no
/// observations. Releases the GIL while games run.
#[pyfunction]
#[pyo3(signature = (seed_start, games, threads, full_simulations, fast_simulations, full_probability, config=None, net=None, nets_dir=None, gumbel=None))]
#[allow(clippy::too_many_arguments)]
fn selfplay_explore<'py>(
    py: Python<'py>,
    seed_start: &Bound<'py, PyAny>,
    games: &Bound<'py, PyAny>,
    threads: &Bound<'py, PyAny>,
    full_simulations: &Bound<'py, PyAny>,
    fast_simulations: &Bound<'py, PyAny>,
    full_probability: f64,
    config: Option<&Bound<'py, PyDict>>,
    net: Option<&str>,
    nets_dir: Option<PathBuf>,
    gumbel: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyList>> {
    use settler_bots::heuristic::{HeuristicEvaluator, DEFAULT_WEIGHTS};
    use settler_bots::netbot::{load_net, DEFAULT_NETS_DIR, NET_BATCH};
    use settler_bots::selfplay::{run_with, Exploration};
    use settler_net::evaluator::NetEvaluator;
    let seed_start: u64 = int_arg(seed_start, "seed_start")?;
    let games: u64 = int_arg(games, "games")?;
    let threads: usize = int_arg(threads, "threads")?;
    let ex = match gumbel {
        Some(n) => Exploration::gumbel(int_arg(n, "gumbel")?),
        None => Exploration {
            full_simulations: int_arg(full_simulations, "full_simulations")?,
            fast_simulations: int_arg(fast_simulations, "fast_simulations")?,
            full_probability,
            ..Exploration::standard()
        },
    };
```

(The rest of the function, from `ex.validate().map_err(value_error)?;`, is unchanged.)

- [ ] **Step 4: Write the Python implementation**

In `alphasettler/selfplay.py`, replace the module docstring's last sentence `Decisions carry no observations; the seed and actions reproduce them."""` with:

```python
Decisions carry no observations; the seed and actions reproduce them. With `gumbel` (P2), every
move is a Gumbel root search and each decision carries its improved policy as `target`; the block
then holds `gumbel_simulations` instead of the exploration settings."""
```

Replace `settings` with:

```python
def check_gumbel(simulations: int) -> None:
    """ValueError unless `simulations` is a budget Gumbel self-play accepts."""
    if not (isinstance(simulations, int) and SELFPLAY_MIN_SIMULATIONS <= simulations <= MAX_SIMULATIONS):
        raise ValueError(f"gumbel simulations must be {SELFPLAY_MIN_SIMULATIONS}..={MAX_SIMULATIONS}, "
                         f"got {simulations!r}")


def settings(net: str | None, nets_dir: str | Path, full_simulations: int, fast_simulations: int,
             full_probability: float, gumbel: int | None = None) -> dict:
    """The `selfplay` block. Checks every argument, and loads the net, without playing a game.
    With `gumbel`, the PUCT arguments are not used."""
    if gumbel is not None:
        check_gumbel(gumbel)
    else:
        for what, n in (("full", full_simulations), ("fast", fast_simulations)):
            if not SELFPLAY_MIN_SIMULATIONS <= n <= MAX_SIMULATIONS:
                raise ValueError(f"{what} simulations must be {SELFPLAY_MIN_SIMULATIONS}..={MAX_SIMULATIONS}, got {n}")
        if not (math.isfinite(full_probability) and 0 <= full_probability <= 1):
            raise ValueError(f"the full-search probability must be in [0, 1], got {full_probability}")
    path = None if net is None else resolve_net(net, str(nets_dir))
    who = {
        "bot": "ismcts" if net is None else f"net:{net}",
        "net": net,
        "net_sha256": None if path is None else files.sha256_file(path),
    }
    batch = SEARCH_BATCH if net is None else NET_BATCH
    if gumbel is not None:
        return {**who, "gumbel_simulations": gumbel, "batch": batch}
    return {
        **who,
        "full_simulations": full_simulations,
        "fast_simulations": fast_simulations,
        "full_probability": full_probability,
        "noise_weight": NOISE_WEIGHT,
        "noise_alpha": NOISE_ALPHA,
        "sample_turns": SAMPLE_TURNS,
        "batch": batch,
    }
```

In `write_explore`, change the signature's last line `counts: dict | None = None) -> dict:` to `counts: dict | None = None, gumbel: int | None = None) -> dict:`, the `meta = ...` line to `meta = settings(net, nets_dir, full_simulations, fast_simulations, full_probability, gumbel)`, and the native call to:

```python
            batch = selfplay_explore(lo, min(BATCH_GAMES, end - lo), threads, full_simulations, fast_simulations,
                                     full_probability, config, net, str(nets_dir), gumbel)
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `PATH=$HOME/.cargo/bin:$PATH cargo test --release -p settler-bots --test selfplay`
Expected: PASS, including the 4 new tests, `standard_exploration_plays_the_games_it_played_before_p2` and `fixed_exploration_is_3a_self_play`.

Rebuild the extension, then run: `.venv/bin/pytest -q tests/python/test_training_data.py`
Expected: PASS, including the 4 new cases; `test_explore_records_carry_their_settings_and_no_observations` passes unchanged.

- [ ] **Step 6: Format, lint, the full suite, rebuild, pytest**

Run the Global Constraints sequence (fmt, fmt --check, clippy, `cargo test --workspace --release`, rebuild the extension, `.venv/bin/pytest -q -rs`). Expected: all pass, nothing skipped.

- [ ] **Step 7: Commit**

```bash
git add bots/src/selfplay.rs bindings/src/lib.rs alphasettler/selfplay.py bots/tests/selfplay.rs tests/python/test_training_data.py
git diff --cached | grep -nE 'sk-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{36}|password=[^[:space:]]+'
git -c user.email=sean.may101@gmail.com commit -m "selfplay: Gumbel self-play (every move a Gumbel search with draws) records each decision's improved policy as its target"
```

---

### Task 4: Training data trains on `target`

**Files:**
- Modify: `net/src/data.rs` (`RecordedDecision`, `Batch` docs, `Batch::push`, `check_game`, `TARGET_TOLERANCE`), `bindings/src/train.rs` (`recorded_game`, the `TrainingSet` and `policy_decisions` docs)
- Test: `net/tests/data.rs`, `tests/python/test_training_data.py`

**Interfaces:**
- Consumes: Task 3's record dicts (`"target"` on Gumbel decisions) and `write_explore(..., gumbel=)`.
- Produces:
  - `settler_net::data::RecordedDecision` gains `pub target: Option<Vec<f32>>`; `pub const TARGET_TOLERANCE: f64 = 1e-4`.
  - `Batch.policy` row = the target at each legal move's policy id when the decision has one, else as before; `Batch.full` = 1 when the decision has a target, else as before; `TrainingSet::policy_decisions` / `sample_full` count and draw decisions with a target.
  - `TrainingSet::new` / `extend` errors: `move {i}: {n} target entries for {m} legal moves`, `move {i}: target entry {j} is {x}, not a probability`, `move {i}: the target sums to {total}, not 1`, each after `game {g} (seed {s}): `.
  - The Python record parser reads an optional `"target"` (a list of numbers, or absent / null).

- [ ] **Step 1: Write the failing tests**

In `net/tests/data.rs`, in `recorded_every`, add `target: None,` after `full_search: decisions.len() % 2 == 0,`. After `a_record_that_does_not_replay_is_an_error`, add:

```rust
/// `g` with an improved policy on every decision ((j + 1) / Σ over its legal moves), every visit
/// count zero and every search marked fast: only the targets make policy targets.
fn with_targets(g: &RecordedGame) -> RecordedGame {
    let mut g = g.clone();
    for d in &mut g.decisions {
        let n = d.visits.len();
        let total = (n * (n + 1) / 2) as f32;
        d.target = Some((1..=n).map(|j| j as f32 / total).collect());
        d.visits.iter_mut().for_each(|v| *v = 0);
        d.full_search = false;
    }
    g
}

#[test]
fn a_decision_with_a_target_trains_on_it_whatever_its_visits() {
    let g = with_targets(&recorded(1, decided()));
    let set = TrainingSet::new(vec![g.clone()]).unwrap();
    assert_eq!(set.policy_decisions(), g.decisions.len());
    assert_eq!(set.full_decisions(), 0, "full_decisions still counts the flag");
    let mut s = State::new(g.seed, g.config);
    let mut next = 0;
    for (i, &a) in g.actions.iter().enumerate() {
        if next < g.decisions.len() && g.decisions[next].index as usize == i {
            let d = &g.decisions[next];
            let mut b = Batch::default();
            set.position(0, next, &mut b);
            let t = d.target.as_ref().unwrap();
            let legal = s.legal_actions();
            for (j, &l) in legal.iter().enumerate() {
                assert_eq!(b.policy[policy_id(l, d.actor)], t[j], "move {i}");
            }
            let row: f64 = b.policy.iter().map(|&x| x as f64).sum();
            assert!((row - 1.0).abs() < 1e-5, "move {i}: {row}");
            assert_eq!(b.full, vec![1], "move {i}");
            next += 1;
        }
        s.apply(a);
    }
    let a = set.sample_full(64, 3, 2).unwrap();
    assert!(a.full.iter().all(|&f| f == 1));
}

#[test]
fn a_bad_target_is_an_error_naming_the_decision() {
    let g = with_targets(&recorded(1, decided()));
    let k = g
        .decisions
        .iter()
        .position(|d| d.visits.len() >= 2)
        .expect("a decision with a choice");
    let at = g.decisions[k].index;
    let bad = |edit: &dyn Fn(&mut Vec<f32>)| {
        let mut b = g.clone();
        edit(b.decisions[k].target.as_mut().unwrap());
        err(b)
    };
    for (e, want) in [
        (
            bad(&|t| {
                t.pop();
            }),
            "target entries",
        ),
        (bad(&|t| t.push(0.0)), "target entries"),
        (bad(&|t| t.iter_mut().for_each(|x| *x *= 0.9)), "sums to"),
        (bad(&|t| t[0] = f32::NAN), "target entry 0"),
        (
            bad(&|t| {
                t[1] += t[0] + 1.0;
                t[0] = -1.0;
            }),
            "target entry 0",
        ),
    ] {
        assert!(e.starts_with("game 0 (seed 1): "), "{e}");
        assert!(e.contains(&format!("move {at}: ")) && e.contains(want), "{e}");
    }
}
```

In `tests/python/test_training_data.py`, after `test_malformed_records_are_value_errors_not_crashes`, add:

```python
def test_a_gumbel_record_trains_on_its_improved_policy(tmp_path):
    games = explore_games(tmp_path, 2, gumbel=6)
    ts = TrainingSet(games)
    assert ts.policy_decisions == ts.decisions == ts.full_decisions
    checked = 0
    for gi, g in enumerate(games):
        for di, d in enumerate(g["decisions"][:30]):
            b = ts.position(gi, di)
            policy = floats(b["policy"])
            for a, t in zip(d["legal"], d["target"]):
                assert policy[policy_id(a, d["actor"])] == pytest.approx(t, abs=1e-7)
            assert sum(policy) == pytest.approx(1.0, abs=1e-4)
            assert list(b["full"]) == [1]
            checked += 1
    assert checked > 30


@pytest.mark.parametrize("edit, message", [
    (lambda t: t.pop(), "target entries"),
    (lambda t: t.append(0.0), "target entries"),
    (lambda t: t.__setitem__(0, t[0] + 0.5), "sums to"),
    (lambda t: t.__setitem__(0, -1.0), "target entry 0"),
    (lambda t: t.__setitem__(0, "x"), "game 0"),
], ids=["short", "long", "sum", "negative", "not-a-number"])
def test_a_malformed_target_is_a_value_error_naming_the_decision(tmp_path, edit, message):
    (g,) = explore_games(tmp_path, 1, gumbel=6)
    bad = json.loads(json.dumps(g))
    edit(bad["decisions"][0]["target"])
    with pytest.raises(ValueError, match=message):
        TrainingSet([bad])
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `PATH=$HOME/.cargo/bin:$PATH cargo test --release -p settler-net --test data`
Expected: FAIL to compile with `struct RecordedDecision has no field named target`.

- [ ] **Step 3: Write the implementation**

In `net/src/data.rs`, after `pub const MAX_SAMPLE: usize = 1 << 16;` add:

```rust
/// How far a recorded `target` may sum from 1.
pub const TARGET_TOLERANCE: f64 = 1e-4;
```

In `RecordedDecision`, after `pub full_search: bool,` add:

```rust
    /// A Gumbel search's improved policy (spec P2 Section 5), aligned with the legal moves and
    /// summing to 1. When present it is the policy target, whatever the visits.
    pub target: Option<Vec<f32>>,
```

In `Batch`, replace the `policy` and `full` doc comments with:

```rust
    /// n × ACTION_SPACE_SIZE: the decision's `target` if it has one, else root visits normalised
    /// over the mask (uniform if none), at the same policy ids.
    pub policy: Vec<f32>,
    /// n: 1 for a policy target: a decision with a `target`, or a full search with at least one
    /// visit.
    pub full: Vec<u8>,
```

In `Batch::push`, replace from `for (a, &v) in legal.iter().zip(&d.visits) {` through `self.full.push((d.full_search && total > 0) as u8);` with:

```rust
        for (j, (a, &v)) in legal.iter().zip(&d.visits).enumerate() {
            let k = m0 + policy_id(*a, actor);
            self.mask[k] = 1;
            self.policy[k] = match &d.target {
                Some(t) => t[j],
                None if total > 0 => v as f32 / total as f32,
                None => 1.0 / legal.len() as f32,
            };
        }
        self.full
            .push((d.target.is_some() || (d.full_search && total > 0)) as u8);
```

Replace `check_game`'s doc comment (both lines) with:

```rust
/// Checks that `g` replays, and returns the indices of its decisions that are policy targets
/// (`Batch::full` 1: a decision with a valid `target`, or a full search with at least one visit).
```

and in `check_game`, replace

```rust
            let visited = d.visits.iter().any(|&v| v > 0);
            if d.full_search && visited {
                targets.push(k as u32);
            }
```

with:

```rust
            if let Some(t) = &d.target {
                if t.len() != buf.len() {
                    return Err(format!(
                        "move {i}: {} target entries for {} legal moves",
                        t.len(),
                        buf.len()
                    ));
                }
                if let Some((j, x)) = t
                    .iter()
                    .enumerate()
                    .find(|(_, x)| !(x.is_finite() && **x >= 0.0))
                {
                    return Err(format!(
                        "move {i}: target entry {j} is {x}, not a probability"
                    ));
                }
                let total: f64 = t.iter().map(|&x| x as f64).sum();
                if (total - 1.0).abs() > TARGET_TOLERANCE {
                    return Err(format!("move {i}: the target sums to {total}, not 1"));
                }
            }
            let visited = d.visits.iter().any(|&v| v > 0);
            if d.target.is_some() || (d.full_search && visited) {
                targets.push(k as u32);
            }
```

Replace the doc comment of `TrainingSet::policy_decisions` (`/// Panics unless ...` and `/// Decisions that are policy targets ...`) with:

```rust
    /// Decisions that are policy targets (`Batch::full` 1): decisions with a target, and full
    /// searches with at least one visit.
```

In `bindings/src/train.rs`, in `recorded_game`, replace the `decisions.push(RecordedDecision { ... });` with:

```rust
        let target = match d.get_item("target")? {
            Some(t) if !t.is_none() => Some(t.extract::<Vec<f32>>()?),
            _ => None,
        };
        decisions.push(RecordedDecision {
            index: int_arg(&field(d, "index", "decision")?, "decision index")?,
            actor: seat(&field(d, "actor", "decision")?, "decision actor")?,
            visits: int_list(&field(d, "visits", "decision")?, "visits")?,
            full_search: field(d, "full_search", "decision")?.extract()?,
            target,
        });
```

Replace the `PyTrainingSet` doc comment's first two lines with:

```rust
/// Recorded games (dicts with `seed`, `config`, `actions`, `decisions[]` with `index`, `actor`,
/// `visits`, `full_search` and an optional `target`, `winner` and `vp`), each replayed once and
/// checked. Build it a record
```

and the `policy_decisions` getter's doc with `/// Decisions that are policy targets (with a target, or full searches with at least one visit).`

- [ ] **Step 4: Run the tests to verify they pass**

Run: `PATH=$HOME/.cargo/bin:$PATH cargo test --release -p settler-net --test data`
Expected: PASS, including the 2 new tests; `a_position_equals_the_encoder_on_the_replayed_state` (records without targets) passes unchanged.

Rebuild the extension, then run: `.venv/bin/pytest -q tests/python/test_training_data.py`
Expected: PASS, including the 6 new cases.

- [ ] **Step 5: Format, lint, the full suite, rebuild, pytest**

Run the Global Constraints sequence (fmt, fmt --check, clippy, `cargo test --workspace --release`, rebuild the extension, `.venv/bin/pytest -q -rs`). Expected: all pass, nothing skipped.

- [ ] **Step 6: Commit**

```bash
git add net/src/data.rs bindings/src/train.rs net/tests/data.rs tests/python/test_training_data.py
git diff --cached | grep -nE 'sk-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{36}|password=[^[:space:]]+'
git -c user.email=sean.may101@gmail.com commit -m "net: a recorded target is the decision's policy target whatever its visits; bad targets are errors naming the move"
```

---

### Task 5: `fast-loop --gumbel SIMS` and `train-loop --gumbel SIMS`

**Files:**
- Modify: `alphasettler/train/fastloop.py` (docstring, `FastLoop.__init__`, `init`, `_generation`, `_selfplay`), `alphasettler/train/loop.py` (`LoopSettings`, `_check`, `Loop.__init__`, `_mode`, `_selfplay`), `alphasettler/cli.py` (both parsers, `_fast_loop`, `_train_loop`)
- Test: `tests/python/test_fast_loop.py`, `tests/python/test_train_loop.py`

**Interfaces:**
- Consumes: Task 3's `selfplay.check_gumbel` and `write_explore(..., gumbel=)`; Task 4's training on targets.
- Produces:
  - `FastLoop.init(root, seed_dirs, start_net, first, trades=False, gumbel=None)` writes `"gumbel"` (null or SIMS) into `state.json`; `FastLoop(...).gumbel` (None for a state file without the key); each generation's `selfplay` entry has `"gumbel"`.
  - `LoopSettings.gumbel: int | None = None`; `Loop(...).gumbel`; `manifest.json` key `"gumbel"`; generation plans (`selfplay.json`) have `"gumbel"` only when set.
  - CLI: `fast-loop --gumbel SIMS` (first run only) and `train-loop --gumbel SIMS`; a mismatch on resume prints `error: <dir> was set up without --gumbel, and a loop keeps the self-play mode it was set up with` (or `with --gumbel N`) and exits 2.

- [ ] **Step 1: Write the failing tests**

In `tests/python/test_fast_loop.py`, add at the end:

```python
def test_cli_fast_loop_with_gumbel_runs_two_generations_of_gumbel_self_play(tmp_path):
    seed = generation(tmp_path, "seed-gen", seed_start=0)
    save(Net(8, 1), tmp_path / "start.bin")
    root = tmp_path / "loop"
    r = cli("fast-loop", "--dir", str(root), "--generations", "2", "--first", "1", "--gumbel", "100",
            "--seed-dirs", str(seed), "--start-net", str(tmp_path / "start.bin"), "--games", "2",
            "--window-games", "3", "--minutes", "0.01", "--gate-seeds", "1", "--gate-simulations", "6",
            "--device", "cpu", "--threads", "2")
    assert r.returncode == 0, r.stderr
    state = json.loads((root / STATE).read_text())
    assert state["gumbel"] == 100 and [g["gen"] for g in state["generations"]] == [1, 2]
    for g in state["generations"]:
        sp = g["selfplay"]
        assert sp["gumbel"] == 100 and sp["belief_resets"] == 0
        assert sp["full_decisions"] == sp["decisions"] > 0
        assert "+g" not in g["gate"]["candidate"] and "+g" not in g["gate"]["baseline"], "gates are unchanged"
    for k in (1, 2):
        paths = sorted((root / gen_name(k)).glob("selfplay-local-*.jsonl.gz"))
        assert paths
        for path in paths:
            for game in records.read(path):
                assert game["selfplay"]["gumbel_simulations"] == 100
                for d in game["decisions"]:
                    assert d["full_search"] is True and sum(d["visits"]) == 99
                    assert len(d["target"]) == len(d["legal"])
                    assert sum(d["target"]) == pytest.approx(1.0, abs=1e-4)


@pytest.mark.parametrize("set_up, resume, message", [
    (None, ["--gumbel", "100"], "without --gumbel"),
    (100, ["--gumbel", "50"], "with --gumbel 100"),
], ids=["set-up-without", "other-value"])
def test_cli_fast_loop_keeps_the_self_play_mode_it_was_set_up_with(tmp_path, set_up, resume, message):
    seed = generation(tmp_path, "seed-gen", seed_start=0)
    save(Net(8, 1), tmp_path / "start.bin")
    root = tmp_path / "loop"
    FastLoop.init(root, [seed], tmp_path / "start.bin", first=1, gumbel=set_up)
    assert FastLoop(root, FastLoopSettings(generations=0, **TINY)).gumbel == set_up
    r = cli("fast-loop", "--dir", str(root), "--generations", "0", *resume)
    assert r.returncode == 2 and message in r.stderr, r.stderr
    r = cli("fast-loop", "--dir", str(root), "--generations", "0")
    assert r.returncode == 0, r.stderr
    assert json.loads((root / STATE).read_text())["gumbel"] == set_up, "a resume without the flag keeps it"


def test_a_loop_set_up_before_gumbel_existed_reads_as_set_up_without_it(tmp_path):
    root, _ = setup(tmp_path)
    state = json.loads((root / STATE).read_text())
    del state["gumbel"]
    (root / STATE).write_text(json.dumps(state))
    assert FastLoop(root, FastLoopSettings(generations=0, **TINY)).gumbel is None
    r = cli("fast-loop", "--dir", str(root), "--generations", "0", "--gumbel", "100")
    assert r.returncode == 2 and "without --gumbel" in r.stderr


def test_a_gumbel_budget_below_two_is_refused_before_anything_is_written(tmp_path):
    seed = generation(tmp_path, "seed-gen", seed_start=0)
    save(Net(8, 1), tmp_path / "start.bin")
    with pytest.raises(ValueError, match="gumbel simulations"):
        FastLoop.init(tmp_path / "bad", [seed], tmp_path / "start.bin", first=1, gumbel=1)
    assert not (tmp_path / "bad").exists()
```

In `tests/python/test_train_loop.py`, add `{"gumbel": 1}` to the `parametrize("change", [...])` list of `test_bad_settings_are_an_error_before_anything_is_written` (after `{"width": 2.5}`), add `from alphasettler import files, records` in place of `from alphasettler import files`, and add at the end:

```python
def test_a_gumbel_train_loop_records_its_mode_and_improved_policies(tmp_path):
    d = tmp_path / "g"
    s = replace(TINY, generations=0, gumbel=4)
    assert Loop(d, s).run() == 3
    m = manifest(d)
    assert m["gumbel"] == 4 and m["generations"][0]["selfplay"]["gumbel"] == 4
    paths = sorted((d / "gen-000").glob("selfplay-local-*.jsonl.gz"))
    assert paths
    for path in paths:
        for g in records.read(path):
            assert g["selfplay"]["gumbel_simulations"] == 4
            assert all(len(x["target"]) == len(x["legal"]) for x in g["decisions"])
    assert Loop(d, replace(s, gumbel=None)).gumbel == 4, "a resume without the flag keeps it"
    with pytest.raises(ValueError, match="with --gumbel 4"):
        Loop(d, replace(s, gumbel=8))


def test_a_train_loop_set_up_without_gumbel_refuses_it(reference):
    assert manifest(reference)["gumbel"] is None
    with pytest.raises(ValueError, match="without --gumbel"):
        Loop(reference, replace(TINY, gumbel=4))
    r = cli("train-loop", "--dir", str(reference), "--generations", "1", "--gumbel", "4")
    assert r.returncode == 2 and "without --gumbel" in r.stderr
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest -q tests/python/test_fast_loop.py tests/python/test_train_loop.py -k "gumbel"`
Expected: FAIL (`unexpected keyword argument 'gumbel'`, `unrecognized arguments: --gumbel`).

- [ ] **Step 3: Write the fast-loop implementation**

In `alphasettler/train/fastloop.py`, append to the module docstring (before its closing `"""`):

```python
With `gumbel` (set when the loop is set up: `init(..., gumbel=SIMS)`, `fast-loop --gumbel SIMS`;
spec P2 Section 5), every self-play move is a Gumbel root search of SIMS simulations whose improved
policy is the decision's policy target; the gate is unchanged. A resumed loop keeps that too.
```

In `FastLoop.__init__`, after the `self.trades = ...` line add:

```python
        self.gumbel = self.state.get("gumbel")  # state files from before P2 have none: PUCT self-play
```

Replace `init` with:

```python
    @staticmethod
    def init(root: str | Path, seed_dirs: list[str | Path], start_net: str | Path, first: int,
             trades: bool = False, gumbel: int | None = None) -> None:
        """Create `root` with `seed_dirs` (oldest first) as the starting records and `start_net`
        as the best net; generation numbering starts at `first`; `trades` plays the trading game;
        `gumbel` (simulations) makes every self-play move a Gumbel root search."""
        root = Path(root)
        if (root / STATE).exists():
            raise ValueError(f"{root / STATE} exists: the loop is already set up (resume it with fast-loop)")
        if gumbel is not None:
            selfplay_mod.check_gumbel(gumbel)
        nets = root / "nets"
        nets.mkdir(parents=True, exist_ok=True)
        files.atomic_copy(start_net, nets / "start.bin")
        files.atomic_copy(start_net, nets / "best.bin")
        dirs = [str(Path(d).resolve()) for d in seed_dirs]
        state = {"seed_dirs": [{"dir": d, "games": count_games(Path(d))} for d in dirs],
                 "first": first, "trades": trades, "gumbel": gumbel, "best": "start", "latest": "start",
                 "generations": []}
        files.atomic_write_text(root / STATE, json.dumps(state, indent=1) + "\n")
```

In `_generation`, replace `print(f"{name}: selfplay with net:{best}", flush=True)` with:

```python
        mode = "" if self.gumbel is None else f" (Gumbel root, {self.gumbel} simulations)"
        print(f"{name}: selfplay with net:{best}{mode}", flush=True)
```

In `_selfplay`, add `gumbel=self.gumbel` to the `selfplay_mod.write_explore(...)` call (after `threads=self.s.threads`), and replace `return {"net": best, "seed_start": start, **totals}` with `return {"net": best, "seed_start": start, "gumbel": self.gumbel, **totals}`.

- [ ] **Step 4: Write the train-loop implementation**

In `alphasettler/train/loop.py`, add to `LoopSettings` after `threads: int | None = None`:

```python
    gumbel: int | None = None  # Gumbel self-play simulations, kept from the run's first start
```

In `_check`, after the `selfplay_mod.settings(None, ".", ...)` line add:

```python
    if s.gumbel is not None:
        selfplay_mod.check_gumbel(s.gumbel)
```

In `Loop.__init__`, after `self.manifest = self._load()` add `self.gumbel = self._mode(settings.gumbel)`, and after `_load` add:

```python
    def _mode(self, asked: int | None) -> int | None:
        """The self-play mode: Gumbel simulations, or None for PUCT self-play. Set when the run
        starts (`--gumbel SIMS`) and kept on every resume, as the fast loop keeps `--trades`; a
        manifest from before P2 has none. Asking for another mode is a ValueError."""
        if not (self.root / MANIFEST).exists():
            self.manifest["gumbel"] = asked
            return asked
        stored = self.manifest.get("gumbel")
        if asked is not None and asked != stored:
            was = "without --gumbel" if stored is None else f"with --gumbel {stored}"
            raise ValueError(f"{self.root} was set up {was}, and a loop keeps the self-play mode it was set up with")
        self.manifest["gumbel"] = stored
        return stored
```

In `_selfplay`, after the `plan = {...}` dict add:

```python
        if self.gumbel is not None:  # only then: plans written before P2 still match
            plan["gumbel"] = self.gumbel
```

and add `gumbel=self.gumbel` to its `selfplay_mod.write_explore(...)` call (after `threads=self.s.threads`).

- [ ] **Step 5: Write the CLI**

In `alphasettler/cli.py`, after the fast-loop parser's `--trades` argument add:

```python
    fl.add_argument("--gumbel", type=_positive_int, default=None, metavar="SIMS",
                    help="first run only: every self-play move a Gumbel root search of SIMS simulations")
```

after the train-loop parser's `--threads` argument add:

```python
    lp.add_argument("--gumbel", type=_positive_int, default=None, metavar="SIMS",
                    help="first run only: every self-play move a Gumbel root search of SIMS simulations")
```

In `_fast_loop`, change `FastLoop.init(root, args.seed_dirs, args.start_net, args.first, trades=args.trades)` to `FastLoop.init(root, args.seed_dirs, args.start_net, args.first, trades=args.trades, gumbel=args.gumbel)`, and after the `if args.trades and not loop.trades:` block add:

```python
    if args.gumbel is not None and args.gumbel != loop.gumbel:
        was = "without --gumbel" if loop.gumbel is None else f"with --gumbel {loop.gumbel}"
        print(f"error: {root} was set up {was}, and a loop keeps the self-play mode it was set up with",
              file=sys.stderr)
        return 2
```

In `_train_loop`, add `gumbel=args.gumbel,` to the `LoopSettings(...)` call (after `device=args.device, threads=args.threads,`).

- [ ] **Step 6: Run the tests to verify they pass**

Run: `.venv/bin/pytest -q tests/python/test_fast_loop.py tests/python/test_train_loop.py`
Expected: PASS, including the 7 new test cases (5 in `test_fast_loop.py`, 2 in `test_train_loop.py`) and the new `{"gumbel": 1}` case; every existing loop test passes unchanged.

- [ ] **Step 7: Format, lint, the full suite, pytest**

Run the Global Constraints sequence (fmt, fmt --check, clippy, `cargo test --workspace --release`, rebuild the extension, `.venv/bin/pytest -q -rs`). Expected: all pass, nothing skipped.

- [ ] **Step 8: Commit**

```bash
git add alphasettler/train/fastloop.py alphasettler/train/loop.py alphasettler/cli.py \
  tests/python/test_fast_loop.py tests/python/test_train_loop.py
git diff --cached | grep -nE 'sk-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{36}|password=[^[:space:]]+'
git -c user.email=sean.may101@gmail.com commit -m "loops: fast-loop and train-loop --gumbel SIMS for Gumbel self-play, kept from set-up like --trades"
```

---

### Task 6: The measurements and `docs/bot/results-p2-gumbel.md`

**Files:**
- Create: `docs/bot/results-p2-gumbel.md`
- Run outputs (not committed; `runs/` is ignored): `runs/p2/` (scripts, logs, match records, `A/`, `B/`, the state copies, `nets/A.bin`, `nets/B.bin`)

**Interfaces:**
- Consumes: `+g` bots (Task 2), `fast-loop --gumbel` (Task 5), `alphasettler sprt` and `arena` with `--trades` / `--no-trades`; `runs/3c/nets/gen-034.bin`, `runs/3b-v2/gen-002` .. `gen-005`, `runs/3c/gen-030` .. `gen-034`.
- Produces: the run outputs in `runs/p2/` and `docs/bot/results-p2-gumbel.md`.

Every step but Step 8 is run by the controller in the background, on an otherwise idle machine. If a step fails, record the failure with its numbers in the results; never rerun a match on the same seeds after changing anything.

- [ ] **Step 1: Build at Task 5's commit (controller runs this)**

```bash
git log -1 --format=%s
PATH=$HOME/.cargo/bin:$PATH cargo build --release -p settler-bots
mkdir -p runs/p2
ls runs/3c/nets/gen-034.bin runs/3b-v2/gen-002 runs/3b-v2/gen-005 runs/3c/gen-030 runs/3c/gen-034 > /dev/null
test ! -e runs/p2/A && test ! -e runs/p2/B
pgrep -fl "alphasettler|settler" || echo idle
```

Then rebuild the extension per the Global Constraints. Pass: the commit subject is Task 5's; every command exits 0; the last line is `idle`.

- [ ] **Step 2: Criterion 1 at 32 simulations (controller runs this)**

```bash
.venv/bin/alphasettler sprt --candidate net:gen-034@32+g --baseline net:gen-034@32 --no-trades \
  --seed-start 3000000000 --nets-dir runs/3c/nets --out runs/p2/sprt-32.jsonl 2>&1 | tee runs/p2/sprt-32.log
```

The defaults are the spec's: p0 0.25, p1 0.30, alpha = beta = 0.05, batches of 12 seeds, at most 400 seeds. Pass: exit 0 and a last-but-one line starting `sprt: ` (H1, H0 or undecided); recorded either way.

- [ ] **Step 3: Criterion 1 at 100 simulations (controller runs this)**

```bash
.venv/bin/alphasettler sprt --candidate net:gen-034@100+g --baseline net:gen-034@100 --no-trades \
  --seed-start 3000001000 --nets-dir runs/3c/nets --out runs/p2/sprt-100.jsonl 2>&1 | tee runs/p2/sprt-100.log
```

Pass: as Step 2.

- [ ] **Step 4: Criterion 1 at 600 simulations (controller runs this)**

```bash
.venv/bin/alphasettler arena --candidate net:gen-034@600+g --baseline net:gen-034@600 --no-trades --seeds 200 \
  --seed-start 3000002000 --nets-dir runs/3c/nets --out runs/p2/arena-600.jsonl 2>&1 | tee runs/p2/arena-600.log
for f in runs/p2/sprt-32.jsonl runs/p2/sprt-100.jsonl; do
  .venv/bin/python3 -c "
import json, sys
rs = [json.loads(l) for l in open(sys.argv[1])]
print(f'{sys.argv[1]}: {len(rs)} games, belief resets {sum(r[\"belief_resets\"] for r in rs)}')
" "$f" | tee -a runs/p2/resets.log
done
```

Pass: exit 0; `belief resets: 0` in `arena-600.log` and `belief resets 0` on both lines of `resets.log`. The bar is z > -2 against 0.25 (the summary line's z); recorded either way.

- [ ] **Step 5: Criterion 2, the two 3-hour loops (controller runs this)**

Create `runs/p2/arm.sh`:

```bash
#!/bin/bash
# One arm of P2 criterion 2: a fresh fast-loop from gen-034 in runs/p2/$1 (extra arguments go to
# fast-loop), stopped at the first 10-second check at or past BUDGET seconds after this script
# starts. runs/p2/$1-state-at-cut.json is the state at the cut: only generations finished by then
# count. Every log line is prefixed with its epoch second.
set -u
cd /Users/seanmay/Desktop/CurrentProjects/AlphaSettler
arm=$1
shift
root=runs/p2/$arm
log=runs/p2/$arm.log
budget=10800
if [ -e "$root" ]; then
  echo "$root exists: each arm starts in a fresh root" >&2
  exit 2
fi
start=$(date +%s)
echo "$start start $arm $*" > "$log"
.venv/bin/alphasettler fast-loop --dir "$root" --generations 99 --first 35 --trades \
  --seed-dirs runs/3b-v2/gen-002 runs/3b-v2/gen-003 runs/3b-v2/gen-004 runs/3b-v2/gen-005 \
              runs/3c/gen-030 runs/3c/gen-031 runs/3c/gen-032 runs/3c/gen-033 runs/3c/gen-034 \
  --start-net runs/3c/nets/gen-034.bin "$@" \
  > >(while IFS= read -r line; do echo "$(date +%s) $line"; done >> "$log") 2>&1 &
pid=$!
while kill -0 "$pid" 2>/dev/null; do
  if (( $(date +%s) - start >= budget )); then
    cp "$root/state.json" "runs/p2/$arm-state-at-cut.json"
    cut=$(date +%s)
    # SIGTERM: a background job of a non-interactive bash ignores SIGINT.
    kill -TERM "$pid"
    wait "$pid"
    echo "$cut cut $arm after $((cut - start)) s" >> "$log"
    exit 0
  fi
  sleep 10
done
wait "$pid"
code=$?
echo "$(date +%s) loop exited early with $code" >> "$log"
exit 1
```

and `runs/p2/both.sh`:

```bash
#!/bin/bash
cd /Users/seanmay/Desktop/CurrentProjects/AlphaSettler
bash runs/p2/arm.sh A && bash runs/p2/arm.sh B --gumbel 100
echo "=== both exit $?" >> runs/p2/both.out
```

Start it detached (it runs about 6 hours, longer than a background tool call may live), with the machine otherwise idle:

```bash
chmod +x runs/p2/arm.sh runs/p2/both.sh
pgrep -fl "alphasettler|settler" || echo idle
nohup bash runs/p2/both.sh >> runs/p2/both.out 2>&1 &
```

Poll `runs/p2/both.out` until it holds `=== both exit 0`. Pass: `=== both exit 0`; `runs/p2/A.log` and `runs/p2/B.log` each end with `cut A after 108NN s` / `cut B after 108NN s` (10,800 to 10,810); `runs/p2/A-state-at-cut.json` has `"gumbel": null`, `runs/p2/B-state-at-cut.json` has `"gumbel": 100`; every generation in both has `belief_resets` 0 in self-play and gate.

- [ ] **Step 6: The tables and the final nets (controller runs this)**

Create `runs/p2/tables.py`:

```python
"""P2 criterion 2: each arm's generations finished within the budget, from its state at the cut and
its timestamped log. Usage: .venv/bin/python3 runs/p2/tables.py A B"""
import json
import re
import sys

for arm in sys.argv[1:]:
    state = json.load(open(f"runs/p2/{arm}-state-at-cut.json"))
    stamps = {}
    for line in open(f"runs/p2/{arm}.log"):
        m = re.match(r"(\d+) (gen-\d{3}): (selfplay|train|gate)\b", line)
        if m:
            stamps[(m[2], m[3])] = int(m[1])
    games = seconds = 0
    print(f"### {arm}: best net {state['best']}, {len(state['generations'])} generations finished in the budget")
    print()
    print("| gen | games | decisions | full searches | steps | held-out score | gate wins / games | z | promoted "
          "| self-play minutes |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for g in state["generations"]:
        name = f"gen-{g['gen']:03d}"
        sp, tr, gate = g["selfplay"], g["train"], g["gate"]
        secs = stamps[(name, "train")] - stamps[(name, "selfplay")]
        games += sp["games"]
        seconds += secs
        z = "-" if gate["z"] is None else f"{gate['z']:.2f}"
        print(f"| {g['gen']} | {sp['games']:,} | {sp['decisions']:,} | {sp['full_decisions']:,} | {tr['steps']:,} | "
              f"{tr['best']['score']:.4f} | {gate['wins']} / {gate['games']} | {z} | "
              f"{'yes' if gate['promoted'] else 'no'} | {secs / 60:.1f} |")
    print()
    if seconds:
        print(f"self-play: {games:,} games in {seconds / 3600:.2f} h of self-play, "
              f"{games / (seconds / 3600):,.0f} games per hour")
    else:
        print("self-play: no generation finished in the budget")
    print()
```

Then:

```bash
.venv/bin/python3 runs/p2/tables.py A B | tee runs/p2/tables.md
mkdir -p runs/p2/nets
for arm in A B; do
  best=$(.venv/bin/python3 -c "import json; print(json.load(open('runs/p2/$arm-state-at-cut.json'))['best'])")
  cp "runs/p2/$arm/nets/$best.bin" "runs/p2/nets/$arm.bin"
  echo "$arm: $best" | tee -a runs/p2/finals.log
done
if cmp -s runs/p2/nets/A.bin runs/p2/nets/B.bin; then echo "the final nets are the same file" | tee -a runs/p2/finals.log; fi
```

Pass: `tables.md` has both tables and both games-per-hour lines; `finals.log` names each arm's best net.

- [ ] **Step 7: Criterion 2's SPRT (controller runs this)**

If `finals.log` says the final nets are the same file, skip this step (the ruling records H0). Otherwise:

```bash
.venv/bin/alphasettler sprt --candidate net:B@100+t --baseline net:A@100+t --trades \
  --seed-start 3000003000 --nets-dir runs/p2/nets --out runs/p2/sprt-B-vs-A.jsonl 2>&1 | tee runs/p2/sprt-B-vs-A.log
.venv/bin/python3 -c "
import json
rs = [json.loads(l) for l in open('runs/p2/sprt-B-vs-A.jsonl')]
print(f'{len(rs)} games, belief resets {sum(r[\"belief_resets\"] for r in rs)}')
" | tee -a runs/p2/sprt-B-vs-A.log
```

Pass: exit 0, a line starting `sprt: `, and `belief resets 0`. H1: Gumbel self-play becomes the mode new loops are set up with; H0 or undecided: the control stays. Recorded either way.

- [ ] **Step 8: Write `docs/bot/results-p2-gumbel.md`**

Every number is copied from the files named here; nothing is estimated. The file:

```markdown
# Performance track P2 (Gumbel root search): results

Date: <date of the last run>. Machine: <`sysctl -n machdep.cpu.brand_string`>, macOS (<`uname -r`>).
Commit: <`git log -1 --format=%h`> (Plan 8, Task 5), plus the commit adding this file.

Spec: [`docs/superpowers/specs/2026-10-08-gumbel-root-design.md`](../superpowers/specs/2026-10-08-gumbel-root-design.md).
Plan: [`docs/superpowers/plans/2026-10-08-plan-8-gumbel-root.md`](../superpowers/plans/2026-10-08-plan-8-gumbel-root.md).

## Play strength, no training (done criterion 1)

`+g` searches the root by sequential halving with no Gumbel draws; the baseline is the same net
with a PUCT root. The no-trade game; `runs/3c/nets/gen-034.bin` on every seat.

| Run | Result | Bar | Outcome |
|---|---|---|---|
| `sprt --candidate net:gen-034@32+g --baseline net:gen-034@32 --no-trades`, seeds from 3,000,000,000 | <the summary line and the sprt line of runs/p2/sprt-32.log> | recorded (p0 0.25, p1 0.30, alpha = beta = 0.05) | <H1 / H0 / undecided> |
| `sprt --candidate net:gen-034@100+g --baseline net:gen-034@100 --no-trades`, seeds from 3,000,001,000 | <the summary line and the sprt line of runs/p2/sprt-100.log> | recorded (same test) | <H1 / H0 / undecided> |
| `arena --candidate net:gen-034@600+g --baseline net:gen-034@600 --no-trades`, 200 seeds from 3,000,002,000 | <the summary line of runs/p2/arena-600.log> | z above -2 | <met / not met> |

Belief resets: <the lines of runs/p2/resets.log and the belief resets line of runs/p2/arena-600.log>.

## Training at equal wall time (done criterion 2)

Two `fast-loop` runs from gen-034, one after the other on an otherwise idle machine, each stopped
3 hours after it started (`runs/p2/arm.sh`): fresh roots, the trading game, seed directories
`runs/3b-v2/gen-002` to `gen-005` and `runs/3c/gen-030` to `gen-034`, 2,000 games a generation,
a window of 40,000 games, 10 minutes of training, a gate of 100 seeds at 100 simulations (PUCT,
`+t` on both sides). A is today's self-play (600 / 100 simulations, p 0.25, Dirichlet noise, 8
sampled turns); B is `--gumbel 100` (every move a Gumbel search of 100 simulations with draws,
every one a policy target). Only generations finished within the 3 hours count.

<the A table and its self-play line, from runs/p2/tables.md>

<the B table and its self-play line, from runs/p2/tables.md>

Final nets: <the lines of runs/p2/finals.log>.

| Run | Result | Bar | Outcome |
|---|---|---|---|
| `sprt --candidate net:B@100+t --baseline net:A@100+t --trades`, seeds from 3,000,003,000 | <the summary line and the sprt line of runs/p2/sprt-B-vs-A.log, or "not run: both final nets are the same file"> | H1 (p1 0.30 against p0 0.25, alpha = beta = 0.05) | <H1: Gumbel self-play becomes the default / H0: the control stays / undecided: the control stays> |

Belief resets: <the belief resets line of runs/p2/sprt-B-vs-A.log, or "-">.

## Rulings the build made

- Gumbel is a root mode of a bot's search config (`+g` bots: no draws; Gumbel self-play: draws
  from the search's own stream); PUCT inside the tree; PUCT bots and PUCT self-play play exactly
  the games they played before (three digests pinned at eb1f739).
- The prior is floored at the smallest positive f32 before its log; π in v_mix and the improved
  policy is the softmax of those logits.
- Ties in the candidates, the halving and the final pick go to the lower action id. A candidate
  left unvisited because the budget ran out is ranked by v_mix.
- The last phase spends what is left round-robin from the leading candidate; v̂ is the root
  evaluation's value for the searching player.
- A decision with a target is a policy target whatever its visits; targets must sum to 1 within
  1e-4. PUCT records carry no target and are unchanged.
- Loops keep their self-play mode from set-up; files from before P2 read as PUCT self-play.
- Each arm's budget is 10,800 s from the start of its script, set-up included; the state at the
  cut decides its generations and its final net. Both arms played the same self-play seeds.
- Fresh seeds: 3,000,000,000, 3,000,001,000, 3,000,002,000 and 3,000,003,000.
- Games per hour counts self-play time only (the log's `selfplay` to `train` lines).

## Next

<one paragraph: if H1, set up the next training loop (3d) with --gumbel 100 and say what the
criterion-1 lines suggest for play-time +g; if H0 or undecided, what the two tables and the games
per hour point to (P3, batching across games, is next on the performance track either way)>
```

Fill every `<...>` from the named source before committing. The committed file holds numbers and text only, with no angle brackets.

- [ ] **Step 9: Commit**

Run: `grep -c '<' docs/bot/results-p2-gumbel.md`
Expected: `0`.

```bash
git add docs/bot/results-p2-gumbel.md
git diff --cached | grep -nE 'sk-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{36}|password=[^[:space:]]+'
git -c user.email=sean.may101@gmail.com commit -m "docs: P2 results, Gumbel root search at play time and in self-play at equal wall time"
```

---

## Self-review

### Spec coverage

| Spec requirement | Task |
|---|---|
| §1 the motivation (policy targets at 100 simulations) | 3, 4 (every Gumbel decision a target), 6 (criterion 2) |
| §2.1 `+g` vs plain at 32 and 100 (SPRT, p0 0.25, p1 0.30, alpha = beta = 0.05, fresh seeds, no-trade game) and at 600 (200 fresh seeds, z > -2) | 2 (`+g`), 6 (Steps 2–4) |
| §2.2 A and B `fast-loop` runs from gen-034, trading game, same window, minutes and gate, 3 hours each, one after the other on an idle machine; then `net:B@100+t` vs three `net:A@100+t` by SPRT; H1 makes Gumbel the default, H0 recorded | 5 (`--gumbel`), 6 (Steps 5–7) |
| §2.3 results file: criterion 1's three lines, both generation tables (games, steps, held-out score, gate), games per hour, criterion 2's line | 6 (Steps 6, 8) |
| §3.1 the root's prior renormalised over A, ℓ = ln π, v̂ the actor's root value | 1 (`logits`, `root_value`, `with_nothing_visited_the_improved_policy_is_the_prior`) |
| §3.2 Gumbel draws from the search's Rng in self-play, zero in play; m = min(16, \|A\|) by g + ℓ, ties to the lower id | 1 (`candidates`, `without_draws_the_candidates_are_the_sixteen_likeliest_moves`, `the_same_seed_gives_the_same_gumbel_search_and_draws_change_the_candidates`) |
| §3.3 n = simulations − 1 over ceil(log2 m) phases, max(1, floor(n / (phases · \|remaining\|))) each, forced root step, PUCT below, keep ceil(\|remaining\| / 2), leftover round-robin, one candidate left ends the halving | 1 (`gumbel_root`, `phases`, `per_candidate`, `sequential_halving_spends_the_budget_phase_by_phase_on_the_lowest_ids_when_all_else_ties`) |
| §3.4 q̂ and σ with c_visit 50, c_scale 1.0 | 1 (`sigma`, `root_stats`) |
| §3.5 the move: best remaining by g + ℓ + σ(q̂) | 1 (`a_move_that_wins_on_the_spot_is_chosen_and_draws_the_improved_policy`) |
| §3.6 π' = softmax(ℓ + σ(completedQ)), v_mix eq. 33 | 1 (`v_mix`, `completed_q`, `improved_policy` and their unit tests) |
| §3 ISMCTS: the forced step asserted; virtual loss kept; round-robin `batch` at a time | 1 (`Sim::descend` assertion, `Sim::run`, `batches_larger_than_the_candidates_leave_no_virtual_loss`) |
| §3 rulings: root only, m = 16 | Global Constraints; 1 |
| §4 `+g` in any order, at most once, `net:` and `ismcts`, g = 0, `+sK` kept, nothing changes without it | 1 (pins), 2 |
| §5 Exploration's Gumbel mode, full searches with targets, noise / PCR / sampling off; optional `target` aligned with `legal` summing to 1; the sampler uses it, else normalised visits; a target is a policy target whatever its visits; `--gumbel SIMS` for both loops, kept like `--trades`; gates unchanged | 3, 4, 5 |
| §6 out of scope (interior rule, Gumbel at chance or opponent nodes, the gate, P3, GPU) | not built; Task 1 changes the root only; Task 5's test checks the gate has no `+g` |
| §7 search tests (top-m by prior, exactly simulations − 1 root visits, equal within one per phase, the halving's winner, π' sums to 1 / equals softmax(ℓ) unvisited / moves mass to value, a single move unsearched, same seed same move and π') | 1 |
| §7 bots tests (any order, refused when repeated, non-`+g` digests unchanged) | 1 (pins), 2 |
| §7 self-play test (a target on every searched decision summing to 1; belief resets 0) | 3 |
| §7 data tests (trains on the target at the legal ids; without, unchanged; wrong length or sum an error naming the decision) | 4 |
| §7 Python tests (`fast-loop --gumbel 100` runs two tiny generations; a loop set up without refuses it on resume) | 5 |

### Placeholder scan

Searched for "TBD", "TODO", "implement later", "fill in", "similar to Task", "appropriate", "handle edge cases": none in the code steps. The one template is Task 6 Step 8's results file, whose `<...>` marks are run-time measurements, each naming its source; Step 9 refuses to commit while one is left. The three digests in Task 1 Step 1 are real values, computed on the base commit.

### Type consistency

- Rust: `RootMode::{Puct, Gumbel { draw }}`, `SearchConfig::root`, `SearchResult::improved: Option<Vec<f32>>` (Task 1) are the names Task 2 (`with_gumbel`, `config().root`, `take_last_search().improved`) and Task 3 (`play_with`'s `root:`, `target: r.improved`) use. `gumbel::{MAX_CANDIDATES, logits, candidates, phases, per_candidate, sigma, v_mix, completed_q, ranked, improved_policy, gumbel}` are defined in Task 1 Step 4 and used with those signatures in Step 5 and the tests. `split_gumbel -> Option<(String, bool)>` is used by `make_bot`, `Roster::new`, `Roster::make` and the tests. `Exploration::gumbel(u32)`, `Exploration.gumbel: bool`, `Decision.target: Option<Vec<f32>>` (Task 3) match the bindings' `Exploration::gumbel(int_arg(n, "gumbel")?)` and `dec.target`. `RecordedDecision.target: Option<Vec<f32>>`, `TARGET_TOLERANCE` (Task 4) match `recorded_every`, `with_targets` and `bindings/src/train.rs`.
- Python: `check_gumbel`, `settings(..., gumbel=None)`, `write_explore(..., gumbel=None)` (Task 3) are what `FastLoop.init`, `FastLoop._selfplay`, `loop._check` and `Loop._selfplay` (Task 5) call. `FastLoop.gumbel`, `Loop.gumbel`, `LoopSettings.gumbel` and the `"gumbel"` keys are used with those names by the CLI and the tests. `explore_games(..., gumbel=6)` passes through `**kw` to `write_explore`.

### Review Focus

Each of the five lines has its test in the owning task (1, 2, 3: Task 1; 4: Task 4; 5: Task 5). Checked and not added: a root action missing from a sampled world (the spec asserts it; the search panics naming the move, and no position reachable from an observation produces it for the viewer's own decision); a `+g` bot answering a trade or a discard (the root is still the bot's own decision, covered by the trading-game comparison in Task 2); a Gumbel record read by a build from before Task 4 (it ignores `target` and trains on visits, as old records do; no such build reads P2 records).
