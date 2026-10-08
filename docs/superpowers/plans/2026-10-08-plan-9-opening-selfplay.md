# Plan 9: The Opening Model in Self-Play (Sub-project 3d-1b) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let self-play's setup settlements come from the opening model (ε-greedy over its top 5 spots, no search, no recorded decision), expose it as `fast-loop --opening FILE` / `train-loop --opening FILE` (gates `+o` on both sides), then measure it: arm C, a 3-hour `fast-loop` from `gen-034` with exactly P2 arm A's settings plus `--opening runs/3c/nets/opening.bin`, settled by an SPRT of `net:C@100+t+o` against three `net:A@100+t+o`.

**Architecture:** `settler_bots::opening_model` gains `settlement_scores` (the scoring `+o` already does, now shared, with `choose_settlement` built on it). `settler_bots::selfplay` gains `OPENING_EPSILON` / `OPENING_TOP`, `explore_settlement` (the ε-greedy pick from the game's exploration stream) and `play_with_opening` / `run_with_opening` (`Option<&OpeningModel>`); `play_with` / `run_with` call them with `None`, so every existing caller and pinned digest is unchanged. The binding `selfplay_explore` gains `opening=PATH`. Python threads it through `selfplay.write_explore(opening=)` (the record's `selfplay` block names the model by sha256), `FastLoop` (`state.json`) and `Loop` (`manifest.json`), which copy the model to `<dir>/nets/opening.bin` at set-up and keep it on every resume. The controller then runs arm C with `runs/p2/arm.sh` unchanged, the SPRT, and writes `docs/bot/results-3d1b.md`.

**Tech Stack:** Rust 1.80+ (edition 2021, no new crates), PyO3 0.29.3 + maturin 1.15.0, Python 3.12 stdlib, PyTorch 2.14.1 (`train` extra only), pytest 9.1.1.

**Spec:** `docs/superpowers/specs/2026-10-08-opening-selfplay-design.md` (binding). Builds on Plan 7 (`docs/superpowers/plans/2026-10-08-plan-7-opening-model.md`: the `ASOPEN` file, `OpeningModel::load`, `choose_settlement`, `+o`; `docs/bot/results-3d1.md`) and Plan 8 (`docs/superpowers/plans/2026-10-08-plan-8-gumbel-root.md`: how `--gumbel` is kept by both loops, `selfplay.mode_mismatch`; P2's arm A in `runs/p2/A`, `runs/p2/arm.sh`, `runs/p2/tables.py`; `docs/bot/results-p2-gumbel.md`).

## Global Constraints

- Python is `.venv/bin/python3`, tests `.venv/bin/pytest`, the CLI `.venv/bin/alphasettler`. Never `python`, `pip`, `python3` outside the venv, or `pip3` outside the venv.
- Rust is `PATH=$HOME/.cargo/bin:$PATH cargo ...` (cargo is not on PATH; `~/.cargo/bin/cargo` is the same binary). MSRV 1.80 (`rust-version = "1.80"`): use nothing stabilised after 1.80 (no `Option::is_none_or`, no `is_multiple_of`). No new external crates and no new Python dependencies. No numpy anywhere.
- Rebuild the extension after any Rust change used from Python. Another measurement may hold the extension: **never run `maturin develop`**. Build a wheel and rename the new library over the old one (an atomic rename, so running processes keep their mapped copy):
  ```bash
  rm -rf target/wheels-3d1b && PATH=$HOME/.cargo/bin:$PATH .venv/bin/maturin build --release -q -o target/wheels-3d1b \
    && unzip -q -o target/wheels-3d1b/alphasettler-*.whl 'alphasettler/_engine*.so' -d target/wheels-3d1b/x \
    && cp target/wheels-3d1b/x/alphasettler/_engine.cpython-312-darwin.so alphasettler/._engine.new.so \
    && mv -f alphasettler/._engine.new.so alphasettler/_engine.cpython-312-darwin.so
  ```
  "Rebuild the extension" in any step below means exactly this command.
- Every task ends with `PATH=$HOME/.cargo/bin:$PATH cargo fmt --all` then `PATH=$HOME/.cargo/bin:$PATH cargo fmt --check` (no output), `PATH=$HOME/.cargo/bin:$PATH cargo clippy --workspace --release --all-targets -- -D warnings` (exit 0), the full `PATH=$HOME/.cargo/bin:$PATH cargo test --workspace --release` (all pass) and, after the rebuild, `.venv/bin/pytest -q -rs` (all pass, the summary lists no skipped tests).
- Nothing under `engine/`, `search/` or `net/` is edited; under `bots/src/` only `selfplay.rs` and `opening_model.rs` change (`ismcts.rs`, `lib.rs` and `netbot.rs` are not edited). `engine/tests/golden_trace.rs` passes unchanged. These digests are never re-pinned, and pass unchanged after every task: 3a's `the_generic_refactor_leaves_3a_ismcts_games_unchanged` (`bots/tests/ismcts.rs`) and `fixed_exploration_is_3a_self_play` (`bots/tests/selfplay.rs`); P2's `bots_without_the_gumbel_suffix_play_the_games_they_played_before_p2` (`bots/tests/ismcts.rs`), `net_bots_without_the_gumbel_suffix_play_the_games_they_played_before_p2` (`bots/tests/netbot.rs`) and `standard_exploration_plays_the_games_it_played_before_p2` (`bots/tests/selfplay.rs`). Self-play without a model plays exactly as at cf24caf, and every `+o` test of Plan 7 (`bots/tests/opening_model.rs`) passes unchanged.
- The self-play choice (spec Section 2): at a setup-settlement decision with a model, every legal settlement is scored as `+o` scores it (placed on a copy, the actor's own-win probability from the actor's observation); with probability 1 − ε the highest (ties to the first in engine order), else one drawn uniformly from the `K` highest (all of them if fewer are legal). ε = `OPENING_EPSILON` = 0.25, K = `OPENING_TOP` = 5, both draws from the game's exploration stream (`Rng::new(mix(seed, EXPLORE_SALT))`). No search runs and no `Decision` is recorded for those picks; setup roads and every later move are searched and recorded as now.
- The model is fixed for the whole loop: copied to `<dir>/nets/opening.bin` at set-up, where the gates' `+o` find it beside every generation's net. `--opening` and `--gumbel` together are refused (spec Section 3), in Rust and in Python, before anything is written.
- The trading game is exactly `{"max_offers_per_turn": 2, "max_trade_cards": 1}` (`TRADES` in `alphasettler/arena.py`, `trading()` in Rust tests).
- The test (spec Section 1.2): arm C reuses arm A's exact settings and self-play seeds (`runs/p2/arm.sh`: fresh root, the trading game, seed directories `runs/3b-v2/gen-002` .. `gen-005` and `runs/3c/gen-030` .. `gen-034`, start net `runs/3c/nets/gen-034.bin`, `--first 35`, 2,000 games, window 40,000 games, 10 training minutes, a gate of 100 seeds at 100 simulations; set up untimed, then 10,800 s of generations) plus `--opening runs/3c/nets/opening.bin`. Then `net:C@100+t+o` against three `net:A@100+t+o`, `alphasettler sprt --trades` with p0 0.25, p1 0.30, alpha = beta = 0.05, fresh seeds from 4,000,000,000. A is `gen-040` (`runs/p2/nets/A.bin`), C is arm C's best at the cut.
- Determinism: same seed, weights and model give the same games and decisions, independent of thread count. Budgets are simulation counts, never wall time (arm C's 3 hours is the experiment's wall clock, not a search budget).
- No module outside `alphasettler/train/` imports torch.
- Git: one commit per task with the message given in the task, run as `git -c user.email=sean.may101@gmail.com commit -m "..."`. No Co-Authored-By or AI attribution, never `--no-verify`, never push. Before every commit run `git diff --cached | grep -nE 'sk-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{36}|password=[^[:space:]]+'` and expect no output.
- Testing: integration tests over mocks (real games, real model files read by the real Rust loader, real record files read back, real loops through the CLI); test behaviour, not implementation; independent tests (each builds its own temp files); descriptive names; pytest fixtures and `parametrize` where cases repeat.
- Long runs (arm C's 3-hour loop, the SPRT) are run by the controller in the background, never by implementer subagents. Steps marked **controller runs this** give the exact command and the pass condition.
- No README or docs beyond `docs/bot/results-3d1b.md` (Task 3).

## Review Focus

1. Resuming a loop with another opening model, with one on a loop set up without, or a `state.json` / `manifest.json` written before 3d-1b (no `opening` key). Expected: refused with exit 2 (the message says how the loop was set up), the same file (by sha256) accepted, a resume with no flag keeps the model, an old file reads as set up without one. Pinned by Task 2's `test_cli_fast_loop_keeps_the_opening_model_it_was_set_up_with`, `test_a_loop_set_up_before_the_opening_model_existed_reads_as_set_up_without_it`, `test_an_opening_train_loop_self_plays_and_gates_with_its_model` and `test_a_train_loop_set_up_without_an_opening_model_refuses_one`.
2. The loop's copy `nets/opening.bin` replaced or deleted after set-up (someone retrains the model in place, or copies a nets directory by hand). Expected: the loop stops before any self-play or gate with an error naming the file and the sha256 it was set up with; nothing is played with the wrong model. Pinned by Task 2's `test_a_changed_opening_model_in_the_nets_directory_stops_the_loop`.
3. A model that scores spots exactly the same (a zero model: every spot 1/4). Expected: the tie goes to engine order, so the greedy pick is the first legal settlement and every exploratory pick one of the first five. Pinned by Task 1's `an_opening_model_picks_every_setup_settlement_from_its_top_spots_unrecorded` (its tied case).
4. `--opening` with `--gumbel`, through any entry point (`fast-loop`, `FastLoop.init`, `Loop`, `write_explore`, `selfplay_explore`). Expected: a ValueError (exit 2 from the CLI) before any file or directory is written. Pinned by Task 1's `opening_model_self_play_refuses_gumbel_self_play` and `test_opening_self_play_refuses_gumbel_and_bad_model_files`, Task 2's `test_a_bad_opening_model_or_one_with_gumbel_is_refused_before_anything_is_written` and `test_an_opening_train_loop_self_plays_and_gates_with_its_model`.
5. A bad model file at set-up: missing, cut short, of other features (hash). Expected: refused before anything is written, by the loader every `+o` bot uses, with an error naming the file. Pinned by Task 1's `test_opening_self_play_refuses_gumbel_and_bad_model_files`, Task 2's `test_a_bad_opening_model_or_one_with_gumbel_is_refused_before_anything_is_written` and the `{"opening": "missing-opening.bin"}` case of `test_bad_settings_are_an_error_before_anything_is_written`.

## File Structure

| File | Responsibility | Task |
|---|---|---|
| `bots/src/opening_model.rs` | `settlement_scores` (every legal spot's own-win probability, engine order); `choose_settlement` built on it, unchanged in behaviour | 1 |
| `bots/src/selfplay.rs` | `OPENING_EPSILON`, `OPENING_TOP`, `explore_settlement`, `play_with_opening`, `run_with_opening` (refuses Gumbel); `play_with` / `run_with` delegate with `None` | 1 |
| `bindings/src/lib.rs` | `selfplay_explore(..., opening=None)`; constants `OPENING_EPSILON`, `OPENING_TOP` | 1 |
| `alphasettler/selfplay.py` | `OPENING_WITH_GUMBEL`, `check_opening`, `settings(..., opening=)`, `write_explore(..., opening=)` (1); `opening_mismatch` (2) | 1, 2 |
| `alphasettler/train/fastloop.py` | `init(..., opening=)`, `FastLoop.opening`, `_opening_file`, `+o` gates, `"opening"` in `state.json` and each generation's `selfplay` | 2 |
| `alphasettler/train/loop.py` | `LoopSettings.opening`, `_check`, `Loop.opening`, `_opening_model`, `_place_opening`, `+o` gates, `"opening"` in `manifest.json` and in plans when set | 2 |
| `alphasettler/cli.py` | `fast-loop --opening FILE`, `train-loop --opening FILE` | 2 |
| `docs/bot/results-3d1b.md` (new) | the measurement | 3 |
| tests: `bots/tests/selfplay.rs` (1), `bots/tests/opening_model.rs` (1), `tests/python/opening_games.py` (1, helper), `tests/python/test_training_data.py` (1), `tests/python/test_fast_loop.py` (2), `tests/python/test_train_loop.py` (2) | | 1–2 |

## Where the code forces a choice the spec does not make

Each is a **Ruling** (with its reason) the results file repeats, so it can be overridden.

- **The model is a parameter of new entry points, not a field of `Exploration`.** `Exploration` is `Copy` and its literals use struct update (`..ex`) throughout the tests; a model is not `Copy`. `play_with_opening(..., opening: Option<&OpeningModel>)` and `run_with_opening(..., threads, opening)` hold the game loop; `play_with` and `run_with` keep their signatures and call them with `None`, so every caller and every pinned digest is unchanged. ε and K are constants, as `SAMPLE_TURNS` is.
- **A model pick's draws.** At a setup settlement with a model, the exploration stream gives one `unit` (explore if below ε) and, when exploring, one `below(min(K, spots))`; the full-search, noise and sampling draws of a searched move are not made for that move (no search runs). Without a model the stream is drawn exactly as before.
- **The ranking is a stable sort by score, best first,** so equal scores keep engine order and the greedy pick is exactly `choose_settlement`'s spot; both read `settlement_scores`, so `+o` and self-play cannot disagree on a score. Exploratory picks draw from the top K including the best, so about 1 − ε + ε/K = 0.8 of picks are the best spot.
- **The skipped seat's bot still gets its events** (the feed delivers them before the pick), so its belief stays live; it simply does not decide, as a `+o` bot's `act` does not search.
- **`--opening` with `--gumbel` is refused in Rust** (`run_with_opening`, so `selfplay_explore` too) **and in Python** (`check_opening`), at set-up and in `write_explore`.
- **Records name the model.** With a model, the `selfplay` block gains `opening_sha256`, `opening_epsilon` and `opening_top`; without one it is unchanged, so plain records stay byte-identical.
- **Loops keep the model by its sha256** under `"opening"` in `state.json` (`fast-loop`) and `manifest.json` (`train-loop`), null without one; files from before 3d-1b have no key and read as without. A resume without `--opening` keeps the model; `--opening` with the same file (same sha256) is accepted; another file, or `--opening` on a loop set up without one, is refused (exit 2). This is how the spec's "a loop set up without it refuses it on resume (and the reverse)" is read: the reverse is a loop set up with a model refusing another model; it is never resumed without its model, because a resume without the flag keeps it.
- **The copy is checked before use.** `fast-loop` checks `nets/opening.bin` against the stored sha256 before every self-play and gate; `train-loop` copies it on its first `run` and checks it at the start of every `run`. A missing or changed copy stops the loop with an error naming the file.
- **Gates.** `fast-loop`: `net:gen-NNN@SIMS+t+o` against `net:BEST@SIMS+t+o` in the trading game (`+t` first, as the spec writes `net:C@100+t+o`); `train-loop` (the no-trade game): `net:gen-NNN@SIMS+o` against `net:gen-BEST@SIMS+o`.
- **`train-loop`'s warm start uses the model too**: generation 0's heuristic self-play also has its setup settlements from the model (the spec: self-play's setup settlements are chosen by the model; the model does not depend on the net). Its per-generation `selfplay.json` plan gains `"opening"` only when set, so runs started before 3d-1b still match their plans.
- **No `--opening` for `alphasettler selfplay`**: the spec names only the loops.
- **Arm C runs through `runs/p2/arm.sh` itself, unchanged** (`bash runs/p2/arm.sh C --opening runs/3c/nets/opening.bin`): the script's extra arguments go to the untimed set-up, so C's settings and seeds are arm A's by construction, and its outputs (`runs/p2/C`, `runs/p2/C.log`, `runs/p2/C-state-at-cut.json`) sit beside A's for `runs/p2/tables.py A C`. Arm A is not rerun: it is P2's recorded control, cut after 10,809 s.
- **The SPRT's nets directory is `runs/p2/nets`**, which already holds `A.bin` (`gen-040`); `C.bin` (C's best at the cut) and `opening.bin` (a copy of `runs/3c/nets/opening.bin`, which every `+o` name loads beside the net) are copied in.
- **Fresh seeds** for the SPRT start at 4,000,000,000 (at most 400 seeds): no self-play (below 1e9), gate (1e9 + 0..200) or SPRT so far (2e9.., 3e9..) used them.
- **H1 makes `--opening` the setting new loops are set up with**; no CLI default changes in this plan, since a resumed loop keeps its setting anyway. H0 or undecided: recorded, and the loop without it stays the default.
- **Task boundaries:** the self-play choice and what records carry (1), the loops (2), the run (3). Records (1) and loops (2) are separate gates: a reviewer can accept the choice rule and still reject how a loop keeps its model.

---

### Task 1: Self-play's setup settlements from the opening model

**Files:**
- Modify: `bots/src/opening_model.rs` (`settlement_scores`, `choose_settlement`), `bots/src/selfplay.rs` (module doc, imports, constants, `explore_settlement`, `play_with`, `play_with_opening`, `run_with`, `run_with_opening`), `bindings/src/lib.rs` (`selfplay_explore`, two constants), `alphasettler/selfplay.py` (docstring, imports, `OPENING_WITH_GUMBEL`, `check_opening`, `settings`, `write_explore`)
- Test: `bots/tests/selfplay.rs`, `bots/tests/opening_model.rs`, `tests/python/opening_games.py` (helper), `tests/python/test_training_data.py`

**Interfaces:**
- Consumes: `settler_bots::opening_model::{OpeningModel, OpeningLayer, choose_settlement}`, `settler_bots::opening::{opening_features, OPENING_FEATURES}`, `settler_search::unit`, `settler_engine::rng::Rng::below`; Python `alphasettler._engine.opening_forward`, `alphasettler.opening.write_model`.
- Produces:
  - `settler_bots::opening_model::settlement_scores(model: &OpeningModel, obs: &Observation, legal: &[Action]) -> Vec<(Action, f32)>` (every `BuildSettlement` of `legal`, in `legal`'s order, with the viewer's own-win probability).
  - `settler_bots::selfplay::{OPENING_EPSILON: f64 = 0.25, OPENING_TOP: usize = 5}`; `explore_settlement(model: &OpeningModel, obs: &Observation, legal: &[Action], rng: &mut Rng) -> Action`; `play_with_opening(seed: u64, config: GameConfig, ex: &Exploration, batch: usize, make_eval: &F, opening: Option<&OpeningModel>) -> SelfPlayGame`; `run_with_opening(seeds: Range<u64>, config: GameConfig, ex: &Exploration, batch: usize, make_eval: &F, threads: usize, opening: Option<&OpeningModel>) -> Result<Vec<SelfPlayGame>, String>` (an error containing `Gumbel` when `ex.gumbel` and a model).
  - `_engine.selfplay_explore(seed_start, games, threads, full_simulations, fast_simulations, full_probability, config=None, net=None, nets_dir=None, gumbel=None, opening=None)` (`opening`: a path to an `ASOPEN` file); `_engine.OPENING_EPSILON` (float 0.25), `_engine.OPENING_TOP` (int 5).
  - `alphasettler.selfplay.OPENING_WITH_GUMBEL: str` (starts `--opening and --gumbel cannot be used together`); `check_opening(path, gumbel: int | None = None) -> str` (the file's sha256 after the Rust loader read it; ValueError otherwise); `settings(net, nets_dir, full_simulations, fast_simulations, full_probability, gumbel=None, opening=None) -> dict`; `write_explore(..., gumbel=None, opening: str | Path | None = None) -> dict`.
  - Test helper `tests/python/opening_games.py::opening_model_file(path: Path, scale: float = 0.02) -> Path` (a valid 78 → 4 model; `scale=0.0` writes a zero model).

- [ ] **Step 1: Write the failing Rust tests**

In `bots/tests/selfplay.rs`, replace the import block

```rust
use settler_bots::selfplay::{
    play, play_with, run, run_with, Exploration, SelfPlayGame, FAST_SIMULATIONS, FULL_PROBABILITY,
    FULL_SIMULATIONS, MIN_SIMULATIONS, NOISE, SAMPLE_TURNS,
};
```

with

```rust
use settler_bots::opening::OPENING_FEATURES;
use settler_bots::opening_model::{settlement_scores, OpeningLayer, OpeningModel};
use settler_bots::selfplay::{
    play, play_with, play_with_opening, run, run_with, run_with_opening, Exploration,
    SelfPlayGame, FAST_SIMULATIONS, FULL_PROBABILITY, FULL_SIMULATIONS, MIN_SIMULATIONS, NOISE,
    OPENING_EPSILON, OPENING_TOP, SAMPLE_TURNS,
};
```

and add at the end of the file:

```rust
/// A 78 -> 16 -> 4 opening model with small weights drawn from `seed` (as tests/opening_model.rs
/// builds them): its spots score apart.
fn opening_model(seed: u64) -> OpeningModel {
    let mut rng = Rng::new(seed);
    let mut layers = Vec::new();
    for (cols, rows) in [(OPENING_FEATURES, 16), (16, NUM_PLAYERS)] {
        let mut x = || (rng.next_u64() >> 40) as f32 / (1u64 << 24) as f32 * 0.2 - 0.1;
        let w = (0..rows * cols).map(|_| x()).collect();
        let b = (0..rows).map(|_| x()).collect();
        layers.push(OpeningLayer { rows, cols, w, b });
    }
    OpeningModel { layers }
}

/// Every weight 0: every spot scores 1/4, a tie broken by engine order.
fn tied_model() -> OpeningModel {
    OpeningModel {
        layers: vec![OpeningLayer {
            rows: NUM_PLAYERS,
            cols: OPENING_FEATURES,
            w: vec![0.0; NUM_PLAYERS * OPENING_FEATURES],
            b: vec![0.0; NUM_PLAYERS],
        }],
    }
}

/// The model's settlements in `s` (the actor's view), best first, equal scores in engine order.
fn ranked_spots(model: &OpeningModel, s: &State) -> Vec<Action> {
    let legal = s.legal_actions();
    let mut scored = settlement_scores(model, &s.observation(s.current_actor()), &legal);
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    scored.into_iter().map(|(a, _)| a).collect()
}

/// With an opening model (spec 3d-1b Section 2) self-play's setup settlements are the model's:
/// none is searched or recorded, each is one of its OPENING_TOP best spots, about four in five
/// are its best one, and the setup roads are still searched and recorded.
#[test]
fn an_opening_model_picks_every_setup_settlement_from_its_top_spots_unrecorded() {
    assert_eq!((OPENING_EPSILON, OPENING_TOP), (0.25, 5));
    for (what, model) in [("random", opening_model(7)), ("tied", tied_model())] {
        let (mut picks, mut off_best, mut road_decisions) = (0, 0, 0);
        for seed in 0..6 {
            let g = play_with_opening(seed, trading(), &small_explore(), 8, &heuristic, Some(&model));
            assert_eq!(g.belief_resets, 0, "{what} seed {seed}");
            let mut s = State::new(seed, trading());
            let mut decisions = g.decisions.iter().peekable();
            for (i, &a) in g.actions.iter().enumerate() {
                let decision = decisions.next_if(|d| d.index as usize == i);
                match s.phase {
                    Phase::SetupSettlement => {
                        assert!(
                            decision.is_none(),
                            "{what} seed {seed} move {i}: a setup settlement was recorded"
                        );
                        let ranked = ranked_spots(&model, &s);
                        let top = &ranked[..ranked.len().min(OPENING_TOP)];
                        assert!(top.contains(&a), "{what} seed {seed} move {i}: {a:?} not in {top:?}");
                        if what == "tied" {
                            // Ties go to engine order: the first five legal settlements.
                            assert!(s.legal_actions()[..OPENING_TOP].contains(&a), "seed {seed} move {i}");
                        }
                        picks += 1;
                        off_best += (a != ranked[0]) as u32;
                    }
                    Phase::SetupRoad { .. } => road_decisions += decision.is_some() as u32,
                    _ => {}
                }
                s.apply(a);
            }
        }
        assert_eq!(picks, 6 * 8, "{what}");
        // 1 - ε + ε / K = 0.8 of the 48 picks are expected on the best spot.
        assert!(
            off_best > 0 && off_best < picks / 2,
            "{what}: {off_best} of {picks} picks off the best spot"
        );
        assert!(road_decisions > 0, "{what}: setup roads are searched and recorded");
    }
}

#[test]
fn opening_model_self_play_is_deterministic_and_thread_independent() {
    let model = opening_model(7);
    let ex = small_explore();
    let a = run_with_opening(0..4, trading(), &ex, 8, &heuristic, 1, Some(&model)).unwrap();
    assert_eq!(
        a,
        run_with_opening(0..4, trading(), &ex, 8, &heuristic, 3, Some(&model)).unwrap()
    );
    assert_eq!(
        a[2],
        play_with_opening(2, trading(), &ex, 8, &heuristic, Some(&model))
    );
    let plain = run_with(0..4, trading(), &ex, 8, &heuristic, 1).unwrap();
    assert!(
        a.iter().zip(&plain).any(|(x, y)| x.actions[..16] != y.actions[..16]),
        "the model changes the openings"
    );
}

#[test]
fn opening_model_self_play_refuses_gumbel_self_play() {
    let model = opening_model(7);
    let ex = Exploration::gumbel(8);
    let e = run_with_opening(0..1, no_trades(), &ex, 8, &heuristic, 1, Some(&model)).unwrap_err();
    assert!(e.contains("Gumbel"), "{e}");
    assert!(run_with_opening(0..1, no_trades(), &ex, 8, &heuristic, 1, None).is_ok());
}

/// Without a model `play_with_opening` is `play_with`: the 3a and P2 digests hold through it.
#[test]
fn self_play_without_an_opening_model_plays_the_games_it_played_before_3d1b() {
    assert_eq!(
        digest(&play_with_opening(2, trading(), &small_explore(), 8, &heuristic, None)),
        0x94f447a9c3baf5b2
    );
    assert_eq!(
        digest(&play_with_opening(3, no_trades(), &Exploration::fixed(10), 8, &heuristic, None)),
        0x517b49ad6e41ec1e
    );
}
```

In `bots/tests/opening_model.rs`, after `a_model_that_ties_picks_the_first_legal_spot`, add:

```rust
#[test]
fn the_scores_cover_every_legal_spot_in_engine_order_and_the_plus_o_spot_is_the_first_best() {
    let m = model(&[16], 3);
    for seed in 0..4 {
        let s = State::new(seed, trading());
        let legal = s.legal_actions();
        let obs = s.observation(0);
        let scores = settlement_scores(&m, &obs, &legal);
        assert_eq!(scores.iter().map(|&(a, _)| a).collect::<Vec<_>>(), legal);
        assert!(scores.iter().all(|&(_, p)| p > 0.0 && p < 1.0), "seed {seed}");
        let best = scores.iter().map(|&(_, p)| p).fold(f32::NEG_INFINITY, f32::max);
        let first_best = scores.iter().find(|&&(_, p)| p == best).unwrap().0;
        assert_eq!(choose_settlement(&m, &obs, &legal), first_best, "seed {seed}");
    }
}
```

- [ ] **Step 2: Run the Rust tests to verify they fail**

Run: `PATH=$HOME/.cargo/bin:$PATH cargo test --release -p settler-bots --test selfplay --test opening_model`
Expected: FAIL to compile with `unresolved imports` for `settlement_scores`, `play_with_opening`, `run_with_opening`, `OPENING_EPSILON`, `OPENING_TOP`.

- [ ] **Step 3: Write `settlement_scores`**

In `bots/src/opening_model.rs`, replace `choose_settlement` (its doc comment and body) with:

```rust
/// Each legal setup settlement with the model's own-win probability for the viewer, in `legal`'s
/// (engine) order: the spot placed on a copy of the position and scored from the viewer's seat
/// (spec 3d-1 Section 4.5). Actions other than settlements are left out. `+o`
/// (`choose_settlement`) and self-play's `selfplay::explore_settlement` both score with this.
pub fn settlement_scores(
    model: &OpeningModel,
    obs: &Observation,
    legal: &[Action],
) -> Vec<(Action, f32)> {
    legal
        .iter()
        .filter_map(|&a| {
            let Action::BuildSettlement(n) = a else {
                return None;
            };
            let mut placed = obs.settlements;
            placed[obs.viewer as usize] |= 1u64 << n;
            let p = model.probabilities(&opening_features(&obs.board, &placed, obs.viewer))[0];
            Some((a, p))
        })
        .collect()
}

/// The setup settlement a `+o` bot plays (spec 3d-1 Section 4.5): the highest own-win
/// probability of `settlement_scores`; ties go to the first in `legal` (engine order). No search
/// runs.
pub fn choose_settlement(model: &OpeningModel, obs: &Observation, legal: &[Action]) -> Action {
    let mut best: Option<(Action, f32)> = None;
    for (a, p) in settlement_scores(model, obs, legal) {
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

- [ ] **Step 4: Write the self-play choice**

In `bots/src/selfplay.rs`, append to the module doc (after the P2 lines):

```rust
//! 3d-1b: with an opening model (`play_with_opening`, `run_with_opening`) the model picks every
//! setup settlement (`explore_settlement`: ε-greedy over its best spots), with no search and no
//! recorded decision; without one, self-play is unchanged.
```

Replace the imports

```rust
use crate::Bot;
use settler_engine::legal::legal_actions;
use settler_engine::rng::{mix, Rng};
use settler_engine::{Action, GameConfig, PlayerId, State, NUM_PLAYERS};
```

with

```rust
use crate::opening_model::{settlement_scores, OpeningModel};
use crate::Bot;
use settler_engine::legal::legal_actions;
use settler_engine::rng::{mix, Rng};
use settler_engine::{Action, GameConfig, Observation, Phase, PlayerId, State, NUM_PLAYERS};
```

After `pub const SAMPLE_TURNS: u32 = 8;` add:

```rust
/// 3d-1b (spec Section 2): with an opening model, the chance a setup settlement is drawn from
/// the model's `OPENING_TOP` best spots instead of being its best spot.
pub const OPENING_EPSILON: f64 = 0.25;
pub const OPENING_TOP: usize = 5;
```

After the `impl Exploration { ... }` block add:

```rust
/// The setup settlement self-play plays with an opening model (spec 3d-1b Section 2): every legal
/// settlement scored as `+o` scores it (`settlement_scores`); with probability
/// `1 - OPENING_EPSILON` the best (ties to the first in engine order: `choose_settlement`'s
/// spot), else one drawn uniformly from the `OPENING_TOP` best (all of them if fewer are legal).
/// Both draws come from `rng`, the game's exploration stream.
pub fn explore_settlement(
    model: &OpeningModel,
    obs: &Observation,
    legal: &[Action],
    rng: &mut Rng,
) -> Action {
    let mut scored = settlement_scores(model, obs, legal);
    // A stable sort, best first: equal scores keep engine order.
    scored.sort_by(|a, b| b.1.total_cmp(&a.1));
    let explore = unit(rng) < OPENING_EPSILON;
    match scored.len() {
        0 => legal[0],
        n if explore => scored[rng.below(n.min(OPENING_TOP) as u32) as usize].0,
        _ => scored[0].0,
    }
}
```

Replace `play_with` (its doc comment, signature and whole body) with:

```rust
/// One self-play game; `make_eval` builds each seat's evaluator, `batch` is the leaf batch.
pub fn play_with<E, F>(
    seed: u64,
    config: GameConfig,
    ex: &Exploration,
    batch: usize,
    make_eval: &F,
) -> SelfPlayGame
where
    E: Evaluator + Send,
    F: Fn() -> E,
{
    play_with_opening(seed, config, ex, batch, make_eval, None)
}

/// `play_with`, and with `opening` the model picks every setup settlement
/// (`explore_settlement`, spec 3d-1b Section 2): no search runs and no decision is recorded for
/// those moves. Without a model this is `play_with` exactly.
pub fn play_with_opening<E, F>(
    seed: u64,
    config: GameConfig,
    ex: &Exploration,
    batch: usize,
    make_eval: &F,
    opening: Option<&OpeningModel>,
) -> SelfPlayGame
where
    E: Evaluator + Send,
    F: Fn() -> E,
{
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
    let mut bots: Vec<IsmctsBot<E>> = (0..NUM_PLAYERS)
        .map(|p| IsmctsBot::with_evaluator("selfplay", bot_seed(seed, p), cfg, make_eval()))
        .collect();
    let mut rng = Rng::new(mix(seed, EXPLORE_SALT));
    let mut s = State::new(seed, config);
    let (mut feed, mut buf, mut step) = (EventFeed::default(), Vec::new(), Vec::new());
    let (mut actions, mut decisions) = (Vec::new(), Vec::new());
    while !s.is_over() {
        legal_actions(&s, &mut buf);
        let actor = s.current_actor() as usize;
        feed.deliver(actor, &mut bots[actor]);
        let a = match (s.phase, opening) {
            // 3d-1b: the model's pick, from this game's exploration stream. The seat's bot has
            // its events (above) but does not decide, so nothing is searched or recorded.
            (Phase::SetupSettlement, Some(model)) => {
                explore_settlement(model, &s.observation(actor as PlayerId), &buf, &mut rng)
            }
            _ => {
                let full = unit(&mut rng) < ex.full_probability;
                let opts = MoveOptions {
                    simulations: if full {
                        ex.full_simulations
                    } else {
                        ex.fast_simulations
                    },
                    noise: match ex.noise {
                        Some(n) if full => Some((n, rng.next_u64())),
                        _ => None,
                    },
                    sample: (s.turn < ex.sample_turns).then(|| rng.next_u64()),
                };
                let a = bots[actor].act_with(&s.observation(actor as PlayerId), &buf, &opts);
                if let Some(r) = bots[actor].take_last_search() {
                    decisions.push(Decision {
                        index: actions.len() as u32,
                        actor: actor as PlayerId,
                        legal: buf.clone(),
                        visits: r.visits,
                        full_search: full,
                        target: r.improved,
                    });
                }
                a
            }
        };
        assert!(
            buf.contains(&a),
            "self-play chose illegal action {a:?} in phase {:?}",
            s.phase
        );
        actions.push(a);
        step.clear();
        s.apply_with(a, None, &mut step);
        feed.push(&step);
    }
    SelfPlayGame {
        seed,
        actions,
        decisions,
        winner: s.winner(),
        vp: std::array::from_fn(|p| s.total_vp(p)),
        turns: s.turn,
        belief_resets: belief_resets(bots.iter().map(|b| b as &dyn Bot)),
    }
}
```

Replace `run_with` (its doc comment, signature and whole body) with:

```rust
/// Every seed in `seeds`, spread over `threads`; sorted by seed and independent of `threads`.
pub fn run_with<E, F>(
    seeds: Range<u64>,
    config: GameConfig,
    ex: &Exploration,
    batch: usize,
    make_eval: &F,
    threads: usize,
) -> Result<Vec<SelfPlayGame>, String>
where
    E: Evaluator + Send,
    F: Fn() -> E + Sync,
{
    run_with_opening(seeds, config, ex, batch, make_eval, threads, None)
}

/// `run_with` with `play_with_opening`'s `opening`. A model with Gumbel self-play is refused
/// (spec 3d-1b Section 3).
pub fn run_with_opening<E, F>(
    seeds: Range<u64>,
    config: GameConfig,
    ex: &Exploration,
    batch: usize,
    make_eval: &F,
    threads: usize,
    opening: Option<&OpeningModel>,
) -> Result<Vec<SelfPlayGame>, String>
where
    E: Evaluator + Send,
    F: Fn() -> E + Sync,
{
    config.validate()?;
    ex.validate()?;
    if ex.gumbel && opening.is_some() {
        return Err("the opening model is not used with Gumbel self-play: Gumbel self-play \
                    stays the P2 measurement it was (spec 3d-1b Section 3)"
            .into());
    }
    if batch == 0 {
        return Err("the leaf batch must be positive".into());
    }
    let n = seeds.end.saturating_sub(seeds.start);
    if n > MAX_SEEDS {
        return Err(format!("at most {MAX_SEEDS} games per call, got {n}"));
    }
    let mut games: Vec<SelfPlayGame> = std::thread::scope(|scope| {
        let workers: Vec<_> = split_seeds(seeds, threads)
            .into_iter()
            .map(|part| {
                scope.spawn(move || {
                    part.map(|seed| play_with_opening(seed, config, ex, batch, make_eval, opening))
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        workers
            .into_iter()
            .flat_map(|w| {
                w.join()
                    .unwrap_or_else(|panic| std::panic::resume_unwind(panic))
            })
            .collect()
    });
    games.sort_by_key(|g| g.seed);
    Ok(games)
}
```

(`play`, `run` and everything else in the file are unchanged.)

- [ ] **Step 5: Run the Rust tests to verify they pass**

Run: `PATH=$HOME/.cargo/bin:$PATH cargo test --release -p settler-bots --test selfplay --test opening_model`
Expected: PASS, including the 4 new self-play tests, the new opening-model test, `fixed_exploration_is_3a_self_play`, `standard_exploration_plays_the_games_it_played_before_p2` and every `+o` test.

- [ ] **Step 6: Write the failing Python tests**

In `tests/python/opening_games.py`, replace

```python
from alphasettler import records
from alphasettler._engine import resolve_config
from alphasettler.arena import TRADES
```

with

```python
from alphasettler import records
from alphasettler._engine import OPENING_FEATURES, resolve_config
from alphasettler.arena import TRADES
from alphasettler.opening import write_model
```

and add at the end:

```python
def opening_model_file(path: Path, scale: float = 0.02) -> Path:
    """A one-layer opening model (78 features -> 4 seats) with fixed weights times `scale`, written
    through the Rust loader: its spots score apart, so its picks are not engine order. `scale=0.0`
    writes a zero model (every spot ties)."""
    w = [((i * 37) % 11 - 5) * scale for i in range(4 * OPENING_FEATURES)]
    write_model(path, [(4, OPENING_FEATURES, w, [0.0] * 4)])
    return Path(path)
```

In `tests/python/test_training_data.py`, replace the `_engine` import with

```python
from alphasettler._engine import (
    NET_BATCH, NET_HIDDEN_LEN, NET_PUBLIC_LEN, NOISE_ALPHA, NOISE_WEIGHT, OPENING_EPSILON, OPENING_TOP, SAMPLE_TURNS,
    SEARCH_BATCH, TrainingSet, action_space_size, encode, net_forward, resolve_net, selfplay_explore,
    write_random_net,
)
```

add `from opening_games import opening_model_file` after `from alphasettler.selfplay import write_explore`, and after `test_gumbel_self_play_needs_at_least_two_simulations` add:

```python
def test_opening_self_play_records_no_setup_settlement_and_names_its_model(tmp_path):
    model = opening_model_file(tmp_path / "opening.bin")
    games = explore_games(tmp_path, 2, config=TRADING, opening=model)
    plain = explore_games(tmp_path, 2, config=TRADING, name="plain.jsonl.gz")
    assert (OPENING_EPSILON, OPENING_TOP) == (0.25, 5)
    for g, p in zip(games, plain):
        assert g["selfplay"] == {**p["selfplay"], "opening_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
                                 "opening_epsilon": OPENING_EPSILON, "opening_top": OPENING_TOP}
        assert g["belief_resets"] == 0 and records.replay_mismatches(g) == []
        recorded = {d["index"] for d in g["decisions"]}
        replay = Game(g["seed"], g["config"])
        settlements = roads = 0
        for i, a in enumerate(g["actions"]):
            if replay.phase == "setup_settlement":
                assert i not in recorded, f"game {g['seed']} move {i}: a setup settlement was recorded"
                settlements += 1
            elif replay.phase == "setup_road":
                roads += i in recorded
            replay.apply(a)
        assert settlements == 8 and roads > 0
    assert any(g["actions"][:16] != p["actions"][:16] for g, p in zip(games, plain)), "the model opens the games"


def test_opening_self_play_refuses_gumbel_and_bad_model_files(tmp_path):
    model = opening_model_file(tmp_path / "opening.bin")
    bad = tmp_path / "bad.bin"
    bad.write_bytes(model.read_bytes()[:-4])
    for kw, message in [({"opening": model, "gumbel": 6}, "--opening and --gumbel"),
                        ({"opening": bad}, "opening model .*bad.bin.*truncated"),
                        ({"opening": tmp_path / "missing.bin"}, "cannot read opening model")]:
        with pytest.raises(ValueError, match=message):
            write_explore(tmp_path / "no.jsonl.gz", 0, 1, config=CONFIG, full_simulations=6, fast_simulations=3,
                          full_probability=0.25, **kw)
        assert not (tmp_path / "no.jsonl.gz").exists()
    with pytest.raises(ValueError, match="Gumbel self-play"):  # the native check, below the Python one
        selfplay_explore(0, 1, 1, 6, 3, 0.25, CONFIG, None, None, 6, str(model))
```

- [ ] **Step 7: Write the binding**

In `bindings/src/lib.rs`, replace `selfplay_explore`'s doc comment, signature attribute and parameter list, and its `use` lines, with:

```rust
/// 3b self-play (spec Section 4): playout cap randomization (`full_simulations` with
/// probability `full_probability`, else `fast_simulations`), root noise on full searches and
/// opening sampling. With `gumbel` (a simulation count), every move is instead a Gumbel root
/// search of that many simulations, recorded with its improved policy as `target` (spec P2
/// Section 5); the three PUCT arguments are then not used. With `opening` (an `ASOPEN` file),
/// the opening model picks every setup settlement, epsilon-greedy over its best spots, with no
/// search and no recorded decision (spec 3d-1b Section 2); refused with `gumbel`. With `net`,
/// every seat is the network `<nets_dir>/<net>.bin` (loaded once); otherwise the heuristic.
/// Decisions carry no observations. Releases the GIL while games run.
#[pyfunction]
#[pyo3(signature = (seed_start, games, threads, full_simulations, fast_simulations, full_probability, config=None, net=None, nets_dir=None, gumbel=None, opening=None))]
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
    opening: Option<PathBuf>,
) -> PyResult<Bound<'py, PyList>> {
    use settler_bots::heuristic::{HeuristicEvaluator, DEFAULT_WEIGHTS};
    use settler_bots::netbot::{load_net, DEFAULT_NETS_DIR, NET_BATCH};
    use settler_bots::opening_model::OpeningModel;
    use settler_bots::selfplay::{run_with_opening, Exploration};
    use settler_net::evaluator::NetEvaluator;
```

after the `let weights = match net { ... };` statement add:

```rust
    let model = match &opening {
        Some(path) => Some(OpeningModel::load(path).map_err(value_error)?),
        None => None,
    };
```

and replace the `let played = py.detach(...)` statement with:

```rust
    let played = py
        .detach(|| match &weights {
            None => run_with_opening(
                seed_start..end,
                config,
                &ex,
                settler_search::SearchConfig::default().batch,
                &|| HeuristicEvaluator::new(DEFAULT_WEIGHTS, 0),
                threads,
                model.as_ref(),
            ),
            Some(w) => run_with_opening(
                seed_start..end,
                config,
                &ex,
                NET_BATCH,
                &|| NetEvaluator::new(w.clone()),
                threads,
                model.as_ref(),
            ),
        })
        .map_err(value_error)?;
```

In `_engine` (the `#[pymodule]`), after `m.add("SAMPLE_TURNS", settler_bots::selfplay::SAMPLE_TURNS)?;` add:

```rust
    m.add("OPENING_EPSILON", settler_bots::selfplay::OPENING_EPSILON)?;
    m.add("OPENING_TOP", settler_bots::selfplay::OPENING_TOP)?;
```

- [ ] **Step 8: Write the Python implementation**

In `alphasettler/selfplay.py`, replace the module docstring's last sentence `the block then holds `gumbel_simulations` instead of the exploration settings."""` with:

```python
the block then holds `gumbel_simulations` instead of the exploration settings. With `opening`
(3d-1b), an opening model file picks every setup settlement, epsilon-greedy over its best spots, with
no search and no decision; the block then also holds `opening_sha256`, `opening_epsilon` and
`opening_top`."""
```

Replace the `_engine` import with:

```python
from alphasettler._engine import (
    DEFAULT_NETS_DIR, MAX_SIMULATIONS, NET_BATCH, NOISE_ALPHA, NOISE_WEIGHT, OPENING_EPSILON, OPENING_TOP, SAMPLE_TURNS,
    SEARCH_BATCH, SELFPLAY_MIN_SIMULATIONS, opening_forward, resolve_net, selfplay_explore,
)
```

After `BATCH_GAMES = 100` add:

```python
OPENING_WITH_GUMBEL = ("--opening and --gumbel cannot be used together: Gumbel self-play stays the P2 measurement "
                       "it was (spec 3d-1b Section 3)")
```

After `check_gumbel` add:

```python
def check_opening(path: str | Path, gumbel: int | None = None) -> str:
    """The sha256 of the opening model file `path`, once the Rust loader every `+o` bot uses has read
    it (its ValueError, naming the file, otherwise). With `gumbel` it is a ValueError: the opening
    model is not used with Gumbel self-play."""
    if gumbel is not None:
        raise ValueError(OPENING_WITH_GUMBEL)
    opening_forward(str(path), [])
    return files.sha256_file(path)
```

Replace `settings` with:

```python
def settings(net: str | None, nets_dir: str | Path, full_simulations: int, fast_simulations: int,
             full_probability: float, gumbel: int | None = None, opening: str | Path | None = None) -> dict:
    """The `selfplay` block. Checks every argument, loads the net and reads the opening model,
    without playing a game. With `gumbel`, the PUCT arguments are not used; `opening` with `gumbel`
    is a ValueError."""
    opening_sha256 = None if opening is None else check_opening(opening, gumbel)
    if gumbel is not None:
        check_gumbel(gumbel)
    else:
        for what, n in (("full", full_simulations), ("fast", fast_simulations)):
            if not SELFPLAY_MIN_SIMULATIONS <= n <= MAX_SIMULATIONS:
                raise ValueError(
                    f"{what} simulations must be {SELFPLAY_MIN_SIMULATIONS}..={MAX_SIMULATIONS}, got {n}")
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
    block = {
        **who,
        "full_simulations": full_simulations,
        "fast_simulations": fast_simulations,
        "full_probability": full_probability,
        "noise_weight": NOISE_WEIGHT,
        "noise_alpha": NOISE_ALPHA,
        "sample_turns": SAMPLE_TURNS,
        "batch": batch,
    }
    if opening_sha256 is not None:  # only then: records without a model are unchanged
        block.update(opening_sha256=opening_sha256, opening_epsilon=OPENING_EPSILON, opening_top=OPENING_TOP)
    return block
```

In `write_explore`, change the signature's last line `counts: dict | None = None, gumbel: int | None = None) -> dict:` to `counts: dict | None = None, gumbel: int | None = None, opening: str | Path | None = None) -> dict:`, the `meta = ...` line to `meta = settings(net, nets_dir, full_simulations, fast_simulations, full_probability, gumbel, opening)`, and the native call to:

```python
            batch = selfplay_explore(lo, min(BATCH_GAMES, end - lo), threads, full_simulations, fast_simulations,
                                     full_probability, config, net, str(nets_dir), gumbel,
                                     None if opening is None else str(opening))
```

- [ ] **Step 9: Run the Python tests to verify they pass**

Rebuild the extension, then run: `.venv/bin/pytest -q tests/python/test_training_data.py`
Expected: PASS, including the 2 new tests; `test_explore_records_carry_their_settings_and_no_observations` and `test_record_files_are_byte_identical_for_the_same_games` pass unchanged.

- [ ] **Step 10: Format, lint, the full suite, rebuild, pytest**

Run the Global Constraints sequence (fmt, fmt --check, clippy, `cargo test --workspace --release`, rebuild the extension, `.venv/bin/pytest -q -rs`). Expected: all pass, nothing skipped.

- [ ] **Step 11: Commit**

```bash
git add bots/src/opening_model.rs bots/src/selfplay.rs bindings/src/lib.rs alphasettler/selfplay.py \
  bots/tests/selfplay.rs bots/tests/opening_model.rs tests/python/opening_games.py tests/python/test_training_data.py
git diff --cached | grep -nE 'sk-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{36}|password=[^[:space:]]+'
git -c user.email=sean.may101@gmail.com commit -m "selfplay: an opening model picks self-play's setup settlements (epsilon-greedy over its top 5), unsearched and unrecorded"
```

---

### Task 2: `fast-loop --opening FILE` and `train-loop --opening FILE`

**Files:**
- Modify: `alphasettler/selfplay.py` (`opening_mismatch`), `alphasettler/train/fastloop.py` (docstring, import, `FastLoop.__init__`, `init`, `_opening_file`, `_generation`, `_selfplay`, `_gate`), `alphasettler/train/loop.py` (docstring, import, `LoopSettings`, `_check`, `Loop.__init__`, `_opening_model`, `run`, `_place_opening`, `_selfplay`, `_gate`), `alphasettler/cli.py` (both parsers, `_fast_loop`, `_train_loop`)
- Test: `tests/python/test_fast_loop.py`, `tests/python/test_train_loop.py`

**Interfaces:**
- Consumes: Task 1's `selfplay.check_opening(path, gumbel=None) -> str`, `selfplay.OPENING_WITH_GUMBEL`, `write_explore(..., opening=)`, the record keys `opening_sha256`, and the test helper `opening_games.opening_model_file(path, scale=0.02)`; `_engine.OPENING_FILE` (`"opening.bin"`, Plan 7).
- Produces:
  - `alphasettler.selfplay.opening_mismatch(root, stored: str | None, asked: str | Path | None) -> str | None` (`<root> was set up without --opening, and a loop keeps the opening model it was set up with`, or `... set up with another opening model (sha256 <hex>), ...`).
  - `FastLoop.init(root, seed_dirs, start_net, first, trades=False, gumbel=None, opening=None)` copies the file to `<root>/nets/opening.bin` and writes `"opening"` (the sha256, or null) into `state.json`; `FastLoop(...).opening` (None for a state file without the key); each generation's `selfplay` entry has `"opening"`; gates `net:gen-NNN@SIMS+t+o` / `net:BEST@SIMS+t+o` (`+o` alone without trades).
  - `LoopSettings.opening: str | None = None`; `Loop(...).opening` (the sha256 or None); `manifest.json` key `"opening"`; generation plans (`selfplay.json`) have `"opening"` only when set; gates `net:gen-NNN@SIMS+o`.
  - CLI: `fast-loop --opening FILE` (first run only) and `train-loop --opening FILE`; a mismatch on resume prints the `opening_mismatch` message after `error: ` and exits 2.

- [ ] **Step 1: Write the failing tests**

In `tests/python/test_fast_loop.py`, replace the import block's first lines

```python
import json

import pytest

from alphasettler import records
from alphasettler.arena import TRADES
```

with

```python
import hashlib
import json

import pytest

from alphasettler import records
from alphasettler._engine import OPENING_FILE
from alphasettler.arena import TRADES
```

add `from opening_games import opening_model_file` after `from test_train import cli, generation`, and add at the end:

```python
SETUP_SETTLEMENTS = {0, 2, 4, 6, 8, 10, 12, 14}  # the first 16 moves: settlement, road, eight times


def test_cli_fast_loop_with_an_opening_model_self_plays_and_gates_with_it(tmp_path):
    seed = generation(tmp_path, "seed-gen", seed_start=0)
    save(Net(8, 1), tmp_path / "start.bin")
    model = opening_model_file(tmp_path / "model.bin")
    digest = hashlib.sha256(model.read_bytes()).hexdigest()
    root = tmp_path / "loop"
    r = cli("fast-loop", "--dir", str(root), "--generations", "2", "--first", "1", "--trades", "--opening", str(model),
            "--seed-dirs", str(seed), "--start-net", str(tmp_path / "start.bin"), "--games", "2",
            "--window-games", "3", "--minutes", "0.01", "--gate-seeds", "1", "--gate-simulations", "6",
            "--device", "cpu", "--threads", "2")
    assert r.returncode == 0, r.stderr
    assert (root / "nets" / OPENING_FILE).read_bytes() == model.read_bytes()
    state = json.loads((root / STATE).read_text())
    assert state["opening"] == digest and [g["gen"] for g in state["generations"]] == [1, 2]
    for g in state["generations"]:
        assert g["selfplay"]["opening"] == digest and g["selfplay"]["belief_resets"] == 0
        assert g["gate"]["candidate"].endswith("@6+t+o") and g["gate"]["baseline"].endswith("@6+t+o"), g["gate"]
        assert g["gate"]["belief_resets"] == 0
    for k in (1, 2):
        paths = sorted((root / gen_name(k)).glob("selfplay-local-*.jsonl.gz"))
        assert paths
        for path in paths:
            for game in records.read(path):
                assert game["selfplay"]["opening_sha256"] == digest
                assert not SETUP_SETTLEMENTS & {d["index"] for d in game["decisions"]}


@pytest.mark.parametrize("set_up, message", [
    (False, "without --opening"),
    (True, "with another opening model"),
], ids=["set-up-without", "another-model"])
def test_cli_fast_loop_keeps_the_opening_model_it_was_set_up_with(tmp_path, set_up, message):
    seed = generation(tmp_path, "seed-gen", seed_start=0)
    save(Net(8, 1), tmp_path / "start.bin")
    model = opening_model_file(tmp_path / "model.bin")
    other = opening_model_file(tmp_path / "other.bin", scale=0.0)
    root = tmp_path / "loop"
    FastLoop.init(root, [seed], tmp_path / "start.bin", first=1, opening=model if set_up else None)
    want = hashlib.sha256(model.read_bytes()).hexdigest() if set_up else None
    assert FastLoop(root, FastLoopSettings(generations=0, **TINY)).opening == want
    r = cli("fast-loop", "--dir", str(root), "--generations", "0", "--opening", str(other if set_up else model))
    assert r.returncode == 2 and message in r.stderr, r.stderr
    r = cli("fast-loop", "--dir", str(root), "--generations", "0")
    assert r.returncode == 0, r.stderr
    assert json.loads((root / STATE).read_text())["opening"] == want, "a resume without the flag keeps it"
    if set_up:
        r = cli("fast-loop", "--dir", str(root), "--generations", "0", "--opening", str(model))
        assert r.returncode == 0, "the same file repeats the setting"


def test_a_loop_set_up_before_the_opening_model_existed_reads_as_set_up_without_it(tmp_path):
    root, _ = setup(tmp_path)
    state = json.loads((root / STATE).read_text())
    del state["opening"]
    (root / STATE).write_text(json.dumps(state))
    assert FastLoop(root, FastLoopSettings(generations=0, **TINY)).opening is None
    model = opening_model_file(tmp_path / "model.bin")
    r = cli("fast-loop", "--dir", str(root), "--generations", "0", "--opening", str(model))
    assert r.returncode == 2 and "without --opening" in r.stderr


def test_a_bad_opening_model_or_one_with_gumbel_is_refused_before_anything_is_written(tmp_path):
    seed = generation(tmp_path, "seed-gen", seed_start=0)
    save(Net(8, 1), tmp_path / "start.bin")
    model = opening_model_file(tmp_path / "model.bin")
    bad = tmp_path / "bad.bin"
    bad.write_bytes(model.read_bytes()[:-4])
    for kw, message in [({"opening": bad}, "truncated"), ({"opening": tmp_path / "missing.bin"}, "cannot read opening model"),
                        ({"opening": model, "gumbel": 100}, "--opening and --gumbel")]:
        with pytest.raises(ValueError, match=message):
            FastLoop.init(tmp_path / "bad-loop", [seed], tmp_path / "start.bin", first=1, **kw)
        assert not (tmp_path / "bad-loop").exists()
    r = cli("fast-loop", "--dir", str(tmp_path / "cli"), "--generations", "0", "--first", "1", "--seed-dirs", str(seed),
            "--start-net", str(tmp_path / "start.bin"), "--opening", str(model), "--gumbel", "100")
    assert r.returncode == 2 and "--opening and --gumbel" in r.stderr
    assert not (tmp_path / "cli").exists()


def test_a_changed_opening_model_in_the_nets_directory_stops_the_loop(tmp_path):
    seed = generation(tmp_path, "seed-gen", seed_start=0)
    save(Net(8, 1), tmp_path / "start.bin")
    root = tmp_path / "loop"
    FastLoop.init(root, [seed], tmp_path / "start.bin", first=1, opening=opening_model_file(tmp_path / "model.bin"))
    opening_model_file(root / "nets" / OPENING_FILE, scale=0.0)  # replaced in place after set-up
    with pytest.raises(ValueError, match="not the opening model"):
        FastLoop(root, FastLoopSettings(generations=1, **TINY)).run()
    assert not list((root / "gen-001").glob("selfplay-local-*.jsonl.gz")), "nothing was played with it"
```

In `tests/python/test_train_loop.py`, replace `from alphasettler._engine import NET_MAX_ROUNDS, NET_MAX_WIDTH` with `from alphasettler._engine import NET_MAX_ROUNDS, NET_MAX_WIDTH, OPENING_FILE`, add `from opening_games import opening_model_file` after the `alphasettler.train.loop` import, add `{"opening": "missing-opening.bin"}` to the `parametrize("change", [...])` list of `test_bad_settings_are_an_error_before_anything_is_written` (after `{"gumbel": 1}`), and add at the end:

```python
SETUP_SETTLEMENTS = {0, 2, 4, 6, 8, 10, 12, 14}


def test_an_opening_train_loop_self_plays_and_gates_with_its_model(tmp_path):
    model = opening_model_file(tmp_path / "model.bin")
    with pytest.raises(ValueError, match="--opening and --gumbel"):
        Loop(tmp_path / "both", replace(TINY, opening=str(model), gumbel=4))
    assert not (tmp_path / "both").exists()
    d = tmp_path / "o"
    s = replace(TINY, opening=str(model))
    assert Loop(d, s).run() == 6
    m = manifest(d)
    assert m["opening"] == sha256(model)
    assert (d / "nets" / OPENING_FILE).read_bytes() == model.read_bytes()
    g0, g1 = m["generations"]
    assert g0["selfplay"]["opening"] == g1["selfplay"]["opening"] == sha256(model)
    assert g1["gate"]["candidate"] == "net:gen-001@4+o" and g1["gate"]["baseline"].endswith("@4+o")
    for k in (0, 1):
        paths = sorted((d / f"gen-00{k}").glob("selfplay-local-*.jsonl.gz"))
        assert paths
        for path in paths:
            for g in records.read(path):
                assert g["selfplay"]["opening_sha256"] == sha256(model)
                assert not SETUP_SETTLEMENTS & {x["index"] for x in g["decisions"]}
    assert Loop(d, replace(s, opening=None)).opening == sha256(model), "a resume without the flag keeps it"
    assert Loop(d, replace(s, opening=None)).run() == 0
    other = opening_model_file(tmp_path / "other.bin", scale=0.0)
    with pytest.raises(ValueError, match="with another opening model"):
        Loop(d, replace(s, opening=str(other)))


def test_a_train_loop_set_up_without_an_opening_model_refuses_one(tmp_path, reference):
    model = opening_model_file(tmp_path / "model.bin")
    assert manifest(reference)["opening"] is None
    assert all("+o" not in manifest(reference)["generations"][1]["gate"][side] for side in ("candidate", "baseline"))
    with pytest.raises(ValueError, match="without --opening"):
        Loop(reference, replace(TINY, opening=str(model)))
    r = cli("train-loop", "--dir", str(reference), "--generations", "1", "--opening", str(model))
    assert r.returncode == 2 and "without --opening" in r.stderr
    # A run set up before 3d-1b has no "opening" key: it reads as set up without one.
    old = tmp_path / "before-3d1b"
    shutil.copytree(reference, old)
    m = manifest(old)
    del m["opening"]
    (old / "manifest.json").write_text(json.dumps(m))
    with pytest.raises(ValueError, match="without --opening"):
        Loop(old, replace(TINY, opening=str(model)))
    assert Loop(old, TINY).opening is None
    assert Loop(old, TINY).run() == 0, "resuming without the flag is accepted, nothing redone"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest -q tests/python/test_fast_loop.py tests/python/test_train_loop.py -k "opening"`
Expected: FAIL (`unexpected keyword argument 'opening'`, `unrecognized arguments: --opening`, `KeyError: 'opening'`).

- [ ] **Step 3: Write `opening_mismatch`**

In `alphasettler/selfplay.py`, after `mode_mismatch` add:

```python
def opening_mismatch(root: str | Path, stored: str | None, asked: str | Path | None) -> str | None:
    """The error for asking `--opening FILE` of a loop in `root` that was set up with the opening
    model whose sha256 is `stored` (None: without one, as every loop from before 3d-1b was), or None
    if the ask is allowed: no flag keeps the loop's model, the same file (by sha256) repeats it."""
    if asked is None:
        return None
    if files.sha256_file(asked) == stored:
        return None
    was = "without --opening" if stored is None else f"with another opening model (sha256 {stored})"
    return f"{root} was set up {was}, and a loop keeps the opening model it was set up with"
```

- [ ] **Step 4: Write the fast-loop implementation**

In `alphasettler/train/fastloop.py`, append to the module docstring (before its closing `"""`):

```python
With `opening` (set when the loop is set up: `init(..., opening=FILE)`, `fast-loop --opening FILE`;
spec 3d-1b), the opening model FILE is copied to `nets/opening.bin`, picks every self-play setup
settlement, and both gate sides are `+o`. A resumed loop keeps it (state.json holds its sha256, and
the copy is checked before every self-play and gate); it cannot be combined with `gumbel`.
```

After `from alphasettler import selfplay as selfplay_mod` add `from alphasettler._engine import OPENING_FILE`.

In `FastLoop.__init__`, after the `self.gumbel = ...` line add:

```python
        self.opening = self.state.get("opening")  # the model's sha256; state files from before 3d-1b have none
```

Replace `init` with:

```python
    @staticmethod
    def init(root: str | Path, seed_dirs: list[str | Path], start_net: str | Path, first: int,
             trades: bool = False, gumbel: int | None = None, opening: str | Path | None = None) -> None:
        """Create `root` with `seed_dirs` (oldest first) as the starting records and `start_net`
        as the best net; generation numbering starts at `first`; `trades` plays the trading game;
        `gumbel` (simulations) makes every self-play move a Gumbel root search; `opening` (an
        opening model file, copied to `nets/opening.bin`) picks self-play's setup settlements."""
        root = Path(root)
        if (root / STATE).exists():
            raise ValueError(f"{root / STATE} exists: the loop is already set up (resume it with fast-loop)")
        if gumbel is not None:
            selfplay_mod.check_gumbel(gumbel)
        opening_sha256 = None if opening is None else selfplay_mod.check_opening(opening, gumbel)
        nets = root / "nets"
        nets.mkdir(parents=True, exist_ok=True)
        files.atomic_copy(start_net, nets / "start.bin")
        files.atomic_copy(start_net, nets / "best.bin")
        if opening is not None:
            files.atomic_copy(opening, nets / OPENING_FILE)
        dirs = [str(Path(d).resolve()) for d in seed_dirs]
        state = {"seed_dirs": [{"dir": d, "games": count_games(Path(d))} for d in dirs],
                 "first": first, "trades": trades, "gumbel": gumbel, "opening": opening_sha256, "best": "start",
                 "latest": "start", "generations": []}
        files.atomic_write_text(root / STATE, json.dumps(state, indent=1) + "\n")
```

After `_save` add:

```python
    def _opening_file(self) -> Path | None:
        """`nets/opening.bin` if the loop was set up with an opening model, checked against the
        sha256 it was set up with (ValueError if missing or changed); None without one."""
        if self.opening is None:
            return None
        path = self.nets / OPENING_FILE
        if not path.is_file() or files.sha256_file(path) != self.opening:
            raise ValueError(f"{path} is not the opening model {self.root} was set up with (sha256 {self.opening}): "
                             f"restore that file")
        return path
```

In `_generation`, replace

```python
        mode = "" if self.gumbel is None else f" (Gumbel root, {self.gumbel} simulations)"
```

with

```python
        mode = "" if self.gumbel is None else f" (Gumbel root, {self.gumbel} simulations)"
        if self.opening is not None:
            mode += " (setup settlements from the opening model)"
```

In `_selfplay`, add `opening=self._opening_file()` to the `selfplay_mod.write_explore(...)` call (after `gumbel=self.gumbel`), and replace `return {"net": best, "seed_start": start, "gumbel": self.gumbel, **totals}` with `return {"net": best, "seed_start": start, "gumbel": self.gumbel, "opening": self.opening, **totals}`.

In `_gate`, replace

```python
        offers = "+t" if self.trades else ""
        candidate, baseline = f"net:{gen_name(k)}@{sims}{offers}", f"net:{best}@{sims}{offers}"
```

with

```python
        self._opening_file()  # a +o gate reads nets/opening.bin: check it first
        suffix = ("+t" if self.trades else "") + ("" if self.opening is None else "+o")
        candidate, baseline = f"net:{gen_name(k)}@{sims}{suffix}", f"net:{best}@{sims}{suffix}"
```

- [ ] **Step 5: Write the train-loop implementation**

In `alphasettler/train/loop.py`, replace the module docstring's last line `the best net, trains from generation k-1's weights and is gated against the best net."""` with:

```python
the best net, trains from generation k-1's weights and is gated against the best net.

With `opening` (`--opening FILE`, set when the run starts and kept on every resume, as `--gumbel`
is; spec 3d-1b), the opening model FILE is copied to `nets/opening.bin`, picks every self-play setup
settlement (generation 0's too), and both gate sides are `+o`. The manifest holds its sha256."""
```

Replace the `_engine` import with:

```python
from alphasettler._engine import (
    MAX_SEEDS, MAX_SIMULATIONS, NET_DEFAULT_ROUNDS, NET_DEFAULT_WIDTH, OPENING_FILE, resolve_config,
)
```

In `LoopSettings`, after `gumbel: int | None = None ...` add:

```python
    opening: str | None = None  # an opening model file, copied to nets/opening.bin when the run starts
```

In `_check`, after the `if s.gumbel is not None:` block add:

```python
    if s.opening is not None:
        selfplay_mod.check_opening(s.opening, s.gumbel)
```

In `Loop.__init__`, after `self.gumbel = self._mode(settings.gumbel)` add `self.opening = self._opening_model(settings.opening)`, and after `_mode` add:

```python
    def _opening_model(self, asked: str | None) -> str | None:
        """The opening model's sha256, or None for self-play without one. Set when the run starts
        (`--opening FILE`; `run` copies it to `nets/opening.bin`) and kept on every resume; a
        manifest from before 3d-1b has none. Asking for another model is a ValueError."""
        if not (self.root / MANIFEST).exists():
            digest = None if asked is None else files.sha256_file(asked)
            self.manifest["opening"] = digest
            return digest
        stored = self.manifest.get("opening")
        error = selfplay_mod.opening_mismatch(self.root, stored, asked)
        if error is not None:
            raise ValueError(error)
        self.manifest["opening"] = stored
        return stored

    def _place_opening(self) -> None:
        """Put the run's opening model at `nets/opening.bin` (copied from `--opening FILE` when it
        is not there) and check it is the model the run was set up with."""
        if self.opening is None:
            return
        path = self.nets / OPENING_FILE
        if not path.exists() and self.s.opening is not None:
            self.nets.mkdir(parents=True, exist_ok=True)
            files.atomic_copy(self.s.opening, path)
        if not path.is_file() or files.sha256_file(path) != self.opening:
            raise ValueError(f"{path} is not the opening model {self.root} was set up with (sha256 {self.opening}): "
                             f"restore it, or resume with --opening and that file")
```

In `run`, after `self._sync_best()` add `self._place_opening()`.

In `_selfplay`, after the `if self.gumbel is not None: ...` lines add:

```python
        if self.opening is not None:  # only then: plans written before 3d-1b still match
            plan["opening"] = self.opening
```

and add `opening=None if self.opening is None else self.nets / OPENING_FILE` to its `selfplay_mod.write_explore(...)` call (after `gumbel=self.gumbel`).

In `_gate`, replace

```python
        candidate, baseline = f"net:{gen_name(k)}@{sims}", f"net:{gen_name(best)}@{sims}"
```

with

```python
        suffix = "" if self.opening is None else "+o"
        candidate, baseline = f"net:{gen_name(k)}@{sims}{suffix}", f"net:{gen_name(best)}@{sims}{suffix}"
```

- [ ] **Step 6: Write the CLI**

In `alphasettler/cli.py`, after the fast-loop parser's `--gumbel` argument add:

```python
    fl.add_argument("--opening", default=None, metavar="FILE",
                    help="first run only: an opening model (ASOPEN) picks self-play's setup settlements, "
                         "and both gate sides are +o")
```

after the train-loop parser's `--gumbel` argument add:

```python
    lp.add_argument("--opening", default=None, metavar="FILE",
                    help="first run only: an opening model (ASOPEN) picks self-play's setup settlements, "
                         "and both gate sides are +o")
```

In `_fast_loop`, change `from alphasettler.selfplay import mode_mismatch` to `from alphasettler.selfplay import mode_mismatch, opening_mismatch`, change the `FastLoop.init(...)` call to

```python
        FastLoop.init(root, args.seed_dirs, args.start_net, args.first, trades=args.trades,
                      gumbel=args.gumbel, opening=args.opening)
```

and after the `mode_mismatch` block add:

```python
    error = opening_mismatch(root, loop.opening, args.opening)
    if error is not None:
        print(f"error: {error}", file=sys.stderr)
        return 2
```

In `_train_loop`, change `gumbel=args.gumbel,` in the `LoopSettings(...)` call to `gumbel=args.gumbel, opening=args.opening,`.

- [ ] **Step 7: Run the tests to verify they pass**

Run: `.venv/bin/pytest -q tests/python/test_fast_loop.py tests/python/test_train_loop.py`
Expected: PASS, including the 6 new fast-loop cases, the 2 new train-loop tests and the new `{"opening": "missing-opening.bin"}` case; every existing loop test passes unchanged.

- [ ] **Step 8: Format, lint, the full suite, pytest**

Run the Global Constraints sequence (fmt, fmt --check, clippy, `cargo test --workspace --release`, rebuild the extension, `.venv/bin/pytest -q -rs`). Expected: all pass, nothing skipped.

- [ ] **Step 9: Commit**

```bash
git add alphasettler/selfplay.py alphasettler/train/fastloop.py alphasettler/train/loop.py alphasettler/cli.py \
  tests/python/test_fast_loop.py tests/python/test_train_loop.py
git diff --cached | grep -nE 'sk-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{36}|password=[^[:space:]]+'
git -c user.email=sean.may101@gmail.com commit -m "loops: fast-loop and train-loop --opening FILE, the model copied beside the nets and kept from set-up; gates play +o"
```

---

### Task 3: Arm C, the SPRT and `docs/bot/results-3d1b.md`

**Files:**
- Create: `docs/bot/results-3d1b.md`
- Run outputs (not committed; `runs/` is ignored): `runs/p2/C/`, `runs/p2/C.log`, `runs/p2/C.out`, `runs/p2/C-state-at-cut.json`, `runs/p2/tables-3d1b.md`, `runs/p2/finals-3d1b.log`, `runs/p2/nets/C.bin`, `runs/p2/nets/opening.bin`, `runs/p2/sprt-C-vs-A.jsonl`, `runs/p2/sprt-C-vs-A.log`

**Interfaces:**
- Consumes: `fast-loop --opening` (Task 2); `+o` names (Plan 7) and `alphasettler sprt --trades`; `runs/p2/arm.sh` and `runs/p2/tables.py` (P2, unchanged); arm A's `runs/p2/A-state-at-cut.json`, `runs/p2/A.log` and `runs/p2/nets/A.bin` (`gen-040`); `runs/3c/nets/opening.bin`, `runs/3c/nets/gen-034.bin`, `runs/3b-v2/gen-002` .. `gen-005`, `runs/3c/gen-030` .. `gen-034`.
- Produces: the run outputs above and `docs/bot/results-3d1b.md`.

Every step but Step 6 is run by the controller in the background, on an otherwise idle machine. If a step fails, record the failure with its numbers in the results; never rerun a match on the same seeds after changing anything.

- [ ] **Step 1: Build at Task 2's commit (controller runs this)**

```bash
git log -1 --format=%s
PATH=$HOME/.cargo/bin:$PATH cargo build --release -p settler-bots
ls runs/p2/arm.sh runs/p2/tables.py runs/p2/A-state-at-cut.json runs/p2/A.log runs/p2/nets/A.bin \
  runs/3c/nets/gen-034.bin runs/3c/nets/opening.bin runs/3b-v2/gen-002 runs/3b-v2/gen-005 runs/3c/gen-030 runs/3c/gen-034 > /dev/null
shasum -a 256 runs/3c/nets/opening.bin runs/p2/nets/A.bin runs/p2/A/nets/gen-040.bin
test ! -e runs/p2/C && test ! -e runs/p2/nets/C.bin && test ! -e runs/p2/sprt-C-vs-A.jsonl
pgrep -fl "alphasettler|settler" || echo idle
```

Then rebuild the extension per the Global Constraints. Pass: the commit subject is Task 2's; every command exits 0; `runs/3c/nets/opening.bin` has sha256 `9ab63da1309e35f2a87990edc85064ee9888057de84d52fb0e09ae382ced922b`; `runs/p2/nets/A.bin` and `runs/p2/A/nets/gen-040.bin` share `79690c2992539f7995e9bc295c73606f01aa3b705e26586e1077bb5f24db7f88`; the last line is `idle`.

- [ ] **Step 2: Arm C, the 3-hour loop (controller runs this)**

Arm C reuses arm A's exact settings and self-play seeds: it runs through `runs/p2/arm.sh` itself, unchanged, whose extra arguments go only to the untimed set-up (the same seed directories, start net, `--first 35`, `--trades`, and every default A used: 2,000 games, window 40,000 games, 10 training minutes, a gate of 100 seeds at 100 simulations; generation k self-plays seeds from k × 10,000,000 and gates on 1,000,000,000 + 0..100, as A did). The only difference is `--opening runs/3c/nets/opening.bin`. Start it detached (about 3 hours plus the set-up, longer than a background tool call may live), with the machine otherwise idle:

```bash
pgrep -fl "alphasettler|settler" || echo idle
nohup bash runs/p2/arm.sh C --opening runs/3c/nets/opening.bin >> runs/p2/C.out 2>&1 &
```

Poll `runs/p2/C.log` until its last line holds `cut C after`, `loop exited early` or `setup failed`. Then:

```bash
tail -1 runs/p2/C.log
.venv/bin/python3 -c "
import hashlib, json
a, c = (json.load(open(f'runs/p2/{x}-state-at-cut.json')) for x in 'AC')
model = hashlib.sha256(open('runs/3c/nets/opening.bin', 'rb').read()).hexdigest()
assert a['seed_dirs'] == c['seed_dirs'] and a['first'] == c['first'] == 35 and a['trades'] is c['trades'] is True
assert a['gumbel'] is None and c['gumbel'] is None and a.get('opening') is None and c['opening'] == model
for ga, gc in zip(a['generations'], c['generations']):
    assert ga['gen'] == gc['gen'] and ga['selfplay']['seed_start'] == gc['selfplay']['seed_start'], gc['gen']
    assert ga['selfplay']['games'] == gc['selfplay']['games'] == 2000 and ga['gate']['seeds'] == gc['gate']['seeds'] == 100
for g in c['generations']:
    assert g['selfplay']['opening'] == model and g['selfplay']['belief_resets'] == 0 and g['gate']['belief_resets'] == 0, g['gen']
    assert g['gate']['candidate'].endswith('@100+t+o') and g['gate']['baseline'].endswith('@100+t+o'), g['gate']
print(f'C matches A: {len(c[\"generations\"])} generations, same seed directories, seeds and gate; +o gates; no belief resets')
" | tee -a runs/p2/finals-3d1b.log
cmp runs/3c/nets/opening.bin runs/p2/C/nets/opening.bin && echo "C's opening.bin is runs/3c/nets/opening.bin"
```

Pass: `runs/p2/C.log` ends with `cut C after 108NN s` (10,800 to 10,810); the check prints `C matches A: ...`; the `cmp` line prints.

- [ ] **Step 3: The tables and the final nets (controller runs this)**

```bash
.venv/bin/python3 runs/p2/tables.py A C | tee runs/p2/tables-3d1b.md
best=$(.venv/bin/python3 -c "import json; print(json.load(open('runs/p2/C-state-at-cut.json'))['best'])")
cp "runs/p2/C/nets/$best.bin" runs/p2/nets/C.bin
cp runs/3c/nets/opening.bin runs/p2/nets/opening.bin
echo "A: gen-040 (runs/p2/nets/A.bin)" | tee -a runs/p2/finals-3d1b.log
echo "C: $best (runs/p2/nets/C.bin)" | tee -a runs/p2/finals-3d1b.log
shasum -a 256 runs/p2/nets/A.bin runs/p2/nets/C.bin runs/p2/nets/opening.bin | tee -a runs/p2/finals-3d1b.log
```

`runs/p2/nets` held `A.bin` (and `B.bin`) only; the `+o` names of the SPRT load `opening.bin` beside the net, so the model is copied in. Pass: `tables-3d1b.md` has both tables and both games-per-hour lines; `finals-3d1b.log` names each arm's net; `runs/p2/nets/opening.bin` has the Step 1 sha256.

- [ ] **Step 4: The SPRT (controller runs this)**

```bash
.venv/bin/alphasettler sprt --candidate net:C@100+t+o --baseline net:A@100+t+o --trades \
  --seed-start 4000000000 --nets-dir runs/p2/nets --out runs/p2/sprt-C-vs-A.jsonl 2>&1 | tee runs/p2/sprt-C-vs-A.log
.venv/bin/python3 -c "
import json
rs = [json.loads(l) for l in open('runs/p2/sprt-C-vs-A.jsonl')]
print(f'{len(rs)} games, belief resets {sum(r[\"belief_resets\"] for r in rs)}')
" | tee -a runs/p2/sprt-C-vs-A.log
```

The defaults are the spec's: p0 0.25, p1 0.30, alpha = beta = 0.05, batches of 12 seeds, at most 400 seeds (4,000,000,000 .. 4,000,000,399). Pass: exit 0, a line starting `sprt: ` (H1, H0 or undecided) and `belief resets 0`. H1: new loops use `--opening`; H0 or undecided: recorded, and the loop without it stays the default.

- [ ] **Step 5: Check the SPRT line against the records (controller runs this)**

```bash
grep -E '^net:C@100\+t\+o vs net:A@100\+t\+o: |^sprt: ' runs/p2/sprt-C-vs-A.log
```

Pass: two lines, the summary and the verdict; these are what Step 6 copies.

- [ ] **Step 6: Write `docs/bot/results-3d1b.md`**

Every number is copied from the files named here; nothing is estimated. The file:

```markdown
# Sub-project 3d-1b (the opening model in self-play): results

Date: <date of the SPRT>. Machine: <`sysctl -n machdep.cpu.brand_string`>, macOS (<`uname -r`>).
Commit: <`git log -1 --format=%h`> (Plan 9, Task 2), plus the commit adding this file.

Spec: [`docs/superpowers/specs/2026-10-08-opening-selfplay-design.md`](../superpowers/specs/2026-10-08-opening-selfplay-design.md).
Plan: [`docs/superpowers/plans/2026-10-08-plan-9-opening-selfplay.md`](../superpowers/plans/2026-10-08-plan-9-opening-selfplay.md).
Builds on: [`results-3d1.md`](results-3d1.md) (the opening model and `+o`) and
[`results-p2-gumbel.md`](results-p2-gumbel.md) (arm A, the control).

## Training at equal wall time (done criteria 2 and 3)

Arm C is a `fast-loop` from gen-034 with exactly arm A's settings, run through the same script
(`bash runs/p2/arm.sh C --opening runs/3c/nets/opening.bin`): a fresh root, the trading game, seed
directories `runs/3b-v2/gen-002` to `gen-005` and `runs/3c/gen-030` to `gen-034`, 2,000 games a
generation, a window of 40,000 games, 10 minutes of training, a gate of 100 seeds at 100
simulations; set up untimed, then 10,800 s of generations (C cut after <N from runs/p2/C.log> s).
Arm A is P2's control (`runs/p2/A`, cut after 10,809 s), not rerun. Both arms played the same
self-play and gate seeds. In C the opening model (`runs/3c/nets/opening.bin`, sha256
<from runs/p2/finals-3d1b.log>) picked every self-play setup settlement: its best spot with
probability 0.75, else one of its 5 best uniformly, with no search and no recorded decision; setup
roads and every later move were searched as in A (600 / 100 simulations, p 0.25, Dirichlet noise, 8
sampled turns). C's gates were `+t+o` on both sides, A's `+t`, so the two gate columns measure
different bots.

<the A table and its self-play line, from runs/p2/tables-3d1b.md>

<the C table and its self-play line, from runs/p2/tables-3d1b.md>

Final nets: <the A and C lines of runs/p2/finals-3d1b.log>.

| Run | Result | Bar | Outcome |
|---|---|---|---|
| `sprt --candidate net:C@100+t+o --baseline net:A@100+t+o --trades`, seeds from 4,000,000,000 | <the summary line and the sprt line of runs/p2/sprt-C-vs-A.log> | H1 (p1 0.30 against p0 0.25, alpha = beta = 0.05) | <H1: new loops use --opening / H0: recorded, the loop without it stays the default / undecided: recorded, the loop without it stays the default> |

Belief resets: <the belief resets line of runs/p2/sprt-C-vs-A.log>.

## Rulings the build made

- The model is a parameter of new self-play entry points (`play_with_opening`,
  `run_with_opening`); without a model self-play is unchanged (the 3a and P2 digests pass through
  them).
- A model pick draws one number from the game's exploration stream (explore if below 0.25) and,
  when exploring, one more (the spot among the 5 best); the ranking is a stable sort, so ties go to
  engine order and the greedy pick is `+o`'s spot. About 0.8 of picks are the best spot.
- The seat's bot still receives the game's events at a model pick; it does not decide.
- `--opening` with `--gumbel` is refused, in Rust and in Python, before anything is written.
- Records with a model carry `opening_sha256`, `opening_epsilon` and `opening_top` in their
  `selfplay` block; records without one are byte-identical to before.
- Loops keep the model by its sha256 (`state.json`, `manifest.json`); a resume without the flag
  keeps it, another file is refused, and a missing or changed `nets/opening.bin` stops the loop.
  `train-loop`'s warm start uses the model too.
- Arm C ran through `runs/p2/arm.sh` unchanged; arm A was not rerun. The SPRT's nets directory
  is `runs/p2/nets` with `C.bin` and `opening.bin` copied in beside `A.bin`.
- Fresh seeds: 4,000,000,000.

## Next

<one paragraph: if H1, the next loop (3d-2) is set up with --opening, and retraining the opening
model on the loop's games is the follow-up the spec names; if H0 or undecided, what the two tables
(held-out scores, gate results, games per hour) point to, and that the loop without the model
stays the default>
```

Fill every `<...>` from the named source before committing. The committed file holds numbers and text only, with no angle brackets.

- [ ] **Step 7: Commit**

Run: `grep -c '<' docs/bot/results-3d1b.md`
Expected: `0`.

```bash
git add docs/bot/results-3d1b.md
git diff --cached | grep -nE 'sk-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{36}|password=[^[:space:]]+'
git -c user.email=sean.may101@gmail.com commit -m "docs: 3d-1b results, the opening model in self-play against P2's control arm at equal wall time"
```

---

## Self-review

### Spec coverage

| Spec requirement | Task |
|---|---|
| §1.1 `fast-loop` and `train-loop` gain `--opening PATH` (an `ASOPEN` file); self-play's setup settlements by the model; gates `+o` on both sides; kept like `--trades` and `--gumbel` | 1 (self-play), 2 (both loops, gates, resume rules) |
| §1.2 arm C from gen-034 with exactly arm A's settings (`runs/p2/arm.sh`) plus `--opening runs/3c/nets/opening.bin`, set up untimed, 10,800 s; `net:C@100+t+o` vs three `net:A@100+t+o`, SPRT p0 0.25, p1 0.30, alpha = beta = 0.05, fresh seeds from 4,000,000,000; A is gen-040, C arm C's best at the cut; H1 / H0 / undecided | 3 (Steps 2–4) |
| §1.3 `docs/bot/results-3d1b.md`: arm C's table beside arm A's (as `tables.py` prints them), the SPRT line | 3 (Steps 3, 6) |
| §2 every legal settlement scored as `+o` (`choose_settlement`: placed on a copy, own-win probability) | 1 (`settlement_scores`, shared by `choose_settlement`; `the_scores_cover_every_legal_spot_in_engine_order_and_the_plus_o_spot_is_the_first_best`) |
| §2 1 − ε the best (ties to the first in engine order), else uniform over the top K (all if fewer); ε 0.25, K 5; from the exploration stream so a seed replays | 1 (`explore_settlement`, `an_opening_model_picks_every_setup_settlement_from_its_top_spots_unrecorded`, `opening_model_self_play_is_deterministic_and_thread_independent`) |
| §2 no search and no decision for those picks; setup roads and later moves searched as now | 1 (same test: no setup-settlement decision, road decisions recorded; Python `test_opening_self_play_records_no_setup_settlement_and_names_its_model`) |
| §2 ruling: the model fixed for the loop, copied into the loop's nets directory at set-up | 2 (`init`, `_place_opening`, `_opening_file`; `test_a_changed_opening_model_in_the_nets_directory_stops_the_loop`) |
| §3 out of scope: `--gumbel` with `--opening` refused; no retraining, no roads by the model, no draft search | 1 (`run_with_opening`, `check_opening`), 2 (set-up refusals); nothing else built |
| §4 Rust: no decision at any setup settlement, every pick within the top K, at least one non-argmax pick over a few seeds, deterministic per seed and thread count, `belief_resets` 0; without a model every pinned digest unchanged | 1 (the four new tests; the never-re-pin list in the Global Constraints) |
| §4 Python: `fast-loop --opening FILE` runs two tiny generations, gate names `+o` on both sides, a loop set up without it refuses it on resume (and the reverse), a bad file refused at set-up, `--opening` with `--gumbel` refused | 2 (`test_cli_fast_loop_with_an_opening_model_self_plays_and_gates_with_it`, `test_cli_fast_loop_keeps_the_opening_model_it_was_set_up_with`, `test_a_bad_opening_model_or_one_with_gumbel_is_refused_before_anything_is_written`; train-loop tests) |

### Placeholder scan

Searched for "TBD", "TODO", "implement later", "fill in", "similar to Task", "appropriate", "handle edge cases": none in the code steps. The one template is Task 3 Step 6's results file, whose `<...>` marks are run-time measurements, each naming its source; Step 7 refuses to commit while one is left. The two digests in Task 1's `self_play_without_an_opening_model_plays_the_games_it_played_before_3d1b` are the existing pinned values (`standard_exploration_plays_the_games_it_played_before_p2`, `fixed_exploration_is_3a_self_play`'s `PLAY_3_AT_10`); the sha256 values in Task 3 Step 1 were read from the files at cf24caf.

### Type consistency

- Rust: `settlement_scores(&OpeningModel, &Observation, &[Action]) -> Vec<(Action, f32)>` (Task 1 Step 3) is what `choose_settlement`, `explore_settlement` and the tests' `ranked_spots` call. `play_with_opening(seed, config, ex, batch, make_eval, opening)` and `run_with_opening(seeds, config, ex, batch, make_eval, threads, opening)` with `opening: Option<&OpeningModel>` are what the tests and the binding (`model.as_ref()`) call; `OPENING_EPSILON: f64`, `OPENING_TOP: usize` are what the tests assert and the binding exports.
- Python: `check_opening(path, gumbel=None) -> str`, `OPENING_WITH_GUMBEL`, `settings(..., opening=None)` and `write_explore(..., opening=None)` (Task 1) are what `FastLoop.init`, `FastLoop._selfplay`, `loop._check` and `Loop._selfplay` (Task 2) call; `opening_mismatch(root, stored, asked)` is used by `_fast_loop` and `Loop._opening_model`. `FastLoop.opening`, `Loop.opening`, `LoopSettings.opening` and the `"opening"` keys (`state.json`, each generation's `selfplay` entry, `manifest.json`, `selfplay.json` plans) are used with those names by the CLI and the tests; record blocks use `opening_sha256`, `opening_epsilon`, `opening_top` in both tasks' tests. `opening_model_file(path, scale=0.02)` (Task 1 helper) is what Task 2's tests import.

### Review Focus

Each of the five lines has its test in the owning task: 1 and 2 in Task 2 (resume rules, the changed copy), 3 in Task 1 (the tied model), 4 and 5 in Tasks 1 and 2 (the refusals at every entry point).
