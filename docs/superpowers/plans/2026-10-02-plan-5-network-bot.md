# Plan 5: Network Bot (Sub-project 3b) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace 3a's hand-set heuristic evaluator with a value/policy network trained by self-play (inference in Rust, training in PyTorch), and beat 3a's own `ismcts` bot with it.

**Architecture:** A new Rust crate `net/` (`settler-net`, depends on `engine` and `search` only) holds the one feature encoder, the weights file, a hand-written batched MLP forward pass, `NetEvaluator` (a `search::Evaluator`) and the replay-based training-data sampler. `bots/` makes `IsmctsBot` generic over its evaluator, adds `net:NAME[@N[+rD]]` bots through a `Roster` that loads each net once, and gains playout cap randomization, Dirichlet root noise and opening sampling in self-play. `bindings/` exposes encode, the sampler, the forward pass and 3b self-play; a new Python package `alphasettler/train/` (PyTorch, `train` extra only) trains, exports and runs the resumable generation loop.

**Tech Stack:** Rust 1.80+ (edition 2021, no new external crates), PyO3 0.29.3 + maturin 1.15.0, Python 3.12 stdlib (gzip, json, hashlib, struct, array), PyTorch 2.14.1 (`train` extra only, MPS for training), pytest 9.1.1, Catanatron 3.3.0 (oracle extra only).

**Spec:** `docs/superpowers/specs/2026-10-02-bot-3b-network-design.md` (Sections 1–5). It builds on `docs/superpowers/specs/2026-10-01-bot-3a-ismcts-design.md` and its results, `docs/bot/results-3a.md`.

## Global Constraints

- Rust MSRV 1.80, edition 2021. No new external crates anywhere (std plus existing workspace crates). `settler-net` depends only on `settler-engine` and `settler-search` (path). Dependencies run one way: `bots → net → search → engine`; `net` never depends on `bots`. `net` joins the workspace `members` and `default-members`.
- Commands: Rust is `~/.cargo/bin/cargo` (cargo is not on PATH). Python tools are `.venv/bin/python3`, `.venv/bin/pytest`, `.venv/bin/pip3`, `.venv/bin/pip-audit`. Never `python` or `pip`.
- Rebuild the extension after any Rust change used from Python: `PATH=$HOME/.cargo/bin:$PATH .venv/bin/maturin develop --release -q`.
- Every task ends with `~/.cargo/bin/cargo fmt --check` printing nothing and `~/.cargo/bin/cargo clippy --workspace --release --all-targets -- -D warnings` passing. (3a's branch broke fmt; run `~/.cargo/bin/cargo fmt --all` before the check.)
- Full checks: `~/.cargo/bin/cargo test --workspace --release` and `.venv/bin/pytest -q`.
- The engine does not change: no file under `engine/` is edited, and `engine/tests/golden_trace.rs` stays green **without re-pinning**.
- Determinism: same seed and weights give the same games, independent of thread count. Budgets are simulation counts, never wall time. Training on MPS is not bit-reproducible; the guarantee covers self-play, search and arenas given a weights file, not training.
- 3b never trades: `OfferTrade` ids are always masked; bots decline every offer.
- Only `oracle/` imports Catanatron. The base install and every Rust crate stay free of PyTorch: `torch==2.14.1` lives only in the new optional extra `train`, and no module outside `alphasettler/train/` imports it. No numpy: tensors come from the bindings' bytes through `torch.frombuffer`.
- Network (spec Section 2): `public features → 512 → 512 → 512` trunk (ReLU); policy `512 → 665` logits, masked, softmax; value `[trunk 512 ‖ hidden 30] → 256 → 4`, softmax; aux `256 → 4` (final VP / 10, training only). The hidden block enters only the value head.
- Weights file: little-endian; header magic `ASNET\0\0\0`, format version (u32), `ENCODING_HASH` (u64), layer count and each layer's shape; then f32 weights and biases in layer order. Wrong magic, version, hash or shape, or a short file, is a clear error.
- Bot names: `ismcts` keeps meaning 3a's heuristic bot. `net:NAME` (1,000 simulations), `net:NAME@N`, `net:NAME@N+rD`, with the same limits on N and D as `ismcts` (1 ≤ N ≤ 1,000,000; 1 ≤ D ≤ 200). NAME is `[A-Za-z0-9_-]{1,64}` and resolves to `<nets-dir>/NAME.bin`; `--nets-dir` defaults to `runs/3b/nets`. A malformed name, a missing file or a file that fails validation is an error before any game starts.
- Self-play: each move is a full search with probability 0.25 (600 simulations) or a fast search (100), drawn from the game's seeded rng. Root noise on full searches only: `prior' = 0.75 · prior + 0.25 · Dir(α)`, α = 10 / (number of legal moves), gamma variates by Marsaglia–Tsang from the engine `Rng`. During setup and the first 8 turns, sample the move in proportion to root visits; after that, play the most-visited move.
- Training: window = the most recent 4 generations; loss = policy cross-entropy on full-search positions only + value cross-entropy (one-hot winner, ¼ each at the turn cap) + 0.25 × MSE(aux, final VP / 10) + L2 1e-4; Adam, batch 1,024, lr 1e-3 for net₀ and 3e-4 afterwards; steps = 2 × new positions / 1,024; export, then a Rust/PyTorch parity check.
- Gating: native CRN arena, no trades, the new net 1 vs 3 copies of the best net at 600 simulations over 200 seeds; promoted if its win rate is at least 0.25 (a tie at exactly 0.25 promotes).
- Git: one commit per task with the message given in the task, run as `git -c user.email=sean.may101@gmail.com commit -m "..."`. No Co-Authored-By or AI attribution, never `--no-verify`, never push. Before every commit run the secret scan `git diff --cached | grep -nE 'sk-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{36}|password=[^[:space:]]+'` and expect no output.
- Long runs (self-play generations of about 6 h, the AlphaBeta 200-seed arena of about 20 min, every 200-seed net arena) are run by the controller in the background, never by implementer subagents. Steps marked **controller runs this** give the exact command and the pass condition.
- No README or docs beyond `docs/perf/net-search.md` and `docs/bot/results-3b.md`.

## Review Focus

1. A weights file trained under an older encoder loaded after the encoder changed: `ENCODING_HASH` is derived from every layout and scaling constant, so any change alters it, and loading refuses with both hashes in the message instead of silently reading misaligned features (Tasks 1 and 2).
2. `train-loop` interrupted mid-step or mid-rename (Ctrl-C, crash) leaving a hidden `.NAME.tmp-PID` file: no reader ever treats a temp file as finished, the rerun clears it and redoes only the incomplete step, and the final manifest equals an uninterrupted run's (Task 7).
3. Record files in the training window that are empty, truncated (a killed writer), hold games with zero decisions, or repeat seeds because a second machine wrote into the same generation directory: each is skipped or deduplicated with a printed notice, and a window with nothing to train on is an error before any step (Task 6).
4. MPS unavailable (another machine, a CI box, a broken PyTorch build): training falls back to the CPU with a printed notice instead of crashing (Task 6).
5. A `net:` name carrying path characters (`net:../x`, `net:a/b`, `net:a\b`, `net:.`, `net:a.bin`, 65 characters): rejected as an unknown bot by the name check, before the nets directory is touched, everywhere a bot is named (Task 3).

## File Structure

| File | Responsibility | Task |
|---|---|---|
| `Cargo.toml` | add `net` to `members` and `default-members` | 1 |
| `net/Cargo.toml`, `net/src/lib.rs` | the `settler-net` crate | 1 (modules added in 2, 3, 5) |
| `net/src/encode.rs` | the only encoder: public features, hidden block, layout constants, `ENCODING_HASH` | 1 |
| `net/src/weights.rs` | weights file: `Arch`, `Layer`, `Weights` (load, validate, save, random) | 2 |
| `net/src/forward.rs` | batched f32 MLP forward pass, `softmax` | 2 |
| `net/src/evaluator.rs` | `NetEvaluator`, `masked_prior`, `absolute_value` | 3 |
| `net/src/data.rs` | recorded games, replay-and-encode training batches, seeded sampling | 5 |
| `net/tests/{common/mod.rs, encode.rs, weights.rs, evaluator.rs, data.rs}` | tests | 1, 2, 3, 5 |
| `search/src/noise.rs` | gamma (Marsaglia–Tsang), Dirichlet, the `Noised` root-noise evaluator wrapper | 4 |
| `search/tests/noise.rs` | tests | 4 |
| `bots/src/ismcts.rs` | `IsmctsBot<E>` generic over the evaluator; `MoveOptions`, `act_with`, `sample_by_visits` | 3, 4 |
| `bots/src/netbot.rs` | `net:` names, `load_net`, `net_bot`, `NET_BATCH`, `DEFAULT_NETS_DIR` | 3 (8 sets `NET_BATCH`) |
| `bots/src/lib.rs` | `Roster`, `unknown_bot` | 3 |
| `bots/src/arena.rs` | `run_match_in`, `run_match_by` | 3 |
| `bots/src/selfplay.rs` | `Exploration`, `play_with`, `run_with` (3a's `play`/`run` become wrappers) | 4 |
| `bots/examples/net_speed.rs` | net throughput and the batch sweep | 8 |
| `bots/tests/{netbot.rs, selfplay.rs}` | tests | 3, 4 |
| `bindings/src/lib.rs` | `Bot(..., nets_dir)`, `run_match(..., nets_dir)`, `selfplay_explore`, constants | 3, 5 |
| `bindings/src/train.rs` | `encode`, `TrainingSet`, `net_forward`, `write_random_net`, `resolve_net` | 3, 5 |
| `alphasettler/files.py` | sha256, temp names, atomic writes | 5 |
| `alphasettler/records.py` | byte-deterministic gzip; replay without observations | 5 |
| `alphasettler/selfplay.py` | 3b self-play writer and its settings block | 5 |
| `alphasettler/arena.py`, `oracle/arena.py` | `nets_dir` pass-through | 3 |
| `alphasettler/cli.py` | `--nets-dir`; `selfplay --net/--pcr`; `train`; `train-loop` | 3, 5, 6, 7 |
| `alphasettler/train/{__init__,model,data,train,loop}.py` | PyTorch twin, data window, training run, generation loop | 6, 7 |
| `pyproject.toml` | the `train` extra | 6 |
| `tests/python/{test_netbot, test_training_data, test_train, test_train_loop}.py`, `test_oracle_arena.py` | tests | 3, 5, 6, 7 |
| `docs/perf/net-search.md`, `docs/bot/results-3b.md` | measurements and results | 8, 9 |

Task order differs from a pure "measure, then train" order in one place: the batch sweep's paired strength match and the self-play speed that sizes a generation both need a trained net, so Task 8 runs the warm start (net₀) first and measures with it.

---

### Task 1: The `net` crate and the feature encoder

**Files:**
- Modify: `Cargo.toml`
- Create: `net/Cargo.toml`, `net/src/lib.rs`, `net/src/encode.rs`
- Test: `net/tests/common/mod.rs`, `net/tests/encode.rs`

**Interfaces:**
- Consumes: `settler_engine::{Observation, State, Phase, PlayerId, Hand, ACTION_SPACE_SIZE, NUM_PLAYERS, NUM_RESOURCES}`, `settler_engine::topology::{topo, NUM_NODES, NUM_EDGES, NUM_TILES}`.
- Produces (`settler_net::encode`):
  - constants `ENCODER_VERSION: u32`, `PUBLIC_LEN: usize` (= 1554), `HIDDEN_LEN: usize` (= 30), `HIDDEN_WIDTH`, `NODE_WIDTH`, `EDGE_WIDTH`, `TILE_WIDTH`, `SEAT_WIDTH`, `NUM_PHASES`, block offsets `NODE_BASE`, `EDGE_BASE`, `TILE_BASE`, `OWN_HAND`, `OWN_DEV_OLD`, `OWN_DEV_NEW`, `SEATS`, `LONGEST_ROAD`, `LARGEST_ARMY`, `BANK`, `DECK`, `PHASE`, `TURN`, `DEV_PLAYED`; scaling denominators `CARD_DEN` … `NODE_PIP_DEN: u32`; `LAYOUT: [u32; 22]`; `ENCODING_HASH: u64`.
  - `pub const fn layout_hash(layout: &[u32]) -> u64`
  - `pub fn rel(p: PlayerId, viewer: PlayerId) -> usize`, `pub fn abs(r: usize, viewer: PlayerId) -> usize`, `pub fn pips(number: u8) -> u32`, `pub fn phase_index(p: Phase) -> usize`
  - `pub fn encode_public(obs: &Observation, out: &mut [f32])` (out.len() == PUBLIC_LEN, every entry overwritten)
  - `pub fn encode_hidden_from(hands: &[Hand; 4], dev_cards: &[[u8; 5]; 4], actor: PlayerId, out: &mut [f32])`, `pub fn encode_hidden(world: &State, actor: PlayerId, out: &mut [f32])`
  - `pub fn encode_state(s: &State, actor: PlayerId, public: &mut [f32], hidden: &mut [f32])`

The public vector, seat-relative to `obs.viewer` (relative seat 0 is the viewer, then turn order):

| Block | Offset | Per item |
|---|---|---|
| 54 nodes × 19 | `NODE_BASE` = 0 | [0..4] settlement of rel seat r; [4..8] city of rel seat r; [8] generic port; [9..14] 2:1 port per resource; [14..19] pips of adjacent tiles per resource / 15 |
| 72 edges × 4 | `EDGE_BASE` = 1026 | road of rel seat r |
| 19 tiles × 8 | `TILE_BASE` = 1314 | [0..5] resource one-hot; [5] desert; [6] pips / 5; [7] robber here |
| own hand | `OWN_HAND` = 1466 | 5 resources / 10 |
| own dev cards | `OWN_DEV_OLD` = 1471, `OWN_DEV_NEW` = 1476 | held before this turn, bought this turn; per kind / 5 |
| 4 seats × 11 | `SEATS` = 1481 | hand count / 20; dev cards / 5; new dev cards / 5; played per kind (5) / 5; knights / 14; public VP / 10; discards owed / 10 |
| longest road, largest army | `LONGEST_ROAD` = 1525, `LARGEST_ARMY` = 1530 | owner one-hot: rel seat 0..3 or [4] none |
| bank | `BANK` = 1535 | 5 resources / 19 |
| dev deck | `DECK` = 1540 | remaining / 25 |
| phase | `PHASE` = 1541 | 11-way one-hot (`phase_index`) |
| turn | `TURN` = 1552 | turn / max_turns |
| dev card played this turn | `DEV_PLAYED` = 1553 | 0 or 1 |

The hidden block: for relative seats 1, 2, 3, that opponent's hand (5, / 10) then dev cards held by kind (5, / 5, VP cards and this turn's purchases included).

- [ ] **Step 1: Write the failing tests**

Edit the root `Cargo.toml`:

```toml
[workspace]
members = ["engine", "bots", "bindings", "search", "net"]
default-members = ["engine", "bots", "search", "net"]
resolver = "2"
```

(keep the `[profile.*]` sections below it unchanged).

Create `net/Cargo.toml`:

```toml
[package]
name = "settler-net"
version = "0.1.0"
edition = "2021"
rust-version = "1.80"
publish = false

[lib]
name = "settler_net"

[dependencies]
settler-engine = { path = "../engine" }
settler-search = { path = "../search" }

[lints.clippy]
needless_range_loop = "allow"
```

Create `net/src/lib.rs`:

```rust
//! The learned evaluator for AlphaSettler (spec:
//! docs/superpowers/specs/2026-10-02-bot-3b-network-design.md): the feature encoder, the weights
//! file, the forward pass, `NetEvaluator` and the training-data sampler.

pub mod encode;
```

Create `net/tests/common/mod.rs`:

```rust
#![allow(dead_code)]

use settler_engine::legal::legal_actions;
use settler_engine::rng::Rng;
use settler_engine::*;
use std::path::PathBuf;

pub fn no_trades() -> GameConfig {
    GameConfig {
        max_offers_per_turn: 0,
        ..GameConfig::default()
    }
}

/// Every `every`-th position (never a finished one) of uniformly random games from `seeds`, at
/// most 3,000 moves each.
pub fn random_positions(seeds: std::ops::Range<u64>, every: usize, config: GameConfig) -> Vec<State> {
    let mut out = Vec::new();
    let mut buf = Vec::new();
    for seed in seeds {
        let mut s = State::new(seed, config);
        let mut rng = Rng::new(seed ^ 0xF00D);
        let mut i = 0;
        while !s.is_over() && i < 3000 {
            legal_actions(&s, &mut buf);
            if i % every == 0 {
                out.push(s);
            }
            s.apply(buf[rng.below(buf.len() as u32) as usize]);
            i += 1;
        }
    }
    out
}

/// A path in the system temp directory unique to this test process and `name`.
pub fn temp_file(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("settler-net-{}-{name}", std::process::id()))
}
```

Create `net/tests/encode.rs`:

```rust
mod common;

use common::*;
use settler_engine::topology::{topo, NUM_NODES, NUM_TILES};
use settler_engine::*;
use settler_net::encode::*;

/// Pips by number token, written out (the encoder computes them).
const PIPS: [u32; 13] = [0, 0, 1, 2, 3, 4, 5, 0, 5, 4, 3, 2, 1];

/// A hand-built position with every scalar block non-trivial. Player 2 is to act.
fn golden_state() -> State {
    let mut s = State::new(42, GameConfig::default());
    s.setup_step = 8;
    s.phase = Phase::Main;
    s.current = 2;
    s.turn = 37;
    s.robber = 7;
    s.bank = [10, 12, 9, 15, 11];
    s.dev_deck_pos = 9;
    s.dev_played_this_turn = true;
    s.longest_road_owner = Some(3);
    let p2 = &mut s.players[2];
    p2.settlements = 1 << 10;
    p2.cities = 1 << 20;
    p2.roads = (1 << 3) | (1 << 4);
    p2.hand = [1, 0, 2, 0, 3];
    p2.dev_hand = [2, 1, 0, 0, 0];
    p2.dev_new = [1, 0, 0, 0, 0];
    p2.knights_played = 1;
    p2.dev_played = [1, 0, 0, 0, 0];
    let p3 = &mut s.players[3];
    p3.settlements = 1;
    p3.roads = 1;
    p3.hand = [0, 4, 0, 0, 0];
    p3.dev_hand = [0, 0, 1, 0, 0];
    p3.discard_remaining = 2;
    s.players[0].cities = 1 << 40;
    s.players[0].hand = [0, 0, 0, 1, 0];
    s.players[1].dev_played = [0, 0, 0, 0, 1];
    s
}

#[test]
fn the_layout_adds_up() {
    assert_eq!(PUBLIC_LEN, 1554);
    assert_eq!(HIDDEN_LEN, 30);
    assert_eq!(
        (EDGE_BASE, TILE_BASE, OWN_HAND, SEATS, LONGEST_ROAD, BANK, PHASE, TURN),
        (1026, 1314, 1466, 1481, 1525, 1535, 1541, 1552)
    );
    assert_eq!(ENCODING_HASH, layout_hash(&LAYOUT));
}

#[test]
fn changing_any_layout_constant_changes_the_hash() {
    for i in 0..LAYOUT.len() {
        let mut changed = LAYOUT;
        changed[i] += 1;
        assert_ne!(layout_hash(&changed), ENCODING_HASH, "layout entry {i}");
    }
    let mut swapped = LAYOUT;
    swapped.swap(12, 13);
    assert_ne!(layout_hash(&swapped), ENCODING_HASH, "order matters");
}

#[test]
fn the_golden_position_encodes_exactly() {
    let s = golden_state();
    let mut got = vec![f32::NAN; PUBLIC_LEN];
    encode_public(&s.observation(2), &mut got);

    let mut want = vec![0.0f32; PUBLIC_LEN];
    let t = topo();
    let b = &s.board;
    // Board blocks, derived tile -> corners (the encoder goes node -> tiles).
    for tile in 0..NUM_TILES {
        let at = TILE_BASE + tile * TILE_WIDTH;
        want[at + b.tile_resource[tile].map_or(5, |r| r.index())] = 1.0;
        let p = PIPS[b.tile_number[tile] as usize];
        want[at + 6] = p as f32 / 5.0;
        if let Some(r) = b.tile_resource[tile] {
            for &n in &t.tile_nodes[tile] {
                want[NODE_BASE + n as usize * NODE_WIDTH + 14 + r.index()] += p as f32;
            }
        }
    }
    for n in 0..NUM_NODES {
        for r in 0..5 {
            want[NODE_BASE + n * NODE_WIDTH + 14 + r] /= 15.0;
        }
    }
    for &(e, kind) in &b.ports {
        let slot = match kind {
            PortKind::Generic => 8,
            PortKind::Specific(r) => 9 + r.index(),
        };
        let (x, y) = t.edge_nodes[e as usize];
        want[NODE_BASE + x as usize * NODE_WIDTH + slot] = 1.0;
        want[NODE_BASE + y as usize * NODE_WIDTH + slot] = 1.0;
    }
    want[TILE_BASE + 7 * TILE_WIDTH + 7] = 1.0; // robber on tile 7
    // Buildings and roads, seat-relative to player 2.
    want[NODE_BASE + 10 * NODE_WIDTH] = 1.0; // player 2 (rel 0) settlement
    want[NODE_BASE + 20 * NODE_WIDTH + 4] = 1.0; // player 2 (rel 0) city
    want[NODE_BASE + 1] = 1.0; // node 0: player 3 (rel 1) settlement
    want[NODE_BASE + 40 * NODE_WIDTH + 4 + 2] = 1.0; // player 0 (rel 2) city
    want[EDGE_BASE + 3 * EDGE_WIDTH] = 1.0;
    want[EDGE_BASE + 4 * EDGE_WIDTH] = 1.0;
    want[EDGE_BASE + 1] = 1.0; // edge 0: player 3 (rel 1)
    // Scalars.
    want[OWN_HAND..OWN_HAND + 5].copy_from_slice(&[1.0 / 10.0, 0.0, 2.0 / 10.0, 0.0, 3.0 / 10.0]);
    want[OWN_DEV_OLD..OWN_DEV_OLD + 5].copy_from_slice(&[1.0 / 5.0, 1.0 / 5.0, 0.0, 0.0, 0.0]);
    want[OWN_DEV_NEW] = 1.0 / 5.0;
    let seat = |r: usize| SEATS + r * SEAT_WIDTH;
    want[seat(0)..seat(0) + SEAT_WIDTH].copy_from_slice(&[
        6.0 / 20.0, 3.0 / 5.0, 1.0 / 5.0, 1.0 / 5.0, 0.0, 0.0, 0.0, 0.0, 1.0 / 14.0, 3.0 / 10.0, 0.0,
    ]);
    want[seat(1)..seat(1) + SEAT_WIDTH].copy_from_slice(&[
        4.0 / 20.0, 1.0 / 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0 / 10.0, 2.0 / 10.0,
    ]);
    want[seat(2)] = 1.0 / 20.0;
    want[seat(2) + 9] = 2.0 / 10.0;
    want[seat(3) + 3 + 4] = 1.0 / 5.0; // player 1 played a monopoly
    want[LONGEST_ROAD + 1] = 1.0;
    want[LARGEST_ARMY + 4] = 1.0; // nobody
    want[BANK..BANK + 5].copy_from_slice(&[10.0 / 19.0, 12.0 / 19.0, 9.0 / 19.0, 15.0 / 19.0, 11.0 / 19.0]);
    want[DECK] = 16.0 / 25.0;
    want[PHASE + 6] = 1.0; // Main
    want[TURN] = 37.0 / 1000.0;
    want[DEV_PLAYED] = 1.0;
    for i in 0..PUBLIC_LEN {
        assert_eq!(got[i], want[i], "feature {i}");
    }

    let mut hidden = vec![f32::NAN; HIDDEN_LEN];
    encode_hidden(&s, 2, &mut hidden);
    let mut want_hidden = vec![0.0f32; HIDDEN_LEN];
    want_hidden[1] = 4.0 / 10.0; // rel 1 = player 3: 4 brick
    want_hidden[5 + 2] = 1.0 / 5.0; // and a road building card
    want_hidden[HIDDEN_WIDTH + 3] = 1.0 / 10.0; // rel 2 = player 0: 1 wheat
    assert_eq!(hidden, want_hidden);
}

fn rotate<T: Copy>(a: [T; 4], k: usize) -> [T; 4] {
    std::array::from_fn(|q| a[(q + 4 - k) % 4])
}

fn rot(p: PlayerId, k: usize) -> PlayerId {
    ((p as usize + k) % 4) as PlayerId
}

/// `o` with every seat p renamed (p + k) % 4, the viewer included.
fn rotated(o: &Observation, k: usize) -> Observation {
    let mut r = *o;
    r.viewer = rot(o.viewer, k);
    r.current = rot(o.current, k);
    r.actor = rot(o.actor, k);
    r.hand_counts = rotate(o.hand_counts, k);
    r.dev_card_counts = rotate(o.dev_card_counts, k);
    r.new_dev_card_counts = rotate(o.new_dev_card_counts, k);
    r.dev_cards_played = rotate(o.dev_cards_played, k);
    r.settlements = rotate(o.settlements, k);
    r.cities = rotate(o.cities, k);
    r.roads = rotate(o.roads, k);
    r.knights_played = rotate(o.knights_played, k);
    r.public_vp = rotate(o.public_vp, k);
    r.longest_road_owner = o.longest_road_owner.map(|p| rot(p, k));
    r.largest_army_owner = o.largest_army_owner.map(|p| rot(p, k));
    r.discard_remaining = rotate(o.discard_remaining, k);
    r.trade = o.trade.map(|t| PendingTrade {
        responses: rotate(t.responses, k),
        next_responder: rot(t.next_responder, k),
        ..t
    });
    r
}

fn public(o: &Observation) -> Vec<f32> {
    let mut v = vec![0.0; PUBLIC_LEN];
    encode_public(o, &mut v);
    v
}

#[test]
fn rotating_every_seat_and_the_actor_together_gives_the_same_features() {
    let (mut checked, mut moved, mut trades) = (0, 0, 0);
    for s in random_positions(0..4, 7, GameConfig::default()) {
        let actor = s.current_actor();
        let obs = s.observation(actor);
        let base = public(&obs);
        let hands = std::array::from_fn(|p| s.players[p].hand);
        let devs = std::array::from_fn(|p| s.players[p].dev_hand);
        let mut hidden = vec![0.0; HIDDEN_LEN];
        encode_hidden_from(&hands, &devs, actor, &mut hidden);
        trades += obs.trade.is_some() as u32;
        for k in 1..4 {
            let r = rotated(&obs, k);
            moved += (r.settlements != obs.settlements) as u32;
            assert_eq!(public(&r), base, "rotation {k}");
            let mut h = vec![0.0; HIDDEN_LEN];
            encode_hidden_from(&rotate(hands, k), &rotate(devs, k), rot(actor, k), &mut h);
            assert_eq!(h, hidden, "rotation {k}");
        }
        checked += 1;
    }
    assert!(checked > 500 && moved > 0 && trades > 0, "{checked} {moved} {trades}");
}

#[test]
fn the_features_are_relative_to_the_viewer() {
    let mut differ = 0;
    for s in random_positions(10..12, 13, no_trades()) {
        let obs = s.observation(s.current_actor());
        let mut seats_only = rotated(&obs, 1);
        seats_only.viewer = obs.viewer; // rotate the seats but not the viewer
        differ += (public(&seats_only) != public(&obs)) as u32;
    }
    assert!(differ > 0);
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `~/.cargo/bin/cargo test --release -p settler-net --test encode`
Expected: compile errors, `unresolved import settler_net::encode::*` items such as `cannot find value PUBLIC_LEN`.

- [ ] **Step 3: Write the encoder**

Create `net/src/encode.rs`:

```rust
//! Observation (+ hidden block) -> seat-relative feature vector (spec Section 2). This is the only
//! encoder: Python gets features through the bindings, so Rust inference and PyTorch training see
//! the same inputs. Any change to the layout or the scaling must bump `ENCODER_VERSION`; the golden
//! test in `net/tests/encode.rs` fails until the test is updated with it.

use settler_engine::topology::{topo, NUM_EDGES, NUM_NODES, NUM_TILES};
use settler_engine::types::*;
use settler_engine::{Observation, Phase, State, ACTION_SPACE_SIZE};

pub const ENCODER_VERSION: u32 = 1;

pub const NODE_WIDTH: usize = 19;
pub const EDGE_WIDTH: usize = 4;
pub const TILE_WIDTH: usize = 8;
pub const SEAT_WIDTH: usize = 11;
pub const NUM_PHASES: usize = 11;

pub const NODE_BASE: usize = 0;
pub const EDGE_BASE: usize = NODE_BASE + NUM_NODES * NODE_WIDTH;
pub const TILE_BASE: usize = EDGE_BASE + NUM_EDGES * EDGE_WIDTH;
pub const OWN_HAND: usize = TILE_BASE + NUM_TILES * TILE_WIDTH;
pub const OWN_DEV_OLD: usize = OWN_HAND + NUM_RESOURCES;
pub const OWN_DEV_NEW: usize = OWN_DEV_OLD + 5;
pub const SEATS: usize = OWN_DEV_NEW + 5;
pub const LONGEST_ROAD: usize = SEATS + NUM_PLAYERS * SEAT_WIDTH;
pub const LARGEST_ARMY: usize = LONGEST_ROAD + NUM_PLAYERS + 1;
pub const BANK: usize = LARGEST_ARMY + NUM_PLAYERS + 1;
pub const DECK: usize = BANK + NUM_RESOURCES;
pub const PHASE: usize = DECK + 1;
pub const TURN: usize = PHASE + NUM_PHASES;
pub const DEV_PLAYED: usize = TURN + 1;
pub const PUBLIC_LEN: usize = DEV_PLAYED + 1;

/// Per opponent (relative seats 1..=3): hand by resource, then dev cards held by kind.
pub const HIDDEN_WIDTH: usize = 10;
pub const HIDDEN_LEN: usize = (NUM_PLAYERS - 1) * HIDDEN_WIDTH;

// A count x is encoded as x / DEN, roughly in [0, 1].
/// Cards of one resource.
pub const CARD_DEN: u32 = 10;
/// A player's total cards.
pub const HAND_DEN: u32 = 20;
/// Dev cards, per kind or in total.
pub const DEV_DEN: u32 = 5;
pub const KNIGHT_DEN: u32 = 14;
pub const VP_DEN: u32 = 10;
pub const DISCARD_DEN: u32 = 10;
pub const BANK_DEN: u32 = 19;
pub const DECK_DEN: u32 = 25;
/// One tile's pips.
pub const PIP_DEN: u32 = 5;
/// A node's pips of one resource.
pub const NODE_PIP_DEN: u32 = 15;

/// Everything the layout depends on, hashed into `ENCODING_HASH`.
pub const LAYOUT: [u32; 22] = [
    ENCODER_VERSION,
    NUM_NODES as u32,
    NODE_WIDTH as u32,
    NUM_EDGES as u32,
    EDGE_WIDTH as u32,
    NUM_TILES as u32,
    TILE_WIDTH as u32,
    SEAT_WIDTH as u32,
    NUM_PHASES as u32,
    PUBLIC_LEN as u32,
    HIDDEN_LEN as u32,
    ACTION_SPACE_SIZE as u32,
    CARD_DEN,
    HAND_DEN,
    DEV_DEN,
    KNIGHT_DEN,
    VP_DEN,
    DISCARD_DEN,
    BANK_DEN,
    DECK_DEN,
    PIP_DEN,
    NODE_PIP_DEN,
];

/// FNV-1a over the little-endian bytes of `layout`.
pub const fn layout_hash(layout: &[u32]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let mut i = 0;
    while i < layout.len() {
        let bytes = layout[i].to_le_bytes();
        let mut j = 0;
        while j < 4 {
            h ^= bytes[j] as u64;
            h = h.wrapping_mul(0x0100_0000_01b3);
            j += 1;
        }
        i += 1;
    }
    h
}

/// Written into every weights file; a file with another value was trained on other features.
pub const ENCODING_HASH: u64 = layout_hash(&LAYOUT);

/// `p`'s seat relative to `viewer`: 0 for the viewer, then in turn order.
#[inline]
pub fn rel(p: PlayerId, viewer: PlayerId) -> usize {
    (p as usize + NUM_PLAYERS - viewer as usize) % NUM_PLAYERS
}

/// The absolute seat of relative seat `r` for `viewer`.
#[inline]
pub fn abs(r: usize, viewer: PlayerId) -> usize {
    (viewer as usize + r) % NUM_PLAYERS
}

/// Dots on a number token: 2 and 12 have 1, 6 and 8 have 5, the desert 0.
pub fn pips(number: u8) -> u32 {
    match number {
        2..=6 => number as u32 - 1,
        8..=12 => 13 - number as u32,
        _ => 0,
    }
}

pub fn phase_index(p: Phase) -> usize {
    match p {
        Phase::SetupSettlement => 0,
        Phase::SetupRoad { .. } => 1,
        Phase::PreRoll => 2,
        Phase::Discard => 3,
        Phase::MoveRobber => 4,
        Phase::Steal => 5,
        Phase::Main => 6,
        Phase::RoadBuilding { .. } => 7,
        Phase::TradeResponse => 8,
        Phase::TradeConfirm => 9,
        Phase::GameOver { .. } => 10,
    }
}

#[inline]
fn scaled(x: u32, den: u32) -> f32 {
    x as f32 / den as f32
}

/// The public features of `obs`, seat-relative to `obs.viewer`, written over all of `out`.
pub fn encode_public(obs: &Observation, out: &mut [f32]) {
    assert_eq!(out.len(), PUBLIC_LEN, "the public block is PUBLIC_LEN floats");
    out.fill(0.0);
    let t = topo();
    let v = obs.viewer;
    let b = &obs.board;
    for n in 0..NUM_NODES {
        let at = NODE_BASE + n * NODE_WIDTH;
        let bit = 1u64 << n;
        for p in 0..NUM_PLAYERS {
            let r = rel(p as PlayerId, v);
            if obs.settlements[p] & bit != 0 {
                out[at + r] = 1.0;
            }
            if obs.cities[p] & bit != 0 {
                out[at + 4 + r] = 1.0;
            }
        }
        if b.generic_port_nodes & bit != 0 {
            out[at + 8] = 1.0;
        }
        for res in 0..NUM_RESOURCES {
            if b.specific_port_nodes[res] & bit != 0 {
                out[at + 9 + res] = 1.0;
            }
        }
        let mut node_pips = [0u32; NUM_RESOURCES];
        for &tile in &t.node_tiles[n] {
            if let Some(res) = b.tile_resource[tile as usize] {
                node_pips[res.index()] += pips(b.tile_number[tile as usize]);
            }
        }
        for res in 0..NUM_RESOURCES {
            out[at + 14 + res] = scaled(node_pips[res], NODE_PIP_DEN);
        }
    }
    for e in 0..NUM_EDGES {
        for p in 0..NUM_PLAYERS {
            if obs.roads[p] & (1u128 << e) != 0 {
                out[EDGE_BASE + e * EDGE_WIDTH + rel(p as PlayerId, v)] = 1.0;
            }
        }
    }
    for tile in 0..NUM_TILES {
        let at = TILE_BASE + tile * TILE_WIDTH;
        match b.tile_resource[tile] {
            Some(res) => out[at + res.index()] = 1.0,
            None => out[at + 5] = 1.0,
        }
        out[at + 6] = scaled(pips(b.tile_number[tile]), PIP_DEN);
        if obs.robber as usize == tile {
            out[at + 7] = 1.0;
        }
    }
    for res in 0..NUM_RESOURCES {
        out[OWN_HAND + res] = scaled(obs.my_hand[res] as u32, CARD_DEN);
    }
    for k in 0..5 {
        let new = obs.my_new_dev_cards[k];
        out[OWN_DEV_OLD + k] = scaled((obs.my_dev_cards[k] - new) as u32, DEV_DEN);
        out[OWN_DEV_NEW + k] = scaled(new as u32, DEV_DEN);
    }
    for p in 0..NUM_PLAYERS {
        let at = SEATS + rel(p as PlayerId, v) * SEAT_WIDTH;
        out[at] = scaled(obs.hand_counts[p] as u32, HAND_DEN);
        out[at + 1] = scaled(obs.dev_card_counts[p] as u32, DEV_DEN);
        out[at + 2] = scaled(obs.new_dev_card_counts[p] as u32, DEV_DEN);
        for k in 0..5 {
            out[at + 3 + k] = scaled(obs.dev_cards_played[p][k] as u32, DEV_DEN);
        }
        out[at + 8] = scaled(obs.knights_played[p] as u32, KNIGHT_DEN);
        out[at + 9] = scaled(obs.public_vp[p] as u32, VP_DEN);
        out[at + 10] = scaled(obs.discard_remaining[p] as u32, DISCARD_DEN);
    }
    let owner = |o: Option<PlayerId>| o.map_or(NUM_PLAYERS, |p| rel(p, v));
    out[LONGEST_ROAD + owner(obs.longest_road_owner)] = 1.0;
    out[LARGEST_ARMY + owner(obs.largest_army_owner)] = 1.0;
    for res in 0..NUM_RESOURCES {
        out[BANK + res] = scaled(obs.bank[res] as u32, BANK_DEN);
    }
    out[DECK] = scaled(obs.dev_deck_remaining as u32, DECK_DEN);
    out[PHASE + phase_index(obs.phase)] = 1.0;
    out[TURN] = scaled(obs.turn, obs.config.max_turns.max(1));
    out[DEV_PLAYED] = obs.dev_played_this_turn as u8 as f32;
}

/// The hidden block for `actor`: each opponent's hand and dev cards (VP cards and cards bought
/// this turn included), seat-relative. Only the value head reads it.
pub fn encode_hidden_from(
    hands: &[Hand; NUM_PLAYERS],
    dev_cards: &[[u8; 5]; NUM_PLAYERS],
    actor: PlayerId,
    out: &mut [f32],
) {
    assert_eq!(out.len(), HIDDEN_LEN, "the hidden block is HIDDEN_LEN floats");
    for r in 1..NUM_PLAYERS {
        let p = abs(r, actor);
        let at = (r - 1) * HIDDEN_WIDTH;
        for res in 0..NUM_RESOURCES {
            out[at + res] = scaled(hands[p][res] as u32, CARD_DEN);
        }
        for k in 0..5 {
            out[at + 5 + k] = scaled(dev_cards[p][k] as u32, DEV_DEN);
        }
    }
}

/// The hidden block read from a (sampled or real) world.
pub fn encode_hidden(world: &State, actor: PlayerId, out: &mut [f32]) {
    let hands = std::array::from_fn(|p| world.players[p].hand);
    let devs = std::array::from_fn(|p| world.players[p].dev_hand);
    encode_hidden_from(&hands, &devs, actor, out);
}

/// Both blocks for `actor` in `s`: public features from `s.observation(actor)`, hidden from `s`.
pub fn encode_state(s: &State, actor: PlayerId, public: &mut [f32], hidden: &mut [f32]) {
    encode_public(&s.observation(actor), public);
    encode_hidden(s, actor, hidden);
}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `~/.cargo/bin/cargo test --release -p settler-net --test encode`
Expected: `test result: ok. 5 passed`.

- [ ] **Step 5: Format, lint, full checks**

Run: `~/.cargo/bin/cargo fmt --all && ~/.cargo/bin/cargo fmt --check`
Expected: no output from the check.
Run: `~/.cargo/bin/cargo clippy --workspace --release --all-targets -- -D warnings`
Expected: exits 0 with no `warning:` or `error:` lines.
Run: `~/.cargo/bin/cargo test --workspace --release`
Expected: every test target passes, `golden_trace` included.

- [ ] **Step 6: Commit**

```bash
git add Cargo.toml Cargo.lock net/
git diff --cached | grep -nE 'sk-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{36}|password=[^[:space:]]+'
git -c user.email=sean.may101@gmail.com commit -m "net: the settler-net crate and the seat-relative feature encoder"
```

Expected: the grep prints nothing; the commit succeeds.

---
### Task 2: The weights file and the forward pass

**Files:**
- Create: `net/src/weights.rs`, `net/src/forward.rs`
- Modify: `net/src/lib.rs`
- Test: `net/tests/weights.rs`

**Interfaces:**
- Consumes: `settler_net::encode::{ENCODING_HASH, PUBLIC_LEN, HIDDEN_LEN, encode_state}` (Task 1); `settler_engine::rng::Rng`.
- Produces:
  - `settler_net::weights`: `MAGIC: [u8; 8]` (= `*b"ASNET\0\0\0"`), `FORMAT_VERSION: u32` (= 1), `NUM_LAYERS` (= 7), `HEADER_LEN` (= 80), `MAX_WIDTH` (= 4096), layer indices `TRUNK0, TRUNK1, TRUNK2, POLICY, VALUE0, VALUE1, AUX` (0..=6); `Arch { trunk: usize, value: usize }` with `Arch::DEFAULT` (512, 256) and `fn shapes(&self) -> [(usize, usize); 7]` (rows = outputs, cols = inputs); `Layer { rows, cols, w: Vec<f32>, b: Vec<f32> }`; `Weights { arch: Arch, layers: Vec<Layer> }` with `random(arch, seed) -> Weights`, `param_count(&self) -> usize`, `to_bytes(&self) -> Vec<u8>`, `from_bytes(&[u8]) -> Result<Weights, String>`, `save(&self, &Path) -> Result<(), String>`, `load(&Path) -> Result<Weights, String>`.
  - `settler_net::forward`: `Outputs { n: usize, policy: Vec<f32>, value: Vec<f32>, aux: Vec<f32> }` (`Default`; row-major: 665 policy logits, 4 seat-relative value logits, 4 aux outputs per position); `Forward::new()` and `Forward::run(&mut self, w: &Weights, public: &[f32], hidden: &[f32], n: usize, out: &mut Outputs)`; `pub fn dot(x: &[f32], y: &[f32]) -> f32`; `pub fn softmax(x: &mut [f32])`.

The file layout: bytes 0..8 magic; 8..12 format version; 12..20 `ENCODING_HASH`; 20..24 layer count; then for layer i, rows at `24 + 8i` and cols at `28 + 8i`; from byte 80, each layer's `rows × cols` weights (row-major) then its `rows` biases, all f32 little-endian. A file may declare any trunk width and value width from 1 to `MAX_WIDTH` (tests use small ones); every other dimension is fixed by the encoder and the action space.

- [ ] **Step 1: Write the failing tests**

Create `net/tests/weights.rs`:

```rust
mod common;

use common::*;
use settler_engine::{GameConfig, ACTION_SPACE_SIZE, NUM_PLAYERS};
use settler_net::encode::{encode_state, ENCODING_HASH, HIDDEN_LEN, PUBLIC_LEN};
use settler_net::forward::{Forward, Outputs};
use settler_net::weights::*;

const SMALL: Arch = Arch { trunk: 8, value: 4 };

fn err(bytes: &[u8]) -> String {
    Weights::from_bytes(bytes).unwrap_err()
}

#[test]
fn the_header_is_80_bytes_and_shapes_follow_the_arch() {
    assert_eq!(HEADER_LEN, 80);
    assert_eq!(MAGIC, *b"ASNET\0\0\0");
    assert_eq!(
        Arch::DEFAULT.shapes(),
        [
            (512, PUBLIC_LEN),
            (512, 512),
            (512, 512),
            (ACTION_SPACE_SIZE, 512),
            (256, 512 + HIDDEN_LEN),
            (NUM_PLAYERS, 256),
            (NUM_PLAYERS, 256),
        ]
    );
    let w = Weights::random(Arch::DEFAULT, 1);
    assert_eq!(w.to_bytes().len(), HEADER_LEN + 4 * w.param_count());
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
    let mut magic = good.clone();
    magic[0] = b'X';
    assert!(err(&magic).contains("magic"), "{}", err(&magic));
    let mut version = good.clone();
    version[8..12].copy_from_slice(&2u32.to_le_bytes());
    assert!(err(&version).contains("format version 2"), "{}", err(&version));
    let mut hash = good.clone();
    hash[12] ^= 1;
    let e = err(&hash);
    assert!(e.contains("encoding hash"), "{e}");
    assert!(e.contains(&format!("{ENCODING_HASH:#018x}")), "names this encoder's hash: {e}");
    assert!(e.contains(&format!("{:#018x}", ENCODING_HASH ^ 1)), "names the file's hash: {e}");
    let mut layers = good.clone();
    layers[20..24].copy_from_slice(&6u32.to_le_bytes());
    assert!(err(&layers).contains("6 layers"), "{}", err(&layers));
    let mut shape = good.clone();
    shape[32..36].copy_from_slice(&7u32.to_le_bytes()); // layer 1's rows: 8 -> 7
    assert!(err(&shape).contains("shapes"), "{}", err(&shape));
    let mut huge = good.clone();
    huge[24..28].copy_from_slice(&u32::MAX.to_le_bytes());
    assert!(err(&huge).contains("shapes"), "{}", err(&huge));
    assert!(err(&good[..good.len() - 1]).contains("truncated"));
    assert!(err(&good[..40]).contains("truncated"));
    assert!(err(&[]).contains("truncated"));
    let mut long = good.clone();
    long.push(0);
    assert!(err(&long).contains("trailing"), "{}", err(&long));
    let mut nan = good.clone();
    nan[HEADER_LEN..HEADER_LEN + 4].copy_from_slice(&f32::NAN.to_le_bytes());
    assert!(err(&nan).contains("NaN"), "{}", err(&nan));
    let missing = Weights::load(&temp_file("no-such.bin")).unwrap_err();
    assert!(missing.contains("cannot read") && missing.contains("no-such.bin"), "{missing}");
}

/// The forward pass written the slow, obvious way, in f64.
fn reference(w: &Weights, public: &[f32], hidden: &[f32]) -> [Vec<f64>; 3] {
    let dense = |l: &Layer, x: &[f64], relu: bool| -> Vec<f64> {
        (0..l.rows)
            .map(|o| {
                let s = l.b[o] as f64
                    + (0..l.cols).map(|k| l.w[o * l.cols + k] as f64 * x[k]).sum::<f64>();
                if relu {
                    s.max(0.0)
                } else {
                    s
                }
            })
            .collect()
    };
    let x: Vec<f64> = public.iter().map(|&v| v as f64).collect();
    let h1 = dense(&w.layers[TRUNK0], &x, true);
    let h2 = dense(&w.layers[TRUNK1], &h1, true);
    let h3 = dense(&w.layers[TRUNK2], &h2, true);
    let policy = dense(&w.layers[POLICY], &h3, false);
    let mut joined = h3.clone();
    joined.extend(hidden.iter().map(|&v| v as f64));
    let v = dense(&w.layers[VALUE0], &joined, true);
    [policy, dense(&w.layers[VALUE1], &v, false), dense(&w.layers[AUX], &v, false)]
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

#[test]
fn the_forward_pass_matches_a_plain_reference() {
    let w = Weights::random(Arch { trunk: 32, value: 16 }, 5);
    let (n, public, hidden) = inputs(97);
    let mut out = Outputs::default();
    Forward::new().run(&w, &public, &hidden, n, &mut out);
    assert_eq!(out.n, n);
    assert_eq!(
        (out.policy.len(), out.value.len(), out.aux.len()),
        (n * ACTION_SPACE_SIZE, n * NUM_PLAYERS, n * NUM_PLAYERS)
    );
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
                assert!((*x as f64 - y).abs() <= 1e-4 * y.abs().max(1.0), "position {i}: {x} vs {y}");
            }
        }
    }
}

#[test]
fn each_row_is_independent_of_the_rest_of_the_batch() {
    let w = Weights::random(Arch { trunk: 32, value: 16 }, 9);
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
        assert_eq!(one.policy[..], all.policy[i * ACTION_SPACE_SIZE..(i + 1) * ACTION_SPACE_SIZE]);
        assert_eq!(one.value[..], all.value[i * NUM_PLAYERS..(i + 1) * NUM_PLAYERS]);
        assert_eq!(one.aux[..], all.aux[i * NUM_PLAYERS..(i + 1) * NUM_PLAYERS]);
    }
}
```

Add to `net/src/lib.rs`, after `pub mod encode;`:

```rust
pub mod forward;
pub mod weights;
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `~/.cargo/bin/cargo test --release -p settler-net --test weights`
Expected: compile errors, `file not found for module forward` / `weights`.

- [ ] **Step 3: Write the weights file**

Create `net/src/weights.rs`:

```rust
//! The weights file (spec Section 2). Little-endian: an 80-byte header (magic, format version,
//! `ENCODING_HASH`, layer count, each layer's rows and columns), then each layer's f32 weights
//! (row-major, rows = outputs) and biases, in layer order.

use crate::encode::{ENCODING_HASH, HIDDEN_LEN, PUBLIC_LEN};
use settler_engine::rng::Rng;
use settler_engine::{ACTION_SPACE_SIZE, NUM_PLAYERS};
use std::path::Path;

pub const MAGIC: [u8; 8] = *b"ASNET\0\0\0";
pub const FORMAT_VERSION: u32 = 1;
pub const NUM_LAYERS: usize = 7;
pub const HEADER_LEN: usize = 8 + 4 + 8 + 4 + NUM_LAYERS * 8;
/// Widest trunk or value layer a file may declare: a guard against a corrupt header.
pub const MAX_WIDTH: usize = 4096;

pub const TRUNK0: usize = 0;
pub const TRUNK1: usize = 1;
pub const TRUNK2: usize = 2;
pub const POLICY: usize = 3;
pub const VALUE0: usize = 4;
pub const VALUE1: usize = 5;
pub const AUX: usize = 6;

/// The two free widths. Everything else follows from the encoder and the action space.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Arch {
    pub trunk: usize,
    pub value: usize,
}

impl Arch {
    pub const DEFAULT: Arch = Arch {
        trunk: 512,
        value: 256,
    };

    /// (rows, cols) of each layer in file order; rows are outputs.
    pub fn shapes(&self) -> [(usize, usize); NUM_LAYERS] {
        let (t, v) = (self.trunk, self.value);
        [
            (t, PUBLIC_LEN),
            (t, t),
            (t, t),
            (ACTION_SPACE_SIZE, t),
            (v, t + HIDDEN_LEN),
            (NUM_PLAYERS, v),
            (NUM_PLAYERS, v),
        ]
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
    /// He-uniform weights and small biases from `seed`, for tests and measurements.
    pub fn random(arch: Arch, seed: u64) -> Weights {
        let mut rng = Rng::new(seed);
        let mut layers = Vec::with_capacity(NUM_LAYERS);
        for (rows, cols) in arch.shapes() {
            let bound = (6.0 / cols as f32).sqrt();
            let w = (0..rows * cols).map(|_| uniform(&mut rng, bound)).collect();
            let b = (0..rows).map(|_| uniform(&mut rng, 0.1)).collect();
            layers.push(Layer { rows, cols, w, b });
        }
        Weights { arch, layers }
    }

    pub fn param_count(&self) -> usize {
        self.layers.iter().map(|l| l.w.len() + l.b.len()).sum()
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(HEADER_LEN + 4 * self.param_count());
        out.extend_from_slice(&MAGIC);
        out.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
        out.extend_from_slice(&ENCODING_HASH.to_le_bytes());
        out.extend_from_slice(&(NUM_LAYERS as u32).to_le_bytes());
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

    pub fn from_bytes(bytes: &[u8]) -> Result<Weights, String> {
        if bytes.len() < HEADER_LEN {
            return Err(format!(
                "truncated: {} bytes is shorter than the {HEADER_LEN}-byte header",
                bytes.len()
            ));
        }
        if bytes[..8] != MAGIC {
            return Err("not an AlphaSettler weights file (wrong magic)".into());
        }
        let u32_at = |i: usize| u32::from_le_bytes(bytes[i..i + 4].try_into().unwrap());
        let version = u32_at(8);
        if version != FORMAT_VERSION {
            return Err(format!(
                "format version {version}, expected {FORMAT_VERSION}"
            ));
        }
        let hash = u64::from_le_bytes(bytes[12..20].try_into().unwrap());
        if hash != ENCODING_HASH {
            return Err(format!(
                "encoding hash {hash:#018x} does not match this encoder's {ENCODING_HASH:#018x}: \
                 the file was trained on a different feature layout"
            ));
        }
        let count = u32_at(20) as usize;
        if count != NUM_LAYERS {
            return Err(format!("{count} layers, expected {NUM_LAYERS}"));
        }
        let shapes: Vec<(usize, usize)> = (0..NUM_LAYERS)
            .map(|i| (u32_at(24 + 8 * i) as usize, u32_at(28 + 8 * i) as usize))
            .collect();
        let arch = Arch {
            trunk: shapes[TRUNK0].0,
            value: shapes[VALUE0].0,
        };
        let widths_ok = (1..=MAX_WIDTH).contains(&arch.trunk) && (1..=MAX_WIDTH).contains(&arch.value);
        if !widths_ok || shapes[..] != arch.shapes()[..] {
            return Err(format!(
                "layer shapes {shapes:?} do not fit the network (trunk and value widths must be 1..={MAX_WIDTH}, \
                 and the other dimensions are fixed by the encoder)"
            ));
        }
        let floats: usize = shapes.iter().map(|&(r, c)| r * c + r).sum();
        let want = HEADER_LEN + 4 * floats;
        if bytes.len() < want {
            return Err(format!("truncated: {} bytes, expected {want}", bytes.len()));
        }
        if bytes.len() > want {
            return Err(format!(
                "{} trailing bytes after the last layer",
                bytes.len() - want
            ));
        }
        let mut at = HEADER_LEN;
        let mut take = |n: usize| -> Vec<f32> {
            let v = bytes[at..at + 4 * n]
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect();
            at += 4 * n;
            v
        };
        let mut layers = Vec::with_capacity(NUM_LAYERS);
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

- [ ] **Step 4: Write the forward pass**

Create `net/src/forward.rs`:

```rust
//! The batched MLP forward pass (f32): the only inference code. `net/tests/weights.rs` checks it
//! against a plain f64 reference and `tests/python/test_train.py` against the PyTorch twin
//! (`alphasettler/train/model.py`).

use crate::encode::{HIDDEN_LEN, PUBLIC_LEN};
use crate::weights::{Layer, Weights, AUX, POLICY, TRUNK0, TRUNK1, TRUNK2, VALUE0, VALUE1};

/// Raw head outputs for `n` positions, row-major: 665 policy logits, 4 seat-relative value logits
/// (before the softmax) and 4 aux outputs (final VP / 10, seat-relative) per position.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Outputs {
    pub n: usize,
    pub policy: Vec<f32>,
    pub value: Vec<f32>,
    pub aux: Vec<f32>,
}

/// Activation buffers reused across calls.
#[derive(Clone, Debug, Default)]
pub struct Forward {
    a: Vec<f32>,
    b: Vec<f32>,
    joined: Vec<f32>,
    v: Vec<f32>,
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

/// In place, numerically stable; entries of -inf become 0.
pub fn softmax(x: &mut [f32]) {
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

/// `y[i][o] = b[o] + w[o] · x[i]` for the `n` rows of `x`, optionally through a ReLU.
fn dense(l: &Layer, x: &[f32], n: usize, relu: bool, y: &mut Vec<f32>) {
    debug_assert_eq!(x.len(), n * l.cols);
    y.clear();
    y.resize(n * l.rows, 0.0);
    for o in 0..l.rows {
        let w = &l.w[o * l.cols..(o + 1) * l.cols];
        for i in 0..n {
            let s = l.b[o] + dot(w, &x[i * l.cols..(i + 1) * l.cols]);
            y[i * l.rows + o] = if relu { s.max(0.0) } else { s };
        }
    }
}

impl Forward {
    pub fn new() -> Forward {
        Forward::default()
    }

    /// Run `n` positions: `public` holds `n × PUBLIC_LEN` floats and `hidden` `n × HIDDEN_LEN`.
    pub fn run(&mut self, w: &Weights, public: &[f32], hidden: &[f32], n: usize, out: &mut Outputs) {
        assert_eq!(public.len(), n * PUBLIC_LEN, "public block length");
        assert_eq!(hidden.len(), n * HIDDEN_LEN, "hidden block length");
        let t = w.arch.trunk;
        dense(&w.layers[TRUNK0], public, n, true, &mut self.a);
        dense(&w.layers[TRUNK1], &self.a, n, true, &mut self.b);
        dense(&w.layers[TRUNK2], &self.b, n, true, &mut self.a);
        dense(&w.layers[POLICY], &self.a, n, false, &mut out.policy);
        self.joined.clear();
        for i in 0..n {
            self.joined.extend_from_slice(&self.a[i * t..(i + 1) * t]);
            self.joined
                .extend_from_slice(&hidden[i * HIDDEN_LEN..(i + 1) * HIDDEN_LEN]);
        }
        dense(&w.layers[VALUE0], &self.joined, n, true, &mut self.v);
        dense(&w.layers[VALUE1], &self.v, n, false, &mut out.value);
        dense(&w.layers[AUX], &self.v, n, false, &mut out.aux);
        out.n = n;
    }
}
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `~/.cargo/bin/cargo test --release -p settler-net`
Expected: `encode` 5 passed, `weights` 5 passed.

- [ ] **Step 6: Format, lint, full checks**

Run: `~/.cargo/bin/cargo fmt --all && ~/.cargo/bin/cargo fmt --check`
Expected: no output from the check.
Run: `~/.cargo/bin/cargo clippy --workspace --release --all-targets -- -D warnings`
Expected: exits 0.
Run: `~/.cargo/bin/cargo test --workspace --release`
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add net/
git diff --cached | grep -nE 'sk-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{36}|password=[^[:space:]]+'
git -c user.email=sean.may101@gmail.com commit -m "net: weights file format and the batched MLP forward pass"
```

---

### Task 3: `NetEvaluator` and `net:` bot names everywhere

**Files:**
- Create: `net/src/evaluator.rs`, `bots/src/netbot.rs`, `bindings/src/train.rs`
- Modify: `net/src/lib.rs`, `bots/Cargo.toml`, `bots/src/lib.rs`, `bots/src/ismcts.rs`, `bots/src/arena.rs`, `bindings/Cargo.toml`, `bindings/src/lib.rs`, `alphasettler/arena.py`, `alphasettler/cli.py`, `oracle/arena.py`
- Test: `net/tests/evaluator.rs`, `bots/tests/netbot.rs`, `tests/python/test_netbot.py`, `tests/python/test_oracle_arena.py` (append)

**Interfaces:**
- Consumes: `Weights`, `Forward`, `Outputs`, `softmax` (Task 2); `encode_state`, `PUBLIC_LEN`, `HIDDEN_LEN` (Task 1); `settler_search::{Evaluator, Eval, Leaf, search_actions, terminal_value, SearchConfig}`; `greedy::choose(obs: &Observation, legal: &[Action]) -> Action`; `ismcts::parse_name(&str) -> Option<(u32, u32)>`.
- Produces:
  - `settler_net::evaluator`: `type RolloutPolicy = fn(&Observation, &[Action]) -> Action`; `NetEvaluator::new(weights: Arc<Weights>) -> NetEvaluator`, `NetEvaluator::with_rollout(weights: Arc<Weights>, rollout: u32, policy: RolloutPolicy) -> NetEvaluator`, `evaluations(&self) -> u64`, `weights(&self) -> &Arc<Weights>`; `impl Evaluator for NetEvaluator`; `pub fn masked_prior(logits: &[f32], legal: &[Action]) -> Vec<f32>`; `pub fn absolute_value(logits: &[f32], actor: PlayerId) -> [f32; 4]`.
  - `settler_bots::ismcts::IsmctsBot<E = HeuristicEvaluator>`: `IsmctsBot::new(seed, simulations, rollout) -> IsmctsBot` (unchanged), `IsmctsBot::with_evaluator(name: &'static str, seed: u64, cfg: SearchConfig, eval: E) -> IsmctsBot<E>`, `take_last_search`, `evaluator(&self) -> &E`, `config(&self) -> &SearchConfig`; `impl<E: Evaluator + Send> Bot for IsmctsBot<E>`.
  - `settler_bots::netbot`: `DEFAULT_NETS_DIR: &str` (= `"runs/3b/nets"`), `MAX_NET_NAME` (= 64), `NET_BATCH: usize` (= 8 until Task 8), `valid_net_name(&str) -> bool`, `parse_net_name(&str) -> Option<(&str, u32, u32)>`, `net_path(&Path, &str) -> PathBuf`, `load_net(nets_dir: &Path, net: &str) -> Result<Arc<Weights>, String>`, `net_bot(seed, weights: Arc<Weights>, simulations, rollout) -> IsmctsBot<NetEvaluator>` (name `"net"`).
  - `settler_bots::{unknown_bot(name: &str) -> String, Roster}`: `Roster::new(names: &[&str], nets_dir: &Path) -> Result<Roster, String>`, `Roster::make(&self, name: &str, seed: u64) -> Option<Box<dyn Bot>>`. `make_bot` still returns `None` for `net:` names (they need a roster).
  - `settler_bots::arena::{run_match_in(candidate, baseline, seeds, config, threads, nets_dir: &Path), run_match_by(seeds, config, threads, make: &(dyn Fn(bool, u64) -> Box<dyn Bot> + Sync))}`; `run_match` keeps its signature and uses `DEFAULT_NETS_DIR`.
  - Python `_engine`: `Bot(name, seed, nets_dir=None)`, `run_match(candidate, baseline, seed_start, seeds, threads, config=None, nets_dir=None)`, `write_random_net(path, seed, trunk=None, value=None)`, constant `DEFAULT_NETS_DIR`.
  - Python: `alphasettler.arena.check(..., nets_dir=None)` and `run(..., nets_dir=None)`; `oracle.arena.check(candidate, baseline, seeds, seed_start=0, nets_dir=None)`, `run_match(candidate, baseline, seeds, seed_start=0, workers=None, nets_dir=None)`; CLI `--nets-dir` (default `runs/3b/nets`) on `arena` and `compare`.

- [ ] **Step 1: Write the failing Rust tests**

Create `net/tests/evaluator.rs`:

```rust
mod common;

use common::*;
use settler_engine::rng::Rng;
use settler_engine::*;
use settler_net::evaluator::{absolute_value, masked_prior, NetEvaluator};
use settler_net::weights::{Arch, Weights};
use settler_search::{search_actions, Belief, Evaluator, Leaf, WorldSampler};
use std::sync::Arc;

fn net() -> Arc<Weights> {
    Arc::new(Weights::random(Arch::DEFAULT, 11))
}

fn searchable(s: &State) -> Vec<Action> {
    let mut legal = Vec::new();
    search_actions(&s.legal_actions(), &mut legal);
    legal
}

#[test]
fn the_prior_depends_only_on_what_the_actor_sees() {
    let mut eval = NetEvaluator::new(net());
    let mut rng = Rng::new(1);
    let (mut checked, mut values_differ) = (0, 0);
    for (i, s) in random_positions(20..26, 11, no_trades()).iter().enumerate() {
        let actor = s.current_actor();
        let obs = s.observation(actor);
        let belief = Belief::from_observation(&obs, 32, i as u64);
        let sampler = WorldSampler::new(&obs, &belief).unwrap();
        let (w1, w2) = (sampler.sample(&belief, &mut rng), sampler.sample(&belief, &mut rng));
        let legal = searchable(s);
        let e = eval.evaluate(&[
            Leaf { world: &w1, actor, legal: &legal },
            Leaf { world: &w2, actor, legal: &legal },
        ]);
        assert_eq!(e[0].prior, e[1].prior, "position {i}");
        values_differ += (e[0].value != e[1].value) as u32;
        checked += 1;
    }
    assert!(checked > 100, "{checked}");
    assert!(values_differ > 0, "the hidden block reaches the value head");
}

#[test]
fn values_and_priors_are_distributions_in_every_phase() {
    let mut eval = NetEvaluator::new(net());
    let mut trade_phases = 0;
    for s in random_positions(0..2, 13, GameConfig::default()) {
        trade_phases += matches!(s.phase, Phase::TradeResponse | Phase::TradeConfirm) as u32;
        let legal = searchable(&s);
        let e = &eval.evaluate(&[Leaf { world: &s, actor: s.current_actor(), legal: &legal }])[0];
        assert_eq!(e.prior.len(), legal.len());
        assert!((e.prior.iter().sum::<f32>() - 1.0).abs() < 1e-4);
        assert!((e.value.iter().sum::<f32>() - 1.0).abs() < 1e-4);
        assert!(e.prior.iter().chain(&e.value).all(|x| x.is_finite() && *x >= 0.0));
    }
    assert!(trade_phases > 0, "pending offers reached the evaluator");
}

#[test]
fn a_single_legal_move_gets_all_the_prior_and_offers_get_none() {
    let mut eval = NetEvaluator::new(net());
    let s = State::new(3, GameConfig::default());
    let actor = s.current_actor();
    let e = eval.evaluate(&[Leaf { world: &s, actor, legal: &[Action::Roll] }]);
    assert_eq!(e[0].prior, vec![1.0]);
    let offer = Action::OfferTrade { give: 0, get: 1 };
    let mixed = [Action::EndTurn, offer, Action::BuyDev];
    let e = eval.evaluate(&[Leaf { world: &s, actor, legal: &mixed }]);
    assert_eq!(e[0].prior[1], 0.0);
    assert!((e[0].prior[0] + e[0].prior[2] - 1.0).abs() < 1e-6);
    assert_eq!(masked_prior(&vec![0.0; ACTION_SPACE_SIZE], &[offer]), vec![1.0]);
}

#[test]
fn values_are_rotated_back_to_absolute_seats() {
    let v = absolute_value(&[0.0, 1.0, 2.0, 3.0], 2);
    let e: Vec<f32> = [0.0f32, 1.0, 2.0, 3.0].iter().map(|x| x.exp()).collect();
    let z: f32 = e.iter().sum();
    // relative seat r is absolute seat (2 + r) % 4
    for (abs_seat, r) in [(2, 0), (3, 1), (0, 2), (1, 3)] {
        assert!((v[abs_seat] - e[r] / z).abs() < 1e-6, "{v:?}");
    }
}

#[test]
fn a_batch_evaluates_each_leaf_as_if_alone() {
    let mut eval = NetEvaluator::new(net());
    let positions = random_positions(3..4, 29, no_trades());
    let legal: Vec<Vec<Action>> = positions.iter().map(searchable).collect();
    let leaves: Vec<Leaf<'_>> = positions
        .iter()
        .zip(&legal)
        .map(|(s, l)| Leaf { world: s, actor: s.current_actor(), legal: l })
        .collect();
    let together = eval.evaluate(&leaves);
    for (i, leaf) in leaves.iter().enumerate() {
        let alone = eval.evaluate(&[Leaf { world: leaf.world, actor: leaf.actor, legal: leaf.legal }]);
        assert_eq!(alone[0], together[i], "leaf {i}");
    }
    assert_eq!(eval.evaluations(), 2 * leaves.len() as u64);
}

fn first(_: &Observation, legal: &[Action]) -> Action {
    legal[0]
}

#[test]
fn rollouts_keep_the_leaf_prior_and_score_the_position_they_reach() {
    let w = net();
    let mut plain = NetEvaluator::new(w.clone());
    let mut rolled = NetEvaluator::with_rollout(w, 4, first);
    let mut differ = 0;
    for s in random_positions(5..6, 17, no_trades()) {
        let legal = searchable(&s);
        let leaf = [Leaf { world: &s, actor: s.current_actor(), legal: &legal }];
        let (a, b) = (plain.evaluate(&leaf), rolled.evaluate(&leaf));
        assert_eq!(a[0].prior, b[0].prior);
        assert!((b[0].value.iter().sum::<f32>() - 1.0).abs() < 1e-4);
        differ += (a[0].value != b[0].value) as u32;
    }
    assert!(differ > 0);
    assert!(rolled.evaluations() > plain.evaluations());
}
```

Create `bots/tests/netbot.rs`:

```rust
use settler_bots::arena::run_match_in;
use settler_bots::netbot::{load_net, parse_net_name, valid_net_name, NET_BATCH};
use settler_bots::{make_bot, Bot, Roster};
use settler_engine::*;
use settler_net::weights::{Arch, Weights};
use std::path::{Path, PathBuf};

fn no_trades() -> GameConfig {
    GameConfig {
        max_offers_per_turn: 0,
        ..GameConfig::default()
    }
}

/// A fresh nets directory holding `tiny.bin` (random weights, trunk 16, value width 8).
fn nets_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("settler-bots-nets-{}-{tag}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    Weights::random(Arch { trunk: 16, value: 8 }, 1)
        .save(&dir.join("tiny.bin"))
        .unwrap();
    dir
}

#[test]
fn net_names_parse_and_bad_ones_are_rejected_without_touching_the_disk() {
    assert_eq!(parse_net_name("net:a"), Some(("a", 1000, 0)));
    assert_eq!(parse_net_name("net:gen-001@600"), Some(("gen-001", 600, 0)));
    assert_eq!(parse_net_name("net:x_Y-9@300+r8"), Some(("x_Y-9", 300, 8)));
    let longest = format!("net:{}", "a".repeat(64));
    assert_eq!(parse_net_name(&longest).map(|p| p.0.len()), Some(64));
    let too_long = format!("net:{}", "a".repeat(65));
    for bad in [
        "net:", "net:@5", "net:../x", "net:..", "net:.", "net:a/b", "net:/abs", "net:a\\b",
        "net:a b", "net:é", "net:a.bin", too_long.as_str(), "net:a@0", "net:a@", "net:a@abc",
        "net:a@1000001", "net:a@300+r0", "net:a@300+r201", "net:a@5@6", "net:a+r8", "netx:a",
        "Net:a", "ismcts:a",
    ] {
        assert_eq!(parse_net_name(bad), None, "{bad}");
        assert!(make_bot(bad, 0).is_none(), "{bad}");
        let e = Roster::new(&[bad], Path::new("/no/such/nets/dir")).err().unwrap();
        assert!(e.starts_with("unknown bot"), "{bad}: {e}");
    }
    assert!(!valid_net_name("") && valid_net_name("A-z_09"));
}

#[test]
fn a_missing_or_stale_weights_file_is_an_error_before_any_game() {
    let dir = nets_dir("errors");
    let e = Roster::new(&["net:missing"], &dir).err().unwrap();
    assert!(e.contains("cannot read") && e.contains("missing.bin"), "{e}");
    let mut stale = std::fs::read(dir.join("tiny.bin")).unwrap();
    stale[12] ^= 1;
    std::fs::write(dir.join("stale.bin"), &stale).unwrap();
    let e = run_match_in("net:stale", "greedy", 0..1, no_trades(), 1, &dir).unwrap_err();
    assert!(e.contains("encoding hash"), "{e}");
    assert!(load_net(&dir, "tiny").is_ok());
    assert!(make_bot("net:tiny", 0).is_none(), "net bots need a roster");
}

#[test]
fn a_net_bot_plays_whole_games_independent_of_thread_count() {
    let dir = nets_dir("arena");
    let a = run_match_in("net:tiny@10", "greedy", 0..2, no_trades(), 1, &dir).unwrap();
    let b = run_match_in("net:tiny@10", "greedy", 0..2, no_trades(), 3, &dir).unwrap();
    assert_eq!(a, b);
    assert_eq!(a.len(), 8);
    assert!(a.iter().all(|r| r.belief_resets == 0 && r.turns > 0));
    let roster = Roster::new(&["net:tiny@10+r2", "greedy"], &dir).unwrap();
    assert_eq!(roster.make("net:tiny@10+r2", 0).unwrap().name(), "net");
    assert_eq!(roster.make("greedy", 0).unwrap().name(), "greedy");
    assert!(roster.make("net:other", 0).is_none());
    assert!(NET_BATCH >= 1);
}
```

- [ ] **Step 2: Run them to verify they fail**

Run: `~/.cargo/bin/cargo test --release -p settler-net --test evaluator; ~/.cargo/bin/cargo test --release -p settler-bots --test netbot`
Expected: compile errors: `could not find evaluator in settler_net`, `could not find netbot in settler_bots`, `use of undeclared crate settler_net` (bots does not depend on it yet).

- [ ] **Step 3: Write `NetEvaluator`**

Add `pub mod evaluator;` to `net/src/lib.rs` (the module list becomes `encode`, `evaluator`, `forward`, `weights`).

Create `net/src/evaluator.rs`:

```rust
//! `NetEvaluator`: `search::Evaluator` backed by the network (spec Section 3). For each leaf it
//! encodes `world.observation(actor)` and the hidden block, runs the batch through one forward
//! pass, masks and softmaxes the policy over `legal`, and rotates the value back to absolute seats.

use crate::encode::{encode_state, HIDDEN_LEN, PUBLIC_LEN};
use crate::forward::{softmax, Forward, Outputs};
use crate::weights::Weights;
use settler_engine::legal::legal_actions;
use settler_engine::{Action, Observation, PlayerId, State, ACTION_SPACE_SIZE, NUM_PLAYERS};
use settler_search::{search_actions, terminal_value, Eval, Evaluator, Leaf};
use std::sync::Arc;

/// Picks a rollout move from an observation and the searchable moves (bots pass GreedyBot's).
pub type RolloutPolicy = fn(&Observation, &[Action]) -> Action;

/// The prior over `legal`: a softmax of the policy logits at their action ids. Domestic offers
/// always get 0 (3b never trades); if `legal` holds nothing else the prior is uniform.
pub fn masked_prior(logits: &[f32], legal: &[Action]) -> Vec<f32> {
    let mut p: Vec<f32> = legal
        .iter()
        .map(|a| match a {
            Action::OfferTrade { .. } => f32::NEG_INFINITY,
            _ => logits[a.encode() as usize],
        })
        .collect();
    if p.iter().all(|&x| x == f32::NEG_INFINITY) {
        let u = 1.0 / p.len() as f32;
        p.fill(u);
        return p;
    }
    softmax(&mut p);
    p
}

/// Win probabilities by absolute seat from the value logits of `actor`'s seat-relative view.
pub fn absolute_value(logits: &[f32], actor: PlayerId) -> [f32; NUM_PLAYERS] {
    let mut p = [logits[0], logits[1], logits[2], logits[3]];
    softmax(&mut p);
    std::array::from_fn(|q| p[(q + NUM_PLAYERS - actor as usize) % NUM_PLAYERS])
}

pub struct NetEvaluator {
    weights: Arc<Weights>,
    rollout: u32,
    policy: Option<RolloutPolicy>,
    forward: Forward,
    out: Outputs,
    public: Vec<f32>,
    hidden: Vec<f32>,
    buf: Vec<Action>,
    moves: Vec<Action>,
    evaluations: u64,
}

impl NetEvaluator {
    /// Weights are shared read-only (`Arc`): load them once per process.
    pub fn new(weights: Arc<Weights>) -> NetEvaluator {
        NetEvaluator {
            weights,
            rollout: 0,
            policy: None,
            forward: Forward::new(),
            out: Outputs::default(),
            public: Vec::new(),
            hidden: Vec::new(),
            buf: Vec::new(),
            moves: Vec::new(),
            evaluations: 0,
        }
    }

    /// The prior still comes from the leaf; the value comes from the position reached after
    /// `rollout` moves of `policy` in the leaf's own sampled world (a finished game scores as
    /// `terminal_value`).
    pub fn with_rollout(weights: Arc<Weights>, rollout: u32, policy: RolloutPolicy) -> NetEvaluator {
        NetEvaluator {
            rollout,
            policy: Some(policy),
            ..NetEvaluator::new(weights)
        }
    }

    /// Positions run through the network so far.
    pub fn evaluations(&self) -> u64 {
        self.evaluations
    }

    pub fn weights(&self) -> &Arc<Weights> {
        &self.weights
    }

    /// Fill the input buffers for `positions` and run them.
    fn run(&mut self, positions: &[(&State, PlayerId)]) {
        let n = positions.len();
        self.public.resize(n * PUBLIC_LEN, 0.0);
        self.hidden.resize(n * HIDDEN_LEN, 0.0);
        for (i, &(s, actor)) in positions.iter().enumerate() {
            encode_state(
                s,
                actor,
                &mut self.public[i * PUBLIC_LEN..(i + 1) * PUBLIC_LEN],
                &mut self.hidden[i * HIDDEN_LEN..(i + 1) * HIDDEN_LEN],
            );
        }
        self.forward
            .run(&self.weights, &self.public, &self.hidden, n, &mut self.out);
        self.evaluations += n as u64;
    }

    fn roll_out(&mut self, world: &State, policy: RolloutPolicy) -> State {
        let mut s = *world;
        for _ in 0..self.rollout {
            if s.is_over() {
                break;
            }
            legal_actions(&s, &mut self.buf);
            search_actions(&self.buf, &mut self.moves);
            let a = policy(&s.observation(s.current_actor()), &self.moves);
            s.apply(a);
        }
        s
    }
}

impl Evaluator for NetEvaluator {
    fn evaluate(&mut self, leaves: &[Leaf<'_>]) -> Vec<Eval> {
        let positions: Vec<(&State, PlayerId)> = leaves.iter().map(|l| (l.world, l.actor)).collect();
        self.run(&positions);
        let mut evals: Vec<Eval> = leaves
            .iter()
            .enumerate()
            .map(|(i, l)| Eval {
                value: absolute_value(&self.out.value[i * NUM_PLAYERS..(i + 1) * NUM_PLAYERS], l.actor),
                prior: masked_prior(
                    &self.out.policy[i * ACTION_SPACE_SIZE..(i + 1) * ACTION_SPACE_SIZE],
                    l.legal,
                ),
            })
            .collect();
        let rollout_policy = if self.rollout > 0 { self.policy } else { None };
        if let Some(policy) = rollout_policy {
            let mut rolled: Vec<(usize, State)> = Vec::new();
            for (i, l) in leaves.iter().enumerate() {
                let s = self.roll_out(l.world, policy);
                if s.is_over() {
                    evals[i].value = terminal_value(&s);
                } else {
                    rolled.push((i, s));
                }
            }
            if !rolled.is_empty() {
                let positions: Vec<(&State, PlayerId)> =
                    rolled.iter().map(|(_, s)| (s, s.current_actor())).collect();
                self.run(&positions);
                for (j, (i, s)) in rolled.iter().enumerate() {
                    evals[*i].value = absolute_value(
                        &self.out.value[j * NUM_PLAYERS..(j + 1) * NUM_PLAYERS],
                        s.current_actor(),
                    );
                }
            }
        }
        evals
    }
}
```

- [ ] **Step 4: Make `IsmctsBot` generic and add the net bots**

In `bots/Cargo.toml`, under `[dependencies]` add:

```toml
settler-net = { path = "../net" }
```

Replace `bots/src/ismcts.rs` from its first line through the end of `fn name` inside `impl Bot for IsmctsBot` with the code below. Only the struct, its constructors, the accessors and the `Bot` impl's header change; `parse_name`, `Tracked` and `set_tracked` are 3a's, copied unchanged:

```rust
//! IsmctsBot: tracks hidden cards from the event log and searches each real decision with
//! single-observer ISMCTS (spec Sections 2-3). Never trades. Generic over the evaluator: the
//! heuristic for `ismcts` names, the network for `net:` names (`crate::netbot`).

use crate::greedy;
use crate::heuristic::{HeuristicEvaluator, DEFAULT_WEIGHTS};
use crate::Bot;
use settler_engine::rng::{mix, Rng};
use settler_engine::{Action, Event, Observation, Phase, PlayerId};
use settler_search::{
    search, search_actions, Belief, Evaluator, SearchConfig, SearchResult, DEFAULT_MAX_STATES,
};

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

/// What the bot knows about the hidden cards.
enum Tracked {
    /// Nothing observed yet.
    Unseen,
    /// The exact posterior, advanced by every event since the game started.
    Live(Belief),
    /// `viewer`'s history contradicted the belief (the reset is already counted); later events
    /// are skipped until `act` rebuilds the belief from an observation.
    Dropped(PlayerId),
}

pub struct IsmctsBot<E = HeuristicEvaluator> {
    name: &'static str,
    seed: u64,
    rng: Rng,
    cfg: SearchConfig,
    eval: E,
    tracked: Tracked,
    last: Option<SearchResult>,
    resets: u64,
    /// `observe` counted a reset since the last `act`, so that decision counts no other.
    counted_in_observe: bool,
    /// Truncations of beliefs already replaced.
    past_truncations: u64,
    searches: u64,
    buf: Vec<Action>,
}

impl IsmctsBot<HeuristicEvaluator> {
    /// 3a's bot: the heuristic evaluator with `rollout` greedy moves per leaf.
    pub fn new(seed: u64, simulations: u32, rollout: u32) -> IsmctsBot {
        IsmctsBot::with_evaluator(
            "ismcts",
            seed,
            SearchConfig {
                simulations,
                ..SearchConfig::default()
            },
            HeuristicEvaluator::new(DEFAULT_WEIGHTS, rollout),
        )
    }
}

impl<E: Evaluator> IsmctsBot<E> {
    /// A bot reporting `name` that searches with `eval` under `cfg`.
    pub fn with_evaluator(name: &'static str, seed: u64, cfg: SearchConfig, eval: E) -> IsmctsBot<E> {
        IsmctsBot {
            name,
            seed,
            rng: Rng::new(seed),
            cfg,
            eval,
            tracked: Tracked::Unseen,
            last: None,
            resets: 0,
            counted_in_observe: false,
            past_truncations: 0,
            searches: 0,
            buf: Vec::new(),
        }
    }

    /// The last decision's search, if it searched; taking it clears it.
    pub fn take_last_search(&mut self) -> Option<SearchResult> {
        self.last.take()
    }

    pub fn evaluator(&self) -> &E {
        &self.eval
    }

    pub fn config(&self) -> &SearchConfig {
        &self.cfg
    }

    /// Replace the tracked state, keeping the old belief's truncation count.
    fn set_tracked(&mut self, t: Tracked) {
        if let Tracked::Live(b) = std::mem::replace(&mut self.tracked, t) {
            self.past_truncations += b.truncations();
        }
    }
}

impl<E: Evaluator + Send> Bot for IsmctsBot<E> {
    fn name(&self) -> &'static str {
        self.name
    }
```

Leave everything after 3a's `fn name` (the `observe`, `act` and `diagnostics` methods and the impl's closing brace) exactly as it is; those bodies use `self.eval`, which now has type `E`.

Create `bots/src/netbot.rs`:

```rust
//! Network bots (spec Section 3): `net:NAME` (1,000 simulations), `net:NAME@N` and
//! `net:NAME@N+rD`, with the same limits on N and D as `ismcts`: IsmctsBot searching with a
//! `NetEvaluator`. NAME is `[A-Za-z0-9_-]{1,64}` and names the file `<nets-dir>/NAME.bin`; the
//! character set keeps every name a plain file name inside the directory.

use crate::greedy;
use crate::ismcts::{self, IsmctsBot};
use settler_net::evaluator::NetEvaluator;
use settler_net::weights::Weights;
use settler_search::SearchConfig;
use std::path::{Path, PathBuf};
use std::sync::Arc;

pub const DEFAULT_NETS_DIR: &str = "runs/3b/nets";
pub const MAX_NET_NAME: usize = 64;
/// Leaves per network call. Set from the batch sweep in `docs/perf/net-search.md`.
pub const NET_BATCH: usize = 8;

pub fn valid_net_name(net: &str) -> bool {
    (1..=MAX_NET_NAME).contains(&net.len())
        && net
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
}

/// `net:NAME` -> (NAME, 1000, 0); `net:NAME@N` -> (NAME, N, 0); `net:NAME@N+rD` -> (NAME, N, D).
pub fn parse_net_name(name: &str) -> Option<(&str, u32, u32)> {
    let rest = name.strip_prefix("net:")?;
    let (net, tail) = match rest.find('@') {
        Some(i) => (&rest[..i], &rest[i..]),
        None => (rest, ""),
    };
    if !valid_net_name(net) {
        return None;
    }
    let (simulations, rollout) = ismcts::parse_name(&format!("ismcts{tail}"))?;
    Some((net, simulations, rollout))
}

pub fn net_path(nets_dir: &Path, net: &str) -> PathBuf {
    nets_dir.join(format!("{net}.bin"))
}

/// Load and validate `<nets_dir>/<net>.bin`. `net` must be a valid NAME (checked first, so a bad
/// name never reaches the filesystem).
pub fn load_net(nets_dir: &Path, net: &str) -> Result<Arc<Weights>, String> {
    if !valid_net_name(net) {
        return Err(format!(
            "bad net name {net:?}: use 1 to {MAX_NET_NAME} of A-Z a-z 0-9 _ -"
        ));
    }
    Weights::load(&net_path(nets_dir, net))
        .map(Arc::new)
        .map_err(|e| format!("net {net:?}: {e}"))
}

pub fn net_bot(seed: u64, weights: Arc<Weights>, simulations: u32, rollout: u32) -> IsmctsBot<NetEvaluator> {
    let eval = if rollout == 0 {
        NetEvaluator::new(weights)
    } else {
        NetEvaluator::with_rollout(weights, rollout, greedy::choose)
    };
    let cfg = SearchConfig {
        simulations,
        batch: NET_BATCH,
        ..SearchConfig::default()
    };
    IsmctsBot::with_evaluator("net", seed, cfg, eval)
}
```

In `bots/src/lib.rs`, add `pub mod netbot;` to the module list (between `ismcts` and `random`), replace the doc comment at the top with

```rust
//! Baseline bots, IsmctsBot (heuristic or network evaluator) and the native arena. Each decision
//! a bot sees an `Observation` and the legal actions; through `observe` it also sees the events
//! since its last move, redacted for its seat, so it can keep its own history.
```

replace `use settler_engine::{Action, Event, Observation, PlayerId};` with

```rust
use settler_engine::{Action, Event, Observation, PlayerId};
use settler_net::weights::Weights;
use std::path::Path;
use std::sync::Arc;
```

and replace everything from `pub const BOT_NAMES` to the end of the file with:

```rust
pub const BOT_NAMES: &[&str] = &["random", "greedy", "ismcts"];

/// A fresh bot by name, with its own random stream seeded from `seed`. Besides `BOT_NAMES`,
/// `ismcts@N` (N simulations) and `ismcts@N+rD` (plus D greedy rollout moves per leaf). `net:`
/// names need their weights loaded first: use a `Roster`.
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

/// The error for a name no bot answers to.
pub fn unknown_bot(name: &str) -> String {
    format!(
        "unknown bot {name:?}; known bots: {BOT_NAMES:?} (ismcts@N for N simulations, ismcts@N+rD \
         to add D greedy rollout moves, net:NAME[@N[+rD]] for <nets-dir>/NAME.bin)"
    )
}

/// Bot names checked, and their networks loaded once and shared read-only across threads, so a
/// match builds bots per game without touching the disk.
pub struct Roster {
    nets: Vec<(String, Arc<Weights>)>,
}

impl Roster {
    /// Every name must be a known bot, and each `net:` name's file in `nets_dir` must load and
    /// validate. The error names the first bad bot. A malformed name is rejected before any file
    /// is read.
    pub fn new(names: &[&str], nets_dir: &Path) -> Result<Roster, String> {
        let mut nets: Vec<(String, Arc<Weights>)> = Vec::new();
        for &name in names {
            if let Some((net, _, _)) = netbot::parse_net_name(name) {
                if !nets.iter().any(|(n, _)| n == net) {
                    nets.push((net.to_owned(), netbot::load_net(nets_dir, net)?));
                }
            } else if make_bot(name, 0).is_none() {
                return Err(unknown_bot(name));
            }
        }
        Ok(Roster { nets })
    }

    /// A fresh bot; `None` for a name this roster was not built with.
    pub fn make(&self, name: &str, seed: u64) -> Option<Box<dyn Bot>> {
        match netbot::parse_net_name(name) {
            Some((net, simulations, rollout)) => {
                let (_, w) = self.nets.iter().find(|(n, _)| n == net)?;
                Some(Box::new(netbot::net_bot(seed, w.clone(), simulations, rollout)))
            }
            None => make_bot(name, seed),
        }
    }
}
```

In `bots/src/arena.rs`, replace the imports with

```rust
use crate::netbot::DEFAULT_NETS_DIR;
use crate::{Bot, Roster};
use settler_engine::legal::legal_actions;
use settler_engine::rng::mix;
use settler_engine::{Event, GameConfig, PlayerId, State, NUM_PLAYERS};
use std::ops::Range;
use std::path::Path;
```

and replace the whole `run_match` function (doc comment included) with:

```rust
/// Every seed in `seeds`, four games each (candidate in seat 0..4), spread over `threads`
/// (capped at the seed count and at 4x the available cores). Records are sorted by
/// (seed, candidate seat) and independent of `threads`. At most `MAX_SEEDS` seeds per call.
/// `net:` names load from `DEFAULT_NETS_DIR`.
pub fn run_match(
    candidate: &str,
    baseline: &str,
    seeds: Range<u64>,
    config: GameConfig,
    threads: usize,
) -> Result<Vec<GameRecord>, String> {
    run_match_in(candidate, baseline, seeds, config, threads, Path::new(DEFAULT_NETS_DIR))
}

/// `run_match` with `net:` names loaded from `nets_dir`; names and networks are checked before
/// any game starts.
pub fn run_match_in(
    candidate: &str,
    baseline: &str,
    seeds: Range<u64>,
    config: GameConfig,
    threads: usize,
    nets_dir: &Path,
) -> Result<Vec<GameRecord>, String> {
    let roster = Roster::new(&[candidate, baseline], nets_dir)?;
    run_match_by(seeds, config, threads, &|is_candidate, seed| {
        let name = if is_candidate { candidate } else { baseline };
        roster.make(name, seed).expect("checked by Roster::new")
    })
}

/// The match loop with bots from `make(is_candidate, bot_seed)`, for tools that build bots
/// directly (for example with a non-default leaf batch).
pub fn run_match_by(
    seeds: Range<u64>,
    config: GameConfig,
    threads: usize,
    make: &(dyn Fn(bool, u64) -> Box<dyn Bot> + Sync),
) -> Result<Vec<GameRecord>, String> {
    config.validate()?;
    let n = seeds.end.saturating_sub(seeds.start);
    if n > MAX_SEEDS {
        return Err(format!("at most {MAX_SEEDS} seeds per match, got {n}"));
    }
    if n == 0 {
        return Ok(Vec::new());
    }
    let parts = split_seeds(seeds, threads);
    let mut records: Vec<GameRecord> = std::thread::scope(|scope| {
        let workers: Vec<_> = parts
            .into_iter()
            .map(|part| {
                scope.spawn(move || {
                    let mut out =
                        Vec::with_capacity((part.end - part.start) as usize * NUM_PLAYERS);
                    for seed in part {
                        for cand in 0..NUM_PLAYERS {
                            let mut bots: Vec<Box<dyn Bot>> = (0..NUM_PLAYERS)
                                .map(|seat| make(seat == cand, bot_seed(seed, seat)))
                                .collect();
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
                        }
                    }
                    out
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
    records.sort_by_key(|r| (r.seed, r.candidate_seat));
    Ok(records)
}
```

- [ ] **Step 5: Run the Rust tests to verify they pass**

Run: `~/.cargo/bin/cargo test --release -p settler-net -p settler-bots`
Expected: all pass, including 3a's `bots/tests/ismcts.rs`, `arena.rs` and `selfplay.rs` unchanged (`unknown_bot_and_bad_config_are_errors` still sees "unknown bot"), `evaluator` 6 passed and `netbot` 3 passed.

- [ ] **Step 6: Write the failing Python tests**

Create `tests/python/test_netbot.py`:

```python
import subprocess
import sys

import pytest

from alphasettler import Bot, Game
from alphasettler._engine import DEFAULT_NETS_DIR, run_match, write_random_net

CONFIG = {"max_offers_per_turn": 0}
BAD = ["net:", "net:../x", "net:a/b", "net:a\\b", "net:.", "net:a b", "net:" + "a" * 65, "net:a@0",
       "net:a@300+r0", "net:a@5@6"]


def cli(*args):
    return subprocess.run([sys.executable, "-m", "alphasettler", *args], capture_output=True, text=True)


@pytest.fixture
def nets(tmp_path):
    d = tmp_path / "nets"
    d.mkdir()
    write_random_net(str(d / "tiny.bin"), 1, 16, 8)
    stale = bytearray((d / "tiny.bin").read_bytes())
    stale[12] ^= 1
    (d / "stale.bin").write_bytes(bytes(stale))
    return d


def test_the_default_nets_dir():
    assert DEFAULT_NETS_DIR == "runs/3b/nets"


def test_a_net_bot_plays_legal_moves_from_python(nets):
    g = Game(4, CONFIG)
    bot = Bot("net:tiny@10", 1, nets_dir=str(nets))
    assert bot.name == "net"
    for _ in range(80):
        if g.is_over:
            break
        a = bot.act(g)
        assert a in g.legal_actions()
        g.apply(a)


def test_bad_names_missing_files_and_stale_hashes_are_errors(nets):
    for bad in BAD:
        with pytest.raises(ValueError, match="unknown bot"):
            Bot(bad, 0, nets_dir=str(nets))
    with pytest.raises(ValueError, match="cannot read"):
        Bot("net:missing", 0, nets_dir=str(nets))
    with pytest.raises(ValueError, match="encoding hash"):
        Bot("net:stale", 0, nets_dir=str(nets))
    with pytest.raises(ValueError, match="encoding hash"):
        run_match("net:stale", "greedy", 0, 1, 1, CONFIG, str(nets))
    with pytest.raises(ValueError, match="trunk"):
        write_random_net(str(nets / "x.bin"), 1, 0, 8)


def test_cli_arena_and_compare_take_net_names(nets, tmp_path):
    out = tmp_path / "a.jsonl"
    r = cli("arena", "--candidate", "net:tiny@5", "--baseline", "greedy", "--seeds", "1", "--no-trades",
            "--nets-dir", str(nets), "--out", str(out))
    assert r.returncode == 0, r.stderr
    assert "belief resets: 0" in r.stdout and len(out.read_text().splitlines()) == 4
    r = cli("compare", "--a", "net:tiny@5", "--b", "greedy", "--baseline", "random", "--seeds", "1",
            "--no-trades", "--nets-dir", str(nets), "--out-dir", str(tmp_path / "cmp"))
    assert r.returncode == 0, r.stderr
    assert "paired net:tiny@5 - greedy:" in r.stdout


def test_cli_rejects_bad_net_names_and_files_before_writing_anything(nets, tmp_path):
    for bad, message in [*[(b, "unknown bot") for b in BAD], ("net:missing", "cannot read"),
                         ("net:stale", "encoding hash")]:
        out = tmp_path / "never.jsonl"
        r = cli("arena", "--candidate", bad, "--baseline", "greedy", "--seeds", "1", "--no-trades",
                "--nets-dir", str(nets), "--out", str(out))
        assert r.returncode == 2 and message in r.stderr, (bad, r.stderr)
        assert not out.exists()
        r = cli("compare", "--a", "greedy", "--b", bad, "--baseline", "random", "--seeds", "1",
                "--nets-dir", str(nets), "--out-dir", str(tmp_path / "never"))
        assert r.returncode == 2 and message in r.stderr, (bad, r.stderr)
        assert not (tmp_path / "never").exists()
```

Append to `tests/python/test_oracle_arena.py`:

```python
def test_net_names_inside_catanatron(tmp_path):
    from alphasettler._engine import write_random_net

    nets = tmp_path / "nets"
    nets.mkdir()
    write_random_net(str(nets / "tiny.bin"), 1, 16, 8)
    stale = bytearray((nets / "tiny.bin").read_bytes())
    stale[12] ^= 1
    (nets / "stale.bin").write_bytes(bytes(stale))
    for bad in ["net:../x", "net:a/b", "net:", "net:a@0"]:
        with pytest.raises(ValueError, match="unknown bot"):
            run_match(bad, "random", seeds=1, nets_dir=str(nets))
    with pytest.raises(ValueError, match="cannot read"):
        run_match("net:missing", "random", seeds=1, nets_dir=str(nets))
    with pytest.raises(ValueError, match="encoding hash"):
        run_match("net:stale", "random", seeds=1, nets_dir=str(nets))
    recs = run_match("net:tiny@5", "random", seeds=1, workers=2, nets_dir=str(nets))
    assert len(recs) == 4
    assert all(r["fallbacks"] == 0 and r["belief_resets"] == 0 for r in recs)
    r = subprocess.run([sys.executable, "-m", "alphasettler", "arena", "--candidate", "net:../x",
                        "--baseline", "catanatron:random", "--seeds", "1"], capture_output=True, text=True)
    assert r.returncode == 2 and "unknown bot" in r.stderr
```

- [ ] **Step 7: Run them to verify they fail**

Run: `PATH=$HOME/.cargo/bin:$PATH .venv/bin/maturin develop --release -q && .venv/bin/pytest -q tests/python/test_netbot.py tests/python/test_oracle_arena.py::test_net_names_inside_catanatron`
Expected: FAIL with `ImportError: cannot import name 'DEFAULT_NETS_DIR'` (and `write_random_net`).

- [ ] **Step 8: Bindings**

In `bindings/Cargo.toml`, under `[dependencies]` add:

```toml
settler-net = { path = "../net" }
settler-search = { path = "../search" }
```

Create `bindings/src/train.rs`:

```rust
//! Network helpers for Python: test weights here; encode, the training sampler and the forward
//! pass join in Plan 5 Task 5.

use crate::convert::*;
use pyo3::prelude::*;
use settler_net::weights::{Arch, Weights, MAX_WIDTH};
use std::path::PathBuf;

/// Random weights (the default shape unless `trunk` / `value` are given) saved to `path`, for
/// tests and measurements.
#[pyfunction]
#[pyo3(signature = (path, seed, trunk=None, value=None))]
fn write_random_net(
    path: PathBuf,
    seed: &Bound<'_, PyAny>,
    trunk: Option<&Bound<'_, PyAny>>,
    value: Option<&Bound<'_, PyAny>>,
) -> PyResult<()> {
    let seed: u64 = int_arg(seed, "seed")?;
    let width = |v: Option<&Bound<'_, PyAny>>, default: usize, what: &str| -> PyResult<usize> {
        let w: usize = match v {
            Some(v) => int_arg(v, what)?,
            None => default,
        };
        if !(1..=MAX_WIDTH).contains(&w) {
            return Err(value_error(format!("{what} must be 1..={MAX_WIDTH}, got {w}")));
        }
        Ok(w)
    };
    let arch = Arch {
        trunk: width(trunk, Arch::DEFAULT.trunk, "trunk")?,
        value: width(value, Arch::DEFAULT.value, "value")?,
    };
    Weights::random(arch, seed).save(&path).map_err(value_error)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(write_random_net, m)?)?;
    Ok(())
}
```

In `bindings/src/lib.rs`:
- after `mod convert;` add `mod train;`;
- replace `PyBot::new` with:

```rust
    #[new]
    #[pyo3(signature = (name, seed, nets_dir=None))]
    fn new(name: &str, seed: &Bound<'_, PyAny>, nets_dir: Option<PathBuf>) -> PyResult<Self> {
        let seed = int_arg::<u64>(seed, "seed")?;
        let dir = nets_dir.unwrap_or_else(|| PathBuf::from(settler_bots::netbot::DEFAULT_NETS_DIR));
        let roster = settler_bots::Roster::new(&[name], &dir).map_err(value_error)?;
        let bot = roster.make(name, seed).expect("checked by Roster::new");
        Ok(PyBot {
            name: bot.name(),
            inner: std::sync::Mutex::new(bot),
        })
    }
```

- add `use std::path::PathBuf;` to the imports;
- in `run_match`, change the signature attribute to `#[pyo3(signature = (candidate, baseline, seed_start, seeds, threads, config=None, nets_dir=None))]`, add `#[allow(clippy::too_many_arguments)]` on the line after it (the function now takes eight), add the parameter `nets_dir: Option<PathBuf>,` after `config`, and replace the `py.detach(...)` call with:

```rust
    let dir = nets_dir.unwrap_or_else(|| PathBuf::from(settler_bots::netbot::DEFAULT_NETS_DIR));
    let records = py
        .detach(|| {
            settler_bots::arena::run_match_in(
                &candidate,
                &baseline,
                seed_start..seed_end,
                config,
                threads,
                &dir,
            )
        })
        .map_err(value_error)?;
```

- in `_engine(...)`, before `Ok(())`, add:

```rust
    m.add("DEFAULT_NETS_DIR", settler_bots::netbot::DEFAULT_NETS_DIR)?;
    train::register(m)?;
```

- [ ] **Step 9: Python pass-through**

In `alphasettler/arena.py`, replace `check` and `run` with:

```python
def check(
    candidate: str,
    baseline: str,
    seeds: int,
    seed_start: int = 0,
    threads: int | None = None,
    config: dict | None = None,
    nets_dir: str | Path | None = None,
) -> None:
    """Raise ValueError for anything `run` would reject, without playing a game."""
    if seeds <= 0:
        raise ValueError("seeds must be positive")
    if seeds > MAX_SEEDS:
        raise ValueError(f"at most {MAX_SEEDS} seeds per match, got {seeds}")
    if seed_start + seeds > SEED_LIMIT:
        raise ValueError(f"seed_start {seed_start} + seeds {seeds} exceeds the u64 seed range")
    # A zero-seed native call checks bot names, nets, config and integer ranges and plays nothing.
    run_match(candidate, baseline, seed_start, 0, threads or os.cpu_count() or 1, config,
              None if nets_dir is None else str(nets_dir))


def run(
    candidate: str,
    baseline: str,
    seeds: int,
    seed_start: int = 0,
    threads: int | None = None,
    config: dict | None = None,
    out: str | Path | TextIO | None = None,
    nets_dir: str | Path | None = None,
) -> tuple[list[dict], Summary]:
    """Play `seeds` seeds from `seed_start` in batches of `BATCH_SEEDS`, appending each batch to
    `out` (a path, opened exclusively, or an open file the caller owns). Inputs are checked and a
    path is opened before any game is played, so a bad argument or path costs nothing. `net:`
    names load from `nets_dir` (default `runs/3b/nets`)."""
    check(candidate, baseline, seeds, seed_start, threads, config, nets_dir)
    threads = threads or os.cpu_count() or 1
    dir_arg = None if nets_dir is None else str(nets_dir)
    owned = isinstance(out, (str, os.PathLike))
    f = open_output(out) if owned else out
    records: list[dict] = []
    try:
        end = seed_start + seeds
        for lo in range(seed_start, end, BATCH_SEEDS):
            batch = run_match(candidate, baseline, lo, min(BATCH_SEEDS, end - lo), threads, config, dir_arg)
            if f is not None:
                _write_records(f, candidate, baseline, batch)
            records.extend(batch)
    finally:
        if owned:
            f.close()
    return records, summarize(records)
```

In `oracle/arena.py`:
- `AlphaSettlerPlayer.__init__` becomes `def __init__(self, color, bot_name: str, seed: int, nets_dir: str | None = None):` with `self.bot = Bot(bot_name, seed, nets_dir)`;
- `play_game` becomes `def play_game(seed: int, candidate: str, baseline: str, candidate_seat: int, nets_dir: str | None = None) -> dict:` and builds the candidate with `AlphaSettlerPlayer(COLORS[i], candidate, seed * 4 + i, nets_dir)`;
- `check` becomes `def check(candidate: str, baseline: str, seeds: int, seed_start: int = 0, nets_dir: str | None = None) -> None:` with `Bot(candidate, 0, nets_dir)  # raises ValueError for a bad name, a missing or invalid net`;
- `run_match` becomes:

```python
def run_match(candidate: str, baseline: str, seeds: int, seed_start: int = 0, workers: int | None = None,
              nets_dir: str | None = None) -> list[dict]:
    check(candidate, baseline, seeds, seed_start, nets_dir)
    jobs = [(s, candidate, baseline, k, nets_dir) for s in range(seed_start, seed_start + seeds) for k in range(4)]
    with hash_seed_zero():
        pool = mp.get_context("spawn").Pool(min(workers or os.cpu_count() or 1, len(jobs)))
    with pool:
        records = pool.starmap(play_game, jobs)
    return sorted(records, key=lambda r: (r["seed"], r["candidate_seat"]))
```

In `alphasettler/cli.py`:
- change the engine import to `from alphasettler._engine import DEFAULT_NETS_DIR, Bot, bot_names`;
- in `_parser`, add to both the `arena` parser (`a`) and the `compare` parser (`c`):

```python
    a.add_argument("--nets-dir", default=DEFAULT_NETS_DIR, help="where net:NAME bots find NAME.bin")
```

(with `c.add_argument(...)` for compare);
- replace `_check_bot_name` with:

```python
def _check_bot_name(name: str, nets_dir: str) -> None:
    """ValueError naming `name` if no bot answers to it (`ismcts@N[+rD]` and `net:NAME[@N[+rD]]`
    included). A well-formed `net:` name whose file is missing or invalid raises that error."""
    try:
        Bot(name, 0, nets_dir)
    except ValueError as e:
        if not str(e).startswith("unknown bot"):
            raise
        raise ValueError(
            f"unknown bot {name!r}; known bots: {bot_names()} (or ismcts@N, ismcts@N+rD, net:NAME[@N[+rD]])"
        ) from None
```

- in `_catanatron_arena`, pass the directory through, resolved so spawned workers see the same path:

```python
    nets_dir = str(Path(args.nets_dir).resolve())
    oracle.arena.check(args.candidate, baseline, args.seeds, args.seed_start, nets_dir)
    # Open the output before the run, so a bad path fails before any game is played.
    with open_output(out) as f:
        records = oracle.arena.run_match(args.candidate, baseline, args.seeds, args.seed_start, args.threads, nets_dir)
```

- in `main`, the native arena line becomes `records, summary = run(args.candidate, args.baseline, args.seeds, args.seed_start, args.threads, config, out, args.nets_dir)`; in the compare branch, `_check_bot_name(name, args.nets_dir)`, `check(name, args.baseline, args.seeds, args.seed_start, args.threads, config, args.nets_dir)` and `run(name, args.baseline, args.seeds, args.seed_start, args.threads, config, f, args.nets_dir)`.

- [ ] **Step 10: Run all tests**

Run: `PATH=$HOME/.cargo/bin:$PATH .venv/bin/maturin develop --release -q && .venv/bin/pytest -q`
Expected: all pass (3a's `test_cli_compare_validates_both_bots_before_running` still prints `unknown bot 'nope'`).

- [ ] **Step 11: Format, lint, full checks**

Run: `~/.cargo/bin/cargo fmt --all && ~/.cargo/bin/cargo fmt --check`
Expected: no output from the check.
Run: `~/.cargo/bin/cargo clippy --workspace --release --all-targets -- -D warnings`
Expected: exits 0.
Run: `~/.cargo/bin/cargo test --workspace --release`
Expected: all pass.

- [ ] **Step 12: Commit**

```bash
git add Cargo.lock net/ bots/ bindings/ alphasettler/arena.py alphasettler/cli.py oracle/arena.py tests/python/test_netbot.py tests/python/test_oracle_arena.py
git diff --cached | grep -nE 'sk-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{36}|password=[^[:space:]]+'
git -c user.email=sean.may101@gmail.com commit -m "net, bots: NetEvaluator and net:NAME bots in every arena"
```

---

### Task 4: Self-play exploration: playout cap randomization, root noise and opening sampling

**Files:**
- Create: `search/src/noise.rs`
- Modify: `search/src/lib.rs`, `bots/src/ismcts.rs`, `bots/src/selfplay.rs`
- Test: `search/tests/noise.rs`, `bots/tests/selfplay.rs` (imports and appended tests), `bots/tests/ismcts.rs` (append), `bots/tests/netbot.rs` (append)

**Interfaces:**
- Consumes: `IsmctsBot::with_evaluator`, `NetEvaluator::new`, `NET_BATCH` (Task 3); `settler_search::{search, Evaluator, Eval, Leaf, SearchConfig}`; `arena::{bot_seed, split_seeds, EventFeed, belief_resets, MAX_SEEDS}`.
- Produces:
  - `settler_search::noise` (re-exported from `settler_search`): `RootNoise { weight: f32, alpha_total: f32 }`; `unit(&mut Rng) -> f64` (uniform in [0, 1)); `normal(&mut Rng) -> f64`; `gamma(shape: f64, &mut Rng) -> f64`; `dirichlet(alpha: f64, n: usize, &mut Rng) -> Vec<f32>`; `Noised::new(inner: &mut E, noise: RootNoise, seed: u64) -> Noised<'_, E>` implementing `Evaluator`.
  - `settler_bots::ismcts`: `MoveOptions { simulations: u32, noise: Option<(RootNoise, u64)>, sample: Option<u64> }`; `sample_by_visits(legal: &[Action], visits: &[u32], seed: u64) -> Option<Action>`; `IsmctsBot::act_with(&mut self, obs: &Observation, legal: &[Action], opts: &MoveOptions) -> Action`. After a sampled move, `take_last_search().action` is the move played.
  - `settler_bots::selfplay`: constants `FULL_SIMULATIONS` (600), `FAST_SIMULATIONS` (100), `FULL_PROBABILITY` (0.25), `NOISE` (`RootNoise { weight: 0.25, alpha_total: 10.0 }`), `SAMPLE_TURNS` (8), `MIN_SIMULATIONS` (2, unchanged); `Exploration { full_simulations, fast_simulations, full_probability: f64, noise: Option<RootNoise>, sample_turns: u32 }` with `Exploration::fixed(simulations)` (3a), `Exploration::standard()` (3b) and `validate(&self) -> Result<(), String>`; `play_with<E, F>(seed, config, ex: &Exploration, batch: usize, make_eval: &F) -> SelfPlayGame` and `run_with<E, F>(seeds, config, ex: &Exploration, batch: usize, make_eval: &F, threads) -> Result<Vec<SelfPlayGame>, String>` where `E: Evaluator + Send, F: Fn() -> E (+ Sync for run_with)`. `play` and `run` keep their 3a signatures and results.

- [ ] **Step 1: Write the failing tests**

Create `search/tests/noise.rs`:

```rust
mod common;

use common::*;
use settler_engine::rng::Rng;
use settler_engine::*;
use settler_search::tree::ROOT;
use settler_search::{
    dirichlet, gamma, search_actions, search_tree, Belief, Evaluator, Leaf, Noised, RootNoise,
    SearchConfig, UniformEvaluator,
};

const NOISE: RootNoise = RootNoise {
    weight: 0.25,
    alpha_total: 10.0,
};

/// Player 0 in Main after setup holding two of every resource: several searchable moves.
fn main_position(seed: u64) -> State {
    edit(&after_setup(seed, no_trades()), no_trades(), |snap| {
        give_cards(snap, 0, [2, 2, 2, 2, 2]);
        snap.phase = Phase::Main;
    })
}

#[test]
fn gamma_variates_have_the_right_mean_and_variance() {
    for shape in [0.2, 1.0, 3.5] {
        let mut rng = Rng::new(7);
        let n = 40_000;
        let xs: Vec<f64> = (0..n).map(|_| gamma(shape, &mut rng)).collect();
        assert!(xs.iter().all(|&x| x >= 0.0 && x.is_finite()));
        let mean = xs.iter().sum::<f64>() / n as f64;
        let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
        assert!((mean - shape).abs() < 4.0 * (shape / n as f64).sqrt(), "shape {shape}: mean {mean}");
        assert!((var - shape).abs() < 0.1 * shape, "shape {shape}: variance {var}");
    }
}

#[test]
fn dirichlet_draws_are_distributions_with_even_means() {
    let mut rng = Rng::new(3);
    let (n, draws) = (5, 20_000);
    let mut mean = [0.0f64; 5];
    for _ in 0..draws {
        let d = dirichlet(10.0 / n as f64, n, &mut rng);
        assert_eq!(d.len(), n);
        assert!((d.iter().map(|&x| x as f64).sum::<f64>() - 1.0).abs() < 1e-5);
        assert!(d.iter().all(|&x| x >= 0.0));
        for i in 0..n {
            mean[i] += d[i] as f64 / draws as f64;
        }
    }
    assert!(mean.iter().all(|m| (m - 0.2).abs() < 0.01), "{mean:?}");
}

#[test]
fn noise_changes_only_the_first_evaluation() {
    let s = main_position(4);
    let mut legal = Vec::new();
    search_actions(&s.legal_actions(), &mut legal);
    let n = legal.len();
    assert!(n >= 3, "{legal:?}");
    let mut inner = UniformEvaluator;
    let mut noised = Noised::new(&mut inner, NOISE, 99);
    let first = noised.evaluate(&[Leaf { world: &s, actor: 0, legal: &legal }]);
    let eta = dirichlet(10.0 / n as f64, n, &mut Rng::new(99));
    for (p, e) in first[0].prior.iter().zip(&eta) {
        assert!((p - (0.75 * (1.0 / n as f32) + 0.25 * e)).abs() < 1e-6);
    }
    assert!((first[0].prior.iter().sum::<f32>() - 1.0).abs() < 1e-5);
    let second = noised.evaluate(&[Leaf { world: &s, actor: 0, legal: &legal }]);
    assert_eq!(second[0].prior, vec![1.0 / n as f32; n]);
}

#[test]
fn the_first_evaluation_of_a_search_is_the_root() {
    let s = main_position(5);
    let obs = s.observation(0);
    let all = s.legal_actions();
    let mut legal = Vec::new();
    search_actions(&all, &mut legal);
    let n = legal.len();
    let belief = Belief::from_observation(&obs, 64, 1);
    let mut inner = UniformEvaluator;
    let mut noised = Noised::new(&mut inner, NOISE, 42);
    let cfg = SearchConfig {
        simulations: 60,
        ..SearchConfig::default()
    };
    let (_, tree) = search_tree(&obs, &all, &belief, &mut noised, &cfg, &mut Rng::new(5)).unwrap();
    let eta = dirichlet(10.0 / n as f64, n, &mut Rng::new(42));
    let root = &tree.nodes[ROOT as usize];
    assert_eq!(root.children.len(), n);
    for ((c, a), e) in root.children.iter().zip(&legal).zip(&eta) {
        assert_eq!(c.key, a.encode());
        assert!((c.prior - (0.75 * (1.0 / n as f32) + 0.25 * e)).abs() < 1e-6, "{a:?}");
    }
}
```

In `bots/tests/selfplay.rs`, replace the two `use settler_bots::...` statements at the top (`use settler_bots::heuristic::{...};` and `use settler_bots::selfplay::{play, run, MIN_SIMULATIONS};`) with:

```rust
use settler_bots::heuristic::{
    fit, log_likelihood, samples_from_games, HeuristicEvaluator, ReplayGame, Sample,
    DEFAULT_WEIGHTS, NUM_FEATURES,
};
use settler_bots::selfplay::{
    play, play_with, run, run_with, Exploration, FAST_SIMULATIONS, FULL_PROBABILITY,
    FULL_SIMULATIONS, MIN_SIMULATIONS, NOISE, SAMPLE_TURNS,
};
```

and append:

```rust
fn small_explore() -> Exploration {
    Exploration {
        full_simulations: 6,
        fast_simulations: 3,
        ..Exploration::standard()
    }
}

fn heuristic() -> HeuristicEvaluator {
    HeuristicEvaluator::new(DEFAULT_WEIGHTS, 0)
}

#[test]
fn the_standard_settings_are_the_specs() {
    let ex = Exploration::standard();
    assert_eq!((ex.full_simulations, ex.fast_simulations), (FULL_SIMULATIONS, FAST_SIMULATIONS));
    assert_eq!((FULL_SIMULATIONS, FAST_SIMULATIONS, FULL_PROBABILITY), (600, 100, 0.25));
    assert_eq!((NOISE.weight, NOISE.alpha_total, SAMPLE_TURNS), (0.25, 10.0, 8));
    assert_eq!(ex.noise, Some(NOISE));
    for bad in [
        Exploration { full_simulations: 1, ..ex },
        Exploration { fast_simulations: 1_000_001, ..ex },
        Exploration { full_probability: 1.5, ..ex },
        Exploration { full_probability: f64::NAN, ..ex },
    ] {
        assert!(bad.validate().is_err(), "{bad:?}");
        assert!(run_with(0..1, no_trades(), &bad, 8, &heuristic, 1).is_err());
    }
}

#[test]
fn about_a_quarter_of_decisions_are_full_searches() {
    let games = run_with(0..16, no_trades(), &small_explore(), 8, &heuristic, 8).unwrap();
    let decisions: Vec<_> = games.iter().flat_map(|g| &g.decisions).collect();
    let n = decisions.len() as f64;
    assert!(n >= 2000.0, "only {n} decisions");
    let full = decisions.iter().filter(|d| d.full_search).count() as f64;
    let se = (0.25 * 0.75 / n).sqrt();
    assert!((full / n - 0.25).abs() < 3.0 * se, "{full} full searches of {n}");
    for d in &decisions {
        let total: u32 = d.visits.iter().sum();
        assert_eq!(total, if d.full_search { 5 } else { 2 });
    }
    assert!(games.iter().all(|g| g.belief_resets == 0));
}

#[test]
fn exploring_self_play_is_deterministic_and_thread_independent() {
    let a = run_with(0..4, no_trades(), &small_explore(), 8, &heuristic, 1).unwrap();
    let b = run_with(0..4, no_trades(), &small_explore(), 8, &heuristic, 3).unwrap();
    assert_eq!(a, b);
    assert_eq!(a[2], play_with(2, no_trades(), &small_explore(), 8, &heuristic));
    assert_ne!(a[0].actions, play(0, no_trades(), 6, 0).actions, "exploration changes the games");
}

#[test]
fn opening_moves_are_sampled_and_later_moves_are_the_most_visited() {
    let games = run_with(0..6, no_trades(), &small_explore(), 8, &heuristic, 3).unwrap();
    let mut off_best = 0;
    for g in &games {
        let mut s = State::new(g.seed, no_trades());
        let mut decisions = g.decisions.iter().peekable();
        for (i, &a) in g.actions.iter().enumerate() {
            if let Some(d) = decisions.next_if(|d| d.index as usize == i) {
                let chosen = d.visits[d.legal.iter().position(|&x| x == a).unwrap()];
                let best = *d.visits.iter().max().unwrap();
                assert!(chosen > 0, "move {i} of seed {} was never searched", g.seed);
                if s.turn < SAMPLE_TURNS {
                    off_best += (chosen < best) as u32;
                } else {
                    assert_eq!(chosen, best, "move {i} of seed {}", g.seed);
                }
            }
            s.apply(a);
        }
    }
    assert!(off_best > 0, "no opening move was sampled away from the most-visited one");
}

#[test]
fn fixed_exploration_is_3a_self_play() {
    let ex = Exploration::fixed(10);
    assert_eq!(play_with(3, no_trades(), &ex, 8, &heuristic), play(3, no_trades(), 10, 0));
    assert!(play(3, no_trades(), 10, 0).decisions.iter().all(|d| d.full_search));
}
```

Append to `bots/tests/ismcts.rs`:

```rust
#[test]
fn act_with_plain_options_is_act_and_a_sampled_move_was_visited() {
    use settler_bots::ismcts::{sample_by_visits, MoveOptions};
    let s = main_with(5, CITY_COST, no_trades());
    let obs = s.observation(0);
    let legal = s.legal_actions();
    let (mut a, mut b) = (IsmctsBot::new(5, 40, 0), IsmctsBot::new(5, 40, 0));
    let plain = MoveOptions { simulations: 40, noise: None, sample: None };
    assert_eq!(decide(&mut a, &s), b.act_with(&obs, &legal, &plain));
    for seed in 0..20 {
        let mut c = IsmctsBot::new(5, 40, 0);
        let x = c.act_with(&obs, &legal, &MoveOptions { sample: Some(seed), ..plain });
        let r = c.take_last_search().unwrap();
        assert_eq!(r.action, x, "the recorded action is the move played");
        assert!(r.visits[legal.iter().position(|&l| l == x).unwrap()] > 0);
    }
    assert_eq!(sample_by_visits(&legal, &vec![0; legal.len()], 1), None);
    let only = legal.len() - 1;
    let mut visits = vec![0; legal.len()];
    visits[only] = 3;
    assert_eq!(sample_by_visits(&legal, &visits, 7), Some(legal[only]));
}
```

Append to `bots/tests/netbot.rs`:

```rust
#[test]
fn net_self_play_explores_deterministically() {
    use settler_bots::selfplay::{run_with, Exploration};
    use settler_net::evaluator::NetEvaluator;
    use std::sync::Arc;
    let w = Arc::new(Weights::random(Arch { trunk: 16, value: 8 }, 3));
    let make = || NetEvaluator::new(w.clone());
    let ex = Exploration { full_simulations: 6, fast_simulations: 3, ..Exploration::standard() };
    let a = run_with(0..2, no_trades(), &ex, NET_BATCH, &make, 1).unwrap();
    let b = run_with(0..2, no_trades(), &ex, NET_BATCH, &make, 2).unwrap();
    assert_eq!(a, b);
    assert!(a.iter().all(|g| g.belief_resets == 0 && !g.decisions.is_empty()));
    assert!(a.iter().flat_map(|g| &g.decisions).any(|d| !d.full_search));
}
```

- [ ] **Step 2: Run them to verify they fail**

Run: `~/.cargo/bin/cargo test --release -p settler-search --test noise; ~/.cargo/bin/cargo test --release -p settler-bots`
Expected: compile errors: `unresolved imports settler_search::dirichlet, gamma, Noised, RootNoise`; `cannot find play_with, run_with, Exploration in selfplay`; `no method named act_with`.

- [ ] **Step 3: Write the noise module**

Create `search/src/noise.rs`:

```rust
//! Root exploration for self-play (spec Section 4): Dirichlet noise mixed into the root prior,
//! from gamma variates drawn with Marsaglia–Tsang from the engine's seeded `Rng`. The search loop
//! itself is unchanged: `Noised` wraps the evaluator.

use crate::evaluator::{Eval, Evaluator, Leaf};
use settler_engine::rng::Rng;

/// `prior' = (1 - weight) · prior + weight · Dir(alpha_total / n)` over the root's n moves.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RootNoise {
    pub weight: f32,
    pub alpha_total: f32,
}

/// Uniform in [0, 1).
pub fn unit(rng: &mut Rng) -> f64 {
    (rng.next_u64() >> 11) as f64 / (1u64 << 53) as f64
}

/// Uniform in (0, 1): safe to take the log of.
fn open_unit(rng: &mut Rng) -> f64 {
    ((rng.next_u64() >> 11) as f64 + 0.5) / (1u64 << 53) as f64
}

/// A standard normal variate (Box–Muller).
pub fn normal(rng: &mut Rng) -> f64 {
    let (u1, u2) = (open_unit(rng), open_unit(rng));
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

/// A Gamma(shape, 1) variate (Marsaglia and Tsang, 2000). Shapes below 1 use the boost
/// Gamma(shape) = Gamma(shape + 1) · U^(1 / shape).
pub fn gamma(shape: f64, rng: &mut Rng) -> f64 {
    assert!(
        shape > 0.0 && shape.is_finite(),
        "gamma shape must be positive, got {shape}"
    );
    if shape < 1.0 {
        let u = open_unit(rng);
        return gamma(shape + 1.0, rng) * u.powf(1.0 / shape);
    }
    let d = shape - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();
    loop {
        let x = normal(rng);
        let v = 1.0 + c * x;
        if v <= 0.0 {
            continue;
        }
        let v = v * v * v;
        let u = open_unit(rng);
        if u < 1.0 - 0.0331 * x.powi(4) || u.ln() < 0.5 * x * x + d * (1.0 - v + v.ln()) {
            return d * v;
        }
    }
}

/// A symmetric Dirichlet(alpha, ..., alpha) draw over `n` entries.
pub fn dirichlet(alpha: f64, n: usize, rng: &mut Rng) -> Vec<f32> {
    let g: Vec<f64> = (0..n).map(|_| gamma(alpha, rng)).collect();
    let total: f64 = g.iter().sum();
    if total > 0.0 {
        g.iter().map(|&x| (x / total) as f32).collect()
    } else {
        vec![1.0 / n as f32; n]
    }
}

/// An evaluator that mixes `noise` into the prior of the first leaf it evaluates. `search`
/// evaluates the root alone in its first batch (the first simulation), so only the root's prior
/// changes. Make a fresh `Noised` for every search.
pub struct Noised<'a, E> {
    inner: &'a mut E,
    noise: RootNoise,
    rng: Rng,
    pending: bool,
}

impl<'a, E: Evaluator> Noised<'a, E> {
    pub fn new(inner: &'a mut E, noise: RootNoise, seed: u64) -> Noised<'a, E> {
        Noised {
            inner,
            noise,
            rng: Rng::new(seed),
            pending: true,
        }
    }
}

impl<E: Evaluator> Evaluator for Noised<'_, E> {
    fn evaluate(&mut self, leaves: &[Leaf<'_>]) -> Vec<Eval> {
        let mut evals = self.inner.evaluate(leaves);
        if self.pending {
            if let Some(e) = evals.first_mut() {
                let n = e.prior.len();
                let eta = dirichlet(self.noise.alpha_total as f64 / n as f64, n, &mut self.rng);
                let w = self.noise.weight;
                for (p, x) in e.prior.iter_mut().zip(eta) {
                    *p = (1.0 - w) * *p + w * x;
                }
                self.pending = false;
            }
        }
        evals
    }
}
```

In `search/src/lib.rs`, add `pub mod noise;` to the module list (after `evaluator`) and, after the `pub use evaluator::...` line:

```rust
pub use noise::{dirichlet, gamma, normal, unit, Noised, RootNoise};
```

- [ ] **Step 4: `act_with` and move sampling**

In `bots/src/ismcts.rs`, change the `settler_search` import to:

```rust
use settler_search::{
    search, search_actions, Belief, Evaluator, Noised, RootNoise, SearchConfig, SearchResult,
    DEFAULT_MAX_STATES,
};
```

Add after `parse_name`:

```rust
/// How one decision is searched and chosen; self-play varies these move by move.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MoveOptions {
    pub simulations: u32,
    /// Dirichlet noise mixed into the root prior, drawn from `Rng::new(seed)`.
    pub noise: Option<(RootNoise, u64)>,
    /// Play a move drawn in proportion to root visits (from `Rng::new(seed)`) instead of the
    /// most-visited one.
    pub sample: Option<u64>,
}

/// An entry of `legal` drawn in proportion to `visits` (aligned with it); `None` if nothing was
/// visited.
pub fn sample_by_visits(legal: &[Action], visits: &[u32], seed: u64) -> Option<Action> {
    let total: u64 = visits.iter().map(|&v| v as u64).sum();
    if total == 0 {
        return None;
    }
    let mut x = ((Rng::new(seed).next_u64() as u128 * total as u128) >> 64) as u64;
    for (a, &v) in legal.iter().zip(visits) {
        if x < v as u64 {
            return Some(*a);
        }
        x -= v as u64;
    }
    unreachable!("x is below the visit total")
}
```

Add to `impl<E: Evaluator> IsmctsBot<E>` (after `config`):

```rust
    /// `act` with this decision's search settings. A decision that does not search (a trade
    /// answer, a single searchable move) ignores `opts`. After a sampled move,
    /// `take_last_search().action` is the move played.
    pub fn act_with(&mut self, obs: &Observation, legal: &[Action], opts: &MoveOptions) -> Action {
        self.last = None;
        let counted_in_observe = std::mem::take(&mut self.counted_in_observe);
        match obs.phase {
            Phase::TradeResponse if legal.contains(&Action::RejectTrade) => {
                return Action::RejectTrade
            }
            Phase::TradeConfirm if legal.contains(&Action::CancelTrade) => {
                return Action::CancelTrade
            }
            _ => {}
        }
        search_actions(legal, &mut self.buf);
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
                self.searches += 1;
                if let Some(a) = opts.sample.and_then(|seed| sample_by_visits(legal, &r.visits, seed)) {
                    r.action = a;
                }
                let a = r.action;
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

Replace the whole `fn act` in `impl<E: Evaluator + Send> Bot for IsmctsBot<E>` with:

```rust
    fn act(&mut self, obs: &Observation, legal: &[Action]) -> Action {
        let opts = MoveOptions {
            simulations: self.cfg.simulations,
            noise: None,
            sample: None,
        };
        self.act_with(obs, legal, &opts)
    }
```

- [ ] **Step 5: Rewrite self-play**

Replace `bots/src/selfplay.rs` with:

```rust
//! Self-play (spec Section 4): IsmctsBot in all four seats, every searched decision recorded.
//! 3a searched every move in full. 3b adds playout cap randomization (full or fast searches),
//! Dirichlet root noise on full searches and proportional move sampling in the opening, for the
//! heuristic evaluator (the warm start) and the network alike.

use crate::arena::{belief_resets, bot_seed, split_seeds, EventFeed, MAX_SEEDS};
use crate::heuristic::{HeuristicEvaluator, DEFAULT_WEIGHTS};
use crate::ismcts::{IsmctsBot, MoveOptions, MAX_SIMULATIONS};
use crate::Bot;
use settler_engine::legal::legal_actions;
use settler_engine::rng::{mix, Rng};
use settler_engine::{Action, GameConfig, PlayerId, State, NUM_PLAYERS};
use settler_search::{unit, Evaluator, RootNoise, SearchConfig};
use std::ops::Range;

/// Fewest simulations self-play accepts: the first simulation only evaluates the root, so with
/// one simulation every recorded visit count would be zero.
pub const MIN_SIMULATIONS: u32 = 2;
pub const FULL_SIMULATIONS: u32 = 600;
pub const FAST_SIMULATIONS: u32 = 100;
pub const FULL_PROBABILITY: f64 = 0.25;
pub const NOISE: RootNoise = RootNoise {
    weight: 0.25,
    alpha_total: 10.0,
};
pub const SAMPLE_TURNS: u32 = 8;
const EXPLORE_SALT: u64 = 0xE7B1;

#[derive(Clone, Debug, PartialEq)]
pub struct Decision {
    /// Index into the game's `actions` of the move this search chose.
    pub index: u32,
    pub actor: PlayerId,
    /// Every legal action (domestic offers included), in engine order.
    pub legal: Vec<Action>,
    /// Root visits aligned with `legal`.
    pub visits: Vec<u32>,
    /// A full search (a policy target); false for a fast search (a value target only).
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
    /// Belief resets over all four bots at the game's end; must be 0 (spec Section 2).
    pub belief_resets: u64,
}

/// How self-play searches and picks its moves.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Exploration {
    pub full_simulations: u32,
    pub fast_simulations: u32,
    /// Chance that a move gets a full search; the rest get fast searches.
    pub full_probability: f64,
    /// Root noise, on full searches only.
    pub noise: Option<RootNoise>,
    /// Moves made while the engine's turn counter is below this (setup is turn 0) are drawn in
    /// proportion to root visits; later moves are the most-visited.
    pub sample_turns: u32,
}

impl Exploration {
    /// 3a: every move a full search of `simulations`, no noise, always the most-visited move.
    pub fn fixed(simulations: u32) -> Exploration {
        Exploration {
            full_simulations: simulations,
            fast_simulations: simulations,
            full_probability: 1.0,
            noise: None,
            sample_turns: 0,
        }
    }

    /// 3b: 600 / 100 simulations with p = 0.25, noise 0.25 · Dir(10 / legal), 8 sampled turns.
    pub fn standard() -> Exploration {
        Exploration {
            full_simulations: FULL_SIMULATIONS,
            fast_simulations: FAST_SIMULATIONS,
            full_probability: FULL_PROBABILITY,
            noise: Some(NOISE),
            sample_turns: SAMPLE_TURNS,
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        for n in [self.full_simulations, self.fast_simulations] {
            if n < MIN_SIMULATIONS {
                return Err(format!(
                    "self-play needs at least {MIN_SIMULATIONS} simulations per search, got {n}"
                ));
            }
            if n > MAX_SIMULATIONS {
                return Err(format!(
                    "self-play allows at most {MAX_SIMULATIONS} simulations per search, got {n}"
                ));
            }
        }
        if !(0.0..=1.0).contains(&self.full_probability) {
            return Err(format!(
                "the full-search probability must be in [0, 1], got {}",
                self.full_probability
            ));
        }
        Ok(())
    }
}

/// One self-play game; `make_eval` builds each seat's evaluator, `batch` is the leaf batch.
pub fn play_with<E, F>(seed: u64, config: GameConfig, ex: &Exploration, batch: usize, make_eval: &F) -> SelfPlayGame
where
    E: Evaluator + Send,
    F: Fn() -> E,
{
    let cfg = SearchConfig {
        simulations: ex.full_simulations,
        batch,
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
        assert!(
            buf.contains(&a),
            "self-play chose illegal action {a:?} in phase {:?}",
            s.phase
        );
        if let Some(r) = bots[actor].take_last_search() {
            decisions.push(Decision {
                index: actions.len() as u32,
                actor: actor as PlayerId,
                legal: buf.clone(),
                visits: r.visits,
                full_search: full,
            });
        }
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
    config.validate()?;
    ex.validate()?;
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
                    part.map(|seed| play_with(seed, config, ex, batch, make_eval))
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

/// 3a self-play: the heuristic bot, every move a full search of `simulations`.
pub fn play(seed: u64, config: GameConfig, simulations: u32, rollout: u32) -> SelfPlayGame {
    let ex = Exploration::fixed(simulations);
    play_with(seed, config, &ex, SearchConfig::default().batch, &|| {
        HeuristicEvaluator::new(DEFAULT_WEIGHTS, rollout)
    })
}

/// 3a self-play over `seeds` (see `run_with`).
pub fn run(
    seeds: Range<u64>,
    config: GameConfig,
    simulations: u32,
    rollout: u32,
    threads: usize,
) -> Result<Vec<SelfPlayGame>, String> {
    let ex = Exploration::fixed(simulations);
    run_with(seeds, config, &ex, SearchConfig::default().batch, &|| {
        HeuristicEvaluator::new(DEFAULT_WEIGHTS, rollout)
    }, threads)
}
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `~/.cargo/bin/cargo test --release -p settler-search -p settler-bots`
Expected: all pass: `noise` 4 passed; `selfplay` passes 3a's 5 tests unchanged plus the 5 new ones; `ismcts` and `netbot` gain one each.

- [ ] **Step 7: Format, lint, full checks**

Run: `~/.cargo/bin/cargo fmt --all && ~/.cargo/bin/cargo fmt --check`
Expected: no output from the check.
Run: `~/.cargo/bin/cargo clippy --workspace --release --all-targets -- -D warnings`
Expected: exits 0.
Run: `~/.cargo/bin/cargo test --workspace --release && PATH=$HOME/.cargo/bin:$PATH .venv/bin/maturin develop --release -q && .venv/bin/pytest -q`
Expected: all pass (the Python self-play tests still see 3a records).

- [ ] **Step 8: Commit**

```bash
git add search/ bots/
git diff --cached | grep -nE 'sk-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{36}|password=[^[:space:]]+'
git -c user.email=sean.may101@gmail.com commit -m "search, bots: playout cap randomization, Dirichlet root noise and opening sampling in self-play"
```

---

### Task 5: Training batches, bindings and `selfplay --net`

**Files:**
- Create: `net/src/data.rs`, `alphasettler/files.py`, `alphasettler/selfplay.py`
- Modify: `net/src/lib.rs`, `bindings/src/train.rs`, `bindings/src/lib.rs`, `alphasettler/records.py`, `alphasettler/cli.py`
- Test: `net/tests/data.rs`, `tests/python/test_training_data.py`

**Interfaces:**
- Consumes: `encode_state`, `rel`, `abs`, `PUBLIC_LEN`, `HIDDEN_LEN` (Task 1); `Weights`, `Forward`, `Outputs` (Task 2); `load_net`, `net_path`, `NET_BATCH`, `DEFAULT_NETS_DIR`, `NetEvaluator` (Task 3); `run_with`, `Exploration`, `SelfPlayGame`, the self-play constants (Task 4).
- Produces:
  - `settler_net::data`: `VP_TARGET_DEN: f32` (10.0); `RecordedDecision { index: u32, actor: PlayerId, visits: Vec<u32>, full_search: bool }`; `RecordedGame { seed: u64, config: GameConfig, actions: Vec<Action>, decisions: Vec<RecordedDecision>, winner: Option<PlayerId>, vp: [u8; 4] }`; `Batch { n, public, hidden, mask: Vec<u8>, policy, full: Vec<u8>, outcome, vp }` (`Default`, `PartialEq`); `TrainingSet::new(Vec<RecordedGame>) -> Result<TrainingSet, String>`, `games()`, `decisions()`, `full_decisions()`, `game_decisions(game) -> usize`, `locate(k) -> (usize, usize)`, `position(game, decision, &mut Batch)`, `batch(picks: &[(usize, usize)], threads) -> Batch`, `sample(n, seed, threads) -> Result<Batch, String>`.
  - Python `_engine`: `encode(game) -> (bytes, bytes)` (public, hidden for the player to act); `TrainingSet(games: list[dict])` with `.games`, `.decisions`, `.full_decisions`, `.sample(n, seed, threads) -> dict`, `.position(game, decision) -> dict` (dict keys `n`, `public`, `hidden`, `mask`, `policy`, `full`, `outcome`, `vp`; f32 fields little-endian, `mask` and `full` one byte each); `net_forward(path, public: bytes, hidden: bytes) -> dict` (keys `policy`, `value`, `aux`, f32 little-endian); `resolve_net(name, nets_dir=None) -> str`; `selfplay_explore(seed_start, games, threads, full_simulations, fast_simulations, full_probability, config=None, net=None, nets_dir=None) -> list[dict]` (decisions without observations); constants `NET_PUBLIC_LEN`, `NET_HIDDEN_LEN`, `NET_ENCODING_HASH`, `NET_FORMAT_VERSION`, `NET_BATCH`, `SEARCH_BATCH`, `SELFPLAY_FULL_SIMULATIONS`, `SELFPLAY_FAST_SIMULATIONS`, `SELFPLAY_FULL_PROBABILITY`, `NOISE_WEIGHT`, `NOISE_ALPHA`, `SAMPLE_TURNS`.
  - Python `alphasettler.files`: `sha256_file(path) -> str`, `temp_path(path) -> Path` (`.NAME.tmp-PID` beside it), `remove_temps(directory, name_glob)`, `atomic_write_bytes(path, data)`, `atomic_write_text(path, text)`, `atomic_copy(src, dst)`.
  - Python `alphasettler.selfplay`: `BATCH_GAMES` (100), `settings(net, nets_dir, full_simulations, fast_simulations, full_probability) -> dict`, `write_explore(out, seed_start, games, *, config, full_simulations, fast_simulations, full_probability, net=None, nets_dir=None, threads=None, counts=None) -> dict` (counts `games`, `decisions`, `full_decisions`, `belief_resets`).
  - `alphasettler.records.open_writer` writes byte-reproducible files; `replay_mismatches` checks observations only where a decision has one.
  - CLI: `alphasettler selfplay --games N (--simulations S | --pcr | --net NAME) [--nets-dir D] [--full-simulations 600] [--fast-simulations 100] [--full-probability 0.25] ...`; 3b output is `<out-dir>/<time>-selfplay-<ismcts|net-NAME>-f<full>-q<fast>.jsonl.gz`.

Records written by `--net` or `--pcr` keep 3a's per-game fields (`config`, `seed`, `actions`, `winner`, `vp`, `turns`, `belief_resets`, and per decision `index`, `actor`, `legal`, `visits`, `full_search`) and add a `selfplay` block; they drop the per-decision `observation`, which the seed and actions reproduce and which the sampler never reads (it would multiply the window's size and parse time).

- [ ] **Step 1: Write the failing Rust tests**

Add `pub mod data;` to `net/src/lib.rs` (the module list becomes `data`, `encode`, `evaluator`, `forward`, `weights`).

Create `net/tests/data.rs`:

```rust
mod common;

use common::*;
use settler_engine::legal::legal_actions;
use settler_engine::rng::Rng;
use settler_engine::*;
use settler_net::data::{Batch, RecordedDecision, RecordedGame, TrainingSet};
use settler_net::encode::{encode_state, HIDDEN_LEN, PUBLIC_LEN};

/// A finished random game recorded with a decision at every 5th move: random visit counts (some
/// all zero), alternating full and fast searches.
fn recorded(seed: u64, config: GameConfig) -> RecordedGame {
    let mut s = State::new(seed, config);
    let mut rng = Rng::new(seed ^ 0xDA7A);
    let (mut actions, mut decisions, mut buf) = (Vec::new(), Vec::new(), Vec::new());
    while !s.is_over() {
        legal_actions(&s, &mut buf);
        if actions.len() % 5 == 0 {
            decisions.push(RecordedDecision {
                index: actions.len() as u32,
                actor: s.current_actor(),
                visits: buf.iter().map(|_| rng.below(4)).collect(),
                full_search: decisions.len() % 2 == 0,
            });
        }
        let a = buf[rng.below(buf.len() as u32) as usize];
        actions.push(a);
        s.apply(a);
    }
    RecordedGame {
        seed,
        config,
        actions,
        decisions,
        winner: s.winner(),
        vp: std::array::from_fn(|p| s.total_vp(p)),
    }
}

fn decided() -> GameConfig {
    GameConfig {
        vp_to_win: 3,
        ..no_trades()
    }
}

/// Trades on (offers among the legal moves) and a turn cap that ends in a draw.
fn drawn() -> GameConfig {
    GameConfig {
        max_turns: 30,
        ..GameConfig::default()
    }
}

#[test]
fn a_position_equals_the_encoder_on_the_replayed_state() {
    let games = vec![recorded(1, decided()), recorded(2, drawn())];
    assert!(games[0].winner.is_some() && games[1].winner.is_none());
    let set = TrainingSet::new(games.clone()).unwrap();
    assert_eq!(set.games(), 2);
    assert_eq!(set.decisions(), games.iter().map(|g| g.decisions.len()).sum::<usize>());
    let mut offers_masked = 0;
    for (gi, g) in games.iter().enumerate() {
        let mut s = State::new(g.seed, g.config);
        let mut next = 0;
        for (i, &a) in g.actions.iter().enumerate() {
            if next < g.decisions.len() && g.decisions[next].index as usize == i {
                let d = &g.decisions[next];
                let mut b = Batch::default();
                set.position(gi, next, &mut b);
                let (mut public, mut hidden) = (vec![0.0; PUBLIC_LEN], vec![0.0; HIDDEN_LEN]);
                encode_state(&s, d.actor, &mut public, &mut hidden);
                assert_eq!((b.n, &b.public, &b.hidden), (1, &public, &hidden));
                let legal = s.legal_actions();
                offers_masked += legal.iter().any(|a| matches!(a, Action::OfferTrade { .. })) as u32;
                let searchable: Vec<(Action, u32)> = legal
                    .iter()
                    .copied()
                    .zip(d.visits.iter().copied())
                    .filter(|(a, _)| !matches!(a, Action::OfferTrade { .. }))
                    .collect();
                let total: u32 = searchable.iter().map(|x| x.1).sum();
                let (mut mask, mut policy) = (vec![0u8; ACTION_SPACE_SIZE], vec![0.0f32; ACTION_SPACE_SIZE]);
                for &(a, v) in &searchable {
                    mask[a.encode() as usize] = 1;
                    policy[a.encode() as usize] = if total > 0 {
                        v as f32 / total as f32
                    } else {
                        1.0 / searchable.len() as f32
                    };
                }
                assert_eq!(b.mask, mask);
                assert_eq!(b.policy, policy);
                assert_eq!(b.full, vec![(d.full_search && total > 0) as u8]);
                let me = d.actor as usize;
                let outcome: Vec<f32> = (0..4)
                    .map(|r| match g.winner {
                        Some(w) => (w as usize == (me + r) % 4) as u8 as f32,
                        None => 0.25,
                    })
                    .collect();
                assert_eq!(b.outcome, outcome);
                let vp: Vec<f32> = (0..4).map(|r| g.vp[(me + r) % 4] as f32 / 10.0).collect();
                assert_eq!(b.vp, vp);
                next += 1;
            }
            s.apply(a);
        }
    }
    assert!(offers_masked > 0, "offers were legal somewhere and stayed out of the mask");
}

#[test]
fn sampling_is_seeded_and_thread_independent() {
    let set = TrainingSet::new(vec![recorded(1, decided()), recorded(3, drawn())]).unwrap();
    let a = set.sample(64, 9, 1).unwrap();
    assert_eq!(a, set.sample(64, 9, 4).unwrap());
    assert_ne!(a, set.sample(64, 10, 1).unwrap());
    assert_eq!(a.n, 64);
    assert_eq!((a.public.len(), a.hidden.len()), (64 * PUBLIC_LEN, 64 * HIDDEN_LEN));
    assert_eq!((a.mask.len(), a.policy.len()), (64 * ACTION_SPACE_SIZE, 64 * ACTION_SPACE_SIZE));
    assert_eq!((a.full.len(), a.outcome.len(), a.vp.len()), (64, 256, 256));
    let picks = [(0, 0), (1, 2), (0, 1)];
    let mut one_by_one = Batch::default();
    for &(g, d) in &picks {
        set.position(g, d, &mut one_by_one);
    }
    assert_eq!(set.batch(&picks, 2), one_by_one);
}

#[test]
fn games_without_decisions_are_never_sampled_and_an_empty_set_is_an_error() {
    let mut empty = recorded(4, decided());
    empty.decisions.clear();
    let set = TrainingSet::new(vec![empty.clone(), recorded(1, decided()), empty.clone()]).unwrap();
    assert_eq!(set.decisions(), set.game_decisions(1));
    assert_eq!(set.locate(0), (1, 0));
    let mut known = Vec::new();
    for d in 0..set.game_decisions(1) {
        let mut b = Batch::default();
        set.position(1, d, &mut b);
        known.push(b.public);
    }
    let b = set.sample(200, 1, 2).unwrap();
    for row in b.public.chunks(PUBLIC_LEN) {
        assert!(known.iter().any(|k| k[..] == *row), "a sampled row came from game 1");
    }
    let e = TrainingSet::new(vec![empty]).unwrap().sample(1, 0, 1).unwrap_err();
    assert!(e.contains("no positions"), "{e}");
    assert!(TrainingSet::new(vec![]).unwrap().sample(1, 0, 1).is_err());
}

fn err(g: RecordedGame) -> String {
    TrainingSet::new(vec![g]).err().unwrap()
}

#[test]
fn a_record_that_does_not_replay_is_an_error() {
    let g = recorded(1, decided());
    let mut actor = g.clone();
    actor.decisions[1].actor = (actor.decisions[1].actor + 1) % 4;
    assert!(err(actor).contains("actor"));
    let mut visits = g.clone();
    visits.decisions[0].visits.push(1);
    assert!(err(visits).contains("visit counts"));
    let mut order = g.clone();
    order.decisions.swap(0, 1);
    assert!(err(order).contains("out of order"));
    let mut winner = g.clone();
    winner.winner = None;
    assert!(err(winner).contains("winner"));
    let mut vp = g.clone();
    vp.vp[0] += 1;
    assert!(err(vp).contains("vp"));
    let mut illegal = g.clone();
    illegal.actions[0] = Action::EndTurn;
    assert!(err(illegal).contains("illegal"));
    let mut short = g.clone();
    short.actions.pop();
    short.decisions.retain(|d| (d.index as usize) < short.actions.len());
    assert!(err(short).contains("does not finish"));
    let mut long = g.clone();
    long.actions.push(Action::EndTurn);
    assert!(err(long).contains("after the game ended"));
    let e = err(RecordedGame { seed: 9, ..g });
    assert!(e.starts_with("game 0 (seed 9)"), "{e}");
}
```

- [ ] **Step 2: Run them to verify they fail**

Run: `~/.cargo/bin/cargo test --release -p settler-net --test data`
Expected: compile error, `file not found for module data`.

- [ ] **Step 3: Write the sampler**

Create `net/src/data.rs`:

```rust
//! Training data (spec Section 4): recorded games replayed to their decisions and encoded.
//! Nothing is held per position: a batch replays each sampled game from its seed (about 60 ns per
//! move) on a thread pool, so the window is limited only by disk.

use crate::encode::{abs, encode_state, rel, HIDDEN_LEN, PUBLIC_LEN};
use settler_engine::legal::legal_actions;
use settler_engine::rng::Rng;
use settler_engine::{Action, GameConfig, PlayerId, State, ACTION_SPACE_SIZE, NUM_PLAYERS};

/// Final VP targets are VP / this.
pub const VP_TARGET_DEN: f32 = 10.0;

#[derive(Clone, Debug, PartialEq)]
pub struct RecordedDecision {
    /// Index into the game's actions of the move this search chose.
    pub index: u32,
    pub actor: PlayerId,
    /// Root visits aligned with the legal moves at `index`, in engine order (offers included).
    pub visits: Vec<u32>,
    pub full_search: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RecordedGame {
    pub seed: u64,
    pub config: GameConfig,
    pub actions: Vec<Action>,
    pub decisions: Vec<RecordedDecision>,
    pub winner: Option<PlayerId>,
    pub vp: [u8; NUM_PLAYERS],
}

/// Inputs and targets for `n` positions, row-major.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Batch {
    pub n: usize,
    /// n × PUBLIC_LEN.
    pub public: Vec<f32>,
    /// n × HIDDEN_LEN.
    pub hidden: Vec<f32>,
    /// n × ACTION_SPACE_SIZE: 1 for each legal move except domestic offers.
    pub mask: Vec<u8>,
    /// n × ACTION_SPACE_SIZE: root visits normalised over the mask (uniform if none).
    pub policy: Vec<f32>,
    /// n: 1 for a full search with at least one visit (a policy target).
    pub full: Vec<u8>,
    /// n × 4, seat-relative: one-hot winner, ¼ each for a turn-cap draw.
    pub outcome: Vec<f32>,
    /// n × 4, seat-relative: final VP / 10.
    pub vp: Vec<f32>,
}

impl Batch {
    fn push(&mut self, s: &State, game: &RecordedGame, d: &RecordedDecision, legal: &[Action]) {
        let actor = d.actor;
        let (p0, h0) = (self.public.len(), self.hidden.len());
        self.public.resize(p0 + PUBLIC_LEN, 0.0);
        self.hidden.resize(h0 + HIDDEN_LEN, 0.0);
        encode_state(s, actor, &mut self.public[p0..], &mut self.hidden[h0..]);
        let m0 = self.mask.len();
        self.mask.resize(m0 + ACTION_SPACE_SIZE, 0);
        self.policy.resize(m0 + ACTION_SPACE_SIZE, 0.0);
        let searchable = || {
            legal
                .iter()
                .zip(&d.visits)
                .filter(|(a, _)| !matches!(a, Action::OfferTrade { .. }))
        };
        let total: u32 = searchable().map(|(_, &v)| v).sum();
        let count = searchable().count();
        for (a, &v) in searchable() {
            let k = m0 + a.encode() as usize;
            self.mask[k] = 1;
            self.policy[k] = if total > 0 {
                v as f32 / total as f32
            } else {
                1.0 / count as f32
            };
        }
        self.full.push((d.full_search && total > 0) as u8);
        let outcome: [f32; NUM_PLAYERS] = match game.winner {
            Some(w) => std::array::from_fn(|r| (r == rel(w, actor)) as u8 as f32),
            None => [1.0 / NUM_PLAYERS as f32; NUM_PLAYERS],
        };
        self.outcome.extend_from_slice(&outcome);
        for r in 0..NUM_PLAYERS {
            self.vp.push(game.vp[abs(r, actor)] as f32 / VP_TARGET_DEN);
        }
        self.n += 1;
    }

    fn append(&mut self, other: Batch) {
        self.n += other.n;
        self.public.extend(other.public);
        self.hidden.extend(other.hidden);
        self.mask.extend(other.mask);
        self.policy.extend(other.policy);
        self.full.extend(other.full);
        self.outcome.extend(other.outcome);
        self.vp.extend(other.vp);
    }
}

/// Replay `g` once, checking it against its record.
fn check_game(g: &RecordedGame, buf: &mut Vec<Action>) -> Result<(), String> {
    g.config.validate()?;
    let mut s = State::new(g.seed, g.config);
    let mut next = g.decisions.iter().peekable();
    for (i, &a) in g.actions.iter().enumerate() {
        if s.is_over() {
            return Err(format!("move {i} comes after the game ended"));
        }
        legal_actions(&s, buf);
        if let Some(d) = next.next_if(|d| d.index as usize == i) {
            if d.actor != s.current_actor() {
                return Err(format!(
                    "move {i}: the record says actor {}, the replay has {}",
                    d.actor,
                    s.current_actor()
                ));
            }
            if d.visits.len() != buf.len() {
                return Err(format!(
                    "move {i}: {} visit counts for {} legal moves",
                    d.visits.len(),
                    buf.len()
                ));
            }
        }
        if !buf.contains(&a) {
            return Err(format!("move {i}: {a:?} is illegal"));
        }
        s.apply(a);
    }
    if let Some(d) = next.next() {
        return Err(format!(
            "the decision at move {} is out of order or past the last move",
            d.index
        ));
    }
    if !s.is_over() {
        return Err("the game does not finish".into());
    }
    if s.winner() != g.winner {
        return Err(format!(
            "the replay's winner is {:?}, the record says {:?}",
            s.winner(),
            g.winner
        ));
    }
    let vp: [u8; NUM_PLAYERS] = std::array::from_fn(|p| s.total_vp(p));
    if vp != g.vp {
        return Err(format!(
            "the replay's final vp is {vp:?}, the record says {:?}",
            g.vp
        ));
    }
    Ok(())
}

/// Recorded games, checked once, from which training batches are replayed.
pub struct TrainingSet {
    games: Vec<RecordedGame>,
    /// `starts[g]` is the global index of game g's first decision; the last entry is the total.
    starts: Vec<usize>,
    full: usize,
}

impl TrainingSet {
    /// Every game is replayed and must match its record exactly.
    pub fn new(games: Vec<RecordedGame>) -> Result<TrainingSet, String> {
        let mut starts = Vec::with_capacity(games.len() + 1);
        let (mut total, mut full) = (0, 0);
        let mut buf = Vec::new();
        for (i, g) in games.iter().enumerate() {
            check_game(g, &mut buf).map_err(|e| format!("game {i} (seed {}): {e}", g.seed))?;
            starts.push(total);
            total += g.decisions.len();
            full += g.decisions.iter().filter(|d| d.full_search).count();
        }
        starts.push(total);
        Ok(TrainingSet { games, starts, full })
    }

    pub fn games(&self) -> usize {
        self.games.len()
    }

    pub fn decisions(&self) -> usize {
        *self.starts.last().expect("starts always ends with the total")
    }

    pub fn full_decisions(&self) -> usize {
        self.full
    }

    pub fn game_decisions(&self, game: usize) -> usize {
        self.games[game].decisions.len()
    }

    /// (game, decision) of global decision `k` (< `decisions()`).
    pub fn locate(&self, k: usize) -> (usize, usize) {
        let g = self.starts.partition_point(|&s| s <= k) - 1;
        (g, k - self.starts[g])
    }

    /// Append decision `decision` of game `game` to `out`.
    pub fn position(&self, game: usize, decision: usize, out: &mut Batch) {
        let g = &self.games[game];
        let d = &g.decisions[decision];
        let mut s = State::new(g.seed, g.config);
        for &a in &g.actions[..d.index as usize] {
            s.apply(a);
        }
        let mut legal = Vec::new();
        legal_actions(&s, &mut legal);
        out.push(&s, g, d, &legal);
    }

    /// `picks` in order, replayed on up to `threads` threads; independent of `threads`.
    pub fn batch(&self, picks: &[(usize, usize)], threads: usize) -> Batch {
        let threads = threads.clamp(1, picks.len().max(1));
        let chunk = picks.len().div_ceil(threads).max(1);
        let parts: Vec<Batch> = std::thread::scope(|scope| {
            let workers: Vec<_> = picks
                .chunks(chunk)
                .map(|part| {
                    scope.spawn(move || {
                        let mut b = Batch::default();
                        for &(g, d) in part {
                            self.position(g, d, &mut b);
                        }
                        b
                    })
                })
                .collect();
            workers
                .into_iter()
                .map(|w| {
                    w.join()
                        .unwrap_or_else(|panic| std::panic::resume_unwind(panic))
                })
                .collect()
        });
        let mut out = Batch::default();
        for p in parts {
            out.append(p);
        }
        out
    }

    /// `n` decisions drawn uniformly (with replacement) from `Rng::new(seed)`, then replayed.
    pub fn sample(&self, n: usize, seed: u64, threads: usize) -> Result<Batch, String> {
        let total = self.decisions();
        if total == 0 {
            return Err("no positions to sample: the records hold no decisions".into());
        }
        let mut rng = Rng::new(seed);
        let picks: Vec<(usize, usize)> = (0..n)
            .map(|_| self.locate(((rng.next_u64() as u128 * total as u128) >> 64) as usize))
            .collect();
        Ok(self.batch(&picks, threads))
    }
}
```

- [ ] **Step 4: Run the Rust tests to verify they pass**

Run: `~/.cargo/bin/cargo test --release -p settler-net --test data`
Expected: `test result: ok. 4 passed`.

- [ ] **Step 5: Write the failing Python tests**

Create `tests/python/test_training_data.py`:

```python
import hashlib
import json
import subprocess
import sys
from array import array

import pytest

from alphasettler import Game, records
from alphasettler._engine import (
    NET_BATCH, NOISE_ALPHA, NOISE_WEIGHT, SAMPLE_TURNS, SEARCH_BATCH, TrainingSet, action_space_size, encode,
    net_forward, resolve_net, write_random_net,
)
from alphasettler.selfplay import write_explore

CONFIG = {"max_offers_per_turn": 0}
A = action_space_size()


def floats(b: bytes) -> list[float]:
    return array("f", b).tolist()


def explore_games(tmp_path, games, config=CONFIG, name="x.jsonl.gz", **kw):
    out = tmp_path / name
    counts = write_explore(out, 0, games, config=config, full_simulations=6, fast_simulations=3,
                           full_probability=0.25, threads=2, **kw)
    back = list(records.read(out))
    assert counts["games"] == games == len(back)
    assert counts["decisions"] == sum(len(g["decisions"]) for g in back)
    return back


def test_explore_records_carry_their_settings_and_no_observations(tmp_path):
    for g in explore_games(tmp_path, 3):
        assert g["selfplay"] == {
            "bot": "ismcts", "net": None, "net_sha256": None, "full_simulations": 6, "fast_simulations": 3,
            "full_probability": 0.25, "noise_weight": NOISE_WEIGHT, "noise_alpha": NOISE_ALPHA,
            "sample_turns": SAMPLE_TURNS, "batch": SEARCH_BATCH,
        }
        assert (NOISE_WEIGHT, NOISE_ALPHA, SAMPLE_TURNS) == (0.25, 10.0, 8)
        assert all("observation" not in d for d in g["decisions"])
        assert {d["full_search"] for d in g["decisions"]} == {True, False}
        for d in g["decisions"]:
            assert sum(d["visits"]) == (5 if d["full_search"] else 2)
        assert g["belief_resets"] == 0
        assert records.replay_mismatches(g) == []


def test_net_self_play_records_the_net_and_its_hash(tmp_path):
    nets = tmp_path / "nets"
    nets.mkdir()
    write_random_net(str(nets / "tiny.bin"), 1, 16, 8)
    (g,) = explore_games(tmp_path, 1, net="tiny", nets_dir=nets)
    assert g["selfplay"]["bot"] == "net:tiny" and g["selfplay"]["net"] == "tiny"
    assert g["selfplay"]["net_sha256"] == hashlib.sha256((nets / "tiny.bin").read_bytes()).hexdigest()
    assert g["selfplay"]["batch"] == NET_BATCH
    with pytest.raises(ValueError, match="bad net name"):
        write_explore(tmp_path / "no.jsonl.gz", 0, 1, config=CONFIG, full_simulations=6, fast_simulations=3,
                      full_probability=0.25, net="../tiny", nets_dir=nets)
    assert not (tmp_path / "no.jsonl.gz").exists()


def test_record_files_are_byte_identical_for_the_same_games(tmp_path):
    explore_games(tmp_path, 2, name="a.jsonl.gz")
    explore_games(tmp_path, 2, name="b.jsonl.gz")
    assert (tmp_path / "a.jsonl.gz").read_bytes() == (tmp_path / "b.jsonl.gz").read_bytes()


def test_a_batch_equals_the_encoder_on_the_replayed_state(tmp_path):
    games = explore_games(tmp_path, 2)
    ts = TrainingSet(games)
    assert ts.games == 2 and ts.decisions == sum(len(g["decisions"]) for g in games)
    assert 0 < ts.full_decisions < ts.decisions
    checked = 0
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
            assert [i for i in range(A) if mask[i]] == sorted(game.legal_actions())
            policy = floats(b["policy"])
            total = sum(d["visits"])
            for a, v in zip(d["legal"], d["visits"]):
                assert policy[a] == pytest.approx(v / total, rel=1e-6)
            assert list(b["full"]) == [int(d["full_search"])]
            checked += 1
    assert checked > 40
    batch = ts.sample(50, 3, 2)
    assert batch == ts.sample(50, 3, 1)
    assert (len(batch["public"]), len(batch["mask"]), len(batch["outcome"])) == (50 * len(public), 50 * A, 50 * 16)


def test_turn_cap_draws_are_a_quarter_each(tmp_path):
    games = explore_games(tmp_path, 2, config={"max_offers_per_turn": 0, "max_turns": 4})
    assert all(g["winner"] is None for g in games)
    b = TrainingSet(games).sample(20, 1, 1)
    assert floats(b["outcome"]) == [0.25] * 80


def test_a_record_that_does_not_replay_is_an_error(tmp_path):
    (g,) = explore_games(tmp_path, 1)
    bad = json.loads(json.dumps(g))
    bad["winner"] = None if g["winner"] is not None else 0
    with pytest.raises(ValueError, match="winner"):
        TrainingSet([bad])
    missing = dict(g)
    del missing["vp"]
    with pytest.raises(ValueError, match="missing"):
        TrainingSet([missing])
    bad_id = json.loads(json.dumps(g))
    bad_id["actions"][0] = 9999
    with pytest.raises(ValueError, match="9999"):
        TrainingSet([bad_id])
    with pytest.raises(ValueError, match="no positions"):
        TrainingSet([{**g, "decisions": []}]).sample(1, 0, 1)


def test_net_forward_runs_the_rust_forward_pass(tmp_path):
    path = tmp_path / "r.bin"
    write_random_net(str(path), 2, 16, 8)
    public, hidden = encode(Game(1, CONFIG))
    out = net_forward(str(path), public * 3, hidden * 3)
    assert (len(out["policy"]), len(out["value"]), len(out["aux"])) == (3 * A * 4, 3 * 16, 3 * 16)
    with pytest.raises(ValueError, match="for some n"):
        net_forward(str(path), public, hidden * 2)
    with pytest.raises(ValueError, match="cannot read"):
        net_forward(str(tmp_path / "none.bin"), public, hidden)


def test_resolve_net_checks_the_name_and_the_file(tmp_path):
    write_random_net(str(tmp_path / "tiny.bin"), 1, 16, 8)
    assert resolve_net("tiny", str(tmp_path)) == str(tmp_path / "tiny.bin")
    for bad in ["../tiny", "a/b", "", "x" * 65]:
        with pytest.raises(ValueError, match="bad net name"):
            resolve_net(bad, str(tmp_path))
    with pytest.raises(ValueError, match="cannot read"):
        resolve_net("missing", str(tmp_path))


def cli(*args):
    return subprocess.run([sys.executable, "-m", "alphasettler", *args], capture_output=True, text=True)


def test_cli_selfplay_with_a_net_or_pcr(tmp_path):
    nets = tmp_path / "nets"
    nets.mkdir()
    write_random_net(str(nets / "tiny.bin"), 1, 16, 8)
    r = cli("selfplay", "--net", "tiny", "--nets-dir", str(nets), "--games", "2", "--full-simulations", "4",
            "--fast-simulations", "2", "--out-dir", str(tmp_path / "out"))
    assert r.returncode == 0, r.stderr
    (path,) = (tmp_path / "out").glob("*-selfplay-net-tiny-f4-q2.jsonl.gz")
    assert "2 games" in r.stdout and "belief resets: 0" in r.stdout
    assert all(g["selfplay"]["net"] == "tiny" for g in records.read(path))
    r = cli("selfplay", "--pcr", "--games", "1", "--full-simulations", "4", "--fast-simulations", "2",
            "--out-dir", str(tmp_path / "out"))
    assert r.returncode == 0, r.stderr
    assert len(list((tmp_path / "out").glob("*-selfplay-ismcts-f4-q2.jsonl.gz"))) == 1
    for bad, message in [
        (["--net", "tiny", "--simulations", "10"], "--simulations"),
        (["--pcr", "--rollout", "4"], "--rollout"),
        (["--net", "../tiny"], "bad net name"),
        (["--net", "missing"], "cannot read"),
        (["--pcr", "--full-simulations", "1"], "full simulations"),
        (["--pcr", "--full-probability", "1.5"], "probability"),
        (["--full-simulations", "4"], "--simulations is required"),
        (["--simulations", "10", "--fast-simulations", "4"], "needs --net or --pcr"),
    ]:
        r = cli("selfplay", "--games", "1", "--nets-dir", str(nets), *bad, "--out-dir", str(tmp_path / "never"))
        assert r.returncode == 2 and message in r.stderr, (bad, r.stderr)
        assert not (tmp_path / "never").exists()
```

- [ ] **Step 6: Run them to verify they fail**

Run: `.venv/bin/pytest -q tests/python/test_training_data.py`
Expected: FAIL with `ImportError: cannot import name 'NET_BATCH' from 'alphasettler._engine'` (and `ModuleNotFoundError: alphasettler.selfplay`).

- [ ] **Step 7: Bindings**

Replace `bindings/src/train.rs` with:

```rust
//! Network and training helpers for Python (spec Section 4): encode a position, replay-and-sample
//! training batches returned as raw little-endian buffers, the Rust forward pass, net lookup and
//! test weights.

use crate::convert::*;
use crate::PyGame;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList};
use settler_bots::netbot::{load_net, net_path, DEFAULT_NETS_DIR};
use settler_engine::{Action, NUM_PLAYERS};
use settler_net::data::{Batch, RecordedDecision, RecordedGame, TrainingSet};
use settler_net::encode::{encode_state, ENCODING_HASH, HIDDEN_LEN, PUBLIC_LEN};
use settler_net::forward::{Forward, Outputs};
use settler_net::weights::{Arch, Weights, FORMAT_VERSION, MAX_WIDTH};
use std::path::PathBuf;

fn le_bytes(xs: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(4 * xs.len());
    for x in xs {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}

fn floats(bytes: &[u8], what: &str) -> PyResult<Vec<f32>> {
    if bytes.len() % 4 != 0 {
        return Err(value_error(format!(
            "{what} holds {} bytes, not a whole number of f32s",
            bytes.len()
        )));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect())
}

fn field<'py>(d: &Bound<'py, PyDict>, key: &str, what: &str) -> PyResult<Bound<'py, PyAny>> {
    d.get_item(key)?
        .ok_or_else(|| value_error(format!("{what} is missing {key:?}")))
}

fn recorded_game(g: &Bound<'_, PyDict>) -> PyResult<RecordedGame> {
    let config_obj = field(g, "config", "game record")?;
    let config = parse_config(Some(config_obj.cast::<PyDict>()?))?;
    let actions = field(g, "actions", "game record")?
        .extract::<Vec<u16>>()?
        .into_iter()
        .map(|id| Action::decode(id).ok_or_else(|| value_error(format!("no action has id {id}"))))
        .collect::<PyResult<Vec<_>>>()?;
    let list_obj = field(g, "decisions", "game record")?;
    let mut decisions = Vec::new();
    for d in list_obj.cast::<PyList>()?.iter() {
        let d = d.cast::<PyDict>()?;
        decisions.push(RecordedDecision {
            index: field(d, "index", "decision")?.extract()?,
            actor: viewer_arg(&field(d, "actor", "decision")?)?,
            visits: field(d, "visits", "decision")?.extract()?,
            full_search: field(d, "full_search", "decision")?.extract()?,
        });
    }
    let vp: Vec<u8> = field(g, "vp", "game record")?.extract()?;
    let vp: [u8; NUM_PLAYERS] = vp
        .try_into()
        .map_err(|_| value_error("a game record's vp needs 4 entries"))?;
    Ok(RecordedGame {
        seed: int_arg(&field(g, "seed", "game record")?, "seed")?,
        config,
        actions,
        decisions,
        winner: field(g, "winner", "game record")?.extract()?,
        vp,
    })
}

fn batch_dict<'py>(py: Python<'py>, b: &Batch) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("n", b.n)?;
    d.set_item("public", PyBytes::new(py, &le_bytes(&b.public)))?;
    d.set_item("hidden", PyBytes::new(py, &le_bytes(&b.hidden)))?;
    d.set_item("mask", PyBytes::new(py, &b.mask))?;
    d.set_item("policy", PyBytes::new(py, &le_bytes(&b.policy)))?;
    d.set_item("full", PyBytes::new(py, &b.full))?;
    d.set_item("outcome", PyBytes::new(py, &le_bytes(&b.outcome)))?;
    d.set_item("vp", PyBytes::new(py, &le_bytes(&b.vp)))?;
    Ok(d)
}

/// The public features and the hidden block for whoever must act in `game`.
#[pyfunction]
fn encode<'py>(
    py: Python<'py>,
    game: PyRef<'_, PyGame>,
) -> PyResult<(Bound<'py, PyBytes>, Bound<'py, PyBytes>)> {
    let s = game.inner.state();
    if s.is_over() {
        return Err(value_error("the game is over"));
    }
    let (mut public, mut hidden) = (vec![0.0; PUBLIC_LEN], vec![0.0; HIDDEN_LEN]);
    encode_state(s, s.current_actor(), &mut public, &mut hidden);
    Ok((
        PyBytes::new(py, &le_bytes(&public)),
        PyBytes::new(py, &le_bytes(&hidden)),
    ))
}

/// Recorded games (dicts with `seed`, `config`, `actions`, `decisions[]` with `index`, `actor`,
/// `visits`, `full_search`, `winner` and `vp`), each replayed once and checked.
#[pyclass(name = "TrainingSet", module = "alphasettler._engine", frozen)]
struct PyTrainingSet {
    inner: TrainingSet,
}

#[pymethods]
impl PyTrainingSet {
    #[new]
    fn new(py: Python<'_>, games: &Bound<'_, PyList>) -> PyResult<Self> {
        let mut parsed = Vec::with_capacity(games.len());
        for g in games.iter() {
            parsed.push(recorded_game(g.cast::<PyDict>()?)?);
        }
        let inner = py
            .detach(|| TrainingSet::new(parsed))
            .map_err(value_error)?;
        Ok(PyTrainingSet { inner })
    }

    #[getter]
    fn games(&self) -> usize {
        self.inner.games()
    }

    #[getter]
    fn decisions(&self) -> usize {
        self.inner.decisions()
    }

    #[getter]
    fn full_decisions(&self) -> usize {
        self.inner.full_decisions()
    }

    /// `n` positions drawn uniformly over every decision from `seed`, replayed on `threads`
    /// threads (the result does not depend on `threads`). Releases the GIL.
    fn sample<'py>(
        &self,
        py: Python<'py>,
        n: &Bound<'py, PyAny>,
        seed: &Bound<'py, PyAny>,
        threads: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let n: usize = int_arg(n, "n")?;
        let seed: u64 = int_arg(seed, "seed")?;
        let threads: usize = int_arg(threads, "threads")?;
        if n == 0 {
            return Err(value_error("n must be positive"));
        }
        let b = py
            .detach(|| self.inner.sample(n, seed, threads))
            .map_err(value_error)?;
        batch_dict(py, &b)
    }

    /// Decision `decision` of game `game` as a one-position batch.
    fn position<'py>(&self, py: Python<'py>, game: usize, decision: usize) -> PyResult<Bound<'py, PyDict>> {
        if game >= self.inner.games() || decision >= self.inner.game_decisions(game) {
            return Err(value_error(format!("no decision {decision} in game {game}")));
        }
        let mut b = Batch::default();
        self.inner.position(game, decision, &mut b);
        batch_dict(py, &b)
    }
}

/// The Rust forward pass of the weights file at `path` on `n` positions (public: n × NET_PUBLIC_LEN
/// f32, hidden: n × NET_HIDDEN_LEN f32, little-endian). Returns raw policy logits, value logits
/// and aux outputs.
#[pyfunction]
fn net_forward<'py>(
    py: Python<'py>,
    path: PathBuf,
    public: &Bound<'py, PyBytes>,
    hidden: &Bound<'py, PyBytes>,
) -> PyResult<Bound<'py, PyDict>> {
    let w = Weights::load(&path).map_err(value_error)?;
    let public = floats(public.as_bytes(), "public")?;
    let hidden = floats(hidden.as_bytes(), "hidden")?;
    let n = public.len() / PUBLIC_LEN;
    if n == 0 || public.len() != n * PUBLIC_LEN || hidden.len() != n * HIDDEN_LEN {
        return Err(value_error(format!(
            "public must hold n × {PUBLIC_LEN} floats and hidden n × {HIDDEN_LEN} for some n ≥ 1, got {} and {}",
            public.len(),
            hidden.len()
        )));
    }
    let mut out = Outputs::default();
    py.detach(|| Forward::new().run(&w, &public, &hidden, n, &mut out));
    let d = PyDict::new(py);
    d.set_item("policy", PyBytes::new(py, &le_bytes(&out.policy)))?;
    d.set_item("value", PyBytes::new(py, &le_bytes(&out.value)))?;
    d.set_item("aux", PyBytes::new(py, &le_bytes(&out.aux)))?;
    Ok(d)
}

/// The path of net `name` in `nets_dir` (default `runs/3b/nets`), after checking the name and
/// loading and validating the file.
#[pyfunction]
#[pyo3(signature = (name, nets_dir=None))]
fn resolve_net(name: &str, nets_dir: Option<PathBuf>) -> PyResult<String> {
    let dir = nets_dir.unwrap_or_else(|| PathBuf::from(DEFAULT_NETS_DIR));
    load_net(&dir, name).map_err(value_error)?;
    Ok(net_path(&dir, name).to_string_lossy().into_owned())
}

/// Random weights (the default shape unless `trunk` / `value` are given) saved to `path`, for
/// tests and measurements.
#[pyfunction]
#[pyo3(signature = (path, seed, trunk=None, value=None))]
fn write_random_net(
    path: PathBuf,
    seed: &Bound<'_, PyAny>,
    trunk: Option<&Bound<'_, PyAny>>,
    value: Option<&Bound<'_, PyAny>>,
) -> PyResult<()> {
    let seed: u64 = int_arg(seed, "seed")?;
    let width = |v: Option<&Bound<'_, PyAny>>, default: usize, what: &str| -> PyResult<usize> {
        let w: usize = match v {
            Some(v) => int_arg(v, what)?,
            None => default,
        };
        if !(1..=MAX_WIDTH).contains(&w) {
            return Err(value_error(format!("{what} must be 1..={MAX_WIDTH}, got {w}")));
        }
        Ok(w)
    };
    let arch = Arch {
        trunk: width(trunk, Arch::DEFAULT.trunk, "trunk")?,
        value: width(value, Arch::DEFAULT.value, "value")?,
    };
    Weights::random(arch, seed).save(&path).map_err(value_error)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTrainingSet>()?;
    m.add_function(wrap_pyfunction!(encode, m)?)?;
    m.add_function(wrap_pyfunction!(net_forward, m)?)?;
    m.add_function(wrap_pyfunction!(resolve_net, m)?)?;
    m.add_function(wrap_pyfunction!(write_random_net, m)?)?;
    m.add("NET_PUBLIC_LEN", PUBLIC_LEN)?;
    m.add("NET_HIDDEN_LEN", HIDDEN_LEN)?;
    m.add("NET_ENCODING_HASH", ENCODING_HASH)?;
    m.add("NET_FORMAT_VERSION", FORMAT_VERSION)?;
    Ok(())
}
```

In `bindings/src/lib.rs`, replace the whole `selfplay` function (doc comment included) with the two functions below and the shared `game_dict` helper:

```rust
/// One self-play game as a record dict. With `observations`, each decision also carries the
/// actor's observation (3a records); 3b records leave it out (the seed and actions reproduce it).
fn game_dict<'py>(
    py: Python<'py>,
    g: &settler_bots::selfplay::SelfPlayGame,
    config: &settler_engine::GameConfig,
    observations: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("seed", g.seed)?;
    d.set_item(
        "actions",
        g.actions.iter().map(|a| a.encode() as u32).collect::<Vec<_>>(),
    )?;
    d.set_item("winner", g.winner)?;
    d.set_item("vp", counts(&g.vp))?;
    d.set_item("turns", g.turns)?;
    d.set_item("belief_resets", g.belief_resets)?;
    let decisions = PyList::empty(py);
    let mut s = settler_engine::State::new(g.seed, *config);
    let mut applied = 0usize;
    for dec in &g.decisions {
        let e = PyDict::new(py);
        e.set_item("index", dec.index)?;
        e.set_item("actor", dec.actor)?;
        e.set_item(
            "legal",
            dec.legal.iter().map(|a| a.encode() as u32).collect::<Vec<_>>(),
        )?;
        e.set_item("visits", dec.visits.clone())?;
        e.set_item("full_search", dec.full_search)?;
        if observations {
            while applied < dec.index as usize {
                s.apply(g.actions[applied]);
                applied += 1;
            }
            e.set_item("observation", observation_dict(py, &s.observation(dec.actor))?)?;
        }
        decisions.append(e)?;
    }
    d.set_item("decisions", decisions)?;
    Ok(d)
}

/// IsmctsBot self-play: `games` games from `seed_start`, every searched decision recorded with
/// the actor's observation. `rollout` defaults to 0. Releases the GIL while games run.
#[pyfunction]
#[pyo3(signature = (seed_start, games, simulations, threads, config=None, rollout=None))]
fn selfplay<'py>(
    py: Python<'py>,
    seed_start: &Bound<'py, PyAny>,
    games: &Bound<'py, PyAny>,
    simulations: &Bound<'py, PyAny>,
    threads: &Bound<'py, PyAny>,
    config: Option<&Bound<'py, PyDict>>,
    rollout: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyList>> {
    use settler_bots::ismcts::{MAX_ROLLOUT, MAX_SIMULATIONS};
    use settler_bots::selfplay::MIN_SIMULATIONS;
    let seed_start: u64 = int_arg(seed_start, "seed_start")?;
    let games: u64 = int_arg(games, "games")?;
    let simulations: u32 = int_arg(simulations, "simulations")?;
    let threads: usize = int_arg(threads, "threads")?;
    let rollout: u32 = match rollout {
        Some(r) => int_arg(r, "rollout")?,
        None => 0,
    };
    if games == 0 {
        return Err(value_error("games must be positive"));
    }
    if !(MIN_SIMULATIONS..=MAX_SIMULATIONS).contains(&simulations) {
        return Err(value_error(format!(
            "simulations must be {MIN_SIMULATIONS}..={MAX_SIMULATIONS}, got {simulations}"
        )));
    }
    if rollout > MAX_ROLLOUT {
        return Err(value_error(format!(
            "rollout must be 0..={MAX_ROLLOUT}, got {rollout}"
        )));
    }
    let end = seed_start
        .checked_add(games)
        .ok_or_else(|| value_error("seed_start + games exceeds the u64 seed range"))?;
    let config = parse_config(config)?;
    let played = py
        .detach(|| {
            settler_bots::selfplay::run(seed_start..end, config, simulations, rollout, threads)
        })
        .map_err(value_error)?;
    let out = PyList::empty(py);
    for g in &played {
        out.append(game_dict(py, g, &config, true)?)?;
    }
    Ok(out)
}

/// 3b self-play (spec Section 4): playout cap randomization (`full_simulations` with
/// probability `full_probability`, else `fast_simulations`), root noise on full searches and
/// opening sampling. With `net`, every seat is the network `<nets_dir>/<net>.bin` (loaded once);
/// otherwise the heuristic. Decisions carry no observations. Releases the GIL while games run.
#[pyfunction]
#[pyo3(signature = (seed_start, games, threads, full_simulations, fast_simulations, full_probability, config=None, net=None, nets_dir=None))]
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
) -> PyResult<Bound<'py, PyList>> {
    use settler_bots::heuristic::{HeuristicEvaluator, DEFAULT_WEIGHTS};
    use settler_bots::netbot::{load_net, DEFAULT_NETS_DIR, NET_BATCH};
    use settler_bots::selfplay::{run_with, Exploration};
    use settler_net::evaluator::NetEvaluator;
    let seed_start: u64 = int_arg(seed_start, "seed_start")?;
    let games: u64 = int_arg(games, "games")?;
    let threads: usize = int_arg(threads, "threads")?;
    let ex = Exploration {
        full_simulations: int_arg(full_simulations, "full_simulations")?,
        fast_simulations: int_arg(fast_simulations, "fast_simulations")?,
        full_probability,
        ..Exploration::standard()
    };
    ex.validate().map_err(value_error)?;
    if games == 0 {
        return Err(value_error("games must be positive"));
    }
    let end = seed_start
        .checked_add(games)
        .ok_or_else(|| value_error("seed_start + games exceeds the u64 seed range"))?;
    let config = parse_config(config)?;
    let dir = nets_dir.unwrap_or_else(|| PathBuf::from(DEFAULT_NETS_DIR));
    let weights = match net {
        Some(n) => Some(load_net(&dir, n).map_err(value_error)?),
        None => None,
    };
    let played = py
        .detach(|| match &weights {
            None => run_with(
                seed_start..end,
                config,
                &ex,
                settler_search::SearchConfig::default().batch,
                &|| HeuristicEvaluator::new(DEFAULT_WEIGHTS, 0),
                threads,
            ),
            Some(w) => run_with(
                seed_start..end,
                config,
                &ex,
                NET_BATCH,
                &|| NetEvaluator::new(w.clone()),
                threads,
            ),
        })
        .map_err(value_error)?;
    let out = PyList::empty(py);
    for g in &played {
        out.append(game_dict(py, g, &config, false)?)?;
    }
    Ok(out)
}
```

In `_engine(...)`, after `m.add_function(wrap_pyfunction!(selfplay, m)?)?;` add:

```rust
    m.add_function(wrap_pyfunction!(selfplay_explore, m)?)?;
    m.add("SEARCH_BATCH", settler_search::SearchConfig::default().batch)?;
    m.add("NET_BATCH", settler_bots::netbot::NET_BATCH)?;
    m.add("SELFPLAY_FULL_SIMULATIONS", settler_bots::selfplay::FULL_SIMULATIONS)?;
    m.add("SELFPLAY_FAST_SIMULATIONS", settler_bots::selfplay::FAST_SIMULATIONS)?;
    m.add("SELFPLAY_FULL_PROBABILITY", settler_bots::selfplay::FULL_PROBABILITY)?;
    m.add("NOISE_WEIGHT", settler_bots::selfplay::NOISE.weight as f64)?;
    m.add("NOISE_ALPHA", settler_bots::selfplay::NOISE.alpha_total as f64)?;
    m.add("SAMPLE_TURNS", settler_bots::selfplay::SAMPLE_TURNS)?;
```

- [ ] **Step 8: Python: files, records, the 3b writer and the CLI**

Create `alphasettler/files.py`:

```python
"""Small file helpers: content hashes and atomic (temp-then-rename) writes.

A temporary file is `.NAME.tmp-PID` beside `NAME`: hidden (readers skip dot files) and unique to
the writing process, so two processes never share one and a reader never mistakes a half-written
file for a finished one."""

from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def temp_path(path: str | Path) -> Path:
    path = Path(path)
    return path.with_name(f".{path.name}.tmp-{os.getpid()}")


def remove_temps(directory: str | Path, name_glob: str) -> None:
    """Delete leftover temporary files of `name_glob` (an interrupted writer's) in `directory`."""
    for p in Path(directory).glob(f".{name_glob}.tmp-*"):
        p.unlink()


def atomic_write_bytes(path: str | Path, data: bytes) -> None:
    tmp = temp_path(path)
    tmp.write_bytes(data)
    os.replace(tmp, path)


def atomic_write_text(path: str | Path, text: str) -> None:
    atomic_write_bytes(path, text.encode())


def atomic_copy(src: str | Path, dst: str | Path) -> None:
    tmp = temp_path(dst)
    shutil.copyfile(src, tmp)
    os.replace(tmp, dst)
```

In `alphasettler/records.py`: add `import io` to the imports, replace the module docstring with

```python
"""Self-play records: one gzipped JSONL line per game (spec Section 4). Files are byte-for-byte
reproducible: the gzip header carries no file name and a zero timestamp."""
```

replace `open_writer` with

```python
@contextmanager
def open_writer(path: str | Path) -> Iterator[Writer]:
    """Create `path` (and its parent directory) exclusively and yield a `Writer` for it; an
    existing file is an error. Leaving the block, by an exception too, closes the gzip stream
    cleanly, so the games written so far stay readable."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "xb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
            with io.TextIOWrapper(gz, encoding="utf-8") as f:
                yield Writer(f)
```

and in `replay_mismatches` replace the observation check with

```python
            if "observation" in d and _plain(g.observation(d["actor"])) != d["observation"]:
                out.append(f"move {i}: observation differs")
```

Create `alphasettler/selfplay.py`:

```python
"""3b self-play records (spec Section 4): playout cap randomization, root noise on full searches
and opening sampling, for the heuristic bot (`ismcts`, the warm start) or a network (`net:NAME`).
Each game carries a `selfplay` block: the bot, the net file's sha256, both simulation caps and the
exploration settings. Decisions carry no observations; the seed and actions reproduce them."""

from __future__ import annotations

import math
import os
from pathlib import Path

from alphasettler import files, records
from alphasettler._engine import (
    DEFAULT_NETS_DIR, MAX_SIMULATIONS, NET_BATCH, NOISE_ALPHA, NOISE_WEIGHT, SAMPLE_TURNS, SEARCH_BATCH,
    SELFPLAY_MIN_SIMULATIONS, resolve_net, selfplay_explore,
)

# Games per native call. Python sees Ctrl-C only between calls, and every finished batch is on disk.
BATCH_GAMES = 100


def settings(net: str | None, nets_dir: str | Path, full_simulations: int, fast_simulations: int,
             full_probability: float) -> dict:
    """The `selfplay` block. Checks every argument, and loads the net, without playing a game."""
    for what, n in (("full", full_simulations), ("fast", fast_simulations)):
        if not SELFPLAY_MIN_SIMULATIONS <= n <= MAX_SIMULATIONS:
            raise ValueError(f"{what} simulations must be {SELFPLAY_MIN_SIMULATIONS}..={MAX_SIMULATIONS}, got {n}")
    if not (math.isfinite(full_probability) and 0 <= full_probability <= 1):
        raise ValueError(f"the full-search probability must be in [0, 1], got {full_probability}")
    path = None if net is None else resolve_net(net, str(nets_dir))
    return {
        "bot": "ismcts" if net is None else f"net:{net}",
        "net": net,
        "net_sha256": None if path is None else files.sha256_file(path),
        "full_simulations": full_simulations,
        "fast_simulations": fast_simulations,
        "full_probability": full_probability,
        "noise_weight": NOISE_WEIGHT,
        "noise_alpha": NOISE_ALPHA,
        "sample_turns": SAMPLE_TURNS,
        "batch": SEARCH_BATCH if net is None else NET_BATCH,
    }


def write_explore(out: str | Path, seed_start: int, games: int, *, config: dict, full_simulations: int,
                  fast_simulations: int, full_probability: float, net: str | None = None,
                  nets_dir: str | Path | None = None, threads: int | None = None,
                  counts: dict | None = None) -> dict:
    """Play `games` games from `seed_start` and write them to the new file `out`. Arguments are
    checked before the file is created. `counts` (if given) is updated as batches finish, so a
    caller that catches KeyboardInterrupt still knows what was kept."""
    nets_dir = DEFAULT_NETS_DIR if nets_dir is None else nets_dir
    meta = settings(net, nets_dir, full_simulations, fast_simulations, full_probability)
    counts = {} if counts is None else counts
    counts.update(games=0, decisions=0, full_decisions=0, belief_resets=0)
    threads = threads or os.cpu_count() or 1
    end = seed_start + games
    with records.open_writer(out) as w:
        for lo in range(seed_start, end, BATCH_GAMES):
            batch = selfplay_explore(lo, min(BATCH_GAMES, end - lo), threads, full_simulations, fast_simulations,
                                     full_probability, config, net, str(nets_dir))
            for g in batch:
                w.write_game({**g, "selfplay": meta}, config)
                counts["games"] += 1
                counts["decisions"] += len(g["decisions"])
                counts["full_decisions"] += sum(d["full_search"] for d in g["decisions"])
                counts["belief_resets"] += g["belief_resets"]
            w.flush()
    return counts
```

In `alphasettler/cli.py`:
- change the module docstring to `"""Command line: arena, compare, oracle-diff, selfplay, fit-heuristic, train and train-loop."""`;
- in `_parser`, replace the `selfplay` parser's `--simulations` line with these lines (keep the other selfplay arguments):

```python
    s.add_argument("--simulations", type=_positive_int, default=None,
                   help="3a self-play: every move a full search of this many simulations")
    s.add_argument("--pcr", action="store_true",
                   help="3b exploration with the heuristic bot (playout cap randomization, root noise, opening sampling)")
    s.add_argument("--net", default=None, help="3b exploration with the network <nets-dir>/NET.bin")
    s.add_argument("--nets-dir", default=DEFAULT_NETS_DIR)
    s.add_argument("--full-simulations", type=int, default=None, help="3b: full-search simulations (default 600)")
    s.add_argument("--fast-simulations", type=int, default=None, help="3b: fast-search simulations (default 100)")
    s.add_argument("--full-probability", type=float, default=None, help="3b: chance of a full search (default 0.25)")
```

- at the top of `_selfplay`, before its imports, add:

```python
    if args.net is not None or args.pcr:
        return _selfplay_explore(args)
    if args.simulations is None:
        raise ValueError("--simulations is required (or use --net NAME or --pcr for 3b exploration)")
    for flag in ("full_simulations", "fast_simulations", "full_probability"):
        if getattr(args, flag) is not None:
            raise ValueError(f"--{flag.replace('_', '-')} needs --net or --pcr")
```

- add after `_selfplay`:

```python
def _selfplay_explore(args) -> int:
    from alphasettler._engine import SELFPLAY_FAST_SIMULATIONS, SELFPLAY_FULL_PROBABILITY, SELFPLAY_FULL_SIMULATIONS
    from alphasettler.selfplay import write_explore

    if args.simulations is not None:
        raise ValueError("--simulations is for 3a self-play; with --net or --pcr use --full-simulations "
                         "and --fast-simulations")
    if args.rollout != 0:
        raise ValueError("--rollout is not supported with --net or --pcr")
    if args.seed_start < 0 or args.seed_start + args.games > SEED_LIMIT:
        raise ValueError(f"seeds {args.seed_start}..{args.seed_start + args.games} are outside the u64 seed range")
    full = SELFPLAY_FULL_SIMULATIONS if args.full_simulations is None else args.full_simulations
    fast = SELFPLAY_FAST_SIMULATIONS if args.fast_simulations is None else args.fast_simulations
    p = SELFPLAY_FULL_PROBABILITY if args.full_probability is None else args.full_probability
    config = {} if args.trades else {"max_offers_per_turn": 0}
    label = "ismcts" if args.net is None else f"net-{args.net}"
    out = Path(args.out_dir) / f"{_stamp()}-selfplay-{label}-f{full}-q{fast}.jsonl.gz"
    counts: dict = {}
    try:
        write_explore(out, args.seed_start, args.games, config=config, full_simulations=full,
                      fast_simulations=fast, full_probability=p, net=args.net, nets_dir=args.nets_dir,
                      threads=args.threads, counts=counts)
    except KeyboardInterrupt:
        print(f"interrupted: kept {counts.get('games', 0)} games, {counts.get('decisions', 0)} searched "
              f"decisions in {out}", file=sys.stderr)
        print(f"belief resets: {counts.get('belief_resets', 0)}", file=sys.stderr)
        return 130
    print(f"{counts['games']} games, {counts['decisions']} searched decisions "
          f"({counts['full_decisions']} full searches), wrote {out}")
    print(f"belief resets: {counts['belief_resets']}")
    return 0
```

(`write_explore` checks the simulations, the probability and the net, `bad net name` / `cannot read` included, before it creates `out`; a bad `--net` name never reaches the output path because `settings` raises first.)

- [ ] **Step 9: Run all tests**

Run: `PATH=$HOME/.cargo/bin:$PATH .venv/bin/maturin develop --release -q && .venv/bin/pytest -q`
Expected: all pass, 3a's `test_selfplay.py` and `test_cli.py` self-play tests included.

- [ ] **Step 10: Format, lint, full checks**

Run: `~/.cargo/bin/cargo fmt --all && ~/.cargo/bin/cargo fmt --check`
Expected: no output from the check.
Run: `~/.cargo/bin/cargo clippy --workspace --release --all-targets -- -D warnings`
Expected: exits 0.
Run: `~/.cargo/bin/cargo test --workspace --release`
Expected: all pass.

- [ ] **Step 11: Commit**

```bash
git add net/ bindings/ alphasettler/files.py alphasettler/records.py alphasettler/selfplay.py alphasettler/cli.py tests/python/test_training_data.py
git diff --cached | grep -nE 'sk-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{36}|password=[^[:space:]]+'
git -c user.email=sean.may101@gmail.com commit -m "net, bindings: replay-and-sample training batches, encode, and 3b self-play records"
```

---

### Task 6: PyTorch training: the model twin, the data window, the training run and `train`

**Files:**
- Modify: `pyproject.toml`, `alphasettler/cli.py`
- Create: `alphasettler/train/__init__.py`, `alphasettler/train/model.py`, `alphasettler/train/data.py`, `alphasettler/train/train.py`
- Test: `tests/python/test_train.py`

**Interfaces:**
- Consumes: `_engine.{TrainingSet, encode, net_forward, NET_PUBLIC_LEN, NET_HIDDEN_LEN, NET_ENCODING_HASH, NET_FORMAT_VERSION, action_space_size}` (Task 5); `alphasettler.records.read`, `alphasettler.files.{sha256_file, temp_path}` (Task 5); `alphasettler.selfplay.write_explore` (tests).
- Produces:
  - `alphasettler.train.model`: `MAGIC`, `ACTIONS`, `SEATS`; `Net(trunk=512, value=256)` (`nn.Module`; attributes `trunk_width`, `value_width`; `layers() -> list[nn.Linear]` in file order; `forward(public, hidden) -> (policy_logits, value_logits, aux)`); `to_bytes(net) -> bytes`; `from_bytes(data) -> Net`; `save(net, path)`; `load(path) -> Net`.
  - `alphasettler.train.data`: `record_files(gen_dir) -> list[Path]`; `compact(game) -> dict`; `read_generation(gen_dir, notices: list[str]) -> list[dict]`; `Window(data: TrainingSet, new_positions: int, files: dict[str, str], notices: list[str])`; `load_window(root, gen_dirs) -> Window`; `Tensors(public, hidden, mask, policy, full, outcome, vp)`; `to_tensors(batch: dict, device) -> Tensors`.
  - `alphasettler.train.train`: `PARITY_POSITIONS` (100), `PARITY_TOLERANCE` (1e-5), `L2` (1e-4), `AUX_WEIGHT` (0.25); `TrainConfig(lr, steps=None, steps_factor=2.0, batch=1024, seed=0, device="auto", threads=None, trunk=512, value=256, log_every=200)`; `pick_device(name) -> torch.device`; `auto_steps(new_positions, batch, factor=2.0) -> int`; `losses(net, t: Tensors) -> dict[str, Tensor]` (keys `policy`, `value`, `aux`, `l2`, `total`); `parity_check(net, path, window) -> float`; `train(window, out, cfg, init=None) -> dict` (keys `steps`, `lr`, `batch`, `init`, `device`, `history`, `parity_max_error`).
  - CLI: `alphasettler train GEN_DIR... --out PATH [--init PATH] [--lr 3e-4] [--steps N] [--batch 1024] [--seed 0] [--device auto|mps|cpu] [--threads T] [--trunk 512] [--value-width 256]` (the last directory is the newest generation).

Loss (spec Section 4): policy cross-entropy against the normalised visits over full-search rows only (`full` is 0 for fast searches); value cross-entropy against the outcome; 0.25 × MSE(aux, final VP / 10); plus 1e-4 × the sum of squared weights of every layer (biases excluded). Adam without weight decay. Parity tolerance: |Rust − PyTorch| ≤ 1e-5 × max(1, |PyTorch|) for every output.

- [ ] **Step 1: Install PyTorch and audit it**

Edit `pyproject.toml`'s `[project.optional-dependencies]` to:

```toml
[project.optional-dependencies]
dev = ["maturin==1.15.0", "pytest==9.1.1"]
oracle = ["catanatron @ git+https://github.com/bcollazo/catanatron.git@ecf931181b9a65bb4116a2153fb78c16f1438e00"]
train = ["torch==2.14.1"]
```

Run: `.venv/bin/pip3 install "torch==2.14.1"`
Expected: `Successfully installed ... torch-2.14.1` (with its dependencies: filelock, fsspec, jinja2, markupsafe, mpmath, networkx, setuptools, sympy, typing-extensions; some may already be present).
Run: `.venv/bin/python3 -c "import torch; print(torch.__version__, torch.backends.mps.is_available())"`
Expected: `2.14.1 True`.
Run: `.venv/bin/pip-audit --version || .venv/bin/pip3 install "pip-audit==2.10.1"` then `.venv/bin/pip-audit --skip-editable`
Expected: `No known vulnerabilities found`. If it lists a vulnerability in torch or a package torch pulled in, stop and report it to the controller instead of continuing.

- [ ] **Step 2: Write the failing tests**

Create `tests/python/test_train.py`:

```python
import random
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from alphasettler import Bot, Game, records
from alphasettler._engine import NET_HIDDEN_LEN, NET_PUBLIC_LEN, encode, net_forward
from alphasettler.selfplay import write_explore
from alphasettler.train.data import load_window, to_tensors
from alphasettler.train.model import Net, from_bytes, to_bytes
from alphasettler.train.train import TrainConfig, auto_steps, losses, parity_check, pick_device, train

ROOT = Path(__file__).resolve().parents[2]
CONFIG = {"max_offers_per_turn": 0}
CPU = torch.device("cpu")


def cli(*args):
    return subprocess.run([sys.executable, "-m", "alphasettler", *args], capture_output=True, text=True)


def write_games(path, seed_start, games):
    write_explore(path, seed_start, games, config=CONFIG, full_simulations=6, fast_simulations=3,
                  full_probability=0.5, threads=2)


def generation(tmp_path, name="gen-000", seed_start=0, games=2):
    d = tmp_path / name
    d.mkdir(parents=True, exist_ok=True)
    write_games(d / f"selfplay-{seed_start}.jsonl.gz", seed_start, games)
    return d


def random_positions(n, seed=0):
    """`n` encoded positions from uniformly random games (trades on), every 7th move."""
    rng = random.Random(seed)
    public, hidden = [], []
    game_seed = 0
    while len(public) < n:
        g = Game(game_seed, None)
        game_seed += 1
        for k in range(400):
            if g.is_over or len(public) == n:
                break
            if k % 7 == 0:
                p, h = encode(g)
                public.append(p)
                hidden.append(h)
            g.apply(rng.choice(g.legal_actions()))
    return b"".join(public), b"".join(hidden)


def test_rust_and_pytorch_forward_passes_agree_on_random_weights(tmp_path):
    torch.manual_seed(0)
    net = Net().eval()
    path = tmp_path / "r.bin"
    path.write_bytes(to_bytes(net))
    public, hidden = random_positions(100)
    rust = net_forward(str(path), public, hidden)
    pt = torch.frombuffer(bytearray(public), dtype=torch.float32).view(100, NET_PUBLIC_LEN)
    ht = torch.frombuffer(bytearray(hidden), dtype=torch.float32).view(100, NET_HIDDEN_LEN)
    with torch.no_grad():
        outs = net(pt, ht)
    for key, ref in zip(("policy", "value", "aux"), outs):
        got = torch.frombuffer(bytearray(rust[key]), dtype=torch.float32).view_as(ref)
        err = ((got - ref).abs() / ref.abs().clamp(min=1.0)).max().item()
        assert err <= 1e-5, (key, err)


def test_weights_round_trip_and_bad_files_are_errors():
    torch.manual_seed(1)
    net = Net(16, 8)
    data = to_bytes(net)
    back = from_bytes(data)
    assert (back.trunk_width, back.value_width) == (16, 8)
    assert all(torch.equal(a, b) for a, b in zip(net.state_dict().values(), back.state_dict().values()))
    shape = bytearray(data)
    shape[32:36] = (15).to_bytes(4, "little")
    for bad, message in [(b"X" + data[1:], "magic"), (data[:12] + bytes([data[12] ^ 1]) + data[13:], "encoding hash"),
                         (bytes(shape), "shapes"), (data[:-1], "truncated"), (data[:40], "truncated"),
                         (data + b"\0", "trailing")]:
        with pytest.raises(ValueError, match=message):
            from_bytes(bad)


def test_steps_are_twice_the_new_positions_over_the_batch():
    assert auto_steps(1024, 1024) == 2
    assert auto_steps(1, 1024) == 1
    assert auto_steps(5000, 1024) == 10
    assert auto_steps(5000, 1024, 3.0) == 15


def test_fast_searches_add_nothing_to_the_policy_loss(tmp_path):
    w = load_window(tmp_path, [generation(tmp_path)])
    t = to_tensors(w.data.sample(64, 5, 2), CPU)
    assert 0 < t.full.sum().item() < 64
    torch.manual_seed(2)
    net = Net(32, 16)
    base = losses(net, t)["policy"]
    fast = t.full == 0
    noise = torch.rand(t.policy.shape) * t.mask
    other = torch.where(fast[:, None], noise / noise.sum(1, keepdim=True), t.policy)
    assert torch.equal(losses(net, replace(t, policy=other))["policy"], base)
    logits, _, _ = net(t.public, t.hidden)
    logp = torch.log_softmax(logits.masked_fill(~t.mask, -1e9), dim=1)
    assert torch.allclose(base, -(t.policy * logp).sum(1)[~fast].mean())


def test_a_tiny_dataset_can_be_overfit(tmp_path):
    w = load_window(tmp_path, [generation(tmp_path)])
    t = to_tensors(w.data.sample(32, 1, 2), CPU)
    torch.manual_seed(3)
    net = Net(64, 32)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    for _ in range(600):
        parts = losses(net, t)
        opt.zero_grad()
        parts["total"].backward()
        opt.step()
    parts = losses(net, t)

    def entropy(p):
        return -(p * torch.log(p.clamp(min=1e-12))).sum(1)

    policy_floor = ((entropy(t.policy) * t.full).sum() / t.full.sum()).item()
    value_floor = entropy(t.outcome).mean().item()
    assert parts["policy"].item() - policy_floor < 0.05
    assert parts["value"].item() - value_floor < 0.05
    assert parts["aux"].item() < 1e-3


def test_without_mps_training_falls_back_to_the_cpu_with_a_notice(monkeypatch, capsys):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert pick_device("auto").type == "cpu"
    assert "MPS is not available" in capsys.readouterr().out
    assert pick_device("mps").type == "cpu"
    assert "MPS is not available" in capsys.readouterr().out
    assert pick_device("cpu").type == "cpu"
    with pytest.raises(ValueError, match="unknown device"):
        pick_device("cuda")


def test_unreadable_empty_and_decisionless_files_are_skipped_with_a_notice(tmp_path):
    gen = generation(tmp_path)
    (gen / "empty.jsonl.gz").write_bytes(b"")
    other = tmp_path / "other"
    other.mkdir()
    write_games(other / "o.jsonl.gz", 100, 3)
    data = (other / "o.jsonl.gz").read_bytes()
    (gen / "truncated.jsonl.gz").write_bytes(data[: len(data) // 2])
    write_games(other / "z.jsonl.gz", 200, 1)
    (lone,) = records.read(other / "z.jsonl.gz")
    records.write(gen / "nodecisions.jsonl.gz", [{**lone, "decisions": []}], CONFIG)
    (gen / ".half.jsonl.gz.tmp-123").write_bytes(b"garbage")
    w = load_window(tmp_path, [gen])
    text = "\n".join(w.notices)
    assert "empty.jsonl.gz: no games" in text
    assert "truncated.jsonl.gz: kept" in text and "unreadable" in text
    assert "nodecisions.jsonl.gz: 1 games but no decisions" in text
    assert ".half" not in text and not any(".tmp-" in k for k in w.files)
    assert w.data.games >= 3 and w.new_positions == w.data.decisions > 0


def test_nothing_to_train_on_is_an_error_before_training(tmp_path):
    empty = tmp_path / "gen-001"
    empty.mkdir()
    (empty / "empty.jsonl.gz").write_bytes(b"")
    with pytest.raises(ValueError, match="no training positions"):
        load_window(tmp_path, [empty])
    old = generation(tmp_path, "gen-000")
    with pytest.raises(ValueError, match="no decisions to train on"):
        load_window(tmp_path, [old, empty])


def test_two_writers_in_one_generation_are_read_once_each(tmp_path):
    gen = generation(tmp_path)
    write_games(gen / "selfplay-elsewhere.jsonl.gz", 0, 2)
    write_games(gen / "selfplay-more.jsonl.gz", 50, 1)
    w = load_window(tmp_path, [gen])
    assert w.data.games == 3
    assert sum("already in gen-000" in n for n in w.notices) == 2
    assert set(w.files) == {"gen-000/selfplay-0.jsonl.gz", "gen-000/selfplay-elsewhere.jsonl.gz",
                            "gen-000/selfplay-more.jsonl.gz"}


def test_train_exports_a_net_the_rust_side_loads_and_checks_parity(tmp_path):
    w = load_window(tmp_path, [generation(tmp_path)])
    nets = tmp_path / "nets"
    nets.mkdir()
    cfg = TrainConfig(lr=1e-3, steps=5, batch=16, device="cpu", trunk=16, value=8, threads=2)
    s = train(w, nets / "t.bin", cfg)
    assert s["steps"] == 5 and s["device"] == "cpu" and s["parity_max_error"] <= 1e-5
    assert not [p for p in nets.iterdir() if p.name.startswith(".")], "no temp file left"
    assert Bot("net:t@5", 0, nets_dir=str(nets)).name == "net"
    train(w, nets / "u.bin", replace(cfg, threads=1))
    assert (nets / "u.bin").read_bytes() == (nets / "t.bin").read_bytes(), "CPU training is reproducible"
    s = train(w, nets / "v.bin", replace(cfg, lr=3e-4, steps=2), init=nets / "t.bin")
    assert s["init"] == str(nets / "t.bin") and from_bytes((nets / "v.bin").read_bytes()).trunk_width == 16
    with pytest.raises(RuntimeError, match="disagree"):
        parity_check(Net(16, 8), nets / "t.bin", w)


def test_cli_train(tmp_path):
    gen = generation(tmp_path)
    out = tmp_path / "nets" / "c.bin"
    r = cli("train", str(gen), "--out", str(out), "--steps", "3", "--batch", "16", "--device", "cpu",
            "--trunk", "16", "--value-width", "8")
    assert r.returncode == 0, r.stderr
    assert f"wrote {out}" in r.stdout and "parity" in r.stdout
    r = cli("train", str(tmp_path / "nothing"), "--out", str(out))
    assert r.returncode == 2 and "no training positions" in r.stderr


def test_the_base_package_does_not_import_torch():
    code = ("import sys, alphasettler, alphasettler.cli, alphasettler.records, alphasettler.selfplay, "
            "alphasettler.files, alphasettler.arena; print('torch' in sys.modules)")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0 and r.stdout.strip() == "False", r.stderr
    outside = [p for p in ROOT.glob("alphasettler/**/*.py") if "train" not in p.relative_to(ROOT).parts]
    assert not [p for p in outside if "import torch" in p.read_text()]
```

- [ ] **Step 3: Run them to verify they fail**

Run: `.venv/bin/pytest -q tests/python/test_train.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'alphasettler.train'`.

- [ ] **Step 4: Write the model twin**

Create `alphasettler/train/__init__.py`:

```python
"""Training the network (spec Section 4). Needs the `train` extra (PyTorch); nothing outside
this package imports torch."""
```

Create `alphasettler/train/model.py`:

```python
"""The PyTorch twin of net/src/forward.rs, and the weights file format of net/src/weights.rs:
little-endian, an 80-byte header (magic, format version, ENCODING_HASH, layer count, each layer's
rows and columns), then each layer's f32 weights (row-major, rows = outputs) and biases."""

from __future__ import annotations

import struct
import sys
from array import array
from pathlib import Path

import torch
from torch import nn

from alphasettler._engine import NET_ENCODING_HASH, NET_FORMAT_VERSION, NET_HIDDEN_LEN, NET_PUBLIC_LEN, action_space_size

MAGIC = b"ASNET\0\0\0"
ACTIONS = action_space_size()
SEATS = 4
NUM_LAYERS = 7
HEADER_LEN = 8 + 4 + 8 + 4 + NUM_LAYERS * 8


class Net(nn.Module):
    """public -> trunk (3 x ReLU) -> policy logits; [trunk || hidden] -> ReLU -> value logits
    and the aux head (final VP / 10). All value and aux outputs are seat-relative."""

    def __init__(self, trunk: int = 512, value: int = 256):
        super().__init__()
        self.trunk_width, self.value_width = trunk, value
        self.trunk = nn.ModuleList([nn.Linear(NET_PUBLIC_LEN, trunk), nn.Linear(trunk, trunk), nn.Linear(trunk, trunk)])
        self.policy = nn.Linear(trunk, ACTIONS)
        self.value0 = nn.Linear(trunk + NET_HIDDEN_LEN, value)
        self.value1 = nn.Linear(value, SEATS)
        self.aux = nn.Linear(value, SEATS)

    def layers(self) -> list[nn.Linear]:
        """In weights-file order."""
        return [*self.trunk, self.policy, self.value0, self.value1, self.aux]

    def forward(self, public: torch.Tensor, hidden: torch.Tensor):
        h = public
        for layer in self.trunk:
            h = torch.relu(layer(h))
        v = torch.relu(self.value0(torch.cat([h, hidden], dim=1)))
        return self.policy(h), self.value1(v), self.aux(v)


def _floats(t: torch.Tensor) -> bytes:
    return array("f", t.detach().to("cpu", torch.float32).contiguous().view(-1).tolist()).tobytes()


def to_bytes(net: Net) -> bytes:
    assert sys.byteorder == "little", "the weights file is little-endian"
    layers = net.layers()
    out = [MAGIC, struct.pack("<IQI", NET_FORMAT_VERSION, NET_ENCODING_HASH, len(layers))]
    out += [struct.pack("<II", layer.out_features, layer.in_features) for layer in layers]
    for layer in layers:
        out += [_floats(layer.weight), _floats(layer.bias)]
    return b"".join(out)


def from_bytes(data: bytes) -> Net:
    if len(data) < HEADER_LEN:
        raise ValueError(f"truncated: {len(data)} bytes is shorter than the {HEADER_LEN}-byte header")
    if data[:8] != MAGIC:
        raise ValueError("not an AlphaSettler weights file (wrong magic)")
    version, encoding, count = struct.unpack_from("<IQI", data, 8)
    if version != NET_FORMAT_VERSION:
        raise ValueError(f"format version {version}, expected {NET_FORMAT_VERSION}")
    if encoding != NET_ENCODING_HASH:
        raise ValueError(f"encoding hash {encoding:#018x} does not match this encoder's {NET_ENCODING_HASH:#018x}")
    if count != NUM_LAYERS:
        raise ValueError(f"{count} layers, expected {NUM_LAYERS}")
    shapes = [struct.unpack_from("<II", data, 24 + 8 * i) for i in range(NUM_LAYERS)]
    net = Net(trunk=shapes[0][0], value=shapes[4][0])
    if [(layer.out_features, layer.in_features) for layer in net.layers()] != shapes:
        raise ValueError(f"layer shapes {shapes} do not fit the network")
    at = HEADER_LEN
    with torch.no_grad():
        for layer in net.layers():
            for p in (layer.weight, layer.bias):
                n = p.numel()
                chunk = data[at : at + 4 * n]
                if len(chunk) != 4 * n:
                    raise ValueError(f"truncated: {len(data)} bytes")
                p.copy_(torch.frombuffer(bytearray(chunk), dtype=torch.float32).view_as(p))
                at += 4 * n
    if at != len(data):
        raise ValueError(f"{len(data) - at} trailing bytes after the last layer")
    return net


def save(net: Net, path: str | Path) -> None:
    Path(path).write_bytes(to_bytes(net))


def load(path: str | Path) -> Net:
    return from_bytes(Path(path).read_bytes())
```

- [ ] **Step 5: Write the data window**

Create `alphasettler/train/data.py`:

```python
"""Training data (spec Section 4): self-play record files -> the Rust sampler -> tensors.

A generation directory holds finished record files `*.jsonl.gz` (other machines may drop more in:
the scale-out path). Hidden temp files are never read. Files that are empty, cut short or hold
no decisions are skipped with a notice, and a seed already read in the same generation (a
second writer playing the same seeds) is read once."""

from __future__ import annotations

import json
import zlib
from dataclasses import dataclass
from pathlib import Path

import torch

from alphasettler import files, records
from alphasettler._engine import NET_HIDDEN_LEN, NET_PUBLIC_LEN, TrainingSet, action_space_size

ACTIONS = action_space_size()


def record_files(gen_dir: str | Path) -> list[Path]:
    """Finished record files in `gen_dir`, by name; temporary (dot) files are not."""
    return sorted(p for p in Path(gen_dir).glob("*.jsonl.gz") if not p.name.startswith("."))


def compact(game: dict) -> dict:
    """Only what the sampler needs."""
    return {
        "seed": game["seed"],
        "config": game["config"],
        "actions": game["actions"],
        "decisions": [{"index": d["index"], "actor": d["actor"], "visits": d["visits"], "full_search": d["full_search"]}
                      for d in game["decisions"]],
        "winner": game["winner"],
        "vp": game["vp"],
    }


def read_generation(gen_dir: str | Path, notices: list[str]) -> list[dict]:
    gen_dir = Path(gen_dir)
    games: list[dict] = []
    seen: set[int] = set()
    for path in record_files(gen_dir):
        kept: list[dict] = []
        broken = False
        try:
            for g in records.read(path):
                if g["seed"] in seen:
                    notices.append(f"{path}: seed {g['seed']} is already in {gen_dir.name}; skipped the duplicate")
                    continue
                seen.add(g["seed"])
                kept.append(compact(g))
        except (EOFError, OSError, zlib.error, json.JSONDecodeError, KeyError) as e:
            broken = True
            notices.append(f"{path}: kept {len(kept)} games, skipped an unreadable rest ({type(e).__name__}: {e})")
        if not kept and not broken:
            notices.append(f"{path}: no games")
        elif kept and not any(g["decisions"] for g in kept):
            notices.append(f"{path}: {len(kept)} games but no decisions")
        games += kept
    return games


@dataclass
class Window:
    data: TrainingSet
    new_positions: int
    files: dict[str, str]
    notices: list[str]


def load_window(root: str | Path, gen_dirs: list[Path]) -> Window:
    """The training set over `gen_dirs` (oldest first; the last is the newest generation), the
    newest generation's decision count, and every record file read (path relative to `root` ->
    sha256)."""
    root = Path(root)
    notices: list[str] = []
    games: list[dict] = []
    new = 0
    for i, d in enumerate(gen_dirs):
        gs = read_generation(d, notices)
        if i == len(gen_dirs) - 1:
            new = sum(len(g["decisions"]) for g in gs)
        games += gs
    data = TrainingSet(games)
    if data.decisions == 0:
        raise ValueError(f"no training positions in {', '.join(str(d) for d in gen_dirs)}")
    if new == 0:
        raise ValueError(f"the newest generation {gen_dirs[-1]} has no decisions to train on")
    read = {str(p.relative_to(root)): files.sha256_file(p) for d in gen_dirs for p in record_files(d)}
    return Window(data, new, read, notices)


@dataclass
class Tensors:
    public: torch.Tensor
    hidden: torch.Tensor
    mask: torch.Tensor
    policy: torch.Tensor
    full: torch.Tensor
    outcome: torch.Tensor
    vp: torch.Tensor


def to_tensors(batch: dict, device: torch.device) -> Tensors:
    """Wrap a sampler batch's little-endian buffers (`torch.frombuffer`, no numpy)."""
    n = batch["n"]

    def f32(key: str, width: int) -> torch.Tensor:
        return torch.frombuffer(bytearray(batch[key]), dtype=torch.float32).view(n, width).to(device)

    def u8(key: str, width: int) -> torch.Tensor:
        return torch.frombuffer(bytearray(batch[key]), dtype=torch.uint8).view(n, width).to(device)

    return Tensors(
        public=f32("public", NET_PUBLIC_LEN),
        hidden=f32("hidden", NET_HIDDEN_LEN),
        mask=u8("mask", ACTIONS).bool(),
        policy=f32("policy", ACTIONS),
        full=u8("full", 1).view(n).float(),
        outcome=f32("outcome", 4),
        vp=f32("vp", 4),
    )
```

- [ ] **Step 6: Write the training run**

Create `alphasettler/train/train.py`:

```python
"""One training run (spec Section 4): Adam on batches from the Rust sampler, then export, then
a parity check of the exported file's Rust forward pass against PyTorch before the file is put
in place. Training on MPS is not bit-reproducible; on the CPU it is."""

from __future__ import annotations

import copy
import math
import os
from dataclasses import dataclass
from pathlib import Path

import torch

from alphasettler import files
from alphasettler._engine import net_forward
from alphasettler.train import model as model_mod
from alphasettler.train.data import Tensors, Window, to_tensors

PARITY_POSITIONS = 100
PARITY_TOLERANCE = 1e-5
PARITY_SEED = 0x9A217
L2 = 1e-4
AUX_WEIGHT = 0.25


@dataclass(frozen=True)
class TrainConfig:
    lr: float
    steps: int | None = None  # None: steps_factor x new positions / batch, rounded up
    steps_factor: float = 2.0
    batch: int = 1024
    seed: int = 0
    device: str = "auto"
    threads: int | None = None
    trunk: int = 512  # for a net trained from scratch; an `init` file brings its own shape
    value: int = 256
    log_every: int = 200


def pick_device(name: str) -> torch.device:
    """`cpu`, or `auto` / `mps`: MPS when available, else the CPU with a printed notice."""
    if name == "cpu":
        return torch.device("cpu")
    if name not in ("auto", "mps"):
        raise ValueError(f"unknown device {name!r}; use auto, mps or cpu")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    print("notice: MPS is not available; training on the CPU", flush=True)
    return torch.device("cpu")


def auto_steps(new_positions: int, batch: int, factor: float = 2.0) -> int:
    """Enough steps to draw each of the newest generation's positions `factor` times on average."""
    return max(1, math.ceil(factor * new_positions / batch))


def losses(net: model_mod.Net, t: Tensors) -> dict[str, torch.Tensor]:
    logits, value, aux = net(t.public, t.hidden)
    logp = torch.log_softmax(logits.masked_fill(~t.mask, -1e9), dim=1)
    per_position = -(t.policy * logp).sum(dim=1)
    policy = (per_position * t.full).sum() / t.full.sum().clamp(min=1.0)
    value_loss = -(t.outcome * torch.log_softmax(value, dim=1)).sum(dim=1).mean()
    aux_loss = torch.nn.functional.mse_loss(aux, t.vp)
    l2 = sum((layer.weight ** 2).sum() for layer in net.layers())
    total = policy + value_loss + AUX_WEIGHT * aux_loss + L2 * l2
    return {"policy": policy, "value": value_loss, "aux": aux_loss, "l2": l2, "total": total}


def parity_check(net: model_mod.Net, path: str | Path, window: Window) -> float:
    """Max relative error between the Rust forward pass of the file at `path` and `net` (on the
    CPU) over `PARITY_POSITIONS` sampled positions; RuntimeError above `PARITY_TOLERANCE`."""
    batch = window.data.sample(PARITY_POSITIONS, PARITY_SEED, 1)
    cpu = copy.deepcopy(net).to("cpu").eval()
    t = to_tensors(batch, torch.device("cpu"))
    with torch.no_grad():
        ref = cpu(t.public, t.hidden)
    rust = net_forward(str(path), batch["public"], batch["hidden"])
    worst = 0.0
    for key, r in zip(("policy", "value", "aux"), ref):
        got = torch.frombuffer(bytearray(rust[key]), dtype=torch.float32).view_as(r)
        worst = max(worst, ((got - r).abs() / r.abs().clamp(min=1.0)).max().item())
    if worst > PARITY_TOLERANCE:
        raise RuntimeError(f"the exported weights disagree with PyTorch: max error {worst:.3g} > {PARITY_TOLERANCE}")
    return worst


def train(window: Window, out: str | Path, cfg: TrainConfig, init: str | Path | None = None) -> dict:
    """Train from `init` (a weights file) or from scratch, write `out` atomically after the
    parity check, and return a summary."""
    out = Path(out)
    torch.manual_seed(cfg.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = pick_device(cfg.device)
    threads = cfg.threads or os.cpu_count() or 1
    net = model_mod.load(init) if init is not None else model_mod.Net(cfg.trunk, cfg.value)
    net.to(device).train()
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    steps = cfg.steps or auto_steps(window.new_positions, cfg.batch, cfg.steps_factor)
    history = []
    for step in range(steps):
        batch = window.data.sample(cfg.batch, (cfg.seed * 2**32 + step) % 2**64, threads)
        parts = losses(net, to_tensors(batch, device))
        opt.zero_grad()
        parts["total"].backward()
        opt.step()
        if step % cfg.log_every == 0 or step == steps - 1:
            row = {"step": step, **{k: round(float(v), 6) for k, v in parts.items()}}
            history.append(row)
            print(f"step {step + 1}/{steps}: total {row['total']:.4f}, policy {row['policy']:.4f}, "
                  f"value {row['value']:.4f}, aux {row['aux']:.5f}", flush=True)
    tmp = files.temp_path(out)
    tmp.write_bytes(model_mod.to_bytes(net))
    try:
        err = parity_check(net, tmp, window)
    except BaseException:
        tmp.unlink()
        raise
    os.replace(tmp, out)
    return {"steps": steps, "lr": cfg.lr, "batch": cfg.batch, "init": None if init is None else str(init),
            "device": device.type, "history": history, "parity_max_error": err}
```

- [ ] **Step 7: The `train` command**

In `alphasettler/cli.py`, add to `_parser` before `return p`:

```python
    t = sub.add_parser("train", help="train a network on self-play records (needs the train extra)")
    t.add_argument("records", nargs="+", help="generation directories, oldest first; the last is the newest")
    t.add_argument("--out", required=True, help="weights file to write")
    t.add_argument("--init", default=None, help="weights file to start from (default: from scratch)")
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--steps", type=_positive_int, default=None, help="default: 2 x new positions / batch")
    t.add_argument("--batch", type=_positive_int, default=1024)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--device", default="auto", choices=["auto", "mps", "cpu"])
    t.add_argument("--threads", type=_positive_int, default=None)
    t.add_argument("--trunk", type=_positive_int, default=512)
    t.add_argument("--value-width", type=_positive_int, default=256)
```

Add the handler after `_fit_heuristic`:

```python
def _train(args) -> int:
    from alphasettler.train.data import load_window
    from alphasettler.train.train import TrainConfig, train

    dirs = [Path(d).resolve() for d in args.records]
    root = Path(os.path.commonpath([str(d) for d in dirs]))
    window = load_window(root, dirs)
    for line in window.notices:
        print(f"notice: {line}")
    print(f"window: {window.data.games} games, {window.data.decisions} positions "
          f"({window.data.full_decisions} full searches), {window.new_positions} new")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cfg = TrainConfig(lr=args.lr, steps=args.steps, batch=args.batch, seed=args.seed, device=args.device,
                      threads=args.threads, trunk=args.trunk, value=args.value_width)
    try:
        s = train(window, out, cfg, Path(args.init) if args.init else None)
    except KeyboardInterrupt:
        print("interrupted: nothing written", file=sys.stderr)
        return 130
    except RuntimeError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(f"trained {s['steps']} steps on {s['device']}; parity max error {s['parity_max_error']:.2e}; wrote {out}")
    return 0
```

In `main`, add the branch `elif args.command == "train": return _train(args)` after the `fit-heuristic` branch.

- [ ] **Step 8: Run the tests to verify they pass**

Run: `.venv/bin/pytest -q tests/python/test_train.py`
Expected: 12 passed.
Run: `.venv/bin/pytest -q -rs`
Expected: all pass and the summary lists no skipped tests (PyTorch and Catanatron are both installed).

- [ ] **Step 9: Format, lint, full checks**

Run: `~/.cargo/bin/cargo fmt --check`
Expected: no output.
Run: `~/.cargo/bin/cargo clippy --workspace --release --all-targets -- -D warnings`
Expected: exits 0.

- [ ] **Step 10: Commit**

```bash
git add pyproject.toml alphasettler/train/ alphasettler/cli.py tests/python/test_train.py
git diff --cached | grep -nE 'sk-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{36}|password=[^[:space:]]+'
git -c user.email=sean.may101@gmail.com commit -m "train: PyTorch twin, data window, training run with parity check, and alphasettler train"
```

---

### Task 7: `train-loop`: generations, manifest, resume and gating

**Files:**
- Create: `alphasettler/train/loop.py`
- Modify: `alphasettler/cli.py`
- Test: `tests/python/test_train_loop.py`

**Interfaces:**
- Consumes: `alphasettler.selfplay.{write_explore, settings}`, `alphasettler.files.*`, `alphasettler.records.read` (Task 5); `alphasettler.arena.run(candidate, baseline, seeds, seed_start, threads, config, out, nets_dir)` (Task 3); `alphasettler.train.data.load_window`, `alphasettler.train.train.{TrainConfig, train}` (Task 6).
- Produces (`alphasettler.train.loop`): `MANIFEST` (`"manifest.json"`), `MANIFEST_FORMAT` (1), `STEPS` (`("selfplay", "train", "gate")`), `SEEDS_PER_GENERATION` (10,000,000), `GATE_SEED_START` (1,000,000,000); `LoopSettings(generations, games=10_000, warm_games=20_000, chunk_games=500, full_simulations=600, fast_simulations=100, full_probability=0.25, window=4, warm_lr=1e-3, lr=3e-4, steps_factor=2.0, batch=1024, trunk=512, value=256, gate_seeds=200, gate_simulations=600, max_turns=1000, device="auto", threads=None)`; `gen_name(k) -> str` (`gen-NNN`); `promoted(wins, games) -> bool`; `Loop(root, settings)` with `.manifest`, `.nets` and `run(stop_after=None) -> int` (steps executed). CLI: `alphasettler train-loop --dir DIR --generations N [--games 10000] [--warm-games 20000] [--chunk-games 500] [--full-simulations 600] [--fast-simulations 100] [--full-probability 0.25] [--window 4] [--warm-lr 1e-3] [--lr 3e-4] [--steps-factor 2.0] [--batch 1024] [--trunk 512] [--value-width 256] [--gate-seeds 200] [--gate-simulations 600] [--max-turns 1000] [--device auto] [--threads T]`.

Layout under `--dir` (for example `runs/3b`):

| Path | Written by | Contents |
|---|---|---|
| `manifest.json` | every step | `{"format": 1, "best": k, "generations": [{"gen": k, "selfplay": {...}, "train": {...}, "gate": {...}}]}` with the outputs' sha256 |
| `gen-NNN/selfplay.json` | self-play | the step's settings, written first; a rerun with other settings on a started step is an error |
| `gen-NNN/selfplay-local-NNNNN.jsonl.gz` | self-play | `chunk_games` games each (other machines may add `*.jsonl.gz` files of their own before training) |
| `gen-NNN/net.bin`, `nets/gen-NNN.bin` | train | the weights, and a copy for `net:gen-NNN` |
| `gen-NNN/gate.jsonl`, `gen-NNN/gate.json`, `nets/best.bin` | gate | the match records, the result, and the best net for `net:best` |

`--generations N` counts the learned generations after the warm start: `0` runs generation 0 only. Generation 0 self-plays `warm_games` games with `ismcts` and the same exploration, trains from scratch at `warm_lr`, and its gate promotes it without a match. Generation k ≥ 1 self-plays `games` games with the best net, trains from generation k−1's `net.bin` at `lr`, and plays `net:gen-k@600` against three `net:gen-best@600` over `gate_seeds` seeds from `GATE_SEED_START` in the native arena with trades off. Self-play seeds of generation k start at `k × SEEDS_PER_GENERATION`; the training seed of generation k is k. Changed flags on a rerun apply to steps not yet started (this is how extra generations change hyperparameters); completed steps are never redone.

- [ ] **Step 1: Write the failing tests**

Create `tests/python/test_train_loop.py`:

```python
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import replace

import pytest

from alphasettler.train.loop import Loop, LoopSettings, promoted

TINY = LoopSettings(generations=1, games=2, warm_games=2, chunk_games=1, full_simulations=4, fast_simulations=2,
                    full_probability=0.5, batch=16, trunk=16, value=8, gate_seeds=1, gate_simulations=4,
                    max_turns=60, device="cpu", threads=2)


def manifest(d):
    return json.loads((d / "manifest.json").read_text())


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def temps(d):
    return [p for p in d.rglob(".*.tmp-*")]


@pytest.fixture(scope="module")
def reference(tmp_path_factory):
    d = tmp_path_factory.mktemp("reference")
    assert Loop(d, TINY).run() == 6
    return d


def test_the_promotion_rule_includes_a_tie_at_a_quarter():
    assert promoted(200, 800) and promoted(50, 200) and promoted(1, 4)
    assert promoted(201, 800)
    assert not promoted(199, 800) and not promoted(0, 4)


def test_a_full_run_writes_every_output_and_redoes_nothing(reference):
    m = manifest(reference)
    assert [sorted(g) for g in m["generations"]] == [["gate", "gen", "selfplay", "train"]] * 2
    g0, g1 = m["generations"]
    assert g0["selfplay"]["bot"] == "ismcts" and g1["selfplay"]["bot"] == "net:gen-000"
    assert g0["selfplay"]["belief_resets"] == 0 and g1["selfplay"]["belief_resets"] == 0
    assert g0["gate"]["promoted"] is True
    assert (g0["train"]["init"], g1["train"]["init"]) == (None, "gen-000")
    assert (g0["train"]["lr"], g1["train"]["lr"]) == (1e-3, 3e-4)
    assert g1["train"]["window"] == ["gen-000", "gen-001"]
    assert g1["train"]["parity_max_error"] <= 1e-5
    assert g1["gate"]["games"] == 4 and g1["gate"]["promoted"] == promoted(g1["gate"]["wins"], 4)
    assert g1["gate"]["belief_resets"] == 0
    assert m["best"] == (1 if g1["gate"]["promoted"] else 0)
    nets = reference / "nets"
    assert (nets / "best.bin").read_bytes() == (nets / f"gen-00{m['best']}.bin").read_bytes()
    for k, g in enumerate(m["generations"]):
        gdir = reference / f"gen-00{k}"
        assert g["train"]["net"] == sha256(gdir / "net.bin") == sha256(nets / f"gen-00{k}.bin")
        assert g["selfplay"]["chunks"] == {name: sha256(gdir / name)
                                           for name in ("selfplay-local-00000.jsonl.gz", "selfplay-local-00001.jsonl.gz")}
    assert temps(reference) == []
    assert Loop(reference, TINY).run() == 0, "nothing is redone"


def test_stopping_after_each_step_and_resuming_gives_the_same_manifest(tmp_path, reference):
    executed = []
    while (n := Loop(tmp_path, TINY).run(stop_after=1)) > 0:
        executed.append(n)
    assert executed == [1] * 6
    assert manifest(tmp_path) == manifest(reference)


@pytest.mark.parametrize("interrupt_at", [2, 5, 9, 17])
def test_an_interrupt_during_a_rename_resumes_to_the_same_manifest(tmp_path, reference, monkeypatch, interrupt_at):
    # os.replace calls in order: 1 gen-000/selfplay.json, 2-3 its chunks, 4 manifest, 5 gen-000/net.bin,
    # 6 nets/gen-000.bin, 7 manifest, 8 nets/best.bin, 9 manifest, ..., 17 gen-001/gate.jsonl.
    real = os.replace
    calls = {"n": 0}

    def flaky(src, dst):
        calls["n"] += 1
        if calls["n"] == interrupt_at:
            raise KeyboardInterrupt
        return real(src, dst)

    monkeypatch.setattr(os, "replace", flaky)
    with pytest.raises(KeyboardInterrupt):
        Loop(tmp_path, TINY).run()
    monkeypatch.setattr(os, "replace", real)
    assert temps(tmp_path), "the interrupted write left its temp file"
    Loop(tmp_path, TINY).run()
    assert manifest(tmp_path) == manifest(reference)
    assert temps(tmp_path) == []


def test_other_settings_on_a_started_self_play_step_are_an_error(tmp_path, monkeypatch):
    real = os.replace
    calls = {"n": 0}

    def flaky(src, dst):
        calls["n"] += 1
        if calls["n"] == 2:
            raise KeyboardInterrupt
        return real(src, dst)

    monkeypatch.setattr(os, "replace", flaky)
    with pytest.raises(KeyboardInterrupt):
        Loop(tmp_path, TINY).run()
    monkeypatch.setattr(os, "replace", real)
    with pytest.raises(ValueError, match="other settings"):
        Loop(tmp_path, replace(TINY, full_simulations=6)).run()


def test_new_flags_apply_only_to_steps_not_yet_run(tmp_path, reference):
    Loop(tmp_path, replace(TINY, generations=0)).run()
    Loop(tmp_path, replace(TINY, lr=1e-4)).run()
    m, ref = manifest(tmp_path), manifest(reference)
    assert m["generations"][0] == ref["generations"][0]
    assert m["generations"][1]["train"]["lr"] == 1e-4


def cli(*args):
    return subprocess.run([sys.executable, "-m", "alphasettler", *args], capture_output=True, text=True)


def test_cli_train_loop(tmp_path):
    args = ["train-loop", "--dir", str(tmp_path), "--generations", "0", "--warm-games", "2", "--chunk-games", "1",
            "--full-simulations", "4", "--fast-simulations", "2", "--batch", "16", "--trunk", "16",
            "--value-width", "8", "--max-turns", "60", "--device", "cpu", "--threads", "2"]
    r = cli(*args)
    assert r.returncode == 0, r.stderr
    assert "3 steps run" in r.stdout and "best net: gen-000" in r.stdout
    assert (tmp_path / "nets" / "best.bin").exists()
    r = cli(*args)
    assert r.returncode == 0 and "nothing to do" in r.stdout
    r = cli("train-loop", "--dir", str(tmp_path / "x"), "--generations", "-1")
    assert r.returncode == 2 and "--generations" in r.stderr
    r = cli("train-loop", "--dir", str(tmp_path / "y"), "--generations", "0", "--full-probability", "2")
    assert r.returncode == 2 and "probability" in r.stderr
    assert not (tmp_path / "y").exists()
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/bin/pytest -q tests/python/test_train_loop.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'alphasettler.train.loop'`.

- [ ] **Step 3: Write the loop**

Create `alphasettler/train/loop.py`:

```python
"""The generation loop (spec Section 4): for each generation, self-play, train, gate.

`<dir>/gen-NNN/` holds a generation's records (`selfplay-local-NNNNN.jsonl.gz`, plus any record
files other machines drop in before training), its weights (`net.bin`) and its gating result
(`gate.jsonl`, `gate.json`). `<dir>/nets/` holds every generation's weights as `gen-NNN.bin` and
the best one as `best.bin`, so `net:gen-NNN` and `net:best` resolve with `--nets-dir <dir>/nets`.
`<dir>/manifest.json` lists the completed steps with their outputs' hashes, and a rerun resumes
at the first incomplete step. Every output is written under a temporary name and renamed when
complete; a step deletes the leftovers of its own interrupted writes before it reruns.

Generation 0 is the warm start: 3a's `ismcts` self-plays with the same exploration, net 0 is
trained from scratch and becomes the best net without a match. Generation k >= 1 self-plays with
the best net, trains from generation k-1's weights and is gated against the best net."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from alphasettler import files, records
from alphasettler import selfplay as selfplay_mod
from alphasettler.arena import run as arena_run

MANIFEST = "manifest.json"
MANIFEST_FORMAT = 1
STEPS = ("selfplay", "train", "gate")
SEEDS_PER_GENERATION = 10_000_000
GATE_SEED_START = 1_000_000_000


@dataclass(frozen=True)
class LoopSettings:
    generations: int
    games: int = 10_000
    warm_games: int = 20_000
    chunk_games: int = 500
    full_simulations: int = 600
    fast_simulations: int = 100
    full_probability: float = 0.25
    window: int = 4
    warm_lr: float = 1e-3
    lr: float = 3e-4
    steps_factor: float = 2.0
    batch: int = 1024
    trunk: int = 512
    value: int = 256
    gate_seeds: int = 200
    gate_simulations: int = 600
    max_turns: int = 1000
    device: str = "auto"
    threads: int | None = None


def gen_name(k: int) -> str:
    return f"gen-{k:03d}"


def promoted(wins: int, games: int) -> bool:
    """A win rate of at least 1/4 against three copies of the best net promotes; a tie at
    exactly 0.25 promotes. Draws count as losses."""
    return 4 * wins >= games


class Loop:
    def __init__(self, root: str | Path, settings: LoopSettings):
        if settings.generations < 0:
            raise ValueError(f"--generations must be 0 or more, got {settings.generations}")
        # Check the self-play settings now, not hours into a run (no net is loaded here).
        selfplay_mod.settings(None, ".", settings.full_simulations, settings.fast_simulations,
                              settings.full_probability)
        self.root = Path(root)
        self.s = settings
        self.nets = self.root / "nets"
        self.config = {"max_offers_per_turn": 0, "max_turns": settings.max_turns}
        self.manifest = self._load()

    def _load(self) -> dict:
        path = self.root / MANIFEST
        if not path.exists():
            return {"format": MANIFEST_FORMAT, "best": None, "generations": []}
        m = json.loads(path.read_text())
        if m.get("format") != MANIFEST_FORMAT:
            raise ValueError(f"{path} has format {m.get('format')!r}, expected {MANIFEST_FORMAT}")
        return m

    def _save(self) -> None:
        files.atomic_write_text(self.root / MANIFEST, json.dumps(self.manifest, indent=1, sort_keys=True) + "\n")

    def _entry(self, k: int) -> dict:
        gens = self.manifest["generations"]
        while len(gens) <= k:
            gens.append({"gen": len(gens)})
        return gens[k]

    def run(self, stop_after: int | None = None) -> int:
        """Run every incomplete step of generations 0..=`generations` in order and return how
        many ran. With `stop_after`, stop once that many have run."""
        self.root.mkdir(parents=True, exist_ok=True)
        files.remove_temps(self.root, MANIFEST)
        done = 0
        for k in range(self.s.generations + 1):
            for step in STEPS:
                if step in self._entry(k):
                    continue
                if stop_after is not None and done >= stop_after:
                    return done
                print(f"{gen_name(k)}: {step}", flush=True)
                result = getattr(self, f"_{step}")(k)
                self._entry(k)[step] = result
                if step == "gate" and result["promoted"]:
                    self.manifest["best"] = k
                self._save()
                done += 1
        return done

    def _selfplay(self, k: int) -> dict:
        gdir = self.root / gen_name(k)
        gdir.mkdir(parents=True, exist_ok=True)
        files.remove_temps(gdir, "selfplay.json")
        files.remove_temps(gdir, "selfplay-local-*")
        net = None if k == 0 else gen_name(self.manifest["best"])
        games = self.s.warm_games if k == 0 else self.s.games
        plan = {
            "bot": "ismcts" if net is None else f"net:{net}",
            "seed_start": k * SEEDS_PER_GENERATION,
            "games": games,
            "chunk_games": self.s.chunk_games,
            "full_simulations": self.s.full_simulations,
            "fast_simulations": self.s.fast_simulations,
            "full_probability": self.s.full_probability,
            "config": self.config,
        }
        plan_path = gdir / "selfplay.json"
        if plan_path.exists():
            if json.loads(plan_path.read_text()) != plan:
                raise ValueError(f"{gdir.name}'s self-play was started with other settings ({plan_path}); "
                                 f"delete {gdir} to start it again")
        else:
            files.atomic_write_text(plan_path, json.dumps(plan, sort_keys=True) + "\n")
        start, end = plan["seed_start"], plan["seed_start"] + games
        totals = {"games": 0, "decisions": 0, "full_decisions": 0, "belief_resets": 0}
        chunks = {}
        for i, lo in enumerate(range(start, end, self.s.chunk_games)):
            path = gdir / f"selfplay-local-{i:05d}.jsonl.gz"
            if not path.exists():
                tmp = files.temp_path(path)
                selfplay_mod.write_explore(
                    tmp, lo, min(self.s.chunk_games, end - lo), config=self.config,
                    full_simulations=self.s.full_simulations, fast_simulations=self.s.fast_simulations,
                    full_probability=self.s.full_probability, net=net, nets_dir=self.nets, threads=self.s.threads)
                os.replace(tmp, path)
                print(f"  {path.name}: seeds {lo}..{min(end, lo + self.s.chunk_games)}", flush=True)
            for g in records.read(path):
                totals["games"] += 1
                totals["decisions"] += len(g["decisions"])
                totals["full_decisions"] += sum(d["full_search"] for d in g["decisions"])
                totals["belief_resets"] += g["belief_resets"]
            chunks[path.name] = files.sha256_file(path)
        print(f"  {totals['games']} games, {totals['decisions']} decisions, belief resets {totals['belief_resets']}",
              flush=True)
        return {**plan, **totals, "chunks": chunks}

    def _train(self, k: int) -> dict:
        from alphasettler.train.data import load_window
        from alphasettler.train.train import TrainConfig, train

        dirs = [self.root / gen_name(j) for j in range(max(0, k - self.s.window + 1), k + 1)]
        window = load_window(self.root, dirs)
        for line in window.notices:
            print(f"notice: {line}", flush=True)
        gdir = self.root / gen_name(k)
        files.remove_temps(gdir, "net.bin")
        cfg = TrainConfig(lr=self.s.warm_lr if k == 0 else self.s.lr, steps_factor=self.s.steps_factor,
                          batch=self.s.batch, seed=k, device=self.s.device, threads=self.s.threads,
                          trunk=self.s.trunk, value=self.s.value)
        init = None if k == 0 else self.root / gen_name(k - 1) / "net.bin"
        summary = train(window, gdir / "net.bin", cfg, init)
        self.nets.mkdir(parents=True, exist_ok=True)
        files.remove_temps(self.nets, f"{gen_name(k)}.bin")
        files.atomic_copy(gdir / "net.bin", self.nets / f"{gen_name(k)}.bin")
        return {
            "window": [d.name for d in dirs],
            "inputs": window.files,
            "games": window.data.games,
            "positions": window.data.decisions,
            "new_positions": window.new_positions,
            "init": None if k == 0 else gen_name(k - 1),
            "lr": cfg.lr,
            "steps": summary["steps"],
            "batch": cfg.batch,
            "device": summary["device"],
            "history": summary["history"],
            "parity_max_error": summary["parity_max_error"],
            "net": files.sha256_file(gdir / "net.bin"),
        }

    def _gate(self, k: int) -> dict:
        self.nets.mkdir(parents=True, exist_ok=True)
        files.remove_temps(self.nets, "best.bin")
        if k == 0:
            files.atomic_copy(self.nets / f"{gen_name(0)}.bin", self.nets / "best.bin")
            return {"promoted": True, "reason": "warm start: net 0 becomes the best net"}
        best = self.manifest["best"]
        gdir = self.root / gen_name(k)
        files.remove_temps(gdir, "gate.jsonl")
        files.remove_temps(gdir, "gate.json")
        sims = self.s.gate_simulations
        candidate, baseline = f"net:{gen_name(k)}@{sims}", f"net:{gen_name(best)}@{sims}"
        tmp = files.temp_path(gdir / "gate.jsonl")
        recs, summary = arena_run(candidate, baseline, self.s.gate_seeds, GATE_SEED_START, self.s.threads,
                                  self.config, tmp, self.nets)
        os.replace(tmp, gdir / "gate.jsonl")
        result = {
            "candidate": candidate,
            "baseline": baseline,
            "seeds": self.s.gate_seeds,
            "seed_start": GATE_SEED_START,
            "wins": summary.wins,
            "games": summary.games,
            "draws": summary.draws,
            "win_rate": round(summary.win_rate, 6),
            "z": round(summary.z_vs_null, 6),
            "belief_resets": sum(r["belief_resets"] for r in recs),
            "promoted": promoted(summary.wins, summary.games),
            "records": files.sha256_file(gdir / "gate.jsonl"),
        }
        files.atomic_write_text(gdir / "gate.json", json.dumps(result, indent=1, sort_keys=True) + "\n")
        if result["promoted"]:
            files.atomic_copy(self.nets / f"{gen_name(k)}.bin", self.nets / "best.bin")
        verdict = "promoted" if result["promoted"] else "not promoted; the best net stays"
        print(f"  {candidate} vs {baseline}: {summary.wins} wins of {summary.games} "
              f"(win rate {summary.win_rate:.3f}, z {summary.z_vs_null:.2f}); {verdict}", flush=True)
        return result
```

- [ ] **Step 4: The `train-loop` command**

In `alphasettler/cli.py`, add to `_parser` before `return p`:

```python
    lp = sub.add_parser("train-loop", help="self-play, train and gate generations; rerun to resume (needs the train extra)")
    lp.add_argument("--dir", required=True)
    lp.add_argument("--generations", type=int, required=True,
                    help="learned generations after the warm start (0: the warm start only)")
    lp.add_argument("--games", type=_positive_int, default=10_000, help="self-play games per learned generation")
    lp.add_argument("--warm-games", type=_positive_int, default=20_000)
    lp.add_argument("--chunk-games", type=_positive_int, default=500)
    lp.add_argument("--full-simulations", type=_positive_int, default=600)
    lp.add_argument("--fast-simulations", type=_positive_int, default=100)
    lp.add_argument("--full-probability", type=float, default=0.25)
    lp.add_argument("--window", type=_positive_int, default=4)
    lp.add_argument("--warm-lr", type=float, default=1e-3)
    lp.add_argument("--lr", type=float, default=3e-4)
    lp.add_argument("--steps-factor", type=float, default=2.0)
    lp.add_argument("--batch", type=_positive_int, default=1024)
    lp.add_argument("--trunk", type=_positive_int, default=512)
    lp.add_argument("--value-width", type=_positive_int, default=256)
    lp.add_argument("--gate-seeds", type=_positive_int, default=200)
    lp.add_argument("--gate-simulations", type=_positive_int, default=600)
    lp.add_argument("--max-turns", type=_positive_int, default=1000)
    lp.add_argument("--device", default="auto", choices=["auto", "mps", "cpu"])
    lp.add_argument("--threads", type=_positive_int, default=None)
```

Add the handler after `_train`:

```python
def _train_loop(args) -> int:
    from alphasettler.train.loop import Loop, LoopSettings, gen_name

    s = LoopSettings(
        generations=args.generations, games=args.games, warm_games=args.warm_games, chunk_games=args.chunk_games,
        full_simulations=args.full_simulations, fast_simulations=args.fast_simulations,
        full_probability=args.full_probability, window=args.window, warm_lr=args.warm_lr, lr=args.lr,
        steps_factor=args.steps_factor, batch=args.batch, trunk=args.trunk, value=args.value_width,
        gate_seeds=args.gate_seeds, gate_simulations=args.gate_simulations, max_turns=args.max_turns,
        device=args.device, threads=args.threads,
    )
    loop = Loop(Path(args.dir), s)
    try:
        n = loop.run()
    except KeyboardInterrupt:
        print("interrupted: rerun the same command to resume at the first incomplete step", file=sys.stderr)
        return 130
    print(f"{n} steps run" if n else "nothing to do: every step is complete")
    best = loop.manifest["best"]
    if best is not None:
        print(f"best net: {gen_name(best)} ({loop.nets / 'best.bin'})")
    return 0
```

In `main`, add `elif args.command == "train-loop": return _train_loop(args)` after the `train` branch.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/bin/pytest -q tests/python/test_train_loop.py`
Expected: 10 passed (about a minute: each tiny loop runs two generations).
Run: `.venv/bin/pytest -q -rs`
Expected: all pass, none skipped.

- [ ] **Step 6: Format, lint, full checks**

Run: `~/.cargo/bin/cargo fmt --check`
Expected: no output.
Run: `~/.cargo/bin/cargo clippy --workspace --release --all-targets -- -D warnings`
Expected: exits 0.
Run: `~/.cargo/bin/cargo test --workspace --release`
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add alphasettler/train/loop.py alphasettler/cli.py tests/python/test_train_loop.py
git diff --cached | grep -nE 'sk-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{36}|password=[^[:space:]]+'
git -c user.email=sean.may101@gmail.com commit -m "train: the resumable generation loop with manifest, atomic outputs and gating"
```

---

### Task 8: Warm start, net search throughput and the batch sweep

**Files:**
- Create: `bots/examples/net_speed.rs`, `docs/perf/net-search.md`
- Modify: `bots/src/netbot.rs` (`NET_BATCH` only, Step 7)

**Interfaces:**
- Consumes: `IsmctsBot::{with_evaluator, evaluator, take_last_search}`, `NetEvaluator::{new, evaluations}`, `Weights::{load, random}`, `Arch::DEFAULT` (Tasks 2–3); `arena::{play_game, run_match_by, GameRecord}` (Task 3); every CLI from Tasks 3–7.
- Produces: the measured default `NET_BATCH`; `docs/perf/net-search.md` with the line `Self-play games per generation: N` that Task 9 reads.

Steps 1–2 are an implementer's; Steps 3–8 are the controller's (long runs in the background).

- [ ] **Step 1: Write the speed and sweep example**

Create `bots/examples/net_speed.rs`:

```rust
//! Network search throughput and the leaf-batch sweep (spec Sections 3 and 5).
//!
//!   net_speed speed PATH DECISIONS SIMULATIONS BATCH
//!       one net bot (seat 0) against three GreedyBots on one core, timing only decisions that
//!       searched: simulations/s, network evaluations/s and ms per decision
//!   net_speed paired PATH SEEDS SIMULATIONS BASELINE
//!       the net bot at batch 8, 16 and 32 against three BASELINE bots on the same seeds (all
//!       cores, trades off), then each larger batch's paired difference from batch 8
//!
//! PATH is a weights file, or `random` for random weights of the default shape.

use settler_bots::arena::{play_game, run_match_by, GameRecord};
use settler_bots::greedy::GreedyBot;
use settler_bots::ismcts::IsmctsBot;
use settler_bots::{make_bot, Bot};
use settler_engine::{Action, Event, GameConfig, Observation, PlayerId};
use settler_net::evaluator::NetEvaluator;
use settler_net::weights::{Arch, Weights};
use settler_search::SearchConfig;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

fn weights(path: &str) -> Arc<Weights> {
    Arc::new(if path == "random" {
        Weights::random(Arch::DEFAULT, 1)
    } else {
        Weights::load(Path::new(path)).unwrap_or_else(|e| panic!("{e}"))
    })
}

fn net_bot(w: &Arc<Weights>, seed: u64, simulations: u32, batch: usize) -> IsmctsBot<NetEvaluator> {
    let cfg = SearchConfig {
        simulations,
        batch,
        ..SearchConfig::default()
    };
    IsmctsBot::with_evaluator("net", seed, cfg, NetEvaluator::new(w.clone()))
}

fn no_trades() -> GameConfig {
    GameConfig {
        max_offers_per_turn: 0,
        ..GameConfig::default()
    }
}

#[derive(Default)]
struct Totals {
    decisions: u64,
    simulations: u64,
    evaluations: u64,
    time: Duration,
}

struct Timed {
    inner: IsmctsBot<NetEvaluator>,
    totals: Arc<Mutex<Totals>>,
}

impl Bot for Timed {
    fn name(&self) -> &'static str {
        "timed-net"
    }

    fn observe(&mut self, viewer: PlayerId, events: &[Event]) {
        self.inner.observe(viewer, events);
    }

    fn act(&mut self, obs: &Observation, legal: &[Action]) -> Action {
        let before = self.inner.evaluator().evaluations();
        let start = Instant::now();
        let a = self.inner.act(obs, legal);
        let elapsed = start.elapsed();
        if let Some(r) = self.inner.take_last_search() {
            let mut t = self.totals.lock().unwrap();
            t.decisions += 1;
            t.simulations += r.simulations as u64;
            t.evaluations += self.inner.evaluator().evaluations() - before;
            t.time += elapsed;
        }
        a
    }
}

fn speed(path: &str, decisions: u64, simulations: u32, batch: usize) {
    let w = weights(path);
    let totals = Arc::new(Mutex::new(Totals::default()));
    let mut seed = 0;
    while totals.lock().unwrap().decisions < decisions {
        let mut bots: Vec<Box<dyn Bot>> = vec![Box::new(Timed {
            inner: net_bot(&w, seed, simulations, batch),
            totals: totals.clone(),
        })];
        bots.extend((1..4).map(|p| Box::new(GreedyBot::new(p)) as Box<dyn Bot>));
        play_game(seed, no_trades(), &mut bots);
        seed += 1;
    }
    let t = totals.lock().unwrap();
    let secs = t.time.as_secs_f64();
    println!(
        "batch {batch}: {} searched decisions over {seed} games at {simulations} simulations: {:.0} simulations/s, {:.0} evaluations/s, {:.2} ms/decision",
        t.decisions,
        t.simulations as f64 / secs,
        t.evaluations as f64 / secs,
        1000.0 * secs / t.decisions as f64
    );
}

/// Per-seed candidate win rates; records are sorted by (seed, seat), four per seed.
fn per_seed(records: &[GameRecord]) -> Vec<f64> {
    records
        .chunks(4)
        .map(|c| c.iter().filter(|r| r.winner == Some(r.candidate_seat)).count() as f64 / 4.0)
        .collect()
}

fn mean_se(xs: &[f64]) -> (f64, f64) {
    let n = xs.len() as f64;
    let mean = xs.iter().sum::<f64>() / n;
    let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0).max(1.0);
    (mean, (var / n).sqrt())
}

fn paired(path: &str, seeds: u64, simulations: u32, baseline: &str) {
    let w = weights(path);
    assert!(
        make_bot(baseline, 0).is_some(),
        "unknown baseline {baseline:?} (net: names are not supported here)"
    );
    let threads = std::thread::available_parallelism().map_or(1, |c| c.get());
    let mut rates: Vec<(usize, Vec<f64>)> = Vec::new();
    for batch in [8usize, 16, 32] {
        let start = Instant::now();
        let records = run_match_by(0..seeds, no_trades(), threads, &|is_candidate, seed| {
            if is_candidate {
                Box::new(net_bot(&w, seed, simulations, batch)) as Box<dyn Bot>
            } else {
                make_bot(baseline, seed).expect("checked above")
            }
        })
        .unwrap_or_else(|e| panic!("{e}"));
        let r = per_seed(&records);
        let (mean, se) = mean_se(&r);
        let resets: u64 = records.iter().map(|x| x.belief_resets).sum();
        println!(
            "batch {batch} at {simulations} simulations vs three {baseline}: win rate {mean:.3} ± {se:.3} (z vs 0.25 = {:.2}) over {seeds} seeds, belief resets {resets}, {:.0} s",
            (mean - 0.25) / se,
            start.elapsed().as_secs_f64()
        );
        rates.push((batch, r));
    }
    for (batch, r) in &rates[1..] {
        let diffs: Vec<f64> = r.iter().zip(&rates[0].1).map(|(a, b)| a - b).collect();
        let (mean, se) = mean_se(&diffs);
        let z = if se > 0.0 { mean / se } else { 0.0 };
        println!("paired batch {batch} - batch 8: mean diff {mean:+.3} ± {se:.3} (z = {z:.2}) over {seeds} seeds");
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let num = |i: usize| -> u64 {
        args.get(i)
            .and_then(|a| a.parse().ok())
            .unwrap_or_else(|| panic!("argument {i} must be a number"))
    };
    match args.first().map(String::as_str) {
        Some("speed") if args.len() == 5 => speed(&args[1], num(2), num(3) as u32, num(4) as usize),
        Some("paired") if args.len() == 5 => paired(&args[1], num(2), num(3) as u32, &args[4]),
        _ => {
            eprintln!("usage: net_speed speed PATH DECISIONS SIMULATIONS BATCH | net_speed paired PATH SEEDS SIMULATIONS BASELINE");
            std::process::exit(2)
        }
    }
}
```

- [ ] **Step 2: Smoke-run it and check formatting**

Run: `~/.cargo/bin/cargo run --release -p settler-bots --example net_speed -- speed random 40 200 8`
Expected: one line starting `batch 8: ` with simulations/s and evaluations/s.
Run: `~/.cargo/bin/cargo run --release -p settler-bots --example net_speed -- paired random 2 20 greedy`
Expected: three `batch N at 20 simulations vs three greedy:` lines, each with `belief resets 0`, then two `paired batch` lines.
Run: `~/.cargo/bin/cargo fmt --all && ~/.cargo/bin/cargo fmt --check && ~/.cargo/bin/cargo clippy --workspace --release --all-targets -- -D warnings`
Expected: no fmt output; clippy exits 0.

- [ ] **Step 3: The warm start (controller runs this)**

```bash
mkdir -p runs/3b
.venv/bin/alphasettler train-loop --dir runs/3b --generations 0 --threads 10 2>&1 | tee -a runs/3b/train-loop.log
```

This is generation 0: 20,000 `ismcts` self-play games at 600/100 simulations with root noise and opening sampling (minutes), then net₀ trained from scratch on MPS at lr 1e-3 for 2 × positions / 1,024 steps. Rerun the same command after any interruption; it resumes.
Pass: exit 0 and `3 steps run`; in `runs/3b/manifest.json`, generation 0 has `selfplay.games == 20000`, `selfplay.belief_resets == 0`, `train.device == "mps"`, `train.parity_max_error <= 1e-5` and `gate.promoted == true`; `runs/3b/nets/gen-000.bin` and `runs/3b/nets/best.bin` exist. Any `notice:` line in the log is investigated before going on.

- [ ] **Step 4: Throughput (controller runs this)**

On an otherwise idle machine:

```bash
for b in 8 16 32; do for i in 1 2 3; do ~/.cargo/bin/cargo run --release -p settler-bots --example net_speed -- speed runs/3b/nets/gen-000.bin 500 1000 $b; done; done 2>&1 | tee runs/3b/net-speed.log
```

Pass: nine `batch N:` lines (about 20 minutes in all).

- [ ] **Step 5: The paired batch match (controller runs this)**

```bash
~/.cargo/bin/cargo run --release -p settler-bots --example net_speed -- paired runs/3b/nets/gen-000.bin 200 1000 ismcts@1000 2>&1 | tee runs/3b/net-batch-paired.log
```

Pass: three `batch N at 1000 simulations vs three ismcts@1000:` lines, each with `belief resets 0`, and two `paired batch` lines (about an hour).

- [ ] **Step 6: Choose the default batch**

The rule: a batch B in {16, 32} qualifies if the mean of its three simulations/s readings is at least 10% above batch 8's mean **and** its paired difference from batch 8 has z > −2. The default is the qualifying batch with the higher simulations/s; if neither qualifies, it stays 8. (Budgets are simulations, so a larger batch only pays for itself in throughput, and only if it does not weaken play.)

- [ ] **Step 7: Set `NET_BATCH` and recheck**

If Step 6 chose 16 or 32, edit the one line in `bots/src/netbot.rs`:

```rust
pub const NET_BATCH: usize = 16;
```

(with the chosen value; leave it at 8 otherwise), and update its doc comment to `/// Leaves per network call, chosen by the batch sweep in docs/perf/net-search.md.`

Run: `~/.cargo/bin/cargo test --workspace --release && PATH=$HOME/.cargo/bin:$PATH .venv/bin/maturin develop --release -q && .venv/bin/pytest -q -rs`
Expected: all pass, none skipped.

- [ ] **Step 8: Self-play speed and the generation size (controller runs this)**

```bash
/usr/bin/time -p .venv/bin/alphasettler selfplay --net gen-000 --nets-dir runs/3b/nets --games 200 --seed-start 900000000 --threads 10 --out-dir runs/3b/speed 2>&1 | tee runs/3b/selfplay-speed.log
```

Pass: `200 games`, `belief resets: 0` and a `real R` line. Then the games per generation are G = max(1000, floor(21600 × 200 / R / 100) × 100): six hours of self-play at net₀'s speed, in hundreds. (Each generation's gate adds roughly 800 games at 600 simulations in all four seats, about 2.7 times a self-play game each, and training adds well under an hour; together they keep a generation inside one night.)

Then the warm-start check (spec done criterion 3, recorded, not a gate):

```bash
.venv/bin/alphasettler arena --candidate net:gen-000 --baseline ismcts@1000 --seeds 200 --no-trades --nets-dir runs/3b/nets --out runs/3b/warm-start-vs-ismcts.jsonl 2>&1 | tee runs/3b/warm-start.log
```

Pass: a summary line and `belief resets: 0`.

- [ ] **Step 9: Write `docs/perf/net-search.md`**

Record, with every number copied from the logs above (nothing estimated):
- machine (`sysctl -n machdep.cpu.brand_string`), commit (`git log -1 --format=%h`, or "uncommitted on top of <hash>"), net (`gen-000`, its sha256 from the manifest), config (trades off, exact belief with 65,536 states, c_puct 1.5, 1,000 simulations, one core for `speed`, all ten cores for `paired`);
- a table of the nine `speed` lines (simulations/s, evaluations/s, ms/decision) and each batch's mean, beside 3a's heuristic 700k simulations/s per core (`docs/perf/search.md`);
- the three `paired` summary lines and the two paired differences, then the rule from Step 6 applied in one sentence and the chosen `NET_BATCH`;
- the self-play timing (`real R` for 200 games on 10 threads) and, on its own line, exactly `Self-play games per generation: G` with the G from Step 8.

- [ ] **Step 10: Commit**

Run: `~/.cargo/bin/cargo fmt --check && ~/.cargo/bin/cargo clippy --workspace --release --all-targets -- -D warnings`
Expected: no fmt output; clippy exits 0.

```bash
git add bots/examples/net_speed.rs bots/src/netbot.rs docs/perf/net-search.md
git diff --cached | grep -nE 'sk-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{36}|password=[^[:space:]]+'
git -c user.email=sean.may101@gmail.com commit -m "perf: net search throughput, the leaf-batch sweep and the default batch"
```

---

### Task 9: Generations, the headline runs and the results record

**Files:**
- Create: `docs/bot/results-3b.md`

**Interfaces:**
- Consumes: `alphasettler train-loop` (Task 7), `alphasettler arena` with `--nets-dir` (Task 3), `docs/perf/net-search.md`'s `Self-play games per generation:` line (Task 8).

Every step below except Step 6 is run by the controller in the background. Each `train-loop` command resumes where it stopped: after a Ctrl-C, a crash or a reboot, rerun the same command.

- [ ] **Step 1: Generations 1 to 5 (controller runs this)**

```bash
GAMES=$(grep -oE 'Self-play games per generation: [0-9]+' docs/perf/net-search.md | grep -oE '[0-9]+$')
.venv/bin/alphasettler train-loop --dir runs/3b --generations 5 --games "$GAMES" --threads 10 2>&1 | tee -a runs/3b/train-loop.log
```

About one night per generation. Pass: exit 0; `runs/3b/manifest.json` has generations 0–5 each with `selfplay`, `train` and `gate`; every `selfplay.belief_resets` and `gate.belief_resets` is 0; every `train.parity_max_error` is at most 1e-5 and every `train.device` is `mps`. A generation that is not promoted is not a failure (its data stays in the window).

- [ ] **Step 2: The headline runs (controller runs this)**

```bash
.venv/bin/alphasettler arena --candidate net:best --baseline ismcts@1000 --seeds 200 --no-trades --nets-dir runs/3b/nets --out runs/3b/headline-vs-ismcts-$(date +%Y%m%d-%H%M).jsonl 2>&1 | tee -a runs/3b/headline.log
.venv/bin/alphasettler arena --candidate net:best --baseline catanatron:alphabeta --seeds 200 --nets-dir runs/3b/nets --out runs/3b/headline-vs-alphabeta-$(date +%Y%m%d-%H%M).jsonl 2>&1 | tee -a runs/3b/headline.log
```

Note which generation `net:best` is (`best` in the manifest) beside each result. Pass for the runs themselves: `belief resets: 0` on both and `fallbacks: 0` on the Catanatron run. **The headline is met** when the first prints z > 3 (against 0.25) **and** the second prints a win rate above 0.383 with z > 3. If it is met, skip to Step 4.

- [ ] **Step 3: Up to three extra generations (controller runs this)**

Only if Step 2 missed. One generation at a time, each changing only training hyperparameters, each followed by both Step 2 commands; stop at the first that meets the headline:

```bash
GAMES=$(grep -oE 'Self-play games per generation: [0-9]+' docs/perf/net-search.md | grep -oE '[0-9]+$')
# generation 6: a lower learning rate
.venv/bin/alphasettler train-loop --dir runs/3b --generations 6 --games "$GAMES" --threads 10 --lr 1e-4 2>&1 | tee -a runs/3b/train-loop.log
# generation 7: also a wider window and more steps
.venv/bin/alphasettler train-loop --dir runs/3b --generations 7 --games "$GAMES" --threads 10 --lr 1e-4 --window 6 --steps-factor 3 2>&1 | tee -a runs/3b/train-loop.log
# generation 8: also larger simulation caps
.venv/bin/alphasettler train-loop --dir runs/3b --generations 8 --games "$GAMES" --threads 10 --lr 1e-4 --window 6 --steps-factor 3 --full-simulations 800 --fast-simulations 150 2>&1 | tee -a runs/3b/train-loop.log
```

Pass conditions as in Step 1. If the headline is still missed after generation 8, record the gap in Step 5 and go on; 3c follows either way.

- [ ] **Step 4: The strength curve (controller runs this)**

```bash
for n in 100 300 1000 3000; do .venv/bin/alphasettler arena --candidate net:best@$n --baseline greedy --seeds 200 --no-trades --nets-dir runs/3b/nets --out runs/3b/curve-$n.jsonl; done 2>&1 | tee runs/3b/curve.log
```

Pass: four summary lines, each with `belief resets: 0`.

- [ ] **Step 5: Write `docs/bot/results-3b.md`**

Record, with every number copied from the logs and the manifest (no placeholders):
- date, machine, Python and rustc versions, `git log -1 --format=%h` (or "uncommitted on top of <hash>"), PyTorch 2.14.1, Catanatron 3.3.0 at ecf931181b9a65bb4116a2153fb78c16f1438e00; a link to the spec;
- search speed with the net (link `docs/perf/net-search.md`) and the chosen `NET_BATCH`;
- a table of every generation: games, decisions, full searches, training steps, final losses (policy, value, aux), parity error, gate wins / games / win rate / z, promoted or not, and the best net after it; the hyperparameters of any extra generation;
- the warm-start check (`runs/3b/warm-start.log`): net₀ at 1,000 simulations vs three `ismcts@1000`;
- the headline runs (every attempt, with the generation of `net:best`, wall time, fallbacks and belief resets) and whether the headline is met; if not, the gap (win rate and z for each) and its suspected cause read from the runs, for example the training curves in the manifest or which phases the net's moves differ from `ismcts`'s;
- the strength curve table (100 / 300 / 1,000 / 3,000 simulations vs greedy) beside 3a's (0.330, 0.388, 0.443, 0.435), and one sentence on whether 3a's plateau is gone;
- a checklist of the spec's Section 5 done criteria, each with its evidence (test names, commands, this file's numbers): the encoder golden and rotation tests (`net/tests/encode.rs`), prior invariance (`net/tests/evaluator.rs`), parity (`tests/python/test_train.py::test_rust_and_pytorch_forward_passes_agree_on_random_weights`), the weights-file errors (`net/tests/weights.rs`, `test_weights_round_trip_and_bad_files_are_errors`), the training-data tests (`net/tests/data.rs`, `tests/python/test_training_data.py`, `test_fast_searches_add_nothing_to_the_policy_loss`, `test_a_tiny_dataset_can_be_overfit`), the self-play tests (`bots/tests/selfplay.rs`), the loop tests (`tests/python/test_train_loop.py`), the gating rule, and the name tests in all three places (`bots/tests/netbot.rs`, `tests/python/test_netbot.py`, `test_net_names_inside_catanatron`).

- [ ] **Step 6: Final checks and commit**

Run: `~/.cargo/bin/cargo test --workspace --release`
Expected: all pass, `golden_trace` included.
Run: `git diff --stat 6842e96 -- engine/`
Expected: no output (the engine is unchanged since this plan started).
Run: `PATH=$HOME/.cargo/bin:$PATH .venv/bin/maturin develop --release -q && .venv/bin/pytest -q -rs`
Expected: all pass, the summary lists no skipped tests.
Run: `~/.cargo/bin/cargo fmt --check && ~/.cargo/bin/cargo clippy --workspace --release --all-targets -- -D warnings`
Expected: no fmt output; clippy exits 0.

```bash
git add docs/bot/results-3b.md
git diff --cached | grep -nE 'sk-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{36}|password=[^[:space:]]+'
git -c user.email=sean.may101@gmail.com commit -m "docs: 3b generations, headline runs and results"
```

---

## Spec coverage (self-review)

| Spec requirement | Task |
|---|---|
| `net/` crate (`settler-net`) depending on engine and search only; `bots → net → search → engine` | 1, 3 |
| The Rust encoder is the only encoder; Python gets features through the bindings | 1, 5 |
| PyTorch only in the `train` extra, pinned; `pip-audit` after adding it | 6 |
| Seat-relative view; value rotated back to absolute seats | 1, 3 |
| Public features (nodes, edges, tiles, scalars), fixed scaling constants | 1 |
| Hidden block (opponents' hands and dev cards, 30 floats), value head only | 1, 2, 3 |
| Network shape: trunk 512×3, policy 665, value [512‖30]→256→4, aux 256→4 | 2, 6 |
| Prior depends only on the actor's observation (property test through `NetEvaluator`) | 3 |
| `OfferTrade` always masked | 3, 5 |
| Weights file: magic, version, `ENCODING_HASH`, shapes, f32 LE; clear errors | 1, 2, 6 |
| `NetEvaluator`: encode, one forward pass per batch, mask, softmax, rotate; `Arc` weights | 3 |
| Leaf batch default from an 8/16/32 sweep (simulations/s and a paired match at equal simulations) | 8 |
| Bot names `net:NAME`, `@N`, `+rD`; NAME charset; `--nets-dir` default; errors before any game, in the native arena, Catanatron arena and `compare` | 3 |
| Search loop, belief, sampler, turn cap, determinism, no trades unchanged | 3, 4 |
| Playout cap randomization (p 0.25, 600/100) from the game's seeded rng; only full searches are policy targets | 4, 5, 6 |
| Dirichlet root noise on full searches, α = 10 / legal, Marsaglia–Tsang from the engine `Rng` | 4 |
| Proportional sampling during setup and the first 8 turns, most-visited after | 4 |
| Records: 3a format plus net name and hash, both caps, exploration settings; fast searches with visits and `full_search: false` | 4, 5 |
| `selfplay --net NAME` | 5 |
| Warm start: 20,000 `ismcts` games with PCR, net₀ by supervision | 7, 8 |
| Rust sampler: compact actions, seeded (game, decision) draws, replay on a thread pool, raw LE buffers, `torch.frombuffer` | 5, 6 |
| Loss terms and weights; Adam, batch 1,024, lr 1e-3 / 3e-4; steps = 2 × new / 1,024; train from the previous generation | 6, 7 |
| Export plus a post-export parity check | 6 |
| Window of the most recent 4 generations | 7 |
| Gating: 1 vs 3 best at 600 simulations, 200 seeds, ≥ 0.25 promotes, data stays | 7 |
| `train-loop`: `gen-NNN/`, manifest with hashes, resume at the first incomplete step, temp-then-rename; scale-out by dropping record files | 5, 6, 7 |
| Tests: encoder golden and rotation; prior invariance; parity 100 positions 1e-5; weights errors; batch = encoder on replay; fast searches masked; overfit; PCR fraction within 3 SE over 2,000 decisions; determinism and thread independence; interrupt-and-resume manifest; gating tie; name errors everywhere | 1–7 |
| Done 1: all tests pass with the `train` extra, none skipped; 0 fallbacks, 0 belief resets | 6, 7, 9 |
| Done 2: net throughput and the batch sweep in `docs/perf/` | 8 |
| Done 3: warm-start check vs three `ismcts@1000` | 8, 9 |
| Done 4: headline vs `ismcts@1000` and `catanatron:alphabeta` | 9 |
| Done 5: strength curve 100/300/1,000/3,000 vs greedy | 9 |
| At most 3 extra generations changing only training hyperparameters; record the gap | 9 |
