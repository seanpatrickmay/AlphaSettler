# Plan 10: Encoder v4, Road-Network and Endgame-Threat Inputs (Sub-project 3d-2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the net the road network and the endgame as inputs (encoder v4: per node 4 seat-relative road distances and an open-spot flag, per edge 4 longest-trail flags, per seat a 6-slot threat block), upgrade v3 nets to v4 with identical outputs (`alphasettler net-upgrade`), then measure it: arm D, a 3-hour `fast-loop` from gen-034 upgraded, with the settings of the arm that won 3d-1b, settled by an SPRT of `net:D@100+t[+o]` against three copies of that arm's best, plus search speed v3 against v4.

**Architecture:** The engine's longest-road walk becomes `settler_engine::rules::awards::longest_trail(roads, blocked) -> (u8, u128)` (length and the trail's edges); `longest_road` returns its length, so no game changes. `settler_net::encode` appends the new slots to the end of the node, edge and global blocks (`NODE_WIDTH` 19 → 24, `EDGE_WIDTH` 4 → 8, `THREATS` after the trade block, `PUBLIC_LEN` 1581 → 2163) and computes them from the `Observation`: a three-ring BFS per seat, the engine's `longest_trail` per seat, and the threat arithmetic. Everything downstream (forward pass, sampler, PyTorch twin, parity test) reads the widths from the constants. `settler_net::weights` gains encoder v3's hash and widths, `Arch::shapes_for`, `Layer::widened` and `Weights::random_v3`; `alphasettler.netformat` reads files of any encoder's widths; `alphasettler.netupgrade` widens each encoder's first layer with zero columns, from encoder v2 (Task 2, keeping the v2 trade-row reset) and v3 (Task 3). The controller then upgrades gen-034, measures speed and runs arm D through `runs/p2/arm.sh` unchanged, the SPRT, and writes `docs/bot/results-3d2.md`.

**Tech Stack:** Rust 1.80+ (edition 2021, no new crates), PyO3 0.29.3 + maturin 1.15.0, Python 3.12 stdlib, PyTorch 2.14.1 (`train` extra only), pytest 9.1.1.

**Spec:** `docs/superpowers/specs/2026-10-08-encoder-v4-roads-threats-design.md` (binding). Builds on encoder v3 (`net/src/encode.rs`, `ENCODER_VERSION` 3, `PUBLIC_LEN` 1581), the v2 → v3 upgrade tool (`alphasettler/netupgrade.py`, c6b2646), Plan 8 (`docs/superpowers/plans/2026-10-08-plan-8-gumbel-root.md`: `runs/p2/arm.sh`, `runs/p2/tables.py`, arm A) and Plan 9 (`docs/superpowers/plans/2026-10-08-plan-9-opening-selfplay.md`: `fast-loop --opening`, arm C, `docs/bot/results-3d1b.md`). Plan 9 must be finished (its results committed) before Task 1 starts: Task 4 reads its verdict, and arm D runs after arm C on an idle machine.

## Global Constraints

- Python is `.venv/bin/python3`, tests `.venv/bin/pytest`, the CLI `.venv/bin/alphasettler`. Never `python`, `pip`, `python3` outside the venv, or `pip3` outside the venv.
- Rust is `PATH=$HOME/.cargo/bin:$PATH cargo ...` (cargo is not on PATH; `~/.cargo/bin/cargo` is the same binary). MSRV 1.80 (`rust-version = "1.80"`): use nothing stabilised after 1.80 (no `Option::is_none_or`, no `is_multiple_of`). No new external crates and no new Python dependencies. No numpy anywhere.
- Rebuild the extension after any Rust change used from Python. Another measurement may hold the extension: **never run `maturin develop`**. Build a wheel and rename the new library over the old one (an atomic rename, so running processes keep their mapped copy):
  ```bash
  rm -rf target/wheels-3d2 && PATH=$HOME/.cargo/bin:$PATH .venv/bin/maturin build --release -q -o target/wheels-3d2 \
    && unzip -q -o target/wheels-3d2/alphasettler-*.whl 'alphasettler/_engine*.so' -d target/wheels-3d2/x \
    && cp target/wheels-3d2/x/alphasettler/_engine.cpython-312-darwin.so alphasettler/._engine.new.so \
    && mv -f alphasettler/._engine.new.so alphasettler/_engine.cpython-312-darwin.so
  ```
  "Rebuild the extension" in any step below means exactly this command.
- Every task ends with `PATH=$HOME/.cargo/bin:$PATH cargo fmt --all` then `PATH=$HOME/.cargo/bin:$PATH cargo fmt --check` (no output), `PATH=$HOME/.cargo/bin:$PATH cargo clippy --workspace --release --all-targets -- -D warnings` (exit 0), the full `PATH=$HOME/.cargo/bin:$PATH cargo test --workspace --release` (all pass) and, after the rebuild, `.venv/bin/pytest -q -rs` (all pass, the summary lists no skipped tests).
- The layout (spec Section 2), all seat-relative (r0 = the viewer, then turn order), every new slot appended at the end of its block: per node `+5` (r0..r3 road distance at slots 19..22, open spot at 23; `NODE_WIDTH` 24), per edge `+4` (r0..r3 on the longest trail at 4..7; `EDGE_WIDTH` 8), globals `+24` after the trade block (4 seats × 6: longest road length / 15; roads to the longest-road award / 5; knights to the largest-army award / 3; points to win / `vp_to_win`; settlements left / 5; cities left / 4). `PUBLIC_LEN` = 1581 + 54·5 + 72·4 + 24 = 2163; `HIDDEN_LEN` stays 30; `ENCODER_VERSION` 4.
  - Road distance: the fewest roads seat r must build to touch the node with its network (every node its roads or buildings touch); paths may not use another seat's road or pass through a node another seat has built on; capped at 3 and divided by 3; unreachable within 3 is 1.0; a node on the network is 0.
  - Open spot: 1 if the node is empty and no neighbour is built on.
  - On the longest trail: 1 if the edge lies on the one longest trail `awards::longest_road` measures (ties to the first found, a deterministic order).
  - Roads to the award: 0 for the holder, else `min(5, max(5, holder length + 1) − own length)`, no holder `max(0, 5 − own length)`. Knights: the same rule with 3 and played knights. Points to win: `max(0, vp_to_win − points) / vp_to_win`, points = public VP for opponents, total (VP cards included) for the viewer.
- Under `engine/` only `engine/src/rules/awards.rs` changes (and engine tests). Nothing under `search/` or `bots/src/` is edited; `net/src/forward.rs`, `net/src/evaluator.rs` and `net/src/data.rs` are not edited (they read the widths from the constants). No change to the heads, the width, the rounds or the message passing (spec Section 4).
- **Tests and digests that read the encoder, and how each is handled** (nothing else is re-pinned):
  - `engine/tests/golden_trace.rs` passes unchanged: `longest_road` returns `longest_trail`'s length, the same maximum over the same walks (Task 1).
  - 3a's `the_generic_refactor_leaves_3a_ismcts_games_unchanged` (`bots/tests/ismcts.rs`) and `fixed_exploration_is_3a_self_play` (`bots/tests/selfplay.rs`), P2's `bots_without_the_gumbel_suffix_play_the_games_they_played_before_p2` (`bots/tests/ismcts.rs`) and `standard_exploration_plays_the_games_it_played_before_p2` (`bots/tests/selfplay.rs`), 3d-1b's `self_play_without_an_opening_model_plays_the_games_it_played_before_3d1b` (`bots/tests/selfplay.rs`), every `+o` test (`bots/tests/opening_model.rs`, whose model reads its own 78 features, not the encoder) and `engine/tests/topology.rs`'s digest: heuristic bots and the engine only, never re-pinned, pass unchanged after every task.
  - P2's `net_bots_without_the_gumbel_suffix_play_the_games_they_played_before_p2` (`bots/tests/netbot.rs`, digest `0xf4d5c47b40f849a3`) is the one digest that reads the encoder: its `tiny.bin` is `Weights::random` at the current layout, so v4 would change both the net and its inputs. It is not re-pinned. Its net becomes `Weights::random_v3(Arch { width: 8, rounds: 1 }, 1)`: v3's draws, upgraded with zero columns, which ignores every new input (Task 2). The wider sums round differently in Accelerate (measured while planning: up to 1.9e-6 on padded random layers, never bit-equal at batch 1, sometimes bit-equal at larger row counts), but the pinned games do not move: while planning this plan, encoder v3's forward pass was run with exactly those padded encoder layers (zero weight columns, nonzero inputs; outputs differed from v3 by 2 ulps) and the pin passed with `0xf4d5c47b40f849a3`.
  - Every other net test builds its net at the current layout with `Weights::random` or the twin (`Net`, `write_random_net`), so it loads under v4 and needs no change; no test reads a v3 net file from disk (none is checked in).
  - The encoder's own tests are re-derived, not re-pinned: `the_layout_adds_up` (new offsets, arithmetic shown), `changing_any_layout_constant_changes_the_hash` (unchanged code, more entries), `reordering_slots_or_blocks_in_the_description_changes_the_hash` (three mutations of the new text added), `the_description_matches_the_constants` (new scales and the threat block), `the_golden_position_encodes_exactly` (the new slots worked out by hand, below), `the_trade_block_encodes_exactly` (its slice now ends at `THREATS`), the seat-rotation test (extended to count the new blocks), `net/tests/graph.rs` (139 globals), `net/tests/weights.rs` (shapes, the parameter count 248,446 + 48 × (5 + 4 + 24) = 250,030, the hand-computed tiny network's encoder widths, the v2 file test).
- Records store raw actions, so every existing self-play directory (`runs/3b-v2/gen-*`, `runs/3c/gen-*`) re-encodes under v4 with no conversion; no training data is cached.
- The trading game is exactly `{"max_offers_per_turn": 2, "max_trade_cards": 1}` (`TRADES` in `alphasettler/arena.py`, `trading()` in Rust tests).
- The test (spec Section 1.3): arm D runs `bash runs/p2/arm.sh D ...` with `runs/p2/arm.sh` unchanged (fresh root, the trading game, seed directories `runs/3b-v2/gen-002` .. `gen-005` and `runs/3c/gen-030` .. `gen-034`, `--first 35`, 2,000 games, window 40,000 games, 10 training minutes, a gate of 100 seeds at 100 simulations; set up untimed, then 10,800 s of generations), its start net overridden by a second `--start-net runs/3d2/gen-034-v4.bin`, plus `--opening runs/3c/nets/opening.bin` exactly when 3d-1b's SPRT was H1. Control: arm C if 3d-1b was H1, else arm A. Then `net:D@100+t[+o]` against three copies of the control's best, upgraded to v4, `alphasettler sprt --trades` with p0 0.25, p1 0.30, alpha = beta = 0.05, fresh seeds from 5,000,000,000.
- Determinism: same seed and weights give the same games, decisions and features, independent of thread count. Budgets are simulation counts, never wall time (arm D's 3 hours is the experiment's wall clock). The forward pass is Accelerate-based: batch results and widened layers equal the originals only within 1e-5, so no test compares outputs across different batch compositions or across layer widths bit for bit.
- No module outside `alphasettler/train/` imports torch (`alphasettler/netupgrade.py` and `alphasettler/netformat.py` stay torch-free).
- Git: one commit per task with the message given in the task, run as `git -c user.email=sean.may101@gmail.com commit -m "..."`. Stage only the files the task names (other agents may be committing in this tree). No Co-Authored-By or AI attribution, never `--no-verify`, never push. Before every commit run `git diff --cached | grep -nE 'sk-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{36}|password=[^[:space:]]+'` and expect no output.
- Testing: integration tests over mocks (real positions encoded by the real encoder, real weights files read by the real Rust loader, real CLI runs, a real loop); test behaviour, not implementation; independent tests (each builds its own temp files); descriptive names; pytest fixtures and `parametrize` where cases repeat. Hand-computed expectations show their arithmetic in a comment.
- Long runs (the v3 build, the speed runs, arm D's 3-hour loop, the SPRT) are run by the controller in the background, never by implementer subagents. Steps marked **controller runs this** give the exact command and the pass condition.
- No README or docs beyond `docs/bot/results-3d2.md` (Task 4).

## Review Focus

1. A seat with nothing on the board (every seat at the first setup decision; a seat that has not placed yet). Expected: every road distance 1.0 (no network, nothing reachable), no trail edge, every node an open spot on an empty board, its threat block `[0, 1, 1, 1, 1, 1]` (no road, 5 roads and 3 knights to go, every point to win, every piece in supply); no panic on the empty road set. Pinned by Task 2's `an_empty_board_reaches_nothing_and_every_spot_is_open` (and r3 of the golden position).
2. A road that ends at a node an opponent has since built on (the usual way a road is cut). Expected: the node counts as touched (distance 0) but no path leaves it, so the nodes beyond it are reached only some other way; the longest trail may end there. Pinned by Task 2's `a_road_into_an_opponents_building_reaches_it_but_not_past_it`.
3. Hidden VP cards. Expected: the viewer's points to win count its own VP cards (bought this turn included); an opponent's cards never show (they are not in the observation), so its threat block uses public VP only. Pinned by Task 2's `the_road_and_threat_blocks_encode_exactly` (player 3 holds a VP card the viewer cannot see; the viewer holds one it counts) and the golden position.
4. The longest-road award unowned while seats are past its minimum (a broken holder with tied challengers, `assign_longest_road`'s None case). Expected: the spec's no-holder branch as written, `max(0, 5 − own)`: 0 to go for each seat at 5 or more, 1.0 for a seat with no road. Pinned by Task 2's `an_unowned_award_past_its_minimum_reads_zero_to_go`.
5. An old net where only a v4 net loads, or a v4 net given to the upgrade: a v3 or v2 file named by a `net:` bot, read by the twin or by `net_forward`; `net-upgrade` on a v4 file, a file of another hash, or onto an existing output. Expected: the loaders refuse with an error naming `alphasettler net-upgrade`; the upgrade refuses with "already an encoder v4 net", "neither encoder v2's ... nor encoder v3's" or "exists", and writes nothing. Pinned by Task 3's `test_loaders_send_an_older_net_to_net_upgrade`, `test_only_encoder_v2_and_v3_nets_are_upgraded`, `test_cli_net_upgrade` and Rust's `an_encoder_v2_or_v3_file_is_refused_with_the_way_to_upgrade_it`.

## File Structure

| File | Responsibility | Task |
|---|---|---|
| `engine/src/rules/awards.rs` | `longest_trail` (length and edges, first found on a tie); `longest_road` returns its length; `LocalGraph` keeps each local edge's board id and its walk returns the edges used | 1 |
| `net/src/encode.rs` | encoder v4: widths, offsets, scales, `LAYOUT`, `LAYOUT_DESCRIPTION`; `road_distances`, `network`, `to_award`; the new slots in `encode_public` | 2 |
| `net/src/graph.rs` | the comment on the 139 globals | 2 |
| `net/src/weights.rs` | the layer table; `V3_ENCODING_HASH`, `V3_NODE_WIDTH`, `V3_EDGE_WIDTH`, `V3_GLOBAL_WIDTH`; `Arch::shapes_for`; `Layer::widened`; `Weights::random_v3` (2); the loader's hint for v3 files (3) | 2, 3 |
| `bindings/src/train.rs` | `NET_V3_ENCODING_HASH`, `NET_V3_NODE_WIDTH`, `NET_V3_EDGE_WIDTH`, `NET_V3_GLOBAL_WIDTH` | 2 |
| `alphasettler/netformat.py` | `shapes` and `read` for any encoder's input widths; `NODE_ENCODER`, `EDGE_ENCODER` | 2 |
| `alphasettler/netupgrade.py` | `SOURCES`, `read_old`, `widened`, `upgrade`, `upgrade_file` to v4: from v2 (2), from v3 (3) | 2, 3 |
| `alphasettler/train/model.py` | docstring (2); the v3 hint (3) | 2, 3 |
| `alphasettler/cli.py` | `net-upgrade` texts and its summary line | 2, 3 |
| `docs/bot/results-3d2.md` (new) | the measurements | 4 |
| tests: `engine/tests/awards.rs`, `engine/tests/longest_road_oracle.rs` (1); `net/tests/encode.rs`, `net/tests/graph.rs`, `net/tests/weights.rs`, `bots/tests/netbot.rs`, `tests/python/test_net_upgrade.py`, `tests/python/test_train.py` (2); `net/tests/weights.rs`, `tests/python/test_net_upgrade.py` (3) | | 1–3 |

## Where the code forces a choice the spec does not make

Each is a **Ruling** (with its reason) the results file repeats, so it can be overridden.

- **A node another seat built on, at the end of a seat's road, is on that seat's network (distance 0) but no path leaves it.** The spec defines the network as every node the seat's roads or buildings touch, and forbids passing through a built-on node; both hold. A built-on node can be reached (distance 1 or 2) but not left.
- **A seat with nothing on the board has no network: every node reads 1.0.** "Unreachable within 3" covers it; there is nothing to measure from.
- **Distances read the board only:** a seat out of road pieces, or without the cards, still gets its distances. The spec's input is where the network can still reach, not what the seat can afford.
- **Ties between longest trails go to the engine walk's first:** starts in the order their nodes first appear among the seat's roads in ascending edge order, and from each node its roads in that same order; a later trail replaces the best only if strictly longer. The trail may end at, never pass, a node another seat built on (the engine's rule), and a seat without roads has no trail edge.
- **The holder's length in "roads to the award" is the trail length the encoder computes for the holder** (the same function and the same blocked nodes as the engine's cached length, so equal in every legal position), and the subtraction saturates at 0 (it cannot go negative while the holder has the longest trail, which the engine keeps true).
- **No holder and a seat at or past the minimum reads 0 to go**, as the spec writes `max(0, 5 − own length)`, even when another seat ties it and one more road would be needed (Review Focus 4).
- **The viewer's points include VP cards bought this turn** (`Observation::my_dev_cards`, as `State::total_vp` counts them). **`vp_to_win` is a config scale** (like `max_turns`), written `/vp_to_win` in the description and not in `LAYOUT`.
- **Six new scales enter `LAYOUT` and the description's `scales:` line:** `distance=3`, `road_length=15` (`MAX_ROADS`), `road_award=5`, `army_award=3`, `settlements=5` (`MAX_SETTLEMENTS`), `cities=4` (`MAX_CITIES`). Supply is the maximum minus the pieces on the board (a settlement upgraded to a city returns to supply).
- **"Bit-equal at batch 1" does not hold, so the upgrade's check is the forward pass's 1e-5.** The zero columns contribute exactly zero, but Accelerate's `cblas_sgemm` sums a 24-, 8- or 139-column row in another order than a 19-, 4- or 115-column one (measured while planning: differences up to 1.9e-6, at batch 1 too). An upgraded net "gives the same outputs" within the parity rule (|a − b| ≤ 1e-5 × max(1, |a|)); the twin, which receives the same weights, gives exactly the original net's.
- **Encoder v2 nets upgrade straight to v4.** "The v2 → v3 path stays" is kept as what it did (zero columns for the trade block, the trade policy rows reset), applied in one step together with the v4 columns: this build cannot load a v3 file, so writing one would be useless. `net-upgrade` always writes this build's encoder.
- **The P2 net pin keeps its digest through `Weights::random_v3`** (Global Constraints): v3's draws over v3's shapes, then zero columns, exactly what `net-upgrade` makes of a v3 file.
- **The loaders' hint covers v3 files** (`net-upgrade` can convert them from Task 3 on), worded as v2's: "it is an encoder v3 net (before the road and threat inputs): convert it with `alphasettler net-upgrade OLD NEW`".
- **`net-upgrade` prints what it zeroed:** `upgraded OLD (encoder vN) to encoder v4: width W, R rounds; 5 node, 4 edge and G global inputs zeroed[; 407 trade policy rows reset (offer bias -3)]; wrote NEW` (G = 48 from v2, 24 from v3).
- **Arm D runs through `runs/p2/arm.sh` itself, unchanged:** the script passes its extra arguments after its own `--start-net runs/3c/nets/gen-034.bin`, and argparse keeps the last value of a plain option, so `bash runs/p2/arm.sh D --start-net runs/3d2/gen-034-v4.bin [--opening runs/3c/nets/opening.bin]` starts from the upgraded net with every other setting and seed of A and C. Task 4 checks `runs/p2/D/nets/start.bin` against the upgraded file.
- **The SPRT plays two v4 nets:** the control's best (C's best or A's `gen-040`) is upgraded into `runs/3d2/nets/`, beside `D.bin` (and `opening.bin` when `+o`); an upgraded control plays as its v3 self within 1e-5. If `D.bin` and the control are the same file (neither loop promoted past an identical start), the SPRT is not run and is recorded as H0.
- **The v3 side of the speed and output comparisons is a build of the commit before Task 1** (`git archive` into `runs/3d2/v3-src`, its own target directory), with the same `net_speed speed PATH 500 1000 8` command as `docs/perf/net-search.md`, three runs each, alternating.
- **H1 and H0 change no code:** the build is v4 either way (a v4 net with zero new-input weights is exactly its v3 net). H1: new loops start from arm D's best. H0 or undecided: recorded; new loops start from the control's best, upgraded.
- **Fresh seeds** for the SPRT start at 5,000,000,000 (at most 400 seeds): no self-play (below 1e9), gate (1e9 + 0..200) or SPRT so far (2e9.., 3e9.., 4e9..) used them.
- **Task boundaries:** the engine walk (1), the encoder and everything that must follow for the build to stay green, the v2 upgrade included (2), v3 nets (3), the runs (4). A reviewer can accept the walk and reject the encoder, or accept the encoder and reject how v3 files are converted.

---

### Task 1: `longest_trail`, the longest road's edges

**Files:**
- Modify: `engine/src/rules/awards.rs:9-48` (`longest_road`, new `longest_trail`), `:52-123` (`LocalGraph`: `edge`, `new`, `walk`)
- Test: `engine/tests/awards.rs`, `engine/tests/longest_road_oracle.rs`

**Interfaces:**
- Consumes: nothing new (`Topology`, `bits128`).
- Produces: `settler_engine::rules::awards::longest_trail(roads: u128, blocked: u64) -> (u8, u128)` — the longest trail's length and its edges as a mask of board edge ids (`(0, 0)` for no roads); `longest_road(roads, blocked) -> u8` unchanged in signature and value.

- [ ] **Step 1: Write the failing tests**

Append to `engine/tests/awards.rs`:

```rust
#[test]
fn the_longest_trail_names_its_edges_and_a_tie_goes_to_the_first_found() {
    // A Y at node 4: edges 1 (0-4), 2 (1-4) and 7 (4-8). Every trail through node 4 uses two of
    // them (length 2). The walk starts at node 0 (the first node of the lowest edge, 1), crosses
    // node 4 and tries edge 2 before edge 7: 0-4-1 is found first and a later trail of the same
    // length does not replace it.
    let y = (1u128 << 1) | (1 << 2) | (1 << 7);
    assert_eq!(longest_trail(y, 0), (2, (1 << 1) | (1 << 2)));
    assert_eq!(longest_road(y, 0), 2);
    assert_eq!(longest_trail(0, 0), (0, 0));
    // A ring: all six edges.
    let ring = topo().tile_nodes[9];
    let mut nodes = ring.to_vec();
    nodes.push(ring[0]);
    let roads = mask128(&path_edges(&nodes));
    assert_eq!(longest_trail(roads, 0), (6, roads));
}

#[test]
fn a_blocked_node_ends_every_trail_through_it() {
    // The Y blocked at node 4: no trail crosses it, so the longest is one edge. The walk from
    // node 0 meets node 4 after edge 1 and stops: edge 1 is found first.
    let y = (1u128 << 1) | (1 << 2) | (1 << 7);
    assert_eq!(longest_trail(y, 1 << 4), (1, 1 << 1));
    assert_eq!(longest_road(y, 1 << 4), 1);
}
```

In `engine/tests/longest_road_oracle.rs`, replace `use settler_engine::rules::awards::longest_road;` with `use settler_engine::rules::awards::{longest_road, longest_trail};` and append:

```rust
/// `longest_trail` returns the length the reference finds, and edges that are the player's,
/// as many as the length, and one trail: the longest trail within them uses all of them.
#[test]
fn the_longest_trail_is_one_trail_of_the_longest_length() {
    let mut rng = Rng::new(7);
    for case in 0..20_000 {
        let count = 1 + rng.below(15);
        let roads = random_roads(&mut rng, count);
        let blocked = if rng.below(2) == 0 {
            0
        } else {
            rng.next_u64() & rng.next_u64() & ((1u64 << 54) - 1)
        };
        let (len, trail) = longest_trail(roads, blocked);
        let at = format!("case {case}: roads {roads:#x} blocked {blocked:#x}");
        assert_eq!(len, reference(roads, blocked), "{at}");
        assert_eq!(trail & !roads, 0, "{at}: the trail is the player's roads");
        assert_eq!(trail.count_ones(), len as u32, "{at}");
        assert_eq!(longest_road(trail, blocked), len, "{at}: its edges are one trail");
    }
}
```

- [ ] **Step 2: Run them to verify they fail**

Run: `PATH=$HOME/.cargo/bin:$PATH cargo test --release -p settler-engine --test awards --test longest_road_oracle`
Expected: FAIL to compile with "cannot find function `longest_trail`".

- [ ] **Step 3: Implement**

In `engine/src/rules/awards.rs`, replace everything from the doc comment `/// Longest trail (no edge reused) in \`roads\`; ...` through the end of `fn longest_road` with:

```rust
/// Longest trail (no edge reused) in `roads`; trails may end at but not pass through `blocked` nodes.
pub fn longest_road(roads: u128, blocked: u64) -> u8 {
    longest_trail(roads, blocked).0
}

/// The longest trail in `roads` (no edge reused; it may end at but not pass through a `blocked`
/// node): its length and its edges, as a mask of board edge ids. Of equally long trails the
/// first found is returned: starts in the order their nodes first appear among `roads` in
/// ascending edge order, from each node its roads in that same order, and a later trail
/// replaces the best only if strictly longer (spec 3d-2: a deterministic order).
///
/// Searches only from nodes where some longest trail must start: dead ends, branch points, and
/// blocked nodes. A trail starting at an unblocked degree-2 node either extends backwards
/// (so it wasn't longest) or is closed and can be rotated onto a non-degree-2 node; the one
/// exception is a component that is a pure cycle, which gets one start of its own.
pub fn longest_trail(roads: u128, blocked: u64) -> (u8, u128) {
    if roads == 0 {
        return (0, 0);
    }
    let g = LocalGraph::new(roads, blocked);
    let mut starts = 0u32;
    for v in 0..g.nodes {
        if g.blocked & (1 << v) != 0 || g.degree[v] != 2 {
            starts |= 1 << v;
        }
    }
    // Any component the starts can't reach is a pure cycle: give it one start.
    let all = if g.nodes == 32 {
        u32::MAX
    } else {
        (1u32 << g.nodes) - 1
    };
    let mut reached = g.flood(starts);
    while all & !reached != 0 {
        let v = (all & !reached).trailing_zeros();
        starts |= 1 << v;
        reached |= g.flood(1 << v);
    }
    let mut best = (0u8, 0u32);
    let mut rest = starts;
    while rest != 0 {
        let v = rest.trailing_zeros() as usize;
        rest &= rest - 1;
        let found = g.walk(v, 0, 0);
        if found.0 > best.0 {
            best = found;
        }
    }
    let mut trail = 0u128;
    let mut used = best.1;
    while used != 0 {
        let i = used.trailing_zeros() as usize;
        used &= used - 1;
        trail |= 1u128 << g.edge[i];
    }
    (best.0, trail)
}
```

In `struct LocalGraph`, after the `adj` field add:

```rust
    /// The board edge id of each local edge (local edges number `roads` in ascending order).
    edge: [u8; 16],
```

In `LocalGraph::new`, add `edge: [0; 16],` to the `LocalGraph { ... }` literal, and in the loop `for (i, e) in bits128(roads).enumerate() {` add as its first line `g.edge[i] = e;`.

Replace `fn walk` with:

```rust
    /// The longest trail from `v` having used the local edges in `used` (length `len` so far):
    /// its length and every local edge it used. The first found wins a tie.
    fn walk(&self, v: usize, used: u32, len: u8) -> (u8, u32) {
        if len > 0 && self.blocked & (1 << v) != 0 {
            return (len, used);
        }
        let mut best = (len, used);
        for &(e, w) in &self.adj[v][..self.degree[v] as usize] {
            if used & (1 << e) == 0 {
                let found = self.walk(w as usize, used | (1 << e), len + 1);
                if found.0 > best.0 {
                    best = found;
                }
            }
        }
        best
    }
```

`longest_road`'s value is unchanged: it is the maximum over the same walks from the same starts (the old code took `best.max(...)`, the new keeps the first strictly longer, which has the same length).

- [ ] **Step 4: Run them to verify they pass, and the golden trace**

Run: `PATH=$HOME/.cargo/bin:$PATH cargo test --release -p settler-engine --test awards --test longest_road_oracle --test golden_trace`
Expected: PASS (`golden_trace.rs` unchanged).

- [ ] **Step 5: The gates**

Run the Global Constraints' gates (fmt, clippy, the full `cargo test --workspace --release`, rebuild the extension, `.venv/bin/pytest -q -rs`). Expected: all pass, no skips; every pinned digest of the Global Constraints passes unchanged.

- [ ] **Step 6: Commit**

```bash
git add engine/src/rules/awards.rs engine/tests/awards.rs engine/tests/longest_road_oracle.rs
git diff --cached | grep -nE 'sk-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{36}|password=[^[:space:]]+'
git -c user.email=sean.may101@gmail.com commit -m "engine: longest_trail returns the longest trail's edges beside its length; longest_road delegates to it"
```

---

### Task 2: Encoder v4 and everything that reads its widths

**Files:**
- Modify: `net/src/encode.rs` (constants `:11-51`, scales after `:85`, `LAYOUT` `:89-118`, `LAYOUT_DESCRIPTION` `:126-174`, imports `:7-9`, new private functions, the end of `encode_public`), `net/src/graph.rs:11`, `net/src/weights.rs` (doc table `:22-24`, constants after `:60`, `Arch::shapes`, `Layer`, `Weights::random`), `bindings/src/train.rs:23-25,399-400`, `alphasettler/netformat.py`, `alphasettler/netupgrade.py` (rewritten), `alphasettler/train/model.py:1-3`, `alphasettler/cli.py` (`net-upgrade` texts, `_net_upgrade`)
- Test: `net/tests/encode.rs`, `net/tests/graph.rs`, `net/tests/weights.rs`, `bots/tests/netbot.rs`, `tests/python/test_net_upgrade.py` (rewritten), `tests/python/test_train.py`

**Interfaces:**
- Consumes: `settler_engine::rules::awards::longest_trail` (Task 1).
- Produces:
  - `settler_net::encode`: `ENCODER_VERSION = 4`, `NODE_WIDTH = 24`, `EDGE_WIDTH = 8`, `NODE_DISTANCE = 19` (r0..r3 at 19..=22), `NODE_OPEN = 23`, `EDGE_TRAIL = 4` (r0..r3 at 4..=7), `THREATS = 2139`, `THREAT_WIDTH = 6` (`[length, roads to award, knights to award, points to win, settlements left, cities left]`), `PUBLIC_LEN = 2163`; scales `DISTANCE_DEN = 3`, `ROAD_LENGTH_DEN = 15`, `ROAD_AWARD_MIN = 5`, `ARMY_AWARD_MIN = 3`, `SETTLEMENT_DEN = 5`, `CITY_DEN = 4` (all `u32`); `LAYOUT: [u32; 33]`.
  - `settler_net::graph::GLOBAL_WIDTH = 139`.
  - `settler_net::weights`: `V3_ENCODING_HASH: u64 = 0x6d02_356c_b617_53c2`, `V3_NODE_WIDTH = 19`, `V3_EDGE_WIDTH = 4`, `V3_GLOBAL_WIDTH = 115` (`usize`); `Arch::shapes_for(&self, node: usize, edge: usize, global: usize) -> Vec<(usize, usize)>`; `Layer::widened(&self, cols: usize) -> Layer`; `Weights::random_v3(arch: Arch, seed: u64) -> Weights`.
  - `_engine.NET_V3_ENCODING_HASH`, `NET_V3_NODE_WIDTH`, `NET_V3_EDGE_WIDTH`, `NET_V3_GLOBAL_WIDTH`.
  - `alphasettler.netformat`: `NODE_ENCODER = 0`, `EDGE_ENCODER = 1`; `shapes(width, rounds, global_width=NET_GLOBAL_WIDTH, node_width=NET_NODE_WIDTH, edge_width=NET_EDGE_WIDTH)`; `read(data, global_width, check_encoding, layout, node_width=NET_NODE_WIDTH, edge_width=NET_EDGE_WIDTH)`.
  - `alphasettler.netupgrade`: `SOURCES: dict[int, tuple[int, int, int, int]]` (hash → (encoder version, node, edge, global widths)); `read_old(data) -> (version, width, rounds, layers)`; `widened(layer, cols)`; `upgrade(data) -> bytes`; `upgrade_file(old, new) -> {"from": int, "width": int, "rounds": int, "zero_inputs": {"node": int, "edge": int, "global": int}, "trade_rows": int}`.
  - Test helpers in `tests/python/test_net_upgrade.py`: `old_net(widths, width=8, rounds=1, seed=0) -> Net`, `old_bytes(net, widths) -> bytes`, `V2` (hash, node, edge, global widths of encoder v2).

- [ ] **Step 1: Write the failing Rust tests**

In `net/tests/encode.rs`:

Replace `use settler_net::encode::*;` with:

```rust
use settler_net::encode::*;
use settler_net::weights::{V2_ENCODING_HASH, V3_ENCODING_HASH};
```

After `const PIPS: ...;` add:

```rust
/// Each node's road-distance slots (r0 .. r3) in an encoded public block.
fn got_distances(public: &[f32]) -> Vec<[f32; NUM_PLAYERS]> {
    (0..NUM_NODES)
        .map(|n| {
            let at = NODE_BASE + n * NODE_WIDTH + NODE_DISTANCE;
            std::array::from_fn(|r| public[at + r])
        })
        .collect()
}

/// The road-distance slots expected when `near[r]` lists relative seat r's nodes 0, 1 and 2
/// roads away: d / 3 for those, 1.0 (3 or more roads, or never) for every other node.
fn want_distances(near: [[&[usize]; 3]; NUM_PLAYERS]) -> Vec<[f32; NUM_PLAYERS]> {
    let mut want = vec![[1.0f32; NUM_PLAYERS]; NUM_NODES];
    for (r, rings) in near.iter().enumerate() {
        for (d, nodes) in rings.iter().enumerate() {
            for &n in *nodes {
                want[n][r] = d as f32 / 3.0;
            }
        }
    }
    want
}

/// The nodes whose open-spot slot is 1; every other node's must be 0.
fn open_nodes(public: &[f32]) -> Vec<usize> {
    (0..NUM_NODES)
        .filter(|&n| {
            let x = public[NODE_BASE + n * NODE_WIDTH + NODE_OPEN];
            assert!(x == 0.0 || x == 1.0, "node {n}: {x}");
            x == 1.0
        })
        .collect()
}

/// The edges on relative seat `r`'s longest trail, ascending.
fn trail_edges(public: &[f32], r: usize) -> Vec<usize> {
    (0..NUM_EDGES)
        .filter(|&e| public[EDGE_BASE + e * EDGE_WIDTH + EDGE_TRAIL + r] == 1.0)
        .collect()
}

/// Relative seat `r`'s threat block.
fn threat(public: &[f32], r: usize) -> &[f32] {
    &public[THREATS + r * THREAT_WIDTH..THREATS + (r + 1) * THREAT_WIDTH]
}

/// A mask of board edges.
fn edges(es: &[u8]) -> u128 {
    es.iter().fold(0u128, |m, &e| m | (1u128 << e))
}
```

Replace `fn the_layout_adds_up` with:

```rust
#[test]
fn the_layout_adds_up() {
    assert_eq!(
        (NODE_WIDTH, EDGE_WIDTH, NODE_DISTANCE, NODE_OPEN, EDGE_TRAIL),
        (24, 8, 19, 23, 4)
    );
    // v3's 1581 + 54 nodes x 5 + 72 edges x 4 + 4 seats x 6 = 1581 + 270 + 288 + 24 = 2163.
    assert_eq!(PUBLIC_LEN, 2163);
    assert_eq!(HIDDEN_LEN, 30);
    // Nodes 54 x 24 = 1296, edges 72 x 8 = 576 (1296 + 576 = 1872), tiles 19 x 8 = 152: the
    // globals start at 2024, each 270 + 288 = 558 after v3's (1466 + 558 = 2024, ...).
    assert_eq!(
        (
            EDGE_BASE,
            TILE_BASE,
            OWN_HAND,
            SEATS,
            LONGEST_ROAD,
            BANK,
            PHASE,
            TURN
        ),
        (1296, 1872, 2024, 2039, 2079, 2089, 2095, 2106)
    );
    assert_eq!(
        (DEV_PLAYED, CURRENT, RETURN_PHASE, FREE_ROADS),
        (2107, 2108, 2112, 2114)
    );
    assert_eq!(
        (
            TRADE_GIVE,
            TRADE_GET,
            TRADE_RESPONSES,
            OFFERS_MADE,
            OFFERS_ALLOWED
        ),
        (2115, 2120, 2125, 2137, 2138)
    );
    // The threat block follows the trade block: 2139 + 4 x 6 = 2163.
    assert_eq!((THREATS, THREAT_WIDTH), (2139, 6));
    assert_eq!(ENCODER_VERSION, 4);
    assert_eq!(ENCODING_HASH, layout_hash(&LAYOUT, LAYOUT_DESCRIPTION));
    assert_ne!(ENCODING_HASH, V2_ENCODING_HASH, "encoder v2's hash");
    assert_ne!(ENCODING_HASH, V3_ENCODING_HASH, "encoder v3's hash");
}
```

In `fn reordering_slots_or_blocks_in_the_description_changes_the_hash`, after the line `exchange(LAYOUT_DESCRIPTION, "public trade_give", "public trade_get"),` add:

```rust
        exchange(LAYOUT_DESCRIPTION, "road_distance_r0", "road_distance_r1"),
        exchange(LAYOUT_DESCRIPTION, "longest_trail_r2", "longest_trail_r3"),
        exchange(
            LAYOUT_DESCRIPTION,
            "points_to_win/vp_to_win",
            "settlements_left/settlements",
        ),
        exchange(LAYOUT_DESCRIPTION, "public offers_allowed", "public threat"),
```

In `fn the_description_matches_the_constants`: after `("free_roads", FREE_ROAD_DEN),` add

```rust
        ("distance", DISTANCE_DEN),
        ("road_length", ROAD_LENGTH_DEN),
        ("road_award", ROAD_AWARD_MIN),
        ("army_award", ARMY_AWARD_MIN),
        ("settlements", SETTLEMENT_DEN),
        ("cities", CITY_DEN),
```

replace `["max_turns", "max_trade_cards", "max_offers_per_turn"].contains(&scale);` with `["max_turns", "max_trade_cards", "max_offers_per_turn", "vp_to_win"].contains(&scale);`, and after `("public", "offers_allowed", 1, 1, OFFERS_ALLOWED),` add `("public", "threat", NUM_PLAYERS, THREAT_WIDTH, THREATS),`.

In `fn the_golden_position_encodes_exactly`, insert before `for i in 0..PUBLIC_LEN {`:

```rust
    // v4, seen by player 2: r0 = player 2, r1 = 3, r2 = 0, r3 = 1. Occupied: 0 (player 3), 10 and
    // 20 (player 2), 40 (player 0).
    // Road distances (a seat's network is distance 0; a path may not use another seat's road or
    // leave another seat's building):
    // r0 = player 2, network {1, 2, 5, 10, 20} (roads 1-5-2 and the two buildings). One road: 4
    //   (from 1), 6 (from 2 and 10), 9 (from 5), 14, 15 (from 10), 25, 26 (from 20). Two: 0 (from
    //   4: player 3's settlement, reached), 8 (from 4), 13 (from 9), 19 (from 14 and 25), 31 (from
    //   25), 32 (from 26).
    // r1 = player 3, network {0, 3}. One: 4 (from 0), 7 (from 3). Two: 1, 8 (from 4), 11, 12 (from 7).
    // r2 = player 0, network {40}. One: 35, 44, 45. Two: 29, 30 (from 35), 39, 48 (from 44), 41, 49
    //   (from 45).
    // r3 = player 1: nothing on the board, every node 1.0.
    let distances = want_distances([
        [&[1, 2, 5, 10, 20], &[4, 6, 9, 14, 15, 25, 26], &[0, 8, 13, 19, 31, 32]],
        [&[0, 3], &[4, 7], &[1, 8, 11, 12]],
        [&[40], &[35, 44, 45], &[29, 30, 39, 41, 48, 49]],
        [&[], &[], &[]],
    ]);
    for (n, d) in distances.iter().enumerate() {
        let at = NODE_BASE + n * NODE_WIDTH + NODE_DISTANCE;
        want[at..at + NUM_PLAYERS].copy_from_slice(d);
    }
    // Open spots: every node but the 4 occupied and their neighbours (0: 3, 4; 10: 6, 14, 15;
    // 20: 15, 25, 26; 40: 35, 44, 45): 54 - 14 = 40 nodes.
    let closed = [0, 3, 4, 6, 10, 14, 15, 20, 25, 26, 35, 40, 44, 45];
    for n in (0..NUM_NODES).filter(|n| !closed.contains(n)) {
        want[NODE_BASE + n * NODE_WIDTH + NODE_OPEN] = 1.0;
    }
    // Longest trails: player 2's 1-5-2 (edges 3 and 4, length 2), player 3's edge 0 (length 1).
    want[EDGE_BASE + 3 * EDGE_WIDTH + EDGE_TRAIL] = 1.0;
    want[EDGE_BASE + 4 * EDGE_WIDTH + EDGE_TRAIL] = 1.0;
    want[EDGE_BASE + EDGE_TRAIL + 1] = 1.0;
    // Threats: [trail / 15, roads to award / 5, knights to award / 3, points to win / 10,
    // settlements left / 5, cities left / 4]. Player 3 holds the longest road with a trail of 1, so
    // a challenger needs max(5, 1 + 1) - own; nobody holds the largest army, so max(0, 3 - own).
    let at = |r: usize| THREATS + r * THREAT_WIDTH;
    // r0 = player 2: trail 2; 5 - 2 = 3 roads; 3 - 1 knight = 2; a settlement, a city and its VP
    // card = 4 points, 6 to go; 4 settlements and 3 cities left.
    want[at(0)..at(0) + 6].copy_from_slice(&[
        2.0 / 15.0,
        3.0 / 5.0,
        2.0 / 3.0,
        6.0 / 10.0,
        4.0 / 5.0,
        3.0 / 4.0,
    ]);
    // r1 = player 3: trail 1, the holder (0); 3 knights; a settlement and the road = 3 public
    // points, 7 to go; 4 settlements, 4 cities.
    want[at(1)..at(1) + 6].copy_from_slice(&[
        1.0 / 15.0,
        0.0,
        3.0 / 3.0,
        7.0 / 10.0,
        4.0 / 5.0,
        4.0 / 4.0,
    ]);
    // r2 = player 0: no road, 5 - 0 = 5; 3 knights; a city = 2 points, 8 to go; 5 settlements, 3 cities.
    want[at(2)..at(2) + 6].copy_from_slice(&[
        0.0,
        5.0 / 5.0,
        3.0 / 3.0,
        8.0 / 10.0,
        5.0 / 5.0,
        3.0 / 4.0,
    ]);
    // r3 = player 1: nothing: 5 roads, 3 knights, 10 points to go, every piece left.
    want[at(3)..at(3) + 6].copy_from_slice(&[
        0.0,
        5.0 / 5.0,
        3.0 / 3.0,
        10.0 / 10.0,
        5.0 / 5.0,
        4.0 / 4.0,
    ]);
```

Replace `fn rotating_every_seat_and_the_actor_together_gives_the_same_features` with:

```rust
#[test]
fn rotating_every_seat_and_the_actor_together_gives_the_same_features() {
    let (mut checked, mut moved, mut trades) = (0, 0, 0);
    // Positions where the actor is not the current player, and Road Building positions.
    let (mut off_turn, mut road_building) = (0, 0);
    // Positions where v4's blocks say something about an opponent: a node one or two roads from
    // it, a road on its longest trail, a threat block unlike the viewer's.
    let (mut near, mut trails, mut threats) = (0, 0, 0);
    for s in random_positions(0..4, 7, GameConfig::default()) {
        let actor = s.current_actor();
        let obs = s.observation(actor);
        let base = public(&obs);
        let hands = std::array::from_fn(|p| s.players[p].hand);
        let devs = std::array::from_fn(|p| s.players[p].dev_hand);
        let mut hidden = vec![0.0; HIDDEN_LEN];
        encode_hidden_from(&hands, &devs, actor, &mut hidden);
        trades += obs.trade.is_some() as u32;
        off_turn += (obs.current != obs.viewer) as u32;
        road_building += matches!(obs.phase, Phase::RoadBuilding { .. }) as u32;
        let d = got_distances(&base);
        near += (1..NUM_PLAYERS).any(|r| d.iter().any(|x| x[r] > 0.0 && x[r] < 1.0)) as u32;
        trails += (1..NUM_PLAYERS).any(|r| !trail_edges(&base, r).is_empty()) as u32;
        threats += (1..NUM_PLAYERS).any(|r| threat(&base, r) != threat(&base, 0)) as u32;
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
    assert!(
        checked > 500
            && moved > 0
            && trades > 0
            && off_turn > 0
            && road_building > 0
            && near > 0
            && trails > 0
            && threats > 0,
        "{checked} {moved} {trades} {off_turn} {road_building} {near} {trails} {threats}"
    );
}
```

In `fn the_trade_block_encodes_exactly`, replace `assert_eq!(&got[TRADE_GIVE..PUBLIC_LEN], &want[..]);` with `assert_eq!(&got[TRADE_GIVE..THREATS], &want[..]);` and `assert!(zero[TRADE_GIVE..PUBLIC_LEN].iter().all(|&x| x == 0.0));` with `assert!(zero[TRADE_GIVE..THREATS].iter().all(|&x| x == 0.0));`.

Append to `net/tests/encode.rs`:

```rust
/// A built position for v4's blocks, seen by player 1 (r0 = 1, r1 = 2, r2 = 3, r3 = 0).
/// Player 0: a settlement on node 4 and a Y of roads from it, edges 1 (0-4), 2 (1-4), 7 (4-8).
/// Player 1: settlements on nodes 5 and 13, roads 3 (1-5) and 8 (5-9), 2 knights, a VP card.
/// Player 2: a city on node 30, 3 knights, the largest army.
/// Player 3: a settlement on node 47 and the path 47-51-48-52-49-53 (edges 66 to 70), the
/// longest road, 1 knight, a VP card (hidden from player 1).
fn threats_state() -> State {
    let mut s = State::new(42, GameConfig::default());
    s.setup_step = 8;
    s.phase = Phase::Main;
    s.current = 1;
    s.longest_road_owner = Some(3);
    s.largest_army_owner = Some(2);
    let p0 = &mut s.players[0];
    p0.settlements = 1 << 4;
    p0.roads = edges(&[1, 2, 7]);
    let p1 = &mut s.players[1];
    p1.settlements = (1 << 5) | (1 << 13);
    p1.roads = edges(&[3, 8]);
    p1.knights_played = 2;
    p1.dev_played = [2, 0, 0, 0, 0];
    p1.dev_hand = [0, 1, 0, 0, 0];
    let p2 = &mut s.players[2];
    p2.cities = 1 << 30;
    p2.knights_played = 3;
    p2.dev_played = [3, 0, 0, 0, 0];
    let p3 = &mut s.players[3];
    p3.settlements = 1 << 47;
    p3.roads = edges(&[66, 67, 68, 69, 70]);
    p3.knights_played = 1;
    p3.dev_played = [1, 0, 0, 0, 0];
    p3.dev_hand = [0, 1, 0, 0, 0];
    s
}

#[test]
fn the_road_and_threat_blocks_encode_exactly() {
    let s = threats_state();
    let mut got = vec![f32::NAN; PUBLIC_LEN];
    encode_public(&s.observation(1), &mut got);
    // Road distances. Occupied: 4 (player 0), 5 and 13 (player 1), 30 (player 2), 47 (player 3).
    // r0 = player 1, network {1, 5, 9, 13}. Node 1's edge 2 (to 4) and node 8's edge 7 (to 4) are
    //   player 0's roads, so node 4 is never reached. One road: 2 (from 5), 14 (from 9), 8, 18
    //   (from 13). Two: 6 (from 2), 10, 19 (from 14), 12 (from 8), 23, 24 (from 18).
    // r1 = player 2, network {30}. One: 24, 35, 36. Two: 18, 19 (from 24), 29, 40 (from 35), 31,
    //   41 (from 36).
    // r2 = player 3, network {47, 48, 49, 51, 52, 53}. One: 43 (from 47), 44 (from 48), 45 (from
    //   49), 50 (from 53). Two: 38, 39 (from 43), 40 (from 44), 41 (from 45), 46 (from 50).
    // r3 = player 0, network {0, 1, 4, 8}. One: 3 (from 0), 12, 13 (from 8); edge 3 (1-5) is player
    //   1's road, so 5 is not reached from 1 (blocked path: otherwise 1 road). Two: 7 (from 3 and
    //   12), 17 (from 12). Node 13 holds player 1's settlement: reached, but no path leaves it, so 9
    //   and 18 stay 3+ roads away (otherwise 2).
    assert_eq!(
        got_distances(&got),
        want_distances([
            [&[1, 5, 9, 13], &[2, 8, 14, 18], &[6, 10, 12, 19, 23, 24]],
            [&[30], &[24, 35, 36], &[18, 19, 29, 31, 40, 41]],
            [
                &[47, 48, 49, 51, 52, 53],
                &[43, 44, 45, 50],
                &[38, 39, 40, 41, 46]
            ],
            [&[0, 1, 4, 8], &[3, 12, 13], &[7, 17]],
        ])
    );
    // Open spots: all but the 5 occupied nodes and their neighbours (4: 0, 1, 8; 5: 1, 2, 9;
    // 13: 8, 9, 18; 30: 24, 35, 36; 47: 43, 51): 54 - 16 = 38 nodes.
    let closed = [0, 1, 2, 4, 5, 8, 9, 13, 18, 24, 30, 35, 36, 43, 47, 51];
    let open: Vec<usize> = (0..NUM_NODES).filter(|n| !closed.contains(n)).collect();
    assert_eq!(open_nodes(&got), open);
    // Longest trails. Player 0's Y has three trails of 2 through node 4; the walk starts at node 0
    // (an end of the lowest edge, 1) and turns onto edge 2 before edge 7: 0-4-1 is kept (the tie).
    assert_eq!(trail_edges(&got, 0), [3, 8]);
    assert!(trail_edges(&got, 1).is_empty());
    assert_eq!(trail_edges(&got, 2), [66, 67, 68, 69, 70]);
    assert_eq!(trail_edges(&got, 3), [1, 2]);
    // Threats: [trail / 15, roads to award / 5, knights to award / 3, points to win / 10,
    // settlements left / 5, cities left / 4]. Player 3 holds the road with 5: a challenger needs
    // max(5, 5 + 1) - own, at most 5. Player 2 holds the army with 3: max(3, 3 + 1) - own, at most 3.
    // r0 = player 1: trail 2, 6 - 2 = 4 roads; 4 - 2 = 2 knights; 2 settlements + its VP card = 3
    // points, 7 to go; 3 settlements, 4 cities left.
    assert_eq!(
        threat(&got, 0),
        [2.0 / 15.0, 4.0 / 5.0, 2.0 / 3.0, 7.0 / 10.0, 3.0 / 5.0, 4.0 / 4.0]
    );
    // r1 = player 2: no road, 6 - 0 = 6, capped at 5; the army's holder, 0; a city and the army =
    // 4 public points, 6 to go; 5 settlements, 3 cities left.
    assert_eq!(
        threat(&got, 1),
        [0.0, 5.0 / 5.0, 0.0, 6.0 / 10.0, 5.0 / 5.0, 3.0 / 4.0]
    );
    // r2 = player 3: trail 5, the road's holder, 0; 4 - 1 = 3 knights; a settlement and the road =
    // 3 public points (its VP card stays hidden), 7 to go; 4 settlements, 4 cities left.
    assert_eq!(
        threat(&got, 2),
        [5.0 / 15.0, 0.0, 3.0 / 3.0, 7.0 / 10.0, 4.0 / 5.0, 4.0 / 4.0]
    );
    // r3 = player 0: trail 2, 4 roads; 4 - 0 = 4, capped at 3; a settlement = 1 point, 9 to go;
    // 4 settlements, 4 cities left.
    assert_eq!(
        threat(&got, 3),
        [2.0 / 15.0, 4.0 / 5.0, 3.0 / 3.0, 9.0 / 10.0, 4.0 / 5.0, 4.0 / 4.0]
    );
}

#[test]
fn an_empty_board_reaches_nothing_and_every_spot_is_open() {
    let s = State::new(7, GameConfig::default());
    let mut got = vec![f32::NAN; PUBLIC_LEN];
    encode_public(&s.observation(0), &mut got);
    assert_eq!(got_distances(&got), vec![[1.0f32; NUM_PLAYERS]; NUM_NODES]);
    assert_eq!(open_nodes(&got), (0..NUM_NODES).collect::<Vec<_>>());
    for r in 0..NUM_PLAYERS {
        assert!(trail_edges(&got, r).is_empty(), "seat {r}");
        // No road; 5 - 0 roads and 3 - 0 knights to go; 10 of 10 points; every piece left.
        assert_eq!(threat(&got, r), [0.0, 1.0, 1.0, 1.0, 1.0, 1.0], "seat {r}");
    }
}

#[test]
fn a_road_into_an_opponents_building_reaches_it_but_not_past_it() {
    // Player 0's roads 0 (0-3) and 6 (3-7) end at node 7, where player 1 has built.
    let mut s = State::new(42, GameConfig::default());
    s.setup_step = 8;
    s.phase = Phase::Main;
    s.players[0].roads = edges(&[0, 6]);
    s.players[1].settlements = 1 << 7;
    let mut got = vec![f32::NAN; PUBLIC_LEN];
    encode_public(&s.observation(0), &mut got);
    // r0 = player 0: network {0, 3, 7} (node 7 is touched: 0 roads), but no path leaves node 7.
    // One road: 4 (from 0). Two: 1, 8 (from 4). Nodes 11 and 12, beside node 7, stay 3+ away.
    // r1 = player 1: network {7}; edge 6 is player 0's road. One road: 11, 12. Two: 16 (from 11),
    // 8, 17 (from 12). Node 3 is never reached.
    assert_eq!(
        got_distances(&got),
        want_distances([
            [&[0, 3, 7], &[4], &[1, 8]],
            [&[7], &[11, 12], &[8, 16, 17]],
            [&[], &[], &[]],
            [&[], &[], &[]],
        ])
    );
    // The trail may end at player 1's settlement: 0-3-7, length 2.
    assert_eq!(trail_edges(&got, 0), [0, 6]);
    assert_eq!(got[THREATS], 2.0 / 15.0);
}

#[test]
fn an_unowned_award_past_its_minimum_reads_zero_to_go() {
    // Players 0 and 1 have trails of 5 and nobody holds the award (as after a broken holder with
    // tied challengers): max(0, 5 - 5) = 0 roads to go for both, as the spec writes it; players 2
    // and 3 have none, max(0, 5 - 0) = 5.
    let mut s = State::new(42, GameConfig::default());
    s.setup_step = 8;
    s.phase = Phase::Main;
    s.players[0].roads = edges(&[0, 6, 10, 18, 23]); // 0-3-7-11-16-21
    s.players[1].roads = edges(&[66, 67, 68, 69, 70]); // 47-51-48-52-49-53
    let mut got = vec![f32::NAN; PUBLIC_LEN];
    encode_public(&s.observation(0), &mut got);
    let slots = |r: usize| (threat(&got, r)[0], threat(&got, r)[1]);
    assert_eq!(
        [slots(0), slots(1), slots(2), slots(3)],
        [(5.0 / 15.0, 0.0), (5.0 / 15.0, 0.0), (0.0, 1.0), (0.0, 1.0)]
    );
}
```

In `net/tests/graph.rs`, replace

```rust
#[test]
fn the_global_block_is_the_115_scalars_after_the_tiles() {
    assert_eq!((GLOBAL_BASE, GLOBAL_WIDTH), (OWN_HAND, 115));
```

with

```rust
#[test]
fn the_global_block_is_the_139_scalars_after_the_tiles() {
    // v3's 115 and the threat block's 4 x 6.
    assert_eq!((GLOBAL_BASE, GLOBAL_WIDTH), (OWN_HAND, 139));
```

In `net/tests/weights.rs`:
- replace `(3, 19), (3, 4), (3, 8), (3, 115),` with `(3, 24), (3, 8), (3, 8), (3, 139),`;
- replace both `248_446` with `250_030` (v3's 248,446 + 48 × (5 + 4 + 24) = 248,446 + 1,584);
- in `fn a_hand_computed_tiny_network` replace `sparse(d, 19, &[], &[1.0, 2.0]),  // every node (1, 2)` with `sparse(d, 24, &[], &[1.0, 2.0]),  // every node (1, 2)`, `sparse(d, 4, &[], &[3.0, 4.0]),   // every edge (3, 4)` with `sparse(d, 8, &[], &[3.0, 4.0]),   // every edge (3, 4)` and `sparse(d, 115, &[], &[7.0, 8.0]), // global (7, 8)` with `sparse(d, 139, &[], &[7.0, 8.0]), // global (7, 8)` (every encoder weight is zero, so the expected outputs do not change);
- in `fn an_encoder_v2_file_is_refused_with_the_way_to_upgrade_it`, replace

```rust
    assert_eq!(
        settler_net::graph::GLOBAL_WIDTH - V2_GLOBAL_WIDTH,
        24,
        "the trade block"
    );
```

with

```rust
    assert_eq!(V3_GLOBAL_WIDTH - V2_GLOBAL_WIDTH, 24, "the trade block");
    assert_eq!(
        settler_net::graph::GLOBAL_WIDTH - V3_GLOBAL_WIDTH,
        24,
        "the threat block"
    );
    assert_eq!(
        (NODE_WIDTH - V3_NODE_WIDTH, EDGE_WIDTH - V3_EDGE_WIDTH),
        (5, 4)
    );
    assert_ne!(V3_ENCODING_HASH, ENCODING_HASH);
```

- append:

```rust
/// `random_v3` is v3's net with zero weights on v4's inputs: whatever those inputs hold, the
/// outputs are bit for bit the same (same widths, same order of sums, and every product with a
/// zero weight is exactly zero).
#[test]
fn a_random_v3_net_ignores_every_input_v4_added() {
    let w = Weights::random_v3(SMALL, 4);
    let shapes: Vec<(usize, usize)> = w.layers.iter().map(|l| (l.rows, l.cols)).collect();
    assert_eq!(shapes, SMALL.shapes());
    for (i, from) in [
        (NODE_ENCODER, V3_NODE_WIDTH),
        (EDGE_ENCODER, V3_EDGE_WIDTH),
        (GLOBAL_ENCODER, V3_GLOBAL_WIDTH),
    ] {
        let l = &w.layers[i];
        assert!(l.w.chunks_exact(l.cols).all(|row| row[from..].iter().all(|&x| x == 0.0)));
        assert!(l.w.chunks_exact(l.cols).any(|row| row[..from].iter().any(|&x| x != 0.0)));
    }
    let (n, public, hidden) = inputs(40);
    let mut blanked = public.clone();
    for row in blanked.chunks_exact_mut(PUBLIC_LEN) {
        for node in 0..NUM_NODES {
            let at = NODE_BASE + node * NODE_WIDTH;
            row[at + V3_NODE_WIDTH..at + NODE_WIDTH].fill(0.0);
        }
        for e in 0..NUM_EDGES {
            let at = EDGE_BASE + e * EDGE_WIDTH;
            row[at + V3_EDGE_WIDTH..at + EDGE_WIDTH].fill(0.0);
        }
        row[THREATS..].fill(0.0);
    }
    assert_ne!(blanked, public, "the positions fill v4's inputs");
    let (mut a, mut b) = (Outputs::default(), Outputs::default());
    Forward::new().run(&w, &public, &hidden, n, &mut a);
    Forward::new().run(&w, &blanked, &hidden, n, &mut b);
    assert_eq!(a, b);
    assert_eq!(
        Layer {
            rows: 1,
            cols: 2,
            w: vec![1.0, 2.0],
            b: vec![3.0]
        }
        .widened(4),
        Layer {
            rows: 1,
            cols: 4,
            w: vec![1.0, 2.0, 0.0, 0.0],
            b: vec![3.0]
        }
    );
}
```

In `bots/tests/netbot.rs`, after `fn nets_dir` add:

```rust
/// A fresh nets directory holding `tiny.bin`: encoder v3's random net of width 8, 1 round, seed
/// 1 (the net the P2 pin was computed with), upgraded to encoder v4 with zero weights for the
/// new inputs (`Weights::random_v3`), so it plays as it did under v3.
fn v3_nets_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("settler-bots-nets-{}-{tag}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    Weights::random_v3(
        Arch {
            width: 8,
            rounds: 1,
        },
        1,
    )
    .save(&dir.join("tiny.bin"))
    .unwrap();
    dir
}
```

and replace

```rust
/// The digest of a trading-game match of `net:tiny@16+t` (random weights, seed 1), computed at
/// eb1f739 before the Gumbel root (P2): net bots without `+g` must play the same games.
#[test]
fn net_bots_without_the_gumbel_suffix_play_the_games_they_played_before_p2() {
    let dir = nets_dir("p2-pin");
```

with

```rust
/// The digest of a trading-game match of `net:tiny@16+t` (random weights, seed 1), computed at
/// eb1f739 before the Gumbel root (P2): net bots without `+g` must play the same games. Encoder
/// v4 (3d-2) added inputs; the pin's net is the same v3 net upgraded (`v3_nets_dir`), which
/// ignores them. Its wider sums round differently (by a few ulps), which moves no move of this
/// match: checked while planning 3d-2 by running encoder v3 with the padded encoder layers.
#[test]
fn net_bots_without_the_gumbel_suffix_play_the_games_they_played_before_p2() {
    let dir = v3_nets_dir("p2-pin");
```

- [ ] **Step 2: Run them to verify they fail**

Run: `PATH=$HOME/.cargo/bin:$PATH cargo test --release -p settler-net -p settler-bots --no-run`
Expected: FAIL to compile with "cannot find value `NODE_DISTANCE`" (and `THREATS`, `V3_ENCODING_HASH`, `random_v3`, ...).

- [ ] **Step 3: Implement the encoder**

In `net/src/encode.rs`, replace the imports

```rust
use settler_engine::topology::{topo, NUM_EDGES, NUM_NODES, NUM_TILES};
use settler_engine::types::*;
use settler_engine::{Observation, Phase, Response, State, ACTION_SPACE_SIZE};
```

with

```rust
use settler_engine::rules::awards::longest_trail;
use settler_engine::topology::{topo, NUM_EDGES, NUM_NODES, NUM_TILES};
use settler_engine::types::*;
use settler_engine::{Observation, Phase, Response, State, ACTION_SPACE_SIZE};
```

Replace `pub const ENCODER_VERSION: u32 = 3;` through `pub const EDGE_WIDTH: usize = 4;` with:

```rust
pub const ENCODER_VERSION: u32 = 4;

pub const NODE_WIDTH: usize = 24;
pub const EDGE_WIDTH: usize = 8;
/// v4 (spec 3d-2): each node's road distance for r0 .. r3 at these 4 slots, then the open spot.
pub const NODE_DISTANCE: usize = 19;
pub const NODE_OPEN: usize = NODE_DISTANCE + NUM_PLAYERS;
/// v4: each edge's "on r's longest trail" for r0 .. r3 at these 4 slots, after the road owners.
pub const EDGE_TRAIL: usize = 4;
```

Replace `pub const PUBLIC_LEN: usize = OFFERS_ALLOWED + 1;` with:

```rust
/// v4: per seat (seat-relative), longest road length / 15, roads to the longest-road award / 5,
/// knights to the largest-army award / 3, points to win / vp_to_win, settlements left / 5, cities
/// left / 4.
pub const THREATS: usize = OFFERS_ALLOWED + 1;
pub const THREAT_WIDTH: usize = 6;
pub const PUBLIC_LEN: usize = THREATS + NUM_PLAYERS * THREAT_WIDTH;
```

After `pub const FREE_ROAD_DEN: u32 = 2;` add:

```rust
/// v4: road distances are capped here (3 or more roads, or never, read 1).
pub const DISTANCE_DEN: u32 = 3;
/// v4: a seat's longest road: at most its 15 roads.
pub const ROAD_LENGTH_DEN: u32 = MAX_ROADS;
/// v4: the longest road is won at 5 roads, the largest army at 3 knights; "to the award" is over
/// these.
pub const ROAD_AWARD_MIN: u32 = 5;
pub const ARMY_AWARD_MIN: u32 = 3;
/// v4: pieces in supply.
pub const SETTLEMENT_DEN: u32 = MAX_SETTLEMENTS;
pub const CITY_DEN: u32 = MAX_CITIES;
```

Replace `pub const LAYOUT: [u32; 27] = [` with `pub const LAYOUT: [u32; 33] = [` and after its last entry `FREE_ROAD_DEN,` add:

```rust
    DISTANCE_DEN,
    ROAD_LENGTH_DEN,
    ROAD_AWARD_MIN,
    ARMY_AWARD_MIN,
    SETTLEMENT_DEN,
    CITY_DEN,
```

In `LAYOUT_DESCRIPTION`, make these replacements (every other line stays):
- `settler-net features v3` → `settler-net features v4`
- the scales line's end `dev_year_of_plenty=2 dev_monopoly=2 vp=10 discard=10 bank=19 deck=25 pip=5 node_pip=15 free_roads=2` → `dev_year_of_plenty=2 dev_monopoly=2 vp=10 discard=10 bank=19 deck=25 pip=5 node_pip=15 free_roads=2 \` followed by a new line `distance=3 road_length=15 road_award=5 army_award=3 settlements=5 cities=4`
- the node line's end `pips_brick/node_pip pips_sheep/node_pip pips_wheat/node_pip pips_ore/node_pip` → `pips_brick/node_pip pips_sheep/node_pip pips_wheat/node_pip pips_ore/node_pip \` followed by `road_distance_r0/distance road_distance_r1/distance road_distance_r2/distance \` and `road_distance_r3/distance open_spot`
- `public edge x72: road_r0 road_r1 road_r2 road_r3` → `public edge x72: road_r0 road_r1 road_r2 road_r3 longest_trail_r0 longest_trail_r1 longest_trail_r2 \` followed by `longest_trail_r3`
- after `public offers_allowed x1: allowed` add the lines `public threat x4: road_length/road_length roads_to_award/road_award knights_to_award/army_award \` and `points_to_win/vp_to_win settlements_left/settlements cities_left/cities`

so that those parts read:

```text
scales: card=19 hand=20 dev_total=25 dev_knight=14 dev_victory_point=5 dev_road_building=2 \
dev_year_of_plenty=2 dev_monopoly=2 vp=10 discard=10 bank=19 deck=25 pip=5 node_pip=15 free_roads=2 \
distance=3 road_length=15 road_award=5 army_award=3 settlements=5 cities=4
public node x54: settlement_r0 settlement_r1 settlement_r2 settlement_r3 city_r0 city_r1 city_r2 \
city_r3 port_generic port_wood port_brick port_sheep port_wheat port_ore pips_wood/node_pip \
pips_brick/node_pip pips_sheep/node_pip pips_wheat/node_pip pips_ore/node_pip \
road_distance_r0/distance road_distance_r1/distance road_distance_r2/distance \
road_distance_r3/distance open_spot
public edge x72: road_r0 road_r1 road_r2 road_r3 longest_trail_r0 longest_trail_r1 longest_trail_r2 \
longest_trail_r3
```

and

```text
public offers_allowed x1: allowed
public threat x4: road_length/road_length roads_to_award/road_award knights_to_award/army_award \
points_to_win/vp_to_win settlements_left/settlements cities_left/cities
hidden opponent x3: wood/card brick/card sheep/card wheat/card ore/card \
```

(A `\` at a line's end in the Rust string joins the next line; the space before it stays, so the slots stay space-separated.)

After `fn scaled` add:

```rust
/// Every node seat `p`'s roads or buildings touch: its network (v4).
fn network(obs: &Observation, p: usize) -> u64 {
    let t = topo();
    bits128(obs.roads[p]).fold(obs.settlements[p] | obs.cities[p], |m, e| {
        m | t.edge_node_mask[e as usize]
    })
}

/// v4's road distances: the fewest roads that make `network` touch each node, capped at
/// `DISTANCE_DEN`. A network node is 0, a node another seat built on included; a path may not
/// leave a `blocked` node (another seat's building) or use a `closed` edge (another seat's road).
fn road_distances(network: u64, blocked: u64, closed: u128) -> [u8; NUM_NODES] {
    let t = topo();
    let cap = DISTANCE_DEN as u8;
    let mut dist = [cap; NUM_NODES];
    for n in bits64(network) {
        dist[n as usize] = 0;
    }
    let (mut seen, mut frontier) = (network, network);
    // Rings 1 .. cap - 1; every node further out keeps the cap.
    for d in 1..cap {
        let mut next = 0u64;
        for n in bits64(frontier & !blocked) {
            for e in bits128(t.node_edge_mask[n as usize] & !closed) {
                next |= t.edge_node_mask[e as usize];
            }
        }
        next &= !seen;
        for n in bits64(next) {
            dist[n as usize] = d;
        }
        seen |= next;
        frontier = next;
    }
    dist
}

/// v4: pieces seat `p` still lacks to take an award won at `min` (5 roads, 3 knights), holding
/// `own`: 0 for the holder, else `min(min, max(min, held + 1) - own)` with `held` the holder's
/// count, and `max(0, min - own)` with no holder (spec 3d-2 Section 2).
fn to_award(own: u32, p: usize, holder: Option<(usize, u32)>, min: u32) -> u32 {
    match holder {
        Some((h, _)) if h == p => 0,
        Some((_, held)) => min.max(held + 1).saturating_sub(own).min(min),
        None => min.saturating_sub(own),
    }
}
```

At the end of `encode_public`, after the `if max_offers > 0 { ... }` block, add:

```rust
    // v4 (spec 3d-2 Section 2): road distances, open spots, longest trails, the threat block.
    let buildings: [u64; NUM_PLAYERS] = std::array::from_fn(|p| obs.settlements[p] | obs.cities[p]);
    let occupied = buildings.iter().fold(0u64, |m, b| m | b);
    let roads = obs.roads.iter().fold(0u128, |m, r| m | r);
    // The trail `awards::longest_road` measures: another seat's building ends it.
    let trails: [(u8, u128); NUM_PLAYERS] =
        std::array::from_fn(|p| longest_trail(obs.roads[p], occupied & !buildings[p]));
    for p in 0..NUM_PLAYERS {
        let r = rel(p as PlayerId, v);
        let dist = road_distances(
            network(obs, p),
            occupied & !buildings[p],
            roads & !obs.roads[p],
        );
        for (n, &d) in dist.iter().enumerate() {
            out[NODE_BASE + n * NODE_WIDTH + NODE_DISTANCE + r] = scaled(d as u32, DISTANCE_DEN);
        }
        for e in bits128(trails[p].1) {
            out[EDGE_BASE + e as usize * EDGE_WIDTH + EDGE_TRAIL + r] = 1.0;
        }
    }
    for n in 0..NUM_NODES {
        if occupied & ((1u64 << n) | t.node_neighbor_mask[n]) == 0 {
            out[NODE_BASE + n * NODE_WIDTH + NODE_OPEN] = 1.0;
        }
    }
    let road_holder = obs
        .longest_road_owner
        .map(|h| (h as usize, trails[h as usize].0 as u32));
    let army_holder = obs
        .largest_army_owner
        .map(|h| (h as usize, obs.knights_played[h as usize] as u32));
    let vp_to_win = obs.config.vp_to_win as u32;
    for p in 0..NUM_PLAYERS {
        let at = THREATS + rel(p as PlayerId, v) * THREAT_WIDTH;
        let length = trails[p].0 as u32;
        // The viewer knows its VP cards; an opponent's are not in the observation.
        let mut points = obs.public_vp[p] as u32;
        if p == v as usize {
            points += obs.my_dev_cards[DevCard::VictoryPoint.index()] as u32;
        }
        let knights = obs.knights_played[p] as u32;
        out[at] = scaled(length, ROAD_LENGTH_DEN);
        out[at + 1] = scaled(
            to_award(length, p, road_holder, ROAD_AWARD_MIN),
            ROAD_AWARD_MIN,
        );
        out[at + 2] = scaled(
            to_award(knights, p, army_holder, ARMY_AWARD_MIN),
            ARMY_AWARD_MIN,
        );
        out[at + 3] = scaled(vp_to_win.saturating_sub(points), vp_to_win);
        out[at + 4] = scaled(
            MAX_SETTLEMENTS.saturating_sub(obs.settlements[p].count_ones()),
            SETTLEMENT_DEN,
        );
        out[at + 5] = scaled(
            MAX_CITIES.saturating_sub(obs.cities[p].count_ones()),
            CITY_DEN,
        );
    }
```

(`t` is the `let t = topo();` at the top of `encode_public`; the `if let Some(t) = obs.trade` block's `t` ends with that block. `vp_to_win` is at least 3, `GameConfig::validate`.)

In `net/src/graph.rs` replace the comment `/// The global scalars: every slot after the tile block, from the actor's own hand on (115: encoder v3's globals, the trade block included).` with `/// The global scalars: every slot after the tile block, from the actor's own hand on (139: encoder v4's globals, the trade and threat blocks included).`

- [ ] **Step 4: Implement the weights file's v3 constants and `random_v3`**

In `net/src/weights.rs`:
- in the doc table replace `| 0 | node encoder | d × 19 | a node's 19 features |` with `| 0 | node encoder | d × 24 | a node's 24 features |`, `| 1 | edge encoder | d × 4 | an edge's 4 features |` with `| 1 | edge encoder | d × 8 | an edge's 8 features |` and `| 3 | global encoder | d × 115 | the 115 global scalars (\`graph::GLOBAL_BASE..PUBLIC_LEN\`) |` with `| 3 | global encoder | d × 139 | the 139 global scalars (\`graph::GLOBAL_BASE..PUBLIC_LEN\`) |`;
- after `pub const V2_GLOBAL_WIDTH: usize = 91;` add:

```rust
/// Encoder v3's `ENCODING_HASH` (3c's and P2's nets, before the road and threat inputs) and its
/// input widths. Encoder v2's node and edge inputs are v3's. `alphasettler net-upgrade` rewrites
/// such files for this encoder.
pub const V3_ENCODING_HASH: u64 = 0x6d02_356c_b617_53c2;
pub const V3_NODE_WIDTH: usize = 19;
pub const V3_EDGE_WIDTH: usize = 4;
pub const V3_GLOBAL_WIDTH: usize = 115;
```

- replace `pub fn shapes(&self) -> Vec<(usize, usize)> {` and the four lines after it

```rust
    /// (rows, cols) of each layer in file order; rows are outputs.
    pub fn shapes(&self) -> Vec<(usize, usize)> {
        let d = self.width;
        let mut s = vec![
            (d, NODE_WIDTH),
            (d, EDGE_WIDTH),
            (d, TILE_WIDTH),
            (d, GLOBAL_WIDTH),
        ];
```

with

```rust
    /// (rows, cols) of each layer in file order; rows are outputs.
    pub fn shapes(&self) -> Vec<(usize, usize)> {
        self.shapes_for(NODE_WIDTH, EDGE_WIDTH, GLOBAL_WIDTH)
    }

    /// `shapes` for encoders of `node`, `edge` and `global` inputs (an older encoder's); the tile
    /// encoder and every later layer are fixed.
    pub fn shapes_for(&self, node: usize, edge: usize, global: usize) -> Vec<(usize, usize)> {
        let d = self.width;
        let mut s = vec![(d, node), (d, edge), (d, TILE_WIDTH), (d, global)];
```

- after `pub struct Layer { ... }` add:

```rust
impl Layer {
    /// This layer with `cols` inputs: its own first, then zero weights for the inputs appended
    /// after them (how `net-upgrade` widens an encoder). Panics below its own width.
    pub fn widened(&self, cols: usize) -> Layer {
        assert!(
            cols >= self.cols,
            "cannot narrow a layer from {} to {cols} inputs",
            self.cols
        );
        let mut w = Vec::with_capacity(self.rows * cols);
        for row in self.w.chunks_exact(self.cols) {
            w.extend_from_slice(row);
            w.resize(w.len() + cols - self.cols, 0.0);
        }
        Layer {
            rows: self.rows,
            cols,
            w,
            b: self.b.clone(),
        }
    }
}
```

- replace `pub fn random(arch: Arch, seed: u64) -> Weights {` and its body with:

```rust
    pub fn random(arch: Arch, seed: u64) -> Weights {
        Weights::random_shaped(arch, seed, arch.shapes())
    }

    /// Encoder v3's `Weights::random(arch, seed)` upgraded to this encoder as `net-upgrade`
    /// upgrades a v3 file: the same draws over v3's layer shapes, then zero weights for every
    /// input v4 added. It ignores those inputs, so it plays as the v3 net did (within the forward
    /// pass's 1e-5: the wider sums round differently). For tests whose pinned games predate v4.
    pub fn random_v3(arch: Arch, seed: u64) -> Weights {
        let v3 = arch.shapes_for(V3_NODE_WIDTH, V3_EDGE_WIDTH, V3_GLOBAL_WIDTH);
        let mut w = Weights::random_shaped(arch, seed, v3);
        for (i, cols) in [
            (NODE_ENCODER, NODE_WIDTH),
            (EDGE_ENCODER, EDGE_WIDTH),
            (GLOBAL_ENCODER, GLOBAL_WIDTH),
        ] {
            w.layers[i] = w.layers[i].widened(cols);
        }
        w
    }

    fn random_shaped(arch: Arch, seed: u64, shapes: Vec<(usize, usize)>) -> Weights {
        if let Err(e) = arch.check() {
            panic!("{e}");
        }
        let mut rng = Rng::new(seed);
        let layers = shapes
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
```

(`random`'s doc comment above it stays. The draws are the old body's, in the same layer order.)

In `bindings/src/train.rs` replace

```rust
use settler_net::weights::{
    Arch, Weights, FORMAT_VERSION, MAX_ROUNDS, MAX_WIDTH, V2_ENCODING_HASH, V2_GLOBAL_WIDTH,
};
```

with

```rust
use settler_net::weights::{
    Arch, Weights, FORMAT_VERSION, MAX_ROUNDS, MAX_WIDTH, V2_ENCODING_HASH, V2_GLOBAL_WIDTH,
    V3_EDGE_WIDTH, V3_ENCODING_HASH, V3_GLOBAL_WIDTH, V3_NODE_WIDTH,
};
```

and after `m.add("NET_V2_GLOBAL_WIDTH", V2_GLOBAL_WIDTH)?;` add:

```rust
    m.add("NET_V3_ENCODING_HASH", V3_ENCODING_HASH)?;
    m.add("NET_V3_NODE_WIDTH", V3_NODE_WIDTH)?;
    m.add("NET_V3_EDGE_WIDTH", V3_EDGE_WIDTH)?;
    m.add("NET_V3_GLOBAL_WIDTH", V3_GLOBAL_WIDTH)?;
```

- [ ] **Step 5: Run the Rust tests to verify they pass**

Run: `PATH=$HOME/.cargo/bin:$PATH cargo test --release -p settler-net -p settler-bots -p settler-engine`
Expected: PASS, `net_bots_without_the_gumbel_suffix_play_the_games_they_played_before_p2` included (digest `0xf4d5c47b40f849a3`). If that pin fails, stop and report BLOCKED with the new digest: it must not be re-pinned.

- [ ] **Step 6: Write the failing Python tests**

Replace `tests/python/test_net_upgrade.py` with:

```python
"""`alphasettler net-upgrade` (specs 3c Section 4 and 3d-2 Section 3): an older net rewritten for
this build's encoder, v4. Needs the `train` extra for the PyTorch twin, like test_train.py."""

import random
import struct
import subprocess
import sys

import pytest
import torch

from alphasettler import Bot, Game, describe_action, netformat
from alphasettler._engine import (
    NET_GLOBAL_WIDTH, NET_HIDDEN_LEN, NET_PUBLIC_LEN, NET_V2_ENCODING_HASH, NET_V2_GLOBAL_WIDTH, NET_V3_EDGE_WIDTH,
    NET_V3_GLOBAL_WIDTH, NET_V3_NODE_WIDTH, encode, net_forward,
)
from alphasettler import netupgrade
from alphasettler.netupgrade import ANSWER_IDS, OFFER_BIAS, OFFER_IDS, TRADE_SLOTS, pooled_row, upgrade
from alphasettler.train.model import ACTIONS, MAGIC, TILE_END, Net, _floats, from_bytes, shapes, to_bytes

NO_TRADES = {"max_offers_per_turn": 0}
# (encoding hash, node, edge and global input widths) of encoder v2; its node and edge inputs are v3's.
V2 = (NET_V2_ENCODING_HASH, NET_V3_NODE_WIDTH, NET_V3_EDGE_WIDTH, NET_V2_GLOBAL_WIDTH)
# Encoder v3's trade block in a v4 feature vector: globals 91 .. 115.
TRADE_BLOCK = slice(TILE_END + NET_V2_GLOBAL_WIDTH, TILE_END + NET_V3_GLOBAL_WIDTH)


def cli(*args):
    return subprocess.run([sys.executable, "-m", "alphasettler", *args], capture_output=True, text=True)


def old_net(widths, width=8, rounds=1, seed=0) -> Net:
    """A random twin with zero encoder weights for every input past an older encoder's `widths`
    (hash, node, edge, global): that older net, as this build reads it."""
    _, node, edge, glob = widths
    torch.manual_seed(seed)
    net = Net(width, rounds).eval()
    with torch.no_grad():
        for layer, keep in ((net.node_encoder, node), (net.edge_encoder, edge), (net.global_encoder, glob)):
            layer.weight[:, keep:] = 0.0
    return net


def old_bytes(net: Net, widths) -> bytes:
    """`net` written as an older encoder's file: its hash, and each encoder's first input columns only."""
    encoding, node, edge, glob = widths
    keep = {0: node, 1: edge, 3: glob}
    layers = net.layers()
    out = [MAGIC, struct.pack("<IQIII", 2, encoding, net.width, net.rounds, len(layers))]
    out += [struct.pack("<II", r, c) for r, c in netformat.shapes(net.width, net.rounds, glob, node, edge)]
    for i, layer in enumerate(layers):
        w = layer.weight.detach()
        out += [_floats(w[:, :keep[i]] if i in keep else w), _floats(layer.bias)]
    return b"".join(out)


def no_trade_positions(n: int):
    """`n` encoded positions from random no-trade games, and each one's legal-move mask."""
    rng = random.Random(1)
    public, hidden, mask = [], [], torch.zeros(n, ACTIONS, dtype=torch.bool)
    seed = 0
    while len(public) < n:
        g = Game(seed, NO_TRADES)
        seed += 1
        for k in range(300):
            if g.is_over or len(public) == n:
                break
            if k % 5 == 0:
                p, h = encode(g)
                mask[len(public), g.legal_actions()] = True
                public.append(p)
                hidden.append(h)
            g.apply(rng.choice(g.legal_actions()))
    return b"".join(public), b"".join(hidden), mask


def _close(got, ref) -> float:
    return ((got - ref).abs() / ref.abs().clamp(min=1.0)).max().item()


@pytest.mark.parametrize("width,rounds", [(48, 2), (5, 3)])
def test_an_upgraded_v2_net_keeps_every_no_trade_output(tmp_path, width, rounds):
    """`src` reads only a v2 net's inputs, so its outputs are that v2 net's. The upgrade must be
    `src` exactly but for the trade rows (zero weights, the fixed biases), so in a no-trade game,
    where the trade block is all zero, it gives `src`'s value, aux and logits on every slot but the
    trade rows, and the Rust forward pass the same prior over legal moves."""
    src = old_net(V2, width, rounds, seed=3)
    data = upgrade(old_bytes(src, V2))
    up = from_bytes(data).eval()
    for i, (a, b) in enumerate(zip(up.layers(), src.layers())):
        if a is not up.other_policy:
            assert torch.equal(a.weight, b.weight) and torch.equal(a.bias, b.bias), i
    path = tmp_path / "up.bin"
    path.write_bytes(data)
    n = 100
    public, hidden, mask = no_trade_positions(n)
    pt = torch.frombuffer(bytearray(public), dtype=torch.float32).view(n, NET_PUBLIC_LEN)
    ht = torch.frombuffer(bytearray(hidden), dtype=torch.float32).view(n, NET_HIDDEN_LEN)
    assert (pt[:, TRADE_BLOCK] == 0).all(), "no-trade games leave the trade block zero"
    with torch.no_grad():
        ref, got = src(pt, ht), up(pt, ht)
    trade = torch.zeros(ACTIONS, dtype=torch.bool)
    trade[list(TRADE_SLOTS)] = True
    assert not mask[:, trade].any(), "no trade move is legal in a no-trade game"
    assert _close(got[0][:, ~trade], ref[0][:, ~trade]) <= 1e-5
    assert (got[0][:, list(OFFER_IDS)] == OFFER_BIAS).all()
    assert (got[0][:, list(ANSWER_IDS)] == 0).all()
    assert _close(got[1], ref[1]) <= 1e-5 and _close(got[2], ref[2]) <= 1e-5
    rust = net_forward(str(path), public, hidden)
    rp = torch.frombuffer(bytearray(rust["policy"]), dtype=torch.float32).view(n, ACTIONS)
    rv = torch.frombuffer(bytearray(rust["value"]), dtype=torch.float32).view(n, 4)

    def prior(x):
        return torch.softmax(x.double().masked_fill(~mask, float("-inf")), 1)

    assert (prior(rp) - prior(ref[0])).abs().max().item() <= 1e-5
    assert (torch.softmax(rv.double(), 1) - torch.softmax(ref[1].double(), 1)).abs().max().item() <= 1e-5


def test_only_encoder_v2_nets_are_upgraded():
    old = old_bytes(old_net(V2), V2)
    with pytest.raises(ValueError, match="already an encoder v4 net"):
        upgrade(to_bytes(Net(8, 1)))
    other = bytearray(old)
    other[12] ^= 1
    with pytest.raises(ValueError, match="not encoder v2's"):
        upgrade(bytes(other))
    with pytest.raises(ValueError, match="wrong magic"):
        upgrade(b"XXNET\0\0\0" + old[8:])
    one = bytearray(old)
    one[8:12] = (1).to_bytes(4, "little")
    with pytest.raises(ValueError, match="format version 1"):
        upgrade(bytes(one))
    with pytest.raises(ValueError, match="truncated"):
        upgrade(old[:20])
    with pytest.raises(ValueError, match="truncated"):
        upgrade(old[:-4])
    with pytest.raises(ValueError, match="trailing bytes"):
        upgrade(old + b"\0\0\0\0")
    wide = bytearray(old)
    struct.pack_into("<I", wide, 20, 2**31)
    with pytest.raises(ValueError, match="width 2147483648 is out of range"):
        upgrade(bytes(wide))
    nan = bytearray(old)
    struct.pack_into("<f", nan, len(nan) - 4, float("nan"))
    with pytest.raises(ValueError, match="NaN or infinite"):
        upgrade(bytes(nan))
    assert from_bytes(upgrade(old)).width == 8


def test_loaders_send_an_encoder_v2_net_to_net_upgrade(tmp_path):
    nets = tmp_path / "nets"
    nets.mkdir()
    (nets / "old.bin").write_bytes(old_bytes(old_net(V2), V2))
    p, h = encode(Game(0, NO_TRADES))
    loads = [lambda: net_forward(str(nets / "old.bin"), p, h),
             lambda: from_bytes((nets / "old.bin").read_bytes()),
             lambda: Bot("net:old", 0, str(nets))]
    for load in loads:
        with pytest.raises(ValueError, match="net-upgrade"):
            load()


def test_the_trade_rows_are_the_engines_offer_answer_and_confirmation_ids():
    assert len(OFFER_IDS) == 400 and all(describe_action(a).startswith("OfferTrade") for a in OFFER_IDS)
    assert [describe_action(a) for a in ANSWER_IDS] == [
        "AcceptTrade", "RejectTrade", "ConfirmTrade(0)", "ConfirmTrade(1)", "ConfirmTrade(2)", "ConfirmTrade(3)",
        "CancelTrade"]
    assert (pooled_row(0), pooled_row(24), pooled_row(258), pooled_row(664)) == (0, 24, 59, 465)
    with pytest.raises(ValueError, match="entity head"):
        pooled_row(25)
    assert netupgrade.shapes(48, 2, NET_GLOBAL_WIDTH) == shapes(48, 2)


def test_cli_net_upgrade(tmp_path):
    old, new = tmp_path / "old.bin", tmp_path / "new.bin"
    old.write_bytes(old_bytes(old_net(V2), V2))
    r = cli("net-upgrade", str(old), str(new))
    assert r.returncode == 0, r.stderr
    # 24 - 19 node, 8 - 4 edge, 139 - 91 global inputs; offers 258 .. 664: 407 rows.
    assert f"(encoder v2) to encoder v4: width 8, 1 rounds; 5 node, 4 edge and 48 global inputs zeroed; " \
           f"407 trade policy rows reset (offer bias -3); wrote {new}" in r.stdout
    assert from_bytes(new.read_bytes()).width == 8
    r = cli("net-upgrade", str(old), str(new))
    assert r.returncode == 2 and "exists" in r.stderr
    r = cli("net-upgrade", str(new), str(tmp_path / "again.bin"))
    assert r.returncode == 2 and "already an encoder v4 net" in r.stderr
    assert not (tmp_path / "again.bin").exists()


def test_net_upgrade_leaves_nothing_behind_when_the_rust_loader_refuses(tmp_path, monkeypatch, capsys):
    """NEW is published by a rename only after the Rust loader has read the temp file, so a refusal
    leaves neither NEW nor a temp, and a rerun is not stopped by "exists"."""
    from alphasettler import cli as cli_mod

    old, new = tmp_path / "old.bin", tmp_path / "new.bin"
    old.write_bytes(old_bytes(old_net(V2), V2))

    def refuse(path, public, hidden):
        assert path != str(new) and (tmp_path / path).exists(), "the check reads the temp file, not NEW"
        raise ValueError("refused")

    monkeypatch.setattr(netupgrade, "net_forward", refuse)
    assert cli_mod.main(["net-upgrade", str(old), str(new)]) == 2
    assert "the Rust loader refused the upgrade: refused" in capsys.readouterr().err
    assert sorted(p.name for p in tmp_path.iterdir()) == ["old.bin"]
    monkeypatch.undo()
    assert cli_mod.main(["net-upgrade", str(old), str(new)]) == 0
    assert from_bytes(new.read_bytes()).width == 8
```

In `tests/python/test_train.py` replace the imports

```python
from alphasettler._engine import (
    NET_DEFAULT_ROUNDS, NET_DEFAULT_WIDTH, NET_ENCODING_HASH, NET_HIDDEN_LEN, NET_MAX_ROUNDS, NET_MAX_WIDTH,
    NET_PUBLIC_LEN, encode, net_forward,
)
```

with

```python
from alphasettler._engine import (
    NET_DEFAULT_ROUNDS, NET_DEFAULT_WIDTH, NET_EDGE_WIDTH, NET_ENCODING_HASH, NET_GLOBAL_WIDTH, NET_HIDDEN_LEN,
    NET_MAX_ROUNDS, NET_MAX_WIDTH, NET_NODE_WIDTH, NET_PUBLIC_LEN, NET_V3_EDGE_WIDTH, NET_V3_GLOBAL_WIDTH,
    NET_V3_NODE_WIDTH, encode, net_forward,
)
```

and `from alphasettler.train.model import ACTIONS, FIXED_HEADER_LEN, Net, from_bytes, shapes, to_bytes` with `from alphasettler.train.model import ACTIONS, EDGE_END, EDGES, FIXED_HEADER_LEN, NODE_END, NODES, Net, from_bytes, shapes, to_bytes`; after `test_rust_and_pytorch_forward_passes_agree_on_random_weights` add:

```python
def test_the_parity_positions_fill_every_v4_input_block():
    """The parity test above compares the two forward passes on these positions: they must reach
    every block encoder v4 added, or a twin that misread one would still pass."""
    public, _, _ = random_positions(100)
    pt = torch.frombuffer(bytearray(public), dtype=torch.float32).view(100, NET_PUBLIC_LEN)
    nodes = pt[:, :NODE_END].reshape(100, NODES, NET_NODE_WIDTH)
    edges = pt[:, NODE_END:EDGE_END].reshape(100, EDGES, NET_EDGE_WIDTH)
    distance = nodes[:, :, NET_V3_NODE_WIDTH:NET_V3_NODE_WIDTH + 4]
    assert ((distance > 0) & (distance < 1)).any(), "a node one or two roads from some seat"
    open_spot = nodes[:, :, NET_NODE_WIDTH - 1]
    assert (open_spot == 1).any() and (open_spot == 0).any()
    assert (edges[:, :, NET_V3_EDGE_WIDTH:] == 1).any(), "a road on a longest trail"
    threats = pt[:, NET_PUBLIC_LEN - (NET_GLOBAL_WIDTH - NET_V3_GLOBAL_WIDTH):]
    assert (threats != threats[:1]).any(), "the threat block changes over the games"
```

- [ ] **Step 7: Rebuild the extension and run them to verify they fail**

Rebuild the extension. Run: `.venv/bin/pytest -q tests/python/test_net_upgrade.py tests/python/test_train.py`
Expected: FAIL: `test_net_upgrade.py` errors on `netformat.shapes(..., glob, node, edge)` (it takes 3 arguments) and the old `upgrade` reads a v2 file with v4's node width ("layer shapes ... do not fit").

- [ ] **Step 8: Implement the Python side**

In `alphasettler/netformat.py`:
- in the module docstring replace `Only the encoding hash check is the caller's: the twin wants this encoder's, the upgrade encoder v2's."""` with `Only the encoding hash check is the caller's: the twin wants this encoder's, the upgrade an older encoder's (whose input widths it passes)."""`;
- replace the import block with:

```python
from alphasettler._engine import (
    NET_EDGE_WIDTH, NET_ENCODING_HASH, NET_FORMAT_VERSION, NET_GLOBAL_WIDTH, NET_HIDDEN_LEN, NET_MAX_ROUNDS,
    NET_MAX_WIDTH, NET_NODE_WIDTH, NET_TILE_WIDTH, action_space_size, net_adjacency,
)
```

- replace `GLOBAL_ENCODER = 3` with:

```python
NODE_ENCODER = 0
EDGE_ENCODER = 1
GLOBAL_ENCODER = 3
```

- replace `def shapes` with:

```python
def shapes(width: int, rounds: int, global_width: int = NET_GLOBAL_WIDTH, node_width: int = NET_NODE_WIDTH,
           edge_width: int = NET_EDGE_WIDTH) -> list[tuple[int, int]]:
    """(rows, cols) of each layer in file order, for encoders of `node_width`, `edge_width` and
    `global_width` inputs (this build's by default, an older encoder's for the upgrade); rows are
    outputs (as Rust's `Arch::shapes_for`)."""
    d = width
    out = [(d, node_width), (d, edge_width), (d, NET_TILE_WIDTH), (d, global_width)]
    out += [(d, 3 * d), (d, 5 * d), (d, 3 * d), (d, 4 * d)] * rounds
    return out + [(2, 2 * d), (1, 2 * d), (1, 2 * d), (4 * d, 4 * d), (OTHER, 4 * d),
                  (4 * d, 4 * d + NET_HIDDEN_LEN), (SEATS, 4 * d), (SEATS, 4 * d)]
```

- replace `def read(data: bytes, global_width: int, check_encoding: Callable[[int], None],` and its next line `         layout: str) -> tuple[int, int, list[tuple[int, int]], array]:` with

```python
def read(data: bytes, global_width: int, check_encoding: Callable[[int], None], layout: str,
         node_width: int = NET_NODE_WIDTH, edge_width: int = NET_EDGE_WIDTH,
         ) -> tuple[int, int, list[tuple[int, int]], array]:
```

  in its docstring replace `whose\n    global encoder takes \`global_width\` inputs;` with `whose\n    encoders take \`node_width\`, \`edge_width\` and \`global_width\` inputs;`, and replace `    if declared != shapes(width, rounds, global_width):` with `    if declared != shapes(width, rounds, global_width, node_width, edge_width):`.

Replace `alphasettler/netupgrade.py` with:

```python
"""`alphasettler net-upgrade OLD NEW` (specs 3c Section 4 and 3d-2 Section 3): an older net
rewritten for this build's encoder, v4. Encoder v4 appended 5 inputs to each node, 4 to each edge
and the 24-input threat block to the globals; encoder v3 had appended the 24-input trade block to
encoder v2's 91 globals.

- Each encoder's first layer gets zero weights for the inputs the old net never saw, so the net
  ignores them.
- From encoder v2 (3b's nets), the pooled head's rows for every domestic offer (OFFER_IDS), the
  answers and the confirmations (ANSWER_IDS, CancelTrade included) get zero weights, and biases of
  OFFER_BIAS for offers and 0 for the rest. 3b never trained them: offers were masked out, and
  every answer was a hard-coded reject or cancel. All 400 offer rows are reset, not only the 20
  one-for-one offers the trading game allows, because the same reason holds for all of them. An
  upgraded v2 net makes the v2 net's moves in every no-trade game.
- Every other byte is copied.

No torch, so it runs without the train extra: the file is read by `alphasettler.netformat`, the
one Python reader of the weights format (net/src/weights.rs's, version 2), which the PyTorch twin
shares."""

from __future__ import annotations

import os
from array import array
from pathlib import Path

from alphasettler import files
from alphasettler._engine import (
    NET_EDGE_WIDTH, NET_ENCODING_HASH, NET_GLOBAL_WIDTH, NET_NODE_WIDTH, NET_SETTLEMENT_BASE, NET_V2_ENCODING_HASH,
    NET_V2_GLOBAL_WIDTH, NET_V3_EDGE_WIDTH, NET_V3_NODE_WIDTH, Game, encode, net_forward,
)
from alphasettler.netformat import (
    EDGE_ENCODER, ENTITY_SLOTS, GLOBAL_ENCODER, NODE_ENCODER, OTHER_POLICY, head_layer, header, read,
)
from alphasettler.netformat import shapes  # noqa: F401  (part of this module's interface)

OFFER_BIAS = -3.0
# Engine action ids (engine/src/action.rs): offers, then accept, reject, the four confirmations, cancel.
OFFER_IDS = range(258, 658)
ANSWER_IDS = range(658, 665)
TRADE_SLOTS = range(258, 665)

# The encoders a net is upgraded from: encoding hash -> (encoder version, node, edge and global input
# widths). Encoder v2's node and edge inputs are v3's.
SOURCES = {
    NET_V2_ENCODING_HASH: (2, NET_V3_NODE_WIDTH, NET_V3_EDGE_WIDTH, NET_V2_GLOBAL_WIDTH),
}
# Each encoder layer and its input count in this build.
TARGET = {"node": (NODE_ENCODER, NET_NODE_WIDTH), "edge": (EDGE_ENCODER, NET_EDGE_WIDTH),
          "global": (GLOBAL_ENCODER, NET_GLOBAL_WIDTH)}


def pooled_row(slot: int) -> int:
    """The pooled head's output row for policy slot `slot` (net/src/policy.rs `other_slot`, inverted)."""
    if NET_SETTLEMENT_BASE <= slot < NET_SETTLEMENT_BASE + ENTITY_SLOTS:
        raise ValueError(f"slot {slot} comes from an entity head")
    return slot if slot < NET_SETTLEMENT_BASE else slot - ENTITY_SLOTS


def _check_source(encoding: int) -> None:
    if encoding == NET_ENCODING_HASH:
        raise ValueError("already an encoder v4 net (this build's encoding hash): nothing to upgrade")
    if encoding not in SOURCES:
        raise ValueError(f"encoding hash {encoding:#018x} is not encoder v2's {NET_V2_ENCODING_HASH:#018x}: "
                         "only encoder v2 nets can be upgraded")


def read_old(data: bytes) -> tuple[int, int, int, list[tuple[int, int, array, array]]]:
    """(encoder version, width, rounds, layers) of a weights file this tool upgrades, each layer
    (rows, cols, weights, biases); ValueError for any other file. Everything is checked before
    anything is sized from it."""
    # The input widths follow from the encoding hash, which `read` checks only after the magic and the
    # version: peek at it, and let `read` report whatever is wrong in its own order.
    peek = int.from_bytes(data[12:20], "little") if len(data) >= 20 else None
    version, node, edge, glob = SOURCES.get(peek, (0, NET_NODE_WIDTH, NET_EDGE_WIDTH, NET_GLOBAL_WIDTH))
    width, rounds, declared, floats = read(data, glob, _check_source, f"an encoder v{version} net",
                                           node_width=node, edge_width=edge)
    layers, at = [], 0
    for rows, cols in declared:
        w = floats[at:at + rows * cols]
        at += rows * cols
        b = floats[at:at + rows]
        at += rows
        layers.append((rows, cols, w, b))
    return version, width, rounds, layers


def widened(layer: tuple[int, int, array, array], cols: int) -> tuple[int, int, array, array]:
    """`layer` with `cols` inputs: its own first, then zero weights for the inputs appended after them."""
    rows, old, w, b = layer
    wide = array("f")
    for r in range(rows):
        wide.extend(w[r * old:(r + 1) * old])
        wide.extend(array("f", [0.0]) * (cols - old))
    return rows, cols, wide, b


def _upgrade(data: bytes) -> tuple[bytes, dict]:
    version, width, rounds, layers = read_old(data)
    zero = {}
    for name, (index, cols) in TARGET.items():
        zero[name] = cols - layers[index][1]
        layers[index] = widened(layers[index], cols)
    trade_rows = 0
    if version == 2:
        rows, cols, w, b = layers[head_layer(rounds, OTHER_POLICY)]
        for slot in TRADE_SLOTS:
            r = pooled_row(slot)
            w[r * cols:(r + 1) * cols] = array("f", [0.0]) * cols
            b[r] = OFFER_BIAS if slot in OFFER_IDS else 0.0
        trade_rows = len(TRADE_SLOTS)
    out = [header(width, rounds, [(r, c) for r, c, _, _ in layers])]
    for _, _, w, b in layers:
        out += [w.tobytes(), b.tobytes()]
    summary = {"from": version, "width": width, "rounds": rounds, "zero_inputs": zero, "trade_rows": trade_rows}
    return b"".join(out), summary


def upgrade(data: bytes) -> bytes:
    """The encoder v4 weights file for the older file `data` (see the module docstring)."""
    return _upgrade(data)[0]


def upgrade_file(old: str | Path, new: str | Path) -> dict:
    """Write the upgrade of `old` to `new` (which must not exist, so nothing is overwritten) and
    return what was done: the source encoder (`from`), width, rounds, the zero inputs per encoder
    and the trade rows reset. The upgrade goes to a temp file beside `new` and is published by a
    rename only after the Rust loader has read it as every bot will, so a failure at any step leaves
    no `new` (and no temp) behind."""
    old, new = Path(old), Path(new)
    if new.exists():
        raise ValueError(f"{new} exists: choose a new path")
    try:
        data, summary = _upgrade(old.read_bytes())
    except ValueError as e:
        raise ValueError(f"{old}: {e}") from None
    tmp = files.temp_path(new)
    try:
        with open(tmp, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        p, h = encode(Game(0, {"max_offers_per_turn": 0}))
        try:
            net_forward(str(tmp), p, h)
        except ValueError as e:
            raise ValueError(f"{old}: the Rust loader refused the upgrade: {e}") from None
        os.replace(tmp, new)
    finally:
        tmp.unlink(missing_ok=True)
    return summary
```

In `alphasettler/train/model.py` replace `net over the encoder's 54 nodes, 72 edges, 19 tiles and 115 global scalars (encoder v3), read from` with `net over the encoder's 54 nodes, 72 edges, 19 tiles and 139 global scalars (encoder v4), read from`.

In `alphasettler/cli.py`:
- replace `` `net-upgrade` rewrites a 3b (encoder v2) net for encoder v3.`` with `` `net-upgrade` rewrites an older net (encoder v2, 3b's) for this build's encoder v4.``;
- replace `nu = sub.add_parser("net-upgrade", help="rewrite an encoder v2 net (3b's) for encoder v3, same no-trade moves")` with `nu = sub.add_parser("net-upgrade", help="rewrite an encoder v2 net (3b's) for encoder v4, same no-trade moves")`;
- replace the body of `_net_upgrade` from `    s = netupgrade.upgrade_file(args.old, args.new)` through `    return 0` with:

```python
    s = netupgrade.upgrade_file(args.old, args.new)
    z = s["zero_inputs"]
    reset = (f"; {s['trade_rows']} trade policy rows reset (offer bias {netupgrade.OFFER_BIAS:g})"
             if s["trade_rows"] else "")
    print(f"upgraded {args.old} (encoder v{s['from']}) to encoder v4: width {s['width']}, {s['rounds']} rounds; "
          f"{z['node']} node, {z['edge']} edge and {z['global']} global inputs zeroed{reset}; wrote {args.new}")
    return 0
```

- [ ] **Step 9: Run them to verify they pass**

Run: `.venv/bin/pytest -q tests/python/test_net_upgrade.py tests/python/test_train.py`
Expected: PASS (the parity test now runs at the v4 widths).

- [ ] **Step 10: The gates**

Run the Global Constraints' gates (fmt, clippy, the full `cargo test --workspace --release`, rebuild the extension, `.venv/bin/pytest -q -rs`). Expected: all pass, no skips; every pinned digest passes unchanged.

- [ ] **Step 11: Commit**

```bash
git add net/src/encode.rs net/src/graph.rs net/src/weights.rs bindings/src/train.rs \
  alphasettler/netformat.py alphasettler/netupgrade.py alphasettler/train/model.py alphasettler/cli.py \
  net/tests/encode.rs net/tests/graph.rs net/tests/weights.rs bots/tests/netbot.rs \
  tests/python/test_net_upgrade.py tests/python/test_train.py
git diff --cached | grep -nE 'sk-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{36}|password=[^[:space:]]+'
git -c user.email=sean.may101@gmail.com commit -m "net: encoder v4, road distances, open spots, longest trails and the threat block; v2 nets upgrade to v4"
```

---

### Task 3: `net-upgrade` for encoder v3 nets

**Files:**
- Modify: `net/src/weights.rs` (`from_bytes`'s hash error), `alphasettler/netupgrade.py` (docstring, imports, `SOURCES`, `_check_source`), `alphasettler/train/model.py` (imports, `_check_encoding`), `alphasettler/cli.py` (`net-upgrade` texts)
- Test: `net/tests/weights.rs`, `tests/python/test_net_upgrade.py`

**Interfaces:**
- Consumes: `V3_ENCODING_HASH`, `NET_V3_*` (Task 2); `netupgrade.SOURCES`, `old_net`, `old_bytes`, `V2` (Task 2); `test_train.generation`, `test_train.random_positions`, `test_fast_loop.TINY`, `FastLoop.init(root, seed_dirs, start_net, first, ...)`, `FastLoop(root, FastLoopSettings(...)).run()`.
- Produces: `netupgrade.SOURCES[NET_V3_ENCODING_HASH] == (3, 19, 4, 115)`; `upgrade_file` summary `{"from": 3, ..., "zero_inputs": {"node": 5, "edge": 4, "global": 24}, "trade_rows": 0}`; the loaders' hint for v3 files; test helper `V3`.

- [ ] **Step 1: Write the failing tests**

In `net/tests/weights.rs` replace `fn an_encoder_v2_file_is_refused_with_the_way_to_upgrade_it() {` with `fn an_encoder_v2_or_v3_file_is_refused_with_the_way_to_upgrade_it() {`, and in it replace

```rust
    let mut other = Weights::random(SMALL, 1).to_bytes();
    other[12] ^= 1;
    assert!(
        !err(&other).contains("net-upgrade"),
        "only v2's hash gets the hint"
    );
```

with

```rust
    let mut v3 = Weights::random(SMALL, 1).to_bytes();
    v3[12..20].copy_from_slice(&V3_ENCODING_HASH.to_le_bytes());
    let e = err(&v3);
    assert!(
        e.contains("encoder v3") && e.contains("alphasettler net-upgrade"),
        "{e}"
    );
    let mut other = Weights::random(SMALL, 1).to_bytes();
    other[12] ^= 1;
    assert!(
        !err(&other).contains("net-upgrade"),
        "only v2's and v3's hashes get the hint"
    );
```

In `tests/python/test_net_upgrade.py`:
- add `import json` after `import random`; extend the `_engine` import with `NET_V3_ENCODING_HASH`; add after the `model` import:

```python
from alphasettler.train.fastloop import STATE, FastLoop, FastLoopSettings

from test_fast_loop import TINY
from test_train import generation, random_positions
```

- after the `V2 = ...` line add:

```python
V3 = (NET_V3_ENCODING_HASH, NET_V3_NODE_WIDTH, NET_V3_EDGE_WIDTH, NET_V3_GLOBAL_WIDTH)
```

- replace `test_only_encoder_v2_nets_are_upgraded`, `test_loaders_send_an_encoder_v2_net_to_net_upgrade` and `test_cli_net_upgrade` with:

```python
@pytest.mark.parametrize("widths", [V2, V3], ids=["v2", "v3"])
def test_only_encoder_v2_and_v3_nets_are_upgraded(widths):
    old = old_bytes(old_net(widths), widths)
    with pytest.raises(ValueError, match="already an encoder v4 net"):
        upgrade(to_bytes(Net(8, 1)))
    other = bytearray(old)
    other[12] ^= 1
    with pytest.raises(ValueError, match="neither encoder v2's .* nor encoder v3's"):
        upgrade(bytes(other))
    with pytest.raises(ValueError, match="wrong magic"):
        upgrade(b"XXNET\0\0\0" + old[8:])
    one = bytearray(old)
    one[8:12] = (1).to_bytes(4, "little")
    with pytest.raises(ValueError, match="format version 1"):
        upgrade(bytes(one))
    with pytest.raises(ValueError, match="truncated"):
        upgrade(old[:20])
    with pytest.raises(ValueError, match="truncated"):
        upgrade(old[:-4])
    with pytest.raises(ValueError, match="trailing bytes"):
        upgrade(old + b"\0\0\0\0")
    wide = bytearray(old)
    struct.pack_into("<I", wide, 20, 2**31)
    with pytest.raises(ValueError, match="width 2147483648 is out of range"):
        upgrade(bytes(wide))
    nan = bytearray(old)
    struct.pack_into("<f", nan, len(nan) - 4, float("nan"))
    with pytest.raises(ValueError, match="NaN or infinite"):
        upgrade(bytes(nan))
    assert from_bytes(upgrade(old)).width == 8


@pytest.mark.parametrize("widths", [V2, V3], ids=["v2", "v3"])
def test_loaders_send_an_older_net_to_net_upgrade(tmp_path, widths):
    nets = tmp_path / "nets"
    nets.mkdir()
    (nets / "old.bin").write_bytes(old_bytes(old_net(widths), widths))
    p, h = encode(Game(0, NO_TRADES))
    loads = [lambda: net_forward(str(nets / "old.bin"), p, h),
             lambda: from_bytes((nets / "old.bin").read_bytes()),
             lambda: Bot("net:old", 0, str(nets))]
    for load in loads:
        with pytest.raises(ValueError, match="net-upgrade"):
            load()


@pytest.mark.parametrize("widths,did", [
    # 24 - 19 node, 8 - 4 edge, 139 - 91 global inputs; offers 258 .. 664: 407 rows.
    (V2, "(encoder v2) to encoder v4: width 8, 1 rounds; 5 node, 4 edge and 48 global inputs zeroed; "
         "407 trade policy rows reset (offer bias -3); wrote"),
    # 139 - 115 global inputs; v3 nets trained their trade rows: none reset.
    (V3, "(encoder v3) to encoder v4: width 8, 1 rounds; 5 node, 4 edge and 24 global inputs zeroed; wrote"),
], ids=["v2", "v3"])
def test_cli_net_upgrade(tmp_path, widths, did):
    old, new = tmp_path / "old.bin", tmp_path / "new.bin"
    old.write_bytes(old_bytes(old_net(widths), widths))
    r = cli("net-upgrade", str(old), str(new))
    assert r.returncode == 0, r.stderr
    assert f"{did} {new}" in r.stdout
    assert from_bytes(new.read_bytes()).width == 8
    r = cli("net-upgrade", str(old), str(new))
    assert r.returncode == 2 and "exists" in r.stderr
    r = cli("net-upgrade", str(new), str(tmp_path / "again.bin"))
    assert r.returncode == 2 and "already an encoder v4 net" in r.stderr
    assert not (tmp_path / "again.bin").exists()
```

- append:

```python
@pytest.mark.parametrize("width,rounds", [(48, 2), (5, 3)])
def test_an_upgraded_v3_net_gives_the_v3_nets_outputs_on_every_position(tmp_path, width, rounds):
    """A v3 net reads only the v3 slots of a v4 feature vector, which are its v3 features: `src` is
    that net, with zero weights on the rest. The upgrade must be `src` exactly (nothing reset), and
    its Rust forward pass must agree with `src` by the parity rule on positions of random trading
    games, where every block v4 added is live."""
    src = old_net(V3, width, rounds, seed=3)
    data = upgrade(old_bytes(src, V3))
    up = from_bytes(data).eval()
    for i, (a, b) in enumerate(zip(up.layers(), src.layers())):
        assert torch.equal(a.weight, b.weight) and torch.equal(a.bias, b.bias), i
    path = tmp_path / "up.bin"
    path.write_bytes(data)
    n = 100
    public, hidden, mask = random_positions(n)
    pt = torch.frombuffer(bytearray(public), dtype=torch.float32).view(n, NET_PUBLIC_LEN)
    ht = torch.frombuffer(bytearray(hidden), dtype=torch.float32).view(n, NET_HIDDEN_LEN)
    assert (pt[:, TILE_END + NET_V3_GLOBAL_WIDTH:] != 0).any(), "the threat block is live"
    with torch.no_grad():
        ref = src(pt, ht)
    rust = net_forward(str(path), public, hidden)
    got = {key: torch.frombuffer(bytearray(rust[key]), dtype=torch.float32).view_as(r)
           for key, r in zip(("policy", "value", "aux"), ref)}
    for key, r in zip(("policy", "value", "aux"), ref):
        assert _close(got[key], r) <= 1e-5, key

    def prior(x):
        return torch.softmax(x.double().masked_fill(~mask, float("-inf")), 1)

    assert (prior(got["policy"]) - prior(ref[0])).abs().max().item() <= 1e-5
    assert (torch.softmax(got["value"].double(), 1) - torch.softmax(ref[1].double(), 1)).abs().max().item() <= 1e-5


def test_a_fast_loop_runs_from_an_upgraded_v3_net(tmp_path):
    seed = generation(tmp_path, "seed-gen", seed_start=0)
    old, start = tmp_path / "v3.bin", tmp_path / "start.bin"
    old.write_bytes(old_bytes(old_net(V3), V3))
    r = cli("net-upgrade", str(old), str(start))
    assert r.returncode == 0, r.stderr
    root = tmp_path / "loop"
    FastLoop.init(root, [seed], start, first=1)
    assert FastLoop(root, FastLoopSettings(generations=1, **TINY)).run() == 1
    state = json.loads((root / STATE).read_text())
    assert [g["gen"] for g in state["generations"]] == [1]
    assert (root / "nets" / "start.bin").read_bytes() == start.read_bytes()
```

- [ ] **Step 2: Run them to verify they fail**

Run: `PATH=$HOME/.cargo/bin:$PATH cargo test --release -p settler-net --test weights an_encoder_v2_or_v3` and `.venv/bin/pytest -q tests/python/test_net_upgrade.py`
Expected: Rust FAIL on `e.contains("encoder v3")`; Python FAIL: the v3 cases raise "is not encoder v2's", the v2 refusal does not match "neither ... nor".

- [ ] **Step 3: Implement**

In `net/src/weights.rs` `from_bytes`, replace

```rust
            let why = if hash == V2_ENCODING_HASH {
                "; it is an encoder v2 net (before the trade block): convert it with \
                 `alphasettler net-upgrade OLD NEW`"
            } else {
                ""
            };
```

with

```rust
            let why = if hash == V2_ENCODING_HASH {
                "; it is an encoder v2 net (before the trade block): convert it with \
                 `alphasettler net-upgrade OLD NEW`"
            } else if hash == V3_ENCODING_HASH {
                "; it is an encoder v3 net (before the road and threat inputs): convert it with \
                 `alphasettler net-upgrade OLD NEW`"
            } else {
                ""
            };
```

In `alphasettler/train/model.py`, extend the `_engine` import with `NET_V3_ENCODING_HASH` and replace `_check_encoding` with:

```python
# Older encoders' hashes, for the error's hint (`alphasettler net-upgrade` converts them).
_OLDER = {NET_V2_ENCODING_HASH: "an encoder v2 net (before the trade block)",
          NET_V3_ENCODING_HASH: "an encoder v3 net (before the road and threat inputs)"}


def _check_encoding(encoding: int) -> None:
    if encoding != NET_ENCODING_HASH:
        why = (f"; it is {_OLDER[encoding]}: convert it with `alphasettler net-upgrade OLD NEW`"
               if encoding in _OLDER else "")
        raise ValueError(f"encoding hash {encoding:#018x} does not match this encoder's {NET_ENCODING_HASH:#018x}: "
                         f"the file was trained on a different feature layout{why}")
```

In `alphasettler/netupgrade.py`:
- in the docstring, after the line `  upgraded v2 net makes the v2 net's moves in every no-trade game.` add:

```text
- From encoder v3 (3c's and P2's nets) nothing else changes: an upgraded v3 net gives the v3 net's
  outputs on every position, within the forward pass's 1e-5 (Accelerate sums the wider rows in
  another order).
```

- extend the `_engine` import with `NET_V3_ENCODING_HASH` and `NET_V3_GLOBAL_WIDTH`;
- in `SOURCES` add after the v2 entry: `    NET_V3_ENCODING_HASH: (3, NET_V3_NODE_WIDTH, NET_V3_EDGE_WIDTH, NET_V3_GLOBAL_WIDTH),`;
- replace the second `raise` of `_check_source` with:

```python
        raise ValueError(f"encoding hash {encoding:#018x} is neither encoder v2's {NET_V2_ENCODING_HASH:#018x} "
                         f"nor encoder v3's {NET_V3_ENCODING_HASH:#018x}: only encoder v2 and v3 nets can be upgraded")
```

In `alphasettler/cli.py`:
- replace `` `net-upgrade` rewrites an older net (encoder v2, 3b's) for this build's encoder v4.`` with `` `net-upgrade` rewrites an older net (encoder v2 or v3) for this build's encoder v4.``;
- replace `help="rewrite an encoder v2 net (3b's) for encoder v4, same no-trade moves")` with `help="rewrite an encoder v2 or v3 net for encoder v4, same moves")` and `nu.add_argument("old", help="the encoder v2 weights file")` with `nu.add_argument("old", help="the encoder v2 or v3 weights file")`.

- [ ] **Step 4: Run them to verify they pass**

Rebuild the extension. Run: `PATH=$HOME/.cargo/bin:$PATH cargo test --release -p settler-net --test weights` and `.venv/bin/pytest -q tests/python/test_net_upgrade.py`
Expected: PASS.

- [ ] **Step 5: The gates**

Run the Global Constraints' gates. Expected: all pass, no skips.

- [ ] **Step 6: Commit**

```bash
git add net/src/weights.rs net/tests/weights.rs alphasettler/netupgrade.py alphasettler/train/model.py \
  alphasettler/cli.py tests/python/test_net_upgrade.py
git diff --cached | grep -nE 'sk-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{36}|password=[^[:space:]]+'
git -c user.email=sean.may101@gmail.com commit -m "net-upgrade: encoder v3 nets to v4 with zero weights for the new inputs; loaders name net-upgrade for v3 nets"
```

---

### Task 4: The upgrade of gen-034, speed, arm D, the SPRT and `docs/bot/results-3d2.md`

**Files:**
- Create: `docs/bot/results-3d2.md`
- Run outputs (not committed; `runs/` is ignored): `runs/3d2/v3-src/`, `runs/3d2/v3-target/`, `runs/3d2/v3-wheels/`, `runs/3d2/forward_dump.py`, `runs/3d2/forward_compare.py`, `runs/3d2/gen-034-v4.bin`, `runs/3d2/upgrade.log`, `runs/3d2/fwd-v3-*.bin`, `runs/3d2/fwd-v4-*.bin`, `runs/3d2/forward.log`, `runs/3d2/speed.log`, `runs/p2/D/`, `runs/p2/D.log`, `runs/p2/D.out`, `runs/p2/D-state-at-cut.json`, `runs/3d2/tables.md`, `runs/3d2/finals.log`, `runs/3d2/nets/`, `runs/3d2/sprt-D.jsonl`, `runs/3d2/sprt-D.log`

**Interfaces:**
- Consumes: `alphasettler net-upgrade` (Task 3); `bots/examples/net_speed.rs` (`net_speed speed PATH DECISIONS SIMULATIONS BATCH`); `runs/p2/arm.sh` and `runs/p2/tables.py` (unchanged); 3d-1b's `runs/p2/sprt-C-vs-A.log`, `runs/p2/C-state-at-cut.json`, `runs/p2/C.log`, `runs/p2/nets/C.bin`; P2's `runs/p2/A-state-at-cut.json`, `runs/p2/A.log`, `runs/p2/nets/A.bin`; `runs/3c/nets/gen-034.bin`, `runs/3c/nets/opening.bin`.
- Produces: the run outputs above and `docs/bot/results-3d2.md`.

Every step but Step 10 is run by the controller in the background, on an otherwise idle machine. If a step fails, record the failure with its numbers in the results; never rerun a match on the same seeds after changing anything.

- [ ] **Step 1: Build at Task 3's commit and read 3d-1b's verdict (controller runs this)**

```bash
git log -1 --format=%s
PATH=$HOME/.cargo/bin:$PATH cargo build --release -p settler-bots --example net_speed
ls runs/p2/arm.sh runs/p2/tables.py runs/p2/A-state-at-cut.json runs/p2/A.log runs/p2/nets/A.bin \
  runs/p2/C-state-at-cut.json runs/p2/C.log runs/p2/nets/C.bin runs/p2/sprt-C-vs-A.log docs/bot/results-3d1b.md \
  runs/3c/nets/gen-034.bin runs/3c/nets/opening.bin > /dev/null
shasum -a 256 runs/3c/nets/gen-034.bin runs/p2/nets/A.bin runs/3c/nets/opening.bin
test ! -e runs/p2/D && test ! -e runs/3d2
grep -q '^sprt: H1' runs/p2/sprt-C-vs-A.log && echo "branch O: control C, --opening" || echo "branch N: control A"
pgrep -fl "alphasettler|settler" || echo idle
```

Then rebuild the extension per the Global Constraints. Pass: the commit subject is Task 3's; every command exits 0; sha256s `5843c12edb4615ec784f89fe039c0dfb399e57fff96ba84557ad13b755713092` (gen-034), `79690c2992539f7995e9bc295c73606f01aa3b705e26586e1077bb5f24db7f88` (A), `9ab63da1309e35f2a87990edc85064ee9888057de84d52fb0e09ae382ced922b` (opening); the last line is `idle`. The branch line decides every later step: **branch O** (3d-1b was H1) uses control C and `--opening`; **branch N** (H0 or undecided) uses control A and no `--opening`.

- [ ] **Step 2: The encoder v3 build (controller runs this)**

The commit before Task 1, built in its own tree and target directory (the working tree stays at Task 3):

```bash
base=$(git rev-parse "$(git log -1 --format=%H --grep='^engine: longest_trail')^")
mkdir -p runs/3d2/v3-src
git log -1 --format='%h %s' "$base" | tee runs/3d2/base.txt
git archive "$base" | tar -x -C runs/3d2/v3-src
PATH=$HOME/.cargo/bin:$PATH cargo build --release --manifest-path runs/3d2/v3-src/Cargo.toml \
  --target-dir runs/3d2/v3-target -p settler-bots --example net_speed
(cd runs/3d2/v3-src && CARGO_TARGET_DIR=../v3-target PATH=$HOME/.cargo/bin:$PATH ../../../.venv/bin/maturin build --release -q -o ../v3-wheels)
unzip -q -o runs/3d2/v3-wheels/alphasettler-*.whl 'alphasettler/_engine*.so' -d runs/3d2/v3-src
PYTHONPATH=runs/3d2/v3-src .venv/bin/python3 -c "import alphasettler, alphasettler._engine as e; print(alphasettler.__file__, hex(e.NET_ENCODING_HASH))"
.venv/bin/python3 -c "import alphasettler, alphasettler._engine as e; print(alphasettler.__file__, hex(e.NET_ENCODING_HASH))"
```

Pass: every command exits 0; the base line is Plan 9's last commit; the first print names a file under `runs/3d2/v3-src/alphasettler/` and `0x6d02356cb61753c2` (encoder v3), the second the working tree's `alphasettler/` and a different hash (encoder v4).

- [ ] **Step 3: Upgrade gen-034 and compare the two builds' outputs (controller runs this)**

Write `runs/3d2/forward_dump.py`:

```python
"""Plan 10 Task 4: the Rust forward pass of the net at NET on 200 positions of seeded random trading
games, written to OUT-{policy,value,aux}-{batch,single}.bin (one call of 200, and 200 calls of 1),
for comparing two builds: the engine plays the same games in each. Usage: forward_dump.py NET OUT"""
import random
import sys

from alphasettler._engine import NET_ENCODING_HASH, Game, encode, net_forward

net, out = sys.argv[1], sys.argv[2]
rng = random.Random(1)
public, hidden, seed = [], [], 0
while len(public) < 200:
    g = Game(seed, {"max_offers_per_turn": 2, "max_trade_cards": 1})
    seed += 1
    for k in range(400):
        if g.is_over or len(public) == 200:
            break
        if k % 7 == 0:
            p, h = encode(g)
            public.append(p)
            hidden.append(h)
        g.apply(rng.choice(g.legal_actions()))
batch = net_forward(net, b"".join(public), b"".join(hidden))
single = [net_forward(net, p, h) for p, h in zip(public, hidden)]
for key in ("policy", "value", "aux"):
    with open(f"{out}-{key}-batch.bin", "wb") as f:
        f.write(batch[key])
    with open(f"{out}-{key}-single.bin", "wb") as f:
        f.write(b"".join(r[key] for r in single))
print(f"encoding {NET_ENCODING_HASH:#018x}: {len(public)} positions from {seed} games")
```

and `runs/3d2/forward_compare.py`:

```python
"""Plan 10 Task 4: two forward_dump.py outputs (A: the encoder v3 build and net, B: v4 and the
upgrade) by the parity rule, |a - b| <= 1e-5 x max(1, |a|) on every output. Usage: forward_compare.py A B"""
import math
import sys
from array import array

a, b = sys.argv[1], sys.argv[2]
worst = 0.0
for mode in ("batch", "single"):
    for key in ("policy", "value", "aux"):
        x, y = array("f"), array("f")
        with open(f"{a}-{key}-{mode}.bin", "rb") as f:
            x.frombytes(f.read())
        with open(f"{b}-{key}-{mode}.bin", "rb") as f:
            y.frombytes(f.read())
        assert len(x) == len(y) > 0 and all(map(math.isfinite, x)) and all(map(math.isfinite, y)), (mode, key)
        err = max(abs(p - q) / max(1.0, abs(p)) for p, q in zip(x, y))
        same = sum(p == q for p, q in zip(x, y))
        worst = max(worst, err)
        print(f"{mode} {key}: {len(x)} outputs, max relative difference {err:.3g}, bit-equal {same} of {len(x)}")
print(f"worst {worst:.3g}: {'pass' if worst <= 1e-5 else 'FAIL'} (parity rule 1e-5)")
```

Then:

```bash
.venv/bin/alphasettler net-upgrade runs/3c/nets/gen-034.bin runs/3d2/gen-034-v4.bin | tee runs/3d2/upgrade.log
shasum -a 256 runs/3c/nets/gen-034.bin runs/3d2/gen-034-v4.bin | tee -a runs/3d2/upgrade.log
PYTHONPATH=runs/3d2/v3-src .venv/bin/python3 runs/3d2/forward_dump.py runs/3c/nets/gen-034.bin runs/3d2/fwd-v3 | tee runs/3d2/forward.log
.venv/bin/python3 runs/3d2/forward_dump.py runs/3d2/gen-034-v4.bin runs/3d2/fwd-v4 | tee -a runs/3d2/forward.log
.venv/bin/python3 runs/3d2/forward_compare.py runs/3d2/fwd-v3 runs/3d2/fwd-v4 | tee -a runs/3d2/forward.log
```

Pass: `upgrade.log` starts `upgraded runs/3c/nets/gen-034.bin (encoder v3) to encoder v4: width 48, 2 rounds; 5 node, 4 edge and 24 global inputs zeroed; wrote runs/3d2/gen-034-v4.bin`; both dumps report 200 positions from the same number of games (v3's hash `0x6d02356cb61753c2` first); the last line of `forward.log` ends `pass (parity rule 1e-5)`. The bit-equal counts are recorded, not required (Rulings).

- [ ] **Step 4: Search speed, v3 against v4 (controller runs this)**

The command of `docs/perf/net-search.md` (one net bot against three GreedyBots, one core, batch 8, 1,000 simulations, 500 searched decisions), three runs of each build, alternating, with nothing else running:

```bash
pgrep -fl "alphasettler|settler" || echo idle
for i in 1 2 3; do
  echo "v3 run $i: $(runs/3d2/v3-target/release/examples/net_speed speed runs/3c/nets/gen-034.bin 500 1000 8)" | tee -a runs/3d2/speed.log
  echo "v4 run $i: $(target/release/examples/net_speed speed runs/3d2/gen-034-v4.bin 500 1000 8)" | tee -a runs/3d2/speed.log
done
```

Pass: six lines, each `batch 8: N searched decisions over G games at 1000 simulations: X simulations/s, Y evaluations/s, Z ms/decision`. The v3 and v4 runs play the same games (same net up to 1e-5); `G` should match between builds, and a mismatch is recorded.

- [ ] **Step 5: Arm D, the 3-hour loop (controller runs this)**

Arm D runs through `runs/p2/arm.sh` itself, unchanged: the script's own `--start-net runs/3c/nets/gen-034.bin` comes first and the second `--start-net` (argparse keeps the last) replaces it; every other setting and seed is A's and C's. Start it detached, with the machine otherwise idle.

Branch O (3d-1b was H1):

```bash
pgrep -fl "alphasettler|settler" || echo idle
nohup bash runs/p2/arm.sh D --start-net runs/3d2/gen-034-v4.bin --opening runs/3c/nets/opening.bin >> runs/p2/D.out 2>&1 &
```

Branch N (H0 or undecided):

```bash
pgrep -fl "alphasettler|settler" || echo idle
nohup bash runs/p2/arm.sh D --start-net runs/3d2/gen-034-v4.bin >> runs/p2/D.out 2>&1 &
```

Once `runs/p2/D.log` has its `start D` line, `cmp runs/p2/D/nets/start.bin runs/3d2/gen-034-v4.bin && echo "D starts from gen-034 upgraded"` must print. Poll `runs/p2/D.log` until its last line holds `cut D after`, `loop exited early` or `setup failed`. Then (every block from here on starts with `ctl=C` in branch O, `ctl=A` in branch N):

```bash
ctl=C  # branch O; ctl=A in branch N
tail -1 runs/p2/D.log
.venv/bin/python3 - "$ctl" <<'EOF' | tee -a runs/3d2/finals.log
import hashlib, json, sys
name = sys.argv[1]
ctl, d = (json.load(open(f"runs/p2/{x}-state-at-cut.json")) for x in (name, "D"))
model = hashlib.sha256(open("runs/3c/nets/opening.bin", "rb").read()).hexdigest()
plus = "+t+o" if name == "C" else "+t"
assert ctl["seed_dirs"] == d["seed_dirs"] and ctl["first"] == d["first"] == 35 and ctl["trades"] is d["trades"] is True
assert ctl.get("gumbel") is None and d["gumbel"] is None
assert d.get("opening") == ctl.get("opening") == (model if name == "C" else None)
for gc, gd in zip(ctl["generations"], d["generations"]):
    assert gc["gen"] == gd["gen"] and gc["selfplay"]["seed_start"] == gd["selfplay"]["seed_start"], gd["gen"]
    assert gc["selfplay"]["games"] == gd["selfplay"]["games"] == 2000 and gc["gate"]["seeds"] == gd["gate"]["seeds"] == 100
for g in d["generations"]:
    assert g["selfplay"]["belief_resets"] == 0 and g["gate"]["belief_resets"] == 0, g["gen"]
    assert g["gate"]["candidate"].endswith("@100" + plus) and g["gate"]["baseline"].endswith("@100" + plus), g["gate"]
print(f"D matches {name}: {len(d['generations'])} generations, same seed directories, seeds and gate; {plus} gates; no belief resets")
EOF
```

Pass: `runs/p2/D.log` ends with `cut D after 108NN s` (10,800 to 10,810); the `cmp` line printed; the check prints `D matches C: ...` (branch O) or `D matches A: ...` (branch N).

- [ ] **Step 6: The tables, the final nets and the control's upgrade (controller runs this)**

```bash
ctl=C  # branch O; ctl=A in branch N
.venv/bin/python3 runs/p2/tables.py "$ctl" D | tee runs/3d2/tables.md
mkdir -p runs/3d2/nets
best=$(.venv/bin/python3 -c "import json; print(json.load(open('runs/p2/D-state-at-cut.json'))['best'])")
cp "runs/p2/D/nets/$best.bin" runs/3d2/nets/D.bin
.venv/bin/alphasettler net-upgrade "runs/p2/nets/$ctl.bin" "runs/3d2/nets/$ctl.bin" | tee -a runs/3d2/finals.log
echo "D: $best (runs/3d2/nets/D.bin)" | tee -a runs/3d2/finals.log
shasum -a 256 runs/3d2/nets/D.bin "runs/3d2/nets/$ctl.bin" runs/3d2/gen-034-v4.bin | tee -a runs/3d2/finals.log
```

Branch O only: `cp runs/3c/nets/opening.bin runs/3d2/nets/opening.bin` (the `+o` names load it beside the net). Pass: `tables.md` has both tables and both games-per-hour lines; `finals.log` names D's net and the control's upgrade line (`(encoder v3) to encoder v4 ... 24 global inputs zeroed`). If `runs/3d2/nets/D.bin` and `runs/3d2/nets/$ctl.bin` have the same sha256, skip Step 7 and record H0 (the bots are identical).

- [ ] **Step 7: The SPRT (controller runs this)**

Branch O:

```bash
.venv/bin/alphasettler sprt --candidate net:D@100+t+o --baseline net:C@100+t+o --trades \
  --seed-start 5000000000 --nets-dir runs/3d2/nets --out runs/3d2/sprt-D.jsonl 2>&1 | tee runs/3d2/sprt-D.log
```

Branch N:

```bash
.venv/bin/alphasettler sprt --candidate net:D@100+t --baseline net:A@100+t --trades \
  --seed-start 5000000000 --nets-dir runs/3d2/nets --out runs/3d2/sprt-D.jsonl 2>&1 | tee runs/3d2/sprt-D.log
```

Then:

```bash
.venv/bin/python3 -c "
import json
rs = [json.loads(l) for l in open('runs/3d2/sprt-D.jsonl')]
print(f'{len(rs)} games, belief resets {sum(r[\"belief_resets\"] for r in rs)}')
" | tee -a runs/3d2/sprt-D.log
```

The defaults are the spec's: p0 0.25, p1 0.30, alpha = beta = 0.05, batches of 12 seeds, at most 400 seeds (5,000,000,000 .. 5,000,000,399). Pass: exit 0, a line starting `sprt: ` (H1, H0 or undecided) and `belief resets 0`.

- [ ] **Step 8: Check the SPRT line against the records (controller runs this)**

```bash
grep -E '^net:D@100\+t(\+o)? vs net:[AC]@100\+t(\+o)?: |^sprt: ' runs/3d2/sprt-D.log
```

Pass: two lines, the summary and the verdict; these are what Step 10 copies.

- [ ] **Step 9: Machine and commit facts (controller runs this)**

```bash
sysctl -n machdep.cpu.brand_string; uname -r; git log -1 --format=%h
```

- [ ] **Step 10: Write `docs/bot/results-3d2.md`**

Every number is copied from the files named here; nothing is estimated. The file:

```markdown
# Sub-project 3d-2 (encoder v4: road-network and endgame-threat inputs): results

Date: <date of the SPRT>. Machine: <Step 9's CPU line>, macOS (<Step 9's uname -r>).
Commit: <Step 9's hash> (Plan 10, Task 3), plus the commit adding this file. Encoder v3 build: <runs/3d2/base.txt>.

Spec: [`docs/superpowers/specs/2026-10-08-encoder-v4-roads-threats-design.md`](../superpowers/specs/2026-10-08-encoder-v4-roads-threats-design.md).
Plan: [`docs/superpowers/plans/2026-10-08-plan-10-encoder-v4.md`](../superpowers/plans/2026-10-08-plan-10-encoder-v4.md).
Builds on: [`results-3d1b.md`](results-3d1b.md) (arm C, the opening model in self-play) and
[`results-p2-gumbel.md`](results-p2-gumbel.md) (arm A).

## Encoder v4

Per node, the fewest roads each seat must build to reach it (0 to 3, a third each; 3 or more, or
never, is 1) and whether it is an open spot; per edge, whether it lies on each seat's longest trail;
per seat, its longest road / 15, roads to the longest-road award / 5, knights to the largest-army
award / 3, points to win / 10 (the viewer's own VP cards counted) and settlements and cities left.
`PUBLIC_LEN` 1,581 → 2,163.

## The upgrade (done criterion 2)

<the first line of runs/3d2/upgrade.log>

<the two sha256 lines of runs/3d2/upgrade.log>

The v3 build's gen-034 and the v4 build's upgrade on the same 200 positions of random trading
games (runs/3d2/forward.log):

<the six "batch ..."/"single ..." lines and the "worst" line of runs/3d2/forward.log>

## Search speed (done criterion 4)

`net_speed speed PATH 500 1000 8`: one core, batch 8, 1,000 simulations, the no-trade game against
three GreedyBots; v3 is gen-034 in the encoder v3 build, v4 the same net upgraded.

| run | v3 simulations/s | v4 simulations/s | v3 ms/decision | v4 ms/decision |
|---|---|---|---|---|
| 1 | <v3 run 1> | <v4 run 1> | <v3 run 1> | <v4 run 1> |
| 2 | <v3 run 2> | <v4 run 2> | <v3 run 2> | <v4 run 2> |
| 3 | <v3 run 3> | <v4 run 3> | <v3 run 3> | <v4 run 3> |
| mean | <mean of the three> | <mean of the three> | <mean> | <mean> |

v4 / v3: <v4 mean / v3 mean, two decimals>.

## Training at equal wall time (done criterion 3)

Arm D is a `fast-loop` from gen-034 upgraded to v4, with exactly arm <C/A>'s settings, run through
the same script (`bash runs/p2/arm.sh D --start-net runs/3d2/gen-034-v4.bin<, --opening runs/3c/nets/opening.bin>`):
a fresh root, the trading game, seed directories `runs/3b-v2/gen-002` to `gen-005` and
`runs/3c/gen-030` to `gen-034` (v3-era records, re-encoded under v4 from their actions), 2,000
games a generation, a window of 40,000 games, 10 minutes of training, a gate of 100 seeds at 100
simulations; set up untimed, then 10,800 s of generations (D cut after <N from runs/p2/D.log> s).
Arm <C/A> is the control (3d-1b <H1/H0 or undecided>), not rerun. Both arms played the same
self-play and gate seeds.

<the control's table and its self-play line, from runs/3d2/tables.md>

<the D table and its self-play line, from runs/3d2/tables.md>

Final nets: <the D line, the control's upgrade line and the sha256 lines of runs/3d2/finals.log>.

| Run | Result | Bar | Outcome |
|---|---|---|---|
| `sprt --candidate net:D@100+t<+o> --baseline net:<C/A>@100+t<+o> --trades`, seeds from 5,000,000,000 | <the summary line and the sprt line of runs/3d2/sprt-D.log> | H1 (p1 0.30 against p0 0.25, alpha = beta = 0.05) | <H1: new loops start from D's best, encoder v4 / H0: recorded; new loops start from the control's best, upgraded / undecided: recorded, as H0> |

Belief resets: <the belief resets line of runs/3d2/sprt-D.log>.

## Rulings the build made

- A node another seat built on at the end of a seat's road is on that seat's network (distance 0),
  but no path leaves it. A seat with nothing on the board reads 1 everywhere. Distances ignore road
  supply and cards.
- Ties between longest trails go to the engine walk's first (starts and edges in ascending edge
  order, strictly longer replaces); `longest_road` returns `longest_trail`'s length, and the golden
  trace did not move.
- Roads to the award use the holder's trail length as the encoder computes it; with no holder, a
  seat at 5 or more reads 0 to go even when tied (the spec's formula as written).
- The viewer's points include VP cards bought this turn; `vp_to_win` scales by the game's config.
  Six scales entered the layout: distance 3, road length 15, road award 5, army award 3,
  settlements 5, cities 4.
- An upgraded net equals the original within the forward pass's 1e-5, not bit for bit: Accelerate
  sums the wider rows in another order (the bit-equal counts above).
- Encoder v2 nets upgrade straight to v4 (the trade rows reset as before); `net-upgrade` always
  writes this build's encoder. Loaders name `net-upgrade` for v2 and v3 files.
- P2's net pin keeps its digest: its net is v3's random net upgraded (`Weights::random_v3`).
- Arm D ran through `runs/p2/arm.sh` unchanged, its start net replaced by a second `--start-net`;
  the control was not rerun, and its best net was upgraded for the SPRT (`runs/3d2/nets`).
- Speed and outputs were compared against a build of the commit before Task 1.
- Fresh seeds: 5,000,000,000.

## Next

<one paragraph: if H1, new loops start from D's best and 3d-3 builds on encoder v4; if H0 or
undecided, what the two tables (held-out scores, gate results, games per hour) and the speed ratio
point to, and that new loops start from the control's best, upgraded (the code stays at v4, which
costs only the speed above)>
```

Fill every `<...>` from the named source before committing. The committed file holds numbers and text only, with no angle brackets.

- [ ] **Step 11: Commit**

Run: `grep -c '<' docs/bot/results-3d2.md`
Expected: `0`.

```bash
git add docs/bot/results-3d2.md
git diff --cached | grep -nE 'sk-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{36}|password=[^[:space:]]+'
git -c user.email=sean.may101@gmail.com commit -m "docs: 3d-2 results, encoder v4 against the 3d-1b control at equal wall time"
```

---

## Self-review

### Spec coverage

| Spec requirement | Task |
|---|---|
| §1.1 `ENCODER_VERSION` 4, the new hash, the golden position re-encoded by hand, seat rotation tests extended to the new blocks; the twin and the parity test follow from the constants | 2 (`the_layout_adds_up`, `the_golden_position_encodes_exactly`, the rotation test's `near`/`trails`/`threats` counts, `test_the_parity_positions_fill_every_v4_input_block`) |
| §1.2 `net-upgrade` v3 → v4 with identical outputs (new columns zero), checked by the Rust loader and a forward-pass comparison on real positions; gen-034 upgraded is the start net | 3 (`test_an_upgraded_v3_net_gives_the_v3_nets_outputs_on_every_position`, `upgrade_file`'s loader check), 4 (Step 3: the two builds on 200 positions; Step 5: D's start net) |
| §1.3 arm D from the upgraded gen-034, the winning 3d-1b arm's settings and seeds, 10,800 s after an untimed set-up; `net:D@100+t[+o]` against three of that arm's best; SPRT p0 0.25, p1 0.30, alpha = beta = 0.05, seeds from 5,000,000,000; H1 / H0 | 4 (Steps 1, 5–8, both branches) |
| §1.4 results: speed (one core, batch 8, 1,000 simulations, v3 vs v4 on the same net upgraded), D's table beside its control's, the SPRT line | 4 (Steps 4, 6, 10) |
| §2 per node +5 (r0..r3 road distance capped at 3 / 3, unreachable 1.0, network 0, no other seat's road, no passing another seat's building; open spot) | 2 (`road_distances`, `network`; `the_road_and_threat_blocks_encode_exactly`, `a_road_into_an_opponents_building_reaches_it_but_not_past_it`, `an_empty_board_reaches_nothing_and_every_spot_is_open`) |
| §2 per edge +4 (on one longest trail of seat r, ties to the first found) | 1 (`longest_trail`, the tie and blocked tests, the oracle property), 2 (edge slots; the tie in `the_road_and_threat_blocks_encode_exactly`) |
| §2 globals +24 after the trade block (length / 15, roads to the award, knights to the award, points to win with public VP for opponents and total for the viewer, settlements / 5, cities / 4) | 2 (`to_award`, the threat loop; `the_road_and_threat_blocks_encode_exactly`, `an_unowned_award_past_its_minimum_reads_zero_to_go`, the golden position) |
| §2 `PUBLIC_LEN` 2163, `HIDDEN_LEN` unchanged | 2 (`the_layout_adds_up`) |
| §2 Ruling: trail and length from the engine's walk, `longest_road` delegating, golden trace guards it | 1 |
| §2 Ruling: distances capped at 3 | 2 (`DISTANCE_DEN`) |
| §3 each encoder's first layer gains zero columns at the end (node 19 → 24, edge 4 → 8, global 115 → 139); every other byte copied; v4 hash; Rust loader; same outputs within 1e-5 (bit-equal at batch 1 if exact); the v2 → v3 path stays | 2 (`widened`, `_upgrade`, the v2 path to v4), 3 (v3), 4 Step 3 (bit-equal counts recorded; Rulings: not bit-equal) |
| §4 out of scope: resource coverage, a wider or deeper net, more rounds, heads, retraining the opening model, 3d-3, 3d-4 | nothing built; the Global Constraints forbid edits to the forward pass and the heads |
| §5 Rust: hand-checked distances, trail edges, threats (a blocked path, a tie); rotation; golden; layout and hash tests; `longest_road` unchanged on every existing test and the golden trace | 1, 2 |
| §5 Python: `net-upgrade` v3 → v4 identical outputs on real positions, refuses a v4 input and an existing output; parity at the new widths; a tiny `fast-loop` from an upgraded net runs | 2 (parity), 3 (`test_an_upgraded_v3_net_...`, `test_only_encoder_v2_and_v3_nets_are_upgraded`, `test_cli_net_upgrade`, `test_a_fast_loop_runs_from_an_upgraded_v3_net`) |

### Placeholder scan

Searched for "TBD", "TODO", "implement later", "fill in", "similar to Task", "appropriate", "handle edge cases": none in the code steps. The one template is Task 4 Step 10's results file, whose `<...>` marks are run-time measurements, each naming its source; Step 11 refuses to commit while one is left. The hand-computed values (distances, open spots, trails, threats, offsets, the parameter count) show their arithmetic beside them and were cross-checked while planning against the board's adjacency (`net_adjacency`) with an independent BFS. `V3_ENCODING_HASH` (`0x6d02_356c_b617_53c2`) is the encoding hash of the build at 1aeca4c and of `runs/3c/nets/gen-034.bin`'s header; the sha256 values in Task 4 Step 1 were read from the files at 1aeca4c (the opening model's from Plan 9).

### Type consistency

- Rust: `longest_trail(u128, u64) -> (u8, u128)` (Task 1) is what `encode_public` calls (Task 2). `NODE_DISTANCE`, `NODE_OPEN`, `EDGE_TRAIL`, `THREATS`, `THREAT_WIDTH`, the six `*_DEN`/`*_MIN` constants (`u32`) and `LAYOUT: [u32; 33]` are what the tests import through `settler_net::encode::*`. `V3_ENCODING_HASH: u64`, `V3_NODE_WIDTH`, `V3_EDGE_WIDTH`, `V3_GLOBAL_WIDTH: usize`, `Arch::shapes_for(node, edge, global)`, `Layer::widened(cols)` and `Weights::random_v3(arch, seed)` (Task 2) are what `bots/tests/netbot.rs`, `net/tests/weights.rs`, `bindings/src/train.rs` and Task 3's loader message use.
- Python: `netformat.shapes(width, rounds, global_width, node_width, edge_width)` and `read(..., node_width=, edge_width=)` are what `netupgrade.read_old` and the test helper `old_bytes` call; `SOURCES` values are `(version, node, edge, global)` in both tasks; `upgrade_file`'s summary keys `from`, `width`, `rounds`, `zero_inputs` (`node`, `edge`, `global`), `trade_rows` are what `_net_upgrade` prints and `test_cli_net_upgrade` checks; the helpers `old_net(widths, width, rounds, seed)`, `old_bytes(net, widths)`, `V2`, `V3` take `widths` = (hash, node, edge, global) throughout.

### Review Focus

Each of the five lines has its test in the owning task: 1–4 in Task 2 (the empty board, the road into a building, hidden VP cards, the unowned award), 5 in Task 3 (the loaders' hint, the upgrade's refusals).
