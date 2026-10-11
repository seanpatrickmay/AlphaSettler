# Targets v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the fast loop denser training targets. It records the root search value q for every searched decision and trains value on a z/q mix. It adds training-only heads for final-state targets (ownership, VP parts, award holders, turns left). An optional EMA-of-weights mode keeps Adam's state with a constant learning rate.

**Architecture:**
- **Rust** (`search`, `bots`, `net`, `bindings`):
  - computes `root_q` in the search;
  - records it as `q` in self-play records;
  - derives final-state targets once per game when records are checked;
  - writes every target into `Batch`.
- **Python** (`alphasettler/train`):
  - adds an `Extras` module of training-only heads on `Net.trunk`;
  - extends `losses`;
  - persists the extras (and, with EMA, the raw weights and Adam state) in a `<net>.train.pt` sidecar that `fast-loop` carries from generation to generation.

The exported weights file and the Rust forward pass do not change.

**Tech Stack:** Rust 1.80+ (edition 2021, no new crates), PyO3 0.29.3 + maturin 1.15.0, Python 3.12, PyTorch (already a `train` extra dependency), pytest.

**Spec:** `docs/superpowers/specs/2026-10-11-targets-v2-design.md` (binding). Written against HEAD 68f3e1d.

## Global Constraints

**Commands**
- **Python:** `.venv/bin/python3`, tests with `.venv/bin/pytest`, the CLI with `.venv/bin/alphasettler`. Never `python`, `pip`, a bare `python3`, or a bare `pip3`.
- **Rust:** `PATH=$HOME/.cargo/bin:$PATH cargo ...`. MSRV 1.80. No new crates, no new Python dependencies, no numpy.
- **Extension rebuild** after any Rust change that Python uses. Never run `maturin develop`. "Rebuild the extension" means exactly:
  ```bash
  rm -rf target/wheels-tv2 && PATH=$HOME/.cargo/bin:$PATH .venv/bin/maturin build --release -q -o target/wheels-tv2 \
    && unzip -q -o target/wheels-tv2/alphasettler-*.whl 'alphasettler/_engine*.so' -d target/wheels-tv2/x \
    && cp target/wheels-tv2/x/alphasettler/_engine.cpython-312-darwin.so alphasettler/._engine.new.so \
    && mv -f alphasettler/._engine.new.so alphasettler/_engine.cpython-312-darwin.so
  ```
- **Every task ends with:**
  1. `PATH=$HOME/.cargo/bin:$PATH cargo fmt --all`, then `cargo fmt --check` (no output);
  2. `PATH=$HOME/.cargo/bin:$PATH cargo clippy --workspace --release --all-targets -- -D warnings` (exit 0);
  3. `PATH=$HOME/.cargo/bin:$PATH cargo test --workspace --release` (all pass);
  4. the rebuild, if Rust changed;
  5. `.venv/bin/pytest -q -rs` (all pass). JVM tests that need the jar may already be failing for a missing jar; record any such failure, and do not fix it. Every other test must pass.

**Pins that must keep passing, unchanged** (never re-pin a digest):
- `engine/tests/golden_trace.rs`;
- every `*_plays_the_games_it_played_before_*`, `fixed_exploration_is_3a_self_play` and `the_generic_refactor_leaves_3a_ismcts_games_unchanged` test.

Adding `q` must not change any move, visit count or record field that exists today.

**Compatibility**
- Old record files have no `"q"`; they must load and train exactly as before.
- With `--targets-v2` and `--ema` both off, `losses`, `fast_train` and `fast-loop` behave exactly as before.

**Spec constants, verbatim**
- `Q_MIX = 0.5`
- `TURNS_DEN = 100`
- loss weights: node 0.25, edge 0.25, parts 0.25, each holder 0.1, turns 0.1
- `EMA_LR = 2.5e-4`
- `EMA_DECAY = 0.9998`
- q validation: entries finite and in [−1e-4, 1 + 1e-4], sum within 1e-3 of 1

**Sidecar**
- The sidecar path is `Path(str(net_path) + ".train.pt")`.
- Load it with `torch.load(path, weights_only=True)`, never unpickling arbitrary objects.
- It holds `{"bin_sha256": str, "extras": state_dict | None, "raw": state_dict | None, "optimizer": state_dict | None}`.

**Git**
- One commit per task, with the given message: `git -c user.email=sean.may101@gmail.com commit -m "..."`.
- No AI attribution, never `--no-verify`.
- Before committing, run `git diff --cached | grep -nE 'sk-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{36}|password=[^[:space:]]+'` and expect no output.
- Stage only the files the task names.

**Tests:** behaviour, not implementation; descriptive names; real games and real records (the existing helpers `write_explore`, `generation` in `tests/python/test_train.py`, and `recorded` in `net/tests/data.rs`).

**Machine:** a diagnostic (`runs/diag/noise_floor.py`) may be running. That is fine: it is not timed. Do not kill it.

## Review Focus

1. **Seat-relativity of every new target.** Absolute seat p must become `rel(p, actor)` for `q`, the node and edge classes, `parts` and `holders`. A test must place a known owner and check the class seen by each of the four actors (Task 3), and do the same for q (Task 2).
2. **Old records and old nets.** A record without `"q"`, an init `.bin` without a sidecar, and a `state.json` without `"targets"` or `"ema"` must each work as before (Tasks 2, 5, 6).
3. **Draws.** A turn-cap draw has winner None and no award holder, or one. Targets must still be defined, and `turns_left` must be ≥ 0 (Task 3, the `drawn()` config).
4. **Sidecar mismatch.** A sidecar whose `bin_sha256` does not match its `.bin` is a ValueError, never silently loaded (Task 5).
5. **The forward pass stays identical.** `Net.trunk` refactoring must not change `forward`'s outputs: same ops in the same order, and the existing Rust parity tests in `test_train.py` keep passing (Task 4). The implementer also checks once, before and after the refactor, that `forward` on fixed seeded inputs is `torch.equal`, and reports it.

---

### Task 1: Root search value in `SearchResult`

**Files:**
- Modify: `search/src/search.rs` (`SearchResult` at line 63, the early return at line 657, the final result at line 730).
- Test: `search/tests/search.rs`.

**Interfaces:**
- Produces: `SearchResult.root_q: Option<[f32; NUM_PLAYERS]>`, in absolute seats. It is `None` exactly when `simulations == 0`.

- [ ] **Step 1: Write the failing tests** (append to `search/tests/search.rs`):

```rust
#[test]
fn root_q_is_the_mean_value_of_every_simulation() {
    let s = city_wins(4);
    let obs = s.observation(0);
    let legal = s.legal_actions();
    let b = Belief::from_observation(&obs, 64, 4);
    let (r, tree) = search_tree(&obs, &legal, &b, &mut UniformEvaluator, &cfg(200), &mut Rng::new(1)).unwrap();
    let q = r.root_q.expect("a search that ran has a root value");
    let root = &tree.nodes[ROOT as usize];
    let visits: u32 = root.children.iter().map(|c| c.visits).sum();
    for p in 0..NUM_PLAYERS {
        // UniformEvaluator's root evaluation is 1/4 for every seat.
        let sum: f32 = 0.25 + root.children.iter().map(|c| c.value_sum[p]).sum::<f32>();
        assert!((q[p] - sum / (1 + visits) as f32).abs() < 1e-5, "seat {p}");
    }
    assert!((q.iter().sum::<f32>() - 1.0).abs() < 1e-4);
    assert!(q[0] > 0.25, "the winning city pulls the actor's value up: {q:?}");
}

#[test]
fn a_decision_returned_without_searching_has_no_root_q() {
    let s = after_setup(3, no_trades());
    let obs = s.observation(0);
    let legal = s.legal_actions();
    let b = Belief::from_observation(&obs, 8, 0);
    let r = search(&obs, &legal, &b, &mut UniformEvaluator, &cfg(500), &mut Rng::new(0)).unwrap();
    assert_eq!((r.simulations, r.root_q), (0, None));
}
```

Also add `NUM_PLAYERS` to the imports if `settler_engine::*` does not already bring it. (It does, through `use settler_engine::*`.) Add a Gumbel case to `search/tests/gumbel.rs`, using that file's `run` helper and its existing state and config: assert that `r.root_q` is `Some` and sums to 1 within 1e-4.

- [ ] **Step 2: Run them.** Command: `PATH=$HOME/.cargo/bin:$PATH cargo test -p settler-search --release --test search root_q`. Expected: a compile error, because `root_q` does not exist. (Check the crate name in `search/Cargo.toml`; use it.)

- [ ] **Step 3: Implement.** In `SearchResult`, add after `improved`:

```rust
    /// The mean of every simulation's backed-up value at the root, in absolute seats: the root's
    /// first evaluation plus every root child's value sum, over 1 + their visits (targets v2,
    /// spec Section 3). `None` when the decision was returned without searching.
    pub root_q: Option<[f32; NUM_PLAYERS]>,
```

In the early return (`root_keys.len() == 1`), set `root_q: None`. Before building the final result, compute:

```rust
    let root_q = sim.root_value.map(|v| {
        let mut sum = v;
        let mut n = 1u32;
        for c in &root.children {
            n += c.visits;
            for p in 0..NUM_PLAYERS {
                sum[p] += c.value_sum[p];
            }
        }
        sum.map(|x| x / n as f32)
    });
```

Pass `root_q` in the struct literal. `root` is the existing `let root = &sim.tree.nodes[ROOT as usize];`. Compute `root_q` before `sim.tree` is moved out.

- [ ] **Step 4: Run the tests.** The search tests pass. Then run the full end-of-task checks from Global Constraints. Any other crate that builds `SearchResult` with a literal, if one exists, needs `root_q`. `grep -rn "SearchResult {" --include=*.rs` lists none outside `search/src/search.rs`.

- [ ] **Step 5: Commit.** Run `git add search/src/search.rs search/tests/search.rs search/tests/gumbel.rs`, then commit with the message `search: the root search value, every simulation's mean backed-up value at the root`.

---

### Task 2: q through self-play records, the training set and the batch

**Files:**
- Modify:
  - `bots/src/selfplay.rs` (`Decision` line 55, the push at line 397);
  - `bindings/src/lib.rs` (the decision dict near line 365);
  - `net/src/data.rs` (`RecordedDecision`, `check_game`, `Batch`, `Batch::push`, `Batch::append`);
  - `bindings/src/train.rs` (`recorded_game` near line 100, `batch_dict` near line 150, and the `TrainingSet` doc string);
  - `alphasettler/train/data.py` (`Tensors`, `to_tensors`).
- Test:
  - `bots/tests/selfplay.rs`;
  - `net/tests/data.rs` (its `RecordedDecision` literal gains `q: None`);
  - `tests/python/test_training_data.py`.

**Interfaces:**
- Consumes: `SearchResult.root_q` (Task 1).
- Produces:
  - `Decision.q: Option<[f32; 4]>` and `RecordedDecision.q: Option<[f32; 4]>`, in absolute seats;
  - the record dict key `"q"` (a list of 4 floats, omitted when `None`);
  - `Batch.q: Vec<f32>` (n × 4, seat-relative, zeros when absent) and `Batch.has_q: Vec<u8>` (n);
  - the batch dict keys `"q"` (f32 LE bytes) and `"has_q"` (u8 bytes);
  - `Tensors.q` (n × 4 float) and `Tensors.has_q` (n float, 0 or 1).

- [ ] **Step 1: Write the failing tests.**

  `bots/tests/selfplay.rs`:

```rust
#[test]
fn every_searched_decision_records_its_root_value() {
    let g = play(3, no_trades(), 10, 0);
    for d in &g.decisions {
        let q = d.q.expect("a searched decision records q");
        assert!((q.iter().sum::<f32>() - 1.0).abs() < 1e-3, "{q:?}");
        assert!(q.iter().all(|&x| (0.0..=1.0 + 1e-4).contains(&x)));
    }
}
```

  `net/tests/data.rs`: add a helper and two tests.

```rust
/// `recorded`, with q on every decision: 0.7 for seat 1 and 0.1 for the others.
fn with_q(g: &RecordedGame) -> RecordedGame {
    let mut g = g.clone();
    for d in &mut g.decisions {
        d.q = Some([0.1, 0.7, 0.1, 0.1]);
    }
    g
}

#[test]
fn q_is_written_seat_relative_to_the_actor_and_flagged() {
    let games = vec![with_q(&recorded(1, decided())), recorded(2, decided())];
    let set = TrainingSet::new(games.clone()).unwrap();
    for (game, has) in [(0usize, 1u8), (1, 0)] {
        for (k, d) in games[game].decisions.iter().enumerate() {
            let mut b = Batch::default();
            set.position(game, k, &mut b);
            assert_eq!(b.has_q, vec![has]);
            if has == 1 {
                let r = (1 + NUM_PLAYERS - d.actor as usize) % NUM_PLAYERS; // seat 1 relative to the actor
                for i in 0..NUM_PLAYERS {
                    assert_eq!(b.q[i], if i == r { 0.7 } else { 0.1 });
                }
            } else {
                assert_eq!(b.q, vec![0.0; 4]);
            }
        }
    }
}

#[test]
fn a_bad_q_is_an_error_naming_the_decision() {
    for bad in [[0.5, 0.5, 0.5, 0.5], [f32::NAN, 0.5, 0.25, 0.25], [-0.5, 0.5, 0.5, 0.5]] {
        let mut g = recorded(1, decided());
        g.decisions[0].q = Some(bad);
        let e = TrainingSet::new(vec![g]).unwrap_err();
        assert!(e.contains("move 0") && e.contains("q"), "{e}");
    }
}
```

  Add `q: None` to the existing `RecordedDecision { .. }` literal in `recorded_every`.

  `tests/python/test_training_data.py`:

```python
def test_self_play_records_q_and_the_batch_carries_it_seat_relative(tmp_path):
    games = explore_games(tmp_path, 2)
    for g in games:
        for d in g["decisions"]:
            assert len(d["q"]) == 4 and abs(sum(d["q"]) - 1) < 1e-3
    ts = TrainingSet(games)
    t = to_tensors(ts.sample(64, 0, 1), torch.device("cpu"))
    assert t.has_q.eq(1).all()
    assert torch.allclose(t.q.sum(1), torch.ones(64), atol=1e-3)


def test_records_without_q_still_train_with_has_q_zero(tmp_path):
    games = explore_games(tmp_path, 1)
    for d in games[0]["decisions"]:
        del d["q"]
    t = to_tensors(TrainingSet(games).sample(16, 0, 1), torch.device("cpu"))
    assert t.has_q.eq(0).all() and t.q.eq(0).all()
```

  Import `torch`, `TrainingSet` and `to_tensors` in that file if they are not already there. Look at the file's existing imports first, and follow them.

- [ ] **Step 2: Run them.** Expect compile errors and a KeyError.

- [ ] **Step 3: Implement.**
  - `bots/src/selfplay.rs`:
    - add `q` to `Decision`, with the doc comment `/// The search's root value (absolute seats, targets v2); None if it did not search.`;
    - set `q: r.root_q` in the push.
  - `bindings/src/lib.rs`: after the `target` block, add `if let Some(q) = &dec.q { e.set_item("q", q.to_vec())?; }`.
  - `net/src/data.rs`:
    - add `pub q: Option<[f32; NUM_PLAYERS]>` to `RecordedDecision`, with the doc comment `/// The search's root value, absolute seats (targets v2); None in records from before it.`;
    - add `pub const Q_TOLERANCE: f64 = 1e-3;` and `pub const Q_RANGE_TOLERANCE: f32 = 1e-4;`;
    - in `check_game`, after the target checks:

```rust
            if let Some(q) = &d.q {
                if let Some((j, x)) = q.iter().enumerate().find(|(_, x)| {
                    !(x.is_finite() && **x >= -Q_RANGE_TOLERANCE && **x <= 1.0 + Q_RANGE_TOLERANCE)
                }) {
                    return Err(format!("move {i}: q entry {j} is {x}, not a probability"));
                }
                let total: f64 = q.iter().map(|&x| x as f64).sum();
                if (total - 1.0).abs() > Q_TOLERANCE {
                    return Err(format!("move {i}: q sums to {total}, not 1"));
                }
            }
```

    - `Batch`: add the fields

```rust
    /// n × 4, seat-relative: the decision's root search value; zeros without one.
    pub q: Vec<f32>,
    /// n: 1 if the decision has a root search value.
    pub has_q: Vec<u8>,
```

    - in `push`:

```rust
        match d.q {
            Some(q) => {
                for r in 0..NUM_PLAYERS {
                    self.q.push(q[abs(r, actor)]);
                }
                self.has_q.push(1);
            }
            None => {
                self.q.extend_from_slice(&[0.0; NUM_PLAYERS]);
                self.has_q.push(0);
            }
        }
```

    - in `append`, extend `q` and `has_q`.
  - `bindings/src/train.rs`:
    - in `recorded_game`, read `q` like `target`: `Some(t) if !t.is_none() => Some(<[f32; 4]>::try_from(t.extract::<Vec<f32>>()?).map_err(|_| value_error("a decision's q needs 4 entries"))?)`;
    - `batch_dict` sets `"q"` (`le_bytes`) and `"has_q"`;
    - add `q` to the `TrainingSet` doc string's list of optional fields.
  - `alphasettler/train/data.py`: in `Tensors` add `q: torch.Tensor` and `has_q: torch.Tensor`; in `to_tensors` add `q=f32("q", SEATS), has_q=u8("has_q", 1).view(n).float()`.

- [ ] **Step 4: Run the tests.** Run the end-of-task checks, including the rebuild and the full pytest. The pinned digests must pass unchanged.

- [ ] **Step 5: Commit.** Commit the named files with the message `records: the root search value q on every searched decision, seat-relative in the batch`.

---

### Task 3: Final-state targets in the batch

**Files:**
- Modify:
  - `net/src/data.rs` (`check_game` returns the final targets; `TrainingSet` stores them; `Batch` fields; `push`; `append`);
  - `bindings/src/train.rs` (`batch_dict`);
  - `alphasettler/train/data.py`.
- Test: `net/tests/data.rs`, `tests/python/test_training_data.py`.

**Interfaces:**
- Produces:
  - **The constants:**
    - `pub const TURNS_DEN: f32 = 100.0;`
    - `pub const NODE_CLASSES: usize = 9;`
    - `pub const EDGE_CLASSES: usize = 5;`
    - `pub const PART_DENS: [f32; 3] = [5.0, 4.0, 5.0];`
  - **`Batch` fields:**
    - `own_node: Vec<u8>` (n × 54);
    - `own_edge: Vec<u8>` (n × 72);
    - `parts: Vec<f32>` (n × 12, ordered relative seat r major, then settlements, cities, VP cards);
    - `holders: Vec<u8>` (n × 2: road, then army; 0 none, 1 + relative seat);
    - `turns_left: Vec<f32>` (n).
  - **The batch dict keys** carry the same names.
  - **`Tensors` fields:**
    - `own_node` (long n × 54);
    - `own_edge` (long n × 72);
    - `parts` (float n × 12);
    - `holders` (long n × 2);
    - `turns_left` (float n).

  Node i is bit i of `settlements` and `cities`, and edge i is bit i of `roads`, the encoder's order. Check this against `net/src/encode.rs`, which reads the same bitboards, and use the topology constants `NUM_NODES` and `NUM_EDGES` (54 and 72).

- [ ] **Step 1: Write the failing tests** in `net/tests/data.rs`.

```rust
/// The final state of `g`, replayed.
fn final_state(g: &RecordedGame) -> State {
    let mut s = State::new(g.seed, g.config);
    for &a in &g.actions {
        s.apply(a);
    }
    s
}

#[test]
fn final_ownership_parts_holders_and_turns_left_are_seat_relative() {
    for (seed, config) in [(1, decided()), (7, drawn())] {
        let g = recorded(seed, config);
        let f = final_state(&g);
        let set = TrainingSet::new(vec![g.clone()]).unwrap();
        for (k, d) in g.decisions.iter().enumerate() {
            let mut b = Batch::default();
            set.position(0, k, &mut b);
            let rel = |p: usize| (p + NUM_PLAYERS - d.actor as usize) % NUM_PLAYERS;
            for n in 0..54 {
                let want = (0..NUM_PLAYERS)
                    .find_map(|p| {
                        if f.players[p].settlements >> n & 1 == 1 { Some(1 + rel(p) as u8) }
                        else if f.players[p].cities >> n & 1 == 1 { Some(5 + rel(p) as u8) }
                        else { None }
                    })
                    .unwrap_or(0);
                assert_eq!(b.own_node[n], want, "node {n}");
            }
            for e in 0..72 {
                let want = (0..NUM_PLAYERS)
                    .find(|&p| f.players[p].roads >> e & 1 == 1)
                    .map_or(0, |p| 1 + rel(p) as u8);
                assert_eq!(b.own_edge[e], want, "edge {e}");
            }
            for p in 0..NUM_PLAYERS {
                let r = rel(p);
                let pl = &f.players[p];
                assert_eq!(b.parts[r * 3], pl.settlements.count_ones() as f32 / 5.0);
                assert_eq!(b.parts[r * 3 + 1], pl.cities.count_ones() as f32 / 4.0);
                assert_eq!(b.parts[r * 3 + 2], pl.dev_hand[DevCard::VictoryPoint.index()] as f32 / 5.0);
            }
            let holder = |o: Option<PlayerId>| o.map_or(0, |p| 1 + rel(p as usize) as u8);
            assert_eq!(b.holders, vec![holder(f.longest_road_owner), holder(f.largest_army_owner)]);
            let mut s = State::new(g.seed, g.config);
            for &a in &g.actions[..d.index as usize] {
                s.apply(a);
            }
            assert_eq!(b.turns_left, vec![(f.turn - s.turn) as f32 / 100.0]);
            assert!(b.turns_left[0] >= 0.0);
        }
    }
}

#[test]
fn a_batch_of_many_positions_lays_the_new_targets_out_row_major() {
    let set = TrainingSet::new(vec![recorded(1, decided()), recorded(2, drawn())]).unwrap();
    let b = set.sample(40, 3, 2).unwrap();
    assert_eq!((b.own_node.len(), b.own_edge.len(), b.parts.len(), b.holders.len(), b.turns_left.len()),
               (40 * 54, 40 * 72, 40 * 12, 40 * 2, 40));
}
```

  Import `DevCard` and `PlayerId` if `settler_engine::*` does not bring them in.

  In `tests/python/test_training_data.py`:

```python
def test_the_batch_carries_the_final_state_targets_with_their_shapes(tmp_path):
    t = to_tensors(TrainingSet(explore_games(tmp_path, 2)).sample(32, 0, 1), torch.device("cpu"))
    assert t.own_node.shape == (32, 54) and t.own_node.max() <= 8
    assert t.own_edge.shape == (32, 72) and t.own_edge.max() <= 4
    assert t.parts.shape == (32, 12) and t.holders.shape == (32, 2) and t.turns_left.shape == (32,)
    assert (t.turns_left >= 0).all()
    assert (t.own_node > 0).any(), "finished games have buildings"
```

- [ ] **Step 2: Run them.** They fail to compile, or fail with a KeyError.

- [ ] **Step 3: Implement.**
  - In `net/src/data.rs`, add a private struct:

```rust
/// A game's final-state targets (targets v2, spec Section 4), in absolute seats; ~150 bytes.
#[derive(Clone, Debug)]
struct Final {
    node: [u8; NUM_NODES],   // 0 empty, 1 + p settlement, 5 + p city
    edge: [u8; NUM_EDGES],   // 0 none, 1 + p road
    parts: [[u8; 3]; NUM_PLAYERS], // settlements, cities, VP cards held
    road: Option<PlayerId>,
    army: Option<PlayerId>,
    turn: u32,
}

impl Final {
    fn of(s: &State) -> Final { /* fill from s.players[p].settlements/cities/roads, dev_hand[VictoryPoint],
                                  s.longest_road_owner, s.largest_army_owner, s.turn */ }
}
```

  - `check_game` returns `Result<(Vec<u32>, Final), String>`, building `Final::of(&s)` after the final checks.
  - `TrainingSet` gains `finals: Vec<Final>`, extended in `extend` in the same order as `games`. `Default` gives an empty vec.
  - `position` passes `&self.finals[game]` and the state's `turn` to `push`.
  - `push(&mut self, s, game, d, legal, f: &Final)` appends:
    - per node, `match f.node[n] { 0 => 0, c @ 1..=4 => 1 + rel(c - 1, actor), c => 5 + rel(c - 5, actor) }`, with `rel` taking a `PlayerId`, so cast;
    - per edge, the same with 1..=4;
    - `parts[r * 3 + j] = f.parts[abs(r, actor)][j] as f32 / PART_DENS[j]`;
    - `holders` as 0 or 1 + `rel`;
    - `turns_left = (f.turn - s.turn) as f32 / TURNS_DEN`.

    Use `saturating_sub`: a decision's turn is never past the final turn.
  - `append` extends all five fields.
  - `bindings/src/train.rs` `batch_dict`: add `"own_node"`, `"own_edge"` and `"holders"` as raw u8 bytes, and `"parts"` and `"turns_left"` through `le_bytes`.
  - `alphasettler/train/data.py`: add the five `Tensors` fields:
    - `own_node=u8("own_node", NODES).long()`;
    - `own_edge=u8("own_edge", EDGES).long()`;
    - `parts=f32("parts", 12)`;
    - `holders=u8("holders", 2).long()`;
    - `turns_left=f32("turns_left", 1).view(n)`.

    `NODES` and `EDGES` come from `alphasettler.netformat`, which already defines them; import from there.

- [ ] **Step 4: Run the tests.** Run the end-of-task checks (rebuild included).

- [ ] **Step 5: Commit.** Use the message `data: final-state targets (ownership, VP parts, award holders, turns left) in every batch`.

---

### Task 4: Training-only heads and the targets-v2 loss

**Files:**
- Create: `alphasettler/train/extras.py`.
- Modify:
  - `alphasettler/train/model.py` (`Net.trunk`, `forward` built on it);
  - `alphasettler/train/train.py` (`losses`, `held_out_losses`).
- Test: `tests/python/test_targets_v2.py` (new).

**Interfaces:**
- Consumes: `Tensors.q`, `has_q`, `own_node`, `own_edge`, `parts`, `holders`, `turns_left` (Tasks 2 and 3).
- Produces:
  - `Net.trunk(public, hidden) -> tuple[node (b×54×d), edge (b×72×d), tile (b×19×d), glob (b×d), pooled (b×4d), v (b×4d)]`;
  - `extras.Extras(d)`, an `nn.Module` with `forward(node, edge, glob, v) -> dict[str, Tensor]` and the keys `node` (b×54×9), `edge` (b×72×5), `parts` (b×12), `road` (b×5), `army` (b×5), `turns` (b);
  - `extras.extras_losses(out: dict, t: Tensors) -> dict[str, Tensor]`, with the keys `extras_node`, `extras_edge`, `extras_parts`, `extras_road`, `extras_army`, `extras_turns`, `extras` (the weighted sum);
  - the constants `Q_MIX = 0.5` and `WEIGHTS = {"node": 0.25, "edge": 0.25, "parts": 0.25, "road": 0.1, "army": 0.1, "turns": 0.1}` in `extras.py`;
  - `losses(net, t, value_rows=None, extras=None)`. With `extras` (an `Extras`), the value target on rows with `has_q` is `(1 - Q_MIX) * outcome + Q_MIX * q`, and `total` adds `parts["extras"]`, computed on the `value_rows`. The returned dict gains the `extras_*` keys. With `extras=None`, the result is identical to today's;
  - `held_out_losses(net, batches, extras=None)`. Value is always against `outcome` (z) on every row. With `extras`, it also reports the mean `extras_*` terms. The `score` stays `policy + value`.

- [ ] **Step 1: Write the failing tests** (`tests/python/test_targets_v2.py`):

```python
"""Targets v2 (spec 2026-10-11): the training-only heads, the q mix and the extra losses."""

import torch

from alphasettler._engine import TrainingSet
from alphasettler.train import model as model_mod
from alphasettler.train.data import to_tensors
from alphasettler.train.extras import Q_MIX, WEIGHTS, Extras, extras_losses
from alphasettler.train.train import held_out_losses, losses

from test_training_data import explore_games


def batch(tmp_path, n=32):
    return to_tensors(TrainingSet(explore_games(tmp_path, 2)).sample(n, 0, 1), torch.device("cpu"))


def test_forward_is_unchanged_by_the_trunk_refactor(tmp_path):
    torch.manual_seed(0)
    net = model_mod.Net(8, 1).eval()
    t = batch(tmp_path)
    p, v, a = net(t.public, t.hidden)
    node, edge, tile, glob, pooled, h = net.trunk(t.public, t.hidden)
    assert node.shape[1:] == (54, 8) and edge.shape[1:] == (72, 8) and glob.shape[1:] == (8,)
    assert torch.equal(v, net.value(h)) and torch.equal(a, net.aux(h))
    # A reloaded copy is the same function; bit-identity with the pre-refactor code is pinned by
    # the Rust parity tests (test_train.py), which compare against an unchanged Rust forward pass.
    p2, v2, a2 = model_mod.from_bytes(model_mod.to_bytes(net))(t.public, t.hidden)
    assert torch.equal(p, p2) and torch.equal(v, v2) and torch.equal(a, a2)


def test_without_extras_the_losses_are_todays(tmp_path):
    torch.manual_seed(0)
    net = model_mod.Net(8, 1)
    t = batch(tmp_path)
    plain = losses(net, t)
    assert set(plain) == {"policy", "value", "aux", "l2", "total"}


def test_the_value_target_mixes_z_and_q_on_rows_with_q(tmp_path):
    torch.manual_seed(0)
    net, ex = model_mod.Net(8, 1), Extras(8)
    t = batch(tmp_path)
    t.q = torch.zeros_like(t.q); t.q[:, 0] = 1.0  # q says relative seat 0 wins
    t.has_q = torch.ones_like(t.has_q)
    logits = net(t.public, t.hidden)[1]
    want = -(((1 - Q_MIX) * t.outcome + Q_MIX * t.q) * torch.log_softmax(logits, 1)).sum(1).mean()
    assert torch.allclose(losses(net, t, None, ex)["value"], want, atol=1e-6)
    t.has_q = torch.zeros_like(t.has_q)
    z_only = -(t.outcome * torch.log_softmax(logits, 1)).sum(1).mean()
    assert torch.allclose(losses(net, t, None, ex)["value"], z_only, atol=1e-6)


def test_the_extra_losses_are_weighted_into_the_total_and_train_the_heads(tmp_path):
    torch.manual_seed(0)
    net, ex = model_mod.Net(8, 1), Extras(8)
    t = batch(tmp_path)
    parts = losses(net, t, None, ex)
    assert {"extras_node", "extras_edge", "extras_parts", "extras_road", "extras_army", "extras_turns"} <= set(parts)
    weighted = sum(WEIGHTS[k] * parts[f"extras_{k}"] for k in WEIGHTS)
    assert torch.allclose(parts["extras"], weighted)
    base = parts["policy"] + parts["value"] + 0.25 * parts["aux"] + 1e-4 * parts["l2"]
    assert torch.allclose(parts["total"], base + parts["extras"], atol=1e-5)
    opt = torch.optim.Adam(list(net.parameters()) + list(ex.parameters()), lr=1e-2)
    first = parts["extras"].item()
    for _ in range(30):
        p = losses(net, t, None, ex); opt.zero_grad(); p["total"].backward(); opt.step()
    assert losses(net, t, None, ex)["extras"].item() < first


def test_held_out_value_is_scored_against_z_even_with_extras(tmp_path):
    torch.manual_seed(0)
    net, ex = model_mod.Net(8, 1), Extras(8)
    t = batch(tmp_path)
    plain, with_extras = held_out_losses(net, [t]), held_out_losses(net, [t], ex)
    assert with_extras["value"] == plain["value"] and with_extras["score"] == plain["score"]
    assert "extras_node" in with_extras and "extras_node" not in plain
```

  `tests/python` is on the path for sibling imports, as `test_fast_train.py` imports `test_train`.

- [ ] **Step 2: Run them.** Command: `.venv/bin/pytest -q tests/python/test_targets_v2.py`. Expected: an ImportError.

- [ ] **Step 3: Implement.**

  `model.py`: move the body of `forward`, up to `v = relu(...)`, into `trunk`, returning `(node, edge, tile, glob, pooled, v)`. `forward` becomes:

```python
    def forward(self, public: torch.Tensor, hidden: torch.Tensor):
        node, edge, tile, glob, pooled, v = self.trunk(public, hidden)
        build = self.node_policy(torch.cat([node, _spread(glob, NODES)], 2))
        road = self.edge_policy(torch.cat([edge, _spread(glob, EDGES)], 2))[:, :, 0]
        robber = self.tile_policy(torch.cat([tile, _spread(glob, TILES)], 2))[:, :, 0]
        other = self.other_policy(torch.relu(self.other_hidden(pooled)))
        s = NET_SETTLEMENT_BASE
        policy = torch.cat([other[:, :s], build[:, :, 0], build[:, :, 1], road, robber, other[:, s:]], 1)
        return policy, self.value(v), self.aux(v)
```

  The operations and their order must be exactly today's: same ops, same order.

  `extras.py`:

```python
"""Targets v2's training-only heads (spec 2026-10-11 Section 4): final ownership per node and edge,
VP parts, the award holders and turns left, on the trunk's node, edge and global embeddings and the
value head's hidden layer. They are never in the weights file: `fast_train` keeps them in the
`.train.pt` sidecar."""

from __future__ import annotations

import torch
from torch import nn

from alphasettler.netformat import EDGES, NODES

Q_MIX = 0.5
NODE_CLASSES, EDGE_CLASSES, HOLDER_CLASSES, PARTS = 9, 5, 5, 12
WEIGHTS = {"node": 0.25, "edge": 0.25, "parts": 0.25, "road": 0.1, "army": 0.1, "turns": 0.1}


class Extras(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.d = d
        self.node = nn.Linear(2 * d, NODE_CLASSES)
        self.edge = nn.Linear(2 * d, EDGE_CLASSES)
        self.scalar = nn.Linear(4 * d, PARTS + 2 * HOLDER_CLASSES + 1)

    def forward(self, node, edge, glob, v) -> dict[str, torch.Tensor]:
        g = glob.unsqueeze(1)
        s = self.scalar(v)
        return {"node": self.node(torch.cat([node, g.expand(-1, NODES, -1)], 2)),
                "edge": self.edge(torch.cat([edge, g.expand(-1, EDGES, -1)], 2)),
                "parts": s[:, :PARTS], "road": s[:, PARTS:PARTS + 5], "army": s[:, PARTS + 5:PARTS + 10],
                "turns": s[:, -1]}


def extras_losses(out: dict, t) -> dict[str, torch.Tensor]:
    ce = nn.functional.cross_entropy
    parts = {
        "extras_node": ce(out["node"].reshape(-1, NODE_CLASSES), t.own_node.reshape(-1)),
        "extras_edge": ce(out["edge"].reshape(-1, EDGE_CLASSES), t.own_edge.reshape(-1)),
        "extras_parts": nn.functional.mse_loss(out["parts"], t.parts),
        "extras_road": ce(out["road"], t.holders[:, 0]),
        "extras_army": ce(out["army"], t.holders[:, 1]),
        "extras_turns": nn.functional.mse_loss(out["turns"], t.turns_left),
    }
    parts["extras"] = sum(WEIGHTS[k] * parts[f"extras_{k}"] for k in WEIGHTS)
    return parts
```

  `train.py` `losses(net, t, value_rows=None, extras=None)`:
  - when `extras is None`, keep today's code path exactly;
  - otherwise call `net.trunk` once and build `policy, value, aux` from it, with the same head code as `forward`. To avoid duplicating it, add a `Net.heads(node, edge, tile, glob, pooled, v)` that `forward` calls too, and use that;
  - compute `out = extras(node, edge, glob, v)`;
  - the value target is `torch.where(t.has_q[:, None] > 0, (1 - Q_MIX) * t.outcome + Q_MIX * t.q, t.outcome)`;
  - select `value_rows` for value, aux and every extras tensor, and for the `Tensors` fields they read. Build a row-sliced view object: `types.SimpleNamespace(**{k: getattr(t, k)[rows] ...})` over the fields `extras_losses` reads;
  - `total += parts["extras"]`.

  `held_out_losses(net, batches, extras=None)`:
  - put `extras` in eval mode too, and restore its mode afterwards;
  - for each batch, compute `losses(net, t)` (z only, so the value and score are unchanged) and, with `extras`, add the `extras_*` terms of `losses(net, t, None, extras)`;
  - average every key it reports.

- [ ] **Step 4: Run the tests.** `tests/python/test_targets_v2.py`, `test_train.py` and `test_fast_train.py` all pass, and so does the full pytest.

- [ ] **Step 5: Commit.** Use the message `train: training-only heads for the final-state targets and the z/q value mix (targets v2)`.

---

### Task 5: `--targets-v2` in fast-train and fast-loop, and the sidecar

**Files:**
- Modify:
  - `alphasettler/train/fast.py` (`FastConfig.targets`, `fast_train`, `sidecar`);
  - `alphasettler/train/fastloop.py` (state `"targets"`, `_train`);
  - `alphasettler/cli.py` (the `fast-train --targets-v2` and `fast-loop --targets-v2` flags, and the mismatch error).
- Test: `tests/python/test_fast_train.py`, `tests/python/test_fast_loop.py`.

**Interfaces:**
- Consumes: `Extras`, the extended `losses` and `held_out_losses` (Task 4).
- Produces:
  - `fast.sidecar(path) -> Path`, which is `Path(str(path) + ".train.pt")`;
  - `FastConfig.targets: int = 1`, where `check()` accepts only 1 or 2;
  - `fast_train`:
    - with `targets == 2`, it creates `Extras(net.width)`, or loads `extras` from `sidecar(init)`;
    - it trains the extras on the value rows;
    - after writing `out`, it writes `sidecar(out)` atomically, as a temp file then `os.replace`, holding `{"bin_sha256": files.sha256_file(out), "extras": extras.state_dict(), "raw": None, "optimizer": None}`;
    - the summary gains `"targets"`.
  - **Loading.** If `init` is given and `sidecar(init)` exists, it is loaded with `torch.load(p, weights_only=True)`. If its `bin_sha256 != files.sha256_file(init)`, that is `ValueError(f"{p} is not the sidecar of {init} (sha256 mismatch)")`. If it is missing and `targets == 2`, `fast_train` prints `notice: no sidecar for {init}; the training-only heads start fresh`.
  - `fast-loop`:
    - `init(..., targets_v2=False)` stores `"targets": 2 if targets_v2 else 1`;
    - `FastLoop.targets = state.get("targets", 1)`;
    - `_train` passes `targets=self.targets` and, after copying `out` to `nets/gen-NNN.bin`, also copies `sidecar(out)` to `sidecar(nets/gen-NNN.bin)` when it exists.
  - CLI:
    - `fast-train --targets-v2`;
    - `fast-loop --targets-v2` (first run only). A resume that passes `--targets-v2` to a loop set up without it is the error `{root} was set up without --targets-v2, and a loop keeps the setting it was set up with`, exit 2, matching `--trades`.

- [ ] **Step 1: Write the failing tests.**

  `test_fast_train.py`:

```python
def test_a_targets_v2_run_writes_a_sidecar_that_the_next_run_continues_from(tmp_path):
    pool = load_pool(tmp_path, two_gens(tmp_path))
    cfg = dataclasses.replace(TINY, targets=2)
    first = tmp_path / "a.bin"
    s = fast_train(pool, first, cfg)
    side = torch.load(sidecar(first), weights_only=True)
    assert s["targets"] == 2 and side["bin_sha256"] == files.sha256_file(first)
    assert side["extras"] and side["raw"] is None and side["optimizer"] is None
    assert "extras_node" in s["best"]
    second = tmp_path / "b.bin"
    fast_train(pool, second, cfg, first)  # loads a.bin.train.pt
    assert sidecar(second).exists()


def test_a_sidecar_that_does_not_match_its_net_is_an_error(tmp_path):
    pool = load_pool(tmp_path, two_gens(tmp_path))
    cfg = dataclasses.replace(TINY, targets=2)
    a, b = tmp_path / "a.bin", tmp_path / "b.bin"
    fast_train(pool, a, cfg)
    fast_train(pool, b, dataclasses.replace(cfg, seed=1))
    sidecar(b).replace(sidecar(a))  # a's sidecar now names b's sha256
    with pytest.raises(ValueError, match="sha256 mismatch"):
        fast_train(pool, tmp_path / "c.bin", cfg, a)


def test_targets_v2_from_a_net_without_a_sidecar_starts_fresh_heads(tmp_path, capsys):
    pool = load_pool(tmp_path, two_gens(tmp_path))
    a = tmp_path / "a.bin"
    fast_train(pool, a, TINY)  # targets 1 writes no sidecar
    assert not sidecar(a).exists()
    fast_train(pool, tmp_path / "b.bin", dataclasses.replace(TINY, targets=2), a)
    assert "training-only heads start fresh" in capsys.readouterr().out


def test_targets_must_be_1_or_2():
    with pytest.raises(ValueError):
        dataclasses.replace(TINY, targets=3).check()
```

  Add the imports `dataclasses`, `from alphasettler import files` and `sidecar` (from `alphasettler.train.fast`).

  `test_fast_loop.py`: use the file's existing `setup` helper and its CLI-driving pattern (read `test_a_trading_loop_plays_and_gates_in_the_trading_game` and `test_cli_fast_loop_keeps_the_trading_setting_it_was_set_up_with`, then follow them exactly). Add:
  - `test_a_targets_v2_loop_records_the_setting_and_carries_the_sidecar_between_generations`. Set up with `--targets-v2` and run two tiny generations. Assert:
    - `state["targets"] == 2`;
    - `nets/gen-NNN.bin.train.pt` exists for both;
    - the second generation's training printed no "start fresh" notice;
  - `test_cli_fast_loop_keeps_the_targets_setting_it_was_set_up_with`: a loop set up without the flag, resumed with `--targets-v2`, exits 2 with the message;
  - `test_a_loop_set_up_before_targets_v2_reads_as_targets_1`: delete `"targets"` from `state.json`, and `FastLoop(...).targets == 1`.

- [ ] **Step 2: Run them.** They fail.

- [ ] **Step 3: Implement**, as the Interfaces block says.
  - In `fast_train`, when `targets == 2`:
    - include the extras' parameters in the one Adam optimizer;
    - move the extras to the device;
    - call `losses(net, batch, rows, extras)`;
    - call `held_out_losses(net, evals, extras)`;
    - keep the best extras state together with the best net state, so the sidecar's heads match the exported weights.
  - Write the sidecar only for `targets == 2` (or with `ema`, in Task 6).
  - The summary's `"best"` row carries the `extras_*` keys (they come from `held_out_losses`).

- [ ] **Step 4: Run the tests.** Run the full pytest and the end-of-task checks. No Rust changed, so no rebuild is needed.

- [ ] **Step 5: Commit.** Use the message `fast loop: --targets-v2, the training-only heads carried generation to generation in a sidecar`.

---

### Task 6: `--ema` (weight averaging, Adam carried over, a constant learning rate)

**Files:**
- Modify: `alphasettler/train/fast.py`, `alphasettler/train/fastloop.py`, `alphasettler/cli.py`.
- Test: `tests/python/test_fast_train.py`, `tests/python/test_fast_loop.py`.

**Interfaces:**
- Consumes: `sidecar` and the sidecar format (Task 5).
- Produces:
  - `EMA_DECAY = 0.9998` and `EMA_LR = 2.5e-4` in `fast.py`;
  - `FastConfig.ema: bool = False`;
  - `lr_at(cfg, step, elapsed, warm=True)`. With `cfg.ema`, it returns the linear warmup over `cfg.warmup` steps when `warm`, then the constant `cfg.lr`. Without `ema`, it is today's schedule;
  - `fast_train` with `cfg.ema`:
    - the raw net, extras and Adam continue from `sidecar(init)["raw"]`, `["extras"]` and `["optimizer"]` when present (the warmup is skipped when `optimizer` is present);
    - otherwise the raw net starts from `init`;
    - the EMA copy starts from `init`'s weights, which are the previous EMA, or from the raw net when there is no init;
    - after every step, `ema_p.mul_(EMA_DECAY).add_(p.detach(), alpha=1 - EMA_DECAY)` for every parameter;
    - held-out evaluation and the best-state selection use the EMA net, with the raw extras;
    - the export is the best EMA weights;
    - the sidecar holds the final step's (not the best step's) raw state dict, extras and optimizer state dict;
  - `fast-loop`:
    - `init(..., ema=False)` stores `"ema": bool`;
    - `FastLoop.ema`;
    - `_train` passes `ema=self.ema` and `lr=EMA_LR if self.ema else self.s.lr`;
  - CLI: `fast-train --ema`, and `fast-loop --ema` (first run only), with the mismatch error `{root} was set up without --ema, and a loop keeps the setting it was set up with`.

- [ ] **Step 1: Write the failing tests.**

```python
def test_with_ema_the_learning_rate_is_constant_after_a_warmup_only_when_asked():
    cfg = FastConfig(minutes=1.0, lr=2.5e-4, warmup=10, ema=True)
    assert lr_at(cfg, 0, 0.0) == pytest.approx(2.5e-5)
    assert lr_at(cfg, 50, 59.0) == pytest.approx(2.5e-4)
    assert lr_at(cfg, 0, 0.0, warm=False) == pytest.approx(2.5e-4)


def test_an_ema_run_exports_the_average_and_carries_raw_weights_and_adam_on(tmp_path):
    pool = load_pool(tmp_path, two_gens(tmp_path))
    cfg = dataclasses.replace(TINY, targets=2, ema=True)
    a = tmp_path / "a.bin"
    fast_train(pool, a, cfg)
    side = torch.load(sidecar(a), weights_only=True)
    assert side["raw"] and side["optimizer"] and side["extras"]
    exported = model_mod.load(a).state_dict()
    assert any(not torch.equal(exported[k], side["raw"][k]) for k in exported), "the export is the average"
    b = tmp_path / "b.bin"
    s = fast_train(pool, b, cfg, a)
    assert s["warmup_skipped"] is True
```

  The summary gains `"warmup_skipped": bool` (true when Adam state was loaded) and `"ema": bool`. Import `model_mod` from `alphasettler.train`.

  `test_fast_loop.py`:
  - `test_an_ema_loop_records_the_setting_and_trains_at_the_constant_rate`. Set up with `--targets-v2 --ema`, run two generations, and assert:
    - `state["ema"] is True`;
    - each generation's `net.bin.json` config has `lr == 2.5e-4` and `ema` true;
    - the second generation's summary has `warmup_skipped` true;
  - the mismatch error for `--ema`;
  - `test_a_loop_set_up_before_ema_reads_as_without_it`.

- [ ] **Step 2: Run them.** They fail.

- [ ] **Step 3: Implement**, as the Interfaces block says. Load optimizer state with `opt.load_state_dict`, after the optimizer is built over the same parameter list in the same order: the net's parameters, then the extras'.

- [ ] **Step 4: Run the tests.** Run the full pytest and the end-of-task checks.

- [ ] **Step 5: Commit.** Use the message `fast loop: --ema, an exported weight average with Adam carried over and a constant learning rate`.

---

### Task 7 (controller): the arms

This task is not dispatched to a subagent. The controller writes `runs/tv2/arms.sh` (force-added) and launches it in the background:

1. **Wait for the machine.** Wait until `noise_floor.py` has finished and `bash runs/jsettlers/idle.sh` passes.
2. **Arm T.**
   - Set up `fast-loop --dir runs/tv2/T --generations 40 --first 41 --trades --targets-v2`, with `long-A.sh`'s exact `--seed-dirs` and `--start-net runs/3d2/nets/A.bin`.
   - Run the generations up to 56 under `caffeinate -dims`.
   - Copy the best net to `runs/tv2/nets/T.bin`, and copy A to `runs/tv2/nets/A.bin`.
   - SPRT `net:T@100+t` against `net:A@100+t`: `--trades --p0 0.25 --p1 0.30 --alpha 0.05 --beta 0.05 --batch 12 --max-seeds 400 --threads 8 --seed-start 12000000000 --nets-dir runs/tv2/nets`, writing `runs/tv2/sprt-T-vs-A.{jsonl,log}`.
3. **D5.** Run `runs/diag/learning-curve.sh`, which was deferred so its 45-minute timed trainings get an idle machine (ledger Ruling).
4. **Arm E.**
   - Set up `runs/tv2/E` the same as T, plus `--ema`. Run generations 41..56. Copy the best net to `runs/tv2/nets/E.bin`.
   - SPRT it against T's best if T's SPRT ended in H1, otherwise against A, from seed 13,000,000,000, writing `runs/tv2/sprt-E-vs-{T,A}.{jsonl,log}`.
5. **Results.** Then the controller writes `docs/bot/results-targets-v2.md` (spec Done criterion 4) and fills in spec Section 1.
