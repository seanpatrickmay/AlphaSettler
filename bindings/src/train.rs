//! Network and training helpers for Python (spec Section 4): encode a position, replay-and-sample
//! training batches returned as raw little-endian buffers, the Rust forward pass, network v2's
//! adjacency for the PyTorch twin, net lookup and test weights.
//!
//! Records are files someone else wrote: every field is type- and range-checked here, and
//! `TrainingSet::extend` replays each game against its record, so a malformed record is a
//! ValueError naming the game, never a panic or an out-of-range index.

use crate::convert::*;
use crate::PyGame;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};
use settler_bots::netbot::{load_net, net_path, DEFAULT_NETS_DIR};
use settler_engine::{PlayerId, NUM_PLAYERS};
use settler_net::data::{Batch, RecordedDecision, RecordedGame, TrainingSet};
use settler_net::encode::{
    encode_state, EDGE_WIDTH, ENCODING_HASH, HIDDEN_LEN, NODE_WIDTH, PUBLIC_LEN, TILE_WIDTH,
};
use settler_net::forward::{Forward, Outputs};
use settler_net::graph::{graph, GLOBAL_WIDTH};
use settler_net::policy::SETTLEMENT_BASE;
use settler_net::weights::{
    Arch, Weights, FORMAT_VERSION, MAX_ROUNDS, MAX_WIDTH, V2_ENCODING_HASH, V2_GLOBAL_WIDTH,
};
use std::path::PathBuf;

fn le_bytes(xs: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(4 * xs.len());
    for x in xs {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}

/// Little-endian f32s. (Written for the workspace's Rust 1.80: no `is_multiple_of` or
/// `as_chunks`.)
fn floats(bytes: &[u8], what: &str) -> PyResult<Vec<f32>> {
    let n = bytes.len() / 4;
    if 4 * n != bytes.len() {
        return Err(value_error(format!(
            "{what} holds {} bytes, not a whole number of f32s",
            bytes.len()
        )));
    }
    Ok((0..n)
        .map(|i| {
            f32::from_le_bytes([
                bytes[4 * i],
                bytes[4 * i + 1],
                bytes[4 * i + 2],
                bytes[4 * i + 3],
            ])
        })
        .collect())
}

fn field<'py>(d: &Bound<'py, PyDict>, key: &str, what: &str) -> PyResult<Bound<'py, PyAny>> {
    d.get_item(key)?
        .ok_or_else(|| value_error(format!("{what} is missing {key:?}")))
}

/// A list of ints, each range-checked into `T`.
fn int_list<T: TryFrom<i128>>(v: &Bound<'_, PyAny>, what: &str) -> PyResult<Vec<T>> {
    v.extract::<Vec<Bound<'_, PyAny>>>()?
        .iter()
        .map(|x| int_arg(x, what))
        .collect()
}

/// A seat 0..NUM_PLAYERS.
fn seat(v: &Bound<'_, PyAny>, what: &str) -> PyResult<PlayerId> {
    let p: u8 = int_arg(v, what)?;
    if p as usize >= NUM_PLAYERS {
        return Err(value_error(format!(
            "{what} {p} out of range 0..{NUM_PLAYERS}"
        )));
    }
    Ok(p)
}

fn recorded_game(g: &Bound<'_, PyAny>) -> PyResult<RecordedGame> {
    let g = g.cast::<PyDict>()?;
    let config_obj = field(g, "config", "game record")?;
    let config = parse_config(Some(config_obj.cast::<PyDict>()?))?;
    let actions = field(g, "actions", "game record")?
        .extract::<Vec<Bound<'_, PyAny>>>()?
        .iter()
        .map(action_from_id)
        .collect::<PyResult<Vec<_>>>()?;
    let mut decisions = Vec::new();
    for d in field(g, "decisions", "game record")?.try_iter()? {
        let d = d?;
        let d = d.cast::<PyDict>()?;
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
    }
    let vp: Vec<u8> = int_list(&field(g, "vp", "game record")?, "vp")?;
    let vp: [u8; NUM_PLAYERS] = vp
        .try_into()
        .map_err(|_| value_error("a game record's vp needs 4 entries"))?;
    let winner = field(g, "winner", "game record")?;
    let winner = if winner.is_none() {
        None
    } else {
        Some(seat(&winner, "winner")?)
    };
    Ok(RecordedGame {
        seed: int_arg(&field(g, "seed", "game record")?, "seed")?,
        config,
        actions,
        decisions,
        winner,
        vp,
    })
}

/// Every game of the iterable `games`, parsed. An `Exception` (a malformed record, or a reader
/// error such as a truncated file's EOFError) becomes a ValueError naming the game's index in the
/// whole set, with the original as its `__cause__`; anything else (KeyboardInterrupt,
/// SystemExit) propagates unchanged so Ctrl-C handlers still fire.
fn recorded_games(games: &Bound<'_, PyAny>, first: usize) -> PyResult<Vec<RecordedGame>> {
    let py = games.py();
    let mut parsed = Vec::new();
    for (i, g) in games.try_iter()?.enumerate() {
        match g.and_then(|g| recorded_game(&g)) {
            Ok(game) => parsed.push(game),
            Err(e) if e.is_instance_of::<PyException>(py) => {
                let wrapped = value_error(format!("game {}: {e}", first + i));
                wrapped.set_cause(py, Some(e));
                return Err(wrapped);
            }
            Err(e) => return Err(e),
        }
    }
    Ok(parsed)
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

/// The public features (NET_PUBLIC_LEN f32) and the hidden block (NET_HIDDEN_LEN f32) for
/// whoever must act in `game`, little-endian.
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
/// `visits`, `full_search` and an optional `target`, `winner` and `vp`), each replayed once and
/// checked. Build it a record file at a time with `extend`, so the window is never all Python
/// dicts at once.
#[pyclass(name = "TrainingSet", module = "alphasettler._engine")]
struct PyTrainingSet {
    inner: TrainingSet,
}

#[pymethods]
impl PyTrainingSet {
    #[new]
    #[pyo3(signature = (games=None))]
    fn new(py: Python<'_>, games: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        let mut set = PyTrainingSet {
            inner: TrainingSet::default(),
        };
        if let Some(games) = games {
            set.extend(py, games)?;
        }
        Ok(set)
    }

    /// Add every game of the iterable `games` (a list, or `records.read(path)`). A bad record is a
    /// ValueError naming its index in the whole set, and adds nothing.
    fn extend(&mut self, py: Python<'_>, games: &Bound<'_, PyAny>) -> PyResult<()> {
        let parsed = recorded_games(games, self.inner.games())?;
        let inner = &mut self.inner;
        py.detach(|| inner.extend(parsed)).map_err(value_error)
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

    /// `n` (1..=MAX_SAMPLE) positions drawn uniformly over every decision from `seed`, replayed on
    /// `threads` threads (the result does not depend on `threads`). Releases the GIL.
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
        let inner = &self.inner;
        let b = py
            .detach(|| inner.sample(n, seed, threads))
            .map_err(value_error)?;
        batch_dict(py, &b)
    }

    /// Decisions that are policy targets (with a target, or full searches with at least one visit).
    #[getter]
    fn policy_decisions(&self) -> usize {
        self.inner.policy_decisions()
    }

    /// As `sample`, drawn uniformly over the policy targets alone.
    fn sample_full<'py>(
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
        let inner = &self.inner;
        let b = py
            .detach(|| inner.sample_full(n, seed, threads))
            .map_err(value_error)?;
        batch_dict(py, &b)
    }

    /// Decision `decision` of game `game` as a one-position batch.
    fn position<'py>(
        &self,
        py: Python<'py>,
        game: &Bound<'py, PyAny>,
        decision: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let game: usize = int_arg(game, "game")?;
        let decision: usize = int_arg(decision, "decision")?;
        if game >= self.inner.games() || decision >= self.inner.game_decisions(game) {
            return Err(value_error(format!(
                "no decision {decision} in game {game}"
            )));
        }
        let mut b = Batch::default();
        self.inner.position(game, decision, &mut b);
        batch_dict(py, &b)
    }
}

/// The Rust forward pass of the weights file at `path` on `n` positions (public: n × NET_PUBLIC_LEN
/// f32, hidden: n × NET_HIDDEN_LEN f32, little-endian). Returns raw policy logits (n × 665), value
/// logits (n × 4) and aux outputs (n × 4), little-endian f32.
#[pyfunction]
fn net_forward<'py>(
    py: Python<'py>,
    path: PathBuf,
    public: &Bound<'py, PyBytes>,
    hidden: &Bound<'py, PyBytes>,
) -> PyResult<Bound<'py, PyDict>> {
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
    let w = Weights::load(&path).map_err(value_error)?;
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

/// Network v2's static adjacency, for the PyTorch twin: a dict of `node_tiles`, `node_edges`,
/// `node_neighbours`, `edge_nodes` and `tile_nodes`, each a list (per entity) of index lists in
/// the order the Rust forward pass sums them. `edge_nodes` is summed; the others are averaged.
#[pyfunction]
fn net_adjacency(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
    let g = graph();
    let d = PyDict::new(py);
    for (name, links) in [
        ("node_tiles", &g.node_tiles),
        ("node_edges", &g.node_edges),
        ("node_neighbours", &g.node_neighbours),
        ("edge_nodes", &g.edge_nodes),
        ("tile_nodes", &g.tile_nodes),
    ] {
        d.set_item(name, links.lists())?;
    }
    Ok(d)
}

/// Random network v2 weights (width `NET_DEFAULT_WIDTH` and `NET_DEFAULT_ROUNDS` rounds unless
/// given) saved to `path`, for tests and measurements.
#[pyfunction]
#[pyo3(signature = (path, seed, width=None, rounds=None))]
fn write_random_net(
    path: PathBuf,
    seed: &Bound<'_, PyAny>,
    width: Option<&Bound<'_, PyAny>>,
    rounds: Option<&Bound<'_, PyAny>>,
) -> PyResult<()> {
    let seed: u64 = int_arg(seed, "seed")?;
    let dim = |v: Option<&Bound<'_, PyAny>>, default: usize, what: &str| -> PyResult<usize> {
        match v {
            Some(v) => int_arg(v, what),
            None => Ok(default),
        }
    };
    let arch = Arch {
        width: dim(width, Arch::DEFAULT.width, "width")?,
        rounds: dim(rounds, Arch::DEFAULT.rounds, "rounds")?,
    };
    arch.check().map_err(value_error)?;
    Weights::random(arch, seed).save(&path).map_err(value_error)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTrainingSet>()?;
    m.add_function(wrap_pyfunction!(encode, m)?)?;
    m.add_function(wrap_pyfunction!(net_forward, m)?)?;
    m.add_function(wrap_pyfunction!(resolve_net, m)?)?;
    m.add_function(wrap_pyfunction!(write_random_net, m)?)?;
    m.add_function(wrap_pyfunction!(net_adjacency, m)?)?;
    m.add("NET_PUBLIC_LEN", PUBLIC_LEN)?;
    m.add("NET_HIDDEN_LEN", HIDDEN_LEN)?;
    m.add("NET_ENCODING_HASH", ENCODING_HASH)?;
    m.add("NET_FORMAT_VERSION", FORMAT_VERSION)?;
    m.add("NET_MAX_SAMPLE", settler_net::data::MAX_SAMPLE)?;
    m.add("NET_MAX_WIDTH", MAX_WIDTH)?;
    m.add("NET_MAX_ROUNDS", MAX_ROUNDS)?;
    m.add("NET_DEFAULT_WIDTH", Arch::DEFAULT.width)?;
    m.add("NET_DEFAULT_ROUNDS", Arch::DEFAULT.rounds)?;
    m.add("NET_NODE_WIDTH", NODE_WIDTH)?;
    m.add("NET_EDGE_WIDTH", EDGE_WIDTH)?;
    m.add("NET_TILE_WIDTH", TILE_WIDTH)?;
    m.add("NET_GLOBAL_WIDTH", GLOBAL_WIDTH)?;
    m.add("NET_V2_ENCODING_HASH", V2_ENCODING_HASH)?;
    m.add("NET_V2_GLOBAL_WIDTH", V2_GLOBAL_WIDTH)?;
    m.add("NET_SETTLEMENT_BASE", SETTLEMENT_BASE)?;
    Ok(())
}
