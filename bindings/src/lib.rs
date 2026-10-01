//! Python bindings: the extension module `alphasettler._engine`. Conversions only; rules live
//! in `settler-engine`, bots and the arena in `settler-bots`.

mod convert;

use convert::*;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use settler_engine::{ACTION_SPACE_SIZE, NUM_PLAYERS};

#[pyclass(name = "Game", module = "alphasettler._engine")]
struct PyGame {
    inner: settler_engine::Game,
}

#[pymethods]
impl PyGame {
    #[new]
    #[pyo3(signature = (seed, config=None))]
    fn new(seed: &Bound<'_, PyAny>, config: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        Ok(PyGame {
            inner: settler_engine::Game::new(int_arg(seed, "seed")?, parse_config(config)?),
        })
    }

    fn legal_actions(&self) -> Vec<u32> {
        self.inner
            .legal_actions()
            .iter()
            .map(|a| a.encode() as u32)
            .collect()
    }

    fn apply(&mut self, action: &Bound<'_, PyAny>) -> PyResult<()> {
        let a = action_from_id(action)?;
        self.inner
            .apply(a)
            .map_err(|e| value_error(format!("{e:?}")))
    }

    fn apply_forced(
        &mut self,
        action: &Bound<'_, PyAny>,
        chance: &Bound<'_, PyDict>,
    ) -> PyResult<()> {
        let a = action_from_id(action)?;
        let c = parse_chance(chance)?;
        self.inner
            .apply_forced(a, Some(c))
            .map_err(|e| value_error(format!("{e:?}")))
    }

    fn observation<'py>(
        &self,
        py: Python<'py>,
        viewer: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyDict>> {
        observation_dict(py, &self.inner.observation(viewer_arg(viewer)?))
    }

    fn log<'py>(
        &self,
        py: Python<'py>,
        viewer: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyList>> {
        let viewer = viewer_arg(viewer)?;
        let out = PyList::empty(py);
        for e in self.inner.log_for(viewer) {
            out.append(event_dict(py, &e)?)?;
        }
        Ok(out)
    }

    /// Final victory points (including hidden VP cards); only available once the game is over.
    fn final_vp(&self) -> PyResult<Vec<u32>> {
        let s = self.inner.state();
        if !s.is_over() {
            return Err(value_error(
                "final_vp is only available once the game is over",
            ));
        }
        Ok((0..NUM_PLAYERS).map(|p| s.total_vp(p) as u32).collect())
    }

    /// A game continuing from `snapshot` (see `snapshot()` for the dict schema). Randomness the
    /// snapshot does not fix (dice, steals) comes from `seed`.
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
        let inner =
            settler_engine::Game::from_snapshot(seed, config, &snap).map_err(value_error)?;
        Ok(PyGame { inner })
    }

    /// The full position as a plain dict, accepted back by `Game.from_snapshot`. A pending
    /// domestic trade is not captured: during `trade_response` / `trade_confirm` the dict
    /// carries only the phase name, and `from_snapshot` refuses it.
    fn snapshot<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        snapshot_dict(py, &self.inner.state().snapshot())
    }

    /// An independent copy of this game, event log included.
    fn copy(&self) -> Self {
        PyGame {
            inner: self.inner.clone(),
        }
    }

    fn __copy__(&self) -> Self {
        self.copy()
    }

    fn __deepcopy__(&self, _memo: &Bound<'_, PyAny>) -> Self {
        self.copy()
    }

    #[getter]
    fn seed(&self) -> u64 {
        self.inner.state().seed
    }

    #[getter]
    fn is_over(&self) -> bool {
        self.inner.state().is_over()
    }

    #[getter]
    fn winner(&self) -> Option<u8> {
        self.inner.state().winner()
    }

    #[getter]
    fn current_actor(&self) -> u8 {
        let s = self.inner.state();
        if s.is_over() {
            s.current
        } else {
            s.current_actor()
        }
    }

    #[getter]
    fn turn(&self) -> u32 {
        self.inner.state().turn
    }

    #[getter]
    fn phase(&self) -> &'static str {
        phase_name(self.inner.state().phase)
    }
}

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
            value_error(format!(
                "unknown bot {name:?}; known bots: {:?}",
                settler_bots::BOT_NAMES
            ))
        })?;
        Ok(PyBot {
            name: bot.name(),
            inner: std::sync::Mutex::new(bot),
        })
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
        if legal.is_empty() {
            // Bots require a non-empty `legal`; refuse here rather than panic inside one.
            return Err(value_error("no legal actions in this position"));
        }
        let obs = s.observation(s.current_actor());
        let mut bot = self
            .inner
            .lock()
            .map_err(|_| value_error("bot state is poisoned"))?;
        Ok(bot.act(&obs, &legal).encode() as u32)
    }
}

#[pyfunction]
fn action_space_size() -> usize {
    ACTION_SPACE_SIZE
}

#[pyfunction]
fn describe_action(action: &Bound<'_, PyAny>) -> PyResult<String> {
    Ok(format!("{:?}", action_from_id(action)?))
}

#[pyfunction]
fn bot_names() -> Vec<&'static str> {
    settler_bots::BOT_NAMES.to_vec()
}

/// Native arena: `seeds` seeds starting at `seed_start`, each played four times with the
/// candidate rotated through every seat, at most `MAX_SEEDS` seeds per call. Releases the GIL
/// while games run.
#[pyfunction]
#[pyo3(signature = (candidate, baseline, seed_start, seeds, threads, config=None))]
fn run_match<'py>(
    py: Python<'py>,
    candidate: &str,
    baseline: &str,
    seed_start: &Bound<'py, PyAny>,
    seeds: &Bound<'py, PyAny>,
    threads: &Bound<'py, PyAny>,
    config: Option<&Bound<'py, PyDict>>,
) -> PyResult<Bound<'py, PyList>> {
    let seed_start: u64 = int_arg(seed_start, "seed_start")?;
    let seeds: u64 = int_arg(seeds, "seeds")?;
    let threads: usize = int_arg(threads, "threads")?;
    let seed_end = seed_start.checked_add(seeds).ok_or_else(|| {
        value_error(format!(
            "seed_start {seed_start} + seeds {seeds} exceeds the u64 seed range"
        ))
    })?;
    let config = parse_config(config)?;
    let (candidate, baseline) = (candidate.to_owned(), baseline.to_owned());
    let records = py
        .detach(|| {
            settler_bots::arena::run_match(
                &candidate,
                &baseline,
                seed_start..seed_end,
                config,
                threads,
            )
        })
        .map_err(value_error)?;
    let out = PyList::empty(py);
    for r in records {
        let d = PyDict::new(py);
        d.set_item("seed", r.seed)?;
        d.set_item("candidate_seat", r.candidate_seat)?;
        d.set_item("winner", r.winner)?;
        d.set_item("vp", counts(&r.vp))?;
        d.set_item("turns", r.turns)?;
        d.set_item("actions", r.actions)?;
        out.append(d)?;
    }
    Ok(out)
}

#[pymodule]
fn _engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyGame>()?;
    m.add_class::<PyBot>()?;
    m.add_function(wrap_pyfunction!(action_space_size, m)?)?;
    m.add_function(wrap_pyfunction!(describe_action, m)?)?;
    m.add_function(wrap_pyfunction!(bot_names, m)?)?;
    m.add_function(wrap_pyfunction!(run_match, m)?)?;
    m.add("MAX_SEEDS", settler_bots::arena::MAX_SEEDS)?;
    Ok(())
}
