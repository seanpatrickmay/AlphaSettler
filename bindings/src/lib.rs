//! Python bindings: the extension module `alphasettler._engine`. Conversions only; rules live
//! in `settler-engine`, bots and the arena in `settler-bots`.

mod convert;
mod opening;
mod train;

use convert::*;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use settler_engine::{ACTION_SPACE_SIZE, NUM_PLAYERS};
use std::path::PathBuf;

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

/// A native bot. `act` runs with the GIL released, so the mutex is what serialises two threads
/// acting with the same bot.
#[pyclass(name = "Bot", module = "alphasettler._engine")]
struct PyBot {
    name: &'static str,
    inner: std::sync::Mutex<Box<dyn settler_bots::Bot>>,
}

#[pymethods]
impl PyBot {
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

    #[getter]
    fn name(&self) -> &'static str {
        self.name
    }

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
        // `obs` and `legal` are owned: let go of the game's borrow before the GIL is released, so
        // another Python thread can `apply` to this game while the bot searches.
        drop(game);
        let a = py
            .detach(|| {
                let mut bot = self.inner.lock().map_err(|_| ())?;
                Ok::<_, ()>(bot.act(&obs, &legal))
            })
            .map_err(|_| value_error("bot state is poisoned"))?;
        Ok(a.encode() as u32)
    }

    /// Show the bot new events as `viewer` saw them (dicts shaped like `Game.log` entries).
    fn observe(&self, viewer: &Bound<'_, PyAny>, events: &Bound<'_, PyList>) -> PyResult<()> {
        let viewer = viewer_arg(viewer)?;
        let events = events
            .iter()
            .map(|e| parse_event(&e))
            .collect::<PyResult<Vec<_>>>()?;
        let mut bot = self
            .inner
            .lock()
            .map_err(|_| value_error("bot state is poisoned"))?;
        bot.observe(viewer, &events);
        Ok(())
    }

    /// The bot's named counters (e.g. `belief_resets` for ismcts); empty for bots without any.
    fn diagnostics<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let bot = self
            .inner
            .lock()
            .map_err(|_| value_error("bot state is poisoned"))?;
        let d = PyDict::new(py);
        for (k, v) in bot.diagnostics() {
            d.set_item(k, v)?;
        }
        Ok(d)
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

/// Every `GameConfig` field as a dict: the defaults with `config`'s overrides applied (and
/// validated), so a stored config replays under its own rules even if the defaults change.
#[pyfunction]
#[pyo3(signature = (config=None))]
fn resolve_config<'py>(
    py: Python<'py>,
    config: Option<&Bound<'py, PyDict>>,
) -> PyResult<Bound<'py, PyDict>> {
    config_dict(py, &parse_config(config)?)
}

/// Native arena: `seeds` seeds starting at `seed_start`, each played four times with the
/// candidate rotated through every seat, at most `MAX_SEEDS` seeds per call. Releases the GIL
/// while games run.
#[pyfunction]
#[pyo3(signature = (candidate, baseline, seed_start, seeds, threads, config=None, nets_dir=None))]
#[allow(clippy::too_many_arguments)]
fn run_match<'py>(
    py: Python<'py>,
    candidate: &str,
    baseline: &str,
    seed_start: &Bound<'py, PyAny>,
    seeds: &Bound<'py, PyAny>,
    threads: &Bound<'py, PyAny>,
    config: Option<&Bound<'py, PyDict>>,
    nets_dir: Option<PathBuf>,
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
    let out = PyList::empty(py);
    for r in records {
        let d = PyDict::new(py);
        d.set_item("seed", r.seed)?;
        d.set_item("candidate_seat", r.candidate_seat)?;
        d.set_item("winner", r.winner)?;
        d.set_item("vp", counts(&r.vp))?;
        d.set_item("turns", r.turns)?;
        d.set_item("actions", r.actions)?;
        d.set_item("belief_resets", r.belief_resets)?;
        d.set_item("bank_trades", r.bank_trades.to_vec())?;
        d.set_item("searches", r.searches.to_vec())?;
        d.set_item("compound_decisions", r.compound_decisions.to_vec())?;
        out.append(d)?;
    }
    Ok(out)
}

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
        g.actions
            .iter()
            .map(|a| a.encode() as u32)
            .collect::<Vec<_>>(),
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
            dec.legal
                .iter()
                .map(|a| a.encode() as u32)
                .collect::<Vec<_>>(),
        )?;
        e.set_item("visits", dec.visits.clone())?;
        e.set_item("full_search", dec.full_search)?;
        if let Some(t) = &dec.target {
            e.set_item("target", t.clone())?;
        }
        if let Some(q) = &dec.q {
            // Six places: an f32's full double repr would make the record files ~3x bigger.
            let q: Vec<f64> = q
                .iter()
                .map(|&x| ((x as f64) * 1e6).round() / 1e6)
                .collect();
            e.set_item("q", q)?;
        }
        if observations {
            while applied < dec.index as usize {
                s.apply(g.actions[applied]);
                applied += 1;
            }
            e.set_item(
                "observation",
                observation_dict(py, &s.observation(dec.actor))?,
            )?;
        }
        decisions.append(e)?;
    }
    d.set_item("decisions", decisions)?;
    // 3d-3: only a league game has seats to name, so every other record is unchanged.
    if let Some(seats) = &g.league {
        d.set_item("league", seats.to_vec())?;
    }
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
/// opening sampling. With `gumbel` (a simulation count), every move is instead a Gumbel root
/// search of that many simulations, recorded with its improved policy as `target` (spec P2
/// Section 5); the three PUCT arguments are then not used. With `opening` (an `ASOPEN` file),
/// the opening model picks every setup settlement, epsilon-greedy over its best spots, with no
/// search and no recorded decision (spec 3d-1b Section 2); refused with `gumbel`. With
/// `league_nets` and `league_styles` (bot names, both or neither, loaded from `nets_dir`), about
/// half the games seat two of them, a past net and a style, whose moves are not recorded; a
/// league game's dict names them under `league` (spec 3d-3 Section 2); refused with `gumbel`.
/// With `compound`, every learner seat searches compound build moves (spec 3d-4; league seats play
/// their pool names as given); refused with `gumbel`. With `net`, every learner seat is the
/// network `<nets_dir>/<net>.bin` (loaded once); otherwise the heuristic.
/// Decisions carry no observations. Releases the GIL while games run.
#[pyfunction]
#[pyo3(signature = (seed_start, games, threads, full_simulations, fast_simulations, full_probability, config=None, net=None, nets_dir=None, gumbel=None, opening=None, league_nets=None, league_styles=None, compound=false))]
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
    league_nets: Option<Vec<String>>,
    league_styles: Option<Vec<String>>,
    compound: bool,
) -> PyResult<Bound<'py, PyList>> {
    use settler_bots::heuristic::{HeuristicEvaluator, DEFAULT_WEIGHTS};
    use settler_bots::netbot::{load_net, DEFAULT_NETS_DIR, NET_BATCH};
    use settler_bots::opening_model::OpeningModel;
    use settler_bots::selfplay::{run_with_league, Exploration, League};
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
    let ex = Exploration { compound, ..ex };
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
    let model = match &opening {
        Some(path) => Some(OpeningModel::load(path).map_err(value_error)?),
        None => None,
    };
    let league = match (&league_nets, &league_styles) {
        (None, None) => None,
        (Some(nets), Some(styles)) => {
            let nets: Vec<&str> = nets.iter().map(String::as_str).collect();
            let styles: Vec<&str> = styles.iter().map(String::as_str).collect();
            Some(League::new(&nets, &styles, &dir).map_err(value_error)?)
        }
        _ => {
            return Err(value_error(
                "league_nets and league_styles are given together",
            ))
        }
    };
    let played = py
        .detach(|| match &weights {
            None => run_with_league(
                seed_start..end,
                config,
                &ex,
                settler_search::SearchConfig::default().batch,
                &|| HeuristicEvaluator::new(DEFAULT_WEIGHTS, 0),
                threads,
                model.as_ref(),
                league.as_ref(),
            ),
            Some(w) => run_with_league(
                seed_start..end,
                config,
                &ex,
                NET_BATCH,
                &|| NetEvaluator::new(w.clone()),
                threads,
                model.as_ref(),
                league.as_ref(),
            ),
        })
        .map_err(value_error)?;
    let out = PyList::empty(py);
    for g in &played {
        out.append(game_dict(py, g, &config, false)?)?;
    }
    Ok(out)
}

/// Fit the heuristic's weights on recorded games (dicts with `seed`, `config`, `actions`,
/// `winner` and `decisions[].index`). Returns the weights, the sample count and the mean
/// log-likelihood under the default and the fitted weights. `iterations` defaults to 25; `l2`
/// must be finite and non-negative. Records with no samples (none, or only draws) are an error.
#[pyfunction]
#[pyo3(signature = (games, iterations=None, l2=0.001))]
fn fit_heuristic<'py>(
    py: Python<'py>,
    games: &Bound<'py, PyList>,
    iterations: Option<&Bound<'py, PyAny>>,
    l2: f64,
) -> PyResult<Bound<'py, PyDict>> {
    use settler_bots::heuristic::{
        fit, log_likelihood, samples_from_games, ReplayGame, DEFAULT_WEIGHTS,
    };
    let iterations: u32 = match iterations {
        Some(n) => int_arg(n, "iterations")?,
        None => 25,
    };
    if !(l2.is_finite() && l2 >= 0.0) {
        return Err(value_error(format!(
            "l2 must be finite and non-negative, got {l2}"
        )));
    }
    let mut replay = Vec::new();
    for g in games.iter() {
        let g = g.cast::<PyDict>()?;
        let get = |k: &str| {
            g.get_item(k)?
                .ok_or_else(|| value_error(format!("game record is missing {k:?}")))
        };
        let config_obj = get("config")?;
        let config = parse_config(Some(config_obj.cast::<PyDict>()?))?;
        let actions = get("actions")?
            .extract::<Vec<Bound<'_, PyAny>>>()?
            .iter()
            .map(action_from_id)
            .collect::<PyResult<Vec<_>>>()?;
        let at = get("decisions")?
            .extract::<Vec<Bound<'_, PyDict>>>()?
            .iter()
            .map(|d| {
                d.get_item("index")?
                    .ok_or_else(|| value_error("decision is missing \"index\""))?
                    .extract::<u32>()
            })
            .collect::<PyResult<Vec<_>>>()?;
        let winner = get("winner")?.extract::<Option<u8>>()?;
        replay.push(ReplayGame {
            seed: int_arg(&get("seed")?, "seed")?,
            config,
            actions,
            winner,
            at,
        });
    }
    let (samples, weights) = py
        .detach(|| {
            let samples = samples_from_games(&replay)?;
            if samples.is_empty() {
                return Err(
                    "no samples to fit: the records hold no decided game with a recorded decision"
                        .to_owned(),
                );
            }
            let weights = fit(&samples, DEFAULT_WEIGHTS, iterations, l2);
            Ok::<_, String>((samples, weights))
        })
        .map_err(value_error)?;
    let d = PyDict::new(py);
    d.set_item("weights", weights.to_vec())?;
    d.set_item("samples", samples.len())?;
    d.set_item(
        "log_likelihood_before",
        log_likelihood(&samples, &DEFAULT_WEIGHTS),
    )?;
    d.set_item("log_likelihood_after", log_likelihood(&samples, &weights))?;
    Ok(d)
}

#[pymodule]
fn _engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyGame>()?;
    m.add_class::<PyBot>()?;
    m.add_function(wrap_pyfunction!(action_space_size, m)?)?;
    m.add_function(wrap_pyfunction!(describe_action, m)?)?;
    m.add_function(wrap_pyfunction!(bot_names, m)?)?;
    m.add_function(wrap_pyfunction!(resolve_config, m)?)?;
    m.add_function(wrap_pyfunction!(run_match, m)?)?;
    m.add_function(wrap_pyfunction!(selfplay, m)?)?;
    m.add_function(wrap_pyfunction!(selfplay_explore, m)?)?;
    m.add(
        "SEARCH_BATCH",
        settler_search::SearchConfig::default().batch,
    )?;
    m.add("NET_BATCH", settler_bots::netbot::NET_BATCH)?;
    m.add(
        "SELFPLAY_FULL_SIMULATIONS",
        settler_bots::selfplay::FULL_SIMULATIONS,
    )?;
    m.add(
        "SELFPLAY_FAST_SIMULATIONS",
        settler_bots::selfplay::FAST_SIMULATIONS,
    )?;
    m.add(
        "SELFPLAY_FULL_PROBABILITY",
        settler_bots::selfplay::FULL_PROBABILITY,
    )?;
    m.add("NOISE_WEIGHT", settler_bots::selfplay::NOISE.weight as f64)?;
    m.add(
        "NOISE_ALPHA",
        settler_bots::selfplay::NOISE.alpha_total as f64,
    )?;
    m.add("SAMPLE_TURNS", settler_bots::selfplay::SAMPLE_TURNS)?;
    m.add("OPENING_EPSILON", settler_bots::selfplay::OPENING_EPSILON)?;
    m.add("OPENING_TOP", settler_bots::selfplay::OPENING_TOP)?;
    m.add(
        "LEAGUE_PROBABILITY",
        settler_bots::selfplay::LEAGUE_PROBABILITY,
    )?;
    m.add("LEAGUE_SEATS", settler_bots::selfplay::LEAGUE_SEATS)?;
    m.add(
        "LEAGUE_STYLES",
        settler_bots::heuristic::STYLES
            .iter()
            .map(|(name, _)| *name)
            .collect::<Vec<_>>(),
    )?;
    m.add_function(wrap_pyfunction!(fit_heuristic, m)?)?;
    m.add("MAX_SEEDS", settler_bots::arena::MAX_SEEDS)?;
    m.add(
        "SELFPLAY_MIN_SIMULATIONS",
        settler_bots::selfplay::MIN_SIMULATIONS,
    )?;
    m.add("MAX_SIMULATIONS", settler_bots::ismcts::MAX_SIMULATIONS)?;
    m.add("MAX_ROLLOUT", settler_bots::ismcts::MAX_ROLLOUT)?;
    m.add("DEFAULT_NETS_DIR", settler_bots::netbot::DEFAULT_NETS_DIR)?;
    train::register(m)?;
    opening::register(m)?;
    Ok(())
}
