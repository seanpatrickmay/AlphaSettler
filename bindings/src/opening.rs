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
/// (`settler_net::evaluator::position_value`): the evaluation a search leaf of the same world
/// gets, equal to it within float rounding whatever the batch shape. `net` is a bare NAME, as in
/// `net:NAME`; a bad name or file raises what `Bot(...)` raises for it.
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
