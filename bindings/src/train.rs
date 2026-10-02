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
            return Err(value_error(format!(
                "{what} must be 1..={MAX_WIDTH}, got {w}"
            )));
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
