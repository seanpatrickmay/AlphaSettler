//! The opening model (spec 3d-1 Sections 4.3 and 4.5): an MLP over the 78 opening features that
//! gives each seat's win probability, its file `<nets_dir>/opening.bin`, and the `+o` choice of a
//! setup settlement.
//!
//! The file, little-endian:
//!
//! | bytes | field |
//! |---|---|
//! | 0..8 | magic `ASOPEN\0\0` |
//! | 8..12 | format version, u32 = 1 |
//! | 12..20 | `OPENING_HASH`, u64 |
//! | 20..24 | layer count L, u32 (1..=`MAX_OPENING_LAYERS`) |
//! | 24..24 + 8L | per layer: rows, then cols (u32 each); rows are outputs |
//! | then | per layer, in order: rows × cols weights (row-major), then rows biases (f32) |
//!
//! Layer 0 takes the 78 features, each later layer the previous one's outputs, and the last gives
//! 4 seat-relative win logits (entry 0 is the seat the features were computed for), softmaxed.
//! Every layer but the last ends in a ReLU. `alphasettler opening-train` writes 78 → 64 → 64 → 4.

use crate::opening::{opening_features, OPENING_FEATURES, OPENING_HASH};
use settler_engine::{Action, Observation, NUM_PLAYERS};
use settler_net::forward::{dot, softmax};
use std::path::Path;

pub const OPENING_MAGIC: [u8; 8] = *b"ASOPEN\0\0";
pub const OPENING_FORMAT_VERSION: u32 = 1;
/// The model's file name inside a nets directory.
pub const OPENING_FILE: &str = "opening.bin";
/// Bounds checked before anything is sized from a file: a guard against a corrupt header.
pub const MAX_OPENING_LAYERS: usize = 8;
pub const MAX_OPENING_WIDTH: usize = 4096;
/// Magic, version, hash and layer count: the bytes before the layer shapes.
pub const OPENING_HEADER_LEN: usize = 8 + 4 + 8 + 4;

#[derive(Clone, Debug, PartialEq)]
pub struct OpeningLayer {
    pub rows: usize,
    pub cols: usize,
    /// `rows × cols`, row-major.
    pub w: Vec<f32>,
    pub b: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct OpeningModel {
    pub layers: Vec<OpeningLayer>,
}

impl OpeningModel {
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&OPENING_MAGIC);
        out.extend_from_slice(&OPENING_FORMAT_VERSION.to_le_bytes());
        out.extend_from_slice(&OPENING_HASH.to_le_bytes());
        out.extend_from_slice(&(self.layers.len() as u32).to_le_bytes());
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

    /// Every check runs before anything is sized from the file: the layer count and every
    /// dimension are bounded first, then the shapes must chain from 78 inputs to 4 outputs.
    pub fn from_bytes(bytes: &[u8]) -> Result<OpeningModel, String> {
        if bytes.len() < OPENING_HEADER_LEN {
            return Err(format!(
                "truncated: {} bytes is shorter than the {OPENING_HEADER_LEN}-byte header",
                bytes.len()
            ));
        }
        if bytes[..8] != OPENING_MAGIC {
            return Err("not an AlphaSettler opening model (wrong magic)".into());
        }
        let u32_at = |i: usize| u32::from_le_bytes(bytes[i..i + 4].try_into().unwrap());
        let version = u32_at(8);
        if version != OPENING_FORMAT_VERSION {
            return Err(format!(
                "format version {version}, expected {OPENING_FORMAT_VERSION}"
            ));
        }
        let hash = u64::from_le_bytes(bytes[12..20].try_into().unwrap());
        if hash != OPENING_HASH {
            return Err(format!(
                "opening feature hash {hash:#018x} does not match this build's \
                 {OPENING_HASH:#018x}: the model was trained on different features"
            ));
        }
        let count = u32_at(20) as usize;
        if !(1..=MAX_OPENING_LAYERS).contains(&count) {
            return Err(format!("{count} layers, expected 1..={MAX_OPENING_LAYERS}"));
        }
        let header = OPENING_HEADER_LEN + 8 * count;
        if bytes.len() < header {
            return Err(format!(
                "truncated: {} bytes is shorter than the {header}-byte header",
                bytes.len()
            ));
        }
        let shapes: Vec<(usize, usize)> = (0..count)
            .map(|i| {
                let at = OPENING_HEADER_LEN + 8 * i;
                (u32_at(at) as usize, u32_at(at + 4) as usize)
            })
            .collect();
        let mut inputs = OPENING_FEATURES;
        for (i, &(rows, cols)) in shapes.iter().enumerate() {
            let dim_ok = |d: usize| (1..=MAX_OPENING_WIDTH).contains(&d);
            if !dim_ok(rows) || !dim_ok(cols) {
                return Err(format!(
                    "layer {i} is {rows} x {cols}: each dimension must be 1..={MAX_OPENING_WIDTH}"
                ));
            }
            if cols != inputs {
                return Err(if i == 0 {
                    format!("layer 0 takes {cols} inputs, expected the {OPENING_FEATURES} opening features")
                } else {
                    format!(
                        "layer {i} takes {cols} inputs, but layer {} gives {inputs}",
                        i - 1
                    )
                });
            }
            inputs = rows;
        }
        if inputs != NUM_PLAYERS {
            return Err(format!(
                "the last layer gives {inputs} outputs, expected {NUM_PLAYERS} (one per seat)"
            ));
        }
        let floats: usize = shapes.iter().map(|&(r, c)| r * c + r).sum();
        let want = header + 4 * floats;
        if bytes.len() < want {
            return Err(format!("truncated: {} bytes, expected {want}", bytes.len()));
        }
        if bytes.len() > want {
            return Err(format!(
                "{} trailing bytes after the last layer",
                bytes.len() - want
            ));
        }
        let mut at = header;
        let mut take = |n: usize| -> Vec<f32> {
            let v = bytes[at..at + 4 * n]
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect();
            at += 4 * n;
            v
        };
        let mut layers = Vec::with_capacity(count);
        for &(rows, cols) in &shapes {
            let w = take(rows * cols);
            let b = take(rows);
            layers.push(OpeningLayer { rows, cols, w, b });
        }
        if layers
            .iter()
            .any(|l| l.w.iter().chain(&l.b).any(|x| !x.is_finite()))
        {
            return Err("a weight is NaN or infinite".into());
        }
        Ok(OpeningModel { layers })
    }

    pub fn load(path: &Path) -> Result<OpeningModel, String> {
        let bytes = std::fs::read(path)
            .map_err(|e| format!("cannot read opening model {}: {e}", path.display()))?;
        OpeningModel::from_bytes(&bytes)
            .map_err(|e| format!("opening model {}: {e}", path.display()))
    }

    pub fn save(&self, path: &Path) -> Result<(), String> {
        std::fs::write(path, self.to_bytes())
            .map_err(|e| format!("cannot write {}: {e}", path.display()))
    }

    /// Win probabilities, seat-relative to the seat `features` were computed for (entry 0).
    /// The shapes are the loader's: a model built by hand must chain from 78 inputs to 4 outputs.
    pub fn probabilities(&self, features: &[f32; OPENING_FEATURES]) -> [f32; NUM_PLAYERS] {
        let mut x = features.to_vec();
        let last = self.layers.len() - 1;
        for (i, l) in self.layers.iter().enumerate() {
            let mut y: Vec<f32> = (0..l.rows)
                .map(|r| dot(&l.w[r * l.cols..(r + 1) * l.cols], &x) + l.b[r])
                .collect();
            if i < last {
                y.iter_mut().for_each(|v| *v = v.max(0.0));
            }
            x = y;
        }
        let mut p = [x[0], x[1], x[2], x[3]];
        softmax(&mut p);
        p
    }
}

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
