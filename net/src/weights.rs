//! The weights file (spec Section 2): an 80-byte header, then the parameters. The file layout:
//! bytes 0..8 magic; 8..12 format version; 12..20 `ENCODING_HASH`; 20..24 layer count; then for
//! layer i, rows at `24 + 8i` and cols at `28 + 8i`; from byte 80, each layer's `rows × cols`
//! weights (row-major, rows = outputs) then its `rows` biases, all f32 little-endian. A file may
//! declare any trunk width and value width from 1 to `MAX_WIDTH` (tests use small ones); every
//! other dimension is fixed by the encoder and the action space.

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
        let widths_ok =
            (1..=MAX_WIDTH).contains(&arch.trunk) && (1..=MAX_WIDTH).contains(&arch.value);
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
