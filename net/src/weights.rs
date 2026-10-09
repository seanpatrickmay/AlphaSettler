//! The weights file, format version 2: network v2, the entity-structured graph net (spec
//! Section 2). Little-endian throughout. A file declares its width `d` (1..=`MAX_WIDTH`) and its
//! round count `R` (1..=`MAX_ROUNDS`); every other dimension is fixed by the encoder and the
//! action space. A version 1 file (the flat MLP) is refused.
//!
//! | bytes | field |
//! |---|---|
//! | 0..8 | magic `ASNET\0\0\0` |
//! | 8..12 | format version, u32 = 2 |
//! | 12..20 | `ENCODING_HASH`, u64 |
//! | 20..24 | width `d`, u32 |
//! | 24..28 | rounds `R`, u32 |
//! | 28..32 | layer count `L`, u32 = 12 + 4R |
//! | 32..32 + 8L | per layer: rows, then cols (u32 each); rows are outputs |
//! | then | per layer, in order: rows × cols weights (row-major), then rows biases (f32) |
//!
//! Layers, in file order (the PyTorch twin's `Net.layers()` returns its modules in this order):
//!
//! | index | layer | rows × cols | input, in column order |
//! |---|---|---|---|
//! | 0 | node encoder | d × 24 | a node's 24 features |
//! | 1 | edge encoder | d × 8 | an edge's 8 features |
//! | 2 | tile encoder | d × 8 | a tile's 8 features |
//! | 3 | global encoder | d × 139 | the 139 global scalars (`graph::GLOBAL_BASE..PUBLIC_LEN`) |
//! | 4 + 4r | round r, tile | d × 3d | [tile, mean of its 6 nodes, global] |
//! | 5 + 4r | round r, node | d × 5d | [node, mean of its tiles, mean of its edges, mean of its neighbouring nodes, global] |
//! | 6 + 4r | round r, edge | d × 3d | [edge, sum of its 2 nodes, global] |
//! | 7 + 4r | round r, global | d × 4d | [global, mean of the nodes, mean of the edges, mean of the tiles] |
//! | P = 4 + 4R | node policy | 2 × 2d | [node, global] -> BuildSettlement, BuildCity logits |
//! | P + 1 | edge policy | 1 × 2d | [edge, global] -> BuildRoad logit |
//! | P + 2 | tile policy | 1 × 2d | [tile, global] -> MoveRobber logit |
//! | P + 3 | other hidden | 4d × 4d | pooled = [global, mean of the nodes, mean of the edges, mean of the tiles] |
//! | P + 4 | other policy | 466 × 4d | the other hidden layer -> every other slot's logit, in slot order |
//! | P + 5 | value hidden | 4d × (4d + 30) | [pooled, hidden block] |
//! | P + 6 | value | 4 × 4d | the value hidden layer -> seat-relative win logits |
//! | P + 7 | aux | 4 × 4d | the value hidden layer -> seat-relative final VP / 10 |
//!
//! The encoders, the other hidden layer and the value hidden layer end in a ReLU. A round layer
//! updates its entities as x ← x + ReLU(W·input + b); a round runs its four layers in the order
//! listed, each reading the values the earlier ones just wrote (`forward.rs`).

use crate::encode::{EDGE_WIDTH, ENCODING_HASH, HIDDEN_LEN, NODE_WIDTH, TILE_WIDTH};
use crate::graph::GLOBAL_WIDTH;
use crate::policy::OTHER_POLICY_LEN;
use settler_engine::rng::Rng;
use settler_engine::NUM_PLAYERS;
use std::path::Path;

pub const MAGIC: [u8; 8] = *b"ASNET\0\0\0";
pub const FORMAT_VERSION: u32 = 2;
/// Magic, version, hash, width, rounds and layer count: the bytes before the layer shapes.
pub const FIXED_HEADER_LEN: usize = 8 + 4 + 8 + 4 + 4 + 4;
/// Widest `d` a file may declare: a guard against a corrupt header.
pub const MAX_WIDTH: usize = 1024;
/// Most rounds a file may declare.
pub const MAX_ROUNDS: usize = 16;
/// Encoder v2's `ENCODING_HASH` (3b's nets, before the trade block) and its global width. The
/// loader refuses such a file; `alphasettler net-upgrade` rewrites it for this encoder.
pub const V2_ENCODING_HASH: u64 = 0x28dd_1f78_0ff2_a1b3;
pub const V2_GLOBAL_WIDTH: usize = 91;
/// Encoder v3's `ENCODING_HASH` (3c's and P2's nets, before the road and threat inputs) and its
/// input widths. Encoder v2's node and edge inputs are v3's. `alphasettler net-upgrade` rewrites
/// such files for this encoder.
pub const V3_ENCODING_HASH: u64 = 0x6d02_356c_b617_53c2;
pub const V3_NODE_WIDTH: usize = 19;
pub const V3_EDGE_WIDTH: usize = 4;
pub const V3_GLOBAL_WIDTH: usize = 115;

pub const NODE_ENCODER: usize = 0;
pub const EDGE_ENCODER: usize = 1;
pub const TILE_ENCODER: usize = 2;
pub const GLOBAL_ENCODER: usize = 3;
/// Round r's layers start at `ROUND_BASE + LAYERS_PER_ROUND * r` (`Arch::round_layer`).
pub const ROUND_BASE: usize = 4;
pub const LAYERS_PER_ROUND: usize = 4;
pub const ROUND_TILE: usize = 0;
pub const ROUND_NODE: usize = 1;
pub const ROUND_EDGE: usize = 2;
pub const ROUND_GLOBAL: usize = 3;
/// The heads follow the last round (`Arch::head_layer`).
pub const HEAD_LAYERS: usize = 8;
pub const NODE_POLICY: usize = 0;
pub const EDGE_POLICY: usize = 1;
pub const TILE_POLICY: usize = 2;
pub const OTHER_HIDDEN: usize = 3;
pub const OTHER_POLICY: usize = 4;
pub const VALUE_HIDDEN: usize = 5;
pub const VALUE: usize = 6;
pub const AUX: usize = 7;

/// The two free dimensions: the width `d` and the round count `R`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Arch {
    pub width: usize,
    pub rounds: usize,
}

impl Arch {
    /// Spec Section 2's defaults: about 2.8M multiply-adds per position.
    pub const DEFAULT: Arch = Arch {
        width: 48,
        rounds: 2,
    };

    /// An error naming the dimension out of range, if either is.
    pub fn check(&self) -> Result<(), String> {
        if !(1..=MAX_WIDTH).contains(&self.width) {
            return Err(format!(
                "width {} is out of range 1..={MAX_WIDTH}",
                self.width
            ));
        }
        if !(1..=MAX_ROUNDS).contains(&self.rounds) {
            return Err(format!(
                "rounds {} is out of range 1..={MAX_ROUNDS}",
                self.rounds
            ));
        }
        Ok(())
    }

    pub fn layer_count(&self) -> usize {
        ROUND_BASE + LAYERS_PER_ROUND * self.rounds + HEAD_LAYERS
    }

    pub fn header_len(&self) -> usize {
        FIXED_HEADER_LEN + 8 * self.layer_count()
    }

    /// The index of round `r`'s layer `which` (`ROUND_TILE` ..= `ROUND_GLOBAL`).
    pub fn round_layer(&self, r: usize, which: usize) -> usize {
        ROUND_BASE + LAYERS_PER_ROUND * r + which
    }

    /// The index of head layer `which` (`NODE_POLICY` ..= `AUX`).
    pub fn head_layer(&self, which: usize) -> usize {
        ROUND_BASE + LAYERS_PER_ROUND * self.rounds + which
    }

    /// (rows, cols) of each layer in file order; rows are outputs.
    pub fn shapes(&self) -> Vec<(usize, usize)> {
        self.shapes_for(NODE_WIDTH, EDGE_WIDTH, GLOBAL_WIDTH)
    }

    /// `shapes` for encoders of `node`, `edge` and `global` inputs (an older encoder's); the tile
    /// encoder and every later layer are fixed.
    pub fn shapes_for(&self, node: usize, edge: usize, global: usize) -> Vec<(usize, usize)> {
        let d = self.width;
        let mut s = vec![(d, node), (d, edge), (d, TILE_WIDTH), (d, global)];
        for _ in 0..self.rounds {
            s.extend([(d, 3 * d), (d, 5 * d), (d, 3 * d), (d, 4 * d)]);
        }
        s.extend([
            (2, 2 * d),
            (1, 2 * d),
            (1, 2 * d),
            (4 * d, 4 * d),
            (OTHER_POLICY_LEN, 4 * d),
            (4 * d, 4 * d + HIDDEN_LEN),
            (NUM_PLAYERS, 4 * d),
            (NUM_PLAYERS, 4 * d),
        ]);
        s
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

#[derive(Clone, Debug, PartialEq)]
pub struct Weights {
    pub arch: Arch,
    pub layers: Vec<Layer>,
}

fn uniform(rng: &mut Rng, bound: f32) -> f32 {
    ((rng.next_u64() >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0) * bound
}

impl Weights {
    /// He-uniform weights and small biases from `seed`, for tests and measurements. Panics if
    /// `arch` is out of range.
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

    pub fn param_count(&self) -> usize {
        self.layers.iter().map(|l| l.w.len() + l.b.len()).sum()
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.arch.header_len() + 4 * self.param_count());
        out.extend_from_slice(&MAGIC);
        out.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
        out.extend_from_slice(&ENCODING_HASH.to_le_bytes());
        out.extend_from_slice(&(self.arch.width as u32).to_le_bytes());
        out.extend_from_slice(&(self.arch.rounds as u32).to_le_bytes());
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

    /// Every check runs before anything is sized from the file: the width and round count are
    /// bounded first, then the layer count and shapes must be the ones they imply.
    pub fn from_bytes(bytes: &[u8]) -> Result<Weights, String> {
        if bytes.len() < FIXED_HEADER_LEN {
            return Err(format!(
                "truncated: {} bytes is shorter than the {FIXED_HEADER_LEN}-byte header",
                bytes.len()
            ));
        }
        if bytes[..8] != MAGIC {
            return Err("not an AlphaSettler weights file (wrong magic)".into());
        }
        let u32_at = |i: usize| u32::from_le_bytes(bytes[i..i + 4].try_into().unwrap());
        let version = u32_at(8);
        if version != FORMAT_VERSION {
            let why = if version == 1 {
                "; version 1 holds the flat MLP that network v2 replaced: retrain the net"
            } else {
                ""
            };
            return Err(format!(
                "format version {version}, expected {FORMAT_VERSION}{why}"
            ));
        }
        let hash = u64::from_le_bytes(bytes[12..20].try_into().unwrap());
        if hash != ENCODING_HASH {
            let why = if hash == V2_ENCODING_HASH {
                "; it is an encoder v2 net (before the trade block): convert it with \
                 `alphasettler net-upgrade OLD NEW`"
            } else if hash == V3_ENCODING_HASH {
                "; it is an encoder v3 net (before the road and threat inputs): convert it with \
                 `alphasettler net-upgrade OLD NEW`"
            } else {
                ""
            };
            return Err(format!(
                "encoding hash {hash:#018x} does not match this encoder's {ENCODING_HASH:#018x}: \
                 the file was trained on a different feature layout{why}"
            ));
        }
        let arch = Arch {
            width: u32_at(20) as usize,
            rounds: u32_at(24) as usize,
        };
        arch.check()?;
        let count = u32_at(28) as usize;
        if count != arch.layer_count() {
            return Err(format!(
                "{count} layers, expected {} for {} rounds",
                arch.layer_count(),
                arch.rounds
            ));
        }
        let header = arch.header_len();
        if bytes.len() < header {
            return Err(format!(
                "truncated: {} bytes is shorter than the {header}-byte header",
                bytes.len()
            ));
        }
        let shapes: Vec<(usize, usize)> = (0..count)
            .map(|i| {
                let at = FIXED_HEADER_LEN + 8 * i;
                (u32_at(at) as usize, u32_at(at + 4) as usize)
            })
            .collect();
        if shapes != arch.shapes() {
            return Err(format!(
                "layer shapes {shapes:?} do not fit network v2 with width {} and {} rounds",
                arch.width, arch.rounds
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
