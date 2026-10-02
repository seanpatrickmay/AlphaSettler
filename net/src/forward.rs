//! The batched MLP forward pass (f32): the only inference code. `net/tests/weights.rs` checks it
//! against a plain f64 reference and `tests/python/test_train.py` against the PyTorch twin
//! (`alphasettler/train/model.py`).

use crate::encode::{HIDDEN_LEN, PUBLIC_LEN};
use crate::weights::{Layer, Weights, AUX, POLICY, TRUNK0, TRUNK1, TRUNK2, VALUE0, VALUE1};

/// Raw head outputs for `n` positions, row-major: 665 policy logits, 4 seat-relative value logits
/// (before the softmax) and 4 aux outputs (final VP / 10, seat-relative) per position.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Outputs {
    pub n: usize,
    pub policy: Vec<f32>,
    pub value: Vec<f32>,
    pub aux: Vec<f32>,
}

/// Activation buffers reused across calls.
#[derive(Clone, Debug, Default)]
pub struct Forward {
    a: Vec<f32>,
    b: Vec<f32>,
    joined: Vec<f32>,
    v: Vec<f32>,
}

/// Eight independent partial sums so the compiler can vectorise, combined in a fixed order: a
/// row's result never depends on the rest of the batch.
#[inline]
pub fn dot(x: &[f32], y: &[f32]) -> f32 {
    debug_assert_eq!(x.len(), y.len());
    let (xc, yc) = (x.chunks_exact(8), y.chunks_exact(8));
    let (xr, yr) = (xc.remainder(), yc.remainder());
    let mut acc = [0.0f32; 8];
    for (a, b) in xc.zip(yc) {
        for j in 0..8 {
            acc[j] += a[j] * b[j];
        }
    }
    let mut tail = 0.0;
    for (a, b) in xr.iter().zip(yr) {
        tail += a * b;
    }
    ((acc[0] + acc[1]) + (acc[2] + acc[3])) + ((acc[4] + acc[5]) + (acc[6] + acc[7])) + tail
}

/// In place, numerically stable; entries of -inf become 0. Requires at least one finite entry and
/// none of +inf: otherwise the result is NaN (`evaluator::masked_prior` handles that case itself).
pub fn softmax(x: &mut [f32]) {
    let m = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    debug_assert!(m.is_finite(), "softmax needs a finite maximum, got {m}");
    let mut total = 0.0;
    for v in x.iter_mut() {
        *v = (*v - m).exp();
        total += *v;
    }
    for v in x.iter_mut() {
        *v /= total;
    }
}

/// `y[i][o] = b[o] + w[o] · x[i]` for the `n` rows of `x`, optionally through a ReLU.
fn dense(l: &Layer, x: &[f32], n: usize, relu: bool, y: &mut Vec<f32>) {
    debug_assert_eq!(x.len(), n * l.cols);
    y.clear();
    y.resize(n * l.rows, 0.0);
    for o in 0..l.rows {
        let w = &l.w[o * l.cols..(o + 1) * l.cols];
        for i in 0..n {
            let s = l.b[o] + dot(w, &x[i * l.cols..(i + 1) * l.cols]);
            y[i * l.rows + o] = if relu { s.max(0.0) } else { s };
        }
    }
}

impl Forward {
    pub fn new() -> Forward {
        Forward::default()
    }

    /// Run `n` positions: `public` holds `n × PUBLIC_LEN` floats and `hidden` `n × HIDDEN_LEN`.
    pub fn run(
        &mut self,
        w: &Weights,
        public: &[f32],
        hidden: &[f32],
        n: usize,
        out: &mut Outputs,
    ) {
        assert_eq!(public.len(), n * PUBLIC_LEN, "public block length");
        assert_eq!(hidden.len(), n * HIDDEN_LEN, "hidden block length");
        let t = w.layers[TRUNK2].rows;
        dense(&w.layers[TRUNK0], public, n, true, &mut self.a);
        dense(&w.layers[TRUNK1], &self.a, n, true, &mut self.b);
        dense(&w.layers[TRUNK2], &self.b, n, true, &mut self.a);
        dense(&w.layers[POLICY], &self.a, n, false, &mut out.policy);
        self.joined.clear();
        for i in 0..n {
            self.joined.extend_from_slice(&self.a[i * t..(i + 1) * t]);
            self.joined
                .extend_from_slice(&hidden[i * HIDDEN_LEN..(i + 1) * HIDDEN_LEN]);
        }
        dense(&w.layers[VALUE0], &self.joined, n, true, &mut self.v);
        dense(&w.layers[VALUE1], &self.v, n, false, &mut out.value);
        dense(&w.layers[AUX], &self.v, n, false, &mut out.aux);
        out.n = n;
    }
}
