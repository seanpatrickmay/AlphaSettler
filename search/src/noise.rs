//! Root exploration for self-play (spec Section 4): Dirichlet noise mixed into the root prior,
//! from gamma variates drawn with Marsaglia–Tsang from the engine's seeded `Rng`. The search loop
//! itself is unchanged: `Noised` wraps the evaluator.

use crate::evaluator::{Eval, Evaluator, Leaf};
use settler_engine::rng::Rng;

/// `prior' = (1 - weight) · prior + weight · Dir(alpha_total / n)` over the root's n moves.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RootNoise {
    pub weight: f32,
    pub alpha_total: f32,
}

/// Uniform in [0, 1).
pub fn unit(rng: &mut Rng) -> f64 {
    (rng.next_u64() >> 11) as f64 / (1u64 << 53) as f64
}

/// Uniform in (0, 1): safe to take the log of.
fn open_unit(rng: &mut Rng) -> f64 {
    ((rng.next_u64() >> 11) as f64 + 0.5) / (1u64 << 53) as f64
}

/// A standard normal variate (Box–Muller).
pub fn normal(rng: &mut Rng) -> f64 {
    let (u1, u2) = (open_unit(rng), open_unit(rng));
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

/// A Gamma(shape, 1) variate (Marsaglia and Tsang, 2000). Shapes below 1 use the boost
/// Gamma(shape) = Gamma(shape + 1) · U^(1 / shape).
pub fn gamma(shape: f64, rng: &mut Rng) -> f64 {
    assert!(
        shape > 0.0 && shape.is_finite(),
        "gamma shape must be positive, got {shape}"
    );
    if shape < 1.0 {
        let u = open_unit(rng);
        return gamma(shape + 1.0, rng) * u.powf(1.0 / shape);
    }
    let d = shape - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();
    loop {
        let x = normal(rng);
        let v = 1.0 + c * x;
        if v <= 0.0 {
            continue;
        }
        let v = v * v * v;
        let u = open_unit(rng);
        if u < 1.0 - 0.0331 * x.powi(4) || u.ln() < 0.5 * x * x + d * (1.0 - v + v.ln()) {
            return d * v;
        }
    }
}

/// A symmetric Dirichlet(alpha, ..., alpha) draw over `n` entries.
pub fn dirichlet(alpha: f64, n: usize, rng: &mut Rng) -> Vec<f32> {
    let g: Vec<f64> = (0..n).map(|_| gamma(alpha, rng)).collect();
    let total: f64 = g.iter().sum();
    if total > 0.0 {
        g.iter().map(|&x| (x / total) as f32).collect()
    } else {
        vec![1.0 / n as f32; n]
    }
}

/// An evaluator that mixes `noise` into the prior of the first leaf it evaluates. `search`
/// evaluates the root alone in its first batch (the first simulation), so only the root's prior
/// changes. Make a fresh `Noised` for every search.
pub struct Noised<'a, E> {
    inner: &'a mut E,
    noise: RootNoise,
    rng: Rng,
    pending: bool,
}

impl<'a, E: Evaluator> Noised<'a, E> {
    pub fn new(inner: &'a mut E, noise: RootNoise, seed: u64) -> Noised<'a, E> {
        Noised {
            inner,
            noise,
            rng: Rng::new(seed),
            pending: true,
        }
    }
}

impl<E: Evaluator> Evaluator for Noised<'_, E> {
    fn evaluate(&mut self, leaves: &[Leaf<'_>]) -> Vec<Eval> {
        let mut evals = self.inner.evaluate(leaves);
        if self.pending {
            if let Some(e) = evals.first_mut() {
                let n = e.prior.len();
                let eta = dirichlet(self.noise.alpha_total as f64 / n as f64, n, &mut self.rng);
                let w = self.noise.weight;
                for (p, x) in e.prior.iter_mut().zip(eta) {
                    *p = (1.0 - w) * *p + w * x;
                }
                self.pending = false;
            }
        }
        evals
    }
}
