//! Gumbel root search arithmetic (spec docs/superpowers/specs/2026-10-08-gumbel-root-design.md
//! Section 3; Danihelka et al., ICLR 2022, Algorithm 2 and appendix eq. 33), kept apart from the
//! tree walk like `puct` so each rule can be tested on plain numbers. Every slice is aligned with
//! the root actions (the searchable moves of `search_moves`, in engine order); `keys` are their
//! action ids, which break ties (the lower first). Values are the root actor's win probabilities
//! in [0, 1], as the paper assumes.

use crate::noise::open_unit;
use settler_engine::rng::Rng;

/// Candidates sequential halving starts from: m = min(MAX_CANDIDATES, |A|).
pub const MAX_CANDIDATES: usize = 16;
/// σ(q) = (C_VISIT + max_b N(b)) · C_SCALE · q, the paper's values.
pub const C_VISIT: f32 = 50.0;
pub const C_SCALE: f32 = 1.0;
/// Root prior probabilities are floored here before their log: a zero prior gets a finite logit
/// (about -87.3).
pub const MIN_PRIOR: f32 = f32::MIN_POSITIVE;

/// A Gumbel(0, 1) draw: -ln(-ln U), U uniform in (0, 1).
pub fn gumbel(rng: &mut Rng) -> f32 {
    (-(-open_unit(rng).ln()).ln()) as f32
}

/// ℓ = ln π: the root prior renormalised over the root actions (uniform if it sums to nothing or
/// is not finite), each probability floored at `MIN_PRIOR`.
pub fn logits(prior: &[f32]) -> Vec<f32> {
    let total: f32 = prior.iter().sum();
    let uniform = 1.0 / prior.len() as f32;
    prior
        .iter()
        .map(|&p| {
            let p = if total > 0.0 && total.is_finite() {
                p / total
            } else {
                uniform
            };
            p.max(MIN_PRIOR).ln()
        })
        .collect()
}

fn softmax64(x: &[f32]) -> Vec<f64> {
    let top = x.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let e: Vec<f64> = x.iter().map(|&v| (v as f64 - top).exp()).collect();
    let total: f64 = e.iter().sum();
    e.into_iter().map(|v| v / total).collect()
}

/// softmax(x), computed in f64.
pub fn softmax(x: &[f32]) -> Vec<f32> {
    softmax64(x).into_iter().map(|p| p as f32).collect()
}

/// `of` (indices into the root actions) ordered by `score`, the largest first, ties to the lower
/// action id.
fn rank(of: &[usize], score: impl Fn(usize) -> f32, keys: &[u16]) -> Vec<usize> {
    let mut out = of.to_vec();
    out.sort_by(|&a, &b| score(b).total_cmp(&score(a)).then(keys[a].cmp(&keys[b])));
    out
}

/// The candidates: the min(MAX_CANDIDATES, |A|) root actions with the largest g + ℓ (`base`),
/// best first.
pub fn candidates(base: &[f32], keys: &[u16]) -> Vec<usize> {
    let all: Vec<usize> = (0..base.len()).collect();
    let mut c = rank(&all, |i| base[i], keys);
    c.truncate(MAX_CANDIDATES);
    c
}

/// ceil(log2 m) halving phases for m candidates (0 for one).
pub fn phases(m: usize) -> u32 {
    if m <= 1 {
        0
    } else {
        usize::BITS - (m - 1).leading_zeros()
    }
}

/// Simulations each remaining candidate gets in a phase: max(1, floor(n / (phases · remaining))).
/// `phases` and `remaining` must be positive.
pub fn per_candidate(n: u32, phases: u32, remaining: usize) -> u32 {
    (n / (phases * remaining as u32)).max(1)
}

/// σ(q) at a root whose most-visited action has `max_visits` visits.
pub fn sigma(q: f32, max_visits: u32) -> f32 {
    (C_VISIT + max_visits as f32) * C_SCALE * q
}

/// v_mix (appendix eq. 33): the root value `v_hat` mixed with the prior-weighted mean q̂ of the
/// visited actions, π = softmax(ℓ). With nothing visited it is `v_hat`.
pub fn v_mix(logits: &[f32], visits: &[u32], q: &[f32], v_hat: f32) -> f32 {
    let total: u32 = visits.iter().sum();
    if total == 0 {
        return v_hat;
    }
    let pi = softmax64(logits);
    let (mut pi_sum, mut pi_q) = (0.0f64, 0.0f64);
    for ((&p, &n), &q) in pi.iter().zip(visits).zip(q) {
        if n > 0 {
            pi_sum += p;
            pi_q += p * q as f64;
        }
    }
    ((v_hat as f64 + total as f64 / pi_sum * pi_q) / (1.0 + total as f64)) as f32
}

/// completedQ: q̂ for a visited action, v_mix for the rest.
pub fn completed_q(logits: &[f32], visits: &[u32], q: &[f32], v_hat: f32) -> Vec<f32> {
    let mix = v_mix(logits, visits, q, v_hat);
    visits
        .iter()
        .zip(q)
        .map(|(&n, &q)| if n > 0 { q } else { mix })
        .collect()
}

/// `of` ordered by g + ℓ + σ(completedQ), best first, ties to the lower action id: sequential
/// halving keeps the front half, and the move is the first.
pub fn ranked(of: &[usize], base: &[f32], keys: &[u16], cq: &[f32], max_visits: u32) -> Vec<usize> {
    rank(of, |i| base[i] + sigma(cq[i], max_visits), keys)
}

/// The improved policy π' = softmax(ℓ + σ(completedQ)) over every root action.
pub fn improved_policy(logits: &[f32], cq: &[f32], max_visits: u32) -> Vec<f32> {
    let x: Vec<f32> = logits
        .iter()
        .zip(cq)
        .map(|(&l, &q)| l + sigma(q, max_visits))
        .collect();
    softmax(&x)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn halving_takes_ceil_log2_phases() {
        assert_eq!(
            [1, 2, 3, 4, 5, 8, 9, 13, 16].map(phases),
            [0, 1, 2, 2, 3, 3, 4, 4, 4]
        );
    }

    #[test]
    fn every_phase_gives_each_candidate_at_least_one_simulation() {
        assert_eq!(per_candidate(99, 4, 16), 1);
        assert_eq!(per_candidate(99, 4, 8), 3);
        assert_eq!(per_candidate(10, 4, 16), 1);
        assert_eq!(per_candidate(599, 4, 4), 37);
    }

    #[test]
    fn candidates_are_the_largest_scores_ties_to_the_lower_id() {
        assert_eq!(
            candidates(&[0.5, 2.0, 2.0, -1.0], &[40, 30, 10, 20]),
            vec![2, 1, 0, 3]
        );
        let many: Vec<f32> = (0..20).map(|i| i as f32).collect();
        let ids: Vec<u16> = (0..20).collect();
        assert_eq!(
            candidates(&many, &ids),
            (4..20).rev().collect::<Vec<usize>>()
        );
    }

    #[test]
    fn v_mix_is_the_root_value_until_something_is_visited() {
        let l = logits(&[0.5, 0.3, 0.2]);
        assert_eq!(v_mix(&l, &[0, 0, 0], &[0.0; 3], 0.4), 0.4);
        // (0.4 + 3 / 0.8 · (0.5 · 0.6 + 0.3 · 0.3)) / 4 = 0.465625
        let m = v_mix(&l, &[2, 1, 0], &[0.6, 0.3, 0.0], 0.4);
        assert!((m - 0.465625).abs() < 1e-6, "{m}");
        let cq = completed_q(&l, &[2, 1, 0], &[0.6, 0.3, 0.0], 0.4);
        assert_eq!(cq[..2], [0.6, 0.3]);
        assert!((cq[2] - 0.465625).abs() < 1e-6);
    }

    #[test]
    fn the_improved_policy_is_the_prior_with_nothing_visited_and_favours_value() {
        let prior = [0.5, 0.3, 0.2];
        let l = logits(&prior);
        let cq = completed_q(&l, &[0; 3], &[0.0; 3], 0.4);
        for (p, want) in improved_policy(&l, &cq, 0).iter().zip(prior) {
            assert!((p - want).abs() < 1e-6, "{p} vs {want}");
        }
        let cq = completed_q(&l, &[2, 1, 0], &[0.6, 0.3, 0.0], 0.4);
        let p = improved_policy(&l, &cq, 2);
        assert!((p.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert!(p[0] > prior[0] && p[1] < prior[1], "{p:?}");
    }

    #[test]
    fn a_zero_prior_has_a_finite_logit() {
        let l = logits(&[1.0, 0.0, 0.0]);
        assert!(l.iter().all(|x| x.is_finite()), "{l:?}");
        assert_eq!(l[0], 0.0);
        assert_eq!(
            logits(&[0.0, 0.0]),
            logits(&[0.5, 0.5]),
            "nothing to renormalise: uniform"
        );
        let p = improved_policy(&l, &[0.25; 3], 0);
        assert!((p.iter().sum::<f32>() - 1.0).abs() < 1e-6 && p.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn gumbel_draws_have_the_standard_mean_and_variance() {
        let mut rng = Rng::new(3);
        let n = 40_000;
        let xs: Vec<f64> = (0..n).map(|_| gumbel(&mut rng) as f64).collect();
        assert!(xs.iter().all(|x| x.is_finite()));
        let mean = xs.iter().sum::<f64>() / n as f64;
        let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
        // Gumbel(0, 1): mean the Euler–Mascheroni constant 0.5772, variance π²/6 = 1.6449.
        assert!((mean - 0.5772).abs() < 0.03, "{mean}");
        assert!((var - 1.6449).abs() < 0.08, "{var}");
    }
}
