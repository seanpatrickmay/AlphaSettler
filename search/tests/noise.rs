mod common;

use common::*;
use settler_engine::rng::Rng;
use settler_engine::*;
use settler_search::tree::ROOT;
use settler_search::{
    dirichlet, gamma, search_actions, search_tree, Belief, Evaluator, Leaf, Noised, RootNoise,
    SearchConfig, UniformEvaluator,
};

const NOISE: RootNoise = RootNoise {
    weight: 0.25,
    alpha_total: 10.0,
};

/// Player 0 in Main after setup holding two of every resource: several searchable moves.
fn main_position(seed: u64) -> State {
    edit(&after_setup(seed, no_trades()), no_trades(), |snap| {
        give_cards(snap, 0, [2, 2, 2, 2, 2]);
        snap.phase = Phase::Main;
    })
}

#[test]
fn gamma_variates_have_the_right_mean_and_variance() {
    for shape in [0.2, 1.0, 3.5] {
        let mut rng = Rng::new(7);
        let n = 40_000;
        let xs: Vec<f64> = (0..n).map(|_| gamma(shape, &mut rng)).collect();
        assert!(xs.iter().all(|&x| x >= 0.0 && x.is_finite()));
        let mean = xs.iter().sum::<f64>() / n as f64;
        let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
        assert!(
            (mean - shape).abs() < 4.0 * (shape / n as f64).sqrt(),
            "shape {shape}: mean {mean}"
        );
        assert!(
            (var - shape).abs() < 0.1 * shape,
            "shape {shape}: variance {var}"
        );
    }
}

#[test]
fn dirichlet_draws_are_distributions_with_even_means() {
    let mut rng = Rng::new(3);
    let (n, draws) = (5, 20_000);
    let mut mean = [0.0f64; 5];
    for _ in 0..draws {
        let d = dirichlet(10.0 / n as f64, n, &mut rng);
        assert_eq!(d.len(), n);
        assert!((d.iter().map(|&x| x as f64).sum::<f64>() - 1.0).abs() < 1e-5);
        assert!(d.iter().all(|&x| x >= 0.0));
        for i in 0..n {
            mean[i] += d[i] as f64 / draws as f64;
        }
    }
    assert!(mean.iter().all(|m| (m - 0.2).abs() < 0.01), "{mean:?}");
}

#[test]
fn noise_changes_only_the_first_evaluation() {
    let s = main_position(4);
    let mut legal = Vec::new();
    search_actions(&s.legal_actions(), &mut legal);
    let n = legal.len();
    assert!(n >= 3, "{legal:?}");
    let mut inner = UniformEvaluator;
    let mut noised = Noised::new(&mut inner, NOISE, 99);
    let first = noised.evaluate(&[Leaf {
        world: &s,
        actor: 0,
        legal: &legal,
    }]);
    let eta = dirichlet(10.0 / n as f64, n, &mut Rng::new(99));
    for (p, e) in first[0].prior.iter().zip(&eta) {
        assert!((p - (0.75 * (1.0 / n as f32) + 0.25 * e)).abs() < 1e-6);
    }
    assert!((first[0].prior.iter().sum::<f32>() - 1.0).abs() < 1e-5);
    let second = noised.evaluate(&[Leaf {
        world: &s,
        actor: 0,
        legal: &legal,
    }]);
    assert_eq!(second[0].prior, vec![1.0 / n as f32; n]);
}

#[test]
fn the_first_evaluation_of_a_search_is_the_root() {
    let s = main_position(5);
    let obs = s.observation(0);
    let all = s.legal_actions();
    let mut legal = Vec::new();
    search_actions(&all, &mut legal);
    let n = legal.len();
    let belief = Belief::from_observation(&obs, 64, 1);
    let mut inner = UniformEvaluator;
    let mut noised = Noised::new(&mut inner, NOISE, 42);
    let cfg = SearchConfig {
        simulations: 60,
        ..SearchConfig::default()
    };
    let (_, tree) = search_tree(&obs, &all, &belief, &mut noised, &cfg, &mut Rng::new(5)).unwrap();
    let eta = dirichlet(10.0 / n as f64, n, &mut Rng::new(42));
    let root = &tree.nodes[ROOT as usize];
    assert_eq!(root.children.len(), n);
    for ((c, a), e) in root.children.iter().zip(&legal).zip(&eta) {
        assert_eq!(c.key, a.encode());
        assert!(
            (c.prior - (0.75 * (1.0 / n as f32) + 0.25 * e)).abs() < 1e-6,
            "{a:?}"
        );
    }
}
