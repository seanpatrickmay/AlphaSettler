use settler_bots::heuristic::{fit, log_likelihood, samples_from_games, ReplayGame, Sample, DEFAULT_WEIGHTS, NUM_FEATURES};
use settler_bots::selfplay::{play, run};
use settler_engine::rng::Rng;
use settler_engine::*;

fn no_trades() -> GameConfig {
    GameConfig { max_offers_per_turn: 0, ..GameConfig::default() }
}

#[test]
fn every_searched_decision_is_recorded_and_replays() {
    let g = play(3, no_trades(), 10, 0);
    assert!(!g.decisions.is_empty());
    let mut s = State::new(3, no_trades());
    let mut decisions = g.decisions.iter().peekable();
    for (i, &a) in g.actions.iter().enumerate() {
        if let Some(d) = decisions.next_if(|d| d.index as usize == i) {
            assert_eq!(d.actor, s.current_actor());
            assert_eq!(d.legal, s.legal_actions());
            assert!(d.legal.contains(&a));
            assert_eq!(d.visits.len(), d.legal.len());
            assert_eq!(d.visits.iter().sum::<u32>(), 9);
            assert!(d.full_search);
        }
        s.apply(a);
    }
    assert!(decisions.next().is_none());
    assert_eq!(s.winner(), g.winner);
    assert_eq!(s.turn, g.turns);
}

#[test]
fn self_play_does_not_depend_on_thread_count() {
    assert_eq!(run(0..4, no_trades(), 10, 0, 1).unwrap(), run(0..4, no_trades(), 10, 0, 3).unwrap());
}

#[test]
fn the_fit_recovers_known_weights() {
    let truth = [0.8f32, -0.5, 0.3, 0.0, 0.6, -0.2, 0.1, 0.4];
    let mut rng = Rng::new(11);
    let mut uniform = || (rng.next_u64() >> 40) as f32 / (1u64 << 24) as f32;
    let samples: Vec<Sample> = (0..6000)
        .map(|_| {
            let features: [[f32; NUM_FEATURES]; 4] = std::array::from_fn(|_| std::array::from_fn(|_| 3.0 * uniform()));
            let scores: Vec<f32> = features.iter().map(|f| f.iter().zip(&truth).map(|(x, w)| x * w).sum()).collect();
            let m = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let e: Vec<f32> = scores.iter().map(|s| (s - m).exp()).collect();
            let total: f32 = e.iter().sum();
            let (mut u, mut winner) = (uniform() * total, 3);
            for (p, &x) in e.iter().enumerate() {
                if u < x {
                    winner = p;
                    break;
                }
                u -= x;
            }
            Sample { features, winner }
        })
        .collect();
    let w = fit(&samples, [0.0; NUM_FEATURES], 25, 0.0);
    for i in 0..NUM_FEATURES {
        assert!((w[i] - truth[i]).abs() < 0.15, "weight {i}: {} vs {}", w[i], truth[i]);
    }
    assert!(log_likelihood(&samples, &w) >= log_likelihood(&samples, &[0.0; NUM_FEATURES]));
}

#[test]
fn fitting_on_real_games_does_not_lower_the_likelihood() {
    let games: Vec<ReplayGame> = run(0..6, no_trades(), 10, 0, 6)
        .unwrap()
        .into_iter()
        .map(|g| ReplayGame {
            seed: g.seed,
            config: no_trades(),
            at: g.decisions.iter().map(|d| d.index).collect(),
            actions: g.actions,
            winner: g.winner,
        })
        .collect();
    let samples = samples_from_games(&games).unwrap();
    assert!(samples.len() > 100);
    let w = fit(&samples, DEFAULT_WEIGHTS, 25, 1e-3);
    assert!(log_likelihood(&samples, &w) >= log_likelihood(&samples, &DEFAULT_WEIGHTS));
}
