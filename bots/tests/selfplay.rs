mod common;

use common::trading;
use settler_bots::heuristic::{
    fit, log_likelihood, samples_from_games, HeuristicEvaluator, ReplayGame, Sample,
    DEFAULT_WEIGHTS, NUM_FEATURES,
};
use settler_bots::selfplay::{
    play, play_with, run, run_with, Exploration, SelfPlayGame, FAST_SIMULATIONS, FULL_PROBABILITY,
    FULL_SIMULATIONS, MIN_SIMULATIONS, NOISE, SAMPLE_TURNS,
};
use settler_engine::rng::Rng;
use settler_engine::*;

fn no_trades() -> GameConfig {
    GameConfig {
        max_offers_per_turn: 0,
        ..GameConfig::default()
    }
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
    assert_eq!(g.belief_resets, 0);
}

#[test]
fn self_play_does_not_depend_on_thread_count() {
    assert_eq!(
        run(0..4, no_trades(), 10, 0, 1).unwrap(),
        run(0..4, no_trades(), 10, 0, 3).unwrap()
    );
}

#[test]
fn self_play_refuses_searches_too_small_to_record_visits() {
    assert_eq!(MIN_SIMULATIONS, 2);
    let e = run(0..1, no_trades(), 1, 0, 1).unwrap_err();
    assert!(e.contains("at least 2 simulations"), "{e}");
    assert!(run(0..1, no_trades(), 2, 0, 1).unwrap()[0]
        .decisions
        .iter()
        .all(|d| d.visits.iter().sum::<u32>() == 1));
}

#[test]
fn the_fit_recovers_known_weights() {
    let truth = [0.8f32, -0.5, 0.3, 0.0, 0.6, -0.2, 0.1, 0.4];
    let mut rng = Rng::new(11);
    let mut uniform = || (rng.next_u64() >> 40) as f32 / (1u64 << 24) as f32;
    let samples: Vec<Sample> = (0..6000)
        .map(|_| {
            let features: [[f32; NUM_FEATURES]; 4] =
                std::array::from_fn(|_| std::array::from_fn(|_| 3.0 * uniform()));
            let scores: Vec<f32> = features
                .iter()
                .map(|f| f.iter().zip(&truth).map(|(x, w)| x * w).sum())
                .collect();
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
        assert!(
            (w[i] - truth[i]).abs() < 0.15,
            "weight {i}: {} vs {}",
            w[i],
            truth[i]
        );
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

/// FNV-1a over everything a self-play game records, in order.
fn digest(g: &SelfPlayGame) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let mut eat = |bytes: &[u8]| {
        for &b in bytes {
            h ^= b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    };
    eat(&g.seed.to_le_bytes());
    for a in &g.actions {
        eat(&a.encode().to_le_bytes());
    }
    for d in &g.decisions {
        eat(&d.index.to_le_bytes());
        eat(&[d.actor, d.full_search as u8]);
        for (a, v) in d.legal.iter().zip(&d.visits) {
            eat(&a.encode().to_le_bytes());
            eat(&v.to_le_bytes());
        }
    }
    eat(&[g.winner.unwrap_or(u8::MAX)]);
    eat(&g.vp);
    eat(&g.turns.to_le_bytes());
    eat(&g.belief_resets.to_le_bytes());
    h
}

fn small_explore() -> Exploration {
    Exploration {
        full_simulations: 6,
        fast_simulations: 3,
        ..Exploration::standard()
    }
}

fn heuristic() -> HeuristicEvaluator {
    HeuristicEvaluator::new(DEFAULT_WEIGHTS, 0)
}

#[test]
fn the_standard_settings_are_the_specs() {
    let ex = Exploration::standard();
    assert_eq!(
        (ex.full_simulations, ex.fast_simulations),
        (FULL_SIMULATIONS, FAST_SIMULATIONS)
    );
    assert_eq!(
        (FULL_SIMULATIONS, FAST_SIMULATIONS, FULL_PROBABILITY),
        (600, 100, 0.25)
    );
    assert_eq!(
        (NOISE.weight, NOISE.alpha_total, SAMPLE_TURNS),
        (0.25, 10.0, 8)
    );
    assert_eq!(ex.noise, Some(NOISE));
    for bad in [
        Exploration {
            full_simulations: 1,
            ..ex
        },
        Exploration {
            fast_simulations: 1_000_001,
            ..ex
        },
        Exploration {
            full_probability: 1.5,
            ..ex
        },
        Exploration {
            full_probability: f64::NAN,
            ..ex
        },
    ] {
        assert!(bad.validate().is_err(), "{bad:?}");
        assert!(run_with(0..1, no_trades(), &bad, 8, &heuristic, 1).is_err());
    }
}

#[test]
fn about_a_quarter_of_decisions_are_full_searches() {
    let games = run_with(0..16, no_trades(), &small_explore(), 8, &heuristic, 8).unwrap();
    let decisions: Vec<_> = games.iter().flat_map(|g| &g.decisions).collect();
    let n = decisions.len() as f64;
    assert!(n >= 2000.0, "only {n} decisions");
    let full = decisions.iter().filter(|d| d.full_search).count() as f64;
    let se = (0.25 * 0.75 / n).sqrt();
    assert!(
        (full / n - 0.25).abs() < 3.0 * se,
        "{full} full searches of {n}"
    );
    for d in &decisions {
        let total: u32 = d.visits.iter().sum();
        assert_eq!(total, if d.full_search { 5 } else { 2 });
    }
    assert!(games.iter().all(|g| g.belief_resets == 0));
}

#[test]
fn exploring_self_play_is_deterministic_and_thread_independent() {
    let a = run_with(0..4, no_trades(), &small_explore(), 8, &heuristic, 1).unwrap();
    let b = run_with(0..4, no_trades(), &small_explore(), 8, &heuristic, 3).unwrap();
    assert_eq!(a, b);
    assert_eq!(
        a[2],
        play_with(2, no_trades(), &small_explore(), 8, &heuristic)
    );
    assert_ne!(
        a[0].actions,
        play(0, no_trades(), 6, 0).actions,
        "exploration changes the games"
    );
}

#[test]
fn opening_moves_are_sampled_and_later_moves_are_the_most_visited() {
    let games = run_with(0..6, no_trades(), &small_explore(), 8, &heuristic, 3).unwrap();
    let mut off_best = 0;
    for g in &games {
        let mut s = State::new(g.seed, no_trades());
        let mut decisions = g.decisions.iter().peekable();
        for (i, &a) in g.actions.iter().enumerate() {
            if let Some(d) = decisions.next_if(|d| d.index as usize == i) {
                let chosen = d.visits[d.legal.iter().position(|&x| x == a).unwrap()];
                let best = *d.visits.iter().max().unwrap();
                assert!(chosen > 0, "move {i} of seed {} was never searched", g.seed);
                if s.turn < SAMPLE_TURNS {
                    off_best += (chosen < best) as u32;
                } else {
                    assert_eq!(chosen, best, "move {i} of seed {}", g.seed);
                }
            }
            s.apply(a);
        }
    }
    assert!(
        off_best > 0,
        "no opening move was sampled away from the most-visited one"
    );
}

/// The digests of 3a's self-play games, computed at d5e2a97 before self-play was rewritten on
/// top of `play_with` (F1): `play` and `Exploration::fixed` must not change a single game. The
/// second pins the rollout path.
#[test]
fn fixed_exploration_is_3a_self_play() {
    const PLAY_3_AT_10: u64 = 0x517b49ad6e41ec1e;
    const PLAY_5_AT_12_ROLLOUT_3: u64 = 0x8a2fe1b5ac54c2fc;
    let g = play(3, no_trades(), 10, 0);
    assert_eq!(digest(&g), PLAY_3_AT_10);
    assert!(g.decisions.iter().all(|d| d.full_search));
    let ex = Exploration::fixed(10);
    assert_eq!(
        digest(&play_with(3, no_trades(), &ex, 8, &heuristic)),
        PLAY_3_AT_10
    );
    assert_eq!(digest(&play(5, no_trades(), 12, 3)), PLAY_5_AT_12_ROLLOUT_3);
}

/// The digest of a trading-game self-play game under the 3b exploration at 6 / 3 simulations,
/// computed at eb1f739 before the Gumbel root (P2): PUCT self-play must not change.
#[test]
fn standard_exploration_plays_the_games_it_played_before_p2() {
    assert_eq!(
        digest(&play_with(2, trading(), &small_explore(), 8, &heuristic)),
        0x94f447a9c3baf5b2
    );
}

/// Noise is drawn only for full searches: with no full searches, noise on or off plays the same
/// games (fast searches stay un-noised).
#[test]
fn fast_searches_are_never_noised() {
    let fast_only = Exploration {
        full_probability: 0.0,
        ..small_explore()
    };
    assert_eq!(fast_only.noise, Some(NOISE));
    let quiet = Exploration {
        noise: None,
        ..fast_only
    };
    let noised = run_with(0..3, no_trades(), &fast_only, 8, &heuristic, 3).unwrap();
    assert_eq!(
        noised,
        run_with(0..3, no_trades(), &quiet, 8, &heuristic, 3).unwrap()
    );
    assert!(noised
        .iter()
        .flat_map(|g| &g.decisions)
        .all(|d| !d.full_search));
}

/// The noise reaches the search: at 40 simulations, noised and plain root searches record
/// different visits on some position of a game.
#[test]
fn root_noise_changes_the_recorded_visits() {
    use settler_bots::ismcts::{IsmctsBot, MoveOptions};
    let g = play(3, no_trades(), 10, 0);
    let mut s = State::new(3, no_trades());
    let (mut compared, mut differ) = (0, 0);
    for (i, &a) in g.actions.iter().enumerate() {
        if i % 7 == 0 && !s.is_over() {
            let (obs, legal) = (s.observation(s.current_actor()), s.legal_actions());
            let visits = |noise| {
                let mut bot = IsmctsBot::new(11, 40, 0);
                let opts = MoveOptions {
                    simulations: 40,
                    noise,
                    sample: None,
                };
                bot.act_with(&obs, &legal, &opts);
                bot.take_last_search().map(|r| r.visits)
            };
            if let (Some(plain), Some(noised)) = (visits(None), visits(Some((NOISE, i as u64)))) {
                compared += 1;
                differ += (plain != noised) as u32;
            }
        }
        s.apply(a);
    }
    assert!(compared >= 10, "only {compared} searched positions");
    assert!(differ > 0, "noise changed none of {compared} searches");
}

/// Trading self-play: every seat may offer, and offers, answers and confirmations are recorded
/// like any other searched decision, visits aligned with the legal moves (spec 3c Section 5).
/// Moves are drawn in proportion to visits, so offers with any visits get played.
#[test]
fn trading_self_play_records_offers_answers_and_confirmations() {
    let ex = Exploration {
        sample_turns: u32::MAX,
        ..Exploration::fixed(20)
    };
    let (mut offers, mut answers, mut confirmations) = (0, 0, 0);
    for seed in 0..3 {
        let g = play_with(seed, trading(), &ex, 8, &heuristic);
        assert_eq!(g.belief_resets, 0, "seed {seed}");
        let mut s = State::new(seed, trading());
        let mut decisions = g.decisions.iter().peekable();
        for (i, &a) in g.actions.iter().enumerate() {
            if let Some(d) = decisions.next_if(|d| d.index as usize == i) {
                assert_eq!(d.legal, s.legal_actions());
                assert_eq!(d.visits.len(), d.legal.len());
                offers += d
                    .legal
                    .iter()
                    .zip(&d.visits)
                    .filter(|(a, v)| matches!(a, Action::OfferTrade { .. }) && **v > 0)
                    .count();
                answers += (s.phase == Phase::TradeResponse) as u32;
                confirmations += (s.phase == Phase::TradeConfirm) as u32;
            }
            s.apply(a);
        }
    }
    assert!(
        offers > 0 && answers > 0 && confirmations > 0,
        "{offers} visited offers, {answers} answers, {confirmations} confirmations"
    );
}
