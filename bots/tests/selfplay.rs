mod common;

use common::trading;
use settler_bots::heuristic::{
    fit, log_likelihood, samples_from_games, HeuristicEvaluator, ReplayGame, Sample,
    DEFAULT_WEIGHTS, NUM_FEATURES,
};
use settler_bots::opening::OPENING_FEATURES;
use settler_bots::opening_model::{settlement_scores, OpeningLayer, OpeningModel};
use settler_bots::selfplay::{
    play, play_with, play_with_league, play_with_opening, run, run_with, run_with_league,
    run_with_opening, Exploration, League, SelfPlayGame, FAST_SIMULATIONS, FULL_PROBABILITY,
    FULL_SIMULATIONS, LEAGUE_PROBABILITY, LEAGUE_SEATS, MIN_SIMULATIONS, NOISE, OPENING_EPSILON,
    OPENING_TOP, SAMPLE_TURNS,
};
use settler_engine::rng::Rng;
use settler_engine::*;
use settler_net::weights::{Arch, Weights};
use std::path::{Path, PathBuf};

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

#[test]
fn gumbel_self_play_records_an_improved_policy_on_every_searched_decision() {
    let ex = Exploration::gumbel(8);
    assert!(ex.validate().is_ok());
    for (seed, config) in [(0, no_trades()), (1, no_trades()), (2, trading())] {
        let g = play_with(seed, config, &ex, 8, &heuristic);
        assert_eq!(g.belief_resets, 0, "seed {seed}");
        assert!(!g.decisions.is_empty());
        let mut s = State::new(seed, config);
        let mut decisions = g.decisions.iter().peekable();
        for (i, &a) in g.actions.iter().enumerate() {
            if let Some(d) = decisions.next_if(|d| d.index as usize == i) {
                assert_eq!(d.legal, s.legal_actions());
                assert!(d.full_search, "seed {seed} move {i}");
                assert_eq!(d.visits.iter().sum::<u32>(), 7, "seed {seed} move {i}");
                let t = d
                    .target
                    .as_ref()
                    .expect("a Gumbel search records its improved policy");
                assert_eq!(t.len(), d.legal.len());
                assert!(t.iter().all(|x| x.is_finite() && *x >= 0.0));
                let total: f64 = t.iter().map(|&x| x as f64).sum();
                assert!((total - 1.0).abs() < 1e-4, "seed {seed} move {i}: {total}");
            }
            s.apply(a);
        }
    }
}

/// Draws are on in self-play. Without them (g = 0) the move played is the halving winner, which
/// differs from the target's argmax on only a few decisions (ties, and the unvisited moves'
/// completedQ): 6 of 375 on this game. With g drawn it is 66 of 255 here (46% on real data). So a
/// tenth of the decisions playing a move other than the target's argmax separates the two.
#[test]
fn gumbel_self_play_draws_so_the_move_played_often_is_not_the_target_argmax() {
    let g = play_with(0, no_trades(), &Exploration::gumbel(8), 8, &heuristic);
    let differing = g
        .decisions
        .iter()
        .filter(|d| {
            let target = d
                .target
                .as_ref()
                .expect("a Gumbel search records its target");
            let mut argmax = 0;
            for (i, &t) in target.iter().enumerate() {
                if t > target[argmax] {
                    argmax = i;
                }
            }
            d.legal[argmax] != g.actions[d.index as usize]
        })
        .count();
    assert!(
        differing * 10 > g.decisions.len(),
        "only {differing} of {} decisions played other than the target's argmax: the draws look off",
        g.decisions.len()
    );
}

#[test]
fn gumbel_self_play_is_deterministic_and_thread_independent() {
    let ex = Exploration::gumbel(6);
    let a = run_with(0..4, no_trades(), &ex, 8, &heuristic, 1).unwrap();
    assert_eq!(
        a,
        run_with(0..4, no_trades(), &ex, 8, &heuristic, 3).unwrap()
    );
    assert_ne!(
        a[0].actions,
        play_with(0, no_trades(), &Exploration::fixed(6), 8, &heuristic).actions,
        "the Gumbel root plays other games"
    );
}

#[test]
fn gumbel_self_play_refuses_noise_sampling_and_fast_searches() {
    let ex = Exploration::gumbel(100);
    assert_eq!(
        (
            ex.full_simulations,
            ex.fast_simulations,
            ex.full_probability,
            ex.noise,
            ex.sample_turns,
            ex.gumbel
        ),
        (100, 100, 1.0, None, 0, true)
    );
    assert!(!Exploration::standard().gumbel && !Exploration::fixed(5).gumbel);
    for bad in [
        Exploration {
            noise: Some(NOISE),
            ..ex
        },
        Exploration {
            sample_turns: 1,
            ..ex
        },
        Exploration {
            full_probability: 0.25,
            ..ex
        },
        Exploration {
            fast_simulations: 50,
            ..ex
        },
        Exploration {
            gumbel: true,
            ..Exploration::standard()
        },
        Exploration::gumbel(1),
    ] {
        assert!(bad.validate().is_err(), "{bad:?}");
        assert!(run_with(0..1, no_trades(), &bad, 8, &heuristic, 1).is_err());
    }
}

#[test]
fn puct_self_play_records_no_target() {
    let g = play_with(2, trading(), &small_explore(), 8, &heuristic);
    assert!(g.decisions.iter().all(|d| d.target.is_none()));
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

/// A 78 -> 16 -> 4 opening model with small weights drawn from `seed` (as tests/opening_model.rs
/// builds them): its spots score apart.
fn opening_model(seed: u64) -> OpeningModel {
    let mut rng = Rng::new(seed);
    let mut layers = Vec::new();
    for (cols, rows) in [(OPENING_FEATURES, 16), (16, NUM_PLAYERS)] {
        let mut x = || (rng.next_u64() >> 40) as f32 / (1u64 << 24) as f32 * 0.2 - 0.1;
        let w = (0..rows * cols).map(|_| x()).collect();
        let b = (0..rows).map(|_| x()).collect();
        layers.push(OpeningLayer { rows, cols, w, b });
    }
    OpeningModel { layers }
}

/// Every weight 0: every spot scores 1/4, a tie broken by engine order.
fn tied_model() -> OpeningModel {
    OpeningModel {
        layers: vec![OpeningLayer {
            rows: NUM_PLAYERS,
            cols: OPENING_FEATURES,
            w: vec![0.0; NUM_PLAYERS * OPENING_FEATURES],
            b: vec![0.0; NUM_PLAYERS],
        }],
    }
}

/// The model's settlements in `s` (the actor's view), best first, equal scores in engine order.
fn ranked_spots(model: &OpeningModel, s: &State) -> Vec<Action> {
    let legal = s.legal_actions();
    let mut scored = settlement_scores(model, &s.observation(s.current_actor()), &legal);
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    scored.into_iter().map(|(a, _)| a).collect()
}

/// With an opening model (spec 3d-1b Section 2) self-play's setup settlements are the model's:
/// none is searched or recorded, each is one of its OPENING_TOP best spots, about four in five
/// are its best one, and the setup roads are still searched and recorded.
#[test]
fn an_opening_model_picks_every_setup_settlement_from_its_top_spots_unrecorded() {
    assert_eq!((OPENING_EPSILON, OPENING_TOP), (0.25, 5));
    for (what, model) in [("random", opening_model(7)), ("tied", tied_model())] {
        let (mut picks, mut off_best, mut road_decisions) = (0, 0, 0);
        for seed in 0..6 {
            let g = play_with_opening(
                seed,
                trading(),
                &small_explore(),
                8,
                &heuristic,
                Some(&model),
            );
            assert_eq!(g.belief_resets, 0, "{what} seed {seed}");
            let mut s = State::new(seed, trading());
            let mut decisions = g.decisions.iter().peekable();
            for (i, &a) in g.actions.iter().enumerate() {
                let decision = decisions.next_if(|d| d.index as usize == i);
                match s.phase {
                    Phase::SetupSettlement => {
                        assert!(
                            decision.is_none(),
                            "{what} seed {seed} move {i}: a setup settlement was recorded"
                        );
                        let ranked = ranked_spots(&model, &s);
                        let top = &ranked[..ranked.len().min(OPENING_TOP)];
                        assert!(
                            top.contains(&a),
                            "{what} seed {seed} move {i}: {a:?} not in {top:?}"
                        );
                        if what == "tied" {
                            // Ties go to engine order: the first five legal settlements.
                            assert!(
                                s.legal_actions()[..OPENING_TOP].contains(&a),
                                "seed {seed} move {i}"
                            );
                        }
                        picks += 1;
                        off_best += (a != ranked[0]) as u32;
                    }
                    Phase::SetupRoad { .. } => road_decisions += decision.is_some() as u32,
                    _ => {}
                }
                s.apply(a);
            }
        }
        assert_eq!(picks, 6 * 8, "{what}");
        // 1 - ε + ε / K = 0.8 of the 48 picks are expected on the best spot.
        assert!(
            off_best > 0 && off_best < picks / 2,
            "{what}: {off_best} of {picks} picks off the best spot"
        );
        assert!(
            road_decisions > 0,
            "{what}: setup roads are searched and recorded"
        );
    }
}

#[test]
fn opening_model_self_play_is_deterministic_and_thread_independent() {
    let model = opening_model(7);
    let ex = small_explore();
    let a = run_with_opening(0..4, trading(), &ex, 8, &heuristic, 1, Some(&model)).unwrap();
    assert_eq!(
        a,
        run_with_opening(0..4, trading(), &ex, 8, &heuristic, 3, Some(&model)).unwrap()
    );
    assert_eq!(
        a[2],
        play_with_opening(2, trading(), &ex, 8, &heuristic, Some(&model))
    );
    let plain = run_with(0..4, trading(), &ex, 8, &heuristic, 1).unwrap();
    assert!(
        a.iter()
            .zip(&plain)
            .any(|(x, y)| x.actions[..16] != y.actions[..16]),
        "the model changes the openings"
    );
}

#[test]
fn opening_model_self_play_refuses_gumbel_self_play() {
    let model = opening_model(7);
    let ex = Exploration::gumbel(8);
    let e = run_with_opening(0..1, no_trades(), &ex, 8, &heuristic, 1, Some(&model)).unwrap_err();
    assert!(e.contains("Gumbel"), "{e}");
    assert!(run_with_opening(0..1, no_trades(), &ex, 8, &heuristic, 1, None).is_ok());
}

/// Without a model `play_with_opening` is `play_with`: the 3a and P2 digests hold through it.
#[test]
fn self_play_without_an_opening_model_plays_the_games_it_played_before_3d1b() {
    assert_eq!(
        digest(&play_with_opening(
            2,
            trading(),
            &small_explore(),
            8,
            &heuristic,
            None
        )),
        0x94f447a9c3baf5b2
    );
    assert_eq!(
        digest(&play_with_opening(
            3,
            no_trades(),
            &Exploration::fixed(10),
            8,
            &heuristic,
            None
        )),
        0x517b49ad6e41ec1e
    );
}

/// A fresh nets directory holding `tiny.bin` (random weights at this build's layout, width 8, 1
/// round), as bots/tests/netbot.rs makes them.
fn nets_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "settler-selfplay-nets-{}-{tag}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    Weights::random(
        Arch {
            width: 8,
            rounds: 1,
        },
        1,
    )
    .save(&dir.join("tiny.bin"))
    .unwrap();
    dir
}

/// These tests' pool: a net and an `ismcts` bot as the past half, the three styles.
const PAST: [&str; 2] = ["net:tiny@6+t", "ismcts@5+t"];
const STYLE_BOTS: [&str; 3] = ["style:roads@6+t", "style:army@6+t", "style:production@6+t"];
const NONE: [&str; 0] = [];

fn league(dir: &Path) -> League {
    League::new(&PAST, &STYLE_BOTS, dir).unwrap()
}

/// 3d-3 (spec Section 2): about half the games are league games. A league game seats two learner
/// bots and two pool bots, one past net and one style, at drawn seats; only the learner seats'
/// decisions are recorded; a plain game is exactly the game self-play without a league plays.
#[test]
fn a_league_game_seats_two_learners_a_past_net_and_a_style_and_records_only_the_learners() {
    assert_eq!((LEAGUE_PROBABILITY, LEAGUE_SEATS), (0.5, 2));
    let dir = nets_dir("seats");
    let pool = league(&dir);
    assert_eq!((pool.nets().len(), pool.styles().len()), (2, 3));
    let ex = small_explore();
    let games =
        run_with_league(0..40, trading(), &ex, 8, &heuristic, 4, None, Some(&pool)).unwrap();
    let (mut league_games, mut pool_seats) = (0, [0u32; NUM_PLAYERS]);
    for g in &games {
        assert_eq!(g.belief_resets, 0, "seed {}", g.seed);
        let Some(seats) = &g.league else {
            assert_eq!(
                *g,
                play_with(g.seed, trading(), &ex, 8, &heuristic),
                "seed {}",
                g.seed
            );
            continue;
        };
        league_games += 1;
        let names: Vec<&str> = seats.iter().flatten().map(String::as_str).collect();
        assert_eq!(names.len(), LEAGUE_SEATS, "seed {}: {seats:?}", g.seed);
        assert_eq!(
            names.iter().filter(|n| PAST.contains(*n)).count(),
            1,
            "seed {}: {seats:?}",
            g.seed
        );
        assert_eq!(
            names.iter().filter(|n| STYLE_BOTS.contains(*n)).count(),
            1,
            "seed {}: {seats:?}",
            g.seed
        );
        for (p, seat) in seats.iter().enumerate() {
            let recorded = g.decisions.iter().any(|d| d.actor as usize == p);
            assert_eq!(
                recorded,
                seat.is_none(),
                "seed {} seat {p}: {seats:?}",
                g.seed
            );
            pool_seats[p] += seat.is_some() as u32;
        }
        let mut s = State::new(g.seed, trading());
        for &a in &g.actions {
            assert!(s.legal_actions().contains(&a), "seed {}: {a:?}", g.seed);
            s.apply(a);
        }
        assert_eq!((s.winner(), s.turn), (g.winner, g.turns), "seed {}", g.seed);
    }
    // 40 games at p = 0.5: 20 expected, standard deviation 3.2.
    assert!(
        (10..=30).contains(&league_games),
        "{league_games} league games of 40"
    );
    assert!(
        pool_seats.iter().all(|&n| n > 0),
        "pool seats {pool_seats:?}"
    );
}

#[test]
fn league_self_play_is_deterministic_and_thread_independent() {
    let dir = nets_dir("threads");
    let pool = league(&dir);
    let ex = small_explore();
    let first = (0..).find(|&s| pool.draw(s).is_some()).unwrap();
    let seeds = first..first + 4;
    let a = run_with_league(
        seeds.clone(),
        trading(),
        &ex,
        8,
        &heuristic,
        1,
        None,
        Some(&pool),
    )
    .unwrap();
    assert_eq!(
        a,
        run_with_league(
            seeds.clone(),
            trading(),
            &ex,
            8,
            &heuristic,
            3,
            None,
            Some(&pool)
        )
        .unwrap()
    );
    assert_eq!(
        a[0],
        play_with_league(first, trading(), &ex, 8, &heuristic, None, Some(&pool))
    );
    assert!(a[0].league.is_some());
    // The seats depend on the seed alone: the same with an opening model, and in another league
    // built from the same names.
    let model = opening_model(7);
    let b = run_with_league(
        seeds.clone(),
        trading(),
        &ex,
        8,
        &heuristic,
        2,
        Some(&model),
        Some(&pool),
    )
    .unwrap();
    assert_eq!(
        b,
        run_with_league(
            seeds,
            trading(),
            &ex,
            8,
            &heuristic,
            1,
            Some(&model),
            Some(&pool)
        )
        .unwrap()
    );
    assert_eq!(
        a.iter().map(|g| &g.league).collect::<Vec<_>>(),
        b.iter().map(|g| &g.league).collect::<Vec<_>>()
    );
    assert_eq!(pool.draw(first), league(&dir).draw(first));
}

#[test]
fn a_league_needs_both_halves_and_known_bots_and_refuses_gumbel_self_play() {
    let dir = nets_dir("bad");
    for (nets, styles, message) in [
        (&NONE[..], &STYLE_BOTS[..], "at least one past net"),
        (&PAST[..], &NONE[..], "at least one style"),
        (&["net:missing@6"][..], &STYLE_BOTS[..], "missing"),
        (&PAST[..], &["style:tempo@6"][..], "unknown bot"),
        (&PAST[..], &["style:roads@6+o"][..], "unknown bot"),
    ] {
        match League::new(nets, styles, &dir) {
            Ok(_) => panic!("accepted {nets:?} {styles:?}"),
            Err(e) => assert!(e.contains(message), "{e}"),
        }
    }
    let pool = league(&dir);
    let gumbel = Exploration::gumbel(8);
    let e = run_with_league(
        0..1,
        no_trades(),
        &gumbel,
        8,
        &heuristic,
        1,
        None,
        Some(&pool),
    )
    .unwrap_err();
    assert!(e.contains("Gumbel"), "{e}");
    assert!(run_with_league(0..1, no_trades(), &gumbel, 8, &heuristic, 1, None, None).is_ok());
}

/// Without a league `play_with_league` is `play_with_opening`: the 3a and P2 digests hold
/// through it.
#[test]
fn self_play_without_a_league_plays_the_games_it_played_before_3d3() {
    let g = play_with_league(2, trading(), &small_explore(), 8, &heuristic, None, None);
    assert_eq!(digest(&g), 0x94f447a9c3baf5b2);
    assert_eq!(g.league, None);
    assert_eq!(
        digest(&play_with_league(
            3,
            no_trades(),
            &Exploration::fixed(10),
            8,
            &heuristic,
            None,
            None
        )),
        0x517b49ad6e41ec1e
    );
}
