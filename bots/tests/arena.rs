use settler_bots::arena::{bot_seed, play_game, run_match, GameRecord, MAX_SEEDS};
use settler_bots::{make_bot, Bot};
use settler_engine::{Action, GameConfig, Observation};

fn per_seed_rates(records: &[GameRecord]) -> Vec<f64> {
    records
        .chunks(4)
        .map(|g| {
            g.iter()
                .filter(|r| r.winner == Some(r.candidate_seat))
                .count() as f64
                / 4.0
        })
        .collect()
}

fn z_vs_quarter(rates: &[f64]) -> f64 {
    let n = rates.len() as f64;
    let mean = rates.iter().sum::<f64>() / n;
    let var = rates.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
    (mean - 0.25) / (var / n).sqrt()
}

#[test]
fn every_seed_is_played_in_all_four_seats() {
    let recs = run_match("random", "random", 10..15, GameConfig::default(), 2).unwrap();
    assert_eq!(recs.len(), 20);
    for (i, r) in recs.iter().enumerate() {
        assert_eq!(r.seed, 10 + (i / 4) as u64);
        assert_eq!(r.candidate_seat, (i % 4) as u8);
        assert!(r.turns > 0 && r.actions > 0);
        if let Some(w) = r.winner {
            assert!(r.vp[w as usize] >= 10);
        }
    }
}

#[test]
fn results_do_not_depend_on_thread_count() {
    let a = run_match("greedy", "random", 0..12, GameConfig::default(), 1).unwrap();
    let b = run_match("greedy", "random", 0..12, GameConfig::default(), 5).unwrap();
    assert_eq!(a, b);
    // An absurd thread count is capped (seed count, 4x cores), not spawned.
    let c = run_match("greedy", "random", 0..12, GameConfig::default(), usize::MAX).unwrap();
    assert_eq!(a, c);
}

#[test]
fn rotations_of_a_seed_are_the_same_game_when_bots_are_identical() {
    // CRN: each seat's bot stream depends on (seed, seat) alone. With candidate == baseline,
    // rotating the candidate changes no bot anywhere, so all four games of a seed are one game.
    let recs = run_match("random", "random", 0..30, GameConfig::default(), 4).unwrap();
    assert_eq!(recs.len(), 120);
    let game = |r: &GameRecord| (r.winner, r.vp, r.turns, r.actions);
    for g in recs.chunks(4) {
        let seats: Vec<u8> = g.iter().map(|r| r.candidate_seat).collect();
        assert_eq!(seats, [0, 1, 2, 3]);
        for r in g {
            assert_eq!(r.seed, g[0].seed);
            assert_eq!(
                game(r),
                game(&g[0]),
                "seed {} seat {}",
                r.seed,
                r.candidate_seat
            );
        }
    }
}

#[test]
fn bot_seeds_are_distinct_over_seed_and_seat() {
    let seeds: std::collections::HashSet<u64> = (0..50)
        .flat_map(|seed| (0..4).map(move |seat| bot_seed(seed, seat)))
        .collect();
    assert_eq!(seeds.len(), 200);
}

#[test]
fn unknown_bot_and_bad_config_are_errors() {
    let e = run_match("nope", "random", 0..1, GameConfig::default(), 1).unwrap_err();
    assert!(e.contains("unknown bot"), "{e}");
    let bad = GameConfig {
        max_trade_cards: 3,
        ..GameConfig::default()
    };
    assert!(run_match("random", "random", 0..1, bad, 1).is_err());
    assert!(
        run_match("random", "random", 5..5, GameConfig::default(), 1)
            .unwrap()
            .is_empty()
    );
}

#[test]
fn a_range_over_the_seed_cap_is_an_error() {
    for seeds in [0..u64::MAX, 7..u64::MAX, 0..MAX_SEEDS + 1] {
        let e = run_match("random", "random", seeds, GameConfig::default(), 4).unwrap_err();
        assert!(e.contains(&format!("at most {MAX_SEEDS} seeds")), "{e}");
    }
}

#[test]
fn a_range_ending_at_the_top_of_u64_splits_without_overflow() {
    let end = u64::MAX;
    let recs = run_match("random", "random", end - 9..end, GameConfig::default(), 8).unwrap();
    let seeds: Vec<u64> = recs.iter().step_by(4).map(|r| r.seed).collect();
    assert_eq!(seeds, (end - 9..end).collect::<Vec<_>>());
}

struct Broken;
impl Bot for Broken {
    fn name(&self) -> &'static str {
        "broken"
    }
    fn act(&mut self, _obs: &Observation, _legal: &[Action]) -> Action {
        Action::BuildCity(53)
    }
}

#[test]
#[should_panic(expected = "bot broken chose illegal action BuildCity(53) in phase SetupSettlement")]
fn illegal_bot_action_panics_with_bot_name() {
    let mut bots: Vec<Box<dyn Bot>> = vec![
        Box::new(Broken),
        make_bot("random", 1).unwrap(),
        make_bot("random", 2).unwrap(),
        make_bot("random", 3).unwrap(),
    ];
    play_game(0, GameConfig::default(), &mut bots);
}

#[test]
fn greedy_beats_random_decisively() {
    let recs = run_match("greedy", "random", 0..200, GameConfig::default(), 8).unwrap();
    let z = z_vs_quarter(&per_seed_rates(&recs));
    assert!(z > 3.0, "z = {z}");
}
