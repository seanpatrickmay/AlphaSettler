use settler_bots::arena::{bot_seed, play_game, run_match, GameRecord, MAX_SEEDS};
use settler_bots::{make_bot, Bot};
use settler_engine::{Action, GameConfig, Observation};
use settler_bots::greedy::GreedyBot;
use settler_engine::{Event, Game, PlayerId};
use std::sync::{Arc, Mutex};

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

/// GreedyBot that records the events it is shown and appends its moves to a shared list.
struct Recorder {
    inner: GreedyBot,
    viewer: Option<PlayerId>,
    seen: Arc<Mutex<Vec<Event>>>,
    moves: Arc<Mutex<Vec<Action>>>,
}

impl Bot for Recorder {
    fn name(&self) -> &'static str {
        "recorder"
    }

    fn observe(&mut self, viewer: PlayerId, events: &[Event]) {
        assert!(self.viewer.map_or(true, |v| v == viewer), "a seat's viewer never changes");
        self.viewer = Some(viewer);
        self.seen.lock().unwrap().extend_from_slice(events);
    }

    fn act(&mut self, obs: &Observation, legal: &[Action]) -> Action {
        let a = self.inner.act(obs, legal);
        self.moves.lock().unwrap().push(a);
        a
    }
}

#[test]
fn each_bot_is_shown_its_own_redacted_log_before_it_moves() {
    for seed in 0..5 {
        let moves = Arc::new(Mutex::new(Vec::new()));
        let seen: Vec<Arc<Mutex<Vec<Event>>>> = (0..4).map(|_| Arc::new(Mutex::new(Vec::new()))).collect();
        let mut bots: Vec<Box<dyn Bot>> = (0..4)
            .map(|p| {
                Box::new(Recorder {
                    inner: GreedyBot::new(0),
                    viewer: None,
                    seen: seen[p].clone(),
                    moves: moves.clone(),
                }) as Box<dyn Bot>
            })
            .collect();
        play_game(seed, GameConfig::default(), &mut bots);

        // Replay the same moves with the logging Game and note each seat's log length at its last move.
        let mut g = Game::new(seed, GameConfig::default());
        let mut at_last_move = [0usize; 4];
        for &a in moves.lock().unwrap().iter() {
            let actor = g.state().current_actor() as usize;
            at_last_move[actor] = g.log().len();
            g.apply(a).unwrap();
        }
        assert!(g.state().is_over());
        for p in 0..4 {
            let shown = seen[p].lock().unwrap();
            assert_eq!(shown.len(), at_last_move[p], "seat {p} saw every event up to its last move");
            assert_eq!(&shown[..], &g.log_for(p as u8)[..shown.len()], "seat {p} saw its redacted log");
        }
    }
}

#[test]
fn split_seeds_covers_the_range_in_order() {
    use settler_bots::arena::split_seeds;
    let parts = split_seeds(10..33, 4);
    assert_eq!(parts.first().unwrap().start, 10);
    assert_eq!(parts.last().unwrap().end, 33);
    assert!(parts.windows(2).all(|w| w[0].end == w[1].start));
    assert!(split_seeds(5..5, 3).is_empty());
    assert_eq!(split_seeds(0..3, 100).len(), 3);
}
