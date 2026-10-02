mod common;

use common::*;
use settler_engine::legal::legal_actions;
use settler_engine::rng::Rng;
use settler_engine::*;
use settler_net::data::{Batch, RecordedDecision, RecordedGame, TrainingSet, MAX_SAMPLE};
use settler_net::encode::{encode_state, HIDDEN_LEN, PUBLIC_LEN};

/// A finished random game recorded with a decision at every 5th move: random visit counts (some
/// all zero), alternating full and fast searches.
fn recorded(seed: u64, config: GameConfig) -> RecordedGame {
    let mut s = State::new(seed, config);
    let mut rng = Rng::new(seed ^ 0xDA7A);
    let (mut actions, mut decisions, mut buf) = (Vec::new(), Vec::new(), Vec::new());
    while !s.is_over() {
        legal_actions(&s, &mut buf);
        if actions.len() % 5 == 0 {
            decisions.push(RecordedDecision {
                index: actions.len() as u32,
                actor: s.current_actor(),
                visits: buf.iter().map(|_| rng.below(4)).collect(),
                full_search: decisions.len() % 2 == 0,
            });
        }
        let a = buf[rng.below(buf.len() as u32) as usize];
        actions.push(a);
        s.apply(a);
    }
    RecordedGame {
        seed,
        config,
        actions,
        decisions,
        winner: s.winner(),
        vp: std::array::from_fn(|p| s.total_vp(p)),
    }
}

fn decided() -> GameConfig {
    GameConfig {
        vp_to_win: 3,
        ..no_trades()
    }
}

/// Trades on (offers among the legal moves) and a turn cap that ends in a draw.
fn drawn() -> GameConfig {
    GameConfig {
        max_turns: 30,
        ..GameConfig::default()
    }
}

#[test]
fn a_position_equals_the_encoder_on_the_replayed_state() {
    let games = vec![recorded(1, decided()), recorded(2, drawn())];
    assert!(games[0].winner.is_some() && games[1].winner.is_none());
    let set = TrainingSet::new(games.clone()).unwrap();
    assert_eq!(set.games(), 2);
    assert_eq!(
        set.decisions(),
        games.iter().map(|g| g.decisions.len()).sum::<usize>()
    );
    let mut offers_masked = 0;
    for (gi, g) in games.iter().enumerate() {
        let mut s = State::new(g.seed, g.config);
        let mut next = 0;
        for (i, &a) in g.actions.iter().enumerate() {
            if next < g.decisions.len() && g.decisions[next].index as usize == i {
                let d = &g.decisions[next];
                let mut b = Batch::default();
                set.position(gi, next, &mut b);
                let (mut public, mut hidden) = (vec![0.0; PUBLIC_LEN], vec![0.0; HIDDEN_LEN]);
                encode_state(&s, d.actor, &mut public, &mut hidden);
                assert_eq!((b.n, &b.public, &b.hidden), (1, &public, &hidden));
                let legal = s.legal_actions();
                offers_masked +=
                    legal.iter().any(|a| matches!(a, Action::OfferTrade { .. })) as u32;
                let searchable: Vec<(Action, u32)> = legal
                    .iter()
                    .copied()
                    .zip(d.visits.iter().copied())
                    .filter(|(a, _)| !matches!(a, Action::OfferTrade { .. }))
                    .collect();
                let total: u32 = searchable.iter().map(|x| x.1).sum();
                let (mut mask, mut policy) = (
                    vec![0u8; ACTION_SPACE_SIZE],
                    vec![0.0f32; ACTION_SPACE_SIZE],
                );
                for &(a, v) in &searchable {
                    mask[a.encode() as usize] = 1;
                    policy[a.encode() as usize] = if total > 0 {
                        v as f32 / total as f32
                    } else {
                        1.0 / searchable.len() as f32
                    };
                }
                assert_eq!(b.mask, mask);
                assert_eq!(b.policy, policy);
                assert_eq!(b.full, vec![(d.full_search && total > 0) as u8]);
                let me = d.actor as usize;
                let outcome: Vec<f32> = (0..4)
                    .map(|r| match g.winner {
                        Some(w) => (w as usize == (me + r) % 4) as u8 as f32,
                        None => 0.25,
                    })
                    .collect();
                assert_eq!(b.outcome, outcome);
                let vp: Vec<f32> = (0..4).map(|r| g.vp[(me + r) % 4] as f32 / 10.0).collect();
                assert_eq!(b.vp, vp);
                next += 1;
            }
            s.apply(a);
        }
    }
    assert!(
        offers_masked > 0,
        "offers were legal somewhere and stayed out of the mask"
    );
}

#[test]
fn sampling_is_seeded_and_thread_independent() {
    let set = TrainingSet::new(vec![recorded(1, decided()), recorded(3, drawn())]).unwrap();
    let a = set.sample(64, 9, 1).unwrap();
    assert_eq!(a, set.sample(64, 9, 4).unwrap());
    assert_ne!(a, set.sample(64, 10, 1).unwrap());
    assert_eq!(a.n, 64);
    assert_eq!(
        (a.public.len(), a.hidden.len()),
        (64 * PUBLIC_LEN, 64 * HIDDEN_LEN)
    );
    assert_eq!(
        (a.mask.len(), a.policy.len()),
        (64 * ACTION_SPACE_SIZE, 64 * ACTION_SPACE_SIZE)
    );
    assert_eq!((a.full.len(), a.outcome.len(), a.vp.len()), (64, 256, 256));
    let picks = [(0, 0), (1, 2), (0, 1)];
    let mut one_by_one = Batch::default();
    for &(g, d) in &picks {
        set.position(g, d, &mut one_by_one);
    }
    assert_eq!(set.batch(&picks, 2), one_by_one);
}

#[test]
fn games_without_decisions_are_never_sampled_and_an_empty_set_is_an_error() {
    let mut empty = recorded(4, decided());
    empty.decisions.clear();
    let set = TrainingSet::new(vec![empty.clone(), recorded(1, decided()), empty.clone()]).unwrap();
    assert_eq!(set.decisions(), set.game_decisions(1));
    assert_eq!(set.locate(0), (1, 0));
    let mut known = Vec::new();
    for d in 0..set.game_decisions(1) {
        let mut b = Batch::default();
        set.position(1, d, &mut b);
        known.push(b.public);
    }
    let b = set.sample(200, 1, 2).unwrap();
    for row in b.public.chunks(PUBLIC_LEN) {
        assert!(
            known.iter().any(|k| k[..] == *row),
            "a sampled row came from game 1"
        );
    }
    let e = TrainingSet::new(vec![empty])
        .unwrap()
        .sample(1, 0, 1)
        .unwrap_err();
    assert!(e.contains("no positions"), "{e}");
    assert!(TrainingSet::new(vec![]).unwrap().sample(1, 0, 1).is_err());
}

fn err(g: RecordedGame) -> String {
    TrainingSet::new(vec![g]).err().unwrap()
}

#[test]
fn a_record_that_does_not_replay_is_an_error() {
    let g = recorded(1, decided());
    let mut actor = g.clone();
    actor.decisions[1].actor = (actor.decisions[1].actor + 1) % 4;
    assert!(err(actor).contains("actor"));
    let mut visits = g.clone();
    visits.decisions[0].visits.push(1);
    assert!(err(visits).contains("visit counts"));
    let mut order = g.clone();
    order.decisions.swap(0, 1);
    assert!(err(order).contains("out of order"));
    let mut winner = g.clone();
    winner.winner = None;
    assert!(err(winner).contains("winner"));
    let mut vp = g.clone();
    vp.vp[0] += 1;
    assert!(err(vp).contains("vp"));
    let mut illegal = g.clone();
    illegal.actions[0] = Action::EndTurn;
    assert!(err(illegal).contains("illegal"));
    let mut short = g.clone();
    short.actions.pop();
    short
        .decisions
        .retain(|d| (d.index as usize) < short.actions.len());
    assert!(err(short).contains("does not finish"));
    let mut long = g.clone();
    long.actions.push(Action::EndTurn);
    assert!(err(long).contains("after the game ended"));
    let e = err(RecordedGame { seed: 9, ..g });
    assert!(e.starts_with("game 0 (seed 9)"), "{e}");
}

/// F2: records join a set a file at a time; a bad batch is reported by its index in the whole set
/// and leaves the set as it was.
#[test]
fn records_can_be_added_a_batch_at_a_time() {
    let (a, b, c) = (
        recorded(1, decided()),
        recorded(2, drawn()),
        recorded(3, decided()),
    );
    let whole = TrainingSet::new(vec![a.clone(), b.clone(), c.clone()]).unwrap();
    let mut parts = TrainingSet::new(vec![a.clone()]).unwrap();
    parts.extend(vec![b.clone(), c.clone()]).unwrap();
    let sizes = |t: &TrainingSet| (t.games(), t.decisions(), t.full_decisions());
    assert_eq!(sizes(&parts), sizes(&whole));
    let sample = whole.sample(64, 5, 2).unwrap();
    assert_eq!(parts.sample(64, 5, 2).unwrap(), sample);
    let mut bad = c.clone();
    bad.winner = None;
    let e = parts.extend(vec![a.clone(), bad]).unwrap_err();
    assert!(e.starts_with("game 4 (seed 3)"), "{e}");
    assert_eq!(sizes(&parts), sizes(&whole));
    assert_eq!(parts.sample(64, 5, 2).unwrap(), sample);
    let mut empty = TrainingSet::default();
    assert_eq!((empty.games(), empty.decisions()), (0, 0));
    assert!(empty.sample(1, 0, 1).is_err());
    empty.extend(vec![]).unwrap();
    empty.extend(vec![a]).unwrap();
    assert_eq!(empty.games(), 1);
}

#[test]
fn a_sample_larger_than_the_cap_is_an_error() {
    let set = TrainingSet::new(vec![recorded(1, decided())]).unwrap();
    let e = set.sample(MAX_SAMPLE + 1, 0, 1).unwrap_err();
    assert!(e.contains("at most"), "{e}");
    assert_eq!(
        set.sample(3, 0, 10_000).unwrap().n,
        3,
        "threads are capped, not refused"
    );
}

#[test]
fn visit_counts_near_the_u32_limit_normalise_without_overflow() {
    let mut g = recorded(1, decided());
    for d in &mut g.decisions {
        d.visits.iter_mut().for_each(|v| *v = u32::MAX);
    }
    let set = TrainingSet::new(vec![g]).unwrap();
    let mut b = Batch::default();
    set.position(0, 0, &mut b);
    let total: f32 = b.policy.iter().sum();
    assert!((total - 1.0).abs() < 1e-5, "{total}");
    assert!(b.policy.iter().all(|&p| p.is_finite() && p >= 0.0));
}
