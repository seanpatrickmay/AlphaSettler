mod common;

use common::*;
use settler_engine::legal::legal_actions;
use settler_engine::rng::Rng;
use settler_engine::*;
use settler_net::data::{Batch, RecordedDecision, RecordedGame, TrainingSet, MAX_SAMPLE};
use settler_net::encode::{encode_state, HIDDEN_LEN, PUBLIC_LEN};
use settler_net::evaluator::masked_prior;
use settler_net::policy::policy_id;
use settler_search::search_actions;

/// A finished random game recorded with a decision at every 5th move: random visit counts (some
/// all zero), alternating full and fast searches.
fn recorded(seed: u64, config: GameConfig) -> RecordedGame {
    recorded_every(seed, config, 5)
}

/// `recorded` with a decision at every `every`-th move.
fn recorded_every(seed: u64, config: GameConfig, every: usize) -> RecordedGame {
    let mut s = State::new(seed, config);
    let mut rng = Rng::new(seed ^ 0xDA7A);
    let (mut actions, mut decisions, mut buf) = (Vec::new(), Vec::new(), Vec::new());
    while !s.is_over() {
        legal_actions(&s, &mut buf);
        if actions.len() % every == 0 {
            decisions.push(RecordedDecision {
                index: actions.len() as u32,
                actor: s.current_actor(),
                visits: buf.iter().map(|_| rng.below(4)).collect(),
                full_search: decisions.len() % 2 == 0,
                target: None,
                q: None,
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

/// `recorded`, with q on every decision: 0.7 for seat 1 and 0.1 for the others.
fn with_q(g: &RecordedGame) -> RecordedGame {
    let mut g = g.clone();
    for d in &mut g.decisions {
        d.q = Some([0.1, 0.7, 0.1, 0.1]);
    }
    g
}

#[test]
fn q_is_written_seat_relative_to_the_actor_and_flagged() {
    let games = vec![with_q(&recorded(1, decided())), recorded(2, decided())];
    let set = TrainingSet::new(games.clone()).unwrap();
    for (game, has) in [(0usize, 1u8), (1, 0)] {
        for (k, d) in games[game].decisions.iter().enumerate() {
            let mut b = Batch::default();
            set.position(game, k, &mut b);
            assert_eq!(b.has_q, vec![has]);
            if has == 1 {
                let r = (1 + NUM_PLAYERS - d.actor as usize) % NUM_PLAYERS; // seat 1 relative to the actor
                for i in 0..NUM_PLAYERS {
                    assert_eq!(b.q[i], if i == r { 0.7 } else { 0.1 });
                }
            } else {
                assert_eq!(b.q, vec![0.0; 4]);
            }
        }
    }
}

#[test]
fn a_bad_q_is_an_error_naming_the_decision() {
    for bad in [
        [0.5, 0.5, 0.5, 0.5],
        [f32::NAN, 0.5, 0.25, 0.25],
        [-0.5, 0.5, 0.5, 0.5],
    ] {
        let mut g = recorded(1, decided());
        g.decisions[0].q = Some(bad);
        let e = TrainingSet::new(vec![g]).unwrap_err();
        assert!(e.contains("move 0") && e.contains("q"), "{e}");
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
    let (mut offers_in_mask, mut offers_targeted) = (0, 0);
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
                let total: u32 = d.visits.iter().sum();
                let (mut mask, mut policy) = (
                    vec![0u8; ACTION_SPACE_SIZE],
                    vec![0.0f32; ACTION_SPACE_SIZE],
                );
                for (&a, &v) in legal.iter().zip(&d.visits) {
                    let k = policy_id(a, d.actor);
                    mask[k] = 1;
                    policy[k] = if total > 0 {
                        v as f32 / total as f32
                    } else {
                        1.0 / legal.len() as f32
                    };
                    if matches!(a, Action::OfferTrade { .. }) {
                        offers_in_mask += b.mask[k] as u32;
                        offers_targeted += (b.policy[k] > 0.0) as u32;
                    }
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
        offers_in_mask > 0 && offers_targeted > 0,
        "offers sit in the mask ({offers_in_mask}) and in the targets ({offers_targeted})"
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

/// `g` with an improved policy on every decision ((j + 1) / Σ over its legal moves), every visit
/// count zero and every search marked fast: only the targets make policy targets.
fn with_targets(g: &RecordedGame) -> RecordedGame {
    let mut g = g.clone();
    for d in &mut g.decisions {
        let n = d.visits.len();
        let total = (n * (n + 1) / 2) as f32;
        d.target = Some((1..=n).map(|j| j as f32 / total).collect());
        d.visits.iter_mut().for_each(|v| *v = 0);
        d.full_search = false;
    }
    g
}

#[test]
fn a_decision_with_a_target_trains_on_it_whatever_its_visits() {
    let g = with_targets(&recorded(1, decided()));
    let set = TrainingSet::new(vec![g.clone()]).unwrap();
    assert_eq!(set.policy_decisions(), g.decisions.len());
    assert_eq!(
        set.full_decisions(),
        0,
        "full_decisions still counts the flag"
    );
    let mut s = State::new(g.seed, g.config);
    let mut next = 0;
    for (i, &a) in g.actions.iter().enumerate() {
        if next < g.decisions.len() && g.decisions[next].index as usize == i {
            let d = &g.decisions[next];
            let mut b = Batch::default();
            set.position(0, next, &mut b);
            let t = d.target.as_ref().unwrap();
            let legal = s.legal_actions();
            for (j, &l) in legal.iter().enumerate() {
                assert_eq!(b.policy[policy_id(l, d.actor)], t[j], "move {i}");
            }
            let row: f64 = b.policy.iter().map(|&x| x as f64).sum();
            assert!((row - 1.0).abs() < 1e-5, "move {i}: {row}");
            assert_eq!(b.full, vec![1], "move {i}");
            next += 1;
        }
        s.apply(a);
    }
    let a = set.sample_full(64, 3, 2).unwrap();
    assert!(a.full.iter().all(|&f| f == 1));
}

#[test]
fn a_bad_target_is_an_error_naming_the_decision() {
    let g = with_targets(&recorded(1, decided()));
    let k = g
        .decisions
        .iter()
        .position(|d| d.visits.len() >= 2)
        .expect("a decision with a choice");
    let at = g.decisions[k].index;
    let bad = |edit: &dyn Fn(&mut Vec<f32>)| {
        let mut b = g.clone();
        edit(b.decisions[k].target.as_mut().unwrap());
        err(b)
    };
    for (e, want) in [
        (
            bad(&|t| {
                t.pop();
            }),
            "target entries",
        ),
        (bad(&|t| t.push(0.0)), "target entries"),
        (bad(&|t| t.iter_mut().for_each(|x| *x *= 0.9)), "sums to"),
        (bad(&|t| t[0] = f32::NAN), "target entry 0"),
        (
            bad(&|t| {
                t[1] += t[0] + 1.0;
                t[0] = -1.0;
            }),
            "target entry 0",
        ),
    ] {
        assert!(e.starts_with("game 0 (seed 1): "), "{e}");
        assert!(
            e.contains(&format!("move {at}: ")) && e.contains(want),
            "{e}"
        );
    }
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

/// Seat-relative ids in both places (spec Section 2). With finite logits only in the slots the
/// sampler's mask marks, `masked_prior` must give every searchable move its softmax share of the
/// logit at that move's `policy_id`: the evaluator reads exactly the slots the training targets
/// fill, at every decision of three games, steals by every seat included.
#[test]
fn the_sampler_and_the_evaluator_use_the_same_policy_ids() {
    let config = GameConfig {
        max_turns: 150,
        ..no_trades()
    };
    let games: Vec<RecordedGame> = (1..4).map(|seed| recorded_every(seed, config, 1)).collect();
    let set = TrainingSet::new(games.clone()).unwrap();
    let (mut checked, mut relative_steals) = (0, 0);
    let mut legal = Vec::new();
    for (gi, g) in games.iter().enumerate() {
        let mut s = State::new(g.seed, g.config);
        for (di, d) in g.decisions.iter().enumerate() {
            assert_eq!(d.index as usize, di, "a decision at every move");
            let mut b = Batch::default();
            set.position(gi, di, &mut b);
            search_actions(&s.legal_actions(), &mut legal);
            let slots: Vec<usize> = (0..ACTION_SPACE_SIZE).filter(|&k| b.mask[k] == 1).collect();
            assert_eq!(slots.len(), legal.len(), "one slot per searchable move");
            let logit = |k: usize| k as f32 / 100.0;
            let mut logits = vec![f32::NEG_INFINITY; ACTION_SPACE_SIZE];
            for &k in &slots {
                logits[k] = logit(k);
            }
            let z: f32 = slots.iter().map(|&k| logit(k).exp()).sum();
            let prior = masked_prior(&logits, &legal, d.actor);
            for (a, p) in legal.iter().zip(&prior) {
                let k = policy_id(*a, d.actor);
                assert!(slots.contains(&k), "{a:?} by seat {}", d.actor);
                assert!(
                    (p - logit(k).exp() / z).abs() < 1e-5,
                    "{a:?} by seat {}",
                    d.actor
                );
                relative_steals +=
                    (matches!(a, Action::StealFrom(_)) && k != a.encode() as usize) as u32;
            }
            checked += 1;
            s.apply(g.actions[di]);
        }
    }
    assert!(checked > 1000, "{checked} decisions");
    assert!(
        relative_steals > 0,
        "steals by seats other than 0 were checked"
    );
}

#[test]
fn sample_full_draws_only_policy_targets() {
    let set = TrainingSet::new(vec![recorded(1, decided()), recorded(3, drawn())]).unwrap();
    let mut every = Batch::default();
    for g in 0..set.games() {
        for d in 0..set.game_decisions(g) {
            set.position(g, d, &mut every);
        }
    }
    let marked = every.full.iter().filter(|&&f| f == 1).count();
    assert!(marked > 0 && marked < every.n);
    assert_eq!(set.policy_decisions(), marked);
    let a = set.sample_full(256, 9, 1).unwrap();
    assert_eq!(a, set.sample_full(256, 9, 4).unwrap());
    assert_ne!(a, set.sample_full(256, 10, 1).unwrap());
    assert_eq!(a.n, 256);
    assert!(
        a.full.iter().all(|&f| f == 1),
        "every row is a policy target"
    );
    let mut none = recorded(1, decided());
    for d in &mut none.decisions {
        d.full_search = false;
    }
    let e = TrainingSet::new(vec![none])
        .unwrap()
        .sample_full(1, 0, 1)
        .unwrap_err();
    assert!(e.contains("no full-search"), "{e}");
    let e = set.sample_full(MAX_SAMPLE + 1, 0, 1).unwrap_err();
    assert!(e.contains("at most"), "{e}");
}

/// In the trading game every legal move, offers and seat-relative confirmations included, has
/// one slot, and the evaluator's prior reads the same slots.
#[test]
fn trade_moves_sit_in_the_same_slots_for_the_sampler_and_the_evaluator() {
    let config = GameConfig {
        max_turns: 150,
        ..trading()
    };
    let games: Vec<RecordedGame> = (1..4).map(|seed| recorded_every(seed, config, 1)).collect();
    let set = TrainingSet::new(games.clone()).unwrap();
    let (mut offers, mut relative_confirms) = (0, 0);
    for (gi, g) in games.iter().enumerate() {
        let mut s = State::new(g.seed, g.config);
        for (di, d) in g.decisions.iter().enumerate() {
            let mut b = Batch::default();
            set.position(gi, di, &mut b);
            let legal = s.legal_actions();
            let slots: Vec<usize> = (0..ACTION_SPACE_SIZE).filter(|&k| b.mask[k] == 1).collect();
            assert_eq!(
                slots.len(),
                legal.len(),
                "one slot per legal move, offers included"
            );
            let prior = masked_prior(&vec![0.0; ACTION_SPACE_SIZE], &legal, d.actor);
            assert!(prior
                .iter()
                .all(|&p| (p - 1.0 / legal.len() as f32).abs() < 1e-6));
            for a in &legal {
                let k = policy_id(*a, d.actor);
                assert!(slots.contains(&k), "{a:?} by seat {}", d.actor);
                offers += matches!(a, Action::OfferTrade { .. }) as u32;
                relative_confirms +=
                    (matches!(a, Action::ConfirmTrade(_)) && k != a.encode() as usize) as u32;
            }
            s.apply(g.actions[di]);
        }
    }
    assert!(
        offers > 0 && relative_confirms > 0,
        "{offers} offers, {relative_confirms} seat-relative confirmations"
    );
}
