mod common;

use common::*;
use settler_engine::rng::Rng;
use settler_engine::*;
use settler_search::{Belief, HandTracker, DEFAULT_MAX_STATES};
use std::collections::HashMap;

#[test]
fn the_full_log_reproduces_every_hand() {
    let configs = (0..40)
        .map(|s| (s, no_trades()))
        .chain((40..48).map(|s| (s, GameConfig::default())));
    for (seed, config) in configs {
        let mut t = HandTracker::new();
        let mut rng = Rng::new(0);
        let mut seen = 0;
        random_game(seed, config, |g| {
            for e in &g.log()[seen..] {
                assert!(
                    t.step(e, &mut rng),
                    "seed {seed}: {e:?} rejected on the true history"
                );
            }
            seen = g.log().len();
            for p in 0..NUM_PLAYERS {
                assert_eq!(
                    t.hands[p],
                    g.state().players[p].hand,
                    "seed {seed} player {p}"
                );
            }
        });
    }
}

#[test]
fn every_state_matches_the_visible_counts_and_the_truth_stays_possible() {
    let mut largest = 0;
    for seed in 0..12 {
        let mut beliefs: Vec<Belief> = (0..4)
            .map(|p| Belief::new(p, DEFAULT_MAX_STATES, seed))
            .collect();
        let mut seen = [0usize; 4];
        random_game(seed, no_trades(), |g| {
            let true_hands: [Hand; 4] = std::array::from_fn(|q| g.state().players[q].hand);
            for p in 0..4u8 {
                let log = g.log_for(p);
                let b = &mut beliefs[p as usize];
                b.observe(&log[seen[p as usize]..]).unwrap();
                seen[p as usize] = log.len();
                b.check(&g.observation(p)).unwrap();
                assert!(
                    b.states().iter().any(|(t, _)| t.hands == true_hands),
                    "seed {seed} viewer {p}: the true hands were lost"
                );
                largest = largest.max(b.states().len());
            }
        });
        for b in &beliefs {
            assert_eq!(b.truncations(), 0, "seed {seed}");
        }
    }
    eprintln!("largest support seen across all games and viewers: {largest}");
    assert!(largest < DEFAULT_MAX_STATES);
}

/// Exact posterior over all four hands: branch on each hidden steal, weighted by the victim's hand.
fn exact(log: &[Event]) -> HashMap<[Hand; 4], f64> {
    let mut states = vec![(HandTracker::new(), 1.0f64)];
    let mut rng = Rng::new(0);
    for e in log {
        let mut next = Vec::new();
        for (t, w) in states {
            if let Event::Stole {
                thief,
                victim,
                resource: None,
            } = *e
            {
                let v = t.hands[victim as usize];
                let total = hand_total(&v) as f64;
                for r in Resource::ALL {
                    if v[r.index()] == 0 {
                        continue;
                    }
                    let mut u = t;
                    if u.step(
                        &Event::Stole {
                            thief,
                            victim,
                            resource: Some(r),
                        },
                        &mut rng,
                    ) {
                        next.push((u, w * v[r.index()] as f64 / total));
                    }
                }
            } else {
                let mut u = t;
                if u.step(e, &mut rng) {
                    next.push((u, w));
                }
            }
        }
        states = next;
    }
    let z: f64 = states.iter().map(|s| s.1).sum();
    let mut out = HashMap::new();
    for (t, w) in states {
        *out.entry(t.hands).or_insert(0.0) += w / z;
    }
    out
}

fn assert_matches_exact(log: &[Event], seed: u64) {
    let truth = exact(log);
    let mut b = Belief::new(0, DEFAULT_MAX_STATES, seed);
    b.observe(log).unwrap();
    assert_eq!(b.states().len(), truth.len(), "seed {seed}: support size");
    for (t, w) in b.states() {
        let p = truth
            .get(&t.hands)
            .unwrap_or_else(|| panic!("seed {seed}: {:?} is ruled out", t.hands));
        assert!(
            (w - p).abs() < 1e-9,
            "seed {seed}: {:?} exact {p} tracked {w}",
            t.hands
        );
    }
}

#[test]
fn hand_built_histories_match_the_exact_posterior() {
    let mut log = setup_events();
    log.extend([
        Event::Produced {
            player: 1,
            resources: [1, 0, 0, 0, 2],
        },
        Event::Produced {
            player: 3,
            resources: [0, 2, 0, 1, 0],
        },
        Event::Stole {
            thief: 2,
            victim: 1,
            resource: None,
        },
        Event::Stole {
            thief: 2,
            victim: 3,
            resource: None,
        },
        Event::Discarded {
            player: 2,
            resource: Resource::Ore,
        },
    ]);
    assert_matches_exact(&log, 1);
}

/// A random small history that is feasible for some true hands: production, hidden steals between
/// players 1-3, discards and paid roads of cards the (simulated) true holder has.
fn random_history(seed: u64) -> Vec<Event> {
    let mut rng = Rng::new(seed);
    let mut log = setup_events();
    let mut truth = HandTracker::new();
    for e in &log {
        assert!(truth.step(e, &mut rng));
    }
    let mut hidden = 0;
    while log.len() < 30 {
        let p = 1 + rng.below(3) as u8;
        let e = match rng.below(4) {
            0 => {
                let mut h = [0u8; 5];
                h[rng.below(5) as usize] += 1 + rng.below(2) as u8;
                Event::Produced {
                    player: p,
                    resources: h,
                }
            }
            1 if hidden < 4 => {
                let q = 1 + (p as u32 + rng.below(2)) % 3;
                if hand_total(&truth.hands[q as usize]) == 0 {
                    continue;
                }
                hidden += 1;
                let r = settler_engine::rules::roll::pick_card(&truth.hands[q as usize], &mut rng);
                let mut t = truth;
                assert!(t.step(
                    &Event::Stole {
                        thief: p,
                        victim: q as u8,
                        resource: Some(r)
                    },
                    &mut rng
                ));
                truth = t;
                log.push(Event::Stole {
                    thief: p,
                    victim: q as u8,
                    resource: None,
                });
                continue;
            }
            2 => {
                let h = truth.hands[p as usize];
                let Some(r) = Resource::ALL.into_iter().find(|r| h[r.index()] > 0) else {
                    continue;
                };
                Event::Discarded {
                    player: p,
                    resource: r,
                }
            }
            _ => {
                if !covers(&truth.hands[p as usize], &ROAD_COST) {
                    continue;
                }
                Event::BuiltRoad {
                    player: p,
                    edge: 40,
                }
            }
        };
        assert!(truth.step(&e, &mut rng));
        log.push(e);
    }
    log
}

#[test]
fn random_histories_match_the_exact_posterior() {
    for seed in 0..25 {
        assert_matches_exact(&random_history(seed), seed);
    }
}

#[test]
fn a_tight_cap_resamples_but_stays_inside_the_exact_support() {
    // History 3 has a single possible state; history 6 is the first with more than 8.
    let log = random_history(6);
    let truth = exact(&log);
    let mut b = Belief::new(0, 4, 9);
    b.observe(&log).unwrap();
    assert!(
        b.truncations() > 0,
        "history 6 has more than 4 possible joint hands"
    );
    assert!(b.states().len() <= 4);
    assert!((b.states().iter().map(|s| s.1).sum::<f64>() - 1.0).abs() < 1e-9);
    assert!(b.states().iter().all(|(t, _)| truth.contains_key(&t.hands)));
}

#[test]
fn an_impossible_history_is_an_error() {
    let mut log = setup_events();
    log.extend([
        Event::Produced {
            player: 1,
            resources: [1, 0, 0, 0, 0],
        },
        Event::Discarded {
            player: 1,
            resource: Resource::Ore,
        },
    ]);
    assert!(Belief::new(0, DEFAULT_MAX_STATES, 0).observe(&log).is_err());
}

#[test]
fn a_belief_from_an_observation_deals_the_unseen_cards() {
    let mut n = 0;
    random_game(9, no_trades(), |g| {
        n += 1;
        if n % 50 != 0 {
            return;
        }
        let obs = g.observation(2);
        let b = Belief::from_observation(&obs, 32, n);
        b.check(&obs).unwrap();
        assert!((b.states().iter().map(|s| s.1).sum::<f64>() - 1.0).abs() < 1e-9);
        for (t, _) in b.states() {
            for r in 0..NUM_RESOURCES {
                let held: u8 = (0..NUM_PLAYERS).map(|p| t.hands[p][r]).sum();
                assert_eq!(held + obs.bank[r], BANK_PER_RESOURCE);
            }
        }
    });
}

#[test]
fn road_building_roads_are_free() {
    let mut log = setup_events();
    log.extend([
        Event::PlayedDev {
            player: 1,
            card: DevCard::RoadBuilding,
        },
        Event::BuiltRoad {
            player: 1,
            edge: 50,
        },
        Event::BuiltRoad {
            player: 1,
            edge: 51,
        },
    ]);
    let mut t = HandTracker::new();
    let mut rng = Rng::new(0);
    assert!(
        log.iter().all(|e| t.step(e, &mut rng)),
        "two free roads after Road Building"
    );
    assert!(
        !t.step(
            &Event::BuiltRoad {
                player: 1,
                edge: 52
            },
            &mut rng
        ),
        "a third road is paid"
    );
}

#[test]
fn sampling_hands_follows_the_weights() {
    let mut log = setup_events();
    log.extend([
        Event::Produced {
            player: 1,
            resources: [3, 1, 0, 0, 0],
        },
        Event::Stole {
            thief: 2,
            victim: 1,
            resource: None,
        },
    ]);
    let mut b = Belief::new(0, DEFAULT_MAX_STATES, 0);
    b.observe(&log).unwrap();
    assert_eq!(b.states().len(), 2);
    let mut rng = Rng::new(4);
    let n = 20_000;
    let wood = (0..n)
        .filter(|_| b.sample_hands(&mut rng)[2][0] == 1)
        .count();
    assert!((wood as f64 / n as f64 - 0.75).abs() < 0.015);
}

#[test]
fn an_intervening_event_closes_the_road_building_window() {
    let mut log = setup_events();
    log.extend([
        Event::PlayedDev {
            player: 1,
            card: DevCard::RoadBuilding,
        },
        Event::BuiltRoad {
            player: 1,
            edge: 50,
        },
    ]);
    let mut t = HandTracker::new();
    let mut rng = Rng::new(0);
    assert!(
        log.iter().all(|e| t.step(e, &mut rng)),
        "one free road after Road Building"
    );
    assert!(t.step(&Event::TurnEnded { player: 1 }, &mut rng));
    assert!(
        !t.step(
            &Event::BuiltRoad {
                player: 1,
                edge: 51
            },
            &mut rng
        ),
        "the window closed, so the road is paid"
    );
}

/// Player 1 holds one of resource 0 and one of resource 4; player 2 steals one unseen, so player 2
/// holds either (50% each).
fn two_possible_thefts() -> Vec<Event> {
    let mut log = setup_events();
    log.extend([
        Event::Produced {
            player: 1,
            resources: [1, 0, 0, 0, 1],
        },
        Event::Stole {
            thief: 2,
            victim: 1,
            resource: None,
        },
    ]);
    log
}

#[test]
fn an_offer_of_a_card_only_one_state_holds_prunes_the_others() {
    let mut log = two_possible_thefts();
    let mut b = Belief::new(0, DEFAULT_MAX_STATES, 0);
    b.observe(&log).unwrap();
    assert_eq!(b.states().len(), 2);
    log.extend([
        Event::TradeOffered {
            player: 2,
            give: [1, 0, 0, 0, 0],
            get: [0, 1, 0, 0, 0],
        },
        Event::TradeCancelled { player: 2 },
    ]);
    let mut b = Belief::new(0, DEFAULT_MAX_STATES, 0);
    b.observe(&log).unwrap();
    assert_eq!(b.states().len(), 1);
    let (t, w) = &b.states()[0];
    assert!((w - 1.0).abs() < 1e-9);
    assert_eq!(t.hands[2], [1, 0, 0, 0, 0]);
    assert_eq!(t.hands[1], [0, 0, 0, 0, 1]);
}

#[test]
fn an_accept_requires_holding_what_the_offerer_wants() {
    // Player 3 offers resource 1 for resource 0; players 0 and 1 reject; player 2 accepts.
    let offer = [
        Event::Produced {
            player: 3,
            resources: [0, 1, 0, 0, 0],
        },
        Event::TradeOffered {
            player: 3,
            give: [0, 1, 0, 0, 0],
            get: [1, 0, 0, 0, 0],
        },
        Event::TradeResponded {
            player: 0,
            accepted: false,
        },
        Event::TradeResponded {
            player: 1,
            accepted: false,
        },
    ];

    // A rejection by player 2 says nothing about their hand: both states stay.
    let mut log = two_possible_thefts();
    log.extend(offer);
    log.push(Event::TradeResponded {
        player: 2,
        accepted: false,
    });
    log.push(Event::TradeCancelled { player: 3 });
    let mut b = Belief::new(0, DEFAULT_MAX_STATES, 0);
    b.observe(&log).unwrap();
    assert_eq!(b.states().len(), 2);

    // An acceptance proves player 2 holds resource 0, and the confirmed trade then moves cards.
    let mut log = two_possible_thefts();
    log.extend(offer);
    log.push(Event::TradeResponded {
        player: 2,
        accepted: true,
    });
    let mut b = Belief::new(0, DEFAULT_MAX_STATES, 0);
    b.observe(&log).unwrap();
    assert_eq!(b.states().len(), 1);
    assert_eq!(b.states()[0].0.hands[2], [1, 0, 0, 0, 0]);
    log.push(Event::TradeConfirmed {
        offerer: 3,
        partner: 2,
        offerer_gave: [0, 1, 0, 0, 0],
        partner_gave: [1, 0, 0, 0, 0],
    });
    b.observe(&log[log.len() - 1..]).unwrap();
    let t = &b.states()[0].0;
    assert_eq!(t.hands[3], [1, 0, 0, 0, 0]);
    assert_eq!(t.hands[2], [0, 1, 0, 0, 0]);
}

#[test]
fn different_parents_that_reach_the_same_child_are_merged() {
    // Player 1 holds [2, 1, ..]. Player 2 steals twice. Orders: wood,wood (2/3 * 1/2), wood,ore
    // (2/3 * 1/2) and ore,wood (1/3 * 1) -- the last two end in the same joint hand.
    let mut log = setup_events();
    log.extend([
        Event::Produced {
            player: 1,
            resources: [2, 1, 0, 0, 0],
        },
        Event::Stole {
            thief: 2,
            victim: 1,
            resource: None,
        },
        Event::Stole {
            thief: 2,
            victim: 1,
            resource: None,
        },
    ]);
    let truth = exact(&log);
    let mut b = Belief::new(0, DEFAULT_MAX_STATES, 0);
    b.observe(&log).unwrap();
    let branches = 3;
    assert!(
        b.states().len() < branches,
        "merged support {} < {branches} branches",
        b.states().len()
    );
    assert_eq!(b.states().len(), 2);
    assert_eq!(b.states().len(), truth.len());
    for (t, w) in b.states() {
        let p = truth[&t.hands];
        assert!((w - p).abs() < 1e-9, "{:?} exact {p} tracked {w}", t.hands);
    }
    let merged = b
        .states()
        .iter()
        .find(|(t, _)| t.hands[2] == [1, 1, 0, 0, 0])
        .expect("the merged state")
        .1;
    assert!((merged - 2.0 / 3.0).abs() < 1e-9);
}
