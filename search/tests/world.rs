mod common;

use common::*;
use settler_engine::rng::Rng;
use settler_engine::*;
use settler_search::{Belief, WorldSampler, DEFAULT_MAX_STATES};

fn trade_phase(p: Phase) -> bool {
    matches!(p, Phase::TradeResponse | Phase::TradeConfirm)
}

#[test]
fn sampled_worlds_are_valid_and_look_exactly_like_the_observation() {
    let games = (0..24).map(|s| (s, no_trades())).chain((24..30).map(|s| (s, GameConfig::default())));
    let mut checked = 0;
    for (seed, config) in games {
        let mut beliefs: Vec<Belief> = (0..4).map(|p| Belief::new(p, DEFAULT_MAX_STATES, seed)).collect();
        let mut seen = [0usize; 4];
        let mut rng = Rng::new(seed);
        let mut n = 0;
        random_game(seed, config, |g| {
            n += 1;
            for p in 0..4u8 {
                let log = g.log_for(p);
                beliefs[p as usize].observe(&log[seen[p as usize]..]).unwrap();
                seen[p as usize] = log.len();
            }
            if n % 5 != 0 || g.state().is_over() || trade_phase(g.state().phase) {
                return;
            }
            for p in 0..4u8 {
                let obs = g.observation(p);
                let sampler = WorldSampler::new(&obs, &beliefs[p as usize]).unwrap();
                for _ in 0..3 {
                    let w = sampler.sample(&beliefs[p as usize], &mut rng);
                    w.check_invariants().unwrap();
                    assert_eq!(w.observation(p), obs, "seed {seed} step {n} viewer {p}");
                    assert_eq!(State::from_snapshot(w.seed, w.config, &w.snapshot()).unwrap(), w);
                    checked += 1;
                }
            }
        });
    }
    assert!(checked > 10_000, "checked {checked} worlds");
}

#[test]
fn worlds_use_fresh_randomness() {
    let s = after_setup(3, no_trades());
    let obs = s.observation(0);
    let b = Belief::from_observation(&obs, 16, 1);
    let sampler = WorldSampler::new(&obs, &b).unwrap();
    let mut rng = Rng::new(0);
    let seeds: std::collections::HashSet<u64> = (0..20).map(|_| sampler.sample(&b, &mut rng).seed).collect();
    assert_eq!(seeds.len(), 20);
    assert!(!seeds.contains(&s.seed));
}

#[test]
fn hidden_dev_cards_are_dealt_uniformly() {
    // Player 1 holds one dev card the viewer cannot see; with the rest of the deck it is one of
    // the 25 cards minus the viewer's own, so a knight shows up 14/25 of the time.
    let base = after_setup(4, no_trades());
    let s = edit(&base, no_trades(), |snap| give_dev(snap, 1, DevCard::Monopoly, 1));
    let obs = s.observation(0);
    let b = Belief::from_observation(&obs, 16, 2);
    let sampler = WorldSampler::new(&obs, &b).unwrap();
    let mut rng = Rng::new(5);
    let n = 5000;
    let knights = (0..n).filter(|_| sampler.sample(&b, &mut rng).players[1].dev_hand[DevCard::Knight.index()] == 1).count();
    let rate = knights as f64 / n as f64;
    assert!((rate - 14.0 / 25.0).abs() < 0.03, "knight rate {rate}");
}

/// Dev cards of `kind` in `w`'s deck, not yet drawn.
fn in_deck(w: &State, kind: DevCard) -> usize {
    w.dev_deck[w.dev_deck_pos as usize..].iter().filter(|&&c| c == kind).count()
}

#[test]
fn the_opponent_on_turn_is_never_dealt_a_win() {
    // With 3 VP to win, player 1 (on turn, 2 public VP) would already have won holding a VP card.
    let cfg = GameConfig { vp_to_win: 3, ..no_trades() };
    let base = after_setup(6, cfg);
    let s = edit(&base, cfg, |snap| {
        give_dev(snap, 1, DevCard::Knight, 1);
        give_dev(snap, 1, DevCard::Monopoly, 1);
        give_dev(snap, 2, DevCard::YearOfPlenty, 1);
        snap.current = 1;
        snap.phase = Phase::PreRoll;
    });
    let obs = s.observation(0);
    let b = Belief::from_observation(&obs, 16, 3);
    let sampler = WorldSampler::new(&obs, &b).unwrap();
    let mut rng = Rng::new(7);
    let n = 10_000;
    let (mut knights, mut next_is_vp, mut p2_vp) = (0, 0, 0);
    for _ in 0..n {
        let w = sampler.sample(&b, &mut rng);
        assert_eq!(w.players[1].dev_hand[DevCard::VictoryPoint.index()], 0);
        w.check_invariants().unwrap();
        // Player 1 holds none of the 5 VP cards: they are in the deck or with player 2.
        let p2_cards = w.players[2].dev_hand[DevCard::VictoryPoint.index()] as usize;
        assert_eq!(in_deck(&w, DevCard::VictoryPoint) + p2_cards, 5, "the VP cards are not player 1's");
        knights += w.players[1].dev_hand[DevCard::Knight.index()] as u32;
        next_is_vp += (w.dev_deck[w.dev_deck_pos as usize] == DevCard::VictoryPoint) as u32;
        p2_vp += w.players[2].dev_hand[DevCard::VictoryPoint.index()] as u32;
    }
    // Two cards drawn uniformly from the 20 non-VP cards (14 of them knights): 2 * 14 / 20 knights.
    let mean = knights as f64 / n as f64;
    assert!((mean - 1.4).abs() < 0.03, "mean knights {mean}");
    // The 23 cards left after player 1's two (5 of them VP) are in uniform order: the next draw and
    // player 2's one card are each a VP card with probability 5/23.
    let exact = 5.0 / 23.0;
    let next = next_is_vp as f64 / n as f64;
    assert!((next - exact).abs() < 0.03, "next deck card is VP {next}, exact {exact}");
    let p2 = p2_vp as f64 / n as f64;
    assert!((p2 - exact).abs() < 0.03, "player 2 holds VP {p2}, exact {exact}");
}

#[test]
fn an_on_turn_opponent_holding_most_of_the_pool_is_still_sampled() {
    // The viewer holds 5 knights, leaving 20 unseen cards (9 knights, 5 VP, 6 others). Player 1, on
    // turn one VP short with 14 of them, can only hold non-VP cards; a rejection sampler accepts
    // such a deal about 4 times in 10,000.
    let cfg = GameConfig { vp_to_win: 3, ..no_trades() };
    let base = after_setup(8, cfg);
    let s = edit(&base, cfg, |snap| {
        give_dev(snap, 0, DevCard::Knight, 5);
        give_dev(snap, 1, DevCard::Knight, 9);
        give_dev(snap, 1, DevCard::Monopoly, 2);
        give_dev(snap, 1, DevCard::YearOfPlenty, 2);
        give_dev(snap, 1, DevCard::RoadBuilding, 1);
        snap.current = 1;
        snap.phase = Phase::PreRoll;
    });
    assert_eq!(s.players[1].dev_hand.iter().sum::<u8>(), 14);
    let obs = s.observation(0);
    let b = Belief::from_observation(&obs, 16, 3);
    let sampler = WorldSampler::new(&obs, &b).unwrap();
    let mut rng = Rng::new(9);
    let n = 2000;
    let mut all_nine = 0;
    for _ in 0..n {
        let w = sampler.sample(&b, &mut rng);
        assert_eq!(w.players[1].dev_hand.iter().sum::<u8>(), 14);
        assert_eq!(w.players[1].dev_hand[DevCard::VictoryPoint.index()], 0);
        assert_eq!(in_deck(&w, DevCard::VictoryPoint), 5);
        w.check_invariants().unwrap();
        all_nine += (w.players[1].dev_hand[DevCard::Knight.index()] == 9) as u32;
    }
    // 14 of the 15 non-VP cards: every knight is held unless the one left out is a knight (9/15).
    let rate = all_nine as f64 / n as f64;
    assert!((rate - 0.4).abs() < 0.03, "all-nine-knights rate {rate}");
}

#[test]
fn sampled_hands_follow_the_beliefs_weights() {
    // The first position in a random game whose belief for some viewer has 2 to 6 joint hands.
    let mut case = None;
    random_game(24, GameConfig::default(), |g| {
        if case.is_some() || g.state().is_over() || trade_phase(g.state().phase) {
            return;
        }
        // Rebuild each viewer's belief from the whole log.
        for p in 0..4u8 {
            let mut b = Belief::new(p, DEFAULT_MAX_STATES, 0);
            b.observe(&g.log_for(p)).unwrap();
            if (2..=6).contains(&b.states().len()) && case.is_none() {
                case = Some((g.observation(p), b));
            }
        }
    });
    let (obs, b) = case.expect("a random game has a position with an ambiguous hand");
    let sampler = WorldSampler::new(&obs, &b).unwrap();
    let mut rng = Rng::new(11);
    let n = 5000;
    let mut counts: std::collections::HashMap<[Hand; 4], usize> = Default::default();
    for _ in 0..n {
        let w = sampler.sample(&b, &mut rng);
        *counts.entry(std::array::from_fn(|q| w.players[q].hand)).or_default() += 1;
    }
    assert!(counts.len() >= 2, "only {} distinct hand sets", counts.len());
    assert_eq!(counts.len(), b.states().len());
    for (t, weight) in b.states() {
        let freq = counts.get(&t.hands).copied().unwrap_or(0) as f64 / n as f64;
        assert!((freq - weight).abs() < 0.03, "hands {:?}: frequency {freq}, weight {weight}", t.hands);
    }
}

#[test]
fn a_pending_trade_cannot_be_sampled() {
    let mut found = false;
    random_game(1, GameConfig::default(), |g| {
        if found || g.state().phase != Phase::TradeResponse {
            return;
        }
        found = true;
        let actor = g.state().current_actor();
        let obs = g.observation(actor);
        let b = Belief::from_observation(&obs, 8, 0);
        assert!(WorldSampler::new(&obs, &b).is_err());
    });
    assert!(found, "a random game with trades reaches TradeResponse");
}
