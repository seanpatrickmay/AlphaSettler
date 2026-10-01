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

#[test]
fn the_opponent_on_turn_is_never_dealt_a_win() {
    // With 3 VP to win, player 1 (on turn, 2 public VP) would already have won holding a VP card.
    let cfg = GameConfig { vp_to_win: 3, ..no_trades() };
    let base = after_setup(6, cfg);
    let s = edit(&base, cfg, |snap| {
        give_dev(snap, 1, DevCard::Knight, 1);
        give_dev(snap, 1, DevCard::Monopoly, 1);
        snap.current = 1;
        snap.phase = Phase::PreRoll;
    });
    let obs = s.observation(0);
    let b = Belief::from_observation(&obs, 16, 3);
    let sampler = WorldSampler::new(&obs, &b).unwrap();
    let mut rng = Rng::new(7);
    let mut vp_in_deck = 0;
    for _ in 0..2000 {
        let w = sampler.sample(&b, &mut rng);
        assert_eq!(w.players[1].dev_hand[DevCard::VictoryPoint.index()], 0);
        w.check_invariants().unwrap();
        vp_in_deck += w.dev_deck[w.dev_deck_pos as usize..].contains(&DevCard::VictoryPoint) as u32;
    }
    assert_eq!(vp_in_deck, 2000, "the VP cards are all in the deck");
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
