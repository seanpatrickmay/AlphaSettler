use settler_bots::{make_bot, Bot, BOT_NAMES};
use settler_engine::legal::legal_actions;
use settler_engine::*;

/// Play one game with four random bots. Returns the final state, the actions taken and the
/// number of decision points where an `OfferTrade` was legal.
fn play(bot_seed: u64, game_seed: u64) -> (State, Vec<Action>, usize) {
    let mut bots: Vec<Box<dyn Bot>> = (0..4)
        .map(|i| make_bot("random", bot_seed + i).unwrap())
        .collect();
    let mut s = State::new(game_seed, GameConfig::default());
    let mut buf = Vec::new();
    let mut taken = Vec::new();
    let mut offers_legal = 0;
    while !s.is_over() {
        legal_actions(&s, &mut buf);
        if buf.iter().any(|a| matches!(a, Action::OfferTrade { .. })) {
            offers_legal += 1;
        }
        let actor = s.current_actor();
        let a = bots[actor as usize].act(&s.observation(actor), &buf);
        assert!(buf.contains(&a), "{a:?} not legal");
        taken.push(a);
        s.apply(a);
    }
    (s, taken, offers_legal)
}

#[test]
fn random_bot_plays_legal_moves_to_the_end() {
    for seed in 0..20 {
        let (s, taken, _) = play(1, seed);
        assert!(s.is_over());
        assert!(!taken.is_empty());
    }
}

#[test]
fn random_bot_never_offers_trades() {
    // Default config allows 3 offers per turn, so offers are legal and must be filtered out.
    let mut offers_legal = 0;
    for seed in 0..20 {
        let (_, taken, legal) = play(2, seed);
        offers_legal += legal;
        assert!(!taken.iter().any(|a| matches!(a, Action::OfferTrade { .. })));
    }
    assert!(
        offers_legal > 0,
        "no decision point offered a trade; the test is vacuous"
    );
}

#[test]
fn random_bot_is_deterministic_per_seed() {
    assert_eq!(play(3, 7).1, play(3, 7).1);
    assert_ne!(play(3, 7).1, play(4, 7).1);
}

#[test]
fn bot_registry() {
    assert!(make_bot("nope", 0).is_none());
    assert!(BOT_NAMES.contains(&"random"));
    assert_eq!(make_bot("random", 0).unwrap().name(), "random");
}
