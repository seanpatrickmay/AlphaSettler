use settler_bots::{make_bot, Bot, BOT_NAMES};
use settler_engine::legal::legal_actions;
use settler_engine::*;

fn play(bot_seed: u64, game_seed: u64) -> (State, Vec<Action>) {
    let mut bots: Vec<Box<dyn Bot>> = (0..4).map(|i| make_bot("random", bot_seed + i).unwrap()).collect();
    let mut s = State::new(game_seed, GameConfig::default());
    let mut buf = Vec::new();
    let mut taken = Vec::new();
    while !s.is_over() {
        legal_actions(&s, &mut buf);
        let actor = s.current_actor();
        let a = bots[actor as usize].act(&s.observation(actor), &buf);
        assert!(buf.contains(&a), "{a:?} not legal");
        taken.push(a);
        s.apply(a);
    }
    (s, taken)
}

#[test]
fn random_bot_plays_legal_moves_to_the_end() {
    for seed in 0..20 {
        let (s, taken) = play(1, seed);
        assert!(s.is_over());
        assert!(!taken.is_empty());
    }
}

#[test]
fn random_bot_never_offers_trades() {
    // Default config allows 3 offers per turn, so offers are legal and must be filtered out.
    for seed in 0..20 {
        let (_, taken) = play(2, seed);
        assert!(!taken.iter().any(|a| matches!(a, Action::OfferTrade { .. })));
    }
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
