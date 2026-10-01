#![allow(dead_code)]

use settler_engine::legal::legal_actions;
use settler_engine::rng::Rng;
use settler_engine::*;

pub fn no_trades() -> GameConfig {
    GameConfig { max_offers_per_turn: 0, ..GameConfig::default() }
}

/// Plays uniformly random legal moves from game `seed`, calling `f(&game)` after every move.
pub fn random_game(seed: u64, config: GameConfig, mut f: impl FnMut(&Game)) {
    let mut g = Game::new(seed, config);
    let mut rng = Rng::new(seed ^ 0x5EED);
    while !g.state().is_over() {
        let legal = g.legal_actions();
        g.apply(legal[rng.below(legal.len() as u32) as usize]).unwrap();
        f(&g);
    }
}

/// The position right after setup in game `seed` (player 0 to roll), with random placements.
pub fn after_setup(seed: u64, config: GameConfig) -> State {
    let mut s = State::new(seed, config);
    let mut rng = Rng::new(seed);
    let mut buf = Vec::new();
    while matches!(s.phase, Phase::SetupSettlement | Phase::SetupRoad { .. }) {
        legal_actions(&s, &mut buf);
        s.apply(buf[rng.below(buf.len() as u32) as usize]);
    }
    s
}

/// `s` with `f` applied to its snapshot, rebuilt (and validated) under `config`.
pub fn edit(s: &State, config: GameConfig, f: impl FnOnce(&mut Snapshot)) -> State {
    let mut snap = s.snapshot();
    f(&mut snap);
    State::from_snapshot(s.seed, config, &snap).expect("the edited position is valid")
}

/// Move `n` cards of `card` from the deck into player `p`'s hand (bought on an earlier turn).
pub fn give_dev(snap: &mut Snapshot, p: usize, card: DevCard, n: u8) {
    for _ in 0..n {
        let i = snap.dev_deck.iter().position(|&c| c == card).expect("the card is left in the deck");
        snap.dev_deck.remove(i);
        snap.players[p].dev_hand[card.index()] += 1;
    }
}

/// Move `cards` from the bank to player `p`.
pub fn give_cards(snap: &mut Snapshot, p: usize, cards: Hand) {
    for r in 0..NUM_RESOURCES {
        snap.bank[r] -= cards[r];
        snap.players[p].hand[r] += cards[r];
    }
}

/// The 16 free setup placements (snake order) as events, so later builds are paid.
pub fn setup_events() -> Vec<Event> {
    let mut out = Vec::new();
    for (i, &p) in settler_engine::state::SETUP_ORDER.iter().enumerate() {
        out.push(Event::BuiltSettlement { player: p, node: i as u8 });
        out.push(Event::BuiltRoad { player: p, edge: i as u8 });
    }
    out
}
