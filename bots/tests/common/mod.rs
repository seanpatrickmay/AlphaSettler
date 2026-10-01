#![allow(dead_code)]

use settler_engine::*;

/// A fresh game past setup: no buildings, empty hands, player 0 to act in `phase`.
pub fn blank(seed: u64, phase: Phase) -> State {
    let mut s = State::new(seed, GameConfig::default());
    s.setup_step = 8;
    s.current = 0;
    s.phase = phase;
    s
}

pub fn give(s: &mut State, p: usize, h: Hand) {
    hand_sub(&mut s.bank, &h);
    hand_add(&mut s.players[p].hand, &h);
}

pub fn act(bot: &mut dyn settler_bots::Bot, s: &State) -> Action {
    let legal = s.legal_actions();
    let a = bot.act(&s.observation(s.current_actor()), &legal);
    assert!(legal.contains(&a), "{a:?} not legal");
    a
}
