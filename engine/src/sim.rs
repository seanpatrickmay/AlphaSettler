//! Uniformly random playouts (benchmarks, property tests, rollout baselines).

use crate::action::Action;
use crate::legal::legal_actions;
use crate::rng::Rng;
use crate::state::State;

/// Play uniformly random legal actions until the game ends. Returns the number of actions.
pub fn play_random(s: &mut State, rng: &mut Rng, buf: &mut Vec<Action>) -> u64 {
    let mut steps = 0;
    while !s.is_over() {
        legal_actions(s, buf);
        let a = buf[rng.below(buf.len() as u32) as usize];
        s.apply(a);
        steps += 1;
    }
    steps
}
