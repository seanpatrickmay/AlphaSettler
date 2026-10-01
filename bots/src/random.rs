//! Uniform over legal actions, except it never proposes domestic trades.

use crate::Bot;
use settler_engine::rng::Rng;
use settler_engine::{Action, Observation};

pub struct RandomBot {
    rng: Rng,
    pool: Vec<Action>,
}

impl RandomBot {
    pub fn new(seed: u64) -> RandomBot {
        RandomBot {
            rng: Rng::new(seed),
            pool: Vec::new(),
        }
    }
}

impl Bot for RandomBot {
    fn name(&self) -> &'static str {
        "random"
    }

    fn act(&mut self, _obs: &Observation, legal: &[Action]) -> Action {
        self.pool.clear();
        self.pool.extend(
            legal
                .iter()
                .copied()
                .filter(|a| !matches!(a, Action::OfferTrade { .. })),
        );
        let pool: &[Action] = if self.pool.is_empty() {
            legal
        } else {
            &self.pool
        };
        pool[self.rng.below(pool.len() as u32) as usize]
    }
}
