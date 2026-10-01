//! Baseline bots and the native arena. Bots see only an `Observation` and the legal actions.

pub mod arena;
pub mod greedy;
pub mod random;

use settler_engine::{Action, Observation};

pub trait Bot: Send {
    fn name(&self) -> &'static str;
    /// Choose one of `legal` (never empty) for `obs.viewer`, the player who must act now.
    fn act(&mut self, obs: &Observation, legal: &[Action]) -> Action;
}

pub const BOT_NAMES: &[&str] = &["random", "greedy"];

/// A fresh bot by name, with its own random stream seeded from `seed`.
pub fn make_bot(name: &str, seed: u64) -> Option<Box<dyn Bot>> {
    match name {
        "random" => Some(Box::new(random::RandomBot::new(seed))),
        "greedy" => Some(Box::new(greedy::GreedyBot::new(seed))),
        _ => None,
    }
}
