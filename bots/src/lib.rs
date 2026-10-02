//! Baseline bots and the native arena. Bots see only an `Observation` and the legal actions.

pub mod arena;
pub mod greedy;
pub mod heuristic;
pub mod ismcts;
pub mod random;

use settler_engine::{Action, Event, Observation, PlayerId};

pub trait Bot: Send {
    fn name(&self) -> &'static str;

    /// Events since this bot last moved, as `viewer` (its seat) may see them, in log order. The
    /// arena calls this before every `act`; bots that keep no history ignore it.
    fn observe(&mut self, _viewer: PlayerId, _events: &[Event]) {}

    /// Choose one of `legal` (never empty) for `obs.viewer`, the player who must act now.
    fn act(&mut self, obs: &Observation, legal: &[Action]) -> Action;

    /// Named counters for reports and tests, e.g. how often a belief had to be rebuilt.
    fn diagnostics(&self) -> Vec<(&'static str, u64)> {
        Vec::new()
    }
}

pub const BOT_NAMES: &[&str] = &["random", "greedy", "ismcts"];

/// A fresh bot by name, with its own random stream seeded from `seed`. Besides `BOT_NAMES`,
/// `ismcts@N` (N simulations) and `ismcts@N+rD` (plus D greedy rollout moves per leaf).
pub fn make_bot(name: &str, seed: u64) -> Option<Box<dyn Bot>> {
    match name {
        "random" => Some(Box::new(random::RandomBot::new(seed))),
        "greedy" => Some(Box::new(greedy::GreedyBot::new(seed))),
        _ => {
            let (sims, rollout) = ismcts::parse_name(name)?;
            Some(Box::new(ismcts::IsmctsBot::new(seed, sims, rollout)))
        }
    }
}
