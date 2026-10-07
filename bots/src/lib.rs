//! Baseline bots, IsmctsBot (heuristic or network evaluator) and the native arena. Each decision
//! a bot sees an `Observation` and the legal actions; through `observe` it also sees the events
//! since its last move, redacted for its seat, so it can keep its own history.

pub mod arena;
pub mod greedy;
pub mod heuristic;
pub mod ismcts;
pub mod netbot;
pub mod random;
pub mod selfplay;

use settler_engine::{Action, Event, Observation, PlayerId};
use settler_net::weights::Weights;
use std::path::Path;
use std::sync::Arc;

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
/// `ismcts@N` (N simulations) and `ismcts@N+rD` (plus D greedy rollout moves per leaf), either
/// with `+sK` to search setup settlements with K times the simulations and `+t` to make domestic
/// offers, the suffixes in any order. `net:` names need their weights loaded first: use a `Roster`.
pub fn make_bot(name: &str, seed: u64) -> Option<Box<dyn Bot>> {
    match name {
        "random" => Some(Box::new(random::RandomBot::new(seed))),
        "greedy" => Some(Box::new(greedy::GreedyBot::new(seed))),
        _ => {
            let (name, offers) = ismcts::split_trades(name)?;
            let (base, factor) = ismcts::split_setup(&name)?;
            let (sims, rollout) = ismcts::parse_name(base)?;
            Some(Box::new(
                ismcts::IsmctsBot::new(seed, sims, rollout)
                    .with_setup_factor(factor)
                    .with_offers(offers),
            ))
        }
    }
}

/// The error for a name no bot answers to.
pub fn unknown_bot(name: &str) -> String {
    format!(
        "unknown bot {name:?}; known bots: {BOT_NAMES:?} (ismcts@N for N simulations, ismcts@N+rD \
         to add D greedy rollout moves, net:NAME[@N[+rD]] for <nets-dir>/NAME.bin; +sK on either \
         searches setup settlements with K times the simulations, +t lets either make domestic \
         offers)"
    )
}

/// Bot names checked, and their networks loaded once and shared read-only across threads, so a
/// match builds bots per game without touching the disk.
pub struct Roster {
    nets: Vec<(String, Arc<Weights>)>,
}

impl Roster {
    /// Every name must be a known bot, and each `net:` name's file in `nets_dir` must load and
    /// validate. The error names the first bad bot. A malformed name is rejected before any file
    /// is read.
    pub fn new(names: &[&str], nets_dir: &Path) -> Result<Roster, String> {
        let mut nets: Vec<(String, Arc<Weights>)> = Vec::new();
        for &name in names {
            if let Some((net, ..)) = netbot::parse_net_name(name) {
                if !nets.iter().any(|(n, _)| n == net) {
                    nets.push((net.to_owned(), netbot::load_net(nets_dir, net)?));
                }
            } else if make_bot(name, 0).is_none() {
                return Err(unknown_bot(name));
            }
        }
        Ok(Roster { nets })
    }

    /// A fresh bot; `None` for an unknown name or a `net:` name whose network this roster did not
    /// load.
    pub fn make(&self, name: &str, seed: u64) -> Option<Box<dyn Bot>> {
        match netbot::parse_net_name(name) {
            Some((net, simulations, rollout, factor, offers)) => {
                let (_, w) = self.nets.iter().find(|(n, _)| n == net)?;
                Some(Box::new(
                    netbot::net_bot(seed, w.clone(), simulations, rollout)
                        .with_setup_factor(factor)
                        .with_offers(offers),
                ))
            }
            None => make_bot(name, seed),
        }
    }
}
