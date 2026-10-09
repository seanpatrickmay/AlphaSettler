//! Baseline bots, IsmctsBot (heuristic or network evaluator) and the native arena. Each decision
//! a bot sees an `Observation` and the legal actions; through `observe` it also sees the events
//! since its last move, redacted for its seat, so it can keep its own history.

pub mod arena;
pub mod greedy;
pub mod heuristic;
pub mod ismcts;
pub mod netbot;
pub mod opening;
pub mod opening_model;
pub mod random;
pub mod selfplay;

use opening_model::{OpeningModel, OPENING_FILE};
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

/// The prefix of a league style's name (spec 3d-3 Section 3): `style:STYLE` with an `ismcts`
/// name's budget and suffixes, STYLE one of `heuristic::STYLES`.
pub const STYLE_PREFIX: &str = "style:";

/// A fresh bot by name, with its own random stream seeded from `seed`. Besides `BOT_NAMES`,
/// `ismcts@N` (N simulations) and `ismcts@N+rD` (plus D greedy rollout moves per leaf), either
/// with `+sK` to search setup settlements with K times the simulations, `+t` to make domestic
/// offers and `+g` to search the root by Gumbel sequential halving, the suffixes in any order.
/// `style:STYLE` takes everything an `ismcts` name takes after `ismcts` (3d-3: the same bot with
/// the style's heuristic weights). `net:` names (and `+o`, which only they take) need files
/// loaded first: use a `Roster`.
pub fn make_bot(name: &str, seed: u64) -> Option<Box<dyn Bot>> {
    match name {
        "random" => Some(Box::new(random::RandomBot::new(seed))),
        "greedy" => Some(Box::new(greedy::GreedyBot::new(seed))),
        _ => {
            let (name, gumbel) = ismcts::split_gumbel(name)?;
            let (name, offers) = ismcts::split_trades(&name)?;
            let (base, factor) = ismcts::split_setup(&name)?;
            let bot = match base.strip_prefix(STYLE_PREFIX) {
                Some(rest) => style_bot(rest, seed)?,
                None => {
                    let (sims, rollout) = ismcts::parse_name(base)?;
                    ismcts::IsmctsBot::new(seed, sims, rollout)
                }
            };
            Some(Box::new(
                bot.with_setup_factor(factor)
                    .with_offers(offers)
                    .with_gumbel(gumbel),
            ))
        }
    }
}

/// `STYLE[@N[+rD]]` (a `style:` name without its prefix and flag suffixes): the bot
/// `IsmctsBot::new` builds for `ismcts[@N[+rD]]`, reporting the name `style`, with the style's
/// weights (`heuristic::style_weights`).
fn style_bot(rest: &str, seed: u64) -> Option<ismcts::IsmctsBot> {
    let end = rest.find(['@', '+']).unwrap_or(rest.len());
    let weights = heuristic::style_weights(&rest[..end])?;
    let (simulations, rollout) = ismcts::parse_name(&format!("ismcts{}", &rest[end..]))?;
    Some(ismcts::IsmctsBot::with_evaluator(
        "style",
        seed,
        settler_search::SearchConfig {
            simulations,
            ..settler_search::SearchConfig::default()
        },
        heuristic::HeuristicEvaluator::new(weights, rollout),
    ))
}

/// The error for a name no bot answers to.
pub fn unknown_bot(name: &str) -> String {
    format!(
        "unknown bot {name:?}; known bots: {BOT_NAMES:?} (ismcts@N for N simulations, ismcts@N+rD \
         to add D greedy rollout moves, net:NAME[@N[+rD]] for <nets-dir>/NAME.bin, style:STYLE[@N[+rD]] for a \
         league style (roads, army or production); +sK on any \
         searches setup settlements with K times the simulations, +t lets any make domestic \
         offers, +g makes any search the root by Gumbel sequential halving, +o lets a net: bot \
         pick its setup settlements with <nets-dir>/{OPENING_FILE})"
    )
}

/// Bot names checked, and their networks (and the opening model, for `+o`) loaded once and shared
/// read-only across threads, so a match builds bots per game without touching the disk.
pub struct Roster {
    nets: Vec<(String, Arc<Weights>)>,
    /// `<nets_dir>/opening.bin`, loaded only if some name has `+o`.
    opening: Option<Arc<OpeningModel>>,
}

impl Roster {
    /// Every name must be a known bot, and each `net:` name's file in `nets_dir` must load and
    /// validate, as must `<nets_dir>/opening.bin` if a name has `+o`. The error names the first
    /// bad bot or file. A malformed name is rejected before any file is read.
    pub fn new(names: &[&str], nets_dir: &Path) -> Result<Roster, String> {
        let mut nets: Vec<(String, Arc<Weights>)> = Vec::new();
        let mut opening = None;
        for &name in names {
            // `+g` changes how a bot searches, not which files it loads.
            let base = ismcts::split_gumbel(name).map(|(b, _)| b);
            if let Some(n) = base.as_deref().and_then(netbot::parse_net_name) {
                if !nets.iter().any(|(net, _)| net == n.net) {
                    nets.push((n.net.to_owned(), netbot::load_net(nets_dir, n.net)?));
                }
                if n.opening && opening.is_none() {
                    opening = Some(Arc::new(OpeningModel::load(&nets_dir.join(OPENING_FILE))?));
                }
            } else if make_bot(name, 0).is_none() {
                return Err(unknown_bot(name));
            }
        }
        Ok(Roster { nets, opening })
    }

    /// A fresh bot; `None` for an unknown name, or a `net:` name whose network (or, with `+o`,
    /// opening model) this roster did not load.
    pub fn make(&self, name: &str, seed: u64) -> Option<Box<dyn Bot>> {
        let (base, gumbel) = ismcts::split_gumbel(name)?;
        match netbot::parse_net_name(&base) {
            Some(n) => {
                let (_, w) = self.nets.iter().find(|(net, _)| net == n.net)?;
                let mut bot = netbot::net_bot(seed, w.clone(), n.simulations, n.rollout)
                    .with_setup_factor(n.setup_factor)
                    .with_offers(n.offers)
                    .with_gumbel(gumbel);
                if n.opening {
                    bot = bot.with_opening(self.opening.clone()?);
                }
                Some(Box::new(bot))
            }
            None => make_bot(name, seed),
        }
    }
}
