//! IsmctsBot: tracks hidden cards from the event log and searches each real decision with
//! single-observer ISMCTS (spec Sections 2-3). Never trades.

use crate::greedy;
use crate::heuristic::{HeuristicEvaluator, DEFAULT_WEIGHTS};
use crate::Bot;
use settler_engine::rng::{mix, Rng};
use settler_engine::{Action, Event, Observation, Phase, PlayerId};
use settler_search::{search, search_actions, Belief, SearchConfig, SearchResult, DEFAULT_MAX_STATES};

pub const DEFAULT_SIMULATIONS: u32 = 1000;
pub const MAX_SIMULATIONS: u32 = 1_000_000;
pub const MAX_ROLLOUT: u32 = 200;
const BELIEF_SALT: u64 = 0xBE11EF;

/// `ismcts` -> (1000, 0); `ismcts@N` -> (N, 0); `ismcts@N+rD` -> (N, D). Digits only, no sign.
pub fn parse_name(name: &str) -> Option<(u32, u32)> {
    let rest = name.strip_prefix("ismcts")?;
    if rest.is_empty() {
        return Some((DEFAULT_SIMULATIONS, 0));
    }
    let rest = rest.strip_prefix('@')?;
    let number = |s: &str| -> Option<u32> {
        if s.is_empty() || !s.bytes().all(|b| b.is_ascii_digit()) {
            return None;
        }
        s.parse().ok()
    };
    let (sims, rollout) = match rest.split_once("+r") {
        Some((s, r)) => (number(s)?, number(r)?),
        None => (number(rest)?, 0),
    };
    let sims_ok = (1..=MAX_SIMULATIONS).contains(&sims);
    let rollout_ok = rest.contains("+r") == (rollout > 0) && rollout <= MAX_ROLLOUT;
    (sims_ok && rollout_ok).then_some((sims, rollout))
}

/// What the bot knows about the hidden cards.
enum Tracked {
    /// Nothing observed yet.
    Unseen,
    /// The exact posterior, advanced by every event since the game started.
    Live(Belief),
    /// `viewer`'s history contradicted the belief (the reset is already counted); later events
    /// are skipped until `act` rebuilds the belief from an observation.
    Dropped(PlayerId),
}

pub struct IsmctsBot {
    seed: u64,
    rng: Rng,
    cfg: SearchConfig,
    eval: HeuristicEvaluator,
    tracked: Tracked,
    last: Option<SearchResult>,
    resets: u64,
    /// Truncations of beliefs already replaced.
    past_truncations: u64,
    searches: u64,
    buf: Vec<Action>,
}

impl IsmctsBot {
    pub fn new(seed: u64, simulations: u32, rollout: u32) -> IsmctsBot {
        IsmctsBot {
            seed,
            rng: Rng::new(seed),
            cfg: SearchConfig { simulations, ..SearchConfig::default() },
            eval: HeuristicEvaluator::new(DEFAULT_WEIGHTS, rollout),
            tracked: Tracked::Unseen,
            last: None,
            resets: 0,
            past_truncations: 0,
            searches: 0,
            buf: Vec::new(),
        }
    }

    /// The last decision's search, if it searched; taking it clears it.
    pub fn take_last_search(&mut self) -> Option<SearchResult> {
        self.last.take()
    }

    /// Replace the tracked state, keeping the old belief's truncation count.
    fn set_tracked(&mut self, t: Tracked) {
        if let Tracked::Live(b) = std::mem::replace(&mut self.tracked, t) {
            self.past_truncations += b.truncations();
        }
    }
}

impl Bot for IsmctsBot {
    fn name(&self) -> &'static str {
        "ismcts"
    }

    /// The first events a bot sees as `viewer` start a game: the belief begins from empty hands.
    /// A contradiction counts one reset here, at the error.
    fn observe(&mut self, viewer: PlayerId, events: &[Event]) {
        match &self.tracked {
            Tracked::Live(b) if b.viewer() == viewer => {}
            Tracked::Dropped(v) if *v == viewer => return,
            _ => self.set_tracked(Tracked::Live(Belief::new(viewer, DEFAULT_MAX_STATES, mix(self.seed, BELIEF_SALT)))),
        }
        let Tracked::Live(b) = &mut self.tracked else { unreachable!("live after the match above") };
        if b.observe(events).is_err() {
            self.resets += 1;
            self.set_tracked(Tracked::Dropped(viewer));
        }
    }

    fn act(&mut self, obs: &Observation, legal: &[Action]) -> Action {
        self.last = None;
        match obs.phase {
            Phase::TradeResponse if legal.contains(&Action::RejectTrade) => return Action::RejectTrade,
            Phase::TradeConfirm if legal.contains(&Action::CancelTrade) => return Action::CancelTrade,
            _ => {}
        }
        search_actions(legal, &mut self.buf);
        if self.buf.len() == 1 {
            return self.buf[0];
        }
        // A live belief that disagrees with `obs`, or no history for this viewer at all, is a
        // reset; a belief dropped in `observe` was counted there.
        let rebuild = match &self.tracked {
            Tracked::Live(b) if b.viewer() == obs.viewer => b.check(obs).is_err().then_some(true),
            Tracked::Dropped(v) if *v == obs.viewer => Some(false),
            _ => Some(true),
        };
        if let Some(count) = rebuild {
            self.resets += count as u64;
            let b = Belief::from_observation(obs, DEFAULT_MAX_STATES, self.rng.next_u64());
            self.set_tracked(Tracked::Live(b));
        }
        let Tracked::Live(belief) = &self.tracked else { unreachable!("live after the rebuild above") };
        match search(obs, legal, belief, &mut self.eval, &self.cfg, &mut self.rng) {
            Ok(r) => {
                self.searches += 1;
                let a = r.action;
                self.last = Some(r);
                a
            }
            Err(_) => {
                // Counted now; the next searched decision rebuilds without counting again.
                self.resets += 1;
                self.set_tracked(Tracked::Dropped(obs.viewer));
                greedy::choose(obs, legal)
            }
        }
    }

    fn diagnostics(&self) -> Vec<(&'static str, u64)> {
        let live = match &self.tracked {
            Tracked::Live(b) => b.truncations(),
            _ => 0,
        };
        vec![
            ("belief_resets", self.resets),
            ("belief_truncations", self.past_truncations + live),
            ("searches", self.searches),
        ]
    }
}
