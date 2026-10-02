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

pub struct IsmctsBot {
    seed: u64,
    rng: Rng,
    cfg: SearchConfig,
    eval: HeuristicEvaluator,
    belief: Option<Belief>,
    last: Option<SearchResult>,
    resets: u64,
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
            belief: None,
            last: None,
            resets: 0,
            searches: 0,
            buf: Vec::new(),
        }
    }

    /// The last decision's search, if it searched; taking it clears it.
    pub fn take_last_search(&mut self) -> Option<SearchResult> {
        self.last.take()
    }
}

impl Bot for IsmctsBot {
    fn name(&self) -> &'static str {
        "ismcts"
    }

    fn observe(&mut self, viewer: PlayerId, events: &[Event]) {
        if self.belief.as_ref().map_or(true, |b| b.viewer() != viewer) {
            self.belief = Some(Belief::new(viewer, DEFAULT_MAX_STATES, mix(self.seed, BELIEF_SALT)));
        }
        let ok = self.belief.as_mut().expect("set above").observe(events).is_ok();
        if !ok {
            self.belief = None; // rebuilt from the next observation; `act` counts the reset
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
        let usable = self.belief.as_ref().is_some_and(|b| b.viewer() == obs.viewer && b.check(obs).is_ok());
        if !usable {
            self.resets += 1;
            self.belief = Some(Belief::from_observation(obs, DEFAULT_MAX_STATES, self.rng.next_u64()));
        }
        let belief = self.belief.as_ref().expect("set above");
        match search(obs, legal, belief, &mut self.eval, &self.cfg, &mut self.rng) {
            Ok(r) => {
                self.searches += 1;
                let a = r.action;
                self.last = Some(r);
                a
            }
            Err(_) => {
                self.resets += 1;
                greedy::choose(obs, legal)
            }
        }
    }

    fn diagnostics(&self) -> Vec<(&'static str, u64)> {
        vec![
            ("belief_resets", self.resets),
            ("belief_truncations", self.belief.as_ref().map_or(0, |b| b.truncations())),
            ("searches", self.searches),
        ]
    }
}
