//! IsmctsBot: tracks hidden cards from the event log and searches each real decision with
//! single-observer ISMCTS (spec Sections 2-3). Never trades. Generic over the evaluator: the
//! heuristic for `ismcts` names, the network for `net:` names (`crate::netbot`).

use crate::greedy;
use crate::heuristic::{HeuristicEvaluator, DEFAULT_WEIGHTS};
use crate::Bot;
use settler_engine::rng::{mix, Rng};
use settler_engine::{Action, Event, Observation, Phase, PlayerId};
use settler_search::{
    search, search_actions, Belief, Evaluator, Noised, RootNoise, SearchConfig, SearchResult,
    DEFAULT_MAX_STATES,
};

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

/// How one decision is searched and chosen; self-play varies these move by move.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MoveOptions {
    pub simulations: u32,
    /// Dirichlet noise mixed into the root prior, drawn from `Rng::new(seed)`.
    pub noise: Option<(RootNoise, u64)>,
    /// Play a move drawn in proportion to root visits (from `Rng::new(seed)`) instead of the
    /// most-visited one.
    pub sample: Option<u64>,
}

/// An entry of `legal` drawn in proportion to `visits` (aligned with it); `None` if nothing was
/// visited.
pub fn sample_by_visits(legal: &[Action], visits: &[u32], seed: u64) -> Option<Action> {
    let total: u64 = visits.iter().map(|&v| v as u64).sum();
    if total == 0 {
        return None;
    }
    let mut x = ((Rng::new(seed).next_u64() as u128 * total as u128) >> 64) as u64;
    for (a, &v) in legal.iter().zip(visits) {
        if x < v as u64 {
            return Some(*a);
        }
        x -= v as u64;
    }
    unreachable!("x is below the visit total")
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

pub struct IsmctsBot<E = HeuristicEvaluator> {
    name: &'static str,
    seed: u64,
    rng: Rng,
    cfg: SearchConfig,
    eval: E,
    tracked: Tracked,
    last: Option<SearchResult>,
    resets: u64,
    /// `observe` counted a reset since the last `act`, so that decision counts no other.
    counted_in_observe: bool,
    /// Truncations of beliefs already replaced.
    past_truncations: u64,
    searches: u64,
    buf: Vec<Action>,
}

impl IsmctsBot<HeuristicEvaluator> {
    /// 3a's bot: the heuristic evaluator with `rollout` greedy moves per leaf.
    pub fn new(seed: u64, simulations: u32, rollout: u32) -> IsmctsBot {
        IsmctsBot::with_evaluator(
            "ismcts",
            seed,
            SearchConfig {
                simulations,
                ..SearchConfig::default()
            },
            HeuristicEvaluator::new(DEFAULT_WEIGHTS, rollout),
        )
    }
}

impl<E: Evaluator> IsmctsBot<E> {
    /// A bot reporting `name` that searches with `eval` under `cfg`.
    pub fn with_evaluator(
        name: &'static str,
        seed: u64,
        cfg: SearchConfig,
        eval: E,
    ) -> IsmctsBot<E> {
        IsmctsBot {
            name,
            seed,
            rng: Rng::new(seed),
            cfg,
            eval,
            tracked: Tracked::Unseen,
            last: None,
            resets: 0,
            counted_in_observe: false,
            past_truncations: 0,
            searches: 0,
            buf: Vec::new(),
        }
    }

    /// The last decision's search, if it searched; taking it clears it.
    pub fn take_last_search(&mut self) -> Option<SearchResult> {
        self.last.take()
    }

    pub fn evaluator(&self) -> &E {
        &self.eval
    }

    pub fn config(&self) -> &SearchConfig {
        &self.cfg
    }

    /// Replace the tracked state, keeping the old belief's truncation count.
    fn set_tracked(&mut self, t: Tracked) {
        if let Tracked::Live(b) = std::mem::replace(&mut self.tracked, t) {
            self.past_truncations += b.truncations();
        }
    }

    /// `act` with this decision's search settings. A decision that does not search (a trade
    /// answer, a single searchable move) ignores `opts`. After a sampled move,
    /// `take_last_search().action` is the move played.
    pub fn act_with(&mut self, obs: &Observation, legal: &[Action], opts: &MoveOptions) -> Action {
        self.last = None;
        let counted_in_observe = std::mem::take(&mut self.counted_in_observe);
        match obs.phase {
            Phase::TradeResponse if legal.contains(&Action::RejectTrade) => {
                return Action::RejectTrade
            }
            Phase::TradeConfirm if legal.contains(&Action::CancelTrade) => {
                return Action::CancelTrade
            }
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
        let counted = counted_in_observe || rebuild == Some(true);
        if let Some(count) = rebuild {
            self.resets += count as u64;
            let b = Belief::from_observation(obs, DEFAULT_MAX_STATES, self.rng.next_u64());
            self.set_tracked(Tracked::Live(b));
        }
        let Tracked::Live(belief) = &self.tracked else {
            unreachable!("live after the rebuild above")
        };
        let cfg = SearchConfig {
            simulations: opts.simulations,
            ..self.cfg
        };
        let result = match opts.noise {
            Some((noise, seed)) => {
                let mut noised = Noised::new(&mut self.eval, noise, seed);
                search(obs, legal, belief, &mut noised, &cfg, &mut self.rng)
            }
            None => search(obs, legal, belief, &mut self.eval, &cfg, &mut self.rng),
        };
        match result {
            Ok(mut r) => {
                self.searches += 1;
                if let Some(a) = opts
                    .sample
                    .and_then(|seed| sample_by_visits(legal, &r.visits, seed))
                {
                    r.action = a;
                }
                let a = r.action;
                self.last = Some(r);
                a
            }
            Err(_) => {
                // At most one reset per decision: not again if `observe` or the rebuild above
                // already counted one. The next searched decision rebuilds without counting.
                if !counted {
                    self.resets += 1;
                }
                self.set_tracked(Tracked::Dropped(obs.viewer));
                greedy::choose(obs, legal)
            }
        }
    }
}

impl<E: Evaluator + Send> Bot for IsmctsBot<E> {
    fn name(&self) -> &'static str {
        self.name
    }

    /// The first events a bot sees as `viewer` start a game: the belief begins from empty hands.
    /// A contradiction counts one reset here, at the error.
    fn observe(&mut self, viewer: PlayerId, events: &[Event]) {
        match &self.tracked {
            Tracked::Live(b) if b.viewer() == viewer => {}
            Tracked::Dropped(v) if *v == viewer => return,
            _ => self.set_tracked(Tracked::Live(Belief::new(
                viewer,
                DEFAULT_MAX_STATES,
                mix(self.seed, BELIEF_SALT),
            ))),
        }
        let Tracked::Live(b) = &mut self.tracked else {
            unreachable!("live after the match above")
        };
        if b.observe(events).is_err() {
            self.resets += 1;
            self.counted_in_observe = true;
            self.set_tracked(Tracked::Dropped(viewer));
        }
    }

    fn act(&mut self, obs: &Observation, legal: &[Action]) -> Action {
        let opts = MoveOptions {
            simulations: self.cfg.simulations,
            noise: None,
            sample: None,
        };
        self.act_with(obs, legal, &opts)
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
