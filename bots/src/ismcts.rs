//! IsmctsBot: tracks hidden cards from the event log and searches each real decision with
//! single-observer ISMCTS (spec Sections 2-3), domestic-trade answers and confirmations included
//! (spec 3c Section 3). It makes offers only with `with_offers` (a `+t` name), and with
//! `with_opening` (a `net:` name's `+o`) an opening model picks its setup settlements without a
//! search (spec 3d-1 Section 4.5). Generic over the evaluator: the heuristic for `ismcts` names,
//! the network for `net:` names (`crate::netbot`).
//! 3d-4: with `with_compound` (a `+c` name) the search's move set holds compound build moves; the
//! bot plays a compound's first step and its later ones without a search while the game goes as
//! planned (`plan_step`).

use crate::greedy;
use crate::heuristic::{HeuristicEvaluator, DEFAULT_WEIGHTS};
use crate::opening_model::{choose_settlement, OpeningModel};
use crate::Bot;
use settler_engine::rng::{mix, Rng};
use settler_engine::types::{hand_total, NUM_RESOURCES};
use settler_engine::{Action, Event, Observation, Phase, PlayerId};
use settler_search::{
    search, search_moves, Belief, Evaluator, Noised, RootMode, RootNoise, SearchConfig,
    SearchResult, DEFAULT_MAX_STATES,
};
use std::sync::Arc;

pub const DEFAULT_SIMULATIONS: u32 = 1000;
pub const MAX_SIMULATIONS: u32 = 1_000_000;
pub const MAX_ROLLOUT: u32 = 200;
/// Most a `+sK` suffix multiplies setup settlements' simulations by.
pub const MAX_SETUP_FACTOR: u32 = 100;
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

/// A searching bot's name with an optional `+sK` suffix split off: `ismcts@100+s10` ->
/// ("ismcts@100", 10), a name without one -> (name, 1). K is digits only, 2..=MAX_SETUP_FACTOR.
pub fn split_setup(name: &str) -> Option<(&str, u32)> {
    let Some(i) = name.rfind("+s") else {
        return Some((name, 1));
    };
    let k = &name[i + 2..];
    if k.is_empty() || !k.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    let k: u32 = k.parse().ok()?;
    (2..=MAX_SETUP_FACTOR)
        .contains(&k)
        .then_some((&name[..i], k))
}

/// The flag suffixes of a searching bot's name.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Flags {
    /// `+t`: the bot may make domestic offers.
    pub offers: bool,
    /// `+o`: an opening model picks the setup settlements (`net:` names only).
    pub opening: bool,
}

/// A searching bot's name with its `+t` and `+o` suffixes removed and its `+rD` / `+sK` suffixes
/// put in the order `parse_name` and `split_setup` read (`+rD` before `+sK`), and which flags it
/// had. The suffixes are everything from the first `+`, in any order, each kind at most once;
/// `None` for a repeated or unknown one (`+t5`, `+T`, `+oo`, `+g` (split off first by
/// `split_gumbel`), a bare `+`). Values are checked by those parsers.
pub fn split_suffixes(name: &str) -> Option<(String, Flags)> {
    let Some(i) = name.find('+') else {
        return Some((name.to_owned(), Flags::default()));
    };
    let (mut rollout, mut setup, mut flags) = (None, None, Flags::default());
    for part in name[i + 1..].split('+') {
        match part.as_bytes().first() {
            Some(b'r') if rollout.is_none() => rollout = Some(part),
            Some(b's') if setup.is_none() => setup = Some(part),
            Some(b't') if part.len() == 1 && !flags.offers => flags.offers = true,
            Some(b'o') if part.len() == 1 && !flags.opening => flags.opening = true,
            _ => return None,
        }
    }
    let mut out = name[..i].to_owned();
    for part in [rollout, setup].into_iter().flatten() {
        out.push('+');
        out.push_str(part);
    }
    Some((out, flags))
}

/// `split_suffixes` for bots that take no opening model (`ismcts` names): the name and whether it
/// had `+t`; `None` if it has `+o`.
pub fn split_trades(name: &str) -> Option<(String, bool)> {
    let (canon, flags) = split_suffixes(name)?;
    (!flags.opening).then_some((canon, flags.offers))
}

/// `name` with every `+FLAG` part removed (`flag` is the part after the `+`), and whether it had
/// one; `None` for a repeat. Every other part is kept in order.
fn split_flag(name: &str, flag: &str) -> Option<(String, bool)> {
    let Some(i) = name.find('+') else {
        return Some((name.to_owned(), false));
    };
    let (mut out, mut found) = (name[..i].to_owned(), false);
    for part in name[i + 1..].split('+') {
        if part == flag {
            if found {
                return None;
            }
            found = true;
        } else {
            out.push('+');
            out.push_str(part);
        }
    }
    Some((out, found))
}

/// A searching bot's name with its `+g` suffix (spec P2 Section 4: Gumbel root search, no
/// draws) removed, and whether it had one. `+g` may sit anywhere among the suffixes, at most once
/// (`None` for a repeat); every other part is kept in order for `split_trades` / `split_suffixes`,
/// which refuse any other `g` part (`+g2`, `+gt`) and anything malformed.
pub fn split_gumbel(name: &str) -> Option<(String, bool)> {
    split_flag(name, "g")
}

/// A searching bot's name with its `+c` suffix (spec 3d-4 Section 4: compound build moves)
/// removed, and whether it had one; anywhere among the suffixes, at most once.
pub fn split_compound(name: &str) -> Option<(String, bool)> {
    split_flag(name, "c")
}

/// `split_compound`, then `split_gumbel`: the name for `split_trades` / `parse_net_name`, whether
/// it had `+g` and whether it had `+c`. `None` for a repeat, and for both: compound moves are not
/// searched with a Gumbel root (spec 3d-4 Section 4).
pub fn split_search_flags(name: &str) -> Option<(String, bool, bool)> {
    let (rest, compound) = split_compound(name)?;
    let (rest, gumbel) = split_gumbel(&rest)?;
    (!(compound && gumbel)).then_some((rest, gumbel, compound))
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
    sample_move(legal, visits, &[], seed).map(|(a, _)| a)
}

/// `sample_by_visits`'s draw, with compounds (spec 3d-4 Section 3): a compound's visits sit on its
/// first trade's entry, so a draw landing there picks among the compounds starting with that trade
/// by their visits, and returns the compound's first step with its other steps. Without compounds
/// (or on any other entry) the plan is empty.
pub fn sample_move(
    legal: &[Action],
    visits: &[u32],
    compounds: &[(Vec<Action>, u32)],
    seed: u64,
) -> Option<(Action, Vec<Action>)> {
    let total: u64 = visits.iter().map(|&v| v as u64).sum();
    if total == 0 {
        return None;
    }
    let mut x = ((Rng::new(seed).next_u64() as u128 * total as u128) >> 64) as u64;
    for (a, &v) in legal.iter().zip(visits) {
        if x < v as u64 {
            for (steps, cv) in compounds.iter().filter(|(m, _)| m.first() == Some(a)) {
                if x < *cv as u64 {
                    let (first, rest) = steps.split_first().expect("a compound has steps");
                    return Some((*first, rest.to_vec()));
                }
                x -= *cv as u64;
            }
            debug_assert!(
                compounds.iter().all(|(m, _)| m.first() != Some(a)),
                "a trade's visits come from its compounds"
            );
            return Some((*a, Vec::new()));
        }
        x -= v as u64;
    }
    unreachable!("x is below the visit total")
}

/// Whether `events` are exactly `viewer`'s own maritime trade `a`: one `MaritimeTraded` by its
/// seat giving only cards of `a`'s resource and getting one card of the other.
fn is_own_trade(events: &[Event], viewer: PlayerId, a: Action) -> bool {
    let Action::MaritimeTrade { give, get } = a else {
        return false;
    };
    match events {
        [Event::MaritimeTraded { player, gave, got }] => {
            let mut one = [0u8; NUM_RESOURCES];
            one[get.index()] = 1;
            *player == viewer
                && *got == one
                && gave[give.index()] > 0
                && hand_total(gave) == gave[give.index()] as u32
        }
        _ => false,
    }
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
    /// Setup settlements search `cfg.simulations` x this (1: no extra search).
    setup_factor: u32,
    /// `+o`: picks setup settlements without a search.
    opening: Option<Arc<OpeningModel>>,
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
    /// 3d-4: the steps of the compound under way still to play, the step last played (whose event
    /// the next decision expects) and the events seen since it.
    plan: Vec<Action>,
    planned: Option<Action>,
    since: Vec<Event>,
    plan_steps: u64,
    /// Searched decisions whose move set held a compound move.
    compound_decisions: u64,
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
            setup_factor: 1,
            opening: None,
            eval,
            tracked: Tracked::Unseen,
            last: None,
            resets: 0,
            counted_in_observe: false,
            past_truncations: 0,
            searches: 0,
            buf: Vec::new(),
            plan: Vec::new(),
            planned: None,
            since: Vec::new(),
            plan_steps: 0,
            compound_decisions: 0,
        }
    }

    /// Search setup settlements with `factor` (1..=MAX_SETUP_FACTOR) times the simulations of
    /// every other decision, capped at MAX_SIMULATIONS. Only `act` applies it; self-play's
    /// `act_with` takes its budget from `MoveOptions`.
    pub fn with_setup_factor(mut self, factor: u32) -> IsmctsBot<E> {
        assert!(
            (1..=MAX_SETUP_FACTOR).contains(&factor),
            "setup factor {factor} out of range"
        );
        self.setup_factor = factor;
        self
    }

    /// Make domestic offers searched moves (a `+t` name). Without it the bot still answers and
    /// confirms offers by searching, but never offers, and assumes nobody does.
    pub fn with_offers(mut self, offers: bool) -> IsmctsBot<E> {
        self.cfg.offers = offers;
        self
    }

    /// Search the root by Gumbel sequential halving with no Gumbel draws (a `+g` name, spec P2
    /// Section 4); below the root the search is PUCT as before. Off, the root is PUCT.
    pub fn with_gumbel(mut self, gumbel: bool) -> IsmctsBot<E> {
        self.cfg.root = if gumbel {
            RootMode::Gumbel { draw: false }
        } else {
            RootMode::Puct
        };
        self
    }

    /// Search compound build moves in place of bare maritime trades at the bot's own Main
    /// decisions (a `+c` name, spec 3d-4), and play a chosen compound's later steps unsearched
    /// while the game goes as planned. Never with `with_gumbel(true)` (the search refuses it).
    pub fn with_compound(mut self, compound: bool) -> IsmctsBot<E> {
        self.cfg.compound = compound;
        self
    }

    /// The next step of the compound under way, if the game went as planned since the bot's last
    /// step (spec 3d-4 Section 3): the events since then are exactly that step's own trade, and the
    /// next step is legal now. Otherwise the plan is dropped and `None` returned.
    fn plan_step(
        &mut self,
        obs: &Observation,
        legal: &[Action],
        since: &[Event],
    ) -> Option<Action> {
        let next = *self.plan.first()?;
        let played = self.planned.take();
        let on_plan =
            played.is_some_and(|a| is_own_trade(since, obs.viewer, a)) && legal.contains(&next);
        if !on_plan {
            self.plan.clear();
            return None;
        }
        self.plan.remove(0);
        self.planned = (!self.plan.is_empty()).then_some(next);
        Some(next)
    }

    /// Keep `plan`, the steps after `played`, as the compound under way (none for a plain move).
    fn start_plan(&mut self, played: Action, plan: Vec<Action>) {
        self.planned = (!plan.is_empty()).then_some(played);
        self.plan = plan;
    }

    /// Let `model` pick every setup settlement (a `+o` name): no search runs for them; every
    /// other decision is unchanged. `act` applies it; self-play's `act_with` does not.
    pub fn with_opening(mut self, model: Arc<OpeningModel>) -> IsmctsBot<E> {
        self.opening = Some(model);
        self
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

    /// `act` with this decision's search settings. A decision that does not search (a single
    /// searchable move, or a compound's later step) ignores `opts`. After a searched move,
    /// `take_last_search().action` is the move played and `.plan` its later steps.
    pub fn act_with(&mut self, obs: &Observation, legal: &[Action], opts: &MoveOptions) -> Action {
        self.last = None;
        let counted_in_observe = std::mem::take(&mut self.counted_in_observe);
        // 3d-4: the next step of a compound under way, unsearched, while the game went as planned.
        let since = std::mem::take(&mut self.since);
        if let Some(a) = self.plan_step(obs, legal, &since) {
            self.plan_steps += 1;
            return a;
        }
        search_moves(legal, self.cfg.offers, &mut self.buf);
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
                if r.simulations == 0 {
                    // Only a compound move set comes back unsearched here: one move once the bare
                    // trades are gone (spec 3d-4). Like a single searchable move, it is played
                    // without counting a search and records nothing.
                    self.start_plan(r.action, r.plan);
                    return r.action;
                }
                self.searches += 1;
                self.compound_decisions += u64::from(!r.compound_visits.is_empty());
                if let Some((a, plan)) = opts
                    .sample
                    .and_then(|seed| sample_move(legal, &r.visits, &r.compound_visits, seed))
                {
                    r.action = a;
                    r.plan = plan;
                }
                let a = r.action;
                self.start_plan(a, r.plan.clone());
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
        // 3d-4: what happened since a compound's last step decides whether its plan goes on.
        if !self.plan.is_empty() {
            self.since.extend_from_slice(events);
        }
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
        if let (Phase::SetupSettlement, Some(model)) = (obs.phase, self.opening.clone()) {
            // As `act_with` does for a decision it does not search.
            self.last = None;
            self.counted_in_observe = false;
            return choose_settlement(&model, obs, legal);
        }
        let factor = if obs.phase == Phase::SetupSettlement {
            self.setup_factor
        } else {
            1
        };
        let opts = MoveOptions {
            simulations: self
                .cfg
                .simulations
                .saturating_mul(factor)
                .min(MAX_SIMULATIONS),
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
            // 1 for a `+g` bot (Gumbel root search), 0 for a PUCT root.
            (
                "gumbel_root",
                matches!(self.cfg.root, RootMode::Gumbel { .. }) as u64,
            ),
            // 3d-4: 1 for a `+c` bot; its searched decisions whose move set held a compound move;
            // the compound steps it played without a search.
            ("compound", self.cfg.compound as u64),
            ("compound_decisions", self.compound_decisions),
            ("plan_steps", self.plan_steps),
        ]
    }
}
