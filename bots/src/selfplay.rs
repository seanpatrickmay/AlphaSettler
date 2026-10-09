//! Self-play (spec Section 4): IsmctsBot in all four seats, every searched decision recorded.
//! 3a searched every move in full. 3b adds playout cap randomization (full or fast searches),
//! Dirichlet root noise on full searches and proportional move sampling in the opening, for the
//! heuristic evaluator (the warm start) and the network alike.
//! 3c: every seat may offer exactly when the game allows offers (the trading game's self-play has
//! every seat `+t`), and offers, answers and confirmations are recorded like any other searched
//! decision.
//! P2: Gumbel self-play (`Exploration::gumbel`) searches every move with the Gumbel root and
//! records its improved policy as the decision's `target`.
//! 3d-1b: with an opening model (`play_with_opening`, `run_with_opening`) the model picks every
//! setup settlement (`explore_settlement`: ε-greedy over its best spots), with no search and no
//! recorded decision; without one, self-play is unchanged.
//! 3d-3: with a league (`play_with_league`, `run_with_league`) about half the games
//! (`LEAGUE_PROBABILITY`) seat two pool bots, a past net and a scripted style, built from their
//! names by a `Roster` (`League`); their moves are neither searched by the learner nor recorded.
//! 3d-4: with `Exploration::compound`, every learner seat searches compound build moves (`+c`); a
//! compound's later steps are played unsearched and record no decision, and its visits sit on its
//! first trade's entry of the decision's `visits`. League seats play their pool names as given.

use crate::arena::{belief_resets, bot_seed, split_seeds, EventFeed, MAX_SEEDS};
use crate::heuristic::{HeuristicEvaluator, DEFAULT_WEIGHTS};
use crate::ismcts::{IsmctsBot, MoveOptions, MAX_SIMULATIONS};
use crate::opening_model::{settlement_scores, OpeningModel};
use crate::{Bot, Roster};
use settler_engine::legal::legal_actions;
use settler_engine::rng::{mix, Rng};
use settler_engine::{Action, GameConfig, Observation, Phase, PlayerId, State, NUM_PLAYERS};
use settler_search::{unit, Evaluator, RootMode, RootNoise, SearchConfig};
use std::ops::Range;
use std::path::Path;

/// Fewest simulations self-play accepts: the first simulation only evaluates the root, so with
/// one simulation every recorded visit count would be zero.
pub const MIN_SIMULATIONS: u32 = 2;
pub const FULL_SIMULATIONS: u32 = 600;
pub const FAST_SIMULATIONS: u32 = 100;
pub const FULL_PROBABILITY: f64 = 0.25;
pub const NOISE: RootNoise = RootNoise {
    weight: 0.25,
    alpha_total: 10.0,
};
pub const SAMPLE_TURNS: u32 = 8;
/// 3d-1b (spec Section 2): with an opening model, the chance a setup settlement is drawn from
/// the model's `OPENING_TOP` best spots instead of being its best spot.
pub const OPENING_EPSILON: f64 = 0.25;
pub const OPENING_TOP: usize = 5;
/// 3d-3 (spec Section 2): with a league, the chance a self-play game is a league game, and the
/// league seats of one (a past net and a style; the other seats are the learner's).
pub const LEAGUE_PROBABILITY: f64 = 0.5;
pub const LEAGUE_SEATS: usize = 2;
const EXPLORE_SALT: u64 = 0xE7B1;
const LEAGUE_SALT: u64 = 0x1EA6;

#[derive(Clone, Debug, PartialEq)]
pub struct Decision {
    /// Index into the game's `actions` of the move this search chose.
    pub index: u32,
    pub actor: PlayerId,
    /// Every legal action (domestic offers included), in engine order.
    pub legal: Vec<Action>,
    /// Root visits aligned with `legal`.
    pub visits: Vec<u32>,
    /// A full search (a policy target); false for a fast search (a value target only).
    pub full_search: bool,
    /// Gumbel self-play: the search's improved policy, aligned with `legal` and summing to 1
    /// (spec P2 Section 5); `None` otherwise.
    pub target: Option<Vec<f32>>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SelfPlayGame {
    pub seed: u64,
    pub actions: Vec<Action>,
    pub decisions: Vec<Decision>,
    pub winner: Option<PlayerId>,
    pub vp: [u8; NUM_PLAYERS],
    pub turns: u32,
    /// Belief resets over all four bots at the game's end; must be 0 (spec Section 2).
    pub belief_resets: u64,
    /// 3d-3: a league game's seats (spec Section 2), per seat the pool bot's name or `None` for a
    /// learner seat; `None` for a plain game.
    pub league: Option<[Option<String>; NUM_PLAYERS]>,
}

/// How self-play searches and picks its moves.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Exploration {
    pub full_simulations: u32,
    pub fast_simulations: u32,
    /// Chance that a move gets a full search; the rest get fast searches.
    pub full_probability: f64,
    /// Root noise, on full searches only.
    pub noise: Option<RootNoise>,
    /// Moves made while the engine's turn counter is below this (setup is turn 0) are drawn in
    /// proportion to root visits; later moves are the most-visited.
    pub sample_turns: u32,
    /// Gumbel root search on every move (spec P2 Section 5), with Gumbel draws, each a full
    /// search of `full_simulations` recorded with its improved policy. Noise, playout cap
    /// randomization and visit sampling must be off: the draw is the exploration (`validate`).
    pub gumbel: bool,
    /// 3d-4: every learner seat searches compound build moves (`+c`); league seats keep their pool
    /// names. Not with `gumbel` (`validate`).
    pub compound: bool,
}

impl Exploration {
    /// 3a: every move a full search of `simulations`, no noise, always the most-visited move.
    pub fn fixed(simulations: u32) -> Exploration {
        Exploration {
            full_simulations: simulations,
            fast_simulations: simulations,
            full_probability: 1.0,
            noise: None,
            sample_turns: 0,
            gumbel: false,
            compound: false,
        }
    }

    /// 3b: 600 / 100 simulations with p = 0.25, noise 0.25 · Dir(10 / legal), 8 sampled turns.
    pub fn standard() -> Exploration {
        Exploration {
            full_simulations: FULL_SIMULATIONS,
            fast_simulations: FAST_SIMULATIONS,
            full_probability: FULL_PROBABILITY,
            noise: Some(NOISE),
            sample_turns: SAMPLE_TURNS,
            gumbel: false,
            compound: false,
        }
    }

    /// P2: every move a Gumbel search of `simulations` with Gumbel draws; every search full, no
    /// noise, no sampled turns.
    pub fn gumbel(simulations: u32) -> Exploration {
        Exploration {
            full_simulations: simulations,
            fast_simulations: simulations,
            full_probability: 1.0,
            noise: None,
            sample_turns: 0,
            gumbel: true,
            compound: false,
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        for n in [self.full_simulations, self.fast_simulations] {
            if n < MIN_SIMULATIONS {
                return Err(format!(
                    "self-play needs at least {MIN_SIMULATIONS} simulations per search, got {n}"
                ));
            }
            if n > MAX_SIMULATIONS {
                return Err(format!(
                    "self-play allows at most {MAX_SIMULATIONS} simulations per search, got {n}"
                ));
            }
        }
        if !(0.0..=1.0).contains(&self.full_probability) {
            return Err(format!(
                "the full-search probability must be in [0, 1], got {}",
                self.full_probability
            ));
        }
        if self.gumbel
            && (self.noise.is_some()
                || self.sample_turns > 0
                || self.full_probability != 1.0
                || self.fast_simulations != self.full_simulations)
        {
            return Err(
                "Gumbel self-play searches every move in full, with no root noise and no \
                        visit sampling: the Gumbel draw is the exploration"
                    .into(),
            );
        }
        if self.compound && self.gumbel {
            return Err(
                "compound moves are not searched with Gumbel self-play: the Gumbel root does not \
                 take them (spec 3d-4 Section 4)"
                    .into(),
            );
        }
        Ok(())
    }
}

/// The setup settlement self-play plays with an opening model (spec 3d-1b Section 2): every legal
/// settlement scored as `+o` scores it (`settlement_scores`); with probability
/// `1 - OPENING_EPSILON` the best (ties to the first in engine order: `choose_settlement`'s
/// spot), else one drawn uniformly from the `OPENING_TOP` best (all of them if fewer are legal).
/// Both draws come from `rng`, the game's exploration stream.
pub fn explore_settlement(
    model: &OpeningModel,
    obs: &Observation,
    legal: &[Action],
    rng: &mut Rng,
) -> Action {
    let mut scored = settlement_scores(model, obs, legal);
    // A stable sort, best first: equal scores keep engine order.
    scored.sort_by(|a, b| b.1.total_cmp(&a.1));
    let explore = unit(rng) < OPENING_EPSILON;
    match scored.len() {
        0 => legal[0],
        n if explore => scored[rng.below(n.min(OPENING_TOP) as u32) as usize].0,
        _ => scored[0].0,
    }
}

/// A league's pool (spec 3d-3 Section 3): bot names of past nets and of scripted styles, checked
/// and their files loaded once by a `Roster`, shared read-only by every game.
pub struct League {
    nets: Vec<String>,
    styles: Vec<String>,
    roster: Roster,
}

impl League {
    /// Both halves need a name, and every name must be a bot whose files load from `nets_dir`
    /// (the `Roster` error names the first that is not).
    pub fn new(nets: &[&str], styles: &[&str], nets_dir: &Path) -> Result<League, String> {
        fn owned(names: &[&str]) -> Vec<String> {
            names.iter().map(|n| n.to_string()).collect()
        }
        if nets.is_empty() {
            return Err("a league needs at least one past net".into());
        }
        if styles.is_empty() {
            return Err("a league needs at least one style".into());
        }
        let names: Vec<&str> = nets.iter().chain(styles).copied().collect();
        let roster = Roster::new(&names, nets_dir)?;
        Ok(League {
            nets: owned(nets),
            styles: owned(styles),
            roster,
        })
    }

    pub fn nets(&self) -> &[String] {
        &self.nets
    }

    pub fn styles(&self) -> &[String] {
        &self.styles
    }

    /// Game `seed`'s league seats: `None` for a plain game (probability
    /// `1 - LEAGUE_PROBABILITY`), else per seat the pool bot's name (`None` at a learner seat):
    /// two seats drawn uniformly from the four, the first a past net and the second a style,
    /// each uniform over its half. Every draw comes from the game's league stream, so the seats
    /// depend on the seed and the pool alone.
    pub fn draw(&self, seed: u64) -> Option<[Option<&str>; NUM_PLAYERS]> {
        let mut rng = Rng::new(mix(seed, LEAGUE_SALT));
        if unit(&mut rng) >= LEAGUE_PROBABILITY {
            return None;
        }
        let first = rng.below(NUM_PLAYERS as u32) as usize;
        let mut second = rng.below(NUM_PLAYERS as u32 - 1) as usize;
        if second >= first {
            second += 1;
        }
        let mut seats = [None; NUM_PLAYERS];
        seats[first] = Some(self.nets[rng.below(self.nets.len() as u32) as usize].as_str());
        seats[second] = Some(self.styles[rng.below(self.styles.len() as u32) as usize].as_str());
        Some(seats)
    }

    fn make(&self, name: &str, seed: u64) -> Box<dyn Bot> {
        self.roster
            .make(name, seed)
            .expect("every pool name was checked by League::new")
    }
}

/// One self-play game; `make_eval` builds each seat's evaluator, `batch` is the leaf batch.
pub fn play_with<E, F>(
    seed: u64,
    config: GameConfig,
    ex: &Exploration,
    batch: usize,
    make_eval: &F,
) -> SelfPlayGame
where
    E: Evaluator + Send,
    F: Fn() -> E,
{
    play_with_opening(seed, config, ex, batch, make_eval, None)
}

/// `play_with`, and with `opening` the model picks every setup settlement
/// (`explore_settlement`, spec 3d-1b Section 2): no search runs and no decision is recorded for
/// those moves. Without a model this is `play_with` exactly.
pub fn play_with_opening<E, F>(
    seed: u64,
    config: GameConfig,
    ex: &Exploration,
    batch: usize,
    make_eval: &F,
    opening: Option<&OpeningModel>,
) -> SelfPlayGame
where
    E: Evaluator + Send,
    F: Fn() -> E,
{
    play_with_league(seed, config, ex, batch, make_eval, opening, None)
}

/// `play_with_opening`, and with `league` the game may be a league game (`League::draw`): its two
/// league seats play their pool bots, whose moves are not recorded (spec 3d-3 Section 2). Without
/// a league, and in a plain game, this is `play_with_opening` exactly.
pub fn play_with_league<E, F>(
    seed: u64,
    config: GameConfig,
    ex: &Exploration,
    batch: usize,
    make_eval: &F,
    opening: Option<&OpeningModel>,
    league: Option<&League>,
) -> SelfPlayGame
where
    E: Evaluator + Send,
    F: Fn() -> E,
{
    // Every seat searches offers exactly when the game allows them (spec 3c Section 5).
    let cfg = SearchConfig {
        simulations: ex.full_simulations,
        batch,
        offers: config.max_offers_per_turn > 0,
        // Gumbel self-play (spec P2 Section 5): every search draws its own Gumbel noise.
        root: if ex.gumbel {
            RootMode::Gumbel { draw: true }
        } else {
            RootMode::Puct
        },
        // 3d-4: compound build moves for every learner seat, each bot playing its compounds
        // through. League seats play their pool bots as their names say (no `+c`, Ruling).
        compound: ex.compound,
        ..SearchConfig::default()
    };
    let mut bots: Vec<IsmctsBot<E>> = (0..NUM_PLAYERS)
        .map(|p| IsmctsBot::with_evaluator("selfplay", bot_seed(seed, p), cfg, make_eval()))
        .collect();
    // 3d-3: this game's league seats, from the league's own stream (a plain game has none). The
    // learner bot of a league seat is never shown an event or asked to move.
    let seats: [Option<&str>; NUM_PLAYERS] = league
        .and_then(|l| l.draw(seed))
        .unwrap_or([None; NUM_PLAYERS]);
    let mut pool: Vec<Option<Box<dyn Bot>>> = seats
        .iter()
        .enumerate()
        .map(|(p, name)| {
            name.map(|n| {
                league
                    .expect("league seats are drawn from a league")
                    .make(n, bot_seed(seed, p))
            })
        })
        .collect();
    let mut rng = Rng::new(mix(seed, EXPLORE_SALT));
    let mut s = State::new(seed, config);
    let (mut feed, mut buf, mut step) = (EventFeed::default(), Vec::new(), Vec::new());
    let (mut actions, mut decisions) = (Vec::new(), Vec::new());
    while !s.is_over() {
        legal_actions(&s, &mut buf);
        let actor = s.current_actor() as usize;
        let a = if let Some(bot) = pool[actor].as_mut() {
            // 3d-3: a league seat plays its own bot (its name's search and budget). Its moves are
            // no policy targets: nothing is recorded and the exploration stream is not drawn.
            feed.deliver(actor, &mut **bot);
            bot.act(&s.observation(actor as PlayerId), &buf)
        } else {
            feed.deliver(actor, &mut bots[actor]);
            match (s.phase, opening) {
                // 3d-1b: the model's pick, from this game's exploration stream. The seat's bot
                // has its events (above) but does not decide, so nothing is searched or recorded.
                (Phase::SetupSettlement, Some(model)) => {
                    explore_settlement(model, &s.observation(actor as PlayerId), &buf, &mut rng)
                }
                _ => {
                    let full = unit(&mut rng) < ex.full_probability;
                    let opts = MoveOptions {
                        simulations: if full {
                            ex.full_simulations
                        } else {
                            ex.fast_simulations
                        },
                        noise: match ex.noise {
                            Some(n) if full => Some((n, rng.next_u64())),
                            _ => None,
                        },
                        sample: (s.turn < ex.sample_turns).then(|| rng.next_u64()),
                    };
                    let a = bots[actor].act_with(&s.observation(actor as PlayerId), &buf, &opts);
                    if let Some(r) = bots[actor].take_last_search() {
                        decisions.push(Decision {
                            index: actions.len() as u32,
                            actor: actor as PlayerId,
                            legal: buf.clone(),
                            visits: r.visits,
                            full_search: full,
                            target: r.improved,
                        });
                    }
                    a
                }
            }
        };
        assert!(
            buf.contains(&a),
            "self-play chose illegal action {a:?} in phase {:?}",
            s.phase
        );
        actions.push(a);
        step.clear();
        s.apply_with(a, None, &mut step);
        feed.push(&step);
    }
    SelfPlayGame {
        seed,
        actions,
        decisions,
        winner: s.winner(),
        vp: std::array::from_fn(|p| s.total_vp(p)),
        turns: s.turn,
        belief_resets: belief_resets(bots.iter().map(|b| b as &dyn Bot))
            + belief_resets(pool.iter().flatten().map(|b| b.as_ref())),
        league: seats
            .iter()
            .any(Option::is_some)
            .then(|| seats.map(|n| n.map(str::to_owned))),
    }
}

/// Every seed in `seeds`, spread over `threads`; sorted by seed and independent of `threads`.
pub fn run_with<E, F>(
    seeds: Range<u64>,
    config: GameConfig,
    ex: &Exploration,
    batch: usize,
    make_eval: &F,
    threads: usize,
) -> Result<Vec<SelfPlayGame>, String>
where
    E: Evaluator + Send,
    F: Fn() -> E + Sync,
{
    run_with_opening(seeds, config, ex, batch, make_eval, threads, None)
}

/// `run_with` with `play_with_opening`'s `opening`. A model with Gumbel self-play is refused
/// (spec 3d-1b Section 3).
pub fn run_with_opening<E, F>(
    seeds: Range<u64>,
    config: GameConfig,
    ex: &Exploration,
    batch: usize,
    make_eval: &F,
    threads: usize,
    opening: Option<&OpeningModel>,
) -> Result<Vec<SelfPlayGame>, String>
where
    E: Evaluator + Send,
    F: Fn() -> E + Sync,
{
    run_with_league(seeds, config, ex, batch, make_eval, threads, opening, None)
}

/// `run_with_opening` with `play_with_league`'s `league`. A league with Gumbel self-play is
/// refused (spec 3d-3 Section 4), as a model is.
#[allow(clippy::too_many_arguments)]
pub fn run_with_league<E, F>(
    seeds: Range<u64>,
    config: GameConfig,
    ex: &Exploration,
    batch: usize,
    make_eval: &F,
    threads: usize,
    opening: Option<&OpeningModel>,
    league: Option<&League>,
) -> Result<Vec<SelfPlayGame>, String>
where
    E: Evaluator + Send,
    F: Fn() -> E + Sync,
{
    config.validate()?;
    ex.validate()?;
    if ex.gumbel && opening.is_some() {
        return Err(
            "the opening model is not used with Gumbel self-play: Gumbel self-play \
                    stays the P2 measurement it was (spec 3d-1b Section 3)"
                .into(),
        );
    }
    if ex.gumbel && league.is_some() {
        return Err(
            "a league is not used with Gumbel self-play: Gumbel self-play stays the P2 \
                    measurement it was (spec 3d-3 Section 4)"
                .into(),
        );
    }
    if batch == 0 {
        return Err("the leaf batch must be positive".into());
    }
    let n = seeds.end.saturating_sub(seeds.start);
    if n > MAX_SEEDS {
        return Err(format!("at most {MAX_SEEDS} games per call, got {n}"));
    }
    let mut games: Vec<SelfPlayGame> = std::thread::scope(|scope| {
        let workers: Vec<_> = split_seeds(seeds, threads)
            .into_iter()
            .map(|part| {
                scope.spawn(move || {
                    part.map(|seed| {
                        play_with_league(seed, config, ex, batch, make_eval, opening, league)
                    })
                    .collect::<Vec<_>>()
                })
            })
            .collect();
        workers
            .into_iter()
            .flat_map(|w| {
                w.join()
                    .unwrap_or_else(|panic| std::panic::resume_unwind(panic))
            })
            .collect()
    });
    games.sort_by_key(|g| g.seed);
    Ok(games)
}

/// 3a self-play: the heuristic bot, every move a full search of `simulations`.
pub fn play(seed: u64, config: GameConfig, simulations: u32, rollout: u32) -> SelfPlayGame {
    let ex = Exploration::fixed(simulations);
    play_with(seed, config, &ex, SearchConfig::default().batch, &|| {
        HeuristicEvaluator::new(DEFAULT_WEIGHTS, rollout)
    })
}

/// 3a self-play over `seeds` (see `run_with`).
pub fn run(
    seeds: Range<u64>,
    config: GameConfig,
    simulations: u32,
    rollout: u32,
    threads: usize,
) -> Result<Vec<SelfPlayGame>, String> {
    let ex = Exploration::fixed(simulations);
    run_with(
        seeds,
        config,
        &ex,
        SearchConfig::default().batch,
        &|| HeuristicEvaluator::new(DEFAULT_WEIGHTS, rollout),
        threads,
    )
}
