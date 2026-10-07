//! Self-play (spec Section 4): IsmctsBot in all four seats, every searched decision recorded.
//! 3a searched every move in full. 3b adds playout cap randomization (full or fast searches),
//! Dirichlet root noise on full searches and proportional move sampling in the opening, for the
//! heuristic evaluator (the warm start) and the network alike.
//! 3c: every seat may offer exactly when the game allows offers (the trading game's self-play has
//! every seat `+t`), and offers, answers and confirmations are recorded like any other searched
//! decision.

use crate::arena::{belief_resets, bot_seed, split_seeds, EventFeed, MAX_SEEDS};
use crate::heuristic::{HeuristicEvaluator, DEFAULT_WEIGHTS};
use crate::ismcts::{IsmctsBot, MoveOptions, MAX_SIMULATIONS};
use crate::Bot;
use settler_engine::legal::legal_actions;
use settler_engine::rng::{mix, Rng};
use settler_engine::{Action, GameConfig, PlayerId, State, NUM_PLAYERS};
use settler_search::{unit, Evaluator, RootNoise, SearchConfig};
use std::ops::Range;

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
const EXPLORE_SALT: u64 = 0xE7B1;

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
        Ok(())
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
    // Every seat searches offers exactly when the game allows them (spec 3c Section 5).
    let cfg = SearchConfig {
        simulations: ex.full_simulations,
        batch,
        offers: config.max_offers_per_turn > 0,
        ..SearchConfig::default()
    };
    let mut bots: Vec<IsmctsBot<E>> = (0..NUM_PLAYERS)
        .map(|p| IsmctsBot::with_evaluator("selfplay", bot_seed(seed, p), cfg, make_eval()))
        .collect();
    let mut rng = Rng::new(mix(seed, EXPLORE_SALT));
    let mut s = State::new(seed, config);
    let (mut feed, mut buf, mut step) = (EventFeed::default(), Vec::new(), Vec::new());
    let (mut actions, mut decisions) = (Vec::new(), Vec::new());
    while !s.is_over() {
        legal_actions(&s, &mut buf);
        let actor = s.current_actor() as usize;
        feed.deliver(actor, &mut bots[actor]);
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
        assert!(
            buf.contains(&a),
            "self-play chose illegal action {a:?} in phase {:?}",
            s.phase
        );
        if let Some(r) = bots[actor].take_last_search() {
            decisions.push(Decision {
                index: actions.len() as u32,
                actor: actor as PlayerId,
                legal: buf.clone(),
                visits: r.visits,
                full_search: full,
            });
        }
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
        belief_resets: belief_resets(bots.iter().map(|b| b as &dyn Bot)),
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
    config.validate()?;
    ex.validate()?;
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
                    part.map(|seed| play_with(seed, config, ex, batch, make_eval))
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
