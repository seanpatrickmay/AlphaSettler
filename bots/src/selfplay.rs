//! Self-play: IsmctsBot in all four seats, recording every searched decision (spec Section 4).

use crate::arena::{bot_seed, split_seeds, EventFeed, MAX_SEEDS};
use crate::ismcts::IsmctsBot;
use crate::Bot;
use settler_engine::legal::legal_actions;
use settler_engine::{Action, GameConfig, PlayerId, State, NUM_PLAYERS};
use std::ops::Range;

/// Fewest simulations self-play accepts: the first simulation only evaluates the root, so with
/// one simulation every recorded visit count would be zero.
pub const MIN_SIMULATIONS: u32 = 2;

#[derive(Clone, Debug, PartialEq)]
pub struct Decision {
    /// Index into the game's `actions` of the move this search chose.
    pub index: u32,
    pub actor: PlayerId,
    /// Every legal action (domestic offers included), in engine order.
    pub legal: Vec<Action>,
    /// Root visits aligned with `legal`.
    pub visits: Vec<u32>,
    /// Always true in 3a; 3b's playout cap randomization records fast searches as false.
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
}

pub fn play(seed: u64, config: GameConfig, simulations: u32, rollout: u32) -> SelfPlayGame {
    let mut bots: Vec<IsmctsBot> = (0..NUM_PLAYERS)
        .map(|p| IsmctsBot::new(bot_seed(seed, p), simulations, rollout))
        .collect();
    let mut s = State::new(seed, config);
    let (mut feed, mut buf, mut step) = (EventFeed::default(), Vec::new(), Vec::new());
    let (mut actions, mut decisions) = (Vec::new(), Vec::new());
    while !s.is_over() {
        legal_actions(&s, &mut buf);
        let actor = s.current_actor() as usize;
        feed.deliver(actor, &mut bots[actor]);
        let a = bots[actor].act(&s.observation(actor as PlayerId), &buf);
        assert!(
            buf.contains(&a),
            "ismcts chose illegal action {a:?} in phase {:?}",
            s.phase
        );
        if let Some(r) = bots[actor].take_last_search() {
            decisions.push(Decision {
                index: actions.len() as u32,
                actor: actor as PlayerId,
                legal: buf.clone(),
                visits: r.visits,
                full_search: true,
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
    }
}

/// Every seed in `seeds`, spread over `threads`; sorted by seed and independent of `threads`.
pub fn run(
    seeds: Range<u64>,
    config: GameConfig,
    simulations: u32,
    rollout: u32,
    threads: usize,
) -> Result<Vec<SelfPlayGame>, String> {
    config.validate()?;
    if simulations < MIN_SIMULATIONS {
        return Err(format!(
            "self-play needs at least {MIN_SIMULATIONS} simulations per search, got {simulations}"
        ));
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
                    part.map(|seed| play(seed, config, simulations, rollout))
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
