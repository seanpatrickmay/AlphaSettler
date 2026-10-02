//! Native arena: one candidate against three copies of a baseline, each seed played four
//! times with the candidate rotated through every seat.

use crate::{make_bot, Bot, BOT_NAMES};
use settler_engine::legal::legal_actions;
use settler_engine::rng::mix;
use settler_engine::{Event, GameConfig, PlayerId, State, NUM_PLAYERS};
use std::ops::Range;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GameRecord {
    pub seed: u64,
    pub candidate_seat: u8,
    pub winner: Option<u8>,
    /// Final victory points including hidden VP cards.
    pub vp: [u8; NUM_PLAYERS],
    pub turns: u32,
    pub actions: u32,
}

/// Largest seed range one `run_match` call accepts.
pub const MAX_SEEDS: u64 = 10_000_000;

const BOT_SALT: u64 = 0xB075;

/// A bot's random stream depends only on (seed, seat), so swapping the candidate leaves every
/// baseline's stream unchanged (common random numbers).
pub fn bot_seed(seed: u64, seat: usize) -> u64 {
    mix(mix(seed, BOT_SALT), seat as u64)
}

/// Events each seat has not been shown yet, already redacted for that seat.
#[derive(Default)]
pub struct EventFeed {
    pending: [Vec<Event>; NUM_PLAYERS],
}

impl EventFeed {
    pub fn push(&mut self, events: &[Event]) {
        for (p, q) in self.pending.iter_mut().enumerate() {
            q.extend(events.iter().map(|e| e.redacted_for(p as PlayerId)));
        }
    }

    /// Show the bot in `seat` everything since it last moved.
    pub fn deliver(&mut self, seat: usize, bot: &mut dyn Bot) {
        bot.observe(seat as PlayerId, &self.pending[seat]);
        self.pending[seat].clear();
    }
}

/// Contiguous parts of `seeds` for up to `threads` workers (capped at the seed count and at 4x
/// the available cores), split by offset in u64 so no bound passes `seeds.end` (a range can end
/// at u64::MAX). Empty for an empty range.
pub fn split_seeds(seeds: Range<u64>, threads: usize) -> Vec<Range<u64>> {
    let n = seeds.end.saturating_sub(seeds.start);
    if n == 0 {
        return Vec::new();
    }
    let cores = std::thread::available_parallelism().map_or(1, |c| c.get());
    let threads = threads.clamp(1, (n.min(usize::MAX as u64) as usize).min(cores * 4));
    let chunk = n.div_ceil(threads as u64);
    (0..threads as u64)
        .map(|i| i * chunk)
        .take_while(|&off| off < n)
        .map(|off| seeds.start + off..seeds.start + off + chunk.min(n - off))
        .collect()
}

/// Play one game. Returns (winner, final VP, turns, actions). Panics if a bot picks an illegal action.
pub fn play_game(
    seed: u64,
    config: GameConfig,
    bots: &mut [Box<dyn Bot>],
) -> (Option<u8>, [u8; NUM_PLAYERS], u32, u32) {
    assert_eq!(bots.len(), NUM_PLAYERS, "a game needs exactly 4 bots");
    let mut s = State::new(seed, config);
    let mut buf = Vec::with_capacity(512);
    let mut feed = EventFeed::default();
    let mut step = Vec::with_capacity(16);
    let mut actions = 0u32;
    while !s.is_over() {
        legal_actions(&s, &mut buf);
        let actor = s.current_actor() as usize;
        feed.deliver(actor, bots[actor].as_mut());
        let a = bots[actor].act(&s.observation(actor as u8), &buf);
        assert!(
            buf.contains(&a),
            "bot {} chose illegal action {a:?} in phase {:?}",
            bots[actor].name(),
            s.phase
        );
        step.clear();
        s.apply_with(a, None, &mut step);
        feed.push(&step);
        actions += 1;
    }
    (
        s.winner(),
        std::array::from_fn(|p| s.total_vp(p)),
        s.turn,
        actions,
    )
}

/// Every seed in `seeds`, four games each (candidate in seat 0..4), spread over `threads`
/// (capped at the seed count and at 4x the available cores). Records are sorted by
/// (seed, candidate seat) and independent of `threads`. At most `MAX_SEEDS` seeds per call.
pub fn run_match(
    candidate: &str,
    baseline: &str,
    seeds: Range<u64>,
    config: GameConfig,
    threads: usize,
) -> Result<Vec<GameRecord>, String> {
    for name in [candidate, baseline] {
        if make_bot(name, 0).is_none() {
            return Err(format!(
                "unknown bot {name:?}; known bots: {BOT_NAMES:?} (ismcts@N for N simulations, ismcts@N+rD to add D greedy rollout moves)"
            ));
        }
    }
    config.validate()?;
    let n = seeds.end.saturating_sub(seeds.start);
    if n > MAX_SEEDS {
        return Err(format!("at most {MAX_SEEDS} seeds per match, got {n}"));
    }
    if n == 0 {
        return Ok(Vec::new());
    }
    let parts = split_seeds(seeds, threads);
    let mut records: Vec<GameRecord> = std::thread::scope(|scope| {
        let workers: Vec<_> = parts
            .into_iter()
            .map(|part| {
                scope.spawn(move || {
                    let mut out =
                        Vec::with_capacity((part.end - part.start) as usize * NUM_PLAYERS);
                    for seed in part {
                        for cand in 0..NUM_PLAYERS {
                            let mut bots: Vec<Box<dyn Bot>> = (0..NUM_PLAYERS)
                                .map(|seat| {
                                    let name = if seat == cand { candidate } else { baseline };
                                    make_bot(name, bot_seed(seed, seat)).expect("validated above")
                                })
                                .collect();
                            let (winner, vp, turns, actions) = play_game(seed, config, &mut bots);
                            out.push(GameRecord {
                                seed,
                                candidate_seat: cand as u8,
                                winner,
                                vp,
                                turns,
                                actions,
                            });
                        }
                    }
                    out
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
    records.sort_by_key(|r| (r.seed, r.candidate_seat));
    Ok(records)
}
