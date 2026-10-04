//! Training data (spec Section 4): recorded games replayed to their decisions and encoded.
//! Nothing is held per position: a batch replays each sampled game from its seed (about 60 ns per
//! move) on a thread pool, so the window costs only its records (seeds, actions, visit counts).
//! Records come from files, so `TrainingSet::extend` replays and checks every game before any of
//! it is used: a record that does not replay is an error, never a panic or a bad index.

use crate::encode::{abs, encode_state, rel, HIDDEN_LEN, PUBLIC_LEN};
use crate::policy::policy_id;
use settler_engine::legal::legal_actions;
use settler_engine::rng::Rng;
use settler_engine::{Action, GameConfig, PlayerId, State, ACTION_SPACE_SIZE, NUM_PLAYERS};

/// Final VP targets are VP / this.
pub const VP_TARGET_DEN: f32 = 10.0;
/// Most positions one `sample` call returns (about 9.7 KB each, so about 640 MB).
pub const MAX_SAMPLE: usize = 1 << 16;

#[derive(Clone, Debug, PartialEq)]
pub struct RecordedDecision {
    /// Index into the game's actions of the move this search chose.
    pub index: u32,
    pub actor: PlayerId,
    /// Root visits aligned with the legal moves at `index`, in engine order (offers included).
    pub visits: Vec<u32>,
    pub full_search: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RecordedGame {
    pub seed: u64,
    pub config: GameConfig,
    pub actions: Vec<Action>,
    pub decisions: Vec<RecordedDecision>,
    pub winner: Option<PlayerId>,
    pub vp: [u8; NUM_PLAYERS],
}

/// Inputs and targets for `n` positions, row-major.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Batch {
    pub n: usize,
    /// n × PUBLIC_LEN.
    pub public: Vec<f32>,
    /// n × HIDDEN_LEN.
    pub hidden: Vec<f32>,
    /// n × ACTION_SPACE_SIZE: 1 at the policy id (`policy::policy_id`, seat-relative for steals
    /// and trade confirmations) of each legal move except domestic offers.
    pub mask: Vec<u8>,
    /// n × ACTION_SPACE_SIZE: root visits normalised over the mask (uniform if none), at the same
    /// policy ids.
    pub policy: Vec<f32>,
    /// n: 1 for a full search with at least one visit (a policy target).
    pub full: Vec<u8>,
    /// n × 4, seat-relative: one-hot winner, ¼ each for a turn-cap draw.
    pub outcome: Vec<f32>,
    /// n × 4, seat-relative: final VP / 10.
    pub vp: Vec<f32>,
}

impl Batch {
    fn push(&mut self, s: &State, game: &RecordedGame, d: &RecordedDecision, legal: &[Action]) {
        let actor = d.actor;
        let (p0, h0) = (self.public.len(), self.hidden.len());
        self.public.resize(p0 + PUBLIC_LEN, 0.0);
        self.hidden.resize(h0 + HIDDEN_LEN, 0.0);
        encode_state(s, actor, &mut self.public[p0..], &mut self.hidden[h0..]);
        let m0 = self.mask.len();
        self.mask.resize(m0 + ACTION_SPACE_SIZE, 0);
        self.policy.resize(m0 + ACTION_SPACE_SIZE, 0.0);
        let searchable = || {
            legal
                .iter()
                .zip(&d.visits)
                .filter(|(a, _)| !matches!(a, Action::OfferTrade { .. }))
        };
        // u64: visit counts come from files and may sum past u32::MAX.
        let total: u64 = searchable().map(|(_, &v)| v as u64).sum();
        let count = searchable().count();
        for (a, &v) in searchable() {
            let k = m0 + policy_id(*a, actor);
            self.mask[k] = 1;
            self.policy[k] = if total > 0 {
                v as f32 / total as f32
            } else {
                1.0 / count as f32
            };
        }
        self.full.push((d.full_search && total > 0) as u8);
        let outcome: [f32; NUM_PLAYERS] = match game.winner {
            Some(w) => std::array::from_fn(|r| (r == rel(w, actor)) as u8 as f32),
            None => [1.0 / NUM_PLAYERS as f32; NUM_PLAYERS],
        };
        self.outcome.extend_from_slice(&outcome);
        for r in 0..NUM_PLAYERS {
            self.vp.push(game.vp[abs(r, actor)] as f32 / VP_TARGET_DEN);
        }
        self.n += 1;
    }

    fn append(&mut self, other: Batch) {
        self.n += other.n;
        self.public.extend(other.public);
        self.hidden.extend(other.hidden);
        self.mask.extend(other.mask);
        self.policy.extend(other.policy);
        self.full.extend(other.full);
        self.outcome.extend(other.outcome);
        self.vp.extend(other.vp);
    }
}

/// Replay `g` once, checking it against its record.
fn check_game(g: &RecordedGame, buf: &mut Vec<Action>) -> Result<(), String> {
    g.config.validate()?;
    let mut s = State::new(g.seed, g.config);
    let mut next = g.decisions.iter().peekable();
    for (i, &a) in g.actions.iter().enumerate() {
        if s.is_over() {
            return Err(format!("move {i} comes after the game ended"));
        }
        legal_actions(&s, buf);
        if let Some(d) = next.next_if(|d| d.index as usize == i) {
            if d.actor != s.current_actor() {
                return Err(format!(
                    "move {i}: the record says actor {}, the replay has {}",
                    d.actor,
                    s.current_actor()
                ));
            }
            if d.visits.len() != buf.len() {
                return Err(format!(
                    "move {i}: {} visit counts for {} legal moves",
                    d.visits.len(),
                    buf.len()
                ));
            }
        }
        if !buf.contains(&a) {
            return Err(format!("move {i}: {a:?} is illegal"));
        }
        s.apply(a);
    }
    if let Some(d) = next.next() {
        return Err(format!(
            "the decision at move {} is out of order or past the last move",
            d.index
        ));
    }
    if !s.is_over() {
        return Err("the game does not finish".into());
    }
    if s.winner() != g.winner {
        return Err(format!(
            "the replay's winner is {:?}, the record says {:?}",
            s.winner(),
            g.winner
        ));
    }
    let vp: [u8; NUM_PLAYERS] = std::array::from_fn(|p| s.total_vp(p));
    if vp != g.vp {
        return Err(format!(
            "the replay's final vp is {vp:?}, the record says {:?}",
            g.vp
        ));
    }
    Ok(())
}

/// Recorded games, checked once, from which training batches are replayed.
#[derive(Clone, Debug)]
pub struct TrainingSet {
    games: Vec<RecordedGame>,
    /// `starts[g]` is the global index of game g's first decision; the last entry is the total.
    starts: Vec<usize>,
    full: usize,
}

/// The empty set.
impl Default for TrainingSet {
    fn default() -> TrainingSet {
        TrainingSet {
            games: Vec::new(),
            starts: vec![0],
            full: 0,
        }
    }
}

impl TrainingSet {
    /// Every game is replayed and must match its record exactly.
    pub fn new(games: Vec<RecordedGame>) -> Result<TrainingSet, String> {
        let mut set = TrainingSet::default();
        set.extend(games)?;
        Ok(set)
    }

    /// Add `games` (one record file's, say), each replayed and checked as in `new`. An error
    /// names the game by its index in the whole set and leaves the set unchanged.
    pub fn extend(&mut self, games: Vec<RecordedGame>) -> Result<(), String> {
        let mut buf = Vec::new();
        for (i, g) in games.iter().enumerate() {
            check_game(g, &mut buf)
                .map_err(|e| format!("game {} (seed {}): {e}", self.games.len() + i, g.seed))?;
        }
        let mut total = self.decisions();
        self.starts.pop();
        for g in &games {
            self.starts.push(total);
            total += g.decisions.len();
            self.full += g.decisions.iter().filter(|d| d.full_search).count();
        }
        self.starts.push(total);
        self.games.extend(games);
        Ok(())
    }

    pub fn games(&self) -> usize {
        self.games.len()
    }

    pub fn decisions(&self) -> usize {
        *self
            .starts
            .last()
            .expect("starts always ends with the total")
    }

    pub fn full_decisions(&self) -> usize {
        self.full
    }

    /// Panics unless `game < games()`.
    pub fn game_decisions(&self, game: usize) -> usize {
        self.games[game].decisions.len()
    }

    /// (game, decision) of global decision `k` (< `decisions()`).
    pub fn locate(&self, k: usize) -> (usize, usize) {
        let g = self.starts.partition_point(|&s| s <= k) - 1;
        (g, k - self.starts[g])
    }

    /// Append decision `decision` of game `game` to `out`. Panics unless `game < games()` and
    /// `decision < game_decisions(game)`.
    pub fn position(&self, game: usize, decision: usize, out: &mut Batch) {
        let g = &self.games[game];
        let d = &g.decisions[decision];
        let mut s = State::new(g.seed, g.config);
        for &a in &g.actions[..d.index as usize] {
            s.apply(a);
        }
        let mut legal = Vec::new();
        legal_actions(&s, &mut legal);
        out.push(&s, g, d, &legal);
    }

    /// `picks` in order, replayed on up to `threads` threads (at most 4 per core); independent of
    /// `threads`. Panics on a pick `position` would panic on.
    pub fn batch(&self, picks: &[(usize, usize)], threads: usize) -> Batch {
        let cores = std::thread::available_parallelism().map_or(1, |c| c.get());
        let threads = threads.clamp(1, picks.len().min(cores * 4).max(1));
        let chunk = picks.len().div_ceil(threads).max(1);
        let parts: Vec<Batch> = std::thread::scope(|scope| {
            let workers: Vec<_> = picks
                .chunks(chunk)
                .map(|part| {
                    scope.spawn(move || {
                        let mut b = Batch::default();
                        for &(g, d) in part {
                            self.position(g, d, &mut b);
                        }
                        b
                    })
                })
                .collect();
            workers
                .into_iter()
                .map(|w| {
                    w.join()
                        .unwrap_or_else(|panic| std::panic::resume_unwind(panic))
                })
                .collect()
        });
        let mut out = Batch::default();
        for p in parts {
            out.append(p);
        }
        out
    }

    /// `n` (at most `MAX_SAMPLE`) decisions drawn uniformly (with replacement) from
    /// `Rng::new(seed)`, then replayed.
    pub fn sample(&self, n: usize, seed: u64, threads: usize) -> Result<Batch, String> {
        let total = self.decisions();
        if total == 0 {
            return Err("no positions to sample: the records hold no decisions".into());
        }
        if n > MAX_SAMPLE {
            return Err(format!(
                "at most {MAX_SAMPLE} positions per sample, asked for {n}"
            ));
        }
        let mut rng = Rng::new(seed);
        let picks: Vec<(usize, usize)> = (0..n)
            .map(|_| self.locate(((rng.next_u64() as u128 * total as u128) >> 64) as usize))
            .collect();
        Ok(self.batch(&picks, threads))
    }
}
