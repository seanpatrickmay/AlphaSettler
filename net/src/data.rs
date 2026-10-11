//! Training data (spec Section 4): recorded games replayed to their decisions and encoded.
//! Nothing is held per position: a batch replays each sampled game from its seed (about 60 ns per
//! move) on a thread pool, so the window costs only its records (seeds, actions, visit counts).
//! Records come from files, so `TrainingSet::extend` replays and checks every game before any of
//! it is used: a record that does not replay is an error, never a panic or a bad index.

use crate::encode::{abs, encode_state, rel, HIDDEN_LEN, PUBLIC_LEN};
use crate::policy::policy_id;
use settler_engine::legal::legal_actions;
use settler_engine::rng::Rng;
use settler_engine::topology::{NUM_EDGES, NUM_NODES};
use settler_engine::{
    Action, DevCard, GameConfig, PlayerId, State, ACTION_SPACE_SIZE, NUM_PLAYERS,
};

/// Final VP targets are VP / this.
pub const VP_TARGET_DEN: f32 = 10.0;
/// Turns-left targets are turns / this (targets v2).
pub const TURNS_DEN: f32 = 100.0;
/// Final node classes: empty, a settlement per relative seat, a city per relative seat.
pub const NODE_CLASSES: usize = 9;
/// Final edge classes: none, a road per relative seat.
pub const EDGE_CLASSES: usize = 5;
/// Final settlements, cities and VP cards held are counts / these.
pub const PART_DENS: [f32; 3] = [5.0, 4.0, 5.0];
/// Most positions one `sample` call returns (about 9.7 KB each, so about 640 MB).
pub const MAX_SAMPLE: usize = 1 << 16;
/// How far a recorded `target` may sum from 1.
pub const TARGET_TOLERANCE: f64 = 1e-4;
/// How far a recorded `q` may sum from 1.
pub const Q_TOLERANCE: f64 = 1e-3;
/// How far outside [0, 1] a recorded `q` entry may lie.
pub const Q_RANGE_TOLERANCE: f32 = 1e-4;

#[derive(Clone, Debug, PartialEq)]
pub struct RecordedDecision {
    /// Index into the game's actions of the move this search chose.
    pub index: u32,
    pub actor: PlayerId,
    /// Root visits aligned with the legal moves at `index`, in engine order (offers included).
    pub visits: Vec<u32>,
    pub full_search: bool,
    /// A Gumbel search's improved policy (spec P2 Section 5), aligned with the legal moves and
    /// summing to 1. When present it is the policy target, whatever the visits.
    pub target: Option<Vec<f32>>,
    /// The search's root value, absolute seats (targets v2); None in records from before it.
    pub q: Option<[f32; NUM_PLAYERS]>,
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
    /// and trade confirmations) of each legal move, domestic offers included.
    pub mask: Vec<u8>,
    /// n × ACTION_SPACE_SIZE: the decision's `target` if it has one, else root visits normalised
    /// over the mask (uniform if none), at the same policy ids.
    pub policy: Vec<f32>,
    /// n: 1 for a policy target: a decision with a `target`, or a full search with at least one
    /// visit.
    pub full: Vec<u8>,
    /// n × 4, seat-relative: one-hot winner, ¼ each for a turn-cap draw.
    pub outcome: Vec<f32>,
    /// n × 4, seat-relative: final VP / 10.
    pub vp: Vec<f32>,
    /// n × 4, seat-relative: the decision's root search value; zeros without one.
    pub q: Vec<f32>,
    /// n: 1 if the decision has a root search value.
    pub has_q: Vec<u8>,
    /// n × NUM_NODES, seat-relative final node class: 0 empty, 1 + r settlement, 5 + r city.
    pub own_node: Vec<u8>,
    /// n × NUM_EDGES, seat-relative final edge class: 0 none, 1 + r road.
    pub own_edge: Vec<u8>,
    /// n × 12: relative seat r major, then final settlements, cities and VP cards held, each
    /// over its `PART_DENS`.
    pub parts: Vec<f32>,
    /// n × 2: the final Longest Road, then Largest Army, holder: 0 none, 1 + relative seat.
    pub holders: Vec<u8>,
    /// n: turns from the decision to the game's end / TURNS_DEN.
    pub turns_left: Vec<f32>,
}

/// A game's final-state targets (targets v2, spec Section 4), in absolute seats; ~150 bytes.
#[derive(Clone, Debug)]
struct Final {
    /// 0 empty, 1 + p settlement, 5 + p city.
    node: [u8; NUM_NODES],
    /// 0 none, 1 + p road.
    edge: [u8; NUM_EDGES],
    /// Settlements, cities and VP cards held.
    parts: [[u8; 3]; NUM_PLAYERS],
    road: Option<PlayerId>,
    army: Option<PlayerId>,
    turn: u32,
}

impl Final {
    /// Node i is bit i of `settlements` and `cities`, edge i bit i of `roads` (the encoder's
    /// order). The engine keeps cities and settlements disjoint; a node in both reads as a city.
    fn of(s: &State) -> Final {
        let mut node = [0u8; NUM_NODES];
        let mut edge = [0u8; NUM_EDGES];
        for (p, pl) in s.players.iter().enumerate() {
            for (n, c) in node.iter_mut().enumerate() {
                if pl.cities >> n & 1 == 1 {
                    *c = 5 + p as u8;
                } else if pl.settlements >> n & 1 == 1 {
                    *c = 1 + p as u8;
                }
            }
            for (e, c) in edge.iter_mut().enumerate() {
                if pl.roads >> e & 1 == 1 {
                    *c = 1 + p as u8;
                }
            }
        }
        Final {
            node,
            edge,
            parts: std::array::from_fn(|p| {
                let pl = &s.players[p];
                [
                    pl.settlements.count_ones() as u8,
                    pl.cities.count_ones() as u8,
                    pl.dev_hand[DevCard::VictoryPoint.index()],
                ]
            }),
            road: s.longest_road_owner,
            army: s.largest_army_owner,
            turn: s.turn,
        }
    }
}

/// 0 for none, else 1 + `p`'s seat relative to `actor`.
fn holder(p: Option<PlayerId>, actor: PlayerId) -> u8 {
    p.map_or(0, |p| 1 + rel(p, actor) as u8)
}

impl Batch {
    fn push(
        &mut self,
        s: &State,
        game: &RecordedGame,
        d: &RecordedDecision,
        legal: &[Action],
        f: &Final,
    ) {
        let actor = d.actor;
        let (p0, h0) = (self.public.len(), self.hidden.len());
        self.public.resize(p0 + PUBLIC_LEN, 0.0);
        self.hidden.resize(h0 + HIDDEN_LEN, 0.0);
        encode_state(s, actor, &mut self.public[p0..], &mut self.hidden[h0..]);
        let m0 = self.mask.len();
        self.mask.resize(m0 + ACTION_SPACE_SIZE, 0);
        self.policy.resize(m0 + ACTION_SPACE_SIZE, 0.0);
        // u64: visit counts come from files and may sum past u32::MAX.
        let total: u64 = d.visits.iter().map(|&v| v as u64).sum();
        for (j, (a, &v)) in legal.iter().zip(&d.visits).enumerate() {
            let k = m0 + policy_id(*a, actor);
            self.mask[k] = 1;
            self.policy[k] = match &d.target {
                Some(t) => t[j],
                None if total > 0 => v as f32 / total as f32,
                None => 1.0 / legal.len() as f32,
            };
        }
        self.full
            .push((d.target.is_some() || (d.full_search && total > 0)) as u8);
        let outcome: [f32; NUM_PLAYERS] = match game.winner {
            Some(w) => std::array::from_fn(|r| (r == rel(w, actor)) as u8 as f32),
            None => [1.0 / NUM_PLAYERS as f32; NUM_PLAYERS],
        };
        self.outcome.extend_from_slice(&outcome);
        for r in 0..NUM_PLAYERS {
            self.vp.push(game.vp[abs(r, actor)] as f32 / VP_TARGET_DEN);
        }
        match d.q {
            Some(q) => {
                for r in 0..NUM_PLAYERS {
                    self.q.push(q[abs(r, actor)]);
                }
                self.has_q.push(1);
            }
            None => {
                self.q.extend_from_slice(&[0.0; NUM_PLAYERS]);
                self.has_q.push(0);
            }
        }
        // Classes 1..=4 and 5..=8 hold absolute seat c - 1 and c - 5 (`Final`).
        self.own_node.extend(f.node.iter().map(|&c| match c {
            0 => 0,
            c @ 1..=4 => 1 + rel(c - 1, actor) as u8,
            c => 5 + rel(c - 5, actor) as u8,
        }));
        self.own_edge.extend(f.edge.iter().map(|&c| match c {
            0 => 0,
            c => 1 + rel(c - 1, actor) as u8,
        }));
        for r in 0..NUM_PLAYERS {
            for (j, den) in PART_DENS.iter().enumerate() {
                self.parts.push(f.parts[abs(r, actor)][j] as f32 / den);
            }
        }
        self.holders.push(holder(f.road, actor));
        self.holders.push(holder(f.army, actor));
        // A decision's turn is never past the final turn.
        self.turns_left
            .push(f.turn.saturating_sub(s.turn) as f32 / TURNS_DEN);
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
        self.q.extend(other.q);
        self.has_q.extend(other.has_q);
        self.own_node.extend(other.own_node);
        self.own_edge.extend(other.own_edge);
        self.parts.extend(other.parts);
        self.holders.extend(other.holders);
        self.turns_left.extend(other.turns_left);
    }
}

/// Checks that `g` replays, and returns the indices of its decisions that are policy targets
/// (`Batch::full` 1: a decision with a valid `target`, or a full search with at least one visit)
/// and its final-state targets.
fn check_game(g: &RecordedGame, buf: &mut Vec<Action>) -> Result<(Vec<u32>, Final), String> {
    let mut targets = Vec::new();
    g.config.validate()?;
    let mut s = State::new(g.seed, g.config);
    let mut next = g.decisions.iter().enumerate().peekable();
    for (i, &a) in g.actions.iter().enumerate() {
        if s.is_over() {
            return Err(format!("move {i} comes after the game ended"));
        }
        legal_actions(&s, buf);
        if let Some((k, d)) = next.next_if(|(_, d)| d.index as usize == i) {
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
            if let Some(t) = &d.target {
                if t.len() != buf.len() {
                    return Err(format!(
                        "move {i}: {} target entries for {} legal moves",
                        t.len(),
                        buf.len()
                    ));
                }
                if let Some((j, x)) = t
                    .iter()
                    .enumerate()
                    .find(|(_, x)| !(x.is_finite() && **x >= 0.0))
                {
                    return Err(format!(
                        "move {i}: target entry {j} is {x}, not a probability"
                    ));
                }
                let total: f64 = t.iter().map(|&x| x as f64).sum();
                if (total - 1.0).abs() > TARGET_TOLERANCE {
                    return Err(format!("move {i}: the target sums to {total}, not 1"));
                }
            }
            if let Some(q) = &d.q {
                if let Some((j, x)) = q.iter().enumerate().find(|(_, x)| {
                    !(x.is_finite() && **x >= -Q_RANGE_TOLERANCE && **x <= 1.0 + Q_RANGE_TOLERANCE)
                }) {
                    return Err(format!("move {i}: q entry {j} is {x}, not a probability"));
                }
                let total: f64 = q.iter().map(|&x| x as f64).sum();
                if (total - 1.0).abs() > Q_TOLERANCE {
                    return Err(format!("move {i}: q sums to {total}, not 1"));
                }
            }
            let visited = d.visits.iter().any(|&v| v > 0);
            if d.target.is_some() || (d.full_search && visited) {
                targets.push(k as u32);
            }
        }
        if !buf.contains(&a) {
            return Err(format!("move {i}: {a:?} is illegal"));
        }
        s.apply(a);
    }
    if let Some((_, d)) = next.next() {
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
    Ok((targets, Final::of(&s)))
}

/// Recorded games, checked once, from which training batches are replayed.
#[derive(Clone, Debug)]
pub struct TrainingSet {
    games: Vec<RecordedGame>,
    /// `starts[g]` is the global index of game g's first decision; the last entry is the total.
    starts: Vec<usize>,
    full: usize,
    /// (game, decision) of every policy target (`check_game`).
    targets: Vec<(u32, u32)>,
    /// Each game's final-state targets, in game order.
    finals: Vec<Final>,
}

/// The empty set.
impl Default for TrainingSet {
    fn default() -> TrainingSet {
        TrainingSet {
            games: Vec::new(),
            starts: vec![0],
            full: 0,
            targets: Vec::new(),
            finals: Vec::new(),
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
        let mut targets = Vec::new();
        let mut finals = Vec::with_capacity(games.len());
        for (i, g) in games.iter().enumerate() {
            let game = (self.games.len() + i) as u32;
            let (found, last) = check_game(g, &mut buf)
                .map_err(|e| format!("game {} (seed {}): {e}", game, g.seed))?;
            targets.extend(found.into_iter().map(|k| (game, k)));
            finals.push(last);
        }
        let mut total = self.decisions();
        self.starts.pop();
        for g in &games {
            self.starts.push(total);
            total += g.decisions.len();
            self.full += g.decisions.iter().filter(|d| d.full_search).count();
        }
        self.targets.extend(targets);
        self.finals.extend(finals);
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

    /// Decisions that are policy targets (`Batch::full` 1): decisions with a target, and full
    /// searches with at least one visit.
    pub fn policy_decisions(&self) -> usize {
        self.targets.len()
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
        out.push(&s, g, d, &legal, &self.finals[game]);
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

    /// `n` positions drawn uniformly over the policy targets alone (`policy_decisions`), from
    /// `seed`; the result does not depend on `threads`.
    pub fn sample_full(&self, n: usize, seed: u64, threads: usize) -> Result<Batch, String> {
        let total = self.targets.len();
        if total == 0 {
            return Err("no full-search decisions with visits to sample".into());
        }
        if n > MAX_SAMPLE {
            return Err(format!(
                "at most {MAX_SAMPLE} positions per sample, asked for {n}"
            ));
        }
        let mut rng = Rng::new(seed);
        let picks: Vec<(usize, usize)> = (0..n)
            .map(|_| {
                let (g, d) =
                    self.targets[((rng.next_u64() as u128 * total as u128) >> 64) as usize];
                (g as usize, d as usize)
            })
            .collect();
        Ok(self.batch(&picks, threads))
    }
}
