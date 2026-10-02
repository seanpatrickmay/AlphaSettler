//! The hand-written evaluator 3a searches with; 3b replaces it with a network (spec Section 4).

use crate::greedy::{self, pips};
use settler_engine::legal::legal_actions;
use settler_engine::topology::{topo, NUM_TILES};
use settler_engine::types::*;
use settler_engine::{Action, Observation, State};
use settler_search::{search_actions, terminal_value, Eval, Evaluator, Leaf};

pub const NUM_FEATURES: usize = 8;
pub const FEATURE_NAMES: [&str; NUM_FEATURES] = [
    "vp",
    "production",
    "hand",
    "over_limit",
    "dev_cards",
    "road_length",
    "knights",
    "settlement_spots",
];
/// Weights of the per-player strength score. Hand-set until `alphasettler fit-heuristic` refits
/// them (docs/bot/results-3a.md records each fit).
pub const DEFAULT_WEIGHTS: [f32; NUM_FEATURES] = [1.0, 0.08, 0.05, -0.1, 0.3, 0.05, 0.1, 0.2];

/// Per resource: the board's mean pips per resource over this resource's pips (scarce resources
/// weigh more). 0 for a resource with no tiles.
fn scarcity(s: &State) -> [f32; NUM_RESOURCES] {
    let mut total = [0u32; NUM_RESOURCES];
    for tile in 0..NUM_TILES {
        if let Some(r) = s.board.tile_resource[tile] {
            total[r.index()] += pips(s.board.tile_number[tile]);
        }
    }
    let mean = total.iter().sum::<u32>() as f32 / NUM_RESOURCES as f32;
    total.map(|t| if t == 0 { 0.0 } else { mean / t as f32 })
}

/// Free nodes at the end of player `p`'s roads where the distance rule allows a settlement.
fn settlement_spots(s: &State, p: usize) -> u32 {
    let t = topo();
    let occupied = s.occupied_nodes();
    let mut reach = 0u64;
    for e in bits128(s.players[p].roads) {
        let (a, b) = t.edge_nodes[e as usize];
        reach |= (1u64 << a) | (1u64 << b);
    }
    bits64(reach)
        .filter(|&n| {
            occupied & (1u64 << n) == 0 && occupied & t.node_neighbor_mask[n as usize] == 0
        })
        .count() as u32
}

/// Player `p`'s features in `s` (which may be a sampled world), in `FEATURE_NAMES` order.
pub fn features(s: &State, p: usize) -> [f32; NUM_FEATURES] {
    let t = topo();
    let pl = &s.players[p];
    let scarce = scarcity(s);
    let mut production = 0.0;
    for tile in 0..NUM_TILES {
        if tile as u8 == s.robber {
            continue;
        }
        let Some(r) = s.board.tile_resource[tile] else {
            continue;
        };
        let mask = t.tile_node_mask[tile];
        let weight = (pl.settlements & mask).count_ones() + 2 * (pl.cities & mask).count_ones();
        production += (weight * pips(s.board.tile_number[tile])) as f32 * scarce[r.index()];
    }
    let cards = hand_total(&pl.hand) as f32;
    let dev = pl.dev_hand.iter().sum::<u8>() - pl.dev_hand[DevCard::VictoryPoint.index()];
    [
        s.total_vp(p) as f32,
        production,
        cards.min(7.0),
        (cards - 7.0).max(0.0),
        dev as f32,
        pl.longest_road_len as f32,
        pl.knights_played as f32,
        settlement_spots(s, p) as f32,
    ]
}

fn softmax(x: &mut [f32]) {
    let m = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut total = 0.0;
    for v in x.iter_mut() {
        *v = (*v - m).exp();
        total += *v;
    }
    for v in x.iter_mut() {
        *v /= total;
    }
}

/// Win probabilities: a softmax over the four players' weighted feature scores.
pub fn value(s: &State, w: &[f32; NUM_FEATURES]) -> [f32; NUM_PLAYERS] {
    let mut v: [f32; NUM_PLAYERS] =
        std::array::from_fn(|p| features(s, p).iter().zip(w).map(|(f, w)| f * w).sum());
    softmax(&mut v);
    v
}

/// Logits: GreedyBot's choice +2, a city or settlement +1, buying a dev card +0.5, everything
/// else 0; softmaxed over `legal`. Reads only the actor's observation.
pub fn prior(obs: &Observation, legal: &[Action]) -> Vec<f32> {
    let pick = greedy::choose(obs, legal);
    let mut logits: Vec<f32> = legal
        .iter()
        .map(|&a| {
            let base = match a {
                Action::BuildCity(_) | Action::BuildSettlement(_) => 1.0,
                Action::BuyDev => 0.5,
                _ => 0.0,
            };
            if a == pick {
                base + 2.0
            } else {
                base
            }
        })
        .collect();
    softmax(&mut logits);
    logits
}

pub struct HeuristicEvaluator {
    pub weights: [f32; NUM_FEATURES],
    /// Greedy moves played from each leaf (in its own sampled world) before scoring it; 0 = none.
    pub rollout: u32,
    buf: Vec<Action>,
    moves: Vec<Action>,
}

impl HeuristicEvaluator {
    pub fn new(weights: [f32; NUM_FEATURES], rollout: u32) -> HeuristicEvaluator {
        HeuristicEvaluator {
            weights,
            rollout,
            buf: Vec::new(),
            moves: Vec::new(),
        }
    }

    fn leaf_value(&mut self, world: &State) -> [f32; NUM_PLAYERS] {
        if self.rollout == 0 {
            return value(world, &self.weights);
        }
        let mut s = *world;
        for _ in 0..self.rollout {
            if s.is_over() {
                return terminal_value(&s);
            }
            legal_actions(&s, &mut self.buf);
            search_actions(&self.buf, &mut self.moves);
            let actor = s.current_actor();
            let a = greedy::choose(&s.observation(actor), &self.moves);
            s.apply(a);
        }
        if s.is_over() {
            terminal_value(&s)
        } else {
            value(&s, &self.weights)
        }
    }
}

impl Evaluator for HeuristicEvaluator {
    fn evaluate(&mut self, leaves: &[Leaf<'_>]) -> Vec<Eval> {
        let mut out = Vec::with_capacity(leaves.len());
        for l in leaves {
            let prior = prior(&l.world.observation(l.actor), l.legal);
            let value = self.leaf_value(l.world);
            out.push(Eval { value, prior });
        }
        out
    }
}

/// One position for the weight fit: every player's features, and who went on to win.
#[derive(Clone, Debug, PartialEq)]
pub struct Sample {
    pub features: [[f32; NUM_FEATURES]; NUM_PLAYERS],
    pub winner: usize,
}

/// A recorded game to replay for samples: its moves and the move indices to sample at.
#[derive(Clone, Debug, PartialEq)]
pub struct ReplayGame {
    pub seed: u64,
    pub config: settler_engine::GameConfig,
    pub actions: Vec<Action>,
    pub winner: Option<PlayerId>,
    pub at: Vec<u32>,
}

/// The true position before each sampled move, labelled with the winner. Draws are skipped.
pub fn samples_from_games(games: &[ReplayGame]) -> Result<Vec<Sample>, String> {
    let mut out = Vec::new();
    let mut buf = Vec::new();
    for g in games {
        let Some(winner) = g.winner else { continue };
        let mut s = State::new(g.seed, g.config);
        let mut at = g.at.iter().peekable();
        for (i, &a) in g.actions.iter().enumerate() {
            if at.next_if(|&&k| k as usize == i).is_some() {
                out.push(Sample {
                    features: std::array::from_fn(|p| features(&s, p)),
                    winner: winner as usize,
                });
            }
            legal_actions(&s, &mut buf);
            if !buf.contains(&a) {
                return Err(format!(
                    "game {}: move {i} ({a:?}) is illegal on replay",
                    g.seed
                ));
            }
            s.apply(a);
        }
        if s.winner() != g.winner {
            return Err(format!(
                "game {}: the replay ends with winner {:?}, the record says {:?}",
                g.seed,
                s.winner(),
                g.winner
            ));
        }
    }
    Ok(out)
}

fn scores(w: &[f64; NUM_FEATURES], f: &[[f32; NUM_FEATURES]; NUM_PLAYERS]) -> [f64; NUM_PLAYERS] {
    std::array::from_fn(|p| (0..NUM_FEATURES).map(|i| w[i] * f[p][i] as f64).sum())
}

fn probabilities(z: [f64; NUM_PLAYERS]) -> [f64; NUM_PLAYERS] {
    let m = z.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let e = z.map(|x| (x - m).exp());
    let total: f64 = e.iter().sum();
    e.map(|x| x / total)
}

fn mean_ll(samples: &[Sample], w: &[f64; NUM_FEATURES]) -> f64 {
    let total: f64 = samples
        .iter()
        .map(|s| probabilities(scores(w, &s.features))[s.winner].ln())
        .sum();
    total / samples.len().max(1) as f64
}

/// Mean log-likelihood of the winners under `weights`.
pub fn log_likelihood(samples: &[Sample], weights: &[f32; NUM_FEATURES]) -> f64 {
    mean_ll(samples, &weights.map(|x| x as f64))
}

/// Solve `a x = b` by Gaussian elimination with partial pivoting; None if singular.
fn solve(
    mut a: [[f64; NUM_FEATURES]; NUM_FEATURES],
    mut b: [f64; NUM_FEATURES],
) -> Option<[f64; NUM_FEATURES]> {
    let n = NUM_FEATURES;
    for col in 0..n {
        let pivot = (col..n).max_by(|&i, &j| a[i][col].abs().total_cmp(&a[j][col].abs()))?;
        if a[pivot][col].abs() < 1e-12 {
            return None;
        }
        a.swap(col, pivot);
        b.swap(col, pivot);
        for row in col + 1..n {
            let f = a[row][col] / a[col][col];
            for k in col..n {
                a[row][k] -= f * a[col][k];
            }
            b[row] -= f * b[col];
        }
    }
    let mut x = [0.0; NUM_FEATURES];
    for row in (0..n).rev() {
        let rest: f64 = (row + 1..n).map(|k| a[row][k] * x[k]).sum();
        x[row] = (b[row] - rest) / a[row][row];
    }
    Some(x)
}

/// Maximum-likelihood weights of the conditional logit P(winner) = softmax(w · features),
/// maximising mean log-likelihood minus `l2`·|w|², by Newton's method with step halving.
pub fn fit(
    samples: &[Sample],
    start: [f32; NUM_FEATURES],
    iterations: u32,
    l2: f64,
) -> [f32; NUM_FEATURES] {
    let n = samples.len().max(1) as f64;
    let objective =
        |w: &[f64; NUM_FEATURES]| mean_ll(samples, w) - l2 * w.iter().map(|x| x * x).sum::<f64>();
    let mut w = start.map(|x| x as f64);
    for _ in 0..iterations {
        let mut g = [0.0f64; NUM_FEATURES];
        let mut h = [[0.0f64; NUM_FEATURES]; NUM_FEATURES];
        for s in samples {
            let pi = probabilities(scores(&w, &s.features));
            let f = s.features.map(|r| r.map(|x| x as f64));
            let mut mean = [0.0f64; NUM_FEATURES];
            for p in 0..NUM_PLAYERS {
                for i in 0..NUM_FEATURES {
                    mean[i] += pi[p] * f[p][i];
                }
            }
            for i in 0..NUM_FEATURES {
                g[i] += f[s.winner][i] - mean[i];
                for j in 0..NUM_FEATURES {
                    let second: f64 = (0..NUM_PLAYERS).map(|p| pi[p] * f[p][i] * f[p][j]).sum();
                    h[i][j] -= second - mean[i] * mean[j];
                }
            }
        }
        for i in 0..NUM_FEATURES {
            g[i] = g[i] / n - 2.0 * l2 * w[i];
            for j in 0..NUM_FEATURES {
                h[i][j] /= n;
            }
            h[i][i] -= 2.0 * l2;
        }
        let Some(step) = solve(h, g) else { break };
        let base = objective(&w);
        // Halve the step until the objective does not fall; if even a tiny step lowers it,
        // stop with `w` as it is rather than accept a worse point.
        let mut t = 1.0;
        let next = loop {
            let cand: [f64; NUM_FEATURES] = std::array::from_fn(|i| w[i] - t * step[i]);
            if objective(&cand) >= base {
                break Some(cand);
            }
            t /= 2.0;
            if t < 1e-6 {
                break None;
            }
        };
        let Some(next) = next else { break };
        let moved: f64 = (0..NUM_FEATURES).map(|i| (next[i] - w[i]).abs()).sum();
        w = next;
        if moved < 1e-9 {
            break;
        }
    }
    w.map(|x| x as f32)
}
