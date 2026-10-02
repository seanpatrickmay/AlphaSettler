//! A simple greedy baseline: builds whenever it can, using pip-count placement scoring.
//! Never offers domestic trades and rejects every offer.

use crate::Bot;
use settler_engine::topology::topo;
use settler_engine::types::*;
use settler_engine::{Action, Observation, Phase};

pub struct GreedyBot;

impl GreedyBot {
    pub fn new(_seed: u64) -> GreedyBot {
        GreedyBot
    }
}

/// Dots on a number token: 2 and 12 have 1, 6 and 8 have 5, the desert 0.
pub fn pips(number: u8) -> u32 {
    if number == 0 {
        0
    } else {
        6 - (7 - number as i32).unsigned_abs()
    }
}

/// Pips of a node's producing tiles plus 1 per distinct resource.
pub fn node_value(obs: &Observation, n: u8) -> u32 {
    let mut value = 0;
    let mut seen = [false; NUM_RESOURCES];
    for &tile in &topo().node_tiles[n as usize] {
        if let Some(r) = obs.board.tile_resource[tile as usize] {
            value += pips(obs.board.tile_number[tile as usize]);
            if !seen[r.index()] {
                seen[r.index()] = true;
                value += 1;
            }
        }
    }
    value
}

fn occupied(obs: &Observation) -> u64 {
    (0..NUM_PLAYERS).fold(0, |m, p| m | obs.settlements[p] | obs.cities[p])
}

/// A free node where the distance rule allows a settlement.
fn spot_ok(obs: &Observation, n: u8) -> bool {
    let occ = occupied(obs);
    occ & (1u64 << n) == 0 && occ & topo().node_neighbor_mask[n as usize] == 0
}

/// The best settlement spot a road on `e` brings within reach (its endpoints and their neighbours).
pub fn road_value(obs: &Observation, e: u8) -> u32 {
    let t = topo();
    let (a, b) = t.edge_nodes[e as usize];
    let mut best = 0;
    for x in [a, b] {
        if spot_ok(obs, x) {
            best = best.max(node_value(obs, x));
        }
        for &y in &t.node_neighbors[x as usize] {
            if spot_ok(obs, y) {
                best = best.max(node_value(obs, y));
            }
        }
    }
    best
}

/// The legal action with the highest score (`None` = not a candidate); first wins ties.
fn best_by(legal: &[Action], score: impl Fn(Action) -> Option<u32>) -> Option<Action> {
    let mut best: Option<(u32, Action)> = None;
    for &a in legal {
        if let Some(v) = score(a) {
            if best.map_or(true, |(bv, _)| v > bv) {
                best = Some((v, a));
            }
        }
    }
    best.map(|(_, a)| a)
}

fn my_buildings(obs: &Observation) -> u64 {
    let me = obs.viewer as usize;
    obs.settlements[me] | obs.cities[me]
}

fn robber_on_me(obs: &Observation) -> bool {
    topo().tile_node_mask[obs.robber as usize] & my_buildings(obs) != 0
}

/// Cost of the build the bot is saving for.
fn target_cost(obs: &Observation) -> Hand {
    let me = obs.viewer as usize;
    if obs.settlements[me] != 0 && obs.cities[me].count_ones() < MAX_CITIES {
        CITY_COST
    } else if obs.settlements[me].count_ones() < MAX_SETTLEMENTS {
        SETTLEMENT_COST
    } else {
        DEV_COST
    }
}

fn deficit(obs: &Observation) -> Hand {
    let cost = target_cost(obs);
    std::array::from_fn(|r| cost[r].saturating_sub(obs.my_hand[r]))
}

fn year_of_plenty_choice(obs: &Observation, legal: &[Action]) -> Option<Action> {
    let need = deficit(obs);
    best_by(legal, |a| match a {
        Action::PlayYearOfPlenty(x, y) => {
            let mut left = need;
            let mut covered = 0;
            for r in [x, y] {
                if left[r.index()] > 0 {
                    left[r.index()] -= 1;
                    covered += 1;
                }
            }
            (covered > 0).then_some(covered)
        }
        _ => None,
    })
}

fn maritime_choice(obs: &Observation, legal: &[Action]) -> Option<Action> {
    let cost = target_cost(obs);
    let need = deficit(obs);
    let buildings = my_buildings(obs);
    best_by(legal, |a| match a {
        Action::MaritimeTrade { give, get } => {
            let rate = obs.board.maritime_rate(buildings, give);
            let surplus = obs.my_hand[give.index()] >= rate + cost[give.index()];
            (need[get.index()] > 0 && surplus).then_some(need[get.index()] as u32)
        }
        _ => None,
    })
}

fn main_choice(obs: &Observation, legal: &[Action]) -> Option<Action> {
    let city = best_by(legal, |a| match a {
        Action::BuildCity(n) => Some(node_value(obs, n)),
        _ => None,
    });
    if city.is_some() {
        return city;
    }
    let settlement = best_by(legal, |a| match a {
        Action::BuildSettlement(n) => Some(node_value(obs, n)),
        _ => None,
    });
    if settlement.is_some() {
        return settlement;
    }
    if legal.contains(&Action::PlayKnight) && robber_on_me(obs) {
        return Some(Action::PlayKnight);
    }
    if legal.contains(&Action::PlayRoadBuilding) {
        return Some(Action::PlayRoadBuilding);
    }
    if let Some(y) = year_of_plenty_choice(obs, legal) {
        return Some(y);
    }
    let road = best_by(legal, |a| match a {
        Action::BuildRoad(e) => Some(road_value(obs, e)).filter(|&v| v > 0),
        _ => None,
    });
    if road.is_some() {
        return road;
    }
    if legal.contains(&Action::BuyDev) {
        return Some(Action::BuyDev);
    }
    if let Some(m) = maritime_choice(obs, legal) {
        return Some(m);
    }
    Some(Action::EndTurn)
}

fn robber_choice(obs: &Observation, legal: &[Action]) -> Option<Action> {
    let t = topo();
    let mine = my_buildings(obs);
    best_by(legal, |a| match a {
        Action::MoveRobber(tile) => {
            let mask = t.tile_node_mask[tile as usize];
            if mask & mine != 0 {
                return None;
            }
            let p = pips(obs.board.tile_number[tile as usize]);
            let mut score = 0;
            for q in 0..NUM_PLAYERS {
                if q == obs.viewer as usize {
                    continue;
                }
                let weight = (obs.settlements[q] & mask).count_ones()
                    + 2 * (obs.cities[q] & mask).count_ones();
                if weight > 0 {
                    score += weight * p + 1;
                }
            }
            Some(score)
        }
        _ => None,
    })
}

/// GreedyBot's move for `obs.viewer` among `legal` (non-empty). Also the heuristic evaluator's prior.
pub fn choose(obs: &Observation, legal: &[Action]) -> Action {
    let pick = match obs.phase {
        Phase::SetupSettlement => best_by(legal, |a| match a {
            Action::BuildSettlement(n) => Some(node_value(obs, n)),
            _ => None,
        }),
        Phase::SetupRoad { .. } | Phase::RoadBuilding { .. } => best_by(legal, |a| match a {
            Action::BuildRoad(e) => Some(road_value(obs, e)),
            _ => None,
        }),
        Phase::PreRoll => {
            if legal.contains(&Action::PlayKnight) && robber_on_me(obs) {
                Some(Action::PlayKnight)
            } else {
                Some(Action::Roll)
            }
        }
        Phase::Discard => best_by(legal, |a| match a {
            Action::Discard(r) => Some(obs.my_hand[r.index()] as u32),
            _ => None,
        }),
        Phase::MoveRobber => robber_choice(obs, legal),
        Phase::Steal => best_by(legal, |a| match a {
            Action::StealFrom(p) => {
                Some(obs.public_vp[p as usize] as u32 * 100 + obs.hand_counts[p as usize] as u32)
            }
            _ => None,
        }),
        Phase::Main => main_choice(obs, legal),
        Phase::TradeResponse => Some(Action::RejectTrade),
        Phase::TradeConfirm => Some(Action::CancelTrade),
        Phase::GameOver { .. } => None,
    };
    pick.filter(|a| legal.contains(a)).unwrap_or(legal[0])
}

impl Bot for GreedyBot {
    fn name(&self) -> &'static str {
        "greedy"
    }

    fn act(&mut self, obs: &Observation, legal: &[Action]) -> Action {
        choose(obs, legal)
    }
}
