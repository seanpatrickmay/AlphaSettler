//! What one player can infer about everyone's resource cards from the events they have seen.
//!
//! Every resource movement is public except a steal the viewer neither made nor suffered. Given
//! the resource taken in each such steal, every hand follows exactly, so the belief is the exact
//! posterior over all four hands: a hidden steal splits each possible state by the resource
//! taken, weighted by the victim's holding; every other event moves each state deterministically
//! and drops the impossible ones. Behavioural evidence ("they would have built if they could") is
//! ignored in 3a (spec Section 2).

use settler_engine::rng::Rng;
use settler_engine::rules::roll::{pick_card, sample_cards};
use settler_engine::types::*;
use settler_engine::{Event, Observation, Phase};
use std::collections::BTreeMap;

/// Settlements and roads placed during setup (two of each per player), which cost nothing.
const SETUP_PIECES: u8 = 8;

/// All four hands, advanced by public events; also the exact tracker for a fully known log.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct HandTracker {
    pub hands: [Hand; NUM_PLAYERS],
    setup_settlements: u8,
    setup_roads: u8,
    /// Free Road Building roads still available to `free_road_player`.
    free_roads: u8,
    free_road_player: PlayerId,
    /// What the pending trade's offerer wants back (all zero when no trade is pending); every
    /// acceptor must hold it. Identical across the states of one belief.
    pending_get: Hand,
}

impl HandTracker {
    /// Hands at the start of a game: all empty, setup not begun.
    pub fn new() -> HandTracker {
        HandTracker::default()
    }

    /// Known `hands` at the position `obs` shows, with the free-build bookkeeping it implies.
    pub fn resume(hands: [Hand; NUM_PLAYERS], obs: &Observation) -> HandTracker {
        let placed = |count: &dyn Fn(usize) -> u32| {
            (0..NUM_PLAYERS)
                .map(count)
                .sum::<u32>()
                .min(SETUP_PIECES as u32) as u8
        };
        let (free_roads, free_road_player) = match obs.phase {
            Phase::RoadBuilding { roads_left } => (roads_left, obs.current),
            _ => (0, 0),
        };
        HandTracker {
            hands,
            setup_settlements: placed(&|p| (obs.settlements[p] | obs.cities[p]).count_ones()),
            setup_roads: placed(&|p| obs.roads[p].count_ones()),
            free_roads,
            free_road_player,
            pending_get: obs.trade.map_or([0; NUM_RESOURCES], |t| t.get),
        }
    }

    /// Advance by one event. `rng` draws the resource of a steal whose resource is hidden.
    /// Returns false (leaving the tracker unspecified) when these hands make `e` impossible.
    pub fn step(&mut self, e: &Event, rng: &mut Rng) -> bool {
        if let Some((p, cost)) = self.cost(e) {
            if !covers(&self.hands[p], &cost) {
                return false;
            }
            hand_sub(&mut self.hands[p], &cost);
        }
        let h = &mut self.hands;
        match *e {
            Event::Produced { player, resources }
            | Event::YearOfPlentyTaken { player, resources } => {
                hand_add(&mut h[player as usize], &resources);
            }
            Event::Discarded { player, resource } => {
                let c = &mut h[player as usize][resource.index()];
                if *c == 0 {
                    return false;
                }
                *c -= 1;
            }
            Event::Stole {
                thief,
                victim,
                resource,
            } => {
                let v = victim as usize;
                let r = match resource {
                    Some(r) if h[v][r.index()] > 0 => r,
                    Some(_) => return false,
                    None if hand_total(&h[v]) > 0 => pick_card(&h[v], rng),
                    None => return false,
                };
                h[v][r.index()] -= 1;
                h[thief as usize][r.index()] += 1;
            }
            Event::MonopolyTaken {
                player,
                resource,
                from,
            } => {
                let (p, r) = (player as usize, resource.index());
                if (0..NUM_PLAYERS).any(|q| q != p && h[q][r] != from[q]) {
                    return false;
                }
                for q in 0..NUM_PLAYERS {
                    if q != p {
                        h[q][r] = 0;
                    }
                }
                h[p][r] += from.iter().sum::<u8>();
            }
            Event::MaritimeTraded { player, gave, got } => {
                let mine = &mut h[player as usize];
                if !covers(mine, &gave) {
                    return false;
                }
                hand_sub(mine, &gave);
                hand_add(mine, &got);
            }
            Event::TradeConfirmed {
                offerer,
                partner,
                offerer_gave,
                partner_gave,
            } => {
                let (o, q) = (offerer as usize, partner as usize);
                if !covers(&h[o], &offerer_gave) || !covers(&h[q], &partner_gave) {
                    return false;
                }
                hand_sub(&mut h[o], &offerer_gave);
                hand_add(&mut h[q], &offerer_gave);
                hand_sub(&mut h[q], &partner_gave);
                hand_add(&mut h[o], &partner_gave);
                self.pending_get = [0; NUM_RESOURCES];
            }
            Event::TradeOffered { player, give, get } => {
                if !covers(&h[player as usize], &give) {
                    return false;
                }
                self.pending_get = get;
            }
            Event::TradeResponded {
                player,
                accepted: true,
            } => {
                if !covers(&h[player as usize], &self.pending_get) {
                    return false;
                }
            }
            Event::TradeCancelled { .. } => self.pending_get = [0; NUM_RESOURCES],
            _ => {}
        }
        true
    }

    /// Who paid what for `e`, updating which builds are free. A Road Building window closes at
    /// the first event that is not one of its roads.
    fn cost(&mut self, e: &Event) -> Option<(usize, Hand)> {
        let window = std::mem::take(&mut self.free_roads);
        match *e {
            Event::BuiltSettlement { player, .. } => {
                if self.setup_settlements < SETUP_PIECES {
                    self.setup_settlements += 1;
                    None
                } else {
                    Some((player as usize, SETTLEMENT_COST))
                }
            }
            Event::BuiltRoad { player, .. } => {
                if self.setup_roads < SETUP_PIECES {
                    self.setup_roads += 1;
                    None
                } else if window > 0 && player == self.free_road_player {
                    self.free_roads = window - 1;
                    None
                } else {
                    Some((player as usize, ROAD_COST))
                }
            }
            Event::BuiltCity { player, .. } => Some((player as usize, CITY_COST)),
            Event::BoughtDev { player, .. } => Some((player as usize, DEV_COST)),
            Event::PlayedDev {
                player,
                card: DevCard::RoadBuilding,
            } => {
                self.free_roads = 2;
                self.free_road_player = player;
                None
            }
            _ => None,
        }
    }
}

/// Joint-hand states kept before the support is resampled down (never reached in the test games).
pub const DEFAULT_MAX_STATES: usize = 65_536;
/// Deals drawn for a belief built from an observation alone.
const OBSERVATION_DEALS: usize = 1024;

/// Opponents' hands dealt uniformly from the cards neither the bank nor the viewer holds, each
/// opponent getting their visible count.
fn deal_hidden_hands(obs: &Observation, rng: &mut Rng) -> [Hand; NUM_PLAYERS] {
    let me = obs.viewer as usize;
    let mut pool: Hand = std::array::from_fn(|r| {
        BANK_PER_RESOURCE
            .saturating_sub(obs.bank[r])
            .saturating_sub(obs.my_hand[r])
    });
    let mut hands = [[0u8; NUM_RESOURCES]; NUM_PLAYERS];
    hands[me] = obs.my_hand;
    for q in (0..NUM_PLAYERS).filter(|&q| q != me) {
        let k = (obs.hand_counts[q] as u32).min(hand_total(&pool)) as u8;
        hands[q] = sample_cards(&pool, k, rng);
        hand_sub(&mut pool, &hands[q]);
    }
    hands
}

/// Equal-weight states from `draws` (duplicates merged), in sorted order.
fn merged(draws: Vec<HandTracker>) -> Vec<(HandTracker, f64)> {
    let n = draws.len() as f64;
    let mut counts: BTreeMap<HandTracker, f64> = BTreeMap::new();
    for t in draws {
        *counts.entry(t).or_insert(0.0) += 1.0 / n;
    }
    counts.into_iter().collect()
}

/// One player's belief about every hand: the distinct joint hands still possible, with their
/// probabilities (summing to 1), in sorted order so everything built on it is deterministic.
#[derive(Clone, Debug)]
pub struct Belief {
    viewer: PlayerId,
    max_states: usize,
    states: Vec<(HandTracker, f64)>,
    rng: Rng,
    truncations: u64,
}

impl Belief {
    /// The belief at the start of a game: every hand empty, with certainty.
    pub fn new(viewer: PlayerId, max_states: usize, seed: u64) -> Belief {
        assert!(max_states > 0, "a belief needs room for at least one state");
        Belief {
            viewer,
            max_states,
            states: vec![(HandTracker::new(), 1.0)],
            rng: Rng::new(seed),
            truncations: 0,
        }
    }

    /// A belief for a viewer who has not seen the history: the cards nobody can see are dealt
    /// uniformly (`deal_hidden_hands`), approximated by 1,024 deals. Later events refine it.
    pub fn from_observation(obs: &Observation, max_states: usize, seed: u64) -> Belief {
        assert!(max_states > 0, "a belief needs room for at least one state");
        let mut rng = Rng::new(seed);
        let draws = (0..OBSERVATION_DEALS.min(max_states))
            .map(|_| HandTracker::resume(deal_hidden_hands(obs, &mut rng), obs))
            .collect();
        Belief {
            viewer: obs.viewer,
            max_states,
            states: merged(draws),
            rng,
            truncations: 0,
        }
    }

    pub fn viewer(&self) -> PlayerId {
        self.viewer
    }

    /// The possible joint hands and their probabilities.
    pub fn states(&self) -> &[(HandTracker, f64)] {
        &self.states
    }

    /// How many times the support outgrew `max_states` and was resampled down.
    pub fn truncations(&self) -> u64 {
        self.truncations
    }

    /// All four hands, drawn with their probability.
    pub fn sample_hands(&self, rng: &mut Rng) -> [Hand; NUM_PLAYERS] {
        let mut u = (rng.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
        for (t, w) in &self.states {
            if u < *w {
                return t.hands;
            }
            u -= w;
        }
        self.states.last().expect("a belief is never empty").0.hands
    }

    /// Advance by `events` as the viewer saw them, in log order. Err when no joint hands are
    /// consistent with the history (an inconsistent feed, or a tracker bug); the belief is then
    /// left as it was before the event that ruled every state out.
    pub fn observe(&mut self, events: &[Event]) -> Result<(), String> {
        let mut unused = Rng::new(0); // `step` draws only for hidden steals, which branch here instead
        for e in events {
            let next: Vec<(HandTracker, f64)> = if let Event::Stole {
                thief,
                victim,
                resource: None,
            } = *e
            {
                let mut next: BTreeMap<HandTracker, f64> = BTreeMap::new();
                for &(t, w) in &self.states {
                    let v = t.hands[victim as usize];
                    let total = hand_total(&v) as f64;
                    for r in Resource::ALL {
                        let k = v[r.index()];
                        if k == 0 {
                            continue;
                        }
                        let mut u = t;
                        if u.step(
                            &Event::Stole {
                                thief,
                                victim,
                                resource: Some(r),
                            },
                            &mut unused,
                        ) {
                            *next.entry(u).or_insert(0.0) += w * k as f64 / total;
                        }
                    }
                }
                next.into_iter().collect()
            } else {
                // Deterministic and injective on the hands, so states stay distinct and sorted.
                self.states
                    .iter()
                    .filter_map(|&(mut t, w)| t.step(e, &mut unused).then_some((t, w)))
                    .collect()
            };
            if next.is_empty() {
                return Err(format!(
                    "player {}'s belief: no hands are consistent with event {e:?}",
                    self.viewer
                ));
            }
            self.states = next;
            let z: f64 = self.states.iter().map(|s| s.1).sum();
            for s in &mut self.states {
                s.1 /= z;
            }
            if self.states.len() > self.max_states {
                self.truncate();
            }
        }
        Ok(())
    }

    /// Resample the support down to `max_states` equal-weight draws (merged).
    fn truncate(&mut self) {
        self.truncations += 1;
        let mut draws = Vec::with_capacity(self.max_states);
        for _ in 0..self.max_states {
            let mut u = (self.rng.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
            let mut pick = self.states.len() - 1;
            for (i, s) in self.states.iter().enumerate() {
                if u < s.1 {
                    pick = i;
                    break;
                }
                u -= s.1;
            }
            draws.push(self.states[pick].0);
        }
        self.states = merged(draws);
    }

    /// Err if some state disagrees with `obs`: the viewer's own hand or anyone's card count.
    pub fn check(&self, obs: &Observation) -> Result<(), String> {
        if obs.viewer != self.viewer {
            return Err(format!(
                "belief of player {} checked against player {}'s view",
                self.viewer, obs.viewer
            ));
        }
        for (t, _) in &self.states {
            if t.hands[self.viewer as usize] != obs.my_hand {
                return Err(format!(
                    "belief holds {:?} for player {}, who holds {:?}",
                    t.hands[self.viewer as usize], self.viewer, obs.my_hand
                ));
            }
            for q in 0..NUM_PLAYERS {
                if hand_total(&t.hands[q]) != obs.hand_counts[q] as u32 {
                    return Err(format!(
                        "belief gives player {q} {} cards, they hold {}",
                        hand_total(&t.hands[q]),
                        obs.hand_counts[q]
                    ));
                }
            }
        }
        Ok(())
    }
}
