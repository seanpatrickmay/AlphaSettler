//! Compound build moves (sub-project 3d-4, docs/superpowers/specs/2026-10-08-compound-moves-design.md
//! Section 2). At the searching player's own Main decisions, with `SearchConfig::compound`, the
//! search's move set keeps every legal action but the bare maritime trades, and adds one compound
//! move for each build target that is not affordable now but is after at most `MAX_TRADES`
//! maritime trades: those trades, then the build. The engine's action space is unchanged: a
//! compound is a search edge keyed `COMPOUND_BIT | build id`, applied step by step.

use settler_engine::rules::{build, dev};
use settler_engine::{
    Action, Hand, Phase, PlayerId, Resource, State, CITY_COST, DEV_COST, NUM_RESOURCES, ROAD_COST,
    SETTLEMENT_COST,
};

/// Set in a compound move's key; the rest of the key is its build's action id (all below 665).
pub const COMPOUND_BIT: u16 = 0x8000;
/// Most maritime trades a compound makes before its build.
pub const MAX_TRADES: usize = 2;
/// A hand covering every build cost: the targets are what the position allows, whatever the hand.
const EVERY_COST: Hand = [1, 1, 1, 2, 3];

/// One or two maritime trades, then the build they make affordable.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Compound {
    /// In the order they are made.
    pub trades: Vec<Action>,
    /// A road, settlement, city or development card.
    pub build: Action,
}

impl Compound {
    pub fn key(&self) -> u16 {
        compound_key(self.build)
    }

    /// Every step: the trades, then the build.
    pub fn actions(&self) -> Vec<Action> {
        let mut v = self.trades.clone();
        v.push(self.build);
        v
    }
}

/// The key of the compound that ends in `build`: there is at most one per target.
pub fn compound_key(build: Action) -> u16 {
    COMPOUND_BIT | build.encode()
}

/// What `build` costs; panics for an action that is not a build target.
pub fn cost(build: Action) -> Hand {
    match build {
        Action::BuildRoad(_) => ROAD_COST,
        Action::BuildSettlement(_) => SETTLEMENT_COST,
        Action::BuildCity(_) => CITY_COST,
        Action::BuyDev => DEV_COST,
        _ => panic!("{build:?} is not a build target"),
    }
}

/// Whether `s` is a decision that searches compound moves: a Main decision of `viewer` (the
/// searching player). Opponents' decisions keep their plain moves (spec Section 5).
pub fn applies(s: &State, viewer: PlayerId) -> bool {
    s.phase == Phase::Main && s.current_actor() == viewer
}

/// Every build the position lets the player to move make, ignoring only the hand (spec Section 2):
/// the engine's own rules (`legal_main_builds`, and `legal_main_dev`'s `BuyDev`) on a copy whose
/// hand covers every cost. Roads, settlements and cities in the engine's order, then the dev card.
pub fn targets(s: &State) -> Vec<Action> {
    let mut rich = *s;
    rich.players[s.current as usize].hand = EVERY_COST;
    let mut out = Vec::new();
    build::legal_main_builds(&rich, &mut out);
    let mut dev_moves = Vec::new();
    dev::legal_main_dev(&rich, &mut dev_moves);
    if dev_moves.contains(&Action::BuyDev) {
        out.push(Action::BuyDev);
    }
    out
}

/// The maritime trades that make `build` affordable for the player to move (spec Section 2), or
/// `None` if it is affordable now, needs more than `MAX_TRADES` trades, or the rule finds no trade
/// the bank can fill. The shortfall is covered one card at a time, resources in index order; each
/// trade gives, among the other resources whose cards beyond the target's cost cover their rate,
/// the one with the most such cards (ties to the lower index), at the player's rate for it. The
/// trades run on a copy of the hand and the bank, so each is legal where it is made.
pub fn trades_for(s: &State, build: Action) -> Option<Vec<Action>> {
    let p = s.current as usize;
    let cost = cost(build);
    let mut hand = s.players[p].hand;
    let short: usize = (0..NUM_RESOURCES)
        .map(|r| cost[r].saturating_sub(hand[r]) as usize)
        .sum();
    if short == 0 || short > MAX_TRADES {
        return None;
    }
    let buildings = s.buildings(p);
    let mut bank = s.bank;
    let mut trades = Vec::with_capacity(short);
    for get in Resource::ALL {
        while hand[get.index()] < cost[get.index()] {
            if bank[get.index()] == 0 {
                return None;
            }
            // (resource, cards to spare, rate)
            let mut best: Option<(Resource, u8, u8)> = None;
            for give in Resource::ALL {
                if give == get {
                    continue;
                }
                let spare = hand[give.index()].saturating_sub(cost[give.index()]);
                let rate = s.board.maritime_rate(buildings, give);
                if spare >= rate && best.map_or(true, |(_, b, _)| spare > b) {
                    best = Some((give, spare, rate));
                }
            }
            let (give, _, rate) = best?;
            hand[give.index()] -= rate;
            bank[give.index()] += rate;
            hand[get.index()] += 1;
            bank[get.index()] -= 1;
            trades.push(Action::MaritimeTrade { give, get });
        }
    }
    Some(trades)
}

/// Every compound move of the player to move in `s`, in `targets` order, into `out` (cleared).
pub fn compounds(s: &State, out: &mut Vec<Compound>) {
    out.clear();
    for build in targets(s) {
        if let Some(trades) = trades_for(s, build) {
            out.push(Compound { trades, build });
        }
    }
}

fn is_trade(a: &Action) -> bool {
    matches!(a, Action::MaritimeTrade { .. })
}

/// The keys of a node's moves into `out` (cleared): at a compound node, `plain`'s moves but the
/// bare maritime trades, then the compounds; otherwise every move of `plain`.
pub fn move_keys(
    plain: &[Action],
    compounds: &[Compound],
    compound_node: bool,
    out: &mut Vec<u16>,
) {
    out.clear();
    if !compound_node {
        out.extend(plain.iter().map(|a| a.encode()));
        return;
    }
    out.extend(plain.iter().filter(|a| !is_trade(a)).map(|a| a.encode()));
    out.extend(compounds.iter().map(Compound::key));
}

/// The priors of a compound node's moves, aligned with `move_keys(plain, compounds, true, ..)`
/// (spec Section 2): kept moves keep their prior from `prior` (aligned with `plain`); a compound
/// gets its first trade's, split equally among the compounds starting with that trade; then all
/// are renormalised, or made uniform if nothing is left.
pub fn move_priors(plain: &[Action], prior: &[f32], compounds: &[Compound]) -> Vec<f32> {
    let mut p: Vec<f32> = plain
        .iter()
        .zip(prior)
        .filter(|(a, _)| !is_trade(a))
        .map(|(_, &x)| x)
        .collect();
    for c in compounds {
        let first = c.trades[0];
        let sharing = compounds.iter().filter(|d| d.trades[0] == first).count() as f32;
        let x = plain
            .iter()
            .position(|&a| a == first)
            .map_or(0.0, |i| prior[i]);
        p.push(x / sharing);
    }
    let total: f32 = p.iter().sum();
    if total > 0.0 && total.is_finite() {
        for x in p.iter_mut() {
            *x /= total;
        }
    } else {
        let u = 1.0 / p.len() as f32;
        p.iter_mut().for_each(|x| *x = u);
    }
    p
}

/// The compound keyed `key` among `compounds`; `None` for a plain move's key.
pub fn find(compounds: &[Compound], key: u16) -> Option<&Compound> {
    if key & COMPOUND_BIT == 0 {
        return None;
    }
    compounds.iter().find(|c| c.key() == key)
}

/// The move keyed `key`: its first step and the steps after it (none for a plain move).
pub fn move_of(key: u16, compounds: &[Compound]) -> (Action, Vec<Action>) {
    match find(compounds, key) {
        Some(c) => (
            c.trades[0],
            c.trades[1..].iter().copied().chain([c.build]).collect(),
        ),
        None => (
            Action::decode(key).expect("a move key is an action id or a compound key"),
            Vec::new(),
        ),
    }
}
