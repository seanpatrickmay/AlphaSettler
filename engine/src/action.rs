//! Every move as one integer in a fixed space (see the table in Plan 1, Task 4).

use crate::types::{Hand, PlayerId, Resource};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Action {
    Roll,
    EndTurn,
    BuyDev,
    PlayKnight,
    PlayRoadBuilding,
    PlayMonopoly(Resource),
    /// Canonical order: first <= second.
    PlayYearOfPlenty(Resource, Resource),
    BuildSettlement(u8),
    BuildCity(u8),
    BuildRoad(u8),
    MoveRobber(u8),
    StealFrom(PlayerId),
    Discard(Resource),
    MaritimeTrade { give: Resource, get: Resource },
    /// Indices into the trade bundles (see `bundle`).
    OfferTrade { give: u8, get: u8 },
    AcceptTrade,
    RejectTrade,
    ConfirmTrade(PlayerId),
    CancelTrade,
}

pub const ACTION_SPACE_SIZE: usize = 665;

/// Unordered resource pairs (i <= j): Year of Plenty choices and 2-card trade bundles.
pub const PAIRS: [(u8, u8); 15] = [
    (0, 0), (0, 1), (0, 2), (0, 3), (0, 4),
    (1, 1), (1, 2), (1, 3), (1, 4),
    (2, 2), (2, 3), (2, 4),
    (3, 3), (3, 4),
    (4, 4),
];

/// Bundles 0..5 are one card of a resource; 5..20 are the `PAIRS`.
pub const NUM_BUNDLES: usize = 20;

pub fn bundle(i: u8) -> Hand {
    let mut h = [0u8; 5];
    if i < 5 {
        h[i as usize] = 1;
    } else {
        let (a, b) = PAIRS[(i - 5) as usize];
        h[a as usize] += 1;
        h[b as usize] += 1;
    }
    h
}

#[inline]
pub fn bundle_size(i: u8) -> u8 {
    if i < 5 {
        1
    } else {
        2
    }
}

fn pair_index(a: Resource, b: Resource) -> u16 {
    let (a, b) = (a.index().min(b.index()) as u8, a.index().max(b.index()) as u8);
    PAIRS.iter().position(|&p| p == (a, b)).unwrap() as u16
}

impl Action {
    pub fn encode(self) -> u16 {
        match self {
            Action::Roll => 0,
            Action::EndTurn => 1,
            Action::BuyDev => 2,
            Action::PlayKnight => 3,
            Action::PlayRoadBuilding => 4,
            Action::PlayMonopoly(r) => 5 + r.index() as u16,
            Action::PlayYearOfPlenty(a, b) => 10 + pair_index(a, b),
            Action::BuildSettlement(n) => 25 + n as u16,
            Action::BuildCity(n) => 79 + n as u16,
            Action::BuildRoad(e) => 133 + e as u16,
            Action::MoveRobber(t) => 205 + t as u16,
            Action::StealFrom(p) => 224 + p as u16,
            Action::Discard(r) => 228 + r.index() as u16,
            Action::MaritimeTrade { give, get } => 233 + 5 * give.index() as u16 + get.index() as u16,
            Action::OfferTrade { give, get } => 258 + 20 * give as u16 + get as u16,
            Action::AcceptTrade => 658,
            Action::RejectTrade => 659,
            Action::ConfirmTrade(p) => 660 + p as u16,
            Action::CancelTrade => 664,
        }
    }

    pub fn decode(id: u16) -> Option<Action> {
        let r = |i: u16| Resource::from_index(i as usize);
        Some(match id {
            0 => Action::Roll,
            1 => Action::EndTurn,
            2 => Action::BuyDev,
            3 => Action::PlayKnight,
            4 => Action::PlayRoadBuilding,
            5..=9 => Action::PlayMonopoly(r(id - 5)),
            10..=24 => {
                let (a, b) = PAIRS[(id - 10) as usize];
                Action::PlayYearOfPlenty(r(a as u16), r(b as u16))
            }
            25..=78 => Action::BuildSettlement((id - 25) as u8),
            79..=132 => Action::BuildCity((id - 79) as u8),
            133..=204 => Action::BuildRoad((id - 133) as u8),
            205..=223 => Action::MoveRobber((id - 205) as u8),
            224..=227 => Action::StealFrom((id - 224) as u8),
            228..=232 => Action::Discard(r(id - 228)),
            233..=257 => {
                let k = id - 233;
                let (give, get) = (k / 5, k % 5);
                if give == get {
                    return None;
                }
                Action::MaritimeTrade { give: r(give), get: r(get) }
            }
            258..=657 => {
                let k = id - 258;
                Action::OfferTrade { give: (k / 20) as u8, get: (k % 20) as u8 }
            }
            658 => Action::AcceptTrade,
            659 => Action::RejectTrade,
            660..=663 => Action::ConfirmTrade((id - 660) as u8),
            664 => Action::CancelTrade,
            _ => return None,
        })
    }
}
