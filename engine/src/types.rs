//! Core value types shared across the engine.

pub const NUM_PLAYERS: usize = 4;
pub const NUM_RESOURCES: usize = 5;

pub type PlayerId = u8;
/// Resource counts indexed by `Resource::index()`.
pub type Hand = [u8; NUM_RESOURCES];

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(u8)]
pub enum Resource {
    Wood = 0,
    Brick = 1,
    Sheep = 2,
    Wheat = 3,
    Ore = 4,
}

impl Resource {
    pub const ALL: [Resource; NUM_RESOURCES] = [
        Resource::Wood,
        Resource::Brick,
        Resource::Sheep,
        Resource::Wheat,
        Resource::Ore,
    ];

    #[inline]
    pub fn index(self) -> usize {
        self as usize
    }

    #[inline]
    pub fn from_index(i: usize) -> Resource {
        Self::ALL[i]
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum DevCard {
    Knight = 0,
    VictoryPoint = 1,
    RoadBuilding = 2,
    YearOfPlenty = 3,
    Monopoly = 4,
}

impl DevCard {
    pub const ALL: [DevCard; 5] = [
        DevCard::Knight,
        DevCard::VictoryPoint,
        DevCard::RoadBuilding,
        DevCard::YearOfPlenty,
        DevCard::Monopoly,
    ];

    #[inline]
    pub fn index(self) -> usize {
        self as usize
    }

    #[inline]
    pub fn from_index(i: usize) -> DevCard {
        Self::ALL[i]
    }
}

/// Cards per type, indexed by `DevCard::index()`.
pub const DEV_DECK_COUNTS: [u8; 5] = [14, 5, 2, 2, 2];
pub const DEV_DECK_SIZE: usize = 25;
pub const BANK_PER_RESOURCE: u8 = 19;

pub const MAX_SETTLEMENTS: u32 = 5;
pub const MAX_CITIES: u32 = 4;
pub const MAX_ROADS: u32 = 15;

pub const ROAD_COST: Hand = [1, 1, 0, 0, 0];
pub const SETTLEMENT_COST: Hand = [1, 1, 1, 1, 0];
pub const CITY_COST: Hand = [0, 0, 0, 2, 3];
pub const DEV_COST: Hand = [0, 0, 1, 1, 1];

#[inline]
pub fn covers(hand: &Hand, cost: &Hand) -> bool {
    (0..NUM_RESOURCES).all(|i| hand[i] >= cost[i])
}

#[inline]
pub fn hand_sub(hand: &mut Hand, x: &Hand) {
    for i in 0..NUM_RESOURCES {
        hand[i] -= x[i];
    }
}

#[inline]
pub fn hand_add(hand: &mut Hand, x: &Hand) {
    for i in 0..NUM_RESOURCES {
        hand[i] += x[i];
    }
}

#[inline]
pub fn hand_total(hand: &Hand) -> u32 {
    hand.iter().map(|&c| c as u32).sum()
}

/// Indices of set bits, lowest first.
pub fn bits64(mut m: u64) -> impl Iterator<Item = u8> {
    std::iter::from_fn(move || {
        if m == 0 {
            return None;
        }
        let i = m.trailing_zeros() as u8;
        m &= m - 1;
        Some(i)
    })
}

/// Indices of set bits, lowest first.
pub fn bits128(mut m: u128) -> impl Iterator<Item = u8> {
    std::iter::from_fn(move || {
        if m == 0 {
            return None;
        }
        let i = m.trailing_zeros() as u8;
        m &= m - 1;
        Some(i)
    })
}
