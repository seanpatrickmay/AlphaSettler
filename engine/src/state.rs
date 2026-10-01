//! The complete game state: a fixed-size `Copy` value.

use crate::board::Board;
use crate::config::GameConfig;
use crate::rng::{mix, Rng};
use crate::types::*;

const BOARD_SALT: u64 = 0xB0A2D;
const DECK_SALT: u64 = 0xDEC4;
const STEAL_SALT: u64 = 0x57EA1;
const MISC_SALT: u64 = 0x3155C;

/// Who places in each of the 8 setup rounds.
pub const SETUP_ORDER: [PlayerId; 8] = [0, 1, 2, 3, 3, 2, 1, 0];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    SetupSettlement,
    SetupRoad {
        node: u8,
    },
    PreRoll,
    /// Players with `discard_remaining > 0` discard one card at a time, lowest seat first.
    Discard,
    MoveRobber,
    Steal,
    Main,
    RoadBuilding {
        roads_left: u8,
    },
    /// Opponents answer the pending offer in seat order after the offerer.
    TradeResponse,
    TradeConfirm,
    GameOver {
        winner: Option<PlayerId>,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct PlayerState {
    pub hand: Hand,
    /// Every dev card held, including victory points and cards bought this turn.
    pub dev_hand: [u8; 5],
    /// Cards bought this turn (not yet playable).
    pub dev_new: [u8; 5],
    pub knights_played: u8,
    /// Dev cards played so far, by kind (knights included).
    pub dev_played: [u8; 5],
    pub settlements: u64,
    pub cities: u64,
    pub roads: u128,
    pub longest_road_len: u8,
    pub discard_remaining: u8,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Response {
    Pending,
    Accepted,
    Rejected,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PendingTrade {
    /// What the offerer gives.
    pub give: Hand,
    /// What the offerer wants back.
    pub get: Hand,
    pub responses: [Response; NUM_PLAYERS],
    pub next_responder: PlayerId,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct State {
    pub config: GameConfig,
    pub seed: u64,
    pub board: Board,
    pub players: [PlayerState; NUM_PLAYERS],
    pub bank: Hand,
    pub dev_deck: [DevCard; DEV_DECK_SIZE],
    pub dev_deck_pos: u8,
    pub robber: u8,
    pub current: PlayerId,
    pub phase: Phase,
    pub turn: u32,
    pub setup_step: u8,
    pub dev_played_this_turn: bool,
    pub offers_this_turn: u8,
    pub trade: Option<PendingTrade>,
    /// Phase resumed after the robber is resolved or Road Building ends: PreRoll when the card
    /// was played before rolling, otherwise Main.
    pub return_phase: Phase,
    pub longest_road_owner: Option<PlayerId>,
    pub largest_army_owner: Option<PlayerId>,
    pub rng_steal: Rng,
    pub rng_misc: Rng,
}

impl State {
    pub fn new(seed: u64, config: GameConfig) -> State {
        let board = Board::random(&mut Rng::new(mix(seed, BOARD_SALT)));
        State::with_board(seed, config, board)
    }

    pub fn with_board(seed: u64, config: GameConfig, board: Board) -> State {
        config.validate().expect("invalid GameConfig");
        let mut dev_deck = [DevCard::Knight; DEV_DECK_SIZE];
        let mut k = 0;
        for (i, &count) in DEV_DECK_COUNTS.iter().enumerate() {
            for _ in 0..count {
                dev_deck[k] = DevCard::from_index(i);
                k += 1;
            }
        }
        Rng::new(mix(seed, DECK_SALT)).shuffle(&mut dev_deck);
        State {
            config,
            seed,
            board,
            players: [PlayerState::default(); NUM_PLAYERS],
            bank: [BANK_PER_RESOURCE; NUM_RESOURCES],
            dev_deck,
            dev_deck_pos: 0,
            robber: board.desert(),
            current: SETUP_ORDER[0],
            phase: Phase::SetupSettlement,
            turn: 0,
            setup_step: 0,
            dev_played_this_turn: false,
            offers_this_turn: 0,
            trade: None,
            return_phase: Phase::Main,
            longest_road_owner: None,
            largest_army_owner: None,
            rng_steal: Rng::new(mix(seed, STEAL_SALT)),
            rng_misc: Rng::new(mix(seed, MISC_SALT)),
        }
    }

    /// Replace the game seed and the steal and misc streams with the ones `State::new(seed, ..)`
    /// would derive. Future dice follow the new seed; the board, hands, deck order and everything
    /// else stay as they are. Search uses this so a sampled world's future is not the real game's.
    pub fn reseed(&mut self, seed: u64) {
        self.seed = seed;
        self.rng_steal = Rng::new(mix(seed, STEAL_SALT));
        self.rng_misc = Rng::new(mix(seed, MISC_SALT));
    }

    /// The player who must choose the next action.
    pub fn current_actor(&self) -> PlayerId {
        match self.phase {
            Phase::Discard => (0..NUM_PLAYERS)
                .find(|&p| self.players[p].discard_remaining > 0)
                .expect("Discard phase with nobody discarding")
                as PlayerId,
            Phase::TradeResponse => {
                self.trade
                    .expect("TradeResponse without a trade")
                    .next_responder
            }
            _ => self.current,
        }
    }

    #[inline]
    pub fn is_over(&self) -> bool {
        matches!(self.phase, Phase::GameOver { .. })
    }

    pub fn winner(&self) -> Option<PlayerId> {
        match self.phase {
            Phase::GameOver { winner } => winner,
            _ => None,
        }
    }

    #[inline]
    pub fn buildings(&self, p: usize) -> u64 {
        self.players[p].settlements | self.players[p].cities
    }

    #[inline]
    pub fn occupied_nodes(&self) -> u64 {
        (0..NUM_PLAYERS).fold(0, |m, p| m | self.buildings(p))
    }

    #[inline]
    pub fn occupied_edges(&self) -> u128 {
        self.players.iter().fold(0, |m, p| m | p.roads)
    }

    /// Victory points visible to everyone (excludes hidden VP cards).
    pub fn public_vp(&self, p: usize) -> u8 {
        let pl = &self.players[p];
        let mut vp = pl.settlements.count_ones() as u8 + 2 * pl.cities.count_ones() as u8;
        if self.longest_road_owner == Some(p as PlayerId) {
            vp += 2;
        }
        if self.largest_army_owner == Some(p as PlayerId) {
            vp += 2;
        }
        vp
    }

    pub fn total_vp(&self, p: usize) -> u8 {
        self.public_vp(p) + self.players[p].dev_hand[DevCard::VictoryPoint.index()]
    }
}
