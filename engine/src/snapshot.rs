//! Plain-data snapshots of a full position: exported for comparison with another engine and
//! imported to continue from another engine's position (differential testing, playing inside
//! Catanatron).

use crate::board::Board;
use crate::config::GameConfig;
use crate::rules::awards::longest_road;
use crate::state::{Phase, State};
use crate::types::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct PlayerSnapshot {
    pub hand: Hand,
    pub dev_hand: [u8; 5],
    pub dev_new: [u8; 5],
    pub dev_played: [u8; 5],
    pub knights_played: u8,
    pub settlements: u64,
    pub cities: u64,
    pub roads: u128,
    pub discard_remaining: u8,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Snapshot {
    pub board: Board,
    pub robber: u8,
    pub bank: Hand,
    /// Cards left in the deck, next draw first.
    pub dev_deck: Vec<DevCard>,
    pub players: [PlayerSnapshot; NUM_PLAYERS],
    pub current: PlayerId,
    pub phase: Phase,
    pub return_phase: Phase,
    pub turn: u32,
    pub setup_step: u8,
    pub dev_played_this_turn: bool,
    /// Domestic trade offers the current player has made this turn.
    pub offers_this_turn: u8,
    pub longest_road_owner: Option<PlayerId>,
    pub largest_army_owner: Option<PlayerId>,
}

impl State {
    pub fn snapshot(&self) -> Snapshot {
        Snapshot {
            board: self.board,
            robber: self.robber,
            bank: self.bank,
            dev_deck: self.dev_deck[self.dev_deck_pos as usize..].to_vec(),
            players: std::array::from_fn(|p| {
                let pl = &self.players[p];
                PlayerSnapshot {
                    hand: pl.hand,
                    dev_hand: pl.dev_hand,
                    dev_new: pl.dev_new,
                    dev_played: pl.dev_played,
                    knights_played: pl.knights_played,
                    settlements: pl.settlements,
                    cities: pl.cities,
                    roads: pl.roads,
                    discard_remaining: pl.discard_remaining,
                }
            }),
            current: self.current,
            phase: self.phase,
            return_phase: self.return_phase,
            turn: self.turn,
            setup_step: self.setup_step,
            dev_played_this_turn: self.dev_played_this_turn,
            offers_this_turn: self.offers_this_turn,
            longest_road_owner: self.longest_road_owner,
            largest_army_owner: self.largest_army_owner,
        }
    }

    /// A state at exactly this position. Randomness the snapshot does not fix (dice, steals)
    /// comes from `seed`. Pending domestic trades are not representable.
    pub fn from_snapshot(seed: u64, config: GameConfig, snap: &Snapshot) -> Result<State, String> {
        config.validate()?;
        snap.board.validate()?;
        if matches!(snap.phase, Phase::TradeResponse | Phase::TradeConfirm) {
            return Err("snapshots with a pending domestic trade are not supported".into());
        }
        if snap.offers_this_turn > config.max_offers_per_turn {
            return Err(format!(
                "offers_this_turn {} exceeds max_offers_per_turn {}",
                snap.offers_this_turn, config.max_offers_per_turn
            ));
        }
        if snap.dev_deck.len() > DEV_DECK_SIZE {
            return Err(format!(
                "dev_deck has {} cards, the deck has only {DEV_DECK_SIZE}",
                snap.dev_deck.len()
            ));
        }
        // Cards already drawn first (in kind order), then the remaining cards in draw order.
        let mut drawn = DEV_DECK_COUNTS.map(|c| c as i32);
        for c in &snap.dev_deck {
            drawn[c.index()] -= 1;
        }
        if let Some(k) = (0..5).find(|&k| drawn[k] < 0) {
            return Err(format!(
                "dev_deck holds more {:?} cards than the deck has",
                DevCard::from_index(k)
            ));
        }
        let pos = DEV_DECK_SIZE - snap.dev_deck.len();
        let mut deck = [DevCard::Knight; DEV_DECK_SIZE];
        let mut i = 0;
        for (k, &n) in drawn.iter().enumerate() {
            for _ in 0..n {
                deck[i] = DevCard::from_index(k);
                i += 1;
            }
        }
        deck[pos..].copy_from_slice(&snap.dev_deck);

        let mut s = State::with_board(seed, config, snap.board);
        s.dev_deck = deck;
        s.dev_deck_pos = pos as u8;
        for (p, ps) in snap.players.iter().enumerate() {
            if ps.roads.count_ones() > MAX_ROADS {
                return Err(format!(
                    "player {p} has {} roads, the limit is {MAX_ROADS}",
                    ps.roads.count_ones()
                ));
            }
            let pl = &mut s.players[p];
            pl.hand = ps.hand;
            pl.dev_hand = ps.dev_hand;
            pl.dev_new = ps.dev_new;
            pl.dev_played = ps.dev_played;
            pl.knights_played = ps.knights_played;
            pl.settlements = ps.settlements;
            pl.cities = ps.cities;
            pl.roads = ps.roads;
            pl.discard_remaining = ps.discard_remaining;
        }
        s.bank = snap.bank;
        s.robber = snap.robber;
        s.current = snap.current;
        s.phase = snap.phase;
        s.return_phase = snap.return_phase;
        s.turn = snap.turn;
        s.setup_step = snap.setup_step;
        s.dev_played_this_turn = snap.dev_played_this_turn;
        s.offers_this_turn = snap.offers_this_turn;
        s.trade = None;
        s.longest_road_owner = snap.longest_road_owner;
        s.largest_army_owner = snap.largest_army_owner;
        let occupied = s.occupied_nodes();
        for p in 0..NUM_PLAYERS {
            if s.players[p].roads >> crate::topology::NUM_EDGES != 0 {
                return Err(format!(
                    "player {p} roads use edge ids beyond {}",
                    crate::topology::NUM_EDGES - 1
                ));
            }
            s.players[p].longest_road_len =
                longest_road(s.players[p].roads, occupied & !s.buildings(p));
        }
        s.check_invariants()?;
        Ok(s)
    }
}
