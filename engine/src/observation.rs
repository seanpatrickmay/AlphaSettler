//! What one player can see. Opponents' hands and dev cards appear only as counts.

use crate::board::Board;
use crate::config::GameConfig;
use crate::state::{PendingTrade, Phase, State};
use crate::types::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Observation {
    pub viewer: PlayerId,
    pub config: GameConfig,
    pub phase: Phase,
    pub current: PlayerId,
    pub actor: PlayerId,
    pub turn: u32,
    pub board: Board,
    pub robber: u8,
    pub bank: Hand,
    pub dev_deck_remaining: u8,
    pub my_hand: Hand,
    pub my_dev_cards: [u8; 5],
    pub my_new_dev_cards: [u8; 5],
    pub hand_counts: [u8; NUM_PLAYERS],
    pub dev_card_counts: [u8; NUM_PLAYERS],
    /// Dev cards each player bought this turn (count only; the kinds stay hidden).
    pub new_dev_card_counts: [u8; NUM_PLAYERS],
    /// Dev cards each player has played, by kind (knights also appear in `knights_played`).
    pub dev_cards_played: [[u8; 5]; NUM_PLAYERS],
    pub settlements: [u64; NUM_PLAYERS],
    pub cities: [u64; NUM_PLAYERS],
    pub roads: [u128; NUM_PLAYERS],
    pub knights_played: [u8; NUM_PLAYERS],
    pub public_vp: [u8; NUM_PLAYERS],
    pub longest_road_owner: Option<PlayerId>,
    pub largest_army_owner: Option<PlayerId>,
    pub trade: Option<PendingTrade>,
    pub dev_played_this_turn: bool,
    pub offers_this_turn: u8,
    /// Cards each player still owes in the Discard phase.
    pub discard_remaining: [u8; NUM_PLAYERS],
    /// Phase resumed once the robber is resolved (PreRoll after a pre-roll knight).
    pub robber_return: Phase,
}

impl State {
    pub fn observation(&self, viewer: PlayerId) -> Observation {
        let me = &self.players[viewer as usize];
        let per = |f: &dyn Fn(usize) -> u8| -> [u8; NUM_PLAYERS] { std::array::from_fn(f) };
        Observation {
            viewer,
            config: self.config,
            phase: self.phase,
            current: self.current,
            actor: if self.is_over() {
                self.current
            } else {
                self.current_actor()
            },
            turn: self.turn,
            board: self.board,
            robber: self.robber,
            bank: self.bank,
            dev_deck_remaining: (DEV_DECK_SIZE - self.dev_deck_pos as usize) as u8,
            my_hand: me.hand,
            my_dev_cards: me.dev_hand,
            my_new_dev_cards: me.dev_new,
            hand_counts: per(&|p| hand_total(&self.players[p].hand) as u8),
            dev_card_counts: per(&|p| self.players[p].dev_hand.iter().sum()),
            new_dev_card_counts: per(&|p| self.players[p].dev_new.iter().sum()),
            dev_cards_played: std::array::from_fn(|p| self.players[p].dev_played),
            settlements: std::array::from_fn(|p| self.players[p].settlements),
            cities: std::array::from_fn(|p| self.players[p].cities),
            roads: std::array::from_fn(|p| self.players[p].roads),
            knights_played: per(&|p| self.players[p].knights_played),
            public_vp: per(&|p| self.public_vp(p)),
            longest_road_owner: self.longest_road_owner,
            largest_army_owner: self.largest_army_owner,
            trade: self.trade,
            dev_played_this_turn: self.dev_played_this_turn,
            offers_this_turn: self.offers_this_turn,
            discard_remaining: per(&|p| self.players[p].discard_remaining),
            robber_return: self.robber_return,
        }
    }
}
