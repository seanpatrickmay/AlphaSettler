//! Observation (+ hidden block) -> seat-relative feature vector (spec Section 2). This is the only
//! encoder: Python gets features through the bindings, so Rust inference and PyTorch training see
//! the same inputs. Any change to the layout or the scaling must bump `ENCODER_VERSION` and update
//! `LAYOUT_DESCRIPTION`; the golden test in `net/tests/encode.rs` fails until the test is updated
//! with it.

use settler_engine::topology::{topo, NUM_EDGES, NUM_NODES, NUM_TILES};
use settler_engine::types::*;
use settler_engine::{Observation, Phase, State, ACTION_SPACE_SIZE};

pub const ENCODER_VERSION: u32 = 1;

pub const NODE_WIDTH: usize = 19;
pub const EDGE_WIDTH: usize = 4;
pub const TILE_WIDTH: usize = 8;
pub const SEAT_WIDTH: usize = 11;
pub const NUM_PHASES: usize = 11;

pub const NODE_BASE: usize = 0;
pub const EDGE_BASE: usize = NODE_BASE + NUM_NODES * NODE_WIDTH;
pub const TILE_BASE: usize = EDGE_BASE + NUM_EDGES * EDGE_WIDTH;
pub const OWN_HAND: usize = TILE_BASE + NUM_TILES * TILE_WIDTH;
pub const OWN_DEV_OLD: usize = OWN_HAND + NUM_RESOURCES;
pub const OWN_DEV_NEW: usize = OWN_DEV_OLD + 5;
pub const SEATS: usize = OWN_DEV_NEW + 5;
pub const LONGEST_ROAD: usize = SEATS + NUM_PLAYERS * SEAT_WIDTH;
pub const LARGEST_ARMY: usize = LONGEST_ROAD + NUM_PLAYERS + 1;
pub const BANK: usize = LARGEST_ARMY + NUM_PLAYERS + 1;
pub const DECK: usize = BANK + NUM_RESOURCES;
pub const PHASE: usize = DECK + 1;
pub const TURN: usize = PHASE + NUM_PHASES;
pub const DEV_PLAYED: usize = TURN + 1;
pub const PUBLIC_LEN: usize = DEV_PLAYED + 1;

/// Per opponent (relative seats 1..=3): hand by resource, then dev cards held by kind.
pub const HIDDEN_WIDTH: usize = 10;
pub const HIDDEN_LEN: usize = (NUM_PLAYERS - 1) * HIDDEN_WIDTH;

// A count x is encoded as x / DEN, roughly in [0, 1].
/// Cards of one resource.
pub const CARD_DEN: u32 = 10;
/// A player's total cards.
pub const HAND_DEN: u32 = 20;
/// Dev cards, per kind or in total.
pub const DEV_DEN: u32 = 5;
pub const KNIGHT_DEN: u32 = 14;
pub const VP_DEN: u32 = 10;
pub const DISCARD_DEN: u32 = 10;
pub const BANK_DEN: u32 = 19;
pub const DECK_DEN: u32 = 25;
/// One tile's pips.
pub const PIP_DEN: u32 = 5;
/// A node's pips of one resource.
pub const NODE_PIP_DEN: u32 = 15;

/// Everything the layout depends on, hashed into `ENCODING_HASH`.
pub const LAYOUT: [u32; 22] = [
    ENCODER_VERSION,
    NUM_NODES as u32,
    NODE_WIDTH as u32,
    NUM_EDGES as u32,
    EDGE_WIDTH as u32,
    NUM_TILES as u32,
    TILE_WIDTH as u32,
    SEAT_WIDTH as u32,
    NUM_PHASES as u32,
    PUBLIC_LEN as u32,
    HIDDEN_LEN as u32,
    ACTION_SPACE_SIZE as u32,
    CARD_DEN,
    HAND_DEN,
    DEV_DEN,
    KNIGHT_DEN,
    VP_DEN,
    DISCARD_DEN,
    BANK_DEN,
    DECK_DEN,
    PIP_DEN,
    NODE_PIP_DEN,
];

/// The whole layout as canonical text: every block in vector order with its item count and one
/// token per slot in slot order (`name/scale` when the slot is a count divided by that scale),
/// plus the scales and `ENCODER_VERSION`. Hashed into `ENCODING_HASH` with `LAYOUT`, so reordering
/// blocks or slots changes the hash even when no numeric constant moves. Keep it in step with
/// `encode_public` and `encode_hidden_from`: `net/tests/encode.rs` checks its offsets, widths,
/// scales and version against the constants.
pub const LAYOUT_DESCRIPTION: &str = "\
settler-net features v1
seats relative to the viewer: r0 is the viewer, then r1 r2 r3 in turn order
resources: wood brick sheep wheat ore
dev kinds: knight victory_point road_building year_of_plenty monopoly
scales: card=10 hand=20 dev=5 knight=14 vp=10 discard=10 bank=19 deck=25 pip=5 node_pip=15
public node x54: settlement_r0 settlement_r1 settlement_r2 settlement_r3 city_r0 city_r1 city_r2 \
city_r3 port_generic port_wood port_brick port_sheep port_wheat port_ore pips_wood/node_pip \
pips_brick/node_pip pips_sheep/node_pip pips_wheat/node_pip pips_ore/node_pip
public edge x72: road_r0 road_r1 road_r2 road_r3
public tile x19: wood brick sheep wheat ore desert pips/pip robber
public own_hand x1: wood/card brick/card sheep/card wheat/card ore/card
public own_dev_old x1: knight/dev victory_point/dev road_building/dev year_of_plenty/dev \
monopoly/dev
public own_dev_new x1: knight/dev victory_point/dev road_building/dev year_of_plenty/dev \
monopoly/dev
public seat x4: hand_count/hand dev_count/dev new_dev_count/dev played_knight/dev \
played_victory_point/dev played_road_building/dev played_year_of_plenty/dev played_monopoly/dev \
knights/knight public_vp/vp discard_owed/discard
public longest_road x1: r0 r1 r2 r3 none
public largest_army x1: r0 r1 r2 r3 none
public bank x1: wood/bank brick/bank sheep/bank wheat/bank ore/bank
public deck x1: remaining/deck
public phase x1: setup_settlement setup_road pre_roll discard move_robber steal main \
road_building trade_response trade_confirm game_over
public turn x1: turn/max_turns
public dev_played x1: dev_played
hidden opponent x3: wood/card brick/card sheep/card wheat/card ore/card knight/dev \
victory_point/dev road_building/dev year_of_plenty/dev monopoly/dev
";

/// FNV-1a over the little-endian bytes of `layout`, then the bytes of `description`.
pub const fn layout_hash(layout: &[u32], description: &str) -> u64 {
    const PRIME: u64 = 0x0100_0000_01b3;
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let mut i = 0;
    while i < layout.len() {
        let bytes = layout[i].to_le_bytes();
        let mut j = 0;
        while j < 4 {
            h ^= bytes[j] as u64;
            h = h.wrapping_mul(PRIME);
            j += 1;
        }
        i += 1;
    }
    let text = description.as_bytes();
    let mut k = 0;
    while k < text.len() {
        h ^= text[k] as u64;
        h = h.wrapping_mul(PRIME);
        k += 1;
    }
    h
}

/// Written into every weights file; a file with another value was trained on other features.
pub const ENCODING_HASH: u64 = layout_hash(&LAYOUT, LAYOUT_DESCRIPTION);

/// `p`'s seat relative to `viewer`: 0 for the viewer, then in turn order.
#[inline]
pub fn rel(p: PlayerId, viewer: PlayerId) -> usize {
    (p as usize + NUM_PLAYERS - viewer as usize) % NUM_PLAYERS
}

/// The absolute seat of relative seat `r` for `viewer`.
#[inline]
pub fn abs(r: usize, viewer: PlayerId) -> usize {
    (viewer as usize + r) % NUM_PLAYERS
}

/// Dots on a number token: 2 and 12 have 1, 6 and 8 have 5, the desert 0.
pub fn pips(number: u8) -> u32 {
    match number {
        2..=6 => number as u32 - 1,
        8..=12 => 13 - number as u32,
        _ => 0,
    }
}

pub fn phase_index(p: Phase) -> usize {
    match p {
        Phase::SetupSettlement => 0,
        Phase::SetupRoad { .. } => 1,
        Phase::PreRoll => 2,
        Phase::Discard => 3,
        Phase::MoveRobber => 4,
        Phase::Steal => 5,
        Phase::Main => 6,
        Phase::RoadBuilding { .. } => 7,
        Phase::TradeResponse => 8,
        Phase::TradeConfirm => 9,
        Phase::GameOver { .. } => 10,
    }
}

#[inline]
fn scaled(x: u32, den: u32) -> f32 {
    x as f32 / den as f32
}

/// The public features of `obs`, seat-relative to `obs.viewer`, written over all of `out`.
pub fn encode_public(obs: &Observation, out: &mut [f32]) {
    assert_eq!(
        out.len(),
        PUBLIC_LEN,
        "the public block is PUBLIC_LEN floats"
    );
    out.fill(0.0);
    let t = topo();
    let v = obs.viewer;
    let b = &obs.board;
    for n in 0..NUM_NODES {
        let at = NODE_BASE + n * NODE_WIDTH;
        let bit = 1u64 << n;
        for p in 0..NUM_PLAYERS {
            let r = rel(p as PlayerId, v);
            if obs.settlements[p] & bit != 0 {
                out[at + r] = 1.0;
            }
            if obs.cities[p] & bit != 0 {
                out[at + 4 + r] = 1.0;
            }
        }
        if b.generic_port_nodes & bit != 0 {
            out[at + 8] = 1.0;
        }
        for res in 0..NUM_RESOURCES {
            if b.specific_port_nodes[res] & bit != 0 {
                out[at + 9 + res] = 1.0;
            }
        }
        let mut node_pips = [0u32; NUM_RESOURCES];
        for &tile in &t.node_tiles[n] {
            if let Some(res) = b.tile_resource[tile as usize] {
                node_pips[res.index()] += pips(b.tile_number[tile as usize]);
            }
        }
        for res in 0..NUM_RESOURCES {
            out[at + 14 + res] = scaled(node_pips[res], NODE_PIP_DEN);
        }
    }
    for e in 0..NUM_EDGES {
        for p in 0..NUM_PLAYERS {
            if obs.roads[p] & (1u128 << e) != 0 {
                out[EDGE_BASE + e * EDGE_WIDTH + rel(p as PlayerId, v)] = 1.0;
            }
        }
    }
    for tile in 0..NUM_TILES {
        let at = TILE_BASE + tile * TILE_WIDTH;
        match b.tile_resource[tile] {
            Some(res) => out[at + res.index()] = 1.0,
            None => out[at + 5] = 1.0,
        }
        out[at + 6] = scaled(pips(b.tile_number[tile]), PIP_DEN);
        if obs.robber as usize == tile {
            out[at + 7] = 1.0;
        }
    }
    for res in 0..NUM_RESOURCES {
        out[OWN_HAND + res] = scaled(obs.my_hand[res] as u32, CARD_DEN);
    }
    for k in 0..5 {
        let new = obs.my_new_dev_cards[k];
        out[OWN_DEV_OLD + k] = scaled((obs.my_dev_cards[k] - new) as u32, DEV_DEN);
        out[OWN_DEV_NEW + k] = scaled(new as u32, DEV_DEN);
    }
    for p in 0..NUM_PLAYERS {
        let at = SEATS + rel(p as PlayerId, v) * SEAT_WIDTH;
        out[at] = scaled(obs.hand_counts[p] as u32, HAND_DEN);
        out[at + 1] = scaled(obs.dev_card_counts[p] as u32, DEV_DEN);
        out[at + 2] = scaled(obs.new_dev_card_counts[p] as u32, DEV_DEN);
        for k in 0..5 {
            out[at + 3 + k] = scaled(obs.dev_cards_played[p][k] as u32, DEV_DEN);
        }
        out[at + 8] = scaled(obs.knights_played[p] as u32, KNIGHT_DEN);
        out[at + 9] = scaled(obs.public_vp[p] as u32, VP_DEN);
        out[at + 10] = scaled(obs.discard_remaining[p] as u32, DISCARD_DEN);
    }
    let owner = |o: Option<PlayerId>| o.map_or(NUM_PLAYERS, |p| rel(p, v));
    out[LONGEST_ROAD + owner(obs.longest_road_owner)] = 1.0;
    out[LARGEST_ARMY + owner(obs.largest_army_owner)] = 1.0;
    for res in 0..NUM_RESOURCES {
        out[BANK + res] = scaled(obs.bank[res] as u32, BANK_DEN);
    }
    out[DECK] = scaled(obs.dev_deck_remaining as u32, DECK_DEN);
    out[PHASE + phase_index(obs.phase)] = 1.0;
    out[TURN] = scaled(obs.turn, obs.config.max_turns.max(1));
    out[DEV_PLAYED] = obs.dev_played_this_turn as u8 as f32;
}

/// The hidden block for `actor`: each opponent's hand and dev cards (VP cards and cards bought
/// this turn included), seat-relative. Only the value head reads it.
pub fn encode_hidden_from(
    hands: &[Hand; NUM_PLAYERS],
    dev_cards: &[[u8; 5]; NUM_PLAYERS],
    actor: PlayerId,
    out: &mut [f32],
) {
    assert_eq!(
        out.len(),
        HIDDEN_LEN,
        "the hidden block is HIDDEN_LEN floats"
    );
    for r in 1..NUM_PLAYERS {
        let p = abs(r, actor);
        let at = (r - 1) * HIDDEN_WIDTH;
        for res in 0..NUM_RESOURCES {
            out[at + res] = scaled(hands[p][res] as u32, CARD_DEN);
        }
        for k in 0..5 {
            out[at + 5 + k] = scaled(dev_cards[p][k] as u32, DEV_DEN);
        }
    }
}

/// The hidden block read from a (sampled or real) world.
pub fn encode_hidden(world: &State, actor: PlayerId, out: &mut [f32]) {
    let hands = std::array::from_fn(|p| world.players[p].hand);
    let devs = std::array::from_fn(|p| world.players[p].dev_hand);
    encode_hidden_from(&hands, &devs, actor, out);
}

/// Both blocks for `actor` in `s`: public features from `s.observation(actor)`, hidden from `s`.
pub fn encode_state(s: &State, actor: PlayerId, public: &mut [f32], hidden: &mut [f32]) {
    encode_public(&s.observation(actor), public);
    encode_hidden(s, actor, hidden);
}
