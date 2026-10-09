//! Observation (+ hidden block) -> seat-relative feature vector (spec Section 2). This is the only
//! encoder: Python gets features through the bindings, so Rust inference and PyTorch training see
//! the same inputs. Any change to the layout or the scaling must bump `ENCODER_VERSION` and update
//! `LAYOUT_DESCRIPTION`; the golden test in `net/tests/encode.rs` fails until the test is updated
//! with it.

use settler_engine::rules::awards::longest_trail;
use settler_engine::topology::{topo, NUM_EDGES, NUM_NODES, NUM_TILES};
use settler_engine::types::*;
use settler_engine::{Observation, Phase, Response, State, ACTION_SPACE_SIZE};

pub const ENCODER_VERSION: u32 = 4;

pub const NODE_WIDTH: usize = 24;
pub const EDGE_WIDTH: usize = 8;
/// v4 (spec 3d-2): each node's road distance for r0 .. r3 at these 4 slots, then the open spot.
pub const NODE_DISTANCE: usize = 19;
pub const NODE_OPEN: usize = NODE_DISTANCE + NUM_PLAYERS;
/// v4: each edge's "on r's longest trail" for r0 .. r3 at these 4 slots, after the road owners.
pub const EDGE_TRAIL: usize = 4;
pub const TILE_WIDTH: usize = 8;
pub const SEAT_WIDTH: usize = 10;
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
/// Whose turn it is (seat-relative one-hot); differs from the actor in Discard and trade replies.
pub const CURRENT: usize = DEV_PLAYED + 1;
/// The phase resumed after the robber or Road Building: PreRoll, Main.
pub const RETURN_PHASE: usize = CURRENT + NUM_PLAYERS;
/// Free roads left in Road Building (0 outside it).
pub const FREE_ROADS: usize = RETURN_PHASE + 2;
/// The pending domestic offer (all 0 when none): what the offerer gives, then what it asks for,
/// per resource, each over `max_trade_cards`.
pub const TRADE_GIVE: usize = FREE_ROADS + 1;
pub const TRADE_GET: usize = TRADE_GIVE + NUM_RESOURCES;
/// Each seat's answer to the pending offer, seat-relative, one-hot: pending, accepted, rejected
/// (all 0 when no offer is pending; the offerer's own seat stays pending).
pub const TRADE_RESPONSES: usize = TRADE_GET + NUM_RESOURCES;
pub const RESPONSE_WIDTH: usize = 3;
/// Offers the player on turn has made this turn / `max_offers_per_turn` (0 when offers are off).
pub const OFFERS_MADE: usize = TRADE_RESPONSES + NUM_PLAYERS * RESPONSE_WIDTH;
/// 1 when the game allows domestic offers (`max_offers_per_turn > 0`).
pub const OFFERS_ALLOWED: usize = OFFERS_MADE + 1;
/// v4: per seat (seat-relative), longest road length / 15, roads to the longest-road award / 5,
/// knights to the largest-army award / 3, points to win / vp_to_win, settlements left / 5, cities
/// left / 4.
pub const THREATS: usize = OFFERS_ALLOWED + 1;
pub const THREAT_WIDTH: usize = 6;
pub const PUBLIC_LEN: usize = THREATS + NUM_PLAYERS * THREAT_WIDTH;

/// Per opponent (relative seats 1..=3): hand by resource, then dev cards held by kind.
pub const HIDDEN_WIDTH: usize = 10;
pub const HIDDEN_LEN: usize = (NUM_PLAYERS - 1) * HIDDEN_WIDTH;

// A count x is encoded as x / DEN. Each DEN is the engine's hard maximum where that is small
// (so the feature stays in [0, 1]), or the realistic maximum where the hard one is far out.
/// Cards of one resource in a hand: at most the 19 the bank starts with.
pub const CARD_DEN: u32 = 19;
/// A player's total cards. The hard maximum is 95 (every card); a 7 halves any hand over 7, so
/// hands rarely pass 20 and this feature rarely passes 1. Not clamped.
pub const HAND_DEN: u32 = 20;
/// Dev cards in total (held, or bought this turn): at most the whole deck.
pub const DEV_TOTAL_DEN: u32 = DEV_DECK_SIZE as u32;
/// Dev cards of one kind (held or played): at most that kind's count in the deck (14 knights, 5
/// VP, 2 of each other kind), in `DevCard` order.
pub const DEV_KIND_DEN: [u32; 5] = [
    DEV_DECK_COUNTS[0] as u32,
    DEV_DECK_COUNTS[1] as u32,
    DEV_DECK_COUNTS[2] as u32,
    DEV_DECK_COUNTS[3] as u32,
    DEV_DECK_COUNTS[4] as u32,
];
/// Public VP: the game ends at 10.
pub const VP_DEN: u32 = 10;
/// Cards owed in a discard: half a hand, so realistically at most 10 (hard maximum 47).
pub const DISCARD_DEN: u32 = 10;
pub const BANK_DEN: u32 = 19;
/// Dev cards left in the deck.
pub const DECK_DEN: u32 = DEV_DECK_SIZE as u32;
/// One tile's pips.
pub const PIP_DEN: u32 = 5;
/// A node's pips of one resource.
pub const NODE_PIP_DEN: u32 = 15;
/// Free roads left in Road Building (1 or 2 during it).
pub const FREE_ROAD_DEN: u32 = 2;
/// v4: road distances are capped here (3 or more roads, or never, read 1).
pub const DISTANCE_DEN: u32 = 3;
/// v4: a seat's longest road: at most its 15 roads.
pub const ROAD_LENGTH_DEN: u32 = MAX_ROADS;
/// v4: the longest road is won at 5 roads, the largest army at 3 knights; "to the award" is over
/// these.
pub const ROAD_AWARD_MIN: u32 = 5;
pub const ARMY_AWARD_MIN: u32 = 3;
/// v4: pieces in supply.
pub const SETTLEMENT_DEN: u32 = MAX_SETTLEMENTS;
pub const CITY_DEN: u32 = MAX_CITIES;

/// Everything the layout depends on, hashed into `ENCODING_HASH`.
pub const LAYOUT: [u32; 33] = [
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
    DEV_TOTAL_DEN,
    DEV_KIND_DEN[0],
    DEV_KIND_DEN[1],
    DEV_KIND_DEN[2],
    DEV_KIND_DEN[3],
    DEV_KIND_DEN[4],
    VP_DEN,
    DISCARD_DEN,
    BANK_DEN,
    DECK_DEN,
    PIP_DEN,
    NODE_PIP_DEN,
    FREE_ROAD_DEN,
    DISTANCE_DEN,
    ROAD_LENGTH_DEN,
    ROAD_AWARD_MIN,
    ARMY_AWARD_MIN,
    SETTLEMENT_DEN,
    CITY_DEN,
];

/// The whole layout as canonical text: every block in vector order with its item count and one
/// token per slot in slot order (`name/scale` when the slot is a count divided by that scale),
/// plus the scales and `ENCODER_VERSION`. Hashed into `ENCODING_HASH` with `LAYOUT`, so reordering
/// blocks or slots changes the hash even when no numeric constant moves. Keep it in step with
/// `encode_public` and `encode_hidden_from`: `net/tests/encode.rs` checks its offsets, widths,
/// scales and version against the constants.
pub const LAYOUT_DESCRIPTION: &str = "\
settler-net features v4
seats relative to the viewer: r0 is the viewer, then r1 r2 r3 in turn order
resources: wood brick sheep wheat ore
dev kinds: knight victory_point road_building year_of_plenty monopoly
scales: card=19 hand=20 dev_total=25 dev_knight=14 dev_victory_point=5 dev_road_building=2 \
dev_year_of_plenty=2 dev_monopoly=2 vp=10 discard=10 bank=19 deck=25 pip=5 node_pip=15 free_roads=2 \
distance=3 road_length=15 road_award=5 army_award=3 settlements=5 cities=4
public node x54: settlement_r0 settlement_r1 settlement_r2 settlement_r3 city_r0 city_r1 city_r2 \
city_r3 port_generic port_wood port_brick port_sheep port_wheat port_ore pips_wood/node_pip \
pips_brick/node_pip pips_sheep/node_pip pips_wheat/node_pip pips_ore/node_pip \
road_distance_r0/distance road_distance_r1/distance road_distance_r2/distance \
road_distance_r3/distance open_spot
public edge x72: road_r0 road_r1 road_r2 road_r3 longest_trail_r0 longest_trail_r1 longest_trail_r2 \
longest_trail_r3
public tile x19: wood brick sheep wheat ore desert pips/pip robber
public own_hand x1: wood/card brick/card sheep/card wheat/card ore/card
public own_dev_old x1: knight/dev_knight victory_point/dev_victory_point \
road_building/dev_road_building year_of_plenty/dev_year_of_plenty monopoly/dev_monopoly
public own_dev_new x1: knight/dev_knight victory_point/dev_victory_point \
road_building/dev_road_building year_of_plenty/dev_year_of_plenty monopoly/dev_monopoly
public seat x4: hand_count/hand dev_count/dev_total new_dev_count/dev_total \
played_knight/dev_knight played_victory_point/dev_victory_point \
played_road_building/dev_road_building played_year_of_plenty/dev_year_of_plenty \
played_monopoly/dev_monopoly public_vp/vp discard_owed/discard
public longest_road x1: r0 r1 r2 r3 none
public largest_army x1: r0 r1 r2 r3 none
public bank x1: wood/bank brick/bank sheep/bank wheat/bank ore/bank
public deck x1: remaining/deck
public phase x1: setup_settlement setup_road pre_roll discard move_robber steal main \
road_building trade_response trade_confirm game_over
public turn x1: turn/max_turns
public dev_played x1: dev_played
public current x1: r0 r1 r2 r3
public return_phase x1: pre_roll main
public free_roads x1: roads_left/free_roads
public trade_give x1: wood/max_trade_cards brick/max_trade_cards sheep/max_trade_cards \
wheat/max_trade_cards ore/max_trade_cards
public trade_get x1: wood/max_trade_cards brick/max_trade_cards sheep/max_trade_cards \
wheat/max_trade_cards ore/max_trade_cards
public trade_response x4: pending accepted rejected
public offers_made x1: offers/max_offers_per_turn
public offers_allowed x1: allowed
public threat x4: road_length/road_length roads_to_award/road_award knights_to_award/army_award \
points_to_win/vp_to_win settlements_left/settlements cities_left/cities
hidden opponent x3: wood/card brick/card sheep/card wheat/card ore/card \
knight/dev_knight victory_point/dev_victory_point road_building/dev_road_building \
year_of_plenty/dev_year_of_plenty monopoly/dev_monopoly
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

/// `return_phase` is always PreRoll or Main (an engine invariant).
fn return_phase_index(p: Phase) -> usize {
    match p {
        Phase::PreRoll => 0,
        Phase::Main => 1,
        other => panic!("return_phase must be PreRoll or Main, got {other:?}"),
    }
}

#[inline]
fn scaled(x: u32, den: u32) -> f32 {
    x as f32 / den as f32
}

/// Every node seat `p`'s roads or buildings touch: its network (v4).
fn network(obs: &Observation, p: usize) -> u64 {
    let t = topo();
    bits128(obs.roads[p]).fold(obs.settlements[p] | obs.cities[p], |m, e| {
        m | t.edge_node_mask[e as usize]
    })
}

/// v4's road distances: the fewest roads that make `network` touch each node, capped at
/// `DISTANCE_DEN`. A network node is 0, a node another seat built on included; a path may not
/// leave a `blocked` node (another seat's building) or use a `closed` edge (another seat's road).
fn road_distances(network: u64, blocked: u64, closed: u128) -> [u8; NUM_NODES] {
    let t = topo();
    let cap = DISTANCE_DEN as u8;
    let mut dist = [cap; NUM_NODES];
    for n in bits64(network) {
        dist[n as usize] = 0;
    }
    let (mut seen, mut frontier) = (network, network);
    // Rings 1 .. cap - 1; every node further out keeps the cap.
    for d in 1..cap {
        let mut next = 0u64;
        for n in bits64(frontier & !blocked) {
            for e in bits128(t.node_edge_mask[n as usize] & !closed) {
                next |= t.edge_node_mask[e as usize];
            }
        }
        next &= !seen;
        for n in bits64(next) {
            dist[n as usize] = d;
        }
        seen |= next;
        frontier = next;
    }
    dist
}

/// v4: pieces seat `p` still lacks to take an award won at `min` (5 roads, 3 knights), holding
/// `own`: 0 for the holder, else `min(min, max(min, held + 1) - own)` with `held` the holder's
/// count, and `max(0, min - own)` with no holder (spec 3d-2 Section 2).
fn to_award(own: u32, p: usize, holder: Option<(usize, u32)>, min: u32) -> u32 {
    match holder {
        Some((h, _)) if h == p => 0,
        Some((_, held)) => min.max(held + 1).saturating_sub(own).min(min),
        None => min.saturating_sub(own),
    }
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
        out[OWN_DEV_OLD + k] = scaled((obs.my_dev_cards[k] - new) as u32, DEV_KIND_DEN[k]);
        out[OWN_DEV_NEW + k] = scaled(new as u32, DEV_KIND_DEN[k]);
    }
    for p in 0..NUM_PLAYERS {
        let at = SEATS + rel(p as PlayerId, v) * SEAT_WIDTH;
        out[at] = scaled(obs.hand_counts[p] as u32, HAND_DEN);
        out[at + 1] = scaled(obs.dev_card_counts[p] as u32, DEV_TOTAL_DEN);
        out[at + 2] = scaled(obs.new_dev_card_counts[p] as u32, DEV_TOTAL_DEN);
        // No separate knights slot: `knights_played` equals `dev_cards_played[p][0]`.
        for k in 0..5 {
            out[at + 3 + k] = scaled(obs.dev_cards_played[p][k] as u32, DEV_KIND_DEN[k]);
        }
        out[at + 8] = scaled(obs.public_vp[p] as u32, VP_DEN);
        out[at + 9] = scaled(obs.discard_remaining[p] as u32, DISCARD_DEN);
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
    out[CURRENT + rel(obs.current, v)] = 1.0;
    out[RETURN_PHASE + return_phase_index(obs.return_phase)] = 1.0;
    if let Phase::RoadBuilding { roads_left } = obs.phase {
        out[FREE_ROADS] = scaled(roads_left as u32, FREE_ROAD_DEN);
    }
    if let Some(t) = obs.trade {
        let cap = obs.config.max_trade_cards as u32;
        for res in 0..NUM_RESOURCES {
            out[TRADE_GIVE + res] = scaled(t.give[res] as u32, cap);
            out[TRADE_GET + res] = scaled(t.get[res] as u32, cap);
        }
        for p in 0..NUM_PLAYERS {
            let answer = match t.responses[p] {
                Response::Pending => 0,
                Response::Accepted => 1,
                Response::Rejected => 2,
            };
            out[TRADE_RESPONSES + rel(p as PlayerId, v) * RESPONSE_WIDTH + answer] = 1.0;
        }
    }
    let max_offers = obs.config.max_offers_per_turn as u32;
    if max_offers > 0 {
        out[OFFERS_MADE] = scaled(obs.offers_this_turn as u32, max_offers);
        out[OFFERS_ALLOWED] = 1.0;
    }
    // v4 (spec 3d-2 Section 2): road distances, open spots, longest trails, the threat block.
    let buildings: [u64; NUM_PLAYERS] = std::array::from_fn(|p| obs.settlements[p] | obs.cities[p]);
    let occupied = buildings.iter().fold(0u64, |m, b| m | b);
    let roads = obs.roads.iter().fold(0u128, |m, r| m | r);
    // The trail `awards::longest_road` measures: another seat's building ends it.
    let trails: [(u8, u128); NUM_PLAYERS] =
        std::array::from_fn(|p| longest_trail(obs.roads[p], occupied & !buildings[p]));
    for p in 0..NUM_PLAYERS {
        let r = rel(p as PlayerId, v);
        let dist = road_distances(
            network(obs, p),
            occupied & !buildings[p],
            roads & !obs.roads[p],
        );
        for (n, &d) in dist.iter().enumerate() {
            out[NODE_BASE + n * NODE_WIDTH + NODE_DISTANCE + r] = scaled(d as u32, DISTANCE_DEN);
        }
        for e in bits128(trails[p].1) {
            out[EDGE_BASE + e as usize * EDGE_WIDTH + EDGE_TRAIL + r] = 1.0;
        }
    }
    for n in 0..NUM_NODES {
        if occupied & ((1u64 << n) | t.node_neighbor_mask[n]) == 0 {
            out[NODE_BASE + n * NODE_WIDTH + NODE_OPEN] = 1.0;
        }
    }
    let road_holder = obs
        .longest_road_owner
        .map(|h| (h as usize, trails[h as usize].0 as u32));
    let army_holder = obs
        .largest_army_owner
        .map(|h| (h as usize, obs.knights_played[h as usize] as u32));
    let vp_to_win = obs.config.vp_to_win as u32;
    for p in 0..NUM_PLAYERS {
        let at = THREATS + rel(p as PlayerId, v) * THREAT_WIDTH;
        let length = trails[p].0 as u32;
        // The viewer knows its VP cards; an opponent's are not in the observation.
        let mut points = obs.public_vp[p] as u32;
        if p == v as usize {
            points += obs.my_dev_cards[DevCard::VictoryPoint.index()] as u32;
        }
        let knights = obs.knights_played[p] as u32;
        out[at] = scaled(length, ROAD_LENGTH_DEN);
        out[at + 1] = scaled(
            to_award(length, p, road_holder, ROAD_AWARD_MIN),
            ROAD_AWARD_MIN,
        );
        out[at + 2] = scaled(
            to_award(knights, p, army_holder, ARMY_AWARD_MIN),
            ARMY_AWARD_MIN,
        );
        out[at + 3] = scaled(vp_to_win.saturating_sub(points), vp_to_win);
        out[at + 4] = scaled(
            MAX_SETTLEMENTS.saturating_sub(obs.settlements[p].count_ones()),
            SETTLEMENT_DEN,
        );
        out[at + 5] = scaled(
            MAX_CITIES.saturating_sub(obs.cities[p].count_ones()),
            CITY_DEN,
        );
    }
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
            out[at + 5 + k] = scaled(dev_cards[p][k] as u32, DEV_KIND_DEN[k]);
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
