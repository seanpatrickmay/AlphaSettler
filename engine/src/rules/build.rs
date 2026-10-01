//! Roads, settlements, cities, and ending the turn.

use crate::action::Action;
use crate::apply::EventSink;
use crate::events::Event;
use crate::rules::awards;
use crate::state::{Phase, State};
use crate::topology::topo;
use crate::types::*;

/// Nodes player `p`'s roads touch.
#[inline]
fn road_nodes(roads: u128) -> u64 {
    let t = topo();
    bits128(roads).fold(0, |m, e| m | t.edge_node_mask[e as usize])
}

/// Every free edge player `p` may build a road on: edges touching `p`'s buildings, or touching
/// a node `p`'s roads reach that no opponent holds.
#[inline]
fn free_road_edges(s: &State, p: usize) -> u128 {
    let t = topo();
    let pl = &s.players[p];
    let mine = pl.settlements | pl.cities;
    let theirs = s.occupied_nodes() & !mine;
    let reach = mine | (road_nodes(pl.roads) & !theirs);
    let edges = bits64(reach).fold(0u128, |m, n| m | t.node_edge_mask[n as usize]);
    edges & !s.occupied_edges()
}

pub fn push_free_roads(s: &State, p: usize, out: &mut Vec<Action>) {
    out.extend(bits128(free_road_edges(s, p)).map(Action::BuildRoad));
}

pub fn has_free_road(s: &State, p: usize) -> bool {
    free_road_edges(s, p) != 0
}

pub fn legal_main_builds(s: &State, out: &mut Vec<Action>) {
    let t = topo();
    let p = s.current as usize;
    let pl = &s.players[p];
    if covers(&pl.hand, &ROAD_COST) && pl.roads.count_ones() < MAX_ROADS {
        push_free_roads(s, p, out);
    }
    if covers(&pl.hand, &SETTLEMENT_COST) && pl.settlements.count_ones() < MAX_SETTLEMENTS {
        let occ = s.occupied_nodes();
        let too_close = bits64(occ).fold(occ, |m, n| m | t.node_neighbor_mask[n as usize]);
        let spots = road_nodes(pl.roads) & !too_close;
        out.extend(bits64(spots).map(Action::BuildSettlement));
    }
    if covers(&pl.hand, &CITY_COST) && pl.cities.count_ones() < MAX_CITIES {
        out.extend(bits64(pl.settlements).map(Action::BuildCity));
    }
}

fn pay(s: &mut State, p: usize, cost: &Hand) {
    hand_sub(&mut s.players[p].hand, cost);
    hand_add(&mut s.bank, cost);
}

pub fn apply_build_road<S: EventSink>(s: &mut State, e: u8, free: bool, sink: &mut S) {
    let p = s.current as usize;
    if !free {
        pay(s, p, &ROAD_COST);
    }
    s.players[p].roads |= 1u128 << e;
    sink.emit(Event::BuiltRoad {
        player: s.current,
        edge: e,
    });
    awards::extend_road_len(s, p, e);
    awards::assign_longest_road(s);
}

pub fn apply_build_settlement<S: EventSink>(s: &mut State, n: u8, sink: &mut S) {
    let p = s.current as usize;
    pay(s, p, &SETTLEMENT_COST);
    s.players[p].settlements |= 1u64 << n;
    sink.emit(Event::BuiltSettlement {
        player: s.current,
        node: n,
    });
    // Only opponents whose roads run through `n` can have been cut.
    let touching = topo().node_edge_mask[n as usize];
    for q in 0..NUM_PLAYERS {
        if q != p && s.players[q].roads & touching != 0 {
            awards::recompute_road_len(s, q);
        }
    }
    awards::assign_longest_road(s);
}

pub fn apply_build_city<S: EventSink>(s: &mut State, n: u8, sink: &mut S) {
    let p = s.current as usize;
    pay(s, p, &CITY_COST);
    s.players[p].settlements &= !(1u64 << n);
    s.players[p].cities |= 1u64 << n;
    sink.emit(Event::BuiltCity {
        player: s.current,
        node: n,
    });
}

pub fn apply_end_turn<S: EventSink>(s: &mut State, sink: &mut S) {
    let p = s.current as usize;
    s.players[p].dev_new = [0; 5];
    s.dev_played_this_turn = false;
    s.offers_this_turn = 0;
    sink.emit(Event::TurnEnded { player: s.current });
    s.current = ((p + 1) % NUM_PLAYERS) as PlayerId;
    s.turn += 1;
    s.phase = Phase::PreRoll;
    if s.turn >= s.config.max_turns {
        s.phase = Phase::GameOver { winner: None };
        sink.emit(Event::GameOver { winner: None });
    }
}
