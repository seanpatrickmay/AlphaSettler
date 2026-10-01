//! Roads, settlements, cities, and ending the turn.

use crate::action::Action;
use crate::apply::EventSink;
use crate::events::Event;
use crate::rules::awards;
use crate::state::{Phase, State};
use crate::topology::{topo, NUM_EDGES, NUM_NODES};
use crate::types::*;

/// Free edge `e` touches player `p`'s building, or `p`'s road through a node no opponent holds.
pub fn road_connected(s: &State, p: usize, e: u8) -> bool {
    let t = topo();
    let pl = &s.players[p];
    let mine = pl.settlements | pl.cities;
    let theirs = s.occupied_nodes() & !mine;
    let (a, b) = t.edge_nodes[e as usize];
    for n in [a, b] {
        let bit = 1u64 << n;
        if mine & bit != 0 {
            return true;
        }
        if theirs & bit != 0 {
            continue;
        }
        if pl.roads & t.node_edge_mask[n as usize] & !(1u128 << e) != 0 {
            return true;
        }
    }
    false
}

pub fn push_free_roads(s: &State, p: usize, out: &mut Vec<Action>) {
    let occ = s.occupied_edges();
    for e in 0..NUM_EDGES as u8 {
        if occ & (1u128 << e) == 0 && road_connected(s, p, e) {
            out.push(Action::BuildRoad(e));
        }
    }
}

pub fn has_free_road(s: &State, p: usize) -> bool {
    let occ = s.occupied_edges();
    (0..NUM_EDGES as u8).any(|e| occ & (1u128 << e) == 0 && road_connected(s, p, e))
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
        for n in 0..NUM_NODES {
            if occ & (1u64 << n) == 0
                && occ & t.node_neighbor_mask[n] == 0
                && pl.roads & t.node_edge_mask[n] != 0
            {
                out.push(Action::BuildSettlement(n as u8));
            }
        }
    }
    if covers(&pl.hand, &CITY_COST) && pl.cities.count_ones() < MAX_CITIES {
        for n in bits64(pl.settlements) {
            out.push(Action::BuildCity(n));
        }
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
    sink.emit(Event::BuiltRoad { player: s.current, edge: e });
    awards::update_longest_road(s);
}

pub fn apply_build_settlement<S: EventSink>(s: &mut State, n: u8, sink: &mut S) {
    let p = s.current as usize;
    pay(s, p, &SETTLEMENT_COST);
    s.players[p].settlements |= 1u64 << n;
    sink.emit(Event::BuiltSettlement { player: s.current, node: n });
    awards::update_longest_road(s);
}

pub fn apply_build_city<S: EventSink>(s: &mut State, n: u8, sink: &mut S) {
    let p = s.current as usize;
    pay(s, p, &CITY_COST);
    s.players[p].settlements &= !(1u64 << n);
    s.players[p].cities |= 1u64 << n;
    sink.emit(Event::BuiltCity { player: s.current, node: n });
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
