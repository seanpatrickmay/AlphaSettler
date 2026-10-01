//! Snake-order initial placement.

use crate::action::Action;
use crate::apply::EventSink;
use crate::events::Event;
use crate::rules::awards;
use crate::state::{Phase, State, SETUP_ORDER};
use crate::topology::{topo, NUM_NODES};
use crate::types::{hand_add, hand_sub, hand_total};

pub fn legal_settlement(s: &State, out: &mut Vec<Action>) {
    let t = topo();
    let occ = s.occupied_nodes();
    for n in 0..NUM_NODES {
        if occ & (1u64 << n) == 0 && occ & t.node_neighbor_mask[n] == 0 {
            out.push(Action::BuildSettlement(n as u8));
        }
    }
}

pub fn legal_road(s: &State, node: u8, out: &mut Vec<Action>) {
    let occ = s.occupied_edges();
    for &e in &topo().node_edges[node as usize] {
        if occ & (1u128 << e) == 0 {
            out.push(Action::BuildRoad(e));
        }
    }
}

pub fn apply_settlement<S: EventSink>(s: &mut State, n: u8, sink: &mut S) {
    let p = s.current as usize;
    s.players[p].settlements |= 1u64 << n;
    sink.emit(Event::BuiltSettlement {
        player: s.current,
        node: n,
    });
    if s.setup_step >= 4 {
        let mut got = [0u8; 5];
        for &tile in &topo().node_tiles[n as usize] {
            if let Some(r) = s.board.tile_resource[tile as usize] {
                got[r.index()] += 1;
            }
        }
        hand_add(&mut s.players[p].hand, &got);
        hand_sub(&mut s.bank, &got);
        if hand_total(&got) > 0 {
            sink.emit(Event::Produced {
                player: s.current,
                resources: got,
            });
        }
    }
    s.phase = Phase::SetupRoad { node: n };
}

pub fn apply_road<S: EventSink>(s: &mut State, node: u8, e: u8, sink: &mut S) {
    debug_assert!(topo().node_edges[node as usize].contains(&e));
    let p = s.current as usize;
    s.players[p].roads |= 1u128 << e;
    sink.emit(Event::BuiltRoad {
        player: s.current,
        edge: e,
    });
    awards::recompute_road_len(s, p);
    awards::assign_longest_road(s);
    s.setup_step += 1;
    if s.setup_step as usize == SETUP_ORDER.len() {
        s.current = 0;
        s.phase = Phase::PreRoll;
    } else {
        s.current = SETUP_ORDER[s.setup_step as usize];
        s.phase = Phase::SetupSettlement;
    }
}
