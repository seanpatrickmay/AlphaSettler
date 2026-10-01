//! Moving the robber and stealing.

use crate::action::Action;
use crate::apply::{Chance, EventSink};
use crate::events::Event;
use crate::rules::roll::pick_card;
use crate::state::{Phase, State};
use crate::topology::{topo, NUM_TILES};
use crate::types::*;

pub fn legal_move_robber(s: &State, out: &mut Vec<Action>) {
    for t in 0..NUM_TILES as u8 {
        if t != s.robber {
            out.push(Action::MoveRobber(t));
        }
    }
}

/// Opponents with a building on the robber's tile and at least one card.
pub fn victims(s: &State) -> [bool; NUM_PLAYERS] {
    let mask = topo().tile_node_mask[s.robber as usize];
    let mut v = [false; NUM_PLAYERS];
    for p in 0..NUM_PLAYERS {
        v[p] = p != s.current as usize
            && s.buildings(p) & mask != 0
            && hand_total(&s.players[p].hand) > 0;
    }
    v
}

pub fn apply_move_robber<S: EventSink>(s: &mut State, tile: u8, sink: &mut S) {
    s.robber = tile;
    sink.emit(Event::RobberMoved { player: s.current, tile });
    s.phase = if victims(s).iter().any(|&v| v) { Phase::Steal } else { s.robber_return };
}

pub fn legal_steal(s: &State, out: &mut Vec<Action>) {
    for (p, &v) in victims(s).iter().enumerate() {
        if v {
            out.push(Action::StealFrom(p as PlayerId));
        }
    }
}

pub fn apply_steal<S: EventSink>(s: &mut State, victim: PlayerId, chance: Option<Chance>, sink: &mut S) {
    let v = victim as usize;
    let hand = s.players[v].hand;
    let r = match chance {
        None => pick_card(&hand, &mut s.rng_steal),
        Some(Chance::Steal(r)) => {
            assert!(hand[r.index()] > 0, "forced steal of {r:?} but victim has none");
            r
        }
        Some(c) => panic!("chance {c:?} does not match StealFrom"),
    };
    s.players[v].hand[r.index()] -= 1;
    s.players[s.current as usize].hand[r.index()] += 1;
    sink.emit(Event::Stole { thief: s.current, victim, resource: Some(r) });
    s.phase = s.robber_return;
}
