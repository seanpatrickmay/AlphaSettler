//! Longest road, largest army, and the win check.

use crate::apply::EventSink;
use crate::events::Event;
use crate::state::{Phase, State};
use crate::topology::topo;
use crate::types::*;

/// Longest trail (no edge reused) in `roads`; trails may end at but not pass through `blocked` nodes.
pub fn longest_road(roads: u128, blocked: u64) -> u8 {
    let t = topo();
    let mut best = 0;
    for e in bits128(roads) {
        let (a, b) = t.edge_nodes[e as usize];
        let rest = roads & !(1u128 << e);
        best = best.max(extend(a, rest, blocked, 1)).max(extend(b, rest, blocked, 1));
    }
    best
}

fn extend(node: u8, roads: u128, blocked: u64, len: u8) -> u8 {
    if blocked & (1u64 << node) != 0 {
        return len;
    }
    let t = topo();
    let mut best = len;
    for e in bits128(roads & t.node_edge_mask[node as usize]) {
        let (a, b) = t.edge_nodes[e as usize];
        let next = if a == node { b } else { a };
        best = best.max(extend(next, roads & !(1u128 << e), blocked, len + 1));
    }
    best
}

/// Recompute every player's longest road and reassign the award (min 5; holder keeps ties).
pub fn update_longest_road(s: &mut State) {
    let occ = s.occupied_nodes();
    let mut lens = [0u8; NUM_PLAYERS];
    for p in 0..NUM_PLAYERS {
        let blocked = occ & !s.buildings(p);
        lens[p] = longest_road(s.players[p].roads, blocked);
        s.players[p].longest_road_len = lens[p];
    }
    let max = *lens.iter().max().unwrap();
    if max < 5 {
        s.longest_road_owner = None;
        return;
    }
    if let Some(h) = s.longest_road_owner {
        if lens[h as usize] == max {
            return;
        }
    }
    let leaders = lens.iter().filter(|&&l| l == max).count();
    s.longest_road_owner = if leaders == 1 {
        Some(lens.iter().position(|&l| l == max).unwrap() as PlayerId)
    } else {
        None
    };
}

/// After player `p` plays a knight: award at 3+, taken only by strictly more knights.
pub fn update_largest_army(s: &mut State, p: usize) {
    let k = s.players[p].knights_played;
    if k < 3 {
        return;
    }
    match s.largest_army_owner {
        None => s.largest_army_owner = Some(p as PlayerId),
        Some(h) if h as usize != p && k > s.players[h as usize].knights_played => {
            s.largest_army_owner = Some(p as PlayerId)
        }
        _ => {}
    }
}

/// The current player wins as soon as they reach the target on their own turn.
pub fn check_win<S: EventSink>(s: &mut State, sink: &mut S) {
    if s.is_over() {
        return;
    }
    let p = s.current;
    if s.total_vp(p as usize) >= s.config.vp_to_win {
        s.phase = Phase::GameOver { winner: Some(p) };
        sink.emit(Event::GameOver { winner: Some(p) });
    }
}
