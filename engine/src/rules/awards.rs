//! Longest road, largest army, and the win check.

use crate::apply::EventSink;
use crate::events::Event;
use crate::state::{Phase, State};
use crate::topology::{topo, Topology};
use crate::types::*;

/// Longest trail (no edge reused) in `roads`; trails may end at but not pass through `blocked` nodes.
///
/// Searches only from nodes where some longest trail must start: dead ends, branch points, and
/// blocked nodes. A trail starting at an unblocked degree-2 node either extends backwards
/// (so it wasn't longest) or is closed and can be rotated onto a non-degree-2 node; the one
/// exception is a component that is a pure cycle, which gets one start of its own.
pub fn longest_road(roads: u128, blocked: u64) -> u8 {
    if roads == 0 {
        return 0;
    }
    let g = LocalGraph::new(roads, blocked);
    let mut starts = 0u32;
    for v in 0..g.nodes {
        if g.blocked & (1 << v) != 0 || g.degree[v] != 2 {
            starts |= 1 << v;
        }
    }
    // Any component the starts can't reach is a pure cycle: give it one start.
    let all = if g.nodes == 32 {
        u32::MAX
    } else {
        (1u32 << g.nodes) - 1
    };
    let mut reached = g.flood(starts);
    while all & !reached != 0 {
        let v = (all & !reached).trailing_zeros();
        starts |= 1 << v;
        reached |= g.flood(1 << v);
    }
    let mut best = 0;
    let mut rest = starts;
    while rest != 0 {
        let v = rest.trailing_zeros() as usize;
        rest &= rest - 1;
        best = best.max(g.walk(v, 0, 0));
    }
    best
}

/// One player's road network, renumbered compactly so the trail search runs on `u32` masks
/// and small adjacency arrays instead of the whole board.
struct LocalGraph {
    nodes: usize,
    blocked: u32,
    degree: [u8; 32],
    /// Up to three (local edge, other local node) pairs per node.
    adj: [[(u8, u8); 3]; 32],
}

impl LocalGraph {
    fn new(roads: u128, blocked: u64) -> LocalGraph {
        let t = topo();
        // 16 roads touch at most 32 nodes, the width of the local masks. A player has 15.
        assert!(
            roads.count_ones() <= 16,
            "trail search supports at most 16 roads"
        );
        let mut local = [u8::MAX; 54];
        let mut g = LocalGraph {
            nodes: 0,
            blocked: 0,
            degree: [0; 32],
            adj: [[(0, 0); 3]; 32],
        };
        let mut id = |n: u8, g: &mut LocalGraph| -> u8 {
            if local[n as usize] == u8::MAX {
                local[n as usize] = g.nodes as u8;
                if blocked & (1u64 << n) != 0 {
                    g.blocked |= 1 << g.nodes;
                }
                g.nodes += 1;
            }
            local[n as usize]
        };
        for (i, e) in bits128(roads).enumerate() {
            let (a, b) = t.edge_nodes[e as usize];
            let (la, lb) = (id(a, &mut g), id(b, &mut g));
            g.adj[la as usize][g.degree[la as usize] as usize] = (i as u8, lb);
            g.degree[la as usize] += 1;
            g.adj[lb as usize][g.degree[lb as usize] as usize] = (i as u8, la);
            g.degree[lb as usize] += 1;
        }
        g
    }

    fn flood(&self, from: u32) -> u32 {
        let mut reached = from;
        let mut frontier = from;
        while frontier != 0 {
            let mut next = 0u32;
            let mut f = frontier;
            while f != 0 {
                let v = f.trailing_zeros() as usize;
                f &= f - 1;
                for &(_, w) in &self.adj[v][..self.degree[v] as usize] {
                    next |= 1 << w;
                }
            }
            frontier = next & !reached;
            reached |= next;
        }
        reached
    }

    fn walk(&self, v: usize, used: u32, len: u8) -> u8 {
        if len > 0 && self.blocked & (1 << v) != 0 {
            return len;
        }
        let mut best = len;
        for &(e, w) in &self.adj[v][..self.degree[v] as usize] {
            if used & (1 << e) == 0 {
                best = best.max(self.walk(w as usize, used | (1 << e), len + 1));
            }
        }
        best
    }
}

/// Nodes connected to `from` through `roads`.
fn flood(t: &Topology, roads: u128, from: u64) -> u64 {
    let mut reached = from;
    let mut frontier = from;
    while frontier != 0 {
        let mut next = 0u64;
        for n in bits64(frontier) {
            for e in bits128(roads & t.node_edge_mask[n as usize]) {
                next |= t.edge_node_mask[e as usize];
            }
        }
        frontier = next & !reached;
        reached |= next;
    }
    reached
}

/// Refresh player `p`'s cached `longest_road_len`. Call after `p` places a road, or after an
/// opponent's building lands on a node `p`'s roads touch; no other change can alter it.
pub fn recompute_road_len(s: &mut State, p: usize) {
    let blocked = s.occupied_nodes() & !s.buildings(p);
    s.players[p].longest_road_len = longest_road(s.players[p].roads, blocked);
}

/// Refresh player `p`'s cached `longest_road_len` after `p` placed the road on edge `e`.
/// A new road can only lengthen trails in its own connected network, so only that network is
/// searched and the cached value of the rest stands.
pub fn extend_road_len(s: &mut State, p: usize, e: u8) {
    let t = topo();
    let roads = s.players[p].roads;
    let network = flood(t, roads, t.edge_node_mask[e as usize]);
    let network_roads =
        bits64(network).fold(0u128, |m, n| m | t.node_edge_mask[n as usize]) & roads;
    let blocked = s.occupied_nodes() & !s.buildings(p);
    let len = longest_road(network_roads, blocked);
    let cached = &mut s.players[p].longest_road_len;
    *cached = (*cached).max(len);
}

/// Reassign the award from the cached lengths (min 5; holder keeps ties; a broken holder
/// with tied challengers leaves it unowned).
pub fn assign_longest_road(s: &mut State) {
    let lens: [u8; NUM_PLAYERS] = std::array::from_fn(|p| s.players[p].longest_road_len);
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
