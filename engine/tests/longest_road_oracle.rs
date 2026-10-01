//! The engine's `longest_road` against a straightforward reference: start a trail from both
//! ends of every road. Kept as an oracle so faster implementations can't drift.

use settler_engine::legal::legal_actions;
use settler_engine::rng::Rng;
use settler_engine::rules::awards::longest_road;
use settler_engine::topology::{topo, NUM_EDGES};
use settler_engine::*;

fn reference(roads: u128, blocked: u64) -> u8 {
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
    let t = topo();
    let mut best = 0;
    for e in bits128(roads) {
        let (a, b) = t.edge_nodes[e as usize];
        let rest = roads & !(1u128 << e);
        best = best
            .max(extend(a, rest, blocked, 1))
            .max(extend(b, rest, blocked, 1));
    }
    best
}

/// A connected-ish random road network: grow from a random edge, sometimes jumping.
fn random_roads(rng: &mut Rng, count: u32) -> u128 {
    let t = topo();
    let mut roads = 1u128 << rng.below(NUM_EDGES as u32);
    while roads.count_ones() < count {
        let frontier: Vec<u8> = (0..NUM_EDGES as u8)
            .filter(|&e| {
                roads & (1u128 << e) == 0 && {
                    let (a, b) = t.edge_nodes[e as usize];
                    roads & (t.node_edge_mask[a as usize] | t.node_edge_mask[b as usize]) != 0
                }
            })
            .collect();
        let e = if frontier.is_empty() || rng.below(8) == 0 {
            rng.below(NUM_EDGES as u32) as u8
        } else {
            frontier[rng.below(frontier.len() as u32) as usize]
        };
        roads |= 1u128 << e;
    }
    roads
}

#[test]
fn matches_reference_on_random_networks() {
    let mut rng = Rng::new(99);
    for case in 0..20_000 {
        let count = 1 + rng.below(15);
        let roads = random_roads(&mut rng, count);
        let blocked = if rng.below(2) == 0 {
            0
        } else {
            rng.next_u64() & rng.next_u64() & ((1u64 << 54) - 1)
        };
        assert_eq!(
            longest_road(roads, blocked),
            reference(roads, blocked),
            "case {case}: roads {roads:#x} blocked {blocked:#x}"
        );
    }
}

#[test]
fn matches_reference_on_rings_and_figure_eights() {
    let t = topo();
    for tile in 0..19 {
        let ring: u128 = (0..6)
            .map(|i| {
                let (a, b) = (t.tile_nodes[tile][i], t.tile_nodes[tile][(i + 1) % 6]);
                let e = t.node_edges[a as usize]
                    .iter()
                    .copied()
                    .find(|&e| t.node_edges[b as usize].contains(&e))
                    .unwrap();
                1u128 << e
            })
            .sum();
        for blocked in [0u64, 1u64 << t.tile_nodes[tile][0], t.tile_node_mask[tile]] {
            assert_eq!(
                longest_road(ring, blocked),
                reference(ring, blocked),
                "ring {tile}"
            );
        }
        for &nb in &t.tile_neighbors[tile] {
            let ring2: u128 = (0..6)
                .map(|i| {
                    let (a, b) = (
                        t.tile_nodes[nb as usize][i],
                        t.tile_nodes[nb as usize][(i + 1) % 6],
                    );
                    let e = t.node_edges[a as usize]
                        .iter()
                        .copied()
                        .find(|&e| t.node_edges[b as usize].contains(&e))
                        .unwrap();
                    1u128 << e
                })
                .sum();
            let both = ring | ring2;
            assert_eq!(
                longest_road(both, 0),
                reference(both, 0),
                "rings {tile},{nb}"
            );
        }
    }
}

#[test]
fn matches_reference_in_random_games() {
    let mut rng = Rng::new(5);
    let mut buf = Vec::new();
    for seed in 0..300 {
        let mut s = State::new(
            seed,
            GameConfig {
                max_offers_per_turn: 0,
                ..GameConfig::default()
            },
        );
        while !s.is_over() {
            legal_actions(&s, &mut buf);
            let a = buf[rng.below(buf.len() as u32) as usize];
            s.apply(a);
            for p in 0..NUM_PLAYERS {
                let blocked = s.occupied_nodes() & !s.buildings(p);
                assert_eq!(
                    longest_road(s.players[p].roads, blocked),
                    reference(s.players[p].roads, blocked)
                );
            }
        }
    }
}
