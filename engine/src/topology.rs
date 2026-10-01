//! The fixed graph of the standard 19-tile board, built once.
//!
//! Tiles use axial coordinates (q, r) in a radius-2 hexagon, ordered by r then q.
//! Pointy-top corners are placed on an integer lattice: a tile's center is
//! (2q + r, 3r) and its corners are offset by `CORNER_OFFSETS`. Nodes are numbered
//! top-to-bottom, left-to-right; edges are sorted by (low node, high node).

use std::collections::{BTreeMap, BTreeSet};
use std::sync::LazyLock;

pub const NUM_TILES: usize = 19;
pub const NUM_NODES: usize = 54;
pub const NUM_EDGES: usize = 72;

/// Clockwise from the top corner (y grows downward).
const CORNER_OFFSETS: [(i32, i32); 6] = [(0, -2), (1, -1), (1, 1), (0, 2), (-1, 1), (-1, -1)];
const AXIAL_DIRS: [(i32, i32); 6] = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)];

pub struct Topology {
    pub tile_coords: [(i8, i8); NUM_TILES],
    pub tile_nodes: [[u8; 6]; NUM_TILES],
    pub tile_node_mask: [u64; NUM_TILES],
    pub tile_neighbors: Vec<Vec<u8>>,
    pub node_tiles: Vec<Vec<u8>>,
    pub node_neighbors: Vec<Vec<u8>>,
    pub node_neighbor_mask: [u64; NUM_NODES],
    pub node_edges: Vec<Vec<u8>>,
    pub node_edge_mask: [u128; NUM_NODES],
    pub edge_nodes: [(u8, u8); NUM_EDGES],
    pub coastal_edges: Vec<u8>,
}

static TOPOLOGY: LazyLock<Topology> = LazyLock::new(Topology::build);

#[inline]
pub fn topo() -> &'static Topology {
    &TOPOLOGY
}

fn corner((q, r): (i32, i32), i: usize) -> (i32, i32) {
    let (dx, dy) = CORNER_OFFSETS[i];
    (2 * q + r + dx, 3 * r + dy)
}

impl Topology {
    fn build() -> Topology {
        let mut coords: Vec<(i32, i32)> = Vec::new();
        for r in -2i32..=2 {
            for q in -2i32..=2 {
                if (q + r).abs() <= 2 {
                    coords.push((q, r));
                }
            }
        }
        assert_eq!(coords.len(), NUM_TILES);

        let mut points: Vec<(i32, i32)> = coords
            .iter()
            .flat_map(|&c| (0..6).map(move |i| corner(c, i)))
            .collect();
        points.sort_by_key(|&(x, y)| (y, x));
        points.dedup();
        assert_eq!(points.len(), NUM_NODES);
        let node_of: BTreeMap<(i32, i32), u8> = points
            .iter()
            .enumerate()
            .map(|(i, &p)| (p, i as u8))
            .collect();

        let mut tile_coords = [(0i8, 0i8); NUM_TILES];
        let mut tile_nodes = [[0u8; 6]; NUM_TILES];
        let mut tile_node_mask = [0u64; NUM_TILES];
        for (t, &c) in coords.iter().enumerate() {
            tile_coords[t] = (c.0 as i8, c.1 as i8);
            for i in 0..6 {
                let n = node_of[&corner(c, i)];
                tile_nodes[t][i] = n;
                tile_node_mask[t] |= 1u64 << n;
            }
        }

        let mut edge_set: BTreeSet<(u8, u8)> = BTreeSet::new();
        for nodes in &tile_nodes {
            for i in 0..6 {
                let (a, b) = (nodes[i], nodes[(i + 1) % 6]);
                edge_set.insert((a.min(b), a.max(b)));
            }
        }
        assert_eq!(edge_set.len(), NUM_EDGES);
        let edge_list: Vec<(u8, u8)> = edge_set.into_iter().collect();
        let mut edge_nodes = [(0u8, 0u8); NUM_EDGES];
        edge_nodes.copy_from_slice(&edge_list);

        let mut node_tiles: Vec<Vec<u8>> = vec![Vec::new(); NUM_NODES];
        for (t, nodes) in tile_nodes.iter().enumerate() {
            for &n in nodes {
                node_tiles[n as usize].push(t as u8);
            }
        }

        let mut node_neighbors: Vec<Vec<u8>> = vec![Vec::new(); NUM_NODES];
        let mut node_edges: Vec<Vec<u8>> = vec![Vec::new(); NUM_NODES];
        let mut node_neighbor_mask = [0u64; NUM_NODES];
        let mut node_edge_mask = [0u128; NUM_NODES];
        for (e, &(a, b)) in edge_nodes.iter().enumerate() {
            node_neighbors[a as usize].push(b);
            node_neighbors[b as usize].push(a);
            node_edges[a as usize].push(e as u8);
            node_edges[b as usize].push(e as u8);
            node_neighbor_mask[a as usize] |= 1u64 << b;
            node_neighbor_mask[b as usize] |= 1u64 << a;
            node_edge_mask[a as usize] |= 1u128 << e;
            node_edge_mask[b as usize] |= 1u128 << e;
        }

        let tile_neighbors: Vec<Vec<u8>> = coords
            .iter()
            .map(|&(q, r)| {
                AXIAL_DIRS
                    .iter()
                    .filter_map(|&(dq, dr)| {
                        coords
                            .iter()
                            .position(|&c| c == (q + dq, r + dr))
                            .map(|i| i as u8)
                    })
                    .collect()
            })
            .collect();

        // Coastal edges border exactly one tile; walk them into a cycle.
        let mut edge_tile_count = [0u8; NUM_EDGES];
        for nodes in &tile_nodes {
            for i in 0..6 {
                let (a, b) = (nodes[i], nodes[(i + 1) % 6]);
                let e = edge_list.binary_search(&(a.min(b), a.max(b))).unwrap();
                edge_tile_count[e] += 1;
            }
        }
        let is_coastal = |e: u8| edge_tile_count[e as usize] == 1;
        let start = (0..NUM_EDGES as u8).find(|&e| is_coastal(e)).unwrap();
        let mut coastal_edges = vec![start];
        let mut prev = start;
        let mut node = edge_nodes[start as usize].1;
        loop {
            let next = *node_edges[node as usize]
                .iter()
                .find(|&&e| e != prev && is_coastal(e))
                .unwrap();
            if next == start {
                break;
            }
            coastal_edges.push(next);
            let (a, b) = edge_nodes[next as usize];
            node = if a == node { b } else { a };
            prev = next;
        }
        assert_eq!(coastal_edges.len(), 30);

        Topology {
            tile_coords,
            tile_nodes,
            tile_node_mask,
            tile_neighbors,
            node_tiles,
            node_neighbors,
            node_neighbor_mask,
            node_edges,
            node_edge_mask,
            edge_nodes,
            coastal_edges,
        }
    }
}
