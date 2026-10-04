//! The board as a graph for network v2 (spec Section 2): where the global scalars sit in the
//! public vector, and the static adjacency, built once from the engine's topology. The PyTorch
//! twin gets the same lists through the bindings (`net_adjacency`).

use crate::encode::{
    EDGE_BASE, EDGE_WIDTH, NODE_BASE, NODE_WIDTH, OWN_HAND, PUBLIC_LEN, TILE_BASE, TILE_WIDTH,
};
use settler_engine::topology::{topo, NUM_EDGES, NUM_NODES, NUM_TILES};
use std::sync::LazyLock;

/// The global scalars: every slot after the tile block, from the actor's own hand on (91).
pub const GLOBAL_BASE: usize = OWN_HAND;
pub const GLOBAL_WIDTH: usize = PUBLIC_LEN - GLOBAL_BASE;

// The entity blocks tile the public vector in order: nodes, edges, tiles, then the globals.
const _: () = assert!(EDGE_BASE == NODE_BASE + NUM_NODES * NODE_WIDTH);
const _: () = assert!(TILE_BASE == EDGE_BASE + NUM_EDGES * EDGE_WIDTH);
const _: () = assert!(GLOBAL_BASE == TILE_BASE + NUM_TILES * TILE_WIDTH);

/// Each entity's neighbours of one kind, flattened: entity `j`'s are `index[start[j]..start[j + 1]]`
/// in the order they are summed, and the sum is multiplied by `scale[j]` (1 / their count for a
/// mean, 1 for a sum).
#[derive(Clone, Debug, PartialEq)]
pub struct Links {
    pub start: Vec<usize>,
    pub index: Vec<u8>,
    pub scale: Vec<f32>,
}

impl Links {
    fn build(lists: &[Vec<u8>], mean: bool) -> Links {
        let mut links = Links {
            start: vec![0],
            index: Vec::new(),
            scale: Vec::new(),
        };
        for l in lists {
            assert!(!l.is_empty(), "every entity has a neighbour of each kind");
            links.index.extend_from_slice(l);
            links.start.push(links.index.len());
            links
                .scale
                .push(if mean { 1.0 / l.len() as f32 } else { 1.0 });
        }
        links
    }

    /// How many entities the table covers.
    pub fn count(&self) -> usize {
        self.scale.len()
    }

    /// Entity `j`'s neighbours.
    pub fn of(&self, j: usize) -> &[u8] {
        &self.index[self.start[j]..self.start[j + 1]]
    }

    /// Every entity's neighbours, as lists (for the bindings and tests).
    pub fn lists(&self) -> Vec<Vec<usize>> {
        (0..self.count())
            .map(|j| self.of(j).iter().map(|&k| k as usize).collect())
            .collect()
    }
}

pub struct Graph {
    /// Mean over a node's 1 to 3 tiles.
    pub node_tiles: Links,
    /// Mean over a node's 2 or 3 edges.
    pub node_edges: Links,
    /// Mean over a node's 2 or 3 neighbouring nodes.
    pub node_neighbours: Links,
    /// Sum over an edge's two nodes.
    pub edge_nodes: Links,
    /// Mean over a tile's six nodes.
    pub tile_nodes: Links,
}

static GRAPH: LazyLock<Graph> = LazyLock::new(|| {
    let t = topo();
    let edge_nodes: Vec<Vec<u8>> = t.edge_nodes.iter().map(|&(a, b)| vec![a, b]).collect();
    let tile_nodes: Vec<Vec<u8>> = t.tile_nodes.iter().map(|n| n.to_vec()).collect();
    Graph {
        node_tiles: Links::build(&t.node_tiles, true),
        node_edges: Links::build(&t.node_edges, true),
        node_neighbours: Links::build(&t.node_neighbors, true),
        edge_nodes: Links::build(&edge_nodes, false),
        tile_nodes: Links::build(&tile_nodes, true),
    }
});

pub fn graph() -> &'static Graph {
    &GRAPH
}
