use settler_engine::topology::{topo, NUM_EDGES, NUM_NODES, NUM_TILES};
use settler_net::encode::{OWN_HAND, PUBLIC_LEN};
use settler_net::graph::*;

fn lists(v: &[Vec<u8>]) -> Vec<Vec<usize>> {
    v.iter()
        .map(|l| l.iter().map(|&x| x as usize).collect())
        .collect()
}

#[test]
fn the_global_block_is_the_139_scalars_after_the_tiles() {
    // v3's 115 and the threat block's 4 x 6.
    assert_eq!((GLOBAL_BASE, GLOBAL_WIDTH), (OWN_HAND, 139));
    assert_eq!(GLOBAL_BASE + GLOBAL_WIDTH, PUBLIC_LEN);
}

#[test]
fn the_tables_are_the_engines_topology() {
    let (g, t) = (graph(), topo());
    assert_eq!(g.node_tiles.lists(), lists(&t.node_tiles));
    assert_eq!(g.node_edges.lists(), lists(&t.node_edges));
    assert_eq!(g.node_neighbours.lists(), lists(&t.node_neighbors));
    let ends: Vec<Vec<u8>> = t.edge_nodes.iter().map(|&(a, b)| vec![a, b]).collect();
    assert_eq!(g.edge_nodes.lists(), lists(&ends));
    let corners: Vec<Vec<u8>> = t.tile_nodes.iter().map(|c| c.to_vec()).collect();
    assert_eq!(g.tile_nodes.lists(), lists(&corners));
    assert_eq!(
        (
            g.node_tiles.count(),
            g.node_edges.count(),
            g.node_neighbours.count(),
            g.edge_nodes.count(),
            g.tile_nodes.count()
        ),
        (NUM_NODES, NUM_NODES, NUM_NODES, NUM_EDGES, NUM_TILES)
    );
}

#[test]
fn the_tables_agree_with_each_other() {
    let g = graph();
    // 19 hexes with 6 corners each; 72 edges with 2 ends each.
    assert_eq!(g.node_tiles.index.len(), 6 * NUM_TILES);
    assert_eq!(g.tile_nodes.index.len(), 6 * NUM_TILES);
    assert_eq!(g.node_edges.index.len(), 2 * NUM_EDGES);
    assert_eq!(g.node_neighbours.index.len(), 2 * NUM_EDGES);
    for n in 0..NUM_NODES {
        let me = n as u8;
        assert!((1..=3).contains(&g.node_tiles.of(n).len()), "node {n}");
        assert!((2..=3).contains(&g.node_edges.of(n).len()), "node {n}");
        for &t in g.node_tiles.of(n) {
            assert!(
                g.tile_nodes.of(t as usize).contains(&me),
                "node {n}, tile {t}"
            );
        }
        for &e in g.node_edges.of(n) {
            assert!(
                g.edge_nodes.of(e as usize).contains(&me),
                "node {n}, edge {e}"
            );
        }
        for &m in g.node_neighbours.of(n) {
            assert!(g.node_neighbours.of(m as usize).contains(&me), "symmetric");
            assert!(
                g.node_edges
                    .of(n)
                    .iter()
                    .any(|&e| g.edge_nodes.of(e as usize).contains(&m)),
                "node {m} is the far end of one of node {n}'s edges"
            );
        }
    }
    for links in [
        &g.node_tiles,
        &g.node_edges,
        &g.node_neighbours,
        &g.tile_nodes,
    ] {
        for j in 0..links.count() {
            assert_eq!(links.scale[j], 1.0 / links.of(j).len() as f32, "a mean");
        }
    }
    for e in 0..NUM_EDGES {
        let ends = g.edge_nodes.of(e);
        assert!(ends.len() == 2 && ends[0] != ends[1], "edge {e}");
        assert_eq!(g.edge_nodes.scale[e], 1.0, "a sum");
    }
}
