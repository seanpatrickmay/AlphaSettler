use settler_engine::topology::*;

#[test]
fn counts() {
    let t = topo();
    assert_eq!(t.tile_nodes.len(), NUM_TILES);
    assert_eq!(t.edge_nodes.len(), NUM_EDGES);
    assert_eq!(t.node_tiles.len(), NUM_NODES);
    assert_eq!((NUM_TILES, NUM_NODES, NUM_EDGES), (19, 54, 72));
}

#[test]
fn every_tile_has_six_distinct_nodes() {
    let t = topo();
    for (i, nodes) in t.tile_nodes.iter().enumerate() {
        let mut v = nodes.to_vec();
        v.sort();
        v.dedup();
        assert_eq!(v.len(), 6, "tile {i}");
        assert_eq!(t.tile_node_mask[i].count_ones(), 6, "tile {i}");
    }
}

#[test]
fn node_tile_counts_and_degrees() {
    let t = topo();
    let tile_slots: usize = t.node_tiles.iter().map(|v| v.len()).sum();
    assert_eq!(tile_slots, NUM_TILES * 6);
    let degree_sum: usize = t.node_neighbors.iter().map(|v| v.len()).sum();
    assert_eq!(degree_sum, 2 * NUM_EDGES);
    for n in 0..NUM_NODES {
        assert!((1..=3).contains(&t.node_tiles[n].len()), "node {n}");
        assert!((2..=3).contains(&t.node_neighbors[n].len()), "node {n}");
        assert_eq!(t.node_neighbors[n].len(), t.node_edges[n].len());
    }
}

#[test]
fn masks_agree_with_lists() {
    let t = topo();
    for n in 0..NUM_NODES {
        let nm: u64 = t.node_neighbors[n].iter().map(|&x| 1u64 << x).sum();
        assert_eq!(nm, t.node_neighbor_mask[n]);
        let em: u128 = t.node_edges[n].iter().map(|&e| 1u128 << e).sum();
        assert_eq!(em, t.node_edge_mask[n]);
    }
    for (e, &(a, b)) in t.edge_nodes.iter().enumerate() {
        assert!(a < b);
        assert!(t.node_edges[a as usize].contains(&(e as u8)));
        assert!(t.node_edges[b as usize].contains(&(e as u8)));
    }
}

#[test]
fn tile_neighbors() {
    let t = topo();
    assert_eq!(t.tile_coords[9], (0, 0));
    assert_eq!(t.tile_neighbors[9].len(), 6);
    assert_eq!(t.tile_coords[0], (0, -2));
    assert_eq!(t.tile_neighbors[0].len(), 3);
    for (i, ns) in t.tile_neighbors.iter().enumerate() {
        for &j in ns {
            assert!(t.tile_neighbors[j as usize].contains(&(i as u8)));
            let shared = (t.tile_node_mask[i] & t.tile_node_mask[j as usize]).count_ones();
            assert_eq!(shared, 2, "neighboring tiles {i},{j} share an edge");
        }
    }
}

#[test]
fn coastline_is_a_30_edge_cycle() {
    let t = topo();
    let c = &t.coastal_edges;
    assert_eq!(c.len(), 30);
    let mut sorted = c.clone();
    sorted.sort();
    sorted.dedup();
    assert_eq!(sorted.len(), 30);
    for i in 0..30 {
        let (a, b) = t.edge_nodes[c[i] as usize];
        let (x, y) = t.edge_nodes[c[(i + 1) % 30] as usize];
        assert!(
            a == x || a == y || b == x || b == y,
            "coastal edges {i} and {} not adjacent",
            i + 1
        );
    }
}
