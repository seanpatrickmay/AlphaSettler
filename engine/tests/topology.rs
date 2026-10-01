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

// Golden pins. Node, edge, and tile numbering are frozen contracts: the policy head's action
// ids and the Catanatron mapping both depend on them. If one of these fails, the numbering
// changed; that breaks the action space, so fix the change rather than the expected values.

#[test]
fn golden_tile_coords() {
    let expected: [(i8, i8); NUM_TILES] = [
        (0, -2),
        (1, -2),
        (2, -2),
        (-1, -1),
        (0, -1),
        (1, -1),
        (2, -1),
        (-2, 0),
        (-1, 0),
        (0, 0),
        (1, 0),
        (2, 0),
        (-2, 1),
        (-1, 1),
        (0, 1),
        (1, 1),
        (-2, 2),
        (-1, 2),
        (0, 2),
    ];
    assert_eq!(topo().tile_coords, expected);
}

#[test]
fn golden_tile_nodes() {
    let t = topo();
    assert_eq!(t.tile_nodes[0], [0, 4, 8, 12, 7, 3]);
    assert_eq!(t.tile_nodes[9], [18, 24, 30, 35, 29, 23]);
    assert_eq!(t.tile_nodes[18], [41, 46, 50, 53, 49, 45]);
}

/// 64-bit FNV-1a.
fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

#[test]
fn golden_topology_hash() {
    let t = topo();
    let mut bytes = Vec::new();
    for &(a, b) in &t.edge_nodes {
        bytes.extend_from_slice(&[a, b]);
    }
    for nodes in &t.tile_nodes {
        bytes.extend_from_slice(nodes);
    }
    bytes.extend_from_slice(&t.coastal_edges);
    assert_eq!(bytes.len(), 2 * NUM_EDGES + 6 * NUM_TILES + 30);
    // Changing this value breaks the action space (see the golden-pin note above).
    assert_eq!(fnv1a(&bytes), 0x71a3_7ac2_6534_a23f);
}
