mod common;

use common::*;
use settler_engine::topology::{topo, NUM_EDGES, NUM_NODES, NUM_TILES};
use settler_engine::{GameConfig, ACTION_SPACE_SIZE, NUM_PLAYERS};
use settler_net::encode::*;
use settler_net::forward::{softmax, Forward, Outputs};
use settler_net::weights::*;

const SMALL: Arch = Arch {
    width: 4,
    rounds: 1,
};

fn err(bytes: &[u8]) -> String {
    Weights::from_bytes(bytes).unwrap_err()
}

#[test]
fn the_header_and_the_layer_shapes_follow_the_arch() {
    assert_eq!(FIXED_HEADER_LEN, 32);
    assert_eq!(MAGIC, *b"ASNET\0\0\0");
    assert_eq!(FORMAT_VERSION, 2);
    assert_eq!(
        Arch::DEFAULT,
        Arch {
            width: 48,
            rounds: 2
        }
    );
    let a = Arch {
        width: 3,
        rounds: 2,
    };
    assert_eq!(a.layer_count(), 20);
    assert_eq!(a.header_len(), 32 + 8 * 20);
    #[rustfmt::skip]
    let want = vec![
        (3, 19), (3, 4), (3, 8), (3, 115),
        (3, 9), (3, 15), (3, 9), (3, 12),
        (3, 9), (3, 15), (3, 9), (3, 12),
        (2, 6), (1, 6), (1, 6), (12, 12), (466, 12), (12, 42), (4, 12), (4, 12),
    ];
    assert_eq!(a.shapes(), want);
    assert_eq!(
        (
            a.round_layer(0, ROUND_TILE),
            a.round_layer(1, ROUND_EDGE),
            a.head_layer(NODE_POLICY),
            a.head_layer(VALUE)
        ),
        (4, 10, 12, 18)
    );
    let w = Weights::random(Arch::DEFAULT, 1);
    assert_eq!(w.param_count(), 248_446);
    assert_eq!(w.to_bytes().len(), Arch::DEFAULT.header_len() + 4 * 248_446);
    let (n, public, hidden) = inputs(301);
    let mut out = Outputs::default();
    Forward::new().run(&w, &public, &hidden, n, &mut out);
    assert_eq!(
        (out.n, out.policy.len(), out.value.len(), out.aux.len()),
        (n, n * ACTION_SPACE_SIZE, n * NUM_PLAYERS, n * NUM_PLAYERS)
    );
    assert!(out.policy.iter().chain(&out.value).all(|x| x.is_finite()));
}

#[test]
fn widths_and_round_counts_are_bounded() {
    let ok = |width, rounds| Arch { width, rounds }.check();
    assert!(ok(1, 1).is_ok() && ok(MAX_WIDTH, MAX_ROUNDS).is_ok());
    assert_eq!(
        ok(0, 1).unwrap_err(),
        format!("width 0 is out of range 1..={MAX_WIDTH}")
    );
    assert!(ok(MAX_WIDTH + 1, 1).unwrap_err().contains("width"));
    assert_eq!(
        ok(4, 0).unwrap_err(),
        format!("rounds 0 is out of range 1..={MAX_ROUNDS}")
    );
    assert!(ok(4, MAX_ROUNDS + 1).unwrap_err().contains("rounds"));
    let deepest = Weights::random(
        Arch {
            width: 1,
            rounds: MAX_ROUNDS,
        },
        2,
    );
    assert_eq!(Weights::from_bytes(&deepest.to_bytes()).unwrap(), deepest);
}

#[test]
fn weights_round_trip_through_bytes_and_files() {
    let w = Weights::random(SMALL, 7);
    assert_eq!(Weights::from_bytes(&w.to_bytes()).unwrap(), w);
    let path = temp_file("round-trip.bin");
    w.save(&path).unwrap();
    assert_eq!(Weights::load(&path).unwrap(), w);
    std::fs::remove_file(&path).unwrap();
    assert_ne!(Weights::random(SMALL, 8), w);
}

#[test]
fn a_bad_file_is_an_error_naming_what_is_wrong() {
    let good = Weights::random(SMALL, 3).to_bytes();
    let patch = |at: usize, v: u32| {
        let mut b = good.clone();
        b[at..at + 4].copy_from_slice(&v.to_le_bytes());
        b
    };
    let mut magic = good.clone();
    magic[0] = b'X';
    assert!(err(&magic).contains("magic"), "{}", err(&magic));
    let v1 = err(&patch(8, 1));
    assert!(
        v1.contains("format version 1, expected 2") && v1.contains("flat MLP"),
        "{v1}"
    );
    assert_eq!(err(&patch(8, 3)), "format version 3, expected 2");
    let mut hash = good.clone();
    hash[12] ^= 1;
    let e = err(&hash);
    assert!(e.contains("encoding hash"), "{e}");
    assert!(
        e.contains(&format!("{ENCODING_HASH:#018x}")),
        "names this encoder's hash: {e}"
    );
    assert!(
        e.contains(&format!("{:#018x}", ENCODING_HASH ^ 1)),
        "names the file's hash: {e}"
    );
    for (at, v, what) in [
        (20, 0, "width 0 is out of range"),
        (20, MAX_WIDTH as u32 + 1, "width 1025 is out of range"),
        (20, u32::MAX, "width 4294967295 is out of range"),
        (24, 0, "rounds 0 is out of range"),
        (24, MAX_ROUNDS as u32 + 1, "rounds 17 is out of range"),
    ] {
        assert!(err(&patch(at, v)).contains(what), "{}", err(&patch(at, v)));
    }
    // A declared width at the bound passes the bound and fails on the shapes it implies.
    assert!(err(&patch(20, MAX_WIDTH as u32)).contains("do not fit network v2"));
    assert!(err(&patch(24, 2)).contains("16 layers, expected 20 for 2 rounds"));
    assert!(err(&patch(28, 15)).contains("15 layers, expected 16"));
    let shape = patch(FIXED_HEADER_LEN + 8, 7); // the edge encoder's rows: 4 -> 7
    assert!(err(&shape).contains("shapes"), "{}", err(&shape));
    assert!(err(&good[..good.len() - 1]).contains("truncated"));
    assert!(err(&good[..40]).contains("shorter than the 160-byte header"));
    assert!(err(&good[..20]).contains("shorter than the 32-byte header"));
    assert!(err(&[]).contains("truncated"));
    let mut long = good.clone();
    long.push(0);
    assert!(err(&long).contains("trailing"), "{}", err(&long));
    let first = SMALL.header_len();
    let mut nan = good.clone();
    nan[first..first + 4].copy_from_slice(&f32::NAN.to_le_bytes());
    assert!(err(&nan).contains("NaN"), "{}", err(&nan));
    let mut inf = good.clone();
    let last = inf.len() - 4; // the aux head's last bias
    inf[last..].copy_from_slice(&f32::INFINITY.to_le_bytes());
    assert!(err(&inf).contains("infinite"), "{}", err(&inf));
    let missing = Weights::load(&temp_file("no-such.bin")).unwrap_err();
    assert!(
        missing.contains("cannot read") && missing.contains("no-such.bin"),
        "{missing}"
    );
}

/// A layer whose weights are all zero but for `(row, col, value)` entries, with these biases.
fn sparse(rows: usize, cols: usize, entries: &[(usize, usize, f32)], b: &[f32]) -> Layer {
    let mut w = vec![0.0; rows * cols];
    for &(r, c, v) in entries {
        w[r * cols + c] = v;
    }
    assert_eq!(b.len(), rows);
    Layer {
        rows,
        cols,
        w,
        b: b.to_vec(),
    }
}

/// d = 2, R = 1, every encoder weight zero: each entity type starts from its bias, all entities of
/// a type equal, so every mean is that value and an edge's sum is twice its nodes'. Each layer
/// picks out a few input columns, so the outputs below follow by hand from the column order.
#[test]
fn a_hand_computed_tiny_network() {
    let d = 2;
    let id: Vec<(usize, usize, f32)> = (0..4 * d).map(|k| (k, k, 1.0)).collect();
    let other_b: Vec<f32> = (0..466).map(|j| j as f32 / 1000.0).collect();
    let other: Vec<(usize, usize, f32)> = (0..466).map(|j| (j, j % 8, 1.0)).collect();
    let layers = vec![
        sparse(d, 19, &[], &[1.0, 2.0]),  // every node (1, 2)
        sparse(d, 4, &[], &[3.0, 4.0]),   // every edge (3, 4)
        sparse(d, 8, &[], &[5.0, 6.0]),   // every tile (5, 6)
        sparse(d, 115, &[], &[7.0, 8.0]), // global (7, 8)
        // tile + ReLU([mean node 0, global 1 - 10]) = (5, 6) + (1, 0) = (6, 6)
        sparse(d, 3 * d, &[(0, 2, 1.0), (1, 5, 1.0)], &[0.0, -10.0]),
        // node + ReLU([mean tile 0 + 0.5, mean edge 1 + 10 x mean neighbour 1]) = (1, 2) + (6.5, 24)
        sparse(
            d,
            5 * d,
            &[(0, 2, 1.0), (1, 5, 1.0), (1, 7, 10.0)],
            &[0.5, 0.0],
        ),
        // edge + ReLU([node sum 0, edge 0]) = (3, 4) + (15, 3) = (18, 7)
        sparse(d, 3 * d, &[(0, 2, 1.0), (1, 0, 1.0)], &[0.0, 0.0]),
        // global + ReLU([mean node 0 + mean edge 1, mean tile 0]) = (7, 8) + (14.5, 6) = (21.5, 14)
        sparse(
            d,
            4 * d,
            &[(0, 2, 1.0), (0, 5, 1.0), (1, 6, 1.0)],
            &[0.0, 0.0],
        ),
        // settlement = node 0, city = global 1 + 1
        sparse(2, 2 * d, &[(0, 0, 1.0), (1, 3, 1.0)], &[0.0, 1.0]),
        // road = edge 1 + global 0
        sparse(1, 2 * d, &[(0, 1, 1.0), (0, 2, 1.0)], &[0.0]),
        // robber = 0.25 - tile 0
        sparse(1, 2 * d, &[(0, 0, -1.0)], &[0.25]),
        sparse(4 * d, 4 * d, &id, &[0.0; 8]),
        // other slot j = pooled[j % 8] + j / 1000
        sparse(466, 4 * d, &other, &other_b),
        // value hidden = (pooled 0, hidden 0, hidden 29, 0, ...)
        sparse(
            4 * d,
            4 * d + HIDDEN_LEN,
            &[(0, 0, 1.0), (1, 8, 1.0), (2, 37, 1.0)],
            &[0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0],
        ),
        sparse(
            4,
            4 * d,
            &[(0, 0, 1.0), (1, 1, 1.0), (2, 2, 1.0)],
            &[0.0, 0.0, 0.0, 1.0],
        ),
        sparse(
            4,
            4 * d,
            &[(0, 0, 1.0), (1, 0, 2.0), (2, 0, 3.0), (3, 0, 4.0)],
            &[0.0; 4],
        ),
    ];
    let w = Weights {
        arch: Arch {
            width: d,
            rounds: 1,
        },
        layers,
    };
    assert_eq!(
        w.layers
            .iter()
            .map(|l| (l.rows, l.cols))
            .collect::<Vec<_>>(),
        w.arch.shapes()
    );
    let w = Weights::from_bytes(&w.to_bytes()).unwrap();
    let mut hidden = vec![0.0; HIDDEN_LEN];
    hidden[0] = 0.5;
    hidden[29] = 0.25;
    let mut out = Outputs::default();
    Forward::new().run(&w, &vec![0.0; PUBLIC_LEN], &hidden, 1, &mut out);
    let pooled = [21.5f32, 14.0, 7.5, 26.0, 18.0, 7.0, 6.0, 6.0];
    let close = |got: f32, want: f32, what: String| {
        assert!((got - want).abs() <= 1e-4, "{what}: {got} vs {want}");
    };
    for n in 0..NUM_NODES {
        close(out.policy[25 + n], 7.5, format!("settlement {n}"));
        close(out.policy[79 + n], 15.0, format!("city {n}"));
    }
    for e in 0..NUM_EDGES {
        close(out.policy[133 + e], 28.5, format!("road {e}"));
    }
    for t in 0..NUM_TILES {
        close(out.policy[205 + t], -5.75, format!("robber {t}"));
    }
    for j in 0..466 {
        let id = if j < 25 { j } else { j + 199 };
        close(
            out.policy[id],
            pooled[j % 8] + j as f32 / 1000.0,
            format!("slot {id}"),
        );
    }
    for (s, want) in [21.5, 0.5, 0.25, 1.0].into_iter().enumerate() {
        close(out.value[s], want, format!("value {s}"));
    }
    for (s, want) in [21.5, 43.0, 64.5, 86.0].into_iter().enumerate() {
        close(out.aux[s], want, format!("aux {s}"));
    }
}

/// Network v2 written the slow, obvious way, in f64, from the engine's topology (not
/// `settler_net::graph`): one position. Slot numbers are spelled out, not taken from `policy`.
fn reference(w: &Weights, public: &[f32], hidden: &[f32]) -> [Vec<f64>; 3] {
    let t = topo();
    let d = w.arch.width;
    let lin = |l: &Layer, x: &[f64]| -> Vec<f64> {
        (0..l.rows)
            .map(|o| {
                l.b[o] as f64
                    + (0..l.cols)
                        .map(|k| l.w[o * l.cols + k] as f64 * x[k])
                        .sum::<f64>()
            })
            .collect()
    };
    let relu = |v: Vec<f64>| -> Vec<f64> { v.into_iter().map(|x| x.max(0.0)).collect() };
    let plus =
        |a: &[f64], b: Vec<f64>| -> Vec<f64> { a.iter().zip(b).map(|(x, y)| x + y).collect() };
    let mean = |rows: &[Vec<f64>], idx: &[usize]| -> Vec<f64> {
        (0..d)
            .map(|c| idx.iter().map(|&i| rows[i][c]).sum::<f64>() / idx.len() as f64)
            .collect()
    };
    let ids = |v: &[u8]| -> Vec<usize> { v.iter().map(|&x| x as usize).collect() };
    let x: Vec<f64> = public.iter().map(|&v| v as f64).collect();
    let item = |base: usize, width: usize, i: usize| &x[base + i * width..base + (i + 1) * width];
    let mut node: Vec<Vec<f64>> = (0..NUM_NODES)
        .map(|n| relu(lin(&w.layers[NODE_ENCODER], item(NODE_BASE, NODE_WIDTH, n))))
        .collect();
    let mut edge: Vec<Vec<f64>> = (0..NUM_EDGES)
        .map(|e| relu(lin(&w.layers[EDGE_ENCODER], item(EDGE_BASE, EDGE_WIDTH, e))))
        .collect();
    let mut tile: Vec<Vec<f64>> = (0..NUM_TILES)
        .map(|i| relu(lin(&w.layers[TILE_ENCODER], item(TILE_BASE, TILE_WIDTH, i))))
        .collect();
    let mut global = relu(lin(&w.layers[GLOBAL_ENCODER], &x[OWN_HAND..]));
    let every = |k: usize| (0..k).collect::<Vec<usize>>();
    for r in 0..w.arch.rounds {
        let l = |which: usize| &w.layers[w.arch.round_layer(r, which)];
        tile = (0..NUM_TILES)
            .map(|i| {
                let input = [
                    tile[i].clone(),
                    mean(&node, &ids(&t.tile_nodes[i])),
                    global.clone(),
                ]
                .concat();
                plus(&tile[i], relu(lin(l(ROUND_TILE), &input)))
            })
            .collect();
        node = (0..NUM_NODES)
            .map(|n| {
                let input = [
                    node[n].clone(),
                    mean(&tile, &ids(&t.node_tiles[n])),
                    mean(&edge, &ids(&t.node_edges[n])),
                    mean(&node, &ids(&t.node_neighbors[n])),
                    global.clone(),
                ]
                .concat();
                plus(&node[n], relu(lin(l(ROUND_NODE), &input)))
            })
            .collect();
        edge = (0..NUM_EDGES)
            .map(|e| {
                let (p, q) = t.edge_nodes[e];
                let sum: Vec<f64> = (0..d)
                    .map(|c| node[p as usize][c] + node[q as usize][c])
                    .collect();
                let input = [edge[e].clone(), sum, global.clone()].concat();
                plus(&edge[e], relu(lin(l(ROUND_EDGE), &input)))
            })
            .collect();
        let input = [
            global.clone(),
            mean(&node, &every(NUM_NODES)),
            mean(&edge, &every(NUM_EDGES)),
            mean(&tile, &every(NUM_TILES)),
        ]
        .concat();
        global = plus(&global, relu(lin(l(ROUND_GLOBAL), &input)));
    }
    let head = |which: usize| &w.layers[w.arch.head_layer(which)];
    let pooled = [
        global.clone(),
        mean(&node, &every(NUM_NODES)),
        mean(&edge, &every(NUM_EDGES)),
        mean(&tile, &every(NUM_TILES)),
    ]
    .concat();
    let mut policy = vec![0.0; ACTION_SPACE_SIZE];
    for n in 0..NUM_NODES {
        let y = lin(
            head(NODE_POLICY),
            &[node[n].clone(), global.clone()].concat(),
        );
        policy[25 + n] = y[0];
        policy[79 + n] = y[1];
    }
    for e in 0..NUM_EDGES {
        policy[133 + e] = lin(
            head(EDGE_POLICY),
            &[edge[e].clone(), global.clone()].concat(),
        )[0];
    }
    for i in 0..NUM_TILES {
        policy[205 + i] = lin(
            head(TILE_POLICY),
            &[tile[i].clone(), global.clone()].concat(),
        )[0];
    }
    let other = lin(head(OTHER_POLICY), &relu(lin(head(OTHER_HIDDEN), &pooled)));
    for (j, v) in other.into_iter().enumerate() {
        policy[if j < 25 { j } else { j + 199 }] = v;
    }
    let mut input = pooled;
    input.extend(hidden.iter().map(|&v| v as f64));
    let v = relu(lin(head(VALUE_HIDDEN), &input));
    [policy, lin(head(VALUE), &v), lin(head(AUX), &v)]
}

fn inputs(n_every: usize) -> (usize, Vec<f32>, Vec<f32>) {
    let positions = random_positions(0..2, n_every, GameConfig::default());
    let n = positions.len();
    let (mut public, mut hidden) = (vec![0.0; n * PUBLIC_LEN], vec![0.0; n * HIDDEN_LEN]);
    for (i, s) in positions.iter().enumerate() {
        encode_state(
            s,
            s.current_actor(),
            &mut public[i * PUBLIC_LEN..(i + 1) * PUBLIC_LEN],
            &mut hidden[i * HIDDEN_LEN..(i + 1) * HIDDEN_LEN],
        );
    }
    (n, public, hidden)
}

/// d = 5 (not a multiple of the 8-lane dot product) and 2 rounds, on real positions.
#[test]
fn the_forward_pass_matches_a_plain_reference() {
    let w = Weights::random(
        Arch {
            width: 5,
            rounds: 2,
        },
        5,
    );
    let (n, public, hidden) = inputs(97);
    assert!(n > 20, "{n} positions");
    let mut out = Outputs::default();
    Forward::new().run(&w, &public, &hidden, n, &mut out);
    for i in 0..n {
        let [p, v, a] = reference(
            &w,
            &public[i * PUBLIC_LEN..(i + 1) * PUBLIC_LEN],
            &hidden[i * HIDDEN_LEN..(i + 1) * HIDDEN_LEN],
        );
        let got = [
            &out.policy[i * ACTION_SPACE_SIZE..(i + 1) * ACTION_SPACE_SIZE],
            &out.value[i * NUM_PLAYERS..(i + 1) * NUM_PLAYERS],
            &out.aux[i * NUM_PLAYERS..(i + 1) * NUM_PLAYERS],
        ];
        for (g, want) in got.iter().zip([p, v, a]) {
            for (x, y) in g.iter().zip(&want) {
                assert!(
                    (*x as f64 - y).abs() <= 1e-4 * y.abs().max(1.0),
                    "position {i}: {x} vs {y}"
                );
            }
        }
    }
}

#[test]
fn each_row_is_independent_of_the_rest_of_the_batch() {
    let w = Weights::random(Arch::DEFAULT, 9);
    let (n, public, hidden) = inputs(211);
    let (mut fwd, mut all, mut one) = (Forward::new(), Outputs::default(), Outputs::default());
    fwd.run(&w, &public, &hidden, n, &mut all);
    for i in 0..n {
        fwd.run(
            &w,
            &public[i * PUBLIC_LEN..(i + 1) * PUBLIC_LEN],
            &hidden[i * HIDDEN_LEN..(i + 1) * HIDDEN_LEN],
            1,
            &mut one,
        );
        assert_eq!(
            one.policy[..],
            all.policy[i * ACTION_SPACE_SIZE..(i + 1) * ACTION_SPACE_SIZE]
        );
        assert_eq!(
            one.value[..],
            all.value[i * NUM_PLAYERS..(i + 1) * NUM_PLAYERS]
        );
        assert_eq!(one.aux[..], all.aux[i * NUM_PLAYERS..(i + 1) * NUM_PLAYERS]);
    }
}

#[test]
fn softmax_sums_to_one_and_sends_minus_infinity_to_zero() {
    let mut x = [1.0f32, f32::NEG_INFINITY, 3.0, -2.0, f32::NEG_INFINITY];
    softmax(&mut x);
    assert!((x.iter().sum::<f32>() - 1.0).abs() < 1e-6, "{x:?}");
    assert_eq!((x[1], x[4]), (0.0, 0.0));
    let z = 1f32.exp() + 3f32.exp() + (-2f32).exp();
    assert!((x[2] - 3f32.exp() / z).abs() < 1e-6, "{x:?}");
    let mut big = [1000.0f32, 999.0];
    softmax(&mut big);
    assert!(
        big.iter().all(|v| v.is_finite()),
        "stable for large logits: {big:?}"
    );
    assert!((big[0] - 1.0 / (1.0 + (-1f32).exp())).abs() < 1e-6);
}

#[test]
fn an_encoder_v2_file_is_refused_with_the_way_to_upgrade_it() {
    let mut bytes = Weights::random(SMALL, 1).to_bytes();
    bytes[12..20].copy_from_slice(&V2_ENCODING_HASH.to_le_bytes());
    let e = err(&bytes);
    assert!(e.contains("encoding hash"), "{e}");
    assert!(
        e.contains("encoder v2") && e.contains("alphasettler net-upgrade"),
        "{e}"
    );
    assert_ne!(V2_ENCODING_HASH, ENCODING_HASH);
    assert_eq!(
        settler_net::graph::GLOBAL_WIDTH - V2_GLOBAL_WIDTH,
        24,
        "the trade block"
    );
    let mut other = Weights::random(SMALL, 1).to_bytes();
    other[12] ^= 1;
    assert!(
        !err(&other).contains("net-upgrade"),
        "only v2's hash gets the hint"
    );
}
