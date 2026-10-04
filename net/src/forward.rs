//! The batched forward pass of network v2 (f32), the only inference code (spec Section 2; the
//! layer order is in `weights.rs`). Each per-entity map runs as one matrix product over every
//! entity of its type in the batch: its rows are the batch's nodes (or edges, or tiles), its
//! columns the layer's inputs side by side. Every row's arithmetic is the same whatever else is in
//! the batch. `net/tests/weights.rs` checks it against a hand-computed tiny network and a plain
//! f64 reference, and `tests/python/test_train.py` against the PyTorch twin
//! (`alphasettler/train/model.py`).

use crate::encode::{
    EDGE_BASE, EDGE_WIDTH, HIDDEN_LEN, NODE_BASE, NODE_WIDTH, PUBLIC_LEN, TILE_BASE, TILE_WIDTH,
};
use crate::graph::{graph, Links, GLOBAL_BASE, GLOBAL_WIDTH};
use crate::policy::{
    other_slot, CITY_BASE, OTHER_POLICY_LEN, ROAD_BASE, ROBBER_BASE, SETTLEMENT_BASE,
};
use crate::weights::{
    Layer, Weights, AUX, EDGE_ENCODER, EDGE_POLICY, GLOBAL_ENCODER, NODE_ENCODER, NODE_POLICY,
    OTHER_HIDDEN, OTHER_POLICY, ROUND_EDGE, ROUND_GLOBAL, ROUND_NODE, ROUND_TILE, TILE_ENCODER,
    TILE_POLICY, VALUE, VALUE_HIDDEN,
};
use settler_engine::topology::{NUM_EDGES, NUM_NODES, NUM_TILES};
use settler_engine::{ACTION_SPACE_SIZE, NUM_PLAYERS};

/// Raw head outputs for `n` positions, row-major: 665 policy logits, 4 seat-relative value logits
/// (before the softmax) and 4 aux outputs (final VP / 10, seat-relative) per position.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Outputs {
    pub n: usize,
    pub policy: Vec<f32>,
    pub value: Vec<f32>,
    pub aux: Vec<f32>,
}

/// Activation buffers reused across calls: the entity states (`n × count × d`), the pooled
/// readout (`n × 4d`), and scratch for aggregates, concatenated inputs and layer outputs.
#[derive(Clone, Debug, Default)]
pub struct Forward {
    node: Vec<f32>,
    edge: Vec<f32>,
    tile: Vec<f32>,
    global: Vec<f32>,
    pooled: Vec<f32>,
    a: Vec<f32>,
    b: Vec<f32>,
    c: Vec<f32>,
    cat: Vec<f32>,
    y: Vec<f32>,
}

/// Eight independent partial sums so the compiler can vectorise, combined in a fixed order: a
/// row's result never depends on the rest of the batch.
#[inline]
pub fn dot(x: &[f32], y: &[f32]) -> f32 {
    debug_assert_eq!(x.len(), y.len());
    let (xc, yc) = (x.chunks_exact(8), y.chunks_exact(8));
    let (xr, yr) = (xc.remainder(), yc.remainder());
    let mut acc = [0.0f32; 8];
    for (a, b) in xc.zip(yc) {
        for j in 0..8 {
            acc[j] += a[j] * b[j];
        }
    }
    let mut tail = 0.0;
    for (a, b) in xr.iter().zip(yr) {
        tail += a * b;
    }
    ((acc[0] + acc[1]) + (acc[2] + acc[3])) + ((acc[4] + acc[5]) + (acc[6] + acc[7])) + tail
}

/// In place, numerically stable; entries of -inf become 0. Requires at least one finite entry and
/// none of +inf: otherwise the result is NaN (`evaluator::masked_prior` handles that case itself).
pub fn softmax(x: &mut [f32]) {
    let m = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    debug_assert!(m.is_finite(), "softmax needs a finite maximum, got {m}");
    let mut total = 0.0;
    for v in x.iter_mut() {
        *v = (*v - m).exp();
        total += *v;
    }
    for v in x.iter_mut() {
        *v /= total;
    }
}

/// `y[i][o] = b[o] + w[o] · x[i]` for the `n` rows of `x`, optionally through a ReLU. Row by row,
/// so a small layer's weights stay in cache across all the rows of an entity type.
fn dense(l: &Layer, x: &[f32], n: usize, relu: bool, y: &mut Vec<f32>) {
    debug_assert_eq!(x.len(), n * l.cols);
    y.clear();
    y.resize(n * l.rows, 0.0);
    for (xi, yi) in x.chunks_exact(l.cols).zip(y.chunks_exact_mut(l.rows)) {
        for (o, out) in yi.iter_mut().enumerate() {
            let s = l.b[o] + dot(&l.w[o * l.cols..(o + 1) * l.cols], xi);
            *out = if relu { s.max(0.0) } else { s };
        }
    }
}

/// x ← x + ReLU(layer(input)) for `n` rows: a round's residual update.
fn residual(l: &Layer, input: &[f32], n: usize, y: &mut Vec<f32>, x: &mut [f32]) {
    dense(l, input, n, true, y);
    for (v, d) in x.iter_mut().zip(y.iter()) {
        *v += *d;
    }
}

/// Each position's `count` items of `width` floats at `base` of its public block, as `n × count`
/// rows.
fn block(public: &[f32], n: usize, base: usize, count: usize, width: usize, out: &mut Vec<f32>) {
    out.clear();
    for i in 0..n {
        let at = i * PUBLIC_LEN + base;
        out.extend_from_slice(&public[at..at + count * width]);
    }
}

/// For each position and each entity `j` of `links`: `scale[j]` × the sum, in link order, of the
/// `d`-float rows of `src` (`src_count` rows per position) it links to. `n × links.count()` rows.
fn aggregate(links: &Links, src: &[f32], src_count: usize, n: usize, d: usize, out: &mut Vec<f32>) {
    let count = links.count();
    out.clear();
    out.resize(n * count * d, 0.0);
    for i in 0..n {
        let s = &src[i * src_count * d..(i + 1) * src_count * d];
        for j in 0..count {
            let o = &mut out[(i * count + j) * d..(i * count + j + 1) * d];
            for &k in links.of(j) {
                let k = k as usize;
                for (v, x) in o.iter_mut().zip(&s[k * d..(k + 1) * d]) {
                    *v += *x;
                }
            }
            let scale = links.scale[j];
            for v in o.iter_mut() {
                *v *= scale;
            }
        }
    }
}

/// The mean of each position's `count` rows of `src`: `n` rows of `d`.
fn mean_rows(src: &[f32], count: usize, n: usize, d: usize, out: &mut Vec<f32>) {
    out.clear();
    out.resize(n * d, 0.0);
    for (o, rows) in out.chunks_exact_mut(d).zip(src.chunks_exact(count * d)) {
        for r in rows.chunks_exact(d) {
            for (v, x) in o.iter_mut().zip(r) {
                *v += *x;
            }
        }
        for v in o.iter_mut() {
            *v /= count as f32;
        }
    }
}

/// One input block of a layer: a row per entity, or a row per position shared by its entities.
enum Part<'a> {
    Each(&'a [f32]),
    Shared(&'a [f32]),
}

/// For each position and each of its `count` entities, the parts' `d`-float rows side by side.
fn concat(parts: &[Part<'_>], n: usize, count: usize, d: usize, out: &mut Vec<f32>) {
    out.clear();
    for i in 0..n {
        for j in 0..count {
            for p in parts {
                out.extend_from_slice(match p {
                    Part::Each(x) => &x[(i * count + j) * d..(i * count + j + 1) * d],
                    Part::Shared(x) => &x[i * d..(i + 1) * d],
                });
            }
        }
    }
}

impl Forward {
    pub fn new() -> Forward {
        Forward::default()
    }

    /// Run `n` positions: `public` holds `n × PUBLIC_LEN` floats and `hidden` `n × HIDDEN_LEN`.
    pub fn run(
        &mut self,
        w: &Weights,
        public: &[f32],
        hidden: &[f32],
        n: usize,
        out: &mut Outputs,
    ) {
        use Part::{Each, Shared};
        assert_eq!(public.len(), n * PUBLIC_LEN, "public block length");
        assert_eq!(hidden.len(), n * HIDDEN_LEN, "hidden block length");
        let (arch, g) = (w.arch, graph());
        let d = arch.width;
        let layer = |i: usize| &w.layers[i];
        let Forward {
            node,
            edge,
            tile,
            global,
            pooled,
            a,
            b,
            c,
            cat,
            y,
        } = self;

        block(public, n, NODE_BASE, NUM_NODES, NODE_WIDTH, cat);
        dense(layer(NODE_ENCODER), cat, n * NUM_NODES, true, node);
        block(public, n, EDGE_BASE, NUM_EDGES, EDGE_WIDTH, cat);
        dense(layer(EDGE_ENCODER), cat, n * NUM_EDGES, true, edge);
        block(public, n, TILE_BASE, NUM_TILES, TILE_WIDTH, cat);
        dense(layer(TILE_ENCODER), cat, n * NUM_TILES, true, tile);
        block(public, n, GLOBAL_BASE, 1, GLOBAL_WIDTH, cat);
        dense(layer(GLOBAL_ENCODER), cat, n, true, global);

        for r in 0..arch.rounds {
            // tile ← [tile, mean of its nodes, global]
            aggregate(&g.tile_nodes, node, NUM_NODES, n, d, a);
            concat(&[Each(tile), Each(a), Shared(global)], n, NUM_TILES, d, cat);
            residual(
                layer(arch.round_layer(r, ROUND_TILE)),
                cat,
                n * NUM_TILES,
                y,
                tile,
            );
            // node ← [node, mean of its tiles, mean of its edges, mean of its neighbours, global]
            aggregate(&g.node_tiles, tile, NUM_TILES, n, d, a);
            aggregate(&g.node_edges, edge, NUM_EDGES, n, d, b);
            aggregate(&g.node_neighbours, node, NUM_NODES, n, d, c);
            concat(
                &[Each(node), Each(a), Each(b), Each(c), Shared(global)],
                n,
                NUM_NODES,
                d,
                cat,
            );
            residual(
                layer(arch.round_layer(r, ROUND_NODE)),
                cat,
                n * NUM_NODES,
                y,
                node,
            );
            // edge ← [edge, sum of its two nodes, global]
            aggregate(&g.edge_nodes, node, NUM_NODES, n, d, a);
            concat(&[Each(edge), Each(a), Shared(global)], n, NUM_EDGES, d, cat);
            residual(
                layer(arch.round_layer(r, ROUND_EDGE)),
                cat,
                n * NUM_EDGES,
                y,
                edge,
            );
            // global ← [global, mean of the nodes, mean of the edges, mean of the tiles]
            mean_rows(node, NUM_NODES, n, d, a);
            mean_rows(edge, NUM_EDGES, n, d, b);
            mean_rows(tile, NUM_TILES, n, d, c);
            concat(&[Each(global), Each(a), Each(b), Each(c)], n, 1, d, cat);
            residual(layer(arch.round_layer(r, ROUND_GLOBAL)), cat, n, y, global);
        }

        // pooled = [global, mean of the nodes, mean of the edges, mean of the tiles]
        mean_rows(node, NUM_NODES, n, d, a);
        mean_rows(edge, NUM_EDGES, n, d, b);
        mean_rows(tile, NUM_TILES, n, d, c);
        concat(&[Each(global), Each(a), Each(b), Each(c)], n, 1, d, pooled);

        out.policy.clear();
        out.policy.resize(n * ACTION_SPACE_SIZE, 0.0);
        concat(&[Each(node), Shared(global)], n, NUM_NODES, d, cat);
        dense(
            layer(arch.head_layer(NODE_POLICY)),
            cat,
            n * NUM_NODES,
            false,
            y,
        );
        for (p, logits) in out
            .policy
            .chunks_exact_mut(ACTION_SPACE_SIZE)
            .zip(y.chunks_exact(2 * NUM_NODES))
        {
            for (k, pair) in logits.chunks_exact(2).enumerate() {
                p[SETTLEMENT_BASE + k] = pair[0];
                p[CITY_BASE + k] = pair[1];
            }
        }
        concat(&[Each(edge), Shared(global)], n, NUM_EDGES, d, cat);
        dense(
            layer(arch.head_layer(EDGE_POLICY)),
            cat,
            n * NUM_EDGES,
            false,
            y,
        );
        for (p, logits) in out
            .policy
            .chunks_exact_mut(ACTION_SPACE_SIZE)
            .zip(y.chunks_exact(NUM_EDGES))
        {
            p[ROAD_BASE..ROAD_BASE + NUM_EDGES].copy_from_slice(logits);
        }
        concat(&[Each(tile), Shared(global)], n, NUM_TILES, d, cat);
        dense(
            layer(arch.head_layer(TILE_POLICY)),
            cat,
            n * NUM_TILES,
            false,
            y,
        );
        for (p, logits) in out
            .policy
            .chunks_exact_mut(ACTION_SPACE_SIZE)
            .zip(y.chunks_exact(NUM_TILES))
        {
            p[ROBBER_BASE..ROBBER_BASE + NUM_TILES].copy_from_slice(logits);
        }
        dense(layer(arch.head_layer(OTHER_HIDDEN)), pooled, n, true, a);
        dense(layer(arch.head_layer(OTHER_POLICY)), a, n, false, y);
        for (p, logits) in out
            .policy
            .chunks_exact_mut(ACTION_SPACE_SIZE)
            .zip(y.chunks_exact(OTHER_POLICY_LEN))
        {
            for (j, &v) in logits.iter().enumerate() {
                p[other_slot(j)] = v;
            }
        }

        // value and aux from [pooled, hidden]
        cat.clear();
        for (p, h) in pooled
            .chunks_exact(4 * d)
            .zip(hidden.chunks_exact(HIDDEN_LEN))
        {
            cat.extend_from_slice(p);
            cat.extend_from_slice(h);
        }
        dense(layer(arch.head_layer(VALUE_HIDDEN)), cat, n, true, a);
        dense(layer(arch.head_layer(VALUE)), a, n, false, &mut out.value);
        dense(layer(arch.head_layer(AUX)), a, n, false, &mut out.aux);
        debug_assert_eq!(out.value.len(), n * NUM_PLAYERS);
        out.n = n;
    }
}
