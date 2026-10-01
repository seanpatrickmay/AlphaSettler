#![allow(dead_code)]

use settler_engine::apply::{Chance, NoEvents};
use settler_engine::topology::topo;
use settler_engine::*;

pub fn cfg() -> GameConfig {
    GameConfig::default()
}

/// A fresh game skipped past setup: no buildings, empty hands, player 0 to act in `phase`.
pub fn blank(seed: u64, phase: Phase) -> State {
    blank_with(seed, phase, cfg())
}

pub fn blank_with(seed: u64, phase: Phase, config: GameConfig) -> State {
    let mut s = State::new(seed, config);
    s.setup_step = 8;
    s.current = 0;
    s.phase = phase;
    s
}

/// Play setup by always taking the first legal action.
pub fn after_setup(seed: u64) -> State {
    let mut s = State::new(seed, cfg());
    while matches!(s.phase, Phase::SetupSettlement | Phase::SetupRoad { .. }) {
        let a = s.legal_actions()[0];
        s.apply(a);
    }
    s
}

/// Move cards from the bank to player `p` (keeps resource totals conserved).
pub fn give(s: &mut State, p: usize, h: Hand) {
    hand_sub(&mut s.bank, &h);
    hand_add(&mut s.players[p].hand, &h);
}

pub fn dice_for_sum(sum: u8) -> (u8, u8) {
    let a = (sum - 1).min(6);
    (a, sum - a)
}

/// Roll a specific total.
pub fn roll(s: &mut State, sum: u8) {
    s.apply_with(
        Action::Roll,
        Some(Chance::Roll { dice: dice_for_sum(sum), discards: None }),
        &mut NoEvents,
    );
}

pub fn mask64(xs: &[u8]) -> u64 {
    xs.iter().fold(0, |m, &x| m | (1u64 << x))
}

pub fn mask128(xs: &[u8]) -> u128 {
    xs.iter().fold(0, |m, &x| m | (1u128 << x))
}

pub fn edge_between(a: u8, b: u8) -> u8 {
    let t = topo();
    *t.node_edges[a as usize]
        .iter()
        .find(|&&e| {
            let (x, y) = t.edge_nodes[e as usize];
            (x == a && y == b) || (x == b && y == a)
        })
        .expect("nodes are not adjacent")
}

/// A simple path of `len` edges starting at `start`, avoiding nodes in `avoid`. Returns its nodes.
pub fn path_nodes(start: u8, len: usize, avoid: u64) -> Vec<u8> {
    fn go(path: &mut Vec<u8>, len: usize, avoid: u64) -> bool {
        if path.len() == len + 1 {
            return true;
        }
        let last = *path.last().unwrap();
        for &nb in &topo().node_neighbors[last as usize] {
            if !path.contains(&nb) && avoid & (1u64 << nb) == 0 {
                path.push(nb);
                if go(path, len, avoid) {
                    return true;
                }
                path.pop();
            }
        }
        false
    }
    let mut path = vec![start];
    assert!(go(&mut path, len, avoid), "no path of length {len} from {start}");
    path
}

pub fn path_edges(nodes: &[u8]) -> Vec<u8> {
    nodes.windows(2).map(|w| edge_between(w[0], w[1])).collect()
}

/// Nodes of tile `t` that touch no other tile.
pub fn exclusive_nodes(t: usize) -> Vec<u8> {
    topo().tile_nodes[t]
        .iter()
        .copied()
        .filter(|&n| topo().node_tiles[n as usize].len() == 1)
        .collect()
}

/// A non-desert tile with at least two nodes that touch only it.
pub fn lonely_tile(s: &State) -> usize {
    (0..19)
        .find(|&t| s.board.tile_resource[t].is_some() && exclusive_nodes(t).len() >= 2)
        .unwrap()
}

pub fn sorted(mut v: Vec<Action>) -> Vec<Action> {
    v.sort_by_key(|a| a.encode());
    v
}
