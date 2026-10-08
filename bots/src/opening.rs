//! The opening model's inputs (spec 3d-1 Section 4.2): 78 hand-built whole-board features of a
//! setup position for one seat `p`, seat-relative (r0 is `p`, then the others in turn order, as
//! the net's inputs name them). One function serves training (through the bindings) and play
//! (`opening_model::choose_settlement`), so the two cannot disagree.
//!
//! A spot's pips are the dice dots (`settler_net::encode::pips`) of the tiles it touches; a seat's
//! pips sum over its settlements, so a tile two of them touch counts twice (it pays twice).
//! "Legal" is the distance rule against every settlement placed. "Distance 2" is two steps along
//! the node graph, whoever owns the edges; nodes one step away are never legal.

use settler_engine::topology::{topo, NUM_NODES, NUM_TILES};
use settler_engine::{bits64, Board, PlayerId, State, NUM_PLAYERS, NUM_RESOURCES};
use settler_net::encode::{abs, layout_hash, pips};

pub const OPENING_FEATURES: usize = 78;

/// Block offsets in vector order. A per-seat block of width `w` holds r0 at the offset, r1 at
/// offset + w, and so on.
pub const SEAT_PIPS: usize = 0;
pub const SEAT_TOTAL: usize = SEAT_PIPS + NUM_PLAYERS * NUM_RESOURCES;
pub const SEAT_RESOURCES: usize = SEAT_TOTAL + NUM_PLAYERS;
pub const SEAT_NUMBERS: usize = SEAT_RESOURCES + NUM_PLAYERS;
/// Per seat: 3:1, then the 2:1 ports for wood brick sheep wheat ore.
pub const SEAT_PORTS: usize = SEAT_NUMBERS + NUM_PLAYERS;
pub const PORT_KINDS: usize = 1 + NUM_RESOURCES;
pub const SEAT_PLACED: usize = SEAT_PORTS + NUM_PLAYERS * PORT_KINDS;
pub const SEAT_REACH: usize = SEAT_PLACED + NUM_PLAYERS;
pub const PLACED_ALL: usize = SEAT_REACH + NUM_PLAYERS;
pub const PICK_LEFT: usize = PLACED_ALL + 1;
pub const BEST_RESOURCE: usize = PICK_LEFT + 1;
pub const BEST_SPOT: usize = BEST_RESOURCE + NUM_RESOURCES;
pub const RICH_SPOTS: usize = BEST_SPOT + 1;
pub const BOARD_RESOURCE: usize = RICH_SPOTS + 1;
const _: () = assert!(BOARD_RESOURCE + NUM_RESOURCES == OPENING_FEATURES);

/// Pips a legal spot needs to count as rich.
pub const RICH_PIPS: u32 = 10;

/// The feature list as canonical text: each `NAME xN: tokens` line is a block of N items (N = 4:
/// one per relative seat) with one token per feature (`name/scale` for a count over that scale).
/// `OPENING_HASH` hashes it, so renaming, reordering or rescaling a feature changes the hash and
/// a model trained on the old list is refused. Keep it in step with `opening_features`.
pub const OPENING_LAYOUT: &str = "\
settler-bots opening features v1
seats relative to p: r0 is p, then r1 r2 r3 in turn order; a seat block holds r0 r1 r2 r3
resources: wood brick sheep wheat ore
pips: dice dots of the tiles a spot touches, summed over a seat's settlements
legal: the distance rule against every settlement placed; distance 2 is two steps on the node graph
seat_pips x4: wood/15 brick/15 sheep/15 wheat/15 ore/15
seat_total x4: pips/30
seat_resources x4: touched/5
seat_numbers x4: distinct/6
seat_ports x4: generic wood brick sheep wheat ore
seat_placed x4: settlements/2
seat_reach x4: best_legal_at_distance_2/15
placed_all x1: settlements/8
pick_left x1: p
best_legal x1: wood/5 brick/5 sheep/5 wheat/5 ore/5
best_legal_spot x1: pips/15
rich_legal_spots x1: at_least_10/10
board x1: wood/25 brick/25 sheep/25 wheat/25 ore/25
";

/// Written into every opening model file; a file with another value was trained on other features.
pub const OPENING_HASH: u64 = layout_hash(&[OPENING_FEATURES as u32], OPENING_LAYOUT);

/// Every feature's name in vector order: `block[i].token`.
pub fn opening_feature_names() -> Vec<String> {
    let mut names = Vec::new();
    for line in OPENING_LAYOUT.lines() {
        let Some((head, tokens)) = line.split_once(": ") else {
            continue;
        };
        let Some((block, count)) = head.split_once(" x") else {
            continue;
        };
        let Ok(count) = count.parse::<usize>() else {
            continue;
        };
        if block.contains(' ') {
            continue;
        }
        for i in 0..count {
            for token in tokens.split(' ') {
                names.push(format!("{block}[{i}].{token}"));
            }
        }
    }
    names
}

/// Node `n`'s pips of each resource.
pub fn spot_pips(board: &Board, n: usize) -> [u32; NUM_RESOURCES] {
    let mut out = [0; NUM_RESOURCES];
    for &tile in &topo().node_tiles[n] {
        if let Some(r) = board.tile_resource[tile as usize] {
            out[r.index()] += pips(board.tile_number[tile as usize]);
        }
    }
    out
}

/// Nodes where a settlement may still go: empty, with no settlement on a neighbour.
pub fn legal_spots(settlements: &[u64; NUM_PLAYERS]) -> u64 {
    let t = topo();
    let occupied = settlements.iter().fold(0u64, |m, s| m | s);
    let blocked = bits64(occupied).fold(occupied, |m, n| m | t.node_neighbor_mask[n as usize]);
    ((1u64 << NUM_NODES) - 1) & !blocked
}

fn scaled(x: u32, den: u32) -> f32 {
    x as f32 / den as f32
}

fn flag(b: bool) -> f32 {
    b as u8 as f32
}

/// The 78 opening features for seat `p` (spec 3d-1 Section 4.2) of a position whose settlements
/// are `settlements` (by absolute seat). `p` is the seat that just picked, or would pick.
pub fn opening_features(
    board: &Board,
    settlements: &[u64; NUM_PLAYERS],
    p: PlayerId,
) -> [f32; OPENING_FEATURES] {
    let t = topo();
    let mut out = [0.0; OPENING_FEATURES];
    let legal = legal_spots(settlements);
    let spots: Vec<[u32; NUM_RESOURCES]> = (0..NUM_NODES).map(|n| spot_pips(board, n)).collect();
    let total = |n: usize| spots[n].iter().sum::<u32>();
    for r in 0..NUM_PLAYERS {
        let mine = settlements[abs(r, p)];
        let mut res = [0u32; NUM_RESOURCES];
        let mut touched = [false; NUM_RESOURCES];
        let mut numbers = 0u16;
        let mut two_steps = 0u64;
        for n in bits64(mine) {
            let n = n as usize;
            for &tile in &t.node_tiles[n] {
                if let Some(x) = board.tile_resource[tile as usize] {
                    let number = board.tile_number[tile as usize];
                    res[x.index()] += pips(number);
                    touched[x.index()] = true;
                    numbers |= 1u16 << number;
                }
            }
            for &m in &t.node_neighbors[n] {
                two_steps |= t.node_neighbor_mask[m as usize];
            }
        }
        for k in 0..NUM_RESOURCES {
            out[SEAT_PIPS + NUM_RESOURCES * r + k] = scaled(res[k], 15);
        }
        out[SEAT_TOTAL + r] = scaled(res.iter().sum(), 30);
        out[SEAT_RESOURCES + r] = scaled(touched.iter().filter(|&&x| x).count() as u32, 5);
        out[SEAT_NUMBERS + r] = scaled(numbers.count_ones(), 6);
        out[SEAT_PORTS + PORT_KINDS * r] = flag(mine & board.generic_port_nodes != 0);
        for k in 0..NUM_RESOURCES {
            out[SEAT_PORTS + PORT_KINDS * r + 1 + k] =
                flag(mine & board.specific_port_nodes[k] != 0);
        }
        out[SEAT_PLACED + r] = scaled(mine.count_ones(), 2);
        let reach = bits64(two_steps & legal)
            .map(|n| total(n as usize))
            .max()
            .unwrap_or(0);
        out[SEAT_REACH + r] = scaled(reach, 15);
    }
    let placed: u32 = settlements.iter().map(|s| s.count_ones()).sum();
    out[PLACED_ALL] = scaled(placed, 8);
    out[PICK_LEFT] = flag(settlements[p as usize].count_ones() < 2);
    let (mut best, mut best_spot, mut rich) = ([0u32; NUM_RESOURCES], 0u32, 0u32);
    for n in bits64(legal) {
        let n = n as usize;
        for k in 0..NUM_RESOURCES {
            best[k] = best[k].max(spots[n][k]);
        }
        best_spot = best_spot.max(total(n));
        rich += (total(n) >= RICH_PIPS) as u32;
    }
    for k in 0..NUM_RESOURCES {
        out[BEST_RESOURCE + k] = scaled(best[k], 5);
    }
    out[BEST_SPOT] = scaled(best_spot, 15);
    out[RICH_SPOTS] = scaled(rich, 10);
    let mut board_res = [0u32; NUM_RESOURCES];
    for tile in 0..NUM_TILES {
        if let Some(x) = board.tile_resource[tile] {
            board_res[x.index()] += pips(board.tile_number[tile]);
        }
    }
    for k in 0..NUM_RESOURCES {
        out[BOARD_RESOURCE + k] = scaled(board_res[k], 25);
    }
    out
}

/// `opening_features` of a game state (its board and every seat's settlements).
pub fn state_features(s: &State, p: PlayerId) -> [f32; OPENING_FEATURES] {
    let settlements = std::array::from_fn(|q| s.players[q].settlements);
    opening_features(&s.board, &settlements, p)
}
