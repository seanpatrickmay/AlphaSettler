//! The per-game board: terrain, number tokens, and ports.

use crate::rng::Rng;
use crate::topology::{topo, NUM_TILES};
use crate::types::Resource;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PortKind {
    /// 3:1 any resource.
    Generic,
    /// 2:1 for one resource.
    Specific(Resource),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Board {
    /// `None` is the desert.
    pub tile_resource: [Option<Resource>; NUM_TILES],
    /// 0 on the desert.
    pub tile_number: [u8; NUM_TILES],
    /// (edge id, kind) for each of the 9 ports.
    pub ports: [(u8, PortKind); 9],
    pub generic_port_nodes: u64,
    pub specific_port_nodes: [u64; 5],
}

use Resource::*;

pub const STANDARD_TERRAIN: [Option<Resource>; NUM_TILES] = [
    Some(Wood),
    Some(Wood),
    Some(Wood),
    Some(Wood),
    Some(Brick),
    Some(Brick),
    Some(Brick),
    Some(Sheep),
    Some(Sheep),
    Some(Sheep),
    Some(Sheep),
    Some(Wheat),
    Some(Wheat),
    Some(Wheat),
    Some(Wheat),
    Some(Ore),
    Some(Ore),
    Some(Ore),
    None,
];

pub const STANDARD_NUMBERS: [u8; 18] = [2, 3, 3, 4, 4, 5, 5, 6, 6, 8, 8, 9, 9, 10, 10, 11, 11, 12];

pub const STANDARD_PORT_KINDS: [PortKind; 9] = [
    PortKind::Generic,
    PortKind::Generic,
    PortKind::Generic,
    PortKind::Generic,
    PortKind::Specific(Wood),
    PortKind::Specific(Brick),
    PortKind::Specific(Sheep),
    PortKind::Specific(Wheat),
    PortKind::Specific(Ore),
];

/// Positions in `Topology::coastal_edges` that hold ports (gaps of 3, 3, 4 around the coast).
pub const PORT_SLOTS: [usize; 9] = [0, 3, 6, 10, 13, 16, 20, 23, 26];

impl Board {
    pub fn new(
        tile_resource: [Option<Resource>; NUM_TILES],
        tile_number: [u8; NUM_TILES],
        ports: [(u8, PortKind); 9],
    ) -> Board {
        let t = topo();
        let mut generic = 0u64;
        let mut specific = [0u64; 5];
        for &(e, kind) in &ports {
            let (a, b) = t.edge_nodes[e as usize];
            let m = (1u64 << a) | (1u64 << b);
            match kind {
                PortKind::Generic => generic |= m,
                PortKind::Specific(r) => specific[r.index()] |= m,
            }
        }
        Board {
            tile_resource,
            tile_number,
            ports,
            generic_port_nodes: generic,
            specific_port_nodes: specific,
        }
    }

    /// Random standard board: shuffled terrain, tokens with no adjacent 6/8, shuffled port kinds.
    pub fn random(rng: &mut Rng) -> Board {
        let t = topo();
        let mut terrain = STANDARD_TERRAIN;
        rng.shuffle(&mut terrain);
        let mut numbers = STANDARD_NUMBERS;
        let tile_number = loop {
            rng.shuffle(&mut numbers);
            let mut tn = [0u8; NUM_TILES];
            let mut k = 0;
            for i in 0..NUM_TILES {
                if terrain[i].is_some() {
                    tn[i] = numbers[k];
                    k += 1;
                }
            }
            if !red_adjacent(&tn) {
                break tn;
            }
        };
        let mut kinds = STANDARD_PORT_KINDS;
        rng.shuffle(&mut kinds);
        let mut ports = [(0u8, PortKind::Generic); 9];
        for i in 0..9 {
            ports[i] = (t.coastal_edges[PORT_SLOTS[i]], kinds[i]);
        }
        Board::new(terrain, tile_number, ports)
    }

    /// Checks a board built outside `Board::random`, e.g. imported from another engine.
    pub fn validate(&self) -> Result<(), String> {
        let deserts = self.tile_resource.iter().filter(|r| r.is_none()).count();
        if deserts != 1 {
            return Err(format!(
                "board must have exactly one desert, found {deserts}"
            ));
        }
        for t in 0..NUM_TILES {
            let n = self.tile_number[t];
            match self.tile_resource[t] {
                None if n != 0 => return Err(format!("desert tile {t} has number {n}")),
                Some(_) if !(2..=12).contains(&n) || n == 7 => {
                    return Err(format!("tile {t} has number {n}"))
                }
                _ => {}
            }
        }
        let coastal = &topo().coastal_edges;
        for (i, &(e, _)) in self.ports.iter().enumerate() {
            if !coastal.contains(&e) {
                return Err(format!(
                    "port {i} is on edge {e}, which is not on the coast"
                ));
            }
            if self.ports[..i].iter().any(|&(f, _)| f == e) {
                return Err(format!("two ports share edge {e}"));
            }
        }
        if *self != Board::new(self.tile_resource, self.tile_number, self.ports) {
            return Err("port node masks do not match the ports".into());
        }
        Ok(())
    }

    pub fn desert(&self) -> u8 {
        self.tile_resource
            .iter()
            .position(|r| r.is_none())
            .expect("board has no desert") as u8
    }

    /// Cards of `r` needed for one bank card, given the player's building nodes.
    #[inline]
    pub fn maritime_rate(&self, buildings: u64, r: Resource) -> u8 {
        if buildings & self.specific_port_nodes[r.index()] != 0 {
            2
        } else if buildings & self.generic_port_nodes != 0 {
            3
        } else {
            4
        }
    }
}

fn red_adjacent(tile_number: &[u8; NUM_TILES]) -> bool {
    let t = topo();
    let red = |n: u8| n == 6 || n == 8;
    (0..NUM_TILES).any(|i| {
        red(tile_number[i])
            && t.tile_neighbors[i]
                .iter()
                .any(|&j| red(tile_number[j as usize]))
    })
}
