//! Structural consistency checks, for states built outside normal play (snapshots, tests).

use crate::rules::awards::longest_road;
use crate::rules::build::has_free_road;
use crate::rules::robber::victims;
use crate::state::{Phase, State, SETUP_ORDER};
use crate::topology::{topo, NUM_EDGES, NUM_NODES, NUM_TILES};
use crate::types::*;

impl State {
    /// Every structural invariant normal play maintains. Award ownership is not checked: an
    /// imported position may follow another engine's award rules.
    pub fn check_invariants(&self) -> Result<(), String> {
        self.config.validate()?;
        self.board.validate()?;
        for r in 0..NUM_RESOURCES {
            let held: u32 = self.players.iter().map(|p| p.hand[r] as u32).sum();
            let total = held + self.bank[r] as u32;
            if total != BANK_PER_RESOURCE as u32 {
                return Err(format!(
                    "resource {r}: hands and bank hold {total} cards, expected {BANK_PER_RESOURCE}"
                ));
            }
        }
        let t = topo();
        let mut nodes = 0u64;
        let mut edges = 0u128;
        for (i, p) in self.players.iter().enumerate() {
            if (p.settlements | p.cities) >> NUM_NODES != 0 {
                return Err(format!(
                    "player {i} settlements/cities use node ids beyond {}",
                    NUM_NODES - 1
                ));
            }
            if p.roads >> NUM_EDGES != 0 {
                return Err(format!(
                    "player {i} roads use edge ids beyond {}",
                    NUM_EDGES - 1
                ));
            }
            for (piece, n, max) in [
                ("settlements", p.settlements.count_ones(), MAX_SETTLEMENTS),
                ("cities", p.cities.count_ones(), MAX_CITIES),
                ("roads", p.roads.count_ones(), MAX_ROADS),
            ] {
                if n > max {
                    return Err(format!("player {i} has {n} {piece}, the limit is {max}"));
                }
            }
            if p.settlements & p.cities != 0 {
                return Err(format!(
                    "player {i} has a settlement and a city on one node"
                ));
            }
            let b = p.settlements | p.cities;
            if nodes & b != 0 {
                return Err("two players share a node".into());
            }
            nodes |= b;
            if edges & p.roads != 0 {
                return Err("two players share an edge".into());
            }
            edges |= p.roads;
            for c in 0..5 {
                if p.dev_new[c] > p.dev_hand[c] {
                    return Err(format!(
                        "player {i} bought more {:?} cards than they hold",
                        DevCard::from_index(c)
                    ));
                }
            }
            if p.knights_played != p.dev_played[DevCard::Knight.index()] {
                return Err(format!(
                    "player {i}: knights_played disagrees with dev_played"
                ));
            }
            if p.dev_played[DevCard::VictoryPoint.index()] != 0 {
                return Err(format!("player {i}: victory point cards are never played"));
            }
        }
        for n in bits64(nodes) {
            if nodes & t.node_neighbor_mask[n as usize] != 0 {
                return Err(format!("distance rule broken at node {n}"));
            }
        }
        let left = &self.dev_deck[self.dev_deck_pos as usize..];
        for (k, &count) in DEV_DECK_COUNTS.iter().enumerate() {
            let out: u32 = self
                .players
                .iter()
                .map(|p| p.dev_hand[k] as u32 + p.dev_played[k] as u32)
                .sum();
            let in_deck = left.iter().filter(|c| c.index() == k).count() as u32;
            if out + in_deck != count as u32 {
                return Err(format!(
                    "{:?}: {out} held or played + {in_deck} in the deck, expected {count}",
                    DevCard::from_index(k)
                ));
            }
        }
        if self.robber as usize >= NUM_TILES {
            return Err(format!(
                "robber on tile {}, beyond {}",
                self.robber,
                NUM_TILES - 1
            ));
        }
        if self.current as usize >= NUM_PLAYERS {
            return Err(format!("current player {} out of range", self.current));
        }
        for (i, p) in self.players.iter().enumerate() {
            if i != self.current as usize && p.dev_new != [0; 5] {
                return Err(format!(
                    "player {i} holds cards bought this turn, but it is player {}'s turn",
                    self.current
                ));
            }
        }
        for owner in [self.longest_road_owner, self.largest_army_owner]
            .into_iter()
            .flatten()
        {
            if owner as usize >= NUM_PLAYERS {
                return Err(format!("award owner {owner} out of range"));
            }
        }
        let owing = self.players.iter().any(|p| p.discard_remaining > 0);
        if owing != (self.phase == Phase::Discard) {
            return Err("discard_remaining must be set exactly in the Discard phase".into());
        }
        for (i, p) in self.players.iter().enumerate() {
            let held = hand_total(&p.hand);
            if p.discard_remaining as u32 > held {
                return Err(format!(
                    "player {i} must discard {} but holds only {held} cards",
                    p.discard_remaining
                ));
            }
        }
        let setup = matches!(self.phase, Phase::SetupSettlement | Phase::SetupRoad { .. });
        if setup {
            let step = self.setup_step as usize;
            if step >= SETUP_ORDER.len() {
                return Err(format!(
                    "setup_step {step} does not match phase {:?}",
                    self.phase
                ));
            }
            if self.current != SETUP_ORDER[step] {
                return Err(format!(
                    "setup_step {step} is player {}'s placement, but current is {}",
                    SETUP_ORDER[step], self.current
                ));
            }
            if let Some(i) = self.players.iter().position(|p| p.cities != 0) {
                return Err(format!(
                    "player {i} has a city, but there are no cities during setup"
                ));
            }
            // Every earlier step placed a settlement and its road; a SetupRoad step's
            // settlement is down too. Fewer pieces could leave no legal placement.
            let settlements: u32 = self
                .players
                .iter()
                .map(|p| p.settlements.count_ones())
                .sum();
            let roads: u32 = self.players.iter().map(|p| p.roads.count_ones()).sum();
            let placed = step as u32 + matches!(self.phase, Phase::SetupRoad { .. }) as u32;
            if settlements != placed {
                return Err(format!(
                    "setup_step {step} needs {placed} settlements on the board, found {settlements}"
                ));
            }
            if roads != step as u32 {
                return Err(format!(
                    "setup_step {step} needs {step} roads on the board, found {roads}"
                ));
            }
            if step >= 4 {
                if let Some(r) = (0..NUM_RESOURCES).find(|&r| self.bank[r] < 3) {
                    return Err(format!(
                        "setup_step {step} pays out a second settlement, but the bank holds only {} of resource {r}",
                        self.bank[r]
                    ));
                }
            }
        } else if self.setup_step as usize != SETUP_ORDER.len() {
            return Err(format!(
                "setup_step must be 8 outside setup, got {} in {:?}",
                self.setup_step, self.phase
            ));
        }
        if !self.is_over() && self.turn >= self.config.max_turns {
            return Err(format!(
                "turn {} reached max_turns {}, but the game is not over",
                self.turn, self.config.max_turns
            ));
        }
        if !matches!(self.return_phase, Phase::PreRoll | Phase::Main) {
            return Err(format!(
                "return_phase must be PreRoll or Main, got {:?}",
                self.return_phase
            ));
        }
        if self.phase == Phase::Discard && self.return_phase != Phase::Main {
            return Err(format!(
                "the Discard phase must return to Main, got return_phase {:?}",
                self.return_phase
            ));
        }
        let trading = matches!(self.phase, Phase::TradeResponse | Phase::TradeConfirm);
        if trading != self.trade.is_some() {
            return Err("a pending trade must exist exactly in the trade phases".into());
        }
        let cur = self.current as usize;
        if self.phase == Phase::Steal && !victims(self).iter().any(|&v| v) {
            return Err(format!(
                "Steal phase with no one to steal from on tile {}",
                self.robber
            ));
        }
        if let Phase::SetupRoad { node } = self.phase {
            if node as usize >= NUM_NODES {
                return Err(format!(
                    "SetupRoad node {node} is out of range (beyond {})",
                    NUM_NODES - 1
                ));
            }
            if self.players[cur].settlements & (1u64 << node) == 0 {
                return Err(format!(
                    "SetupRoad at node {node}, which has no settlement of the current player"
                ));
            }
            if t.node_edge_mask[node as usize] & !self.occupied_edges() == 0 {
                return Err(format!("SetupRoad at node {node}, which has no free edge"));
            }
        }
        if let Phase::RoadBuilding { roads_left } = self.phase {
            if !(1..=2).contains(&roads_left) {
                return Err(format!("RoadBuilding with {roads_left} roads left"));
            }
            let pieces = MAX_ROADS - self.players[cur].roads.count_ones();
            if roads_left as u32 > pieces {
                return Err(format!(
                    "RoadBuilding with {roads_left} roads left, but player {cur} has only {pieces} road pieces"
                ));
            }
            if !has_free_road(self, cur) {
                return Err(format!(
                    "RoadBuilding, but player {cur} has nowhere to build a road"
                ));
            }
        }
        if let Phase::GameOver { winner: Some(w) } = self.phase {
            if w as usize >= NUM_PLAYERS {
                return Err(format!("winner {w} out of range"));
            }
        }
        // The player on turn wins the moment they reach the target, so a game still going on
        // cannot have them there. Others may be: they win on their own turn.
        if !self.is_over() && self.total_vp(cur) >= self.config.vp_to_win {
            return Err(format!(
                "player {cur} has {} VP on their own turn (vp_to_win {}), but the game is not over",
                self.total_vp(cur),
                self.config.vp_to_win
            ));
        }
        let occupied = self.occupied_nodes();
        for p in 0..NUM_PLAYERS {
            let len = longest_road(self.players[p].roads, occupied & !self.buildings(p));
            if self.players[p].longest_road_len != len {
                return Err(format!("cached longest road of player {p} is stale"));
            }
        }
        Ok(())
    }
}
