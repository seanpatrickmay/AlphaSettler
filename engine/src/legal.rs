//! Legal action generation, dispatched on phase.

use crate::action::Action;
use crate::rules::setup;
use crate::state::{Phase, State};

pub fn legal_actions(s: &State, out: &mut Vec<Action>) {
    out.clear();
    match s.phase {
        Phase::SetupSettlement => setup::legal_settlement(s, out),
        Phase::SetupRoad { node } => setup::legal_road(s, node, out),
        Phase::GameOver { .. } => {}
        Phase::PreRoll
        | Phase::Discard
        | Phase::MoveRobber
        | Phase::Steal
        | Phase::Main
        | Phase::RoadBuilding { .. }
        | Phase::TradeResponse
        | Phase::TradeConfirm => {}
    }
}

impl State {
    pub fn legal_actions(&self) -> Vec<Action> {
        let mut v = Vec::new();
        legal_actions(self, &mut v);
        v
    }
}
