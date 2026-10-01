//! Legal action generation, dispatched on phase.

use crate::action::Action;
use crate::rules::{build, dev, maritime, robber, roll, setup};
use crate::state::{Phase, State};

pub fn legal_actions(s: &State, out: &mut Vec<Action>) {
    out.clear();
    match s.phase {
        Phase::SetupSettlement => setup::legal_settlement(s, out),
        Phase::SetupRoad { node } => setup::legal_road(s, node, out),
        Phase::PreRoll => {
            out.push(Action::Roll);
            dev::legal_knight(s, out);
        }
        Phase::Discard => roll::legal_discard(s, out),
        Phase::MoveRobber => robber::legal_move_robber(s, out),
        Phase::Steal => robber::legal_steal(s, out),
        Phase::GameOver { .. } => {}
        Phase::Main => {
            build::legal_main_builds(s, out);
            dev::legal_main_dev(s, out);
            maritime::legal(s, out);
            out.push(Action::EndTurn);
        }
        Phase::RoadBuilding { .. } => dev::legal_road_building(s, out),
        Phase::TradeResponse | Phase::TradeConfirm => {}
    }
}

impl State {
    pub fn legal_actions(&self) -> Vec<Action> {
        let mut v = Vec::new();
        legal_actions(self, &mut v);
        v
    }
}
