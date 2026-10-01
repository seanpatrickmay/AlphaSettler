//! Applying actions. Actions must be legal (see `legal_actions`); illegal ones panic or corrupt.

use crate::action::Action;
use crate::events::Event;
use crate::rules::setup;
use crate::state::{Phase, State};
use crate::types::{DevCard, Hand, Resource, NUM_PLAYERS};

/// A random outcome supplied by the caller instead of drawn from the state's streams.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Chance {
    /// `discards` is only used in `catanatron_compat` mode on a 7.
    Roll { dice: (u8, u8), discards: Option<[Hand; NUM_PLAYERS]> },
    Steal(Resource),
    Dev(DevCard),
}

pub trait EventSink {
    fn emit(&mut self, e: Event);
}

/// Discards events; compiles away in search.
pub struct NoEvents;

impl EventSink for NoEvents {
    #[inline(always)]
    fn emit(&mut self, _e: Event) {}
}

impl EventSink for Vec<Event> {
    fn emit(&mut self, e: Event) {
        self.push(e);
    }
}

impl State {
    pub fn apply(&mut self, a: Action) {
        self.apply_with(a, None, &mut NoEvents);
    }

    pub fn apply_with<S: EventSink>(&mut self, a: Action, chance: Option<Chance>, sink: &mut S) {
        match (self.phase, a) {
            (Phase::SetupSettlement, Action::BuildSettlement(n)) => setup::apply_settlement(self, n, sink),
            (Phase::SetupRoad { node }, Action::BuildRoad(e)) => setup::apply_road(self, node, e, sink),
            (phase, a) => panic!("illegal action {a:?} in phase {phase:?} (chance {chance:?})"),
        }
    }
}
