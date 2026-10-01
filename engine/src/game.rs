//! A game with legality checking and an event log, for arenas and bindings.
//! Search should use `State` directly.

use crate::action::Action;
use crate::apply::Chance;
use crate::config::GameConfig;
use crate::events::Event;
use crate::legal::legal_actions;
use crate::observation::Observation;
use crate::state::{Phase, State};
use crate::types::PlayerId;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IllegalAction {
    pub action: Action,
    pub phase: Phase,
}

#[derive(Clone, Debug)]
pub struct Game {
    state: State,
    log: Vec<Event>,
    buf: Vec<Action>,
}

impl Game {
    pub fn new(seed: u64, config: GameConfig) -> Game {
        Game::from_state(State::new(seed, config))
    }

    pub fn from_state(state: State) -> Game {
        Game {
            state,
            log: Vec::new(),
            buf: Vec::new(),
        }
    }

    pub fn state(&self) -> &State {
        &self.state
    }

    pub fn legal_actions(&self) -> Vec<Action> {
        self.state.legal_actions()
    }

    pub fn apply(&mut self, a: Action) -> Result<(), IllegalAction> {
        self.apply_forced(a, None)
    }

    /// Apply `a` if it is legal, otherwise return `IllegalAction` and leave the game untouched.
    ///
    /// `chance` forces the outcome of a random event (dice, steal, dev draw). It is trusted
    /// input meant for tests and replay: an impossible forced outcome panics.
    pub fn apply_forced(&mut self, a: Action, chance: Option<Chance>) -> Result<(), IllegalAction> {
        // Year of Plenty is unordered: canonicalize to ascending resource order.
        // Every other action passes through untouched so out-of-range payloads are rejected.
        let a = match a {
            Action::PlayYearOfPlenty(x, y) if x.index() > y.index() => {
                Action::PlayYearOfPlenty(y, x)
            }
            other => other,
        };
        legal_actions(&self.state, &mut self.buf);
        if !self.buf.contains(&a) {
            return Err(IllegalAction {
                action: a,
                phase: self.state.phase,
            });
        }
        self.state.apply_with(a, chance, &mut self.log);
        Ok(())
    }

    pub fn observation(&self, viewer: PlayerId) -> Observation {
        self.state.observation(viewer)
    }

    /// The full event log as `viewer` is allowed to see it.
    pub fn log_for(&self, viewer: PlayerId) -> Vec<Event> {
        self.log.iter().map(|e| e.redacted_for(viewer)).collect()
    }
}
