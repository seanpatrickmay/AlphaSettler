//! A game with legality checking and an event log, for arenas and bindings.
//! Search should use `State` directly.

use crate::action::Action;
use crate::apply::{check_chance, Chance};
use crate::config::GameConfig;
use crate::events::Event;
use crate::legal::legal_actions;
use crate::observation::Observation;
use crate::snapshot::Snapshot;
use crate::state::{Phase, State};
use crate::types::PlayerId;

/// Why `Game::apply` / `Game::apply_forced` refused an action. The game is left untouched.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ApplyError {
    /// `action` is not legal in `phase`.
    Illegal { action: Action, phase: Phase },
    /// The forced `chance` cannot happen for `action` in the current state.
    ImpossibleChance {
        action: Action,
        chance: Chance,
        reason: &'static str,
    },
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

    /// A game continuing from `snap` (see `State::from_snapshot`), with an empty event log.
    pub fn from_snapshot(seed: u64, config: GameConfig, snap: &Snapshot) -> Result<Game, String> {
        Ok(Game::from_state(State::from_snapshot(seed, config, snap)?))
    }

    pub fn state(&self) -> &State {
        &self.state
    }

    pub fn legal_actions(&self) -> Vec<Action> {
        self.state.legal_actions()
    }

    pub fn apply(&mut self, a: Action) -> Result<(), ApplyError> {
        self.apply_forced(a, None)
    }

    /// Apply `a` if it is legal, otherwise return `ApplyError::Illegal` and leave the game untouched.
    ///
    /// `chance` forces the outcome of a random event (dice, steal, dev draw), for tests and
    /// replay. An outcome that cannot happen here returns `ApplyError::ImpossibleChance`, also
    /// leaving the game untouched.
    pub fn apply_forced(&mut self, a: Action, chance: Option<Chance>) -> Result<(), ApplyError> {
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
            return Err(ApplyError::Illegal {
                action: a,
                phase: self.state.phase,
            });
        }
        if let Some(c) = chance {
            check_chance(&self.state, a, c).map_err(|reason| ApplyError::ImpossibleChance {
                action: a,
                chance: c,
                reason,
            })?;
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
