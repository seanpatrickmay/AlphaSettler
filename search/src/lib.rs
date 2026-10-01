//! Information-set Monte Carlo tree search for AlphaSettler
//! (spec: docs/superpowers/specs/2026-10-01-bot-3a-ismcts-design.md).

pub mod belief;

pub use belief::{Belief, HandTracker, DEFAULT_MAX_STATES};
