//! Information-set Monte Carlo tree search for AlphaSettler
//! (spec: docs/superpowers/specs/2026-10-01-bot-3a-ismcts-design.md).

pub mod belief;
pub mod evaluator;
pub mod gumbel;
pub mod noise;
pub mod puct;
pub mod search;
pub mod tree;
pub mod world;

pub use belief::{Belief, HandTracker, DEFAULT_MAX_STATES};
pub use evaluator::{Eval, Evaluator, Leaf, UniformEvaluator};
pub use noise::{dirichlet, gamma, normal, unit, Noised, RootNoise};
pub use search::{
    search, search_actions, search_moves, search_tree, terminal_value, visible_chance, RootMode,
    SearchConfig, SearchResult,
};
pub use world::WorldSampler;
