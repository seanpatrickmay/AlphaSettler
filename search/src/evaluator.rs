//! The seam between the search and whatever scores positions: a hand-written heuristic in 3a, a
//! network in 3b (spec Section 4).

use settler_engine::{Action, PlayerId, State, NUM_PLAYERS};

/// A position to evaluate: a sampled world, the player to act, and their legal moves.
pub struct Leaf<'a> {
    pub world: &'a State,
    pub actor: PlayerId,
    pub legal: &'a [Action],
}

#[derive(Clone, Debug, PartialEq)]
pub struct Eval {
    /// Win probability per player, summing to 1. May use the whole sampled world.
    pub value: [f32; NUM_PLAYERS],
    /// One probability per entry of `legal`, summing to 1. Must depend only on
    /// `world.observation(actor)`, so opponents in the tree do not play as if they saw hidden cards.
    pub prior: Vec<f32>,
}

pub trait Evaluator {
    /// One `Eval` per leaf, in order.
    fn evaluate(&mut self, leaves: &[Leaf<'_>]) -> Vec<Eval>;
}

/// Equal values and a uniform prior: the search's own baseline and a test double.
pub struct UniformEvaluator;

impl Evaluator for UniformEvaluator {
    fn evaluate(&mut self, leaves: &[Leaf<'_>]) -> Vec<Eval> {
        leaves
            .iter()
            .map(|l| Eval {
                value: [1.0 / NUM_PLAYERS as f32; NUM_PLAYERS],
                prior: vec![1.0 / l.legal.len() as f32; l.legal.len()],
            })
            .collect()
    }
}
