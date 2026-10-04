//! The learned evaluator for AlphaSettler (spec:
//! docs/superpowers/specs/2026-10-02-bot-3b-network-design.md): the feature encoder, the board
//! graph and policy ids, the weights file, the forward pass, `NetEvaluator` and the training-data
//! sampler.

pub mod data;
pub mod encode;
pub mod evaluator;
pub mod forward;
pub mod graph;
pub mod policy;
pub mod weights;
