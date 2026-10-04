//! Policy ids (spec Section 2): the network's 665 policy slots. A move's slot is its engine id
//! (`Action::encode`), except that `StealFrom(p)` and `ConfirmTrade(p)` name `p` relative to the
//! actor, the way the inputs name players: "steal from the next player" has one slot whoever
//! acts. The training sampler (`data.rs`) and `NetEvaluator` both go through `policy_id`, so
//! targets and priors sit in the same slots. Slots `SETTLEMENT_BASE..ENTITY_END` come from network
//! v2's per-entity heads (settlements and cities per node, roads per edge, robber moves per
//! tile); every other slot comes from the pooled head, in slot order (`other_slot`).

use crate::encode::rel;
use settler_engine::topology::{NUM_EDGES, NUM_NODES, NUM_TILES};
use settler_engine::{Action, PlayerId, ACTION_SPACE_SIZE};

/// `BuildSettlement(n)` is slot `SETTLEMENT_BASE + n`; cities, roads and robber moves follow
/// contiguously (`net/tests/policy.rs` checks each base against `Action::encode`).
pub const SETTLEMENT_BASE: usize = 25;
pub const CITY_BASE: usize = SETTLEMENT_BASE + NUM_NODES;
pub const ROAD_BASE: usize = CITY_BASE + NUM_NODES;
pub const ROBBER_BASE: usize = ROAD_BASE + NUM_EDGES;
/// One past the last robber slot.
pub const ENTITY_END: usize = ROBBER_BASE + NUM_TILES;
/// The slots no entity head produces: 0..25 and 224..665.
pub const OTHER_POLICY_LEN: usize = ACTION_SPACE_SIZE - (ENTITY_END - SETTLEMENT_BASE);

/// The slot of the pooled head's output `j` (`j < OTHER_POLICY_LEN`).
#[inline]
pub fn other_slot(j: usize) -> usize {
    if j < SETTLEMENT_BASE {
        j
    } else {
        j + (ENTITY_END - SETTLEMENT_BASE)
    }
}

/// The policy slot of `a` played by `actor`.
#[inline]
pub fn policy_id(a: Action, actor: PlayerId) -> usize {
    let a = match a {
        Action::StealFrom(p) => Action::StealFrom(rel(p, actor) as PlayerId),
        Action::ConfirmTrade(p) => Action::ConfirmTrade(rel(p, actor) as PlayerId),
        other => other,
    };
    a.encode() as usize
}
