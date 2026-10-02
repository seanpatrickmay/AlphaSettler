//! The search tree: nodes in one arena; a decision node's children are keyed by action id, a
//! chance node's by the outcome the searching player sees (dice sum, resource or dev card index).

use settler_engine::{Action, NUM_PLAYERS};

pub const ROOT: u32 = 0;
pub const UNEXPANDED: u32 = u32::MAX;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NodeKind {
    /// The player to act (read from each world: it is public) chooses an action.
    Decision,
    /// `action` was chosen; its random outcome is visible to the searching player.
    Chance { action: Action },
}

#[derive(Clone, Debug, PartialEq)]
pub struct Child {
    pub key: u16,
    /// Index of the child node, or `UNEXPANDED` until a simulation first goes there.
    pub node: u32,
    pub prior: f32,
    pub visits: u32,
    /// Descents through the parent in which this action was legal (ISMCTS availability). Not
    /// exactly simulations: a descent abandoned as `Busy` later in the tree still counts.
    pub available: u32,
    /// Simulations through this child waiting for their leaf evaluation (virtual loss).
    pub pending: u32,
    pub value_sum: [f32; NUM_PLAYERS],
}

impl Child {
    pub fn new(key: u16, prior: f32) -> Child {
        Child {
            key,
            node: UNEXPANDED,
            prior,
            visits: 0,
            available: 0,
            pending: 0,
            value_sum: [0.0; NUM_PLAYERS],
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Node {
    pub kind: NodeKind,
    pub children: Vec<Child>,
    /// A decision node is expanded once evaluated; chance nodes are created expanded.
    pub expanded: bool,
    /// A simulation in the current batch is waiting to evaluate this node.
    pub awaiting_eval: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Tree {
    pub nodes: Vec<Node>,
}

impl Default for Tree {
    fn default() -> Self {
        Tree::new()
    }
}

impl Tree {
    /// A tree holding only the unexpanded root decision node.
    pub fn new() -> Tree {
        let mut t = Tree { nodes: Vec::new() };
        t.add(NodeKind::Decision);
        t
    }

    pub fn add(&mut self, kind: NodeKind) -> u32 {
        self.nodes.push(Node {
            kind,
            children: Vec::new(),
            expanded: matches!(kind, NodeKind::Chance { .. }),
            awaiting_eval: false,
        });
        (self.nodes.len() - 1) as u32
    }

    /// Index (in `node`'s children) of the child keyed `key`, added with `prior` if missing.
    pub fn child(&mut self, node: u32, key: u16, prior: f32) -> usize {
        let children = &mut self.nodes[node as usize].children;
        match children.iter().position(|c| c.key == key) {
            Some(i) => i,
            None => {
                children.push(Child::new(key, prior));
                children.len() - 1
            }
        }
    }

    /// The node under `parent`'s child `i`, created as `kind` on first visit.
    pub fn descend_to(&mut self, parent: u32, i: usize, kind: NodeKind) -> u32 {
        let existing = self.nodes[parent as usize].children[i].node;
        if existing != UNEXPANDED {
            return existing;
        }
        let n = self.add(kind);
        self.nodes[parent as usize].children[i].node = n;
        n
    }
}
