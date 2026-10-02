//! The simulation loop: single-observer ISMCTS with PUCT, 4-player value vectors, chance nodes
//! for outcomes the searching player sees, and leaf batching with virtual loss (spec Section 3).

use crate::belief::Belief;
use crate::evaluator::{Evaluator, Leaf};
use crate::puct;
use crate::tree::{Child, NodeKind, Tree, ROOT};
use crate::world::WorldSampler;
use settler_engine::legal::legal_actions;
use settler_engine::rng::Rng;
use settler_engine::rules::roll::pick_card;
use settler_engine::{Action, Chance, NoEvents, Observation, PlayerId, State, NUM_PLAYERS};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SearchConfig {
    /// Simulations per decision: the budget (never wall time).
    pub simulations: u32,
    pub c_puct: f32,
    /// Leaves per `Evaluator` call.
    pub batch: usize,
}

impl Default for SearchConfig {
    fn default() -> Self {
        SearchConfig {
            simulations: 1000,
            c_puct: 1.5,
            batch: 8,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct SearchResult {
    pub action: Action,
    /// Root visit counts aligned with the `legal` slice given to `search` (0 for unsearched
    /// actions). They sum to `simulations - 1`: the first simulation evaluates the root.
    pub visits: Vec<u32>,
    pub simulations: u32,
    pub nodes: usize,
}

/// The actions the search considers: everything legal except domestic offers (3a never trades).
pub fn search_actions(legal: &[Action], out: &mut Vec<Action>) {
    out.clear();
    out.extend(
        legal
            .iter()
            .copied()
            .filter(|a| !matches!(a, Action::OfferTrade { .. })),
    );
}

/// A finished game's value: 1 for the winner, ¼ each when the turn cap ended it.
pub fn terminal_value(s: &State) -> [f32; NUM_PLAYERS] {
    match s.winner() {
        Some(w) => {
            let mut v = [0.0; NUM_PLAYERS];
            v[w as usize] = 1.0;
            v
        }
        None => [1.0 / NUM_PLAYERS as f32; NUM_PLAYERS],
    }
}

/// Whether `viewer` sees the random outcome of `a` in `s`: every roll, their own dev-card draws,
/// and steals they make or suffer.
pub fn visible_chance(s: &State, a: Action, viewer: PlayerId) -> bool {
    match a {
        Action::Roll => true,
        Action::BuyDev => s.current == viewer,
        Action::StealFrom(v) => s.current == viewer || v == viewer,
        _ => false,
    }
}

/// Apply `a` with a sampled outcome and return the outcome code the viewer sees.
fn apply_visible(s: &mut State, a: Action, rng: &mut Rng) -> u16 {
    match a {
        Action::Roll => {
            let dice = (1 + rng.below(6) as u8, 1 + rng.below(6) as u8);
            s.apply_with(
                a,
                Some(Chance::Roll {
                    dice,
                    discards: None,
                }),
                &mut NoEvents,
            );
            (dice.0 + dice.1) as u16
        }
        Action::StealFrom(v) => {
            let r = pick_card(&s.players[v as usize].hand, rng);
            s.apply_with(a, Some(Chance::Steal(r)), &mut NoEvents);
            r.index() as u16
        }
        Action::BuyDev => {
            let card = s.dev_deck[s.dev_deck_pos as usize];
            s.apply(a);
            card.index() as u16
        }
        _ => unreachable!("{a:?} has no visible chance outcome"),
    }
}

/// The 665 action ids as a bitset.
#[derive(Clone, Copy, Default)]
struct ActionSet([u64; 11]);

impl ActionSet {
    fn of(actions: &[Action]) -> ActionSet {
        let mut s = ActionSet::default();
        for a in actions {
            let k = a.encode() as usize;
            s.0[k / 64] |= 1 << (k % 64);
        }
        s
    }

    fn contains(&self, key: u16) -> bool {
        (self.0[key as usize / 64] >> (key % 64)) & 1 == 1
    }

    fn remove(&mut self, key: u16) {
        self.0[key as usize / 64] &= !(1u64 << (key % 64));
    }
}

type Path = Vec<(u32, usize)>;

struct Pending {
    path: Path,
    node: u32,
    world: State,
    legal: Vec<Action>,
}

// A Descent is moved straight into the batch, so boxing the large variant would only add an allocation.
#[allow(clippy::large_enum_variant)]
enum Descent {
    Terminal {
        path: Path,
        value: [f32; NUM_PLAYERS],
    },
    Leaf(Pending),
    /// Reached a node another simulation in this batch is already evaluating.
    Busy,
}

fn descend(
    tree: &mut Tree,
    mut world: State,
    viewer: PlayerId,
    c_puct: f32,
    rng: &mut Rng,
    buf: &mut Vec<Action>,
    legal: &mut Vec<Action>,
) -> Descent {
    let mut node = ROOT;
    let mut path = Path::new();
    loop {
        if world.is_over() {
            return Descent::Terminal {
                value: terminal_value(&world),
                path,
            };
        }
        if let NodeKind::Chance { action } = tree.nodes[node as usize].kind {
            let key = apply_visible(&mut world, action, rng);
            let i = tree.child(node, key, 0.0);
            path.push((node, i));
            node = tree.descend_to(node, i, NodeKind::Decision);
            continue;
        }
        legal_actions(&world, buf);
        search_actions(buf, legal);
        let n = &mut tree.nodes[node as usize];
        if !n.expanded {
            if n.awaiting_eval {
                return Descent::Busy;
            }
            n.awaiting_eval = true;
            return Descent::Leaf(Pending {
                path,
                node,
                world,
                legal: legal.clone(),
            });
        }
        let allowed = ActionSet::of(legal);
        let mut unseen = allowed;
        for c in n.children.iter_mut() {
            if allowed.contains(c.key) {
                c.available += 1;
                unseen.remove(c.key);
            }
        }
        // Actions this world allows that earlier worlds did not: uniform prior.
        let uniform = 1.0 / legal.len() as f32;
        for a in legal.iter() {
            if unseen.contains(a.encode()) {
                n.children.push(Child {
                    available: 1,
                    ..Child::new(a.encode(), uniform)
                });
            }
        }
        let actor = world.current_actor() as usize;
        let i = puct::select(&n.children, |c| allowed.contains(c.key), actor, c_puct)
            .expect("a live decision node has a legal action");
        let a = Action::decode(n.children[i].key).expect("children are keyed by valid action ids");
        path.push((node, i));
        if visible_chance(&world, a, viewer) {
            node = tree.descend_to(node, i, NodeKind::Chance { action: a });
        } else {
            world.apply(a);
            node = tree.descend_to(node, i, NodeKind::Decision);
        }
    }
}

fn set_pending(tree: &mut Tree, path: &[(u32, usize)], on: bool) {
    for &(n, i) in path {
        let c = &mut tree.nodes[n as usize].children[i];
        if on {
            c.pending += 1;
        } else {
            c.pending -= 1;
        }
    }
}

fn backup(tree: &mut Tree, path: &[(u32, usize)], value: &[f32; NUM_PLAYERS]) {
    for &(n, i) in path {
        let c = &mut tree.nodes[n as usize].children[i];
        c.visits += 1;
        for p in 0..NUM_PLAYERS {
            c.value_sum[p] += value[p];
        }
    }
}

fn expand(tree: &mut Tree, node: u32, legal: &[Action], prior: &[f32]) {
    assert_eq!(
        prior.len(),
        legal.len(),
        "the evaluator's prior must align with the legal actions"
    );
    let n = &mut tree.nodes[node as usize];
    n.expanded = true;
    n.awaiting_eval = false;
    n.children
        .extend(legal.iter().zip(prior).map(|(a, &p)| Child {
            available: 1,
            ..Child::new(a.encode(), p)
        }));
}

/// Search `obs.viewer`'s decision among `legal` (see `search`), also returning the tree.
pub fn search_tree<E: Evaluator>(
    obs: &Observation,
    legal: &[Action],
    belief: &Belief,
    eval: &mut E,
    cfg: &SearchConfig,
    rng: &mut Rng,
) -> Result<(SearchResult, Tree), String> {
    assert!(
        cfg.simulations > 0 && cfg.batch > 0,
        "simulations and batch must be positive"
    );
    let mut root_actions = Vec::new();
    search_actions(legal, &mut root_actions);
    match root_actions.len() {
        0 => return Err("nothing to search: no legal action".into()),
        1 => {
            let r = SearchResult {
                action: root_actions[0],
                visits: vec![0; legal.len()],
                simulations: 0,
                nodes: 0,
            };
            return Ok((r, Tree::new()));
        }
        _ => {}
    }
    let sampler = WorldSampler::new(obs, belief)?;
    let mut tree = Tree::new();
    let (mut buf, mut scratch) = (Vec::new(), Vec::new());
    let mut batch: Vec<Pending> = Vec::with_capacity(cfg.batch);
    let mut done = 0u32;
    while done < cfg.simulations {
        batch.clear();
        while batch.len() < cfg.batch && done + (batch.len() as u32) < cfg.simulations {
            let world = sampler.sample(belief, rng);
            match descend(
                &mut tree,
                world,
                obs.viewer,
                cfg.c_puct,
                rng,
                &mut buf,
                &mut scratch,
            ) {
                Descent::Terminal { path, value } => {
                    backup(&mut tree, &path, &value);
                    done += 1;
                }
                Descent::Leaf(p) => {
                    set_pending(&mut tree, &p.path, true);
                    batch.push(p);
                }
                Descent::Busy => break,
            }
        }
        if batch.is_empty() {
            continue;
        }
        let leaves: Vec<Leaf<'_>> = batch
            .iter()
            .map(|p| Leaf {
                world: &p.world,
                actor: p.world.current_actor(),
                legal: &p.legal,
            })
            .collect();
        let evals = eval.evaluate(&leaves);
        assert_eq!(
            evals.len(),
            batch.len(),
            "the evaluator must return one evaluation per leaf"
        );
        for (p, e) in batch.iter().zip(&evals) {
            expand(&mut tree, p.node, &p.legal, &e.prior);
            set_pending(&mut tree, &p.path, false);
            backup(&mut tree, &p.path, &e.value);
            done += 1;
        }
    }
    let root = &tree.nodes[ROOT as usize];
    let best = root
        .children
        .iter()
        .max_by(|a, b| {
            a.visits
                .cmp(&b.visits)
                .then(a.prior.total_cmp(&b.prior))
                .then(b.key.cmp(&a.key))
        })
        .expect("the first simulation expands the root");
    let action = Action::decode(best.key).expect("children are keyed by valid action ids");
    let visits = legal
        .iter()
        .map(|a| {
            root.children
                .iter()
                .find(|c| c.key == a.encode())
                .map_or(0, |c| c.visits)
        })
        .collect();
    let nodes = tree.nodes.len();
    Ok((
        SearchResult {
            action,
            visits,
            simulations: done,
            nodes,
        },
        tree,
    ))
}

/// The most-visited move for `obs.viewer` among `legal` after `cfg.simulations` simulations, each
/// in a world sampled from `belief`. A single searchable move is returned without searching.
pub fn search<E: Evaluator>(
    obs: &Observation,
    legal: &[Action],
    belief: &Belief,
    eval: &mut E,
    cfg: &SearchConfig,
    rng: &mut Rng,
) -> Result<SearchResult, String> {
    search_tree(obs, legal, belief, eval, cfg, rng).map(|(r, _)| r)
}
