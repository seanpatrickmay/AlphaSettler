//! The simulation loop: single-observer ISMCTS with PUCT, 4-player value vectors, chance nodes
//! for outcomes the searching player sees, and leaf batching with virtual loss (spec Section 3).
//! P2: the root may instead be searched by Gumbel sequential halving (`RootMode::Gumbel`,
//! docs/superpowers/specs/2026-10-08-gumbel-root-design.md Section 3); below the root it is PUCT.
//! 3d-4: with `SearchConfig::compound`, the searching player's own Main decisions search compound
//! build moves (`crate::compound`) instead of bare maritime trades.

use crate::belief::Belief;
use crate::compound::{self, Compound};
use crate::evaluator::{Evaluator, Leaf};
use crate::gumbel;
use crate::puct;
use crate::tree::{Child, NodeKind, Tree, ROOT};
use crate::world::WorldSampler;
use settler_engine::legal::legal_actions;
use settler_engine::rng::Rng;
use settler_engine::rules::roll::pick_card;
use settler_engine::{
    Action, Chance, NoEvents, Observation, Phase, PlayerId, State, ACTION_SPACE_SIZE, NUM_PLAYERS,
};

/// How the root's simulations are spent and its move chosen.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RootMode {
    /// PUCT at the root as below it; the most-visited move is played (3a to 3c).
    Puct,
    /// Gumbel sequential halving over the root's candidates (spec P2 Section 3). `draw`: the
    /// Gumbel noise comes from the search's `Rng` (self-play); off, it is zero (play).
    Gumbel { draw: bool },
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SearchConfig {
    /// Simulations per decision: the budget (never wall time).
    pub simulations: u32,
    pub c_puct: f32,
    /// Leaves per `Evaluator` call.
    pub batch: usize,
    /// Domestic offers are searched moves for every seat in the tree: a bot that may offer
    /// assumes its opponents may too. Off, no seat offers in the tree (3a and 3b never offered).
    pub offers: bool,
    /// The root's rule: PUCT unless the bot asks for Gumbel (`+g`, Gumbel self-play).
    pub root: RootMode,
    /// 3d-4 (`+c`): the searching player's own Main decisions search compound build moves in
    /// place of bare maritime trades. Never with a Gumbel root.
    pub compound: bool,
}

impl Default for SearchConfig {
    fn default() -> Self {
        SearchConfig {
            simulations: 1000,
            c_puct: 1.5,
            batch: 8,
            offers: false,
            root: RootMode::Puct,
            compound: false,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct SearchResult {
    /// The chosen move's first step (the move itself unless it is a compound).
    pub action: Action,
    /// Root visit counts aligned with the `legal` slice given to `search` (0 for unsearched
    /// actions), each compound's visits added to its first trade's entry (spec 3d-4 Section 3).
    /// They sum to `simulations - 1`: the first simulation evaluates the root.
    pub visits: Vec<u32>,
    pub simulations: u32,
    pub nodes: usize,
    /// A Gumbel search's improved policy softmax(ℓ + σ(completedQ)), aligned with `legal`,
    /// summing to 1, 0 for a move the search does not consider (an offer without `offers`).
    /// `None` for PUCT and for a decision returned without searching.
    pub improved: Option<Vec<f32>>,
    /// The chosen compound's steps after `action` (empty for a plain move).
    pub plan: Vec<Action>,
    /// Every root compound move (all its steps) with its visits, in move order; empty when the
    /// root searched none.
    pub compound_visits: Vec<(Vec<Action>, u32)>,
}

/// The actions the search considers: everything legal, with domestic offers only when `offers`.
pub fn search_moves(legal: &[Action], offers: bool, out: &mut Vec<Action>) {
    out.clear();
    out.extend(
        legal
            .iter()
            .copied()
            .filter(|a| offers || !matches!(a, Action::OfferTrade { .. })),
    );
}

/// `search_moves` without offers: what a bot that never offers searches, and what rollouts play.
pub fn search_actions(legal: &[Action], out: &mut Vec<Action>) {
    search_moves(legal, false, out);
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

/// Move keys as a bitset: the 665 action ids, then the compound keys (`compound::COMPOUND_BIT` |
/// a build's id) above them.
#[derive(Clone, Copy, Default)]
struct ActionSet([u64; 21]);

fn slot(key: u16) -> usize {
    if key & compound::COMPOUND_BIT == 0 {
        key as usize
    } else {
        ACTION_SPACE_SIZE + (key & !compound::COMPOUND_BIT) as usize
    }
}

impl ActionSet {
    fn of(keys: &[u16]) -> ActionSet {
        let mut s = ActionSet::default();
        for &k in keys {
            let i = slot(k);
            s.0[i / 64] |= 1 << (i % 64);
        }
        s
    }

    fn contains(&self, key: u16) -> bool {
        let i = slot(key);
        (self.0[i / 64] >> (i % 64)) & 1 == 1
    }

    fn remove(&mut self, key: u16) {
        let i = slot(key);
        self.0[i / 64] &= !(1u64 << (i % 64));
    }
}

type Path = Vec<(u32, usize)>;

struct Pending {
    path: Path,
    node: u32,
    world: State,
    legal: Vec<Action>,
    /// A compound node's compounds (its moves are `compound::move_keys` of `legal` and these);
    /// `None` for a plain node, whose moves are `legal`.
    compounds: Option<Vec<Compound>>,
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
    let keys: Vec<u16> = legal.iter().map(|a| a.encode()).collect();
    expand_keys(tree, node, legal, prior, &keys, prior);
}

/// `expand` for a compound node (spec 3d-4 Section 2): its children are the move set, with the
/// priors `compound::move_priors` forms from the evaluator's prior over the engine's moves.
fn expand_compound(
    tree: &mut Tree,
    node: u32,
    legal: &[Action],
    prior: &[f32],
    compounds: &[Compound],
) {
    let mut keys = Vec::new();
    compound::move_keys(legal, compounds, true, &mut keys);
    let priors = compound::move_priors(legal, prior, compounds);
    expand_keys(tree, node, legal, prior, &keys, &priors);
}

/// Mark `node` expanded with one child per key, at `priors` (aligned with `keys`); `prior` is the
/// evaluator's, checked against `legal`.
fn expand_keys(
    tree: &mut Tree,
    node: u32,
    legal: &[Action],
    prior: &[f32],
    keys: &[u16],
    priors: &[f32],
) {
    assert_eq!(
        prior.len(),
        legal.len(),
        "the evaluator's prior must align with the legal actions"
    );
    let n = &mut tree.nodes[node as usize];
    n.expanded = true;
    n.awaiting_eval = false;
    n.children
        .extend(keys.iter().zip(priors).map(|(&k, &p)| Child {
            available: 1,
            ..Child::new(k, p)
        }));
}

/// One search's working state: the tree, the world sampler and the leaf batch.
struct Sim<'a, E> {
    obs: &'a Observation,
    belief: &'a Belief,
    sampler: WorldSampler,
    eval: &'a mut E,
    cfg: &'a SearchConfig,
    rng: &'a mut Rng,
    tree: Tree,
    buf: Vec<Action>,
    scratch: Vec<Action>,
    /// This node's compounds (empty at a plain node) and move keys.
    compounds: Vec<Compound>,
    keys: Vec<u16>,
    batch: Vec<Pending>,
    /// Simulations completed (each backed up).
    done: u32,
    /// The root's evaluation: its backup path is empty, so the tree does not keep it.
    root_value: Option<[f32; NUM_PLAYERS]>,
}

impl<E: Evaluator> Sim<'_, E> {
    /// One descent from the root in `world`: to a terminal, to a leaf to evaluate, or `Busy`.
    /// `force` is the first step from the root (a Gumbel phase's candidate); `None`, and every
    /// step below the root, is PUCT. With `cfg.compound`, the searching player's own Main
    /// decisions search compound moves instead of bare maritime trades (spec 3d-4 Section 2); a
    /// compound edge applies its trades, then its build, each asserted legal.
    fn descend(&mut self, mut world: State, mut force: Option<u16>) -> Descent {
        let mut node = ROOT;
        let mut path = Path::new();
        loop {
            if world.is_over() {
                return Descent::Terminal {
                    value: terminal_value(&world),
                    path,
                };
            }
            if let NodeKind::Chance { action } = self.tree.nodes[node as usize].kind {
                let key = apply_visible(&mut world, action, self.rng);
                let i = self.tree.child(node, key, 0.0);
                path.push((node, i));
                node = self.tree.descend_to(node, i, NodeKind::Decision);
                continue;
            }
            legal_actions(&world, &mut self.buf);
            search_moves(&self.buf, self.cfg.offers, &mut self.scratch);
            let compound_node = self.cfg.compound && compound::applies(&world, self.obs.viewer);
            self.compounds.clear();
            if compound_node {
                compound::compounds(&world, &mut self.compounds);
            }
            compound::move_keys(
                &self.scratch,
                &self.compounds,
                compound_node,
                &mut self.keys,
            );
            let n = &mut self.tree.nodes[node as usize];
            if !n.expanded {
                if n.awaiting_eval {
                    return Descent::Busy;
                }
                n.awaiting_eval = true;
                return Descent::Leaf(Pending {
                    path,
                    node,
                    world,
                    legal: self.scratch.clone(),
                    compounds: compound_node.then(|| self.compounds.clone()),
                });
            }
            // Availability counts every descent through here, including one that then stops at a
            // node another simulation in the batch is evaluating (`Busy`), which is not counted as
            // a simulation: at batch 8 that over-counts the root's simulations by about 0.3%.
            let allowed = ActionSet::of(&self.keys);
            let mut unseen = allowed;
            for c in n.children.iter_mut() {
                if allowed.contains(c.key) {
                    c.available += 1;
                    unseen.remove(c.key);
                }
            }
            // Moves this world allows that earlier worlds did not: uniform prior.
            let uniform = 1.0 / self.keys.len() as f32;
            for &key in self.keys.iter() {
                if unseen.contains(key) {
                    n.children.push(Child {
                        available: 1,
                        ..Child::new(key, uniform)
                    });
                }
            }
            let i = match force.take() {
                // The root decision is the searching player's own, so its moves are the same in
                // every world (spec P2 Section 3).
                Some(key) => {
                    assert!(
                        allowed.contains(key),
                        "the forced root move {:?} is not legal in a sampled world",
                        Action::decode(key)
                    );
                    n.children
                        .iter()
                        .position(|c| c.key == key)
                        .expect("an allowed move is a child")
                }
                None => {
                    let actor = world.current_actor() as usize;
                    puct::select(
                        &n.children,
                        |c| allowed.contains(c.key),
                        actor,
                        self.cfg.c_puct,
                    )
                    .expect("a live decision node has a legal action")
                }
            };
            let key = n.children[i].key;
            path.push((node, i));
            let a = match compound::find(&self.compounds, key) {
                // A compound edge (spec 3d-4 Section 2): its trades in order, then its build, each
                // a legal engine action at that point. The build goes on as a plain move does
                // (a dev card through its chance node).
                Some(c) => {
                    for &t in &c.trades {
                        legal_actions(&world, &mut self.buf);
                        assert!(
                            self.buf.contains(&t),
                            "the compound trade {t:?} (for {:?}) is not legal in a sampled world",
                            c.build
                        );
                        world.apply(t);
                    }
                    legal_actions(&world, &mut self.buf);
                    assert!(
                        self.buf.contains(&c.build),
                        "the compound build {:?} is not legal after its trades",
                        c.build
                    );
                    c.build
                }
                None => Action::decode(key).expect("children are keyed by valid action ids"),
            };
            if visible_chance(&world, a, self.obs.viewer) {
                node = self
                    .tree
                    .descend_to(node, i, NodeKind::Chance { action: a });
            } else {
                world.apply(a);
                node = self.tree.descend_to(node, i, NodeKind::Decision);
            }
        }
    }

    /// `count` more simulations, each in a world sampled from the belief, at most `cfg.batch`
    /// leaves per evaluator call. Simulation `k` of this call takes its first step from the root
    /// to the child keyed `step(k)` (`None`: PUCT). A simulation that meets a node another one in
    /// the batch is evaluating (`Busy`) ends the batch and is retried, in a new world, in the next.
    fn run(&mut self, count: usize, step: impl Fn(usize) -> Option<u16>) {
        let mut next = 0;
        while next < count {
            self.batch.clear();
            while self.batch.len() < self.cfg.batch && next < count {
                let world = self.sampler.sample(self.belief, self.rng);
                match self.descend(world, step(next)) {
                    Descent::Terminal { path, value } => {
                        backup(&mut self.tree, &path, &value);
                        self.done += 1;
                        next += 1;
                    }
                    Descent::Leaf(p) => {
                        set_pending(&mut self.tree, &p.path, true);
                        self.batch.push(p);
                        next += 1;
                    }
                    Descent::Busy => break,
                }
            }
            if self.batch.is_empty() {
                continue;
            }
            let leaves: Vec<Leaf<'_>> = self
                .batch
                .iter()
                .map(|p| Leaf {
                    world: &p.world,
                    actor: p.world.current_actor(),
                    legal: &p.legal,
                })
                .collect();
            let evals = self.eval.evaluate(&leaves);
            assert_eq!(
                evals.len(),
                self.batch.len(),
                "the evaluator must return one evaluation per leaf"
            );
            for (p, e) in self.batch.iter().zip(&evals) {
                if p.node == ROOT {
                    self.root_value = Some(e.value);
                }
                match &p.compounds {
                    None => expand(&mut self.tree, p.node, &p.legal, &e.prior),
                    Some(cs) => expand_compound(&mut self.tree, p.node, &p.legal, &e.prior, cs),
                }
                set_pending(&mut self.tree, &p.path, false);
                backup(&mut self.tree, &p.path, &e.value);
                self.done += 1;
            }
        }
    }
}

/// The most-visited root move among `root_keys` (ties: the higher prior, then the lower key).
/// Root children come from each world's moves; only one the caller's move set holds is played.
fn most_visited(tree: &Tree, root_keys: &[u16]) -> Result<u16, String> {
    let searchable = ActionSet::of(root_keys);
    let best = tree.nodes[ROOT as usize]
        .children
        .iter()
        .filter(|c| searchable.contains(c.key))
        .max_by(|a, b| {
            a.visits
                .cmp(&b.visits)
                .then(a.prior.total_cmp(&b.prior))
                .then(b.key.cmp(&a.key))
        })
        .ok_or("no searched root move is among the legal actions")?;
    Ok(best.key)
}

/// Visits and mean `actor` value of the root children at `at` (indices into the root's
/// children); the mean is 0 for an unvisited child.
fn root_stats(tree: &Tree, at: &[usize], actor: usize) -> (Vec<u32>, Vec<f32>) {
    let children = &tree.nodes[ROOT as usize].children;
    at.iter()
        .map(|&i| {
            let c = &children[i];
            let q = if c.visits > 0 {
                c.value_sum[actor] / c.visits as f32
            } else {
                0.0
            };
            (c.visits, q)
        })
        .unzip()
}

/// The root candidates' visits, their completedQ and the most visits any has: what the halving
/// ranks by and the improved policy is built from.
fn root_completed_q(
    tree: &Tree,
    at: &[usize],
    actor: usize,
    logits: &[f32],
    v_hat: f32,
) -> (Vec<f32>, u32) {
    let (visits, q) = root_stats(tree, at, actor);
    let cq = gumbel::completed_q(logits, &visits, &q, v_hat);
    (cq, visits.iter().copied().max().unwrap_or(0))
}

/// Gumbel root search (spec P2 Section 3): the first simulation evaluates the root, then
/// sequential halving over the candidates spends the other `simulations - 1`, each forced to its
/// candidate at the root and PUCT below. Returns the move and the improved policy over
/// `root_actions` (aligned with it).
fn gumbel_root<E: Evaluator>(
    sim: &mut Sim<'_, E>,
    root_actions: &[Action],
    draw: bool,
) -> (Action, Vec<f32>) {
    // A single simulation always ends at the unexpanded root: it evaluates the root alone.
    sim.run(1, |_| None);
    let actor = sim.obs.viewer as usize;
    let v_hat = sim
        .root_value
        .expect("the first simulation evaluates the root")[actor];
    let keys: Vec<u16> = root_actions.iter().map(|a| a.encode()).collect();
    let (at, prior): (Vec<usize>, Vec<f32>) = {
        let children = &sim.tree.nodes[ROOT as usize].children;
        keys.iter()
            .map(|&k| {
                let i = children.iter().position(|c| c.key == k).unwrap_or_else(|| {
                    panic!(
                        "the root move {:?} is not legal in the first sampled world",
                        Action::decode(k)
                    )
                });
                (i, children[i].prior)
            })
            .unzip()
    };
    let logits = gumbel::logits(&prior);
    // One draw per root action, in root-action order, right after the root's evaluation.
    let base: Vec<f32> = logits
        .iter()
        .map(|&l| if draw { l + gumbel::gumbel(sim.rng) } else { l })
        .collect();
    let mut remaining = gumbel::candidates(&base, &keys);
    // `search_tree` refuses an empty budget (its assert, as for PUCT) before it gets here, so this
    // cannot underflow.
    let n = sim.cfg.simulations - 1;
    let phases = gumbel::phases(remaining.len());
    let mut left = n;
    for phase in 0..phases {
        // The last phase, or a phase with one candidate left, spends the rest round-robin. Keeping
        // ceil(half) over ceil(log2 m) phases always leaves two for the last phase, so the second
        // test cannot fire; it stays for spec Section 3, step 3: "A phase with one candidate left
        // ends the halving and spends the rest on it".
        let last = phase + 1 == phases || remaining.len() == 1;
        let count = if last {
            left
        } else {
            // `.min(left)` caps a phase by the budget left, so the root gets exactly
            // `simulations - 1` visits and at tiny budgets some candidates get none (ledger S3).
            (gumbel::per_candidate(n, phases, remaining.len()) * remaining.len() as u32).min(left)
        };
        let order = &remaining;
        sim.run(count as usize, |k| Some(keys[order[k % order.len()]]));
        left -= count;
        if last {
            break;
        }
        let (cq, most) = root_completed_q(&sim.tree, &at, actor, &logits, v_hat);
        let mut kept = gumbel::ranked(&remaining, &base, &keys, &cq, most);
        kept.truncate(remaining.len().div_ceil(2));
        remaining = kept;
    }
    let (cq, most) = root_completed_q(&sim.tree, &at, actor, &logits, v_hat);
    let best = gumbel::ranked(&remaining, &base, &keys, &cq, most)[0];
    (
        root_actions[best],
        gumbel::improved_policy(&logits, &cq, most),
    )
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
    assert!(
        !(cfg.compound && matches!(cfg.root, RootMode::Gumbel { .. })),
        "compound moves are not searched with a Gumbel root (spec 3d-4 Section 4)"
    );
    let mut root_actions = Vec::new();
    search_moves(legal, cfg.offers, &mut root_actions);
    if root_actions.is_empty() {
        return Err("nothing to search: no legal action".into());
    }
    // 3d-4: at the searching player's own Main decision the moves are compounds, not bare trades,
    // read from the sampler's template (the player's own hand, the public bank and board).
    let compound_root = cfg.compound && obs.phase == Phase::Main && obs.actor == obs.viewer;
    let mut root_compounds = Vec::new();
    let early = if compound_root {
        let sampler = WorldSampler::new(obs, belief)?;
        compound::compounds(sampler.template(), &mut root_compounds);
        Some(sampler)
    } else {
        None
    };
    let mut root_keys = Vec::new();
    compound::move_keys(
        &root_actions,
        &root_compounds,
        compound_root,
        &mut root_keys,
    );
    if root_keys.len() == 1 {
        let (action, plan) = compound::move_of(root_keys[0], &root_compounds);
        let r = SearchResult {
            action,
            visits: vec![0; legal.len()],
            simulations: 0,
            nodes: 0,
            improved: None,
            plan,
            compound_visits: root_compounds.iter().map(|c| (c.actions(), 0)).collect(),
        };
        return Ok((r, Tree::new()));
    }
    let sampler = match early {
        Some(s) => s,
        None => WorldSampler::new(obs, belief)?,
    };
    let mut sim = Sim {
        obs,
        belief,
        sampler,
        eval,
        cfg,
        rng,
        tree: Tree::new(),
        buf: Vec::new(),
        scratch: Vec::new(),
        compounds: Vec::new(),
        keys: Vec::new(),
        batch: Vec::with_capacity(cfg.batch),
        done: 0,
        root_value: None,
    };
    let (action, plan, improved) = match cfg.root {
        RootMode::Puct => {
            sim.run(cfg.simulations as usize, |_| None);
            let key = most_visited(&sim.tree, &root_keys)?;
            let (action, plan) = compound::move_of(key, &root_compounds);
            (action, plan, None)
        }
        RootMode::Gumbel { draw } => {
            let (action, policy) = gumbel_root(&mut sim, &root_actions, draw);
            let improved: Vec<f32> = legal
                .iter()
                .map(|a| {
                    root_actions
                        .iter()
                        .position(|r| r == a)
                        .map_or(0.0, |i| policy[i])
                })
                .collect();
            (action, Vec::new(), Some(improved))
        }
    };
    let root = &sim.tree.nodes[ROOT as usize];
    let child_visits = |key: u16| {
        root.children
            .iter()
            .find(|c| c.key == key)
            .map_or(0, |c| c.visits)
    };
    let mut visits: Vec<u32> = legal.iter().map(|a| child_visits(a.encode())).collect();
    // spec 3d-4 Section 3: a compound's visits go to its first trade's entry of `legal`.
    let mut compound_visits = Vec::with_capacity(root_compounds.len());
    for c in &root_compounds {
        let v = child_visits(c.key());
        let i = legal
            .iter()
            .position(|&a| a == c.trades[0])
            .expect("a compound's first trade is legal at the root");
        visits[i] += v;
        compound_visits.push((c.actions(), v));
    }
    let (simulations, nodes) = (sim.done, sim.tree.nodes.len());
    Ok((
        SearchResult {
            action,
            visits,
            simulations,
            nodes,
            improved,
            plan,
            compound_visits,
        },
        sim.tree,
    ))
}

/// The move for `obs.viewer` among `legal` after `cfg.simulations` simulations, each in a world
/// sampled from `belief`: the most-visited (PUCT root) or sequential halving's winner (Gumbel
/// root); with `cfg.compound`, a compound's first step with the rest in `plan`. A single
/// searchable move is returned without searching.
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
