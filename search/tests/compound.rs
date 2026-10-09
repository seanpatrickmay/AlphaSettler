//! Compound build moves in search (spec docs/superpowers/specs/2026-10-08-compound-moves-design.md
//! Sections 2 and 6): the move set, the trades each target gets, the priors and compound edges.

mod common;

use common::*;
use settler_engine::rng::Rng;
use settler_engine::topology::{topo, NUM_EDGES, NUM_NODES};
use settler_engine::Resource::{Brick, Ore, Sheep, Wheat, Wood};
use settler_engine::*;
use settler_search::compound::{compound_key, compounds, move_priors, Compound, COMPOUND_BIT};
use settler_search::tree::{NodeKind, ROOT};
use settler_search::{
    search, search_tree, Belief, Eval, Evaluator, Leaf, RootMode, SearchConfig, SearchResult,
    UniformEvaluator,
};

fn mt(give: Resource, get: Resource) -> Action {
    Action::MaritimeTrade { give, get }
}

fn with_compound(simulations: u32) -> SearchConfig {
    SearchConfig {
        simulations,
        compound: true,
        ..SearchConfig::default()
    }
}

/// Every port node of `s`'s board.
fn port_nodes(s: &State) -> u64 {
    s.board
        .specific_port_nodes
        .iter()
        .fold(s.board.generic_port_nodes, |m, &x| m | x)
}

/// The lowest node on no port: every rate there is 4.
fn off_port(s: &State) -> u8 {
    (0..NUM_NODES as u8)
        .find(|&n| (port_nodes(s) >> n) & 1 == 0)
        .expect("a node off every port")
}

/// The lowest node of the 2:1 brick port.
fn brick_port(s: &State) -> u8 {
    let m = s.board.specific_port_nodes[Brick.index()];
    assert_ne!(m, 0, "the board has a brick port");
    m.trailing_zeros() as u8
}

/// Player 0 to act in Main on game `seed`'s board with one settlement (at `at(&board)`), no roads
/// and `hand`; player 1 holds `elsewhere` (to empty the bank of something); nobody has built
/// anything else.
fn lone_settlement(seed: u64, at: fn(&State) -> u8, hand: Hand, elsewhere: Hand) -> State {
    let s = State::new(seed, no_trades());
    let node = at(&s);
    edit(&s, no_trades(), |snap| {
        snap.players[0].settlements = 1u64 << node;
        give_cards(snap, 0, hand);
        give_cards(snap, 1, elsewhere);
        snap.setup_step = 8;
        snap.current = 0;
        snap.phase = Phase::Main;
    })
}

fn node_of(s: &State) -> u8 {
    s.players[0].settlements.trailing_zeros() as u8
}

/// The settlement's edges, in id order: the position's road targets (no road is built yet).
fn edges_at(s: &State) -> Vec<u8> {
    let n = node_of(s) as usize;
    (0..NUM_EDGES as u8)
        .filter(|&e| (topo().node_edge_mask[n] >> e) & 1 == 1)
        .collect()
}

fn compounds_of(s: &State) -> Vec<Compound> {
    let mut out = Vec::new();
    compounds(s, &mut out);
    out
}

/// A road compound on every edge of the settlement, each with `trades`.
fn roads(s: &State, trades: &[Action]) -> Vec<Compound> {
    edges_at(s)
        .into_iter()
        .map(|e| Compound {
            trades: trades.to_vec(),
            build: Action::BuildRoad(e),
        })
        .collect()
}

fn dev_card(trades: &[Action]) -> Compound {
    Compound {
        trades: trades.to_vec(),
        build: Action::BuyDev,
    }
}

/// Player 0's search of `s` from `seed` (belief and search streams).
fn run<E: Evaluator>(s: &State, eval: &mut E, cfg: &SearchConfig, seed: u64) -> SearchResult {
    let obs = s.observation(0);
    let b = Belief::from_observation(&obs, 64, seed);
    search(&obs, &s.legal_actions(), &b, eval, cfg, &mut Rng::new(seed)).unwrap()
}

/// Values ¼ each; a bank trade's prior is ten times any other move's.
struct TradeHeavy;

impl Evaluator for TradeHeavy {
    fn evaluate(&mut self, leaves: &[Leaf<'_>]) -> Vec<Eval> {
        leaves
            .iter()
            .map(|l| {
                let w: Vec<f32> = l
                    .legal
                    .iter()
                    .map(|a| {
                        if matches!(a, Action::MaritimeTrade { .. }) {
                            10.0
                        } else {
                            1.0
                        }
                    })
                    .collect();
                let total: f32 = w.iter().sum();
                Eval {
                    value: [0.25; 4],
                    prior: w.iter().map(|x| x / total).collect(),
                }
            })
            .collect()
    }
}

/// Wood 12 and wheat 2 off every port: a road is short a brick (one trade of 4 wood), the dev card
/// a sheep and an ore (two trades, the sheep first), the city three ore (more than two trades: none).
/// No road is built, so no settlement spot is reachable.
#[test]
fn each_target_gets_the_trades_the_rule_picks_and_none_needing_three() {
    for seed in 0..3 {
        let s = lone_settlement(seed, off_port, [12, 0, 0, 2, 0], [0; 5]);
        let mut want = roads(&s, &[mt(Wood, Brick)]);
        want.push(dev_card(&[mt(Wood, Sheep), mt(Wood, Ore)]));
        assert_eq!(compounds_of(&s), want, "seed {seed}");
    }
}

/// Wood 8, wheat 2, ore 1: the city is short two ore, both bought with wood (8 to spare, then 4).
/// With the bank down to one ore the second trade cannot be filled: no city compound. With no ore
/// in the bank, a dev card short one ore gets none either.
#[test]
fn a_target_the_bank_cannot_fill_gets_no_compound() {
    for seed in 0..3 {
        let s = lone_settlement(seed, off_port, [8, 0, 0, 2, 1], [0; 5]);
        let mut want = roads(&s, &[mt(Wood, Brick)]);
        want.push(Compound {
            trades: vec![mt(Wood, Ore), mt(Wood, Ore)],
            build: Action::BuildCity(node_of(&s)),
        });
        want.push(dev_card(&[mt(Wood, Sheep)]));
        assert_eq!(compounds_of(&s), want, "seed {seed}");
        // 19 ore: 1 in player 0's hand, 17 in player 1's, 1 in the bank.
        let short = lone_settlement(seed, off_port, [8, 0, 0, 2, 1], [0, 0, 0, 0, 17]);
        assert_eq!(short.bank[Ore.index()], 1);
        let mut want = roads(&short, &[mt(Wood, Brick)]);
        want.push(dev_card(&[mt(Wood, Sheep)]));
        assert_eq!(
            compounds_of(&short),
            want,
            "seed {seed}: one ore in the bank"
        );
        let empty = lone_settlement(seed, off_port, [4, 4, 1, 1, 0], [0, 0, 0, 0, 19]);
        assert_eq!(empty.bank[Ore.index()], 0);
        assert_eq!(
            compounds_of(&empty),
            vec![],
            "seed {seed}: no ore in the bank"
        );
    }
}

/// Off every port with wood 4, brick 4, sheep 1, wheat 1, the dev card is short one ore: wood and
/// brick both have 4 to spare and the tie goes to wood (the lower index); with brick 5, brick has
/// more to spare. The road is affordable now (no compound); the city is four cards short.
#[test]
fn the_most_cards_to_spare_give_and_ties_go_to_the_lower_resource() {
    for seed in 0..3 {
        let tie = lone_settlement(seed, off_port, [4, 4, 1, 1, 0], [0; 5]);
        assert_eq!(
            compounds_of(&tie),
            vec![dev_card(&[mt(Wood, Ore)])],
            "seed {seed}"
        );
        let more = lone_settlement(seed, off_port, [4, 5, 1, 1, 0], [0; 5]);
        assert_eq!(
            compounds_of(&more),
            vec![dev_card(&[mt(Brick, Ore)])],
            "seed {seed}"
        );
    }
}

/// On the 2:1 brick port, two brick buy the dev card's missing ore; off every port the same hand
/// (rate 4 everywhere) buys nothing. The road needs one of the two brick, so the one left over
/// cannot pay the rate: the road gets no compound.
#[test]
fn a_port_rate_makes_a_trade_and_a_needed_card_is_never_given() {
    for seed in 0..3 {
        let port = lone_settlement(seed, brick_port, [0, 2, 1, 1, 0], [0; 5]);
        assert_eq!(
            compounds_of(&port),
            vec![dev_card(&[mt(Brick, Ore)])],
            "seed {seed}"
        );
        let off = lone_settlement(seed, off_port, [0, 2, 1, 1, 0], [0; 5]);
        assert_eq!(compounds_of(&off), vec![], "seed {seed}");
    }
}

/// EndTurn 0.4, two road compounds sharing Sheep -> Wood (0.3 / 2 each), a dev card on Sheep ->
/// Brick (0.2); Sheep -> Ore starts nothing and is gone. Over the 0.9 left: 4/9, 1/6, 1/6, 2/9.
#[test]
fn a_shared_first_trade_splits_its_prior_and_the_move_set_renormalises() {
    let plain = [
        Action::EndTurn,
        mt(Sheep, Wood),
        mt(Sheep, Brick),
        mt(Sheep, Ore),
    ];
    let cs = [
        Compound {
            trades: vec![mt(Sheep, Wood)],
            build: Action::BuildRoad(1),
        },
        Compound {
            trades: vec![mt(Sheep, Wood)],
            build: Action::BuildRoad(2),
        },
        dev_card(&[mt(Sheep, Brick)]),
    ];
    let p = move_priors(&plain, &[0.4, 0.3, 0.2, 0.1], &cs);
    let want: [f32; 4] = [0.4 / 0.9, 0.15 / 0.9, 0.15 / 0.9, 0.2 / 0.9];
    assert_eq!(p.len(), want.len());
    for (got, want) in p.iter().zip(want) {
        assert!((got - want).abs() < 1e-6, "{p:?}");
    }
    // Nothing left with any prior: the move set is uniform.
    assert_eq!(
        move_priors(&[Action::EndTurn, mt(Sheep, Ore)], &[0.0, 1.0], &[]),
        vec![1.0]
    );
}

/// The +c move set of the wood-12 position: the end of the turn and the compounds, no bare trade
/// (spec Section 2). A uniform prior gives each of the 5 engine moves 1/5: each road compound 1/5
/// split k ways, the dev card's 1/5 (its first trade starts nothing else), the end of the turn
/// 1/5; renormalised over 3/5: 1/3, 1/(3k) each, 1/3. Without +c the root is the engine's moves.
#[test]
fn the_move_set_has_no_bare_trade_and_shared_first_trades_split_their_prior() {
    for seed in 0..3 {
        let s = lone_settlement(seed, off_port, [12, 0, 0, 2, 0], [0; 5]);
        let legal = s.legal_actions();
        assert_eq!(
            legal,
            vec![
                mt(Wood, Brick),
                mt(Wood, Sheep),
                mt(Wood, Wheat),
                mt(Wood, Ore),
                Action::EndTurn
            ]
        );
        let obs = s.observation(0);
        let b = Belief::from_observation(&obs, 64, seed);
        let (r, tree) = search_tree(
            &obs,
            &legal,
            &b,
            &mut UniformEvaluator,
            &with_compound(1),
            &mut Rng::new(seed),
        )
        .unwrap();
        let k = edges_at(&s).len();
        let mut want = vec![(Action::EndTurn.encode(), 1.0 / 3.0)];
        want.extend(
            edges_at(&s)
                .into_iter()
                .map(|e| (compound_key(Action::BuildRoad(e)), 1.0 / (3.0 * k as f32))),
        );
        want.push((compound_key(Action::BuyDev), 1.0 / 3.0));
        let root = &tree.nodes[ROOT as usize];
        let got: Vec<(u16, f32)> = root.children.iter().map(|c| (c.key, c.prior)).collect();
        assert_eq!(got.len(), want.len(), "seed {seed}");
        for (&(gk, gp), &(wk, wp)) in got.iter().zip(&want) {
            assert_eq!(gk, wk, "seed {seed}");
            assert!(
                (gp - wp).abs() < 1e-6,
                "seed {seed}: key {gk:#x} prior {gp}, want {wp}"
            );
        }
        assert!(root.children.iter().all(|c| c.key & COMPOUND_BIT != 0
            || !matches!(Action::decode(c.key), Some(Action::MaritimeTrade { .. }))));
        // One simulation evaluates the root alone; equal priors go to the lower key, a plain move.
        assert_eq!(
            (r.simulations, r.action, r.plan.clone()),
            (1, Action::EndTurn, vec![])
        );
        assert_eq!(r.compound_visits.len(), k + 1);
        let cfg = SearchConfig {
            simulations: 1,
            ..SearchConfig::default()
        };
        let (plain, tree) = search_tree(
            &obs,
            &legal,
            &b,
            &mut UniformEvaluator,
            &cfg,
            &mut Rng::new(seed),
        )
        .unwrap();
        let root = &tree.nodes[ROOT as usize];
        let keys: Vec<u16> = root.children.iter().map(|c| c.key).collect();
        assert_eq!(keys, legal.iter().map(|a| a.encode()).collect::<Vec<_>>());
        assert!(root.children.iter().all(|c| (c.prior - 0.2).abs() < 1e-6));
        assert!(plain.plan.is_empty() && plain.compound_visits.is_empty());
    }
}

/// 200 simulations of the wood-12 position under a prior that favours trades: the move priors are
/// the end of the turn 1/21, each road 10/(21k), the dev card 10/21, so the first descent takes the
/// dev card's compound (two trades, then the dev-card chance node). Every compound step is asserted
/// legal in its sampled world as the search applies it. The root's visits sum to 199 and sit on
/// the end of the turn and on the two first trades.
#[test]
fn compound_edges_apply_their_steps_legally_and_their_visits_land_on_their_first_trades() {
    for seed in 0..3 {
        let s = lone_settlement(seed, off_port, [12, 0, 0, 2, 0], [0; 5]);
        let obs = s.observation(0);
        let legal = s.legal_actions();
        let b = Belief::from_observation(&obs, 64, seed);
        let (r, tree) = search_tree(
            &obs,
            &legal,
            &b,
            &mut TradeHeavy,
            &with_compound(200),
            &mut Rng::new(seed),
        )
        .unwrap();
        assert_eq!(r.simulations, 200);
        assert_eq!(r.visits.iter().sum::<u32>(), 199, "seed {seed}");
        let k = edges_at(&s).len();
        assert_eq!(r.compound_visits.len(), k + 1);
        let on = |first: Action| -> u32 {
            r.compound_visits
                .iter()
                .filter(|(m, _)| m[0] == first)
                .map(|(_, v)| v)
                .sum()
        };
        let at = |a: Action| r.visits[legal.iter().position(|&x| x == a).unwrap()];
        assert_eq!(at(mt(Wood, Brick)), on(mt(Wood, Brick)), "seed {seed}");
        assert_eq!(at(mt(Wood, Sheep)), on(mt(Wood, Sheep)), "seed {seed}");
        assert_eq!(
            (at(mt(Wood, Wheat)), at(mt(Wood, Ore))),
            (0, 0),
            "no compound starts with them"
        );
        assert_eq!(
            at(Action::EndTurn) + on(mt(Wood, Brick)) + on(mt(Wood, Sheep)),
            199
        );
        assert_eq!(
            r.compound_visits.last().unwrap().0,
            vec![mt(Wood, Sheep), mt(Wood, Ore), Action::BuyDev]
        );
        let root = &tree.nodes[ROOT as usize];
        let dev = root
            .children
            .iter()
            .find(|c| c.key == compound_key(Action::BuyDev))
            .unwrap();
        assert!(dev.visits > 0, "seed {seed}");
        assert_eq!(
            tree.nodes[dev.node as usize].kind,
            NodeKind::Chance {
                action: Action::BuyDev
            }
        );
        assert_eq!(
            r,
            run(&s, &mut TradeHeavy, &with_compound(200), seed),
            "a seed replays"
        );
    }
}

/// Wood 4 off every port: four bank trades are legal, but the road would need the wood the trade
/// gives away and nothing else is within two trades, so the +c move set is the end of the turn
/// alone: returned without simulating, as any single move is.
#[test]
fn a_compound_root_with_one_move_is_returned_unsearched() {
    for seed in 0..3 {
        let s = lone_settlement(seed, off_port, [4, 0, 0, 0, 0], [0; 5]);
        assert_eq!(
            s.legal_actions().len(),
            5,
            "seed {seed}: four trades and the end of the turn"
        );
        let r = run(&s, &mut UniformEvaluator, &with_compound(50), seed);
        assert_eq!((r.action, r.simulations, r.nodes), (Action::EndTurn, 0, 0));
        assert!(
            r.visits.iter().all(|&v| v == 0) && r.plan.is_empty() && r.compound_visits.is_empty()
        );
        let plain = SearchConfig {
            simulations: 50,
            ..SearchConfig::default()
        };
        assert_eq!(run(&s, &mut UniformEvaluator, &plain, seed).simulations, 50);
    }
}

#[test]
#[should_panic(expected = "Gumbel")]
fn compound_moves_with_a_gumbel_root_are_refused() {
    let s = lone_settlement(1, off_port, [12, 0, 0, 2, 0], [0; 5]);
    let cfg = SearchConfig {
        root: RootMode::Gumbel { draw: false },
        ..with_compound(10)
    };
    run(&s, &mut UniformEvaluator, &cfg, 1);
}
