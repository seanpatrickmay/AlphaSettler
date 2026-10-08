//! Gumbel root search (spec docs/superpowers/specs/2026-10-08-gumbel-root-design.md Sections 3
//! and 7): candidates, sequential halving's budget, the chosen move and the improved policy.

mod common;

use common::*;
use settler_engine::rng::Rng;
use settler_engine::*;
use settler_search::gumbel::MAX_CANDIDATES;
use settler_search::tree::ROOT;
use settler_search::{
    search, search_moves, search_tree, Belief, Eval, Evaluator, Leaf, RootMode, SearchConfig,
    SearchResult, UniformEvaluator,
};
use std::collections::HashSet;

fn gumbel(simulations: u32, draw: bool) -> SearchConfig {
    SearchConfig {
        simulations,
        root: RootMode::Gumbel { draw },
        ..SearchConfig::default()
    }
}

/// Player 0 in Main after setup holding four of every resource: about 30 searchable moves
/// (roads, settlements, a dev card, 20 bank trades, end turn).
fn rich(seed: u64) -> State {
    edit(&after_setup(seed, no_trades()), no_trades(), |snap| {
        give_cards(snap, 0, [4, 4, 4, 4, 4]);
        snap.phase = Phase::Main;
    })
}

/// Player 0 in Main with city resources and `vp_to_win` 3: either city wins on the spot.
fn city_wins(seed: u64) -> State {
    let config = GameConfig {
        vp_to_win: 3,
        ..no_trades()
    };
    let base = after_setup(seed, config);
    edit(&base, config, |snap| {
        give_cards(snap, 0, CITY_COST);
        snap.phase = Phase::Main;
    })
}

/// Values ¼ each; the prior grows with a move's place in `legal`, (i + 1) / Σ, so the last moves
/// are the likeliest.
struct Graded;

impl Evaluator for Graded {
    fn evaluate(&mut self, leaves: &[Leaf<'_>]) -> Vec<Eval> {
        leaves
            .iter()
            .map(|l| {
                let n = l.legal.len();
                let total = (n * (n + 1) / 2) as f32;
                Eval {
                    value: [0.25; 4],
                    prior: (1..=n).map(|i| i as f32 / total).collect(),
                }
            })
            .collect()
    }
}

/// Values ¼ each; all the prior on the first legal move, exactly none on the rest.
struct FirstOnly;

impl Evaluator for FirstOnly {
    fn evaluate(&mut self, leaves: &[Leaf<'_>]) -> Vec<Eval> {
        leaves
            .iter()
            .map(|l| {
                let mut prior = vec![0.0; l.legal.len()];
                prior[0] = 1.0;
                Eval {
                    value: [0.25; 4],
                    prior,
                }
            })
            .collect()
    }
}

/// Player 0's search of `s` from `seed` (belief and search streams).
fn run<E: Evaluator>(s: &State, eval: &mut E, cfg: &SearchConfig, seed: u64) -> SearchResult {
    let obs = s.observation(0);
    let b = Belief::from_observation(&obs, 64, seed);
    search(&obs, &s.legal_actions(), &b, eval, cfg, &mut Rng::new(seed)).unwrap()
}

fn root_moves(s: &State) -> Vec<Action> {
    let mut moves = Vec::new();
    search_moves(&s.legal_actions(), false, &mut moves);
    moves
}

fn sums_to_one(p: &[f32]) -> bool {
    p.iter().all(|x| x.is_finite() && *x >= 0.0)
        && (p.iter().map(|&x| x as f64).sum::<f64>() - 1.0).abs() < 1e-5
}

#[test]
fn sequential_halving_spends_the_budget_phase_by_phase_on_the_lowest_ids_when_all_else_ties() {
    let s = rich(1);
    let legal = s.legal_actions();
    assert!(legal.len() > MAX_CANDIDATES, "{} moves", legal.len());
    let r = run(&s, &mut UniformEvaluator, &gumbel(100, false), 1);
    assert_eq!(r.simulations, 100);
    // n = 99 over 4 phases: 16 candidates x 1, 8 x 3, 4 x 6, then the last 2 share the 35 left
    // (18 and 17). A uniform prior and equal values tie everything, so the candidates and every
    // halving keep the lowest action ids.
    let mut by_id: Vec<usize> = (0..legal.len()).collect();
    by_id.sort_by_key(|&i| legal[i].encode());
    let want_by_rank = [28, 27, 10, 10, 4, 4, 4, 4, 1, 1, 1, 1, 1, 1, 1, 1];
    for (rank, &i) in by_id.iter().enumerate() {
        let want = want_by_rank.get(rank).copied().unwrap_or(0);
        assert_eq!(r.visits[i], want, "rank {rank}: {:?}", legal[i]);
    }
    assert_eq!(r.visits.iter().sum::<u32>(), 99);
    assert_eq!(r.action, legal[by_id[0]], "the halving's winner");
}

#[test]
fn without_draws_the_candidates_are_the_sixteen_likeliest_moves() {
    let s = rich(2);
    let n = s.legal_actions().len();
    assert!(n > MAX_CANDIDATES, "{n} moves");
    let r = run(&s, &mut Graded, &gumbel(64, false), 2);
    let visited: Vec<bool> = r.visits.iter().map(|&v| v > 0).collect();
    let want: Vec<bool> = (0..n).map(|i| i >= n - MAX_CANDIDATES).collect();
    assert_eq!(visited, want);
    assert_eq!(r.visits.iter().sum::<u32>(), 63);
}

#[test]
fn with_nothing_visited_the_improved_policy_is_the_prior() {
    let s = rich(3);
    let legal = s.legal_actions();
    let n = legal.len();
    let r = run(&s, &mut Graded, &gumbel(1, false), 3);
    assert_eq!((r.simulations, r.visits.iter().sum::<u32>()), (1, 0));
    let improved = r.improved.unwrap();
    let total = (n * (n + 1) / 2) as f32;
    for (i, p) in improved.iter().enumerate() {
        assert!((p - (i + 1) as f32 / total).abs() < 1e-6, "move {i}: {p}");
    }
    assert_eq!(r.action, legal[n - 1], "g = 0: the likeliest move");
}

#[test]
fn a_move_that_wins_on_the_spot_is_chosen_and_draws_the_improved_policy() {
    for seed in 0..5 {
        let s = city_wins(seed);
        let legal = s.legal_actions();
        let r = run(&s, &mut UniformEvaluator, &gumbel(50, false), seed);
        let mut after = s;
        after.apply(r.action);
        assert_eq!(after.winner(), Some(0), "seed {seed}: {:?}", r.action);
        let improved = r.improved.unwrap();
        assert!(sums_to_one(&improved), "seed {seed}: {improved:?}");
        // The prior gives the two cities 2 of at least 3 moves; their value moves nearly all of
        // the improved policy onto them.
        let cities: f32 = legal
            .iter()
            .zip(&improved)
            .filter(|(a, _)| matches!(a, Action::BuildCity(_)))
            .map(|(_, p)| p)
            .sum();
        assert!(cities > 0.9, "seed {seed}: {cities}");
    }
}

#[test]
fn the_improved_policy_covers_the_legal_moves_and_is_zero_off_the_searched_ones() {
    let config = GameConfig::default();
    let s = edit(&after_setup(7, config), config, |snap| {
        give_cards(snap, 0, [1, 1, 1, 1, 1]);
        snap.phase = Phase::Main;
    });
    let legal = s.legal_actions();
    let is_offer = |a: &Action| matches!(a, Action::OfferTrade { .. });
    assert!(legal.iter().any(is_offer), "offers are legal here");
    let r = run(&s, &mut UniformEvaluator, &gumbel(60, false), 4);
    let improved = r.improved.unwrap();
    assert_eq!(improved.len(), legal.len());
    assert!(sums_to_one(&improved));
    for (a, (&p, &v)) in legal.iter().zip(improved.iter().zip(&r.visits)) {
        if is_offer(a) {
            assert_eq!((p, v), (0.0, 0), "{a:?}");
        }
    }
}

#[test]
fn tiny_budgets_spend_exactly_simulations_minus_one_root_visits() {
    for s in [rich(5), city_wins(1)] {
        let moves = root_moves(&s);
        for simulations in [2, 3, 5, 17, 40] {
            for draw in [false, true] {
                let r = run(
                    &s,
                    &mut UniformEvaluator,
                    &gumbel(simulations, draw),
                    simulations as u64,
                );
                let what = format!(
                    "{} moves, {simulations} simulations, draw {draw}",
                    moves.len()
                );
                assert_eq!(r.simulations, simulations, "{what}");
                assert_eq!(r.visits.iter().sum::<u32>(), simulations - 1, "{what}");
                assert!(sums_to_one(r.improved.as_ref().unwrap()), "{what}");
                assert!(moves.contains(&r.action), "{what}");
            }
        }
    }
}

#[test]
fn batches_larger_than_the_candidates_leave_no_virtual_loss() {
    for (s, batch) in [(city_wins(3), 8), (rich(6), 16)] {
        let obs = s.observation(0);
        let legal = s.legal_actions();
        let b = Belief::from_observation(&obs, 64, 6);
        let cfg = SearchConfig {
            batch,
            ..gumbel(300, true)
        };
        let (r, tree) = search_tree(
            &obs,
            &legal,
            &b,
            &mut UniformEvaluator,
            &cfg,
            &mut Rng::new(6),
        )
        .unwrap();
        assert_eq!((r.simulations, r.visits.iter().sum::<u32>()), (300, 299));
        for (i, n) in tree.nodes.iter().enumerate() {
            assert!(!n.awaiting_eval, "node {i} still awaits an evaluation");
            assert!(
                n.children.iter().all(|c| c.pending == 0),
                "node {i} keeps a virtual loss"
            );
        }
        let root: u32 = tree.nodes[ROOT as usize]
            .children
            .iter()
            .map(|c| c.visits)
            .sum();
        assert_eq!(root, 299);
    }
}

#[test]
fn a_prior_with_zeros_still_gives_a_proper_improved_policy() {
    let s = rich(7);
    let legal = s.legal_actions();
    let r = run(&s, &mut FirstOnly, &gumbel(40, false), 7);
    let improved = r.improved.unwrap();
    assert!(sums_to_one(&improved), "{improved:?}");
    assert!(r.visits[0] > 0, "the one likely move is a candidate");
    assert!(legal.contains(&r.action));
}

#[test]
fn the_same_seed_gives_the_same_gumbel_search_and_draws_change_the_candidates() {
    let s = rich(8);
    let drawn = gumbel(64, true);
    let a = run(&s, &mut UniformEvaluator, &drawn, 11);
    assert_eq!(a, run(&s, &mut UniformEvaluator, &drawn, 11));
    let visited = |r: &SearchResult| r.visits.iter().map(|&v| v > 0).collect::<Vec<bool>>();
    let with_draws: HashSet<Vec<bool>> = (0..6)
        .map(|seed| visited(&run(&s, &mut UniformEvaluator, &drawn, seed)))
        .collect();
    assert!(
        with_draws.len() > 1,
        "the draws never changed the candidates"
    );
    let plain = gumbel(64, false);
    let without: HashSet<Vec<bool>> = (0..6)
        .map(|seed| visited(&run(&s, &mut UniformEvaluator, &plain, seed)))
        .collect();
    assert_eq!(without.len(), 1, "without draws the prior alone picks them");
}

/// Values 0.9 for player 0 once `target` is on the board (a road or settlement it built), 0.1
/// before; the prior is uniform. One root move's child is then clearly the best.
struct Rates {
    target: Action,
}

impl Evaluator for Rates {
    fn evaluate(&mut self, leaves: &[Leaf<'_>]) -> Vec<Eval> {
        leaves
            .iter()
            .map(|l| {
                let me = &l.world.players[0];
                let built = match self.target {
                    Action::BuildRoad(e) => me.roads >> e & 1 == 1,
                    Action::BuildSettlement(n) => me.settlements >> n & 1 == 1,
                    other => panic!("{other:?} leaves no mark to rate"),
                };
                let mine = if built { 0.9 } else { 0.1 };
                let rest = (1.0 - mine) / 3.0;
                Eval {
                    value: [mine, rest, rest, rest],
                    prior: vec![1.0 / l.legal.len() as f32; l.legal.len()],
                }
            })
            .collect()
    }
}

#[test]
fn the_halving_winner_is_the_move_a_hand_built_evaluator_rates_best() {
    let s = rich(4);
    let legal = s.legal_actions();
    let mut by_id = root_moves(&s);
    by_id.sort_by_key(|a| a.encode());
    let target = *by_id
        .iter()
        .find(|a| matches!(a, Action::BuildRoad(_) | Action::BuildSettlement(_)))
        .expect("a road or settlement is legal");
    // With a uniform prior and no draws the first 16 ids are the candidates, and a tie goes to
    // the lower id: the target is a candidate, but the value alone must lift it above the first.
    let place = by_id.iter().position(|&a| a == target).unwrap();
    assert!(
        (1..MAX_CANDIDATES).contains(&place),
        "{target:?} is #{place}"
    );
    let r = run(&s, &mut Rates { target }, &gumbel(100, false), 4);
    assert_eq!(r.action, target);
    let improved = r.improved.unwrap();
    let on_target = legal.iter().position(|&a| a == target).unwrap();
    assert!(improved[on_target] > 0.9, "{improved:?}");
}

#[test]
fn a_zero_prior_move_is_still_a_candidate_when_the_root_has_sixteen_moves_or_fewer() {
    let s = city_wins(0);
    let legal = s.legal_actions();
    assert!(
        (2..=MAX_CANDIDATES).contains(&legal.len()),
        "{} moves",
        legal.len()
    );
    let r = run(&s, &mut FirstOnly, &gumbel(50, false), 0);
    assert!(
        r.visits.iter().all(|&v| v > 0),
        "every move is searched: {:?}",
        r.visits
    );
}

#[test]
fn a_single_searchable_move_is_returned_without_searching() {
    let s = after_setup(3, no_trades()); // PreRoll, no dev cards: only Roll
    assert_eq!(s.legal_actions(), vec![Action::Roll]);
    let r = run(&s, &mut UniformEvaluator, &gumbel(500, true), 0);
    assert_eq!(
        (r.action, r.simulations, r.improved),
        (Action::Roll, 0, None)
    );
}

#[test]
fn the_default_root_is_puct_and_reports_no_improved_policy() {
    assert_eq!(SearchConfig::default().root, RootMode::Puct);
    let cfg = SearchConfig {
        simulations: 50,
        ..SearchConfig::default()
    };
    assert_eq!(run(&rich(1), &mut UniformEvaluator, &cfg, 1).improved, None);
}
