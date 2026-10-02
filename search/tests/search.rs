mod common;

use common::*;
use settler_engine::rng::Rng;
use settler_engine::*;
use settler_search::tree::{NodeKind, ROOT};
use settler_search::{search, search_tree, visible_chance, Belief, SearchConfig, UniformEvaluator};

fn cfg(simulations: u32) -> SearchConfig {
    SearchConfig {
        simulations,
        ..SearchConfig::default()
    }
}

/// Player 0 in Main with city resources and `vp_to_win` set so a city wins on the spot.
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

#[test]
fn takes_a_winning_build() {
    for seed in 0..5 {
        let s = city_wins(seed);
        let obs = s.observation(0);
        let legal = s.legal_actions();
        let b = Belief::from_observation(&obs, 64, seed);
        let r = search(
            &obs,
            &legal,
            &b,
            &mut UniformEvaluator,
            &cfg(200),
            &mut Rng::new(seed),
        )
        .unwrap();
        let mut after = s;
        after.apply(r.action);
        assert_eq!(
            after.winner(),
            Some(0),
            "seed {seed}: {:?} does not win",
            r.action
        );
    }
}

#[test]
fn the_same_seed_gives_the_same_search() {
    let s = city_wins(1);
    let obs = s.observation(0);
    let legal = s.legal_actions();
    let b = Belief::from_observation(&obs, 64, 1);
    let run = |seed| {
        search(
            &obs,
            &legal,
            &b,
            &mut UniformEvaluator,
            &cfg(300),
            &mut Rng::new(seed),
        )
        .unwrap()
    };
    assert_eq!(run(9), run(9));
}

#[test]
fn root_visits_count_every_simulation_but_the_first() {
    let s = city_wins(2);
    let obs = s.observation(0);
    let legal = s.legal_actions();
    let b = Belief::from_observation(&obs, 64, 2);
    let r = search(
        &obs,
        &legal,
        &b,
        &mut UniformEvaluator,
        &cfg(250),
        &mut Rng::new(0),
    )
    .unwrap();
    assert_eq!(r.visits.len(), legal.len());
    assert_eq!(r.simulations, 250);
    assert_eq!(r.visits.iter().sum::<u32>(), 249);
}

#[test]
fn a_single_legal_move_is_played_without_searching() {
    let s = after_setup(3, no_trades()); // PreRoll, no dev cards: only Roll
    let obs = s.observation(0);
    let legal = s.legal_actions();
    assert_eq!(legal, vec![Action::Roll]);
    let b = Belief::from_observation(&obs, 8, 0);
    let r = search(
        &obs,
        &legal,
        &b,
        &mut UniformEvaluator,
        &cfg(500),
        &mut Rng::new(0),
    )
    .unwrap();
    assert_eq!((r.action, r.simulations), (Action::Roll, 0));
}

/// Player 0 before rolling, holding a knight: Roll and PlayKnight are both legal.
fn knight_before_roll(seed: u64, config: GameConfig) -> State {
    let base = after_setup(seed, config);
    edit(&base, config, |snap| give_dev(snap, 0, DevCard::Knight, 1))
}

#[test]
fn dice_branch_into_chance_outcomes() {
    let s = knight_before_roll(4, no_trades());
    let obs = s.observation(0);
    let legal = s.legal_actions();
    assert!(legal.contains(&Action::PlayKnight) && legal.contains(&Action::Roll));
    let b = Belief::from_observation(&obs, 64, 4);
    let (_, tree) = search_tree(
        &obs,
        &legal,
        &b,
        &mut UniformEvaluator,
        &cfg(400),
        &mut Rng::new(1),
    )
    .unwrap();
    let roll = tree.nodes[ROOT as usize]
        .children
        .iter()
        .find(|c| c.key == Action::Roll.encode())
        .unwrap();
    let chance = &tree.nodes[roll.node as usize];
    assert_eq!(
        chance.kind,
        NodeKind::Chance {
            action: Action::Roll
        }
    );
    assert!(
        chance.children.len() >= 6,
        "{} dice outcomes seen",
        chance.children.len()
    );
    for c in &chance.children {
        assert!((2..=12).contains(&c.key));
        assert_eq!(tree.nodes[c.node as usize].kind, NodeKind::Decision);
    }
}

#[test]
fn which_chance_outcomes_the_viewer_sees() {
    let s = after_setup(5, no_trades());
    assert!(visible_chance(&s, Action::Roll, 2));
    let mut buying = s;
    buying.current = 1;
    assert!(visible_chance(&buying, Action::BuyDev, 1));
    assert!(!visible_chance(&buying, Action::BuyDev, 0));
    assert!(
        visible_chance(&buying, Action::StealFrom(0), 0),
        "the victim sees what was taken"
    );
    assert!(
        visible_chance(&buying, Action::StealFrom(2), 1),
        "the thief sees what was taken"
    );
    assert!(
        !visible_chance(&buying, Action::StealFrom(2), 0),
        "a bystander does not"
    );
}

#[test]
fn reaching_the_turn_cap_inside_the_tree_is_a_draw_not_a_crash() {
    let config = GameConfig {
        max_turns: 1,
        ..no_trades()
    };
    let s = knight_before_roll(6, config);
    let obs = s.observation(0);
    let legal = s.legal_actions();
    let b = Belief::from_observation(&obs, 32, 6);
    let r = search(
        &obs,
        &legal,
        &b,
        &mut UniformEvaluator,
        &cfg(300),
        &mut Rng::new(2),
    )
    .unwrap();
    assert!(legal.contains(&r.action));
}
