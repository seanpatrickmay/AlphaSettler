mod common;
use common::*;
use settler_engine::topology::topo;
use settler_engine::*;

fn roads_in(legal: &[Action]) -> Vec<u8> {
    let mut v: Vec<u8> = legal.iter().filter_map(|a| if let Action::BuildRoad(e) = a { Some(*e) } else { None }).collect();
    v.sort();
    v
}

#[test]
fn only_end_turn_without_resources() {
    let s = blank(1, Phase::Main);
    assert_eq!(s.legal_actions(), vec![Action::EndTurn]);
}

#[test]
fn road_requires_connection() {
    let mut s = blank(1, Phase::Main);
    let a = 20u8;
    s.players[0].settlements = 1u64 << a;
    give(&mut s, 0, ROAD_COST);
    let mut expected = topo().node_edges[a as usize].clone();
    expected.sort();
    assert_eq!(roads_in(&s.legal_actions()), expected);
    s.apply(Action::BuildRoad(expected[0]));
    assert_eq!(s.players[0].hand, [0; 5]);
    assert_eq!(s.bank, [19; 5]);
    assert_eq!(s.players[0].roads, 1u128 << expected[0]);
}

#[test]
fn roads_extend_from_roads() {
    let mut s = blank(1, Phase::Main);
    let nodes = path_nodes(20, 1, 0);
    let (a, b) = (nodes[0], nodes[1]);
    let ab = edge_between(a, b);
    s.players[0].settlements = 1u64 << a;
    s.players[0].roads = 1u128 << ab;
    give(&mut s, 0, ROAD_COST);
    let legal = roads_in(&s.legal_actions());
    for &e in topo().node_edges[b as usize].iter().chain(&topo().node_edges[a as usize]) {
        if e != ab {
            assert!(legal.contains(&e), "edge {e}");
        }
    }
}

#[test]
fn opponent_settlement_blocks_road_through_node() {
    let mut s = blank(1, Phase::Main);
    let nodes = path_nodes(20, 1, 0);
    let (a, b) = (nodes[0], nodes[1]);
    let ab = edge_between(a, b);
    s.players[0].settlements = 1u64 << a;
    s.players[0].roads = 1u128 << ab;
    s.players[1].settlements = 1u64 << b;
    give(&mut s, 0, ROAD_COST);
    let legal = roads_in(&s.legal_actions());
    for &e in &topo().node_edges[b as usize] {
        assert!(!legal.contains(&e), "edge {e} passes through opponent's settlement");
    }
    assert!(!legal.is_empty());
}

#[test]
fn settlement_needs_own_road_and_distance() {
    let mut s = blank(1, Phase::Main);
    let nodes = path_nodes(20, 2, 0);
    s.players[0].settlements = 1u64 << nodes[0];
    s.players[0].roads = mask128(&path_edges(&nodes));
    give(&mut s, 0, SETTLEMENT_COST);
    let settlements: Vec<Action> =
        s.legal_actions().into_iter().filter(|a| matches!(a, Action::BuildSettlement(_))).collect();
    assert_eq!(settlements, vec![Action::BuildSettlement(nodes[2])]);
    s.apply(Action::BuildSettlement(nodes[2]));
    assert_eq!(s.players[0].settlements, mask64(&[nodes[0], nodes[2]]));
    assert_eq!(s.bank, [19; 5]);
    assert_eq!(s.public_vp(0), 2);
}

#[test]
fn city_upgrades_settlement() {
    let mut s = blank(1, Phase::Main);
    s.players[0].settlements = 1u64 << 20;
    give(&mut s, 0, CITY_COST);
    assert!(s.legal_actions().contains(&Action::BuildCity(20)));
    s.apply(Action::BuildCity(20));
    assert_eq!(s.players[0].settlements, 0);
    assert_eq!(s.players[0].cities, 1u64 << 20);
    assert_eq!(s.public_vp(0), 2);
    assert_eq!(s.bank, [19; 5]);
}

#[test]
fn piece_limits() {
    let mut s = blank(1, Phase::Main);
    let nodes = path_nodes(20, 2, 0);
    s.players[0].settlements = mask64(&[nodes[0], 0, 5, 40, 50]);
    s.players[0].roads = mask128(&path_edges(&nodes));
    give(&mut s, 0, SETTLEMENT_COST);
    assert!(!s.legal_actions().iter().any(|a| matches!(a, Action::BuildSettlement(_))));

    let mut s = blank(1, Phase::Main);
    s.players[0].settlements = 1u64 << topo().edge_nodes[0].0;
    s.players[0].roads = (1u128 << 15) - 1;
    give(&mut s, 0, ROAD_COST);
    assert!(!s.legal_actions().iter().any(|a| matches!(a, Action::BuildRoad(_))));

    let mut s = blank(1, Phase::Main);
    s.players[0].cities = mask64(&[0, 10, 30, 50]);
    s.players[0].settlements = 1u64 << 20;
    give(&mut s, 0, CITY_COST);
    assert!(!s.legal_actions().iter().any(|a| matches!(a, Action::BuildCity(_))));
}

#[test]
fn end_turn_advances() {
    let mut s = blank(1, Phase::Main);
    s.players[0].dev_hand[0] = 1;
    s.players[0].dev_new[0] = 1;
    s.dev_played_this_turn = true;
    s.offers_this_turn = 2;
    s.apply(Action::EndTurn);
    assert_eq!(s.current, 1);
    assert_eq!(s.turn, 1);
    assert_eq!(s.phase, Phase::PreRoll);
    assert_eq!(s.players[0].dev_new, [0; 5]);
    assert!(!s.dev_played_this_turn);
    assert_eq!(s.offers_this_turn, 0);
}

#[test]
fn max_turns_ends_game_without_winner() {
    let c = GameConfig { max_turns: 1, ..cfg() };
    let mut s = blank_with(1, Phase::Main, c);
    s.apply(Action::EndTurn);
    assert_eq!(s.phase, Phase::GameOver { winner: None });
    assert!(s.legal_actions().is_empty());
}
