mod common;
use common::*;
use settler_engine::rules::awards::*;
use settler_engine::topology::topo;
use settler_engine::*;

#[test]
fn straight_path_length() {
    let nodes = path_nodes(0, 6, 0);
    assert_eq!(longest_road(mask128(&path_edges(&nodes)), 0), 6);
}

#[test]
fn opponent_building_splits_road() {
    let nodes = path_nodes(0, 6, 0);
    let roads = mask128(&path_edges(&nodes));
    assert_eq!(longest_road(roads, 1u64 << nodes[3]), 3);
    assert_eq!(longest_road(roads, 1u64 << nodes[1]), 5);
}

#[test]
fn ring_counts_every_edge() {
    let ring = topo().tile_nodes[9];
    let mut nodes = ring.to_vec();
    nodes.push(ring[0]);
    assert_eq!(longest_road(mask128(&path_edges(&nodes)), 0), 6);
}

/// Full recompute: every player's cached length, then the award.
fn update_all(s: &mut State) {
    for p in 0..NUM_PLAYERS {
        recompute_road_len(s, p);
    }
    assign_longest_road(s);
}

fn two_roads(len0: usize, len1: usize) -> State {
    let mut s = blank(1, Phase::Main);
    let n0 = path_nodes(0, len0, 0);
    let avoid = n0.iter().fold(0u64, |m, &n| {
        m | (1u64 << n) | topo().node_neighbor_mask[n as usize]
    });
    let n1 = path_nodes(53, len1, avoid);
    s.players[0].roads = mask128(&path_edges(&n0));
    s.players[1].roads = mask128(&path_edges(&n1));
    s
}

#[test]
fn award_needs_five() {
    let mut s = two_roads(4, 3);
    update_all(&mut s);
    assert_eq!(s.longest_road_owner, None);
    let mut s = two_roads(5, 3);
    update_all(&mut s);
    assert_eq!(s.longest_road_owner, Some(0));
    assert_eq!(s.public_vp(0), 2);
    assert_eq!(s.players[0].longest_road_len, 5);
}

#[test]
fn holder_keeps_award_on_tie() {
    let mut s = two_roads(5, 5);
    s.longest_road_owner = Some(1);
    update_all(&mut s);
    assert_eq!(s.longest_road_owner, Some(1));
}

#[test]
fn longer_road_takes_award() {
    let mut s = two_roads(6, 5);
    s.longest_road_owner = Some(1);
    update_all(&mut s);
    assert_eq!(s.longest_road_owner, Some(0));
}

#[test]
fn broken_holder_with_tied_challengers_leaves_award_unowned() {
    let mut s = two_roads(5, 5);
    s.longest_road_owner = Some(2);
    s.players[2].roads = 0; // holder's road was broken below everyone else
    update_all(&mut s);
    assert_eq!(s.longest_road_owner, None);
}

#[test]
fn settlement_breaking_road_moves_award() {
    let mut s = two_roads(6, 5);
    update_all(&mut s);
    assert_eq!(s.longest_road_owner, Some(0));
    // Player 1 reaches a middle node of player 0's road with a spur and settles there.
    // Cutting a 6-road at index 2, 3, or 4 leaves at most 4 on either side.
    let n0 = path_nodes(0, 6, 0);
    let t = topo();
    let (cut, spur) = (2..=4)
        .find_map(|i| {
            let n = n0[i];
            t.node_edges[n as usize]
                .iter()
                .copied()
                .find(|&e| s.players[0].roads & (1u128 << e) == 0)
                .map(|e| (n, e))
        })
        .expect("a middle node of the path with a third edge");
    s.current = 1;
    s.players[1].roads |= 1u128 << spur;
    give(&mut s, 1, SETTLEMENT_COST);
    s.apply(Action::BuildSettlement(cut));
    assert!(s.players[0].longest_road_len <= 4);
    assert_eq!(s.longest_road_owner, Some(1));
}

#[test]
fn largest_army() {
    let mut s = blank(1, Phase::Main);
    s.players[0].knights_played = 2;
    update_largest_army(&mut s, 0);
    assert_eq!(s.largest_army_owner, None);
    s.players[0].knights_played = 3;
    update_largest_army(&mut s, 0);
    assert_eq!(s.largest_army_owner, Some(0));
    s.players[1].knights_played = 3;
    update_largest_army(&mut s, 1);
    assert_eq!(s.largest_army_owner, Some(0));
    s.players[1].knights_played = 4;
    update_largest_army(&mut s, 1);
    assert_eq!(s.largest_army_owner, Some(1));
}

#[test]
fn building_to_target_vp_wins() {
    let c = GameConfig {
        vp_to_win: 3,
        ..cfg()
    };
    let mut s = blank_with(1, Phase::Main, c);
    let nodes = path_nodes(20, 2, 0);
    s.players[0].roads = mask128(&path_edges(&nodes));
    s.players[0].cities = 1u64 << 0;
    give(&mut s, 0, SETTLEMENT_COST);
    s.apply(Action::BuildSettlement(nodes[2]));
    assert_eq!(s.phase, Phase::GameOver { winner: Some(0) });
}

#[test]
fn players_win_only_on_their_own_turn() {
    let c = GameConfig {
        vp_to_win: 3,
        ..cfg()
    };
    let mut s = blank_with(1, Phase::Main, c);
    s.players[1].cities = 1u64 << 0;
    s.players[1].settlements = 1u64 << 30;
    s.players[0].settlements = 1u64 << 20;
    give(&mut s, 0, CITY_COST);
    s.apply(Action::BuildCity(20));
    assert_eq!(s.phase, Phase::Main);
    s.apply(Action::EndTurn);
    assert_eq!(s.phase, Phase::GameOver { winner: Some(1) });
}

#[test]
fn road_build_updates_longest_road() {
    let mut s = blank(1, Phase::Main);
    let nodes = path_nodes(0, 5, 0);
    let edges = path_edges(&nodes);
    s.players[0].settlements = 1u64 << nodes[0];
    s.players[0].roads = mask128(&edges[..4]);
    give(&mut s, 0, ROAD_COST);
    s.apply(Action::BuildRoad(edges[4]));
    assert_eq!(s.longest_road_owner, Some(0));
}
