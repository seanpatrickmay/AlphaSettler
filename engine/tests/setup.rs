mod common;
use common::*;
use settler_engine::topology::topo;
use settler_engine::*;

#[test]
fn state_fits_in_512_bytes() {
    let size = std::mem::size_of::<State>();
    assert!(size <= 512, "State is {size} bytes");
}

#[test]
fn new_game_starts_in_setup() {
    let s = State::new(1, cfg());
    assert_eq!(s.phase, Phase::SetupSettlement);
    assert_eq!(s.current_actor(), 0);
    assert_eq!(s.robber, s.board.desert());
    assert_eq!(s.bank, [19; 5]);
    let legal = s.legal_actions();
    assert_eq!(legal.len(), 54);
    assert!(legal.iter().all(|a| matches!(a, Action::BuildSettlement(_))));
}

#[test]
fn same_seed_same_game() {
    assert_eq!(State::new(5, cfg()), State::new(5, cfg()));
    assert_ne!(State::new(5, cfg()).dev_deck, State::new(6, cfg()).dev_deck);
}

#[test]
fn settlement_then_road_at_that_node() {
    let mut s = State::new(1, cfg());
    let n = 20u8;
    s.apply(Action::BuildSettlement(n));
    assert_eq!(s.phase, Phase::SetupRoad { node: n });
    let expected: Vec<Action> = topo().node_edges[n as usize].iter().map(|&e| Action::BuildRoad(e)).collect();
    assert_eq!(sorted(s.legal_actions()), sorted(expected));
}

#[test]
fn distance_rule_in_setup() {
    let mut s = State::new(1, cfg());
    let n = 20u8;
    s.apply(Action::BuildSettlement(n));
    let e = topo().node_edges[n as usize][0];
    s.apply(Action::BuildRoad(e));
    assert_eq!(s.current_actor(), 1);
    let legal = s.legal_actions();
    assert!(!legal.contains(&Action::BuildSettlement(n)));
    for &nb in &topo().node_neighbors[n as usize] {
        assert!(!legal.contains(&Action::BuildSettlement(nb)));
    }
    assert_eq!(legal.len(), 54 - 1 - topo().node_neighbors[n as usize].len());
}

#[test]
fn snake_order() {
    let mut s = State::new(2, cfg());
    let mut order = Vec::new();
    while matches!(s.phase, Phase::SetupSettlement | Phase::SetupRoad { .. }) {
        if s.phase == Phase::SetupSettlement {
            order.push(s.current_actor());
        }
        let a = s.legal_actions()[0];
        s.apply(a);
    }
    assert_eq!(order, vec![0, 1, 2, 3, 3, 2, 1, 0]);
    assert_eq!(s.phase, Phase::PreRoll);
    assert_eq!(s.current, 0);
    assert_eq!(s.turn, 0);
}

#[test]
fn only_second_settlement_pays_out() {
    let mut s = State::new(3, cfg());
    let mut placed = 0;
    while matches!(s.phase, Phase::SetupSettlement | Phase::SetupRoad { .. }) {
        let a = s.legal_actions()[0];
        if let Action::BuildSettlement(n) = a {
            let p = s.current as usize;
            let before = s.players[p].hand;
            let mut expected = [0u8; 5];
            if placed >= 4 {
                for &t in &topo().node_tiles[n as usize] {
                    if let Some(r) = s.board.tile_resource[t as usize] {
                        expected[r.index()] += 1;
                    }
                }
            }
            s.apply(a);
            let mut delta = s.players[p].hand;
            hand_sub(&mut delta, &before);
            assert_eq!(delta, expected, "settlement #{placed}");
            placed += 1;
        } else {
            s.apply(a);
        }
    }
    for r in 0..5 {
        let held: u32 = s.players.iter().map(|p| p.hand[r] as u32).sum();
        assert_eq!(held + s.bank[r] as u32, 19);
    }
}

#[test]
#[should_panic(expected = "illegal action")]
fn illegal_action_panics() {
    let mut s = State::new(1, cfg());
    s.apply(Action::Roll);
}
