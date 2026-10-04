use settler_engine::topology::{topo, NUM_EDGES, NUM_TILES};
use settler_engine::*;
use settler_net::policy::*;

#[test]
fn entity_slots_are_the_engine_ids_and_the_pooled_head_has_the_rest() {
    assert_eq!(
        (
            Action::BuildSettlement(0).encode() as usize,
            Action::BuildCity(0).encode() as usize,
            Action::BuildRoad(0).encode() as usize,
            Action::MoveRobber(0).encode() as usize,
        ),
        (SETTLEMENT_BASE, CITY_BASE, ROAD_BASE, ROBBER_BASE)
    );
    assert_eq!(
        Action::BuildRoad(NUM_EDGES as u8 - 1).encode() as usize + 1,
        ROBBER_BASE
    );
    assert_eq!(
        Action::MoveRobber(NUM_TILES as u8 - 1).encode() as usize + 1,
        ENTITY_END
    );
    assert_eq!((ENTITY_END, OTHER_POLICY_LEN), (224, 466));
    let others: Vec<usize> = (0..OTHER_POLICY_LEN).map(other_slot).collect();
    let expected: Vec<usize> = (0..SETTLEMENT_BASE)
        .chain(ENTITY_END..ACTION_SPACE_SIZE)
        .collect();
    assert_eq!(others, expected);
}

#[test]
fn every_actor_maps_the_moves_one_to_one_onto_the_slots() {
    let moves: Vec<Action> = (0..ACTION_SPACE_SIZE as u16)
        .filter_map(Action::decode)
        .collect();
    for actor in 0..NUM_PLAYERS as PlayerId {
        let mut seen = vec![false; ACTION_SPACE_SIZE];
        for &a in &moves {
            let k = policy_id(a, actor);
            assert!(
                k < ACTION_SPACE_SIZE && !seen[k],
                "actor {actor}: {a:?} -> {k}"
            );
            seen[k] = true;
            if !matches!(a, Action::StealFrom(_) | Action::ConfirmTrade(_)) {
                assert_eq!(k, a.encode() as usize, "{a:?}");
            }
        }
    }
}

#[test]
fn steals_and_trade_confirmations_name_the_player_relative_to_the_actor() {
    for actor in 0..NUM_PLAYERS as PlayerId {
        for k in 0..NUM_PLAYERS {
            let p = ((actor as usize + k) % NUM_PLAYERS) as PlayerId;
            assert_eq!(policy_id(Action::StealFrom(p), actor), 224 + k);
            assert_eq!(policy_id(Action::ConfirmTrade(p), actor), 660 + k);
        }
    }
}

/// The same steal from every seat: a hand-built Steal phase whose only victim is the player after
/// the actor. Rotating the actor and the victim together leaves the steal's policy id unchanged.
#[test]
fn rotating_every_seat_leaves_a_steals_policy_id_unchanged() {
    let tile = 9usize;
    let mut ids = Vec::new();
    for actor in 0..NUM_PLAYERS as PlayerId {
        let victim = (actor + 1) % NUM_PLAYERS as PlayerId;
        let mut s = State::new(7, GameConfig::default());
        s.setup_step = 8;
        s.phase = Phase::Steal;
        s.return_phase = Phase::Main;
        s.current = actor;
        s.robber = tile as u8;
        s.players[victim as usize].settlements = 1u64 << topo().tile_nodes[tile][0];
        s.players[victim as usize].hand = [1, 0, 0, 0, 0];
        let legal = s.legal_actions();
        assert_eq!(legal, vec![Action::StealFrom(victim)], "actor {actor}");
        ids.push(policy_id(legal[0], actor));
    }
    assert_eq!(ids, vec![225; NUM_PLAYERS]);
}
