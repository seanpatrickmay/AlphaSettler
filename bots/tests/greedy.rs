mod common;
use common::*;
use settler_bots::greedy::{node_value, pips, road_value, GreedyBot};
use settler_bots::{make_bot, BOT_NAMES};
use settler_engine::topology::topo;
use settler_engine::*;

#[test]
fn pip_counts() {
    assert_eq!([0, 2, 3, 6, 8, 12].map(pips), [0, 1, 2, 5, 5, 1]);
}

#[test]
fn registered() {
    assert!(BOT_NAMES.contains(&"greedy"));
    assert_eq!(make_bot("greedy", 1).unwrap().name(), "greedy");
}

#[test]
fn setup_takes_the_most_valuable_spot() {
    let s = State::new(3, GameConfig::default());
    let obs = s.observation(0);
    let best = (0..54u8).map(|n| node_value(&obs, n)).max().unwrap();
    let a = act(&mut GreedyBot::new(0), &s);
    let Action::BuildSettlement(n) = a else {
        panic!("{a:?}")
    };
    assert_eq!(node_value(&obs, n), best);
}

#[test]
fn setup_road_heads_for_the_best_reachable_spot() {
    let mut s = State::new(3, GameConfig::default());
    let a = act(&mut GreedyBot::new(0), &s);
    s.apply(a);
    let obs = s.observation(0);
    let best = s
        .legal_actions()
        .iter()
        .map(|a| match a {
            Action::BuildRoad(e) => road_value(&obs, *e),
            _ => 0,
        })
        .max()
        .unwrap();
    let Action::BuildRoad(e) = act(&mut GreedyBot::new(0), &s) else {
        panic!()
    };
    assert_eq!(road_value(&obs, e), best);
}

#[test]
fn prefers_city_over_settlement() {
    let mut s = blank(1, Phase::Main);
    let n0 = 20u8;
    let path: Vec<u8> = {
        let t = topo();
        let a = t.node_neighbors[n0 as usize][0];
        let b = *t.node_neighbors[a as usize]
            .iter()
            .find(|&&x| x != n0)
            .unwrap();
        vec![n0, a, b]
    };
    s.players[0].settlements = 1u64 << n0;
    for w in path.windows(2) {
        let e = *topo().node_edges[w[0] as usize]
            .iter()
            .find(|&&e| topo().node_edges[w[1] as usize].contains(&e))
            .unwrap();
        s.players[0].roads |= 1u128 << e;
    }
    give(&mut s, 0, [1, 1, 1, 3, 3]);
    let legal = s.legal_actions();
    assert!(legal
        .iter()
        .any(|a| matches!(a, Action::BuildSettlement(_))));
    assert_eq!(act(&mut GreedyBot::new(0), &s), Action::BuildCity(n0));
}

#[test]
fn ends_turn_with_nothing_useful() {
    let s = blank(1, Phase::Main);
    assert_eq!(act(&mut GreedyBot::new(0), &s), Action::EndTurn);
}

#[test]
fn rejects_every_offer() {
    let mut s = blank(1, Phase::Main);
    give(&mut s, 0, [1, 0, 0, 0, 0]);
    give(&mut s, 1, [0, 1, 0, 0, 0]);
    s.apply(Action::OfferTrade { give: 0, get: 1 });
    assert_eq!(s.current_actor(), 1);
    assert_eq!(act(&mut GreedyBot::new(0), &s), Action::RejectTrade);
}

#[test]
fn never_offers_trades() {
    let mut s = blank(1, Phase::Main);
    give(&mut s, 0, [2, 2, 0, 0, 0]);
    assert!(s
        .legal_actions()
        .iter()
        .any(|a| matches!(a, Action::OfferTrade { .. })));
    let a = act(&mut GreedyBot::new(0), &s);
    assert!(!matches!(a, Action::OfferTrade { .. }), "{a:?}");
}

#[test]
fn discards_its_most_plentiful_resource() {
    let mut s = blank(1, Phase::PreRoll);
    give(&mut s, 0, [1, 5, 2, 0, 1]);
    s.apply_with(
        Action::Roll,
        Some(Chance::Roll {
            dice: (3, 4),
            discards: None,
        }),
        &mut NoEvents,
    );
    assert_eq!(s.phase, Phase::Discard);
    assert_eq!(
        act(&mut GreedyBot::new(0), &s),
        Action::Discard(Resource::Brick)
    );
}

#[test]
fn robber_avoids_own_tiles_and_hits_opponents() {
    let mut s = blank(1, Phase::MoveRobber);
    let t = topo();
    let by_pips = |tile: usize| pips(s.board.tile_number[tile]);
    let mut tiles: Vec<usize> = (0..19).filter(|&x| x as u8 != s.robber).collect();
    tiles.sort_by_key(|&x| std::cmp::Reverse(by_pips(x)));
    let (mine, theirs) = (tiles[0], tiles[1]);
    let my_node = *t.tile_nodes[mine]
        .iter()
        .find(|&&n| t.tile_node_mask[theirs] & (1u64 << n) == 0)
        .unwrap();
    s.players[0].settlements = 1u64 << my_node;
    let opp_node = *t.tile_nodes[theirs]
        .iter()
        .find(|&&n| t.tile_node_mask[mine] & (1u64 << n) == 0)
        .unwrap();
    s.players[1].cities = 1u64 << opp_node;
    let Action::MoveRobber(tile) = act(&mut GreedyBot::new(0), &s) else {
        panic!()
    };
    assert_eq!(
        t.tile_node_mask[tile as usize] & s.players[0].settlements,
        0
    );
    assert_ne!(t.tile_node_mask[tile as usize] & s.players[1].cities, 0);
}

#[test]
fn trades_surplus_with_the_bank_for_a_missing_card() {
    let mut s = blank(1, Phase::Main);
    s.players[0].settlements = 1u64 << 20; // target build: city (needs 2 wheat, 3 ore)
    give(&mut s, 0, [4, 0, 0, 0, 0]);
    assert_eq!(
        act(&mut GreedyBot::new(0), &s),
        Action::MaritimeTrade {
            give: Resource::Wood,
            get: Resource::Ore
        }
    );
}

#[test]
fn greedy_games_finish() {
    for seed in 0..20 {
        let mut bots: Vec<Box<dyn settler_bots::Bot>> =
            (0..4).map(|i| make_bot("greedy", i).unwrap()).collect();
        let mut s = State::new(seed, GameConfig::default());
        let mut steps = 0;
        while !s.is_over() {
            let legal = s.legal_actions();
            let actor = s.current_actor();
            let a = bots[actor as usize].act(&s.observation(actor), &legal);
            assert!(legal.contains(&a));
            s.apply(a);
            steps += 1;
            assert!(steps < 200_000);
        }
    }
}
