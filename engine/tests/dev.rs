mod common;
use common::*;
use settler_engine::apply::{Chance, NoEvents};
use settler_engine::types::DevCard::*;
use settler_engine::*;

fn has(s: &State, a: Action) -> bool {
    s.legal_actions().contains(&a)
}

#[test]
fn buy_dev_pays_and_draws_top_card() {
    let mut s = blank(1, Phase::Main);
    give(&mut s, 0, DEV_COST);
    let top = s.dev_deck[0];
    assert!(has(&s, Action::BuyDev));
    s.apply(Action::BuyDev);
    assert_eq!(s.players[0].dev_hand[top.index()], 1);
    assert_eq!(s.players[0].dev_new[top.index()], 1);
    assert_eq!(s.dev_deck_pos, 1);
    assert_eq!(s.players[0].hand, [0; 5]);
}

#[test]
fn forced_dev_draw() {
    let mut s = blank(1, Phase::Main);
    give(&mut s, 0, DEV_COST);
    s.apply_with(Action::BuyDev, Some(Chance::Dev(Monopoly)), &mut NoEvents);
    assert_eq!(s.players[0].dev_hand[Monopoly.index()], 1);
}

#[test]
#[should_panic(expected = "not left in deck")]
fn forced_dev_card_missing_panics() {
    let mut s = blank(1, Phase::Main);
    s.dev_deck = [Knight; 25];
    give(&mut s, 0, DEV_COST);
    s.apply_with(Action::BuyDev, Some(Chance::Dev(Monopoly)), &mut NoEvents);
}

#[test]
fn empty_deck_cannot_buy() {
    let mut s = blank(1, Phase::Main);
    s.dev_deck_pos = 25;
    give(&mut s, 0, DEV_COST);
    assert!(!has(&s, Action::BuyDev));
}

#[test]
fn cannot_play_card_bought_this_turn() {
    let mut s = blank(1, Phase::Main);
    give(&mut s, 0, DEV_COST);
    s.apply_with(Action::BuyDev, Some(Chance::Dev(Knight)), &mut NoEvents);
    assert!(!has(&s, Action::PlayKnight));
    s.apply(Action::EndTurn);
    assert_eq!(s.players[0].dev_new, [0; 5]);
    s.current = 0;
    s.phase = Phase::Main;
    assert!(has(&s, Action::PlayKnight));
}

#[test]
fn one_dev_card_per_turn() {
    let mut s = blank(1, Phase::Main);
    s.players[0].dev_hand[Knight.index()] = 1;
    s.players[0].dev_hand[Monopoly.index()] = 1;
    assert!(has(&s, Action::PlayKnight));
    assert!(has(&s, Action::PlayMonopoly(Resource::Ore)));
    s.apply(Action::PlayMonopoly(Resource::Ore));
    assert!(!has(&s, Action::PlayKnight));
}

#[test]
fn knight_before_roll_returns_to_preroll() {
    let mut s = blank(1, Phase::PreRoll);
    s.players[0].dev_hand[Knight.index()] = 1;
    assert_eq!(
        sorted(s.legal_actions()),
        vec![Action::Roll, Action::PlayKnight]
    );
    s.apply(Action::PlayKnight);
    assert_eq!(s.phase, Phase::MoveRobber);
    assert_eq!(s.players[0].knights_played, 1);
    let tile = s.legal_actions()[0];
    s.apply(tile);
    assert_eq!(s.phase, Phase::PreRoll);
    assert_eq!(s.legal_actions(), vec![Action::Roll]);
}

#[test]
fn knight_after_roll_returns_to_main() {
    let mut s = blank(1, Phase::Main);
    s.players[0].dev_hand[Knight.index()] = 1;
    s.apply(Action::PlayKnight);
    let tile = s.legal_actions()[0];
    s.apply(tile);
    assert_eq!(s.phase, Phase::Main);
}

#[test]
fn third_knight_takes_largest_army() {
    let mut s = blank(1, Phase::Main);
    s.players[0].dev_hand[Knight.index()] = 1;
    s.players[0].knights_played = 2;
    s.apply(Action::PlayKnight);
    assert_eq!(s.largest_army_owner, Some(0));
}

#[test]
fn road_building_places_two_free_roads() {
    let mut s = blank(1, Phase::Main);
    s.players[0].settlements = 1u64 << 20;
    s.players[0].dev_hand[RoadBuilding.index()] = 1;
    s.apply(Action::PlayRoadBuilding);
    assert_eq!(s.phase, Phase::RoadBuilding { roads_left: 2 });
    let a = s.legal_actions()[0];
    s.apply(a);
    assert_eq!(s.phase, Phase::RoadBuilding { roads_left: 1 });
    let b = s.legal_actions()[0];
    s.apply(b);
    assert_eq!(s.phase, Phase::Main);
    assert_eq!(s.players[0].roads.count_ones(), 2);
    assert_eq!(s.bank, [19; 5]);
}

#[test]
fn road_building_with_one_piece_left_grants_one_road() {
    for start in [Phase::Main, Phase::PreRoll] {
        let mut s = blank(1, start);
        s.players[0].settlements = 1u64 << settler_engine::topology::topo().edge_nodes[0].0;
        s.players[0].roads = (1u128 << 14) - 1; // 14 roads: only one piece left
        s.players[0].dev_hand[RoadBuilding.index()] = 1;
        s.apply(Action::PlayRoadBuilding);
        assert_eq!(s.phase, Phase::RoadBuilding { roads_left: 1 });
        let a = s.legal_actions()[0];
        s.apply(a);
        assert_eq!(s.players[0].roads.count_ones(), MAX_ROADS);
        assert_eq!(s.phase, start);
    }
}

#[test]
fn road_building_before_rolling_ends_early_without_an_edge() {
    // A settlement at v whose only free edge e leads to u, where every other edge is taken:
    // after the first free road there is nowhere for the second.
    let t = settler_engine::topology::topo();
    let (v, u) = t.edge_nodes[0];
    let e = 0u8;
    let others: Vec<u8> = [v, u]
        .iter()
        .flat_map(|&n| t.node_edges[n as usize].iter().copied())
        .filter(|&x| x != e)
        .collect();
    let mut s = blank(1, Phase::PreRoll);
    s.players[0].settlements = 1u64 << v;
    s.players[1].roads = mask128(&others);
    deal_dev(&mut s, 0, DevCard::RoadBuilding);
    s.apply(Action::PlayRoadBuilding);
    assert_eq!(s.phase, Phase::RoadBuilding { roads_left: 2 });
    assert_eq!(s.legal_actions(), vec![Action::BuildRoad(e)]);
    s.apply(Action::BuildRoad(e));
    assert_eq!(s.phase, Phase::PreRoll);
    assert_eq!(s.legal_actions(), vec![Action::Roll]);
}

#[test]
fn year_of_plenty() {
    let mut s = blank(1, Phase::Main);
    s.players[0].dev_hand[YearOfPlenty.index()] = 1;
    s.apply(Action::PlayYearOfPlenty(Resource::Wood, Resource::Ore));
    assert_eq!(s.players[0].hand, [1, 0, 0, 0, 1]);
    assert_eq!(s.bank, [18, 19, 19, 19, 18]);
}

#[test]
fn year_of_plenty_respects_bank() {
    let mut s = blank(1, Phase::Main);
    s.players[0].dev_hand[YearOfPlenty.index()] = 1;
    s.bank[Resource::Wheat.index()] = 1;
    assert!(has(
        &s,
        Action::PlayYearOfPlenty(Resource::Wheat, Resource::Ore)
    ));
    assert!(!has(
        &s,
        Action::PlayYearOfPlenty(Resource::Wheat, Resource::Wheat)
    ));
}

#[test]
fn monopoly_takes_all() {
    let mut s = blank(1, Phase::Main);
    s.players[0].dev_hand[Monopoly.index()] = 1;
    give(&mut s, 1, [3, 1, 0, 0, 0]);
    give(&mut s, 2, [2, 0, 0, 0, 0]);
    s.apply(Action::PlayMonopoly(Resource::Wood));
    assert_eq!(s.players[0].hand, [5, 0, 0, 0, 0]);
    assert_eq!(s.players[1].hand, [0, 1, 0, 0, 0]);
    assert_eq!(s.players[2].hand, [0; 5]);
}

#[test]
fn victory_point_cards_are_hidden_and_never_played() {
    let mut s = blank(1, Phase::Main);
    s.players[0].dev_hand[VictoryPoint.index()] = 2;
    assert_eq!(s.total_vp(0) - s.public_vp(0), 2);
    assert_eq!(s.legal_actions(), vec![Action::EndTurn]);
}

#[test]
fn pre_roll_offers_every_held_dev_card() {
    let mut s = after_setup(1);
    assert_eq!(s.phase, Phase::PreRoll);
    let p = s.current as usize;
    for c in [
        DevCard::Knight,
        DevCard::RoadBuilding,
        DevCard::YearOfPlenty,
        DevCard::Monopoly,
    ] {
        deal_dev(&mut s, p, c);
    }
    let legal = s.legal_actions();
    assert_eq!(legal[0], Action::Roll);
    for a in [
        Action::PlayKnight,
        Action::PlayRoadBuilding,
        Action::PlayYearOfPlenty(Resource::Wood, Resource::Ore),
        Action::PlayMonopoly(Resource::Sheep),
    ] {
        assert!(legal.contains(&a), "{a:?} missing from {legal:?}");
    }
    assert!(!legal.contains(&Action::BuyDev));
}

#[test]
fn year_of_plenty_before_rolling_stays_pre_roll_and_uses_the_turns_card() {
    let mut s = after_setup(2);
    let p = s.current as usize;
    deal_dev(&mut s, p, DevCard::YearOfPlenty);
    deal_dev(&mut s, p, DevCard::Monopoly);
    let before = s.players[p].hand;
    s.apply(Action::PlayYearOfPlenty(Resource::Wood, Resource::Brick));
    assert_eq!(s.phase, Phase::PreRoll);
    assert_eq!(s.players[p].hand[0], before[0] + 1);
    assert_eq!(s.players[p].hand[1], before[1] + 1);
    assert_eq!(s.legal_actions(), vec![Action::Roll]);
}

#[test]
fn monopoly_before_rolling() {
    let mut s = after_setup(3);
    let p = s.current as usize;
    let q = (p + 1) % NUM_PLAYERS;
    deal_dev(&mut s, p, DevCard::Monopoly);
    give(&mut s, q, [0, 0, 3, 0, 0]);
    let mine = s.players[p].hand[2];
    let theirs: u8 = (0..NUM_PLAYERS)
        .filter(|&o| o != p)
        .map(|o| s.players[o].hand[2])
        .sum();
    assert!(theirs >= 3);
    s.apply(Action::PlayMonopoly(Resource::Sheep));
    assert_eq!(s.phase, Phase::PreRoll);
    assert_eq!(s.players[p].hand[2], mine + theirs);
    for o in (0..NUM_PLAYERS).filter(|&o| o != p) {
        assert_eq!(s.players[o].hand[2], 0);
    }
}

#[test]
fn road_building_before_rolling_returns_to_pre_roll() {
    let mut s = after_setup(4);
    let p = s.current as usize;
    deal_dev(&mut s, p, DevCard::RoadBuilding);
    let roads = s.players[p].roads.count_ones();
    s.apply(Action::PlayRoadBuilding);
    assert_eq!(s.phase, Phase::RoadBuilding { roads_left: 2 });
    for _ in 0..2 {
        let a = s.legal_actions()[0];
        assert!(matches!(a, Action::BuildRoad(_)));
        s.apply(a);
    }
    assert_eq!(s.players[p].roads.count_ones(), roads + 2);
    assert_eq!(s.phase, Phase::PreRoll);
    assert_eq!(s.legal_actions(), vec![Action::Roll]);
}

#[test]
fn road_building_needs_a_placeable_road() {
    // No roads or buildings: nowhere to place a free road.
    let mut s = blank(1, Phase::Main);
    deal_dev(&mut s, 0, DevCard::RoadBuilding);
    assert!(!s.legal_actions().contains(&Action::PlayRoadBuilding));
}

#[test]
fn road_building_needs_a_road_piece() {
    let mut s = blank(1, Phase::Main);
    let nodes = path_nodes(0, 15, 0); // 15 edges = MAX_ROADS
    s.players[0].roads = mask128(&path_edges(&nodes));
    assert_eq!(s.players[0].roads.count_ones(), MAX_ROADS);
    deal_dev(&mut s, 0, DevCard::RoadBuilding);
    assert!(!s.legal_actions().contains(&Action::PlayRoadBuilding));
}
