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
    assert_eq!(sorted(s.legal_actions()), vec![Action::Roll, Action::PlayKnight]);
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
fn road_building_with_no_legal_road_returns_to_main() {
    let mut s = blank(1, Phase::Main);
    s.players[0].dev_hand[RoadBuilding.index()] = 1;
    s.apply(Action::PlayRoadBuilding);
    assert_eq!(s.phase, Phase::Main);
    assert_eq!(s.players[0].dev_hand[RoadBuilding.index()], 0);

    let mut s = blank(1, Phase::Main);
    s.players[0].settlements = 1u64 << settler_engine::topology::topo().edge_nodes[0].0;
    s.players[0].roads = (1u128 << 14) - 1; // 14 roads: only one piece left
    s.players[0].dev_hand[RoadBuilding.index()] = 1;
    s.apply(Action::PlayRoadBuilding);
    assert_eq!(s.phase, Phase::RoadBuilding { roads_left: 1 });
    let a = s.legal_actions()[0];
    s.apply(a);
    assert_eq!(s.phase, Phase::Main);
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
    assert!(has(&s, Action::PlayYearOfPlenty(Resource::Wheat, Resource::Ore)));
    assert!(!has(&s, Action::PlayYearOfPlenty(Resource::Wheat, Resource::Wheat)));
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
