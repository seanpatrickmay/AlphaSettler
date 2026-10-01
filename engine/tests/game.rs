mod common;
use common::*;
use settler_engine::apply::Chance;
use settler_engine::*;

#[test]
fn observation_hides_other_hands_and_dev_cards() {
    let mut s = blank(1, Phase::Main);
    give(&mut s, 0, [1, 1, 0, 0, 0]);
    give(&mut s, 1, [2, 0, 1, 0, 0]);
    s.players[1].dev_hand[DevCard::VictoryPoint.index()] = 1;
    s.players[1].cities = 1u64 << 20;
    let o = s.observation(0);
    assert_eq!(o.my_hand, [1, 1, 0, 0, 0]);
    assert_eq!(o.hand_counts, [2, 3, 0, 0]);
    assert_eq!(o.dev_card_counts, [0, 1, 0, 0]);
    assert_eq!(o.public_vp[1], 2);
    assert_eq!(o.my_dev_cards, [0; 5]);
    let o1 = s.observation(1);
    assert_eq!(o1.my_hand, [2, 0, 1, 0, 0]);
    assert_eq!(o1.my_dev_cards[DevCard::VictoryPoint.index()], 1);
}

fn steal_game() -> Game {
    let mut s = blank(1, Phase::MoveRobber);
    let t = lonely_tile(&s) as u8;
    s.players[1].settlements = 1u64 << exclusive_nodes(t as usize)[0];
    give(&mut s, 1, [0, 0, 0, 0, 1]);
    let mut g = Game::from_state(s);
    g.apply(Action::MoveRobber(t)).unwrap();
    g.apply(Action::StealFrom(1)).unwrap();
    g
}

#[test]
fn steals_are_private_to_thief_and_victim() {
    let g = steal_game();
    let stole = |v: PlayerId| *g.log_for(v).iter().rev().find(|e| matches!(e, Event::Stole { .. })).unwrap();
    let full = Event::Stole { thief: 0, victim: 1, resource: Some(Resource::Ore) };
    assert_eq!(stole(0), full);
    assert_eq!(stole(1), full);
    assert_eq!(stole(2), Event::Stole { thief: 0, victim: 1, resource: None });
}

#[test]
fn bought_dev_cards_are_private() {
    let mut s = blank(1, Phase::Main);
    give(&mut s, 0, DEV_COST);
    let mut g = Game::from_state(s);
    g.apply_forced(Action::BuyDev, Some(Chance::Dev(DevCard::Monopoly))).unwrap();
    assert_eq!(g.log_for(0).last(), Some(&Event::BoughtDev { player: 0, card: Some(DevCard::Monopoly) }));
    assert_eq!(g.log_for(3).last(), Some(&Event::BoughtDev { player: 0, card: None }));
}

#[test]
fn illegal_action_is_rejected_without_changing_state() {
    let mut g = Game::new(1, cfg());
    let before = *g.state();
    let err = g.apply(Action::Roll).unwrap_err();
    assert_eq!(err, IllegalAction { action: Action::Roll, phase: Phase::SetupSettlement });
    assert_eq!(*g.state(), before);
    assert!(g.log_for(0).is_empty());
}

#[test]
fn actions_after_game_over_are_rejected() {
    let c = GameConfig { max_turns: 1, ..cfg() };
    let mut g = Game::from_state(blank_with(1, Phase::Main, c));
    g.apply(Action::EndTurn).unwrap();
    assert!(g.state().is_over());
    assert!(g.legal_actions().is_empty());
    assert!(g.apply(Action::Roll).is_err());
    assert_eq!(g.log_for(0).last(), Some(&Event::GameOver { winner: None }));
}

#[test]
fn year_of_plenty_order_is_normalized() {
    let mut s = blank(1, Phase::Main);
    s.players[0].dev_hand[DevCard::YearOfPlenty.index()] = 1;
    let mut g = Game::from_state(s);
    g.apply(Action::PlayYearOfPlenty(Resource::Ore, Resource::Wood)).unwrap();
    assert_eq!(g.state().players[0].hand, [1, 0, 0, 0, 1]);
}

#[test]
fn full_game_log_is_replayable_from_seed() {
    let mut a = Game::new(42, cfg());
    let mut b = Game::new(42, cfg());
    for _ in 0..200 {
        if a.state().is_over() {
            break;
        }
        let act = a.legal_actions()[0];
        a.apply(act).unwrap();
        b.apply(act).unwrap();
    }
    assert_eq!(a.state(), b.state());
    assert_eq!(a.log_for(0), b.log_for(0));
}
