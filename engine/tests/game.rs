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
    let stole = |v: PlayerId| {
        *g.log_for(v)
            .iter()
            .rev()
            .find(|e| matches!(e, Event::Stole { .. }))
            .unwrap()
    };
    let full = Event::Stole {
        thief: 0,
        victim: 1,
        resource: Some(Resource::Ore),
    };
    assert_eq!(stole(0), full);
    assert_eq!(stole(1), full);
    assert_eq!(
        stole(2),
        Event::Stole {
            thief: 0,
            victim: 1,
            resource: None
        }
    );
}

#[test]
fn bought_dev_cards_are_private() {
    let mut s = blank(1, Phase::Main);
    give(&mut s, 0, DEV_COST);
    let mut g = Game::from_state(s);
    g.apply_forced(Action::BuyDev, Some(Chance::Dev(DevCard::Monopoly)))
        .unwrap();
    assert_eq!(
        g.log_for(0).last(),
        Some(&Event::BoughtDev {
            player: 0,
            card: Some(DevCard::Monopoly)
        })
    );
    assert_eq!(
        g.log_for(3).last(),
        Some(&Event::BoughtDev {
            player: 0,
            card: None
        })
    );
}

#[test]
fn illegal_action_is_rejected_without_changing_state() {
    let mut g = Game::new(1, cfg());
    let before = *g.state();
    let err = g.apply(Action::Roll).unwrap_err();
    assert_eq!(
        err,
        IllegalAction {
            action: Action::Roll,
            phase: Phase::SetupSettlement
        }
    );
    assert_eq!(*g.state(), before);
    assert!(g.log_for(0).is_empty());
}

#[test]
fn actions_after_game_over_are_rejected() {
    let c = GameConfig {
        max_turns: 1,
        ..cfg()
    };
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
    g.apply(Action::PlayYearOfPlenty(Resource::Ore, Resource::Wood))
        .unwrap();
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

#[test]
fn out_of_range_actions_are_rejected_not_aliased() {
    let mut s = blank(1, Phase::Main);
    s.players[0].settlements = 1u64 << 0;
    give(&mut s, 0, CITY_COST);
    let mut g = Game::from_state(s);
    assert!(g.legal_actions().contains(&Action::BuildCity(0)));
    let before = *g.state();
    let err = g.apply(Action::BuildSettlement(54)).unwrap_err();
    assert_eq!(err.action, Action::BuildSettlement(54));
    assert_eq!(*g.state(), before);
    let err = g.apply(Action::StealFrom(4)).unwrap_err();
    assert_eq!(err.action, Action::StealFrom(4));
    assert_eq!(*g.state(), before);
    assert!(g.log_for(0).is_empty());
}

#[test]
fn observation_shows_config_and_turn_counters() {
    let c = GameConfig {
        vp_to_win: 8,
        max_offers_per_turn: 2,
        ..cfg()
    };
    let mut s = blank_with(1, Phase::Main, c);
    s.offers_this_turn = 2;
    let o = s.observation(3);
    assert_eq!(o.config, c);
    assert_eq!(o.offers_this_turn, 2);
}

#[test]
fn observation_shows_robber_return_after_knights() {
    let mut s = blank(1, Phase::PreRoll);
    s.players[0].dev_hand[DevCard::Knight.index()] = 2;
    let mut pre = s;
    pre.apply(Action::PlayKnight);
    let o = pre.observation(2);
    assert_eq!(o.phase, Phase::MoveRobber);
    assert_eq!(o.robber_return, Phase::PreRoll);
    assert_eq!(o.dev_cards_played[0][DevCard::Knight.index()], 1);
    let mut main = s;
    main.phase = Phase::Main;
    main.apply(Action::PlayKnight);
    assert_eq!(main.observation(2).robber_return, Phase::Main);
}

#[test]
fn observation_counts_played_dev_cards_by_kind() {
    let mut s = blank(1, Phase::Main);
    s.players[1].dev_hand[DevCard::Monopoly.index()] = 1;
    s.current = 1;
    s.apply(Action::PlayMonopoly(Resource::Wood));
    let o = s.observation(0);
    let mut expected = [[0u8; 5]; 4];
    expected[1][DevCard::Monopoly.index()] = 1;
    assert_eq!(o.dev_cards_played, expected);
    assert_eq!(s.players[1].dev_played, expected[1]);
}

#[test]
fn observation_shows_discards_owed_mid_discard() {
    let mut s = blank(1, Phase::PreRoll);
    give(&mut s, 0, [5, 4, 0, 0, 0]);
    give(&mut s, 2, [0, 0, 4, 4, 2]);
    roll(&mut s, 7);
    assert_eq!(s.phase, Phase::Discard);
    assert_eq!(s.observation(1).discard_remaining, [4, 0, 5, 0]);
    s.apply(Action::Discard(Resource::Wood));
    assert_eq!(s.observation(3).discard_remaining, [3, 0, 5, 0]);
}

#[test]
fn observation_shows_new_dev_card_counts() {
    let mut s = blank(1, Phase::Main);
    give(&mut s, 0, DEV_COST);
    s.apply(Action::BuyDev);
    assert_eq!(s.observation(2).new_dev_card_counts, [1, 0, 0, 0]);
    s.apply(Action::EndTurn);
    assert_eq!(s.observation(2).new_dev_card_counts, [0; 4]);
}

#[test]
fn observation_does_not_depend_on_hidden_information() {
    let mut a = blank(1, Phase::Main);
    give(&mut a, 1, [2, 0, 0, 0, 0]);
    give(&mut a, 2, [0, 2, 0, 0, 0]);
    a.players[1].dev_hand[DevCard::Knight.index()] = 1;
    a.players[2].dev_hand[DevCard::VictoryPoint.index()] = 1;
    a.players[2].dev_new[DevCard::VictoryPoint.index()] = 1;
    a.dev_deck_pos = 2;
    let mut b = a;
    b.seed ^= 0xFFFF;
    b.dev_deck.reverse();
    b.rng_steal = settler_engine::rng::Rng::new(7);
    b.rng_misc = settler_engine::rng::Rng::new(8);
    b.players[1].hand = [0, 2, 0, 0, 0];
    b.players[2].hand = [2, 0, 0, 0, 0];
    b.players[1].dev_hand = [0, 0, 0, 0, 1];
    b.players[2].dev_hand = [1, 0, 0, 0, 0];
    b.players[2].dev_new = [1, 0, 0, 0, 0];
    assert_ne!(a, b);
    assert_eq!(a.observation(0), b.observation(0));
    assert_eq!(a.observation(3), b.observation(3));
}
