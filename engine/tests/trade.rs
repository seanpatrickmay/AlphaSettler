mod common;
use common::*;
use settler_engine::*;

fn offers(s: &State) -> Vec<Action> {
    s.legal_actions().into_iter().filter(|a| matches!(a, Action::OfferTrade { .. })).collect()
}

// Bundle 0 = one Wood, bundle 1 = one Brick.
const WOOD_FOR_BRICK: Action = Action::OfferTrade { give: 0, get: 1 };

#[test]
fn offers_require_holding_the_give_side() {
    let mut s = blank(1, Phase::Main);
    assert!(offers(&s).is_empty());
    give(&mut s, 0, [1, 0, 0, 0, 0]);
    let o = offers(&s);
    // give = 1 Wood; get = any bundle without Wood: 4 singles + 10 pairs.
    assert_eq!(o.len(), 14);
    assert!(o.iter().all(|a| matches!(a, Action::OfferTrade { give: 0, .. })));
}

#[test]
fn trade_size_cap() {
    let c = GameConfig { max_trade_cards: 1, ..cfg() };
    let mut s = blank_with(1, Phase::Main, c);
    give(&mut s, 0, [2, 0, 0, 0, 0]);
    assert_eq!(offers(&s).len(), 4);
}

#[test]
fn trading_disabled_with_zero_offers() {
    let c = GameConfig { max_offers_per_turn: 0, ..cfg() };
    let mut s = blank_with(1, Phase::Main, c);
    give(&mut s, 0, [1, 0, 0, 0, 0]);
    assert!(offers(&s).is_empty());
}

#[test]
fn accepted_trade_exchanges_cards() {
    let mut s = blank(1, Phase::Main);
    give(&mut s, 0, [1, 0, 0, 0, 0]);
    give(&mut s, 1, [0, 1, 0, 0, 0]);
    s.apply(WOOD_FOR_BRICK);
    assert_eq!(s.phase, Phase::TradeResponse);
    assert_eq!(s.current_actor(), 1);
    assert_eq!(sorted(s.legal_actions()), vec![Action::AcceptTrade, Action::RejectTrade]);
    s.apply(Action::AcceptTrade);
    assert_eq!(s.current_actor(), 2);
    assert_eq!(s.legal_actions(), vec![Action::RejectTrade]); // player 2 has no brick
    s.apply(Action::RejectTrade);
    s.apply(Action::RejectTrade);
    assert_eq!(s.phase, Phase::TradeConfirm);
    assert_eq!(s.current_actor(), 0);
    assert_eq!(sorted(s.legal_actions()), vec![Action::ConfirmTrade(1), Action::CancelTrade]);
    s.apply(Action::ConfirmTrade(1));
    assert_eq!(s.players[0].hand, [0, 1, 0, 0, 0]);
    assert_eq!(s.players[1].hand, [1, 0, 0, 0, 0]);
    assert_eq!(s.phase, Phase::Main);
    assert_eq!(s.trade, None);
    assert_eq!(s.offers_this_turn, 1);
}

#[test]
fn all_rejections_return_to_main() {
    let mut s = blank(1, Phase::Main);
    give(&mut s, 0, [1, 0, 0, 0, 0]);
    s.apply(WOOD_FOR_BRICK);
    for _ in 0..3 {
        s.apply(Action::RejectTrade);
    }
    assert_eq!(s.phase, Phase::Main);
    assert_eq!(s.trade, None);
    assert_eq!(s.players[0].hand, [1, 0, 0, 0, 0]);
}

#[test]
fn offerer_can_cancel_after_acceptance() {
    let mut s = blank(1, Phase::Main);
    give(&mut s, 0, [1, 0, 0, 0, 0]);
    give(&mut s, 1, [0, 1, 0, 0, 0]);
    s.apply(WOOD_FOR_BRICK);
    s.apply(Action::AcceptTrade);
    s.apply(Action::RejectTrade);
    s.apply(Action::RejectTrade);
    s.apply(Action::CancelTrade);
    assert_eq!(s.phase, Phase::Main);
    assert_eq!(s.players[0].hand, [1, 0, 0, 0, 0]);
    assert_eq!(s.players[1].hand, [0, 1, 0, 0, 0]);
}

#[test]
fn offer_cap_per_turn_resets_next_turn() {
    let c = GameConfig { max_offers_per_turn: 1, ..cfg() };
    let mut s = blank_with(1, Phase::Main, c);
    give(&mut s, 0, [1, 0, 0, 0, 0]);
    s.apply(WOOD_FOR_BRICK);
    for _ in 0..3 {
        s.apply(Action::RejectTrade);
    }
    assert!(offers(&s).is_empty());
    s.apply(Action::EndTurn);
    assert_eq!(s.offers_this_turn, 0);
}
