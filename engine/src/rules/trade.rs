//! Player-to-player trades: offer, then each opponent accepts or rejects in seat order,
//! then the offerer confirms with one acceptor or cancels.

use crate::action::{bundle, bundle_size, Action, NUM_BUNDLES};
use crate::apply::EventSink;
use crate::events::Event;
use crate::state::{PendingTrade, Phase, Response, State};
use crate::types::*;

pub fn legal_offers(s: &State, out: &mut Vec<Action>) {
    if s.offers_this_turn >= s.config.max_offers_per_turn {
        return;
    }
    let cap = s.config.max_trade_cards;
    let hand = s.players[s.current as usize].hand;
    for gi in 0..NUM_BUNDLES as u8 {
        if bundle_size(gi) > cap {
            continue;
        }
        let g = bundle(gi);
        if !covers(&hand, &g) {
            continue;
        }
        for wi in 0..NUM_BUNDLES as u8 {
            if bundle_size(wi) > cap {
                continue;
            }
            let w = bundle(wi);
            if (0..NUM_RESOURCES).all(|r| g[r] == 0 || w[r] == 0) {
                out.push(Action::OfferTrade { give: gi, get: wi });
            }
        }
    }
}

pub fn apply_offer<S: EventSink>(s: &mut State, give: u8, get: u8, sink: &mut S) {
    let (give, get) = (bundle(give), bundle(get));
    s.offers_this_turn += 1;
    s.trade = Some(PendingTrade {
        give,
        get,
        responses: [Response::Pending; NUM_PLAYERS],
        next_responder: (s.current + 1) % NUM_PLAYERS as PlayerId,
    });
    s.phase = Phase::TradeResponse;
    sink.emit(Event::TradeOffered { player: s.current, give, get });
}

pub fn legal_response(s: &State, out: &mut Vec<Action>) {
    let tr = s.trade.expect("TradeResponse without a trade");
    if covers(&s.players[tr.next_responder as usize].hand, &tr.get) {
        out.push(Action::AcceptTrade);
    }
    out.push(Action::RejectTrade);
}

pub fn apply_response<S: EventSink>(s: &mut State, accepted: bool, sink: &mut S) {
    let mut tr = s.trade.expect("TradeResponse without a trade");
    let r = tr.next_responder;
    tr.responses[r as usize] = if accepted { Response::Accepted } else { Response::Rejected };
    sink.emit(Event::TradeResponded { player: r, accepted });
    let next = (r + 1) % NUM_PLAYERS as PlayerId;
    if next != s.current {
        tr.next_responder = next;
        s.trade = Some(tr);
    } else if tr.responses.contains(&Response::Accepted) {
        s.trade = Some(tr);
        s.phase = Phase::TradeConfirm;
    } else {
        s.trade = None;
        s.phase = Phase::Main;
        sink.emit(Event::TradeCancelled { player: s.current });
    }
}

pub fn legal_confirm(s: &State, out: &mut Vec<Action>) {
    let tr = s.trade.expect("TradeConfirm without a trade");
    for (p, &r) in tr.responses.iter().enumerate() {
        if r == Response::Accepted {
            out.push(Action::ConfirmTrade(p as PlayerId));
        }
    }
    out.push(Action::CancelTrade);
}

pub fn apply_confirm<S: EventSink>(s: &mut State, partner: PlayerId, sink: &mut S) {
    let tr = s.trade.take().expect("TradeConfirm without a trade");
    let (p, q) = (s.current as usize, partner as usize);
    hand_sub(&mut s.players[p].hand, &tr.give);
    hand_add(&mut s.players[q].hand, &tr.give);
    hand_sub(&mut s.players[q].hand, &tr.get);
    hand_add(&mut s.players[p].hand, &tr.get);
    s.phase = Phase::Main;
    sink.emit(Event::TradeConfirmed {
        offerer: s.current,
        partner,
        offerer_gave: tr.give,
        partner_gave: tr.get,
    });
}

pub fn apply_cancel<S: EventSink>(s: &mut State, sink: &mut S) {
    s.trade = None;
    s.phase = Phase::Main;
    sink.emit(Event::TradeCancelled { player: s.current });
}
