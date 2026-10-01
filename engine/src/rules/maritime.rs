//! Trades with the bank at 4:1, or 3:1 / 2:1 through ports.

use crate::action::Action;
use crate::apply::EventSink;
use crate::events::Event;
use crate::state::State;
use crate::types::*;

pub fn legal(s: &State, out: &mut Vec<Action>) {
    let p = s.current as usize;
    let buildings = s.buildings(p);
    for give in Resource::ALL {
        let rate = s.board.maritime_rate(buildings, give);
        if s.players[p].hand[give.index()] < rate {
            continue;
        }
        for get in Resource::ALL {
            if get != give && s.bank[get.index()] >= 1 {
                out.push(Action::MaritimeTrade { give, get });
            }
        }
    }
}

pub fn apply<S: EventSink>(s: &mut State, give: Resource, get: Resource, sink: &mut S) {
    let p = s.current as usize;
    let rate = s.board.maritime_rate(s.buildings(p), give);
    let mut gave = [0u8; NUM_RESOURCES];
    gave[give.index()] = rate;
    let mut got = [0u8; NUM_RESOURCES];
    got[get.index()] = 1;
    hand_sub(&mut s.players[p].hand, &gave);
    hand_add(&mut s.bank, &gave);
    hand_sub(&mut s.bank, &got);
    hand_add(&mut s.players[p].hand, &got);
    sink.emit(Event::MaritimeTraded {
        player: s.current,
        gave,
        got,
    });
}
