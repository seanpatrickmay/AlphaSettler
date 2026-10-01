//! Applying actions. Actions must be legal (see `legal_actions`); illegal ones panic or corrupt.

use crate::action::Action;
use crate::events::Event;
use crate::rules::{awards, build, dev, maritime, robber, roll, setup, trade};
use crate::state::{Phase, State};
use crate::types::{covers, hand_total, DevCard, Hand, Resource, NUM_PLAYERS};

/// A random outcome supplied by the caller instead of drawn from the state's streams.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Chance {
    /// `discards` is only used in `catanatron_compat` mode on a 7.
    Roll {
        dice: (u8, u8),
        discards: Option<[Hand; NUM_PLAYERS]>,
    },
    Steal(Resource),
    Dev(DevCard),
}

pub trait EventSink {
    fn emit(&mut self, e: Event);
}

/// Discards events; compiles away in search.
pub struct NoEvents;

impl EventSink for NoEvents {
    #[inline(always)]
    fn emit(&mut self, _e: Event) {}
}

impl EventSink for Vec<Event> {
    fn emit(&mut self, e: Event) {
        self.push(e);
    }
}

const DISCARD_OWES_NONE: [&str; NUM_PLAYERS] = [
    "forced discard for player 0 who owes none",
    "forced discard for player 1 who owes none",
    "forced discard for player 2 who owes none",
    "forced discard for player 3 who owes none",
];
const DISCARD_WRONG_SIZE: [&str; NUM_PLAYERS] = [
    "forced discard for player 0 has wrong size",
    "forced discard for player 1 has wrong size",
    "forced discard for player 2 has wrong size",
    "forced discard for player 3 has wrong size",
];
const DISCARD_EXCEEDS_HAND: [&str; NUM_PLAYERS] = [
    "forced discard for player 0 exceeds hand",
    "forced discard for player 1 exceeds hand",
    "forced discard for player 2 exceeds hand",
    "forced discard for player 3 exceeds hand",
];

/// Whether forcing outcome `c` for action `a` is possible in `s`, checked before anything
/// mutates. Assumes `a` is legal in `s`; `Err` carries the reason.
pub fn check_chance(s: &State, a: Action, c: Chance) -> Result<(), &'static str> {
    match (a, c) {
        (Action::Roll, Chance::Roll { dice, discards }) => {
            let (d1, d2) = dice;
            if !(1..=6).contains(&d1) || !(1..=6).contains(&d2) {
                return Err("forced dice out of range");
            }
            let Some(forced) = discards else {
                return Ok(());
            };
            if d1 + d2 != 7 {
                return Err("forced discards given but the roll is not a 7");
            }
            if !s.config.catanatron_compat {
                return Err("forced discards given but catanatron_compat is off");
            }
            let owed: [u8; NUM_PLAYERS] = std::array::from_fn(|p| roll::discard_owed(s, p));
            if owed.iter().all(|&k| k == 0) {
                return Err("forced discards given but no player must discard");
            }
            for p in 0..NUM_PLAYERS {
                let n = hand_total(&forced[p]);
                if owed[p] == 0 && n != 0 {
                    return Err(DISCARD_OWES_NONE[p]);
                }
                if n != owed[p] as u32 {
                    return Err(DISCARD_WRONG_SIZE[p]);
                }
                if !covers(&s.players[p].hand, &forced[p]) {
                    return Err(DISCARD_EXCEEDS_HAND[p]);
                }
            }
            Ok(())
        }
        (Action::StealFrom(v), Chance::Steal(r)) => match s.players.get(v as usize) {
            Some(pl) if pl.hand[r.index()] > 0 => Ok(()),
            _ => Err("forced steal of a resource the victim has none of"),
        },
        (Action::BuyDev, Chance::Dev(card)) => {
            if s.dev_deck[s.dev_deck_pos as usize..].contains(&card) {
                Ok(())
            } else {
                Err("forced dev card not left in deck")
            }
        }
        (Action::Roll | Action::StealFrom(_) | Action::BuyDev, _) => {
            Err("chance kind does not match the action")
        }
        _ => Err("action does not take a chance outcome"),
    }
}

impl State {
    pub fn apply(&mut self, a: Action) {
        self.apply_with(a, None, &mut NoEvents);
    }

    /// Apply `a`, forcing the random outcome to `chance` if given. Panics if `chance` is
    /// impossible (see `check_chance`), before anything changes.
    pub fn apply_with<S: EventSink>(&mut self, a: Action, chance: Option<Chance>, sink: &mut S) {
        if let Some(c) = chance {
            if let Err(reason) = check_chance(self, a, c) {
                panic!("impossible chance {c:?} for {a:?}: {reason}");
            }
        }
        match (self.phase, a) {
            (Phase::SetupSettlement, Action::BuildSettlement(n)) => {
                setup::apply_settlement(self, n, sink)
            }
            (Phase::SetupRoad { node }, Action::BuildRoad(e)) => {
                setup::apply_road(self, node, e, sink)
            }
            (Phase::PreRoll, Action::Roll) => roll::apply_roll(self, chance, sink),
            (Phase::Discard, Action::Discard(r)) => roll::apply_discard(self, r, sink),
            (Phase::MoveRobber, Action::MoveRobber(t)) => robber::apply_move_robber(self, t, sink),
            (Phase::Steal, Action::StealFrom(v)) => robber::apply_steal(self, v, chance, sink),
            (Phase::Main, Action::BuildRoad(e)) => build::apply_build_road(self, e, false, sink),
            (Phase::Main, Action::BuildSettlement(n)) => {
                build::apply_build_settlement(self, n, sink)
            }
            (Phase::Main, Action::BuildCity(n)) => build::apply_build_city(self, n, sink),
            (Phase::Main, Action::EndTurn) => build::apply_end_turn(self, sink),
            (Phase::PreRoll | Phase::Main, Action::PlayKnight) => {
                dev::apply_play_knight(self, sink)
            }
            (Phase::Main, Action::BuyDev) => dev::apply_buy_dev(self, chance, sink),
            (Phase::PreRoll | Phase::Main, Action::PlayRoadBuilding) => {
                dev::apply_play_road_building(self, sink)
            }
            (Phase::PreRoll | Phase::Main, Action::PlayYearOfPlenty(a, b)) => {
                dev::apply_year_of_plenty(self, a, b, sink)
            }
            (Phase::PreRoll | Phase::Main, Action::PlayMonopoly(r)) => {
                dev::apply_monopoly(self, r, sink)
            }
            (Phase::RoadBuilding { roads_left }, Action::BuildRoad(e)) => {
                dev::apply_road_building_road(self, e, roads_left, sink)
            }
            (Phase::Main, Action::MaritimeTrade { give, get }) => {
                maritime::apply(self, give, get, sink)
            }
            (Phase::Main, Action::OfferTrade { give, get }) => {
                trade::apply_offer(self, give, get, sink)
            }
            (Phase::TradeResponse, Action::AcceptTrade) => trade::apply_response(self, true, sink),
            (Phase::TradeResponse, Action::RejectTrade) => trade::apply_response(self, false, sink),
            (Phase::TradeConfirm, Action::ConfirmTrade(p)) => trade::apply_confirm(self, p, sink),
            (Phase::TradeConfirm, Action::CancelTrade) => trade::apply_cancel(self, sink),
            (phase, a) => panic!("illegal action {a:?} in phase {phase:?} (chance {chance:?})"),
        }
        awards::check_win(self, sink);
    }
}
