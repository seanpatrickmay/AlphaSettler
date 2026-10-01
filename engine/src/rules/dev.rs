//! Development cards: buying, and playing knight, road building, year of plenty, monopoly.

use crate::action::{Action, PAIRS};
use crate::apply::{Chance, EventSink};
use crate::events::Event;
use crate::rules::{awards, build};
use crate::state::{Phase, State};
use crate::types::*;

/// Not yet played a dev card this turn, and holds one of `card` not bought this turn.
fn playable(s: &State, card: DevCard) -> bool {
    let pl = &s.players[s.current as usize];
    !s.dev_played_this_turn && pl.dev_hand[card.index()] > pl.dev_new[card.index()]
}

pub fn legal_knight(s: &State, out: &mut Vec<Action>) {
    if playable(s, DevCard::Knight) {
        out.push(Action::PlayKnight);
    }
}

pub fn legal_main_dev(s: &State, out: &mut Vec<Action>) {
    let pl = &s.players[s.current as usize];
    if covers(&pl.hand, &DEV_COST) && (s.dev_deck_pos as usize) < DEV_DECK_SIZE {
        out.push(Action::BuyDev);
    }
    legal_knight(s, out);
    if playable(s, DevCard::RoadBuilding) {
        out.push(Action::PlayRoadBuilding);
    }
    if playable(s, DevCard::YearOfPlenty) {
        for &(a, b) in &PAIRS {
            let ok = if a == b {
                s.bank[a as usize] >= 2
            } else {
                s.bank[a as usize] >= 1 && s.bank[b as usize] >= 1
            };
            if ok {
                out.push(Action::PlayYearOfPlenty(
                    Resource::from_index(a as usize),
                    Resource::from_index(b as usize),
                ));
            }
        }
    }
    if playable(s, DevCard::Monopoly) {
        for r in Resource::ALL {
            out.push(Action::PlayMonopoly(r));
        }
    }
}

pub fn legal_road_building(s: &State, out: &mut Vec<Action>) {
    build::push_free_roads(s, s.current as usize, out);
}

fn use_card<S: EventSink>(s: &mut State, card: DevCard, sink: &mut S) {
    s.players[s.current as usize].dev_hand[card.index()] -= 1;
    s.dev_played_this_turn = true;
    sink.emit(Event::PlayedDev {
        player: s.current,
        card,
    });
}

pub fn apply_buy_dev<S: EventSink>(s: &mut State, chance: Option<Chance>, sink: &mut S) {
    let p = s.current as usize;
    hand_sub(&mut s.players[p].hand, &DEV_COST);
    hand_add(&mut s.bank, &DEV_COST);
    let pos = s.dev_deck_pos as usize;
    match chance {
        None => {}
        Some(Chance::Dev(card)) => {
            let j = (pos..DEV_DECK_SIZE)
                .find(|&j| s.dev_deck[j] == card)
                .unwrap_or_else(|| panic!("forced dev card {card:?} not left in deck"));
            s.dev_deck.swap(pos, j);
        }
        Some(c) => panic!("chance {c:?} does not match BuyDev"),
    }
    let card = s.dev_deck[pos];
    s.dev_deck_pos += 1;
    s.players[p].dev_hand[card.index()] += 1;
    s.players[p].dev_new[card.index()] += 1;
    sink.emit(Event::BoughtDev {
        player: s.current,
        card: Some(card),
    });
}

pub fn apply_play_knight<S: EventSink>(s: &mut State, sink: &mut S) {
    let p = s.current as usize;
    let from_preroll = s.phase == Phase::PreRoll;
    use_card(s, DevCard::Knight, sink);
    s.players[p].knights_played += 1;
    awards::update_largest_army(s, p);
    s.robber_return = if from_preroll {
        Phase::PreRoll
    } else {
        Phase::Main
    };
    s.phase = Phase::MoveRobber;
}

fn road_building_phase(s: &State, left: u8) -> Phase {
    if left > 0 && build::has_free_road(s, s.current as usize) {
        Phase::RoadBuilding { roads_left: left }
    } else {
        Phase::Main
    }
}

pub fn apply_play_road_building<S: EventSink>(s: &mut State, sink: &mut S) {
    use_card(s, DevCard::RoadBuilding, sink);
    let pieces_left = MAX_ROADS - s.players[s.current as usize].roads.count_ones();
    s.phase = road_building_phase(s, pieces_left.min(2) as u8);
}

pub fn apply_road_building_road<S: EventSink>(s: &mut State, e: u8, roads_left: u8, sink: &mut S) {
    build::apply_build_road(s, e, true, sink);
    s.phase = road_building_phase(s, roads_left - 1);
}

pub fn apply_year_of_plenty<S: EventSink>(s: &mut State, a: Resource, b: Resource, sink: &mut S) {
    use_card(s, DevCard::YearOfPlenty, sink);
    let mut got = [0u8; NUM_RESOURCES];
    got[a.index()] += 1;
    got[b.index()] += 1;
    hand_add(&mut s.players[s.current as usize].hand, &got);
    hand_sub(&mut s.bank, &got);
    sink.emit(Event::YearOfPlentyTaken {
        player: s.current,
        resources: got,
    });
}

pub fn apply_monopoly<S: EventSink>(s: &mut State, r: Resource, sink: &mut S) {
    use_card(s, DevCard::Monopoly, sink);
    let p = s.current as usize;
    let mut amount = 0u8;
    for q in 0..NUM_PLAYERS {
        if q != p {
            amount += s.players[q].hand[r.index()];
            s.players[q].hand[r.index()] = 0;
        }
    }
    s.players[p].hand[r.index()] += amount;
    sink.emit(Event::MonopolyTaken {
        player: s.current,
        resource: r,
        amount,
    });
}
