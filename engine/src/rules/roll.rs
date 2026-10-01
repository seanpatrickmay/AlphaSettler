//! Dice, production (with the bank-shortage rule), sevens, and discards.

use crate::action::Action;
use crate::apply::{Chance, EventSink};
use crate::events::Event;
use crate::rng::{dice_for, Rng};
use crate::state::{Phase, State};
use crate::topology::{topo, NUM_TILES};
use crate::types::*;

pub fn apply_roll<S: EventSink>(s: &mut State, chance: Option<Chance>, sink: &mut S) {
    let ((d1, d2), forced_discards) = match chance {
        None => (dice_for(s.seed, s.turn), None),
        Some(Chance::Roll { dice, discards }) => (dice, discards),
        Some(c) => panic!("chance {c:?} does not match Roll"),
    };
    sink.emit(Event::Rolled { player: s.current, dice: (d1, d2) });
    let sum = d1 + d2;
    if sum != 7 {
        produce(s, sum, sink);
        s.phase = Phase::Main;
        return;
    }
    let limit = s.config.discard_limit as u32;
    let mut any = false;
    for p in 0..NUM_PLAYERS {
        let n = hand_total(&s.players[p].hand);
        if n > limit {
            s.players[p].discard_remaining = (n / 2) as u8;
            any = true;
        }
    }
    s.robber_return = Phase::Main;
    if any && s.config.catanatron_compat {
        random_discards(s, forced_discards, sink);
        s.phase = Phase::MoveRobber;
    } else if any {
        s.phase = Phase::Discard;
    } else {
        s.phase = Phase::MoveRobber;
    }
}

fn produce<S: EventSink>(s: &mut State, roll: u8, sink: &mut S) {
    let t = topo();
    let mut owed = [[0u8; NUM_RESOURCES]; NUM_PLAYERS];
    for tile in 0..NUM_TILES {
        if s.board.tile_number[tile] != roll || tile as u8 == s.robber {
            continue;
        }
        let Some(r) = s.board.tile_resource[tile] else { continue };
        let mask = t.tile_node_mask[tile];
        for p in 0..NUM_PLAYERS {
            let pl = &s.players[p];
            let n = (pl.settlements & mask).count_ones() + 2 * (pl.cities & mask).count_ones();
            owed[p][r.index()] += n as u8;
        }
    }
    for r in 0..NUM_RESOURCES {
        let total: u32 = (0..NUM_PLAYERS).map(|p| owed[p][r] as u32).sum();
        if total <= s.bank[r] as u32 {
            continue;
        }
        let claimants: Vec<usize> = (0..NUM_PLAYERS).filter(|&p| owed[p][r] > 0).collect();
        if claimants.len() == 1 {
            owed[claimants[0]][r] = s.bank[r];
        } else {
            for p in 0..NUM_PLAYERS {
                owed[p][r] = 0;
            }
        }
    }
    for p in 0..NUM_PLAYERS {
        if hand_total(&owed[p]) > 0 {
            hand_add(&mut s.players[p].hand, &owed[p]);
            hand_sub(&mut s.bank, &owed[p]);
            sink.emit(Event::Produced { player: p as PlayerId, resources: owed[p] });
        }
    }
}

/// One card drawn uniformly from a non-empty hand.
pub fn pick_card(hand: &Hand, rng: &mut Rng) -> Resource {
    let mut x = rng.below(hand_total(hand));
    for r in 0..NUM_RESOURCES {
        if x < hand[r] as u32 {
            return Resource::from_index(r);
        }
        x -= hand[r] as u32;
    }
    unreachable!("pick_card on an empty hand")
}

/// `k` cards drawn without replacement.
pub fn sample_cards(hand: &Hand, k: u8, rng: &mut Rng) -> Hand {
    let mut left = *hand;
    let mut out = [0u8; NUM_RESOURCES];
    for _ in 0..k {
        let r = pick_card(&left, rng).index();
        left[r] -= 1;
        out[r] += 1;
    }
    out
}

fn random_discards<S: EventSink>(s: &mut State, forced: Option<[Hand; NUM_PLAYERS]>, sink: &mut S) {
    for p in 0..NUM_PLAYERS {
        let k = s.players[p].discard_remaining;
        if k == 0 {
            continue;
        }
        let hand = s.players[p].hand;
        let discard = match forced {
            Some(f) => {
                assert_eq!(hand_total(&f[p]), k as u32, "forced discard for player {p} has wrong size");
                assert!(covers(&hand, &f[p]), "forced discard for player {p} exceeds hand");
                f[p]
            }
            None => sample_cards(&hand, k, &mut s.rng_misc),
        };
        for r in 0..NUM_RESOURCES {
            for _ in 0..discard[r] {
                sink.emit(Event::Discarded { player: p as PlayerId, resource: Resource::from_index(r) });
            }
        }
        hand_sub(&mut s.players[p].hand, &discard);
        hand_add(&mut s.bank, &discard);
        s.players[p].discard_remaining = 0;
    }
}

pub fn legal_discard(s: &State, out: &mut Vec<Action>) {
    let p = s.current_actor() as usize;
    for r in Resource::ALL {
        if s.players[p].hand[r.index()] > 0 {
            out.push(Action::Discard(r));
        }
    }
}

pub fn apply_discard<S: EventSink>(s: &mut State, r: Resource, sink: &mut S) {
    let p = s.current_actor() as usize;
    s.players[p].hand[r.index()] -= 1;
    s.bank[r.index()] += 1;
    s.players[p].discard_remaining -= 1;
    sink.emit(Event::Discarded { player: p as PlayerId, resource: r });
    if s.players.iter().all(|pl| pl.discard_remaining == 0) {
        s.phase = Phase::MoveRobber;
    }
}
