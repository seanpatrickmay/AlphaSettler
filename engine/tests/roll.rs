mod common;
use common::*;
use settler_engine::apply::{Chance, NoEvents};
use settler_engine::rng::dice_for;
use settler_engine::topology::topo;
use settler_engine::*;

#[test]
fn preroll_offers_roll() {
    let s = blank(1, Phase::PreRoll);
    assert_eq!(s.legal_actions(), vec![Action::Roll]);
}

#[test]
fn unforced_roll_uses_seed_and_turn() {
    let mut s = after_setup(4);
    let mut log: Vec<Event> = Vec::new();
    s.apply_with(Action::Roll, None, &mut log);
    assert_eq!(log[0], Event::Rolled { player: 0, dice: dice_for(4, 0) });
}

#[test]
fn settlement_collects_one_city_two() {
    let mut s = blank(3, Phase::PreRoll);
    let t = lonely_tile(&s);
    let r = s.board.tile_resource[t].unwrap();
    let ex = exclusive_nodes(t);
    s.players[0].settlements = 1u64 << ex[0];
    s.players[1].cities = 1u64 << ex[1];
    let n = s.board.tile_number[t];
    roll(&mut s, n);
    assert_eq!(s.players[0].hand[r.index()], 1);
    assert_eq!(s.players[1].hand[r.index()], 2);
    assert_eq!(s.bank[r.index()], 16);
    assert_eq!(s.phase, Phase::Main);
}

#[test]
fn robber_blocks_production() {
    let mut s = blank(3, Phase::PreRoll);
    let t = lonely_tile(&s);
    s.players[0].settlements = 1u64 << exclusive_nodes(t)[0];
    s.robber = t as u8;
    let n = s.board.tile_number[t];
    roll(&mut s, n);
    assert_eq!(hand_total(&s.players[0].hand), 0);
}

#[test]
fn bank_shortage_with_two_claimants_pays_nobody() {
    let mut s = blank(3, Phase::PreRoll);
    let t = lonely_tile(&s);
    let r = s.board.tile_resource[t].unwrap().index();
    let ex = exclusive_nodes(t);
    s.players[0].settlements = 1u64 << ex[0];
    s.players[1].settlements = 1u64 << ex[1];
    s.bank[r] = 1;
    let n = s.board.tile_number[t];
    roll(&mut s, n);
    assert_eq!(s.players[0].hand[r], 0);
    assert_eq!(s.players[1].hand[r], 0);
    assert_eq!(s.bank[r], 1);
}

#[test]
fn bank_shortage_with_one_claimant_pays_what_is_left() {
    let mut s = blank(3, Phase::PreRoll);
    let t = lonely_tile(&s);
    let r = s.board.tile_resource[t].unwrap().index();
    s.players[0].cities = 1u64 << exclusive_nodes(t)[0];
    s.bank[r] = 1;
    let n = s.board.tile_number[t];
    roll(&mut s, n);
    assert_eq!(s.players[0].hand[r], 1);
    assert_eq!(s.bank[r], 0);
}

#[test]
fn seven_without_big_hands_goes_to_robber() {
    let mut s = blank(1, Phase::PreRoll);
    give(&mut s, 0, [2, 2, 2, 1, 0]); // exactly 7: no discard
    roll(&mut s, 7);
    assert_eq!(s.phase, Phase::MoveRobber);
    assert_eq!(hand_total(&s.players[0].hand), 7);
}

#[test]
fn seven_makes_big_hands_discard_half() {
    let mut s = blank(1, Phase::PreRoll);
    give(&mut s, 0, [3, 3, 3, 0, 0]);
    roll(&mut s, 7);
    assert_eq!(s.phase, Phase::Discard);
    assert_eq!(s.players[0].discard_remaining, 4);
    assert_eq!(s.current_actor(), 0);
    assert_eq!(
        sorted(s.legal_actions()),
        vec![Action::Discard(Resource::Wood), Action::Discard(Resource::Brick), Action::Discard(Resource::Sheep)]
    );
    for _ in 0..3 {
        s.apply(Action::Discard(Resource::Wood));
    }
    assert_eq!(s.phase, Phase::Discard);
    s.apply(Action::Discard(Resource::Brick));
    assert_eq!(s.players[0].hand, [0, 2, 3, 0, 0]);
    assert_eq!(s.phase, Phase::MoveRobber);
    assert_eq!(s.bank[0], 19);
}

#[test]
fn multiple_discarders_go_in_seat_order() {
    let mut s = blank(1, Phase::PreRoll);
    give(&mut s, 1, [8, 0, 0, 0, 0]);
    give(&mut s, 2, [0, 8, 0, 0, 0]);
    roll(&mut s, 7);
    assert_eq!(s.current_actor(), 1);
    for _ in 0..4 {
        assert_eq!(s.phase, Phase::Discard);
        s.apply(Action::Discard(Resource::Wood));
    }
    assert_eq!(s.current_actor(), 2);
    for _ in 0..4 {
        assert_eq!(s.phase, Phase::Discard);
        s.apply(Action::Discard(Resource::Brick));
    }
    assert_eq!(s.phase, Phase::MoveRobber);
    assert_eq!(s.current_actor(), 0);
}

#[test]
fn compat_mode_discards_randomly() {
    let c = GameConfig { catanatron_compat: true, ..cfg() };
    let mut s = blank_with(1, Phase::PreRoll, c);
    give(&mut s, 0, [3, 3, 3, 0, 0]);
    roll(&mut s, 7);
    assert_eq!(s.phase, Phase::MoveRobber);
    assert_eq!(hand_total(&s.players[0].hand), 5);
    assert_eq!(s.players[0].discard_remaining, 0);
}

#[test]
fn compat_mode_accepts_forced_discards() {
    let c = GameConfig { catanatron_compat: true, ..cfg() };
    let mut s = blank_with(1, Phase::PreRoll, c);
    give(&mut s, 0, [5, 4, 0, 0, 0]);
    let mut discards = [[0u8; 5]; 4];
    discards[0] = [4, 0, 0, 0, 0];
    s.apply_with(Action::Roll, Some(Chance::Roll { dice: (3, 4), discards: Some(discards) }), &mut NoEvents);
    assert_eq!(s.players[0].hand, [1, 4, 0, 0, 0]);
}

#[test]
fn robber_must_move() {
    let s = blank(1, Phase::MoveRobber);
    let legal = s.legal_actions();
    assert_eq!(legal.len(), 18);
    assert!(!legal.contains(&Action::MoveRobber(s.robber)));
}

#[test]
fn steal_from_victim_on_new_tile() {
    let mut s = blank(1, Phase::MoveRobber);
    let t = lonely_tile(&s) as u8;
    s.players[1].settlements = 1u64 << exclusive_nodes(t as usize)[0];
    give(&mut s, 1, [2, 0, 0, 0, 0]);
    s.apply(Action::MoveRobber(t));
    assert_eq!(s.robber, t);
    assert_eq!(s.phase, Phase::Steal);
    assert_eq!(s.legal_actions(), vec![Action::StealFrom(1)]);
    s.apply_with(Action::StealFrom(1), Some(Chance::Steal(Resource::Wood)), &mut NoEvents);
    assert_eq!(s.players[0].hand, [1, 0, 0, 0, 0]);
    assert_eq!(s.players[1].hand, [1, 0, 0, 0, 0]);
    assert_eq!(s.phase, Phase::Main);
}

#[test]
fn unforced_steal_takes_one_card() {
    let mut s = blank(1, Phase::MoveRobber);
    let t = lonely_tile(&s) as u8;
    s.players[1].settlements = 1u64 << exclusive_nodes(t as usize)[0];
    give(&mut s, 1, [1, 1, 1, 0, 0]);
    s.apply(Action::MoveRobber(t));
    s.apply(Action::StealFrom(1));
    assert_eq!(hand_total(&s.players[0].hand), 1);
    assert_eq!(hand_total(&s.players[1].hand), 2);
}

#[test]
fn empty_handed_players_are_not_victims() {
    let mut s = blank(1, Phase::MoveRobber);
    let t = lonely_tile(&s) as u8;
    s.players[1].settlements = 1u64 << exclusive_nodes(t as usize)[0];
    s.apply(Action::MoveRobber(t));
    assert_eq!(s.phase, Phase::Main);
}

#[test]
fn no_steal_when_only_own_buildings() {
    let mut s = blank(1, Phase::MoveRobber);
    let t = lonely_tile(&s) as u8;
    s.players[0].settlements = 1u64 << exclusive_nodes(t as usize)[0];
    give(&mut s, 0, [1, 0, 0, 0, 0]);
    s.apply(Action::MoveRobber(t));
    assert_eq!(s.phase, Phase::Main);
}

#[test]
#[should_panic(expected = "victim has none")]
fn forced_steal_of_missing_resource_panics() {
    let mut s = blank(1, Phase::MoveRobber);
    let t = lonely_tile(&s) as u8;
    s.players[1].settlements = 1u64 << exclusive_nodes(t as usize)[0];
    give(&mut s, 1, [1, 0, 0, 0, 0]);
    s.apply(Action::MoveRobber(t));
    s.apply_with(Action::StealFrom(1), Some(Chance::Steal(Resource::Ore)), &mut NoEvents);
}

#[test]
#[should_panic(expected = "does not match")]
fn mismatched_chance_panics() {
    let mut s = blank(1, Phase::PreRoll);
    s.apply_with(Action::Roll, Some(Chance::Steal(Resource::Wood)), &mut NoEvents);
}

#[test]
fn tile_masks_cover_building_production() {
    // Sanity check for the helper: lonely tile's exclusive nodes are on that tile only.
    let s = blank(9, Phase::PreRoll);
    let t = lonely_tile(&s);
    for n in exclusive_nodes(t) {
        assert_eq!(topo().node_tiles[n as usize], vec![t as u8]);
    }
}

fn forced_roll(s: &mut State, dice: (u8, u8), discards: Option<[Hand; 4]>) {
    s.apply_with(Action::Roll, Some(Chance::Roll { dice, discards }), &mut NoEvents);
}

#[test]
#[should_panic(expected = "out of range")]
fn forced_dice_zero_panics() {
    forced_roll(&mut blank(1, Phase::PreRoll), (0, 0), None);
}

#[test]
#[should_panic(expected = "out of range")]
fn forced_dice_above_six_panics() {
    forced_roll(&mut blank(1, Phase::PreRoll), (9, 9), None);
}

#[test]
#[should_panic(expected = "out of range")]
fn forced_dice_mixed_out_of_range_panics() {
    forced_roll(&mut blank(1, Phase::PreRoll), (0, 7), None);
}

#[test]
#[should_panic(expected = "forced discards given but the roll is not a 7")]
fn forced_discards_on_non_seven_panics() {
    let c = GameConfig { catanatron_compat: true, ..cfg() };
    let mut s = blank_with(1, Phase::PreRoll, c);
    forced_roll(&mut s, (3, 3), Some([[0u8; 5]; 4]));
}

#[test]
#[should_panic(expected = "forced discards given but catanatron_compat is off")]
fn forced_discards_without_compat_panics() {
    let mut s = blank(1, Phase::PreRoll);
    give(&mut s, 0, [5, 4, 0, 0, 0]);
    let mut discards = [[0u8; 5]; 4];
    discards[0] = [4, 0, 0, 0, 0];
    forced_roll(&mut s, (3, 4), Some(discards));
}

#[test]
#[should_panic(expected = "forced discards given but no player must discard")]
fn forced_discards_when_nobody_discards_panics() {
    let c = GameConfig { catanatron_compat: true, ..cfg() };
    let mut s = blank_with(1, Phase::PreRoll, c);
    forced_roll(&mut s, (3, 4), Some([[0u8; 5]; 4]));
}

#[test]
#[should_panic(expected = "forced discard for player 1 who owes none")]
fn forced_discard_for_player_who_owes_none_panics() {
    let c = GameConfig { catanatron_compat: true, ..cfg() };
    let mut s = blank_with(1, Phase::PreRoll, c);
    give(&mut s, 0, [5, 4, 0, 0, 0]);
    give(&mut s, 1, [2, 0, 0, 0, 0]);
    let mut discards = [[0u8; 5]; 4];
    discards[0] = [4, 0, 0, 0, 0];
    discards[1] = [1, 0, 0, 0, 0];
    forced_roll(&mut s, (3, 4), Some(discards));
}
