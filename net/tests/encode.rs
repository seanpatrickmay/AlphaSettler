mod common;

use common::*;
use settler_engine::topology::{topo, NUM_EDGES, NUM_NODES, NUM_TILES};
use settler_engine::*;
use settler_net::encode::*;

/// Pips by number token, written out (the encoder computes them).
const PIPS: [u32; 13] = [0, 0, 1, 2, 3, 4, 5, 0, 5, 4, 3, 2, 1];

/// A hand-built position with every scalar block non-trivial. Player 2 is to act, placing the
/// second free road of a Road Building card played before rolling.
fn golden_state() -> State {
    let mut s = State::new(42, GameConfig::default());
    s.setup_step = 8;
    s.phase = Phase::RoadBuilding { roads_left: 1 };
    s.return_phase = Phase::PreRoll;
    s.current = 2;
    s.turn = 37;
    s.robber = 7;
    s.bank = [10, 12, 9, 15, 11];
    s.dev_deck_pos = 9;
    s.dev_played_this_turn = true;
    s.offers_this_turn = 1;
    s.longest_road_owner = Some(3);
    let p2 = &mut s.players[2];
    p2.settlements = 1 << 10;
    p2.cities = 1 << 20;
    p2.roads = (1 << 3) | (1 << 4);
    p2.hand = [1, 0, 2, 0, 3];
    p2.dev_hand = [2, 1, 0, 0, 0];
    p2.dev_new = [1, 0, 0, 0, 0];
    p2.knights_played = 1;
    p2.dev_played = [1, 0, 1, 0, 0];
    let p3 = &mut s.players[3];
    p3.settlements = 1;
    p3.roads = 1;
    p3.hand = [0, 4, 0, 0, 0];
    p3.dev_hand = [0, 0, 1, 0, 0];
    p3.discard_remaining = 2;
    s.players[0].cities = 1 << 40;
    s.players[0].hand = [0, 0, 0, 1, 0];
    s.players[1].dev_played = [0, 0, 0, 0, 1];
    s
}

#[test]
fn the_layout_adds_up() {
    assert_eq!(PUBLIC_LEN, 1581);
    assert_eq!(HIDDEN_LEN, 30);
    assert_eq!(
        (
            EDGE_BASE,
            TILE_BASE,
            OWN_HAND,
            SEATS,
            LONGEST_ROAD,
            BANK,
            PHASE,
            TURN
        ),
        (1026, 1314, 1466, 1481, 1521, 1531, 1537, 1548)
    );
    assert_eq!(
        (DEV_PLAYED, CURRENT, RETURN_PHASE, FREE_ROADS),
        (1549, 1550, 1554, 1556)
    );
    assert_eq!(
        (
            TRADE_GIVE,
            TRADE_GET,
            TRADE_RESPONSES,
            OFFERS_MADE,
            OFFERS_ALLOWED
        ),
        (1557, 1562, 1567, 1579, 1580)
    );
    assert_eq!(ENCODER_VERSION, 3);
    assert_eq!(ENCODING_HASH, layout_hash(&LAYOUT, LAYOUT_DESCRIPTION));
    assert_ne!(ENCODING_HASH, 0x28dd_1f78_0ff2_a1b3, "encoder v2's hash");
}

#[test]
fn changing_any_layout_constant_changes_the_hash() {
    for i in 0..LAYOUT.len() {
        let mut changed = LAYOUT;
        changed[i] += 1;
        assert_ne!(
            layout_hash(&changed, LAYOUT_DESCRIPTION),
            ENCODING_HASH,
            "layout entry {i}"
        );
    }
    let mut swapped = LAYOUT;
    swapped.swap(12, 13);
    assert_ne!(
        layout_hash(&swapped, LAYOUT_DESCRIPTION),
        ENCODING_HASH,
        "order matters"
    );
}

/// `text` with the first occurrences of `a` and `b` exchanged; panics unless both occur.
fn exchange(text: &str, a: &str, b: &str) -> String {
    let (i, j) = (text.find(a).expect(a), text.find(b).expect(b));
    assert!(
        i + a.len() <= j,
        "`{a}` comes before `{b}` and they do not overlap"
    );
    format!(
        "{}{b}{}{a}{}",
        &text[..i],
        &text[i + a.len()..j],
        &text[j + b.len()..]
    )
}

#[test]
fn reordering_slots_or_blocks_in_the_description_changes_the_hash() {
    let mutations = [
        // Two slots inside one block.
        exchange(LAYOUT_DESCRIPTION, "settlement_r0", "settlement_r1"),
        exchange(LAYOUT_DESCRIPTION, "port_generic", "port_wood"),
        exchange(LAYOUT_DESCRIPTION, "public_vp/vp", "discard_owed/discard"),
        exchange(LAYOUT_DESCRIPTION, "trade_response", "trade_confirm"),
        exchange(LAYOUT_DESCRIPTION, "accepted", "rejected"),
        exchange(LAYOUT_DESCRIPTION, "public trade_give", "public trade_get"),
        // Two whole blocks.
        exchange(
            LAYOUT_DESCRIPTION,
            "public longest_road",
            "public largest_army",
        ),
        // A renamed slot.
        LAYOUT_DESCRIPTION.replacen("robber", "thief", 1),
    ];
    for (i, m) in mutations.iter().enumerate() {
        assert_ne!(m, LAYOUT_DESCRIPTION, "mutation {i} changed the text");
        assert_ne!(layout_hash(&LAYOUT, m), ENCODING_HASH, "mutation {i}");
    }
}

/// The description names every block in order with its item count and one token per slot; its
/// offsets, widths, scales and version must agree with the constants the encoder writes with.
#[test]
fn the_description_matches_the_constants() {
    let lines: Vec<&str> = LAYOUT_DESCRIPTION.lines().collect();
    assert_eq!(lines[0], format!("settler-net features v{ENCODER_VERSION}"));

    let scales: Vec<(&str, u32)> = lines
        .iter()
        .find_map(|l| l.strip_prefix("scales: "))
        .expect("a scales line")
        .split(' ')
        .map(|kv| {
            let (k, v) = kv.split_once('=').expect(kv);
            (k, v.parse().expect(kv))
        })
        .collect();
    let want_scales = [
        ("card", CARD_DEN),
        ("hand", HAND_DEN),
        ("dev_total", DEV_TOTAL_DEN),
        ("dev_knight", DEV_KIND_DEN[0]),
        ("dev_victory_point", DEV_KIND_DEN[1]),
        ("dev_road_building", DEV_KIND_DEN[2]),
        ("dev_year_of_plenty", DEV_KIND_DEN[3]),
        ("dev_monopoly", DEV_KIND_DEN[4]),
        ("vp", VP_DEN),
        ("discard", DISCARD_DEN),
        ("bank", BANK_DEN),
        ("deck", DECK_DEN),
        ("pip", PIP_DEN),
        ("node_pip", NODE_PIP_DEN),
        ("free_roads", FREE_ROAD_DEN),
    ];
    assert_eq!(scales, want_scales);

    // (section, block, items, width, offset)
    let want = [
        ("public", "node", NUM_NODES, NODE_WIDTH, NODE_BASE),
        ("public", "edge", NUM_EDGES, EDGE_WIDTH, EDGE_BASE),
        ("public", "tile", NUM_TILES, TILE_WIDTH, TILE_BASE),
        ("public", "own_hand", 1, NUM_RESOURCES, OWN_HAND),
        ("public", "own_dev_old", 1, 5, OWN_DEV_OLD),
        ("public", "own_dev_new", 1, 5, OWN_DEV_NEW),
        ("public", "seat", NUM_PLAYERS, SEAT_WIDTH, SEATS),
        ("public", "longest_road", 1, NUM_PLAYERS + 1, LONGEST_ROAD),
        ("public", "largest_army", 1, NUM_PLAYERS + 1, LARGEST_ARMY),
        ("public", "bank", 1, NUM_RESOURCES, BANK),
        ("public", "deck", 1, 1, DECK),
        ("public", "phase", 1, NUM_PHASES, PHASE),
        ("public", "turn", 1, 1, TURN),
        ("public", "dev_played", 1, 1, DEV_PLAYED),
        ("public", "current", 1, NUM_PLAYERS, CURRENT),
        ("public", "return_phase", 1, 2, RETURN_PHASE),
        ("public", "free_roads", 1, 1, FREE_ROADS),
        ("public", "trade_give", 1, NUM_RESOURCES, TRADE_GIVE),
        ("public", "trade_get", 1, NUM_RESOURCES, TRADE_GET),
        (
            "public",
            "trade_response",
            NUM_PLAYERS,
            RESPONSE_WIDTH,
            TRADE_RESPONSES,
        ),
        ("public", "offers_made", 1, 1, OFFERS_MADE),
        ("public", "offers_allowed", 1, 1, OFFERS_ALLOWED),
        ("hidden", "opponent", NUM_PLAYERS - 1, HIDDEN_WIDTH, 0),
    ];
    let mut got = Vec::new();
    let mut end = [("public", 0usize), ("hidden", 0usize)];
    for line in &lines {
        let Some((head, slots)) = line.split_once(": ") else {
            continue;
        };
        let words: Vec<&str> = head.split(' ').collect();
        let [section @ ("public" | "hidden"), block, items] = words[..] else {
            continue;
        };
        let items: usize = items.strip_prefix('x').expect(line).parse().expect(line);
        let slots: Vec<&str> = slots.split(' ').collect();
        for slot in &slots {
            if let Some((_, scale)) = slot.split_once('/') {
                // Scales the game's config sets rather than a constant.
                let from_config =
                    ["max_turns", "max_trade_cards", "max_offers_per_turn"].contains(&scale);
                assert!(
                    from_config || want_scales.iter().any(|&(k, _)| k == scale),
                    "slot {slot} names a declared scale"
                );
            }
        }
        let at = end.iter_mut().find(|(s, _)| *s == section).unwrap();
        got.push((section, block, items, slots.len(), at.1));
        at.1 += items * slots.len();
    }
    assert_eq!(got, want);
    assert_eq!(end, [("public", PUBLIC_LEN), ("hidden", HIDDEN_LEN)]);
}

#[test]
fn the_golden_position_encodes_exactly() {
    let s = golden_state();
    let mut got = vec![f32::NAN; PUBLIC_LEN];
    encode_public(&s.observation(2), &mut got);

    let mut want = vec![0.0f32; PUBLIC_LEN];
    let t = topo();
    let b = &s.board;
    // Board blocks, derived tile -> corners (the encoder goes node -> tiles).
    for tile in 0..NUM_TILES {
        let at = TILE_BASE + tile * TILE_WIDTH;
        want[at + b.tile_resource[tile].map_or(5, |r| r.index())] = 1.0;
        let p = PIPS[b.tile_number[tile] as usize];
        want[at + 6] = p as f32 / 5.0;
        if let Some(r) = b.tile_resource[tile] {
            for &n in &t.tile_nodes[tile] {
                want[NODE_BASE + n as usize * NODE_WIDTH + 14 + r.index()] += p as f32;
            }
        }
    }
    for n in 0..NUM_NODES {
        for r in 0..5 {
            want[NODE_BASE + n * NODE_WIDTH + 14 + r] /= 15.0;
        }
    }
    for &(e, kind) in &b.ports {
        let slot = match kind {
            PortKind::Generic => 8,
            PortKind::Specific(r) => 9 + r.index(),
        };
        let (x, y) = t.edge_nodes[e as usize];
        want[NODE_BASE + x as usize * NODE_WIDTH + slot] = 1.0;
        want[NODE_BASE + y as usize * NODE_WIDTH + slot] = 1.0;
    }
    // The robber on tile 7.
    want[TILE_BASE + 7 * TILE_WIDTH + 7] = 1.0;

    // Buildings and roads, seat-relative to player 2.
    want[NODE_BASE + 10 * NODE_WIDTH] = 1.0; // player 2 (rel 0) settlement
    want[NODE_BASE + 20 * NODE_WIDTH + 4] = 1.0; // player 2 (rel 0) city
    want[NODE_BASE + 1] = 1.0; // node 0: player 3 (rel 1) settlement
    want[NODE_BASE + 40 * NODE_WIDTH + 4 + 2] = 1.0; // player 0 (rel 2) city
    want[EDGE_BASE + 3 * EDGE_WIDTH] = 1.0;
    want[EDGE_BASE + 4 * EDGE_WIDTH] = 1.0;
    want[EDGE_BASE + 1] = 1.0; // edge 0: player 3 (rel 1)

    // Scalars. Dev cards of one kind are divided by that kind's deck count (14, 5, 2, 2, 2).
    want[OWN_HAND..OWN_HAND + 5].copy_from_slice(&[1.0 / 19.0, 0.0, 2.0 / 19.0, 0.0, 3.0 / 19.0]);
    want[OWN_DEV_OLD..OWN_DEV_OLD + 5].copy_from_slice(&[1.0 / 14.0, 1.0 / 5.0, 0.0, 0.0, 0.0]);
    want[OWN_DEV_NEW] = 1.0 / 14.0;
    let seat = |r: usize| SEATS + r * SEAT_WIDTH;
    // Player 2 (rel 0): 6 cards, 3 dev cards (1 new), played a knight and a road building, 3 VP.
    want[seat(0)..seat(0) + SEAT_WIDTH].copy_from_slice(&[
        6.0 / 20.0,
        3.0 / 25.0,
        1.0 / 25.0,
        1.0 / 14.0,
        0.0,
        1.0 / 2.0,
        0.0,
        0.0,
        3.0 / 10.0,
        0.0,
    ]);
    // Player 3 (rel 1): 4 cards, 1 dev card, 3 VP (settlement + longest road), owes 2.
    want[seat(1)..seat(1) + SEAT_WIDTH].copy_from_slice(&[
        4.0 / 20.0,
        1.0 / 25.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        3.0 / 10.0,
        2.0 / 10.0,
    ]);
    // Player 0 (rel 2): 1 card, a city.
    want[seat(2)] = 1.0 / 20.0;
    want[seat(2) + 8] = 2.0 / 10.0;
    // Player 1 (rel 3) played a monopoly.
    want[seat(3) + 3 + 4] = 1.0 / 2.0;
    want[LONGEST_ROAD + 1] = 1.0;
    want[LARGEST_ARMY + 4] = 1.0; // nobody
    want[BANK..BANK + 5].copy_from_slice(&[
        10.0 / 19.0,
        12.0 / 19.0,
        9.0 / 19.0,
        15.0 / 19.0,
        11.0 / 19.0,
    ]);
    want[DECK] = 16.0 / 25.0;
    want[PHASE + 7] = 1.0; // RoadBuilding
    want[TURN] = 37.0 / 1000.0;
    want[DEV_PLAYED] = 1.0;
    want[CURRENT] = 1.0; // player 2 is the viewer
    want[RETURN_PHASE] = 1.0; // PreRoll
    want[FREE_ROADS] = 1.0 / 2.0;
    want[OFFERS_MADE] = 1.0 / 3.0; // one of the default config's three offers this turn
    want[OFFERS_ALLOWED] = 1.0;
    for i in 0..PUBLIC_LEN {
        assert_eq!(got[i], want[i], "feature {i}");
    }

    // Seen by player 0, the current player is relative seat 2.
    let mut seen_by_0 = vec![f32::NAN; PUBLIC_LEN];
    encode_public(&s.observation(0), &mut seen_by_0);
    assert_eq!(seen_by_0[CURRENT..CURRENT + 4], [0.0, 0.0, 1.0, 0.0]);

    let mut hidden = vec![f32::NAN; HIDDEN_LEN];
    encode_hidden(&s, 2, &mut hidden);
    let mut want_hidden = vec![0.0f32; HIDDEN_LEN];
    want_hidden[1] = 4.0 / 19.0; // rel 1 = player 3: 4 brick
    want_hidden[5 + 2] = 1.0 / 2.0; // and a road building card
    want_hidden[HIDDEN_WIDTH + 3] = 1.0 / 19.0; // rel 2 = player 0: 1 wheat
    assert_eq!(hidden, want_hidden);
}

fn rotate<T: Copy>(a: [T; 4], k: usize) -> [T; 4] {
    std::array::from_fn(|q| a[(q + 4 - k) % 4])
}

fn rot(p: PlayerId, k: usize) -> PlayerId {
    ((p as usize + k) % 4) as PlayerId
}

/// `o` with every seat p renamed (p + k) % 4, the viewer included.
fn rotated(o: &Observation, k: usize) -> Observation {
    let mut r = *o;
    r.viewer = rot(o.viewer, k);
    r.current = rot(o.current, k);
    r.actor = rot(o.actor, k);
    r.hand_counts = rotate(o.hand_counts, k);
    r.dev_card_counts = rotate(o.dev_card_counts, k);
    r.new_dev_card_counts = rotate(o.new_dev_card_counts, k);
    r.dev_cards_played = rotate(o.dev_cards_played, k);
    r.settlements = rotate(o.settlements, k);
    r.cities = rotate(o.cities, k);
    r.roads = rotate(o.roads, k);
    r.knights_played = rotate(o.knights_played, k);
    r.public_vp = rotate(o.public_vp, k);
    r.longest_road_owner = o.longest_road_owner.map(|p| rot(p, k));
    r.largest_army_owner = o.largest_army_owner.map(|p| rot(p, k));
    r.discard_remaining = rotate(o.discard_remaining, k);
    r.trade = o.trade.map(|t| PendingTrade {
        responses: rotate(t.responses, k),
        next_responder: rot(t.next_responder, k),
        ..t
    });
    r
}

fn public(o: &Observation) -> Vec<f32> {
    let mut v = vec![0.0; PUBLIC_LEN];
    encode_public(o, &mut v);
    v
}

#[test]
fn rotating_every_seat_and_the_actor_together_gives_the_same_features() {
    let (mut checked, mut moved, mut trades) = (0, 0, 0);
    // Positions where the actor is not the current player, and Road Building positions.
    let (mut off_turn, mut road_building) = (0, 0);
    for s in random_positions(0..4, 7, GameConfig::default()) {
        let actor = s.current_actor();
        let obs = s.observation(actor);
        let base = public(&obs);
        let hands = std::array::from_fn(|p| s.players[p].hand);
        let devs = std::array::from_fn(|p| s.players[p].dev_hand);
        let mut hidden = vec![0.0; HIDDEN_LEN];
        encode_hidden_from(&hands, &devs, actor, &mut hidden);
        trades += obs.trade.is_some() as u32;
        off_turn += (obs.current != obs.viewer) as u32;
        road_building += matches!(obs.phase, Phase::RoadBuilding { .. }) as u32;
        for k in 1..4 {
            let r = rotated(&obs, k);
            moved += (r.settlements != obs.settlements) as u32;
            assert_eq!(public(&r), base, "rotation {k}");
            let mut h = vec![0.0; HIDDEN_LEN];
            encode_hidden_from(&rotate(hands, k), &rotate(devs, k), rot(actor, k), &mut h);
            assert_eq!(h, hidden, "rotation {k}");
        }
        checked += 1;
    }
    assert!(
        checked > 500 && moved > 0 && trades > 0 && off_turn > 0 && road_building > 0,
        "{checked} {moved} {trades} {off_turn} {road_building}"
    );
}

#[test]
fn the_features_are_relative_to_the_viewer() {
    let mut differ = 0;
    for s in random_positions(10..12, 13, no_trades()) {
        let obs = s.observation(s.current_actor());
        let mut seats_only = rotated(&obs, 1);
        seats_only.viewer = obs.viewer; // rotate the seats but not the viewer
        differ += (public(&seats_only) != public(&obs)) as u32;
    }
    assert!(differ > 0);
}

/// The trade block of a hand-built TradeConfirm position in the trading game (one card a side,
/// two offers a turn): player 1 offered a brick for an ore, its second offer this turn; players
/// 2 accepted, 3 and 0 rejected. Seen by player 3, so player 3 is r0, 0 is r1, 1 is r2, 2 is r3.
/// Without offers the block is all zero.
#[test]
fn the_trade_block_encodes_exactly() {
    let mut s = State::new(42, trading());
    s.setup_step = 8;
    s.phase = Phase::TradeConfirm;
    s.current = 1;
    s.offers_this_turn = 2;
    s.trade = Some(PendingTrade {
        give: [0, 1, 0, 0, 0],
        get: [0, 0, 0, 0, 1],
        responses: [
            Response::Rejected,
            Response::Pending,
            Response::Accepted,
            Response::Rejected,
        ],
        next_responder: 0,
    });
    let mut got = vec![f32::NAN; PUBLIC_LEN];
    encode_public(&s.observation(3), &mut got);
    #[rustfmt::skip]
    let want: Vec<f32> = vec![
        0.0, 1.0, 0.0, 0.0, 0.0, // give: a brick / 1
        0.0, 0.0, 0.0, 0.0, 1.0, // get: an ore / 1
        0.0, 0.0, 1.0, // r0 = player 3 rejected
        0.0, 0.0, 1.0, // r1 = player 0 rejected
        1.0, 0.0, 0.0, // r2 = player 1, the offerer, stays pending
        0.0, 1.0, 0.0, // r3 = player 2 accepted
        1.0, // 2 of 2 offers made
        1.0, // offers allowed
    ];
    assert_eq!(&got[TRADE_GIVE..PUBLIC_LEN], &want[..]);

    let quiet = State::new(42, no_trades());
    let mut zero = vec![f32::NAN; PUBLIC_LEN];
    encode_public(&quiet.observation(0), &mut zero);
    assert!(zero[TRADE_GIVE..PUBLIC_LEN].iter().all(|&x| x == 0.0));
}
