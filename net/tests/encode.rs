mod common;

use common::*;
use settler_engine::topology::{topo, NUM_EDGES, NUM_NODES, NUM_TILES};
use settler_engine::*;
use settler_net::encode::*;
use settler_net::weights::{V2_ENCODING_HASH, V3_ENCODING_HASH};

/// Pips by number token, written out (the encoder computes them).
const PIPS: [u32; 13] = [0, 0, 1, 2, 3, 4, 5, 0, 5, 4, 3, 2, 1];

/// Each node's road-distance slots (r0 .. r3) in an encoded public block.
fn got_distances(public: &[f32]) -> Vec<[f32; NUM_PLAYERS]> {
    (0..NUM_NODES)
        .map(|n| {
            let at = NODE_BASE + n * NODE_WIDTH + NODE_DISTANCE;
            std::array::from_fn(|r| public[at + r])
        })
        .collect()
}

/// The road-distance slots expected when `near[r]` lists relative seat r's nodes 0, 1 and 2
/// roads away: d / 3 for those, 1.0 (3 or more roads, or never) for every other node.
fn want_distances(near: [[&[usize]; 3]; NUM_PLAYERS]) -> Vec<[f32; NUM_PLAYERS]> {
    let mut want = vec![[1.0f32; NUM_PLAYERS]; NUM_NODES];
    for (r, rings) in near.iter().enumerate() {
        for (d, nodes) in rings.iter().enumerate() {
            for &n in *nodes {
                want[n][r] = d as f32 / 3.0;
            }
        }
    }
    want
}

/// The nodes whose open-spot slot is 1; every other node's must be 0.
fn open_nodes(public: &[f32]) -> Vec<usize> {
    (0..NUM_NODES)
        .filter(|&n| {
            let x = public[NODE_BASE + n * NODE_WIDTH + NODE_OPEN];
            assert!(x == 0.0 || x == 1.0, "node {n}: {x}");
            x == 1.0
        })
        .collect()
}

/// The edges on relative seat `r`'s longest trail, ascending.
fn trail_edges(public: &[f32], r: usize) -> Vec<usize> {
    (0..NUM_EDGES)
        .filter(|&e| public[EDGE_BASE + e * EDGE_WIDTH + EDGE_TRAIL + r] == 1.0)
        .collect()
}

/// Relative seat `r`'s threat block.
fn threat(public: &[f32], r: usize) -> &[f32] {
    &public[THREATS + r * THREAT_WIDTH..THREATS + (r + 1) * THREAT_WIDTH]
}

/// A mask of board edges.
fn edges(es: &[u8]) -> u128 {
    es.iter().fold(0u128, |m, &e| m | (1u128 << e))
}

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
    assert_eq!(
        (NODE_WIDTH, EDGE_WIDTH, NODE_DISTANCE, NODE_OPEN, EDGE_TRAIL),
        (24, 8, 19, 23, 4)
    );
    // v3's 1581 + 54 nodes x 5 + 72 edges x 4 + 4 seats x 6 = 1581 + 270 + 288 + 24 = 2163.
    assert_eq!(PUBLIC_LEN, 2163);
    assert_eq!(HIDDEN_LEN, 30);
    // Nodes 54 x 24 = 1296, edges 72 x 8 = 576 (1296 + 576 = 1872), tiles 19 x 8 = 152: the
    // globals start at 2024, each 270 + 288 = 558 after v3's (1466 + 558 = 2024, ...).
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
        (1296, 1872, 2024, 2039, 2079, 2089, 2095, 2106)
    );
    assert_eq!(
        (DEV_PLAYED, CURRENT, RETURN_PHASE, FREE_ROADS),
        (2107, 2108, 2112, 2114)
    );
    assert_eq!(
        (
            TRADE_GIVE,
            TRADE_GET,
            TRADE_RESPONSES,
            OFFERS_MADE,
            OFFERS_ALLOWED
        ),
        (2115, 2120, 2125, 2137, 2138)
    );
    // The threat block follows the trade block: 2139 + 4 x 6 = 2163.
    assert_eq!((THREATS, THREAT_WIDTH), (2139, 6));
    assert_eq!(ENCODER_VERSION, 4);
    assert_eq!(ENCODING_HASH, layout_hash(&LAYOUT, LAYOUT_DESCRIPTION));
    assert_ne!(ENCODING_HASH, V2_ENCODING_HASH, "encoder v2's hash");
    assert_ne!(ENCODING_HASH, V3_ENCODING_HASH, "encoder v3's hash");
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
        exchange(LAYOUT_DESCRIPTION, "road_distance_r0", "road_distance_r1"),
        exchange(LAYOUT_DESCRIPTION, "longest_trail_r2", "longest_trail_r3"),
        exchange(
            LAYOUT_DESCRIPTION,
            "points_to_win/vp_to_win",
            "settlements_left/settlements",
        ),
        exchange(LAYOUT_DESCRIPTION, "public offers_allowed", "public threat"),
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
        ("distance", DISTANCE_DEN),
        ("road_length", ROAD_LENGTH_DEN),
        ("road_award", ROAD_AWARD_MIN),
        ("army_award", ARMY_AWARD_MIN),
        ("settlements", SETTLEMENT_DEN),
        ("cities", CITY_DEN),
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
        ("public", "threat", NUM_PLAYERS, THREAT_WIDTH, THREATS),
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
                let from_config = [
                    "max_turns",
                    "max_trade_cards",
                    "max_offers_per_turn",
                    "vp_to_win",
                ]
                .contains(&scale);
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
    // v4, seen by player 2: r0 = player 2, r1 = 3, r2 = 0, r3 = 1. Occupied: 0 (player 3), 10 and
    // 20 (player 2), 40 (player 0).
    // Road distances (a seat's network is distance 0; a path may not use another seat's road or
    // leave another seat's building):
    // r0 = player 2, network {1, 2, 5, 10, 20} (roads 1-5-2 and the two buildings). One road: 4
    //   (from 1), 6 (from 2 and 10), 9 (from 5), 14, 15 (from 10), 25, 26 (from 20). Two: 0 (from
    //   4: player 3's settlement, reached), 8 (from 4), 13 (from 9), 19 (from 14 and 25), 31 (from
    //   25), 32 (from 26).
    // r1 = player 3, network {0, 3}. One: 4 (from 0), 7 (from 3). Two: 1, 8 (from 4), 11, 12 (from 7).
    // r2 = player 0, network {40}. One: 35, 44, 45. Two: 29, 30 (from 35), 39, 48 (from 44), 41, 49
    //   (from 45).
    // r3 = player 1: nothing on the board, every node 1.0.
    let distances = want_distances([
        [
            &[1, 2, 5, 10, 20],
            &[4, 6, 9, 14, 15, 25, 26],
            &[0, 8, 13, 19, 31, 32],
        ],
        [&[0, 3], &[4, 7], &[1, 8, 11, 12]],
        [&[40], &[35, 44, 45], &[29, 30, 39, 41, 48, 49]],
        [&[], &[], &[]],
    ]);
    for (n, d) in distances.iter().enumerate() {
        let at = NODE_BASE + n * NODE_WIDTH + NODE_DISTANCE;
        want[at..at + NUM_PLAYERS].copy_from_slice(d);
    }
    // Open spots: every node but the 4 occupied and their neighbours (0: 3, 4; 10: 6, 14, 15;
    // 20: 15, 25, 26; 40: 35, 44, 45): 54 - 14 = 40 nodes.
    let closed = [0, 3, 4, 6, 10, 14, 15, 20, 25, 26, 35, 40, 44, 45];
    for n in (0..NUM_NODES).filter(|n| !closed.contains(n)) {
        want[NODE_BASE + n * NODE_WIDTH + NODE_OPEN] = 1.0;
    }
    // Longest trails: player 2's 1-5-2 (edges 3 and 4, length 2), player 3's edge 0 (length 1).
    want[EDGE_BASE + 3 * EDGE_WIDTH + EDGE_TRAIL] = 1.0;
    want[EDGE_BASE + 4 * EDGE_WIDTH + EDGE_TRAIL] = 1.0;
    want[EDGE_BASE + EDGE_TRAIL + 1] = 1.0;
    // Threats: [trail / 15, roads to award / 5, knights to award / 3, points to win / 10,
    // settlements left / 5, cities left / 4]. Player 3 holds the longest road with a trail of 1, so
    // a challenger needs max(5, 1 + 1) - own; nobody holds the largest army, so max(0, 3 - own).
    let at = |r: usize| THREATS + r * THREAT_WIDTH;
    // r0 = player 2: trail 2; 5 - 2 = 3 roads; 3 - 1 knight = 2; a settlement, a city and its VP
    // card = 4 points, 6 to go; 4 settlements and 3 cities left.
    want[at(0)..at(0) + 6].copy_from_slice(&[
        2.0 / 15.0,
        3.0 / 5.0,
        2.0 / 3.0,
        6.0 / 10.0,
        4.0 / 5.0,
        3.0 / 4.0,
    ]);
    // r1 = player 3: trail 1, the holder (0); 3 knights; a settlement and the road = 3 public
    // points, 7 to go; 4 settlements, 4 cities.
    want[at(1)..at(1) + 6].copy_from_slice(&[
        1.0 / 15.0,
        0.0,
        3.0 / 3.0,
        7.0 / 10.0,
        4.0 / 5.0,
        4.0 / 4.0,
    ]);
    // r2 = player 0: no road, 5 - 0 = 5; 3 knights; a city = 2 points, 8 to go; 5 settlements, 3 cities.
    want[at(2)..at(2) + 6].copy_from_slice(&[
        0.0,
        5.0 / 5.0,
        3.0 / 3.0,
        8.0 / 10.0,
        5.0 / 5.0,
        3.0 / 4.0,
    ]);
    // r3 = player 1: nothing: 5 roads, 3 knights, 10 points to go, every piece left.
    want[at(3)..at(3) + 6].copy_from_slice(&[
        0.0,
        5.0 / 5.0,
        3.0 / 3.0,
        10.0 / 10.0,
        5.0 / 5.0,
        4.0 / 4.0,
    ]);
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
    // Positions where v4's blocks say something about an opponent: a node one or two roads from
    // it, a road on its longest trail, a threat block unlike the viewer's.
    let (mut near, mut trails, mut threats) = (0, 0, 0);
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
        let d = got_distances(&base);
        near += (1..NUM_PLAYERS).any(|r| d.iter().any(|x| x[r] > 0.0 && x[r] < 1.0)) as u32;
        trails += (1..NUM_PLAYERS).any(|r| !trail_edges(&base, r).is_empty()) as u32;
        threats += (1..NUM_PLAYERS).any(|r| threat(&base, r) != threat(&base, 0)) as u32;
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
        checked > 500
            && moved > 0
            && trades > 0
            && off_turn > 0
            && road_building > 0
            && near > 0
            && trails > 0
            && threats > 0,
        "{checked} {moved} {trades} {off_turn} {road_building} {near} {trails} {threats}"
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
    assert_eq!(&got[TRADE_GIVE..THREATS], &want[..]);

    let quiet = State::new(42, no_trades());
    let mut zero = vec![f32::NAN; PUBLIC_LEN];
    encode_public(&quiet.observation(0), &mut zero);
    assert!(zero[TRADE_GIVE..THREATS].iter().all(|&x| x == 0.0));
}

/// A built position for v4's blocks, seen by player 1 (r0 = 1, r1 = 2, r2 = 3, r3 = 0).
/// Player 0: a settlement on node 4 and a Y of roads from it, edges 1 (0-4), 2 (1-4), 7 (4-8).
/// Player 1: settlements on nodes 5 and 13, roads 3 (1-5) and 8 (5-9), 2 knights, a VP card.
/// Player 2: a city on node 30, 3 knights, the largest army.
/// Player 3: a settlement on node 47 and the path 47-51-48-52-49-53 (edges 66 to 70), the
/// longest road, 1 knight, a VP card (hidden from player 1).
fn threats_state() -> State {
    let mut s = State::new(42, GameConfig::default());
    s.setup_step = 8;
    s.phase = Phase::Main;
    s.current = 1;
    s.longest_road_owner = Some(3);
    s.largest_army_owner = Some(2);
    let p0 = &mut s.players[0];
    p0.settlements = 1 << 4;
    p0.roads = edges(&[1, 2, 7]);
    let p1 = &mut s.players[1];
    p1.settlements = (1 << 5) | (1 << 13);
    p1.roads = edges(&[3, 8]);
    p1.knights_played = 2;
    p1.dev_played = [2, 0, 0, 0, 0];
    p1.dev_hand = [0, 1, 0, 0, 0];
    let p2 = &mut s.players[2];
    p2.cities = 1 << 30;
    p2.knights_played = 3;
    p2.dev_played = [3, 0, 0, 0, 0];
    let p3 = &mut s.players[3];
    p3.settlements = 1 << 47;
    p3.roads = edges(&[66, 67, 68, 69, 70]);
    p3.knights_played = 1;
    p3.dev_played = [1, 0, 0, 0, 0];
    p3.dev_hand = [0, 1, 0, 0, 0];
    s
}

#[test]
fn the_road_and_threat_blocks_encode_exactly() {
    let s = threats_state();
    let mut got = vec![f32::NAN; PUBLIC_LEN];
    encode_public(&s.observation(1), &mut got);
    // Road distances. Occupied: 4 (player 0), 5 and 13 (player 1), 30 (player 2), 47 (player 3).
    // r0 = player 1, network {1, 5, 9, 13}. Node 1's edge 2 (to 4) and node 8's edge 7 (to 4) are
    //   player 0's roads, so node 4 is never reached. One road: 2 (from 5), 14 (from 9), 8, 18
    //   (from 13). Two: 6 (from 2), 10, 19 (from 14), 12 (from 8), 23, 24 (from 18).
    // r1 = player 2, network {30}. One: 24, 35, 36. Two: 18, 19 (from 24), 29, 40 (from 35), 31,
    //   41 (from 36).
    // r2 = player 3, network {47, 48, 49, 51, 52, 53}. One: 43 (from 47), 44 (from 48), 45 (from
    //   49), 50 (from 53). Two: 38, 39 (from 43), 40 (from 44), 41 (from 45), 46 (from 50).
    // r3 = player 0, network {0, 1, 4, 8}. One: 3 (from 0), 12, 13 (from 8); edge 3 (1-5) is player
    //   1's road, so 5 is not reached from 1 (blocked path: otherwise 1 road). Two: 7 (from 3 and
    //   12), 17 (from 12). Node 13 holds player 1's settlement: reached, but no path leaves it, so 9
    //   and 18 stay 3+ roads away (otherwise 2).
    assert_eq!(
        got_distances(&got),
        want_distances([
            [&[1, 5, 9, 13], &[2, 8, 14, 18], &[6, 10, 12, 19, 23, 24]],
            [&[30], &[24, 35, 36], &[18, 19, 29, 31, 40, 41]],
            [
                &[47, 48, 49, 51, 52, 53],
                &[43, 44, 45, 50],
                &[38, 39, 40, 41, 46]
            ],
            [&[0, 1, 4, 8], &[3, 12, 13], &[7, 17]],
        ])
    );
    // Open spots: all but the 5 occupied nodes and their neighbours (4: 0, 1, 8; 5: 1, 2, 9;
    // 13: 8, 9, 18; 30: 24, 35, 36; 47: 43, 51): 54 - 16 = 38 nodes.
    let closed = [0, 1, 2, 4, 5, 8, 9, 13, 18, 24, 30, 35, 36, 43, 47, 51];
    let open: Vec<usize> = (0..NUM_NODES).filter(|n| !closed.contains(n)).collect();
    assert_eq!(open_nodes(&got), open);
    // r0 = player 1: 1-5-9, edges 3 and 8.
    assert_eq!(trail_edges(&got, 0), [3, 8]);
    assert!(trail_edges(&got, 1).is_empty());
    assert_eq!(trail_edges(&got, 2), [66, 67, 68, 69, 70]);
    // Longest trails. Player 0's Y has three trails of 2 through node 4; the walk starts at node 0
    // (an end of the lowest edge, 1) and turns onto edge 2 before edge 7: 0-4-1 is kept (the tie).
    assert_eq!(trail_edges(&got, 3), [1, 2]);
    // Threats: [trail / 15, roads to award / 5, knights to award / 3, points to win / 10,
    // settlements left / 5, cities left / 4]. Player 3 holds the road with 5: a challenger needs
    // max(5, 5 + 1) - own, at most 5. Player 2 holds the army with 3: max(3, 3 + 1) - own, at most 3.
    // r0 = player 1: trail 2, 6 - 2 = 4 roads; 4 - 2 = 2 knights; 2 settlements + its VP card = 3
    // points, 7 to go; 3 settlements, 4 cities left.
    assert_eq!(
        threat(&got, 0),
        [
            2.0 / 15.0,
            4.0 / 5.0,
            2.0 / 3.0,
            7.0 / 10.0,
            3.0 / 5.0,
            4.0 / 4.0
        ]
    );
    // r1 = player 2: no road, 6 - 0 = 6, capped at 5; the army's holder, 0; a city and the army =
    // 4 public points, 6 to go; 5 settlements, 3 cities left.
    assert_eq!(
        threat(&got, 1),
        [0.0, 5.0 / 5.0, 0.0, 6.0 / 10.0, 5.0 / 5.0, 3.0 / 4.0]
    );
    // r2 = player 3: trail 5, the road's holder, 0; 4 - 1 = 3 knights; a settlement and the road =
    // 3 public points (its VP card stays hidden), 7 to go; 4 settlements, 4 cities left.
    assert_eq!(
        threat(&got, 2),
        [5.0 / 15.0, 0.0, 3.0 / 3.0, 7.0 / 10.0, 4.0 / 5.0, 4.0 / 4.0]
    );
    // r3 = player 0: trail 2, 4 roads; 4 - 0 = 4, capped at 3; a settlement = 1 point, 9 to go;
    // 4 settlements, 4 cities left.
    assert_eq!(
        threat(&got, 3),
        [
            2.0 / 15.0,
            4.0 / 5.0,
            3.0 / 3.0,
            9.0 / 10.0,
            4.0 / 5.0,
            4.0 / 4.0
        ]
    );
}

#[test]
fn an_empty_board_reaches_nothing_and_every_spot_is_open() {
    let s = State::new(7, GameConfig::default());
    let mut got = vec![f32::NAN; PUBLIC_LEN];
    encode_public(&s.observation(0), &mut got);
    assert_eq!(got_distances(&got), vec![[1.0f32; NUM_PLAYERS]; NUM_NODES]);
    assert_eq!(open_nodes(&got), (0..NUM_NODES).collect::<Vec<_>>());
    for r in 0..NUM_PLAYERS {
        assert!(trail_edges(&got, r).is_empty(), "seat {r}");
        // No road; 5 - 0 roads and 3 - 0 knights to go; 10 of 10 points; every piece left.
        assert_eq!(threat(&got, r), [0.0, 1.0, 1.0, 1.0, 1.0, 1.0], "seat {r}");
    }
}

#[test]
fn a_road_into_an_opponents_building_reaches_it_but_not_past_it() {
    // Player 0's roads 0 (0-3) and 6 (3-7) end at node 7, where player 1 has built.
    let mut s = State::new(42, GameConfig::default());
    s.setup_step = 8;
    s.phase = Phase::Main;
    s.players[0].roads = edges(&[0, 6]);
    s.players[1].settlements = 1 << 7;
    let mut got = vec![f32::NAN; PUBLIC_LEN];
    encode_public(&s.observation(0), &mut got);
    // r0 = player 0: network {0, 3, 7} (node 7 is touched: 0 roads), but no path leaves node 7.
    // One road: 4 (from 0). Two: 1, 8 (from 4). Nodes 11 and 12, beside node 7, stay 3+ away.
    // r1 = player 1: network {7}; edge 6 is player 0's road. One road: 11, 12. Two: 16 (from 11),
    // 8, 17 (from 12). Node 3 is never reached.
    assert_eq!(
        got_distances(&got),
        want_distances([
            [&[0, 3, 7], &[4], &[1, 8]],
            [&[7], &[11, 12], &[8, 16, 17]],
            [&[], &[], &[]],
            [&[], &[], &[]],
        ])
    );
    // The trail may end at player 1's settlement: 0-3-7, length 2.
    assert_eq!(trail_edges(&got, 0), [0, 6]);
    assert_eq!(got[THREATS], 2.0 / 15.0);
}

#[test]
fn an_unowned_award_past_its_minimum_reads_zero_to_go() {
    // Players 0 and 1 have trails of 5 and nobody holds the award (as after a broken holder with
    // tied challengers): max(0, 5 - 5) = 0 roads to go for both, as the spec writes it; players 2
    // and 3 have none, max(0, 5 - 0) = 5.
    let mut s = State::new(42, GameConfig::default());
    s.setup_step = 8;
    s.phase = Phase::Main;
    s.players[0].roads = edges(&[0, 6, 10, 18, 23]); // 0-3-7-11-16-21
    s.players[1].roads = edges(&[66, 67, 68, 69, 70]); // 47-51-48-52-49-53
    let mut got = vec![f32::NAN; PUBLIC_LEN];
    encode_public(&s.observation(0), &mut got);
    let slots = |r: usize| (threat(&got, r)[0], threat(&got, r)[1]);
    assert_eq!(
        [slots(0), slots(1), slots(2), slots(3)],
        [(5.0 / 15.0, 0.0), (5.0 / 15.0, 0.0), (0.0, 1.0), (0.0, 1.0)]
    );
}
