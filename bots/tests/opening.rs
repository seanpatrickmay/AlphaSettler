//! The opening features (spec 3d-1 Section 4.2).

use settler_bots::opening::*;
use settler_engine::rng::Rng;
use settler_engine::*;
use settler_net::encode::layout_hash;

/// Seed 0's board, five settlements in: seat 0 on node 9, seat 1 on 12, seat 2 on 18, seat 3 on
/// 41 and then 26 (its second pick, right after its first); seat 3 is to place its road. What the
/// nodes touch (resource number: pips): 9 wheat 10:3, wheat 8:5, brick 5:4; 12 ore 10:3,
/// brick 9:4, sheep 8:5; 18 sheep 8:5, brick 5:4, ore 11:2; 41 wood 2:1, wheat 9:4, sheep 6:5;
/// 26 sheep 4:3, and it holds the brick 2:1 port.
fn five_placed() -> Game {
    let mut g = Game::new(0, GameConfig::default());
    for (k, n) in [9u8, 12, 18, 41, 26].into_iter().enumerate() {
        g.apply(Action::BuildSettlement(n)).unwrap();
        if k < 4 {
            let road = g.legal_actions()[0];
            g.apply(road).unwrap();
        }
    }
    assert_eq!(
        (g.state().phase, g.state().current),
        (Phase::SetupRoad { node: 26 }, 3)
    );
    g
}

#[test]
fn features_of_a_hand_built_setup_match_hand_counted_values() {
    let g = five_placed();
    let f = state_features(g.state(), 3);
    // Seat-relative: r0 = seat 3, r1 = seat 0, r2 = seat 1, r3 = seat 2.
    let mut want: Vec<f32> = Vec::new();
    // Pips per resource (wood brick sheep wheat ore) / 15. Seat 3: node 41 (wood 1, wheat 4,
    // sheep 5) and node 26 (sheep 3).
    want.extend([1.0, 0.0, 8.0, 4.0, 0.0].map(|x: f32| x / 15.0));
    want.extend([0.0, 4.0, 0.0, 8.0, 0.0].map(|x: f32| x / 15.0));
    want.extend([0.0, 4.0, 5.0, 0.0, 3.0].map(|x: f32| x / 15.0));
    want.extend([0.0, 4.0, 5.0, 0.0, 2.0].map(|x: f32| x / 15.0));
    // Total pips / 30.
    want.extend([13.0, 12.0, 12.0, 11.0].map(|x: f32| x / 30.0));
    // Resources touched / 5.
    want.extend([3.0, 2.0, 3.0, 3.0].map(|x: f32| x / 5.0));
    // Distinct numbers / 6: seat 3 {2, 9, 6, 4}; seat 0 {10, 8, 5}; seat 1 {10, 9, 8}; seat 2
    // {8, 5, 11}.
    want.extend([4.0, 3.0, 3.0, 3.0].map(|x: f32| x / 6.0));
    // Ports (3:1, then 2:1 wood brick sheep wheat ore): only seat 3's node 26, the brick 2:1.
    want.extend([0.0, 0.0, 1.0, 0.0, 0.0, 0.0]);
    want.extend([0.0; 18]);
    // Settlements placed / 2.
    want.extend([1.0, 0.5, 0.5, 0.5]);
    // The richest legal spot two steps from the seat's settlements, pips / 15: seat 3 node 31
    // (brick 5, sheep 4, wheat 9: 11); seat 0 node 19 (brick 5, wood 11, brick 5: 10); seat 1
    // node 22 (brick 9, wood 4, wood 12: 8); seat 2 node 19 (10).
    want.extend([11.0, 10.0, 8.0, 10.0].map(|x: f32| x / 15.0));
    // Settlements placed in all / 8; seat 3 has made both its picks.
    want.extend([5.0 / 8.0, 0.0]);
    // Most pips of each resource on a legal spot / 5: wood node 22 (4), brick node 19 (8), sheep
    // node 29 (5), wheat node 2 (5), ore node 0 (3).
    want.extend([4.0, 8.0, 5.0, 5.0, 3.0].map(|x: f32| x / 5.0));
    // The richest legal spot (node 31, 11) / 15; legal spots with at least 10 pips (19, 31) / 10.
    want.extend([11.0 / 15.0, 2.0 / 10.0]);
    // The board's pips per resource / 25.
    want.extend([7.0, 12.0, 18.0, 14.0, 7.0].map(|x: f32| x / 25.0));
    assert_eq!(want.len(), OPENING_FEATURES);
    assert_eq!(f.to_vec(), want);
}

/// Every per-seat block: (offset, width).
const SEAT_BLOCKS: [(usize, usize); 7] = [
    (SEAT_PIPS, 5),
    (SEAT_TOTAL, 1),
    (SEAT_RESOURCES, 1),
    (SEAT_NUMBERS, 1),
    (SEAT_PORTS, 6),
    (SEAT_PLACED, 1),
    (SEAT_REACH, 1),
];

#[test]
fn every_perspective_sees_the_same_seats_rotated() {
    let g = five_placed();
    let f: Vec<[f32; OPENING_FEATURES]> = (0..4u8).map(|p| state_features(g.state(), p)).collect();
    assert_eq!(
        SEAT_BLOCKS.iter().map(|&(_, w)| 4 * w).sum::<usize>(),
        PLACED_ALL
    );
    for p in 0..4 {
        for q in 0..4 {
            for &(base, w) in &SEAT_BLOCKS {
                for r in 0..4 {
                    // Absolute seat p + r is relative seat p + r - q from q's side.
                    let rq = (p + r + 4 - q) % 4;
                    assert_eq!(
                        f[p][base + w * r..base + w * (r + 1)],
                        f[q][base + w * rq..base + w * (rq + 1)],
                        "block {base}: r{r} from seat {p}, r{rq} from seat {q}"
                    );
                }
            }
            assert_eq!(f[p][PLACED_ALL], f[q][PLACED_ALL]);
            assert_eq!(f[p][BEST_RESOURCE..], f[q][BEST_RESOURCE..]);
        }
    }
    // Only seat 3 has placed both its settlements.
    assert_eq!([0, 1, 2, 3].map(|p| f[p][PICK_LEFT]), [1.0, 1.0, 1.0, 0.0]);
}

#[test]
fn legal_spots_are_the_engines_legal_setup_settlements() {
    let mut checked = 0;
    for seed in 0..6 {
        let mut g = Game::new(seed, GameConfig::default());
        let mut rng = Rng::new(seed);
        while matches!(
            g.state().phase,
            Phase::SetupSettlement | Phase::SetupRoad { .. }
        ) {
            let legal = g.legal_actions();
            if g.state().phase == Phase::SetupSettlement {
                let engine = legal.iter().fold(0u64, |m, a| match a {
                    Action::BuildSettlement(n) => m | (1u64 << *n),
                    _ => m,
                });
                let placed = std::array::from_fn(|q| g.state().players[q].settlements);
                assert_eq!(legal_spots(&placed), engine, "seed {seed}");
                checked += 1;
            }
            g.apply(legal[rng.below(legal.len() as u32) as usize])
                .unwrap();
        }
    }
    assert_eq!(checked, 6 * 8);
}

#[test]
fn the_layout_names_every_feature_once_in_vector_order() {
    let names = opening_feature_names();
    assert_eq!(names.len(), OPENING_FEATURES);
    let unique: std::collections::HashSet<&String> = names.iter().collect();
    assert_eq!(unique.len(), OPENING_FEATURES);
    for (at, name) in [
        (SEAT_PIPS + 5 * 2 + 3, "seat_pips[2].wheat/15"),
        (SEAT_TOTAL + 1, "seat_total[1].pips/30"),
        (SEAT_RESOURCES, "seat_resources[0].touched/5"),
        (SEAT_NUMBERS + 3, "seat_numbers[3].distinct/6"),
        (SEAT_PORTS + 6 + 2, "seat_ports[1].brick"),
        (SEAT_PLACED + 2, "seat_placed[2].settlements/2"),
        (SEAT_REACH + 3, "seat_reach[3].best_legal_at_distance_2/15"),
        (PLACED_ALL, "placed_all[0].settlements/8"),
        (PICK_LEFT, "pick_left[0].p"),
        (BEST_RESOURCE + 4, "best_legal[0].ore/5"),
        (BEST_SPOT, "best_legal_spot[0].pips/15"),
        (RICH_SPOTS, "rich_legal_spots[0].at_least_10/10"),
        (BOARD_RESOURCE, "board[0].wood/25"),
        (OPENING_FEATURES - 1, "board[0].ore/25"),
    ] {
        assert_eq!(names[at], name, "feature {at}");
    }
}

#[test]
fn the_hash_changes_when_a_feature_is_renamed_or_moved() {
    assert_eq!(
        OPENING_HASH,
        layout_hash(&[OPENING_FEATURES as u32], OPENING_LAYOUT)
    );
    let renamed = OPENING_LAYOUT.replace("pick_left x1: p", "pick_left x1: own");
    let moved = OPENING_LAYOUT.replace("wood/5 brick/5", "brick/5 wood/5");
    let block_moved = OPENING_LAYOUT.replace(
        "seat_placed x4: settlements/2\nseat_reach x4: best_legal_at_distance_2/15",
        "seat_reach x4: best_legal_at_distance_2/15\nseat_placed x4: settlements/2",
    );
    for other in [renamed, moved, block_moved] {
        assert_ne!(other, OPENING_LAYOUT, "the edit must apply");
        assert_ne!(
            layout_hash(&[OPENING_FEATURES as u32], &other),
            OPENING_HASH
        );
    }
}
