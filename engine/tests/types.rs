use settler_engine::types::*;

#[test]
fn resource_index_roundtrip() {
    for (i, r) in Resource::ALL.iter().enumerate() {
        assert_eq!(r.index(), i);
        assert_eq!(Resource::from_index(i), *r);
    }
}

#[test]
fn dev_card_index_roundtrip() {
    for (i, c) in DevCard::ALL.iter().enumerate() {
        assert_eq!(c.index(), i);
        assert_eq!(DevCard::from_index(i), *c);
    }
}

#[test]
fn dev_deck_has_25_cards() {
    let total: usize = DEV_DECK_COUNTS.iter().map(|&c| c as usize).sum();
    assert_eq!(total, DEV_DECK_SIZE);
    assert_eq!(DEV_DECK_COUNTS[DevCard::Knight.index()], 14);
    assert_eq!(DEV_DECK_COUNTS[DevCard::VictoryPoint.index()], 5);
}

#[test]
fn costs_match_rulebook() {
    assert_eq!(ROAD_COST, [1, 1, 0, 0, 0]);
    assert_eq!(SETTLEMENT_COST, [1, 1, 1, 1, 0]);
    assert_eq!(CITY_COST, [0, 0, 0, 2, 3]);
    assert_eq!(DEV_COST, [0, 0, 1, 1, 1]);
}

#[test]
fn hand_arithmetic() {
    let mut h: Hand = [2, 1, 0, 0, 0];
    assert!(covers(&h, &ROAD_COST));
    hand_sub(&mut h, &ROAD_COST);
    assert_eq!(h, [1, 0, 0, 0, 0]);
    assert!(!covers(&h, &ROAD_COST));
    hand_add(&mut h, &ROAD_COST);
    assert_eq!(h, [2, 1, 0, 0, 0]);
    assert_eq!(hand_total(&h), 3);
}

#[test]
fn bit_iterators() {
    assert_eq!(bits64(0b1010_0001).collect::<Vec<_>>(), vec![0, 5, 7]);
    assert_eq!(bits128((1u128 << 100) | 1).collect::<Vec<_>>(), vec![0, 100]);
    assert_eq!(bits64(0).count(), 0);
}
