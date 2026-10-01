use settler_engine::action::*;
use settler_engine::types::{hand_total, Resource};

#[test]
fn space_size() {
    assert_eq!(ACTION_SPACE_SIZE, 665);
}

#[test]
fn decode_encode_roundtrip() {
    let mut n = 0;
    for id in 0..ACTION_SPACE_SIZE as u16 {
        if let Some(a) = Action::decode(id) {
            assert_eq!(a.encode(), id, "{a:?}");
            n += 1;
        }
    }
    assert_eq!(n, 660);
    assert_eq!(Action::decode(ACTION_SPACE_SIZE as u16), None);
}

#[test]
fn known_offsets() {
    assert_eq!(Action::Roll.encode(), 0);
    assert_eq!(Action::BuildSettlement(0).encode(), 25);
    assert_eq!(Action::BuildRoad(71).encode(), 204);
    assert_eq!(Action::MoveRobber(18).encode(), 223);
    assert_eq!(Action::OfferTrade { give: 19, get: 19 }.encode(), 657);
    assert_eq!(Action::CancelTrade.encode(), 664);
}

#[test]
fn maritime_same_resource_is_not_an_action() {
    assert_eq!(Action::decode(233), None); // give Wood, get Wood
    assert_eq!(
        Action::decode(234),
        Some(Action::MaritimeTrade {
            give: Resource::Wood,
            get: Resource::Brick
        })
    );
}

#[test]
fn year_of_plenty_order_is_canonical() {
    use Resource::*;
    assert_eq!(
        Action::PlayYearOfPlenty(Ore, Wood).encode(),
        Action::PlayYearOfPlenty(Wood, Ore).encode()
    );
    assert_eq!(
        Action::decode(Action::PlayYearOfPlenty(Ore, Wood).encode()),
        Some(Action::PlayYearOfPlenty(Wood, Ore))
    );
}

#[test]
fn bundles() {
    assert_eq!(bundle(0), [1, 0, 0, 0, 0]);
    assert_eq!(bundle(5), [2, 0, 0, 0, 0]);
    assert_eq!(bundle(6), [1, 1, 0, 0, 0]);
    assert_eq!(bundle(19), [0, 0, 0, 0, 2]);
    let mut all = Vec::new();
    for i in 0..NUM_BUNDLES as u8 {
        assert_eq!(hand_total(&bundle(i)) as u8, bundle_size(i));
        all.push(bundle(i));
    }
    all.sort();
    all.dedup();
    assert_eq!(all.len(), NUM_BUNDLES);
}
