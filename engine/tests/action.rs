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

/// First and last id of every range in the frozen encoding table (Plan 1, Task 4). The policy
/// head and the Catanatron mapping depend on these; never renumber.
#[test]
fn golden_range_bounds() {
    use Resource::*;
    let pins = [
        (Action::Roll, 0),
        (Action::EndTurn, 1),
        (Action::BuyDev, 2),
        (Action::PlayKnight, 3),
        (Action::PlayRoadBuilding, 4),
        (Action::PlayMonopoly(Wood), 5),
        (Action::PlayMonopoly(Ore), 9),
        (Action::PlayYearOfPlenty(Wood, Wood), 10),
        (Action::PlayYearOfPlenty(Ore, Ore), 24),
        (Action::BuildSettlement(0), 25),
        (Action::BuildSettlement(53), 78),
        (Action::BuildCity(0), 79),
        (Action::BuildCity(53), 132),
        (Action::BuildRoad(0), 133),
        (Action::BuildRoad(71), 204),
        (Action::MoveRobber(0), 205),
        (Action::MoveRobber(18), 223),
        (Action::StealFrom(0), 224),
        (Action::StealFrom(3), 227),
        (Action::Discard(Wood), 228),
        (Action::Discard(Ore), 232),
        (
            Action::MaritimeTrade {
                give: Wood,
                get: Brick,
            },
            234,
        ),
        (
            Action::MaritimeTrade {
                give: Ore,
                get: Wheat,
            },
            256,
        ),
        (Action::OfferTrade { give: 0, get: 0 }, 258),
        (Action::OfferTrade { give: 19, get: 19 }, 657),
        (Action::AcceptTrade, 658),
        (Action::RejectTrade, 659),
        (Action::ConfirmTrade(0), 660),
        (Action::ConfirmTrade(3), 663),
        (Action::CancelTrade, 664),
    ];
    for (a, id) in pins {
        assert_eq!(a.encode(), id, "{a:?}");
        assert_eq!(Action::decode(id), Some(a), "id {id}");
    }
}
