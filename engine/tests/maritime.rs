mod common;
use common::*;
use settler_engine::topology::topo;
use settler_engine::*;
use Resource::*;

fn maritime(s: &State) -> Vec<Action> {
    sorted(
        s.legal_actions()
            .into_iter()
            .filter(|a| matches!(a, Action::MaritimeTrade { .. }))
            .collect(),
    )
}

#[test]
fn default_rate_is_four() {
    let mut s = blank(1, Phase::Main);
    give(&mut s, 0, [3, 0, 0, 0, 0]);
    assert!(maritime(&s).is_empty());
    give(&mut s, 0, [1, 0, 0, 0, 0]);
    let expected: Vec<Action> = [Brick, Sheep, Wheat, Ore]
        .iter()
        .map(|&g| Action::MaritimeTrade { give: Wood, get: g })
        .collect();
    assert_eq!(maritime(&s), sorted(expected));
    s.apply(Action::MaritimeTrade {
        give: Wood,
        get: Ore,
    });
    assert_eq!(s.players[0].hand, [0, 0, 0, 0, 1]);
    assert_eq!(s.bank, [19, 19, 19, 19, 18]);
}

#[test]
fn generic_port_is_three() {
    let mut s = blank(1, Phase::Main);
    let (e, _) = *s
        .board
        .ports
        .iter()
        .find(|p| p.1 == PortKind::Generic)
        .unwrap();
    s.players[0].settlements = 1u64 << topo().edge_nodes[e as usize].0;
    give(&mut s, 0, [0, 0, 3, 0, 0]);
    assert_eq!(maritime(&s).len(), 4);
    s.apply(Action::MaritimeTrade {
        give: Sheep,
        get: Wood,
    });
    assert_eq!(s.players[0].hand, [1, 0, 0, 0, 0]);
}

#[test]
fn specific_port_is_two_for_its_resource_only() {
    let mut s = blank(1, Phase::Main);
    let (e, kind) = *s
        .board
        .ports
        .iter()
        .find(|p| p.1 != PortKind::Generic)
        .unwrap();
    let PortKind::Specific(r) = kind else {
        unreachable!()
    };
    s.players[0].cities = 1u64 << topo().edge_nodes[e as usize].1;
    let other = Resource::ALL.into_iter().find(|&x| x != r).unwrap();
    let mut h = [0u8; 5];
    h[r.index()] = 2;
    h[other.index()] = 2;
    give(&mut s, 0, h);
    let legal = maritime(&s);
    assert!(legal
        .iter()
        .all(|a| matches!(a, Action::MaritimeTrade { give, .. } if *give == r)));
    assert_eq!(legal.len(), 4);
}

#[test]
fn cannot_take_from_empty_bank_pile() {
    let mut s = blank(1, Phase::Main);
    give(&mut s, 0, [4, 0, 0, 0, 0]);
    s.bank[Ore.index()] = 0;
    assert!(!maritime(&s).contains(&Action::MaritimeTrade {
        give: Wood,
        get: Ore
    }));
    assert_eq!(maritime(&s).len(), 3);
}
