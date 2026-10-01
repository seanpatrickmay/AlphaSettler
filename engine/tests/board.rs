use settler_engine::board::*;
use settler_engine::config::GameConfig;
use settler_engine::rng::*;
use settler_engine::topology::topo;
use settler_engine::types::Resource;

#[test]
fn rng_is_deterministic_and_bounded() {
    let mut a = Rng::new(5);
    let mut b = Rng::new(5);
    for _ in 0..100 {
        assert_eq!(a.next_u64(), b.next_u64());
    }
    let mut r = Rng::new(1);
    let mut seen = [false; 6];
    for _ in 0..1000 {
        let x = r.below(6);
        assert!(x < 6);
        seen[x as usize] = true;
    }
    assert!(seen.iter().all(|&s| s));
}

#[test]
fn shuffle_is_a_permutation() {
    let mut xs: Vec<u32> = (0..50).collect();
    Rng::new(3).shuffle(&mut xs);
    let mut sorted = xs.clone();
    sorted.sort();
    assert_eq!(sorted, (0..50).collect::<Vec<_>>());
    assert_ne!(xs, sorted);
}

#[test]
fn dice_depend_only_on_seed_and_turn() {
    assert_eq!(dice_for(9, 3), dice_for(9, 3));
    let rolls: Vec<_> = (0..50).map(|t| dice_for(9, t)).collect();
    assert!(rolls
        .iter()
        .all(|&(a, b)| (1..=6).contains(&a) && (1..=6).contains(&b)));
    assert!(rolls.windows(2).any(|w| w[0] != w[1]));
    assert_ne!(
        (0..20).map(|t| dice_for(1, t)).collect::<Vec<_>>(),
        (0..20).map(|t| dice_for(2, t)).collect::<Vec<_>>()
    );
}

#[test]
fn default_config_matches_spec() {
    let c = GameConfig::default();
    assert_eq!(
        (
            c.vp_to_win,
            c.discard_limit,
            c.max_offers_per_turn,
            c.max_trade_cards,
            c.max_turns,
            c.catanatron_compat
        ),
        (10, 7, 3, 2, 1000, false)
    );
    assert!(c.validate().is_ok());
    assert!(GameConfig {
        max_trade_cards: 3,
        ..c
    }
    .validate()
    .is_err());
    assert!(GameConfig {
        max_trade_cards: 0,
        ..c
    }
    .validate()
    .is_err());
}

#[test]
fn standard_board_composition() {
    for seed in 0..50 {
        let b = Board::random(&mut Rng::new(seed));
        let count = |r: Option<Resource>| b.tile_resource.iter().filter(|&&x| x == r).count();
        assert_eq!(count(Some(Resource::Wood)), 4);
        assert_eq!(count(Some(Resource::Brick)), 3);
        assert_eq!(count(Some(Resource::Sheep)), 4);
        assert_eq!(count(Some(Resource::Wheat)), 4);
        assert_eq!(count(Some(Resource::Ore)), 3);
        assert_eq!(count(None), 1);
        let d = b.desert() as usize;
        assert_eq!(b.tile_number[d], 0);
        let mut nums: Vec<u8> = (0..19)
            .filter(|&i| i != d)
            .map(|i| b.tile_number[i])
            .collect();
        nums.sort();
        assert_eq!(nums, STANDARD_NUMBERS.to_vec());
    }
}

#[test]
fn no_adjacent_red_numbers() {
    let t = topo();
    for seed in 0..200 {
        let b = Board::random(&mut Rng::new(seed));
        let red = |n: u8| n == 6 || n == 8;
        for i in 0..19 {
            if red(b.tile_number[i]) {
                for &j in &t.tile_neighbors[i] {
                    assert!(
                        !red(b.tile_number[j as usize]),
                        "seed {seed}: tiles {i},{j}"
                    );
                }
            }
        }
    }
}

#[test]
fn ports_are_standard() {
    let t = topo();
    let b = Board::random(&mut Rng::new(11));
    let mut edges: Vec<u8> = b.ports.iter().map(|p| p.0).collect();
    edges.sort();
    edges.dedup();
    assert_eq!(edges.len(), 9);
    for &(e, _) in &b.ports {
        assert!(t.coastal_edges.contains(&e));
    }
    assert_eq!(
        b.ports.iter().filter(|p| p.1 == PortKind::Generic).count(),
        4
    );
    for r in Resource::ALL {
        assert_eq!(
            b.ports
                .iter()
                .filter(|p| p.1 == PortKind::Specific(r))
                .count(),
            1
        );
    }
}

#[test]
fn maritime_rates_follow_ports() {
    let t = topo();
    let b = Board::random(&mut Rng::new(4));
    assert_eq!(b.maritime_rate(0, Resource::Wood), 4);
    let (ge, _) = *b.ports.iter().find(|p| p.1 == PortKind::Generic).unwrap();
    let gnode = 1u64 << t.edge_nodes[ge as usize].0;
    for r in Resource::ALL {
        assert_eq!(b.maritime_rate(gnode, r), 3);
    }
    let (se, kind) = *b.ports.iter().find(|p| p.1 != PortKind::Generic).unwrap();
    let PortKind::Specific(sr) = kind else {
        unreachable!()
    };
    let snode = 1u64 << t.edge_nodes[se as usize].1;
    assert_eq!(b.maritime_rate(snode, sr), 2);
    let other = Resource::ALL.into_iter().find(|&r| r != sr).unwrap();
    assert_eq!(b.maritime_rate(snode, other), 4);
}

#[test]
fn boards_are_seed_deterministic() {
    assert_eq!(
        Board::random(&mut Rng::new(8)),
        Board::random(&mut Rng::new(8))
    );
    let first = Board::random(&mut Rng::new(0));
    assert!((1..10).any(|s| Board::random(&mut Rng::new(s)) != first));
}
