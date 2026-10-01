use proptest::prelude::*;
use settler_engine::legal::legal_actions;
use settler_engine::rng::Rng;
use settler_engine::sim::play_random;
use settler_engine::topology::topo;
use settler_engine::*;

fn check_invariants(s: &State) {
    for r in 0..NUM_RESOURCES {
        let held: u32 = s.players.iter().map(|p| p.hand[r] as u32).sum();
        assert_eq!(held + s.bank[r] as u32, BANK_PER_RESOURCE as u32, "resource {r} not conserved");
    }
    let t = topo();
    let mut nodes = 0u64;
    let mut edges = 0u128;
    for (i, p) in s.players.iter().enumerate() {
        assert!(p.settlements.count_ones() <= MAX_SETTLEMENTS, "player {i} settlements");
        assert!(p.cities.count_ones() <= MAX_CITIES, "player {i} cities");
        assert!(p.roads.count_ones() <= MAX_ROADS, "player {i} roads");
        assert_eq!(p.settlements & p.cities, 0);
        let b = p.settlements | p.cities;
        assert_eq!(nodes & b, 0, "two players share a node");
        nodes |= b;
        assert_eq!(edges & p.roads, 0, "two players share an edge");
        edges |= p.roads;
        for c in 0..5 {
            assert!(p.dev_new[c] <= p.dev_hand[c]);
        }
    }
    for n in bits64(nodes) {
        assert_eq!(nodes & t.node_neighbor_mask[n as usize], 0, "distance rule broken at {n}");
    }
    let held_dev: u32 = s.players.iter().flat_map(|p| p.dev_hand.iter()).map(|&c| c as u32).sum();
    assert!(held_dev <= s.dev_deck_pos as u32);
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(48))]

    #[test]
    fn random_games_keep_invariants(seed in any::<u64>(), trades in any::<bool>(), compat in any::<bool>()) {
        let cfg = GameConfig {
            max_offers_per_turn: if trades { 3 } else { 0 },
            catanatron_compat: compat,
            ..GameConfig::default()
        };
        let mut s = State::new(seed, cfg);
        let mut rng = Rng::new(seed ^ 0xABCD);
        let mut buf = Vec::new();
        let mut steps = 0u64;
        while !s.is_over() {
            legal_actions(&s, &mut buf);
            prop_assert!(!buf.is_empty(), "no legal actions in {:?}", s.phase);
            for &a in &buf {
                prop_assert_eq!(Action::decode(a.encode()), Some(a));
            }
            let a = buf[rng.below(buf.len() as u32) as usize];
            s.apply(a);
            check_invariants(&s);
            steps += 1;
            prop_assert!(steps < 2_000_000, "game did not terminate");
        }
        match s.winner() {
            Some(w) => prop_assert!(s.total_vp(w as usize) >= cfg.vp_to_win),
            None => prop_assert!(s.turn >= cfg.max_turns),
        }
    }
}

#[test]
fn play_random_finishes_games() {
    let cfg = GameConfig { max_offers_per_turn: 0, ..GameConfig::default() };
    let mut rng = Rng::new(1);
    let mut buf = Vec::new();
    for seed in 0..20 {
        let mut s = State::new(seed, cfg);
        let n = play_random(&mut s, &mut rng, &mut buf);
        assert!(s.is_over());
        assert!(n > 0);
    }
}
