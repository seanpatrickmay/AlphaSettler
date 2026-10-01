use proptest::prelude::*;
use settler_engine::legal::legal_actions;
use settler_engine::rng::Rng;
use settler_engine::sim::play_random;
use settler_engine::*;

fn check_invariants(s: &State) {
    s.check_invariants().unwrap_or_else(|e| panic!("{e}"));
    check_vp(s);
}

/// VP recomputed from raw board fields, independent of `State::public_vp` / `total_vp`.
fn check_vp(s: &State) {
    for (i, p) in s.players.iter().enumerate() {
        let mut expect = p.settlements.count_ones() + 2 * p.cities.count_ones();
        if s.longest_road_owner == Some(i as PlayerId) {
            expect += 2;
        }
        if s.largest_army_owner == Some(i as PlayerId) {
            expect += 2;
        }
        assert_eq!(s.public_vp(i) as u32, expect, "public vp of player {i}");
    }
    if let Some(h) = s.longest_road_owner {
        let len = s.players[h as usize].longest_road_len;
        assert!(len >= 5, "longest road holder {h} has only {len}");
        assert!(
            s.players.iter().all(|p| p.longest_road_len <= len),
            "longer road than holder {h}"
        );
    }
    if let Some(h) = s.largest_army_owner {
        let k = s.players[h as usize].knights_played;
        assert!(k >= 3, "largest army holder {h} has only {k} knights");
        assert!(
            s.players.iter().all(|p| p.knights_played <= k),
            "more knights than holder {h}"
        );
    }
    if !s.is_over() {
        let c = s.current as usize;
        assert!(
            (s.total_vp(c) as u32) < s.config.vp_to_win as u32,
            "player {c} has {} vp but the game is not over",
            s.total_vp(c)
        );
    }
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
            for &other in &buf {
                let mut c = s;
                c.apply(other);
                check_invariants(&c);
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
    let cfg = GameConfig {
        max_offers_per_turn: 0,
        ..GameConfig::default()
    };
    let mut rng = Rng::new(1);
    let mut buf = Vec::new();
    for seed in 0..20 {
        let mut s = State::new(seed, cfg);
        let n = play_random(&mut s, &mut rng, &mut buf);
        assert!(s.is_over());
        assert!(n > 0);
    }
}

#[test]
fn turn_limit_ends_game_as_draw() {
    let cfg = GameConfig {
        max_turns: 5,
        max_offers_per_turn: 0,
        ..GameConfig::default()
    };
    let mut rng = Rng::new(3);
    let mut buf = Vec::new();
    let mut draws = 0;
    for seed in 0..20 {
        let mut s = State::new(seed, cfg);
        play_random(&mut s, &mut rng, &mut buf);
        assert!(s.is_over());
        if s.winner().is_none() {
            draws += 1;
            assert!(s.turn >= 5, "draw before the turn limit (turn {})", s.turn);
        }
    }
    assert!(draws >= 1, "no game hit the turn limit");
}
