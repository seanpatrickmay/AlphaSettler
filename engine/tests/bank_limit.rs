mod common;
use common::*;
use settler_engine::topology::topo;
use settler_engine::*;

fn unlimited() -> GameConfig {
    GameConfig {
        bank_limit: false,
        ..GameConfig::default()
    }
}

/// The first tile with a resource: its resource, its number and two opposite corners (never adjacent).
fn producing_tile(s: &State) -> (Resource, u8, u8, u8) {
    let t = (0..19)
        .find(|&t| s.board.tile_resource[t].is_some())
        .unwrap();
    let nodes = topo().tile_nodes[t];
    (
        s.board.tile_resource[t].unwrap(),
        s.board.tile_number[t],
        nodes[0],
        nodes[3],
    )
}

/// Players 0 and 1 settled on that tile; player 2 holds all 19 of its resource, so the pile is empty.
fn two_claimants(config: GameConfig) -> State {
    let mut s = blank_with(5, Phase::PreRoll, config);
    let (r, _, a, b) = producing_tile(&s);
    s.players[0].settlements |= 1u64 << a;
    s.players[1].settlements |= 1u64 << b;
    s.players[2].hand[r.index()] = 19;
    s.bank[r.index()] = 0;
    s
}

fn held(s: &State, r: usize) -> u32 {
    s.players.iter().map(|p| p.hand[r] as u32).sum()
}

#[test]
fn the_bank_limit_is_on_by_default() {
    assert!(GameConfig::default().bank_limit);
}

#[test]
fn with_the_limit_two_claimants_of_an_empty_pile_get_none_of_it() {
    let mut s = two_claimants(GameConfig::default());
    let (r, n, _, _) = producing_tile(&s);
    roll(&mut s, n);
    assert_eq!(s.players[0].hand[r.index()], 0);
    assert_eq!(s.players[1].hand[r.index()], 0);
    s.check_invariants().unwrap();
}

#[test]
fn without_the_limit_both_claimants_are_paid_and_the_bank_shows_none() {
    let mut s = two_claimants(unlimited());
    let (r, n, _, _) = producing_tile(&s);
    roll(&mut s, n);
    assert!(s.players[0].hand[r.index()] >= 1);
    assert!(s.players[1].hand[r.index()] >= 1);
    assert!(held(&s, r.index()) > 19);
    assert_eq!(s.bank[r.index()], 0);
    s.check_invariants().unwrap();
}

#[test]
fn without_the_limit_maritime_trades_and_year_of_plenty_pay_from_an_empty_pile() {
    for config in [GameConfig::default(), unlimited()] {
        let mut s = blank_with(2, Phase::Main, config);
        give(&mut s, 2, [0, 0, 0, 0, 19]);
        give(&mut s, 0, [4, 0, 0, 0, 0]);
        s.players[0].dev_hand[DevCard::YearOfPlenty.index()] = 1;
        let trade = Action::MaritimeTrade {
            give: Resource::Wood,
            get: Resource::Ore,
        };
        let plenty = Action::PlayYearOfPlenty(Resource::Ore, Resource::Ore);
        let legal = s.legal_actions();
        assert_eq!(legal.contains(&trade), !config.bank_limit, "{config:?}");
        assert_eq!(legal.contains(&plenty), !config.bank_limit, "{config:?}");
        if !config.bank_limit {
            s.apply(trade);
            s.apply(plenty);
            assert_eq!(s.players[0].hand, [0, 0, 0, 0, 3]);
            assert_eq!(s.bank[Resource::Ore.index()], 0);
            assert_eq!(s.bank[Resource::Wood.index()], 19);
        }
    }
}

#[test]
fn a_snapshot_with_more_than_19_of_a_resource_in_hands_imports_only_without_the_limit() {
    let mut s = blank_with(1, Phase::Main, unlimited());
    s.players[0].hand = [20, 0, 0, 0, 0];
    s.refresh_bank();
    assert_eq!(s.bank, [0, 19, 19, 19, 19]);
    let snap = s.snapshot();
    assert!(State::from_snapshot(1, unlimited(), &snap).is_ok());
    let err = State::from_snapshot(1, GameConfig::default(), &snap).unwrap_err();
    assert!(err.contains("hands and bank hold"), "{err}");
}

#[test]
fn games_without_the_limit_show_the_derived_bank_after_every_action() {
    for seed in 0..12u64 {
        let mut s = State::new(seed, unlimited());
        let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        while !s.is_over() {
            let legal = s.legal_actions();
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            s.apply(legal[(x % legal.len() as u64) as usize]);
            for r in 0..NUM_RESOURCES {
                assert_eq!(
                    s.bank[r] as u32,
                    19u32.saturating_sub(held(&s, r)),
                    "seed {seed} resource {r}"
                );
            }
            s.check_invariants().unwrap();
        }
    }
}
