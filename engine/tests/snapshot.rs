mod common;

use settler_engine::rng::Rng;
use settler_engine::state::SETUP_ORDER;
use settler_engine::topology::topo;
use settler_engine::*;

/// Positions from random games, sampled every 23 actions plus the end: one pass without
/// domestic trades and one with the default config. Positions with a pending trade are skipped,
/// since snapshots cannot represent them.
fn sample_states() -> Vec<State> {
    let no_trades = GameConfig {
        max_offers_per_turn: 0,
        ..GameConfig::default()
    };
    let mut out = sample_states_with(no_trades);
    out.extend(sample_states_with(GameConfig::default()));
    out
}

fn sample_states_with(cfg: GameConfig) -> Vec<State> {
    let mut out = Vec::new();
    for seed in 0..30 {
        let mut s = State::new(seed, cfg);
        let mut rng = Rng::new(seed ^ 0x5A);
        let mut step = 0;
        while !s.is_over() {
            if step % 23 == 0 && !matches!(s.phase, Phase::TradeResponse | Phase::TradeConfirm) {
                out.push(s);
            }
            let legal = s.legal_actions();
            s.apply(legal[rng.below(legal.len() as u32) as usize]);
            step += 1;
        }
        out.push(s);
    }
    out
}

#[test]
fn sampled_states_satisfy_invariants() {
    for s in sample_states() {
        s.check_invariants()
            .unwrap_or_else(|e| panic!("seed {}: {e}", s.seed));
    }
}

#[test]
fn snapshot_round_trips() {
    for s in sample_states() {
        let snap = s.snapshot();
        let t = State::from_snapshot(s.seed, s.config, &snap)
            .unwrap_or_else(|e| panic!("seed {}: {e}", s.seed));
        assert_eq!(t.snapshot(), snap);
        assert_eq!(t.legal_actions(), s.legal_actions());
        assert_eq!(t.observation(1), s.observation(1));
    }
}

#[test]
fn samples_cover_offers_made_this_turn() {
    // Without this, the trades-on pass could round-trip only offers_this_turn == 0.
    assert!(sample_states().iter().any(|s| s.offers_this_turn > 0));
}

#[test]
fn game_from_snapshot() {
    let s = common::after_setup(5);
    let g = Game::from_snapshot(5, s.config, &s.snapshot()).unwrap();
    assert_eq!(g.legal_actions(), s.legal_actions());
}

#[test]
fn rejects_inconsistent_snapshots() {
    let base = common::after_setup(3).snapshot();
    let cfg = GameConfig::default();
    let expect_err = |change: &dyn Fn(&mut Snapshot), needle: &str| {
        let mut snap = base.clone();
        change(&mut snap);
        let e = State::from_snapshot(3, cfg, &snap).expect_err(needle);
        assert!(e.contains(needle), "expected {needle:?} in {e:?}");
    };
    expect_err(&|s| s.bank[0] += 1, "resource 0");
    expect_err(
        &|s| s.players[1].settlements |= s.players[0].settlements,
        "share a node",
    );
    expect_err(
        &|s| s.dev_deck.iter_mut().for_each(|c| *c = DevCard::Monopoly),
        "Monopoly",
    );
    expect_err(&|s| s.dev_deck.push(DevCard::Knight), "dev_deck");
    expect_err(
        &|s| {
            let d = s.board.desert() as usize;
            s.board.tile_resource[d] = Some(Resource::Wood);
        },
        "desert",
    );
    expect_err(
        &|s| s.phase = Phase::TradeResponse,
        "pending domestic trade",
    );
    expect_err(&|s| s.players[2].discard_remaining = 3, "Discard");
    expect_err(&|s| s.players[0].roads = u128::MAX, "roads");
    expect_err(&|s| s.players[0].settlements = u64::MAX, "settlements");
    expect_err(&|s| s.robber = 19, "robber");
    expect_err(&|s| s.current = 4, "current");
    expect_err(&|s| s.phase = Phase::GameOver { winner: Some(9) }, "winner");
    expect_err(&|s| s.return_phase = Phase::Discard, "return_phase");
    // Steal with nobody to rob: the robber sits where no opponent of player 0 has a building.
    expect_err(
        &|s| {
            assert_eq!(s.current, 0);
            let others = s.players[1..]
                .iter()
                .fold(0u64, |m, p| m | p.settlements | p.cities);
            let tile = (0..19)
                .find(|&t| topo().tile_node_mask[t] & others == 0)
                .expect("a tile with no opponent building");
            s.robber = tile as u8;
            s.phase = Phase::Steal;
        },
        "no one to steal from",
    );
    // A discard the player cannot pay.
    expect_err(
        &|s| {
            let h = s.players[0].hand;
            hand_add(&mut s.bank, &h);
            s.players[0].hand = [0; NUM_RESOURCES];
            s.players[0].discard_remaining = 1;
            s.phase = Phase::Discard;
        },
        "holds only 0 cards",
    );
    // Road Building with no road network to extend.
    expect_err(
        &|s| {
            let p = &mut s.players[0];
            p.settlements = 0;
            p.cities = 0;
            p.roads = 0;
            s.phase = Phase::RoadBuilding { roads_left: 1 };
        },
        "nowhere to build",
    );
    // Road Building with more roads left than road pieces.
    expect_err(
        &|s| {
            let others = s.players[1..].iter().fold(0u128, |m, p| m | p.roads);
            let mut roads = 0u128;
            for e in 0..72 {
                if others >> e & 1 == 0 && roads.count_ones() < MAX_ROADS {
                    roads |= 1u128 << e;
                }
            }
            s.players[0].roads = roads;
            s.phase = Phase::RoadBuilding { roads_left: 2 };
        },
        "only 0 road pieces",
    );
    expect_err(&|s| s.turn = GameConfig::default().max_turns, "max_turns");
    expect_err(&|s| s.offers_this_turn = 4, "offers_this_turn 4 exceeds");
    expect_err(&|s| s.setup_step = 9, "setup_step must be 8");
    expect_err(
        &|s| {
            let p = (0..NUM_PLAYERS)
                .find(|&p| hand_total(&s.players[p].hand) > 0)
                .unwrap();
            s.players[p].discard_remaining = 1;
            s.phase = Phase::Discard;
            s.return_phase = Phase::PreRoll;
        },
        "Discard phase must return to Main",
    );
}

/// Setup played by always taking the first legal action, stopped at the settlement of `step`.
fn during_setup(seed: u64, step: u8) -> State {
    let mut s = State::new(seed, GameConfig::default());
    while s.setup_step < step || s.phase != Phase::SetupSettlement {
        s.apply(s.legal_actions()[0]);
    }
    s
}

fn reject(snap: &Snapshot, needle: &str) {
    let e = State::from_snapshot(3, GameConfig::default(), snap).expect_err(needle);
    assert!(e.contains(needle), "expected {needle:?} in {e:?}");
}

#[test]
fn rejects_inconsistent_setup_snapshots() {
    let cfg = GameConfig::default();
    let second_round = during_setup(3, 4).snapshot();
    State::from_snapshot(3, cfg, &second_round).unwrap();

    // The bank cannot pay out a second-round settlement.
    let mut snap = second_round.clone();
    let p = snap.current as usize;
    snap.players[p].hand[0] += snap.bank[0] - 2;
    snap.bank[0] = 2;
    reject(&snap, "bank holds only 2");

    // The wrong player placing.
    let mut snap = second_round.clone();
    assert_eq!(snap.current, SETUP_ORDER[4]);
    snap.current = (SETUP_ORDER[4] + 1) % 4;
    reject(&snap, "placement");

    // A setup road with every edge at its node taken. Step 6 has 6 roads down, enough to cover
    // the node's 2 or 3 edges while keeping the road count the step requires.
    let mut s = during_setup(3, 6);
    s.apply(s.legal_actions()[0]);
    let Phase::SetupRoad { node } = s.phase else {
        panic!("expected SetupRoad, got {:?}", s.phase)
    };
    let road_phase = s.snapshot();
    State::from_snapshot(3, cfg, &road_phase).unwrap();
    let mut snap = road_phase.clone();
    let blocked = topo().node_edge_mask[node as usize];
    let mut rest = (0..72u32).map(|e| 1u128 << e).filter(|&b| b & blocked == 0);
    let mut roads = blocked;
    while roads.count_ones() < 6 {
        roads |= rest.next().unwrap();
    }
    for p in snap.players.iter_mut() {
        p.roads = 0;
    }
    snap.players[0].roads = roads;
    reject(&snap, "no free edge");

    // A setup road at a node that does not exist.
    let mut s = during_setup(3, 2);
    s.apply(s.legal_actions()[0]);
    let road_phase = s.snapshot();

    // A setup road at a node that does not exist.
    let mut snap = road_phase.clone();
    snap.phase = Phase::SetupRoad { node: 60 };
    reject(&snap, "node 60 is out of range");
}

#[test]
fn setup_piece_counts_must_match_setup_step() {
    // Step 4 is a settlement turn: 4 settlements and 4 roads placed, no cities.
    let settle = during_setup(3, 4).snapshot();
    let owner = |snap: &Snapshot| {
        (0..NUM_PLAYERS)
            .find(|&p| snap.players[p].settlements != 0)
            .unwrap()
    };
    let mut snap = settle.clone();
    let p = owner(&snap);
    let lowest = snap.players[p].settlements & snap.players[p].settlements.wrapping_neg();
    snap.players[p].settlements &= !lowest;
    reject(
        &snap,
        "setup_step 4 needs 4 settlements on the board, found 3",
    );
    let mut snap = settle.clone();
    let p = (0..NUM_PLAYERS)
        .find(|&p| snap.players[p].roads != 0)
        .unwrap();
    let lowest = snap.players[p].roads & snap.players[p].roads.wrapping_neg();
    snap.players[p].roads &= !lowest;
    reject(&snap, "setup_step 4 needs 4 roads on the board, found 3");
    let mut snap = settle.clone();
    let p = owner(&snap);
    snap.players[p].cities = snap.players[p].settlements;
    snap.players[p].settlements = 0;
    reject(&snap, "no cities during setup");

    // Step 2's road turn: its settlement is down, so 3 settlements and 2 roads.
    let mut s = during_setup(3, 2);
    s.apply(s.legal_actions()[0]);
    let road_phase = s.snapshot();
    State::from_snapshot(3, GameConfig::default(), &road_phase).unwrap();
    let mut snap = road_phase.clone();
    let p = (0..NUM_PLAYERS)
        .find(|&p| p != snap.current as usize && snap.players[p].settlements != 0)
        .unwrap();
    snap.players[p].settlements = 0;
    reject(
        &snap,
        "setup_step 2 needs 3 settlements on the board, found 2",
    );
}

#[test]
fn rejects_an_unclaimed_win_by_the_player_on_turn() {
    // Player 0 is on turn with 2 settlements; Longest Road makes 4, enough to win at 3 or 4.
    let base = common::after_setup(3).snapshot();
    assert_eq!(base.current, 0);
    let cfg = GameConfig {
        vp_to_win: 4,
        ..GameConfig::default()
    };
    State::from_snapshot(3, cfg, &base).unwrap();
    let mut snap = base.clone();
    snap.longest_road_owner = Some(0);
    let e = State::from_snapshot(3, cfg, &snap).expect_err("unclaimed win");
    assert!(e.contains("player 0 has 4 VP"), "{e}");
    // Another player at the target off turn is a legal position: they win on their own turn.
    let mut snap = base.clone();
    snap.longest_road_owner = Some(1);
    State::from_snapshot(3, cfg, &snap).unwrap();
}

#[test]
fn only_the_player_on_turn_holds_cards_bought_this_turn() {
    let base = common::after_setup(3).snapshot();
    assert_eq!(base.current, 0);
    for p in [0, 1] {
        let mut snap = base.clone();
        let card = snap.dev_deck.remove(0);
        snap.players[p].dev_hand[card.index()] += 1;
        snap.players[p].dev_new[card.index()] += 1;
        let r = State::from_snapshot(3, GameConfig::default(), &snap);
        if p == 0 {
            r.unwrap();
        } else {
            let e = r.expect_err("dev_new off turn");
            assert!(e.contains("player 1 holds cards bought this turn"), "{e}");
        }
    }
}

#[test]
fn board_validation() {
    let b = State::new(1, GameConfig::default()).board;
    assert!(b.validate().is_ok());
    let mut bad = b;
    bad.tile_number[(b.desert() as usize + 1) % 19] = 7;
    assert!(bad.validate().unwrap_err().contains("number 7"));
    let mut bad = b;
    bad.ports[1].0 = bad.ports[0].0;
    assert!(bad.validate().unwrap_err().contains("share edge"));
    let mut bad = b;
    bad.generic_port_nodes ^= 1;
    assert!(bad.validate().unwrap_err().contains("port node masks"));
}

#[test]
fn reseed_changes_only_the_random_streams() {
    let fresh = State::new(7, GameConfig::default());
    let mut same = fresh;
    same.reseed(7);
    assert_eq!(same, fresh);
    let mut other = fresh;
    other.reseed(8);
    assert_eq!(other.seed, 8);
    assert_ne!(other.rng_steal, fresh.rng_steal);
    assert_eq!(other.snapshot(), fresh.snapshot());
}
