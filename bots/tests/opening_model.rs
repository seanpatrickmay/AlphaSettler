//! The opening model file, its forward pass, the `+o` name suffix and `+o` bots (spec 3d-1
//! Sections 4.3 and 4.5).

mod common;

use common::trading;
use settler_bots::arena::run_match_in;
use settler_bots::netbot::{parse_net_name, NetName};
use settler_bots::opening::*;
use settler_bots::opening_model::*;
use settler_bots::{make_bot, Bot, Roster};
use settler_engine::rng::Rng;
use settler_engine::topology::topo;
use settler_engine::*;
use settler_net::encode::pips;
use settler_net::weights::{Arch, Weights};
use std::path::PathBuf;

fn temp_dir(tag: &str) -> PathBuf {
    let dir =
        std::env::temp_dir().join(format!("settler-bots-opening-{}-{tag}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// A nets directory holding `tiny.bin` (random weights, width 8, 1 round) and `opening.bin` if
/// `opening` is given (and no `opening.bin` otherwise).
fn nets_dir(tag: &str, opening: Option<&OpeningModel>) -> PathBuf {
    let dir = temp_dir(tag);
    Weights::random(
        Arch {
            width: 8,
            rounds: 1,
        },
        1,
    )
    .save(&dir.join("tiny.bin"))
    .unwrap();
    let path = dir.join(OPENING_FILE);
    match opening {
        Some(m) => m.save(&path).unwrap(),
        None => {
            let _ = std::fs::remove_file(&path);
        }
    }
    dir
}

/// A model with the given hidden widths between the 78 features and the 4 seats, and small
/// weights drawn from `seed`.
fn model(hidden: &[usize], seed: u64) -> OpeningModel {
    let mut rng = Rng::new(seed);
    let mut dims = vec![OPENING_FEATURES];
    dims.extend_from_slice(hidden);
    dims.push(NUM_PLAYERS);
    let mut layers = Vec::new();
    for d in dims.windows(2) {
        let (cols, rows) = (d[0], d[1]);
        let mut x = || (rng.next_u64() >> 40) as f32 / (1u64 << 24) as f32 * 0.2 - 0.1;
        let w = (0..rows * cols).map(|_| x()).collect();
        let b = (0..rows).map(|_| x()).collect();
        layers.push(OpeningLayer { rows, cols, w, b });
    }
    OpeningModel { layers }
}

/// A one-layer model whose own-seat logit is `weight` times feature `k`; every other weight is 0.
fn one_feature(k: usize, weight: f32) -> OpeningModel {
    let mut w = vec![0.0; NUM_PLAYERS * OPENING_FEATURES];
    w[k] = weight;
    OpeningModel {
        layers: vec![OpeningLayer {
            rows: NUM_PLAYERS,
            cols: OPENING_FEATURES,
            w,
            b: vec![0.0; NUM_PLAYERS],
        }],
    }
}

fn set_u32(b: &mut [u8], at: usize, v: u32) {
    b[at..at + 4].copy_from_slice(&v.to_le_bytes());
}

fn searches(bot: &dyn Bot) -> u64 {
    bot.diagnostics()
        .into_iter()
        .find(|(k, _)| *k == "searches")
        .map(|(_, v)| v)
        .unwrap()
}

/// The first legal settlement with the most wheat pips, counted from the board directly.
fn most_wheat(board: &Board, legal: &[Action]) -> Action {
    let wheat = |a: Action| match a {
        Action::BuildSettlement(n) => topo().node_tiles[n as usize]
            .iter()
            .filter(|&&t| board.tile_resource[t as usize] == Some(Resource::Wheat))
            .map(|&t| pips(board.tile_number[t as usize]))
            .sum::<u32>(),
        _ => 0,
    };
    let mut best = legal[0];
    for &a in legal {
        if wheat(a) > wheat(best) {
            best = a;
        }
    }
    best
}

#[test]
fn a_model_round_trips_through_its_file_and_gives_probabilities() {
    let m = model(&[16, 8], 1);
    let bytes = m.to_bytes();
    assert_eq!(&bytes[..8], b"ASOPEN\0\0");
    assert_eq!(OpeningModel::from_bytes(&bytes).unwrap(), m);
    let path = temp_dir("roundtrip").join(OPENING_FILE);
    m.save(&path).unwrap();
    assert_eq!(OpeningModel::load(&path).unwrap(), m);
    let g = Game::new(5, GameConfig::default());
    let p = m.probabilities(&state_features(g.state(), 0));
    assert!((p.iter().sum::<f32>() - 1.0).abs() < 1e-6, "{p:?}");
    assert!(p.iter().all(|&x| x > 0.0), "{p:?}");
}

#[test]
fn the_forward_pass_is_relu_between_layers_and_a_softmax_at_the_end() {
    // Layer 0: unit 0 = 2 x PICK_LEFT, unit 1 = -2 x PICK_LEFT (cut to 0 by the ReLU).
    // Layer 1: own-seat logit = unit 0 + unit 1; the other seats' logits are 0.
    let mut w0 = vec![0.0; 2 * OPENING_FEATURES];
    w0[PICK_LEFT] = 2.0;
    w0[OPENING_FEATURES + PICK_LEFT] = -2.0;
    let mut w1 = vec![0.0; NUM_PLAYERS * 2];
    w1[0] = 1.0;
    w1[1] = 1.0;
    let m = OpeningModel {
        layers: vec![
            OpeningLayer {
                rows: 2,
                cols: OPENING_FEATURES,
                w: w0,
                b: vec![0.0; 2],
            },
            OpeningLayer {
                rows: NUM_PLAYERS,
                cols: 2,
                w: w1,
                b: vec![0.0; NUM_PLAYERS],
            },
        ],
    };
    let mut x = [0.0; OPENING_FEATURES];
    x[PICK_LEFT] = 1.0;
    // Logits [2, 0, 0, 0] (without the ReLU they would be all 0): e^2 / (e^2 + 3) for the seat.
    let e2 = 2.0f32.exp();
    let want = [e2, 1.0, 1.0, 1.0].map(|v| v / (e2 + 3.0));
    let got = m.probabilities(&x);
    for k in 0..NUM_PLAYERS {
        assert!((got[k] - want[k]).abs() < 1e-6, "{got:?} vs {want:?}");
    }
}

#[test]
fn bad_model_files_are_refused_with_the_reason() {
    // Layer 0 is 8 x 78 (rows at byte 24, cols at 28), layer 1 is 4 x 8 (rows at 32, cols at 36).
    let good = model(&[8], 2).to_bytes();
    let edit = |f: &dyn Fn(&mut Vec<u8>)| -> Vec<u8> {
        let mut b = good.clone();
        f(&mut b);
        b
    };
    let nan = |x: f32| {
        move |b: &mut Vec<u8>| {
            let n = b.len();
            b[n - 4..].copy_from_slice(&x.to_le_bytes());
        }
    };
    let cases: Vec<(Vec<u8>, &str)> = vec![
        (good[..10].to_vec(), "shorter than the 24-byte header"),
        (edit(&|b: &mut Vec<u8>| b[0] ^= 1), "wrong magic"),
        (
            edit(&|b: &mut Vec<u8>| set_u32(b, 8, 2)),
            "format version 2, expected 1",
        ),
        (edit(&|b: &mut Vec<u8>| b[12] ^= 1), "opening feature hash"),
        (
            edit(&|b: &mut Vec<u8>| set_u32(b, 20, 0)),
            "0 layers, expected 1..=8",
        ),
        (
            edit(&|b: &mut Vec<u8>| set_u32(b, 20, 9)),
            "9 layers, expected 1..=8",
        ),
        (
            edit(&|b: &mut Vec<u8>| set_u32(b, 24, 0)),
            "each dimension must be 1..=4096",
        ),
        (
            edit(&|b: &mut Vec<u8>| set_u32(b, 28, 77)),
            "layer 0 takes 77 inputs",
        ),
        (
            edit(&|b: &mut Vec<u8>| set_u32(b, 36, 9)),
            "layer 1 takes 9 inputs, but layer 0 gives 8",
        ),
        (
            edit(&|b: &mut Vec<u8>| set_u32(b, 32, 3)),
            "the last layer gives 3 outputs",
        ),
        (
            edit(&|b: &mut Vec<u8>| {
                b.pop();
            }),
            "truncated",
        ),
        (
            edit(&|b: &mut Vec<u8>| b.push(0)),
            "1 trailing bytes after the last layer",
        ),
        (edit(&nan(f32::NAN)), "NaN or infinite"),
        (edit(&nan(f32::INFINITY)), "NaN or infinite"),
    ];
    for (bytes, want) in &cases {
        let e = OpeningModel::from_bytes(bytes).unwrap_err();
        assert!(e.contains(want), "{want}: {e}");
    }
    let dir = temp_dir("bad");
    let path = dir.join(OPENING_FILE);
    std::fs::write(&path, &cases[1].0).unwrap();
    let e = OpeningModel::load(&path).unwrap_err();
    assert!(
        e.contains(&path.display().to_string()) && e.contains("wrong magic"),
        "{e}"
    );
    let e = OpeningModel::load(&dir.join("missing.bin")).unwrap_err();
    assert!(
        e.contains("cannot read opening model") && e.contains("missing.bin"),
        "{e}"
    );
}

#[test]
fn the_opening_suffix_parses_in_any_order_and_only_once() {
    let base = NetName {
        net: "g",
        simulations: 100,
        rollout: 0,
        setup_factor: 1,
        offers: false,
        opening: true,
    };
    assert_eq!(parse_net_name("net:g@100+o"), Some(base));
    assert_eq!(
        parse_net_name("net:g+o"),
        Some(NetName {
            simulations: 1000,
            ..base
        })
    );
    let full = NetName {
        rollout: 2,
        setup_factor: 3,
        offers: true,
        ..base
    };
    for name in [
        "net:g@100+o+t+s3+r2",
        "net:g@100+t+o+r2+s3",
        "net:g@100+s3+r2+t+o",
        "net:g@100+r2+o+s3+t",
    ] {
        assert_eq!(parse_net_name(name), Some(full), "{name}");
    }
    let dir = nets_dir("names", None);
    for bad in [
        "net:g@100+o+o",
        "net:g@100+o5",
        "net:g@100+O",
        "net:g@100+oo",
        "net:g+o@100",
        "net:g@100+t+o+t",
        "ismcts@100+o",
        "ismcts+o",
        "greedy+o",
        "random+o",
    ] {
        assert_eq!(parse_net_name(bad), None, "{bad}");
        assert!(make_bot(bad, 0).is_none(), "{bad}");
        let e = Roster::new(&[bad], &dir).err().unwrap();
        assert!(e.starts_with("unknown bot"), "{bad}: {e}");
    }
}

#[test]
fn a_missing_or_bad_opening_model_is_refused_before_any_game() {
    let dir = nets_dir("refused", None);
    let e = Roster::new(&["net:tiny@10+o"], &dir).err().unwrap();
    assert!(
        e.contains("cannot read opening model") && e.contains(OPENING_FILE),
        "{e}"
    );
    let e = run_match_in("net:tiny@10+o", "net:tiny@10", 0..1, trading(), 1, &dir).unwrap_err();
    assert!(e.contains(OPENING_FILE), "{e}");
    let mut stale = model(&[8], 3).to_bytes();
    stale[12] ^= 1;
    std::fs::write(dir.join(OPENING_FILE), &stale).unwrap();
    let e = Roster::new(&["net:tiny@10+o"], &dir).err().unwrap();
    assert!(
        e.contains("opening feature hash") && e.contains(OPENING_FILE),
        "{e}"
    );
    // A name without +o never reads the file.
    assert!(Roster::new(&["net:tiny@10+t"], &dir).is_ok());
}

#[test]
fn a_plus_o_bot_plays_the_spot_its_model_file_scores_highest_and_searches_everything_else() {
    // Own-seat logit = 10 x r0's wheat pips / 15: this model wants wheat.
    let dir = nets_dir("choice", Some(&one_feature(SEAT_PIPS + 3, 10.0)));
    let roster = Roster::new(&["net:tiny@10+o"], &dir).unwrap();
    for seed in 0..4 {
        let mut g = Game::new(seed, GameConfig::default());
        let mut bot = roster.make("net:tiny@10+o", seed).unwrap();
        let mut picks = 0;
        while matches!(
            g.state().phase,
            Phase::SetupSettlement | Phase::SetupRoad { .. }
        ) {
            let s = *g.state();
            let legal = g.legal_actions();
            let before = searches(&*bot);
            let a = bot.act(&g.observation(s.current_actor()), &legal);
            if s.phase == Phase::SetupSettlement {
                assert_eq!(a, most_wheat(&s.board, &legal), "seed {seed}, pick {picks}");
                assert_eq!(
                    searches(&*bot),
                    before,
                    "the model decides without a search"
                );
                picks += 1;
            } else if legal.len() > 1 {
                assert_eq!(searches(&*bot), before + 1, "setup roads are searched");
            }
            g.apply(a).unwrap();
        }
        assert_eq!(picks, 8);
    }
}

#[test]
fn a_model_that_ties_picks_the_first_legal_spot() {
    let zero = one_feature(0, 0.0);
    let g = Game::new(9, GameConfig::default());
    let legal = g.legal_actions();
    assert_eq!(
        choose_settlement(&zero, &g.observation(0), &legal),
        legal[0]
    );
}

#[test]
fn a_spot_scored_before_it_is_placed_has_the_features_training_sees_after() {
    let mut g = Game::new(4, GameConfig::default());
    let mut checked = 0;
    while matches!(
        g.state().phase,
        Phase::SetupSettlement | Phase::SetupRoad { .. }
    ) {
        let legal = g.legal_actions();
        if g.state().phase == Phase::SetupSettlement {
            let p = g.state().current;
            let obs = g.observation(p);
            for &a in legal.iter().step_by(5) {
                let Action::BuildSettlement(n) = a else {
                    unreachable!("only settlements are legal here")
                };
                let mut placed = obs.settlements;
                placed[p as usize] |= 1u64 << n;
                let mut after = g.clone();
                after.apply(a).unwrap();
                assert_eq!(
                    opening_features(&obs.board, &placed, p),
                    state_features(after.state(), p)
                );
                checked += 1;
            }
        }
        g.apply(legal[legal.len() / 2]).unwrap();
    }
    assert!(checked > 40, "{checked}");
}

#[test]
fn plus_o_bots_play_whole_trading_games_independent_of_thread_count() {
    let dir = nets_dir("arena", Some(&model(&[8], 4)));
    let one = run_match_in("net:tiny@10+t+o", "net:tiny@10+t", 0..1, trading(), 1, &dir).unwrap();
    let three = run_match_in("net:tiny@10+t+o", "net:tiny@10+t", 0..1, trading(), 3, &dir).unwrap();
    assert_eq!(one, three);
    assert_eq!(one.len(), 4);
    assert!(one.iter().all(|r| r.belief_resets == 0 && r.turns > 0));
}

#[test]
fn the_opening_suffix_composes_with_the_gumbel_suffix_in_any_order() {
    let dir = nets_dir("gumbel", Some(&one_feature(SEAT_PIPS + 3, 10.0)));
    let names = ["net:tiny@10+o+g", "net:tiny@10+g+o", "net:tiny@10+t+g+s2+o"];
    let roster = Roster::new(&names, &dir).unwrap();
    for name in names {
        let mut bot = roster.make(name, 0).unwrap();
        let d: std::collections::HashMap<_, _> = bot.diagnostics().into_iter().collect();
        assert_eq!(d["gumbel_root"], 1, "{name}");
        let g = Game::new(0, GameConfig::default());
        let legal = g.legal_actions();
        assert_eq!(
            bot.act(&g.observation(0), &legal),
            most_wheat(&g.state().board, &legal),
            "{name}"
        );
        assert_eq!(
            searches(&*bot),
            0,
            "{name}: the model decides without a search"
        );
    }
    for bad in [
        "net:tiny@10+o+g+o",
        "net:tiny@10+g+o+g",
        "net:tiny@10+og",
        "net:tiny@10+go",
        "ismcts@10+g+o",
    ] {
        assert!(make_bot(bad, 0).is_none(), "{bad}");
        let e = Roster::new(&[bad], &dir).err().unwrap();
        assert!(e.starts_with("unknown bot"), "{bad}: {e}");
    }
}
