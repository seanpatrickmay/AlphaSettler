mod common;

use common::*;
use settler_engine::{GameConfig, ACTION_SPACE_SIZE, NUM_PLAYERS};
use settler_net::encode::{encode_state, ENCODING_HASH, HIDDEN_LEN, PUBLIC_LEN};
use settler_net::forward::{Forward, Outputs};
use settler_net::weights::*;

const SMALL: Arch = Arch { trunk: 8, value: 4 };

fn err(bytes: &[u8]) -> String {
    Weights::from_bytes(bytes).unwrap_err()
}

#[test]
fn the_header_is_80_bytes_and_shapes_follow_the_arch() {
    assert_eq!(HEADER_LEN, 80);
    assert_eq!(MAGIC, *b"ASNET\0\0\0");
    assert_eq!(
        Arch::DEFAULT.shapes(),
        [
            (512, PUBLIC_LEN),
            (512, 512),
            (512, 512),
            (ACTION_SPACE_SIZE, 512),
            (256, 512 + HIDDEN_LEN),
            (NUM_PLAYERS, 256),
            (NUM_PLAYERS, 256),
        ]
    );
    let w = Weights::random(Arch::DEFAULT, 1);
    assert_eq!(w.to_bytes().len(), HEADER_LEN + 4 * w.param_count());
}

#[test]
fn weights_round_trip_through_bytes_and_files() {
    let w = Weights::random(SMALL, 7);
    assert_eq!(Weights::from_bytes(&w.to_bytes()).unwrap(), w);
    let path = temp_file("round-trip.bin");
    w.save(&path).unwrap();
    assert_eq!(Weights::load(&path).unwrap(), w);
    std::fs::remove_file(&path).unwrap();
    assert_ne!(Weights::random(SMALL, 8), w);
}

#[test]
fn a_bad_file_is_an_error_naming_what_is_wrong() {
    let good = Weights::random(SMALL, 3).to_bytes();
    let mut magic = good.clone();
    magic[0] = b'X';
    assert!(err(&magic).contains("magic"), "{}", err(&magic));
    let mut version = good.clone();
    version[8..12].copy_from_slice(&2u32.to_le_bytes());
    assert!(
        err(&version).contains("format version 2"),
        "{}",
        err(&version)
    );
    let mut hash = good.clone();
    hash[12] ^= 1;
    let e = err(&hash);
    assert!(e.contains("encoding hash"), "{e}");
    assert!(
        e.contains(&format!("{ENCODING_HASH:#018x}")),
        "names this encoder's hash: {e}"
    );
    assert!(
        e.contains(&format!("{:#018x}", ENCODING_HASH ^ 1)),
        "names the file's hash: {e}"
    );
    let mut layers = good.clone();
    layers[20..24].copy_from_slice(&6u32.to_le_bytes());
    assert!(err(&layers).contains("6 layers"), "{}", err(&layers));
    let mut shape = good.clone();
    shape[32..36].copy_from_slice(&7u32.to_le_bytes()); // layer 1's rows: 8 -> 7
    assert!(err(&shape).contains("shapes"), "{}", err(&shape));
    let mut huge = good.clone();
    huge[24..28].copy_from_slice(&u32::MAX.to_le_bytes());
    assert!(err(&huge).contains("shapes"), "{}", err(&huge));
    assert!(err(&good[..good.len() - 1]).contains("truncated"));
    assert!(err(&good[..40]).contains("truncated"));
    assert!(err(&[]).contains("truncated"));
    let mut long = good.clone();
    long.push(0);
    assert!(err(&long).contains("trailing"), "{}", err(&long));
    let mut nan = good.clone();
    nan[HEADER_LEN..HEADER_LEN + 4].copy_from_slice(&f32::NAN.to_le_bytes());
    assert!(err(&nan).contains("NaN"), "{}", err(&nan));
    let missing = Weights::load(&temp_file("no-such.bin")).unwrap_err();
    assert!(
        missing.contains("cannot read") && missing.contains("no-such.bin"),
        "{missing}"
    );
}

/// The forward pass written the slow, obvious way, in f64.
fn reference(w: &Weights, public: &[f32], hidden: &[f32]) -> [Vec<f64>; 3] {
    let dense = |l: &Layer, x: &[f64], relu: bool| -> Vec<f64> {
        (0..l.rows)
            .map(|o| {
                let s = l.b[o] as f64
                    + (0..l.cols)
                        .map(|k| l.w[o * l.cols + k] as f64 * x[k])
                        .sum::<f64>();
                if relu {
                    s.max(0.0)
                } else {
                    s
                }
            })
            .collect()
    };
    let x: Vec<f64> = public.iter().map(|&v| v as f64).collect();
    let h1 = dense(&w.layers[TRUNK0], &x, true);
    let h2 = dense(&w.layers[TRUNK1], &h1, true);
    let h3 = dense(&w.layers[TRUNK2], &h2, true);
    let policy = dense(&w.layers[POLICY], &h3, false);
    let mut joined = h3.clone();
    joined.extend(hidden.iter().map(|&v| v as f64));
    let v = dense(&w.layers[VALUE0], &joined, true);
    [
        policy,
        dense(&w.layers[VALUE1], &v, false),
        dense(&w.layers[AUX], &v, false),
    ]
}

fn inputs(n_every: usize) -> (usize, Vec<f32>, Vec<f32>) {
    let positions = random_positions(0..2, n_every, GameConfig::default());
    let n = positions.len();
    let (mut public, mut hidden) = (vec![0.0; n * PUBLIC_LEN], vec![0.0; n * HIDDEN_LEN]);
    for (i, s) in positions.iter().enumerate() {
        encode_state(
            s,
            s.current_actor(),
            &mut public[i * PUBLIC_LEN..(i + 1) * PUBLIC_LEN],
            &mut hidden[i * HIDDEN_LEN..(i + 1) * HIDDEN_LEN],
        );
    }
    (n, public, hidden)
}

#[test]
fn the_forward_pass_matches_a_plain_reference() {
    let w = Weights::random(
        Arch {
            trunk: 32,
            value: 16,
        },
        5,
    );
    let (n, public, hidden) = inputs(97);
    let mut out = Outputs::default();
    Forward::new().run(&w, &public, &hidden, n, &mut out);
    assert_eq!(out.n, n);
    assert_eq!(
        (out.policy.len(), out.value.len(), out.aux.len()),
        (n * ACTION_SPACE_SIZE, n * NUM_PLAYERS, n * NUM_PLAYERS)
    );
    for i in 0..n {
        let [p, v, a] = reference(
            &w,
            &public[i * PUBLIC_LEN..(i + 1) * PUBLIC_LEN],
            &hidden[i * HIDDEN_LEN..(i + 1) * HIDDEN_LEN],
        );
        let got = [
            &out.policy[i * ACTION_SPACE_SIZE..(i + 1) * ACTION_SPACE_SIZE],
            &out.value[i * NUM_PLAYERS..(i + 1) * NUM_PLAYERS],
            &out.aux[i * NUM_PLAYERS..(i + 1) * NUM_PLAYERS],
        ];
        for (g, want) in got.iter().zip([p, v, a]) {
            for (x, y) in g.iter().zip(&want) {
                assert!(
                    (*x as f64 - y).abs() <= 1e-4 * y.abs().max(1.0),
                    "position {i}: {x} vs {y}"
                );
            }
        }
    }
}

#[test]
fn each_row_is_independent_of_the_rest_of_the_batch() {
    let w = Weights::random(
        Arch {
            trunk: 32,
            value: 16,
        },
        9,
    );
    let (n, public, hidden) = inputs(211);
    let (mut fwd, mut all, mut one) = (Forward::new(), Outputs::default(), Outputs::default());
    fwd.run(&w, &public, &hidden, n, &mut all);
    for i in 0..n {
        fwd.run(
            &w,
            &public[i * PUBLIC_LEN..(i + 1) * PUBLIC_LEN],
            &hidden[i * HIDDEN_LEN..(i + 1) * HIDDEN_LEN],
            1,
            &mut one,
        );
        assert_eq!(
            one.policy[..],
            all.policy[i * ACTION_SPACE_SIZE..(i + 1) * ACTION_SPACE_SIZE]
        );
        assert_eq!(
            one.value[..],
            all.value[i * NUM_PLAYERS..(i + 1) * NUM_PLAYERS]
        );
        assert_eq!(one.aux[..], all.aux[i * NUM_PLAYERS..(i + 1) * NUM_PLAYERS]);
    }
}
