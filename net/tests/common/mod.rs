#![allow(dead_code)]

use settler_engine::legal::legal_actions;
use settler_engine::rng::Rng;
use settler_engine::*;
use std::path::PathBuf;

pub fn no_trades() -> GameConfig {
    GameConfig {
        max_offers_per_turn: 0,
        ..GameConfig::default()
    }
}

/// Every `every`-th position (never a finished one) of uniformly random games from `seeds`, at
/// most 3,000 moves each.
pub fn random_positions(
    seeds: std::ops::Range<u64>,
    every: usize,
    config: GameConfig,
) -> Vec<State> {
    let mut out = Vec::new();
    let mut buf = Vec::new();
    for seed in seeds {
        let mut s = State::new(seed, config);
        let mut rng = Rng::new(seed ^ 0xF00D);
        let mut i = 0;
        while !s.is_over() && i < 3000 {
            legal_actions(&s, &mut buf);
            if i % every == 0 {
                out.push(s);
            }
            s.apply(buf[rng.below(buf.len() as u32) as usize]);
            i += 1;
        }
    }
    out
}

/// A path in the system temp directory unique to this test process and `name`.
pub fn temp_file(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("settler-net-{}-{name}", std::process::id()))
}
