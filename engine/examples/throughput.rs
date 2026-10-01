//! Games per second and ns per action, on one core and on all cores.
//! Usage: throughput [games_per_thread] [max_offers_per_turn (default 0 = no domestic trades)]

use settler_engine::rng::Rng;
use settler_engine::sim::play_random;
use settler_engine::{GameConfig, State};
use std::time::Instant;

fn run(seeds: std::ops::Range<u64>, cfg: GameConfig) -> (u64, u64) {
    let mut rng = Rng::new(seeds.start ^ 0x5EED);
    let mut buf = Vec::with_capacity(256);
    let (mut games, mut actions) = (0, 0);
    for seed in seeds {
        let mut s = State::new(seed, cfg);
        actions += play_random(&mut s, &mut rng, &mut buf);
        games += 1;
    }
    (games, actions)
}

fn main() {
    let n: u64 = std::env::args()
        .nth(1)
        .and_then(|a| a.parse().ok())
        .unwrap_or(10_000);
    let offers: u8 = std::env::args()
        .nth(2)
        .and_then(|a| a.parse().ok())
        .unwrap_or(0);
    let cfg = GameConfig {
        max_offers_per_turn: offers,
        ..GameConfig::default()
    };
    println!("max_offers_per_turn = {offers}");

    let t0 = Instant::now();
    let (g, a) = run(0..n, cfg);
    let dt = t0.elapsed().as_secs_f64();
    println!(
        "1 thread: {g} games in {dt:.2}s = {:.0} games/s, {:.2}M actions/s, {:.0} actions/game, {:.1} ns/action",
        g as f64 / dt,
        a as f64 / dt / 1e6,
        a as f64 / g as f64,
        dt * 1e9 / a as f64
    );

    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1) as u64;
    let t0 = Instant::now();
    let handles: Vec<_> = (0..threads)
        .map(|i| std::thread::spawn(move || run(n * (i + 1)..n * (i + 2), cfg)))
        .collect();
    let (mut g, mut a) = (0, 0);
    for h in handles {
        let (gg, aa) = h.join().unwrap();
        g += gg;
        a += aa;
    }
    let dt = t0.elapsed().as_secs_f64();
    println!(
        "{threads} threads: {g} games in {dt:.2}s = {:.0} games/s, {:.2}M actions/s",
        g as f64 / dt,
        a as f64 / dt / 1e6
    );
}
