use criterion::{criterion_group, criterion_main, Criterion};
use settler_engine::legal::legal_actions;
use settler_engine::rng::Rng;
use settler_engine::sim::play_random;
use settler_engine::{GameConfig, State};
use std::hint::black_box;

fn bench(c: &mut Criterion) {
    let cfg = GameConfig {
        max_offers_per_turn: 0,
        ..GameConfig::default()
    };
    let mut rng = Rng::new(7);
    let mut buf = Vec::with_capacity(256);

    // A mid-game position from a random playout.
    let mut mid = State::new(7, cfg);
    for _ in 0..400 {
        if mid.is_over() {
            break;
        }
        legal_actions(&mid, &mut buf);
        let a = buf[rng.below(buf.len() as u32) as usize];
        mid.apply(a);
    }

    c.bench_function("clone_state", |b| b.iter(|| black_box(*black_box(&mid))));
    c.bench_function("legal_actions_midgame", |b| {
        b.iter(|| {
            legal_actions(black_box(&mid), &mut buf);
            black_box(buf.len())
        })
    });
    c.bench_function("random_game", |b| {
        let mut seed = 0u64;
        b.iter(|| {
            seed += 1;
            let mut s = State::new(seed, cfg);
            black_box(play_random(&mut s, &mut rng, &mut buf))
        })
    });
}

criterion_group!(benches, bench);
criterion_main!(benches);
