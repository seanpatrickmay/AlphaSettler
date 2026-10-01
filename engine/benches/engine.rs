use criterion::{criterion_group, criterion_main, Criterion};
use settler_engine::legal::legal_actions;
use settler_engine::rng::Rng;
use settler_engine::sim::play_random;
use settler_engine::{Action, GameConfig, Phase, State};
use std::hint::black_box;

/// A mid-game position with real move-generation work: Main phase, at least 8 legal actions, and
/// at least one build or maritime action (so the road/settlement/city/maritime scans actually run).
/// Starts from seed 7 (400 random actions, then random steps, capped at 10,000), then tries
/// further games (seeds 8, 9, ...) with the same rng. Fails loudly if nothing qualifies.
fn rich_main_position(cfg: GameConfig, rng: &mut Rng, buf: &mut Vec<Action>) -> State {
    const MAX_GAMES: u64 = 50;
    for game in 0..MAX_GAMES {
        let seed = 7 + game;
        let mut s = State::new(seed, cfg);
        let mut steps = 0u32;
        while !s.is_over() && steps < 400 + 10_000 {
            legal_actions(&s, buf);
            if steps >= 400
                && s.phase == Phase::Main
                && buf.len() >= 8
                && buf.iter().any(|a| {
                    matches!(
                        a,
                        Action::BuildRoad(_)
                            | Action::BuildSettlement(_)
                            | Action::BuildCity(_)
                            | Action::MaritimeTrade { .. }
                    )
                })
            {
                let (mut roads, mut settlements, mut cities, mut maritime, mut dev, mut other) =
                    (0, 0, 0, 0, 0, 0);
                for a in buf.iter() {
                    match a {
                        Action::BuildRoad(_) => roads += 1,
                        Action::BuildSettlement(_) => settlements += 1,
                        Action::BuildCity(_) => cities += 1,
                        Action::MaritimeTrade { .. } => maritime += 1,
                        Action::BuyDev
                        | Action::PlayKnight
                        | Action::PlayRoadBuilding
                        | Action::PlayMonopoly(_)
                        | Action::PlayYearOfPlenty(..) => dev += 1,
                        _ => other += 1,
                    }
                }
                eprintln!(
                    "legal_actions_midgame position: seed={seed}, step={steps}, phase={:?}, {} legal actions: \
                     {roads} roads, {settlements} settlements, {cities} cities, {maritime} maritime, \
                     {dev} dev, {other} other",
                    s.phase,
                    buf.len()
                );
                return s;
            }
            let a = buf[rng.below(buf.len() as u32) as usize];
            s.apply(a);
            steps += 1;
        }
    }
    panic!("no rich Main-phase position found in {MAX_GAMES} games");
}

fn bench(c: &mut Criterion) {
    let cfg = GameConfig {
        max_offers_per_turn: 0,
        ..GameConfig::default()
    };
    let mut rng = Rng::new(7);
    let mut buf = Vec::with_capacity(256);

    let mid = rich_main_position(cfg, &mut rng, &mut buf);

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
