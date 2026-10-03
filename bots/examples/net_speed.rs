//! Network search throughput and the leaf-batch sweep (spec Sections 3 and 5).
//!
//!   net_speed speed PATH DECISIONS SIMULATIONS BATCH
//!       one net bot (seat 0) against three GreedyBots on one core, timing only decisions that
//!       searched: simulations/s, network evaluations/s and ms per decision
//!   net_speed paired PATH SEEDS SIMULATIONS BASELINE
//!       the net bot at batch 8, 16 and 32 against three BASELINE bots on the same seeds (all
//!       cores, trades off), then each larger batch's paired difference from batch 8
//!
//! PATH is a weights file, or `random` for random weights of the default shape.

use settler_bots::arena::{play_game, run_match_by, GameRecord};
use settler_bots::greedy::GreedyBot;
use settler_bots::ismcts::IsmctsBot;
use settler_bots::netbot::net_bot_with;
use settler_bots::{make_bot, Bot};
use settler_engine::{Action, Event, GameConfig, Observation, PlayerId};
use settler_net::evaluator::NetEvaluator;
use settler_net::weights::{Arch, Weights};
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

fn weights(path: &str) -> Arc<Weights> {
    Arc::new(if path == "random" {
        Weights::random(Arch::DEFAULT, 1)
    } else {
        Weights::load(Path::new(path)).unwrap_or_else(|e| panic!("{e}"))
    })
}

fn no_trades() -> GameConfig {
    GameConfig {
        max_offers_per_turn: 0,
        ..GameConfig::default()
    }
}

#[derive(Default)]
struct Totals {
    decisions: u64,
    simulations: u64,
    evaluations: u64,
    time: Duration,
}

struct Timed {
    inner: IsmctsBot<NetEvaluator>,
    totals: Arc<Mutex<Totals>>,
}

impl Bot for Timed {
    fn name(&self) -> &'static str {
        "timed-net"
    }

    fn observe(&mut self, viewer: PlayerId, events: &[Event]) {
        self.inner.observe(viewer, events);
    }

    fn act(&mut self, obs: &Observation, legal: &[Action]) -> Action {
        let before = self.inner.evaluator().evaluations();
        let start = Instant::now();
        let a = self.inner.act(obs, legal);
        let elapsed = start.elapsed();
        if let Some(r) = self.inner.take_last_search() {
            let mut t = self.totals.lock().unwrap();
            t.decisions += 1;
            t.simulations += r.simulations as u64;
            t.evaluations += self.inner.evaluator().evaluations() - before;
            t.time += elapsed;
        }
        a
    }
}

fn speed(path: &str, decisions: u64, simulations: u32, batch: usize) {
    let w = weights(path);
    let totals = Arc::new(Mutex::new(Totals::default()));
    let mut seed = 0;
    while totals.lock().unwrap().decisions < decisions {
        let mut bots: Vec<Box<dyn Bot>> = vec![Box::new(Timed {
            inner: net_bot_with(seed, w.clone(), simulations, 0, batch),
            totals: totals.clone(),
        })];
        bots.extend((1..4).map(|p| Box::new(GreedyBot::new(p)) as Box<dyn Bot>));
        play_game(seed, no_trades(), &mut bots);
        seed += 1;
    }
    let t = totals.lock().unwrap();
    let secs = t.time.as_secs_f64();
    println!(
        "batch {batch}: {} searched decisions over {seed} games at {simulations} simulations: {:.0} simulations/s, {:.0} evaluations/s, {:.2} ms/decision",
        t.decisions,
        t.simulations as f64 / secs,
        t.evaluations as f64 / secs,
        1000.0 * secs / t.decisions as f64
    );
}

/// Per-seed candidate win rates; records are sorted by (seed, seat), four per seed.
fn per_seed(records: &[GameRecord]) -> Vec<f64> {
    records
        .chunks(4)
        .map(|c| {
            c.iter()
                .filter(|r| r.winner == Some(r.candidate_seat))
                .count() as f64
                / 4.0
        })
        .collect()
}

fn mean_se(xs: &[f64]) -> (f64, f64) {
    let n = xs.len() as f64;
    let mean = xs.iter().sum::<f64>() / n;
    let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0).max(1.0);
    (mean, (var / n).sqrt())
}

fn paired(path: &str, seeds: u64, simulations: u32, baseline: &str) {
    let w = weights(path);
    assert!(
        make_bot(baseline, 0).is_some(),
        "unknown baseline {baseline:?} (net: names are not supported here)"
    );
    let threads = std::thread::available_parallelism().map_or(1, |c| c.get());
    let mut rates: Vec<(usize, Vec<f64>)> = Vec::new();
    for batch in [8usize, 16, 32] {
        let start = Instant::now();
        let records = run_match_by(0..seeds, no_trades(), threads, &|is_candidate, seed| {
            if is_candidate {
                Box::new(net_bot_with(seed, w.clone(), simulations, 0, batch)) as Box<dyn Bot>
            } else {
                make_bot(baseline, seed).expect("checked above")
            }
        })
        .unwrap_or_else(|e| panic!("{e}"));
        let r = per_seed(&records);
        let (mean, se) = mean_se(&r);
        let resets: u64 = records.iter().map(|x| x.belief_resets).sum();
        println!(
            "batch {batch} at {simulations} simulations vs three {baseline}: win rate {mean:.3} ± {se:.3} (z vs 0.25 = {:.2}) over {seeds} seeds, belief resets {resets}, {:.0} s",
            (mean - 0.25) / se,
            start.elapsed().as_secs_f64()
        );
        rates.push((batch, r));
    }
    for (batch, r) in &rates[1..] {
        let diffs: Vec<f64> = r.iter().zip(&rates[0].1).map(|(a, b)| a - b).collect();
        let (mean, se) = mean_se(&diffs);
        let z = if se > 0.0 { mean / se } else { 0.0 };
        println!("paired batch {batch} - batch 8: mean diff {mean:+.3} ± {se:.3} (z = {z:.2}) over {seeds} seeds");
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let num = |i: usize| -> u64 {
        args.get(i)
            .and_then(|a| a.parse().ok())
            .unwrap_or_else(|| panic!("argument {i} must be a number"))
    };
    match args.first().map(String::as_str) {
        Some("speed") if args.len() == 5 => speed(&args[1], num(2), num(3) as u32, num(4) as usize),
        Some("paired") if args.len() == 5 => paired(&args[1], num(2), num(3) as u32, &args[4]),
        _ => {
            eprintln!("usage: net_speed speed PATH DECISIONS SIMULATIONS BATCH | net_speed paired PATH SEEDS SIMULATIONS BASELINE");
            std::process::exit(2)
        }
    }
}
