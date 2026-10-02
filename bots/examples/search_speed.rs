//! IsmctsBot search throughput on one core: IsmctsBot in seat 0 against three GreedyBots, timing
//! only decisions that searched. Usage: search_speed [decisions] [simulations]

use settler_bots::arena::play_game;
use settler_bots::greedy::GreedyBot;
use settler_bots::ismcts::IsmctsBot;
use settler_bots::Bot;
use settler_engine::{Action, Event, GameConfig, Observation, PlayerId};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

#[derive(Default)]
struct Totals {
    decisions: u64,
    simulations: u64,
    nodes: u64,
    time: Duration,
}

struct Timed {
    inner: IsmctsBot,
    totals: Arc<Mutex<Totals>>,
}

impl Bot for Timed {
    fn name(&self) -> &'static str {
        "timed-ismcts"
    }

    fn observe(&mut self, viewer: PlayerId, events: &[Event]) {
        self.inner.observe(viewer, events);
    }

    fn act(&mut self, obs: &Observation, legal: &[Action]) -> Action {
        let start = Instant::now();
        let a = self.inner.act(obs, legal);
        let elapsed = start.elapsed();
        if let Some(r) = self.inner.take_last_search() {
            let mut t = self.totals.lock().unwrap();
            t.decisions += 1;
            t.simulations += r.simulations as u64;
            t.nodes += r.nodes as u64;
            t.time += elapsed;
        }
        a
    }
}

fn main() {
    let args: Vec<u64> = std::env::args().skip(1).map(|a| a.parse().expect("a number")).collect();
    let decisions = args.first().copied().unwrap_or(2000);
    let simulations = args.get(1).copied().unwrap_or(1000) as u32;
    let config = GameConfig { max_offers_per_turn: 0, ..GameConfig::default() };
    let totals = Arc::new(Mutex::new(Totals::default()));
    let mut seed = 0;
    while totals.lock().unwrap().decisions < decisions {
        let mut bots: Vec<Box<dyn Bot>> = vec![Box::new(Timed { inner: IsmctsBot::new(seed, simulations, 0), totals: totals.clone() })];
        bots.extend((1..4).map(|p| Box::new(GreedyBot::new(p)) as Box<dyn Bot>));
        play_game(seed, config, &mut bots);
        seed += 1;
    }
    let t = totals.lock().unwrap();
    let secs = t.time.as_secs_f64();
    println!(
        "{} searched decisions over {seed} games at {simulations} simulations: {:.0} simulations/s, {:.2} ms/decision, {:.0} nodes/decision",
        t.decisions,
        t.simulations as f64 / secs,
        1000.0 * secs / t.decisions as f64,
        t.nodes as f64 / t.decisions as f64
    );
}
