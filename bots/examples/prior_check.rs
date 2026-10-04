//! The network acceptance gate (spec Section 2 and done criterion 3b): how good is a net's prior?
//!
//!   prior_check agree PATH GAMES
//!       GAMES fresh `ismcts` self-play games with the warm start's exploration (playout cap
//!       randomization, root noise, opening sampling; seeds from 800,000,000, trades off) on all
//!       cores. At each full search with two or more moves: does the heuristic prior's top move,
//!       and the net prior's, equal the search's most-visited move? Reported by that move's kind,
//!       then the gate: the net at least 0.5 on setup settlements, robber moves and roads, over at
//!       least 3,000 searches.
//!   prior_check hybrid PATH SIMULATIONS SEEDS
//!       four evaluators, each at SIMULATIONS against three ismcts@SIMULATIONS over seeds
//!       0..SEEDS (all cores, trades off): heuristic prior and value (plays exactly as ismcts),
//!       net prior and value, net prior with the heuristic value, heuristic prior with the net
//!       value. The gate: "net prior + heuristic value" above the flat MLP's 0.140 (measured at
//!       100 simulations over 50 seeds).
//!
//! PATH is a weights file, or `random` for random weights of the default shape.

use settler_bots::arena::{run_match_by, GameRecord};
use settler_bots::heuristic::{self, HeuristicEvaluator, DEFAULT_WEIGHTS};
use settler_bots::ismcts::IsmctsBot;
use settler_bots::selfplay::{run_with, Exploration};
use settler_bots::{make_bot, Bot};
use settler_engine::{Action, GameConfig, Phase, State};
use settler_net::evaluator::NetEvaluator;
use settler_net::weights::{Arch, Weights};
use settler_search::{Eval, Evaluator, Leaf, SearchConfig};
use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

/// Seeds of the agreement games: far from the warm start's 0..100,000 and the gate's 10^9.
const AGREE_SEED_START: u64 = 800_000_000;
const AGREE_GATE: f64 = 0.5;
const AGREE_MIN_SEARCHES: u64 = 3_000;
const AGREE_GATE_KINDS: [&str; 3] = ["setup-settlement", "robber", "road"];
/// The flat MLP's "net prior + heuristic value" win rate at 100 simulations over 50 seeds.
const HYBRID_BAR: f64 = 0.140;

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

fn cores() -> usize {
    std::thread::available_parallelism().map_or(1, |c| c.get())
}

/// The kind of move a search chose; setup settlements and roads apart from later ones.
fn kind(a: &Action, setup: bool) -> &'static str {
    match a {
        Action::BuildSettlement(_) if setup => "setup-settlement",
        Action::BuildRoad(_) if setup => "setup-road",
        Action::BuildSettlement(_) => "settlement",
        Action::BuildCity(_) => "city",
        Action::BuildRoad(_) => "road",
        Action::Roll => "roll",
        Action::EndTurn => "end-turn",
        Action::BuyDev => "buy-dev",
        Action::PlayKnight
        | Action::PlayRoadBuilding
        | Action::PlayMonopoly(_)
        | Action::PlayYearOfPlenty(..) => "play-dev",
        Action::MoveRobber(_) => "robber",
        Action::StealFrom(_) => "steal",
        Action::Discard(_) => "discard",
        Action::MaritimeTrade { .. } => "maritime",
        _ => "other",
    }
}

/// The first index of the largest entry.
fn argmax(v: &[f32]) -> usize {
    let mut best = 0;
    for i in 1..v.len() {
        if v[i] > v[best] {
            best = i;
        }
    }
    best
}

/// [searches, heuristic agrees, net agrees, net == heuristic] by kind.
type Table = BTreeMap<&'static str, [u64; 4]>;

fn agree(path: &str, games: u64) {
    let w = weights(path);
    let cfg = no_trades();
    let make = || HeuristicEvaluator::new(DEFAULT_WEIGHTS, 0);
    let seeds = AGREE_SEED_START..AGREE_SEED_START + games;
    let played = run_with(
        seeds,
        cfg,
        &Exploration::standard(),
        SearchConfig::default().batch,
        &make,
        cores(),
    )
    .unwrap_or_else(|e| panic!("{e}"));
    let mut net = NetEvaluator::new(w);
    let mut table = Table::new();
    for g in &played {
        let mut s = State::new(g.seed, cfg);
        let mut next = 0;
        for d in &g.decisions {
            while next < d.index as usize {
                s.apply(g.actions[next]);
                next += 1;
            }
            let searchable: Vec<(Action, f32)> = d
                .legal
                .iter()
                .zip(&d.visits)
                .filter(|(a, _)| !matches!(a, Action::OfferTrade { .. }))
                .map(|(&a, &v)| (a, v as f32))
                .collect();
            if !d.full_search || searchable.len() < 2 {
                continue;
            }
            let legal: Vec<Action> = searchable.iter().map(|x| x.0).collect();
            let visits: Vec<f32> = searchable.iter().map(|x| x.1).collect();
            let target = argmax(&visits);
            let heuristic = argmax(&heuristic::prior(&s.observation(d.actor), &legal));
            let leaf = Leaf {
                world: &s,
                actor: d.actor,
                legal: &legal,
            };
            let mine = argmax(&net.evaluate(&[leaf])[0].prior);
            let setup = matches!(s.phase, Phase::SetupSettlement | Phase::SetupRoad { .. });
            let row = table.entry(kind(&legal[target], setup)).or_insert([0; 4]);
            row[0] += 1;
            row[1] += (heuristic == target) as u64;
            row[2] += (mine == target) as u64;
            row[3] += (mine == heuristic) as u64;
        }
    }
    let line = |name: &str, r: &[u64; 4]| {
        let f = |k: usize| r[k] as f64 / r[0].max(1) as f64;
        println!(
            "{name:>17}: n {:>6}  heur-top1 {:.3}  net-top1 {:.3}  net==heur {:.3}",
            r[0],
            f(1),
            f(2),
            f(3)
        );
    };
    let mut all = [0u64; 4];
    for (name, r) in &table {
        line(name, r);
        for k in 0..4 {
            all[k] += r[k];
        }
    }
    line("ALL", &all);
    let rate = |name: &str| {
        table
            .get(name)
            .map_or(0.0, |r| r[2] as f64 / r[0].max(1) as f64)
    };
    let pass =
        all[0] >= AGREE_MIN_SEARCHES && AGREE_GATE_KINDS.iter().all(|k| rate(k) >= AGREE_GATE);
    println!(
        "agreement gate (net-top1 >= {AGREE_GATE} on {}, over >= {AGREE_MIN_SEARCHES} searches): {} over {} games, {} searches",
        AGREE_GATE_KINDS.join(", "),
        if pass { "pass" } else { "FAIL" },
        played.len(),
        all[0]
    );
}

#[derive(Clone, Copy)]
enum Mix {
    Heuristic,
    Net,
    NetPriorHeuristicValue,
    HeuristicPriorNetValue,
}

const MIXES: [(&str, Mix); 4] = [
    ("heuristic prior + heuristic value", Mix::Heuristic),
    ("net prior + net value", Mix::Net),
    ("net prior + heuristic value", Mix::NetPriorHeuristicValue),
    ("heuristic prior + net value", Mix::HeuristicPriorNetValue),
];

/// Both evaluators on every leaf; the prior from one and the value from the other.
struct Hybrid {
    heuristic: HeuristicEvaluator,
    net: NetEvaluator,
    mix: Mix,
}

impl Evaluator for Hybrid {
    fn evaluate(&mut self, leaves: &[Leaf<'_>]) -> Vec<Eval> {
        let h = self.heuristic.evaluate(leaves);
        let n = self.net.evaluate(leaves);
        h.into_iter()
            .zip(n)
            .map(|(h, n)| match self.mix {
                Mix::Heuristic => h,
                Mix::Net => n,
                Mix::NetPriorHeuristicValue => Eval {
                    value: h.value,
                    prior: n.prior,
                },
                Mix::HeuristicPriorNetValue => Eval {
                    value: n.value,
                    prior: h.prior,
                },
            })
            .collect()
    }
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

fn hybrid(path: &str, simulations: u32, seeds: u64) {
    let w = weights(path);
    let baseline = format!("ismcts@{simulations}");
    assert!(make_bot(&baseline, 0).is_some(), "{baseline} is a bot");
    let mut gate = 0.0;
    for (name, mix) in MIXES {
        let records = run_match_by(0..seeds, no_trades(), cores(), &|is_candidate, seed| {
            if is_candidate {
                let eval = Hybrid {
                    heuristic: HeuristicEvaluator::new(DEFAULT_WEIGHTS, 0),
                    net: NetEvaluator::new(w.clone()),
                    mix,
                };
                let cfg = SearchConfig {
                    simulations,
                    ..SearchConfig::default()
                };
                Box::new(IsmctsBot::with_evaluator("hybrid", seed, cfg, eval)) as Box<dyn Bot>
            } else {
                make_bot(&baseline, seed).expect("checked above")
            }
        })
        .unwrap_or_else(|e| panic!("{e}"));
        let wins = records
            .iter()
            .filter(|r| r.winner == Some(r.candidate_seat))
            .count();
        let draws = records.iter().filter(|r| r.winner.is_none()).count();
        let rate = wins as f64 / records.len() as f64;
        let (_, se) = mean_se(&per_seed(&records));
        let z = if se > 0.0 { (rate - 0.25) / se } else { 0.0 };
        let vp = records
            .iter()
            .map(|r| r.vp[r.candidate_seat as usize] as f64)
            .sum::<f64>()
            / records.len() as f64;
        let resets: u64 = records.iter().map(|r| r.belief_resets).sum();
        println!(
            "{name:>33} @{simulations} vs three {baseline}: win rate {rate:.3} ({wins}/{}, {draws} draws), z vs 0.25 = {z:.2}, mean VP {vp:.2}, belief resets {resets}",
            records.len()
        );
        if let Mix::NetPriorHeuristicValue = mix {
            gate = rate;
        }
    }
    println!(
        "hybrid gate (net prior + heuristic value above {HYBRID_BAR:.3}): {} at {gate:.3}",
        if gate > HYBRID_BAR { "pass" } else { "FAIL" }
    );
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let num = |i: usize| -> u64 {
        args.get(i)
            .and_then(|a| a.parse().ok())
            .unwrap_or_else(|| panic!("argument {i} must be a number"))
    };
    match args.first().map(String::as_str) {
        Some("agree") if args.len() == 3 => agree(&args[1], num(2)),
        Some("hybrid") if args.len() == 4 => hybrid(&args[1], num(2) as u32, num(3)),
        _ => {
            eprintln!(
                "usage: prior_check agree PATH GAMES | prior_check hybrid PATH SIMULATIONS SEEDS"
            );
            std::process::exit(2)
        }
    }
}
