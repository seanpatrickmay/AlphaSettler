//! Where random-play time goes: legal-action generation vs apply, split by phase.
//! Usage: profile_steps [games] [max_offers_per_turn]. Timer overhead (~20-40 ns per sample)
//! inflates every bucket; read the proportions, not the absolute values.

use settler_engine::legal::legal_actions;
use settler_engine::rng::Rng;
use settler_engine::{Action, GameConfig, Phase, State};
use std::time::Instant;

fn phase_index(p: Phase) -> usize {
    match p {
        Phase::SetupSettlement => 0,
        Phase::SetupRoad { .. } => 1,
        Phase::PreRoll => 2,
        Phase::Discard => 3,
        Phase::MoveRobber => 4,
        Phase::Steal => 5,
        Phase::Main => 6,
        Phase::RoadBuilding { .. } => 7,
        Phase::TradeResponse => 8,
        Phase::TradeConfirm => 9,
        Phase::GameOver { .. } => 10,
    }
}

const PHASES: [&str; 11] = [
    "SetupSettlement",
    "SetupRoad",
    "PreRoll",
    "Discard",
    "MoveRobber",
    "Steal",
    "Main",
    "RoadBuilding",
    "TradeResponse",
    "TradeConfirm",
    "GameOver",
];

fn action_kind(a: Action) -> usize {
    match a {
        Action::BuildRoad(_) => 0,
        Action::BuildSettlement(_) => 1,
        Action::BuildCity(_) => 2,
        Action::OfferTrade { .. } => 3,
        Action::MaritimeTrade { .. } => 4,
        Action::Roll => 5,
        Action::EndTurn => 6,
        _ => 7,
    }
}

const KINDS: [&str; 8] = [
    "BuildRoad",
    "BuildSettlement",
    "BuildCity",
    "OfferTrade",
    "Maritime",
    "Roll",
    "EndTurn",
    "other",
];

fn main() {
    let games: u64 = std::env::args()
        .nth(1)
        .and_then(|a| a.parse().ok())
        .unwrap_or(5000);
    let offers: u8 = std::env::args()
        .nth(2)
        .and_then(|a| a.parse().ok())
        .unwrap_or(0);
    let cfg = GameConfig {
        max_offers_per_turn: offers,
        ..GameConfig::default()
    };
    let mut legal_ns = [0u128; 11];
    let mut legal_n = [0u64; 11];
    let mut legal_len = [0u64; 11];
    let mut apply_ns = [0u128; 8];
    let mut apply_n = [0u64; 8];
    let mut rng = Rng::new(1);
    let mut buf = Vec::with_capacity(512);
    let t_all = Instant::now();
    for seed in 0..games {
        let mut s = State::new(seed, cfg);
        while !s.is_over() {
            let p = phase_index(s.phase);
            let t = Instant::now();
            legal_actions(&s, &mut buf);
            legal_ns[p] += t.elapsed().as_nanos();
            legal_n[p] += 1;
            legal_len[p] += buf.len() as u64;
            let a = buf[rng.below(buf.len() as u32) as usize];
            let k = action_kind(a);
            let t = Instant::now();
            s.apply(a);
            apply_ns[k] += t.elapsed().as_nanos();
            apply_n[k] += 1;
        }
    }
    let total: u128 = legal_ns.iter().sum::<u128>() + apply_ns.iter().sum::<u128>();
    let steps: u64 = legal_n.iter().sum();
    println!(
        "max_offers_per_turn = {offers}, {games} games, {steps} steps, wall {:.2}s",
        t_all.elapsed().as_secs_f64()
    );
    println!(
        "{:<16} {:>10} {:>9} {:>8} {:>7}",
        "legal: phase", "calls", "ns/call", "avg len", "share"
    );
    for i in 0..11 {
        if legal_n[i] > 0 {
            println!(
                "{:<16} {:>10} {:>9.0} {:>8.1} {:>6.1}%",
                PHASES[i],
                legal_n[i],
                legal_ns[i] as f64 / legal_n[i] as f64,
                legal_len[i] as f64 / legal_n[i] as f64,
                100.0 * legal_ns[i] as f64 / total as f64
            );
        }
    }
    println!(
        "{:<16} {:>10} {:>9} {:>8} {:>7}",
        "apply: kind", "calls", "ns/call", "", "share"
    );
    for i in 0..8 {
        if apply_n[i] > 0 {
            println!(
                "{:<16} {:>10} {:>9.0} {:>8} {:>6.1}%",
                KINDS[i],
                apply_n[i],
                apply_ns[i] as f64 / apply_n[i] as f64,
                "",
                100.0 * apply_ns[i] as f64 / total as f64
            );
        }
    }
}
