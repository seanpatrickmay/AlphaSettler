//! Behavior lock for optimization work: a hash over every action, event, and final state of
//! many seeded random games. Any change to the rules engine that alters a single game fails
//! this test. If the change is an intended rules change, recompute and update the constant
//! deliberately; performance work must never change it. The order of legal actions is part
//! of the lock (random play picks by index), so generators must keep emitting ascending order.

use settler_engine::legal::legal_actions;
use settler_engine::rng::Rng;
use settler_engine::*;

struct Fnv(u64);

impl Fnv {
    fn new() -> Fnv {
        Fnv(0xcbf2_9ce4_8422_2325)
    }

    fn bytes(&mut self, bs: &[u8]) {
        for &b in bs {
            self.0 ^= b as u64;
            self.0 = self.0.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
}

fn trace_hash(config: GameConfig, seeds: std::ops::Range<u64>) -> u64 {
    let mut h = Fnv::new();
    let mut buf = Vec::new();
    for seed in seeds {
        let mut s = State::new(seed, config);
        let mut rng = Rng::new(seed ^ 0x601D);
        let mut events: Vec<Event> = Vec::new();
        while !s.is_over() {
            legal_actions(&s, &mut buf);
            for a in &buf {
                h.bytes(&a.encode().to_le_bytes());
            }
            let a = buf[rng.below(buf.len() as u32) as usize];
            s.apply_with(a, None, &mut events);
        }
        h.bytes(format!("{events:?}").as_bytes());
        h.bytes(format!("{s:?}").as_bytes());
    }
    h.0
}

// Re-pinned 2026-10-01 for two rule fixes: every dev card may be played before rolling, and
// Road Building needs a placeable road (Plan 3, Task 1).
const TRADES_OFF: u64 = 2480314488515831906;
const TRADES_ON: u64 = 7874488224759562945;
const COMPAT: u64 = 2937732219135153055;

#[test]
fn golden_trace_trades_off() {
    let cfg = GameConfig {
        max_offers_per_turn: 0,
        ..GameConfig::default()
    };
    assert_eq!(trace_hash(cfg, 0..300), TRADES_OFF);
}

#[test]
fn golden_trace_trades_on() {
    assert_eq!(trace_hash(GameConfig::default(), 0..200), TRADES_ON);
}

#[test]
fn golden_trace_catanatron_compat() {
    let cfg = GameConfig {
        catanatron_compat: true,
        max_offers_per_turn: 0,
        ..GameConfig::default()
    };
    assert_eq!(trace_hash(cfg, 0..100), COMPAT);
}
