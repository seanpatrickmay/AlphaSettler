mod common;

use common::trading;
use settler_bots::arena::{belief_resets, play_game, run_match_in, GameRecord};
use settler_bots::ismcts::IsmctsBot;
use settler_bots::netbot::{
    load_net, net_bot, net_bot_with, parse_net_name, valid_net_name, NET_BATCH,
};
use settler_bots::{make_bot, Bot, Roster};
use settler_engine::*;
use settler_net::evaluator::NetEvaluator;
use settler_net::weights::{Arch, Weights};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

fn no_trades() -> GameConfig {
    GameConfig {
        max_offers_per_turn: 0,
        ..GameConfig::default()
    }
}

/// A fresh nets directory holding `tiny.bin` (random weights, width 8, 1 round).
fn nets_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("settler-bots-nets-{}-{tag}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    Weights::random(
        Arch {
            width: 8,
            rounds: 1,
        },
        1,
    )
    .save(&dir.join("tiny.bin"))
    .unwrap();
    dir
}

#[test]
fn net_names_parse_and_bad_ones_are_rejected_without_touching_the_disk() {
    assert_eq!(parse_net_name("net:a"), Some(("a", 1000, 0, 1, false)));
    assert_eq!(
        parse_net_name("net:gen-001@600"),
        Some(("gen-001", 600, 0, 1, false))
    );
    assert_eq!(
        parse_net_name("net:gen-001@600+s10"),
        Some(("gen-001", 600, 0, 10, false))
    );
    assert_eq!(
        parse_net_name("net:x_Y-9@300+r8"),
        Some(("x_Y-9", 300, 8, 1, false))
    );
    assert_eq!(
        parse_net_name("net:gen-004@600+t"),
        Some(("gen-004", 600, 0, 1, true))
    );
    assert_eq!(
        parse_net_name("net:gen-004@600+s4+t+r2"),
        Some(("gen-004", 600, 2, 4, true))
    );
    let longest = format!("net:{}", "a".repeat(64));
    assert_eq!(parse_net_name(&longest).map(|p| p.0.len()), Some(64));
    let too_long = format!("net:{}", "a".repeat(65));
    for bad in [
        "net:",
        "net:@5",
        "net:../x",
        "net:..",
        "net:.",
        "net:a/b",
        "net:/abs",
        "net:a\\b",
        "net:a b",
        "net:é",
        "net:a.bin",
        too_long.as_str(),
        "net:a@0",
        "net:a@",
        "net:a@abc",
        "net:a@1000001",
        "net:a@300+r0",
        "net:a@300+r201",
        "net:a@5@6",
        "net:a+r8",
        "netx:a",
        "Net:a",
        "ismcts:a",
        "net:a@300+t+t",
        "net:a@300+t5",
        "net:a+t@300",
        "net:a@300+T",
        "net:a@300+g+g",
        "net:a+g@300",
        "net:a@300+g2",
        "net:a@300+G",
    ] {
        assert_eq!(parse_net_name(bad), None, "{bad}");
        assert!(make_bot(bad, 0).is_none(), "{bad}");
        let e = Roster::new(&[bad], Path::new("/no/such/nets/dir"))
            .err()
            .unwrap();
        assert!(e.starts_with("unknown bot"), "{bad}: {e}");
    }
    assert!(!valid_net_name("") && valid_net_name("A-z_09"));
}

#[test]
fn a_missing_or_stale_weights_file_is_an_error_before_any_game() {
    let dir = nets_dir("errors");
    let e = Roster::new(&["net:missing"], &dir).err().unwrap();
    assert!(
        e.contains("cannot read") && e.contains("missing.bin"),
        "{e}"
    );
    let mut stale = std::fs::read(dir.join("tiny.bin")).unwrap();
    stale[12] ^= 1;
    std::fs::write(dir.join("stale.bin"), &stale).unwrap();
    let e = run_match_in("net:stale", "greedy", 0..1, no_trades(), 1, &dir).unwrap_err();
    assert!(e.contains("encoding hash"), "{e}");
    assert!(load_net(&dir, "tiny").is_ok());
    assert!(make_bot("net:tiny", 0).is_none(), "net bots need a roster");
}

#[test]
fn a_net_bot_plays_whole_games_independent_of_thread_count() {
    let dir = nets_dir("arena");
    let a = run_match_in("net:tiny@10", "greedy", 0..2, no_trades(), 1, &dir).unwrap();
    let b = run_match_in("net:tiny@10", "greedy", 0..2, no_trades(), 3, &dir).unwrap();
    assert_eq!(a, b);
    assert_eq!(a.len(), 8);
    assert!(a.iter().all(|r| r.belief_resets == 0 && r.turns > 0));
    let roster = Roster::new(&["net:tiny@10+r2", "greedy"], &dir).unwrap();
    assert_eq!(roster.make("net:tiny@10+r2", 0).unwrap().name(), "net");
    assert_eq!(roster.make("greedy", 0).unwrap().name(), "greedy");
    assert!(roster.make("net:other", 0).is_none());
}

/// FNV-1a over (winner, final VP, turns, actions) of each record, as in tests/ismcts.rs.
fn record_digest(records: &[GameRecord]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let mut eat = |bytes: &[u8]| {
        for &b in bytes {
            h ^= b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    };
    for r in records {
        eat(&[r.winner.unwrap_or(u8::MAX)]);
        eat(&r.vp);
        eat(&r.turns.to_le_bytes());
        eat(&r.actions.to_le_bytes());
    }
    h
}

/// The digest of a trading-game match of `net:tiny@16+t` (random weights, seed 1), computed at
/// eb1f739 before the Gumbel root (P2): net bots without `+g` must play the same games.
#[test]
fn net_bots_without_the_gumbel_suffix_play_the_games_they_played_before_p2() {
    let dir = nets_dir("p2-pin");
    let records = run_match_in("net:tiny@16+t", "greedy", 0..1, trading(), 1, &dir).unwrap();
    assert_eq!(record_digest(&records), 0xf4d5c47b40f849a3);
}

#[test]
fn net_bots_search_with_the_given_leaf_batch() {
    let w = load_net(&nets_dir("batch"), "tiny").unwrap();
    let bot = net_bot(0, w.clone(), 10, 0);
    assert_eq!((bot.name(), bot.config().batch), ("net", NET_BATCH));
    assert_eq!(bot.config().simulations, 10);
    let bot = net_bot_with(0, w.clone(), 20, 3, NET_BATCH + 5);
    assert_eq!(bot.config().batch, NET_BATCH + 5);
    assert_eq!(bot.config().simulations, 20);
    assert!(std::sync::Arc::ptr_eq(bot.evaluator().weights(), &w));
}

#[test]
fn gumbel_net_names_build_the_gumbel_bot_in_any_order() {
    let dir = nets_dir("gumbel");
    let names = ["net:tiny@16+g", "net:tiny@16+t+g", "net:tiny@16+g+s2+t"];
    let roster = Roster::new(&names, &dir).unwrap();
    for name in names {
        assert_eq!(roster.make(name, 0).unwrap().name(), "net", "{name}");
    }
    let gumbel_root = |name: &str| {
        let bot = roster.make(name, 0).unwrap();
        let d: std::collections::HashMap<_, _> = bot.diagnostics().into_iter().collect();
        d["gumbel_root"]
    };
    for name in names {
        assert_eq!(gumbel_root(name), 1, "{name}");
    }
    let plain = ["net:tiny@16", "net:tiny@16+t", "net:tiny@16+s2+t"];
    let roster_plain = Roster::new(&plain, &dir).unwrap();
    for name in plain {
        let bot = roster_plain.make(name, 0).unwrap();
        let d: std::collections::HashMap<_, _> = bot.diagnostics().into_iter().collect();
        assert_eq!(d["gumbel_root"], 0, "{name}");
    }
    let w = load_net(&dir, "tiny").unwrap();
    let mut s = State::new(5, trading());
    let mut rng = settler_engine::rng::Rng::new(5);
    let mut compared = 0;
    for i in 0..400 {
        if s.is_over() {
            break;
        }
        let legal = s.legal_actions();
        if i % 9 == 0 && legal.len() > 1 {
            let obs = s.observation(s.current_actor());
            let named = roster.make("net:tiny@16+t+g", 3).unwrap().act(&obs, &legal);
            let direct = net_bot(3, w.clone(), 16, 0)
                .with_offers(true)
                .with_gumbel(true)
                .act(&obs, &legal);
            assert_eq!(named, direct, "move {i}");
            compared += 1;
        }
        s.apply(legal[rng.below(legal.len() as u32) as usize]);
    }
    assert!(compared >= 10, "only {compared} positions compared");
}

#[test]
fn net_self_play_explores_deterministically() {
    use settler_bots::selfplay::{run_with, Exploration};
    let w = Arc::new(Weights::random(
        Arch {
            width: 8,
            rounds: 1,
        },
        3,
    ));
    let make = || NetEvaluator::new(w.clone());
    let ex = Exploration {
        full_simulations: 6,
        fast_simulations: 3,
        ..Exploration::standard()
    };
    let a = run_with(0..2, no_trades(), &ex, NET_BATCH, &make, 1).unwrap();
    let b = run_with(0..2, no_trades(), &ex, NET_BATCH, &make, 2).unwrap();
    assert_eq!(a, b);
    assert!(a
        .iter()
        .all(|g| g.belief_resets == 0 && !g.decisions.is_empty()));
    assert!(a.iter().flat_map(|g| &g.decisions).any(|d| !d.full_search));
}

/// A net bot that counts the offers it makes and its searched trade answers and confirmations.
struct Counting {
    bot: IsmctsBot<NetEvaluator>,
    offers: Arc<AtomicU32>,
    trade_searches: Arc<AtomicU32>,
}

impl Bot for Counting {
    fn name(&self) -> &'static str {
        "counting"
    }

    fn observe(&mut self, viewer: PlayerId, events: &[Event]) {
        self.bot.observe(viewer, events);
    }

    fn act(&mut self, obs: &Observation, legal: &[Action]) -> Action {
        let a = self.bot.act(obs, legal);
        if matches!(a, Action::OfferTrade { .. }) {
            self.offers.fetch_add(1, Ordering::Relaxed);
        }
        let trade = matches!(obs.phase, Phase::TradeResponse | Phase::TradeConfirm);
        if trade && self.bot.take_last_search().is_some() {
            self.trade_searches.fetch_add(1, Ordering::Relaxed);
        }
        a
    }

    fn diagnostics(&self) -> Vec<(&'static str, u64)> {
        self.bot.diagnostics()
    }
}

#[test]
fn net_bots_with_and_without_the_offer_suffix_play_the_trading_game() {
    let dir = nets_dir("trading");
    let records = run_match_in("net:tiny@10+t", "net:tiny@10", 0..2, trading(), 2, &dir).unwrap();
    assert_eq!(records.len(), 8);
    assert!(records.iter().all(|r| r.belief_resets == 0), "{records:?}");
    // The same table by hand (seat 0 `+t`): offers have priors, so seat 0 offers, the others
    // never do, and trade answers and confirmations are reached and searched.
    let w = load_net(&dir, "tiny").unwrap();
    let counter = || Arc::new(AtomicU32::new(0));
    let offers: Vec<Arc<AtomicU32>> = (0..4).map(|_| counter()).collect();
    let searched = counter();
    for seed in 0..2u64 {
        let mut bots: Vec<Box<dyn Bot>> = (0..4)
            .map(|p| {
                Box::new(Counting {
                    bot: net_bot(seed * 4 + p as u64, w.clone(), 10, 0).with_offers(p == 0),
                    offers: offers[p].clone(),
                    trade_searches: searched.clone(),
                }) as Box<dyn Bot>
            })
            .collect();
        play_game(seed, trading(), &mut bots);
        assert_eq!(belief_resets(bots.iter().map(|b| &**b)), 0, "seed {seed}");
    }
    let made: Vec<u32> = offers.iter().map(|c| c.load(Ordering::Relaxed)).collect();
    assert!(
        made[0] > 0 && made[1..] == [0, 0, 0],
        "offers by seat: {made:?}"
    );
    assert!(
        searched.load(Ordering::Relaxed) > 0,
        "no answer or confirmation was searched"
    );
}
