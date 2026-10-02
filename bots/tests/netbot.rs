use settler_bots::arena::run_match_in;
use settler_bots::netbot::{
    load_net, net_bot, net_bot_with, parse_net_name, valid_net_name, NET_BATCH,
};
use settler_bots::{make_bot, Bot, Roster};
use settler_engine::*;
use settler_net::weights::{Arch, Weights};
use std::path::{Path, PathBuf};

fn no_trades() -> GameConfig {
    GameConfig {
        max_offers_per_turn: 0,
        ..GameConfig::default()
    }
}

/// A fresh nets directory holding `tiny.bin` (random weights, trunk 16, value width 8).
fn nets_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("settler-bots-nets-{}-{tag}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    Weights::random(
        Arch {
            trunk: 16,
            value: 8,
        },
        1,
    )
    .save(&dir.join("tiny.bin"))
    .unwrap();
    dir
}

#[test]
fn net_names_parse_and_bad_ones_are_rejected_without_touching_the_disk() {
    assert_eq!(parse_net_name("net:a"), Some(("a", 1000, 0)));
    assert_eq!(parse_net_name("net:gen-001@600"), Some(("gen-001", 600, 0)));
    assert_eq!(parse_net_name("net:x_Y-9@300+r8"), Some(("x_Y-9", 300, 8)));
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
fn net_self_play_explores_deterministically() {
    use settler_bots::selfplay::{run_with, Exploration};
    use settler_net::evaluator::NetEvaluator;
    use std::sync::Arc;
    let w = Arc::new(Weights::random(
        Arch {
            trunk: 16,
            value: 8,
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
