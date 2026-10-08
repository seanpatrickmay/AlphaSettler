mod common;

use common::trading;
use settler_bots::arena::{belief_resets, play_game, run_match, GameRecord};
use settler_bots::ismcts::{parse_name, IsmctsBot, MoveOptions};
use settler_bots::{make_bot, Bot, BOT_NAMES};
use settler_engine::rng::Rng;
use settler_engine::*;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

fn no_trades() -> GameConfig {
    GameConfig {
        max_offers_per_turn: 0,
        ..GameConfig::default()
    }
}

#[test]
fn names_parse_and_bad_ones_are_rejected() {
    assert!(BOT_NAMES.contains(&"ismcts"));
    assert_eq!(parse_name("ismcts"), Some((1000, 0)));
    assert_eq!(parse_name("ismcts@1"), Some((1, 0)));
    assert_eq!(parse_name("ismcts@1000000"), Some((1_000_000, 0)));
    assert_eq!(parse_name("ismcts@300+r8"), Some((300, 8)));
    for bad in [
        "ismcts@0",
        "ismcts@",
        "ismcts@abc",
        "ismcts@+5",
        "ismcts@1000001",
        "ismcts@300+r0",
        "ismcts@300+r201",
        "ismcts@300+r",
        "ismctsx",
        "ismcts@300/r8",
        "ismcts+r8",
    ] {
        assert_eq!(parse_name(bad), None, "{bad}");
        assert!(make_bot(bad, 0).is_none(), "{bad}");
    }
    assert_eq!(make_bot("ismcts@50", 0).unwrap().name(), "ismcts");
}

#[test]
fn plays_whole_games_without_belief_resets() {
    for seed in 0..2u64 {
        let mut bots: Vec<Box<dyn Bot>> = (0..4u64)
            .map(|p| Box::new(IsmctsBot::new(seed * 4 + p, 20, 0)) as Box<dyn Bot>)
            .collect();
        let (_, _, turns, _) = play_game(seed, no_trades(), &mut bots);
        assert!(turns > 0);
        for b in &bots {
            let d: std::collections::HashMap<_, _> = b.diagnostics().into_iter().collect();
            assert_eq!(d["belief_resets"], 0, "seed {seed}");
            assert!(d["searches"] > 0);
        }
    }
}

#[test]
fn arena_results_do_not_depend_on_thread_count() {
    let a = run_match("ismcts@20", "greedy", 0..3, no_trades(), 1).unwrap();
    let b = run_match("ismcts@20", "greedy", 0..3, no_trades(), 3).unwrap();
    assert_eq!(a, b);
}

#[test]
fn beats_random() {
    let records = run_match("ismcts@100", "random", 0..10, no_trades(), 10).unwrap();
    let wins = records
        .iter()
        .filter(|r| r.winner == Some(r.candidate_seat))
        .count();
    assert!(
        wins as f64 / records.len() as f64 > 0.8,
        "{wins}/{}",
        records.len()
    );
}

/// Answers and confirmations are searched, from a belief rebuilt from the observation alone
/// (each decision is a fresh bot with no history).
#[test]
fn answers_and_confirmations_are_searched() {
    let mut g = Game::new(2, GameConfig::default());
    let mut rng = Rng::new(2);
    let (mut answered, mut confirmed) = (0, 0);
    while !g.state().is_over() && (answered < 3 || confirmed < 3) {
        let s = *g.state();
        let legal = g.legal_actions();
        let actor = s.current_actor();
        if matches!(s.phase, Phase::TradeResponse | Phase::TradeConfirm) && legal.len() > 1 {
            let mut bot = IsmctsBot::new(0, 50, 0);
            let a = bot.act(&s.observation(actor), &legal);
            assert!(legal.contains(&a));
            assert!(
                bot.take_last_search().is_some(),
                "{:?} was answered without a search",
                s.phase
            );
            if s.phase == Phase::TradeResponse {
                answered += 1;
            } else {
                confirmed += 1;
            }
        }
        let pick = if s.phase == Phase::TradeResponse && legal.contains(&Action::AcceptTrade) {
            Action::AcceptTrade
        } else {
            legal[rng.below(legal.len() as u32) as usize]
        };
        g.apply(pick).unwrap();
    }
    assert!(
        answered >= 3 && confirmed >= 3,
        "{answered} answers, {confirmed} confirmations"
    );
}

fn resets(bot: &IsmctsBot) -> u64 {
    let d: std::collections::HashMap<_, _> = bot.diagnostics().into_iter().collect();
    d["belief_resets"]
}

/// Random moves from `g` until player 0, past setup and turn `min_turn` and outside the trade
/// phases, must make a decision that is searched (`searched`: more than one searchable move) or
/// not.
fn advance(g: &mut Game, rng: &mut Rng, min_turn: u32, searched: bool) {
    let mut moves = Vec::new();
    loop {
        let s = *g.state();
        assert!(!s.is_over(), "the game ended before the wanted decision");
        let legal = g.legal_actions();
        settler_search::search_actions(&legal, &mut moves);
        let ready = s.turn >= min_turn
            && s.current_actor() == 0
            && !matches!(
                s.phase,
                Phase::SetupSettlement
                    | Phase::SetupRoad { .. }
                    | Phase::TradeResponse
                    | Phase::TradeConfirm
            );
        if ready && (moves.len() > 1) == searched {
            return;
        }
        g.apply(legal[rng.below(legal.len() as u32) as usize])
            .unwrap();
    }
}

/// Events no history can explain: player 0 discards one more wood than they hold.
fn contradiction(obs: &Observation) -> Vec<Event> {
    let wood = obs.my_hand[Resource::Wood.index()] as usize;
    vec![
        Event::Discarded {
            player: 0,
            resource: Resource::Wood
        };
        wood + 1
    ]
}

/// Act for player 0 at `g`'s current decision, check the move is legal and apply it.
fn act_and_apply(bot: &mut IsmctsBot, g: &mut Game) {
    let legal = g.legal_actions();
    let a = bot.act(&g.observation(0), &legal);
    assert!(legal.contains(&a));
    g.apply(a).unwrap();
}

#[test]
fn a_contradictory_history_is_counted_and_recovered_from() {
    for (inject, expected) in [(false, 0), (true, 1)] {
        let mut g = Game::new(7, no_trades());
        let mut rng = Rng::new(7);
        advance(&mut g, &mut rng, 8, true);
        let mut bot = IsmctsBot::new(1, 50, 0);
        let mut seen = g.log_for(0).len(); // like the arena: everything up to now, then each new event
        bot.observe(0, &g.log_for(0));
        if inject {
            bot.observe(0, &contradiction(&g.observation(0)));
        }
        assert_eq!(
            resets(&bot),
            expected,
            "inject {inject}: counted at the error"
        );
        act_and_apply(&mut bot, &mut g);
        assert!(bot.take_last_search().is_some());
        assert_eq!(
            resets(&bot),
            expected,
            "inject {inject}: the rebuild is not counted again"
        );
        // The rebuilt belief keeps tracking the true history.
        for _ in 0..3 {
            advance(&mut g, &mut rng, 0, true);
            let log = g.log_for(0);
            bot.observe(0, &log[seen..]);
            seen = log.len();
            act_and_apply(&mut bot, &mut g);
            assert!(bot.take_last_search().is_some());
            assert_eq!(resets(&bot), expected, "inject {inject}: later decisions");
        }
    }
}

#[test]
fn a_contradiction_before_an_unsearched_move_is_counted_once() {
    let mut g = Game::new(8, no_trades());
    let mut rng = Rng::new(8);
    advance(&mut g, &mut rng, 8, false);
    let mut bot = IsmctsBot::new(2, 50, 0);
    let mut seen = g.log_for(0).len();
    bot.observe(0, &g.log_for(0));
    bot.observe(0, &contradiction(&g.observation(0)));
    assert_eq!(resets(&bot), 1);
    act_and_apply(&mut bot, &mut g);
    assert!(
        bot.take_last_search().is_none(),
        "a single move is not searched"
    );
    assert_eq!(resets(&bot), 1);
    // Events while the belief is dropped are skipped, then the next search rebuilds uncounted.
    for _ in 0..3 {
        advance(&mut g, &mut rng, 0, true);
        let log = g.log_for(0);
        bot.observe(0, &log[seen..]);
        seen = log.len();
        act_and_apply(&mut bot, &mut g);
        assert!(bot.take_last_search().is_some());
        assert_eq!(resets(&bot), 1);
    }
}

#[test]
fn acting_without_any_history_is_one_reset() {
    let s = main_with(4, CITY_COST, no_trades());
    let mut bot = IsmctsBot::new(4, 50, 0);
    decide(&mut bot, &s);
    assert_eq!(resets(&bot), 1);
    decide(&mut bot, &s); // the belief rebuilt from this observation still matches it
    assert_eq!(resets(&bot), 1);
}

#[test]
fn native_arena_records_report_zero_belief_resets() {
    let records = run_match("ismcts@20", "greedy", 0..2, no_trades(), 2).unwrap();
    assert_eq!(records.len(), 8);
    assert!(records.iter().all(|r| r.belief_resets == 0), "{records:?}");
}

#[test]
fn belief_resets_sum_over_seats_that_report_them() {
    let s = main_with(5, CITY_COST, no_trades());
    let mut counted = IsmctsBot::new(5, 20, 0);
    decide(&mut counted, &s); // no history: one reset
    let greedy = make_bot("greedy", 0).unwrap();
    let clean = IsmctsBot::new(6, 20, 0);
    let seats: [&dyn Bot; 3] = [&counted, greedy.as_ref(), &clean];
    assert_eq!(belief_resets(seats), 1);
}

#[test]
fn a_decision_counts_at_most_one_reset() {
    // An observation the sampler rejects (more hidden dev cards than the deck has) after a
    // reset was already counted for the same decision: the failed search adds none.
    let s = main_with(6, CITY_COST, no_trades());
    let mut broken = s.observation(0);
    broken.dev_card_counts[1] += 30;
    let legal = s.legal_actions();
    let mut fresh = IsmctsBot::new(6, 20, 0);
    assert!(legal.contains(&fresh.act(&broken, &legal)));
    assert!(fresh.take_last_search().is_none(), "the search failed");
    assert_eq!(resets(&fresh), 1, "no history, then a failed search");
    // A later decision whose search fails again is a new reset.
    fresh.act(&broken, &legal);
    assert_eq!(resets(&fresh), 2);

    let mut contradicted = IsmctsBot::new(7, 20, 0);
    contradicted.observe(0, &contradiction(&s.observation(0)));
    assert_eq!(resets(&contradicted), 1);
    contradicted.act(&broken, &legal);
    assert_eq!(
        resets(&contradicted),
        1,
        "a contradiction, then a failed search"
    );
}

#[test]
fn a_new_seat_starts_a_new_game_without_a_reset() {
    let mut bots: Vec<Box<dyn Bot>> = (0..4u64).map(|p| make_bot("greedy", p).unwrap()).collect();
    bots[0] = Box::new(IsmctsBot::new(9, 20, 0));
    play_game(0, no_trades(), &mut bots);
    bots.swap(0, 2);
    play_game(1, no_trades(), &mut bots);
    let d: std::collections::HashMap<_, _> = bots[2].diagnostics().into_iter().collect();
    assert_eq!(d["belief_resets"], 0);
    assert!(d["searches"] > 0);
}

#[test]
fn rollouts_play_whole_games() {
    let mut bots: Vec<Box<dyn Bot>> = (0..4u64).map(|p| make_bot("greedy", p).unwrap()).collect();
    bots[1] = make_bot("ismcts@20+r8", 1).unwrap();
    let (_, _, turns, _) = play_game(3, no_trades(), &mut bots); // panics on an illegal move
    assert!(turns > 0);
    let d: std::collections::HashMap<_, _> = bots[1].diagnostics().into_iter().collect();
    assert_eq!(d["belief_resets"], 0);
    assert!(d["searches"] > 0);
}

/// Player 0 in Main right after setup with `cards` added.
fn main_with(seed: u64, cards: Hand, config: GameConfig) -> State {
    let mut s = State::new(seed, config);
    let mut rng = Rng::new(seed);
    let mut buf = Vec::new();
    while matches!(s.phase, Phase::SetupSettlement | Phase::SetupRoad { .. }) {
        settler_engine::legal::legal_actions(&s, &mut buf);
        s.apply(buf[rng.below(buf.len() as u32) as usize]);
    }
    let mut snap = s.snapshot();
    for r in 0..NUM_RESOURCES {
        snap.bank[r] -= cards[r];
        snap.players[0].hand[r] += cards[r];
    }
    snap.phase = Phase::Main;
    State::from_snapshot(seed, config, &snap).unwrap()
}

fn decide(bot: &mut IsmctsBot, s: &State) -> Action {
    let obs = s.observation(0);
    let legal = s.legal_actions();
    let a = bot.act(&obs, &legal);
    assert!(legal.contains(&a));
    a
}

#[test]
fn builds_an_affordable_city_rather_than_ending_the_turn() {
    for seed in 0..4 {
        let s = main_with(seed, CITY_COST, no_trades());
        let mut bot = IsmctsBot::new(seed, 1000, 0);
        assert!(
            matches!(decide(&mut bot, &s), Action::BuildCity(_)),
            "seed {seed}"
        );
    }
}

#[test]
fn never_robs_its_own_buildings_when_an_opponent_tile_is_free() {
    use settler_engine::topology::topo;
    for seed in 0..4 {
        let mut s = main_with(seed, [0; 5], no_trades());
        let mut snap = s.snapshot();
        let i = snap
            .dev_deck
            .iter()
            .position(|&c| c == DevCard::Knight)
            .unwrap();
        snap.dev_deck.remove(i);
        snap.players[0].dev_hand[DevCard::Knight.index()] = 1;
        s = State::from_snapshot(seed, no_trades(), &snap).unwrap();
        s.apply(Action::PlayKnight);
        assert_eq!(s.phase, Phase::MoveRobber);
        let mine = s.buildings(0);
        let theirs = (1..4).fold(0, |m, p| m | s.buildings(p));
        let touches = |t: u8, m: u64| topo().tile_node_mask[t as usize] & m != 0;
        assert!((0..19).any(|t| t != s.robber && touches(t, theirs) && !touches(t, mine)));
        let mut bot = IsmctsBot::new(seed, 1000, 0);
        let Action::MoveRobber(t) = decide(&mut bot, &s) else {
            panic!("not a robber move")
        };
        assert!(!touches(t, mine), "seed {seed}: robbed own tile {t}");
    }
}

#[test]
fn search_results_are_kept_for_self_play() {
    let s = main_with(5, CITY_COST, no_trades());
    let mut bot = IsmctsBot::new(5, 40, 0);
    decide(&mut bot, &s);
    let r = bot.take_last_search().expect("the decision was searched");
    assert_eq!(r.simulations, 40);
    assert!(bot.take_last_search().is_none());
}

/// FNV-1a over (winner, final VP, turns, actions) of each record, in order.
fn digest(records: &[GameRecord]) -> u64 {
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

/// The digests of 3a's `ismcts` games, computed at 9dc553a before IsmctsBot became generic over
/// its evaluator: the refactor must not change a single game. On these seeds `ismcts@20` plays
/// greedy's moves, so `ismcts@200` (whose games differ) is pinned too.
#[test]
fn the_generic_refactor_leaves_3a_ismcts_games_unchanged() {
    let games = |name| digest(&run_match(name, "greedy", 0..2, no_trades(), 1).unwrap());
    assert_eq!(games("ismcts@20"), 0x30461dca2efcbb79);
    assert_eq!(games("ismcts@200"), 0xce9e1db7dc5ee013);
    assert_ne!(
        games("ismcts@200"),
        games("greedy"),
        "the pin sees the search"
    );
}

/// The digest of a trading-game match of `ismcts@200+t+s2`, computed at eb1f739 before the
/// Gumbel root (P2): bots without `+g` must play the same games (spec P2 Section 4).
#[test]
fn bots_without_the_gumbel_suffix_play_the_games_they_played_before_p2() {
    let records = run_match("ismcts@200+t+s2", "greedy", 0..2, trading(), 1).unwrap();
    assert_eq!(digest(&records), 0x840a576a3ed2426e);
}

#[test]
fn a_sampled_move_was_visited_and_is_the_recorded_action() {
    use settler_bots::ismcts::{sample_by_visits, MoveOptions};
    let s = main_with(5, CITY_COST, no_trades());
    let obs = s.observation(0);
    let legal = s.legal_actions();
    let plain = MoveOptions {
        simulations: 40,
        noise: None,
        sample: None,
    };
    for seed in 0..20 {
        let mut c = IsmctsBot::new(5, 40, 0);
        let x = c.act_with(
            &obs,
            &legal,
            &MoveOptions {
                sample: Some(seed),
                ..plain
            },
        );
        let r = c.take_last_search().unwrap();
        assert_eq!(r.action, x, "the recorded action is the move played");
        assert!(r.visits[legal.iter().position(|&l| l == x).unwrap()] > 0);
    }
    assert_eq!(sample_by_visits(&legal, &vec![0; legal.len()], 1), None);
    let only = legal.len() - 1;
    let mut visits = vec![0; legal.len()];
    visits[only] = 3;
    assert_eq!(sample_by_visits(&legal, &visits, 7), Some(legal[only]));
}

#[test]
fn moves_are_sampled_in_proportion_to_visits() {
    use settler_bots::ismcts::sample_by_visits;
    let legal = [Action::EndTurn, Action::BuyDev, Action::Roll];
    let visits = [6, 3, 1];
    let n = 20_000;
    let mut count = [0u32; 3];
    for seed in 0..n {
        let a = sample_by_visits(&legal, &visits, seed).unwrap();
        count[legal.iter().position(|&l| l == a).unwrap()] += 1;
    }
    for (c, v) in count.iter().zip(visits) {
        let p = v as f64 / 10.0;
        let se = (p * (1.0 - p) / n as f64).sqrt();
        assert!((*c as f64 / n as f64 - p).abs() < 4.0 * se, "{count:?}");
    }
}

#[test]
fn setup_factors_parse_and_bad_ones_are_rejected() {
    use settler_bots::ismcts::{split_setup, MAX_SETUP_FACTOR};
    assert_eq!(split_setup("ismcts@100"), Some(("ismcts@100", 1)));
    assert_eq!(split_setup("ismcts@100+s10"), Some(("ismcts@100", 10)));
    assert_eq!(split_setup("ismcts@100+r5+s3"), Some(("ismcts@100+r5", 3)));
    assert_eq!(
        split_setup("net:gen-004@100+s10"),
        Some(("net:gen-004@100", 10))
    );
    let max = format!("ismcts@1+s{MAX_SETUP_FACTOR}");
    assert_eq!(split_setup(&max), Some(("ismcts@1", MAX_SETUP_FACTOR)));
    let over = format!("ismcts@1+s{}", MAX_SETUP_FACTOR + 1);
    for bad in [
        "ismcts@100+s",
        "ismcts@100+s0",
        "ismcts@100+s1",
        "ismcts@100+s-2",
        "ismcts@100+sx",
        &over,
    ] {
        assert_eq!(split_setup(bad), None, "{bad}");
    }
    assert!(make_bot("ismcts@20+s5", 0).is_some());
    assert!(make_bot("ismcts@20+s0", 0).is_none());
    assert!(
        make_bot("greedy+s5", 0).is_none(),
        "only searching bots take a setup factor"
    );
}

#[test]
fn a_setup_factor_multiplies_the_simulations_of_setup_settlements_only() {
    let mut g = Game::new(3, no_trades());
    let mut bot = IsmctsBot::new(3, 20, 0).with_setup_factor(5);
    assert_eq!(g.observation(0).phase, Phase::SetupSettlement);
    act_and_apply(&mut bot, &mut g);
    assert_eq!(bot.take_last_search().unwrap().simulations, 100);
    assert!(matches!(g.observation(0).phase, Phase::SetupRoad { .. }));
    act_and_apply(&mut bot, &mut g);
    if let Some(r) = bot.take_last_search() {
        assert_eq!(r.simulations, 20, "setup roads keep the base budget");
    }
    let mut plain = IsmctsBot::new(3, 20, 0);
    let mut g = Game::new(3, no_trades());
    act_and_apply(&mut plain, &mut g);
    assert_eq!(plain.take_last_search().unwrap().simulations, 20);
}

#[test]
fn the_offer_suffix_combines_with_the_others_in_any_order() {
    use settler_bots::ismcts::split_trades;
    assert_eq!(
        split_trades("ismcts@100"),
        Some(("ismcts@100".to_owned(), false))
    );
    assert_eq!(
        split_trades("ismcts@100+t"),
        Some(("ismcts@100".to_owned(), true))
    );
    for name in [
        "ismcts@100+t+r5+s3",
        "ismcts@100+r5+t+s3",
        "ismcts@100+s3+r5+t",
        "ismcts@100+s3+t+r5",
    ] {
        assert_eq!(
            split_trades(name),
            Some(("ismcts@100+r5+s3".to_owned(), true)),
            "{name}"
        );
        assert!(make_bot(name, 0).is_some(), "{name}");
    }
    assert_eq!(
        split_trades("net:gen-004@600+s4+t"),
        Some(("net:gen-004@600+s4".to_owned(), true))
    );
    for bad in [
        "ismcts@100+t+t",
        "ismcts@100+t5",
        "ismcts@100+T",
        "ismcts@100+",
        "ismcts@100+r5+r5",
        "ismcts@100+s3+s3",
        "ismcts@100+x",
    ] {
        assert_eq!(split_trades(bad), None, "{bad}");
        assert!(make_bot(bad, 0).is_none(), "{bad}");
    }
    assert!(make_bot("ismcts+t", 0).is_some());
    assert!(
        make_bot("greedy+t", 0).is_none(),
        "only searching bots offer"
    );
    assert!(make_bot("random+t", 0).is_none());
}

#[test]
fn the_gumbel_suffix_parses_in_any_order_and_only_once() {
    use settler_bots::ismcts::{split_gumbel, split_trades};
    assert_eq!(
        split_gumbel("ismcts@100"),
        Some(("ismcts@100".to_owned(), false))
    );
    assert_eq!(
        split_gumbel("ismcts@100+g"),
        Some(("ismcts@100".to_owned(), true))
    );
    assert_eq!(
        split_gumbel("net:gen-034@32+g"),
        Some(("net:gen-034@32".to_owned(), true))
    );
    for name in [
        "ismcts@100+g+t+r5+s3",
        "ismcts@100+t+g+r5+s3",
        "ismcts@100+r5+s3+t+g",
        "ismcts@100+s3+g+r5+t",
    ] {
        let (rest, gumbel) = split_gumbel(name).unwrap();
        assert!(gumbel, "{name}");
        assert_eq!(
            split_trades(&rest),
            Some(("ismcts@100+r5+s3".to_owned(), true)),
            "{name}"
        );
        assert!(make_bot(name, 0).is_some(), "{name}");
    }
    for bad in [
        "ismcts@100+g+g",
        "ismcts@100+g2",
        "ismcts@100+G",
        "ismcts@100+gt",
        "ismcts@100+g+",
        "ismcts+g@100",
        "greedy+g",
        "random+g",
        "+g",
    ] {
        assert!(make_bot(bad, 0).is_none(), "{bad}");
    }
    assert!(make_bot("ismcts+g", 0).is_some());
}

#[test]
fn a_gumbel_name_builds_the_gumbel_bot_and_plain_names_stay_puct() {
    use settler_search::RootMode;
    assert_eq!(IsmctsBot::new(0, 10, 0).config().root, RootMode::Puct);
    assert_eq!(
        IsmctsBot::new(0, 10, 0).with_gumbel(true).config().root,
        RootMode::Gumbel { draw: false }
    );
    let mut differ = 0;
    for seed in 0..12 {
        let s = main_with(seed, [3, 3, 3, 3, 3], no_trades());
        let (obs, legal) = (s.observation(0), s.legal_actions());
        let named = make_bot("ismcts@40+g", 7).unwrap().act(&obs, &legal);
        let mut g = IsmctsBot::new(7, 40, 0).with_gumbel(true);
        let direct = g.act(&obs, &legal);
        let plain_named = make_bot("ismcts@40", 7).unwrap().act(&obs, &legal);
        let mut p = IsmctsBot::new(7, 40, 0);
        let puct = p.act(&obs, &legal);
        assert_eq!(named, direct, "seed {seed}");
        assert_eq!(plain_named, puct, "seed {seed}");
        let (gr, pr) = (g.take_last_search().unwrap(), p.take_last_search().unwrap());
        assert!(gr.improved.is_some(), "seed {seed}: Gumbel improved policy");
        assert!(
            pr.improved.is_none(),
            "seed {seed}: PUCT has no improved policy"
        );
        differ += (gr.visits != pr.visits) as u32;
    }
    assert!(
        differ > 0,
        "the Gumbel root spent its visits as PUCT did on every position"
    );
}

/// Whether a boxed bot reports a Gumbel root search.
fn gumbel_root(bot: &dyn Bot) -> u64 {
    let d: std::collections::HashMap<_, _> = bot.diagnostics().into_iter().collect();
    d["gumbel_root"]
}

#[test]
fn make_bot_gives_gumbel_roots_exactly_to_names_with_the_gumbel_suffix() {
    for (name, expected) in [
        ("ismcts@10", 0),
        ("ismcts@10+t", 0),
        ("ismcts@10+s2+t", 0),
        ("ismcts@10+g", 1),
        ("ismcts@10+t+g", 1),
        ("ismcts@10+g+s2+t", 1),
        ("ismcts@10+r2+g", 1),
    ] {
        let bot = make_bot(name, 0).unwrap();
        assert_eq!(gumbel_root(bot.as_ref()), expected, "{name}");
    }
}

#[test]
fn a_gumbel_bot_keeps_its_setup_factor() {
    let mut g = Game::new(3, no_trades());
    let mut bot = IsmctsBot::new(3, 20, 0)
        .with_setup_factor(5)
        .with_gumbel(true);
    assert_eq!(g.observation(0).phase, Phase::SetupSettlement);
    act_and_apply(&mut bot, &mut g);
    let r = bot.take_last_search().unwrap();
    assert_eq!(r.simulations, 100);
    assert!(
        r.improved.is_some(),
        "the setup settlement was a Gumbel search"
    );
}

#[test]
fn only_a_bot_with_the_offer_suffix_searches_offers() {
    let is_offer = |a: &Action| matches!(a, Action::OfferTrade { .. });
    for seed in 0..3 {
        let s = main_with(seed, [1, 1, 1, 1, 1], trading());
        let legal = s.legal_actions();
        assert!(legal.iter().any(is_offer), "seed {seed}: offers are legal");
        let offer_visits = |offers: bool| {
            let mut bot = IsmctsBot::new(seed, 200, 0).with_offers(offers);
            let a = bot.act(&s.observation(0), &legal);
            let r = bot.take_last_search().expect("the decision was searched");
            let v: u32 = legal
                .iter()
                .zip(&r.visits)
                .filter(|(a, _)| is_offer(a))
                .map(|(_, &v)| v)
                .sum();
            (a, v)
        };
        let (a, v) = offer_visits(false);
        assert!(
            !is_offer(&a) && v == 0,
            "seed {seed}: the answer-only bot searched an offer"
        );
        let (_, v) = offer_visits(true);
        assert!(v > 0, "seed {seed}: the +t bot never searched an offer");
    }
}

/// An IsmctsBot that plays a move drawn in proportion to root visits (as self-play's opening
/// does), so any offer the search visits gets played, and counts its offers and its searched
/// trade answers and confirmations.
struct Sampling {
    bot: IsmctsBot,
    rng: Rng,
    offers: Arc<AtomicU32>,
    trade_searches: Arc<AtomicU32>,
}

impl Bot for Sampling {
    fn name(&self) -> &'static str {
        "sampling"
    }

    fn observe(&mut self, viewer: PlayerId, events: &[Event]) {
        self.bot.observe(viewer, events);
    }

    fn act(&mut self, obs: &Observation, legal: &[Action]) -> Action {
        let opts = MoveOptions {
            simulations: 30,
            noise: None,
            sample: Some(self.rng.next_u64()),
        };
        let a = self.bot.act_with(obs, legal, &opts);
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

/// Whole trading games (every move checked legal by `play_game`): seat 0 may offer, the other
/// seats answer by searching but never offer, and no belief is ever reset.
#[test]
fn trading_games_play_through_and_only_the_offering_seat_offers() {
    let counter = || Arc::new(AtomicU32::new(0));
    let offers: Vec<Arc<AtomicU32>> = (0..4).map(|_| counter()).collect();
    let searched = counter();
    for seed in 0..3u64 {
        let mut bots: Vec<Box<dyn Bot>> = (0..4)
            .map(|p| {
                Box::new(Sampling {
                    bot: IsmctsBot::new(seed * 4 + p as u64, 30, 0).with_offers(p == 0),
                    rng: Rng::new(seed * 4 + p as u64),
                    offers: offers[p].clone(),
                    trade_searches: searched.clone(),
                }) as Box<dyn Bot>
            })
            .collect();
        let (_, _, turns, _) = play_game(seed, trading(), &mut bots);
        assert!(turns > 0);
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

/// Two settlements at nodes the distance rule allows.
fn two_settlements() -> u64 {
    use settler_engine::topology::topo;
    let far = (1..54)
        .find(|&n| topo().node_neighbor_mask[0] & (1u64 << n) == 0)
        .unwrap();
    1 | (1u64 << far)
}

/// The cards `city_offer` deals, as production events, so a belief built from `Belief::new`
/// matches the hand-built position exactly.
fn produced(offerer: PlayerId, ore_holder: PlayerId) -> Vec<Event> {
    vec![
        Event::Produced {
            player: offerer,
            resources: [1, 0, 0, 2, 2],
        },
        Event::Produced {
            player: ore_holder,
            resources: [0, 0, 0, 0, 1],
        },
    ]
}

/// The trading game to 3 VP. `offerer` (two settlements: 2 VP) holds a wood, 2 wheat and 2 ore
/// and has just offered the wood for the one ore its winning city lacks; `ore_holder` holds the
/// only other ore, and nobody else holds anything.
fn city_offer(seed: u64, offerer: PlayerId, ore_holder: PlayerId) -> Game {
    city_offer_to(seed, offerer, ore_holder, 3)
}

/// `city_offer` in a game to `vp_to_win`: at 4 the city does not win.
fn city_offer_to(seed: u64, offerer: PlayerId, ore_holder: PlayerId, vp_to_win: u8) -> Game {
    let config = GameConfig {
        vp_to_win,
        ..trading()
    };
    let mut s = State::new(seed, config);
    s.setup_step = 8;
    s.phase = Phase::Main;
    s.current = offerer;
    s.players[offerer as usize].settlements = two_settlements();
    common::give(&mut s, offerer as usize, [1, 0, 0, 2, 2]);
    common::give(&mut s, ore_holder as usize, [0, 0, 0, 0, 1]);
    let mut g = Game::from_state(s);
    g.apply(Action::OfferTrade { give: 0, get: 4 }).unwrap();
    g
}

#[test]
fn rejects_an_offer_that_hands_the_offerer_the_win() {
    for seed in 0..4 {
        let g = city_offer(seed, 1, 2);
        assert_eq!(
            (g.state().phase, g.state().current_actor()),
            (Phase::TradeResponse, 2)
        );
        let legal = g.legal_actions();
        assert!(legal.contains(&Action::AcceptTrade));
        let mut bot = IsmctsBot::new(seed, 1000, 0);
        bot.observe(2, &[produced(1, 2), g.log_for(2)].concat());
        assert_eq!(
            bot.act(&g.observation(2), &legal),
            Action::RejectTrade,
            "seed {seed}"
        );
        assert!(bot.take_last_search().is_some(), "the answer was searched");
        assert_eq!(resets(&bot), 0);
    }
}

/// The control for `rejects_an_offer_that_hands_the_offerer_the_win` (the heuristic prior
/// favours rejecting either way): the search visits accepting less when the city wins the game
/// than in the same position to 4 VP, where it does not.
#[test]
fn accepting_is_searched_less_when_it_hands_the_offerer_the_win() {
    let accept_visits = |seed: u64, vp_to_win: u8| {
        let g = city_offer_to(seed, 1, 2, vp_to_win);
        let legal = g.legal_actions();
        let mut bot = IsmctsBot::new(seed, 1000, 0);
        bot.observe(2, &[produced(1, 2), g.log_for(2)].concat());
        bot.act(&g.observation(2), &legal);
        let r = bot.take_last_search().expect("the answer was searched");
        assert_eq!(resets(&bot), 0);
        let accept = legal
            .iter()
            .position(|&a| a == Action::AcceptTrade)
            .unwrap();
        r.visits[accept]
    };
    for seed in 0..4 {
        let (wins, control) = (accept_visits(seed, 3), accept_visits(seed, 4));
        assert!(control > 0, "seed {seed}: accepting was never searched");
        assert!(wins < control, "seed {seed}: {wins} vs {control} visits");
    }
}

/// An acceptor cannot build off its turn, so the trade that wins on the spot is the offerer's:
/// confirming completes the city. The heuristic prior favours cancelling; the search overrules it.
#[test]
fn confirms_a_trade_that_wins_the_game_on_the_spot() {
    for seed in 0..4 {
        let mut g = city_offer(seed, 0, 1);
        g.apply(Action::AcceptTrade).unwrap(); // player 1
        g.apply(Action::RejectTrade).unwrap(); // player 2
        g.apply(Action::RejectTrade).unwrap(); // player 3
        assert_eq!(
            (g.state().phase, g.state().current_actor()),
            (Phase::TradeConfirm, 0)
        );
        let legal = g.legal_actions();
        assert_eq!(legal, vec![Action::ConfirmTrade(1), Action::CancelTrade]);
        let mut bot = IsmctsBot::new(seed, 1000, 0);
        bot.observe(0, &[produced(0, 1), g.log_for(0)].concat());
        assert_eq!(
            bot.act(&g.observation(0), &legal),
            Action::ConfirmTrade(1),
            "seed {seed}"
        );
        assert_eq!(resets(&bot), 0);
    }
}
