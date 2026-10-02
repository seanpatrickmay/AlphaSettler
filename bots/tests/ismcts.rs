mod common;

use settler_bots::arena::{play_game, run_match};
use settler_bots::ismcts::{parse_name, IsmctsBot};
use settler_bots::{make_bot, Bot, BOT_NAMES};
use settler_engine::rng::Rng;
use settler_engine::*;

fn no_trades() -> GameConfig {
    GameConfig { max_offers_per_turn: 0, ..GameConfig::default() }
}

#[test]
fn names_parse_and_bad_ones_are_rejected() {
    assert!(BOT_NAMES.contains(&"ismcts"));
    assert_eq!(parse_name("ismcts"), Some((1000, 0)));
    assert_eq!(parse_name("ismcts@1"), Some((1, 0)));
    assert_eq!(parse_name("ismcts@1000000"), Some((1_000_000, 0)));
    assert_eq!(parse_name("ismcts@300+r8"), Some((300, 8)));
    for bad in ["ismcts@0", "ismcts@", "ismcts@abc", "ismcts@+5", "ismcts@1000001", "ismcts@300+r0",
                "ismcts@300+r201", "ismcts@300+r", "ismctsx", "ismcts@300/r8", "ismcts+r8"] {
        assert_eq!(parse_name(bad), None, "{bad}");
        assert!(make_bot(bad, 0).is_none(), "{bad}");
    }
    assert_eq!(make_bot("ismcts@50", 0).unwrap().name(), "ismcts");
}

#[test]
fn plays_whole_games_without_belief_resets() {
    for seed in 0..2u64 {
        let mut bots: Vec<Box<dyn Bot>> =
            (0..4u64).map(|p| Box::new(IsmctsBot::new(seed * 4 + p, 20, 0)) as Box<dyn Bot>).collect();
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
    let wins = records.iter().filter(|r| r.winner == Some(r.candidate_seat)).count();
    assert!(wins as f64 / records.len() as f64 > 0.8, "{wins}/{}", records.len());
}

#[test]
fn answers_offers_without_searching() {
    let mut g = Game::new(2, GameConfig::default());
    let mut rng = Rng::new(2);
    let (mut rejected, mut cancelled) = (false, false);
    while !g.state().is_over() && !(rejected && cancelled) {
        let s = *g.state();
        let legal = g.legal_actions();
        let actor = s.current_actor();
        let mut bot = IsmctsBot::new(0, 50, 0);
        if s.phase == Phase::TradeResponse {
            assert_eq!(bot.act(&s.observation(actor), &legal), Action::RejectTrade);
            assert!(bot.take_last_search().is_none(), "answered an offer by searching");
            rejected = true;
        }
        if s.phase == Phase::TradeConfirm {
            assert_eq!(bot.act(&s.observation(actor), &legal), Action::CancelTrade);
            assert!(bot.take_last_search().is_none(), "answered an offer by searching");
            cancelled = true;
        }
        let pick = if s.phase == Phase::TradeResponse && legal.contains(&Action::AcceptTrade) {
            Action::AcceptTrade
        } else {
            legal[rng.below(legal.len() as u32) as usize]
        };
        g.apply(pick).unwrap();
    }
    assert!(rejected && cancelled, "random play with trades reaches both trade phases");
}

fn resets(bot: &IsmctsBot) -> u64 {
    let d: std::collections::HashMap<_, _> = bot.diagnostics().into_iter().collect();
    d["belief_resets"]
}

/// Random moves from `g` until player 0, past setup and turn `min_turn`, must make a decision
/// that is searched (`searched`: more than one searchable move, not a trade answer) or not.
fn advance(g: &mut Game, rng: &mut Rng, min_turn: u32, searched: bool) {
    let mut moves = Vec::new();
    loop {
        let s = *g.state();
        assert!(!s.is_over(), "the game ended before the wanted decision");
        let legal = g.legal_actions();
        settler_search::search_actions(&legal, &mut moves);
        let ready = s.turn >= min_turn
            && s.current_actor() == 0
            && !matches!(s.phase, Phase::SetupSettlement | Phase::SetupRoad { .. } | Phase::TradeResponse | Phase::TradeConfirm);
        if ready && (moves.len() > 1) == searched {
            return;
        }
        g.apply(legal[rng.below(legal.len() as u32) as usize]).unwrap();
    }
}

/// Events no history can explain: player 0 discards one more wood than they hold.
fn contradiction(obs: &Observation) -> Vec<Event> {
    let wood = obs.my_hand[Resource::Wood.index()] as usize;
    vec![Event::Discarded { player: 0, resource: Resource::Wood }; wood + 1]
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
        assert_eq!(resets(&bot), expected, "inject {inject}: counted at the error");
        act_and_apply(&mut bot, &mut g);
        assert!(bot.take_last_search().is_some());
        assert_eq!(resets(&bot), expected, "inject {inject}: the rebuild is not counted again");
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
    assert!(bot.take_last_search().is_none(), "a single move is not searched");
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
        assert!(matches!(decide(&mut bot, &s), Action::BuildCity(_)), "seed {seed}");
    }
}

#[test]
fn never_robs_its_own_buildings_when_an_opponent_tile_is_free() {
    use settler_engine::topology::topo;
    for seed in 0..4 {
        let mut s = main_with(seed, [0; 5], no_trades());
        let mut snap = s.snapshot();
        let i = snap.dev_deck.iter().position(|&c| c == DevCard::Knight).unwrap();
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
        let Action::MoveRobber(t) = decide(&mut bot, &s) else { panic!("not a robber move") };
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
