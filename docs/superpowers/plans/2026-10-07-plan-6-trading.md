# Plan 6: Domestic Trading (Sub-project 3c) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A bot that offers, answers and confirms player-to-player trades, trained from 3b's `gen-004`, and measured against the spec's three done criteria.

**Architecture:** The engine is untouched. `settler-search` gains `SearchConfig::offers` (offers become searched moves for every seat when set) and samples worlds at trade roots; `IsmctsBot` stops hard-coding its trade answers, searches them, and offers only with a `+t` name suffix. Encoder v3 appends a 24-scalar trade block to the globals (115 instead of 91); `alphasettler net-upgrade` rewrites a v2 net for it with unchanged no-trade moves. Training targets and the network prior include offers, self-play seats offer whenever the game allows it, and `--trades` (the trading game `{"max_offers_per_turn": 2, "max_trade_cards": 1}`) reaches the arena, self-play and the fast loop. The Catanatron oracle translates our offers, Catanatron's answers and our confirmation. The controller then upgrades `gen-004`, runs the trading fast loop and measures the done criteria.

**Tech Stack:** Rust 1.80+ (edition 2021, no new crates), PyO3 0.29.3 + maturin 1.15.0, Python 3.12 stdlib, PyTorch 2.14.1 (`train` extra only), pytest 9.1.1, Catanatron 3.3.0 at `ecf931181b9a65bb4116a2153fb78c16f1438e00` (the `oracle` extra, imported only by `oracle/`).

**Spec:** `docs/superpowers/specs/2026-10-07-trading-design.md` (binding). Builds on `docs/superpowers/plans/2026-10-04-plan-5b-network-v2.md` (network v2, weights format 2) and commit `baf67ba` (the `+sK` suffix).

## Global Constraints

- Python is `.venv/bin/python3`, tests `.venv/bin/pytest`, the CLI `.venv/bin/alphasettler`. Never `python`, `pip`, `python3` outside the venv, or `pip3` outside the venv.
- Rust is `~/.cargo/bin/cargo` (cargo is not on PATH). MSRV 1.80 (`rust-version = "1.80"` in every crate): use nothing stabilised after 1.80. No new external crates and no new Python dependencies. No numpy anywhere.
- Rebuild the extension after any Rust change used from Python. First check `pgrep -fl alphasettler`. If it prints nothing, run `PATH=$HOME/.cargo/bin:$PATH .venv/bin/maturin develop --release -q`. If it prints a process (a long arena or loop has the extension loaded), never overwrite the loaded file in place; build a wheel and rename the new library over the old one (an atomic rename, so running processes keep their mapped copy):
  ```bash
  rm -rf target/wheels-3c && PATH=$HOME/.cargo/bin:$PATH .venv/bin/maturin build --release -q -o target/wheels-3c \
    && unzip -q -o target/wheels-3c/alphasettler-*.whl 'alphasettler/_engine*.so' -d target/wheels-3c/x \
    && mv -f target/wheels-3c/x/alphasettler/_engine.cpython-312-darwin.so alphasettler/_engine.cpython-312-darwin.so
  ```
  "Rebuild the extension" in any step below means exactly this rule.
- Every task ends with `~/.cargo/bin/cargo fmt --all` then `~/.cargo/bin/cargo fmt --check` (no output), `~/.cargo/bin/cargo clippy --workspace --release --all-targets -- -D warnings` (exit 0), the full `~/.cargo/bin/cargo test --workspace --release` (all pass) and, after the rebuild, `.venv/bin/pytest -q -rs` (all pass, the summary lists no skipped tests).
- The engine does not change: no file under `engine/` is edited, and `engine/tests/golden_trace.rs` passes without re-pinning (spec Section 2).
- 3a's golden digests stay as they are, never re-pinned: `the_generic_refactor_leaves_3a_ismcts_games_unchanged` (`bots/tests/ismcts.rs`) and `fixed_exploration_is_3a_self_play` (`bots/tests/selfplay.rs`) (spec Section 3).
- The trading game is exactly `{"max_offers_per_turn": 2, "max_trade_cards": 1}` (spec Section 2), named `TRADES` in `alphasettler/arena.py` and `trading()` in Rust tests.
- `+t` lets a searching bot (`ismcts…`, `net:…`) offer; without it a bot answers and confirms by searching but never offers, and assumes nobody does. `+t` combines with `+rD` and `+sK` in any order (spec Section 3).
- Encoder v3: `NET_GLOBAL_WIDTH` 115, `PUBLIC_LEN` 1581, `ENCODER_VERSION` 3, a new `ENCODING_HASH` (spec Section 4). Weights format stays 2. Library code reads block offsets from the encoder's constants, never as literals.
- Encoder v2's hash is `0x28dd1f780ff2a1b3` (read from `runs/3b-v2/nets/gen-004.bin`'s header and from the build before Task 2). The v3 loaders refuse a v2 file; only `net-upgrade` reads one. `OFFER_BIAS = -3.0` (spec Section 4).
- Determinism: same seed and weights give the same games, independent of thread count. Budgets are simulation counts, never wall time. Only `oracle/` imports Catanatron, and Catanatron's installed source is never edited. No module outside `alphasettler/train/` imports torch (`alphasettler/netupgrade.py` uses `struct` and `array` only).
- Measurement seeds start at 2,000,000,000 (`--seed-start 2000000000`): 200 seeds, four seat rotations each, never used by training or gates (spec Section 1).
- Git: one commit per task with the message given in the task, run as `git -c user.email=sean.may101@gmail.com commit -m "..."`. No Co-Authored-By or AI attribution, never `--no-verify`, never push. Before every commit run `git diff --cached | grep -nE 'sk-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{36}|password=[^[:space:]]+'` and expect no output.
- Long runs (the trading loop, every 200-seed arena) are run by the controller in the background, never by implementer subagents. Steps marked **controller runs this** give the exact command and the pass condition.
- No README or docs beyond `docs/bot/results-3c.md` (Task 6).

## Review Focus

1. An encoder v2 net reaching the v3 build: `net:gen-004 --nets-dir runs/3b-v2/nets`, a resumed `runs/fast-loop`, `fast-train --init` on a 3b file. Expected: it is refused before any game or record is read, with an error that names `alphasettler net-upgrade`. Pinned by Task 3's `an_encoder_v2_file_is_refused_with_the_way_to_upgrade_it` and `test_loaders_send_an_encoder_v2_net_to_net_upgrade`.
2. A decision at a trade root with a belief rebuilt from the observation alone (no history, a reset, a bot driven from Python). Expected: sampled hands still give the offerer what it offers and every acceptor what it was asked for, so `apply_confirm` never wraps a `u8`. Pinned by Task 1's `a_belief_from_the_observation_alone_covers_the_pending_trade`, `answers_and_confirmations_are_searched` and the Python `test_offer_suffixes_build_net_bots_in_any_order`.
3. Bot names with suffixes repeated, malformed or out of order (`+t+t`, `+t5`, `+T`, `net:a+t@600`, `+s3+t+r2`). Expected: equivalent orders build the same bot, and bad names are refused as unknown bots before any file is read. Pinned by Task 1's `the_offer_suffix_combines_with_the_others_in_any_order` and the netbot name test.
4. A fast loop resumed with a different `--trades` setting from the one it was set up with. Expected: the loop keeps its own setting, and asking for trades on a loop set up without them is refused. Pinned by Task 4's `test_cli_fast_loop_keeps_the_trading_setting_it_was_set_up_with`.
5. Our bot offering inside Catanatron: AlphaBeta asked an offer it can take (unpatched, it raises), an offer nobody can take, and a third offer in one turn. Expected: AlphaBeta answers by its own search. An unanswered offer emits `trade_cancelled` and returns to the turn. The snapshot counts the turn's offers, so a third is not legal. Pinned by Task 5's three trade tests.

## File Structure

| File | Responsibility | Task |
|---|---|---|
| `search/src/search.rs`, `search/src/lib.rs` | `SearchConfig::offers`, `search_moves`, offers in the tree | 1 |
| `search/src/world.rs` | worlds at trade roots: template from the Main position plus the pending trade | 1 |
| `search/src/belief.rs` | `deal_hidden_hands` deals the pending trade's cards first | 1 |
| `bots/src/ismcts.rs` | searched answers/confirmations, `with_offers`, `split_trades` | 1 |
| `bots/src/lib.rs`, `bots/src/netbot.rs` | `+t` in `make_bot`, `Roster`, `parse_net_name` (5-tuple) | 1 |
| `net/src/encode.rs` | encoder v3, the trade block | 2 |
| `net/src/graph.rs`, `net/src/weights.rs`, `alphasettler/train/model.py` | docs only in Task 2: both forward passes read the global width from the encoder | 2 |
| `net/src/weights.rs`, `bindings/src/train.rs`, `alphasettler/train/model.py` | `V2_ENCODING_HASH`, `V2_GLOBAL_WIDTH`, the upgrade hint in both loaders | 3 |
| `alphasettler/netupgrade.py` (new), `alphasettler/cli.py` | `net-upgrade OLD NEW` | 3 |
| `net/src/data.rs`, `net/src/evaluator.rs` | offers in the mask, targets and prior | 4 |
| `bots/src/selfplay.rs` | self-play seats offer when the game allows offers | 4 |
| `alphasettler/arena.py`, `alphasettler/cli.py`, `alphasettler/train/fastloop.py` | `TRADES`, `--trades` on arena / selfplay / fast-loop | 4 |
| `oracle/translate.py`, `oracle/events.py`, `oracle/arena.py`, `alphasettler/cli.py` | Catanatron trade translation, the Catanatron arena `--trades` | 5 |
| `docs/bot/results-3c.md` (new) | the measurements | 6 |
| tests: `search/tests/{world,search}.rs`, `bots/tests/{ismcts,netbot,selfplay}.rs`, `net/tests/{encode,graph,weights,data,evaluator}.rs`, `tests/python/{test_netbot,test_train,test_net_upgrade (new),test_cli,test_fast_loop,test_oracle_trades (new)}.py`, `tests/python/net_names.py` | | 1–5 |

## Where the code forces a choice the spec does not make

- **The engine refuses snapshots with a pending trade** (`State::from_snapshot`), and the engine must not change. Two places build positions from snapshots. `WorldSampler` therefore builds its template at the Main position the offer was made from, then sets `phase` and `trade` from the (public) observation (Task 1). The oracle imports the pre-offer position, then replays the offer and every answer through our rules (`our_game`, Task 5).
- **`Belief::from_observation` ignored the pending trade**, so a rebuilt belief could deal an acceptor too few cards. It now deals the offerer's give and each acceptor's get first. This changes nothing when no trade is pending (same RNG draws).
- **`masked_prior` gave every offer probability 0** (3b). Task 4 gives offers their own logits like any move, in the same commit that puts offers in the training mask, so the prior and the targets keep sharing slots.
- **Task boundaries:** self-play's offer setting moves to Task 4 (with recording), not Task 1. The masked prior is part of "the network's view of offers" (Task 4), not search. Tasks 2 and 3 are separate because Task 2 alone leaves every 3b net refused, which is the state Task 3's hint addresses.
- **The upgrade zeroes all 400 offer rows** (bias −3.0), not only the 20 one-for-one ids. The reason the spec gives (untrained noise) holds for all 400, and only the 20 are legal in the trading game, so the measured behaviour is identical. `CancelTrade` (664) counts as a confirmation id (bias 0): gen-4 never searched a confirmation either.
- **"Accepts an offer that wins it the game on the spot":** an acceptor cannot build off its turn, so no answer wins on the spot. The test is the offerer confirming an accepted trade that completes a winning city (the prior favours cancelling, so the search must overrule it), plus the spec's rejection test.
- **AlphaBeta cannot answer an offer in Catanatron 3.3.** Its `execute_spectrum` raises `RuntimeError("Unknown ActionType")` on `ACCEPT_TRADE` and `REJECT_TRADE`, because Catanatron's bots never offer. The oracle adds the four trade-answer action types (deterministic: no dice, no hidden draw) to `catanatron.players.tree_search_utils.DETERMINISTIC_ACTIONS` at run time when trades are on. AlphaBeta then answers by its own depth-2 search, as the spec intends. No Catanatron file is edited.
- **Catanatron asks responders in seat-index order**, ours in seat order after the offerer. The oracle forwards events in Catanatron's order (the belief does not depend on the order) and replays answers in ours.
- **Criterion 3's baseline `net:gen-004` is an encoder v2 file**, which the v3 build refuses. The baseline is the upgraded gen-004. Its no-trade moves are the original's up to float summation order.
- **`selfplay --trades`** used to mean the engine's default config with ismcts never offering. It now means the trading game, and every seat may offer.
- **Generation count:** the spec does not fix one. The trading loop runs 10 generations (30–39), then the criteria are measured once.

---

### Task 1: Search and bots — searched trade answers, offers with `+t`, worlds at trade roots

**Files:**
- Modify: `search/src/search.rs`, `search/src/lib.rs`, `search/src/world.rs`, `search/src/belief.rs`, `bots/src/ismcts.rs`, `bots/src/lib.rs`, `bots/src/netbot.rs`, `alphasettler/cli.py` (one error message)
- Test: `search/tests/world.rs`, `search/tests/search.rs`, `bots/tests/ismcts.rs`, `bots/tests/netbot.rs`, `tests/python/test_netbot.py`, `tests/python/net_names.py`

**Interfaces:**
- Consumes: `settler_engine::{Action, Phase, Response, PendingTrade, covers, hand_sub, hand_add, Game, State}`; `Belief::states() -> &[(HandTracker, f64)]` with `HandTracker::hands: [Hand; 4]`.
- Produces:
  - `settler_search::SearchConfig { simulations, c_puct, batch, offers: bool }` (`Default` has `offers: false`).
  - `settler_search::search_moves(legal: &[Action], offers: bool, out: &mut Vec<Action>)`; `search_actions(legal, out)` is `search_moves(legal, false, out)` (unchanged behaviour, still used by rollouts).
  - `WorldSampler::new` accepts `TradeResponse` / `TradeConfirm` observations; it errs only for `GameOver`, or when a belief state does not cover the pending trade.
  - `settler_bots::ismcts::split_trades(name: &str) -> Option<(String, bool)>`; `IsmctsBot::with_offers(self, offers: bool) -> IsmctsBot<E>`.
  - `settler_bots::netbot::parse_net_name(name) -> Option<(&str, u32, u32, u32, bool)>` (net, simulations, rollout, setup factor, offers).

- [ ] **Step 1: Write the failing tests**

In `search/tests/world.rs`, delete the test `a_pending_trade_cannot_be_sampled` (its behaviour is reversed by the spec) and append:

```rust
/// Every viewer's exact belief, advanced through random games with trades on, samples worlds at
/// TradeResponse and TradeConfirm roots that keep the pending trade and hands that cover it:
/// every legal move, a confirmation included, applies to a valid state.
#[test]
fn worlds_sampled_at_a_trade_root_keep_the_trade_and_hands_that_cover_it() {
    let mut roots = 0;
    for seed in 0..6 {
        let mut beliefs: Vec<Belief> = (0..4)
            .map(|p| Belief::new(p, DEFAULT_MAX_STATES, seed))
            .collect();
        let mut seen = [0usize; 4];
        let mut rng = Rng::new(seed);
        random_game(seed, GameConfig::default(), |g| {
            for p in 0..4u8 {
                let log = g.log_for(p);
                beliefs[p as usize]
                    .observe(&log[seen[p as usize]..])
                    .unwrap();
                seen[p as usize] = log.len();
            }
            let s = g.state();
            if s.is_over() || !trade_phase(s.phase) {
                return;
            }
            roots += 1;
            let t = s.trade.unwrap();
            for p in 0..4u8 {
                let obs = g.observation(p);
                let sampler = WorldSampler::new(&obs, &beliefs[p as usize]).unwrap();
                for _ in 0..3 {
                    let w = sampler.sample(&beliefs[p as usize], &mut rng);
                    w.check_invariants().unwrap();
                    assert_eq!(w.observation(p), obs, "seed {seed} viewer {p}");
                    assert_eq!((w.phase, w.trade), (s.phase, s.trade));
                    assert!(covers(&w.players[s.current as usize].hand, &t.give));
                    for q in 0..4 {
                        if t.responses[q] == Response::Accepted {
                            assert!(covers(&w.players[q].hand, &t.get), "acceptor {q}");
                        }
                    }
                    for a in w.legal_actions() {
                        let mut x = w;
                        x.apply(a);
                        x.check_invariants().unwrap();
                    }
                }
            }
        });
    }
    assert!(roots > 20, "{roots} trade roots");
}

/// A belief rebuilt from the observation alone (no history) still deals the offerer what it
/// gives and every acceptor what it asked for, so a world can be sampled at the trade root.
#[test]
fn a_belief_from_the_observation_alone_covers_the_pending_trade() {
    let mut checked = 0;
    random_game(2, GameConfig::default(), |g| {
        let s = g.state();
        if checked >= 40 || s.is_over() || !trade_phase(s.phase) {
            return;
        }
        let t = s.trade.unwrap();
        for p in 0..4u8 {
            let obs = g.observation(p);
            let b = Belief::from_observation(&obs, 256, p as u64);
            for (st, _) in b.states() {
                assert!(covers(&st.hands[s.current as usize], &t.give));
                for q in 0..4 {
                    if t.responses[q] == Response::Accepted {
                        assert!(covers(&st.hands[q], &t.get), "acceptor {q}");
                    }
                }
            }
            assert!(WorldSampler::new(&obs, &b).is_ok());
        }
        checked += 1;
    });
    assert!(checked >= 10, "{checked} trade roots");
}
```

In `search/tests/search.rs`, change the import to

```rust
use settler_search::{
    search, search_moves, search_tree, terminal_value, visible_chance, Belief, Eval, Evaluator,
    Leaf, SearchConfig, UniformEvaluator,
};
```

and append:

```rust
/// Player 0 in Main holding one card of each resource, in a game that allows offers.
fn main_with_cards(seed: u64) -> State {
    let config = GameConfig::default();
    let base = after_setup(seed, config);
    edit(&base, config, |snap| {
        give_cards(snap, 0, [1, 1, 1, 1, 1]);
        snap.phase = Phase::Main;
    })
}

#[test]
fn offers_are_searched_only_when_the_config_allows_them() {
    let s = main_with_cards(7);
    let obs = s.observation(0);
    let legal = s.legal_actions();
    let is_offer = |a: &Action| matches!(a, Action::OfferTrade { .. });
    assert!(legal.iter().any(is_offer), "offers are legal here");
    let mut moves = Vec::new();
    search_moves(&legal, false, &mut moves);
    assert!(!moves.iter().any(is_offer) && moves.len() < legal.len());
    search_moves(&legal, true, &mut moves);
    assert_eq!(moves, legal);
    let b = Belief::from_observation(&obs, 64, 7);
    for offers in [false, true] {
        let config = SearchConfig {
            simulations: 600,
            offers,
            ..SearchConfig::default()
        };
        let (r, tree) = search_tree(
            &obs,
            &legal,
            &b,
            &mut UniformEvaluator,
            &config,
            &mut Rng::new(3),
        )
        .unwrap();
        let offered: u32 = legal
            .iter()
            .zip(&r.visits)
            .filter(|(a, _)| is_offer(*a))
            .map(|(_, &v)| v)
            .sum();
        assert_eq!(offered > 0, offers, "offers {offers}: {offered} visits on offers");
        // Below a searched offer the opponents answer: a decision node with children.
        let answered = tree.nodes[ROOT as usize].children.iter().any(|c| {
            Action::decode(c.key).is_some_and(|a| is_offer(&a))
                && c.visits > 1
                && !tree.nodes[c.node as usize].children.is_empty()
        });
        assert_eq!(answered, offers, "offers {offers}");
    }
}
```

In `bots/tests/ismcts.rs`, replace the imports at the top with

```rust
mod common;

use settler_bots::arena::{belief_resets, play_game, run_match, GameRecord};
use settler_bots::ismcts::{parse_name, IsmctsBot, MoveOptions};
use settler_bots::{make_bot, Bot, BOT_NAMES};
use settler_engine::rng::Rng;
use settler_engine::*;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
```

and add, after `fn no_trades()`:

```rust
/// The trading game (spec 3c Section 2): two offers a turn, one card a side.
fn trading() -> GameConfig {
    GameConfig {
        max_offers_per_turn: 2,
        max_trade_cards: 1,
        ..GameConfig::default()
    }
}
```

Replace the whole test `answers_offers_without_searching` with:

```rust
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
```

Append to `bots/tests/ismcts.rs`:

```rust
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
    assert!(make_bot("greedy+t", 0).is_none(), "only searching bots offer");
    assert!(make_bot("random+t", 0).is_none());
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
                .filter(|(a, _)| is_offer(*a))
                .map(|(_, &v)| v)
                .sum();
            (a, v)
        };
        let (a, v) = offer_visits(false);
        assert!(!is_offer(&a) && v == 0, "seed {seed}: the answer-only bot searched an offer");
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
        assert_eq!(
            belief_resets(bots.iter().map(|b| &**b)),
            0,
            "seed {seed}"
        );
    }
    let made: Vec<u32> = offers.iter().map(|c| c.load(Ordering::Relaxed)).collect();
    assert!(made[0] > 0 && made[1..] == [0, 0, 0], "offers by seat: {made:?}");
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
    let config = GameConfig {
        vp_to_win: 3,
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
```

In `bots/tests/netbot.rs`, replace the four `parse_net_name` asserts at the top of `net_names_parse_and_bad_ones_are_rejected_without_touching_the_disk` with

```rust
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
```

and add `"net:a@300+t+t"`, `"net:a@300+t5"`, `"net:a+t@300"` and `"net:a@300+T"` to its `bad` list. Append:

```rust
#[test]
fn net_bots_with_and_without_the_offer_suffix_play_the_trading_game() {
    let dir = nets_dir("trading");
    let trading = GameConfig {
        max_offers_per_turn: 2,
        max_trade_cards: 1,
        ..GameConfig::default()
    };
    let records = run_match_in("net:tiny@10+t", "net:tiny@10", 0..2, trading, 2, &dir).unwrap();
    assert_eq!(records.len(), 8);
    assert!(records.iter().all(|r| r.belief_resets == 0), "{records:?}");
}
```

In `tests/python/net_names.py`, add `"net:a@300+t+t"` and `"net:a+t@300"` to `BAD_NET_NAMES`. In `tests/python/test_netbot.py`, append:

```python
def test_offer_suffixes_build_net_bots_in_any_order(nets):
    for name in ("net:tiny@10+t", "net:tiny@10+s2+t", "net:tiny@10+t+r3+s2"):
        assert Bot(name, 1, nets_dir=str(nets)).name == "net", name
    # Bots driven from Python see no events: every decision rebuilds the belief from the
    # observation, trade roots included.
    g = Game(4, {"max_offers_per_turn": 2, "max_trade_cards": 1})
    bot = Bot("net:tiny@10+t", 1, nets_dir=str(nets))
    for _ in range(200):
        if g.is_over:
            break
        a = bot.act(g)
        assert a in g.legal_actions()
        g.apply(a)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `~/.cargo/bin/cargo test --release -p settler-search --test world --test search`
Expected: compile error, `cannot find function search_moves` / `no field offers on type SearchConfig`.
Run: `~/.cargo/bin/cargo test --release -p settler-bots --test ismcts --test netbot`
Expected: compile errors naming `with_offers`, `split_trades` and the 5-tuple.

- [ ] **Step 3: Offers in the search**

In `search/src/search.rs`, replace `SearchConfig`, its `Default`, and `search_actions` with:

```rust
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SearchConfig {
    /// Simulations per decision: the budget (never wall time).
    pub simulations: u32,
    pub c_puct: f32,
    /// Leaves per `Evaluator` call.
    pub batch: usize,
    /// Domestic offers are searched moves for every seat in the tree: a bot that may offer
    /// assumes its opponents may too. Off, no seat offers in the tree (3a and 3b never offered).
    pub offers: bool,
}

impl Default for SearchConfig {
    fn default() -> Self {
        SearchConfig {
            simulations: 1000,
            c_puct: 1.5,
            batch: 8,
            offers: false,
        }
    }
}
```

```rust
/// The actions the search considers: everything legal, with domestic offers only when `offers`.
pub fn search_moves(legal: &[Action], offers: bool, out: &mut Vec<Action>) {
    out.clear();
    out.extend(
        legal
            .iter()
            .copied()
            .filter(|a| offers || !matches!(a, Action::OfferTrade { .. })),
    );
}

/// `search_moves` without offers: what a bot that never offers searches, and what rollouts play.
pub fn search_actions(legal: &[Action], out: &mut Vec<Action>) {
    search_moves(legal, false, out);
}
```

Change `descend` to take the config instead of `c_puct` (the argument count stays at 7, under clippy's limit):

```rust
fn descend(
    tree: &mut Tree,
    mut world: State,
    viewer: PlayerId,
    cfg: &SearchConfig,
    rng: &mut Rng,
    buf: &mut Vec<Action>,
    legal: &mut Vec<Action>,
) -> Descent {
```

Inside it, replace `search_actions(buf, legal);` with `search_moves(buf, cfg.offers, legal);` and the selection line with `let i = puct::select(&n.children, |c| allowed.contains(c.key), actor, cfg.c_puct)`. In `search_tree`, replace `search_actions(legal, &mut root_actions);` with `search_moves(legal, cfg.offers, &mut root_actions);` and the call with:

```rust
            match descend(
                &mut tree,
                world,
                obs.viewer,
                cfg,
                rng,
                &mut buf,
                &mut scratch,
            ) {
```

In `search/src/lib.rs` replace the `search::` re-export with:

```rust
pub use search::{
    search, search_actions, search_moves, search_tree, terminal_value, visible_chance,
    SearchConfig, SearchResult,
};
```

- [ ] **Step 4: Worlds at trade roots, and a rebuilt belief that covers the trade**

In `search/src/belief.rs`, change the engine import to `use settler_engine::{Event, Observation, Phase, Response};` and replace `deal_hidden_hands` with:

```rust
/// Opponents' hands dealt uniformly from the cards neither the bank nor the viewer holds, each
/// opponent getting their visible count. A pending trade is evidence everyone saw: the offerer
/// holds what it gives and every acceptor what it was asked for, so those cards are dealt first.
/// With no trade pending the draws are exactly as before.
fn deal_hidden_hands(obs: &Observation, rng: &mut Rng) -> [Hand; NUM_PLAYERS] {
    let me = obs.viewer as usize;
    let mut pool: Hand = std::array::from_fn(|r| {
        BANK_PER_RESOURCE
            .saturating_sub(obs.bank[r])
            .saturating_sub(obs.my_hand[r])
    });
    let mut must = [[0u8; NUM_RESOURCES]; NUM_PLAYERS];
    if let Some(t) = obs.trade {
        must[obs.current as usize] = t.give;
        for q in 0..NUM_PLAYERS {
            if t.responses[q] == Response::Accepted {
                must[q] = t.get;
            }
        }
    }
    must[me] = [0; NUM_RESOURCES];
    for q in 0..NUM_PLAYERS {
        let capped: Hand = std::array::from_fn(|r| must[q][r].min(pool[r]));
        must[q] = capped;
        hand_sub(&mut pool, &must[q]);
    }
    let mut hands = [[0u8; NUM_RESOURCES]; NUM_PLAYERS];
    hands[me] = obs.my_hand;
    for q in (0..NUM_PLAYERS).filter(|&q| q != me) {
        let k = (obs.hand_counts[q] as u32)
            .saturating_sub(hand_total(&must[q]))
            .min(hand_total(&pool)) as u8;
        hands[q] = sample_cards(&pool, k, rng);
        hand_sub(&mut pool, &hands[q]);
        hand_add(&mut hands[q], &must[q]);
    }
    hands
}
```

In `search/src/world.rs`, update the module doc's last sentence to add: "A pending domestic trade (public) is carried as it is; the template is built at the Main position the offer was made from, because a snapshot cannot hold one." Change the import to `use settler_engine::{Observation, Phase, PlayerSnapshot, Response, Snapshot, State};`. In `WorldSampler::new`, replace

```rust
        if matches!(
            obs.phase,
            Phase::TradeResponse | Phase::TradeConfirm | Phase::GameOver { .. }
        ) {
            return Err(format!("no worlds to sample in phase {:?}", obs.phase));
        }
```

with

```rust
        if matches!(obs.phase, Phase::GameOver { .. }) {
            return Err(format!("no worlds to sample in phase {:?}", obs.phase));
        }
        // Every world must be able to play the pending trade out: the offerer holds what it
        // gives, each acceptor what it was asked for. The belief's events guarantee it; a state
        // that breaks it would wrap a hand below zero on a confirmation.
        if let Some(t) = obs.trade {
            let offerer = obs.current as usize;
            let uncovered = belief.states().iter().any(|(s, _)| {
                !covers(&s.hands[offerer], &t.give)
                    || (0..NUM_PLAYERS).any(|q| {
                        t.responses[q] == Response::Accepted && !covers(&s.hands[q], &t.get)
                    })
            });
            if uncovered {
                return Err("a belief state does not cover the pending trade".into());
            }
        }
```

In the `Snapshot` literal, replace `phase: obs.phase,` with

```rust
            // A snapshot cannot hold a pending trade: build the Main position it was made from.
            phase: if obs.trade.is_some() {
                Phase::Main
            } else {
                obs.phase
            },
```

and replace `sampler.template = State::from_snapshot(0, obs.config, &snap)?;` with

```rust
        sampler.template = State::from_snapshot(0, obs.config, &snap)?;
        sampler.template.phase = obs.phase;
        sampler.template.trade = obs.trade;
```

- [ ] **Step 5: The bots: searched answers, `+t`, any suffix order**

In `bots/src/ismcts.rs`:

Replace the module doc with:

```rust
//! IsmctsBot: tracks hidden cards from the event log and searches each real decision with
//! single-observer ISMCTS (spec Sections 2-3), domestic-trade answers and confirmations included
//! (spec 3c Section 3). It makes offers only with `with_offers` (a `+t` name). Generic over the
//! evaluator: the heuristic for `ismcts` names, the network for `net:` names (`crate::netbot`).
```

Replace the `settler_search` import with:

```rust
use settler_search::{
    search, search_moves, Belief, Evaluator, Noised, RootNoise, SearchConfig, SearchResult,
    DEFAULT_MAX_STATES,
};
```

After `split_setup`, add:

```rust
/// A searching bot's name with its `+t` suffix removed and its `+rD` / `+sK` suffixes put in the
/// order `parse_name` and `split_setup` read (`+rD` before `+sK`), and whether it had `+t`. The
/// suffixes are everything from the first `+`, in any order, each kind at most once; `None` for a
/// repeated or unknown one (`+t5`, `+T`, a bare `+`). Their values are checked by those parsers.
pub fn split_trades(name: &str) -> Option<(String, bool)> {
    let Some(i) = name.find('+') else {
        return Some((name.to_owned(), false));
    };
    let (mut rollout, mut setup, mut offers) = (None, None, false);
    for part in name[i + 1..].split('+') {
        match part.as_bytes().first() {
            Some(b'r') if rollout.is_none() => rollout = Some(part),
            Some(b's') if setup.is_none() => setup = Some(part),
            Some(b't') if part.len() == 1 && !offers => offers = true,
            _ => return None,
        }
    }
    let mut out = name[..i].to_owned();
    for part in [rollout, setup].into_iter().flatten() {
        out.push('+');
        out.push_str(part);
    }
    Some((out, offers))
}
```

After `with_setup_factor`, add:

```rust
    /// Make domestic offers searched moves (a `+t` name). Without it the bot still answers and
    /// confirms offers by searching, but never offers, and assumes nobody does.
    pub fn with_offers(mut self, offers: bool) -> IsmctsBot<E> {
        self.cfg.offers = offers;
        self
    }
```

In `act_with`, change the doc's second sentence to "A decision that does not search (a single searchable move) ignores `opts`." and replace

```rust
        match obs.phase {
            Phase::TradeResponse if legal.contains(&Action::RejectTrade) => {
                return Action::RejectTrade
            }
            Phase::TradeConfirm if legal.contains(&Action::CancelTrade) => {
                return Action::CancelTrade
            }
            _ => {}
        }
        search_actions(legal, &mut self.buf);
```

with

```rust
        search_moves(legal, self.cfg.offers, &mut self.buf);
```

In `bots/src/lib.rs`, replace `make_bot`'s doc and fallback arm, `unknown_bot`, and the two `parse_net_name` uses:

```rust
/// A fresh bot by name, with its own random stream seeded from `seed`. Besides `BOT_NAMES`,
/// `ismcts@N` (N simulations) and `ismcts@N+rD` (plus D greedy rollout moves per leaf), either
/// with `+sK` to search setup settlements with K times the simulations and `+t` to make domestic
/// offers, the suffixes in any order. `net:` names need their weights loaded first: use a `Roster`.
pub fn make_bot(name: &str, seed: u64) -> Option<Box<dyn Bot>> {
    match name {
        "random" => Some(Box::new(random::RandomBot::new(seed))),
        "greedy" => Some(Box::new(greedy::GreedyBot::new(seed))),
        _ => {
            let (name, offers) = ismcts::split_trades(name)?;
            let (base, factor) = ismcts::split_setup(&name)?;
            let (sims, rollout) = ismcts::parse_name(base)?;
            Some(Box::new(
                ismcts::IsmctsBot::new(seed, sims, rollout)
                    .with_setup_factor(factor)
                    .with_offers(offers),
            ))
        }
    }
}

/// The error for a name no bot answers to.
pub fn unknown_bot(name: &str) -> String {
    format!(
        "unknown bot {name:?}; known bots: {BOT_NAMES:?} (ismcts@N for N simulations, ismcts@N+rD \
         to add D greedy rollout moves, net:NAME[@N[+rD]] for <nets-dir>/NAME.bin; +sK on either \
         searches setup settlements with K times the simulations, +t lets either make domestic \
         offers)"
    )
}
```

In `Roster::new`: `if let Some((net, ..)) = netbot::parse_net_name(name) {`. In `Roster::make`:

```rust
        match netbot::parse_net_name(name) {
            Some((net, simulations, rollout, factor, offers)) => {
                let (_, w) = self.nets.iter().find(|(n, _)| n == net)?;
                Some(Box::new(
                    netbot::net_bot(seed, w.clone(), simulations, rollout)
                        .with_setup_factor(factor)
                        .with_offers(offers),
                ))
            }
            None => make_bot(name, seed),
        }
```

In `bots/src/netbot.rs`, change the module doc's first line to "Network bots (spec Section 3): `net:NAME` (1,000 simulations), `net:NAME@N` and `net:NAME@N+rD`, each optionally with `+sK` and `+t` in any order," and replace `parse_net_name`:

```rust
/// `net:NAME` -> (NAME, 1000, 0, 1, false); `net:NAME@N` -> (NAME, N, 0, 1, false);
/// `net:NAME@N+rD` -> (NAME, N, D, 1, false). `+sK` sets the setup factor to K and `+t` the last
/// field (the bot may offer); the suffixes come in any order (`ismcts::split_trades`).
pub fn parse_net_name(name: &str) -> Option<(&str, u32, u32, u32, bool)> {
    let (canon, offers) = ismcts::split_trades(name)?;
    let (canon, factor) = ismcts::split_setup(&canon)?;
    let rest = name.strip_prefix("net:")?;
    let end = rest.find(['@', '+']).unwrap_or(rest.len());
    let net = &rest[..end];
    if !valid_net_name(net) {
        return None;
    }
    let tail = &canon["net:".len() + net.len()..];
    let (simulations, rollout) = ismcts::parse_name(&format!("ismcts{tail}"))?;
    Some((net, simulations, rollout, factor, offers))
}
```

In `alphasettler/cli.py`, in `_check_bot_name`, change the re-raised message's parenthesis to `(or ismcts@N, ismcts@N+rD, net:NAME[@N[+rD]], either with +sK and +t)`.

- [ ] **Step 6: Run the tests to verify they pass**

Run: `~/.cargo/bin/cargo test --release -p settler-search`
Expected: all pass, including `worlds_sampled_at_a_trade_root_keep_the_trade_and_hands_that_cover_it`, `a_belief_from_the_observation_alone_covers_the_pending_trade`, `offers_are_searched_only_when_the_config_allows_them`.
Run: `~/.cargo/bin/cargo test --release -p settler-bots`
Expected: all pass, including the six new or replaced `ismcts` tests and `net_bots_with_and_without_the_offer_suffix_play_the_trading_game`.

- [ ] **Step 7: The 3a digests are unchanged**

Run: `~/.cargo/bin/cargo test --release -p settler-bots --test ismcts the_generic_refactor_leaves_3a_ismcts_games_unchanged && ~/.cargo/bin/cargo test --release -p settler-bots --test selfplay fixed_exploration_is_3a_self_play`
Expected: both `1 passed`, with the pinned constants untouched (`git diff bots/tests/selfplay.rs` shows nothing; in `bots/tests/ismcts.rs` the lines `0x30461dca2efcbb79` and `0xce9e1db7dc5ee013` are unchanged).

- [ ] **Step 8: Rebuild, format, lint, full checks**

Rebuild the extension (Global Constraints). Then run the Global Constraints' end-of-task checks.
Expected: fmt prints nothing; clippy exits 0; every Rust test passes (`golden_trace` included); `.venv/bin/pytest -q -rs` all pass, none skipped, `test_offer_suffixes_build_net_bots_in_any_order` among them.

- [ ] **Step 9: Commit**

```bash
git add search/src/search.rs search/src/lib.rs search/src/world.rs search/src/belief.rs bots/src/ismcts.rs bots/src/lib.rs bots/src/netbot.rs alphasettler/cli.py search/tests/world.rs search/tests/search.rs bots/tests/ismcts.rs bots/tests/netbot.rs tests/python/test_netbot.py tests/python/net_names.py
git diff --cached | grep -nE 'sk-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{36}|password=[^[:space:]]+'
git -c user.email=sean.may101@gmail.com commit -m "search, bots: trade answers and confirmations are searched, offers with +t, worlds sampled at trade roots"
```

---

### Task 2: Encoder v3 — the trade block

**Files:**
- Modify: `net/src/encode.rs`; docs only in `net/src/graph.rs`, `net/src/weights.rs`, `alphasettler/train/model.py`
- Test: `net/tests/encode.rs`, `net/tests/graph.rs`, `net/tests/weights.rs`, `tests/python/test_train.py`

Both forward passes read the global block as `GLOBAL_BASE..PUBLIC_LEN` (`graph::GLOBAL_WIDTH` in Rust, `NET_GLOBAL_WIDTH` from the bindings in `model.py`). So appending the block to the encoder widens the global encoder in both, and in the weights file's layer shapes, with no code change. Only their docs say "91". The Rust/PyTorch parity test (`test_rust_and_pytorch_forward_passes_agree_on_random_weights`) runs on random positions from games with trades on, so it covers the new block.

**Interfaces:**
- Consumes: `settler_engine::Response`, `Observation::{trade, offers_this_turn, config}`.
- Produces: `settler_net::encode::{TRADE_GIVE 1557, TRADE_GET 1562, TRADE_RESPONSES 1567, RESPONSE_WIDTH 3, OFFERS_MADE 1579, OFFERS_ALLOWED 1580}`, `PUBLIC_LEN` 1581, `ENCODER_VERSION` 3; `graph::GLOBAL_WIDTH` 115 and `NET_GLOBAL_WIDTH` 115 follow. A v2 file (hash `0x28dd1f780ff2a1b3`) fails to load with the existing "encoding hash … does not match" error (Task 3 adds the hint).

- [ ] **Step 1: Record encoder v2's hash before anything changes**

Run: `.venv/bin/python3 -c "from alphasettler._engine import NET_ENCODING_HASH as h; print(hex(h))"`
Expected: `0x28dd1f780ff2a1b3`. If it prints anything else, stop and report it: Task 3's `V2_ENCODING_HASH` and the Global Constraints rely on this value.

- [ ] **Step 2: Write the failing tests**

In `net/tests/encode.rs`:

In `golden_state`, after `s.dev_played_this_turn = true;` add `s.offers_this_turn = 1;`.

Replace `the_layout_adds_up` with:

```rust
#[test]
fn the_layout_adds_up() {
    assert_eq!(PUBLIC_LEN, 1581);
    assert_eq!(HIDDEN_LEN, 30);
    assert_eq!(
        (
            EDGE_BASE,
            TILE_BASE,
            OWN_HAND,
            SEATS,
            LONGEST_ROAD,
            BANK,
            PHASE,
            TURN
        ),
        (1026, 1314, 1466, 1481, 1521, 1531, 1537, 1548)
    );
    assert_eq!(
        (DEV_PLAYED, CURRENT, RETURN_PHASE, FREE_ROADS),
        (1549, 1550, 1554, 1556)
    );
    assert_eq!(
        (TRADE_GIVE, TRADE_GET, TRADE_RESPONSES, OFFERS_MADE, OFFERS_ALLOWED),
        (1557, 1562, 1567, 1579, 1580)
    );
    assert_eq!(ENCODER_VERSION, 3);
    assert_eq!(ENCODING_HASH, layout_hash(&LAYOUT, LAYOUT_DESCRIPTION));
    assert_ne!(ENCODING_HASH, 0x28dd_1f78_0ff2_a1b3, "encoder v2's hash");
}
```

In `reordering_slots_or_blocks_in_the_description_changes_the_hash`, add to `mutations`:

```rust
        exchange(LAYOUT_DESCRIPTION, "accepted", "rejected"),
        exchange(LAYOUT_DESCRIPTION, "public trade_give", "public trade_get"),
```

In `the_description_matches_the_constants`, replace the scale assertion inside `for slot in &slots` with:

```rust
            if let Some((_, scale)) = slot.split_once('/') {
                // Scales the game's config sets rather than a constant.
                let from_config =
                    ["max_turns", "max_trade_cards", "max_offers_per_turn"].contains(&scale);
                assert!(
                    from_config || want_scales.iter().any(|&(k, _)| k == scale),
                    "slot {slot} names a declared scale"
                );
            }
```

and add before the `("hidden", "opponent", …)` row of `want`:

```rust
        ("public", "trade_give", 1, NUM_RESOURCES, TRADE_GIVE),
        ("public", "trade_get", 1, NUM_RESOURCES, TRADE_GET),
        ("public", "trade_response", NUM_PLAYERS, RESPONSE_WIDTH, TRADE_RESPONSES),
        ("public", "offers_made", 1, 1, OFFERS_MADE),
        ("public", "offers_allowed", 1, 1, OFFERS_ALLOWED),
```

In `the_golden_position_encodes_exactly`, after `want[FREE_ROADS] = 1.0 / 2.0;` add:

```rust
    want[OFFERS_MADE] = 1.0 / 3.0; // one of the default config's three offers this turn
    want[OFFERS_ALLOWED] = 1.0;
```

Append:

```rust
/// The trade block of a hand-built TradeConfirm position in the trading game (one card a side,
/// two offers a turn): player 1 offered a brick for an ore, its second offer this turn; players
/// 2 accepted, 3 and 0 rejected. Seen by player 3, so player 3 is r0, 0 is r1, 1 is r2, 2 is r3.
/// Without offers the block is all zero.
#[test]
fn the_trade_block_encodes_exactly() {
    let config = GameConfig {
        max_offers_per_turn: 2,
        max_trade_cards: 1,
        ..GameConfig::default()
    };
    let mut s = State::new(42, config);
    s.setup_step = 8;
    s.phase = Phase::TradeConfirm;
    s.current = 1;
    s.offers_this_turn = 2;
    s.trade = Some(PendingTrade {
        give: [0, 1, 0, 0, 0],
        get: [0, 0, 0, 0, 1],
        responses: [
            Response::Rejected,
            Response::Pending,
            Response::Accepted,
            Response::Rejected,
        ],
        next_responder: 0,
    });
    let mut got = vec![f32::NAN; PUBLIC_LEN];
    encode_public(&s.observation(3), &mut got);
    #[rustfmt::skip]
    let want: Vec<f32> = vec![
        0.0, 1.0, 0.0, 0.0, 0.0, // give: a brick / 1
        0.0, 0.0, 0.0, 0.0, 1.0, // get: an ore / 1
        0.0, 0.0, 1.0, // r0 = player 3 rejected
        0.0, 0.0, 1.0, // r1 = player 0 rejected
        1.0, 0.0, 0.0, // r2 = player 1, the offerer, stays pending
        0.0, 1.0, 0.0, // r3 = player 2 accepted
        1.0, // 2 of 2 offers made
        1.0, // offers allowed
    ];
    assert_eq!(&got[TRADE_GIVE..PUBLIC_LEN], &want[..]);

    let quiet = State::new(42, no_trades());
    let mut zero = vec![f32::NAN; PUBLIC_LEN];
    encode_public(&quiet.observation(0), &mut zero);
    assert!(zero[TRADE_GIVE..PUBLIC_LEN].iter().all(|&x| x == 0.0));
}
```

In `net/tests/graph.rs`, replace the first test with:

```rust
#[test]
fn the_global_block_is_the_115_scalars_after_the_tiles() {
    assert_eq!((GLOBAL_BASE, GLOBAL_WIDTH), (OWN_HAND, 115));
    assert_eq!(GLOBAL_BASE + GLOBAL_WIDTH, PUBLIC_LEN);
}
```

In `net/tests/weights.rs`: in `the_header_and_the_layer_shapes_follow_the_arch` change `(3, 91)` to `(3, 115)` and both `247_294` to `248_446` (the global encoder gains 24 × 48 = 1,152 weights); in `a_hand_computed_tiny_network` change `sparse(d, 91, &[], &[7.0, 8.0])` to `sparse(d, 115, &[], &[7.0, 8.0])`.

In `tests/python/test_train.py`, `test_the_default_net_is_width_48_with_2_rounds`: change both `247_294` to `248_446`.

- [ ] **Step 3: Run the tests to verify they fail**

Run: `~/.cargo/bin/cargo test --release -p settler-net --test encode --test graph --test weights`
Expected: compile error, `cannot find value TRADE_GIVE in this scope` (and the other new constants).

- [ ] **Step 4: Encoder v3**

In `net/src/encode.rs`:

Change the engine import to `use settler_engine::{Observation, Phase, Response, State, ACTION_SPACE_SIZE};` and `pub const ENCODER_VERSION: u32 = 3;`.

Replace `pub const PUBLIC_LEN: usize = FREE_ROADS + 1;` with:

```rust
/// The pending domestic offer (all 0 when none): what the offerer gives, then what it asks for,
/// per resource, each over `max_trade_cards`.
pub const TRADE_GIVE: usize = FREE_ROADS + 1;
pub const TRADE_GET: usize = TRADE_GIVE + NUM_RESOURCES;
/// Each seat's answer to the pending offer, seat-relative, one-hot: pending, accepted, rejected
/// (all 0 when no offer is pending; the offerer's own seat stays pending).
pub const TRADE_RESPONSES: usize = TRADE_GET + NUM_RESOURCES;
pub const RESPONSE_WIDTH: usize = 3;
/// Offers the player on turn has made this turn / `max_offers_per_turn` (0 when offers are off).
pub const OFFERS_MADE: usize = TRADE_RESPONSES + NUM_PLAYERS * RESPONSE_WIDTH;
/// 1 when the game allows domestic offers (`max_offers_per_turn > 0`).
pub const OFFERS_ALLOWED: usize = OFFERS_MADE + 1;
pub const PUBLIC_LEN: usize = OFFERS_ALLOWED + 1;
```

In `LAYOUT_DESCRIPTION`, change the first line to `settler-net features v3` and replace the line `public free_roads x1: roads_left/free_roads` with:

```text
public free_roads x1: roads_left/free_roads
public trade_give x1: wood/max_trade_cards brick/max_trade_cards sheep/max_trade_cards \
wheat/max_trade_cards ore/max_trade_cards
public trade_get x1: wood/max_trade_cards brick/max_trade_cards sheep/max_trade_cards \
wheat/max_trade_cards ore/max_trade_cards
public trade_response x4: pending accepted rejected
public offers_made x1: offers/max_offers_per_turn
public offers_allowed x1: allowed
```

(inside the Rust string, a trailing `\` continues the line exactly as the existing entries do).

At the end of `encode_public`, after the `RoadBuilding` block, add:

```rust
    if let Some(t) = obs.trade {
        let cap = obs.config.max_trade_cards as u32;
        for res in 0..NUM_RESOURCES {
            out[TRADE_GIVE + res] = scaled(t.give[res] as u32, cap);
            out[TRADE_GET + res] = scaled(t.get[res] as u32, cap);
        }
        for p in 0..NUM_PLAYERS {
            let answer = match t.responses[p] {
                Response::Pending => 0,
                Response::Accepted => 1,
                Response::Rejected => 2,
            };
            out[TRADE_RESPONSES + rel(p as PlayerId, v) * RESPONSE_WIDTH + answer] = 1.0;
        }
    }
    let max_offers = obs.config.max_offers_per_turn as u32;
    if max_offers > 0 {
        out[OFFERS_MADE] = scaled(obs.offers_this_turn as u32, max_offers);
        out[OFFERS_ALLOWED] = 1.0;
    }
```

Docs: in `net/src/graph.rs` change "(91)" to "(115: encoder v3's globals, the trade block included)"; in `net/src/weights.rs` change the table row to `| 3 | global encoder | d × 115 | the 115 global scalars (`graph::GLOBAL_BASE..PUBLIC_LEN`) |`; in `alphasettler/train/model.py`'s docstring change "91 global scalars" to "115 global scalars (encoder v3)".

- [ ] **Step 5: Run the tests to verify they pass**

Run: `~/.cargo/bin/cargo test --release -p settler-net`
Expected: all pass, including `the_trade_block_encodes_exactly`, the updated golden test, and `rotating_every_seat_and_the_actor_together_gives_the_same_features` (now also rotating the trade answers).
Rebuild the extension, then run: `.venv/bin/pytest -q tests/python/test_train.py -k "default_net or agree"`
Expected: 3 passed.

- [ ] **Step 6: Format, lint, full checks**

Run the Global Constraints' end-of-task checks. Expected as in Task 1.

- [ ] **Step 7: Commit**

```bash
git add net/src/encode.rs net/src/graph.rs net/src/weights.rs alphasettler/train/model.py net/tests/encode.rs net/tests/graph.rs net/tests/weights.rs tests/python/test_train.py
git diff --cached | grep -nE 'sk-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{36}|password=[^[:space:]]+'
git -c user.email=sean.may101@gmail.com commit -m "net: encoder v3, the trade block (24 globals: the pending offer, each seat's answer, offers made and allowed)"
```

---

### Task 3: `alphasettler net-upgrade`

**Files:**
- Create: `alphasettler/netupgrade.py`, `tests/python/test_net_upgrade.py`
- Modify: `net/src/weights.rs`, `bindings/src/train.rs`, `alphasettler/train/model.py`, `alphasettler/cli.py`
- Test: `net/tests/weights.rs`, `tests/python/test_net_upgrade.py`

**Interfaces:**
- Consumes (Task 2): `NET_GLOBAL_WIDTH` 115, `NET_ENCODING_HASH` (v3). From the bindings: `NET_FORMAT_VERSION`, `NET_MAX_WIDTH`, `NET_MAX_ROUNDS`, `NET_NODE_WIDTH`, `NET_EDGE_WIDTH`, `NET_TILE_WIDTH`, `NET_HIDDEN_LEN`, `NET_SETTLEMENT_BASE`, `net_adjacency()`, `action_space_size()`, `encode`, `net_forward`.
- Produces:
  - `settler_net::weights::{V2_ENCODING_HASH: u64 = 0x28dd_1f78_0ff2_a1b3, V2_GLOBAL_WIDTH: usize = 91}`; the bindings export them as `NET_V2_ENCODING_HASH`, `NET_V2_GLOBAL_WIDTH`.
  - Both loaders refuse a v2 file with `encoding hash … does not match this encoder's …: the file was trained on a different feature layout; it is an encoder v2 net (before the trade block): convert it with `alphasettler net-upgrade OLD NEW``.
  - `alphasettler.netupgrade`: `OFFER_BIAS = -3.0`, `OFFER_IDS = range(258, 658)`, `ANSWER_IDS = range(658, 665)`, `TRADE_SLOTS = range(258, 665)`, `shapes(width, rounds, global_width) -> list[tuple[int, int]]`, `pooled_row(slot) -> int`, `read_v2(data: bytes) -> tuple[int, int, list]`, `upgrade(data: bytes) -> bytes`, `upgrade_file(old, new) -> dict` (keys `width`, `rounds`, `zero_inputs`, `trade_rows`).
  - CLI: `alphasettler net-upgrade OLD NEW` (exit 0 and `wrote NEW`; exit 2 with `error: …` for an existing NEW or a file it cannot convert).

- [ ] **Step 1: Write the failing tests**

Append to `net/tests/weights.rs`:

```rust
#[test]
fn an_encoder_v2_file_is_refused_with_the_way_to_upgrade_it() {
    let mut bytes = Weights::random(SMALL, 1).to_bytes();
    bytes[12..20].copy_from_slice(&V2_ENCODING_HASH.to_le_bytes());
    let e = err(&bytes);
    assert!(e.contains("encoding hash"), "{e}");
    assert!(
        e.contains("encoder v2") && e.contains("alphasettler net-upgrade"),
        "{e}"
    );
    assert_ne!(V2_ENCODING_HASH, ENCODING_HASH);
    assert_eq!(
        settler_net::graph::GLOBAL_WIDTH - V2_GLOBAL_WIDTH,
        24,
        "the trade block"
    );
    let mut other = Weights::random(SMALL, 1).to_bytes();
    other[12] ^= 1;
    assert!(!err(&other).contains("net-upgrade"), "only v2's hash gets the hint");
}
```

Create `tests/python/test_net_upgrade.py`:

```python
"""`alphasettler net-upgrade` (spec 3c Section 4): an encoder v2 net rewritten for encoder v3,
with the same outputs on every no-trade position. Needs the `train` extra for the PyTorch twin,
like test_train.py."""

import random
import struct
import subprocess
import sys

import pytest
import torch

from alphasettler import Bot, Game, describe_action
from alphasettler._engine import (
    NET_GLOBAL_WIDTH, NET_HIDDEN_LEN, NET_PUBLIC_LEN, NET_V2_ENCODING_HASH, NET_V2_GLOBAL_WIDTH, encode,
    net_forward,
)
from alphasettler import netupgrade
from alphasettler.netupgrade import ANSWER_IDS, OFFER_BIAS, OFFER_IDS, TRADE_SLOTS, pooled_row, upgrade
from alphasettler.train.model import ACTIONS, MAGIC, Net, _floats, from_bytes, shapes, to_bytes

NO_TRADES = {"max_offers_per_turn": 0}


def cli(*args):
    return subprocess.run([sys.executable, "-m", "alphasettler", *args], capture_output=True, text=True)


def v2_bytes(net: Net) -> bytes:
    """`net` written as an encoder v2 file: v2's hash, and only the global encoder's first 91 input
    columns."""
    layers = net.layers()
    v2_shapes = [(r, NET_V2_GLOBAL_WIDTH if i == 3 else c) for i, (r, c) in enumerate(shapes(net.width, net.rounds))]
    out = [MAGIC, struct.pack("<IQIII", 2, NET_V2_ENCODING_HASH, net.width, net.rounds, len(layers))]
    out += [struct.pack("<II", r, c) for r, c in v2_shapes]
    for i, layer in enumerate(layers):
        w = layer.weight.detach()
        out += [_floats(w[:, :NET_V2_GLOBAL_WIDTH] if i == 3 else w), _floats(layer.bias)]
    return b"".join(out)


def no_trade_positions(n: int):
    """`n` encoded positions from random no-trade games, and each one's legal-move mask."""
    rng = random.Random(1)
    public, hidden, mask = [], [], torch.zeros(n, ACTIONS, dtype=torch.bool)
    seed = 0
    while len(public) < n:
        g = Game(seed, NO_TRADES)
        seed += 1
        for k in range(300):
            if g.is_over or len(public) == n:
                break
            if k % 5 == 0:
                p, h = encode(g)
                mask[len(public), g.legal_actions()] = True
                public.append(p)
                hidden.append(h)
            g.apply(rng.choice(g.legal_actions()))
    return b"".join(public), b"".join(hidden), mask


def _close(got, ref) -> float:
    return ((got - ref).abs() / ref.abs().clamp(min=1.0)).max().item()


@pytest.mark.parametrize("width,rounds", [(48, 2), (5, 3)])
def test_an_upgraded_net_keeps_every_no_trade_output(tmp_path, width, rounds):
    """In a no-trade game the trade block is all zero, so the original v2 net's outputs are `src`'s
    on the v3 features (its 24 extra global inputs meet only zeros). The upgraded net must give
    the same value and aux, the same logits on every slot but the trade rows, exactly the fixed
    biases on those, and the same prior over legal moves in the Rust forward pass."""
    torch.manual_seed(3)
    src = Net(width, rounds).eval()
    data = upgrade(v2_bytes(src))
    up = from_bytes(data).eval()
    assert (up.global_encoder.weight[:, NET_V2_GLOBAL_WIDTH:] == 0).all()
    path = tmp_path / "up.bin"
    path.write_bytes(data)
    n = 100
    public, hidden, mask = no_trade_positions(n)
    pt = torch.frombuffer(bytearray(public), dtype=torch.float32).view(n, NET_PUBLIC_LEN)
    ht = torch.frombuffer(bytearray(hidden), dtype=torch.float32).view(n, NET_HIDDEN_LEN)
    block = NET_GLOBAL_WIDTH - NET_V2_GLOBAL_WIDTH
    assert (pt[:, NET_PUBLIC_LEN - block:] == 0).all(), "no-trade games leave the trade block zero"
    with torch.no_grad():
        ref, got = src(pt, ht), up(pt, ht)
    trade = torch.zeros(ACTIONS, dtype=torch.bool)
    trade[list(TRADE_SLOTS)] = True
    assert not mask[:, trade].any(), "no trade move is legal in a no-trade game"
    assert _close(got[0][:, ~trade], ref[0][:, ~trade]) <= 1e-5
    assert (got[0][:, list(OFFER_IDS)] == OFFER_BIAS).all()
    assert (got[0][:, list(ANSWER_IDS)] == 0).all()
    assert _close(got[1], ref[1]) <= 1e-5 and _close(got[2], ref[2]) <= 1e-5
    rust = net_forward(str(path), public, hidden)
    rp = torch.frombuffer(bytearray(rust["policy"]), dtype=torch.float32).view(n, ACTIONS)
    rv = torch.frombuffer(bytearray(rust["value"]), dtype=torch.float32).view(n, 4)

    def prior(x):
        return torch.softmax(x.double().masked_fill(~mask, float("-inf")), 1)

    assert (prior(rp) - prior(ref[0])).abs().max().item() <= 1e-5
    assert (torch.softmax(rv.double(), 1) - torch.softmax(ref[1].double(), 1)).abs().max().item() <= 1e-5


def test_only_encoder_v2_nets_are_upgraded():
    torch.manual_seed(0)
    old = v2_bytes(Net(8, 1))
    with pytest.raises(ValueError, match="already an encoder v3 net"):
        upgrade(to_bytes(Net(8, 1)))
    other = bytearray(old)
    other[12] ^= 1
    with pytest.raises(ValueError, match="not encoder v2's"):
        upgrade(bytes(other))
    with pytest.raises(ValueError, match="wrong magic"):
        upgrade(b"XXNET\0\0\0" + old[8:])
    one = bytearray(old)
    one[8:12] = (1).to_bytes(4, "little")
    with pytest.raises(ValueError, match="format version 1"):
        upgrade(bytes(one))
    with pytest.raises(ValueError, match="truncated"):
        upgrade(old[:20])
    with pytest.raises(ValueError, match="truncated"):
        upgrade(old[:-4])
    with pytest.raises(ValueError, match="trailing bytes"):
        upgrade(old + b"\0\0\0\0")
    wide = bytearray(old)
    struct.pack_into("<I", wide, 20, 2**31)
    with pytest.raises(ValueError, match="width 2147483648 is out of range"):
        upgrade(bytes(wide))
    nan = bytearray(old)
    struct.pack_into("<f", nan, len(nan) - 4, float("nan"))
    with pytest.raises(ValueError, match="NaN or infinite"):
        upgrade(bytes(nan))
    assert from_bytes(upgrade(old)).width == 8


def test_loaders_send_an_encoder_v2_net_to_net_upgrade(tmp_path):
    torch.manual_seed(0)
    nets = tmp_path / "nets"
    nets.mkdir()
    (nets / "old.bin").write_bytes(v2_bytes(Net(8, 1)))
    p, h = encode(Game(0, NO_TRADES))
    loads = [lambda: net_forward(str(nets / "old.bin"), p, h),
             lambda: from_bytes((nets / "old.bin").read_bytes()),
             lambda: Bot("net:old", 0, str(nets))]
    for load in loads:
        with pytest.raises(ValueError, match="net-upgrade"):
            load()


def test_the_trade_rows_are_the_engines_offer_answer_and_confirmation_ids():
    assert len(OFFER_IDS) == 400 and all(describe_action(a).startswith("OfferTrade") for a in OFFER_IDS)
    assert [describe_action(a) for a in ANSWER_IDS] == [
        "AcceptTrade", "RejectTrade", "ConfirmTrade(0)", "ConfirmTrade(1)", "ConfirmTrade(2)", "ConfirmTrade(3)",
        "CancelTrade"]
    assert (pooled_row(0), pooled_row(24), pooled_row(258), pooled_row(664)) == (0, 24, 59, 465)
    with pytest.raises(ValueError, match="entity head"):
        pooled_row(25)
    assert netupgrade.shapes(48, 2, NET_GLOBAL_WIDTH) == shapes(48, 2)


def test_cli_net_upgrade(tmp_path):
    torch.manual_seed(0)
    old, new = tmp_path / "old.bin", tmp_path / "new.bin"
    old.write_bytes(v2_bytes(Net(8, 1)))
    r = cli("net-upgrade", str(old), str(new))
    assert r.returncode == 0, r.stderr
    assert f"wrote {new}" in r.stdout and "width 8, 1 rounds" in r.stdout
    assert from_bytes(new.read_bytes()).width == 8
    r = cli("net-upgrade", str(old), str(new))
    assert r.returncode == 2 and "exists" in r.stderr
    r = cli("net-upgrade", str(new), str(tmp_path / "again.bin"))
    assert r.returncode == 2 and "already an encoder v3 net" in r.stderr
    assert not (tmp_path / "again.bin").exists()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `~/.cargo/bin/cargo test --release -p settler-net --test weights`
Expected: compile error, `cannot find value V2_ENCODING_HASH`.
Run: `.venv/bin/pytest -q tests/python/test_net_upgrade.py`
Expected: collection error, `cannot import name 'NET_V2_ENCODING_HASH'`.

- [ ] **Step 3: The v2 constants and the loaders' hint**

In `net/src/weights.rs`, after `MAX_ROUNDS`, add:

```rust
/// Encoder v2's `ENCODING_HASH` (3b's nets, before the trade block) and its global width. The
/// loader refuses such a file; `alphasettler net-upgrade` rewrites it for this encoder.
pub const V2_ENCODING_HASH: u64 = 0x28dd_1f78_0ff2_a1b3;
pub const V2_GLOBAL_WIDTH: usize = 91;
```

and replace the hash check in `from_bytes` with:

```rust
        let hash = u64::from_le_bytes(bytes[12..20].try_into().unwrap());
        if hash != ENCODING_HASH {
            let why = if hash == V2_ENCODING_HASH {
                "; it is an encoder v2 net (before the trade block): convert it with \
                 `alphasettler net-upgrade OLD NEW`"
            } else {
                ""
            };
            return Err(format!(
                "encoding hash {hash:#018x} does not match this encoder's {ENCODING_HASH:#018x}: \
                 the file was trained on a different feature layout{why}"
            ));
        }
```

In `bindings/src/train.rs`, change the weights import to `use settler_net::weights::{Arch, Weights, FORMAT_VERSION, MAX_ROUNDS, MAX_WIDTH, V2_ENCODING_HASH, V2_GLOBAL_WIDTH};` and add to `register`, after `NET_GLOBAL_WIDTH`:

```rust
    m.add("NET_V2_ENCODING_HASH", V2_ENCODING_HASH)?;
    m.add("NET_V2_GLOBAL_WIDTH", V2_GLOBAL_WIDTH)?;
```

In `alphasettler/train/model.py`, add `NET_V2_ENCODING_HASH` to the `_engine` import and replace the hash check in `from_bytes` with:

```python
    if encoding != NET_ENCODING_HASH:
        why = ("; it is an encoder v2 net (before the trade block): convert it with `alphasettler net-upgrade OLD NEW`"
               if encoding == NET_V2_ENCODING_HASH else "")
        raise ValueError(f"encoding hash {encoding:#018x} does not match this encoder's {NET_ENCODING_HASH:#018x}: "
                         f"the file was trained on a different feature layout{why}")
```

- [ ] **Step 4: The upgrade**

Create `alphasettler/netupgrade.py`:

```python
"""`alphasettler net-upgrade OLD NEW` (spec 3c Section 4): an encoder v2 net (3b's: 91 global
inputs) rewritten for encoder v3 (115: the trade block appended), with the same moves in every
no-trade game.

- The global encoder gets zero weights for the 24 new inputs, so the net ignores them.
- The pooled head's rows for every domestic offer (OFFER_IDS), the answers and the confirmations
  (ANSWER_IDS, CancelTrade included) get zero weights, and biases of OFFER_BIAS for offers and 0
  for the rest. 3b never trained them: offers were masked out, and every answer was a hard-coded
  reject or cancel. All 400 offer rows are reset, not only the 20 one-for-one offers the trading
  game allows, because the same reason holds for all of them.
- Every other byte is copied.

Plain `struct` and `array`, no torch, so it runs without the train extra. The weights format is
net/src/weights.rs's, version 2."""

from __future__ import annotations

import math
import struct
import sys
from array import array
from pathlib import Path

from alphasettler import files
from alphasettler._engine import (
    NET_EDGE_WIDTH, NET_ENCODING_HASH, NET_FORMAT_VERSION, NET_GLOBAL_WIDTH, NET_HIDDEN_LEN, NET_MAX_ROUNDS,
    NET_MAX_WIDTH, NET_NODE_WIDTH, NET_SETTLEMENT_BASE, NET_TILE_WIDTH, NET_V2_ENCODING_HASH, NET_V2_GLOBAL_WIDTH,
    action_space_size, net_adjacency,
)

MAGIC = b"ASNET\0\0\0"
FIXED_HEADER_LEN = 8 + 4 + 8 + 4 + 4 + 4
SEATS = 4
GLOBAL_ENCODER = 3
OTHER_POLICY = 4  # the pooled head's output layer, after the rounds (net/src/weights.rs)
OFFER_BIAS = -3.0
# Engine action ids (engine/src/action.rs): offers, then accept, reject, the four confirmations, cancel.
OFFER_IDS = range(258, 658)
ANSWER_IDS = range(658, 665)
TRADE_SLOTS = range(258, 665)
_ADJ = net_adjacency()
# Slots NET_SETTLEMENT_BASE .. + ENTITY_SLOTS come from the per-entity heads (net/src/policy.rs).
ENTITY_SLOTS = 2 * len(_ADJ["node_tiles"]) + len(_ADJ["edge_nodes"]) + len(_ADJ["tile_nodes"])
OTHER = action_space_size() - ENTITY_SLOTS


def layer_count(rounds: int) -> int:
    return 4 + 4 * rounds + 8


def shapes(width: int, rounds: int, global_width: int) -> list[tuple[int, int]]:
    """(rows, cols) of each layer in file order for a global encoder of `global_width` inputs
    (`model.shapes` with the global width as an argument)."""
    d = width
    out = [(d, NET_NODE_WIDTH), (d, NET_EDGE_WIDTH), (d, NET_TILE_WIDTH), (d, global_width)]
    out += [(d, 3 * d), (d, 5 * d), (d, 3 * d), (d, 4 * d)] * rounds
    return out + [(2, 2 * d), (1, 2 * d), (1, 2 * d), (4 * d, 4 * d), (OTHER, 4 * d),
                  (4 * d, 4 * d + NET_HIDDEN_LEN), (SEATS, 4 * d), (SEATS, 4 * d)]


def pooled_row(slot: int) -> int:
    """The pooled head's output row for policy slot `slot` (net/src/policy.rs `other_slot`, inverted)."""
    if NET_SETTLEMENT_BASE <= slot < NET_SETTLEMENT_BASE + ENTITY_SLOTS:
        raise ValueError(f"slot {slot} comes from an entity head")
    return slot if slot < NET_SETTLEMENT_BASE else slot - ENTITY_SLOTS


def read_v2(data: bytes) -> tuple[int, int, list[tuple[int, int, array, array]]]:
    """(width, rounds, layers) of an encoder v2 weights file, each layer (rows, cols, weights,
    biases); ValueError for any other file. Everything is checked before anything is sized from it."""
    assert sys.byteorder == "little", "the weights file is little-endian"
    if len(data) < FIXED_HEADER_LEN:
        raise ValueError(f"truncated: {len(data)} bytes is shorter than the {FIXED_HEADER_LEN}-byte header")
    if data[:8] != MAGIC:
        raise ValueError("not an AlphaSettler weights file (wrong magic)")
    version, encoding, width, rounds, count = struct.unpack_from("<IQIII", data, 8)
    if version != NET_FORMAT_VERSION:
        raise ValueError(f"format version {version}, expected {NET_FORMAT_VERSION}")
    if encoding == NET_ENCODING_HASH:
        raise ValueError("already an encoder v3 net (this build's encoding hash): nothing to upgrade")
    if encoding != NET_V2_ENCODING_HASH:
        raise ValueError(f"encoding hash {encoding:#018x} is not encoder v2's {NET_V2_ENCODING_HASH:#018x}: "
                         "only encoder v2 nets can be upgraded")
    if not 1 <= width <= NET_MAX_WIDTH:
        raise ValueError(f"width {width} is out of range 1..={NET_MAX_WIDTH}")
    if not 1 <= rounds <= NET_MAX_ROUNDS:
        raise ValueError(f"rounds {rounds} is out of range 1..={NET_MAX_ROUNDS}")
    if count != layer_count(rounds):
        raise ValueError(f"{count} layers, expected {layer_count(rounds)} for {rounds} rounds")
    header = FIXED_HEADER_LEN + 8 * count
    if len(data) < header:
        raise ValueError(f"truncated: {len(data)} bytes is shorter than the {header}-byte header")
    declared = [struct.unpack_from("<II", data, FIXED_HEADER_LEN + 8 * i) for i in range(count)]
    if declared != shapes(width, rounds, NET_V2_GLOBAL_WIDTH):
        raise ValueError(f"layer shapes {declared} do not fit an encoder v2 net with width {width} and {rounds} rounds")
    want = header + 4 * sum(r * c + r for r, c in declared)
    if len(data) < want:
        raise ValueError(f"truncated: {len(data)} bytes, expected {want}")
    if len(data) > want:
        raise ValueError(f"{len(data) - want} trailing bytes after the last layer")
    floats = array("f")
    floats.frombytes(data[header:])
    if not all(math.isfinite(x) for x in floats):
        raise ValueError("a weight is NaN or infinite")
    layers, at = [], 0
    for rows, cols in declared:
        w = floats[at:at + rows * cols]
        at += rows * cols
        b = floats[at:at + rows]
        at += rows
        layers.append((rows, cols, w, b))
    return width, rounds, layers


def upgrade(data: bytes) -> bytes:
    """The encoder v3 weights file for the encoder v2 file `data` (see the module docstring)."""
    width, rounds, layers = read_v2(data)
    extra = NET_GLOBAL_WIDTH - NET_V2_GLOBAL_WIDTH
    rows, cols, w, b = layers[GLOBAL_ENCODER]
    wide = array("f")
    for r in range(rows):
        wide.extend(w[r * cols:(r + 1) * cols])
        wide.extend(array("f", [0.0]) * extra)
    layers[GLOBAL_ENCODER] = (rows, NET_GLOBAL_WIDTH, wide, b)
    rows, cols, w, b = layers[4 + 4 * rounds + OTHER_POLICY]
    for slot in TRADE_SLOTS:
        r = pooled_row(slot)
        w[r * cols:(r + 1) * cols] = array("f", [0.0]) * cols
        b[r] = OFFER_BIAS if slot in OFFER_IDS else 0.0
    out = [MAGIC, struct.pack("<IQIII", NET_FORMAT_VERSION, NET_ENCODING_HASH, width, rounds, len(layers))]
    out += [struct.pack("<II", r, c) for r, c, _, _ in layers]
    for _, _, w, b in layers:
        out += [w.tobytes(), b.tobytes()]
    return b"".join(out)


def upgrade_file(old: str | Path, new: str | Path) -> dict:
    """Write the upgrade of `old` to `new` (which must not exist, so nothing is overwritten)."""
    old, new = Path(old), Path(new)
    if new.exists():
        raise ValueError(f"{new} exists: choose a new path")
    try:
        data = upgrade(old.read_bytes())
    except ValueError as e:
        raise ValueError(f"{old}: {e}") from None
    files.atomic_write_bytes(new, data)
    width, rounds = struct.unpack_from("<II", data, 20)
    return {"width": width, "rounds": rounds, "zero_inputs": NET_GLOBAL_WIDTH - NET_V2_GLOBAL_WIDTH,
            "trade_rows": len(TRADE_SLOTS)}
```

In `alphasettler/cli.py`: add `` `net-upgrade` `` to the module docstring's command list ("…, `fast-loop`, `train-loop` and `net-upgrade`" with "; `net-upgrade` rewrites a 3b (encoder v2) net for encoder v3" appended to the first sentence). In `_parser`, before `lp = sub.add_parser("train-loop", …)`:

```python
    nu = sub.add_parser("net-upgrade", help="rewrite an encoder v2 net (3b's) for encoder v3, same no-trade moves")
    nu.add_argument("old", help="the encoder v2 weights file")
    nu.add_argument("new", help="the weights file to write (must not exist)")
```

Add the handler after `_train_loop`:

```python
def _net_upgrade(args) -> int:
    from alphasettler import netupgrade
    from alphasettler._engine import Game, encode, net_forward

    s = netupgrade.upgrade_file(args.old, args.new)
    p, h = encode(Game(0, {"max_offers_per_turn": 0}))
    net_forward(str(args.new), p, h)  # the Rust loader checks the file as every bot will read it
    print(f"upgraded {args.old} (encoder v2) to encoder v3: width {s['width']}, {s['rounds']} rounds; "
          f"{s['zero_inputs']} new global inputs and {s['trade_rows']} trade policy rows zeroed "
          f"(offer bias {netupgrade.OFFER_BIAS:g}); wrote {args.new}")
    return 0
```

and in `main`, before the final `else:` (compare): `elif args.command == "net-upgrade":` / `return _net_upgrade(args)`.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `~/.cargo/bin/cargo test --release -p settler-net --test weights`
Expected: all pass, including `an_encoder_v2_file_is_refused_with_the_way_to_upgrade_it`.
Rebuild the extension, then run: `.venv/bin/pytest -q tests/python/test_net_upgrade.py`
Expected: 6 passed.

- [ ] **Step 6: Format, lint, full checks**

Run the Global Constraints' end-of-task checks. Expected as in Task 1.

- [ ] **Step 7: Commit**

```bash
git add net/src/weights.rs bindings/src/train.rs alphasettler/train/model.py alphasettler/netupgrade.py alphasettler/cli.py net/tests/weights.rs tests/python/test_net_upgrade.py
git diff --cached | grep -nE 'sk-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{36}|password=[^[:space:]]+'
git -c user.email=sean.may101@gmail.com commit -m "net, train: net-upgrade turns an encoder v2 net into a v3 one with the same no-trade moves"
```

---

### Task 4: Training data and the trading game — offers as targets, trading self-play, `--trades`

**Files:**
- Modify: `net/src/data.rs`, `net/src/evaluator.rs`, `bots/src/selfplay.rs`, `alphasettler/arena.py`, `alphasettler/cli.py`, `alphasettler/train/fastloop.py`
- Test: `net/tests/data.rs`, `net/tests/evaluator.rs`, `bots/tests/selfplay.rs`, `tests/python/test_cli.py`, `tests/python/test_fast_loop.py`

**Interfaces:**
- Consumes (Task 1): `SearchConfig::offers`, `search_moves`; `+t` names. (Task 3): v3 nets only.
- Produces:
  - `Batch::push` and `check_game`: every legal move is in the mask and the policy target, offers included; `masked_prior(logits, legal, actor)` gives offers their own logits.
  - `selfplay::play_with`: every seat searches offers exactly when `config.max_offers_per_turn > 0`.
  - `alphasettler.arena.TRADES = {"max_offers_per_turn": 2, "max_trade_cards": 1}`.
  - CLI: `arena --trades` (mutually exclusive with `--no-trades`), `selfplay --trades` (the trading game), `fast-loop --trades` (a first-run setting).
  - `FastLoop.init(root, seed_dirs, start_net, first, trades: bool = False)` writes `"trades"` into `state.json`; `FastLoop.trades: bool`; `FastLoop.config` is `{**TRADES, "max_turns": …}` when trading.

- [ ] **Step 1: Write the failing tests**

In `net/tests/data.rs`, in `a_position_equals_the_encoder_on_the_replayed_state`, replace everything from `let mut offers_masked = 0;` to the end of the function so that offers count like any move:

```rust
    let (mut offers_in_mask, mut offers_targeted) = (0, 0);
    for (gi, g) in games.iter().enumerate() {
        let mut s = State::new(g.seed, g.config);
        let mut next = 0;
        for (i, &a) in g.actions.iter().enumerate() {
            if next < g.decisions.len() && g.decisions[next].index as usize == i {
                let d = &g.decisions[next];
                let mut b = Batch::default();
                set.position(gi, next, &mut b);
                let (mut public, mut hidden) = (vec![0.0; PUBLIC_LEN], vec![0.0; HIDDEN_LEN]);
                encode_state(&s, d.actor, &mut public, &mut hidden);
                assert_eq!((b.n, &b.public, &b.hidden), (1, &public, &hidden));
                let legal = s.legal_actions();
                let total: u32 = d.visits.iter().sum();
                let (mut mask, mut policy) = (
                    vec![0u8; ACTION_SPACE_SIZE],
                    vec![0.0f32; ACTION_SPACE_SIZE],
                );
                for (&a, &v) in legal.iter().zip(&d.visits) {
                    let k = policy_id(a, d.actor);
                    mask[k] = 1;
                    policy[k] = if total > 0 {
                        v as f32 / total as f32
                    } else {
                        1.0 / legal.len() as f32
                    };
                    if matches!(a, Action::OfferTrade { .. }) {
                        offers_in_mask += b.mask[k] as u32;
                        offers_targeted += (b.policy[k] > 0.0) as u32;
                    }
                }
                assert_eq!(b.mask, mask);
                assert_eq!(b.policy, policy);
                assert_eq!(b.full, vec![(d.full_search && total > 0) as u8]);
                let me = d.actor as usize;
                let outcome: Vec<f32> = (0..4)
                    .map(|r| match g.winner {
                        Some(w) => (w as usize == (me + r) % 4) as u8 as f32,
                        None => 0.25,
                    })
                    .collect();
                assert_eq!(b.outcome, outcome);
                let vp: Vec<f32> = (0..4).map(|r| g.vp[(me + r) % 4] as f32 / 10.0).collect();
                assert_eq!(b.vp, vp);
                next += 1;
            }
            s.apply(a);
        }
    }
    assert!(
        offers_in_mask > 0 && offers_targeted > 0,
        "offers sit in the mask ({offers_in_mask}) and in the targets ({offers_targeted})"
    );
}
```

Append to `net/tests/data.rs`:

```rust
/// In the trading game every legal move, offers and seat-relative confirmations included, has
/// one slot, and the evaluator's prior reads the same slots.
#[test]
fn trade_moves_sit_in_the_same_slots_for_the_sampler_and_the_evaluator() {
    let config = GameConfig {
        max_offers_per_turn: 2,
        max_trade_cards: 1,
        max_turns: 150,
        ..GameConfig::default()
    };
    let games: Vec<RecordedGame> = (1..4).map(|seed| recorded_every(seed, config, 1)).collect();
    let set = TrainingSet::new(games.clone()).unwrap();
    let (mut offers, mut relative_confirms) = (0, 0);
    for (gi, g) in games.iter().enumerate() {
        let mut s = State::new(g.seed, g.config);
        for (di, d) in g.decisions.iter().enumerate() {
            let mut b = Batch::default();
            set.position(gi, di, &mut b);
            let legal = s.legal_actions();
            let slots: Vec<usize> = (0..ACTION_SPACE_SIZE).filter(|&k| b.mask[k] == 1).collect();
            assert_eq!(slots.len(), legal.len(), "one slot per legal move, offers included");
            let prior = masked_prior(&vec![0.0; ACTION_SPACE_SIZE], &legal, d.actor);
            assert!(prior
                .iter()
                .all(|&p| (p - 1.0 / legal.len() as f32).abs() < 1e-6));
            for a in &legal {
                let k = policy_id(*a, d.actor);
                assert!(slots.contains(&k), "{a:?} by seat {}", d.actor);
                offers += matches!(a, Action::OfferTrade { .. }) as u32;
                relative_confirms +=
                    (matches!(a, Action::ConfirmTrade(_)) && k != a.encode() as usize) as u32;
            }
            s.apply(g.actions[di]);
        }
    }
    assert!(
        offers > 0 && relative_confirms > 0,
        "{offers} offers, {relative_confirms} seat-relative confirmations"
    );
}
```

In `net/tests/evaluator.rs`, replace `a_single_legal_move_gets_all_the_prior_and_offers_get_none` and `a_prior_with_no_finite_logit_is_uniform_over_the_moves_it_may_play` with:

```rust
#[test]
fn a_single_legal_move_gets_all_the_prior_and_offers_read_their_own_logits() {
    let mut eval = NetEvaluator::new(net());
    let s = State::new(3, GameConfig::default());
    let actor = s.current_actor();
    let e = eval.evaluate(&[Leaf {
        world: &s,
        actor,
        legal: &[Action::Roll],
    }]);
    assert_eq!(e[0].prior, vec![1.0]);
    let offer = Action::OfferTrade { give: 0, get: 1 };
    let mixed = [Action::EndTurn, offer, Action::BuyDev];
    let e = eval.evaluate(&[Leaf {
        world: &s,
        actor,
        legal: &mixed,
    }]);
    assert!(e[0].prior[1] > 0.0, "a searched offer has prior mass");
    assert!((e[0].prior.iter().sum::<f32>() - 1.0).abs() < 1e-6);
    let mut logits = vec![0.0f32; ACTION_SPACE_SIZE];
    logits[offer.encode() as usize] = 2.0f32.ln(); // twice EndTurn's and BuyDev's weight
    let p = masked_prior(&logits, &mixed, 0);
    assert!((p[0] - 0.25).abs() < 1e-6 && (p[1] - 0.5).abs() < 1e-6 && (p[2] - 0.25).abs() < 1e-6);
    assert_eq!(
        masked_prior(&vec![0.0; ACTION_SPACE_SIZE], &[offer], 0),
        vec![1.0]
    );
}

#[test]
fn a_prior_with_no_finite_logit_is_uniform_over_the_moves_at_the_maximum() {
    let offer = Action::OfferTrade { give: 0, get: 1 };
    let dead = vec![f32::NEG_INFINITY; ACTION_SPACE_SIZE];
    let third = 1.0 / 3.0;
    assert_eq!(
        masked_prior(&dead, &[Action::EndTurn, offer, Action::BuyDev], 0),
        vec![third, third, third]
    );
    assert_eq!(masked_prior(&dead, &[offer, offer], 0), vec![0.5, 0.5]);
    let mut huge = vec![0.0; ACTION_SPACE_SIZE];
    huge[Action::EndTurn.encode() as usize] = f32::INFINITY;
    let p = masked_prior(&huge, &[Action::EndTurn, offer, Action::BuyDev], 0);
    assert_eq!(p, vec![1.0, 0.0, 0.0], "the softmax's limit, not NaN");
}
```

Append to `bots/tests/selfplay.rs`:

```rust
/// The trading game (spec 3c Section 2).
fn trading() -> GameConfig {
    GameConfig {
        max_offers_per_turn: 2,
        max_trade_cards: 1,
        ..GameConfig::default()
    }
}

/// Trading self-play: every seat may offer, and offers, answers and confirmations are recorded
/// like any other searched decision, visits aligned with the legal moves (spec 3c Section 5).
/// Moves are drawn in proportion to visits, so offers with any visits get played.
#[test]
fn trading_self_play_records_offers_answers_and_confirmations() {
    let ex = Exploration {
        sample_turns: u32::MAX,
        ..Exploration::fixed(20)
    };
    let (mut offers, mut answers, mut confirmations) = (0, 0, 0);
    for seed in 0..3 {
        let g = play_with(seed, trading(), &ex, 8, &heuristic);
        assert_eq!(g.belief_resets, 0, "seed {seed}");
        let mut s = State::new(seed, trading());
        let mut decisions = g.decisions.iter().peekable();
        for (i, &a) in g.actions.iter().enumerate() {
            if let Some(d) = decisions.next_if(|d| d.index as usize == i) {
                assert_eq!(d.legal, s.legal_actions());
                assert_eq!(d.visits.len(), d.legal.len());
                offers += d
                    .legal
                    .iter()
                    .zip(&d.visits)
                    .filter(|(a, v)| matches!(a, Action::OfferTrade { .. }) && **v > 0)
                    .count();
                answers += (s.phase == Phase::TradeResponse) as u32;
                confirmations += (s.phase == Phase::TradeConfirm) as u32;
            }
            s.apply(a);
        }
    }
    assert!(
        offers > 0 && answers > 0 && confirmations > 0,
        "{offers} visited offers, {answers} answers, {confirmations} confirmations"
    );
}
```

Append to `tests/python/test_cli.py`:

```python
def test_cli_arena_trades_plays_the_trading_game(tmp_path, monkeypatch, capsys):
    from alphasettler import cli as cli_module
    from alphasettler.arena import TRADES

    assert TRADES == {"max_offers_per_turn": 2, "max_trade_cards": 1}
    seen = []
    real = cli_module.run

    def spy(candidate, baseline, seeds, seed_start, threads, config, out, nets_dir):
        seen.append(config)
        return real(candidate, baseline, seeds, seed_start, threads, config, out, nets_dir)

    monkeypatch.setattr(cli_module, "run", spy)
    rc = cli_module.main(["arena", "--candidate", "ismcts@10+t", "--baseline", "ismcts@10", "--seeds", "1",
                          "--trades", "--out", str(tmp_path / "t.jsonl")])
    assert rc == 0
    assert seen == [TRADES]
    assert "belief resets: 0" in capsys.readouterr().out


def test_cli_trades_and_no_trades_are_exclusive():
    r = cli("arena", "--candidate", "random", "--baseline", "random", "--seeds", "1", "--trades", "--no-trades")
    assert r.returncode == 2 and "not allowed with" in r.stderr


def test_cli_selfplay_trades_records_the_trading_game(tmp_path):
    from alphasettler import records

    r = cli("selfplay", "--pcr", "--games", "1", "--full-simulations", "4", "--fast-simulations", "2", "--trades",
            "--out-dir", str(tmp_path))
    assert r.returncode == 0, r.stderr
    (path,) = tmp_path.glob("*.jsonl.gz")
    (g,) = list(records.read(path))
    assert (g["config"]["max_offers_per_turn"], g["config"]["max_trade_cards"]) == (2, 1)
    assert records.replay_mismatches(g) == []
```

In `tests/python/test_fast_loop.py`, change the imports to

```python
from alphasettler import records
from alphasettler.arena import TRADES
from alphasettler.train.fastloop import GATE_MIN_Z, STATE, FastLoop, FastLoopSettings, count_games, promoted
from alphasettler.train.loop import SEEDS_PER_GENERATION, gen_name
from alphasettler.train.model import Net, save
```

and append:

```python
def test_a_trading_loop_plays_and_gates_in_the_trading_game(tmp_path):
    seed = generation(tmp_path, "seed-gen", seed_start=0)  # no-trade records start the window
    save(Net(8, 1), tmp_path / "start.bin")
    root = tmp_path / "loop"
    FastLoop.init(root, [seed], tmp_path / "start.bin", first=1, trades=True)
    loop = FastLoop(root, FastLoopSettings(generations=2, **TINY))
    assert loop.trades and loop.config == {**TRADES, "max_turns": 1000}
    assert loop.run() == 2
    state = json.loads((root / STATE).read_text())
    assert state["trades"] is True
    for g in state["generations"]:
        assert g["gate"]["candidate"].endswith("+t") and g["gate"]["baseline"].endswith("+t"), g["gate"]
    for k in (1, 2):
        paths = sorted((root / gen_name(k)).glob("selfplay-local-*.jsonl.gz"))
        assert paths
        for path in paths:
            for game in records.read(path):
                assert (game["config"]["max_offers_per_turn"], game["config"]["max_trade_cards"]) == (2, 1)


def test_cli_fast_loop_keeps_the_trading_setting_it_was_set_up_with(tmp_path):
    root, _ = setup(tmp_path)  # set up without --trades
    assert FastLoop(root, FastLoopSettings(generations=0, **TINY)).trades is False
    r = cli("fast-loop", "--dir", str(root), "--generations", "0", "--trades")
    assert r.returncode == 2 and "without --trades" in r.stderr
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `~/.cargo/bin/cargo test --release -p settler-net --test data --test evaluator`
Expected: FAIL: `offers sit in the mask (0) …`, the slot-count assertion in `trade_moves_sit_in_the_same_slots_for_the_sampler_and_the_evaluator`, and the offer-prior assertions.
Run: `~/.cargo/bin/cargo test --release -p settler-bots --test selfplay trading_self_play`
Expected: FAIL with `0 visited offers` (self-play seats do not offer yet).
Run: `.venv/bin/pytest -q tests/python/test_cli.py tests/python/test_fast_loop.py`
Expected: `test_fast_loop.py` fails to collect (`cannot import name 'TRADES'`); the three new `test_cli.py` tests fail.

- [ ] **Step 3: Offers in the targets and the prior**

In `net/src/data.rs`, change the `mask` doc to "n × ACTION_SPACE_SIZE: 1 at the policy id (`policy::policy_id`, seat-relative for steals and trade confirmations) of each legal move, domestic offers included." In `Batch::push`, replace the `searchable` closure and the loop with:

```rust
        // u64: visit counts come from files and may sum past u32::MAX.
        let total: u64 = d.visits.iter().map(|&v| v as u64).sum();
        for (a, &v) in legal.iter().zip(&d.visits) {
            let k = m0 + policy_id(*a, actor);
            self.mask[k] = 1;
            self.policy[k] = if total > 0 {
                v as f32 / total as f32
            } else {
                1.0 / legal.len() as f32
            };
        }
```

In `check_game`, change its doc's parenthesis to "(`Batch::full` 1: a full search with at least one visit)" and replace the `visited` computation with `let visited = d.visits.iter().any(|&v| v > 0);`.

In `net/src/evaluator.rs`, delete `fn is_offer` and replace `masked_prior` with:

```rust
/// The prior over `legal` for `actor`: a softmax of the policy logits at the moves' policy ids
/// (`policy::policy_id`: steals and trade confirmations seat-relative, as in the training
/// targets). Domestic offers reach it only from a bot that searches them (`+t`) and read their
/// own logits like any move. With no finite maximum (every logit -inf, or some +inf) it is the
/// softmax's limit: uniform over the moves at the maximum, never NaN.
pub fn masked_prior(logits: &[f32], legal: &[Action], actor: PlayerId) -> Vec<f32> {
    let mut p: Vec<f32> = legal.iter().map(|a| logits[policy_id(*a, actor)]).collect();
    let max = p.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if max.is_finite() {
        softmax(&mut p);
        return p;
    }
    for x in p.iter_mut() {
        *x = (*x == max) as u8 as f32;
    }
    let n: f32 = p.iter().sum();
    p.iter_mut().for_each(|x| *x /= n);
    p
}
```

- [ ] **Step 4: Trading self-play**

In `bots/src/selfplay.rs`, add to the module doc: "3c: every seat may offer exactly when the game allows offers (the trading game's self-play has every seat `+t`), and offers, answers and confirmations are recorded like any other searched decision." In `play_with`, replace the `cfg` literal with:

```rust
    // Every seat searches offers exactly when the game allows them (spec 3c Section 5).
    let cfg = SearchConfig {
        simulations: ex.full_simulations,
        batch,
        offers: config.max_offers_per_turn > 0,
        ..SearchConfig::default()
    };
```

- [ ] **Step 5: `--trades` in the arena, self-play and the fast loop**

In `alphasettler/arena.py`, after `SEED_LIMIT`, add:

```python
# The trading game (spec 3c Section 2): two domestic offers a turn, one card a side.
TRADES = {"max_offers_per_turn": 2, "max_trade_cards": 1}
```

In `alphasettler/cli.py`:
- Import: `from alphasettler.arena import SEED_LIMIT, TRADES, _write_records, check, format_summary, open_output, run`.
- In the arena parser replace the `--no-trades` line with:

```python
    rules = a.add_mutually_exclusive_group()
    rules.add_argument("--no-trades", action="store_true", help="disable domestic trade offers")
    rules.add_argument("--trades", action="store_true",
                       help="the trading game: 2 domestic offers a turn, 1 card a side (+t bots offer)")
```

- In the selfplay parser replace the `--trades` line with `s.add_argument("--trades", action="store_true", help="the trading game (2 offers a turn, 1 card a side); every seat may offer")`.
- In both `_selfplay` and `_selfplay_explore`, replace `config = {} if args.trades else {"max_offers_per_turn": 0}` with `config = dict(TRADES) if args.trades else {"max_offers_per_turn": 0}`.
- In the fast-loop parser add `fl.add_argument("--trades", action="store_true", help="first run only: self-play and gate in the trading game, every seat +t")`.
- In `_fast_loop`, pass `trades=args.trades` to `FastLoop.init(...)`, and after `loop = FastLoop(root, s)` add:

```python
    if args.trades and not loop.trades:
        print(f"error: {root} was set up without --trades, and a loop keeps the setting it was set up with",
              file=sys.stderr)
        return 2
```

- In `main`, replace the first `config = …` line with:

```python
    if getattr(args, "trades", False):
        config = dict(TRADES)
    elif getattr(args, "no_trades", False):
        config = {"max_offers_per_turn": 0}
    else:
        config = None
```

In `alphasettler/train/fastloop.py`:
- Add to the module docstring: "With trades (set when the loop is set up: `init(..., trades=True)`, `fast-loop --trades`), self-play and the gate play the trading game (spec 3c Section 5): every self-play seat may offer, and both gate sides are `+t`. A resumed loop keeps the setting it was set up with."
- Import `from alphasettler.arena import TRADES, run as arena_run`.
- Replace `__init__` with:

```python
    def __init__(self, root: str | Path, settings: FastLoopSettings):
        self.root = Path(root).resolve()  # the seed directories are recorded absolute
        self.s = settings
        self.nets = self.root / "nets"
        self.state = json.loads((self.root / STATE).read_text())
        self.trades = bool(self.state.get("trades", False))  # state files from before 3c have none
        rules = dict(TRADES) if self.trades else {"max_offers_per_turn": 0}
        self.config = {**rules, "max_turns": settings.max_turns}
```

- Change `init`'s signature to `def init(root: str | Path, seed_dirs: list[str | Path], start_net: str | Path, first: int, trades: bool = False) -> None:`, its docstring to end "…; generation numbering starts at `first`; `trades` plays the trading game.", and its state to `{"seed_dirs": [...], "first": first, "trades": trades, "best": "start", "latest": "start", "generations": []}`.
- In `_gate`, replace the candidate/baseline line with:

```python
        sims = self.s.gate_simulations
        offers = "+t" if self.trades else ""
        candidate, baseline = f"net:{gen_name(k)}@{sims}{offers}", f"net:{best}@{sims}{offers}"
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `~/.cargo/bin/cargo test --release -p settler-net && ~/.cargo/bin/cargo test --release -p settler-bots --test selfplay`
Expected: all pass, `fixed_exploration_is_3a_self_play` and `trading_self_play_records_offers_answers_and_confirmations` included.
Rebuild the extension, then run: `.venv/bin/pytest -q tests/python/test_cli.py tests/python/test_fast_loop.py tests/python/test_training_data.py`
Expected: all pass.

- [ ] **Step 7: Format, lint, full checks**

Run the Global Constraints' end-of-task checks. Expected as in Task 1.

- [ ] **Step 8: Commit**

```bash
git add net/src/data.rs net/src/evaluator.rs bots/src/selfplay.rs alphasettler/arena.py alphasettler/cli.py alphasettler/train/fastloop.py net/tests/data.rs net/tests/evaluator.rs bots/tests/selfplay.rs tests/python/test_cli.py tests/python/test_fast_loop.py
git diff --cached | grep -nE 'sk-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{36}|password=[^[:space:]]+'
git -c user.email=sean.may101@gmail.com commit -m "train: offers in the policy targets and prior, trading self-play, and the trading game (--trades)"
```

---

### Task 5: Catanatron trading — our offers, Catanatron's answers, our confirmation

**Files:**
- Modify: `oracle/translate.py`, `oracle/events.py`, `oracle/arena.py`, `alphasettler/cli.py` (`_catanatron_arena`)
- Create: `tests/python/test_oracle_trades.py`

**Interfaces:**
- Consumes (Task 4): `alphasettler.arena.TRADES`; (Task 1): `+t` names and searched confirmations.
- Produces:
  - `oracle.translate`: `OFFER_BASE` 258, `ACCEPT_TRADE` 658, `REJECT_TRADE` 659, `CONFIRM_BASE` 660, `CANCEL_TRADE` 664; `confirm_trade(p)`, `bundle(i) -> list[int]`, `bundle_index(counts) -> int`, `offer_trade(give, get) -> int`, `is_offer(a) -> bool`, `offer_value(a) -> tuple`, `offer_action(color, a)`, `our_game(seed, game, config)`. `action_token` translates `OFFER_TRADE`, `ACCEPT_TRADE`, `REJECT_TRADE`, `CONFIRM_TRADE` and `CANCEL_TRADE`. `catan_snapshot` returns phase `"main"` during `DECIDE_TRADE` / `DECIDE_ACCEPTEES`, with `offers_this_turn` counting the turn's offers made before any still pending.
  - `oracle.events.record_events` emits `trade_offered`, `trade_responded` (plus `trade_cancelled` after the last answer when nobody accepted), `trade_confirmed` and `trade_cancelled`.
  - `oracle.arena`: `let_alphabeta_answer_offers()`, `AlphaSettlerPlayer(color, bot_name, seed, nets_dir=None, config=None)`, `play_game(..., nets_dir=None, trades=False)`, `run_match(..., nets_dir=None, trades=False)`.

- [ ] **Step 1: Write the failing tests**

Create `tests/python/test_oracle_trades.py`:

```python
"""Domestic trades inside Catanatron (spec 3c Section 6): our offer, Catanatron's answers (AlphaBeta
by its own search), our confirmation, and the snapshot and events that carry them to our bot."""

import subprocess
import sys

import pytest

pytest.importorskip("catanatron")

from catanatron import Color, Game as CatanGame, Player  # noqa: E402
from catanatron.models.actions import generate_playable_actions  # noqa: E402
from catanatron.models.decks import freqdeck_add, freqdeck_subtract  # noqa: E402
from catanatron.models.enums import Action, ActionPrompt, ActionRecord, ActionType  # noqa: E402
from catanatron.players import AlphaBetaPlayer  # noqa: E402
from catanatron.state_functions import (  # noqa: E402
    get_player_freqdeck, player_freqdeck_add, player_freqdeck_subtract,
)

from alphasettler import describe_action  # noqa: E402
from alphasettler.arena import TRADES  # noqa: E402
from oracle.arena import AlphaSettlerPlayer, _force_seating, let_alphabeta_answer_offers, run_match  # noqa: E402
from oracle.events import record_events, redact  # noqa: E402
from oracle.translate import (  # noqa: E402
    ACCEPT_TRADE, CANCEL_TRADE, REJECT_TRADE, bundle, bundle_index, catan_snapshot, compare, confirm_trade, is_offer,
    offer_trade, offer_value, our_game, our_steps,
)


class StubBot:
    def __init__(self, result):
        self.result = result

    def act(self, game):
        return self.result


class Accepting(Player):
    """Accepts every offer it can; otherwise plays Catanatron's first playable action."""

    def decide(self, game, playable_actions):
        return next((a for a in playable_actions if a.action_type == ActionType.ACCEPT_TRADE), playable_actions[0])


def _set_hand(state, color, hand):
    old = get_player_freqdeck(state, color)
    player_freqdeck_subtract(state, color, old)
    state.resource_freqdeck = freqdeck_add(state.resource_freqdeck, old)
    player_freqdeck_add(state, color, hand)
    state.resource_freqdeck = freqdeck_subtract(state.resource_freqdeck, hand)


def _trading_table(seed=5):
    """Seat 0 is ours (a stub bot), on its turn after a roll of 6, holding one wood. Seat 1
    (AlphaBeta) and seat 2 (accepts everything) hold a brick each; seat 3 (AlphaBeta) holds nothing."""
    ours = AlphaSettlerPlayer(Color.RED, "random", 1, config=TRADES)
    players = [ours, AlphaBetaPlayer(Color.BLUE), Accepting(Color.ORANGE), AlphaBetaPlayer(Color.WHITE)]
    game = CatanGame(players, seed=seed)
    _force_seating(game, players)
    while game.state.is_initial_build_phase:
        game.play_tick(decide_fn=lambda player, g, actions: actions[0])
    st = game.state
    assert st.current_color() == Color.RED and st.current_prompt == ActionPrompt.PLAY_TURN
    roll = Action(Color.RED, ActionType.ROLL, None)
    game.execute(roll, action_record=ActionRecord(action=roll, result=(2, 4)))
    for color, hand in zip(st.colors, ([1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 0, 0, 0])):
        _set_hand(st, color, hand)
    game.playable_actions = generate_playable_actions(st)
    return ours, game


def test_trade_ids_match_the_engine():
    for g in range(20):
        for w in range(20):
            assert describe_action(offer_trade(g, w)) == f"OfferTrade {{ give: {g}, get: {w} }}"
    assert [describe_action(a) for a in (ACCEPT_TRADE, REJECT_TRADE, CANCEL_TRADE)] == [
        "AcceptTrade", "RejectTrade", "CancelTrade"]
    assert [describe_action(confirm_trade(p)) for p in range(4)] == [f"ConfirmTrade({p})" for p in range(4)]
    assert [bundle_index(bundle(i)) for i in range(20)] == list(range(20))
    # Bundle 11 is PAIRS[6] = (brick, sheep): a wood for a brick and a sheep.
    assert offer_value(offer_trade(0, 11)) == (1, 0, 0, 0, 0, 0, 1, 1, 0, 0)


def test_our_offer_is_answered_by_alphabeta_and_confirmed_by_us():
    let_alphabeta_answer_offers()
    ours, game = _trading_table()
    st = game.state
    mirror = our_game(0, game, TRADES)  # our engine, stepped alongside Catanatron
    offer = offer_trade(0, 1)  # one wood for one brick
    assert offer in mirror.legal_actions()
    ours.bot = StubBot(offer)
    chosen = ours.decide(game, game.playable_actions)
    assert chosen == Action(Color.RED, ActionType.OFFER_TRADE, offer_value(offer)) and ours.fallbacks == 0
    events = []
    before = catan_snapshot(game)

    def step(action):
        nonlocal before
        record = game.execute(action)
        for a, chance in our_steps(st, record):
            if chance is None:
                mirror.apply(a)
            else:
                mirror.apply_forced(a, chance)
        after = catan_snapshot(game)
        events.extend(record_events(before, after, record, st))
        before = after

    step(chosen)
    while st.current_prompt == ActionPrompt.DECIDE_TRADE:  # seats 1, 2, 3, in our order too
        step(st.current_player().decide(game, game.playable_actions))
    assert st.current_prompt == ActionPrompt.DECIDE_ACCEPTEES and st.acceptees[2]
    imported = our_game(0, game, TRADES)
    assert imported.phase == mirror.phase == "trade_confirm"
    assert imported.observation(0) == mirror.observation(0)
    assert {confirm_trade(2), CANCEL_TRADE} <= set(imported.legal_actions())
    ours.bot = StubBot(confirm_trade(2))
    confirm = ours.decide(game, game.playable_actions)
    assert confirm.action_type == ActionType.CONFIRM_TRADE and confirm.value[10] == Color.ORANGE
    step(confirm)
    assert ours.fallbacks == 0
    assert get_player_freqdeck(st, Color.RED) == [0, 1, 0, 0, 0]
    assert compare(mirror.snapshot(), catan_snapshot(game)) == []
    assert [e["type"] for e in events] == ["trade_offered"] + ["trade_responded"] * 3 + ["trade_confirmed"]
    for viewer in range(4):
        assert mirror.log(viewer) == [redact(e, viewer) for e in events], viewer


def test_offers_nobody_can_take_are_cancelled_and_a_turn_allows_two():
    ours, game = _trading_table()
    st = game.state
    for n in (1, 2):
        ours.bot = StubBot(offer_trade(0, 4))  # a wood for an ore nobody holds
        before = catan_snapshot(game)
        record = game.execute(ours.decide(game, game.playable_actions))
        events = record_events(before, catan_snapshot(game), record, st)
        while st.current_prompt == ActionPrompt.DECIDE_TRADE:
            before = catan_snapshot(game)
            record = game.execute(st.current_player().decide(game, game.playable_actions))
            events += record_events(before, catan_snapshot(game), record, st)
        assert [e["type"] for e in events] == ["trade_offered"] + ["trade_responded"] * 3 + ["trade_cancelled"]
        assert events[-1] == {"type": "trade_cancelled", "player": 0}
        assert st.current_prompt == ActionPrompt.PLAY_TURN
        assert catan_snapshot(game)["offers_this_turn"] == n
    assert not any(is_offer(a) for a in our_game(0, game, TRADES).legal_actions()), "two offers a turn"
    assert ours.fallbacks == 0


def test_a_trading_bot_plays_whole_games_inside_catanatron(tmp_path):
    records = run_match("ismcts@10+t", "random", seeds=1, workers=4, trades=True)
    assert len(records) == 4
    assert all(r["fallbacks"] == 0 and r["belief_resets"] == 0 for r in records), records
    out = tmp_path / "c.jsonl"
    r = subprocess.run([sys.executable, "-m", "alphasettler", "arena", "--candidate", "ismcts@10+t", "--baseline",
                        "catanatron:random", "--seeds", "1", "--trades", "--out", str(out)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "fallbacks: 0" in r.stdout and "belief resets: 0" in r.stdout
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest -q tests/python/test_oracle_trades.py`
Expected: collection error, `cannot import name 'let_alphabeta_answer_offers'`.

- [ ] **Step 3: Translation**

In `oracle/translate.py`:
- Change the module docstring's first line to "Translation between Catanatron 3.3 and AlphaSettler: boards, seats, actions (domestic trades included), legal sets and full positions."
- Imports: `from catanatron.models.enums import CITY, RESOURCES, SETTLEMENT, Action, ActionPrompt, ActionType` and `from alphasettler import Game`.
- After `def maritime(...)`, add:

```python
# Domestic trades (engine/src/action.rs): an offer is OFFER_BASE + 20 x give bundle + get bundle,
# where bundles 0..5 are one card of a resource and 5..20 the PAIRS.
OFFER_BASE, ACCEPT_TRADE, REJECT_TRADE, CONFIRM_BASE, CANCEL_TRADE = 258, 658, 659, 660, 664


def confirm_trade(p): return CONFIRM_BASE + p
def offer_trade(give, get): return OFFER_BASE + 20 * give + get
def is_offer(a): return OFFER_BASE <= a < ACCEPT_TRADE


def bundle(i: int) -> list[int]:
    """Bundle `i` as counts per resource."""
    counts = [0] * 5
    for r in ((i,) if i < 5 else PAIRS[i - 5]):
        counts[r] += 1
    return counts


def bundle_index(counts) -> int:
    cards = [r for r in range(5) for _ in range(counts[r])]
    if len(cards) == 1:
        return cards[0]
    if len(cards) == 2:
        return 5 + PAIRS.index((cards[0], cards[1]))
    raise Untranslatable("domestic-trade", f"a side of {len(cards)} cards (ours holds 1 or 2)")


def offer_value(a: int) -> tuple[int, ...]:
    """Catanatron's OFFER_TRADE value for our offer id: the give counts, then the get counts."""
    k = a - OFFER_BASE
    return tuple(bundle(k // 20) + bundle(k % 20))


def offer_action(color, a: int):
    return Action(color, ActionType.OFFER_TRADE, offer_value(a))
```

- In `action_token`, replace the final `raise` with:

```python
    if t == ActionType.OFFER_TRADE:
        return offer_trade(bundle_index(v[:5]), bundle_index(v[5:10]))
    if t == ActionType.ACCEPT_TRADE:
        return ACCEPT_TRADE
    if t == ActionType.REJECT_TRADE:
        return REJECT_TRADE
    if t == ActionType.CONFIRM_TRADE:
        return confirm_trade(seat(state, v[10]))
    if t == ActionType.CANCEL_TRADE:
        return CANCEL_TRADE
    raise Untranslatable("domestic-trade", f"{t} has no translation")
```

- In `_phase`, before the final `raise`, add:

```python
    if prompt in (ActionPrompt.DECIDE_TRADE, ActionPrompt.DECIDE_ACCEPTEES):
        # A pending domestic offer: the snapshot is the Main position it was made from
        # (`our_game` replays the offer and its answers onto it).
        return "main", {}
```

- After `_bought_this_turn`, add:

```python
def _offers_made(st) -> int:
    """Domestic offers the current player made this turn, not counting one still pending (a
    snapshot is the position an offer was made from), read from the action records."""
    color = st.colors[st.current_turn_index]
    n = 0
    for record in reversed(st.action_records):
        t = record.action.action_type
        if t == ActionType.END_TURN:
            break
        if t == ActionType.OFFER_TRADE and record.action.color == color:
            n += 1
    return n - int(st.is_resolving_trade)
```

- In `catan_snapshot`, replace the `offers_this_turn` entry with `"offers_this_turn": _offers_made(st),`.
- After `catan_snapshot`, add:

```python
def our_game(seed: int, game, config: dict) -> Game:
    """Catanatron's position as our engine's ``Game`` under ``config``. Our snapshots cannot hold
    a pending domestic offer, so one is replayed onto the position it was made from: the offer,
    then every opponent's answer in our seat order (which leaves the same position as
    Catanatron's order). Only an offer every opponent has answered (DECIDE_ACCEPTEES: the offerer
    to confirm) is a position we play; one still collecting answers raises Untranslatable."""
    ours = Game.from_snapshot(seed, catan_snapshot(game), config)
    st = game.state
    if not st.is_resolving_trade:
        return ours
    if st.current_prompt != ActionPrompt.DECIDE_ACCEPTEES:
        raise Untranslatable("domestic-trade", "an offer still collecting answers")
    v = st.current_trade
    ours.apply(offer_trade(bundle_index(v[:5]), bundle_index(v[5:10])))
    offerer = st.current_turn_index
    for k in range(1, 4):
        ours.apply(ACCEPT_TRADE if st.acceptees[(offerer + k) % 4] else REJECT_TRADE)
    return ours
```

- [ ] **Step 4: Events**

In `oracle/events.py`, before the final `else:` of `record_events`, add:

```python
    elif t == ActionType.OFFER_TRADE:
        ev.append({"type": "trade_offered", "player": p, "give": list(v[:5]), "get": list(v[5:10])})
    elif t in (ActionType.ACCEPT_TRADE, ActionType.REJECT_TRADE):
        ev.append({"type": "trade_responded", "player": p, "accepted": t == ActionType.ACCEPT_TRADE})
        if not state.is_resolving_trade:  # the last answer, and nobody accepted
            ev.append({"type": "trade_cancelled", "player": state.current_turn_index})
    elif t == ActionType.CONFIRM_TRADE:
        ev.append({"type": "trade_confirmed", "offerer": p, "partner": seat(state, v[10]),
                   "offerer_gave": list(v[:5]), "partner_gave": list(v[5:10])})
    elif t == ActionType.CANCEL_TRADE:
        ev.append({"type": "trade_cancelled", "player": p})
```

Catanatron asks responders in seat-index order, ours in seat order after the offerer. The events go out in Catanatron's order, which the belief does not depend on (each answer only filters by the same pending get).

- [ ] **Step 5: The arena**

In `oracle/arena.py`:
- Add to the module docstring: "With trades on (`run_match(..., trades=True)`, `arena --trades`) our candidate plays the trading game: a `+t` candidate may offer, Catanatron's bots answer (AlphaBeta by its own search, see `let_alphabeta_answer_offers`), and our bot confirms or cancels."
- Imports: `from catanatron.models.enums import ActionType`, `from catanatron.players import tree_search_utils`, `from alphasettler import Bot`, `from alphasettler.arena import TRADES`, and from `oracle.translate`: `SETUP_TURNS, Untranslatable, action_token, catan_snapshot, is_offer, move_robber, offer_action, our_game, seat, steal_from` (`Game` is no longer imported here).
- After `CONFIG`, add:

```python
# AlphaBeta expands only the action types its search knows how to execute, and raises on the
# answers to a domestic offer: Catanatron's own bots never offer, so they never meet one. The
# answers and confirmations are deterministic (no dice, no hidden draw), so with our offers on they
# join that set, and AlphaBeta answers by its own search (spec 3c Section 6). Catanatron's source is
# not changed; the set is extended at run time in the processes that play trading games.
TRADE_ANSWERS = (ActionType.ACCEPT_TRADE, ActionType.REJECT_TRADE, ActionType.CONFIRM_TRADE,
                 ActionType.CANCEL_TRADE)


def let_alphabeta_answer_offers() -> None:
    tree_search_utils.DETERMINISTIC_ACTIONS.update(TRADE_ANSWERS)
```

- `AlphaSettlerPlayer.__init__(self, color, bot_name: str, seed: int, nets_dir: str | None = None, config: dict | None = None)` sets `self.config = CONFIG if config is None else config`. In `decide`, replace `ours = Game.from_snapshot(self.seed, catan_snapshot(game), CONFIG)` with:

```python
            ours = our_game(self.seed, game, self.config)
            if ours.phase == "main":
                # Catanatron lists no offers (any one is valid in PLAY_TURN after the roll): ours
                # are the ones our rules allow, none when the config turns offers off.
                for a in ours.legal_actions():
                    if is_offer(a):
                        by_token[a] = offer_action(self.color, a)
```

- `play_game(seed, candidate, baseline, candidate_seat, nets_dir=None, trades=False)`: at its start add `if trades:` / `let_alphabeta_answer_offers()`, and build our player with `AlphaSettlerPlayer(COLORS[i], candidate, seed * 4 + i, nets_dir, TRADES if trades else CONFIG)`.
- `run_match(candidate, baseline, seeds, seed_start=0, workers=None, nets_dir=None, trades=False)`: jobs become `(s, candidate, baseline, k, nets_dir, trades)`.

In `alphasettler/cli.py`, `_catanatron_arena`: change the docstring to "Our candidate inside Catanatron against three of its bots. `--trades` plays the trading game (a `+t` candidate offers, Catanatron's bots answer); otherwise ours plays with offers off, and `--no-trades` is moot: Catanatron's bots never offer." and the run to `records = oracle.arena.run_match(args.candidate, baseline, args.seeds, args.seed_start, args.threads, nets_dir, trades=args.trades)`.

- [ ] **Step 6: Run the tests to verify they pass**

Run: `.venv/bin/pytest -q tests/python/test_oracle_trades.py tests/python/test_oracle_arena.py tests/python/test_oracle_events.py tests/python/test_oracle_translate.py tests/python/test_oracle_diff.py`
Expected: all pass; `test_oracle_trades.py` 4 passed.

- [ ] **Step 7: Format, lint, full checks**

Run the Global Constraints' end-of-task checks (no Rust changed: the extension needs no rebuild). Expected as in Task 1.

- [ ] **Step 8: Commit**

```bash
git add oracle/translate.py oracle/events.py oracle/arena.py alphasettler/cli.py tests/python/test_oracle_trades.py
git diff --cached | grep -nE 'sk-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{36}|password=[^[:space:]]+'
git -c user.email=sean.may101@gmail.com commit -m "oracle: domestic trades inside Catanatron (our offers, AlphaBeta's answers, our confirmation) and arena --trades"
```

---

### Task 6: The trading net and the done criteria

**Files:**
- Create: `docs/bot/results-3c.md`
- Run outputs (not committed): `runs/3c/` (`gen-004-v3.bin`, `state.json`, `nets/`, `gen-030` … `gen-039`, the logs and arena records)

**Interfaces:**
- Consumes: `alphasettler net-upgrade` (Task 3); `fast-loop --trades` (Task 4); `arena --trades` native and Catanatron (Tasks 4–5); `alphasettler.stats.paired`.
- Produces: `runs/3c/nets/T.bin` (the trading net: the loop's best), `runs/3c/nets/gen-004.bin` (upgraded gen-004), the three criteria's records, `docs/bot/results-3c.md`.

Every step but Step 8 is run by the controller in the background. Every `fast-loop` command resumes where it stopped: after a Ctrl-C, a crash or a reboot, rerun it with the same `--dir` and `--generations`.

- [ ] **Step 1: Build at Task 5's commit (controller runs this)**

```bash
git log -1 --format=%s
~/.cargo/bin/cargo build --release -p settler-bots
pgrep -fl alphasettler || true
```

Then rebuild the extension per the Global Constraints (atomic rename if `pgrep` printed a process). Pass: the commit subject is Task 5's; every command exits 0. A process still running from before Task 2 keeps its v2 library mapped and is unaffected. Any process started from here on refuses v2 nets.

- [ ] **Step 2: Upgrade gen-004 (controller runs this)**

```bash
test ! -e runs/3c && mkdir -p runs/3c
.venv/bin/alphasettler net-upgrade runs/3b-v2/nets/gen-004.bin runs/3c/gen-004-v3.bin 2>&1 | tee runs/3c/upgrade.log
shasum -a 256 runs/3b-v2/nets/gen-004.bin
```

Pass: `upgraded runs/3b-v2/nets/gen-004.bin (encoder v2) to encoder v3: width 48, 2 rounds; 24 new global inputs and 407 trade policy rows zeroed (offer bias -3); wrote runs/3c/gen-004-v3.bin`, and the source's sha256 is `7bd4cb337c1a28bdab5cab98a7ece628d33d14f6190a3a7203a57b3cdbe8470b` (it is gen-004 as 3b left it).

- [ ] **Step 3: The trading loop (controller runs this)**

```bash
.venv/bin/alphasettler fast-loop --dir runs/3c --generations 39 --first 30 --trades \
  --seed-dirs runs/3b-v2/gen-002 runs/3b-v2/gen-003 runs/3b-v2/gen-004 runs/3b-v2/gen-005 \
  --start-net runs/3c/gen-004-v3.bin 2>&1 | tee -a runs/3c/fast-loop.log
```

To resume: `.venv/bin/alphasettler fast-loop --dir runs/3c --generations 39 2>&1 | tee -a runs/3c/fast-loop.log` (the loop keeps `--trades` from `state.json`).

Ten generations (30–39), each 2,000 trading self-play games with the best net (generation 30 with the upgraded gen-004), 10 minutes of training on the newest 40,000 games (the four 3b directories, 42,400 no-trade games, turning over to trade games as generations accumulate), and the regression gate (100 seeds from 1,000,000,000 at 100 simulations, both sides `+t`, promoted unless z at most −1). Generation k's self-play seeds are k × 10,000,000 (300M–390M), clear of 3b's (0–59M), the other fast loop's (60M on) and the measurement seeds. Expect about two hours per generation.
Pass: exit 0 and `10 generations run; the best net is gen-0NN (runs/3c/nets/best.bin)` (or `start` if no candidate was promoted); every self-play line shows `belief resets 0`.

Then:

```bash
cp runs/3c/nets/best.bin runs/3c/nets/T.bin
cp runs/3c/gen-004-v3.bin runs/3c/nets/gen-004.bin
.venv/bin/python3 -c "
import json
from pathlib import Path
from alphasettler import records
s = json.loads(Path('runs/3c/state.json').read_text())
print('best', s['best'], 'trades', s['trades'])
for g in s['generations']:
    k = g['gen']
    games = offers = confirmed = 0
    for p in sorted(Path(f'runs/3c/gen-{k:03d}').glob('selfplay-local-*.jsonl.gz')):
        for game in records.read(p):
            games += 1
            offers += sum(258 <= a < 658 for a in game['actions'])
            confirmed += sum(660 <= a < 664 for a in game['actions'])
    t, gate = g['train'], g['gate']
    print(f\"| {k} | {g['selfplay']['net']} | {games} | {offers / games:.2f} | {confirmed / games:.2f} | {t['steps']} | \"
          f\"{t['best']['score']} | {gate['win_rate']} | {gate['z']} | {gate['promoted']} |\")
" | tee runs/3c/generations.md
```

Pass: `trades True`, ten table rows; record whether offers per game is above 0 in the later generations (it is the evidence the net learned to offer; not a gate).

- [ ] **Step 4: Done criterion 1 — trading pays against our own bots (controller runs this)**

```bash
.venv/bin/alphasettler arena --candidate net:T@600+t --baseline net:T@600 --trades --seeds 200 \
  --seed-start 2000000000 --nets-dir runs/3c/nets --out runs/3c/c1-T600t-vs-T600.jsonl 2>&1 | tee runs/3c/c1.log
```

Pass (spec Section 1.1): the summary line's win rate is above 0.25 and its `z vs 0.25` above 3; `belief resets: 0`.

- [ ] **Step 5: Done criterion 3 — no regression without trades (controller runs this)**

```bash
.venv/bin/alphasettler arena --candidate net:T@600 --baseline net:gen-004@600 --no-trades --seeds 200 \
  --seed-start 2000000000 --nets-dir runs/3c/nets --out runs/3c/c3-T600-vs-gen004.jsonl 2>&1 | tee runs/3c/c3.log
```

Pass (spec Section 1.3): `z vs 0.25` above −2; `belief resets: 0`. `net:gen-004` here is the upgraded file (the v3 build refuses the original); its no-trade moves are the original's.

- [ ] **Step 6: Done criterion 2 — trading pays against AlphaBeta (controller runs this)**

Both arms play the trading game, so the only difference is the candidate's offers (the encoder's "offers allowed" input is 1 on both sides). Each arm takes about five hours.

```bash
.venv/bin/alphasettler arena --candidate net:T@1000+t --baseline catanatron:alphabeta --trades --seeds 200 \
  --seed-start 2000000000 --nets-dir runs/3c/nets --out runs/3c/c2-T1000t-vs-alphabeta.jsonl 2>&1 | tee runs/3c/c2.log
.venv/bin/alphasettler arena --candidate net:T@1000 --baseline catanatron:alphabeta --trades --seeds 200 \
  --seed-start 2000000000 --nets-dir runs/3c/nets --out runs/3c/c2-T1000-vs-alphabeta.jsonl 2>&1 | tee -a runs/3c/c2.log
.venv/bin/python3 -c "
import json
from alphasettler.stats import paired
a = [json.loads(l) for l in open('runs/3c/c2-T1000t-vs-alphabeta.jsonl')]
b = [json.loads(l) for l in open('runs/3c/c2-T1000-vs-alphabeta.jsonl')]
p = paired(a, b)
print(f'paired net:T@1000+t - net:T@1000 vs catanatron:alphabeta, trading game: mean diff {p.mean_diff:+.3f} '
      f'± {p.se:.3f} (z = {p.z:.2f}) over {p.seeds} seeds')
" | tee -a runs/3c/c2.log
```

Pass (spec Section 1.2): the paired mean difference is above 0 with z above 2; both arms print `fallbacks: 0` and `belief resets: 0`.

If any criterion fails, still write the results with the numbers and the failure stated plainly. Do not rerun on the same seeds after changing anything.

- [ ] **Step 7: Write `docs/bot/results-3c.md`**

Every number is copied from the logs and files named here; nothing is estimated. The file:

```markdown
# Sub-project 3c (domestic trading): results

Date: <date of the last criterion run>. Machine: <`sysctl -n machdep.cpu.brand_string`>, macOS
(<`uname -r`>). Commit: <`git log -1 --format=%h`> (Plan 6, Task 5), plus the commit adding this file.
Catanatron 3.3.0 at ecf931181b9a65bb4116a2153fb78c16f1438e00.

Spec: [`docs/superpowers/specs/2026-10-07-trading-design.md`](../superpowers/specs/2026-10-07-trading-design.md).
Plan: [`docs/superpowers/plans/2026-10-07-plan-6-trading.md`](../superpowers/plans/2026-10-07-plan-6-trading.md).

## Configuration

The trading game: 2 domestic offers per turn, 1 card a side (`--trades`). `+t` lets a bot offer.
Without it a bot answers and confirms by searching but never offers. The network is encoder v3
(115 globals: the trade block). Its start is gen-004 rewritten by `alphasettler net-upgrade`
(<the line of runs/3c/upgrade.log>).

## The trading loop

`fast-loop --trades`, generations 30 to 39 in `runs/3c`, window 40,000 games, starting from
runs/3b-v2 gen-002 to gen-005. Best net: <best from runs/3c/state.json>, copied to
`runs/3c/nets/T.bin`.

| gen | self-play net | games | offers / game | trades / game | train steps | held-out score | gate win rate | gate z | promoted |
|---|---|---|---|---|---|---|---|---|---|
<the ten rows of runs/3c/generations.md>

## Done criteria (200 fresh seeds from 2,000,000,000, four seat rotations each)

| Criterion | Command | Result | Bar | Met |
|---|---|---|---|---|
| 1. trading pays against our bots | `arena --candidate net:T@600+t --baseline net:T@600 --trades` | <summary line of runs/3c/c1.log> | win rate above 0.25, z above 3 | <yes/no> |
| 2. trading pays against AlphaBeta | `arena --candidate net:T@1000[+t] --baseline catanatron:alphabeta --trades`, both arms | <both summary lines and the paired line of runs/3c/c2.log> | paired diff above 0, z above 2 | <yes/no> |
| 3. no regression without trades | `arena --candidate net:T@600 --baseline net:gen-004@600 --no-trades` | <summary line of runs/3c/c3.log> | z above −2 | <yes/no> |

Belief resets and fallbacks: <the belief resets and fallbacks lines of each run>.

## Rulings the build made

- The engine is unchanged. Worlds at trade roots, and positions imported from Catanatron, start
  from the Main position the offer was made from; the pending trade is then set or replayed.
- AlphaBeta's search raises on trade answers in Catanatron 3.3. The oracle adds the four answer
  types to its deterministic set at run time, so AlphaBeta answers by its own search.
- Criterion 3's baseline is gen-004 upgraded to encoder v3, because the v3 build refuses v2 files.
- The upgrade resets all 400 offer rows (bias −3) and the seven answer and confirmation rows
  (bias 0). Only the 20 one-for-one offers are legal in the trading game.

## Next

<one paragraph: if criterion 1 holds, the spec's first follow-up (max_trade_cards 2); otherwise what the numbers point to, from the table above>
```

Fill every `<...>` from the named source before committing. The committed file holds numbers and text only, with no angle brackets.

- [ ] **Step 8: Commit**

Run: `grep -c '<' docs/bot/results-3c.md`
Expected: `0`.

```bash
git add docs/bot/results-3c.md
git diff --cached | grep -nE 'sk-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{36}|password=[^[:space:]]+'
git -c user.email=sean.may101@gmail.com commit -m "docs: 3c trading results"
```

---

## Self-review

### Spec coverage

| Spec requirement | Task |
|---|---|
| §1 done criteria 1–3, 200 fresh seeds from 2,000,000,000, four rotations | 6 (Steps 4–6) |
| §2 the trading game `{"max_offers_per_turn": 2, "max_trade_cards": 1}`, `arena --trades` beside `--no-trades` | 4 (`TRADES`, exclusive group, test) |
| §2 engine unchanged, `golden_trace` passes | Global Constraints; every task's full test run |
| §3 offers searched when the bot may offer, the same rule for every seat in the tree | 1 (`SearchConfig::offers`, `search_moves`, `descend`; `offers_are_searched_only_when_the_config_allows_them`) |
| §3 answers and confirmations searched by every searching bot; 3a digests unchanged | 1 (`act_with`; `answers_and_confirmations_are_searched`; Step 7) |
| §3 `+t` with `+rD` / `+sK` in any order; answer-only without it | 1 (`split_trades`, `parse_net_name`, `make_bot`, `Roster`; name tests; `trading_games_play_through_and_only_the_offering_seat_offers`) |
| §3 world sampling at trade roots; belief conditions on offers and acceptances | 1 (`WorldSampler::new`, `deal_hidden_hands`; two world tests) |
| §4 encoder v3: 5 + 5 + 12 + 1 + 1, widths 115 / 1581, new hash, v2 refused, format 2 | 2 |
| §4 `net-upgrade OLD NEW`: zero new global columns, zero trade rows, OFFER_BIAS −3.0 for offers, 0 for the rest; parity on no-trade positions | 3 |
| §5 offers in mask and targets; 3b batches unchanged | 4 (`Batch::push`, `check_game`; 3b records hold no offers) |
| §5 trade decisions recorded in self-play | 4 (`trading_self_play_records_offers_answers_and_confirmations`) |
| §5 `fast-loop --trades`: every seat `+t`, window starts with 3b's generations, gate in the trading game with both sides `+t`; generation 1 from upgraded gen-4 | 4 (FastLoop), 6 (Step 3) |
| §6 our offer to OFFER_TRADE, DECIDE_ACCEPTEES to our TradeConfirm, pending trade and `offers_this_turn` in the imported position, trade events | 5 |
| §6 Catanatron arena `--trades`, AlphaBeta answers with its own search | 5 (`let_alphabeta_answer_offers`, CLI) |
| §8 testing list | 1–5 as listed in each task's Step 1 |

### Placeholder scan

Searched for "TBD", "TODO", "implement later", "fill in", "similar to Task", "appropriate", "handle edge cases": none in the code steps. The one template is Task 6 Step 7's results file, whose `<...>` marks are run-time measurements, each naming its source; Step 8 refuses to commit while one is left.

### Type consistency

- `SearchConfig.offers: bool`, `search_moves(&[Action], bool, &mut Vec<Action>)`, `IsmctsBot::with_offers(bool)`, `split_trades(&str) -> Option<(String, bool)>` and the 5-tuple `parse_net_name` are defined in Task 1 and used with those names in Task 1's `lib.rs`, Task 4's `selfplay.rs` and the tests.
- `TRADE_GIVE`, `TRADE_GET`, `TRADE_RESPONSES`, `RESPONSE_WIDTH`, `OFFERS_MADE`, `OFFERS_ALLOWED` (Task 2) are used only in `encode.rs` and `net/tests/encode.rs`.
- `V2_ENCODING_HASH` / `V2_GLOBAL_WIDTH` (Rust, Task 3) are exported as `NET_V2_ENCODING_HASH` / `NET_V2_GLOBAL_WIDTH` and used by `model.py`, `netupgrade.py` and `test_net_upgrade.py`. The loader message "it is an encoder v2 net (before the trade block): convert it with `alphasettler net-upgrade OLD NEW`" is identical in Rust and Python, and the tests match "net-upgrade".
- `TRADES` (Task 4, `alphasettler/arena.py`) is the one definition used by `cli.py`, `fastloop.py`, `oracle/arena.py` and the tests. `FastLoop.init(..., trades=False)` and `FastLoop.trades` agree between `fastloop.py`, `cli.py` and `test_fast_loop.py`.
- Oracle names (`offer_trade`, `confirm_trade`, `is_offer`, `offer_value`, `offer_action`, `bundle`, `bundle_index`, `our_game`, `let_alphabeta_answer_offers`, `run_match(..., trades=)`) are defined in Task 5 and used with those names in `arena.py`, `cli.py` and `test_oracle_trades.py`.

### Review Focus

Each of the five lines has its test in the owning task (1: Task 3; 2 and 3: Task 1; 4: Task 4; 5: Task 5). Checked and not added: a third offer in the native game (the engine's own `legal_offers` limit, already tested in `engine/tests/trade.rs`); an opponent's offer reaching our bot inside Catanatron (Catanatron's bots never offer; `our_game` raises Untranslatable, a counted fallback).
