# Plan 13: JSettlers2 Smart Robots as an Arena Opponent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Play an AlphaSettler bot as one seat against three stock JSettlers2 2.6.10 *smart* robots inside JSettlers' own server (`alphasettler arena --candidate BOT --baseline jsettlers:smart [--trades] [--aligned]`), check the translation by a differential replay of robots-only games (`alphasettler jsettlers-diff`), and measure `net:B@1000+t+o` in three arms (native trades, aligned trades, no trades) beside the Catanatron AlphaBeta line in `docs/bot/results-jsettlers.md`.

**Architecture:** A Java harness under `jsettlers/` (GPL-3.0-or-later, compiled by plain `javac` against the downloaded 2.6.10 server jar) runs one JVM per arena worker: `ArenaServer extends SOCServer` on a string port creates each game with fixed options, our seed's board (`ArenaBoard` through `ArenaBoardFactory`), routed random streams (`RoutedRandom` in `SOCGame.rand`) and fixed seats, and streams every recorded game event (its GameEventLog line, with the server-side position in JSettlers coordinates whenever it changed) as JSON lines on stdout. Three stock robots sit beside a `BridgeClient` whose `BridgeBrain extends SOCRobotBrain` keeps the stock turn machine and asks Python for every decision. In Python, `oracle/jsettlers/` (stdlib only, no Catanatron) maps coordinates (`tables`), positions (`translate`), the message stream (`messages`, `events.Tracker`: our steps with forced chance outcomes and our bot's events) and drives games (`arena.Seat`, `run_match`); `diff` replays robots-only games through our engine. The engine gains `GameConfig::bank_limit` (default on, so nothing existing changes) because JSettlers' bank is unlimited.

**Tech Stack:** Rust 1.80+ (edition 2021, no new crates), PyO3 0.29.3 + maturin 1.15.0, Python 3.12 stdlib, pytest 9.1.1; Java 20 (`javac --release 17`) against `jsettlers-2.6.10-server.jar` only.

**Spec:** `docs/superpowers/specs/2026-10-09-jsettlers2-arena-design.md` (binding). Research behind it: `.superpowers/sdd/jsettlers2-research.md` (JSettlers facts, the three routes, the rulings). Builds on the Catanatron arena (`oracle/{translate,events,arena,diff,allowlist}.py`, `alphasettler/cli.py` `_catanatron_arena` and `_oracle_diff`, `.superpowers/sdd/catanatron-reference.md`). Written against HEAD 1e95ddb. Every JSettlers member this plan calls was read in the `release-2.6.10` sources (`src/main/java/soc/...`); the members only reflection can reach are listed in `jsettlers/api.txt` and checked against the jar by `jsettlers/check-api.sh` before every build.

## Global Constraints

- Python is `.venv/bin/python3`, tests `.venv/bin/pytest`, the CLI `.venv/bin/alphasettler`. Never `python`, `pip`, `python3` outside the venv, or `pip3` outside the venv.
- Rust is `PATH=$HOME/.cargo/bin:$PATH cargo ...` (cargo is not on PATH; `~/.cargo/bin/cargo` is the same binary). MSRV 1.80 (`rust-version = "1.80"`): use nothing stabilised after 1.80. No new external crates and no new Python dependencies: `oracle/jsettlers/` is stdlib Python and the harness is plain Java against the one JSettlers jar. No numpy anywhere.
- Rebuild the extension after any Rust change used from Python. Another measurement may hold the extension: **never run `maturin develop`**. Build a wheel and rename the new library over the old one (an atomic rename, so running processes keep their mapped copy):
  ```bash
  rm -rf target/wheels-js && PATH=$HOME/.cargo/bin:$PATH .venv/bin/maturin build --release -q -o target/wheels-js \
    && unzip -q -o target/wheels-js/alphasettler-*.whl 'alphasettler/_engine*.so' -d target/wheels-js/x \
    && cp target/wheels-js/x/alphasettler/_engine.cpython-312-darwin.so alphasettler/._engine.new.so \
    && mv -f alphasettler/._engine.new.so alphasettler/_engine.cpython-312-darwin.so
  ```
  "Rebuild the extension" in any step below means exactly this command.
- Every task ends with `PATH=$HOME/.cargo/bin:$PATH cargo fmt --all` then `PATH=$HOME/.cargo/bin:$PATH cargo fmt --check` (no output), `PATH=$HOME/.cargo/bin:$PATH cargo clippy --workspace --release --all-targets -- -D warnings` (exit 0), the full `PATH=$HOME/.cargo/bin:$PATH cargo test --workspace --release` (all pass), `bash jsettlers/build.sh` (exit 0; from Task 1 on) and, after the rebuild when Rust changed, `.venv/bin/pytest -q -rs` (all pass, the summary lists no skipped tests).
- Timed machine: a timed measurement (`runs/p2/arm.sh`, the 3d-4 arms) may be running on this machine. Before any `cargo`, `maturin`, `pytest`, `javac` or `jsettlers/build.sh` command, run `pgrep -fl 'arm.sh'`; if it prints anything, wait (poll every 5 minutes) until it prints nothing. Controller-run steps check it too.
- Java: Java 20 is installed (`java -version`). Compile only through `jsettlers/build.sh` (plain `javac --release 17 -Xlint:all,-serial,-deprecation,-removal,-processing,-path -Werror -cp jsettlers/lib/jsettlers-2.6.10-server.jar`). No Gradle, no Maven, no other jar.
- The JSettlers jar is downloaded by `scripts/fetch-jsettlers.sh`, which checks its sha256 (`9b26cb20ec943213600795d660fad2da32fc72629c348e180bd171cc26d6b4e2`), and is never committed: `jsettlers/lib/` and `jsettlers/build/` are in `.gitignore`, and `git status --porcelain` never lists them.
- License: every `.java` file under `jsettlers/src/` starts with the five-line GPL-3.0-or-later header of Task 1 (its second line is `// SPDX-License-Identifier: GPL-3.0-or-later`), and `jsettlers/COPYING` is the notice. Only `jsettlers/` holds code that extends or links JSettlers; Python talks to it through JSON lines on the JVM's stdin and stdout (`oracle/jsettlers/jvm.py`), never by import. `oracle/jsettlers/` imports neither Catanatron nor `oracle.{translate,events,arena,diff,allowlist,play,tables}` (it may import `oracle.compare`, which imports no other project's engine).
- Never re-pin a digest. `bank_limit` (Task 2) defaults on, so every existing game, digest, record file and resolved config is unchanged: `engine/tests/golden_trace.rs` and these pins pass unchanged after every task: `the_generic_refactor_leaves_3a_ismcts_games_unchanged` and `bots_without_the_gumbel_suffix_play_the_games_they_played_before_p2` (`bots/tests/ismcts.rs`), `net_bots_without_the_gumbel_suffix_play_the_games_they_played_before_p2` (`bots/tests/netbot.rs`), `fixed_exploration_is_3a_self_play`, `standard_exploration_plays_the_games_it_played_before_p2`, `self_play_without_an_opening_model_plays_the_games_it_played_before_3d1b` and `self_play_without_a_league_plays_the_games_it_played_before_3d3` (`bots/tests/selfplay.rs`). `resolve_config` writes `bank_limit` only when it is off.
- The trading game is exactly `{"max_offers_per_turn": 2, "max_trade_cards": 1}` (`TRADES` in `alphasettler/arena.py`); JSettlers positions import with `"bank_limit": False` added (and `"max_offers_per_turn": 0` in the no-trades arm and the differential replay).
- The spec's fixed choices: three stock smart robots named `robot 1`..`robot 3` (`SOCServer.getRobotParameters` gives every `robot ` name `SMART_STRATEGY`), our seat JSettlers seat 0 named `robot bridge` (the `robot ` prefix gives its stock fallbacks the smart parameters); game options 4 players, 10 VP, the classic board, `BC` off on injected boards, `NT` on only in the no-trades arm, `PLAY_FO` off; `SOCRobotBrain.BOTS_ONLY_FAST_PAUSE_FACTOR = 0.01f`; `jsettlers.bots.timeout.turn` 600; robots wait 5 s for an answer, ours counts as late from 4.5 s.
- Determinism: a seed fixes the board, the dice, the deck shuffle, the steals' stream and our seat's turn position, in every rotation and every arm (common random numbers). The robots' own randomness and thread timing stay free, so a game does not replay exactly (recorded, as for Catanatron's time-limited AlphaBeta). Bot seeds are `seed * 4 + k` for turn position k, as in the Catanatron arena.
- Git: one commit per task with the message given in the task, run as `git -c user.email=sean.may101@gmail.com commit -m "..."`. No Co-Authored-By or AI attribution, never `--no-verify`, never push. Before every commit run `git diff --cached | grep -nE 'sk-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{36}|password=[^[:space:]]+'` and expect no output. Stage only the files the task names: other work may be committed in this repository concurrently.
- Testing: integration tests over mocks (real JVM games with the real robots, real imports through `Game.from_snapshot`, real bots observing the translated events); test behaviour, not implementation; independent tests; descriptive names; fixtures and `parametrize` where cases repeat. Tests that need the JVM call `require()` from `tests/python/jsettlers_harness.py`, which fails (never skips) with the command to run when the jar or the build is missing.
- Long runs (the 20-game differential replay, the smoke run, the three measurement arms, the native-boards check) are run by the controller in the background, never by implementer subagents. Steps marked **controller runs this** give the exact command and the pass condition.
- No README or docs beyond `docs/bot/results-jsettlers.md` (Task 9) and `jsettlers/COPYING` (Task 1).

## Review Focus

1. A harness JVM that errors, dies or goes silent mid-game (an exception in a server or robot thread, a stalled robot, a command it cannot parse). Expected: the run stops with an error that names the JVM's log under `runs/jsettlers/logs/`, no partial or invented record is written, and the JVM never outlives the Python process that started it. Pinned by Task 3's `test_errors_and_silence_fail_loudly_naming_the_log` and `test_the_jvm_exits_when_its_stdin_closes`.
2. A decision whose position does not import (an intermediate server state, a rule difference) or whose request JSettlers declines (`SOCDeclinePlayerRequest`). Expected: the stock brain decides that hook, the fallback is counted by reason, and the same declined request is not asked again in a loop. Pinned by Task 6's `test_unimportable_declined_or_failing_decisions_fall_back_by_reason` and the fallback-reason checks of `test_a_no_trades_bridge_game_records_every_counter`.
3. Robot offers our trading game cannot hold (two cards for one, a shared resource, trades off), arriving while we are on turn (counter-offers), or answered too slowly. Expected: rejected without asking the bot (or rejected when late), each counted in `unrepresentable`, `counters_rejected` or `late`, never accepted by accident. Pinned by Task 6's `test_answers_reject_counter_offers_and_unrepresentable_offers_without_asking`, `test_answers_import_the_offerers_main_position_with_earlier_seats_rejecting` and `test_late_answers_are_sent_as_rejections_and_counted`, and Task 7's `test_a_native_trading_game_answers_and_counts_the_robots_offers`.
4. More than 19 of a resource in hands (JSettlers' unlimited bank). Expected: the position imports and play goes on, the bank shows 0 of it, the payout is counted in `bank_limit_hits`, and with the limit on (every native game) nothing changes. Pinned by Task 2's `without_the_limit_both_claimants_are_paid_and_the_bank_shows_none`, `a_snapshot_with_more_than_19_of_a_resource_in_hands_imports_only_without_the_limit` and `test_a_position_with_20_cards_of_a_resource_imports_only_without_the_limit`, and Task 4's `test_more_than_19_cards_import_only_without_the_bank_limit` and `test_declines_and_bank_limit_payouts_are_counted`.
5. A game that reaches our turn cap (JSettlers has none). Expected: the driver aborts it, it records no winner and `aborted: true`, and any decision asked after the cap falls back as `turn-cap`; a decision asked after the abort, even one that arrives after the game's `over`, is answered `turn-cap` and not counted. Pinned by Task 6's `test_a_game_at_the_turn_cap_ends_with_no_winner_and_the_jvm_plays_on`, `test_late_messages_of_an_earlier_aborted_game_are_answered_and_dropped` and the `turn-cap` case of `test_unimportable_declined_or_failing_decisions_fall_back_by_reason`.

## File Structure

| File | Responsibility | Task |
|---|---|---|
| `scripts/fetch-jsettlers.sh` (new) | download the 2.6.10 server jar, check its sha256 | 1 |
| `jsettlers/COPYING`, `jsettlers/build.sh`, `jsettlers/check-api.sh`, `jsettlers/api.txt` (new) | the GPL notice; `javac` against the jar; the reflection targets checked by `javap` | 1 |
| `jsettlers/src/alphasettler/jsettlers/Json.java`, `Probe.java` (new) | minimal JSON; the board and constants dump | 1 |
| `tests/python/data/jsettlers_board.json` (new, generated by `Probe`) | JSettlers' classic board and codes, the facts the tables are checked against | 1 |
| `oracle/jsettlers/__init__.py`, `tables.py`, `actions.py` (new) | paths and `missing()`; coordinates and codes; our action ids and `redact` | 1 |
| `.gitignore` | `jsettlers/lib/`, `jsettlers/build/` | 1 |
| `engine/src/config.rs`, `state.rs`, `apply.rs`, `invariants.rs`, `rules/{roll,maritime,dev,setup}.rs`, `bindings/src/convert.rs` | `bank_limit` | 2 |
| `jsettlers/src/alphasettler/jsettlers/{Pipe,GameSpec,ArenaServer,ArenaBoard,ArenaBoardFactory,RoutedRandom,Snapshots,ArenaMain}.java` (new) | the in-JVM server, boards, dice, raw positions, the pipe, the main loop | 3 (ArenaMain again in 6, 7) |
| `oracle/jsettlers/jvm.py`, `boards.py` (new) | the JVM process; boards, layouts, streams, game commands | 3 |
| `oracle/jsettlers/messages.py`, `translate.py`, `events.py` (new) | message lines; raw position → our snapshot; the stream → `Step`s | 4 |
| `oracle/jsettlers/diff.py`, `allowlist.py` (new); `alphasettler/cli.py` (`jsettlers-diff`) | the differential replay | 5 |
| `jsettlers/src/alphasettler/jsettlers/{BridgeClient,BridgeBrain,BridgeOpening,BridgeRobber,BridgeDM}.java` (new) | the bridge seat | 6 (BridgeBrain again in 7) |
| `oracle/jsettlers/arena.py` (new) | `Seat`, `play_game`, `run_match`, `check`, records | 6 |
| `jsettlers/src/alphasettler/jsettlers/{CappedClient,CappedBrain}.java` (new) | the aligned arm's robots | 7 |
| `alphasettler/cli.py` (`arena --baseline jsettlers:...`, `--aligned`, `--native-boards`) | the arena command | 8 |
| `docs/bot/results-jsettlers.md` (new) | the measurements | 9 |
| tests: `tests/python/jsettlers_harness.py` (new, 1), `tests/python/test_jsettlers_tables.py` (1), `engine/tests/bank_limit.rs` and `tests/python/test_bank_limit.py` (2), `tests/python/test_jsettlers_jvm.py` (3), `test_jsettlers_translate.py` and `test_jsettlers_events.py` (4), `test_jsettlers_diff.py` (5), `test_jsettlers_seat.py` and `test_jsettlers_arena.py` (6, 7), `test_jsettlers_cli.py` (8) | | 1–8 |

## Where the code forces a choice the spec does not make

Each is a **Ruling** (with its reason) the results file repeats, so it can be overridden.

- **The stream, not a request, holds the positions.** The harness streams every recorded message as GameEventLog writes it (`all:`, `pN:`, `!pN:`, `fN:` for a client request, then `toString()`), and on the line where the server-side game changed it attaches that game in JSettlers' own coordinates and codes (`Snapshots.raw`); Python translates. A decision imports the latest position in the stream before the request, the one matching the events our bot has seen, never a separate read.
- **Monitor.** `ArenaServer.record` serializes recording with its own lock, because the server's treater thread and `SOCForceEndTurnThread` (holding the game's monitor) can both change the game and record its events: the start line, `Snapshots.raw`, its comparison with the last position and the queued lines are one step per event, and `abort` queues `over` under the same lock, so no event follows it. The game's monitor is not taken (it is not reentrant, and most callers hold it); the game is looked up outside the lock and `destroyGameAndBroadcast` runs after it, so nothing under the lock waits on a JSettlers lock.
- **Stretches.** A client request line starts a stretch; the server's messages up to the next request are what it did (one `Step`). Python closes the open stretch before every decision and at the end of the game. Two stretches are one action of ours when the first leaves a position we have no phase for: a build request and its placement, a Year of Plenty or Monopoly and its pick. A robber move with one victim and JSettlers' automatic robbery are one stretch, so one step of two actions.
- **The end of a game.** The stream ends at the first message recorded once the game is over, which can come before the server reports the winning action (JSettlers announces a winning placement in text first): the winning action is read from its request and checked against the final position. Our rules end the game the moment the player on turn reaches the target; where JSettlers plays on (a knight's Largest Army wins there only after the robber moves), the step carries the position as our rules leave it and nothing after it is a step. A forced end of turn (`SOCTurn` with no end-turn request, in any stretch) ends the turn of ours as well and carries `forced-end-turn`.
- **A forced end in setup.** The server ends a stalled setup turn itself (stock robots: `planInitRoad` can return an edge off the board), removing the settlement awaiting its road and the cards it produced, and the setup order and JSettlers' first player move on. It is its own step, allowlisted `forced-setup-end`: the replay compares everything but that settlement, its cards and the setup turn order, then re-imports. Our engine has no event that removes a settlement, so our bot is shown nothing and takes one counted belief reset at its next searched decision. A stock robot whose own settlement was cancelled keeps it in its copy of the game (`handleCANCELBUILDREQUEST` undoes it only on another player's turn) and plays the rest on a board it does not have; the record's `robot_desync` says so, and Task 9 reports those games apart.
- **Seats.** The bridge is JSettlers seat 0 (seated by `readyGameAskRobotsJoin`'s `robotSeats`, reached by reflection); the first player is scripted to `(4 - k) % 4` so our seat has turn position k. Our player i is JSettlers seat `(first + i) % 4`, so our player 0 moves first as in our engine; records use our numbering and `candidate_seat` is k.
- **Boards.** A seed's board is the native arena's (`Game(seed).snapshot()["board"]`): its tiles and numbers as they are, its nine port kinds in our port order moved onto JSettlers' nine port edges (`SOCBoard4p.PORTS_EDGE_V1`) in their order. The native-boards check arm leaves board generation to JSettlers (`BC` on, no common random numbers for the board).
- **Random streams.** `SOCGame.rand` is replaced by `RoutedRandom`, which routes `nextInt()` by its caller's method name: `startGame` returns the scripted first player, `rollDice` draws from the seed's dice stream, `startGame_setupDevCards` from its deck stream, anything else (steals) from its misc stream. JSettlers reports only a roll's total; our forced roll and our event use the pair `(min(n - 1, 6), n - min(n - 1, 6))`.
- **The unlimited bank.** With `bank_limit` off our engine never runs short (production, maritime trades, Year of Plenty), and its `bank` shows what a 19-card bank would hold, 19 less the cards in hands, never below 0 (refreshed after every action; `from_snapshot` keeps the bank it is given). A belief built from an observation alone (after a reset) can under-deal a resource of which more than 19 are held. The conservation and second-settlement bank invariants are checked only with the limit on. `config_dict` writes `bank_limit` only when it is off, so resolved configs and record files are unchanged. Compound moves (`+c`) keep their limited-bank rule (not measured here). `bank_limit_hits` counts the stretches after which some resource's cards in hands exceed 19 and grew (the payouts our limit would have refused).
- **Discards.** JSettlers' players discard concurrently; the replay applies them in our seat order once the last one is in (the steps before compare nothing), and our bot is shown each discard as it comes. Our seat's discards are chosen one card at a time on a copy of the position where only our seat owes, our bot shown each before it chooses the next (and not again when the stream reports them); a choice that fails after the first card completes with the first discards our rules allow, counted, so the reply is what the bot was shown. Our bot is shown every discard's resources, as our rules show them (its belief has no hidden discard); JSettlers shows opponents only the count, so this is a small information edge for our bot, stated in the results.
- **Bank trades.** One JSettlers bank trade is as many maritime trades of ours as whole rates it gives, paired with the cards it gets in resource order; anything else (an undone trade) is the allowlisted `bank-trade`.
- **Domestic trades.** The replay re-imports the position after an executed trade (allowlisted `domestic-trade`). Our bot is shown each offer when made (`trade_offered`), each rejection, and an executed trade as the accepted offer re-stated, its acceptance and its confirmation, so its belief checks the cards that moved; clearing an open offer is `trade_cancelled`.
- **Pre-roll.** Our bot rolls or plays a knight before the roll; any other card it would play there is left to the stock brain's roll-or-knight (fallback `preroll-card`), because the stock turn machine plays only knights before the roll.
- **Robber victim.** Chosen with the robber move, at the hypothetical steal position, as the Catanatron arena does; JSettlers' own victim request takes it.
- **Our offers** (spec Section 5): sent to the opponents who hold the asked card and whose lone acceptance our bot would confirm (asked at that hypothetical position). An offer with no such opponent is made and declined inside our engine only: our bot is shown its events and decides again (at most `max_offers_per_turn` times). `our_offers` counts every offer our bot chose; `our_offers_accepted` the executed ones.
- **Robot offers.** One made to us while we are the current player (a counter-offer, or an offer to the player on turn) is rejected unasked and counted in `counters_rejected`. One outside our trading game (not one card a side, a resource on both sides, or the no-trades arm) is rejected unasked and counted in `unrepresentable`. The rest are answered at the offerer's Main position with its earlier offers this turn clamped to 1, the offer applied and the seats before ours rejecting; an answer ready 4.5 s or more after the request is sent as a rejection and counted in `late`.
- **Fallbacks** (spec Section 2): a position that does not import, an engine panic or error, a request JSettlers declined since the last decision, a reply the brain cannot carry out, a Java-side timeout (600 s), or a victim JSettlers will not take. The stock brain decides that hook (a free road falls back to the first road JSettlers allows, since the stock placement needs a plan the bridge never made); the record counts `fallbacks` and `fallback_reasons` by reason (`position`, `declined`, `engine`, `choice`, `preroll-card`, `turn-cap`, `java-timeout`, `victim`, `forced-discard`). A refusal is charged to the hook it happened in (every hook checks for one first). A discard of ours the server forced (no reply of ours asked for it) is `forced-discard`. The robber move JSettlers asks after our winning knight (our rules ended the game at the card) is answered `after-win` and not counted.
- **Our seat never trades by the stock negotiator.** The bridge's `makeOffer` and `makeCounterOffer` send nothing, so every offer from our seat is our bot's; an answer that falls back is the stock brain's `considerOffer`.
- **Timeouts.** Built-in robots' forced end of turn (`SOCServer.ROBOT_FORCE_ENDTURN_SECONDS`, 8 s) is raised to 60 s so our thinking during a robot's turn (an answer, a discard) never forces it, and a stubborn robot's (`ROBOT_FORCE_ENDTURN_STUBBORN_SECONDS`, 4 s) likewise; our own turns have `jsettlers.bots.timeout.turn` 600 s, and the bridge waits 540 s for a reply, so a slow one is a counted `java-timeout` before the server's end.
- **Turn cap.** At our `max_turns` (1000) the driver aborts the game: no winner, `aborted: true`.
- **The aligned arm.** `CappedBrain` sends the stock negotiator's offer only if it is one card a side, sends it to every opponent, at most two a turn, and never counter-offers; when the negotiator's offer does not fit, that robot stops offering for the turn. Records count robot offers (`robot_offers`) and those outside these caps (`robot_offers_over_caps`) in every arm.
- **Records**, per game: `seed`, `candidate_seat`, `winner`, `vp`, `turns`, `rounds`, `aborted`, `robot_desync` (a setup settlement was cancelled by a forced end), `last_build_turn` (per seat of ours, the last turn it built a piece or bought a card after setup, null if never: stock robots can stop building at 8-9 VP while nobody holds Longest Road, so Task 9 reports results with and without those games), `fallbacks`, `fallback_reasons`, `belief_resets`, `our_offers`, `our_offers_accepted`, `offers_to_us`, `offers_to_us_accepted`, `unrepresentable`, `counters_rejected`, `late`, `robot_offers`, `robot_offers_over_caps`, `bank_limit_hits`, `arm` (the order of `RECORD_KEYS`).
- **Workers.** One JVM per worker process, games one at a time. A JVM that reports an error or sends nothing for 15 minutes fails the run, naming its log; a JVM exits when its stdin closes.
- **B** is chosen in Task 9 Step 4 by an explicit table from the results files present at run time; the candidate name stays the spec's `net:B@1000+t+o` (no `+c`) whatever B is.
- **Fresh seeds:** the arms from 8,000,000,000 (the spec's), the native-boards check on the first 20 of them, the differential gate from 9,000,000,000, the smoke run from 9,100,000,000; no self-play (below 1e9), gate (1e9 + 0..200) or SPRT (2e9 to 7.001e9) used them. Tests use seeds below 100.
- **Task boundaries:** setup and the tables (1), the engine (2), the server (3), the translation (4), the replay gate (5), the bridge without trades plus the whole Python seat (6), trades across the bridge and the aligned robots (7), the command (8), the runs (9).

---

### Task 1: The jar, the probe and the board tables

**Files:**
- Create: `scripts/fetch-jsettlers.sh`, `jsettlers/COPYING`, `jsettlers/build.sh`, `jsettlers/check-api.sh`, `jsettlers/api.txt`, `jsettlers/src/alphasettler/jsettlers/Json.java`, `jsettlers/src/alphasettler/jsettlers/Probe.java`, `oracle/jsettlers/__init__.py`, `oracle/jsettlers/tables.py`, `oracle/jsettlers/actions.py`, `tests/python/jsettlers_harness.py`, `tests/python/data/jsettlers_board.json` (written by the probe in Step 5)
- Modify: `.gitignore` (one line)
- Test: `tests/python/test_jsettlers_tables.py` (new)

**Interfaces:**
- Consumes: `alphasettler.Game` (`snapshot()`), `alphasettler._engine.net_adjacency` (our tile/edge lists), the jar's `soc.game.SOCBoard4p`, `SOCBoard`, `SOCResourceConstants`, `SOCDevCardConstants`, `SOCPlayingPiece`, `SOCGame` state constants, `soc.robot.SOCRobotDM.SMART_STRATEGY`, `soc.util.Version.version()`.
- Produces:
  - `oracle.jsettlers`: `ROOT`, `VERSION = "2.6.10"`, `JAR`, `JAR_SHA256`, `CLASSES`, `LOGS`, `java_command(main: str, *args) -> list[str]`, `missing(main: str = "ArenaMain") -> str | None`.
  - `oracle.jsettlers.tables`: `TILE_COORDS`, `TILE_NODES`, `EDGE_NODES`, `COASTAL_EDGES`, `HEXNODES`, `OUR_TILE_TO_JS_HEX`, `JS_HEX_TO_OUR_TILE`, `OUR_NODE_TO_JS_NODE`, `JS_NODE_TO_OUR_NODE`, `js_edge_nodes(edge) -> (int, int)`, `OUR_EDGE_TO_JS_EDGE`, `JS_EDGE_TO_OUR_EDGE`, `NUM_TO_HEX` (37), `HEX_NUM`, `PORT_EDGES`, `PORT_HEXNUMS`, `PORT_FACING`, `JS_RES` (JS resource code -> our index), `OUR_RES_TO_JS`, `js5_to_ours(js5) -> list[int]`, `ours_to_js5(hand) -> list[int]`, `JS_DEV_TO_OURS` (JS dev code -> our dev name), `DEV_NAMES`, `RESOURCE_NAMES`, `WATER_HEX`, `DESERT_HEX`, `MISC_PORT`, `ROAD`, `SETTLEMENT`, `CITY`, `STATES` (name -> code), `SMART_STRATEGY`, `SETUP_STATES`, the state constants `START1A, START1B, START2A, START2B, ROLL_OR_CARD, PLAY1, PLACING_ROAD, PLACING_SETTLEMENT, PLACING_CITY, PLACING_ROBBER, PLACING_FREE_ROAD1, PLACING_FREE_ROAD2, WAITING_FOR_DISCARDS, WAITING_FOR_ROB_CHOOSE_PLAYER, WAITING_FOR_DISCOVERY, WAITING_FOR_MONOPOLY, OVER`.
  - `oracle.jsettlers.actions`: `ROLL, END_TURN, BUY_DEV, KNIGHT, ROAD_BUILDING, ACCEPT, REJECT, CANCEL`, `PAIRS`, `monopoly(r)`, `year_of_plenty(a, b)`, `settlement(n)`, `city(n)`, `road(e)`, `move_robber(t)`, `steal_from(p)`, `discard(r)`, `maritime(give, get)`, `bundle(i) -> list[int]`, `bundle_index(hand) -> int | None`, `offer(give_hand, get_hand) -> int | None`, `offer_hands(a) -> (list[int], list[int])`, `is_offer(a)`, `confirm(p)`, `decode(a) -> tuple`, `redact(event, viewer) -> dict`.
  - `tests/python/jsettlers_harness.py`: `require(main: str = "ArenaMain") -> None` (fails the test, never skips, naming the command that builds what is missing).
  - `jsettlers/build.sh` (exit 0 = `check-api.sh` passed and every source compiled to `jsettlers/build/classes`), `jsettlers/check-api.sh` (exit 0 iff every line of `api.txt` is found by `javap -p`).

- [ ] **Step 1: The fetch script, the ignore line and the notice**

Create `scripts/fetch-jsettlers.sh`:
```bash
#!/usr/bin/env bash
# Download JSettlers2 2.6.10's server jar (GPL-3.0-or-later; never committed) and check its sha256.
set -euo pipefail
cd "$(dirname "$0")/.."
URL=https://github.com/jdmonin/JSettlers2/releases/download/release-2.6.10/jsettlers-2.6.10-server.jar
SHA=9b26cb20ec943213600795d660fad2da32fc72629c348e180bd171cc26d6b4e2
OUT=jsettlers/lib/jsettlers-2.6.10-server.jar
mkdir -p jsettlers/lib
if [ -f "$OUT" ] && echo "$SHA  $OUT" | shasum -a 256 -c --status; then
  echo "$OUT is present and checks out"
  exit 0
fi
curl -fsSL "$URL" -o "$OUT.part"
echo "$SHA  $OUT.part" | shasum -a 256 -c --status || { rm -f "$OUT.part"; echo "sha256 mismatch for $URL" >&2; exit 1; }
mv -f "$OUT.part" "$OUT"
echo "fetched $OUT"
```

Append to `.gitignore`, after the `bench/compare/.cache/` line:
```
# JSettlers2's server jar (downloaded by scripts/fetch-jsettlers.sh; jsettlers/build/ is covered by build/)
jsettlers/lib/
```

Create `jsettlers/COPYING`:
```
The Java sources under jsettlers/src/ extend and link classes of JSettlers2
(https://github.com/jdmonin/JSettlers2, release 2.6.10), which is distributed under the
GNU General Public License, version 3 or (at your option) any later version.

These sources are therefore distributed under the same terms: GPL-3.0-or-later.
The full license text is at https://www.gnu.org/licenses/gpl-3.0.txt.

Nothing outside jsettlers/ links JSettlers2. The rest of AlphaSettler talks to the harness
only through JSON lines on the harness process's standard input and output
(oracle/jsettlers/jvm.py), as a separate program.

The JSettlers2 jar itself is not part of this repository; scripts/fetch-jsettlers.sh downloads
the released jsettlers-2.6.10-server.jar and checks its sha256.
```

Every `.java` file this plan creates starts with exactly this header (the class's own javadoc follows it):
```java
// AlphaSettler's JSettlers2 arena harness.
// SPDX-License-Identifier: GPL-3.0-or-later
// Extends JSettlers2 (https://github.com/jdmonin/JSettlers2, GPL-3.0-or-later).
// This program is free software: you can redistribute it and/or modify it under the terms of the
// GNU General Public License, version 3 or any later version; see jsettlers/COPYING.
```

Run: `bash scripts/fetch-jsettlers.sh && git status --porcelain jsettlers/`
Expected: `fetched jsettlers/lib/jsettlers-2.6.10-server.jar` (or `... is present and checks out`), and `git status` lists `jsettlers/COPYING` only (the jar is ignored).

- [ ] **Step 2: The API check and the build script**

The harness reaches a few members of the jar that are not public API (a private field, package-private methods and constants) and some public ones whose exact signature the code depends on. `api.txt` lists each as `class | text javap prints for it`; `check-api.sh` fails, naming the line, if the 2.6.10 jar does not have it.

Create `jsettlers/api.txt`:
```
# class | substring of `javap -p -cp <jar> <class>` output that must be present
soc.game.SOCGame | private java.util.Random rand;
soc.game.SOCGame | private int[] devCardDeck;
soc.game.SOCGame | void startGame_setupDevCards();
soc.game.SOCGame | public void startGame();
soc.game.SOCGame | rollDice();
soc.game.SOCGame | public static soc.game.SOCBoard$BoardFactory boardFactory;
soc.game.SOCGame | public int getNumDevCards();
soc.game.SOCGame | public int getFirstPlayer();
soc.game.SOCGame | public int getRoundCount();
soc.game.SOCBoard4p | static final int[] PORTS_HEXNUM_V1;
soc.game.SOCBoard4p | public int[] getPortsEdges();
soc.game.SOCBoard4p | public int[] getPortsFacing();
soc.game.SOCBoard | public void makeNewBoard(soc.game.SOCGameOptionSet);
soc.game.SOCBoard | public void setHexLayout(int[])
soc.game.SOCBoard | public void setNumberLayout(int[])
soc.game.SOCBoard | public void setRobberHex(int, boolean)
soc.game.SOCBoard | public int getHexNumFromCoord(int)
soc.game.SOCBoard | public int getPortTypeFromNodeCoord(int);
soc.game.SOCPlayingPiece | public int getCoordinates();
soc.game.SOCPlayer | public boolean isPotentialRoad(int);
soc.server.SOCServer | boolean readyGameAskRobotsJoin(soc.game.SOCGame, boolean[], soc.server.genericServer.Connection[], int)
soc.server.SOCServer | protected void startEmptyLog(soc.game.SOCGame, boolean)
soc.server.SOCServer | public soc.game.SOCGame createGameAndBroadcast(soc.server.genericServer.Connection, java.lang.String, soc.game.SOCGameOptionSet, boolean);
soc.server.SOCServer | protected java.util.Vector<soc.server.genericServer.Connection> robots;
soc.server.SOCServer | public static int ROBOT_FORCE_ENDTURN_SECONDS;
soc.server.SOCBoardAtServer$BoardFactoryAtServer | public soc.game.SOCBoard createBoard(soc.game.SOCGameOptionSet, boolean, int)
soc.robot.SOCRobotBrain | public static float BOTS_ONLY_FAST_PAUSE_FACTOR;
soc.robot.SOCRobotBrain | protected void planAndDoActionForPLAY1();
soc.robot.SOCRobotBrain | protected void rollOrPlayKnightOrExpectDice();
soc.robot.SOCRobotBrain | protected void placeIfExpectPlacing();
soc.robot.SOCRobotBrain | protected void moveRobber();
soc.robot.SOCRobotBrain | protected void discard(int);
soc.robot.SOCRobotBrain | protected int considerOffer(soc.game.SOCTradeOffer);
soc.robot.SOCRobotBrain | protected boolean makeOffer(
soc.robot.SOCRobotBrain | protected boolean makeCounterOffer(soc.game.SOCTradeOffer);
soc.robot.SOCRobotBrain | protected void setStrategyFields();
soc.robot.SOCRobotBrain | protected soc.robot.SOCRobotDM createDM();
soc.robot.SOCRobotBrain | protected void resetFieldsAtEndTurn();
soc.robot.SOCRobotDM | protected soc.game.SOCResourceSet resourceChoices;
soc.robot.SOCRobotClient | protected java.lang.String rbclass;
soc.robot.SOCRobotClient | public soc.robot.SOCRobotBrain createBrain(soc.util.SOCRobotParameters, soc.game.SOCGame, soc.util.CappedQueue<soc.message.SOCMessage>);
soc.robot.SOCRobotClient | public boolean printedInitialWelcome;
soc.robot.MonopolyStrategy | public void setMonopolyChoice(int);
soc.robot.RobberStrategy | public int chooseRobberVictim(boolean[], boolean);
soc.robot.OpeningBuildStrategy | public int planInitialSettlements();
soc.robot.OpeningBuildStrategy | public int planSecondSettlement();
soc.robot.OpeningBuildStrategy | public int planInitRoad();
soc.util.Version | public static java.lang.String version();
```

Create `jsettlers/check-api.sh` (written for macOS's bash 3.2: no associative arrays):
```bash
#!/usr/bin/env bash
# Fail, naming the line, unless every member jsettlers/api.txt (or the file given) lists is in the pinned jar.
set -euo pipefail
cd "$(dirname "$0")/.."
JAR=jsettlers/lib/jsettlers-2.6.10-server.jar
LIST=${1:-jsettlers/api.txt}
[ -f "$JAR" ] || { echo "missing $JAR; run: bash scripts/fetch-jsettlers.sh" >&2; exit 1; }
cache=$(mktemp -d)
trap 'rm -rf "$cache"' EXIT
fail=0
n=0
while IFS= read -r line; do
  case "$line" in ''|'#'*) continue ;; esac
  cls=${line%% | *}
  want=${line#* | }
  [ -f "$cache/$cls" ] || javap -p -cp "$JAR" "$cls" > "$cache/$cls" 2>&1 || true
  n=$((n + 1))
  if ! grep -qF -- "$want" "$cache/$cls"; then
    echo "check-api: $cls has no member matching: $want" >&2
    fail=1
  fi
done < "$LIST"
[ "$fail" = 0 ] && echo "check-api: $n members present"
exit "$fail"
```

Create `jsettlers/build.sh`:
```bash
#!/usr/bin/env bash
# Compile the arena harness against the pinned JSettlers2 jar (plain javac; no Gradle or Maven).
set -euo pipefail
cd "$(dirname "$0")/.."
JAR=jsettlers/lib/jsettlers-2.6.10-server.jar
bash jsettlers/check-api.sh
rm -rf jsettlers/build/classes
mkdir -p jsettlers/build/classes
find jsettlers/src -name '*.java' | sort > jsettlers/build/sources.txt
for f in $(cat jsettlers/build/sources.txt); do
  [ "$(sed -n 2p "$f")" = "// SPDX-License-Identifier: GPL-3.0-or-later" ] || { echo "$f lacks the GPL header" >&2; exit 1; }
done
javac --release 17 -Xlint:all,-serial,-deprecation,-removal,-processing,-path -Werror -encoding UTF-8 \
  -cp "$JAR" -d jsettlers/build/classes @jsettlers/build/sources.txt
echo "built $(wc -l < jsettlers/build/sources.txt | tr -d ' ') sources into jsettlers/build/classes"
```

- [ ] **Step 3: Prove the API check fails loudly, then passes**

Run: `pgrep -fl 'arm.sh' || echo idle` (wait while it prints a process), then:
```bash
{ cat jsettlers/api.txt; echo 'soc.game.SOCGame | private int notAMember;'; } > "$TMPDIR/api-check.txt"
bash jsettlers/check-api.sh "$TMPDIR/api-check.txt"; echo "exit $?"
rm -f "$TMPDIR/api-check.txt"
bash jsettlers/check-api.sh
```
Expected: the first run prints `check-api: soc.game.SOCGame has no member matching: private int notAMember;` and `exit 1`; the second prints `check-api: 48 members present` and exits 0. If any real line is reported missing, the jar is not the pinned 2.6.10 or a signature differs: read it with `javap -p -cp jsettlers/lib/jsettlers-2.6.10-server.jar <class>`, report the difference as BLOCKED with the javap line, and do not edit `api.txt` to make it pass.

- [ ] **Step 4: `Json.java` and `Probe.java`**

Create `jsettlers/src/alphasettler/jsettlers/Json.java` (the five header lines first, as in Step 1):
```java
package alphasettler.jsettlers;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * The minimal JSON the pipe needs: objects as insertion-ordered maps, arrays as lists,
 * integers as {@code Long}, other numbers as {@code Double}, strings, booleans and null.
 * {@link Raw} carries text that is already JSON.
 */
public final class Json {
    private Json() {}

    /** Already-serialized JSON, written as is. */
    public record Raw(String json) {}

    public static Map<String, Object> map(Object... kv) {
        if (kv.length % 2 != 0) throw new IllegalArgumentException("map needs key/value pairs");
        Map<String, Object> m = new LinkedHashMap<>();
        for (int i = 0; i < kv.length; i += 2) m.put((String) kv[i], kv[i + 1]);
        return m;
    }

    public static List<Object> list(Object... xs) {
        return new ArrayList<>(Arrays.asList(xs));
    }

    public static List<Object> ints(int[] xs) {
        List<Object> out = new ArrayList<>(xs.length);
        for (int x : xs) out.add(x);
        return out;
    }

    public static String write(Object v) {
        StringBuilder sb = new StringBuilder();
        write(sb, v);
        return sb.toString();
    }

    private static void write(StringBuilder sb, Object v) {
        if (v == null) {
            sb.append("null");
        } else if (v instanceof Raw r) {
            sb.append(r.json());
        } else if (v instanceof String s) {
            quote(sb, s);
        } else if (v instanceof Boolean || v instanceof Integer || v instanceof Long) {
            sb.append(v);
        } else if (v instanceof Double d) {
            if (d.isNaN() || d.isInfinite()) throw new IllegalArgumentException("non-finite number");
            sb.append(d);
        } else if (v instanceof int[] a) {
            write(sb, ints(a));
        } else if (v instanceof Map<?, ?> m) {
            sb.append('{');
            boolean first = true;
            for (Map.Entry<?, ?> e : m.entrySet()) {
                if (!first) sb.append(',');
                first = false;
                quote(sb, (String) e.getKey());
                sb.append(':');
                write(sb, e.getValue());
            }
            sb.append('}');
        } else if (v instanceof List<?> l) {
            sb.append('[');
            for (int i = 0; i < l.size(); i++) {
                if (i > 0) sb.append(',');
                write(sb, l.get(i));
            }
            sb.append(']');
        } else {
            throw new IllegalArgumentException("cannot write " + v.getClass().getName());
        }
    }

    private static void quote(StringBuilder sb, String s) {
        sb.append('"');
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            switch (c) {
                case '"' -> sb.append("\\\"");
                case '\\' -> sb.append("\\\\");
                case '\n' -> sb.append("\\n");
                case '\r' -> sb.append("\\r");
                case '\t' -> sb.append("\\t");
                default -> {
                    if (c < 0x20) sb.append(String.format("\\u%04x", (int) c));
                    else sb.append(c);
                }
            }
        }
        sb.append('"');
    }

    public static Object parse(String s) {
        int[] at = {0};
        Object v = value(s, at);
        skip(s, at);
        if (at[0] != s.length()) throw new IllegalArgumentException("trailing text in JSON at " + at[0]);
        return v;
    }

    @SuppressWarnings("unchecked")
    public static Map<String, Object> object(String s) {
        Object v = parse(s);
        if (!(v instanceof Map)) throw new IllegalArgumentException("expected a JSON object: " + s);
        return (Map<String, Object>) v;
    }

    private static void skip(String s, int[] at) {
        while (at[0] < s.length() && Character.isWhitespace(s.charAt(at[0]))) at[0]++;
    }

    private static Object value(String s, int[] at) {
        skip(s, at);
        if (at[0] >= s.length()) throw new IllegalArgumentException("unexpected end of JSON");
        char c = s.charAt(at[0]);
        if (c == '{') {
            at[0]++;
            Map<String, Object> m = new LinkedHashMap<>();
            skip(s, at);
            if (s.charAt(at[0]) == '}') { at[0]++; return m; }
            while (true) {
                skip(s, at);
                Object k = value(s, at);
                if (!(k instanceof String key)) throw new IllegalArgumentException("object key is not a string");
                skip(s, at);
                expect(s, at, ':');
                m.put(key, value(s, at));
                skip(s, at);
                if (s.charAt(at[0]) == ',') { at[0]++; continue; }
                expect(s, at, '}');
                return m;
            }
        }
        if (c == '[') {
            at[0]++;
            List<Object> l = new ArrayList<>();
            skip(s, at);
            if (s.charAt(at[0]) == ']') { at[0]++; return l; }
            while (true) {
                l.add(value(s, at));
                skip(s, at);
                if (s.charAt(at[0]) == ',') { at[0]++; continue; }
                expect(s, at, ']');
                return l;
            }
        }
        if (c == '"') return string(s, at);
        if (s.startsWith("true", at[0])) { at[0] += 4; return Boolean.TRUE; }
        if (s.startsWith("false", at[0])) { at[0] += 5; return Boolean.FALSE; }
        if (s.startsWith("null", at[0])) { at[0] += 4; return null; }
        int start = at[0];
        while (at[0] < s.length() && "+-0123456789.eE".indexOf(s.charAt(at[0])) >= 0) at[0]++;
        String num = s.substring(start, at[0]);
        if (num.isEmpty()) throw new IllegalArgumentException("bad JSON at " + start);
        if (num.contains(".") || num.contains("e") || num.contains("E")) return Double.parseDouble(num);
        return Long.parseLong(num);
    }

    private static void expect(String s, int[] at, char c) {
        if (at[0] >= s.length() || s.charAt(at[0]) != c)
            throw new IllegalArgumentException("expected '" + c + "' in JSON at " + at[0]);
        at[0]++;
    }

    private static String string(String s, int[] at) {
        expect(s, at, '"');
        StringBuilder sb = new StringBuilder();
        while (true) {
            char c = s.charAt(at[0]++);
            if (c == '"') return sb.toString();
            if (c != '\\') { sb.append(c); continue; }
            char e = s.charAt(at[0]++);
            switch (e) {
                case '"', '\\', '/' -> sb.append(e);
                case 'n' -> sb.append('\n');
                case 'r' -> sb.append('\r');
                case 't' -> sb.append('\t');
                case 'b' -> sb.append('\b');
                case 'f' -> sb.append('\f');
                case 'u' -> { sb.append((char) Integer.parseInt(s.substring(at[0], at[0] + 4), 16)); at[0] += 4; }
                default -> throw new IllegalArgumentException("bad escape \\" + e);
            }
        }
    }

    // Typed reads of a parsed object; each fails loudly on a missing or mistyped key.

    public static long num(Map<String, Object> m, String k) {
        Object v = m.get(k);
        if (!(v instanceof Long l)) throw new IllegalArgumentException("expected an integer at " + k + ": " + m);
        return l;
    }

    public static int i(Map<String, Object> m, String k) {
        return Math.toIntExact(num(m, k));
    }

    public static boolean bool(Map<String, Object> m, String k) {
        Object v = m.get(k);
        if (!(v instanceof Boolean b)) throw new IllegalArgumentException("expected a boolean at " + k + ": " + m);
        return b;
    }

    public static String str(Map<String, Object> m, String k) {
        Object v = m.get(k);
        if (!(v instanceof String s)) throw new IllegalArgumentException("expected a string at " + k + ": " + m);
        return s;
    }

    public static int[] intArray(Map<String, Object> m, String k) {
        Object v = m.get(k);
        if (!(v instanceof List<?> l)) throw new IllegalArgumentException("expected a list at " + k + ": " + m);
        int[] out = new int[l.size()];
        for (int j = 0; j < out.length; j++) out[j] = Math.toIntExact((Long) l.get(j));
        return out;
    }

    public static boolean[] boolArray(Map<String, Object> m, String k) {
        Object v = m.get(k);
        if (!(v instanceof List<?> l)) throw new IllegalArgumentException("expected a list at " + k + ": " + m);
        boolean[] out = new boolean[l.size()];
        for (int j = 0; j < out.length; j++) out[j] = (Boolean) l.get(j);
        return out;
    }

    @SuppressWarnings("unchecked")
    public static Map<String, Object> sub(Map<String, Object> m, String k) {
        Object v = m.get(k);
        if (v == null) return null;
        if (!(v instanceof Map)) throw new IllegalArgumentException("expected an object at " + k + ": " + m);
        return (Map<String, Object>) v;
    }
}
```

Create `jsettlers/src/alphasettler/jsettlers/Probe.java` (header first):
```java
package alphasettler.jsettlers;

import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.TreeSet;
import soc.game.SOCBoard;
import soc.game.SOCBoard4p;
import soc.game.SOCDevCardConstants;
import soc.game.SOCGame;
import soc.game.SOCGameOptionSet;
import soc.game.SOCPlayingPiece;
import soc.game.SOCResourceConstants;
import soc.robot.SOCRobotDM;
import soc.util.Version;

/**
 * Prints, as one JSON object, the facts of the jar's classic 4-player board and codes that
 * oracle/jsettlers/tables.py is checked against: land hexes, their corner nodes in JSettlers'
 * clockwise-from-north order, every land edge and its end nodes, the hex-layout index of each
 * hex, the port edges, facings and hex numbers, and the resource, card, piece, hex, state and
 * strategy constants. Run by Task 1 into tests/python/data/jsettlers_board.json.
 */
public final class Probe {
    private Probe() {}

    public static void main(String[] args) throws ReflectiveOperationException {
        SOCBoard4p b = new SOCBoard4p(SOCGameOptionSet.getAllKnownOptions());
        Map<String, Object> out = Json.map("version", Version.version());
        List<Object> land = new ArrayList<>(), hexNodes = new ArrayList<>(), edges = new ArrayList<>(),
            hexNums = new ArrayList<>();
        TreeSet<Integer> seen = new TreeSet<>();
        for (int h : b.getLandHexCoords()) {
            land.add(h);
            int[] ns = b.getAdjacentNodesToHex_arr(h);
            List<Object> row = Json.list(h);
            for (int n : ns) row.add(n);
            hexNodes.add(row);
            for (int i = 0; i < 6; i++) {
                int e = b.getEdgeBetweenAdjacentNodes(ns[i], ns[(i + 1) % 6]);
                if (seen.add(e)) {
                    int[] ab = b.getAdjacentNodesToEdge_arr(e);
                    edges.add(Json.list(e, ab[0], ab[1]));
                }
            }
        }
        for (int h = 0; h < 0x100; h++) {
            int n = b.getHexNumFromCoord(h);
            if (n >= 0) hexNums.add(Json.list(h, n));
        }
        out.put("land_hexes", land);
        out.put("hex_nodes", hexNodes);
        out.put("edges", edges);
        out.put("hex_nums", hexNums);
        Field hexnum = SOCBoard4p.class.getDeclaredField("PORTS_HEXNUM_V1");
        hexnum.setAccessible(true);
        out.put("port_hexnums", Json.ints((int[]) hexnum.get(null)));
        out.put("port_edges", Json.ints(b.getPortsEdges()));
        out.put("port_facing", Json.ints(b.getPortsFacing()));
        out.put("resources", Json.map("clay", SOCResourceConstants.CLAY, "ore", SOCResourceConstants.ORE,
            "sheep", SOCResourceConstants.SHEEP, "wheat", SOCResourceConstants.WHEAT,
            "wood", SOCResourceConstants.WOOD, "unknown", SOCResourceConstants.UNKNOWN));
        out.put("dev_cards", Json.map("unknown", SOCDevCardConstants.UNKNOWN, "roads", SOCDevCardConstants.ROADS,
            "disc", SOCDevCardConstants.DISC, "mono", SOCDevCardConstants.MONO, "cap", SOCDevCardConstants.CAP,
            "market", SOCDevCardConstants.MARKET, "univ", SOCDevCardConstants.UNIV,
            "temple", SOCDevCardConstants.TEMPLE, "chapel", SOCDevCardConstants.CHAPEL,
            "knight", SOCDevCardConstants.KNIGHT));
        out.put("pieces", Json.map("road", SOCPlayingPiece.ROAD, "settlement", SOCPlayingPiece.SETTLEMENT,
            "city", SOCPlayingPiece.CITY));
        out.put("hex_types", Json.map("water", SOCBoard.WATER_HEX, "desert", SOCBoard.DESERT_HEX,
            "misc_port_hex", SOCBoard.MISC_PORT_HEX, "misc_port", SOCBoard.MISC_PORT));
        out.put("states", Json.map("START1A", SOCGame.START1A, "START1B", SOCGame.START1B,
            "START2A", SOCGame.START2A, "START2B", SOCGame.START2B, "ROLL_OR_CARD", SOCGame.ROLL_OR_CARD,
            "PLAY1", SOCGame.PLAY1, "PLACING_ROAD", SOCGame.PLACING_ROAD,
            "PLACING_SETTLEMENT", SOCGame.PLACING_SETTLEMENT, "PLACING_CITY", SOCGame.PLACING_CITY,
            "PLACING_ROBBER", SOCGame.PLACING_ROBBER, "PLACING_FREE_ROAD1", SOCGame.PLACING_FREE_ROAD1,
            "PLACING_FREE_ROAD2", SOCGame.PLACING_FREE_ROAD2, "WAITING_FOR_DISCARDS", SOCGame.WAITING_FOR_DISCARDS,
            "WAITING_FOR_ROB_CHOOSE_PLAYER", SOCGame.WAITING_FOR_ROB_CHOOSE_PLAYER,
            "WAITING_FOR_DISCOVERY", SOCGame.WAITING_FOR_DISCOVERY,
            "WAITING_FOR_MONOPOLY", SOCGame.WAITING_FOR_MONOPOLY, "OVER", SOCGame.OVER));
        out.put("smart_strategy", SOCRobotDM.SMART_STRATEGY);
        System.out.println(Json.write(out));
    }
}
```

- [ ] **Step 5: Build and run the probe**

Run: `pgrep -fl 'arm.sh' || echo idle`, then:
```bash
bash jsettlers/build.sh
mkdir -p tests/python/data
java -cp jsettlers/build/classes:jsettlers/lib/jsettlers-2.6.10-server.jar alphasettler.jsettlers.Probe \
  > tests/python/data/jsettlers_board.json
.venv/bin/python3 -c "
import json
d = json.load(open('tests/python/data/jsettlers_board.json'))
assert d['version'] == '2.6.10', d['version']
assert len(d['land_hexes']) == 19 and len(d['edges']) == 72, (len(d['land_hexes']), len(d['edges']))
assert len(d['hex_nums']) == 37, len(d['hex_nums'])
print({k: d[k] for k in ('port_edges', 'port_facing', 'port_hexnums', 'resources', 'smart_strategy')})
"
```
Expected: `built 2 sources ...`, then a dict whose `port_edges` is `[39, 90, 156, 204, 201, 165, 114, 66, 36]` (0x27, 0x5A, 0x9C, 0xCC, 0xC9, 0xA5, 0x72, 0x42, 0x24), `port_facing` `[3, 4, 4, 5, 6, 6, 1, 2, 2]`, `port_hexnums` `[0, 2, 8, 21, 32, 35, 33, 22, 9]`, `resources` clay 1, ore 2, sheep 3, wheat 4, wood 5, unknown 6, `smart_strategy` 0. Any other value is a fact the tables below assume: report BLOCKED with the printed dict.

- [ ] **Step 6: Write the failing tests**

Create `tests/python/jsettlers_harness.py`:
```python
"""Tests that need the JSettlers harness call `require()`: a missing jar or build fails the test with
the command that provides it (never a skip; the full suite must run everything)."""

import pytest

from oracle.jsettlers import missing


def require(main: str = "ArenaMain") -> None:
    why = missing(main)
    if why is not None:
        pytest.fail(f"the JSettlers harness is needed: {why}")
```

Create `tests/python/test_jsettlers_tables.py`:
```python
import json
import subprocess
from pathlib import Path

import pytest

from alphasettler import Game
from alphasettler._engine import net_adjacency
from jsettlers_harness import require
from oracle.jsettlers import java_command, missing
from oracle.jsettlers import tables as T
from oracle.jsettlers.actions import (
    PAIRS, bundle, bundle_index, decode, maritime, offer, offer_hands, redact, year_of_plenty,
)

PROBE = json.loads((Path(__file__).parent / "data" / "jsettlers_board.json").read_text())


def test_the_committed_probe_is_what_the_jar_prints():
    require("Probe")
    out = subprocess.run(java_command("Probe"), capture_output=True, text=True, check=True).stdout
    assert json.loads(out) == PROBE


def test_our_topology_rebuilt_in_python_is_the_engines():
    adj = net_adjacency()
    assert [sorted(ns) for ns in T.TILE_NODES] == [sorted(ns) for ns in adj["tile_nodes"]]
    assert [tuple(sorted(p)) for p in adj["edge_nodes"]] == list(T.EDGE_NODES)
    assert len(T.COASTAL_EDGES) == 30


def test_tiles_map_onto_the_land_hexes():
    assert sorted(T.OUR_TILE_TO_JS_HEX) == sorted(PROBE["land_hexes"])
    assert T.OUR_TILE_TO_JS_HEX[9] == 0x77  # our centre tile (q, r) = (0, 0)
    assert all(T.JS_HEX_TO_OUR_TILE[h] == t for t, h in enumerate(T.OUR_TILE_TO_JS_HEX))


def test_hex_corners_agree_in_order():
    corners = {row[0]: row[1:] for row in PROBE["hex_nodes"]}
    for t, h in enumerate(T.OUR_TILE_TO_JS_HEX):
        assert [T.OUR_NODE_TO_JS_NODE[n] for n in T.TILE_NODES[t]] == corners[h]


def test_nodes_and_edges_are_bijections_matching_the_probe():
    assert sorted(T.JS_NODE_TO_OUR_NODE.values()) == list(range(54))
    probe_edges = {e: {a, b} for e, a, b in PROBE["edges"]}
    assert set(T.OUR_EDGE_TO_JS_EDGE) == set(probe_edges)
    for ours, js in enumerate(T.OUR_EDGE_TO_JS_EDGE):
        a, b = T.EDGE_NODES[ours]
        assert probe_edges[js] == {T.OUR_NODE_TO_JS_NODE[a], T.OUR_NODE_TO_JS_NODE[b]}
        assert set(T.js_edge_nodes(js)) == probe_edges[js]
        assert T.JS_EDGE_TO_OUR_EDGE[js] == ours


def test_ports_sit_on_our_coast_with_the_probed_facing_and_hex_numbers():
    assert T.PORT_EDGES == PROBE["port_edges"]
    assert T.PORT_FACING == PROBE["port_facing"]
    assert T.PORT_HEXNUMS == PROBE["port_hexnums"]
    assert all(T.JS_EDGE_TO_OUR_EDGE[e] in T.COASTAL_EDGES for e in T.PORT_EDGES)
    assert len({T.JS_EDGE_TO_OUR_EDGE[e] for e in T.PORT_EDGES}) == 9


def test_hex_layout_indices_match_the_probe():
    assert sorted([h, n] for h, n in PROBE["hex_nums"]) == sorted([h, n] for n, h in enumerate(T.NUM_TO_HEX))
    assert all(T.NUM_TO_HEX[T.HEX_NUM[h]] == h for h in T.NUM_TO_HEX)


def test_codes_match_the_jar():
    res = PROBE["resources"]
    assert {res[k]: T.JS_RES[res[k]] for k in ("clay", "ore", "sheep", "wheat", "wood")} == {
        res["clay"]: 1, res["ore"]: 4, res["sheep"]: 2, res["wheat"]: 3, res["wood"]: 0}
    dev = PROBE["dev_cards"]
    assert T.JS_DEV_TO_OURS == {dev["knight"]: "knight", dev["roads"]: "road_building",
                                dev["disc"]: "year_of_plenty", dev["mono"]: "monopoly",
                                **{dev[k]: "victory_point" for k in ("cap", "market", "univ", "temple", "chapel")}}
    assert (T.ROAD, T.SETTLEMENT, T.CITY) == tuple(PROBE["pieces"][k] for k in ("road", "settlement", "city"))
    assert T.STATES == PROBE["states"]
    assert (T.WATER_HEX, T.DESERT_HEX, T.MISC_PORT) == (
        PROBE["hex_types"]["water"], PROBE["hex_types"]["desert"], PROBE["hex_types"]["misc_port"])
    assert T.SMART_STRATEGY == PROBE["smart_strategy"]
    assert T.ours_to_js5([1, 0, 0, 0, 0]) == [0, 0, 0, 0, 1]  # wood is JSettlers' fifth


@pytest.mark.parametrize("hand", [[1, 0, 0, 0, 0], [0, 2, 3, 0, 1], [4, 5, 6, 7, 8]])
def test_resource_sets_round_trip(hand):
    assert T.js5_to_ours(T.ours_to_js5(hand)) == hand


def test_action_ids_follow_the_engine():
    g = Game(3)
    assert len(g.legal_actions()) == 54 and min(g.legal_actions()) == 25  # setup settlements 25..78
    assert maritime(4, 0) == 253 and year_of_plenty(3, 1) == 10 + PAIRS.index((1, 3))
    for i in range(20):
        assert bundle_index(bundle(i)) == i
    a = offer([0, 1, 0, 0, 0], [0, 0, 0, 0, 1])
    assert a == 258 + 20 * 1 + 4 and offer_hands(a) == ([0, 1, 0, 0, 0], [0, 0, 0, 0, 1])
    assert offer([1, 1, 1, 0, 0], [1, 0, 0, 0, 0]) is None
    assert decode(660 + 2) == ("confirm", 2) and decode(205 + 7) == ("move_robber", 7)


def test_redact_hides_steals_and_draws_from_others():
    stole = {"type": "stole", "thief": 1, "victim": 2, "resource": "ore"}
    assert redact(stole, 0)["resource"] is None and redact(stole, 2) == stole
    bought = {"type": "bought_dev", "player": 3, "card": "knight"}
    assert redact(bought, 1)["card"] is None and redact(bought, 3) == bought


def test_missing_names_the_command_to_run(monkeypatch, tmp_path):
    import oracle.jsettlers as js

    monkeypatch.setattr(js, "JAR", tmp_path / "absent.jar")
    assert "scripts/fetch-jsettlers.sh" in missing("Probe")
```

Run: `.venv/bin/pytest -q tests/python/test_jsettlers_tables.py`
Expected: FAIL at collection with `ModuleNotFoundError: No module named 'oracle.jsettlers'`.

- [ ] **Step 7: Write `oracle/jsettlers/__init__.py`, `tables.py` and `actions.py`**

Create `oracle/jsettlers/__init__.py`:
```python
"""JSettlers2 adapter. Python talks to the GPL harness under jsettlers/ only through JSON lines
on its JVM's stdin and stdout (jvm.py); nothing here imports Catanatron or the Catanatron modules
of `oracle`. This module only names paths, so a caller can say what is missing before any JVM runs."""

from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VERSION = "2.6.10"
JAR = ROOT / "jsettlers" / "lib" / f"jsettlers-{VERSION}-server.jar"
JAR_SHA256 = "9b26cb20ec943213600795d660fad2da32fc72629c348e180bd171cc26d6b4e2"
CLASSES = ROOT / "jsettlers" / "build" / "classes"
LOGS = ROOT / "runs" / "jsettlers" / "logs"
PACKAGE = "alphasettler.jsettlers"


def java_command(main: str, *args: str) -> list[str]:
    """The command that runs harness class `main` (e.g. "ArenaMain") with `args`."""
    return ["java", "-Xss4m", "-cp", f"{CLASSES}:{JAR}", f"{PACKAGE}.{main}", *args]


def missing(main: str = "ArenaMain") -> str | None:
    """What is missing to run harness class `main`, with the command that provides it, or None."""
    if shutil.which("java") is None:
        return "java is not on PATH (install a JDK, version 17 or later)"
    if not JAR.is_file():
        return f"{JAR} is missing; run: bash scripts/fetch-jsettlers.sh && bash jsettlers/build.sh"
    if not (CLASSES / Path(*PACKAGE.split(".")) / f"{main}.class").is_file():
        return f"the harness class {main} is not built; run: bash jsettlers/build.sh"
    return None
```

Create `oracle/jsettlers/tables.py`:
```python
"""Coordinates and codes between JSettlers2 2.6.10's classic 4-player board and AlphaSettler.

Derived by geometry and checked against the probe of the jar (tests/python/data/jsettlers_board.json,
printed by jsettlers/src/alphasettler/jsettlers/Probe.java): tiles, nodes and edges are bijections.
Both engines number a hex's corners clockwise from the top. JSettlers' hex 0xRC has row R and column
C on diagonal axes, so our axial tile (q, r) is the hex ((7 + 2q + 2r) << 4) | (7 + 2q)."""

from __future__ import annotations

from collections import Counter

# --- our topology, rebuilt as engine/src/topology.rs builds it ---
_CORNER_OFFSETS = [(0, -2), (1, -1), (1, 1), (0, 2), (-1, 1), (-1, -1)]


def _our_topology():
    coords = [(q, r) for r in range(-2, 3) for q in range(-2, 3) if abs(q + r) <= 2]

    def corner(c, i):
        (q, r), (dx, dy) = c, _CORNER_OFFSETS[i]
        return (2 * q + r + dx, 3 * r + dy)

    points = sorted({corner(c, i) for c in coords for i in range(6)}, key=lambda p: (p[1], p[0]))
    node_of = {p: n for n, p in enumerate(points)}
    tile_nodes = [[node_of[corner(c, i)] for i in range(6)] for c in coords]
    sides = [(min(a, b), max(a, b)) for ns in tile_nodes for a, b in zip(ns, ns[1:] + ns[:1])]
    edges = sorted(set(sides))
    count = Counter(sides)
    coastal = frozenset(e for e, pair in enumerate(edges) if count[pair] == 1)
    return coords, tile_nodes, edges, coastal


TILE_COORDS, TILE_NODES, EDGE_NODES, COASTAL_EDGES = _our_topology()

# --- JSettlers' v1 encoding (SOCBoard, SOCBoard4p) ---
HEXNODES = (0x01, 0x12, 0x21, 0x10, -0x01, -0x10)  # N, NE, SE, S, SW, NW
OUR_TILE_TO_JS_HEX = [((7 + 2 * q + 2 * r) << 4) | (7 + 2 * q) for q, r in TILE_COORDS]
JS_HEX_TO_OUR_TILE = {h: t for t, h in enumerate(OUR_TILE_TO_JS_HEX)}


def _node_map() -> list[int]:
    js_of: dict[int, int] = {}
    for t, h in enumerate(OUR_TILE_TO_JS_HEX):
        for i, d in enumerate(HEXNODES):
            if js_of.setdefault(TILE_NODES[t][i], h + d) != h + d:
                raise AssertionError(f"our node {TILE_NODES[t][i]} has two JSettlers coordinates")
    return [js_of[n] for n in range(54)]


OUR_NODE_TO_JS_NODE = _node_map()
JS_NODE_TO_OUR_NODE = {js: n for n, js in enumerate(OUR_NODE_TO_JS_NODE)}
assert len(JS_NODE_TO_OUR_NODE) == 54, "the node map is not a bijection"


def js_edge_nodes(edge: int) -> tuple[int, int]:
    """The two nodes of JSettlers edge `edge` (SOCBoard.getAdjacentNodesToEdge_arr, v1 encoding)."""
    if ((edge & 0x0F) + (edge >> 4)) % 2 == 0:
        return edge + 0x01, edge + 0x10
    return edge, edge + 0x11


def _js_edge(a: int, b: int) -> int:
    lo, hi = min(a, b), max(a, b)
    e = lo if hi - lo == 0x11 else lo - 1
    if set(js_edge_nodes(e)) != {lo, hi}:
        raise AssertionError(f"no JSettlers edge joins nodes {lo:#x} and {hi:#x}")
    return e


OUR_EDGE_TO_JS_EDGE = [_js_edge(OUR_NODE_TO_JS_NODE[a], OUR_NODE_TO_JS_NODE[b]) for a, b in EDGE_NODES]
JS_EDGE_TO_OUR_EDGE = {e: i for i, e in enumerate(OUR_EDGE_TO_JS_EDGE)}
assert len(JS_EDGE_TO_OUR_EDGE) == 72, "the edge map is not a bijection"

# SOCBoard.numToHexID: the hex at each index of hexLayout / numberLayout (land and the water ring).
NUM_TO_HEX = [
    0x17, 0x39, 0x5B, 0x7D,
    0x15, 0x37, 0x59, 0x7B, 0x9D,
    0x13, 0x35, 0x57, 0x79, 0x9B, 0xBD,
    0x11, 0x33, 0x55, 0x77, 0x99, 0xBB, 0xDD,
    0x31, 0x53, 0x75, 0x97, 0xB9, 0xDB,
    0x51, 0x73, 0x95, 0xB7, 0xD9,
    0x71, 0x93, 0xB5, 0xD7,
]
HEX_NUM = {h: n for n, h in enumerate(NUM_TO_HEX)}
# SOCBoard4p.PORTS_EDGE_V1 / PORTS_FACING_V1 / PORTS_HEXNUM_V1, clockwise from the upper left.
PORT_EDGES = [0x27, 0x5A, 0x9C, 0xCC, 0xC9, 0xA5, 0x72, 0x42, 0x24]
PORT_FACING = [3, 4, 4, 5, 6, 6, 1, 2, 2]
PORT_HEXNUMS = [0, 2, 8, 21, 32, 35, 33, 22, 9]

WATER_HEX, DESERT_HEX, MISC_PORT = 0, 6, 0
ROAD, SETTLEMENT, CITY = 0, 1, 2
SMART_STRATEGY = 0

RESOURCE_NAMES = ("wood", "brick", "sheep", "wheat", "ore")
DEV_NAMES = ("knight", "victory_point", "road_building", "year_of_plenty", "monopoly")
# JSettlers resource (and land hex) code -> our resource index: clay 1, ore 2, sheep 3, wheat 4, wood 5.
JS_RES = {1: 1, 2: 4, 3: 2, 4: 3, 5: 0}
OUR_RES_TO_JS = {ours: js for js, ours in JS_RES.items()}
JS_DEV_TO_OURS = {9: "knight", 1: "road_building", 2: "year_of_plenty", 3: "monopoly",
                  **{c: "victory_point" for c in range(4, 9)}}


def js5_to_ours(js5) -> list[int]:
    """JSettlers' five known amounts (clay, ore, sheep, wheat, wood) in our order."""
    clay, ore, sheep, wheat, wood = js5
    return [wood, clay, sheep, wheat, ore]


def ours_to_js5(hand) -> list[int]:
    wood, brick, sheep, wheat, ore = hand
    return [brick, ore, sheep, wheat, wood]


STATES = {"START1A": 5, "START1B": 6, "START2A": 10, "START2B": 11, "ROLL_OR_CARD": 15, "PLAY1": 20,
          "PLACING_ROAD": 30, "PLACING_SETTLEMENT": 31, "PLACING_CITY": 32, "PLACING_ROBBER": 33,
          "PLACING_FREE_ROAD1": 40, "PLACING_FREE_ROAD2": 41, "WAITING_FOR_DISCARDS": 50,
          "WAITING_FOR_ROB_CHOOSE_PLAYER": 51, "WAITING_FOR_DISCOVERY": 52, "WAITING_FOR_MONOPOLY": 53,
          "OVER": 1000}
START1A, START1B, START2A, START2B = 5, 6, 10, 11
ROLL_OR_CARD, PLAY1 = 15, 20
PLACING_ROAD, PLACING_SETTLEMENT, PLACING_CITY, PLACING_ROBBER = 30, 31, 32, 33
PLACING_FREE_ROAD1, PLACING_FREE_ROAD2 = 40, 41
WAITING_FOR_DISCARDS, WAITING_FOR_ROB_CHOOSE_PLAYER = 50, 51
WAITING_FOR_DISCOVERY, WAITING_FOR_MONOPOLY = 52, 53
OVER = 1000
SETUP_STATES = frozenset({START1A, START1B, START2A, START2B})
```

Create `oracle/jsettlers/actions.py`:
```python
"""Our engine's action ids (engine/src/action.rs `encode`) and the event redaction of
`Event::redacted_for`, kept here so oracle/jsettlers needs nothing from the Catanatron modules."""

from __future__ import annotations

ROLL, END_TURN, BUY_DEV, KNIGHT, ROAD_BUILDING = 0, 1, 2, 3, 4
ACCEPT, REJECT, CANCEL = 658, 659, 664
PAIRS = [(a, b) for a in range(5) for b in range(a, 5)]


def monopoly(r: int) -> int:
    return 5 + r


def year_of_plenty(a: int, b: int) -> int:
    return 10 + PAIRS.index((min(a, b), max(a, b)))


def settlement(n: int) -> int:
    return 25 + n


def city(n: int) -> int:
    return 79 + n


def road(e: int) -> int:
    return 133 + e


def move_robber(t: int) -> int:
    return 205 + t


def steal_from(p: int) -> int:
    return 224 + p


def discard(r: int) -> int:
    return 228 + r


def maritime(give: int, get: int) -> int:
    return 233 + 5 * give + get


def confirm(p: int) -> int:
    return 660 + p


def bundle(i: int) -> list[int]:
    h = [0] * 5
    if i < 5:
        h[i] = 1
    else:
        a, b = PAIRS[i - 5]
        h[a] += 1
        h[b] += 1
    return h


def bundle_index(hand) -> int | None:
    """The bundle id of a one- or two-card hand, or None for any other hand."""
    n = sum(hand)
    if n == 1:
        return list(hand).index(1)
    if n == 2:
        cards = sorted(r for r in range(5) for _ in range(hand[r]))
        return 5 + PAIRS.index((cards[0], cards[1]))
    return None


def offer(give, get) -> int | None:
    """OfferTrade's id for these hands, or None when either side is not a bundle."""
    g, w = bundle_index(give), bundle_index(get)
    return None if g is None or w is None else 258 + 20 * g + w


def is_offer(a: int) -> bool:
    return 258 <= a < 658


def offer_hands(a: int) -> tuple[list[int], list[int]]:
    k = a - 258
    return bundle(k // 20), bundle(k % 20)


def decode(a: int) -> tuple:
    """A readable form of action id `a`, for messages and tests."""
    named = {ROLL: ("roll",), END_TURN: ("end_turn",), BUY_DEV: ("buy_dev",), KNIGHT: ("knight",),
             ROAD_BUILDING: ("road_building",), ACCEPT: ("accept",), REJECT: ("reject",), CANCEL: ("cancel",)}
    if a in named:
        return named[a]
    for lo, hi, name in ((5, 10, "monopoly"), (10, 25, "year_of_plenty"), (25, 79, "settlement"), (79, 133, "city"),
                         (133, 205, "road"), (205, 224, "move_robber"), (224, 228, "steal_from"),
                         (228, 233, "discard"), (233, 258, "maritime"), (258, 658, "offer"), (660, 664, "confirm")):
        if lo <= a < hi:
            return (name, a - lo)
    raise ValueError(f"action id {a} is not a valid action")


def redact(event: dict, viewer: int) -> dict:
    """`event` as `viewer` may see it (the engine's `Event::redacted_for`)."""
    if event["type"] == "stole" and viewer not in (event["thief"], event["victim"]):
        return {**event, "resource": None}
    if event["type"] == "bought_dev" and viewer != event["player"]:
        return {**event, "card": None}
    return event
```

- [ ] **Step 8: Run the tests to verify they pass**

Run: `.venv/bin/pytest -q tests/python/test_jsettlers_tables.py`
Expected: PASS, 14 tests. A failure in `test_hex_corners_agree_in_order` or `test_nodes_and_edges_are_bijections_matching_the_probe` means the geometry formula is wrong for this jar: report BLOCKED with the failing hex or edge; do not fit the tables to the probe by hand.

- [ ] **Step 9: Format, lint, the full suite**

Run the Global Constraints sequence (nothing in Rust changed: no rebuild). Expected: all pass, nothing skipped; `git status --porcelain` lists neither `jsettlers/lib/` nor `jsettlers/build/`.

- [ ] **Step 10: Commit**

```bash
git add scripts/fetch-jsettlers.sh .gitignore jsettlers/COPYING jsettlers/build.sh jsettlers/check-api.sh jsettlers/api.txt \
  jsettlers/src/alphasettler/jsettlers/Json.java jsettlers/src/alphasettler/jsettlers/Probe.java \
  oracle/jsettlers/__init__.py oracle/jsettlers/tables.py oracle/jsettlers/actions.py \
  tests/python/jsettlers_harness.py tests/python/test_jsettlers_tables.py tests/python/data/jsettlers_board.json
git diff --cached | grep -nE 'sk-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{36}|password=[^[:space:]]+'
git -c user.email=sean.may101@gmail.com commit -m "jsettlers: fetch script, API check, probe and board tables for JSettlers2 2.6.10"
```

---

### Task 2: `bank_limit`, off for JSettlers' unlimited bank

**Files:**
- Modify: `engine/src/config.rs` (`GameConfig.bank_limit`, `Default`), `engine/src/state.rs` (`bank_has`, `bank_pay`, `refresh_bank`), `engine/src/apply.rs` (`apply_with` refreshes the bank), `engine/src/rules/roll.rs` (`produce`), `engine/src/rules/maritime.rs` (`legal`, `apply`), `engine/src/rules/dev.rs` (`legal_dev_plays`, `apply_year_of_plenty`), `engine/src/rules/setup.rs` (`apply_settlement`), `engine/src/invariants.rs` (two checks), `bindings/src/convert.rs` (`parse_config`, `config_dict`)
- Test: `engine/tests/bank_limit.rs` (new), `tests/python/test_bank_limit.py` (new)

**Interfaces:**
- Consumes: `State`, `GameConfig`, `BANK_PER_RESOURCE`, `hand_sub`; the test helpers of `engine/tests/common/mod.rs` (`blank_with`, `roll`).
- Produces: `GameConfig { .., bank_limit: bool }` with `GameConfig::default().bank_limit == true`; `State::bank_has(&self, r: usize, n: u8) -> bool`, `State::bank_pay(&mut self, h: &Hand)`, `State::refresh_bank(&mut self)`; the Python config key `"bank_limit"` (accepted by every function that takes a config; `resolve_config` and observation configs carry it only when it is `False`).

- [ ] **Step 1: Run the pins at the base commit**

Run: `pgrep -fl 'arm.sh' || echo idle` (wait while it prints a process), then `PATH=$HOME/.cargo/bin:$PATH cargo test --release -p settler-engine --test golden_trace && PATH=$HOME/.cargo/bin:$PATH cargo test --release -p settler-bots --test ismcts --test netbot --test selfplay -- 3a before_p2 before_3d1b before_3d3`
Expected: PASS (the golden trace and the seven named pins). They must pass unchanged after every later step.

- [ ] **Step 2: Write the failing Rust tests**

Create `engine/tests/bank_limit.rs`:
```rust
mod common;
use common::*;
use settler_engine::topology::topo;
use settler_engine::*;

fn unlimited() -> GameConfig {
    GameConfig {
        bank_limit: false,
        ..GameConfig::default()
    }
}

/// The first tile with a resource: its resource, its number and two opposite corners (never adjacent).
fn producing_tile(s: &State) -> (Resource, u8, u8, u8) {
    let t = (0..19).find(|&t| s.board.tile_resource[t].is_some()).unwrap();
    let nodes = topo().tile_nodes[t];
    (
        s.board.tile_resource[t].unwrap(),
        s.board.tile_number[t],
        nodes[0],
        nodes[3],
    )
}

/// Players 0 and 1 settled on that tile; player 2 holds all 19 of its resource, so the pile is empty.
fn two_claimants(config: GameConfig) -> State {
    let mut s = blank_with(5, Phase::PreRoll, config);
    let (r, _, a, b) = producing_tile(&s);
    s.players[0].settlements |= 1u64 << a;
    s.players[1].settlements |= 1u64 << b;
    s.players[2].hand[r.index()] = 19;
    s.bank[r.index()] = 0;
    s
}

fn held(s: &State, r: usize) -> u32 {
    s.players.iter().map(|p| p.hand[r] as u32).sum()
}

#[test]
fn the_bank_limit_is_on_by_default() {
    assert!(GameConfig::default().bank_limit);
}

#[test]
fn with_the_limit_two_claimants_of_an_empty_pile_get_none_of_it() {
    let mut s = two_claimants(GameConfig::default());
    let (r, n, _, _) = producing_tile(&s);
    roll(&mut s, n);
    assert_eq!(s.players[0].hand[r.index()], 0);
    assert_eq!(s.players[1].hand[r.index()], 0);
    s.check_invariants().unwrap();
}

#[test]
fn without_the_limit_both_claimants_are_paid_and_the_bank_shows_none() {
    let mut s = two_claimants(unlimited());
    let (r, n, _, _) = producing_tile(&s);
    roll(&mut s, n);
    assert!(s.players[0].hand[r.index()] >= 1);
    assert!(s.players[1].hand[r.index()] >= 1);
    assert!(held(&s, r.index()) > 19);
    assert_eq!(s.bank[r.index()], 0);
    s.check_invariants().unwrap();
}

#[test]
fn without_the_limit_maritime_trades_and_year_of_plenty_pay_from_an_empty_pile() {
    for config in [GameConfig::default(), unlimited()] {
        let mut s = blank_with(2, Phase::Main, config);
        give(&mut s, 2, [0, 0, 0, 0, 19]);
        give(&mut s, 0, [4, 0, 0, 0, 0]);
        s.players[0].dev_hand[DevCard::YearOfPlenty.index()] = 1;
        let trade = Action::MaritimeTrade {
            give: Resource::Wood,
            get: Resource::Ore,
        };
        let plenty = Action::PlayYearOfPlenty(Resource::Ore, Resource::Ore);
        let legal = s.legal_actions();
        assert_eq!(legal.contains(&trade), !config.bank_limit, "{config:?}");
        assert_eq!(legal.contains(&plenty), !config.bank_limit, "{config:?}");
        if !config.bank_limit {
            s.apply(trade);
            s.apply(plenty);
            assert_eq!(s.players[0].hand, [0, 0, 0, 0, 3]);
            assert_eq!(s.bank[Resource::Ore.index()], 0);
            assert_eq!(s.bank[Resource::Wood.index()], 19);
        }
    }
}

#[test]
fn a_snapshot_with_more_than_19_of_a_resource_in_hands_imports_only_without_the_limit() {
    let mut s = blank_with(1, Phase::Main, unlimited());
    s.players[0].hand = [20, 0, 0, 0, 0];
    s.refresh_bank();
    assert_eq!(s.bank, [0, 19, 19, 19, 19]);
    let snap = s.snapshot();
    assert!(State::from_snapshot(1, unlimited(), &snap).is_ok());
    let err = State::from_snapshot(1, GameConfig::default(), &snap).unwrap_err();
    assert!(err.contains("hands and bank hold"), "{err}");
}

#[test]
fn games_without_the_limit_show_the_derived_bank_after_every_action() {
    for seed in 0..12u64 {
        let mut s = State::new(seed, unlimited());
        let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        while !s.is_over() {
            let legal = s.legal_actions();
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            s.apply(legal[(x % legal.len() as u64) as usize]);
            for r in 0..NUM_RESOURCES {
                assert_eq!(
                    s.bank[r] as u32,
                    19u32.saturating_sub(held(&s, r)),
                    "seed {seed} resource {r}"
                );
            }
            s.check_invariants().unwrap();
        }
    }
}
```

Run: `PATH=$HOME/.cargo/bin:$PATH cargo test --release -p settler-engine --test bank_limit`
Expected: FAIL to compile: `struct GameConfig has no field named bank_limit` and `no method named refresh_bank`.

- [ ] **Step 3: Write the engine change**

In `engine/src/config.rs`, add the field after `catanatron_compat` and its default:
```rust
    /// The bank holds 19 cards of each resource and runs short (the default). Off, it never runs
    /// short, as in JSettlers2: production, maritime trades and Year of Plenty always pay, and
    /// `bank` shows what a 19-card bank would hold, 19 less the cards in hands, never below 0.
    pub bank_limit: bool,
```
```rust
            catanatron_compat: false,
            bank_limit: true,
```

In `engine/src/state.rs`, add to `impl State` after `reseed`:
```rust
    /// Whether the bank can pay `n` cards of resource `r`: always without the bank limit.
    #[inline]
    pub fn bank_has(&self, r: usize, n: u8) -> bool {
        !self.config.bank_limit || self.bank[r] >= n
    }

    /// Pay `h` out of the bank. Without the bank limit the bank is only a display (see
    /// `refresh_bank`), so it stops at 0 instead of running short.
    #[inline]
    pub fn bank_pay(&mut self, h: &Hand) {
        if self.config.bank_limit {
            hand_sub(&mut self.bank, h);
        } else {
            for r in 0..NUM_RESOURCES {
                self.bank[r] = self.bank[r].saturating_sub(h[r]);
            }
        }
    }

    /// Without the bank limit: set the bank to 19 less the cards in hands, never below 0.
    pub fn refresh_bank(&mut self) {
        for r in 0..NUM_RESOURCES {
            let held: u32 = self.players.iter().map(|p| p.hand[r] as u32).sum();
            self.bank[r] = (BANK_PER_RESOURCE as u32).saturating_sub(held) as u8;
        }
    }
```

In `engine/src/apply.rs`, `apply_with`, replace the last line of the function body (`awards::check_win(self, sink);`) with:
```rust
        if !self.config.bank_limit {
            self.refresh_bank();
        }
        awards::check_win(self, sink);
```

In `engine/src/rules/roll.rs`, `produce`: wrap the shortage loop and pay through `bank_pay`:
```rust
    if s.config.bank_limit {
        for r in 0..NUM_RESOURCES {
            let total: u32 = (0..NUM_PLAYERS).map(|p| owed[p][r] as u32).sum();
            if total <= s.bank[r] as u32 {
                continue;
            }
            let claimants: Vec<usize> = (0..NUM_PLAYERS).filter(|&p| owed[p][r] > 0).collect();
            if claimants.len() == 1 {
                owed[claimants[0]][r] = s.bank[r];
            } else {
                for p in 0..NUM_PLAYERS {
                    owed[p][r] = 0;
                }
            }
        }
    }
    for p in 0..NUM_PLAYERS {
        if hand_total(&owed[p]) > 0 {
            hand_add(&mut s.players[p].hand, &owed[p]);
            s.bank_pay(&owed[p]);
            sink.emit(Event::Produced {
                player: p as PlayerId,
                resources: owed[p],
            });
        }
    }
```

In `engine/src/rules/maritime.rs`: in `legal`, replace `s.bank[get.index()] >= 1` with `s.bank_has(get.index(), 1)`; in `apply`, replace `hand_sub(&mut s.bank, &got);` with `s.bank_pay(&got);`.

In `engine/src/rules/dev.rs`, `legal_dev_plays`: replace the two bank conditions with
```rust
            let ok = if a == b {
                s.bank_has(a as usize, 2)
            } else {
                s.bank_has(a as usize, 1) && s.bank_has(b as usize, 1)
            };
```
and in `apply_year_of_plenty` replace `hand_sub(&mut s.bank, &got);` with `s.bank_pay(&got);`.

In `engine/src/rules/setup.rs`, `apply_settlement`: replace `hand_sub(&mut s.bank, &got);` with `s.bank_pay(&got);` and drop `hand_sub` from the `use crate::types::{...}` line (clippy flags the unused import).

In `engine/src/invariants.rs`: wrap the conservation loop at the top of `check_invariants` in `if self.config.bank_limit { ... }`, and change the second-settlement check's `if step >= 4 {` to `if step >= 4 && self.config.bank_limit {`. Add one line to the doc comment of `check_invariants`: `/// Without the bank limit the bank is derived from the hands, so its conservation is not checked.`

- [ ] **Step 4: Run the Rust tests to verify they pass**

Run: `PATH=$HOME/.cargo/bin:$PATH cargo test --release -p settler-engine --test bank_limit`
Expected: PASS, 6 tests.

Run: Step 1's two commands.
Expected: PASS, unchanged.

- [ ] **Step 5: Write the failing Python tests**

Create `tests/python/test_bank_limit.py`:
```python
import random

import pytest

from alphasettler import Game
from alphasettler._engine import resolve_config

UNLIMITED = {"bank_limit": False}


def test_resolved_configs_name_the_bank_limit_only_when_it_is_off():
    assert "bank_limit" not in resolve_config({})
    assert "bank_limit" not in resolve_config({"bank_limit": True})
    assert resolve_config(UNLIMITED)["bank_limit"] is False
    assert resolve_config(UNLIMITED) == {**resolve_config({}), "bank_limit": False}


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_an_unlimited_game_shows_19_less_the_cards_in_hands(seed):
    g, rng = Game(seed, UNLIMITED), random.Random(seed)
    assert g.observation(0)["config"]["bank_limit"] is False
    while not g.is_over:
        g.apply(rng.choice(g.legal_actions()))
        s = g.snapshot()
        held = [sum(p["hand"][r] for p in s["players"]) for r in range(5)]
        assert s["bank"] == [max(0, 19 - h) for h in held]


def test_a_position_with_20_cards_of_a_resource_imports_only_without_the_limit():
    snap = Game(4).snapshot()  # the opening position, with player 0 given twenty wood
    snap["players"][0]["hand"] = [20, 0, 0, 0, 0]
    snap["bank"] = [0, 19, 19, 19, 19]
    Game.from_snapshot(4, snap, UNLIMITED)
    with pytest.raises(ValueError, match="hands and bank hold"):
        Game.from_snapshot(4, snap, {})
```

Run: `.venv/bin/pytest -q tests/python/test_bank_limit.py`
Expected: FAIL: `ValueError: unknown config key "bank_limit"`.

- [ ] **Step 6: Write the binding**

In `bindings/src/convert.rs`, `parse_config`, add after the `catanatron_compat` arm:
```rust
                "bank_limit" => c.bank_limit = v.extract()?,
```
and in `config_dict`, after the `catanatron_compat` line:
```rust
    // Written only when off, so every resolved config and record file from before is unchanged.
    if !c.bank_limit {
        d.set_item("bank_limit", false)?;
    }
```

- [ ] **Step 7: Rebuild and run the Python tests**

Rebuild the extension (Global Constraints). Run: `.venv/bin/pytest -q tests/python/test_bank_limit.py tests/python/test_cli.py tests/python/test_engine.py tests/python/test_snapshot.py`
Expected: PASS (5 new tests; the existing files unchanged).

- [ ] **Step 8: Format, lint, the full suite**

Run the Global Constraints sequence. Expected: all pass, nothing skipped; the pins of Step 1 pass unchanged.

- [ ] **Step 9: Commit**

```bash
git add engine/src/config.rs engine/src/state.rs engine/src/apply.rs engine/src/rules/roll.rs engine/src/rules/maritime.rs \
  engine/src/rules/dev.rs engine/src/rules/setup.rs engine/src/invariants.rs bindings/src/convert.rs \
  engine/tests/bank_limit.rs tests/python/test_bank_limit.py
git diff --cached | grep -nE 'sk-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{36}|password=[^[:space:]]+'
git -c user.email=sean.may101@gmail.com commit -m "engine: bank_limit setting (default on, every game unchanged); off, the bank never runs short, as in JSettlers2"
```

---

### Task 3: The in-JVM arena server: injected boards, routed dice and the event stream

**Files:**
- Create: `jsettlers/src/alphasettler/jsettlers/{Pipe,GameSpec,RoutedRandom,ArenaBoard,ArenaBoardFactory,Snapshots,ArenaServer,ArenaMain}.java`, `oracle/jsettlers/jvm.py`, `oracle/jsettlers/boards.py`
- Test: `tests/python/test_jsettlers_jvm.py` (new)

**Interfaces:**
- Consumes: Task 1's `Json`, `tables`, `java_command`, `missing`, `LOGS`; `alphasettler.Game` (`snapshot()["board"]`); the jar members of `jsettlers/api.txt`.
- Produces:
  - The pipe protocol. Python to Java, one JSON object a line: `{"t": "game", "g": NAME, "layout": {"hexes": [37 ints], "numbers": [37 ints], "robber": HEX} | null, "first": JS_SEAT, "dice": LONG, "deck": LONG, "misc": LONG, "nt": BOOL, "bridge": BOOL}`, `{"t": "reply", "id": N, ...}` (Task 6), `{"t": "abort", "g": NAME}`, `{"t": "quit"}`. Java to Python: `{"t": "ready", "version": "2.6.10", "mode": MODE, "robots": [names]}`; `{"t": "start", "g", "first", "seats": [4 names by JS seat], "smart": [4 strategy types], "board": BOARD_RAW, "s": RAW}`; `{"t": "ev", "g", "a": AUDIENCE, "m": MESSAGE_TEXT, "s": RAW (only when the server-side game changed)}`; `{"t": "ask", ...}` and `{"t": "fallback", ...}` (Task 6); `{"t": "over", "g", "winner": JS_SEAT or -1, "vp": [4], "rounds": N, "aborted": BOOL}`; `{"t": "error", "what": TEXT}` (then the JVM exits 1). AUDIENCE is GameEventLog's prefix: `all`, `pN`, `!pN`, `!p[a, b]`, `fN` (a client's request), `fo`, `ob`, `un`.
  - RAW (`Snapshots.raw`): `{"state", "current", "robber", "deck": [JS card codes, next draw first], "players": [{"res": [clay, ore, sheep, wheat, wood], "dev": [10 counts by JS card code], "played": [knights, road building, discovery, monopoly], "settlements": [nodes], "cities": [nodes], "roads": [edges], "discard": N, "played_dev": BOOL, "vp": N} x4], "longest", "largest", "winner"}` (seats, -1 for none). BOARD_RAW (`Snapshots.board`): `{"hexes": [[hex, type, number] x19], "ports": [[edge, port type] x9 in PORT_EDGES order], "robber": HEX}`.
  - `oracle.jsettlers.jvm`: `Jvm(mode="stock", log=None)` with `.ready`, `.log`, `.send(msg)`, `.recv(timeout) -> dict`, `.close()`, context manager; `JvmError(RuntimeError)`; `READY_TIMEOUT = 120.0`.
  - `oracle.jsettlers.boards`: `native_board(seed) -> dict`, `js_layout(board) -> dict`, `our_board(board_raw) -> dict`, `streams(seed) -> {"dice", "deck", "misc"}`, `first_player(k) -> int`, `game_command(name, seed, k, *, injected=True, no_trades=False, bridge=True) -> dict`, `normalized(board) -> dict`.

- [ ] **Step 1: Write the failing tests**

Create `tests/python/test_jsettlers_jvm.py`:
```python
import pytest

from alphasettler import Game
from jsettlers_harness import require
from oracle.jsettlers import tables as T
from oracle.jsettlers.boards import (
    first_player, game_command, js_layout, native_board, normalized, our_board, streams,
)
from oracle.jsettlers.jvm import Jvm, JvmError

GAME_TIMEOUT = 600.0  # a robots-only game at the fast pace takes well under a minute


@pytest.fixture(scope="module")
def jvm(tmp_path_factory):
    require()
    with Jvm(log=tmp_path_factory.mktemp("jvm") / "jvm.log") as j:  # suite runs leave no logs under runs/
        yield j


def play(jvm, cmd, stop_after_rolls=None):
    """Messages of one game until its "over"; with `stop_after_rolls`, abort after that many rolls."""
    jvm.send(cmd)
    msgs, rolls, aborted = [], 0, False
    while True:
        m = jvm.recv(GAME_TIMEOUT)
        assert m.get("g") == cmd["g"], m
        msgs.append(m)
        if m["t"] == "ev" and m["m"].startswith("SOCDiceResult:"):
            rolls += 1
            if stop_after_rolls is not None and rolls >= stop_after_rolls and not aborted:
                jvm.send({"t": "abort", "g": cmd["g"]})
                aborted = True
        if m["t"] == "over":
            return msgs


def rolls_of(msgs):
    return [int(m["m"].rsplit("param=", 1)[1]) for m in msgs if m["t"] == "ev" and m["m"].startswith("SOCDiceResult:")]


def test_streams_and_layouts_are_pure_functions_of_the_seed():
    assert streams(5) == streams(5) and streams(5) != streams(6)
    assert all(0 <= v < 2**63 for v in streams(5).values())
    assert [first_player(k) for k in range(4)] == [0, 3, 2, 1]
    b = native_board(11)
    assert b["tile_resource"] == Game(11).snapshot()["board"]["tile_resource"]
    assert [T.OUR_EDGE_TO_JS_EDGE[e] for e, _ in b["ports"]] == T.PORT_EDGES
    assert sorted(k for _, k in b["ports"]) == sorted(k for _, k in Game(11).snapshot()["board"]["ports"])
    lay = js_layout(b)
    desert = b["tile_resource"].index(None)
    assert lay["robber"] == T.OUR_TILE_TO_JS_HEX[desert]
    assert lay["hexes"][T.HEX_NUM[lay["robber"]]] == T.DESERT_HEX
    assert sum(1 for h in lay["hexes"] if 1 <= h <= 5) == 18
    assert all(lay["hexes"][n] >= 7 for n in T.PORT_HEXNUMS)


def test_a_robots_game_runs_to_a_winner_on_the_injected_board_with_the_scripted_first_player(jvm):
    msgs = play(jvm, game_command("j-full", 7, 1, bridge=False))
    start = next(m for m in msgs if m["t"] == "start")
    over = msgs[-1]
    assert start["first"] == first_player(1) == 3
    assert start["seats"] == ["robot 1", "robot 2", "robot 3", "robot 4"]
    assert start["smart"] == [T.SMART_STRATEGY] * 4
    assert normalized(our_board(start["board"])) == normalized(native_board(7))
    assert start["s"]["robber"] == start["board"]["robber"]
    assert over["aborted"] is False and 0 <= over["winner"] < 4 and over["vp"][over["winner"]] >= 10
    assert any(m["t"] == "ev" and m["a"].startswith("f") for m in msgs), "client requests are recorded"
    assert all("s" not in m or isinstance(m["s"]["players"], list) for m in msgs if m["t"] == "ev")


def test_dice_and_the_deck_follow_the_seed_in_every_rotation(jvm):
    a = play(jvm, game_command("j-crn-a", 21, 0, bridge=False), stop_after_rolls=12)
    b = play(jvm, game_command("j-crn-b", 21, 2, bridge=False), stop_after_rolls=12)
    c = play(jvm, game_command("j-crn-c", 22, 0, bridge=False), stop_after_rolls=12)
    assert rolls_of(a)[:12] == rolls_of(b)[:12]
    assert rolls_of(a)[:12] != rolls_of(c)[:12]
    deck = [next(m for m in g if m["t"] == "start")["s"]["deck"] for g in (a, b, c)]
    assert deck[0] == deck[1] != deck[2] and len(deck[0]) == 25
    assert all(g[-1]["aborted"] is True and g[-1]["winner"] == -1 for g in (a, b, c))


def test_native_boards_are_jsettlers_own(jvm):
    msgs = play(jvm, game_command("j-native", 7, 0, injected=False, bridge=False), stop_after_rolls=1)
    board = our_board(next(m for m in msgs if m["t"] == "start")["board"])
    snap = {**Game(7).snapshot(), "board": board, "robber": board["tile_resource"].index(None)}
    Game.from_snapshot(7, snap, {})  # a valid board of ours
    assert normalized(board) != normalized(native_board(7))


def test_errors_and_silence_fail_loudly_naming_the_log(tmp_path):
    require()
    with Jvm(log=tmp_path / "jvm.log") as j:
        with pytest.raises(TimeoutError, match=str(j.log)):
            j.recv(0.5)
        j.send({"t": "nonsense"})
        with pytest.raises(JvmError, match="unknown command") as e:
            j.recv(60.0)
        assert str(j.log) in str(e.value)
        assert j.proc.wait(timeout=30) == 1


def test_the_jvm_exits_when_its_stdin_closes(tmp_path):
    require()
    j = Jvm(log=tmp_path / "jvm.log")
    j.proc.stdin.close()
    assert j.proc.wait(timeout=30) == 0
    j.close()
```

Run: `.venv/bin/pytest -q tests/python/test_jsettlers_jvm.py`
Expected: FAIL at collection: `ModuleNotFoundError: No module named 'oracle.jsettlers.boards'`.

- [ ] **Step 2: The pipe, the game spec, the routed random source and the board**

Each file starts with the five-line header of Task 1 Step 1.

`jsettlers/src/alphasettler/jsettlers/Pipe.java`:
```java
package alphasettler.jsettlers;

import java.io.PrintStream;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.concurrent.atomic.AtomicLong;

/**
 * JSON lines to the Python driver on the real stdout (JSettlers' own prints go to stderr), written
 * in the order {@link #send} was called by one writer thread, and asks that block their caller
 * until the matching reply arrives on stdin.
 */
final class Pipe {
    private final PrintStream out;
    private final LinkedBlockingQueue<String> queue = new LinkedBlockingQueue<>();
    private final Map<Long, CompletableFuture<Map<String, Object>>> waiting = new ConcurrentHashMap<>();
    private final AtomicLong ids = new AtomicLong();

    Pipe(PrintStream out) {
        this.out = out;
        Thread w = new Thread(this::drain, "pipe-writer");
        w.setDaemon(true);
        w.start();
    }

    void send(Map<String, Object> m) {
        queue.add(Json.write(m));
    }

    private void drain() {
        try {
            while (true) {
                String line = queue.take();
                synchronized (out) {
                    out.println(line);
                    out.flush();
                }
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }

    /** Sends {@code m} with a fresh "id" and waits for its reply; null if none came within {@code seconds}. */
    Map<String, Object> ask(Map<String, Object> m, long seconds) {
        long id = ids.incrementAndGet();
        CompletableFuture<Map<String, Object>> f = new CompletableFuture<>();
        waiting.put(id, f);
        m.put("id", id);
        send(m);
        try {
            return f.get(seconds, TimeUnit.SECONDS);
        } catch (TimeoutException e) {
            return null;
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            return null;
        } catch (ExecutionException e) {
            throw new IllegalStateException(e);
        } finally {
            waiting.remove(id);
        }
    }

    /** A reply from Python; one whose ask already timed out is dropped (the asker has fallen back). */
    void reply(Map<String, Object> m) {
        CompletableFuture<Map<String, Object>> f = waiting.get(Json.num(m, "id"));
        if (f != null) f.complete(m);
    }

    /** Writes what was queued, then exits with {@code code}. */
    void flushAndExit(int code) {
        long deadline = System.currentTimeMillis() + 5000;
        while (!queue.isEmpty() && System.currentTimeMillis() < deadline) {
            try {
                Thread.sleep(10);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                break;
            }
        }
        synchronized (out) {
            out.flush();
        }
        Runtime.getRuntime().halt(code);
    }

    /** Reports a failure to Python and the log, then exits 1. */
    void fatal(String what) {
        System.err.println("arena harness failed: " + what);
        synchronized (out) {
            out.println(Json.write(Json.map("t", "error", "what", what)));
            out.flush();
        }
        Runtime.getRuntime().halt(1);
    }
}
```

`jsettlers/src/alphasettler/jsettlers/GameSpec.java`:
```java
package alphasettler.jsettlers;

import java.util.Map;

/**
 * One game the driver asked for: its name, the injected board (or null for JSettlers' own), the
 * scripted first player, the seeds of the dice, deck and misc streams, the no-trading option and
 * whether JSettlers seat 0 is the bridge.
 */
record GameSpec(String name, int[] hexes, int[] numbers, int robber, int first, long dice, long deck, long misc,
                boolean noTrades, boolean bridge) {

    static GameSpec parse(Map<String, Object> m) {
        Map<String, Object> layout = Json.sub(m, "layout");
        int first = Json.i(m, "first");
        if (first < 0 || first > 3) throw new IllegalArgumentException("first player must be 0..3: " + m);
        int[] hexes = layout == null ? null : Json.intArray(layout, "hexes");
        int[] numbers = layout == null ? null : Json.intArray(layout, "numbers");
        if (hexes != null && (hexes.length != 37 || numbers.length != 37))
            throw new IllegalArgumentException("a layout has 37 hexes and 37 numbers: " + m);
        return new GameSpec(Json.str(m, "g"), hexes, numbers, layout == null ? -1 : Json.i(layout, "robber"), first,
            Json.num(m, "dice"), Json.num(m, "deck"), Json.num(m, "misc"), Json.bool(m, "nt"), Json.bool(m, "bridge"));
    }

    boolean injected() {
        return hexes != null;
    }
}
```

`jsettlers/src/alphasettler/jsettlers/RoutedRandom.java`:
```java
package alphasettler.jsettlers;

import java.util.Random;

/**
 * Replaces {@code SOCGame.rand} (a private field, set by reflection before the game starts).
 * {@code nextInt()} is routed by its caller's method name: {@code startGame} (the first-player
 * draw) gets the scripted seat, {@code rollDice} draws each die from the seed's dice stream,
 * {@code startGame_setupDevCards} shuffles the deck from the deck stream, and every other caller
 * (the steals) uses the misc stream. So a seed's k-th roll and its deck order are the same in
 * every rotation and arm, whatever the robots do in between.
 */
final class RoutedRandom extends Random {
    private final int first;
    private final Random dice;
    private final Random deck;
    private final Random misc;

    RoutedRandom(GameSpec spec) {
        super(spec.misc());
        first = spec.first();
        dice = new Random(spec.dice());
        deck = new Random(spec.deck());
        misc = new Random(spec.misc());
    }

    @Override
    public int nextInt() {
        String caller = StackWalker.getInstance()
            .walk(frames -> frames.skip(1).findFirst().map(StackWalker.StackFrame::getMethodName).orElse(""));
        return switch (caller) {
            case "startGame" -> first;
            case "rollDice" -> dice.nextInt(6);
            case "startGame_setupDevCards" -> deck.nextInt(Integer.MAX_VALUE);
            default -> misc.nextInt(Integer.MAX_VALUE);
        };
    }
}
```

`jsettlers/src/alphasettler/jsettlers/ArenaBoard.java`:
```java
package alphasettler.jsettlers;

import soc.game.SOCBoard4p;
import soc.game.SOCGameOptionSet;

/**
 * The classic 4-player board, laid out from the driver's layout instead of JSettlers' shuffle when
 * the game was created with one (ports are encoded in the hex layout, as the v1 encoding does).
 */
final class ArenaBoard extends SOCBoard4p {
    private final transient GameSpec spec;

    ArenaBoard(SOCGameOptionSet opts, GameSpec spec) {
        super(opts);
        this.spec = spec;
    }

    @Override
    public void makeNewBoard(SOCGameOptionSet opts) {
        if (spec == null || !spec.injected()) {
            super.makeNewBoard(opts);
            return;
        }
        setHexLayout(spec.hexes().clone());
        setNumberLayout(spec.numbers().clone());
        setRobberHex(spec.robber(), false);
    }
}
```

`jsettlers/src/alphasettler/jsettlers/ArenaBoardFactory.java`:
```java
package alphasettler.jsettlers;

import soc.game.SOCBoard;
import soc.game.SOCGameOptionSet;
import soc.server.SOCBoardAtServer;

/**
 * Installed as {@code SOCGame.boardFactory} before the server exists (it must extend
 * BoardFactoryAtServer, or SOCGameListAtServer replaces it). The arena sets {@link #pending} around
 * its own createGameAndBroadcast call, which builds the game's board on the same thread; every other
 * board (the robots' copies of the game) is JSettlers' usual one.
 */
final class ArenaBoardFactory extends SOCBoardAtServer.BoardFactoryAtServer {
    static volatile GameSpec pending;

    @Override
    public SOCBoard createBoard(SOCGameOptionSet opts, boolean largeBoard, int maxPlayers) {
        GameSpec spec = pending;
        if (!largeBoard && maxPlayers == 4 && spec != null) return new ArenaBoard(opts, spec);
        return super.createBoard(opts, largeBoard, maxPlayers);
    }
}
```

- [ ] **Step 3: The snapshots, the server and the main loop**

`jsettlers/src/alphasettler/jsettlers/Snapshots.java`:
```java
package alphasettler.jsettlers;

import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Map;
import soc.game.SOCBoard;
import soc.game.SOCGame;
import soc.game.SOCPlayer;
import soc.game.SOCPlayingPiece;
import soc.game.SOCResourceConstants;
import soc.game.SOCResourceSet;

/** The server-side game in JSettlers' own coordinates and codes; oracle/jsettlers/translate.py maps it. */
final class Snapshots {
    private static final Field DECK;

    static {
        try {
            DECK = SOCGame.class.getDeclaredField("devCardDeck");
            DECK.setAccessible(true);
        } catch (NoSuchFieldException e) {
            throw new ExceptionInInitializerError(e);
        }
    }

    private Snapshots() {}

    static Map<String, Object> raw(SOCGame ga) {
        List<Object> players = new ArrayList<>();
        for (int pn = 0; pn < ga.maxPlayers; pn++) {
            SOCPlayer pl = ga.getPlayer(pn);
            SOCResourceSet r = pl.getResources();
            int[] res = {r.getAmount(SOCResourceConstants.CLAY), r.getAmount(SOCResourceConstants.ORE),
                r.getAmount(SOCResourceConstants.SHEEP), r.getAmount(SOCResourceConstants.WHEAT),
                r.getAmount(SOCResourceConstants.WOOD)};
            int[] dev = new int[10];
            for (int c = 0; c < 10; c++) dev[c] = pl.getInventory().getAmount(c);
            int[] played = {pl.getNumKnights(), pl.numRBCards, pl.numDISCCards, pl.numMONOCards};
            players.add(Json.map("res", res, "dev", dev, "played", played,
                "settlements", coords(pl.getSettlements()), "cities", coords(pl.getCities()),
                "roads", coords(pl.getRoadsAndShips()),
                "discard", pl.getNeedToDiscard() ? pl.getCountToDiscard() : 0,
                "played_dev", pl.hasPlayedDevCard(), "vp", pl.getTotalVP()));
        }
        return Json.map("state", ga.getGameState(), "current", ga.getCurrentPlayerNumber(),
            "robber", ga.getBoard().getRobberHex(), "deck", deck(ga), "players", players,
            "longest", pn(ga.getPlayerWithLongestRoad()), "largest", pn(ga.getPlayerWithLargestArmy()),
            "winner", pn(ga.getPlayerWithWin()));
    }

    static Map<String, Object> board(SOCGame ga) {
        SOCBoard b = ga.getBoard();
        List<Object> hexes = new ArrayList<>();
        for (int h : b.getLandHexCoords()) hexes.add(Json.list(h, b.getHexTypeFromCoord(h), b.getNumberOnHexFromCoord(h)));
        List<Object> ports = new ArrayList<>();
        for (int e : b.getPortsEdges()) ports.add(Json.list(e, b.getPortTypeFromNodeCoord(b.getAdjacentNodesToEdge_arr(e)[0])));
        return Json.map("hexes", hexes, "ports", ports, "robber", b.getRobberHex());
    }

    /** The undrawn cards, the next draw first (SOCGame draws devCardDeck[numDevCards - 1]). */
    static List<Object> deck(SOCGame ga) {
        List<Object> out = new ArrayList<>();
        try {
            int[] d = (int[]) DECK.get(ga);
            if (d == null) return out;
            for (int i = ga.getNumDevCards() - 1; i >= 0; i--) out.add(d[i]);
        } catch (IllegalAccessException e) {
            throw new IllegalStateException(e);
        }
        return out;
    }

    private static int pn(SOCPlayer p) {
        return p == null ? -1 : p.getPlayerNumber();
    }

    private static List<Object> coords(Collection<? extends SOCPlayingPiece> pieces) {
        List<Object> out = new ArrayList<>();
        pieces.stream().map(SOCPlayingPiece::getCoordinates).sorted().forEach(out::add);
        return out;
    }
}
```

`jsettlers/src/alphasettler/jsettlers/ArenaServer.java`:
```java
package alphasettler.jsettlers;

import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import soc.game.SOCGame;
import soc.game.SOCGameOption;
import soc.game.SOCGameOptionSet;
import soc.message.SOCMessage;
import soc.message.SOCMessageForGame;
import soc.server.SOCServer;
import soc.server.genericServer.Connection;

/**
 * JSettlers' own server on a string port, recording every game event (with its audience) to the
 * pipe, plus the server-side position whenever it changed. Games are created with the arena's
 * options, board, routed random source and seats.
 */
final class ArenaServer extends SOCServer {
    static final String PORT = "alphasettler-arena";
    static final String COOKIE = "arena";
    static final String BRIDGE = "robot bridge";
    private static final Field RAND;
    private static final Method ASK_ROBOTS;

    static {
        try {
            RAND = SOCGame.class.getDeclaredField("rand");
            RAND.setAccessible(true);
            ASK_ROBOTS = SOCServer.class.getDeclaredMethod("readyGameAskRobotsJoin",
                SOCGame.class, boolean[].class, Connection[].class, int.class);
            ASK_ROBOTS.setAccessible(true);
        } catch (ReflectiveOperationException e) {
            throw new ExceptionInInitializerError(e);
        }
    }

    private final Pipe pipe;
    private final Map<String, GameSpec> specs = new ConcurrentHashMap<>();
    private final Map<String, String> lastRaw = new ConcurrentHashMap<>();
    private final Set<String> started = ConcurrentHashMap.newKeySet();
    private final Set<String> finished = ConcurrentHashMap.newKeySet();
    private volatile GameSpec creating;
    /** Serializes recording and aborting (see {@link #record}). */
    private final Object recording = new Object();

    ArenaServer(Pipe pipe) {
        super(PORT, props());
        this.pipe = pipe;
    }

    static Properties props() {
        Properties p = new Properties();
        p.setProperty(SOCServer.PROP_JSETTLERS_STARTROBOTS, "0");
        p.setProperty(SOCServer.PROP_JSETTLERS_BOTS_COOKIE, COOKIE);
        p.setProperty(SOCServer.PROP_JSETTLERS_BOTS_TIMEOUT_TURN, "600");
        return p;
    }

    /** The robot connections' names, once every one of {@code names} has logged in. */
    boolean hasRobots(List<String> names) {
        List<String> have = new ArrayList<>();
        synchronized (robots) {
            for (Connection c : robots) have.add(c.getData());
        }
        return have.containsAll(names);
    }

    private Connection robot(String name) {
        synchronized (robots) {
            for (Connection c : robots) if (name.equals(c.getData())) return c;
        }
        throw new IllegalStateException("no robot named " + name + " is connected");
    }

    void create(GameSpec spec) throws ReflectiveOperationException {
        // A finished game's name stays taken: its late events are dropped by name, so a reused name would go silent.
        if (specs.containsKey(spec.name()) || finished.contains(spec.name()))
            throw new IllegalArgumentException("game name already used: " + spec.name());
        SOCGameOptionSet opts = new SOCGameOptionSet();
        for (SOCGameOption opt : knownOpts) {
            if (!((opt.key.charAt(0) == '_') || opt.hasFlag(SOCGameOption.FLAG_INACTIVE_HIDDEN)
                  || opt.hasFlag(SOCGameOption.FLAG_INTERNAL_GAME_PROPERTY)))
                opts.put(opt);
        }
        opts = new SOCGameOptionSet(opts, true);
        opts.get("PL").setIntValue(4);
        if (spec.injected()) opts.get("BC").setBoolValue(false);
        opts.get("NT").setBoolValue(spec.noTrades());
        specs.put(spec.name(), spec);
        creating = spec;
        ArenaBoardFactory.pending = spec;
        SOCGame ga;
        try {
            ga = createGameAndBroadcast(null, spec.name(), opts, true);
        } finally {
            ArenaBoardFactory.pending = null;
            creating = null;
        }
        if (ga == null) throw new IllegalStateException("the server did not create game " + spec.name());
        if (!(ga.getBoard() instanceof ArenaBoard)) throw new IllegalStateException("game " + spec.name() + " did not get the arena's board");
        if (!(RAND.get(ga) instanceof RoutedRandom)) throw new IllegalStateException("game " + spec.name() + " did not get the routed random source");
        String[] names = spec.bridge()
            ? new String[] {BRIDGE, "robot 1", "robot 2", "robot 3"}
            : new String[] {"robot 1", "robot 2", "robot 3", "robot 4"};
        Connection[] seats = new Connection[4];
        for (int i = 0; i < 4; i++) seats[i] = robot(names[i]);
        ga.setGameState(SOCGame.READY);
        if (!(Boolean) ASK_ROBOTS.invoke(this, ga, null, seats, 0))
            throw new IllegalStateException("no robots could join " + spec.name());
    }

    void abort(String g) {
        SOCGame ga = getGame(g);
        if (ga == null) return;
        synchronized (recording) {  // no event of g is queued after its "over"
            if (!finished.add(g)) return;
            pipe.send(over(ga, true));
        }
        destroyGameAndBroadcast(g, "arena abort");  // outside the lock: it takes JSettlers' own locks
        forget(g);
    }

    @Override
    public boolean isRecordGameEventsActive() {
        return true;
    }

    @Override
    public boolean isRecordGameEventsFromClientsActive() {
        return true;
    }

    @Override
    protected void startEmptyLog(SOCGame game, boolean isReset) {
        GameSpec spec = creating;
        if (spec == null || !spec.name().equals(game.getName()))
            throw new IllegalStateException("game " + game.getName() + " was not created by the arena");
        try {
            RAND.set(game, new RoutedRandom(spec));
        } catch (IllegalAccessException e) {
            throw new IllegalStateException(e);
        }
    }

    @Override
    public void recordGameEvent(String g, SOCMessage m) {
        record(g, "all", m);
    }

    @Override
    public void recordGameEventTo(String g, int pn, SOCMessage m) {
        record(g, audience(pn), m);
    }

    @Override
    public void recordGameEventNotTo(String g, int pn, SOCMessage m) {
        record(g, "!p" + pn, m);
    }

    @Override
    public void recordGameEventNotTo(String g, int[] pns, SOCMessage m) {
        record(g, "!p" + Arrays.toString(pns), m);
    }

    @Override
    public void recordClientMessage(String g, int pn, SOCMessageForGame m) {
        record(g, pn == PN_OBSERVER ? "fo" : "f" + pn, m);
    }

    private static String audience(int pn) {
        if (pn == PN_OBSERVER) return "ob";
        if (pn == PN_REPLY_TO_UNDETERMINED) return "un";
        return "p" + pn;
    }

    /**
     * Serialized by {@link #recording}: the server's treater thread and {@code SOCForceEndTurnThread} (which holds
     * the game's monitor) can both record events of one game, so the start block, the raw position, its comparison
     * with the last one and the queued lines are one atomic step per event, in the order recorded. The game's own
     * monitor is not taken ({@code takeMonitor} is not reentrant, and most callers already hold it); the game is
     * looked up outside the lock, and nothing under it waits on a JSettlers lock.
     */
    private void record(String g, String aud, Object m) {
        if (!specs.containsKey(g)) return;
        SOCGame ga = getGame(g);
        if (ga == null) return;
        synchronized (recording) {
            if (finished.contains(g)) return;
            int state = ga.getGameState();
            if (!started.contains(g) && (state < SOCGame.START1A || state >= SOCGame.OVER))
                return;  // joins and sit-downs before the game
            if (started.add(g)) {
                Map<String, Object> raw = Snapshots.raw(ga);
                lastRaw.put(g, Json.write(raw));
                List<Object> seats = new ArrayList<>(), smart = new ArrayList<>();
                for (int pn = 0; pn < 4; pn++) {
                    String name = ga.getPlayer(pn).getName();
                    seats.add(name);
                    smart.add(getRobotParameters(name).getStrategyType());
                }
                pipe.send(Json.map("t", "start", "g", g, "first", ga.getFirstPlayer(), "seats", seats, "smart", smart,
                    "board", Snapshots.board(ga), "s", raw));
            }
            Map<String, Object> ev = Json.map("t", "ev", "g", g, "a", aud, "m", String.valueOf(m));
            String raw = Json.write(Snapshots.raw(ga));
            if (!raw.equals(lastRaw.get(g))) {
                ev.put("s", new Json.Raw(raw));
                lastRaw.put(g, raw);
            }
            pipe.send(ev);
            if (state != SOCGame.OVER || !finished.add(g)) return;
            pipe.send(over(ga, false));
        }
        Thread t = new Thread(() -> {
            try {
                Thread.sleep(200);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
            destroyGameAndBroadcast(g, "arena game over");
            forget(g);
        }, "arena-destroy");
        t.setDaemon(true);
        t.start();
    }

    private void forget(String g) {
        specs.remove(g);
        lastRaw.remove(g);
        started.remove(g);
    }

    private static Map<String, Object> over(SOCGame ga, boolean aborted) {
        int[] vp = new int[4];
        for (int pn = 0; pn < 4; pn++) vp[pn] = ga.getPlayer(pn).getTotalVP();
        int winner = aborted || ga.getPlayerWithWin() == null ? -1 : ga.getPlayerWithWin().getPlayerNumber();
        return Json.map("t", "over", "g", ga.getName(), "winner", winner, "vp", vp, "rounds", ga.getRoundCount(),
            "aborted", aborted);
    }
}
```

`jsettlers/src/alphasettler/jsettlers/ArenaMain.java`:
```java
package alphasettler.jsettlers;

import java.io.BufferedReader;
import java.io.FileDescriptor;
import java.io.FileOutputStream;
import java.io.InputStreamReader;
import java.io.PrintStream;
import java.io.PrintWriter;
import java.io.StringWriter;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import soc.baseclient.ServerConnectInfo;
import soc.game.SOCGame;
import soc.game.SOCGameOptionSet;
import soc.robot.SOCRobotBrain;
import soc.robot.SOCRobotClient;
import soc.server.SOCServer;
import soc.util.Version;

/**
 * The harness process: `java ... alphasettler.jsettlers.ArenaMain stock` (or `aligned`, Task 7).
 * Starts the server and its robots, says "ready", then reads commands from stdin until it closes.
 */
public final class ArenaMain {
    private ArenaMain() {}

    public static void main(String[] args) {
        PrintStream out = new PrintStream(new FileOutputStream(FileDescriptor.out), true, StandardCharsets.UTF_8);
        System.setOut(System.err);  // JSettlers prints progress to stdout; the pipe owns the real one
        Pipe pipe = new Pipe(out);
        Thread.setDefaultUncaughtExceptionHandler((t, e) -> pipe.fatal("uncaught in " + t.getName() + ": " + stack(e)));
        try {
            run(args.length > 0 ? args[0] : "stock", pipe);
        } catch (Throwable e) {
            pipe.fatal(stack(e));
        }
    }

    private static void run(String mode, Pipe pipe) throws Exception {
        if (!mode.equals("stock")) throw new IllegalArgumentException("unknown mode " + mode + " (stock)");
        SOCRobotBrain.BOTS_ONLY_FAST_PAUSE_FACTOR = 0.01f;
        SOCServer.ROBOT_FORCE_ENDTURN_SECONDS = 60;
        SOCGame.boardFactory = new ArenaBoardFactory();
        ArenaServer srv = new ArenaServer(pipe);
        srv.setPriority(5);
        srv.start();
        ServerConnectInfo sci = new ServerConnectInfo(ArenaServer.PORT, ArenaServer.COOKIE);
        List<String> names = new ArrayList<>();
        for (int i = 1; i <= 4; i++) {
            String name = "robot " + i;
            names.add(name);
            start(new SOCRobotClient(sci, name, "pw"), srv);
        }
        long deadline = System.currentTimeMillis() + 60_000;
        while (!srv.hasRobots(names)) {
            if (System.currentTimeMillis() > deadline) throw new IllegalStateException("robots did not connect: " + names);
            Thread.sleep(50);
        }
        pipe.send(Json.map("t", "ready", "version", Version.version(), "mode", mode, "robots", names));
        BufferedReader in = new BufferedReader(new InputStreamReader(System.in, StandardCharsets.UTF_8));
        String line;
        while ((line = in.readLine()) != null) {
            if (line.isBlank()) continue;
            Map<String, Object> m = Json.object(line);
            switch (Json.str(m, "t")) {
                case "game" -> srv.create(GameSpec.parse(m));
                case "reply" -> pipe.reply(m);
                case "abort" -> srv.abort(Json.str(m, "g"));
                case "quit" -> pipe.flushAndExit(0);
                default -> throw new IllegalArgumentException("unknown command: " + line);
            }
        }
        pipe.flushAndExit(0);  // stdin closed: the driver is gone, and so is this JVM
    }

    /** Starts a robot client on its own thread, as SOCLocalRobotClient (package-private) does. */
    static void start(SOCRobotClient client, SOCServer srv) {
        client.knownOpts = new SOCGameOptionSet(srv.knownOpts, true);
        client.printedInitialWelcome = true;
        Thread t = new Thread(client::init, "robot-" + client.getNickname());
        t.setDaemon(true);
        t.start();
    }

    static String stack(Throwable e) {
        StringWriter w = new StringWriter();
        e.printStackTrace(new PrintWriter(w));
        return w.toString();
    }
}
```

- [ ] **Step 4: `jvm.py` and `boards.py`**

Create `oracle/jsettlers/jvm.py`:
```python
"""One harness JVM (jsettlers/ArenaMain) driven over JSON lines on its stdin and stdout. Its stderr,
with JSettlers' own output, goes to a log under runs/jsettlers/logs/ that every error names. The JVM
exits when its stdin closes, so it never outlives the process that started it."""

from __future__ import annotations

import itertools
import json
import os
import queue
import subprocess
import threading
import time
from pathlib import Path

from oracle.jsettlers import LOGS, java_command, missing

READY_TIMEOUT = 120.0
_COUNT = itertools.count()


class JvmError(RuntimeError):
    """The harness reported a failure (and exited)."""


class Jvm:
    def __init__(self, mode: str = "stock", log: Path | None = None):
        why = missing()
        if why is not None:
            raise RuntimeError(f"the JSettlers harness is not ready: {why}")
        LOGS.mkdir(parents=True, exist_ok=True)
        self.log = Path(log) if log is not None else LOGS / f"{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}-{next(_COUNT)}.log"
        self._err = open(self.log, "w")
        self.proc = subprocess.Popen(java_command("ArenaMain", mode), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=self._err, text=True, encoding="utf-8", bufsize=1)
        self._lines: queue.Queue[str | None] = queue.Queue()
        threading.Thread(target=self._read, daemon=True).start()
        try:
            self.ready = self.recv(READY_TIMEOUT)
        except BaseException:
            self.close()
            raise
        if self.ready.get("t") != "ready":
            self.close()
            raise JvmError(f"the JSettlers JVM did not start: {self.ready}; see {self.log}")

    def _read(self) -> None:
        for line in self.proc.stdout:
            self._lines.put(line)
        self._lines.put(None)

    def send(self, msg: dict) -> None:
        try:
            self.proc.stdin.write(json.dumps(msg) + "\n")
            self.proc.stdin.flush()
        except (BrokenPipeError, ValueError, OSError):
            raise EOFError(f"the JSettlers JVM is gone (exit code {self.proc.poll()}); see {self.log}") from None

    def recv(self, timeout: float) -> dict:
        try:
            line = self._lines.get(timeout=timeout)
        except queue.Empty:
            raise TimeoutError(f"the JSettlers JVM sent nothing for {timeout:.1f} s; see {self.log}") from None
        if line is None:
            self.proc.wait(timeout=30)
            raise EOFError(f"the JSettlers JVM exited (code {self.proc.returncode}); see {self.log}")
        msg = json.loads(line)
        if msg.get("t") == "error":
            first = (msg.get("what") or "").strip().splitlines()[:1]
            raise JvmError(f"the JSettlers JVM failed: {first[0] if first else 'no detail'}; see {self.log}")
        return msg

    def close(self) -> None:
        if self.proc.poll() is None:
            try:
                self.proc.stdin.close()
            except OSError:
                pass
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        self._err.close()

    def __enter__(self) -> "Jvm":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
```

Create `oracle/jsettlers/boards.py`:
```python
"""Boards and per-seed streams for JSettlers games. A seed's board is the native arena's (our
`Game(seed)` board): its tiles and numbers as they are and its nine port kinds, in our port order,
on JSettlers' nine port edges in their order. The dice, deck and misc seeds are fixed by the seed,
so every rotation and arm of a seed sees the same board, dice and deck (common random numbers)."""

from __future__ import annotations

from alphasettler import Game
from oracle.jsettlers.tables import (
    DESERT_HEX, HEX_NUM, JS_EDGE_TO_OUR_EDGE, JS_HEX_TO_OUR_TILE, JS_RES, MISC_PORT, OUR_EDGE_TO_JS_EDGE,
    OUR_RES_TO_JS, OUR_TILE_TO_JS_HEX, PORT_EDGES, PORT_FACING, PORT_HEXNUMS, RESOURCE_NAMES, WATER_HEX,
)

_MASK = (1 << 64) - 1
_SALTS = {"dice": 0xD1CE, "deck": 0xDEC4, "misc": 0x3155C}


def _splitmix(x: int) -> int:
    x = (x + 0x9E3779B97F4A7C15) & _MASK
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & _MASK
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & _MASK
    return x ^ (x >> 31)


def streams(seed: int) -> dict:
    """The seeds (63-bit, a Java long) of the game's dice, deck and misc random streams."""
    return {k: _splitmix((seed & _MASK) ^ salt) >> 1 for k, salt in _SALTS.items()}


def first_player(k: int) -> int:
    """The JSettlers first player that puts the bridge (JSettlers seat 0) at turn position k."""
    return (4 - k) % 4


def native_board(seed: int) -> dict:
    b = Game(seed).snapshot()["board"]
    kinds = [kind for _, kind in b["ports"]]
    # Port pairs are tuples, as `Game.snapshot()` writes them, so boards compare with ==.
    return {"tile_resource": list(b["tile_resource"]), "tile_number": list(b["tile_number"]),
            "ports": [(JS_EDGE_TO_OUR_EDGE[e], kind) for e, kind in zip(PORT_EDGES, kinds)]}


def normalized(board: dict) -> dict:
    return {"tile_resource": list(board["tile_resource"]), "tile_number": list(board["tile_number"]),
            "ports": sorted([int(e), k] for e, k in board["ports"])}


def js_layout(board: dict) -> dict:
    """`board` (ports on JSettlers' port edges) as SOCBoard's v1 hexLayout and numberLayout."""
    hexes, numbers, robber = [WATER_HEX] * 37, [-1] * 37, None
    for t, res in enumerate(board["tile_resource"]):
        h = OUR_TILE_TO_JS_HEX[t]
        if res is None:
            hexes[HEX_NUM[h]], robber = DESERT_HEX, h
        else:
            hexes[HEX_NUM[h]] = OUR_RES_TO_JS[RESOURCE_NAMES.index(res)]
            numbers[HEX_NUM[h]] = board["tile_number"][t]
    for i, (e, kind) in enumerate(board["ports"]):
        if OUR_EDGE_TO_JS_EDGE[e] != PORT_EDGES[i]:
            raise ValueError(f"port {i} is on our edge {e}, not on JSettlers' port edge {PORT_EDGES[i]:#x}")
        f = PORT_FACING[i]
        hexes[PORT_HEXNUMS[i]] = f + 6 if kind == "generic" else (f << 4) | OUR_RES_TO_JS[RESOURCE_NAMES.index(kind)]
    return {"hexes": hexes, "numbers": numbers, "robber": robber}


def our_board(raw: dict) -> dict:
    """A JSettlers board (Snapshots.board) as our board dict."""
    tile_resource, tile_number = [None] * 19, [0] * 19
    for h, kind, number in raw["hexes"]:
        t = JS_HEX_TO_OUR_TILE[h]
        if kind != DESERT_HEX:
            tile_resource[t], tile_number[t] = RESOURCE_NAMES[JS_RES[kind]], number
    ports = [(JS_EDGE_TO_OUR_EDGE[e], "generic" if kind == MISC_PORT else RESOURCE_NAMES[JS_RES[kind]])
             for e, kind in raw["ports"]]
    return {"tile_resource": tile_resource, "tile_number": tile_number, "ports": ports}


def game_command(name: str, seed: int, k: int, *, injected: bool = True, no_trades: bool = False,
                 bridge: bool = True) -> dict:
    return {"t": "game", "g": name, "layout": js_layout(native_board(seed)) if injected else None,
            "first": first_player(k), **streams(seed), "nt": no_trades, "bridge": bridge}
```

- [ ] **Step 5: Build and run the tests to verify they pass**

Run: `pgrep -fl 'arm.sh' || echo idle`, then `bash jsettlers/build.sh && .venv/bin/pytest -q tests/python/test_jsettlers_jvm.py`
Expected: `built 10 sources`, then PASS, 6 tests (under five minutes). If `test_a_robots_game_runs_to_a_winner_on_the_injected_board_with_the_scripted_first_player` times out, read the JVM log the error names: a robot thread's exception there is a harness bug to fix, not a reason to raise the timeout. If the routed first player or dice differ, `RoutedRandom`'s caller names do not match the jar's call sites: print `StackWalker` frames from a debug build to the log and report BLOCKED with them.

- [ ] **Step 6: Format, lint, the full suite**

Run the Global Constraints sequence (no Rust change). Expected: all pass, nothing skipped.

- [ ] **Step 7: Commit**

```bash
git add jsettlers/src/alphasettler/jsettlers/Pipe.java jsettlers/src/alphasettler/jsettlers/GameSpec.java \
  jsettlers/src/alphasettler/jsettlers/RoutedRandom.java jsettlers/src/alphasettler/jsettlers/ArenaBoard.java \
  jsettlers/src/alphasettler/jsettlers/ArenaBoardFactory.java jsettlers/src/alphasettler/jsettlers/Snapshots.java \
  jsettlers/src/alphasettler/jsettlers/ArenaServer.java jsettlers/src/alphasettler/jsettlers/ArenaMain.java \
  oracle/jsettlers/jvm.py oracle/jsettlers/boards.py tests/python/test_jsettlers_jvm.py
git diff --cached | grep -nE 'sk-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{36}|password=[^[:space:]]+'
git -c user.email=sean.may101@gmail.com commit -m "jsettlers: in-JVM arena server with injected boards, routed dice and the event stream"
```

---

### Task 4: Positions and events from the JSettlers stream

**Files:**
- Create: `oracle/jsettlers/messages.py`, `oracle/jsettlers/translate.py`, `oracle/jsettlers/events.py`
- Test: `tests/python/test_jsettlers_translate.py`, `tests/python/test_jsettlers_events.py` (new)

**Interfaces:**
- Consumes: Task 1's `tables` and `actions`; Task 3's `boards.our_board`, `boards.game_command`, `jvm.Jvm` and the RAW and AUDIENCE formats; `alphasettler.Game.from_snapshot`, `alphasettler.Bot` (`observe`, `diagnostics`).
- Produces:
  - `oracle.jsettlers.messages`: `SET_KEYS`, `Entry(aud, name, f, raw)` with `.from_client`, `.sender -> int | None`, `.num(key) -> int` (hex for `coord`), and `parse(aud, text, raw=None) -> Entry`; `ours5(rset) -> list[int]`.
  - `oracle.jsettlers.translate`: `Untranslatable(kind, detail)` (`.kind`), `PHASES`, `dev_counts(dev10) -> list[int]`, `Context(board, first, turn=0, rolled=False, drawn=[0]*5, offers=0)` with `.ours(js)` and `.js(ours)`, `snapshot(raw, ctx, max_offers=0) -> dict`, `import_game(seed, raw, ctx, config) -> Game`.
  - `oracle.jsettlers.events`: `dice_pair(total) -> [d1, d2]`, `Step(kind, player, actions, events, after, untranslatable, info)`, `Tracker(start)` with `.feed(msg) -> list[Step]`, `.flush() -> list[Step]`, `.raw`, `.ctx`, `.board`, `.declined` (per our seat), `.bank_limit_hits`, `.over`, `.snapshot(max_offers=0)`, `.hand(raw, seat) -> list[int]`, `.rates(raw, seat) -> list[int]`. Step kinds: `roll`, `setup`, `build`, `road_building`, `robber`, `steal`, `buy`, `knight`, `year_of_plenty`, `monopoly`, `discard`, `maritime`, `trade`, `end_turn`, `offer`, `reject`, `clear`, `noop`, `game_over`.

- [ ] **Step 1: Write the failing translation tests**

Create `tests/python/test_jsettlers_translate.py`:
```python
import pytest

from alphasettler import Game
from oracle.jsettlers import tables as T
from oracle.jsettlers.boards import native_board
from oracle.jsettlers.translate import Context, Untranslatable, dev_counts, import_game, snapshot

UNLIMITED = {"max_offers_per_turn": 0, "bank_limit": False}


def player(res=(0, 0, 0, 0, 0), dev=None, played=(0, 0, 0, 0), settlements=(), cities=(), roads=(), discard=0,
           played_dev=False):
    return {"res": list(res), "dev": list(dev or [0] * 10), "played": list(played),
            "settlements": [T.OUR_NODE_TO_JS_NODE[n] for n in settlements],
            "cities": [T.OUR_NODE_TO_JS_NODE[n] for n in cities],
            "roads": [T.OUR_EDGE_TO_JS_EDGE[e] for e in roads], "discard": discard, "played_dev": played_dev,
            "vp": len(settlements) + 2 * len(cities)}


def raw(state, current=0, players=None, deck=None, robber=None, board=None, longest=-1, largest=-1, winner=-1):
    board = board or native_board(3)
    desert = board["tile_resource"].index(None)
    return {"state": state, "current": current,
            "robber": T.OUR_TILE_TO_JS_HEX[desert] if robber is None else robber,
            "deck": [9] * 14 + [1, 1, 2, 2, 3, 3, 4, 5, 6, 7, 8] if deck is None else deck,
            "players": players or [player() for _ in range(4)],
            "longest": longest, "largest": largest, "winner": winner}


def test_the_opening_imports_with_our_player_0_on_jsettlers_first_seat():
    ctx = Context(native_board(3), first=2)
    s = snapshot(raw(T.START1A, current=2), ctx)
    assert (s["phase"], s["current"], s["setup_step"], s["turn"]) == ("setup_settlement", 0, 0, 0)
    assert s["bank"] == [19] * 5 and len(s["dev_deck"]) == 25
    g = import_game(3, raw(T.START1A, current=2), ctx, UNLIMITED)
    assert g.phase == "setup_settlement" and g.current_actor == 0
    assert (ctx.ours(2), ctx.js(0), ctx.ours(1), ctx.js(3)) == (0, 2, 3, 1)


def test_a_setup_road_position_finds_the_settlement_awaiting_its_road():
    ctx = Context(native_board(3), first=0)
    players = [player(settlements=[10]), player(), player(), player()]
    s = snapshot(raw(T.START1B, current=0, players=players), ctx)
    assert (s["phase"], s["setup_road_node"], s["setup_step"]) == ("setup_road", 10, 0)
    import_game(3, raw(T.START1B, current=0, players=players), ctx, UNLIMITED)


def test_hands_seats_cards_and_awards_map_to_ours():
    ctx = Context(native_board(3), first=1, turn=7, rolled=True)
    ctx.drawn = [0, 1, 0, 0, 0]
    dev = [0] * 10
    dev[9], dev[4], dev[1] = 2, 1, 1  # two knights, a VP card, a road building card
    players = [player() for _ in range(4)]
    players[1] = player(res=(1, 2, 3, 4, 5), dev=dev, played=(1, 0, 0, 0), played_dev=True)
    s = snapshot(raw(T.PLAY1, current=1, players=players, longest=1, largest=3), ctx)
    me = s["players"][0]
    assert me["hand"] == [5, 1, 3, 4, 2]
    assert me["dev_hand"] == [2, 1, 1, 0, 0] and dev_counts(dev) == [2, 1, 1, 0, 0]
    assert me["dev_new"] == [0, 1, 0, 0, 0] and s["players"][1]["dev_new"] == [0] * 5
    assert me["dev_played"] == [1, 0, 0, 0, 0] and me["knights_played"] == 1
    assert (s["current"], s["phase"], s["turn"], s["return_phase"]) == (0, "main", 7, "main")
    assert (s["longest_road_owner"], s["largest_army_owner"], s["dev_played_this_turn"]) == (0, 2, True)
    assert s["bank"] == [14, 18, 16, 15, 17]


def test_more_than_19_cards_import_only_without_the_bank_limit():
    ctx = Context(native_board(3), first=0, rolled=True)
    players = [player(res=(0, 0, 0, 0, 12)), player(res=(0, 0, 0, 0, 9)), player(), player()]
    s = snapshot(raw(T.PLAY1, players=players), ctx)
    assert s["bank"][0] == 0 and s["players"][0]["hand"][0] == 12
    import_game(3, raw(T.PLAY1, players=players), ctx, UNLIMITED)
    with pytest.raises(ValueError, match="hands and bank hold"):
        import_game(3, raw(T.PLAY1, players=players), ctx, {"max_offers_per_turn": 0})


@pytest.mark.parametrize("state,phase,extra", [
    (T.ROLL_OR_CARD, "pre_roll", {}), (T.PLACING_ROBBER, "move_robber", {}),
    (T.PLACING_FREE_ROAD1, "road_building", {"roads_left": 2}), (T.PLACING_FREE_ROAD2, "road_building", {"roads_left": 1}),
    (T.OVER, "game_over", {"winner": 3}),
])
def test_states_map_to_our_phases(state, phase, extra):
    ctx = Context(native_board(3), first=1)
    s = snapshot(raw(state, current=1, winner=0), ctx)
    assert s["phase"] == phase and all(s[k] == v for k, v in extra.items())


@pytest.mark.parametrize("state", [T.PLACING_ROAD, T.PLACING_SETTLEMENT, T.PLACING_CITY, T.WAITING_FOR_DISCOVERY,
                                   T.WAITING_FOR_MONOPOLY, 0, 1])
def test_intermediate_states_are_untranslatable(state):
    with pytest.raises(Untranslatable) as e:
        snapshot(raw(state), Context(native_board(3), first=0))
    assert e.value.kind == "intermediate-state"


def test_discards_owed_appear_only_in_the_discard_phase():
    ctx = Context(native_board(3), first=0, rolled=True)
    players = [player(res=(2, 2, 2, 2, 0), discard=4)] + [player() for _ in range(3)]
    s = snapshot(raw(T.WAITING_FOR_DISCARDS, players=players), ctx)
    assert s["phase"] == "discard" and s["players"][0]["discard_remaining"] == 4
    g = import_game(3, raw(T.WAITING_FOR_DISCARDS, players=players), ctx, UNLIMITED)
    assert g.phase == "discard" and g.current_actor == 0


def test_an_imported_position_compares_equal_to_its_own_translation():
    ctx = Context(native_board(3), first=1)
    theirs = snapshot(raw(T.START1A, current=1), ctx)
    assert import_game(3, raw(T.START1A, current=1), ctx, UNLIMITED).snapshot() == theirs
```

Run: `.venv/bin/pytest -q tests/python/test_jsettlers_translate.py`
Expected: FAIL at collection: `No module named 'oracle.jsettlers.translate'`.

- [ ] **Step 2: Write `messages.py` and `translate.py`**

Create `oracle/jsettlers/messages.py`:
```python
"""One recorded message of the harness's stream: its GameEventLog audience and its toString(),
parsed into fields. Resource sets (`clay=..|ore=..|sheep=..|wheat=..|wood=..|unknown=..`) become
dicts; every other value stays a string."""

from __future__ import annotations

from dataclasses import dataclass

from oracle.jsettlers.tables import js5_to_ours

SET_KEYS = ("clay", "ore", "sheep", "wheat", "wood", "unknown")


@dataclass(frozen=True)
class Entry:
    aud: str  # all, pN, !pN, !p[a, b], fN (a client's request), fo, ob, un
    name: str  # the message class, e.g. "SOCPutPiece"
    f: dict
    raw: dict | None = None  # the server-side position after it, when it changed

    @property
    def from_client(self) -> bool:
        return self.aud.startswith("f")

    @property
    def sender(self) -> int | None:
        return int(self.aud[1:]) if self.aud[:1] == "f" and self.aud[1:].isdigit() else None

    def num(self, key: str) -> int:
        v = self.f[key]
        return int(v, 16) if key == "coord" else int(v)


def parse(aud: str, text: str, raw: dict | None = None) -> Entry:
    name, _, body = text.partition(":")
    tokens = body.split("|") if body else []
    fields: dict = {}
    i = 0
    while i < len(tokens):
        key, _, value = tokens[i].partition("=")
        if value.startswith("clay="):
            parts = [value, *tokens[i + 1:i + 6]]
            rset = {}
            for part in parts:
                k, _, v = part.partition("=")
                rset[k] = int(v)
            if tuple(rset) != SET_KEYS:
                raise ValueError(f"a resource set without the six JSettlers keys in {text!r}")
            fields[key] = rset
            i += 6
        else:
            fields[key] = value
            i += 1
    return Entry(aud, name, fields, raw)


def ours5(rset: dict) -> list[int]:
    """A JSettlers resource set's known cards in our resource order."""
    return js5_to_ours([rset[k] for k in SET_KEYS[:5]])
```

Create `oracle/jsettlers/translate.py`:
```python
"""JSettlers positions (the harness's RAW snapshots) in our `Game.from_snapshot` schema. The stream
adds what a position alone does not say (`Context`): our turn count, whether the current player has
rolled, the cards it bought this turn and the offers it made. Our player i is JSettlers seat
(first + i) % 4, so our player 0 moves first. The bank is derived (19 less the cards in hands, never
below 0): JSettlers' bank is unlimited, and its positions import with `bank_limit` off."""

from __future__ import annotations

from dataclasses import dataclass, field

from alphasettler import Game
from oracle.jsettlers.tables import (
    EDGE_NODES, JS_DEV_TO_OURS, JS_EDGE_TO_OUR_EDGE, JS_HEX_TO_OUR_TILE, JS_NODE_TO_OUR_NODE, OVER,
    PLACING_FREE_ROAD1, PLACING_FREE_ROAD2, PLACING_ROBBER, PLAY1, ROLL_OR_CARD, SETUP_STATES, START1A, START1B,
    START2A, START2B, WAITING_FOR_DISCARDS, WAITING_FOR_ROB_CHOOSE_PLAYER, js5_to_ours,
)

PHASES = {START1A: "setup_settlement", START2A: "setup_settlement", START1B: "setup_road", START2B: "setup_road",
          ROLL_OR_CARD: "pre_roll", PLAY1: "main", PLACING_ROBBER: "move_robber",
          WAITING_FOR_DISCARDS: "discard", WAITING_FOR_ROB_CHOOSE_PLAYER: "steal",
          PLACING_FREE_ROAD1: "road_building", PLACING_FREE_ROAD2: "road_building", OVER: "game_over"}
_NODE_EDGES = [[e for e, (a, b) in enumerate(EDGE_NODES) if n in (a, b)] for n in range(54)]


class Untranslatable(Exception):
    def __init__(self, kind: str, detail: str):
        super().__init__(f"{kind}: {detail}")
        self.kind = kind


@dataclass
class Context:
    board: dict  # our board dict
    first: int  # JSettlers' first player
    turn: int = 0  # turns ended since setup (our `turn`)
    rolled: bool = False  # the current player has rolled this turn
    drawn: list = field(default_factory=lambda: [0] * 5)  # cards it bought this turn, by our card index
    offers: int = 0  # domestic offers it made this turn

    def ours(self, js: int) -> int:
        return (js - self.first) % 4

    def js(self, ours: int) -> int:
        return (self.first + ours) % 4


def dev_counts(dev10) -> list[int]:
    """Cards held by JSettlers code (0..9) in our order: knight, victory point, road building,
    year of plenty, monopoly."""
    return [dev10[9], sum(dev10[4:9]), dev10[1], dev10[2], dev10[3]]


def _setup_road_node(p: dict) -> int:
    roads = set(p["roads"])
    for n in p["settlements"]:
        if not any(e in roads for e in _NODE_EDGES[n]):
            return n
    raise Untranslatable("setup", "no settlement of the current player awaits its road")


def snapshot(raw: dict, ctx: Context, max_offers: int = 0) -> dict:
    state = raw["state"]
    phase = PHASES.get(state)
    if phase is None:
        raise Untranslatable("intermediate-state", f"JSettlers state {state} has no phase of ours")
    cur = ctx.ours(raw["current"]) if raw["current"] >= 0 else 0
    players = []
    for i in range(4):
        p = raw["players"][ctx.js(i)]
        knights, rb, disc, mono = p["played"]
        dev = dev_counts(p["dev"])
        players.append({
            "hand": js5_to_ours(p["res"]),
            "dev_hand": dev,
            "dev_new": [min(a, b) for a, b in zip(ctx.drawn, dev)] if i == cur else [0] * 5,
            "dev_played": [knights, 0, rb, disc, mono],
            "knights_played": knights,
            "settlements": sorted(JS_NODE_TO_OUR_NODE[n] for n in p["settlements"]),
            "cities": sorted(JS_NODE_TO_OUR_NODE[n] for n in p["cities"]),
            "roads": sorted(JS_EDGE_TO_OUR_EDGE[e] for e in p["roads"]),
            "discard_remaining": p["discard"] if phase == "discard" else 0,
        })
    held = [sum(p["hand"][r] for p in players) for r in range(5)]

    def seat(js: int) -> int | None:
        return None if js < 0 else ctx.ours(js)

    setup = state in SETUP_STATES
    return {
        "board": ctx.board,
        "robber": JS_HEX_TO_OUR_TILE[raw["robber"]],
        "bank": [max(0, 19 - h) for h in held],
        "dev_deck": [JS_DEV_TO_OURS[c] for c in raw["deck"]],
        "players": players,
        "current": cur,
        "phase": phase,
        "setup_road_node": _setup_road_node(players[cur]) if phase == "setup_road" else None,
        "roads_left": (2 if state == PLACING_FREE_ROAD1 else 1) if phase == "road_building" else None,
        "winner": seat(raw["winner"]) if phase == "game_over" else None,
        "return_phase": "main" if ctx.rolled else "pre_roll",
        "turn": ctx.turn,
        "setup_step": sum(len(p["roads"]) for p in players) if setup else 8,
        "dev_played_this_turn": bool(raw["players"][raw["current"]]["played_dev"]) if raw["current"] >= 0 else False,
        "offers_this_turn": min(ctx.offers, max_offers),
        "longest_road_owner": seat(raw["longest"]),
        "largest_army_owner": seat(raw["largest"]),
    }


def import_game(seed: int, raw: dict, ctx: Context, config: dict) -> Game:
    """Our engine at this JSettlers position (ValueError if our rules cannot hold it)."""
    return Game.from_snapshot(seed, snapshot(raw, ctx, config.get("max_offers_per_turn", 0)), config)
```

- [ ] **Step 3: Run the translation tests to verify they pass**

Run: `.venv/bin/pytest -q tests/python/test_jsettlers_translate.py`
Expected: PASS, 18 tests.

- [ ] **Step 4: Write the failing event tests**

The scripted sequences below are the classic-board sequences of JSettlers' `doc/Message-Sequences-for-Game-Actions.md` (2.6.10), with the positions the harness would attach. `Script` keeps a RAW and lets a test change it as the server would.

Create `tests/python/test_jsettlers_events.py`:
```python
import copy

import pytest

from alphasettler import Bot
from jsettlers_harness import require
from oracle.jsettlers import tables as T
from oracle.jsettlers.actions import (
    BUY_DEV, END_TURN, KNIGHT, ROLL, city, discard, maritime, monopoly, move_robber, redact, road, settlement, steal_from,
    year_of_plenty,
)
from oracle.jsettlers.boards import game_command, js_layout, native_board
from oracle.jsettlers.events import Tracker, dice_pair
from oracle.jsettlers.jvm import Jvm
from oracle.jsettlers.messages import parse
from oracle.jsettlers.translate import import_game

UNLIMITED = {"max_offers_per_turn": 0, "bank_limit": False}
BOARD = native_board(3)
WOOD, BRICK, SHEEP, WHEAT, ORE = range(5)


def rset(ours=(0, 0, 0, 0, 0), unknown=0):
    js = T.ours_to_js5(list(ours))
    return "clay={}|ore={}|sheep={}|wheat={}|wood={}|unknown={}".format(*js, unknown)


class Script:
    """A Tracker fed by hand: `raw` is the server-side position; `send` records one message."""

    def __init__(self, first=0, state=T.PLAY1, current=0):
        players = [{"res": [0] * 5, "dev": [0] * 10, "played": [0] * 4, "settlements": [], "cities": [], "roads": [],
                    "discard": 0, "played_dev": False, "vp": 0} for _ in range(4)]
        desert = T.OUR_TILE_TO_JS_HEX[BOARD["tile_resource"].index(None)]
        self.raw = {"state": state, "current": current, "robber": desert, "deck": [9] * 14 + [1, 1, 2, 2, 3, 3, 4, 5, 6, 7, 8],
                    "players": players, "longest": -1, "largest": -1, "winner": -1}
        board_raw = {"hexes": [], "ports": [], "robber": desert}
        lay = js_layout(BOARD)
        for t, h in enumerate(T.OUR_TILE_TO_JS_HEX):
            board_raw["hexes"].append([h, lay["hexes"][T.HEX_NUM[h]], max(0, lay["numbers"][T.HEX_NUM[h]])])
        for i, e in enumerate(T.PORT_EDGES):
            kind = BOARD["ports"][i][1]
            board_raw["ports"].append([e, 0 if kind == "generic" else T.OUR_RES_TO_JS[T.RESOURCE_NAMES.index(kind)]])
        self.tracker = Tracker({"t": "start", "g": "x", "first": first, "board": board_raw, "s": copy.deepcopy(self.raw)})
        self.steps = []

    def js(self, ours):
        return self.tracker.ctx.js(ours)

    def hand(self, ours, h):
        self.raw["players"][self.js(ours)]["res"] = T.ours_to_js5(list(h))

    def send(self, aud, text, changed=False):
        msg = {"t": "ev", "g": "x", "a": aud, "m": text}
        if changed:
            msg["s"] = copy.deepcopy(self.raw)
        self.steps += self.tracker.feed(msg)

    def flush(self):
        """Every step so far (the stretches closed by later requests and the one open now)."""
        self.steps += self.tracker.flush()
        return self.steps


def test_parse_reads_fields_and_resource_sets():
    e = parse("all", "SOCBankTrade:game=g|give=clay=0|ore=4|sheep=0|wheat=0|wood=0|unknown=0|get=clay=1|ore=0|sheep=0|wheat=0|wood=0|unknown=0|pn=2")
    assert e.name == "SOCBankTrade" and e.f["pn"] == "2" and e.f["give"]["ore"] == 4 and e.f["get"]["clay"] == 1
    assert parse("f3", "SOCPutPiece:game=g|playerNumber=3|pieceType=0|coord=67").num("coord") == 0x67
    assert parse("f3", "SOCEndTurn:game=g").sender == 3


@pytest.mark.parametrize("total,pair", [(2, [1, 1]), (7, [6, 1]), (12, [6, 6]), (5, [4, 1])])
def test_dice_pairs_have_the_total(total, pair):
    assert dice_pair(total) == pair and sum(pair) == total


def test_a_roll_produces_from_the_hand_differences():
    s = Script(first=1, current=1)
    s.send("f1", "SOCRollDice:game=x")
    s.send("all", "SOCDiceResult:game=x|param=9")
    s.hand(0, [0, 2, 0, 0, 0])
    s.hand(2, [1, 0, 0, 0, 1])
    s.raw["state"] = T.PLAY1
    s.send("all", "SOCGameState:game=x|state=20", changed=True)
    [step] = s.flush()
    assert step.kind == "roll" and step.player == 0 and step.actions == [(ROLL, {"roll": [6, 3]})]
    assert step.events == [{"type": "rolled", "player": 0, "dice": [6, 3]},
                           {"type": "produced", "player": 0, "resources": [0, 2, 0, 0, 0]},
                           {"type": "produced", "player": 2, "resources": [1, 0, 0, 0, 1]}]
    assert s.tracker.ctx.rolled


def test_a_build_request_and_its_placement_are_one_step():
    s = Script()
    s.hand(0, [1, 1, 0, 0, 0])
    s.raw["players"][0]["roads"] = [T.OUR_EDGE_TO_JS_EDGE[3]]
    s.send("all", "SOCGameState:game=x|state=20", changed=True)
    s.send("f0", "SOCBuildRequest:game=x|pieceType=0")
    s.hand(0, [0, 0, 0, 0, 0])
    s.raw["state"] = T.PLACING_ROAD
    s.send("all", "SOCPlayerElements:game=x|playerNum=0|actionType=LOSE|e1=1,e5=1", changed=True)
    s.send("all", "SOCGameState:game=x|state=30")
    js_edge = T.OUR_EDGE_TO_JS_EDGE[4]
    s.send("f0", f"SOCPutPiece:game=x|playerNumber=0|pieceType=0|coord={js_edge:x}")
    s.raw["players"][0]["roads"].append(js_edge)
    s.raw["state"] = T.PLAY1
    s.send("all", f"SOCPutPiece:game=x|playerNumber=0|pieceType=0|coord={js_edge:x}", changed=True)
    s.send("all", "SOCGameState:game=x|state=20")
    [step] = [x for x in s.flush() if x.actions]
    assert step.kind == "build" and step.actions == [(road(4), None)]
    assert step.events == [{"type": "built_road", "player": 0, "edge": 4}]


def test_a_trade_answer_inside_a_build_carries_no_position():
    s = Script()
    s.hand(0, [0, 0, 0, 2, 3])
    s.raw["players"][0]["settlements"] = [T.OUR_NODE_TO_JS_NODE[20]]
    s.send("all", "SOCGameState:game=x|state=20", changed=True)
    s.send("f0", "SOCBuildRequest:game=x|pieceType=2")
    s.hand(0, [0, 0, 0, 0, 0])
    s.raw["state"] = T.PLACING_CITY
    s.send("all", "SOCPlayerElements:game=x|playerNum=0|actionType=LOSE|e2=3,e4=2", changed=True)
    s.send("all", "SOCGameState:game=x|state=32")
    s.send("f2", "SOCRejectOffer:game=x|playerNumber=0")  # a late answer to an earlier offer
    s.send("all", "SOCRejectOffer:game=x|playerNumber=2")
    node = T.OUR_NODE_TO_JS_NODE[20]
    s.send("f0", f"SOCPutPiece:game=x|playerNumber=0|pieceType=2|coord={node:x}")
    s.raw["players"][0]["settlements"], s.raw["players"][0]["cities"] = [], [node]
    s.raw["state"] = T.PLAY1
    s.send("all", f"SOCPutPiece:game=x|playerNumber=0|pieceType=2|coord={node:x}", changed=True)
    s.send("all", "SOCGameState:game=x|state=20")
    reject, build = [x for x in s.flush() if x.kind in ("reject", "build")]
    assert reject.kind == "reject" and reject.after is None
    assert reject.events == [{"type": "trade_responded", "player": 2, "accepted": False}]
    assert build.kind == "build" and build.actions == [(city(20), None)] and build.after["state"] == T.PLAY1


def test_a_second_setup_settlement_produces():
    s = Script(state=T.START2A, current=0)
    node = 20
    s.send("f0", f"SOCPutPiece:game=x|playerNumber=0|pieceType=1|coord={T.OUR_NODE_TO_JS_NODE[node]:x}")
    s.raw["players"][0]["settlements"].append(T.OUR_NODE_TO_JS_NODE[node])
    s.hand(0, [1, 0, 1, 0, 0])
    s.raw["state"] = T.START2B
    s.send("all", f"SOCPutPiece:game=x|playerNumber=0|pieceType=1|coord={T.OUR_NODE_TO_JS_NODE[node]:x}", changed=True)
    s.send("all", "SOCGameState:game=x|state=11")
    [step] = s.flush()
    assert step.kind == "setup" and step.actions == [(settlement(node), None)]
    assert step.events[1] == {"type": "produced", "player": 0, "resources": [1, 0, 1, 0, 0]}


def test_a_robber_move_with_one_victim_is_the_move_and_the_true_steal():
    s = Script()
    s.hand(2, [0, 0, 0, 0, 1])
    s.raw["state"] = T.PLACING_ROBBER
    s.send("all", "SOCGameState:game=x|state=33", changed=True)
    hexes = T.OUR_TILE_TO_JS_HEX
    s.send("f0", f"SOCMoveRobber:game=x|playerNumber=0|coord={hexes[5]:x}")
    s.raw["robber"] = hexes[5]
    s.send("all", f"SOCMoveRobber:game=x|playerNumber=0|coord={hexes[5]:x}", changed=True)
    s.hand(2, [0, 0, 0, 0, 0])
    s.hand(0, [0, 0, 0, 0, 1])
    s.raw["state"] = T.PLAY1
    s.send("p0", "SOCRobberyResult:game=x|perp=0|victim=2|resType=2|amount=1|isGainLose=true", changed=True)
    s.send("p2", "SOCRobberyResult:game=x|perp=0|victim=2|resType=2|amount=1|isGainLose=true")
    s.send("!p[0, 2]", "SOCRobberyResult:game=x|perp=0|victim=2|resType=6|amount=1|isGainLose=true")
    s.send("all", "SOCGameState:game=x|state=20")
    [step] = [x for x in s.flush() if x.actions]
    assert step.actions == [(move_robber(5), None), (steal_from(2), {"steal": "ore"})]
    assert step.events[1] == {"type": "stole", "thief": 0, "victim": 2, "resource": "ore"}


def test_a_dev_card_draw_uses_the_true_card():
    s = Script(first=3, current=3)
    s.send("f3", "SOCBuyDevCardRequest:game=x")
    s.raw["players"][3]["dev"][5] = 1
    s.send("p3", "SOCDevCardAction:game=x|playerNum=3|actionType=DRAW|cardType=5", changed=True)
    s.send("!p3", "SOCDevCardAction:game=x|playerNum=3|actionType=DRAW|cardType=0")
    [step] = s.flush()
    assert step.actions == [(BUY_DEV, {"dev": "victory_point"})] and step.player == 0
    assert s.tracker.ctx.drawn == [0, 1, 0, 0, 0]


def test_year_of_plenty_and_monopoly_join_the_card_and_the_pick():
    s = Script()
    s.send("f0", "SOCPlayDevCardRequest:game=x|devCard=2")
    s.raw["state"] = T.WAITING_FOR_DISCOVERY
    s.send("all", "SOCDevCardAction:game=x|playerNum=0|actionType=PLAY|cardType=2", changed=True)
    s.send("f0", f"SOCPickResources:game=x|resources={rset([0, 0, 0, 1, 1])}")
    s.hand(0, [0, 0, 0, 1, 1])
    s.raw["state"] = T.PLAY1
    s.send("all", f"SOCPickResources:game=x|resources={rset([0, 0, 0, 1, 1])}|pn=0|reason=2", changed=True)
    s.hand(1, [0, 0, 3, 0, 0])
    s.send("f0", "SOCEndTurn:game=x")  # closes the stretch; the end-turn step follows
    steps = s.flush()
    yop = next(x for x in steps if x.kind == "year_of_plenty")
    assert yop.actions == [(year_of_plenty(WHEAT, ORE), None)]
    assert [e["type"] for e in yop.events] == ["played_dev", "year_of_plenty_taken"]
    m = Script()
    m.hand(1, [0, 0, 3, 0, 0])
    m.send("all", "SOCGameState:game=x|state=20", changed=True)
    m.send("f0", "SOCPlayDevCardRequest:game=x|devCard=3")
    m.raw["state"] = T.WAITING_FOR_MONOPOLY
    m.send("all", "SOCDevCardAction:game=x|playerNum=0|actionType=PLAY|cardType=3", changed=True)
    m.send("f0", "SOCPickResourceType:game=x|resType=3")
    m.hand(1, [0, 0, 0, 0, 0])
    m.hand(0, [0, 0, 3, 0, 0])
    m.raw["state"] = T.PLAY1
    m.send("all", "SOCSimpleAction:game=x|pn=0|actType=3|v1=3|v2=3", changed=True)
    [mono] = [x for x in m.flush() if x.actions]
    assert mono.actions == [(monopoly(SHEEP), None)]
    assert mono.events[1] == {"type": "monopoly_taken", "player": 0, "resource": "sheep", "from": [0, 3, 0, 0]}


def test_concurrent_discards_replay_in_seat_order_once_the_last_is_in():
    s = Script(first=0, current=0)
    s.hand(1, [4, 4, 0, 0, 0])
    s.hand(3, [0, 0, 8, 0, 0])
    s.raw["players"][1]["discard"], s.raw["players"][3]["discard"] = 4, 4
    s.raw["state"] = T.WAITING_FOR_DISCARDS
    s.send("all", "SOCGameState:game=x|state=50", changed=True)
    s.send("f3", f"SOCDiscard:game=x|resources={rset([0, 0, 4, 0, 0])}")
    s.hand(3, [0, 0, 4, 0, 0])
    s.raw["players"][3]["discard"] = 0
    s.send("p3", f"SOCDiscard:game=x|playerNum=3|resources={rset([0, 0, 4, 0, 0])}", changed=True)
    s.send("!p3", f"SOCDiscard:game=x|playerNum=3|resources={rset(unknown=4)}")
    s.send("f1", f"SOCDiscard:game=x|resources={rset([2, 2, 0, 0, 0])}")
    s.hand(1, [2, 2, 0, 0, 0])
    s.raw["players"][1]["discard"] = 0
    s.raw["state"] = T.PLACING_ROBBER
    s.send("p1", f"SOCDiscard:game=x|playerNum=1|resources={rset([2, 2, 0, 0, 0])}", changed=True)
    first, last = [x for x in s.flush() if x.kind == "discard"]
    assert first.actions == [] and first.after is None and len(first.events) == 4
    assert last.actions == [(discard(WOOD), None)] * 2 + [(discard(BRICK), None)] * 2 + [(discard(SHEEP), None)] * 4
    assert last.after is not None


@pytest.mark.parametrize("give,get,expected", [
    ([4, 0, 0, 0, 0], [0, 0, 0, 0, 1], [maritime(WOOD, ORE)]),
    ([8, 0, 0, 0, 0], [0, 1, 0, 0, 1], [maritime(WOOD, BRICK), maritime(WOOD, ORE)]),
    ([1, 0, 0, 0, 0], [0, 0, 0, 0, 4], None),  # an undone trade
])
def test_bank_trades_become_our_maritime_trades_or_an_allowlisted_difference(give, get, expected):
    s = Script()
    s.hand(0, give)
    s.send("all", "SOCGameState:game=x|state=20", changed=True)
    s.send("f0", f"SOCBankTrade:game=x|give={rset(give)}|get={rset(get)}")
    s.hand(0, get)
    s.send("all", f"SOCBankTrade:game=x|give={rset(give)}|get={rset(get)}|pn=0", changed=True)
    [step] = [x for x in s.flush() if x.kind == "maritime"]
    if expected is None:
        assert step.untranslatable == "bank-trade" and step.actions == []
    else:
        assert [a for a, _ in step.actions] == expected and step.untranslatable is None


def test_an_accepted_offer_is_restated_responded_and_confirmed():
    s = Script(first=0, current=0)
    s.hand(0, [1, 0, 0, 0, 0])
    s.hand(2, [0, 0, 0, 1, 0])
    s.send("all", "SOCGameState:game=x|state=20", changed=True)
    s.send("f0", f"SOCMakeOffer:game=x|from=0|to=false,true,true,true|give={rset([1, 0, 0, 0, 0])}|get={rset([0, 0, 0, 1, 0])}")
    s.send("all", f"SOCMakeOffer:game=x|from=0|to=false,true,true,true|give={rset([1, 0, 0, 0, 0])}|get={rset([0, 0, 0, 1, 0])}")
    s.send("f2", "SOCAcceptOffer:game=x|accepting=0|offering=0")
    s.hand(0, [0, 0, 0, 1, 0])
    s.hand(2, [1, 0, 0, 0, 0])
    s.send("all", f"SOCAcceptOffer:game=x|accepting=2|offering=0|toAccepting={rset([1, 0, 0, 0, 0])}|toOffering={rset([0, 0, 0, 1, 0])}", changed=True)
    s.send("all", "SOCClearOffer:game=x|playerNumber=-1")
    offer, trade = [x for x in s.flush() if x.kind in ("offer", "trade")]
    assert offer.info == {"from": 0, "to": [1, 2, 3], "give": [1, 0, 0, 0, 0], "get": [0, 0, 0, 1, 0]}
    assert trade.untranslatable == "domestic-trade"
    assert [e["type"] for e in trade.events] == ["trade_offered", "trade_responded", "trade_confirmed"]
    assert trade.events[2] == {"type": "trade_confirmed", "offerer": 0, "partner": 2,
                               "offerer_gave": [1, 0, 0, 0, 0], "partner_gave": [0, 0, 0, 1, 0]}
    assert s.tracker.ctx.offers == 1


def test_ending_a_turn_counts_it_and_ending_inside_road_building_is_allowlisted():
    s = Script(first=0, current=0)
    s.tracker.ctx.rolled = True
    s.send("f0", "SOCEndTurn:game=x")
    s.raw["current"], s.raw["state"] = 1, T.ROLL_OR_CARD
    s.send("all", "SOCClearOffer:game=x|playerNumber=-1")
    s.send("all", "SOCTurn:game=x|playerNumber=1|gameState=15", changed=True)
    [step] = s.flush()
    assert step.actions == [(END_TURN, None)] and step.events == [{"type": "turn_ended", "player": 0}]
    assert (s.tracker.ctx.turn, s.tracker.ctx.rolled) == (1, False)
    r = Script(state=T.PLACING_FREE_ROAD1)
    r.send("f0", "SOCEndTurn:game=x")
    r.raw["current"], r.raw["state"] = 1, T.ROLL_OR_CARD
    r.send("all", "SOCTurn:game=x|playerNumber=1|gameState=15", changed=True)
    [step] = r.flush()
    assert step.untranslatable == "road-building-cancelled"


def test_a_turn_the_server_ends_itself_is_counted_and_allowlisted():
    s = Script(first=0, current=0)
    c = s.tracker.ctx
    c.rolled, c.drawn, c.offers = True, [0, 1, 0, 0, 0], 1
    s.send("f0", "SOCBuildRequest:game=x|pieceType=0")  # a request the player never completes
    s.raw["current"], s.raw["state"] = 1, T.ROLL_OR_CARD
    s.send("all", "SOCTurn:game=x|playerNumber=1|gameState=15", changed=True)  # SOCForceEndTurnThread
    s.send("all", "SOCRollDicePrompt:game=x|playerNumber=1")
    [step] = s.flush()
    assert (step.kind, step.player, step.actions) == ("end_turn", 0, [])
    assert step.untranslatable == "forced-end-turn" and step.events == [{"type": "turn_ended", "player": 0}]
    assert (c.turn, c.rolled, c.drawn, c.offers) == (1, False, [0] * 5, 0)


def forced_end(s, nxt=1):
    """The server ends the current player's stalled turn (SOCForceEndTurnThread) inside the open stretch."""
    s.raw["current"], s.raw["state"] = nxt, T.ROLL_OR_CARD
    s.send("all", f"SOCTurn:game=x|playerNumber={nxt}|gameState=15", changed=True)
    s.send("all", f"SOCRollDicePrompt:game=x|playerNumber={nxt}")


def test_a_roll_then_a_forced_end_of_turn_in_one_stretch_ends_the_turn():
    s = Script(first=0, current=0)
    s.tracker.ctx.drawn = [0, 1, 0, 0, 0]
    s.send("f0", "SOCRollDice:game=x")
    s.send("all", "SOCDiceResult:game=x|param=9")
    s.hand(2, [1, 0, 0, 0, 0])
    s.raw["state"] = T.PLAY1
    s.send("all", "SOCGameState:game=x|state=20", changed=True)
    forced_end(s)
    [step] = s.flush()
    assert step.kind == "roll" and step.untranslatable == "forced-end-turn"
    assert step.events[0]["type"] == "rolled" and step.events[-1] == {"type": "turn_ended", "player": 0}
    c = s.tracker.ctx
    assert (c.turn, c.rolled, c.drawn, c.offers) == (1, False, [0] * 5, 0)
    assert s.tracker.snapshot()["return_phase"] == "pre_roll" and s.tracker.snapshot()["turn"] == 1


def test_a_bank_trade_then_a_forced_end_of_turn_in_one_stretch_ends_the_turn():
    s = Script(first=0, current=0)
    s.tracker.ctx.rolled, s.tracker.ctx.offers = True, 2
    s.hand(0, [4, 0, 0, 0, 0])
    s.send("all", "SOCGameState:game=x|state=20", changed=True)
    s.send("f0", f"SOCBankTrade:game=x|give={rset([4, 0, 0, 0, 0])}|get={rset([0, 0, 0, 0, 1])}")
    s.hand(0, [0, 0, 0, 0, 1])
    s.send("all", f"SOCBankTrade:game=x|give={rset([4, 0, 0, 0, 0])}|get={rset([0, 0, 0, 0, 1])}|pn=0", changed=True)
    forced_end(s)
    [step] = [x for x in s.flush() if x.kind == "maritime"]
    assert step.actions == [(maritime(WOOD, ORE), None)] and step.untranslatable == "forced-end-turn"
    assert step.events[-1] == {"type": "turn_ended", "player": 0} and step.after["current"] == 1
    c = s.tracker.ctx
    assert (c.turn, c.rolled, c.offers) == (1, False, 0)


def test_a_card_left_unpicked_then_a_forced_end_of_turn_ends_the_turn():
    s = Script(first=0, current=0)
    s.tracker.ctx.rolled = True
    s.send("f0", "SOCPlayDevCardRequest:game=x|devCard=2")
    s.raw["state"] = T.WAITING_FOR_DISCOVERY
    s.send("all", "SOCDevCardAction:game=x|playerNum=0|actionType=PLAY|cardType=2", changed=True)
    forced_end(s)
    [step] = s.flush()
    assert (step.kind, step.actions, step.untranslatable) == ("end_turn", [], "forced-end-turn")
    assert step.events == [{"type": "turn_ended", "player": 0}] and s.tracker.ctx.turn == 1


def stalled_setup_road(s, nxt=1, state=T.START2A):
    """Player 0 asks for a road JSettlers refuses, stalls, and the server ends its setup turn: the settlement
    awaiting that road goes, with the cards it produced, and the turn passes to seat `nxt` in `state`."""
    s.send("f0", "SOCPutPiece:game=x|playerNumber=0|pieceType=0|coord=16")
    s.send("p0", "SOCDeclinePlayerRequest:game=x|state=0|reason=4|detail1=0|detail2=22|text=You can't build a road there.")
    s.send("p0", "SOCCancelBuildRequest:game=x|pieceType=0")
    s.raw["players"][0]["settlements"].pop()
    s.hand(0, [0, 0, 0, 0, 0])
    s.send("all", "SOCCancelBuildRequest:game=x|pieceType=1", changed=True)
    s.raw["current"], s.raw["state"] = nxt, state
    s.send("all", "SOCClearOffer:game=x|playerNumber=-1", changed=True)
    s.send("all", f"SOCTurn:game=x|playerNumber={nxt}|gameState={state}")


def test_a_setup_turn_the_server_ends_is_its_own_allowlisted_step_and_a_robot_desync():
    # Stock robots: planInitRoad can return an edge off the board, and the robot whose own setup settlement the
    # server then cancels keeps it in its copy of the game (stall-debug.md, seed 78).
    s = Script(first=0, state=T.START2B, current=0)
    s.raw["players"][0]["settlements"] = [T.OUR_NODE_TO_JS_NODE[n] for n in (10, 20)]
    s.raw["players"][0]["roads"] = [T.OUR_EDGE_TO_JS_EDGE[min(e for e, (a, b) in enumerate(T.EDGE_NODES) if 10 in (a, b))]]
    s.hand(0, [0, 0, 0, 1, 0])
    s.send("all", "SOCGameState:game=x|state=11", changed=True)
    s.tracker.flush()
    stalled_setup_road(s)
    [step] = s.flush()
    assert (step.kind, step.player, step.actions, step.events) == ("forced_setup_end", 0, [], [])
    assert step.untranslatable == "forced-setup-end" and step.after["state"] == T.START2A
    assert s.tracker.setup_cancelled == [0] and s.tracker.ctx.turn == 0


def test_a_setup_turn_passing_without_a_cancelled_settlement_is_no_desync():
    s = Script(first=0, state=T.START1A, current=0)
    s.send("all", "SOCGameState:game=x|state=5", changed=True)
    s.send("all", "SOCTurn:game=x|playerNumber=0|gameState=5")  # the game's first turn: nothing was ended
    assert [x.kind for x in s.flush()] in ([], ["noop"])
    s.raw["current"] = 1
    s.send("all", "SOCTurn:game=x|playerNumber=1|gameState=5", changed=True)  # a stalled first placement
    [step] = s.flush()
    assert step.untranslatable == "forced-setup-end" and s.tracker.setup_cancelled == []


def test_forced_discards_for_two_players_in_one_stretch_are_both_read():
    s = Script(first=0, current=0)
    s.hand(1, [4, 4, 0, 0, 0])
    s.hand(3, [0, 0, 8, 0, 0])
    s.raw["players"][1]["discard"], s.raw["players"][3]["discard"] = 4, 4
    s.raw["state"] = T.WAITING_FOR_DISCARDS
    s.send("all", "SOCGameState:game=x|state=50", changed=True)
    s.send("f0", "SOCMoveRobber:game=x|playerNumber=0|coord=0")  # any request: the stretch below is the server's
    s.hand(1, [2, 2, 0, 0, 0])
    s.hand(3, [0, 0, 4, 0, 0])
    s.raw["players"][1]["discard"], s.raw["players"][3]["discard"] = 0, 0
    s.raw["state"] = T.PLACING_ROBBER
    s.send("p1", f"SOCDiscard:game=x|playerNum=1|resources={rset([2, 2, 0, 0, 0])}", changed=True)
    s.send("!p1", f"SOCDiscard:game=x|playerNum=1|resources={rset(unknown=4)}")
    s.send("p3", f"SOCDiscard:game=x|playerNum=3|resources={rset([0, 0, 4, 0, 0])}")
    s.send("!p3", f"SOCDiscard:game=x|playerNum=3|resources={rset(unknown=4)}")
    [step] = [x for x in s.flush() if x.kind == "discard"]
    assert step.actions == [(discard(WOOD), None)] * 2 + [(discard(BRICK), None)] * 2 + [(discard(SHEEP), None)] * 4
    assert [e["player"] for e in step.events] == [1] * 4 + [3] * 4 and step.after is not None


def test_a_roll_that_takes_cards_produces_nothing_negative():
    s = Script(first=0, current=0)
    s.hand(1, [2, 0, 0, 0, 0])
    s.send("all", "SOCGameState:game=x|state=20", changed=True)
    s.send("f0", "SOCRollDice:game=x")
    s.send("all", "SOCDiceResult:game=x|param=8")
    s.hand(0, [0, 1, 0, 0, 0])
    s.hand(1, [1, 0, 0, 0, 0])  # a hand that shrank on a roll: no production of ours explains it
    s.send("all", "SOCGameState:game=x|state=20", changed=True)
    [step] = [x for x in s.flush() if x.kind == "roll"]
    assert step.untranslatable == "production"
    assert step.events[1:] == [{"type": "produced", "player": 0, "resources": [0, 1, 0, 0, 0]}]


def test_declines_and_bank_limit_payouts_are_counted():
    s = Script()
    s.send("f0", "SOCBuildRequest:game=x|pieceType=2")
    s.send("un", "SOCDeclinePlayerRequest:game=x|reason=1|detail1=0|detail2=0|text=no")
    assert s.flush() == [] and s.tracker.declined == [1, 0, 0, 0]
    s.hand(1, [0, 0, 0, 0, 15])
    s.hand(2, [0, 0, 0, 0, 4])
    s.send("all", "SOCGameState:game=x|state=20", changed=True)
    s.send("f0", "SOCRollDice:game=x")
    s.send("all", "SOCDiceResult:game=x|param=6")
    s.hand(2, [0, 0, 0, 0, 5])
    s.send("all", "SOCGameState:game=x|state=20", changed=True)
    s.flush()
    assert s.tracker.bank_limit_hits == 1


def test_a_winning_placement_ends_with_game_over():
    s = Script()
    edge = T.OUR_EDGE_TO_JS_EDGE[7]
    s.send("f0", f"SOCPutPiece:game=x|playerNumber=0|pieceType=0|coord={edge:x}")
    s.raw["players"][0]["roads"].append(edge)
    s.raw["state"], s.raw["winner"] = T.OVER, 0
    s.send("all", f"SOCPutPiece:game=x|playerNumber=0|pieceType=0|coord={edge:x}", changed=True)
    s.send("all", "SOCGameState:game=x|state=1000")
    [step] = s.flush()
    assert step.events[-1] == {"type": "game_over", "winner": 0}


def test_a_winning_action_cut_short_by_the_end_of_the_stream_is_read_from_its_request():
    # The harness stops at the first message recorded once the game is over, and JSettlers announces a winning
    # placement (or card) in text before it reports the action itself.
    s = Script()
    node = T.OUR_NODE_TO_JS_NODE[20]
    s.raw["players"][0]["settlements"] = [node]
    s.send("all", "SOCGameState:game=x|state=20", changed=True)
    s.send("f0", f"SOCPutPiece:game=x|playerNumber=0|pieceType=2|coord={node:x}")
    s.raw["players"][0]["settlements"], s.raw["players"][0]["cities"] = [], [node]
    s.raw["state"], s.raw["winner"] = T.OVER, 0
    s.send("all", "SOCGameServerText:game=x|text=robot 1 built a city.", changed=True)
    [step] = [x for x in s.flush() if x.kind != "noop"]
    assert step.kind == "build" and step.actions == [(city(20), None)]
    assert step.events == [{"type": "built_city", "player": 0, "node": 20}, {"type": "game_over", "winner": 0}]
    r = Script()
    edge = T.OUR_EDGE_TO_JS_EDGE[7]
    r.raw["players"][0]["roads"] = [edge]
    r.send("all", "SOCGameState:game=x|state=20", changed=True)
    r.send("f0", f"SOCPutPiece:game=x|playerNumber=0|pieceType=0|coord={edge:x}")  # a road it already has
    r.raw["state"], r.raw["winner"] = T.OVER, 0
    r.send("all", "SOCGameServerText:game=x|text=robot 1 won.", changed=True)
    last = r.flush()[-1]
    assert (last.kind, last.actions, last.events) == ("noop", [], [{"type": "game_over", "winner": 0}])
    b = Script(first=1, current=1)
    b.send("f1", "SOCBuyDevCardRequest:game=x")
    b.raw["players"][1]["dev"][5] = 1
    b.raw["deck"] = b.raw["deck"][:15] + b.raw["deck"][16:]  # a victory point card left the deck
    b.raw["state"], b.raw["winner"] = T.OVER, 1
    b.send("all", "SOCPlayerElements:game=x|playerNum=1|actionType=LOSE|e2=1,e3=1,e4=1", changed=True)
    [step] = b.flush()
    assert step.kind == "buy" and step.player == 0 and step.actions == [(BUY_DEV, {"dev": "victory_point"})]
    assert step.events[-1] == {"type": "game_over", "winner": 0}
    e = Script(first=0, current=0)
    e.tracker.ctx.rolled = True
    e.send("f0", "SOCEndTurn:game=x")
    e.raw["current"], e.raw["state"], e.raw["winner"] = 1, T.OVER, 1  # the next player starts its turn with 10
    e.send("all", "SOCClearOffer:game=x|playerNumber=-1", changed=True)
    [step] = e.flush()
    assert step.kind == "end_turn" and step.actions == [(END_TURN, None)]
    assert step.events == [{"type": "turn_ended", "player": 0}, {"type": "game_over", "winner": 1}]


def test_a_knight_that_reaches_the_target_ends_our_game_before_the_robber_moves():
    s = Script()
    s.raw["players"][0]["vp"], s.raw["players"][0]["dev"][9] = 8, 1
    s.raw["players"][0]["settlements"] = [T.OUR_NODE_TO_JS_NODE[n] for n in (0, 10, 20, 30)]
    s.raw["players"][0]["cities"] = [T.OUR_NODE_TO_JS_NODE[n] for n in (40, 50)]
    s.raw["players"][0]["played"] = [2, 0, 0, 0]
    s.raw["deck"] = s.raw["deck"][3:]  # the three knights it held or played
    s.send("all", "SOCGameState:game=x|state=20", changed=True)
    s.send("f0", "SOCPlayDevCardRequest:game=x|devCard=9")
    p = s.raw["players"][0]
    p["dev"][9], p["played"], p["played_dev"], p["vp"] = 0, [3, 0, 0, 0], True, 10
    s.raw["state"], s.raw["largest"] = T.PLACING_ROBBER, 0  # JSettlers awards the win only after the robber moves
    s.send("all", "SOCDevCardAction:game=x|playerNum=0|actionType=PLAY|cardType=9", changed=True)
    s.send("all", "SOCGameState:game=x|state=33")
    hexes = T.OUR_TILE_TO_JS_HEX
    s.send("f0", f"SOCMoveRobber:game=x|playerNumber=0|coord={hexes[5]:x}")
    s.raw["robber"], s.raw["state"], s.raw["winner"] = hexes[5], T.OVER, 0
    s.send("all", f"SOCMoveRobber:game=x|playerNumber=0|coord={hexes[5]:x}", changed=True)
    s.send("all", "SOCGameState:game=x|state=1000")
    steps = [x for x in s.flush() if x.kind != "noop"]
    assert [x.kind for x in steps] == ["knight"] and steps[0].actions == [(KNIGHT, None)]
    assert steps[0].events == [{"type": "played_dev", "player": 0, "card": "knight"},
                               {"type": "game_over", "winner": 0}]
    g = import_game(3, steps[0].after, s.tracker.ctx, UNLIMITED)
    assert g.phase == "game_over" and g.snapshot()["robber"] == BOARD["tile_resource"].index(None)
    assert s.tracker.over


def test_a_robots_game_translates_and_four_observers_keep_their_beliefs(tmp_path):
    require()
    seed = 31
    bots = [Bot("ismcts@20", seed * 4 + i, None) for i in range(4)]
    with Jvm(log=tmp_path / "jvm.log") as jvm:  # suite runs leave no logs under runs/
        jvm.send(game_command("e-full", seed, 2, bridge=False))
        tracker, kinds, imported = None, set(), 0
        while True:
            m = jvm.recv(600.0)
            if m["t"] == "start":
                tracker = Tracker(m)
                for i, b in enumerate(bots):
                    b.observe(i, [])
                continue
            steps = tracker.feed(m) if m["t"] == "ev" else tracker.flush()
            for st in steps:
                kinds.add(st.kind)
                for i, b in enumerate(bots):
                    b.observe(i, [redact(e, i) for e in st.events])
                if st.after is not None:
                    import_game(seed, st.after, tracker.ctx, UNLIMITED)
                    imported += 1
            if m["t"] == "over":
                break
    assert {"setup", "roll", "build", "end_turn"} <= kinds and imported > 100
    assert [b.diagnostics().get("belief_resets", 0) for b in bots] == [0, 0, 0, 0]
    assert tracker.over
```

Run: `.venv/bin/pytest -q tests/python/test_jsettlers_events.py`
Expected: FAIL at collection: `No module named 'oracle.jsettlers.events'`.

- [ ] **Step 5: Write `events.py`**

Create `oracle/jsettlers/events.py`:
```python
"""The harness's stream of one game as steps of ours (spec Section 2, "Events to our bot").

A client's request line starts a stretch; the server's messages up to the next request are what it
did. Each closed stretch becomes at most one `Step`: our actions with their forced chance outcomes
(for the replay), our engine's events (for our bot, before redaction) and the JSettlers position
after it. Two stretches make one action of ours when the first leaves a position we have no phase
for (a build request and its placement, a Year of Plenty or Monopoly card and its pick). Discards
come one player at a time in any order; their actions are released in our seat order once the
last is in. A step our rules cannot replay carries an allowlist kind (`untranslatable`) instead.
Our rules end the game the moment the player on turn reaches the target: when JSettlers plays on (a
knight's Largest Army wins there only after the robber moves) the step carries the position as our
rules leave it, and nothing after our game's end becomes a step. A setup turn the server ended (a stalled
placement) is its own allowlisted step, `forced-setup-end`."""

from __future__ import annotations

from dataclasses import dataclass, field

from oracle.jsettlers import actions as A
from oracle.jsettlers.boards import our_board
from oracle.jsettlers.messages import Entry, ours5, parse
from oracle.jsettlers.tables import (
    CITY, DEV_NAMES, EDGE_NODES, JS_DEV_TO_OURS, JS_EDGE_TO_OUR_EDGE, JS_HEX_TO_OUR_TILE, JS_NODE_TO_OUR_NODE,
    JS_RES, OVER, PLACING_CITY, PLACING_FREE_ROAD1, PLACING_FREE_ROAD2, PLACING_ROAD, PLACING_SETTLEMENT,
    RESOURCE_NAMES, ROAD, SETTLEMENT, SETUP_STATES, WAITING_FOR_DISCARDS, WAITING_FOR_DISCOVERY,
    WAITING_FOR_MONOPOLY, js5_to_ours,
)
from oracle.jsettlers.translate import Context, snapshot

_HALF_DONE = frozenset({PLACING_ROAD, PLACING_SETTLEMENT, PLACING_CITY, WAITING_FOR_DISCOVERY, WAITING_FOR_MONOPOLY})
VP_TO_WIN = 10  # the arena's game option VP, and our vp_to_win


def dice_pair(total: int) -> list[int]:
    """JSettlers reports a roll's total only; our forced roll and event use this pair."""
    a = min(total - 1, 6)
    return [a, total - a]


@dataclass
class Step:
    kind: str
    player: int | None  # our seat that acted
    actions: list = field(default_factory=list)  # [(action id, chance dict or None)], in order
    events: list = field(default_factory=list)  # our engine's events, unredacted
    after: dict | None = None  # the JSettlers position after it; None while the action is incomplete
    untranslatable: str | None = None  # an allowlist kind: the replay re-imports `after`
    info: dict = field(default_factory=dict)


def _played(p: int, card: str) -> dict:
    return {"type": "played_dev", "player": p, "card": card}


def _offered(p: int, give, get) -> dict:
    return {"type": "trade_offered", "player": p, "give": list(give), "get": list(get)}


def _declined(e: Entry) -> bool:
    """The server refused the request: a declined build or card, or a refused offer or bank trade."""
    return (e.name == "SOCDeclinePlayerRequest"
            or (e.name == "SOCDevCardAction" and e.f.get("actionType") == "CANNOT_PLAY")
            or (e.name == "SOCRejectOffer" and "reasonCode" in e.f))


def _won(raw: dict) -> dict | None:
    """The position as our rules leave it when the player on turn has reached the target but JSettlers
    plays on (a knight's Largest Army wins only after the robber moves there): our game is over, won by
    that player. None when JSettlers agrees (or nobody is there)."""
    cur = raw["current"]
    if raw["state"] == OVER or raw["state"] in SETUP_STATES or cur < 0 or raw["players"][cur]["vp"] < VP_TO_WIN:
        return None
    return {**raw, "state": OVER, "winner": cur}


class Tracker:
    def __init__(self, start: dict):
        self.board = our_board(start["board"])
        self.ctx = Context(self.board, start["first"])
        self.raw = start["s"]
        self.declined = [0] * 4
        self.bank_limit_hits = 0
        self.setup_cancelled: list[int] = []  # our seats whose setup settlement a forced end of turn removed
        self.over = False
        self._open: list[Entry] | None = None
        self._before = self.raw
        self._discards: dict[int, list[int]] = {}

    def hand(self, raw: dict, seat: int) -> list[int]:
        return js5_to_ours(raw["players"][self.ctx.js(seat)]["res"])

    def snapshot(self, max_offers: int = 0) -> dict:
        return snapshot(self.raw, self.ctx, max_offers)

    def feed(self, msg: dict) -> list[Step]:
        e = parse(msg["a"], msg["m"], msg.get("s"))
        out: list[Step] = []
        if e.from_client:
            out = self._close()
            self._open, self._before = [e], self.raw
        else:
            if self._open is None:
                self._open, self._before = [], self.raw
            self._open.append(e)
        if e.raw is not None:
            self.raw = e.raw
        return out

    def flush(self) -> list[Step]:
        """Close the open stretch (before a decision, and at the end of the game)."""
        return self._close()

    def _close(self) -> list[Step]:
        entries, before, after = self._open, self._before, self.raw
        self._open, self._before = None, self.raw
        if not entries or self.over:  # after the game is over in ours, nothing more replays
            return []
        req = entries[0] if entries[0].from_client else None
        server = [e for e in entries if not e.from_client]
        if req is not None and req.sender is not None and any(_declined(e) for e in server):
            self.declined[self.ctx.ours(req.sender)] += 1
        for r in range(5):
            b = sum(self.hand(before, q)[r] for q in range(4))
            a = sum(self.hand(after, q)[r] for q in range(4))
            if a > 19 and a > b:
                self.bank_limit_hits += 1
                break
        step = self._action(req, server, before, after)
        if (before["state"] not in SETUP_STATES and any(e.name == "SOCTurn" for e in server)
                and (step is None or step.kind != "end_turn")):
            step = self._forced_end(step, before, after)
        if (step is not None and step.after is not None and step.after["state"] in _HALF_DONE
                and not step.actions and step.untranslatable is None):
            # Events only (an offer or an answer to one) inside another player's build or card pick: the
            # position after it has no phase of ours, and the step that completes the action carries it.
            step.after = None
        steps = [] if step is None else [step]
        won = _won(after)
        if won is not None and steps:
            steps[-1].after = won
        end = won or after
        if end["state"] == OVER:
            self.over = True
            ev = {"type": "game_over", "winner": None if end["winner"] < 0 else self.ctx.ours(end["winner"])}
            if steps:
                steps[-1].events.append(ev)
            else:
                steps.append(Step("game_over", None, events=[ev], after=end))
        return steps

    def _action(self, req, server, before, after) -> Step | None:
        def first(name, pred=lambda e: True):
            return next((e for e in server if e.name == name and pred(e)), None)

        if (dice := first("SOCDiceResult")) is not None:
            return self._roll(int(dice.f["param"]), before, after)
        if (piece := first("SOCPutPiece", lambda e: e.aud == "all")) is not None:
            return self._piece(piece, before, after)
        if (robber := first("SOCMoveRobber", lambda e: e.aud == "all")) is not None:
            return self._robber(robber, server, after)
        if (robbery := self._robbery(server)) is not None:
            thief, victim, r = robbery
            return Step("steal", thief, [(A.steal_from(victim), {"steal": RESOURCE_NAMES[r]})],
                        [{"type": "stole", "thief": thief, "victim": victim, "resource": RESOURCE_NAMES[r]}], after)
        if (draw := first("SOCDevCardAction", lambda e: e.f.get("actionType") == "DRAW"
                          and e.f.get("cardType", "0") != "0")) is not None:
            return self._buy(draw, after)
        if (play := first("SOCDevCardAction", lambda e: e.f.get("actionType") == "PLAY")) is not None:
            return self._play(play, after)
        if (pick := first("SOCPickResources", lambda e: e.f.get("reason") == "2")) is not None:
            return self._plenty(pick, after)
        if (mono := first("SOCSimpleAction", lambda e: e.f.get("actType") == "3")) is not None:
            return self._monopoly(mono, before, after)
        if discs := [e for e in server
                     if e.name == "SOCDiscard" and "playerNum" in e.f and e.f["resources"]["unknown"] == 0]:
            return self._discard(discs, after)
        if (bank := first("SOCBankTrade", lambda e: "pn" in e.f)) is not None:
            return self._bank(bank, before, after)
        if (accept := first("SOCAcceptOffer", lambda e: "toAccepting" in e.f)) is not None:
            return self._trade(accept, after)
        if first("SOCTurn") is not None and before["state"] not in SETUP_STATES:
            return self._end_turn(before, after, requested=req is not None and req.name == "SOCEndTurn")
        cancelled = first("SOCCancelBuildRequest", lambda e: e.aud == "all" and e.f.get("pieceType") == "1")
        if before["state"] in SETUP_STATES and (cancelled is not None or (
                first("SOCTurn") is not None and (after["current"] != before["current"]
                                                  or after["state"] not in SETUP_STATES))):
            return self._forced_setup_end(before, after, cancelled is not None)
        if (made := first("SOCMakeOffer")) is not None:
            return self._offer(made, after)
        if (rejected := first("SOCRejectOffer", lambda e: "reasonCode" not in e.f)) is not None:
            q = self.ctx.ours(int(rejected.f["playerNumber"]))
            return Step("reject", q, [], [{"type": "trade_responded", "player": q, "accepted": False}], after,
                        info={"player": q})
        if (cleared := first("SOCClearOffer", lambda e: int(e.f["playerNumber"]) >= 0)) is not None:
            q = self.ctx.ours(int(cleared.f["playerNumber"]))
            return Step("clear", q, [], [{"type": "trade_cancelled", "player": q}], after)
        if after["state"] == OVER and req is not None and (step := self._cut_short(req, before, after)) is not None:
            return step
        if after["state"] in _HALF_DONE or after == before:
            return None
        return Step("noop", None, after=after)

    def _forced_setup_end(self, before, after, cancelled: bool) -> Step:
        """The server ended a stalled setup turn (SOCForceEndTurnThread): the settlement awaiting its road, if
        any, is removed with the cards it produced, and the setup order moves on (JSettlers may move its first
        player). Our engine has no such end, and no event removes a settlement, so our bot is shown nothing and
        the replay re-imports. A stock robot whose own settlement is cancelled keeps it in its copy of the game
        (`setup_cancelled`, the arena's `robot_desync`)."""
        p = self.ctx.ours(before["current"])
        if cancelled:
            self.setup_cancelled.append(p)
        return Step("forced_setup_end", p, [], [], after, untranslatable="forced-setup-end")

    def _forced_end(self, step: Step | None, before, after) -> Step:
        """The server ended the turn (SOCForceEndTurnThread) in a stretch whose first rule read another
        action (a roll, a bank trade, a forced discard, a card left unpicked): the turn ends after it."""
        if step is None:
            return self._end_turn(before, after, requested=False)
        c = self.ctx
        c.turn, c.rolled, c.drawn, c.offers = c.turn + 1, False, [0] * 5, 0
        step.events.append({"type": "turn_ended", "player": c.ours(before["current"])})
        step.after = after
        if step.untranslatable is None:  # our replay re-imports the position after it
            step.untranslatable = "forced-end-turn"
        return step

    def _cut_short(self, req: Entry, before, after) -> Step | None:
        """The winning action when the stream ends before the server reports it: the harness stops at the
        first message recorded once the game is over, and JSettlers announces a winning placement or card
        in text first. The request says what was asked; the position after it says what was done."""
        if req.sender is None:
            return None
        js = req.sender
        if req.name == "SOCPutPiece" and int(req.f["playerNumber"]) == js:
            kind, coord = int(req.f["pieceType"]), req.num("coord")
            key = {ROAD: "roads", SETTLEMENT: "settlements", CITY: "cities"}.get(kind)
            if key is not None and coord in after["players"][js][key] and coord not in before["players"][js][key]:
                return self._piece(Entry("all", req.name, req.f), before, after)
        if req.name == "SOCBuyDevCardRequest":
            new = [c for c in range(10) if after["players"][js]["dev"][c] > before["players"][js]["dev"][c]]
            if len(new) == 1:
                return self._buy(Entry("all", "SOCDevCardAction", {"playerNum": str(js), "cardType": str(new[0])}),
                                 after)
        if req.name == "SOCEndTurn" and before["current"] == js and after["current"] != js:
            return self._end_turn(before, after)
        return None

    def _gain(self, before, after, q) -> list[int]:
        return [a - b for a, b in zip(self.hand(after, q), self.hand(before, q))]

    def _roll(self, total, before, after) -> Step:
        p = self.ctx.ours(before["current"])
        events = [{"type": "rolled", "player": p, "dice": dice_pair(total)}]
        bad = None
        if total != 7:
            for q in range(4):
                g = self._gain(before, after, q)
                if any(x < 0 for x in g):
                    bad = "production"
                elif any(g):
                    events.append({"type": "produced", "player": q, "resources": g})
        self.ctx.rolled = True
        return Step("roll", p, [(A.ROLL, {"roll": dice_pair(total)})], events, after, untranslatable=bad)

    def _piece(self, e, before, after) -> Step:
        p = self.ctx.ours(int(e.f["playerNumber"]))
        kind, coord = int(e.f["pieceType"]), e.num("coord")
        if kind == ROAD:
            edge = JS_EDGE_TO_OUR_EDGE[coord]
            action, ev = A.road(edge), {"type": "built_road", "player": p, "edge": edge}
        elif kind in (SETTLEMENT, CITY):
            node = JS_NODE_TO_OUR_NODE[coord]
            name = "settlement" if kind == SETTLEMENT else "city"
            action = A.settlement(node) if kind == SETTLEMENT else A.city(node)
            ev = {"type": f"built_{name}", "player": p, "node": node}
        else:
            return Step("noop", p, after=after, untranslatable="piece")
        events = [ev]
        state = before["state"]
        if state in SETUP_STATES:
            if kind == SETTLEMENT and any(g := self._gain(before, after, p)):
                events.append({"type": "produced", "player": p, "resources": g})
            return Step("setup", p, [(action, None)], events, after)
        if state in (PLACING_FREE_ROAD1, PLACING_FREE_ROAD2):
            return Step("road_building", p, [(action, None)], events, after)
        return Step("build", p, [(action, None)], events, after)

    def _robbery(self, server):
        e = next((x for x in server if x.name == "SOCRobberyResult" and x.f.get("resType", "6") != "6"), None)
        if e is None:
            return None
        return self.ctx.ours(int(e.f["perp"])), self.ctx.ours(int(e.f["victim"])), JS_RES[int(e.f["resType"])]

    def _robber(self, e, server, after) -> Step:
        p = self.ctx.ours(int(e.f["playerNumber"]))
        tile = JS_HEX_TO_OUR_TILE[e.num("coord")]
        actions = [(A.move_robber(tile), None)]
        events = [{"type": "robber_moved", "player": p, "tile": tile}]
        if (robbery := self._robbery(server)) is not None:
            thief, victim, r = robbery
            actions.append((A.steal_from(victim), {"steal": RESOURCE_NAMES[r]}))
            events.append({"type": "stole", "thief": thief, "victim": victim, "resource": RESOURCE_NAMES[r]})
        return Step("robber", p, actions, events, after)

    def _buy(self, e, after) -> Step:
        p = self.ctx.ours(int(e.f["playerNum"]))
        card = JS_DEV_TO_OURS[int(e.f["cardType"])]
        self.ctx.drawn[DEV_NAMES.index(card)] += 1
        return Step("buy", p, [(A.BUY_DEV, {"dev": card})], [{"type": "bought_dev", "player": p, "card": card}], after)

    def _play(self, e, after) -> Step | None:
        p = self.ctx.ours(int(e.f["playerNum"]))
        card = int(e.f["cardType"])
        if card == 9:
            return Step("knight", p, [(A.KNIGHT, None)], [_played(p, "knight")], after)
        if card == 1:
            return Step("road_building", p, [(A.ROAD_BUILDING, None)], [_played(p, "road_building")], after)
        if card in (2, 3):
            return None  # completed by the pick that follows
        return Step("noop", p, after=after, untranslatable="dev-card")

    def _plenty(self, e, after) -> Step:
        p = self.ctx.ours(int(e.f["pn"]))
        got = ours5(e.f["resources"])
        cards = [r for r in range(5) for _ in range(got[r])]
        if len(cards) != 2:
            return Step("noop", p, after=after, untranslatable="year-of-plenty")
        return Step("year_of_plenty", p, [(A.year_of_plenty(*cards), None)],
                    [_played(p, "year_of_plenty"), {"type": "year_of_plenty_taken", "player": p, "resources": got}],
                    after)

    def _monopoly(self, e, before, after) -> Step:
        p = self.ctx.ours(int(e.f["pn"]))
        r = JS_RES[int(e.f["v2"])]
        taken = [0 if q == p else self.hand(before, q)[r] - self.hand(after, q)[r] for q in range(4)]
        return Step("monopoly", p, [(A.monopoly(r), None)],
                    [_played(p, "monopoly"),
                     {"type": "monopoly_taken", "player": p, "resource": RESOURCE_NAMES[r], "from": taken}], after)

    def _discard(self, discs: list[Entry], after) -> Step:
        """Every discard of the stretch (the server force-discards for each stalled player at once)."""
        events = []
        for e in discs:
            q = self.ctx.ours(int(e.f["playerNum"]))
            cards = ours5(e.f["resources"])
            self._discards[q] = [a + b for a, b in zip(self._discards.get(q, [0] * 5), cards)]
            events += [{"type": "discarded", "player": q, "resource": RESOURCE_NAMES[r]}
                       for r in range(5) for _ in range(cards[r])]
        q = self.ctx.ours(int(discs[0].f["playerNum"]))
        if after["state"] == WAITING_FOR_DISCARDS:
            return Step("discard", q, [], events, None)
        actions = [(A.discard(r), None) for s in sorted(self._discards) for r in range(5)
                   for _ in range(self._discards[s][r])]
        self._discards = {}
        return Step("discard", q, actions, events, after)

    def rates(self, raw: dict, seat: int) -> list[int]:
        """Seat's maritime rates (our resource order) from its buildings in `raw` and our board's ports."""
        p = raw["players"][self.ctx.js(seat)]
        nodes = {JS_NODE_TO_OUR_NODE[n] for n in p["settlements"] + p["cities"]}
        rates, generic = [4] * 5, False
        for e, kind in self.board["ports"]:
            a, b = EDGE_NODES[e]
            if a in nodes or b in nodes:
                if kind == "generic":
                    generic = True
                else:
                    rates[RESOURCE_NAMES.index(kind)] = 2
        return [min(r, 3) if generic else r for r in rates]

    def _bank(self, e, before, after) -> Step:
        p = self.ctx.ours(int(e.f["pn"]))
        give, get = ours5(e.f["give"]), ours5(e.f["get"])
        rates = self.rates(before, p)
        whole = [{"type": "maritime_traded", "player": p, "gave": give, "got": get}]
        units = []
        for g in range(5):
            if give[g] % rates[g]:
                return Step("maritime", p, [], whole, after, untranslatable="bank-trade")
            units += [g] * (give[g] // rates[g])
        gets = [r for r in range(5) for _ in range(get[r])]
        if len(units) != len(gets) or any(a == b for a, b in zip(units, gets)):
            return Step("maritime", p, [], whole, after, untranslatable="bank-trade")
        actions, events = [], []
        for g, r in zip(units, gets):
            gave, got = [0] * 5, [0] * 5
            gave[g], got[r] = rates[g], 1
            actions.append((A.maritime(g, r), None))
            events.append({"type": "maritime_traded", "player": p, "gave": gave, "got": got})
        return Step("maritime", p, actions, events, after)

    def _trade(self, e, after) -> Step:
        a, o = self.ctx.ours(int(e.f["accepting"])), self.ctx.ours(int(e.f["offering"]))
        give, get = ours5(e.f["toAccepting"]), ours5(e.f["toOffering"])
        events = [_offered(o, give, get), {"type": "trade_responded", "player": a, "accepted": True},
                  {"type": "trade_confirmed", "offerer": o, "partner": a, "offerer_gave": give, "partner_gave": get}]
        return Step("trade", o, [], events, after, untranslatable="domestic-trade",
                    info={"offerer": o, "partner": a, "give": give, "get": get})

    def _end_turn(self, before, after, requested: bool = True) -> Step:
        p = self.ctx.ours(before["current"])
        cancelled = before["state"] in (PLACING_FREE_ROAD1, PLACING_FREE_ROAD2)
        c = self.ctx
        c.turn, c.rolled, c.drawn, c.offers = c.turn + 1, False, [0] * 5, 0
        if not requested:  # the server ended a stalled turn (SOCForceEndTurnThread): no action of ours replays it
            return Step("end_turn", p, [], [{"type": "turn_ended", "player": p}], after, untranslatable="forced-end-turn")
        return Step("end_turn", p, [] if cancelled else [(A.END_TURN, None)], [{"type": "turn_ended", "player": p}],
                    after, untranslatable="road-building-cancelled" if cancelled else None)

    def _offer(self, e, after) -> Step:
        o = self.ctx.ours(int(e.f["from"]))
        to = sorted(self.ctx.ours(i) for i, t in enumerate(e.f["to"].split(",")) if t == "true")
        give, get = ours5(e.f["give"]), ours5(e.f["get"])
        if after["current"] == self.ctx.js(o):
            self.ctx.offers += 1
        return Step("offer", o, [], [_offered(o, give, get)], after, info={"from": o, "to": to, "give": give, "get": get})
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `pgrep -fl 'arm.sh' || echo idle`, then `.venv/bin/pytest -q tests/python/test_jsettlers_translate.py tests/python/test_jsettlers_events.py`
Expected: PASS, 49 tests (18 + 31). If `test_a_robots_game_translates_and_four_observers_keep_their_beliefs` raises inside `import_game` or a bot's `belief_resets` is not 0, the message the failing step came from is the last `m` read: add it to the error, find the rule of `_action` that misread it, and fix that rule (the scripted tests above must still pass).

- [ ] **Step 7: Format, lint, the full suite**

Run the Global Constraints sequence. Expected: all pass, nothing skipped.

- [ ] **Step 8: Commit**

```bash
git add oracle/jsettlers/messages.py oracle/jsettlers/translate.py oracle/jsettlers/events.py \
  tests/python/test_jsettlers_translate.py tests/python/test_jsettlers_events.py
git diff --cached | grep -nE 'sk-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{36}|password=[^[:space:]]+'
git -c user.email=sean.may101@gmail.com commit -m "oracle/jsettlers: positions and events of ours from the JSettlers stream"
```

---

### Task 5: The differential replay of robots-only games (`jsettlers-diff`)

**Files:**
- Create: `oracle/jsettlers/diff.py`, `oracle/jsettlers/allowlist.py`, `oracle/compare.py`
- Modify: `alphasettler/cli.py` (module docstring, the `jsettlers-diff` parser, `_jsettlers_missing`, `_jsettlers_diff`, `main`), `oracle/translate.py` (`compare` moves to `oracle/compare.py`)
- Test: `tests/python/test_jsettlers_diff.py` (new)

**Interfaces:**
- Consumes: Task 3's `Jvm`, `JvmError`, `game_command`; Task 4's `Tracker`, `Step`, `snapshot`, `Untranslatable`; Task 1's `actions.decode`, `missing`; `alphasettler.Game` (`from_snapshot`, `legal_actions`, `apply`, `apply_forced`, `snapshot`); `alphasettler.arena.open_output`, `SEED_LIMIT`.
- Produces:
  - `oracle.jsettlers.allowlist`: `Entry(rule, why)`, `ENTRIES` with keys `domestic-trade`, `road-building-cancelled`, `bank-trade`, `forced-end-turn`.
  - `oracle.jsettlers.diff`: `CONFIG = {"max_offers_per_turn": 0, "bank_limit": False}`, `MAX_TURNS = 1000`, `GameResult(seed, steps, allowlisted, mismatch, trace, winner, turns, capped)`, `compare(ours, theirs) -> list[str]` (re-exported from `oracle.compare`), `excused(ours, theirs, step) -> dict` (a turn end compared with what ending the turn gives), `traded(step)`, `undo_fault(rates, gave, got, last) -> str | None` (an excused bank trade must be the exact reverse of the player's last bank trade this turn, whole maritime trades at its `Tracker.rates`), `play_game(jvm, seed, k=None) -> GameResult` (a failure of the Tracker, a translation or the start import is that game's mismatch; the trace stops at the mismatch), `run(seeds, workers=None) -> list[GameResult]` (one worker runs in the calling process), `Summary`, `summarize(results) -> Summary`, `format_summary(s) -> str`.
  - `alphasettler jsettlers-diff --games N [--seed-start S] [--workers W] [--out PATH]`: exit 0 with no mismatch, 1 with any (the first five printed with their traces; each mismatched game's JSONL line keeps its trace), 2 when the harness is missing, 3 when it fails (the JVM's error, exit or silence).

- [ ] **Step 1: Write the failing tests**

Create `tests/python/test_jsettlers_diff.py`:
```python
import copy
import json

import pytest

import oracle.jsettlers.diff as diff
import oracle.jsettlers.jvm as jvm_module
from alphasettler import Game, cli
from jsettlers_harness import require
from oracle.jsettlers import events
from oracle.jsettlers import tables as T
from oracle.jsettlers.allowlist import ENTRIES
from oracle.jsettlers.boards import js_layout, native_board
from oracle.jsettlers.diff import GameResult, compare, format_summary, play_game, summarize
from oracle.jsettlers.jvm import Jvm

SEED = 3  # the scripted games' seed: the board of their start message is native_board(SEED)
NAME = f"diff-{SEED}-0"


def rset(ours=(0, 0, 0, 0, 0)):
    return "clay={}|ore={}|sheep={}|wheat={}|wood={}|unknown=0".format(*T.ours_to_js5(list(ours)))


def board_raw(board):
    lay = js_layout(board)
    hexes = [[h, lay["hexes"][T.HEX_NUM[h]], max(0, lay["numbers"][T.HEX_NUM[h]])] for h in T.OUR_TILE_TO_JS_HEX]
    ports = [[e, T.MISC_PORT if kind == "generic" else T.OUR_RES_TO_JS[T.RESOURCE_NAMES.index(kind)]]
             for e, (_, kind) in zip(T.PORT_EDGES, board["ports"])]
    return {"hexes": hexes, "ports": ports, "robber": lay["robber"]}


class FakeJvm:
    """The harness's side of one scripted game: `recv` hands out its messages in turn; `sent` keeps
    what the driver sent."""

    def __init__(self, msgs):
        self.msgs, self.sent = list(msgs), []

    def send(self, msg):
        self.sent.append(msg)

    def recv(self, timeout):
        return self.msgs.pop(0)


class Stream:
    """One game's harness messages written by hand. JSettlers' first player is seat 0, so our seat i is
    its seat i; `raw` is the server-side position, and `ev(..., changed=True)` attaches it."""

    def __init__(self, state=T.PLAY1, hands=(), settlements=(), roads=()):
        board = native_board(SEED)
        players = [{"res": [0] * 5, "dev": [0] * 10, "played": [0] * 4, "settlements": [], "cities": [], "roads": [],
                    "discard": 0, "played_dev": False, "vp": 0} for _ in range(4)]
        self.raw = {"state": state, "current": 0, "robber": js_layout(board)["robber"],
                    "deck": [9] * 14 + [1, 1, 2, 2, 3, 3, 4, 5, 6, 7, 8], "players": players,
                    "longest": -1, "largest": -1, "winner": -1}
        for seat, h in hands:
            self.hand(seat, h)
        for seat, node in settlements:
            players[seat]["settlements"].append(T.OUR_NODE_TO_JS_NODE[node])
            players[seat]["vp"] += 1
        for seat, edge in roads:
            players[seat]["roads"].append(T.OUR_EDGE_TO_JS_EDGE[edge])
        self.msgs = [{"t": "start", "g": NAME, "first": 0, "board": board_raw(board), "s": copy.deepcopy(self.raw)}]

    def hand(self, seat, h):
        self.raw["players"][seat]["res"] = T.ours_to_js5(list(h))

    def ev(self, aud, text, changed=False):
        msg = {"t": "ev", "g": NAME, "a": aud, "m": text}
        if changed:
            msg["s"] = copy.deepcopy(self.raw)
        self.msgs.append(msg)

    def play(self):
        jvm = FakeJvm(self.msgs + [{"t": "over", "g": NAME, "winner": -1}])
        return play_game(jvm, SEED, 0), jvm


@pytest.fixture(scope="module")
def jvm(tmp_path_factory):
    require()
    with Jvm(log=tmp_path_factory.mktemp("jvm") / "jvm.log") as j:  # suite runs leave no logs under runs/
        yield j


def test_two_robots_games_replay_with_no_untriaged_mismatch(jvm):
    results = [play_game(jvm, seed) for seed in (41, 42)]
    for r in results:
        assert r.mismatch is None, (r.mismatch, r.trace[-5:])
        assert r.steps > 150 and (r.winner is not None or r.capped)
        assert set(r.allowlisted) <= set(ENTRIES)
    assert "0 mismatches" in format_summary(summarize(results))


def test_corrupted_dice_are_caught(jvm, monkeypatch):
    monkeypatch.setattr(events, "dice_pair", lambda total: [1, 1] if total != 2 else [1, 2])
    r = play_game(jvm, 43)
    # Every roll is replayed as a 2: the first roll that pays out (or a 7) differs in hands or phase.
    assert r.mismatch is not None and " after roll by " in r.mismatch, r.mismatch
    assert "hand'" in r.mismatch or "'phase'" in r.mismatch, r.mismatch
    assert r.trace[-1].startswith("f"), r.trace  # the trace ends at the request that closed the mismatching stretch


def test_compare_names_the_fields_that_differ():
    a = {"board": 1, "robber": 2, "bank": [1] * 5, "current": 0, "phase": "main", "turn": 3, "setup_step": 8,
         "longest_road_owner": None, "largest_army_owner": None, "dev_played_this_turn": False,
         "offers_this_turn": 0, "dev_deck": ["knight", "monopoly"], "return_phase": "main",
         "setup_road_node": None, "roads_left": None, "winner": None,
         "players": [{"hand": [0] * 5, "dev_hand": [0] * 5, "dev_new": [0] * 5, "dev_played": [0] * 5,
                      "knights_played": 0, "settlements": [], "cities": [], "roads": [], "discard_remaining": 0}] * 4}
    b = json.loads(json.dumps(a))
    b["dev_deck"] = ["monopoly", "knight"]  # the deck compares as a multiset
    assert compare(a, b) == []
    b["players"][2]["hand"] = [1, 0, 0, 0, 0]
    b["turn"] = 4
    assert compare(a, b) == ["turn", "players[2].hand"]


def test_every_allowlist_entry_names_its_rule():
    assert set(ENTRIES) == {"domestic-trade", "road-building-cancelled", "bank-trade", "forced-end-turn",
                            "forced-setup-end"}
    assert all(e.rule and e.why for e in ENTRIES.values())


def test_the_command_replays_and_writes_one_line_a_game(tmp_path, capsys, monkeypatch):
    require()
    monkeypatch.setattr(jvm_module, "LOGS", tmp_path / "logs")  # one worker runs in this process: no logs under runs/
    out = tmp_path / "diff.jsonl"
    rc = cli.main(["jsettlers-diff", "--games", "1", "--seed-start", "44", "--workers", "1", "--out", str(out)])
    text = capsys.readouterr().out
    assert rc == 0 and "1 games" in text and "0 mismatches" in text
    [line] = out.read_text().splitlines()
    assert json.loads(line)["seed"] == 44 and json.loads(line)["mismatch"] is None
    assert len(list((tmp_path / "logs").iterdir())) == 1


def test_the_command_says_what_is_missing(monkeypatch, tmp_path, capsys):
    import oracle.jsettlers as js

    monkeypatch.setattr(js, "JAR", tmp_path / "absent.jar")
    assert cli.main(["jsettlers-diff", "--games", "1"]) == 2
    assert "scripts/fetch-jsettlers.sh" in capsys.readouterr().err


def test_a_result_summary_counts_allowlisted_kinds():
    rs = [GameResult(1, steps=10), GameResult(2, steps=5)]
    rs[1].allowlisted["bank-trade"] += 2
    s = summarize(rs)
    assert (s.games, s.steps, s.clean_games, s.allowlisted["bank-trade"]) == (2, 15, 1, 2)


def roll_then_forced_end(stream, gain=None, nxt=1, state=T.ROLL_OR_CARD):
    """Player 0 rolls a 9 (nobody has a building on it) and the server then ends its stalled turn,
    handing it to seat `nxt` in `state` (JSettlers' position as the stream shows it)."""
    stream.ev("f0", "SOCRollDice:game=x")
    stream.ev("all", "SOCDiceResult:game=x|param=9")
    if gain is not None:
        stream.hand(*gain)
    stream.raw["state"] = T.PLAY1
    stream.ev("all", "SOCGameState:game=x|state=20", changed=True)
    stream.raw["current"], stream.raw["state"] = nxt, state
    stream.ev("all", f"SOCTurn:game=x|playerNumber={nxt}|gameState=15", changed=True)  # SOCForceEndTurnThread
    stream.ev("all", f"SOCRollDicePrompt:game=x|playerNumber={nxt}")


def road_building_ended(stream, nxt=1):
    """Player 0 ends its turn with both free roads unplaced; the turn goes to seat `nxt`."""
    stream.ev("f0", "SOCEndTurn:game=x")
    stream.raw["current"], stream.raw["state"] = nxt, T.ROLL_OR_CARD
    stream.ev("all", f"SOCTurn:game=x|playerNumber={nxt}|gameState=15", changed=True)


def domestic_trade(stream, partner_gets):
    """Player 0 gives one wood for player 2's wheat; JSettlers' position shows player 2 holding `partner_gets`."""
    give, get = rset([1, 0, 0, 0, 0]), rset([0, 0, 0, 1, 0])
    stream.ev("f0", f"SOCMakeOffer:game=x|from=0|to=false,true,true,true|give={give}|get={get}")
    stream.ev("all", f"SOCMakeOffer:game=x|from=0|to=false,true,true,true|give={give}|get={get}")
    stream.ev("f2", "SOCAcceptOffer:game=x|accepting=0|offering=0")
    stream.hand(0, [0, 0, 0, 1, 0])
    stream.hand(2, partner_gets)
    stream.ev("all", f"SOCAcceptOffer:game=x|accepting=2|offering=0|toAccepting={give}|toOffering={get}",
              changed=True)
    stream.ev("all", "SOCClearOffer:game=x|playerNumber=-1")


def bank_trades(stream, *trades):
    """Player 0's bank trades, each (give, get, its hand after) in our resource order."""
    for give, get, hand in trades:
        stream.ev("f0", f"SOCBankTrade:game=x|give={rset(give)}|get={rset(get)}")
        stream.hand(0, hand)
        stream.ev("all", f"SOCBankTrade:game=x|give={rset(give)}|get={rset(get)}|pn=0", changed=True)


def last_setup_road():
    """The start of our engine's last setup step on SEED's board (player 0's second road, first choices
    everywhere), as a Stream position, with player 0's second settlement node and what it produced."""
    g = Game(SEED)
    while not (g.phase == "setup_settlement" and g.snapshot()["setup_step"] == 7):
        g.apply(g.legal_actions()[0])
    before = g.snapshot()["players"][0]["hand"]
    g.apply(g.legal_actions()[0])
    snap = g.snapshot()
    node, ps = snap["setup_road_node"], snap["players"]
    produced = [a - b for a, b in zip(ps[0]["hand"], before)]
    start = dict(state=T.START2B, hands=[(i, p["hand"]) for i, p in enumerate(ps)],
                 settlements=[(i, n) for i, p in enumerate(ps) for n in p["settlements"]],
                 roads=[(i, e) for i, p in enumerate(ps) for e in p["roads"]])
    return start, node, produced


LAST_SETUP, SECOND_NODE, PRODUCED = last_setup_road()


def stalled_setup_road(stream, removed=SECOND_NODE, takes_back=True, nxt=1):
    """Player 0 asks for a road JSettlers refuses and stalls; the server ends its setup turn, removing the
    settlement `removed` (and, with `takes_back`, the cards it produced), and JSettlers' play begins at `nxt`."""
    stream.ev("f0", "SOCPutPiece:game=x|playerNumber=0|pieceType=0|coord=16")
    stream.ev("p0", "SOCDeclinePlayerRequest:game=x|state=0|reason=4|detail1=0|detail2=22|text=no")
    p = stream.raw["players"][0]
    p["settlements"].remove(T.OUR_NODE_TO_JS_NODE[removed])
    p["vp"] -= 1
    if takes_back:
        stream.hand(0, [h - x for h, x in zip(T.js5_to_ours(p["res"]), PRODUCED)])
    stream.ev("all", "SOCCancelBuildRequest:game=x|pieceType=1", changed=True)
    stream.raw["current"], stream.raw["state"] = nxt, T.ROLL_OR_CARD
    stream.ev("all", "SOCClearOffer:game=x|playerNumber=-1", changed=True)
    stream.ev("all", f"SOCTurn:game=x|playerNumber={nxt}|gameState=15")


def one(r, n=1):
    return [n if i == r else 0 for i in range(5)]


WOOD, ORE = 0, 4
PORT_EDGE, PORT_KIND = next((e, k) for e, k in native_board(SEED)["ports"] if k != "generic")
K = T.RESOURCE_NAMES.index(PORT_KIND)  # player 0's 2:1 port resource in the port cases
R, Z = [r for r in range(5) if r != K][:2]  # two resources it trades at 4:1
AT_PORT = dict(state=T.PLAY1, hands=((0, one(K, 2)),), settlements=((0, T.EDGE_NODES[PORT_EDGE][0]),))
FOUR_WOOD = dict(state=T.PLAY1, hands=((0, one(WOOD, 4)),))
WOOD_FOR_ORE = (one(WOOD, 4), one(ORE), one(ORE))  # a whole 4:1 maritime trade, replayed as ours
ROAD_BUILDING = dict(state=T.PLACING_FREE_ROAD1, settlements=((0, T.EDGE_NODES[0][0]),), roads=((0, 0),))
TRADERS = dict(hands=((0, [1, 0, 0, 0, 0]), (2, [0, 0, 0, 1, 0])))

# id: (start, script, its arguments, None when the step must pass, else a fragment the mismatch names)
ALLOWLISTED = {
    "forced end after a roll": (dict(state=T.ROLL_OR_CARD), roll_then_forced_end, (), None),
    "forced end after a roll that pays a card no building pays": (
        dict(state=T.ROLL_OR_CARD), roll_then_forced_end, ((2, [1, 0, 0, 0, 0]),), "players[2].hand"),
    "forced end handing the turn to the wrong seat": (
        dict(state=T.ROLL_OR_CARD), roll_then_forced_end, (None, 2), "'current'"),
    "forced end leaving the player in main": (
        dict(state=T.ROLL_OR_CARD), roll_then_forced_end, (None, 0, T.PLAY1), "'phase'"),
    "forced setup end": (LAST_SETUP, stalled_setup_road, (), None),
    "forced setup end keeping the cards the settlement produced": (
        LAST_SETUP, stalled_setup_road, (SECOND_NODE, False), "players[0].hand"),
    "forced setup end removing the settlement that has its road": (
        LAST_SETUP, stalled_setup_road, (LAST_SETUP["settlements"][0][1],), "players[0].settlements"),
    "road building ended": (ROAD_BUILDING, road_building_ended, (), None),
    "road building ended handing the turn to the wrong seat": (
        ROAD_BUILDING, road_building_ended, (2,), "'current'"),
    "domestic trade": (TRADERS, domestic_trade, ([1, 0, 0, 0, 0],), None),
    "domestic trade moving more than the traded cards": (TRADERS, domestic_trade, ([2, 0, 0, 0, 0],),
                                                         "players[2].hand"),
    "bank trade undone": (FOUR_WOOD, bank_trades, (WOOD_FOR_ORE, (one(ORE), one(WOOD, 4), one(WOOD, 4))), None),
    "bank trade undone at a port": (AT_PORT, bank_trades, ((one(K, 2), one(R), one(R)), (one(R), one(K, 2), one(K, 2))),
                                    None),
    "bank trade one for one at no port": (
        FOUR_WOOD, bank_trades, (WOOD_FOR_ORE, (one(ORE), one(WOOD), one(WOOD))), "not the undo of whole maritime"),
    "bank trade of nothing for five": (
        FOUR_WOOD, bank_trades, ((one(WOOD, 0), one(ORE, 5), [4, 0, 0, 0, 5]),), "not the undo of whole maritime"),
    "bank trade at a port the player does not have": (
        AT_PORT, bank_trades, ((one(K, 2), one(R), one(R)), (one(R), one(Z, 2), one(Z, 2))),
        "not the undo of whole maritime"),
    "bank trade undoing no trade of the player": (
        dict(state=T.PLAY1, hands=((0, one(ORE)),)), bank_trades, ((one(ORE), one(WOOD, 4), one(WOOD, 4)),),
        "reverses no bank trade"),
}


@pytest.mark.parametrize("case", list(ALLOWLISTED))
def test_an_allowlisted_step_is_compared_but_for_what_its_rule_changes(case):
    start, script, args, wrong = ALLOWLISTED[case]
    s = Stream(**start)
    script(s, *args)
    r, _ = s.play()
    if wrong is None:
        assert r.mismatch is None and sum(r.allowlisted.values()) == 1, (r.mismatch, r.allowlisted)
    else:
        assert r.mismatch is not None and wrong in r.mismatch, r.mismatch


def test_an_untranslatable_kind_off_the_allowlist_is_a_mismatch():
    s = Stream(state=T.ROLL_OR_CARD, hands=((2, [1, 0, 0, 0, 0]),))
    s.ev("f0", "SOCRollDice:game=x")
    s.ev("all", "SOCDiceResult:game=x|param=9")
    s.hand(2, [0, 0, 0, 0, 0])  # a roll that takes a card: no production of ours does that
    s.raw["state"] = T.PLAY1
    s.ev("all", "SOCGameState:game=x|state=20", changed=True)
    r, _ = s.play()
    assert "production" not in ENTRIES
    assert r.mismatch == "step 0 (roll): our rules cannot replay it (production)" and not r.allowlisted


def test_a_tracker_failure_is_the_games_mismatch_and_the_trace_stops_there():
    s = Stream(state=T.ROLL_OR_CARD)
    s.ev("f0", "SOCRollDice:game=x")
    s.ev("all", "SOCDiceResult:game=x|param=nine")  # a message the Tracker cannot read
    s.ev("f0", "SOCEndTurn:game=x")  # closes the stretch: the Tracker raises here
    s.raw["current"] = 1
    s.ev("all", "SOCTurn:game=x|playerNumber=1|gameState=15", changed=True)
    r, jvm = s.play()
    assert r.mismatch.startswith("step 0 (ev message): ValueError:"), r.mismatch
    assert r.trace[-1] == "f0:SOCEndTurn:game=x" and not any("SOCTurn" in t for t in r.trace)
    assert jvm.sent[-1] == {"t": "abort", "g": NAME} and jvm.msgs == []


def test_a_start_position_that_does_not_import_is_the_games_mismatch():
    s = Stream(state=T.PLACING_ROAD)  # a state with no phase of ours
    r, jvm = s.play()
    assert r.mismatch.startswith("step 0 (start message): Untranslatable:"), r.mismatch
    assert (r.winner, r.turns) == (None, 0) and jvm.sent[-1] == {"t": "abort", "g": NAME}


def test_a_failing_harness_exits_3_not_the_missing_harness_code(monkeypatch, capsys):
    def silent(seeds, workers=None):
        raise TimeoutError("the JSettlers JVM sent nothing for 900.0 s; see x.log")

    monkeypatch.setattr(cli, "_jsettlers_missing", lambda: None)
    monkeypatch.setattr(diff, "run", silent)
    assert cli.main(["jsettlers-diff", "--games", "1"]) == 3
    assert "harness failed: TimeoutError: the JSettlers JVM sent nothing" in capsys.readouterr().err


def test_a_mismatched_games_line_keeps_its_trace(monkeypatch, tmp_path, capsys):
    bad = GameResult(5, steps=3, mismatch="step 3 after roll by 0: ['phase']", trace=["f0:SOCRollDice:game=x"])
    monkeypatch.setattr(cli, "_jsettlers_missing", lambda: None)
    monkeypatch.setattr(diff, "run", lambda seeds, workers=None: [GameResult(4, steps=9), bad])
    out = tmp_path / "diff.jsonl"
    assert cli.main(["jsettlers-diff", "--games", "2", "--seed-start", "4", "--out", str(out)]) == 1
    clean, mismatched = (json.loads(line) for line in out.read_text().splitlines())
    assert "trace" not in clean and mismatched["trace"] == ["f0:SOCRollDice:game=x"]
    assert "mismatch seed 5: step 3 after roll by 0" in capsys.readouterr().err
```

Run: `.venv/bin/pytest -q tests/python/test_jsettlers_diff.py`
Expected: FAIL at collection: `No module named 'oracle.jsettlers.allowlist'`.

- [ ] **Step 2: Write `allowlist.py` and `diff.py`**

Create `oracle/jsettlers/allowlist.py`:
```python
"""The rule differences between JSettlers2 2.6.10 and our engine that the differential replay
allows (spec Sections 3 and 5 and the plan's rulings on trades and the end of a turn). On such a
step only what its rule changes may differ (oracle.jsettlers.diff.excused), and our engine then
re-imports JSettlers' position; anything else our engine cannot replay is a mismatch to triage."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Entry:
    rule: str
    why: str


ENTRIES = {
    "domestic-trade": Entry(
        "a domestic trade executes when the first player accepts; the offerer never confirms, offers go to a "
        "chosen subset, may be two cards for one and are uncapped",
        "our trading game (current player offers to all, answers in seat order, offerer confirms) cannot "
        "replay it, so the position after the trade is re-imported"),
    "road-building-cancelled": Entry(
        "a player may end the turn during Road Building, forfeiting the roads not placed",
        "our Road Building has no way to stop early"),
    "bank-trade": Entry(
        "one bank trade may be any exchange the rates allow, including undoing an earlier one (1 for 2-4)",
        "only bank trades that are whole maritime trades of ours are replayed as such"),
    "forced-end-turn": Entry(
        "the server ends a stalled player's turn itself, with no end-turn request (SOCForceEndTurnThread)",
        "our engine has no forced end of turn, so the position after it is re-imported"),
    "forced-setup-end": Entry(
        "the server ends a stalled setup turn itself, removing the settlement awaiting its road with the cards it "
        "produced, and the setup order (and JSettlers' first player) moves on",
        "our setup has no forced end and no event removes a settlement, so the position after it is re-imported"),
}
```

Create `oracle/compare.py`, moving `compare` out of `oracle/translate.py` unchanged (in `oracle/translate.py` the `def compare` block becomes `from oracle.compare import compare  # noqa: F401 (moved; oracle.diff imports it from here)` among its imports, and its now unused `Counter` import goes), so both differential replays share one comparison:
```python
"""Snapshot comparison shared by the Catanatron and JSettlers differential replays. Imports no engine of
another project, so both oracles may use it."""

from __future__ import annotations

from collections import Counter


def compare(ours: dict, theirs: dict) -> list[str]:
    """Paths where two snapshots disagree. Fields the engines represent differently are compared
    by meaning: the deck as a multiset (draw order beyond the next forced card is irrelevant)
    and the return phase only where it is used."""
    diffs = [k for k in ("board", "robber", "bank", "current", "phase", "turn", "setup_step",
                         "longest_road_owner", "largest_army_owner", "dev_played_this_turn",
                         "offers_this_turn")
             if ours[k] != theirs[k]]
    used = {"setup_road": "setup_road_node", "road_building": "roads_left", "game_over": "winner"}
    if ours["phase"] == theirs["phase"] and ours["phase"] in used:
        k = used[ours["phase"]]
        if ours[k] != theirs[k]:
            diffs.append(k)
    if (ours["phase"] == theirs["phase"] and ours["phase"] in ("move_robber", "road_building", "steal")
            and ours["return_phase"] != theirs["return_phase"]):
        diffs.append("return_phase")
    if Counter(ours["dev_deck"]) != Counter(theirs["dev_deck"]):
        diffs.append("dev_deck")
    for i, (a, b) in enumerate(zip(ours["players"], theirs["players"])):
        for k in ("hand", "dev_hand", "dev_new", "dev_played", "knights_played", "settlements", "cities",
                  "roads", "discard_remaining"):
            if a[k] != b[k]:
                diffs.append(f"players[{i}].{k}")
    return diffs
```

Create `oracle/jsettlers/diff.py`:
```python
"""Differential replay of robots-only JSettlers games through our engine (spec done criterion 2).

Four stock smart robots play in the harness on the seed's board and dice. Every step of the stream
(oracle/jsettlers/events.py) is replayed in our engine with its chance outcomes forced (the roll,
the steal, the card drawn), and the position after it is compared with JSettlers'. A step our rules
cannot express carries an allowlist kind: its translatable actions replay, the position is compared
with what that kind's rule changes done as the rule does it (`excused`), and our engine re-imports
JSettlers' position. Anything else is a mismatch, which ends that game's comparison; so does a
failure of the Tracker or of an import. Positions import with the bank limit off and no offers
(robots' trades are replayed by re-import). A game that reaches our turn cap is stopped."""

from __future__ import annotations

import copy
import multiprocessing as mp
import os
from collections import Counter
from dataclasses import dataclass, field

from alphasettler import Game
from oracle.compare import compare
from oracle.jsettlers import allowlist
from oracle.jsettlers.actions import decode
from oracle.jsettlers.boards import game_command
from oracle.jsettlers.events import Step, Tracker
from oracle.jsettlers.jvm import Jvm
from oracle.jsettlers.tables import RESOURCE_NAMES, TILE_NODES
from oracle.jsettlers.translate import Untranslatable, import_game, snapshot

CONFIG = {"max_offers_per_turn": 0, "bank_limit": False}
MAX_TURNS = 1000
GAME_TIMEOUT = 900.0
TRACE_LEN = 20


@dataclass
class GameResult:
    seed: int
    steps: int = 0  # steps with actions replayed and compared
    allowlisted: Counter = field(default_factory=Counter)
    mismatch: str | None = None
    trace: list = field(default_factory=list)
    winner: int | None = None
    turns: int = 0
    capped: bool = False


def traded(st: Step) -> tuple[list[int], list[int]]:
    """The cards a bank trade step gave and got, summed over its maritime trades."""
    gave, got = [0] * 5, [0] * 5
    for e in st.events:
        if e["type"] == "maritime_traded":
            gave = [a + b for a, b in zip(gave, e["gave"])]
            got = [a + b for a, b in zip(got, e["got"])]
    return gave, got


def undo_fault(rates: list[int], gave: list[int], got: list[int], last) -> str | None:
    """Why an allowlisted bank trade is not JSettlers' undo of the player's last bank trade this turn,
    or None. An undo is the exact reverse of whole maritime trades at the player's rates (`rates`, our
    resource order): it gets back `rate` cards of a resource for each card it gives, never a resource
    on both sides, and it reverses `last`, the player's last bank trade this turn as (gave, got)."""
    units = sum(g // rate for g, rate in zip(got, rates))
    if (not any(gave) or any(g % rate for g, rate in zip(got, rates)) or units != sum(gave)
            or any(a and b for a, b in zip(gave, got))):
        return f"{gave} for {got} is not the undo of whole maritime trades at rates {rates}"
    if last != (got, gave):
        return f"{gave} for {got} reverses no bank trade of the player this turn (last: {last})"
    return None


def excused(ours: dict, theirs: dict, st: Step) -> dict:
    """Our position `ours` (before the step, its translatable actions applied) with what the step's
    allowlist rule changes done as the rule does it: a domestic trade moves exactly the traded cards
    between the two traders, a bank trade exactly the cards traded with the bank, and a turn that ends
    (forced, or inside Road Building) passes to the next player's roll as our end of turn does. A setup turn
    the server ended loses the settlement awaiting its road and what that settlement produced; the setup
    order after it (current player, phase, step, turn) is JSettlers' own, the one field set read from `theirs`.
    Otherwise `theirs` is not read: compared with JSettlers' position, anything that differs is a mismatch."""
    exp = copy.deepcopy(ours)
    hands = [p["hand"] for p in exp["players"]]

    def move(player: int, gain, loss) -> None:
        hands[player][:] = [h + g - x for h, g, x in zip(hands[player], gain, loss)]

    if st.untranslatable == "domestic-trade":
        i = st.info
        move(i["offerer"], i["get"], i["give"])
        move(i["partner"], i["give"], i["get"])
    elif st.untranslatable == "bank-trade":
        gave, got = traded(st)
        move(st.player, got, gave)
    elif st.untranslatable == "forced-setup-end":
        node = exp["setup_road_node"] if exp["phase"] == "setup_road" else None
        if node is not None:
            exp["players"][st.player]["settlements"].remove(node)
            if exp["setup_step"] >= 4:  # a second settlement: what it produced goes back
                res = exp["board"]["tile_resource"]
                for t, nodes in enumerate(TILE_NODES):
                    if node in nodes and res[t] is not None:
                        hands[st.player][RESOURCE_NAMES.index(res[t])] -= 1
        for k in ("current", "phase", "setup_road_node", "setup_step", "turn", "return_phase", "roads_left"):
            exp[k] = theirs[k]
    if st.untranslatable in ("domestic-trade", "bank-trade", "forced-setup-end"):
        exp["bank"] = [max(0, 19 - sum(h[r] for h in hands)) for r in range(5)]  # the bank with its limit off
    if any(e["type"] == "turn_ended" for e in st.events):
        exp.update(current=(exp["current"] + 1) % 4, phase="pre_roll", return_phase="pre_roll", roads_left=None,
                   setup_road_node=None, dev_played_this_turn=False, offers_this_turn=0, turn=exp["turn"] + 1)
        for q in exp["players"]:
            q["dev_new"], q["discard_remaining"] = [0] * 5, 0
    return exp


def _replay(res: GameResult, seed: int, tracker: Tracker, ours: Game, st: Step, trades: dict) -> Game:
    """Replay one step on `ours` and compare; returns the game to go on with (JSettlers' position,
    re-imported, after an allowlisted step). `trades` keeps each player's last bank trade as
    (turn, gave, got), to check an undo against."""
    if st.kind == "end_turn" and tracker.ctx.turn >= MAX_TURNS:
        res.capped = True  # our engine ends the game here; JSettlers would play on
        return ours
    kind = st.untranslatable
    if kind is not None and kind not in allowlist.ENTRIES:
        res.mismatch = f"step {res.steps} ({st.kind}): our rules cannot replay it ({kind})"
        return ours
    turn = ours.turn
    if st.actions:  # an allowlisted step's translatable actions (a roll before a forced end) replay too
        try:
            for a, chance in st.actions:
                if a not in ours.legal_actions():
                    res.mismatch = f"step {res.steps} ({st.kind}): {decode(a)} is not legal in ours ({ours.phase})"
                    return ours
                if chance is None:
                    ours.apply(a)
                else:
                    ours.apply_forced(a, chance)
        except BaseException as e:  # noqa: BLE001 - a panic (PyO3 PanicException) is a finding too
            if isinstance(e, (KeyboardInterrupt, SystemExit)):
                raise
            res.mismatch = f"step {res.steps} ({st.kind}): {type(e).__name__}: {e}"
            return ours
        if kind is None:
            res.steps += 1
    if st.after is None:
        if kind is not None:
            res.mismatch = f"step {res.steps} ({st.kind}): an allowlisted step ({kind}) without JSettlers' position"
        return ours
    try:
        theirs = snapshot(st.after, tracker.ctx, 0)
    except Untranslatable as e:
        res.mismatch = f"step {res.steps} ({st.kind}): JSettlers' position after it does not translate: {e}"
        return ours
    if st.kind == "maritime":
        last = trades.get(st.player)
        trades[st.player] = (turn, *traded(st))
        if kind == "bank-trade":
            why = undo_fault(tracker.rates(st.after, st.player), *traded(st),
                             last[1:] if last is not None and last[0] == turn else None)
            if why is not None:
                res.mismatch = f"step {res.steps} ({st.kind}) by {st.player} ({kind}): {why}"
                return ours
    if kind is None:
        diffs = compare(ours.snapshot(), theirs)
        if diffs:
            res.mismatch = f"step {res.steps} after {st.kind} by {st.player}: {diffs}"
        return ours
    res.allowlisted[kind] += 1
    diffs = compare(excused(ours.snapshot(), theirs, st), theirs)
    if diffs:
        res.mismatch = f"step {res.steps} after {st.kind} by {st.player} ({kind}): beyond what its rule changes: {diffs}"
        return ours
    try:
        return Game.from_snapshot(seed, theirs, CONFIG)
    except ValueError as e:
        res.mismatch = f"step {res.steps} ({st.kind}): JSettlers' position does not import: {e}"
        return ours


def _advance(res: GameResult, seed: int, tracker: Tracker | None, ours: Game | None, m: dict, trades: dict):
    if m["t"] == "start":
        tracker = Tracker(m)
        return tracker, import_game(seed, tracker.raw, tracker.ctx, CONFIG)
    for st in tracker.feed(m) if m["t"] == "ev" else tracker.flush():
        if res.mismatch is None and not res.capped:
            ours = _replay(res, seed, tracker, ours, st, trades)
    return tracker, ours


def play_game(jvm: Jvm, seed: int, k: int | None = None) -> GameResult:
    """One robots-only game on `jvm`, replayed step by step. A broken harness (the JVM's error, exit
    or silence) raises; a failure of ours (the Tracker, a translation, the start position's import)
    is this game's mismatch. The trace holds the stream up to the mismatch or the turn cap."""
    k = seed % 4 if k is None else k
    name = f"diff-{seed}-{k}"
    res = GameResult(seed)
    jvm.send(game_command(name, seed, k, bridge=False))
    tracker, ours, aborted, trades = None, None, False, {}
    while True:
        m = jvm.recv(GAME_TIMEOUT)
        if m.get("g") != name:
            raise RuntimeError(f"a message from another game while replaying {name}: {m}")
        if res.mismatch is None and not res.capped:
            if m["t"] == "ev":
                res.trace = (res.trace + [f"{m['a']}:{m['m']}"])[-TRACE_LEN:]
            try:
                tracker, ours = _advance(res, seed, tracker, ours, m, trades)
            except BaseException as e:  # noqa: BLE001 - a panic (PyO3 PanicException) is a finding too
                if isinstance(e, (KeyboardInterrupt, SystemExit)):
                    raise
                res.mismatch = f"step {res.steps} ({m['t']} message): {type(e).__name__}: {e}"
        if m["t"] == "over":
            res.winner = None if m["winner"] < 0 or tracker is None else tracker.ctx.ours(m["winner"])
            res.turns = 0 if tracker is None else tracker.ctx.turn
            return res
        if (res.mismatch is not None or res.capped) and not aborted:
            jvm.send({"t": "abort", "g": name})
            aborted = True


def _chunk(seeds: list[int]) -> list[GameResult]:
    with Jvm() as jvm:
        return [play_game(jvm, s) for s in seeds]


def run(seeds, workers: int | None = None) -> list[GameResult]:
    """Each worker process runs one JVM and plays its share of the seeds in turn; one worker runs in
    this process."""
    seeds = list(seeds)
    n = max(1, min(workers or os.cpu_count() or 1, len(seeds)))
    if n == 1:
        return sorted(_chunk(seeds), key=lambda r: r.seed)
    chunks = [seeds[i::n] for i in range(n)]
    with mp.get_context("spawn").Pool(n) as pool:
        results = [r for chunk in pool.map(_chunk, chunks) for r in chunk]
    return sorted(results, key=lambda r: r.seed)


@dataclass
class Summary:
    games: int
    steps: int
    clean_games: int
    allowlisted: Counter
    allowlisted_games: Counter
    mismatches: list
    capped: int


def summarize(results) -> Summary:
    results = list(results)
    total, games = Counter(), Counter()
    for r in results:
        total.update(r.allowlisted)
        games.update(r.allowlisted.keys())
    return Summary(games=len(results), steps=sum(r.steps for r in results),
                   clean_games=sum(1 for r in results if not r.allowlisted and r.mismatch is None),
                   allowlisted=total, allowlisted_games=games,
                   mismatches=[r for r in results if r.mismatch is not None], capped=sum(r.capped for r in results))


def format_summary(s: Summary) -> str:
    lines = [f"{s.games} games, {s.steps} steps compared, {s.clean_games} games identical end to end, "
             f"{len(s.mismatches)} mismatches, {s.capped} stopped at the turn cap"]
    for kind, n in sorted(s.allowlisted.items()):
        lines.append(f"  allowlisted {kind}: {n} steps in {s.allowlisted_games[kind]} games "
                     f"({allowlist.ENTRIES[kind].rule})")
    return "\n".join(lines)
```

- [ ] **Step 3: The command**

In `alphasettler/cli.py`:
1. In the module docstring's first line, after `` `oracle-diff`, `` insert `` `jsettlers-diff`, ``, and add one sentence at the end of the docstring: ``` `jsettlers-diff` replays robots-only JSettlers2 games through our engine (needs the harness: scripts/fetch-jsettlers.sh, jsettlers/build.sh; exit 0 with no mismatch, 1 with any, 2 when the harness is missing, 3 when it fails).```
2. In `_parser`, after the `oracle-diff` block:
```python
    jd = sub.add_parser("jsettlers-diff", help="differential replay of robots-only JSettlers2 games (needs the harness)")
    jd.add_argument("--games", type=_positive_int, required=True)
    jd.add_argument("--seed-start", type=int, default=9_000_000_000)
    jd.add_argument("--workers", type=_positive_int, default=None, help="worker processes, one JVM each")
    jd.add_argument("--out", default=None, help="JSONL path, one line per game")
```
3. After `_oracle_diff`:
```python
def _jsettlers_missing() -> str | None:
    from oracle.jsettlers import missing

    return missing()


def _jsettlers_diff(args) -> int:
    why = _jsettlers_missing()
    if why is not None:
        print(f"error: the JSettlers harness is not ready: {why}", file=sys.stderr)
        return 2
    from oracle.jsettlers.diff import format_summary, run, summarize

    if args.seed_start < 0 or args.seed_start + args.games > SEED_LIMIT:
        raise ValueError(f"seeds {args.seed_start}..{args.seed_start + args.games} are outside the u64 seed range")
    with ExitStack() as stack:
        f = stack.enter_context(open_output(args.out)) if args.out else None
        try:
            results = run(range(args.seed_start, args.seed_start + args.games), args.workers)
        except (OSError, EOFError, RuntimeError) as e:  # the JVM's error, exit or silence (TimeoutError is an OSError)
            print(f"error: the JSettlers harness failed: {type(e).__name__}: {e}", file=sys.stderr)
            return 3
        s = summarize(results)
        print(format_summary(s))
        if f is not None:
            for r in results:
                rec = {"seed": r.seed, "steps": r.steps, "allowlisted": dict(r.allowlisted), "mismatch": r.mismatch,
                       "winner": r.winner, "turns": r.turns, "capped": r.capped}
                if r.mismatch is not None:
                    rec["trace"] = r.trace  # the stream up to the mismatch
                f.write(json.dumps(rec) + "\n")
            print(f"wrote {args.out}")
    for r in s.mismatches[:5]:
        print(f"mismatch seed {r.seed}: {r.mismatch}", file=sys.stderr)
        for line in r.trace[-5:]:
            print(f"    {line}", file=sys.stderr)
    return 1 if s.mismatches else 0
```
4. In `main`, after the `oracle-diff` branch: `elif args.command == "jsettlers-diff": return _jsettlers_diff(args)`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pgrep -fl 'arm.sh' || echo idle`, then `.venv/bin/pytest -q tests/python/test_jsettlers_diff.py tests/python/test_oracle_diff.py tests/python/test_oracle_trades.py`
Expected: PASS, 29 tests in the first file; the two oracle files pass unchanged (the move of `compare` changes nothing). A mismatch in `test_two_robots_games_replay_with_no_untriaged_mismatch` is the point of the gate: read the trace it prints, decide whether it is a translation bug (fix it in Task 4's code, with a scripted test in `test_jsettlers_events.py` that reproduces the sequence) or a rule difference (add an allowlist entry with its rule and reason, a classifying test, and a ruling line in the results file of Task 9); never loosen `compare`.

- [ ] **Step 5: Format, lint, the full suite**

Run the Global Constraints sequence. Expected: all pass, nothing skipped.

- [ ] **Step 6: Commit**

```bash
git add oracle/jsettlers/diff.py oracle/jsettlers/allowlist.py oracle/compare.py oracle/translate.py alphasettler/cli.py tests/python/test_jsettlers_diff.py docs/superpowers/plans/2026-10-09-plan-13-jsettlers-arena.md
git diff --cached | grep -nE 'sk-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{36}|password=[^[:space:]]+'
git -c user.email=sean.may101@gmail.com commit -m "oracle/jsettlers: differential replay of robots-only JSettlers games (jsettlers-diff)"
```

---

### Task 6: The bridge seat and the arena driver

**Files:**
- Create: `jsettlers/src/alphasettler/jsettlers/{BridgeClient,BridgeBrain,BridgeOpening,BridgeRobber,BridgeDM}.java`, `oracle/jsettlers/arena.py`
- Modify: `jsettlers/src/alphasettler/jsettlers/ArenaMain.java` (start the bridge client), `jsettlers/api.txt` (seven brain fields and the opening's first settlement)
- Test: `tests/python/test_jsettlers_seat.py`, `tests/python/test_jsettlers_arena.py` (new), `tests/python/data/jsettlers_seed78_k1.jsonl` (new: the harness's lines of the seed-78 bridge game, k = 1, recorded at fdd5a6d; see stall-debug.md); Modify: `oracle/jsettlers/events.py`, `oracle/jsettlers/allowlist.py`, `oracle/jsettlers/diff.py` and their tests (fix round 1: the forced setup end)

**Interfaces:**
- Consumes: Tasks 3-4 (`Jvm`, `game_command`, `Tracker`, `Untranslatable`, the RAW format), Task 1 (`actions`, `tables`, `missing`), `alphasettler.Bot`, `alphasettler.Game`, `alphasettler.arena.TRADES`; in Java the stock `SOCRobotBrain` turn machine and the members in `api.txt`.
- Produces:
  - Asks (Java to Python): `{"t": "ask", "id", "g", "kind", "state", "extra"}` with kind `setup_settlement`, `setup_road`, `pre_roll`, `main`, `road_building`, `move_robber`, `discard` (`extra.n`), `answer` (Task 7). Replies: `{"t": "reply", "id", "fallback": REASON}` (the stock brain decides that hook) or, by kind, `{"node": JS_NODE}`, `{"edge": JS_EDGE}`, `{"do": "roll" | "knight"}`, `{"do": "end"}` / `{"do": "build", "piece": 0|1|2, "coord": JS}` / `{"do": "buy"}` / `{"do": "knight"}` / `{"do": "rb"}` / `{"do": "yop", "res": [js, js]}` / `{"do": "mono", "res": js}` / `{"do": "bank", "give": js5, "get": js5}` (Task 7 adds `{"do": "offer", ...}` and `{"accept": BOOL}`), `{"hex": JS_HEX, "victim": JS_SEAT | -1}`, `{"res": js5}`. Java's own fallbacks: `{"t": "fallback", "g", "reason": "java-timeout" | "victim"}`.
  - `oracle.jsettlers.arena`: `BASELINES = ("smart",)`, `NO_TRADES`, `LATE = 4.5`, `SILENCE = 900.0`, `MAX_TURNS = 1000`, `FALLBACK_REASONS`, `AFTER_WIN`, `COUNTERS`, `RECORD_KEYS`, `Fallback(reason)`, `config(trades) -> dict`, `Seat(bot, seed, config, tracker, seat, max_turns=MAX_TURNS, clock=time.monotonic)` with `.observe(steps)`, `.decide(ask, received) -> dict`, `.fellback(reason)`, `.fallbacks` (Counter), `.counts` (Counter), `.capped`, `.last_build` (per seat of ours); records add `robot_desync`; `play_game(jvm, seed, k, candidate, nets_dir=None, trades=True, arm="native", native_boards=False, max_turns=MAX_TURNS) -> dict`; `check(candidate, baseline, seeds, seed_start=0, nets_dir=None)`; `run_match(candidate, baseline, seeds, seed_start=0, workers=None, nets_dir=None, trades=True, aligned=False, native_boards=False, max_turns=MAX_TURNS, log_dir=None) -> list[dict]` (native boards need trades).

- [ ] **Step 1: Write the failing Seat tests**

These need no JVM: a `Script` (from `test_jsettlers_events.py`) gives a real `Tracker` at a scripted position and a stub bot chooses.

Create `tests/python/test_jsettlers_seat.py`:
```python
from collections import Counter

import pytest

from oracle.jsettlers import tables as T
from oracle.jsettlers.actions import (
    ACCEPT, BUY_DEV, CANCEL, END_TURN, KNIGHT, ROAD_BUILDING, ROLL, confirm, discard, maritime, monopoly,
    move_robber, offer, road, settlement, steal_from, year_of_plenty,
)
from oracle.jsettlers.arena import COUNTERS, NO_TRADES, Seat, config
from test_jsettlers_events import BOARD, Script
from test_jsettlers_events import rset as js_rset

TRADES = config(True)
WOOD, BRICK, SHEEP, WHEAT, ORE = range(5)
HOME = 20  # our settlement
NEAR = {HOME} | {b for a, b in T.EDGE_NODES if a == HOME} | {a for a, b in T.EDGE_NODES if b == HOME}
HOME_ROAD = min(e for e, (a, b) in enumerate(T.EDGE_NODES) if HOME in (a, b))


class StubBot:
    """Chooses with `choose(game) -> action id`; records the positions it was asked at and what it saw."""

    def __init__(self, choose):
        self.choose, self.seen, self.positions, self.chosen = choose, [], [], []

    def act(self, game):
        self.positions.append((game.phase, game.current_actor))
        a = self.choose(game)
        self.chosen.append(a)
        return a

    def observe(self, seat, events):
        self.seen.extend(events)

    def diagnostics(self):
        return {}


def seat_at(state=T.PLAY1, current=0, first=0, rolled=True, hands=None, cfg=TRADES, choose=None, clock=None):
    s = Script(first=first, state=state, current=current)
    for p, h in (hands or {}).items():
        s.hand(p, h)
    s.raw["players"][s.js(0)]["settlements"] = [T.OUR_NODE_TO_JS_NODE[HOME]]
    s.raw["players"][s.js(0)]["roads"] = [T.OUR_EDGE_TO_JS_EDGE[HOME_ROAD]]
    s.send("all", f"SOCGameState:game=x|state={state}", changed=True)
    s.tracker.flush()
    s.tracker.ctx.rolled = rolled
    bot = StubBot(choose or (lambda g: g.legal_actions()[0]))
    seat = Seat(bot, 5, cfg, s.tracker, 0, clock=clock or (lambda: 0.0))
    return s, seat, bot


def ask(kind, **extra):
    return {"t": "ask", "id": 7, "g": "x", "kind": kind, "state": 0, "extra": extra}


def first_road(g):
    return next(a for a in g.legal_actions() if road(0) <= a < road(72))


def test_main_choices_become_bridge_requests():
    _, seat, bot = seat_at(hands={0: [1, 1, 0, 0, 0]}, choose=first_road)
    r = seat.decide(ask("main"), 0.0)
    edge = bot.chosen[0] - road(0)
    assert r == {"t": "reply", "id": 7, "do": "build", "piece": T.ROAD, "coord": T.OUR_EDGE_TO_JS_EDGE[edge]}
    _, seat, _ = seat_at(choose=lambda g: END_TURN)
    assert seat.decide(ask("main"), 0.0)["do"] == "end"
    _, seat, _ = seat_at(choose=lambda g: BUY_DEV, hands={0: [0, 0, 1, 1, 1]})
    assert seat.decide(ask("main"), 0.0)["do"] == "buy"
    s, seat, _ = seat_at(hands={0: [4, 0, 0, 0, 0]}, choose=lambda g: maritime(WOOD, ORE))
    rate = s.tracker.rates(s.raw, 0)[WOOD]
    r = seat.decide(ask("main"), 0.0)
    assert (r["do"], r["give"], r["get"]) == ("bank", T.ours_to_js5([rate, 0, 0, 0, 0]), T.ours_to_js5([0, 0, 0, 0, 1]))


@pytest.mark.parametrize("card,action,expected", [
    (2, year_of_plenty(WHEAT, ORE), {"do": "yop", "res": [T.OUR_RES_TO_JS[WHEAT], T.OUR_RES_TO_JS[ORE]]}),
    (3, monopoly(SHEEP), {"do": "mono", "res": T.OUR_RES_TO_JS[SHEEP]}),
    (9, KNIGHT, {"do": "knight"}),
])
def test_card_plays_become_bridge_requests(card, action, expected):
    s, seat, _ = seat_at(choose=lambda g: action)
    s.raw["players"][s.js(0)]["dev"][card] = 1
    s.raw["deck"].remove(card)  # the card in hand left the deck
    s.send("all", "SOCGameState:game=x|state=20", changed=True)
    s.tracker.flush()
    r = seat.decide(ask("main"), 0.0)
    assert {k: r[k] for k in expected} == expected


def test_unimportable_declined_or_failing_decisions_fall_back_by_reason():
    s, seat, _ = seat_at(state=T.PLACING_ROAD)
    assert seat.decide(ask("main"), 0.0) == {"t": "reply", "id": 7, "fallback": "position"}
    s, seat, _ = seat_at(choose=lambda g: END_TURN)
    s.tracker.declined[0] += 1
    assert seat.decide(ask("main"), 0.0)["fallback"] == "declined"
    assert seat.decide(ask("main"), 0.0)["do"] == "end"  # the same decline is not a reason twice

    def boom(g):
        raise RuntimeError("engine panic")
    _, seat, _ = seat_at(choose=boom)
    assert seat.decide(ask("main"), 0.0)["fallback"] == "engine"
    _, seat, _ = seat_at(state=T.ROLL_OR_CARD, rolled=False, choose=lambda g: ROAD_BUILDING)
    assert seat.decide(ask("pre_roll"), 0.0)["fallback"] == "preroll-card"
    _, seat, _ = seat_at(state=T.ROLL_OR_CARD, rolled=False, choose=lambda g: ROLL)
    assert seat.decide(ask("pre_roll"), 0.0)["do"] == "roll"
    assert seat.fallbacks == Counter()
    seat.fellback("victim")
    assert seat.fallbacks == Counter({"victim": 1})
    seat.capped = True  # the driver aborted the game at our turn cap
    assert seat.decide(ask("main"), 0.0)["fallback"] == "turn-cap"


@pytest.mark.parametrize("state,kind,extra", [
    (T.PLACING_ROBBER, "move_robber", {}),
    (T.PLACING_FREE_ROAD1, "road_building", {}),
    (T.WAITING_FOR_DISCARDS, "discard", {"n": 4}),
])
def test_a_refusal_in_the_robber_free_road_or_discard_hook_falls_back_there(state, kind, extra):
    s, seat, _ = seat_at(state=state, choose=never)
    s.tracker.declined[0] += 1
    assert seat.decide(ask(kind, **extra), 0.0)["fallback"] == "declined"
    assert seat.fallbacks == Counter({"declined": 1})


def test_a_decision_at_the_seats_turn_cap_falls_back():
    s, _, _ = seat_at(choose=never)
    seat = Seat(StubBot(never), 5, TRADES, s.tracker, 0, max_turns=3, clock=lambda: 0.0)
    s.tracker.ctx.turn = 3
    assert seat.decide(ask("main"), 0.0)["fallback"] == "turn-cap"


def test_a_declined_setup_placement_falls_back_instead_of_being_asked_again():
    # JSettlers re-arms the placement after a refusal; answering the same node again would loop until the
    # robot client leaves the game.
    s = Script(state=T.START1A, current=0)
    s.send("all", "SOCGameState:game=x|state=5", changed=True)
    s.tracker.flush()
    bot = StubBot(lambda g: g.legal_actions()[0])
    seat = Seat(bot, 5, TRADES, s.tracker, 0, clock=lambda: 0.0)
    s.tracker.declined[0] += 1
    assert seat.decide(ask("setup_settlement"), 0.0)["fallback"] == "declined"
    r = seat.decide(ask("setup_settlement"), 0.0)
    assert r["node"] == T.OUR_NODE_TO_JS_NODE[bot.chosen[0] - settlement(0)]
    s.tracker.declined[0] += 1
    assert seat.decide(ask("setup_road"), 0.0)["fallback"] == "declined"
    assert seat.fallbacks == Counter({"declined": 2})


def test_the_robber_reply_carries_the_victim_chosen_at_the_steal_position():
    s, seat, bot = seat_at(state=T.PLACING_ROBBER, hands={2: [0, 0, 1, 0, 0]})
    tile = next(t for t in range(19) if BOARD["tile_resource"][t] is not None and not NEAR & set(T.TILE_NODES[t]))
    victim_node = T.TILE_NODES[tile][0]
    s.raw["players"][s.js(2)]["settlements"] = [T.OUR_NODE_TO_JS_NODE[victim_node]]
    s.send("all", "SOCGameState:game=x|state=33", changed=True)
    s.tracker.flush()
    bot.choose = lambda g: move_robber(tile) if g.phase == "move_robber" else steal_from(2)
    r = seat.decide(ask("move_robber"), 0.0)
    assert r["hex"] == T.OUR_TILE_TO_JS_HEX[tile] and r["victim"] == s.js(2)
    assert [p for p, _ in bot.positions] == ["move_robber", "steal"]


def never(g):
    raise AssertionError("the bot must not be asked")


def test_the_robber_move_asked_after_our_winning_knight_is_answered_but_not_a_fallback():
    # Our rules end the game when the knight's Largest Army reaches 10 points; JSettlers still asks us to
    # move the robber before it awards the win.
    s, seat, _ = seat_at(choose=never)
    p = s.raw["players"][s.js(0)]
    p["vp"], p["dev"][9], p["played"] = 8, 1, [2, 0, 0, 0]
    s.raw["deck"] = s.raw["deck"][3:]
    s.send("all", "SOCGameState:game=x|state=20", changed=True)
    s.tracker.flush()
    s.send(f"f{s.js(0)}", "SOCPlayDevCardRequest:game=x|devCard=9")
    p["dev"][9], p["played"], p["played_dev"], p["vp"] = 0, [3, 0, 0, 0], True, 10
    s.raw["state"], s.raw["largest"] = T.PLACING_ROBBER, s.js(0)
    s.send("all", f"SOCDevCardAction:game=x|playerNum={s.js(0)}|actionType=PLAY|cardType=9", changed=True)
    s.send("all", "SOCGameState:game=x|state=33")
    seat.observe(s.flush())
    assert s.tracker.over
    assert seat.decide(ask("move_robber"), 0.0) == {"t": "reply", "id": 7, "fallback": "after-win"}
    assert seat.fallbacks == Counter()


def discard_at(choose=None):
    """Our seat (4 wood, 4 brick) and seat 1 each owe 4 cards after a 7."""
    s, seat, bot = seat_at(state=T.WAITING_FOR_DISCARDS, hands={0: [4, 4, 0, 0, 0], 1: [0, 0, 8, 0, 0]}, choose=choose)
    s.raw["players"][s.js(0)]["discard"], s.raw["players"][s.js(1)]["discard"] = 4, 4
    s.send("all", "SOCGameState:game=x|state=50", changed=True)
    s.tracker.flush()
    return s, seat, bot


def stream_discard(s, seat, cards, hand, forced=False):
    """The stream reports our seat's discard of `cards`: after our request, or forced by the server."""
    js0 = s.js(0)
    if not forced:
        s.send(f"f{js0}", f"SOCDiscard:game=x|resources={js_rset(cards)}")
    s.hand(0, hand)
    s.raw["players"][js0]["discard"] = 0
    s.send("all", f"SOCDiscard:game=x|playerNum={js0}|resources={js_rset(cards)}", changed=True)
    seat.observe(s.flush())
    s.steps = []


def discarded(r, n=1):
    return [{"type": "discarded", "player": 0, "resource": T.RESOURCE_NAMES[r]}] * n


def test_discards_are_chosen_one_card_at_a_time_where_only_our_seat_owes():
    s, seat, bot = discard_at(lambda g: discard(WOOD) if discard(WOOD) in g.legal_actions() else discard(BRICK))
    r = seat.decide(ask("discard", n=4), 0.0)
    assert r["res"] == T.ours_to_js5([4, 0, 0, 0, 0])
    assert all(actor == 0 for _, actor in bot.positions) and len(bot.positions) == 4
    # Each discard is shown before the next is chosen (the bot's belief checks its hand at every decision),
    # and the stream's report of the same discards is not shown again.
    ours = {"type": "discarded", "player": 0, "resource": "wood"}
    assert bot.seen == [ours] * 4
    stream_discard(s, seat, [4, 0, 0, 0, 0], [0, 4, 0, 0, 0])
    assert bot.seen == [ours] * 4 and seat.fallbacks == Counter()


def test_a_discard_cut_short_is_completed_with_the_first_discards_our_rules_allow():
    # The cards already shown stand: the reply holds them and the rest, so the bot's view stays true.
    def two_then_panic(g):
        if len(bot.chosen) >= 2:
            raise RuntimeError("engine panic")
        return discard(BRICK)
    s, seat, bot = discard_at(two_then_panic)
    r = seat.decide(ask("discard", n=4), 0.0)
    assert r["res"] == T.ours_to_js5([2, 2, 0, 0, 0]) and seat.fallbacks == Counter({"engine": 1})
    assert bot.seen == discarded(BRICK, 2) + discarded(WOOD, 2)
    stream_discard(s, seat, [2, 2, 0, 0, 0], [2, 2, 0, 0, 0])
    assert bot.seen == discarded(BRICK, 2) + discarded(WOOD, 2)


def test_a_discard_that_fails_at_its_first_card_is_the_stock_brains_and_shown_from_the_stream():
    def panic(g):
        raise RuntimeError("engine panic")
    s, seat, bot = discard_at(panic)
    assert seat.decide(ask("discard", n=4), 0.0)["fallback"] == "engine" and bot.seen == []
    stream_discard(s, seat, [0, 4, 0, 0, 0], [4, 0, 0, 0, 0])
    assert bot.seen == discarded(BRICK, 4) and seat.fallbacks == Counter({"engine": 1})


def test_discards_the_server_forces_on_our_seat_are_counted_and_shown():
    s, seat, bot = discard_at()
    stream_discard(s, seat, [1, 3, 0, 0, 0], [3, 1, 0, 0, 0], forced=True)  # we were never asked
    assert seat.fallbacks == Counter({"forced-discard": 1})
    assert bot.seen == discarded(WOOD) + discarded(BRICK, 3)


def test_a_discard_forced_before_our_reply_is_counted_and_leaves_nothing_to_swallow_later():
    s, seat, bot = discard_at(lambda g: discard(WOOD))
    seat.decide(ask("discard", n=4), 0.0)  # shown 4 wood while choosing
    stream_discard(s, seat, [0, 4, 0, 0, 0], [4, 0, 0, 0, 0], forced=True)  # the server discarded first
    assert seat.fallbacks == Counter({"forced-discard": 1})
    assert bot.seen == discarded(WOOD, 4) + discarded(BRICK, 4)  # the bot's belief then resets, counted
    s.raw["players"][s.js(0)]["discard"] = 2
    stream_discard(s, seat, [2, 0, 0, 0, 0], [2, 0, 0, 0, 0], forced=True)  # a later forced discard of wood
    assert bot.seen[-2:] == discarded(WOOD, 2) and seat.fallbacks["forced-discard"] == 2


def robot_offer(frm, give, get, to=(True, False, False, False)):
    return {"from": frm, "to": list(to), "give": T.ours_to_js5(give), "get": T.ours_to_js5(get)}


def test_answers_reject_counter_offers_and_unrepresentable_offers_without_asking():
    _, seat, _ = seat_at(current=0, choose=never)  # we are on turn: an offer to us is a counter-offer
    assert seat.decide(ask("answer", **robot_offer(2, [0, 0, 1, 0, 0], [1, 0, 0, 0, 0])), 0.0)["accept"] is False
    assert seat.counts["counters_rejected"] == 1
    s, seat, _ = seat_at(current=2, choose=never, hands={0: [1, 0, 0, 0, 0], 2: [0, 0, 2, 0, 0]})
    assert seat.decide(ask("answer", **robot_offer(2, [0, 0, 2, 0, 0], [1, 0, 0, 0, 0])), 0.0)["accept"] is False
    assert seat.counts["unrepresentable"] == 1
    _, seat, _ = seat_at(current=2, choose=never, cfg=config(False), hands={0: [1, 0, 0, 0, 0], 2: [0, 0, 1, 0, 0]})
    assert seat.decide(ask("answer", **robot_offer(2, [0, 0, 1, 0, 0], [1, 0, 0, 0, 0])), 0.0)["accept"] is False
    assert seat.counts["unrepresentable"] == 1 and seat.counts["offers_to_us"] == 1


def test_answers_import_the_offerers_main_position_with_earlier_seats_rejecting():
    def answer(g):
        assert g.phase == "trade_response" and g.current_actor == 0
        return ACCEPT
    s, seat, bot = seat_at(current=2, choose=answer, hands={0: [1, 0, 0, 0, 0], 2: [0, 0, 1, 0, 0]})
    s.tracker.ctx.offers = 3  # three offers this turn, the last of them this one: earlier ones clamp to 1
    r = seat.decide(ask("answer", **robot_offer(s.js(2), [0, 0, 1, 0, 0], [1, 0, 0, 0, 0])), 0.0)
    assert r["accept"] is True and seat.counts["offers_to_us"] == 1
    assert bot.positions == [("trade_response", 0)]


def test_late_answers_are_sent_as_rejections_and_counted():
    s, seat, _ = seat_at(current=2, choose=lambda g: ACCEPT, hands={0: [1, 0, 0, 0, 0], 2: [0, 0, 1, 0, 0]},
                         clock=lambda: 4.6)
    s.tracker.ctx.offers = 1
    r = seat.decide(ask("answer", **robot_offer(s.js(2), [0, 0, 1, 0, 0], [1, 0, 0, 0, 0])), 0.0)
    assert r["accept"] is False and seat.counts["late"] == 1


def test_our_offer_goes_to_those_we_would_confirm_and_an_unsent_one_is_declined_in_our_engine():
    give, get = [1, 0, 0, 0, 0], [0, 0, 0, 0, 1]
    a = offer(give, get)
    hands = {0: [1, 0, 0, 0, 0], 1: [0, 0, 0, 0, 1], 2: [0, 0, 0, 0, 1], 3: [0, 0, 0, 0, 0]}

    def choose(g):
        if g.phase == "trade_confirm":
            return confirm(2) if confirm(2) in g.legal_actions() else CANCEL
        return a if a in g.legal_actions() else END_TURN
    s, seat, bot = seat_at(hands=hands, choose=choose)
    r = seat.decide(ask("main"), 0.0)
    assert r["do"] == "offer" and r["give"] == T.ours_to_js5(give) and r["get"] == T.ours_to_js5(get)
    assert r["to"] == [i == s.js(2) for i in range(4)]
    assert seat.counts["our_offers"] == 1

    def refuse(g):
        if g.phase == "trade_confirm":
            return CANCEL
        return a if a in g.legal_actions() else END_TURN
    s, seat, bot = seat_at(hands=hands, choose=refuse)
    assert seat.decide(ask("main"), 0.0)["do"] == "end"  # two offers made in our engine only, then the end
    assert seat.counts["our_offers"] == 2
    kinds = [e["type"] for e in bot.seen]
    assert kinds.count("trade_offered") == 2 and "trade_cancelled" in kinds and "trade_confirmed" not in kinds


def test_robot_offers_and_their_caps_are_counted_from_the_stream():
    s, seat, _ = seat_at(current=1)
    rset = "clay=0|ore=0|sheep=0|wheat=0|wood={}|unknown=0"
    js1 = s.js(1)
    s.send(f"f{js1}", "SOCMakeOffer:game=x")
    s.send("all", f"SOCMakeOffer:game=x|from={js1}|to=true,true,true,true|give={rset.format(2)}|get={rset.format(0).replace('ore=0', 'ore=1')}")
    s.send(f"f{js1}", "SOCMakeOffer:game=x")
    s.send("all", f"SOCMakeOffer:game=x|from={js1}|to=true,false,false,true|give={rset.format(1)}|get={rset.format(0).replace('ore=0', 'ore=1')}")
    seat.observe(s.flush())  # both offer steps: the one the second request closed and the one open now
    assert seat.counts["robot_offers"] == 2 and seat.counts["robot_offers_over_caps"] == 2
    assert set(COUNTERS) >= {"robot_offers", "robot_offers_over_caps", "our_offers", "late"}
    assert NO_TRADES == {"max_offers_per_turn": 0}


def test_each_seats_last_build_or_purchase_turn_is_read_from_the_stream():
    # Task 9 reports results with and without games where the robots stopped building (stall-debug.md).
    s, seat, _ = seat_at(current=1)
    assert seat.last_build == [None] * 4
    js1 = s.js(1)
    s.tracker.ctx.turn = 5
    s.send(f"f{js1}", "SOCBuyDevCardRequest:game=x")
    s.raw["players"][js1]["dev"][9] += 1
    s.raw["deck"].remove(9)
    s.send("all", f"SOCDevCardAction:game=x|playerNum={js1}|actionType=DRAW|cardType=9", changed=True)
    seat.observe(s.flush())
    s.steps = []
    assert seat.last_build == [None, 5, None, None]
    s.tracker.ctx.turn = 9
    edge = next(e for e, (a, b) in enumerate(T.EDGE_NODES) if HOME in (a, b) and e != HOME_ROAD)
    js_edge = T.OUR_EDGE_TO_JS_EDGE[edge]
    s.raw["current"] = s.js(0)
    s.send("all", "SOCGameState:game=x|state=20", changed=True)
    s.send(f"f{s.js(0)}", f"SOCPutPiece:game=x|playerNumber={s.js(0)}|pieceType=0|coord={js_edge:x}")
    s.raw["players"][s.js(0)]["roads"].append(js_edge)
    s.send("all", f"SOCPutPiece:game=x|playerNumber={s.js(0)}|pieceType=0|coord={js_edge:x}", changed=True)
    seat.observe(s.flush())
    assert seat.last_build == [9, 5, None, None]
```

Run: `.venv/bin/pytest -q tests/python/test_jsettlers_seat.py`
Expected: FAIL at collection: `No module named 'oracle.jsettlers.arena'`.

- [ ] **Step 2: Write `oracle/jsettlers/arena.py`**

```python
"""An AlphaSettler bot as one seat against three JSettlers2 smart robots in JSettlers' own server
(spec Sections 2 and 5). The bridge brain asks at each decision; `Seat` imports the latest position
in the stream, asks the bot (which sees only its own observation) and answers in JSettlers' terms.
A position or choice the translation cannot express answers `fallback` and the stock brain decides
that hook, counted by reason. Our bot is shown every step's events, redacted for its seat.

Seeds: a seed fixes the board, dice and deck (common random numbers); the candidate takes turn
position k in the seed's k-th game, with bot seed `seed * 4 + k`. Each worker process runs one JVM
and plays its games in turn."""

from __future__ import annotations

import multiprocessing as mp
import os
import time
from collections import Counter
from pathlib import Path

from alphasettler import Bot, Game
from alphasettler.arena import TRADES
from oracle.jsettlers import actions as A
from oracle.jsettlers import missing
from oracle.jsettlers.boards import game_command
from oracle.jsettlers.events import Tracker
from oracle.jsettlers.jvm import Jvm
from oracle.jsettlers.tables import (
    OUR_EDGE_TO_JS_EDGE, OUR_NODE_TO_JS_NODE, OUR_RES_TO_JS, OUR_TILE_TO_JS_HEX, PLAY1, RESOURCE_NAMES, js5_to_ours,
    ours_to_js5,
)
from oracle.jsettlers.translate import Untranslatable, snapshot

BASELINES = ("smart",)
NO_TRADES = {"max_offers_per_turn": 0}
LATE = 4.5  # seconds: robots wait 5 s for answers in robots-only games
SILENCE = 900.0  # seconds without a message before the run fails
MAX_TURNS = 1000
FALLBACK_REASONS = ("position", "declined", "engine", "choice", "preroll-card", "turn-cap", "java-timeout", "victim",
                    "forced-discard")
AFTER_WIN = "after-win"  # a decision JSettlers asks once our rules have ended the game: answered, not counted
COUNTERS = ("our_offers", "our_offers_accepted", "offers_to_us", "offers_to_us_accepted", "unrepresentable",
            "counters_rejected", "late", "robot_offers", "robot_offers_over_caps")
RECORD_KEYS = ("seed", "candidate_seat", "winner", "vp", "turns", "rounds", "aborted", "robot_desync", "last_build_turn",
               "fallbacks", "fallback_reasons", "belief_resets", *COUNTERS, "bank_limit_hits", "arm")
BUILD_STEPS = frozenset({"build", "road_building", "buy"})  # a piece built (setup aside) or a card bought


class Fallback(Exception):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def config(trades: bool) -> dict:
    return {**(TRADES if trades else NO_TRADES), "bank_limit": False}


class Seat:
    def __init__(self, bot, seed: int, config: dict, tracker: Tracker, seat: int, max_turns: int = MAX_TURNS,
                 clock=time.monotonic):
        self.bot, self.seed, self.config, self.tracker, self.seat = bot, seed, config, tracker, seat
        self.max_turns, self.clock = max_turns, clock
        self.fallbacks: Counter = Counter()
        self.counts: Counter = Counter({k: 0 for k in COUNTERS})
        self.capped = False
        # Per seat of ours, the last turn it built a piece or bought a card (None if never): stock robots can stop
        # building at 8-9 VP when nobody holds Longest Road (stall-debug.md), and Task 9 reports those games apart.
        self.last_build: list[int | None] = [None] * 4
        self._declined_seen = 0
        self._virtual, self._virtual_turn = 0, 0  # offers declined inside our engine only, this turn
        self._robot_offers: Counter = Counter()  # (turn, seat) -> offers made
        # Our seat's next discard as the stream should report it: None (none asked), "stock" (the stock brain
        # decides it, unseen by our bot) or the resource names our bot was shown while it chose them.
        self._discard_expect: Counter | str | None = None

    # --- the stream ---

    def observe(self, steps) -> None:
        for st in steps:
            if st.kind in BUILD_STEPS and st.player is not None and st.actions:
                self.last_build[st.player] = self.tracker.ctx.turn
            if st.kind == "offer" and st.player != self.seat:
                self.counts["robot_offers"] += 1
                key = (self.tracker.ctx.turn, st.player)
                self._robot_offers[key] += 1
                i = st.info
                others = sorted(q for q in range(4) if q != st.player)
                within = (sum(i["give"]) == 1 and sum(i["get"]) == 1 and i["give"] != i["get"]
                          and i["to"] == others and self._robot_offers[key] <= 2)
                if not within:
                    self.counts["robot_offers_over_caps"] += 1
            if st.kind == "trade":
                if st.info["offerer"] == self.seat:
                    self.counts["our_offers_accepted"] += 1
                elif st.info["partner"] == self.seat:
                    self.counts["offers_to_us_accepted"] += 1
            self.bot.observe(self.seat, [A.redact(e, self.seat) for e in self._unseen(st.events)])

    def _unseen(self, events: list[dict]) -> list[dict]:
        """`events` less our seat's discards the bot was shown while it chose them (`_discard`). A discard of
        ours that no reply of ours asked for (the server force-discarded, before our reply or with none) is
        counted (`forced-discard`) and shown; what our bot was shown instead then resets its belief, counted.
        Either way the expectation ends with the step that reports our discard."""
        ours = Counter(e["resource"] for e in events if e["type"] == "discarded" and e["player"] == self.seat)
        if not ours:
            return events
        expect, self._discard_expect = self._discard_expect, None
        if expect == ours:
            return [e for e in events if not (e["type"] == "discarded" and e["player"] == self.seat)]
        if expect != "stock":
            self.fallbacks["forced-discard"] += 1
        return events

    def fellback(self, reason: str) -> None:
        self.fallbacks[reason] += 1

    # --- decisions ---

    def decide(self, ask: dict, received: float) -> dict:
        if ask["kind"] == "discard":
            self._discard_expect = "stock"  # unless `_discard` replies with the cards it showed
        if self.tracker.over:
            # Our rules ended the game (a knight's Largest Army reached the target) while JSettlers plays on to
            # the robber move: no decision of ours is owed, so the stock brain moves it, uncounted.
            return {"t": "reply", "id": ask["id"], "fallback": AFTER_WIN}
        try:
            if self.capped or self.tracker.ctx.turn >= self.max_turns:
                raise Fallback("turn-cap")
            reply = getattr(self, "_" + ask["kind"])(ask.get("extra") or {}, received)
        except Fallback as f:
            self.fallbacks[f.reason] += 1
            return {"t": "reply", "id": ask["id"], "fallback": f.reason}
        return {"t": "reply", "id": ask["id"], **reply}

    def _act(self, game: Game) -> int:
        try:
            return self.bot.act(game)
        except BaseException as e:  # noqa: BLE001 - an engine panic (PyO3 PanicException) is a counted fallback
            if isinstance(e, (KeyboardInterrupt, SystemExit)):
                raise
            raise Fallback("engine") from e

    def _snapshot(self) -> dict:
        if self.tracker.ctx.turn != self._virtual_turn:
            self._virtual, self._virtual_turn = 0, self.tracker.ctx.turn
        cap = self.config.get("max_offers_per_turn", 0)
        try:
            snap = snapshot(self.tracker.raw, self.tracker.ctx, cap)
        except Untranslatable:
            raise Fallback("position") from None
        snap["offers_this_turn"] = min(snap["offers_this_turn"] + self._virtual, cap)
        return snap

    def _import(self, snap: dict, *phases: str) -> Game:
        if snap["phase"] not in phases:
            raise Fallback("position")
        try:
            g = Game.from_snapshot(self.seed, snap, self.config)
        except ValueError:
            raise Fallback("position") from None
        if g.current_actor != self.seat:
            raise Fallback("position")
        return g

    def _game(self, *phases: str) -> Game:
        return self._import(self._snapshot(), *phases)

    def _check_declined(self) -> None:
        """A request of ours JSettlers declined since the last decision: the stock brain decides this one,
        so the same refused request is not sent again in a loop."""
        if self.tracker.declined[self.seat] != self._declined_seen:
            self._declined_seen = self.tracker.declined[self.seat]
            raise Fallback("declined")

    def _setup_settlement(self, extra, received) -> dict:
        self._check_declined()
        kind, *v = A.decode(self._act(self._game("setup_settlement")))
        if kind != "settlement":
            raise Fallback("choice")
        return {"node": OUR_NODE_TO_JS_NODE[v[0]]}

    def _setup_road(self, extra, received) -> dict:
        self._check_declined()
        kind, *v = A.decode(self._act(self._game("setup_road")))
        if kind != "road":
            raise Fallback("choice")
        return {"edge": OUR_EDGE_TO_JS_EDGE[v[0]]}

    def _pre_roll(self, extra, received) -> dict:
        self._check_declined()
        a = self._act(self._game("pre_roll"))
        if a == A.ROLL:
            return {"do": "roll"}
        if a == A.KNIGHT:
            return {"do": "knight"}
        raise Fallback("preroll-card")  # the stock turn machine plays only knights before the roll

    def _road_building(self, extra, received) -> dict:
        self._check_declined()
        kind, *v = A.decode(self._act(self._game("road_building")))
        if kind != "road":
            raise Fallback("choice")
        return {"edge": OUR_EDGE_TO_JS_EDGE[v[0]]}

    def _move_robber(self, extra, received) -> dict:
        self._check_declined()
        g = self._game("move_robber")
        a = self._act(g)
        kind, *v = A.decode(a)
        if kind != "move_robber":
            raise Fallback("choice")
        after = g.copy()
        after.apply(a)
        victim = -1
        if after.phase == "steal":
            k2, *w = A.decode(self._act(after))
            if k2 != "steal_from":
                raise Fallback("choice")
            victim = self.tracker.ctx.js(w[0])
        return {"hex": OUR_TILE_TO_JS_HEX[v[0]], "victim": victim}

    def _discard(self, extra, received) -> dict:
        """Our rules discard one card a decision: the bot sees each discard before it chooses the next (its belief
        checks its hand), and the same discards in the stream are not shown to it again. A choice that fails after
        the first card is shown completes with the first discards our rules allow (counted), so the reply is what
        the bot was shown."""
        self._check_declined()
        snap = self._snapshot()
        if snap["phase"] != "discard":
            raise Fallback("position")
        n = int(extra["n"])
        for i, p in enumerate(snap["players"]):
            p["discard_remaining"] = n if i == self.seat else 0
        g = self._import(snap, "discard")
        cards, shown, filling = [0] * 5, Counter(), False
        for i in range(n):
            try:
                if filling:
                    raise Fallback("filling")
                a = self._act(g)
                if A.decode(a)[0] != "discard":
                    raise Fallback("choice")
            except Fallback as f:
                if i == 0:
                    raise
                if not filling:
                    self.fallbacks[f.reason] += 1
                    filling = True
                a = next(x for x in g.legal_actions() if A.decode(x)[0] == "discard")
            r = A.decode(a)[1]
            cards[r] += 1
            seen = len(g.log(self.seat))
            g.apply(a)
            self.bot.observe(self.seat, list(g.log(self.seat))[seen:])
            shown[RESOURCE_NAMES[r]] += 1
        self._discard_expect = shown
        return {"res": ours_to_js5(cards)}

    def _main(self, extra, received) -> dict:
        self._check_declined()
        for _ in range(self.config.get("max_offers_per_turn", 0) + 1):
            g = self._game("main")
            a = self._act(g)
            if A.is_offer(a):
                reply = self._offer(g, a)
                if reply is None:
                    continue  # declined inside our engine: decide again
                return reply
            return self._main_reply(a)
        raise Fallback("choice")  # unreachable: our engine caps the offers a turn

    def _main_reply(self, a: int) -> dict:
        kind, *v = A.decode(a)
        if kind == "end_turn":
            return {"do": "end"}
        if kind == "road":
            return {"do": "build", "piece": 0, "coord": OUR_EDGE_TO_JS_EDGE[v[0]]}
        if kind in ("settlement", "city"):
            return {"do": "build", "piece": 1 if kind == "settlement" else 2, "coord": OUR_NODE_TO_JS_NODE[v[0]]}
        if kind == "buy_dev":
            return {"do": "buy"}
        if kind == "knight":
            return {"do": "knight"}
        if kind == "road_building":
            return {"do": "rb"}
        if kind == "year_of_plenty":
            x, y = A.PAIRS[v[0]]
            return {"do": "yop", "res": [OUR_RES_TO_JS[x], OUR_RES_TO_JS[y]]}
        if kind == "monopoly":
            return {"do": "mono", "res": OUR_RES_TO_JS[v[0]]}
        if kind == "maritime":
            give, get = divmod(v[0], 5)
            gave, got = [0] * 5, [0] * 5
            gave[give], got[get] = self.tracker.rates(self.tracker.raw, self.seat)[give], 1
            return {"do": "bank", "give": ours_to_js5(gave), "get": ours_to_js5(got)}
        raise Fallback("choice")

    def _offer(self, g: Game, a: int) -> dict | None:
        """Our offer goes to the opponents whose lone acceptance our bot would confirm. With none, it is
        made and declined inside our engine only (our bot sees its events), and the bot decides again."""
        self.counts["our_offers"] += 1
        give, get = A.offer_hands(a)
        to = []
        for q in range(4):
            if q == self.seat:
                continue
            h = g.copy()
            h.apply(a)
            ok = True
            while h.phase == "trade_response":
                if h.current_actor == q:
                    if A.ACCEPT not in h.legal_actions():
                        ok = False
                        break
                    h.apply(A.ACCEPT)
                else:
                    h.apply(A.REJECT)
            if ok and h.phase == "trade_confirm" and self._act(h) == A.confirm(q):
                to.append(q)
        if not to:
            h = g.copy()
            h.apply(a)
            while h.phase == "trade_response":
                h.apply(A.REJECT)
            self.bot.observe(self.seat, list(h.log(self.seat)))
            self._virtual += 1
            return None
        js_to = [False] * 4
        for q in to:
            js_to[self.tracker.ctx.js(q)] = True
        return {"do": "offer", "to": js_to, "give": ours_to_js5(give), "get": ours_to_js5(get)}

    def _answer(self, extra, received) -> dict:
        """A robot's offer to us: answered at the offerer's Main position, its earlier offers this turn
        clamped to 1, the offer applied and the seats before ours rejecting."""
        self.counts["offers_to_us"] += 1
        ctx, raw = self.tracker.ctx, self.tracker.raw
        if raw["current"] == ctx.js(self.seat):
            self.counts["counters_rejected"] += 1
            return {"accept": False}
        frm = ctx.ours(int(extra["from"]))
        give, get = js5_to_ours(extra["give"]), js5_to_ours(extra["get"])
        a = A.offer(give, get)
        cap = self.config.get("max_trade_cards", 0) if self.config.get("max_offers_per_turn", 0) else 0
        if (a is None or sum(give) > cap or sum(get) > cap or any(x and y for x, y in zip(give, get))
                or raw["state"] != PLAY1 or frm != ctx.ours(raw["current"])):
            self.counts["unrepresentable"] += 1
            return {"accept": False}
        snap = self._snapshot()
        snap["offers_this_turn"] = min(max(ctx.offers - 1, 0), 1)
        g = self._import_answer(snap, a)
        accept = self._act(g) == A.ACCEPT
        if self.clock() - received >= LATE:
            self.counts["late"] += 1
            return {"accept": False}
        return {"accept": accept}

    def _import_answer(self, snap: dict, a: int) -> Game:
        if snap["phase"] != "main":
            raise Fallback("position")
        try:
            g = Game.from_snapshot(self.seed, snap, self.config)
        except ValueError:
            raise Fallback("position") from None
        if a not in g.legal_actions():
            raise Fallback("position")
        g.apply(a)
        while g.phase == "trade_response" and g.current_actor != self.seat:
            g.apply(A.REJECT)
        if g.phase != "trade_response":
            raise Fallback("position")
        return g


def play_game(jvm: Jvm, seed: int, k: int, candidate: str, nets_dir: str | None = None, trades: bool = True,
              arm: str = "native", native_boards: bool = False, max_turns: int = MAX_TURNS) -> dict:
    """One game: the candidate at turn position k against three smart robots. A broken harness raises."""
    name = f"{arm}-{seed}-{k}"
    cfg = config(trades)
    bot = Bot(candidate, seed * 4 + k, nets_dir)
    jvm.send(game_command(name, seed, k, injected=not native_boards, no_trades=not trades, bridge=True))
    tracker = seat = None
    aborted = False
    while True:
        m = jvm.recv(SILENCE)
        if m.get("g") != name:
            # The bridge of an earlier, aborted game may still ask (or time out) after its "over": answer and drop.
            if m["t"] == "ask":
                jvm.send({"t": "reply", "id": m["id"], "fallback": "turn-cap"})
                continue
            if m["t"] == "fallback":
                continue
            raise RuntimeError(f"a message from another game while playing {name}: {m}")
        t = m["t"]
        if t == "start":
            tracker = Tracker(m)
            seat = Seat(bot, seed, cfg, tracker, tracker.ctx.ours(0), max_turns)
            if seat.seat != k:
                raise RuntimeError(f"{name}: the bridge has turn position {seat.seat}, not {k}")
            bot.observe(seat.seat, [])
            continue
        if t == "ev":
            seat.observe(tracker.feed(m))
        elif t == "ask":
            received = time.monotonic()
            seat.observe(tracker.flush())
            jvm.send(seat.decide(m, received))
        elif t == "fallback":
            seat.fellback(m["reason"])
        elif t == "over":
            seat.observe(tracker.flush())
            ctx = tracker.ctx
            return {
                "seed": seed, "candidate_seat": k,
                "winner": None if m["winner"] < 0 else ctx.ours(m["winner"]),
                "vp": [m["vp"][ctx.js(i)] for i in range(4)],
                "turns": ctx.turn, "rounds": m["rounds"], "aborted": m["aborted"],
                # A setup settlement the server cancelled by a forced end: a stock robot keeps it in its own copy of
                # the game when the turn was its own, and plays the rest of the game on a board it does not have.
                "robot_desync": bool(tracker.setup_cancelled),
                "last_build_turn": list(seat.last_build),
                "fallbacks": sum(seat.fallbacks.values()), "fallback_reasons": dict(seat.fallbacks),
                "belief_resets": bot.diagnostics().get("belief_resets", 0),
                **{c: seat.counts[c] for c in COUNTERS},
                "bank_limit_hits": tracker.bank_limit_hits, "arm": arm,
            }
        else:
            raise RuntimeError(f"{name}: unexpected message {m}")
        if not aborted and tracker.ctx.turn >= max_turns:
            jvm.send({"t": "abort", "g": name})
            aborted = seat.capped = True


def check(candidate: str, baseline: str, seeds: int, seed_start: int = 0, nets_dir: str | None = None) -> None:
    """Raise ValueError for anything `run_match` would reject, without starting a JVM."""
    if baseline not in BASELINES:
        raise ValueError(f"unknown JSettlers robot {baseline!r}; known: {list(BASELINES)}")
    Bot(candidate, 0, nets_dir)  # raises ValueError for a bad name or a missing or invalid net
    if seeds <= 0:
        raise ValueError("seeds must be positive")
    if seed_start < 0 or (seed_start + seeds) * 4 > 2**64:
        raise ValueError(f"seeds {seed_start}..{seed_start + seeds} are outside the bot seed range")
    why = missing()
    if why is not None:
        raise ValueError(f"the JSettlers harness is not ready: {why}")


def arm_name(trades: bool, aligned: bool, native_boards: bool) -> str:
    if native_boards:
        return "native-boards"
    if not trades:
        return "no-trades"
    return "aligned" if aligned else "native"


def _chunk(jobs, candidate, nets_dir, trades, aligned, native_boards, max_turns, log_dir) -> list[dict]:
    arm = arm_name(trades, aligned, native_boards)
    log = None if log_dir is None else Path(log_dir) / f"{arm}-{os.getpid()}.log"
    with Jvm("aligned" if aligned else "stock", log=log) as jvm:
        return [play_game(jvm, s, k, candidate, nets_dir, trades, arm, native_boards, max_turns) for s, k in jobs]


def run_match(candidate: str, baseline: str, seeds: int, seed_start: int = 0, workers: int | None = None,
              nets_dir: str | None = None, trades: bool = True, aligned: bool = False, native_boards: bool = False,
              max_turns: int = MAX_TURNS, log_dir: str | Path | None = None) -> list[dict]:
    """Four games a seed (the candidate in each turn position), one JVM per worker process. Each JVM's log
    goes under `log_dir` when given (else runs/jsettlers/logs/)."""
    if native_boards and not trades:
        raise ValueError("native boards need trades (that check is the native arm on JSettlers' own boards)")
    check(candidate, baseline, seeds, seed_start, nets_dir)
    jobs = [(s, k) for s in range(seed_start, seed_start + seeds) for k in range(4)]
    n = max(1, min(workers or os.cpu_count() or 1, len(jobs)))
    chunks = [jobs[i::n] for i in range(n)]
    args = [(c, candidate, nets_dir, trades, aligned, native_boards, max_turns, log_dir) for c in chunks]
    with mp.get_context("spawn").Pool(n) as pool:
        out = pool.starmap(_chunk, args)
    return sorted((r for chunk in out for r in chunk), key=lambda r: (r["seed"], r["candidate_seat"]))
```
- [ ] **Step 3: Run the Seat tests to verify they pass**

Run: `.venv/bin/pytest -q tests/python/test_jsettlers_seat.py`
Expected: PASS, 23 tests.

- [ ] **Step 4: Write the failing JVM tests**

Create `tests/python/test_jsettlers_arena.py`:
```python
import json
from pathlib import Path

import pytest

from jsettlers_harness import require
from oracle.jsettlers import diff
from oracle.jsettlers import tables as T
from oracle.jsettlers.arena import FALLBACK_REASONS, RECORD_KEYS, check, play_game, run_match
from oracle.jsettlers.boards import first_player, js_layout, native_board
from oracle.jsettlers.jvm import Jvm
from test_jsettlers_events import Script


@pytest.fixture(scope="module")
def jvm(tmp_path_factory):
    require()
    with Jvm(log=tmp_path_factory.mktemp("jvm") / "jvm.log") as j:  # suite runs leave no logs under runs/
        yield j


def test_a_no_trades_bridge_game_records_every_counter(jvm):
    r = play_game(jvm, 51, 1, "ismcts@20", trades=False, arm="no-trades")
    assert tuple(r) == RECORD_KEYS
    assert r["candidate_seat"] == 1 and r["arm"] == "no-trades" and r["aborted"] is False
    assert r["winner"] in (0, 1, 2, 3) and r["vp"][r["winner"]] >= 10
    assert r["turns"] > 10 and r["belief_resets"] == 0
    assert set(r["fallback_reasons"]) <= set(FALLBACK_REASONS)
    assert r["fallbacks"] == sum(r["fallback_reasons"].values()) and r["fallbacks"] < 20
    assert r["our_offers"] == r["offers_to_us"] == r["robot_offers"] == 0
    assert len(r["last_build_turn"]) == 4 and r["last_build_turn"][1] is not None and r["robot_desync"] is False
    assert all(t is None or 0 <= t <= r["turns"] for t in r["last_build_turn"])


def test_a_game_at_the_turn_cap_ends_with_no_winner_and_the_jvm_plays_on(jvm):
    r = play_game(jvm, 52, 0, "ismcts@20", trades=False, arm="no-trades", max_turns=3)
    assert r["aborted"] is True and r["winner"] is None and r["turns"] >= 3
    # The aborted game's bridge may still ask after its "over"; the next game on this JVM answers it.
    r = play_game(jvm, 52, 1, "ismcts@20", trades=False, arm="no-trades", max_turns=3)
    assert r["aborted"] is True and r["candidate_seat"] == 1


class FakeJvm:
    def __init__(self, messages):
        self.messages, self.sent = list(messages), []

    def send(self, m):
        self.sent.append(m)

    def recv(self, timeout):
        return self.messages.pop(0)


def start_message(name, seed, k):
    """A "start" line as the harness writes it, for seed's injected board with our seat at turn position k."""
    s = Script(first=first_player(k), state=T.START1A, current=first_player(k))
    board, lay = native_board(seed), js_layout(native_board(seed))
    hexes = [[h, lay["hexes"][T.HEX_NUM[h]], max(0, lay["numbers"][T.HEX_NUM[h]])] for h in T.OUR_TILE_TO_JS_HEX]
    ports = [[e, 0 if kind == "generic" else T.OUR_RES_TO_JS[T.RESOURCE_NAMES.index(kind)]]
             for e, (_, kind) in zip(T.PORT_EDGES, board["ports"])]
    return {"t": "start", "g": name, "first": first_player(k), "board": {"hexes": hexes, "ports": ports}, "s": s.raw}


def test_late_messages_of_an_earlier_aborted_game_are_answered_and_dropped():
    name = "no-trades-52-2"
    over = {"t": "over", "g": name, "winner": -1, "vp": [2, 2, 2, 2], "rounds": 0, "aborted": True}
    j = FakeJvm([{"t": "ask", "id": 41, "g": "no-trades-52-1", "kind": "main", "state": 20, "extra": {}},
                 {"t": "fallback", "g": "no-trades-52-1", "reason": "java-timeout"},
                 start_message(name, 52, 2), over])
    r = play_game(j, 52, 2, "random", trades=False, arm="no-trades")
    assert j.sent[0]["t"] == "game" and j.sent[1] == {"t": "reply", "id": 41, "fallback": "turn-cap"}
    assert r["fallbacks"] == 0 and r["aborted"] is True and r["last_build_turn"] == [None] * 4
    assert r["robot_desync"] is False
    j = FakeJvm([{"t": "ev", "g": "elsewhere", "a": "all", "m": "SOCGameState:game=elsewhere|state=20"}])
    with pytest.raises(RuntimeError, match="a message from another game"):
        play_game(j, 52, 3, "random", trades=False, arm="no-trades")


SEED78 = Path(__file__).parent / "data" / "jsettlers_seed78_k1.jsonl"  # the harness's lines, bridge game y-78-1


def seed78(name=None):
    msgs = [json.loads(line) for line in SEED78.read_text().splitlines()]
    return msgs if name is None else [{**m, "g": name} for m in msgs if m["t"] in ("start", "ev", "over")]


def test_a_robots_setup_settlement_cancelled_by_a_forced_end_is_a_desync_and_one_belief_reset():
    # Seed 78, k=1 (stall-debug.md): robot 3's planInitRoad returns an edge off the board, the server ends its setup
    # turn and cancels its settlement, and the robot keeps it in its own copy. Our bot is shown no event for the
    # cancelled settlement (no engine event removes one), so its next searched decision resets its belief once.
    r = play_game(FakeJvm(seed78()), 78, 1, "ismcts@20", trades=False, arm="y")
    assert r["robot_desync"] is True and r["belief_resets"] == 1 and r["fallbacks"] == 0
    assert r["last_build_turn"][0] is None  # the desynced robot (our seat 0 here) never builds again


def test_the_seed_78_stream_replays_with_the_forced_setup_end_allowlisted():
    res = diff.play_game(FakeJvm(seed78("diff-78-1")), 78, 1)
    assert res.mismatch is None and res.allowlisted["forced-setup-end"] == 1 and res.steps > 30


def test_seed_78_desyncs_a_robot_in_the_live_harness(jvm):
    r = play_game(jvm, 78, 1, "ismcts@20", trades=False, arm="no-trades", max_turns=1)  # one forced end: about 60 s
    assert r["robot_desync"] is True and r["aborted"] is True


def test_run_match_rotates_the_seat_on_one_board(tmp_path):
    require()
    rs = run_match("ismcts@20", "smart", 1, seed_start=53, workers=2, trades=False, log_dir=tmp_path)
    assert [r["candidate_seat"] for r in rs] == [0, 1, 2, 3] and {r["seed"] for r in rs} == {53}
    assert all(r["belief_resets"] == 0 for r in rs)
    assert len(list(tmp_path.glob("no-trades-*.log"))) == 2


@pytest.mark.parametrize("args,match", [
    (("ismcts@20", "fast", 1), "unknown JSettlers robot"),
    (("no-such-bot", "smart", 1), "unknown bot"),
    (("ismcts@20", "smart", 0), "seeds must be positive"),
])
def test_check_rejects_bad_requests_before_any_game(args, match):
    with pytest.raises(ValueError, match=match):
        check(*args)


def test_native_boards_without_trades_are_refused_before_any_game():
    with pytest.raises(ValueError, match="native boards need trades"):
        run_match("ismcts@20", "smart", 1, trades=False, native_boards=True)
```

Run: `.venv/bin/pytest -q tests/python/test_jsettlers_arena.py`
Expected: FAIL: the bridge game never starts (`no robot named robot bridge is connected`, raised as `JvmError` naming the log), and the no-JVM cases (`check`, the native-boards refusal, the late messages of an aborted game) pass.

- [ ] **Step 5: Write the bridge**

Each file starts with the five-line header of Task 1 Step 1.

`jsettlers/src/alphasettler/jsettlers/BridgeClient.java`:
```java
package alphasettler.jsettlers;

import soc.baseclient.ServerConnectInfo;
import soc.game.SOCGame;
import soc.message.SOCMessage;
import soc.robot.SOCRobotBrain;
import soc.robot.SOCRobotClient;
import soc.util.CappedQueue;
import soc.util.SOCRobotParameters;

/**
 * Our seat: a robot connection (so games stay robots-only and run at the fast pace) named
 * "robot bridge" (so its fallback decisions use the smart parameters), with a third-party brain
 * class (so the long jsettlers.bots.timeout.turn applies to its turns).
 */
final class BridgeClient extends SOCRobotClient {
    private final Pipe pipe;

    BridgeClient(ServerConnectInfo sci, Pipe pipe) {
        super(sci, ArenaServer.BRIDGE, "pw");
        this.pipe = pipe;
        rbclass = BridgeClient.class.getName();
    }

    @Override
    public SOCRobotBrain createBrain(SOCRobotParameters params, SOCGame ga, CappedQueue<SOCMessage> mq) {
        return new BridgeBrain(this, params, ga, mq, pipe);
    }
}
```

`jsettlers/src/alphasettler/jsettlers/BridgeDM.java`:
```java
package alphasettler.jsettlers;

import soc.game.SOCResourceSet;
import soc.robot.SOCRobotBrain;
import soc.robot.SOCRobotDM;

/** The stock decision maker, plus our Year of Plenty picks (sent by the stock WAITING_FOR_DISCOVERY step). */
final class BridgeDM extends SOCRobotDM {
    BridgeDM(SOCRobotBrain br) {
        super(br);
    }

    void choose(int a, int b) {
        resourceChoices = new SOCResourceSet();
        resourceChoices.add(1, a);
        resourceChoices.add(1, b);
    }
}
```

`jsettlers/src/alphasettler/jsettlers/BridgeOpening.java`:
```java
package alphasettler.jsettlers;

import java.util.Map;
import soc.game.SOCGame;
import soc.game.SOCPlayer;
import soc.robot.OpeningBuildStrategy;

/**
 * Initial placement asked of Python; the stock strategy decides when Python falls back. Our first
 * settlement is kept in the stock field, so a stock second settlement plans around it.
 */
final class BridgeOpening extends OpeningBuildStrategy {
    private final BridgeBrain brain;

    BridgeOpening(SOCGame ga, SOCPlayer pl, BridgeBrain br) {
        super(ga, pl, br);
        brain = br;
    }

    @Override
    public int planInitialSettlements() {
        Map<String, Object> r = brain.ask("setup_settlement", null);
        if (r == null) return super.planInitialSettlements();
        firstSettlement = Json.i(r, "node");
        return firstSettlement;
    }

    @Override
    public int planSecondSettlement() {
        Map<String, Object> r = brain.ask("setup_settlement", null);
        return r != null ? Json.i(r, "node") : super.planSecondSettlement();
    }

    @Override
    public int planInitRoad() {
        Map<String, Object> r = brain.ask("setup_road", null);
        return r != null ? Json.i(r, "edge") : super.planInitRoad();
    }
}
```

`jsettlers/src/alphasettler/jsettlers/BridgeRobber.java`:
```java
package alphasettler.jsettlers;

import java.util.Random;
import soc.game.SOCGame;
import soc.game.SOCPlayer;
import soc.robot.RobberStrategy;

/** The victim our bot chose with its robber move, when JSettlers asks for one. */
final class BridgeRobber extends RobberStrategy {
    private final BridgeBrain brain;

    BridgeRobber(SOCGame ga, SOCPlayer pl, BridgeBrain br, Random rand) {
        super(ga, pl, br, rand);
        brain = br;
    }

    @Override
    public int chooseRobberVictim(boolean[] isVictim, boolean canChooseNone) {
        int v = brain.takeVictim();
        if (v >= 0 && v < isVictim.length && isVictim[v]) return v;
        if (v != BridgeBrain.STOCK_ROBBER) brain.fellBack("victim");
        return super.chooseRobberVictim(isVictim, canChooseNone);
    }
}
```

`jsettlers/src/alphasettler/jsettlers/BridgeBrain.java`:
```java
package alphasettler.jsettlers;

import java.util.Map;
import soc.game.SOCCity;
import soc.game.SOCDevCardConstants;
import soc.game.SOCGame;
import soc.game.SOCPlayingPiece;
import soc.game.SOCResourceSet;
import soc.game.SOCRoad;
import soc.game.SOCSettlement;
import soc.message.SOCMessage;
import soc.robot.SOCRobotBrain;
import soc.robot.SOCRobotClient;
import soc.robot.SOCRobotDM;
import soc.util.CappedQueue;
import soc.util.SOCRobotParameters;

/**
 * The stock robot brain with its decision hooks asked of Python: initial placement, roll or knight
 * before the roll, the main-phase choice, free roads, the robber and its victim, discards, and
 * (Task 7) offers and answers. The stock turn machine decides when to ask; a null answer (Python
 * fell back, or no reply in time) lets the stock brain decide that hook. Each reply sets the flags
 * the stock code path it replaces sets, so the stock loop waits for the same server answers.
 */
final class BridgeBrain extends SOCRobotBrain {
    /** Below the server's jsettlers.bots.timeout.turn (600 s), so a slow reply is our counted java-timeout first. */
    static final long ASK_SECONDS = 540;
    static final int STOCK_ROBBER = -2;
    private final Pipe pipe;
    private volatile int victim = STOCK_ROBBER;

    BridgeBrain(SOCRobotClient rc, SOCRobotParameters params, SOCGame ga, CappedQueue<SOCMessage> mq, Pipe pipe) {
        super(rc, params, ga, mq);
        this.pipe = pipe;
    }

    /** Python's reply to a decision of {@code kind}, or null when the stock brain decides it. */
    Map<String, Object> ask(String kind, Map<String, Object> extra) {
        Map<String, Object> q = Json.map("t", "ask", "g", game.getName(), "kind", kind, "state", game.getGameState(),
            "extra", extra == null ? Json.map() : extra);
        Map<String, Object> r = pipe.ask(q, ASK_SECONDS);
        if (r == null) {
            fellBack("java-timeout");
            return null;
        }
        return r.containsKey("fallback") ? null : r;
    }

    void fellBack(String reason) {
        pipe.send(Json.map("t", "fallback", "g", game.getName(), "reason", reason));
    }

    int takeVictim() {
        int v = victim;
        victim = STOCK_ROBBER;
        return v;
    }

    static SOCResourceSet set(int[] js5) {
        return new SOCResourceSet(js5[0], js5[1], js5[2], js5[3], js5[4], 0);
    }

    @Override
    protected void setStrategyFields() {
        super.setStrategyFields();
        openingBuildStrategy = new BridgeOpening(game, ourPlayerData, this);
        robberStrategy = new BridgeRobber(game, ourPlayerData, this, rand);
    }

    @Override
    protected SOCRobotDM createDM() {
        return new BridgeDM(this);
    }

    /** Replaces the stock knight-or-roll choice under the stock guard; flags as in playKnightCard / rollDice. */
    @Override
    protected void rollOrPlayKnightOrExpectDice() {
        if (ourTurn && !waitingForOurTurn && !expectPLAY1 && !expectDISCARD && !expectPLACING_ROBBER
            && !(expectDICERESULT && counter < 4000)) {
            Map<String, Object> r = ask("pre_roll", null);
            if (r != null) {
                expectROLL_OR_CARD = false;
                if ("knight".equals(r.get("do"))) {
                    expectPLACING_ROBBER = true;
                    waitingForGameState = true;
                    counter = 0;
                    client.playDevCard(game, SOCDevCardConstants.KNIGHT);
                } else {
                    expectDICERESULT = true;
                    counter = 0;
                    client.rollDice(game);
                }
                return;
            }
        }
        super.rollOrPlayKnightOrExpectDice();
    }

    @Override
    protected void planAndDoActionForPLAY1() {
        Map<String, Object> r = ask("main", null);
        if (r == null) {
            super.planAndDoActionForPLAY1();
            return;
        }
        main(r);
    }

    /** One main-phase reply, with the flags of the stock path it replaces. Task 7 adds "offer". */
    void main(Map<String, Object> r) {
        switch (Json.str(r, "do")) {
            case "end" -> {  // as planAndDoActionForPLAY1's end of turn
                resetFieldsAtEndTurn();
                client.endTurn(game);
            }
            case "build" -> build(Json.i(r, "piece"), Json.i(r, "coord"));
            case "buy" -> {  // as buildRequestPlannedPiece's CARD
                waitingForDevCard = true;
                client.buyDevCard(game);
            }
            case "knight" -> play(SOCDevCardConstants.KNIGHT);
            case "rb" -> play(SOCDevCardConstants.ROADS);
            case "yop" -> {
                int[] res = Json.intArray(r, "res");
                ((BridgeDM) decisionMaker).choose(res[0], res[1]);
                play(SOCDevCardConstants.DISC);
            }
            case "mono" -> {
                monopolyStrategy.setMonopolyChoice(Json.i(r, "res"));
                play(SOCDevCardConstants.MONO);
            }
            case "bank" -> {  // as buildOrGetResourceByTradeOrCard's tradeWithBank
                waitingForTradeMsg = true;
                counter = 0;
                client.bankTrade(game, set(Json.intArray(r, "give")), set(Json.intArray(r, "get")));
            }
            default -> pipe.fatal("unknown main reply " + r);
        }
    }

    /** As buildOrGetResourceByTradeOrCard's card plays (playKnightCard for a knight). */
    private void play(int card) {
        switch (card) {
            case SOCDevCardConstants.KNIGHT -> expectPLACING_ROBBER = true;
            case SOCDevCardConstants.ROADS -> expectPLACING_FREE_ROAD1 = true;
            case SOCDevCardConstants.DISC -> expectWAITING_FOR_DISCOVERY = true;
            case SOCDevCardConstants.MONO -> expectWAITING_FOR_MONOPOLY = true;
            default -> pipe.fatal("cannot play dev card " + card);
        }
        waitingForGameState = true;
        counter = 0;
        client.playDevCard(game, card);
    }

    /** As buildRequestPlannedPiece: placeIfExpectPlacing then puts whatWeWantToBuild. */
    private void build(int piece, int coord) {
        waitingForGameState = true;
        counter = 0;
        switch (piece) {
            case SOCPlayingPiece.ROAD -> {
                expectPLACING_ROAD = true;
                whatWeWantToBuild = new SOCRoad(ourPlayerData, coord, null);
            }
            case SOCPlayingPiece.SETTLEMENT -> {
                expectPLACING_SETTLEMENT = true;
                whatWeWantToBuild = new SOCSettlement(ourPlayerData, coord, null);
            }
            case SOCPlayingPiece.CITY -> {
                expectPLACING_CITY = true;
                whatWeWantToBuild = new SOCCity(ourPlayerData, coord, null);
            }
            default -> pipe.fatal("unknown piece " + piece);
        }
        client.buildRequest(game, piece);
    }

    /**
     * Free roads are asked of Python. The stock placement takes its roads from a building plan the
     * bridge never made, so a fallback places the first road JSettlers allows.
     */
    @Override
    protected void placeIfExpectPlacing() {
        int gs = game.getGameState();
        if (!waitingForGameState && ourTurn && !waitingForOurTurn
            && (gs == SOCGame.PLACING_FREE_ROAD1 || gs == SOCGame.PLACING_FREE_ROAD2)
            && (expectPLACING_FREE_ROAD1 || expectPLACING_FREE_ROAD2)) {
            Map<String, Object> r = ask("road_building", null);
            int edge = r != null ? Json.i(r, "edge") : anyPotentialRoad();
            expectPLACING_FREE_ROAD1 = false;
            expectPLACING_FREE_ROAD2 = gs == SOCGame.PLACING_FREE_ROAD1;
            if (gs == SOCGame.PLACING_FREE_ROAD2) expectPLAY1 = true;
            waitingForGameState = true;
            counter = 0;
            client.putPiece(game, new SOCRoad(ourPlayerData, edge, null));
            return;
        }
        super.placeIfExpectPlacing();
    }

    private int anyPotentialRoad() {
        for (int e = 0x22; e <= 0xCC; e++) if (ourPlayerData.isPotentialRoad(e)) return e;
        pipe.fatal("no road can be placed for Road Building in " + game.getName());
        return -1;
    }

    @Override
    protected void moveRobber() {
        Map<String, Object> r = ask("move_robber", null);
        if (r == null) {
            victim = STOCK_ROBBER;
            super.moveRobber();
            return;
        }
        victim = Json.i(r, "victim");
        client.moveRobber(game, ourPlayerData, Json.i(r, "hex"));
    }

    @Override
    protected void discard(int numDiscards) {
        Map<String, Object> r = ask("discard", Json.map("n", numDiscards));
        if (r == null) {
            super.discard(numDiscards);
            return;
        }
        client.discard(game, set(Json.intArray(r, "res")));
    }
}
```

In `ArenaMain.run`, after `SOCServer.ROBOT_FORCE_ENDTURN_SECONDS = 60;`, add (a stubborn robot's turn, one force-ended before, gets the same 60 s, so our discards on it get the same window):
```java
        SOCServer.ROBOT_FORCE_ENDTURN_STUBBORN_SECONDS = 60;
```

In `ArenaMain.run`, after the loop that starts the four robots, add:
```java
        start(new BridgeClient(sci, pipe), srv);
        names.add(ArenaServer.BRIDGE);
```

Add these lines to `jsettlers/api.txt` (the brain fields the bridge writes, the stock opening's first settlement, so a stock second settlement plans around ours, and the stubborn robots' force-end time) and re-run `bash jsettlers/check-api.sh` (expected: `57 members present`):
```
soc.robot.SOCRobotBrain | protected boolean expectROLL_OR_CARD;
soc.robot.SOCRobotBrain | protected boolean expectDISCARD;
soc.robot.SOCRobotBrain | protected soc.game.SOCPlayingPiece whatWeWantToBuild;
soc.robot.SOCRobotBrain | protected boolean waitingForTradeResponse;
soc.robot.SOCRobotBrain | protected int tradeResponseTimeoutSec;
soc.robot.SOCRobotBrain | protected boolean[] offerRejections;
soc.robot.SOCRobotBrain | protected boolean doneTrading;
soc.robot.OpeningBuildStrategy | protected int firstSettlement;
soc.server.SOCServer | public static int ROBOT_FORCE_ENDTURN_STUBBORN_SECONDS;
```
- [ ] **Step 6: Build and run the tests to verify they pass**

Run: `pgrep -fl 'arm.sh' || echo idle`, then `bash jsettlers/build.sh && .venv/bin/pytest -q tests/python/test_jsettlers_seat.py tests/python/test_jsettlers_arena.py`
Expected: `built 15 sources`; PASS, 23 + 11 tests. If the bridge game stalls (a `TimeoutError` naming the log), the log's last brain lines say which flag the stock loop was waiting on: compare the reply's flags with the stock code path it replaces (Step 5's comments name each), fix the bridge, and keep the stall's reason in a comment.

- [ ] **Step 7: Format, lint, the full suite**

Run the Global Constraints sequence. Expected: all pass, nothing skipped.

- [ ] **Step 8: Commit**

```bash
git add jsettlers/src/alphasettler/jsettlers/BridgeClient.java jsettlers/src/alphasettler/jsettlers/BridgeBrain.java \
  jsettlers/src/alphasettler/jsettlers/BridgeOpening.java jsettlers/src/alphasettler/jsettlers/BridgeRobber.java \
  jsettlers/src/alphasettler/jsettlers/BridgeDM.java jsettlers/src/alphasettler/jsettlers/ArenaMain.java jsettlers/api.txt \
  oracle/jsettlers/arena.py tests/python/test_jsettlers_seat.py tests/python/test_jsettlers_arena.py
git diff --cached | grep -nE 'sk-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{36}|password=[^[:space:]]+'
git -c user.email=sean.may101@gmail.com commit -m "jsettlers: the bridge seat and the arena driver (no domestic trades yet)"
```

---

### Task 7: Domestic trades across the bridge, and the capped robots of the aligned arm

**Files:**
- Create: `jsettlers/src/alphasettler/jsettlers/CappedClient.java`, `jsettlers/src/alphasettler/jsettlers/CappedBrain.java`
- Modify: `jsettlers/src/alphasettler/jsettlers/BridgeBrain.java` (`considerOffer`, `makeOffer`, `makeCounterOffer`, the `offer` reply), `jsettlers/src/alphasettler/jsettlers/ArenaMain.java` (mode `aligned`), `jsettlers/api.txt`
- Test: `tests/python/test_jsettlers_arena.py` (two tests added)

**Interfaces:**
- Consumes: Task 6's `BridgeBrain.ask`, `BridgeBrain.main`, `Seat._answer` and `Seat._offer` (already written and unit-tested in Task 6), the stock `handleMAKEOFFER` / `handleTradeResponse` / `tradeStopWaitingClearOffer` machinery, `SOCRobotNegotiator.makeOffer(SOCBuildPlan)`.
- Produces: the ask `{"kind": "answer", "extra": {"from": JS_SEAT, "to": [4 bools], "give": js5, "get": js5}}` (the offerer gives `give` and wants `get`) with reply `{"accept": BOOL}`; the main reply `{"do": "offer", "to": [4 bools by JS seat], "give": js5, "get": js5}`; `Jvm("aligned")` (four `CappedClient` robots named `robot 1`..`robot 4`).

- [ ] **Step 1: Write the failing tests**

Append to `tests/python/test_jsettlers_arena.py`:
```python
def test_a_native_trading_game_answers_and_counts_the_robots_offers(jvm):
    r = play_game(jvm, 61, 2, "ismcts@20+t", trades=True, arm="native")
    assert r["belief_resets"] == 0 and r["arm"] == "native"
    assert r["robot_offers"] > 0, "stock smart robots offer in a full trading game"
    assert r["offers_to_us"] >= r["unrepresentable"] + r["counters_rejected"] + r["late"]
    assert r["our_offers_accepted"] <= r["our_offers"] and r["offers_to_us_accepted"] <= r["offers_to_us"]
    assert set(r["fallback_reasons"]) <= set(FALLBACK_REASONS)


def test_aligned_robots_offer_only_within_our_trading_game():
    require()
    with Jvm("aligned") as j:
        assert j.ready["mode"] == "aligned"
        r = play_game(j, 62, 0, "ismcts@20+t", trades=True, arm="aligned")
    assert r["robot_offers_over_caps"] == 0 and r["belief_resets"] == 0 and r["arm"] == "aligned"
```

Run: `.venv/bin/pytest -q tests/python/test_jsettlers_arena.py -k "trading or aligned"`
Expected: FAIL: the native game's `robot_offers` is 0 or its offers are answered by the stock negotiator (no `answer` asks reach Python, so `offers_to_us` stays 0 while robots offer to us), and `Jvm("aligned")` fails with `unknown mode aligned (stock)` named in a `JvmError`.

- [ ] **Step 2: Trades in `BridgeBrain`**

Add imports `java.util.ArrayList`, `java.util.List`, `java.util.concurrent.ConcurrentHashMap`, `soc.game.SOCPlayer`, `soc.game.SOCResourceConstants`, `soc.game.SOCTradeOffer`, `soc.robot.SOCBuildPlan`, `soc.robot.SOCRobotNegotiator`, and add to the class:
```java
    /** Answers by offer (offerer, sets, offer time): the stock brain may consider one offer twice. */
    private final Map<String, Integer> answers = new ConcurrentHashMap<>();

    static int[] js5(SOCResourceSet s) {
        return new int[] {s.getAmount(SOCResourceConstants.CLAY), s.getAmount(SOCResourceConstants.ORE),
            s.getAmount(SOCResourceConstants.SHEEP), s.getAmount(SOCResourceConstants.WHEAT),
            s.getAmount(SOCResourceConstants.WOOD)};
    }

    @Override
    protected int considerOffer(SOCTradeOffer offer) {
        if (!offer.getTo()[getOurPlayerNumber()]) return SOCRobotNegotiator.IGNORE_OFFER;
        SOCPlayer from = game.getPlayer(offer.getFrom());
        String key = offer.getFrom() + "|" + offer.getGiveSet() + "|" + offer.getGetSet() + "|" + from.getCurrentOfferTime();
        Integer known = answers.get(key);
        if (known != null) return known;
        List<Object> toList = new ArrayList<>();
        for (boolean b : offer.getTo()) toList.add(b);
        Map<String, Object> r = ask("answer", Json.map("from", offer.getFrom(), "to", toList,
            "give", js5(offer.getGiveSet()), "get", js5(offer.getGetSet())));
        int answer = r == null ? super.considerOffer(offer)
            : (Json.bool(r, "accept") ? SOCRobotNegotiator.ACCEPT_OFFER : SOCRobotNegotiator.REJECT_OFFER);
        answers.put(key, answer);
        return answer;
    }

    /** Our seat offers only what our bot chose (the "offer" reply); the stock negotiator never offers for it. */
    @Override
    protected boolean makeOffer(SOCBuildPlan buildPlan) {
        doneTrading = true;
        return false;
    }

    /** Our rules have no counter-offers: the stock brain then rejects. */
    @Override
    protected boolean makeCounterOffer(SOCTradeOffer offer) {
        return false;
    }

    @Override
    protected void resetFieldsAtEndTurn() {
        super.resetFieldsAtEndTurn();
        answers.clear();
    }
```
In `BridgeBrain.main`, add before `default`:
```java
            case "offer" -> {
                SOCTradeOffer offer = new SOCTradeOffer(game.getName(), getOurPlayerNumber(), Json.boolArray(r, "to"),
                    set(Json.intArray(r, "give")), set(Json.intArray(r, "get")));
                ourPlayerData.setCurrentOffer(offer);
                for (int pn = 0; pn < game.maxPlayers; pn++) offerRejections[pn] = false;
                waitingForTradeResponse = true;
                tradeResponseTimeoutSec = TRADE_RESPONSE_TIMEOUT_SEC_BOTS_ONLY;
                counter = 0;
                client.offerTrade(game, offer);
            }
```
The stock loop then waits for the answers (an acceptance executes the trade; all rejections, or 5 s, clear the offer) and calls `planAndDoActionForPLAY1` again, which asks Python for the next main decision.

- [ ] **Step 3: The capped robots**

`jsettlers/src/alphasettler/jsettlers/CappedBrain.java` (header first):
```java
package alphasettler.jsettlers;

import soc.game.SOCGame;
import soc.game.SOCTradeOffer;
import soc.message.SOCMessage;
import soc.robot.SOCBuildPlan;
import soc.robot.SOCRobotBrain;
import soc.robot.SOCRobotClient;
import soc.util.CappedQueue;
import soc.util.SOCRobotParameters;

/**
 * A stock smart robot whose offers fit our trading game (the aligned arm, spec Section 5): it sends
 * the stock negotiator's offer only when it is one card a side, sends it to every opponent, at most
 * two a turn, and never counter-offers. When the negotiator's offer does not fit, it stops offering
 * for the turn. Everything else is the stock brain.
 */
final class CappedBrain extends SOCRobotBrain {
    static final int MAX_OFFERS = 2;
    private int offersThisTurn;

    CappedBrain(SOCRobotClient rc, SOCRobotParameters params, SOCGame ga, CappedQueue<SOCMessage> mq) {
        super(rc, params, ga, mq);
    }

    @Override
    protected boolean makeOffer(SOCBuildPlan buildPlan) {
        SOCTradeOffer o = offersThisTurn < MAX_OFFERS ? negotiator.makeOffer(buildPlan) : null;
        negotiator.resetWantsAnotherOffer();
        if (o == null || o.getGiveSet().getTotal() != 1 || o.getGetSet().getTotal() != 1) {
            ourPlayerData.setCurrentOffer(null);
            doneTrading = true;
            waitingForTradeResponse = false;
            return false;
        }
        boolean[] to = new boolean[game.maxPlayers];
        for (int pn = 0; pn < game.maxPlayers; pn++) {
            to[pn] = pn != getOurPlayerNumber() && !game.isSeatVacant(pn);
            offerRejections[pn] = false;
        }
        SOCTradeOffer capped = new SOCTradeOffer(game.getName(), getOurPlayerNumber(), to, o.getGiveSet(), o.getGetSet());
        ourPlayerData.setCurrentOffer(capped);
        waitingForTradeResponse = true;
        tradeResponseTimeoutSec = TRADE_RESPONSE_TIMEOUT_SEC_BOTS_ONLY;
        counter = 0;
        offersThisTurn++;
        client.offerTrade(game, capped);
        return true;
    }

    @Override
    protected boolean makeCounterOffer(SOCTradeOffer offer) {
        return false;
    }

    @Override
    protected void resetFieldsAtEndTurn() {
        super.resetFieldsAtEndTurn();
        offersThisTurn = 0;
    }
}
```

`jsettlers/src/alphasettler/jsettlers/CappedClient.java` (header first):
```java
package alphasettler.jsettlers;

import soc.baseclient.ServerConnectInfo;
import soc.game.SOCGame;
import soc.message.SOCMessage;
import soc.robot.SOCRobotBrain;
import soc.robot.SOCRobotClient;
import soc.util.CappedQueue;
import soc.util.SOCRobotParameters;

/** A robot named "robot N" (smart parameters) whose brain is {@link CappedBrain}. */
final class CappedClient extends SOCRobotClient {
    CappedClient(ServerConnectInfo sci, String name) {
        super(sci, name, "pw");
        rbclass = CappedClient.class.getName();
    }

    @Override
    public SOCRobotBrain createBrain(SOCRobotParameters params, SOCGame ga, CappedQueue<SOCMessage> mq) {
        return new CappedBrain(this, params, ga, mq);
    }
}
```

In `ArenaMain.run`: replace the mode check with
```java
        if (!mode.equals("stock") && !mode.equals("aligned"))
            throw new IllegalArgumentException("unknown mode " + mode + " (stock, aligned)");
```
and the robot construction line with
```java
            start(mode.equals("aligned") ? new CappedClient(sci, name) : new SOCRobotClient(sci, name, "pw"), srv);
```

Add to `jsettlers/api.txt` and re-run `bash jsettlers/check-api.sh` (expected: `62 members present`):
```
soc.robot.SOCRobotNegotiator | public soc.game.SOCTradeOffer makeOffer(soc.robot.SOCBuildPlan);
soc.robot.SOCRobotNegotiator | public void resetWantsAnotherOffer();
soc.robot.SOCRobotBrain | protected static final int TRADE_RESPONSE_TIMEOUT_SEC_BOTS_ONLY;
soc.game.SOCPlayer | public long getCurrentOfferTime();
soc.game.SOCTradeOffer | public soc.game.SOCTradeOffer(java.lang.String, int, boolean[], soc.game.SOCResourceSet, soc.game.SOCResourceSet);
```

- [ ] **Step 4: Build and run the tests to verify they pass**

Run: `pgrep -fl 'arm.sh' || echo idle`, then `bash jsettlers/build.sh && .venv/bin/pytest -q tests/python/test_jsettlers_seat.py tests/python/test_jsettlers_arena.py`
Expected: `built 17 sources`; PASS, 23 + 13 tests.

- [ ] **Step 5: Format, lint, the full suite**

Run the Global Constraints sequence. Expected: all pass, nothing skipped.

- [ ] **Step 6: Commit**

```bash
git add jsettlers/src/alphasettler/jsettlers/BridgeBrain.java jsettlers/src/alphasettler/jsettlers/CappedBrain.java \
  jsettlers/src/alphasettler/jsettlers/CappedClient.java jsettlers/src/alphasettler/jsettlers/ArenaMain.java jsettlers/api.txt \
  tests/python/test_jsettlers_arena.py
git diff --cached | grep -nE 'sk-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{36}|password=[^[:space:]]+'
git -c user.email=sean.may101@gmail.com commit -m "jsettlers: domestic trades across the bridge and the capped robots of the aligned arm"
```

---

### Task 8: `alphasettler arena --baseline jsettlers:smart`

**Files:**
- Modify: `alphasettler/cli.py` (module docstring, the `arena` parser, `_jsettlers_arena` (new), `main`)
- Test: `tests/python/test_jsettlers_cli.py` (new)

**Interfaces:**
- Consumes: Task 6-7's `oracle.jsettlers.arena` (`check`, `run_match`, `COUNTERS`, `FALLBACK_REASONS`), Task 5's `_jsettlers_missing`, the CLI's `open_output`, `_write_records`, `format_summary`, `summarize`, `_resets_line`.
- Produces: `alphasettler arena --candidate BOT --baseline jsettlers:smart --seeds N [--seed-start S] [--threads W] [--trades [--aligned]] [--native-boards] [--nets-dir D] [--out PATH]`: four games a seed, records in the arena's JSONL shape (`candidate`, `baseline`, then Task 6's record keys), a summary with the win rate, its CI and z vs 0.25, mean VP and turns, the fallbacks by reason, the belief resets, every trade counter, the bank-limit hits and the aborted games. `--aligned` or `--native-boards` without `--trades`, or `--aligned` / `--native-boards` with another baseline, is an error before any game.

- [ ] **Step 1: Write the failing tests**

Create `tests/python/test_jsettlers_cli.py`:
```python
import json

import pytest

from alphasettler import cli
from jsettlers_harness import require


def test_arena_plays_four_games_a_seed_and_reports_the_counters(tmp_path, capsys):
    require()
    out = tmp_path / "js.jsonl"
    rc = cli.main(["arena", "--candidate", "ismcts@20", "--baseline", "jsettlers:smart", "--seeds", "1",
                   "--seed-start", "71", "--threads", "2", "--out", str(out)])
    text = capsys.readouterr().out
    assert rc == 0
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert [r["candidate_seat"] for r in rows] == [0, 1, 2, 3]
    assert all(r["candidate"] == "ismcts@20" and r["baseline"] == "jsettlers:smart" and r["arm"] == "no-trades"
               for r in rows)
    for line in ("ismcts@20 vs jsettlers:smart (no-trades): win rate", "belief resets: 0", "fallbacks:",
                 "trades: our_offers", "bank limit hits:", f"wrote {out}"):
        assert line in text, line


@pytest.mark.parametrize("args,message", [
    (["--baseline", "jsettlers:fast"], "unknown JSettlers robot"),
    (["--baseline", "jsettlers:smart", "--aligned"], "--aligned needs --trades"),
    (["--baseline", "jsettlers:smart", "--native-boards"], "--native-boards needs --trades"),
    (["--baseline", "random", "--aligned", "--trades"], "only with a jsettlers: baseline"),
    (["--baseline", "random", "--native-boards"], "only with a jsettlers: baseline"),
])
def test_bad_requests_fail_before_any_game(args, message, tmp_path, capsys):
    require()
    out = tmp_path / "x.jsonl"
    rc = cli.main(["arena", "--candidate", "ismcts@20", "--seeds", "1", "--out", str(out), *args])
    assert rc == 2 and message in capsys.readouterr().err and not out.exists()


def test_a_missing_harness_is_named(monkeypatch, tmp_path, capsys):
    import oracle.jsettlers as js

    monkeypatch.setattr(js, "JAR", tmp_path / "absent.jar")
    rc = cli.main(["arena", "--candidate", "ismcts@20", "--baseline", "jsettlers:smart", "--seeds", "1",
                   "--out", str(tmp_path / "y.jsonl")])
    assert rc == 2 and "scripts/fetch-jsettlers.sh" in capsys.readouterr().err
```

Run: `.venv/bin/pytest -q tests/python/test_jsettlers_cli.py`
Expected: FAIL: `unrecognized arguments: --aligned` / `--native-boards` (argparse exits 2 with its own message, so the parametrized message checks fail), and the first test fails because `jsettlers:smart` is passed to the native arena (`unknown bot`).

- [ ] **Step 2: Write the command**

In `alphasettler/cli.py`:
1. Module docstring: add the sentence ``` `arena --baseline jsettlers:smart` plays the candidate as one seat against three JSettlers2 smart robots in JSettlers' own server (`--trades`, `--aligned`, `--native-boards`; needs the harness).```
2. In `_parser`, after the `arena` parser's `--nets-dir` argument:
```python
    a.add_argument("--aligned", action="store_true",
                   help="jsettlers: the robots' offers capped to our trading game (with --trades)")
    a.add_argument("--native-boards", action="store_true", help="jsettlers: JSettlers' own boards instead of ours")
```
3. After `_catanatron_arena`:
```python
def _jsettlers_arena(args, out: Path) -> int:
    """Our candidate as one seat against three JSettlers2 robots in JSettlers' own server. `--trades` plays
    the trading game with JSettlers' player trading on (`--aligned` caps the robots' offers to it);
    otherwise JSettlers' no-trading option is on and the candidate offers nothing."""
    why = _jsettlers_missing()
    if why is not None:
        print(f"error: the JSettlers harness is not ready: {why}", file=sys.stderr)
        return 2
    import oracle.jsettlers.arena as js_arena

    if args.aligned and not args.trades:
        raise ValueError("--aligned needs --trades")
    if args.native_boards and not args.trades:
        raise ValueError("--native-boards needs --trades (it checks the native arm)")
    baseline = args.baseline.split(":", 1)[1]
    nets_dir = str(Path(args.nets_dir).resolve())
    js_arena.check(args.candidate, baseline, args.seeds, args.seed_start, nets_dir)
    with open_output(out) as f:
        records = js_arena.run_match(args.candidate, baseline, args.seeds, args.seed_start, args.threads, nets_dir,
                                     trades=args.trades, aligned=args.aligned, native_boards=args.native_boards)
        _write_records(f, args.candidate, args.baseline, records)
    print(format_summary(f"{args.candidate} vs {args.baseline} ({records[0]['arm']})", summarize(records)))
    reasons: dict = {}
    for r in records:
        for k, v in r["fallback_reasons"].items():
            reasons[k] = reasons.get(k, 0) + v
    print(f"fallbacks: {sum(r['fallbacks'] for r in records)} {dict(sorted(reasons.items()))}")
    print(_resets_line(records))
    print("trades: " + ", ".join(f"{k} {sum(r[k] for r in records)}" for k in js_arena.COUNTERS))
    print(f"bank limit hits: {sum(r['bank_limit_hits'] for r in records)}, "
          f"aborted at the turn cap: {sum(r['aborted'] for r in records)}")
    print(f"wrote {out}")
    return 0
```
4. In `main`, in the `arena` branch, before the `catanatron:` check:
```python
            if (args.aligned or args.native_boards) and not args.baseline.startswith("jsettlers:"):
                raise ValueError("--aligned and --native-boards are only with a jsettlers: baseline")
            if args.baseline.startswith("jsettlers:"):
                return _jsettlers_arena(args, out)
```
`getattr` is not needed: both flags exist on the `arena` parser only, and these lines run only for `arena`.

- [ ] **Step 3: Run the tests to verify they pass**

Run: `pgrep -fl 'arm.sh' || echo idle`, then `.venv/bin/pytest -q tests/python/test_jsettlers_cli.py tests/python/test_cli.py`
Expected: PASS (6 new tests; `test_cli.py` unchanged).

- [ ] **Step 4: Format, lint, the full suite**

Run the Global Constraints sequence. Expected: all pass, nothing skipped.

- [ ] **Step 5: Commit**

```bash
git add alphasettler/cli.py tests/python/test_jsettlers_cli.py
git diff --cached | grep -nE 'sk-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{36}|password=[^[:space:]]+'
git -c user.email=sean.may101@gmail.com commit -m "cli: arena --baseline jsettlers:smart, with --trades, --aligned and --native-boards"
```

---

### Task 9: The gate, the smoke run, the three arms and `docs/bot/results-jsettlers.md`

**Files:**
- Create: `docs/bot/results-jsettlers.md`
- Run outputs (not committed; `runs/` is ignored): `runs/jsettlers/diff-gate.{jsonl,log}`, `runs/jsettlers/smoke.{jsonl,log}`, `runs/jsettlers/b.log`, `runs/jsettlers/nets/{B.bin,opening.bin,opening.bin.json}`, `runs/jsettlers/{native,aligned,no-trades,native-boards}.{jsonl,log}`, `runs/jsettlers/tables.md`, `runs/jsettlers/logs/` (the JVM logs)

**Interfaces:**
- Consumes: `alphasettler jsettlers-diff` (Task 5), `alphasettler arena --baseline jsettlers:smart` (Task 8), the record keys of Task 6, the results files `docs/bot/results-3d3.md` and (if present) `docs/bot/results-3d4.md` and the nets they name, the Catanatron line in `runs/vs-catanatron/gen034-1000to-vs-alphabeta-trades.log`.
- Produces: the run outputs above and `docs/bot/results-jsettlers.md`.

Every step but Step 8 is run by the controller in the background. Each step starts with `pgrep -fl 'arm.sh' || echo idle` and waits (polling every 5 minutes) until it prints `idle`. If a measurement step fails, record the failure with its numbers in the results; never rerun a measurement on the same seeds after changing anything. The gate (Step 2) is a check of the translation, not a measurement: after a fix it is rerun on its own seeds.

- [ ] **Step 1: Build at Task 8's commit (controller runs this)**

```bash
git log -1 --format=%s
pgrep -fl 'arm.sh' || echo idle
PATH=$HOME/.cargo/bin:$PATH cargo build --release -p settler-bots
bash scripts/fetch-jsettlers.sh && bash jsettlers/build.sh
test ! -e runs/jsettlers/native.jsonl && mkdir -p runs/jsettlers/nets
```
Then rebuild the extension per the Global Constraints. Pass: the commit subject is Task 8's (or a later commit that leaves Tasks 1-8's files as they committed them), every command exits 0, `build.sh` prints `built 17 sources`.

- [ ] **Step 2: The differential gate, 20 robots-only games (controller runs this)**

```bash
.venv/bin/alphasettler jsettlers-diff --games 20 --seed-start 9000000000 --workers 4 \
  --out runs/jsettlers/diff-gate.jsonl 2>&1 | tee runs/jsettlers/diff-gate.log
```
Pass (done criterion 2): exit 0 and the first line ends `0 mismatches, N stopped at the turn cap`. A mismatch goes back to an implementer with its trace (Task 5 Step 4's triage: a translation fix with a scripted test, or an allowlist entry with its rule, reason and test); then `rm runs/jsettlers/diff-gate.*` and rerun this step. Record every triaged mismatch for the results file (seed, step, what it was, the fix or entry).

- [ ] **Step 3: The smoke run, 2 seeds with `ismcts@20+t` (controller runs this)**

```bash
.venv/bin/alphasettler arena --candidate ismcts@20+t --baseline jsettlers:smart --trades --seeds 2 \
  --seed-start 9100000000 --threads 4 --out runs/jsettlers/smoke.jsonl 2>&1 | tee runs/jsettlers/smoke.log
```
Pass (spec Section 7): exit 0, `belief resets: 0`, eight records, and the `fallbacks:`, `trades:` and `bank limit hits:` lines present. A belief reset or a stalled game goes back to an implementer with the JVM log the error names.

- [ ] **Step 4: Decide B (controller runs this)**

B is the best net named by the latest results file when this step runs:

| Latest of `docs/bot/results-3d*.md` | B | Opening model |
|---|---|---|
| `results-3d4.md`, and its row for `sprt --candidate net:F@...` has the outcome `H1:` | `runs/3d4/nets/F.bin` | `runs/3d4/nets/opening.bin` if it exists, else `runs/3c/nets/opening.bin` |
| `results-3d4.md`, any other outcome for F | `runs/3d4/nets/B.bin` | `runs/3d4/nets/opening.bin` if it exists, else `runs/3c/nets/opening.bin` |
| `results-3d3.md` (no 3d-4 results yet) | `runs/3d2/nets/A.bin` (A's gen-040 at encoder v4, the spec's B today) | `runs/3c/nets/opening.bin` |
| anything else | stop: report BLOCKED with the file list | |

```bash
ls docs/bot/results-3d*.md | sort | tail -1
grep -nE 'sprt --candidate net:F@.*\| H1: ' docs/bot/results-3d4.md 2>/dev/null || echo "F not H1 (or no 3d-4 results)"
```
Set `B_SRC` and `O_SRC` from the table, then:
```bash
cp "$B_SRC" runs/jsettlers/nets/B.bin
cp "$O_SRC" runs/jsettlers/nets/opening.bin
if [ -f "$O_SRC.json" ]; then cp "$O_SRC.json" runs/jsettlers/nets/opening.bin.json; fi
{ echo "latest results: $(ls docs/bot/results-3d*.md | sort | tail -1)"; echo "B: $B_SRC"; echo "opening: $O_SRC"
  shasum -a 256 runs/jsettlers/nets/B.bin runs/jsettlers/nets/opening.bin; } | tee runs/jsettlers/b.log
.venv/bin/python3 -c "from alphasettler import Bot; Bot('net:B@1000+t+o', 0, 'runs/jsettlers/nets'); Bot('net:B@1000+o', 0, 'runs/jsettlers/nets'); print('B loads with +o')"
```
Pass: `b.log` names the results file, B and the opening model with their sha256; the last command prints `B loads with +o`.

- [ ] **Step 5: The three arms, 100 seeds each from 8,000,000,000 (controller runs this)**

Run one after another (each uses 8 workers, one JVM each), in this order:
```bash
.venv/bin/alphasettler arena --candidate net:B@1000+t+o --baseline jsettlers:smart --trades --seeds 100 \
  --seed-start 8000000000 --threads 8 --nets-dir runs/jsettlers/nets --out runs/jsettlers/native.jsonl \
  2>&1 | tee runs/jsettlers/native.log
.venv/bin/alphasettler arena --candidate net:B@1000+t+o --baseline jsettlers:smart --trades --aligned --seeds 100 \
  --seed-start 8000000000 --threads 8 --nets-dir runs/jsettlers/nets --out runs/jsettlers/aligned.jsonl \
  2>&1 | tee runs/jsettlers/aligned.log
.venv/bin/alphasettler arena --candidate net:B@1000+o --baseline jsettlers:smart --seeds 100 \
  --seed-start 8000000000 --threads 8 --nets-dir runs/jsettlers/nets --out runs/jsettlers/no-trades.jsonl \
  2>&1 | tee runs/jsettlers/no-trades.log
```
Pass for each: exit 0, 400 records, a summary line (win rate, CI, z vs 0.25, mean VP, mean turns), `belief resets: 0`. A nonzero belief-reset count is recorded in the results, not hidden. A run that fails (an error naming a JVM log) writes no records (they are written when the match ends), only its empty output file: record the error, remove that file, and rerun the arm whole on the same seeds if no code changed, or, after an implementer's fix, on the next fresh range (8,000,001,000), saying so in the results.

- [ ] **Step 6: The native-boards check, 20 seeds (controller runs this)**

```bash
.venv/bin/alphasettler arena --candidate net:B@1000+t+o --baseline jsettlers:smart --trades --native-boards --seeds 20 \
  --seed-start 8000000000 --threads 8 --nets-dir runs/jsettlers/nets --out runs/jsettlers/native-boards.jsonl \
  2>&1 | tee runs/jsettlers/native-boards.log
```
Pass: exit 0, 80 records, `belief resets: 0`. Its boards are JSettlers' own (clump-breaking on), so it shares the dice and deck streams with the native arm but not the boards: it is a check of the board ruling, not a paired comparison.

- [ ] **Step 7: The tables (controller runs this)**

```bash
.venv/bin/python3 - <<'EOF' | tee runs/jsettlers/tables.md
import json
from collections import Counter
from alphasettler.stats import summarize
from oracle.jsettlers.arena import COUNTERS
print("| arm | games | win rate [95% CI] | z vs 0.25 | mean VP | mean turns | aborted | fallbacks | belief resets |")
print("|---|---|---|---|---|---|---|---|---|")
rows = {}
for arm in ("native", "aligned", "no-trades", "native-boards"):
    rs = [json.loads(l) for l in open(f"runs/jsettlers/{arm}.jsonl")]
    rows[arm] = rs
    s = summarize(rs)
    print(f"| {arm} | {s.games} | {s.win_rate:.3f} [{s.ci_low:.3f}, {s.ci_high:.3f}] | {s.z_vs_null:.2f} | "
          f"{s.mean_vp:.2f} | {s.mean_turns:.1f} | {sum(r['aborted'] for r in rs)} | {sum(r['fallbacks'] for r in rs)} | "
          f"{sum(r['belief_resets'] for r in rs)} |")
print()
print("| arm | " + " | ".join(COUNTERS) + " | bank_limit_hits |")
print("|---|" + "---|" * (len(COUNTERS) + 1))
for arm, rs in rows.items():
    print(f"| {arm} | " + " | ".join(str(sum(r[k] for r in rs)) for k in COUNTERS)
          + f" | {sum(r['bank_limit_hits'] for r in rs)} |")
print()
for arm, rs in rows.items():
    reasons = Counter()
    for r in rs:
        reasons.update(r["fallback_reasons"])
    print(f"- {arm} fallbacks by reason: {dict(sorted(reasons.items())) or 'none'}")
EOF
grep -h "win rate" runs/vs-catanatron/gen034-1000to-vs-alphabeta-trades.log
```
Pass: four rows in each table, and the Catanatron line `net:gen-034@1000+t+o vs catanatron:alphabeta: win rate 0.713 [0.665, 0.760] ...`.

- [ ] **Step 8: Write `docs/bot/results-jsettlers.md`**

Every number is copied from the files named here; nothing is estimated. The file:

```markdown
# JSettlers2 2.6.10 smart robots as an arena opponent: results

Date: <date of the last arm>. Machine: <`sysctl -n machdep.cpu.brand_string`>, macOS (<`uname -r`>), Java <`java -version` first line>.
Commit: <`git log -1 --format=%h`> (Plan 13, Task 8, plus any gate fixes), plus the commit adding this file.

Spec: [`docs/superpowers/specs/2026-10-09-jsettlers2-arena-design.md`](../superpowers/specs/2026-10-09-jsettlers2-arena-design.md).
Plan: [`docs/superpowers/plans/2026-10-09-plan-13-jsettlers-arena.md`](../superpowers/plans/2026-10-09-plan-13-jsettlers-arena.md).
Harness: `jsettlers/` (GPL-3.0-or-later, see `jsettlers/COPYING`) against the released `jsettlers-2.6.10-server.jar`
(sha256 `9b26cb20ec943213600795d660fad2da32fc72629c348e180bd171cc26d6b4e2`, fetched by `scripts/fetch-jsettlers.sh`).

The candidate is `net:B@1000+t+o` (the no-trades arm: `net:B@1000+o`), B = <B line of runs/jsettlers/b.log> with the
opening model <opening line of runs/jsettlers/b.log>, chosen from <latest results line of runs/jsettlers/b.log>.
Opponents: three stock JSettlers2 smart robots ("robot 1".."robot 3"), 100 seeds from 8,000,000,000, four games a seed
(the candidate in each turn position), the same board, dice and deck in every game and arm of a seed.

## The translation (done criterion 2)

<the summary lines of runs/jsettlers/diff-gate.log>

Triaged in the gate: <each mismatch found and its fix or allowlist entry, one line each, or "none">.

## The smoke run

<the summary, fallbacks, belief resets, trades and bank-limit lines of runs/jsettlers/smoke.log>

## Against three smart robots (done criteria 1 and 3)

<the first table of runs/jsettlers/tables.md>

Beside the Catanatron line: `net:gen-034@1000+t+o` against three Catanatron AlphaBeta with trades, 0.713
[0.665, 0.760] over 100 seeds (`runs/vs-catanatron/`, z vs 0.25 = 18.91, mean VP 9.05, mean turns 74.1).

## Trades and the unlimited bank

<the second table of runs/jsettlers/tables.md>

<the fallback lines of runs/jsettlers/tables.md>

## Rulings the build made

- The event stream, not a request, holds the positions; a decision imports the latest position in the stream.
- A client request starts a stretch; two stretches are one action of ours where the first leaves no phase of ours.
- The bridge is JSettlers seat 0; the first player is scripted to (4 - k) % 4; our player i is JSettlers seat (first + i) % 4.
- A seed's board is the native arena's, its port kinds in our order on JSettlers' port edges in their order; clump-breaking off on injected boards.
- `SOCGame.rand` is routed by caller: the first player scripted, dice, deck and steals from the seed's three streams.
- A roll's dice are reported as the pair (min(n - 1, 6), rest); only the total is JSettlers'.
- `bank_limit` off: the bank never runs short and shows 19 less the cards in hands; resolved configs name it only when off.
- `bank_limit_hits` counts the stretches after which some resource's cards in hands exceed 19 and grew.
- Discards are replayed in our seat order once the last is in; our discards are chosen where only our seat owes.
- Our bot is shown every discard's resources (our rules make discards public; JSettlers shows opponents only their count): a small information edge for our bot.
- One bank trade is that many maritime trades of ours, paired in resource order; anything else is allowlisted (`bank-trade`).
- Executed domestic trades are re-imported in the replay and shown to our bot as the offer re-stated, accepted and confirmed.
- Before the roll our bot rolls or plays a knight; any other card there falls back (`preroll-card`).
- The robber's victim is chosen with the move, at the hypothetical steal position.
- Our offers go to the opponents whose lone acceptance our bot would confirm; with none the offer is made and declined in our engine only, and the bot decides again; `our_offers` counts both.
- Robot offers to us while we are on turn are rejected unasked (`counters_rejected`); outside our trading game rejected unasked (`unrepresentable`); answered at the offerer's Main position with its earlier offers clamped to 1; 4.5 s or later sent as a rejection (`late`).
- The bridge's stock negotiator never offers for our seat and never counter-offers; a fallback answer is the stock brain's.
- Fallbacks are counted by reason: position, declined, engine, choice, preroll-card, turn-cap, java-timeout, victim.
- Built-in robots' forced end of turn is raised from 8 s to 60 s (so our thinking during a robot's turn never forces it); our turns have 600 s.
- A game at our 1000-turn cap is aborted: no winner, `aborted: true`.
- The aligned arm's robots send only one-card-a-side offers, to every opponent, at most two a turn, no counters; an offer that does not fit ends their offering for the turn.
- B is chosen by an explicit table from the latest results file; the candidate stays `net:B@1000+t+o`.
- Fresh seeds: arms from 8,000,000,000, the native-boards check on the first 20 of them, the gate from 9,000,000,000, the smoke run from 9,100,000,000.

## Reading

<three to five sentences: the headline (native) rate against 0.25 and beside the Catanatron line; what the aligned and
no-trades arms add (how much of the native rate the robots' trading costs or gives); what the counters say about the
native arm's bias (unrepresentable offers, counters, late answers); the native-boards check against the native arm; the
bank-limit hits and fallbacks, and whether either is large enough to qualify the result.>
```

- [ ] **Step 9: Commit**

Run: `grep -c '<' docs/bot/results-jsettlers.md`
Expected: `0`.

```bash
git add docs/bot/results-jsettlers.md
git diff --cached | grep -nE 'sk-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{36}|password=[^[:space:]]+'
git -c user.email=sean.may101@gmail.com commit -m "docs: JSettlers2 results, three trading arms beside the Catanatron line"
```

---

## Self-review

### Spec coverage

| Spec requirement | Task |
|---|---|
| §1.1 `arena --candidate BOT --baseline jsettlers:smart [--trades] [--aligned] --seeds N`, four games a seed, arena JSONL with the Section 5 counters | 6 (`run_match`, records), 7 (`--aligned` robots), 8 (the command; `test_arena_plays_four_games_a_seed_and_reports_the_counters`) |
| §1.2 differential replay of robots-only games, chance forced, allowlisted differences, 0 untriaged mismatches over 20 games | 4 (steps with forced outcomes), 5 (`jsettlers-diff`, allowlist, two-game test, corrupted-dice test), 9 Step 2 (the 20-game gate) |
| §1.3 `net:B@1000+t+o` vs three smart robots, 100 seeds from 8,000,000,000, native / aligned / no-trades arms, win rate with CI and z, mean VP and turns, counters | 9 Steps 4-5, 7 |
| §1.4 `docs/bot/results-jsettlers.md` beside the Catanatron line | 9 Steps 7-8 |
| §2 server on a string port, events with audience and true contents, the server-side game as a snapshot | 3 (`ArenaServer`, `Snapshots`, the stream), 4 (`translate.snapshot`) |
| §2 bridge seat: `BridgeClient extends SOCRobotClient`, `BridgeBrain extends SOCRobotBrain`, the stock turn machine, hooks for placement, roll or card, main plan, robber and victim, discard, monopoly and year of plenty, answering and making offers | 6 (all but offers), 7 (offers and answers) |
| §2 events to our bot, redacted for our seat | 4 (`Tracker` events), 6 (`Seat.observe`, `redact`); belief checked in Task 4's robots game and Tasks 6-7's games |
| §2 fallbacks to the stock brain, counted | 6 (`Fallback`, reasons, Java `fallback` messages), Review Focus 2 |
| §2 JSON lines over stdin and stdout | 3 (`Pipe`, `Jvm`) |
| §2 rulings: route (c); `jsettlers/` under GPL-3.0-or-later with a notice; jar downloaded and sha256-checked, compiled with `javac` | 1 (`COPYING`, headers checked by `build.sh`, `fetch-jsettlers.sh`, `check-api.sh`), 3, 6 |
| §3 bank: `bank_limit` (default on), JSettlers imports with it off, decisions where it bites counted | 2, 4 (`bank_limit_hits`), 6 (records) |
| §3 boards injected by a board factory on JSettlers' nine port edges, clump-breaking off; a native-boards check arm of 20 seeds | 3 (`ArenaBoardFactory`, `ArenaBoard`, `boards.native_board`, `js_layout`), 9 Step 6 |
| §3 dice, steals and deck routed by reflection to per-seed streams; the first player scripted; common random numbers | 3 (`RoutedRandom`, `test_dice_and_the_deck_follow_the_seed_in_every_rotation`) |
| §3 discards and bank trades map to that many one-card steps of ours | 4 (`_discard`, `_bank` and their tests) |
| §3 turn limit: the harness ends a game at our `max_turns`, no winner | 5 (`capped`), 6 (`test_a_game_at_the_turn_cap_ends_with_no_winner`) |
| §3 options: 4 players, 10 VP, classic board, no scenarios, `PLAY_FO` off | 3 (`ArenaServer.create`) |
| §4 three stock smart robots | 3 (names `robot N`, `start["smart"]` asserted) |
| §5 our offers to the opponents we would confirm; the first acceptor's trade stands for our confirm; replayed as offer, answers, confirm; counters rejected; our cap by our engine | 6 (`Seat._offer`, tests), 7 (`offer` reply, `makeCounterOffer`), 4 (`_trade` re-stated events) |
| §5 robot offers within our game answered at the robot's Main position; outside it rejected unasked (`unrepresentable`) | 6 (`Seat._answer`, tests), 7 (`considerOffer`) |
| §5 answer time: late answers counted (`late`) | 6 (`LATE`, `test_late_answers_are_sent_as_rejections_and_counted`) |
| §5 robot-robot trades reach our bot as trade events with real bundles | 4 (`_offer`, `_trade`, `_reject`, `_clear`) |
| §5 rulings: native arm the headline; aligned arm's capped negotiator; our bot answers only one-card offers | 7 (`CappedBrain`), 9; 6 (`unrepresentable`) |
| §5 records: our_offers, our_offers_accepted, offers_to_us, offers_to_us_accepted, unrepresentable, counters_rejected, late, bank_limit_hits, fallbacks, belief_resets | 6 (`RECORD_KEYS`, `test_a_no_trades_bridge_game_records_every_counter`) |
| §6 out of scope: fast robots, STAC, 5-6 players, sea boards, two-card offers by our bot, network play, human play | `BASELINES = ("smart",)`; `PL` 4; injected classic boards; `max_trade_cards` 1; string port only |
| §7 Python: tables are bijections checked against a probe; snapshot round trips; event translation for every classic-board sequence; trade counters on scripted offers | 1 (`test_jsettlers_tables.py`), 4 (`test_jsettlers_translate.py`, `test_jsettlers_events.py`), 6 (`test_jsettlers_seat.py`) |
| §7 Rust: `bank_limit` off lets production and trades exceed 19; with it on every test and digest unchanged | 2 (`engine/tests/bank_limit.rs`, the pins of Step 1) |
| §7 Java: compiles against the pinned jar; a robots-only game runs to a winner under the injected board and routed dice | 1 (`build.sh`), 3 (`test_a_robots_game_runs_to_a_winner_on_the_injected_board_with_the_scripted_first_player`) |
| §7 integration: the differential replay; a 2-seed smoke test with `ismcts@20+t`, belief resets 0, counters reported | 5, 9 Steps 2-3 |

### Placeholder scan

Searched for "TBD", "TODO", "implement later", "fill in", "similar to Task", "appropriate", "handle edge cases" and for instructions to fix code shown in a step: none in the code steps. The templates are Task 9's run-time choices (Step 4's B table, decided from the results files present when it runs) and Step 8's results file, whose `<...>` marks each name their source; Step 9 refuses to commit while one is left. The JSettlers facts the code depends on are checked, not assumed: the members reached by reflection or relied on by signature are in `jsettlers/api.txt` (48, then 55, then 60 lines, checked by `javap` before every build); the board geometry, codes and port facts are checked against the probe dump (Task 1 Step 5 prints and asserts the expected values; a different value is BLOCKED, not fitted); the routed first player, dice and deck are asserted by Task 3's games; the message sequences are those of the 2.6.10 `doc/Message-Sequences-for-Game-Actions.md`, and the robots game of Task 4 and the gate of Task 5 fail loudly on any sequence the rules of `Tracker._action` misread.

### Type consistency

- Java: `GameSpec(name, hexes, numbers, robber, first, dice, deck, misc, noTrades, bridge)` is what `ArenaServer.create`, `ArenaBoard`, `ArenaBoardFactory.pending` and `RoutedRandom` read; `Pipe.send/ask/reply/fatal/flushAndExit` are what `ArenaServer`, `ArenaMain` and `BridgeBrain` call; `ArenaServer.BRIDGE` (`robot bridge`) is the name `BridgeClient` logs in with and `create` seats at JSettlers seat 0; `BridgeBrain.ask(kind, extra)`, `fellBack(reason)`, `takeVictim()`, `STOCK_ROBBER`, `main(reply)` are what `BridgeOpening`, `BridgeRobber` and Task 7's additions use; `BridgeDM.choose(a, b)` is what the `yop` reply calls.
- Protocol: Python's `game_command` keys (`t, g, layout{hexes, numbers, robber}, first, dice, deck, misc, nt, bridge`) are `GameSpec.parse`'s; the `start`, `ev`, `ask`, `fallback`, `over`, `error` messages Java writes are what `Jvm.recv`, `Tracker(start)`, `Tracker.feed`, `diff.play_game` and `arena.play_game` read; each reply shape of Task 6's Interfaces is what `BridgeBrain` parses (`node`, `edge`, `do`, `piece`, `coord`, `res`, `give`, `get`, `to`, `hex`, `victim`, `accept`, `fallback`).
- Python: `Context(board, first, turn, rolled, drawn, offers)` with `.ours/.js` (Task 4) is what `Tracker`, `Seat` and `diff` use; `Step(kind, player, actions, events, after, untranslatable, info)` and its kinds are what `diff._replay` and `Seat.observe` read (`info` keys `from, to, give, get` for offers and `offerer, partner, give, get` for trades); `Tracker.declined`, `.bank_limit_hits`, `.rates(raw, seat)`, `.hand(raw, seat)`, `.snapshot(max_offers)` are what `Seat` calls; `config(trades)` gives the import config with `bank_limit: False` everywhere a JSettlers position is imported (`diff.CONFIG` likewise); `RECORD_KEYS` is the order `play_game` writes and the tests assert; `COUNTERS` and `FALLBACK_REASONS` are what the CLI and Task 9's tables sum.
- Rust: `GameConfig.bank_limit`, `State::bank_has(r: usize, n: u8)`, `State::bank_pay(&Hand)`, `State::refresh_bank()` are what `roll.rs`, `maritime.rs`, `dev.rs`, `setup.rs`, `apply.rs` and the tests call; `config_dict` writes `"bank_limit"` only when false, which `test_resolved_configs_name_the_bank_limit_only_when_it_is_off` asserts.

### Review Focus

Each of the five lines names its pinning test in the owning task: 1 in Task 3 (`test_errors_and_silence_fail_loudly_naming_the_log`, `test_the_jvm_exits_when_its_stdin_closes`); 2 in Task 6 (`test_unimportable_declined_or_failing_decisions_fall_back_by_reason`, and the reasons of its JVM game); 3 in Task 6 (the two `test_answers_*` tests and the late-answer test) with Task 7's native game; 4 in Task 2 (the engine and binding tests) and Task 4 (translation and counter tests); 5 in Task 6 (the turn-cap game and the `turn-cap` fallback case).
