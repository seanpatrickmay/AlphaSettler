# JSettlers2 robots as an arena opponent

Date: 2026-10-09. Builds on the Catanatron arena (`oracle/`: `translate.py` `catan_snapshot` →
`Game.from_snapshot`, `events.py` `record_events`, `arena.py`, `diff.py`; `alphasettler arena
--baseline catanatron:...`) and the trading game (`docs/superpowers/specs/2026-10-07-trading-design.md`,
`TRADES = {"max_offers_per_turn": 2, "max_trade_cards": 1}`).

The user asked (2026-10-08) how the bot does "against the other best bots out there ... winrate in a
4-player game with trades on", and approved adding JSettlers2 after the 3d work ("if you think
jsettlers2 would be stronger, then add it after the current work"). JSettlers2's robots negotiate
domestic trades (they offer, answer and counter-offer), which Catanatron's never do, so it is the
first opponent that tests the bot's answers to foreign offers. Decisions the user did not make are
marked **Ruling**.

## 1. Done criteria

1. **The bridge runs full games:** `alphasettler arena --candidate BOT --baseline jsettlers:smart
   [--trades] [--aligned] --seeds N` plays BOT as one seat against three stock JSettlers2 2.6.10 *smart*
   robots, four games a seed (BOT in each turn position), and writes records in the arena's JSONL shape
   with the JSettlers counters of Section 5.
2. **The translation is checked:** a differential replay of robots-only JSettlers games through our
   engine, with chance outcomes forced, matches the server's state at every step, except the rule
   differences of Section 3 named in an allowlist; 0 untriaged mismatches over 20 games.
3. **The measurement:** `net:B@1000+t+o` (B: the best net named by the latest results file when the
   run starts; today A's gen-040 at encoder v4, `runs/3d2/nets/A.bin`, with `runs/3c/nets/opening.bin`)
   against three smart robots, 100 seeds from 8,000,000,000 (400 games), in three arms:
   - **native** (headline): trading on, the robots on JSettlers' own trading rules;
   - **aligned**: trading on, the robots' offers filtered to our trading game (Section 5);
   - **no trades**: JSettlers' `NT` option on, our bot without `+t`.
   Each arm reports the win rate with its 95% CI and z vs 0.25, mean VP and turns, and the counters.
4. **Results** in `docs/bot/results-jsettlers.md`, beside the Catanatron AlphaBeta line
   (`net:gen-034@1000+t+o` 0.713 [0.665, 0.760], `runs/vs-catanatron/`).

## 2. The bridge

One JVM per arena worker runs JSettlers' own server in-process (a string port, no network), three
stock robots and one bridge seat. The server is the authority on the game.

- **Server:** `ArenaServer extends SOCServer` creates each game with the options of Section 3,
  streams every game event the server records (with its audience and true contents) to the Python
  driver, and exports the server-side `SOCGame` as a snapshot in our `Game.from_snapshot` schema on
  request, read under the game's monitor.
- **Bridge seat:** `BridgeClient extends SOCRobotClient` (so the game counts as robots-only and runs at
  the fast pace) whose brain, `BridgeBrain extends SOCRobotBrain`, keeps the stock turn machine and
  protocol and replaces its decision hooks (initial placement, roll or play a card, the main-phase
  plan, robber and victim, discard, monopoly and year-of-plenty picks, answering an offer, making an
  offer) with a request to Python: the snapshot of the moment and the decision kind; Python imports
  it, asks `Bot.act` on our observation, and returns our action, which the brain maps to the
  JSettlers request.
- **Events to our bot:** the driver turns the server's event stream into our event dicts and feeds
  `bot.observe(seat, [redact(e, seat) ...])` as the Catanatron arena does, so the bot's belief about
  hidden cards is built only from what our seat may see.
- **Fallbacks:** a position or choice the translation cannot express falls back to the stock brain's
  own decision for that hook, counted in `fallbacks` (as the Catanatron arena falls back to the first
  legal action).
- **Pipe:** JSON lines over the JVM's stdin and stdout.

**Ruling:** route (c) of the research note (an in-JVM referee bridge) over a Python client on the
network protocol (route a) or JSettlers' robots ported into our engine (route b). The real server and
the real robots play, our bot gets an exact god's-view snapshot to redact, and boards and dice can be
injected. Cost if wrong: the stock brain decides *when* to ask us, so its own guards (denied-request
caps, offer waits, a forced end of turn) can pre-empt a decision; the fallback and late-answer
counters show it.

**Ruling:** the Java code lives in its own directory (`jsettlers/`) under GPL-3.0-or-later headers
with a copying notice, because it extends JSettlers classes; everything else talks to it over the
pipe. The JSettlers jar is not committed: a setup script downloads release 2.6.10's server jar and
checks its sha256 (`9b26cb20ec943213600795d660fad2da32fc72629c348e180bd171cc26d6b4e2`), and the harness
is compiled with `javac` against it (Java 8+; this Mac has Java 20; no Gradle needed). Cost if wrong:
none to the results; a license question is the user's to settle before anything is distributed.

## 3. Rules alignment

JSettlers2's classic rules match ours on setup order, the win check, the development deck and its
play timing, road building, longest road and largest army (ties included), the discard rule and the
robber. The differences and how each is handled:

- **Bank:** JSettlers' bank is unlimited. **Ruling:** our engine gains a `bank_limit` setting
  (default on, every existing game unchanged) and JSettlers positions import with it off; the
  records count the decisions where our limit would have bitten.
- **Boards:** JSettlers generates its own (spiral numbers, clump-breaking on). **Ruling:** inject our
  boards through a board factory, drawn from our seed on JSettlers' nine port edges, with clump-breaking
  off, so a seed is the same board in all four games and arms. Cost if wrong: the robots see boards
  they would not generate; a small native-boards check arm (20 seeds) reports the difference.
- **Dice and first player:** JSettlers has no seed. **Ruling:** route the game's random source by
  reflection to a per-seed stream for dice, steals and the deck shuffle, and script the first player
  so our seat plays each turn position once a seed (common random numbers across arms). The robots'
  own randomness and thread timing stay free, so a game does not replay exactly; recorded, as with
  Catanatron's time-limited AlphaBeta.
- **Discards and bank trades:** one JSettlers message may discard a set or trade several units; it
  maps to that many of our one-card steps.
- **Turn limit:** none in JSettlers; the harness ends a game at our `max_turns` and records no winner.
- **Options:** 4 players, 10 VP, the classic board, no scenarios or sea board, robots without full
  observation (`PLAY_FO` off).

## 4. Opponents

**Ruling:** three stock *smart* robots (JSettlers' stronger built-in setting); the fast robots are
not measured. The literature puts JSettlers' robots at about AlphaBeta's level (an unreviewed 2026
head-to-head gives each about 20–25% against three of the other), so this measures the bot against
a negotiating opponent more than against a stronger one.

## 5. Domestic trades

JSettlers' trading differs from ours: any player may offer the current player, the current player
offers to a chosen subset, robots offer two cards for one and counter-offer, there is no per-turn
cap, and the first acceptor's trade executes at once (the offerer does not confirm).

- **Our offers** (we are the current player, in Main, within our trading game): the bridge sends the
  offer to the opponents our bot would confirm with, found by asking our bot at the hypothetical
  position where that opponent alone accepted; the first robot to accept gets the trade, which stands
  for our confirm. Our engine replays it as the offer, the answers in seat order and our confirm.
  Counter-offers to our offers are rejected. Our own per-turn cap is enforced by our engine.
- **Robot offers to us:** an offer within our trading game (one card a side) is imported at the
  robot's Main position with the offer applied, and our bot answers it at its TradeResponse decision;
  an offer outside it (a robot's two-for-one) is rejected without asking and counted
  (`unrepresentable`).
- **Answer time:** robots wait 5 s for answers in robots-only games; an answer that misses it is
  declined by the server and counted (`late`).
- **Robot-robot trades** are public and reach our bot as trade events with their real bundles.

**Ruling:** the **native** arm is the headline (the opponents as they are), and the **aligned** arm
replaces the robots' offer-making with a capped negotiator (one card a side, to all, at most two
offers a turn, no counters), which makes the line comparable with the Catanatron trading runs. Cost
if wrong: the native arm is biased against us by every rejected two-for-one and every late answer;
the counters say by how much.

**Ruling:** our bot answers only offers it can represent (one card a side), because it was trained in
that game; evaluating two-card offers is a later, measured change.

Records gain, per game: `our_offers`, `our_offers_accepted`, `offers_to_us`, `offers_to_us_accepted`,
`unrepresentable`, `counters_rejected`, `late`, `bank_limit_hits`, beside `fallbacks` and
`belief_resets`.

## 6. Out of scope

JSettlers' fast robots, the STAC fork's agents, 5–6 players, sea boards and scenarios, two-card offers
by our bot, playing on a JSettlers server over the network, human play through JSettlers' client.

## 7. Testing

- Python: the coordinate, resource and port tables are bijections checked against a probe dump of
  JSettlers' 4-player board; snapshot round trips through `Game.from_snapshot` on recorded positions;
  event translation for every classic-board message sequence; the trade counters on scripted offers.
- Rust: `bank_limit` off lets production and trades exceed 19 cards a resource; with it on every
  existing test and pinned digest is unchanged.
- Java: the harness compiles against the pinned jar; a robots-only game runs to a winner under the
  injected board and routed dice.
- Integration: the differential replay of criterion 2; a 2-seed bridge smoke test with `ismcts@20+t`
  that ends every game with `belief_resets` 0 and reports the counters.
