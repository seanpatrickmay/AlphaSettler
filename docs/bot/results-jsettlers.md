# JSettlers2 2.6.10 smart robots as an arena opponent: results

Date: 2026-10-11. Machine: Apple M1 Pro, macOS (25.5.0), Java java version "20.0.2" 2023-07-18.
Commit: 508d832 (Plan 13, Task 8, plus the final fix waves and the arms script; the gate needed no fixes), plus the commit adding this file.

Spec: [`docs/superpowers/specs/2026-10-09-jsettlers2-arena-design.md`](../superpowers/specs/2026-10-09-jsettlers2-arena-design.md).
Plan: [`docs/superpowers/plans/2026-10-09-plan-13-jsettlers-arena.md`](../superpowers/plans/2026-10-09-plan-13-jsettlers-arena.md).
Harness: `jsettlers/` (GPL-3.0-or-later, see `jsettlers/COPYING`) against the released `jsettlers-2.6.10-server.jar`
(sha256 `9b26cb20ec943213600795d660fad2da32fc72629c348e180bd171cc26d6b4e2`, fetched by `scripts/fetch-jsettlers.sh`).

The candidate is `net:B@1000+t+o` (the no-trades arm: `net:B@1000+o`), B = `runs/3d4/nets/B.bin` (sha256
`31909c7f9595d88bbadd179299a270013216c8e373a2ec33aa09acf171d7fbbe`, A's gen-040 at encoder v4) with the opening model
`runs/3c/nets/opening.bin` (sha256 `9ab63da1309e35f2a87990edc85064ee9888057de84d52fb0e09ae382ced922b`), chosen from
`docs/bot/results-3d4.md`.
Opponents: three stock JSettlers2 smart robots ("robot 1".."robot 3"), 100 seeds from 8,000,000,000, four games a seed
(the candidate in each turn position), the same board, dice and deck in every game and arm of a seed.

## The translation (done criterion 2)

```
20 games, 8658 steps compared, 0 games identical end to end, 0 mismatches, 1 stopped at the turn cap
  allowlisted domestic-trade: 268 steps in 20 games (a domestic trade executes when the first player accepts; the offerer never confirms, offers go to a chosen subset, may be two cards for one and are uncapped)
```

Triaged in the gate: none.

## The smoke run

```
ismcts@20+t vs jsettlers:smart (native): win rate 0.125 [0.000, 0.370] 95% CI over 2 seeds (8 games, 1 wins, 0 draws), z vs 0.25 = -1.00, mean VP 5.50, mean turns 93.9
dropped: 0 games (0 robot desync in setup), 0 seeds left out of the summary
fallbacks: 0 {}
belief resets: 0
forced ends of our turn: 0
trades: our_offers 0, our_offers_accepted 0, offers_to_us 156, offers_to_us_accepted 0, unrepresentable 3, counters_rejected 0, late 0, beaten 0, robot_offers 213, robot_offers_over_caps 155
bank limit hits: 2, aborted at the turn cap: 0 (mean turns since the last build: n/a)
```

## Against three smart robots (done criteria 1 and 3)

| arm | games | dropped | win rate [95% CI] | z vs 0.25 | mean VP | mean turns | turn-cap games | fallbacks | belief resets | forced ends of our turn |
|---|---|---|---|---|---|---|---|---|---|---|
| native | 396 | 1 seeds (4 games) | 0.657 [0.607, 0.706] | 16.09 | 8.95 | 80.6 | 0 | 210 | 8 | 0 |
| native, without turn-cap seeds | 396 | 1 seeds (4 games) | 0.657 [0.607, 0.706] | 16.09 | 8.95 | 80.6 | 0 | 210 | 8 | 0 |
| aligned | 396 | 1 seeds (4 games) | 0.654 [0.604, 0.704] | 15.97 | 8.90 | 80.4 | 0 | 201 | 5 | 0 |
| aligned, without turn-cap seeds | 396 | 1 seeds (4 games) | 0.654 [0.604, 0.704] | 15.97 | 8.90 | 80.4 | 0 | 201 | 5 | 0 |
| no-trades | 396 | 1 seeds (4 games) | 0.689 [0.638, 0.741] | 16.80 | 9.10 | 83.6 | 0 | 221 | 0 | 0 |
| no-trades, without turn-cap seeds | 396 | 1 seeds (4 games) | 0.689 [0.638, 0.741] | 16.80 | 9.10 | 83.6 | 0 | 221 | 0 | 0 |
| native-boards | 80 | 0 seeds (0 games) | 0.688 [0.618, 0.757] | 12.25 | 8.99 | 79.6 | 0 | 39 | 1 | 0 |
| native-boards, without turn-cap seeds | 80 | 0 seeds (0 games) | 0.688 [0.618, 0.757] | 12.25 | 8.99 | 79.6 | 0 | 39 | 1 | 0 |

- native: 0 turn-cap games (mean turns since the last build n/a); late 16 of 7731 offers to us (0.2%), beaten 109
- aligned: 0 turn-cap games (mean turns since the last build n/a); late 15 of 9209 offers to us (0.2%), beaten 127
- no-trades: 0 turn-cap games (mean turns since the last build n/a); late 0 of 0 offers to us (0.0%), beaten 0
- native-boards: 0 turn-cap games (mean turns since the last build n/a); late 4 of 1557 offers to us (0.3%), beaten 24
- native JVM logs: 0 forced ends of the bridge's turn, 0 robot brain exceptions (any seat)
- aligned JVM logs: 0 forced ends of the bridge's turn, 0 robot brain exceptions (any seat)
- no-trades JVM logs: 0 forced ends of the bridge's turn, 0 robot brain exceptions (any seat)
- native-boards JVM logs: 0 forced ends of the bridge's turn, 0 robot brain exceptions (any seat)

The dropped seed is 8,000,000,034 in all three arms (a setup `robot-desync`), so the three arms stay paired over the
same 99 seeds. Paired by seed (each seed's four-game win rate), aligned less native is -0.003 [-0.058, +0.053] and
no-trades less native is +0.033 [-0.023, +0.089] (95%, normal approximation over 99 seeds).

The native-arm condition (`late` at most 5% of `offers_to_us`): met, late 16 of 7731 offers to us (0.2%) at 8 workers;
no rerun.

Belief resets are not 0 in the trading arms (native 8 in 6 games, seeds 8,000,000,006, 029, 048, 055, 057 and 065;
aligned 5; native-boards 1) and are 0 in the no-trades arm, so the gap is in how some trade events reach our bot. Each
reset costs one decision played greedily before the belief is rebuilt. Not triaged yet.

Beside the Catanatron line: `net:gen-034@1000+t+o` against three Catanatron AlphaBeta with trades, 0.713
[0.665, 0.760] over 100 seeds (`runs/vs-catanatron/`, z vs 0.25 = 18.91, mean VP 9.05, mean turns 74.1).

## Trades and the unlimited bank

| arm | our_offers | our_offers_accepted | offers_to_us | offers_to_us_accepted | unrepresentable | counters_rejected | late | beaten | robot_offers | robot_offers_over_caps | bank_limit_hits |
|---|---|---|---|---|---|---|---|---|---|---|---|
| native | 2477 | 368 | 7731 | 241 | 663 | 197 | 16 | 109 | 10857 | 7634 | 35 |
| aligned | 2401 | 356 | 9209 | 227 | 17 | 10 | 15 | 127 | 9209 | 0 | 19 |
| no-trades | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 14 |
| native-boards | 538 | 67 | 1557 | 64 | 156 | 54 | 4 | 24 | 2194 | 1538 | 5 |

- native fallbacks by reason: {'position': 7, 'preroll-card': 203}
- aligned fallbacks by reason: {'position': 14, 'preroll-card': 187}
- no-trades fallbacks by reason: {'preroll-card': 221}
- native-boards fallbacks by reason: {'position': 4, 'preroll-card': 35}

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
- Robot offers to us while we are on turn are rejected unasked (`counters_rejected`); outside our trading game rejected unasked (`unrepresentable`); answered at the offerer's Main position with its earlier offers clamped to 1; 4.5 s or later sent as a rejection (`late`); an acceptance the server refused once the offer was gone is `late` (withdrawn) or `beaten` (another player's trade executed first).
- The bridge's stock negotiator never offers for our seat and never counter-offers; a fallback answer is the stock brain's.
- Fallbacks are counted by reason: position, declined, engine, choice, preroll-card, turn-cap, java-timeout, victim, forced-discard (a discard of ours the server forced, before our reply or with none).
- A turn of ours the server force-ended (after 600 s, e.g. an exception in the bridge brain, which `SOCRobotBrain.run` survives without a fallback) is counted in `forced_end_turns`; the JVM logs are grepped per arm for the same and for brain exceptions.
- Built-in robots' forced end of turn is raised from 8 s to 60 s (so our thinking during a robot's turn never forces it); our turns have 600 s.
- A game at our 1000-turn cap is aborted: no winner, `aborted: true`. Turn-cap games stay in as no-winner losses (as in the Catanatron arena); every table also has a row without each seed that has one (the stock Longest-Road stall: nobody holds Longest Road on a robot's copy of the game, so a robot at 8-9 VP stops building), with the cap count and the mean turns since the last build.
- A setup settlement the server cancelled by a forced end leaves a stock robot on a board it does not have (`robot-desync`): the game is aborted at once and its whole seed (all four rotations) is left out of every summary and table (`oracle.jsettlers.arena.kept`), the dropped seeds and games counted per arm; the arms are then not exactly paired.
- A stale acceptance of ours that executes a later offer from the same robot (`SOCAcceptOffer` names only the offerer) is not counted: the differential replay is robots-only and does not see it.
- The native-boards check's boards are JSettlers' own and unseeded: each record names its board (`board`), and the check is unpaired.
- The arms start only on an idle machine on AC power with the lid open (`runs/jsettlers/idle.sh`); the native arm is valid only with `late` at most 5% of `offers_to_us`, else it is rerun with 4 workers on the next fresh range.
- One worker's error or death stops the whole arm at once (exit 3); the games it finished are kept apart in the output's `.partial` file, never used as a result.
- The aligned arm's robots send only one-card-a-side offers, to every opponent, at most two a turn, no counters; an offer that does not fit stops that robot's offering until the stock brain clears `doneTrading` (a new turn, or a declined request mid-turn), within the same caps.
- B is chosen by an explicit table from the latest results file; the candidate stays `net:B@1000+t+o`.
- Fresh seeds: arms from 8,000,000,000, the native-boards check on the first 20 of them, the gate from 9,000,000,000, the smoke run from 9,100,000,000.

## Caveats

- Our bot is shown the resources of every discard; JSettlers shows opponents only the count: a small information edge for our bot.
- The stock Longest-Road stall: the win rate is given with and without the seeds with a turn-capped game, beside the cap count and the mean turns since the last build. No arm had a turn-cap game.
- Desync seeds are dropped whole (count per arm above), so the arms are not exactly paired. Here the one desync seed is the same in every arm.
- The native-boards check is unpaired (JSettlers' own boards are unseeded).
- Common random numbers cover the board, dice, deck, steals and first player only; the robots' own randomness and thread timing are free.
- The native arm is biased by the robots' offers our trading game cannot hold (`unrepresentable`) and by the counter-offers rejected unasked (`counters_rejected`); the aligned arm is the one comparable to the Catanatron line.
- Our offers go only to the robots that hold the card and whose acceptance our bot would confirm; the first acceptor's trade stands.
- A development card other than a knight before the roll falls back to the stock brain (`preroll-card`).
- A stale acceptance of ours that executes a later offer of the same robot is not counted.
- After our rules end a game nothing ends the JSettlers game; the decisions asked after it are answered `after-win` and not counted.
- The gate checks the translation of robots-only games, not the bridge's action mapping; the fallback counters stand in for that.
- Answer timing: `late` and `beaten` are reported per arm; the state machine's load (8 JVMs beside the bot) moves them.
- The JVM-log grep counts (forced ends of the bridge's turn, robot brain exceptions) are reported per arm.
- The bank is unlimited in JSettlers; `bank_limit_hits` is reported per arm.
- Belief resets in the trading arms (above) are untriaged; each costs one greedy decision.
- The Catanatron line is a different net (gen-034) against a different opponent, so the two rates sit side by side, not as a paired comparison.

## Reading

`net:B@1000+t+o` wins 0.657 [0.607, 0.706] against three stock smart robots with trades on, far above the 0.25 of an
even seat (z 16.09), and a little below the 0.713 [0.665, 0.760] that gen-034 reaches against Catanatron AlphaBeta,
a different net against a different opponent, so the two say only that the smart robots are in the same range as AlphaBeta. The aligned arm is the same as the native one (paired
-0.003 [-0.058, +0.053]): the robots' offers our trading game cannot hold (663 unrepresentable, 197 counters rejected,
of 7731 offers to us) do not move the rate measurably. Trading itself buys our bot nothing here: the no-trades arm wins
0.689, paired +0.033 [-0.023, +0.089] over native, and our bot accepts few robot offers (241 of 7731) while 368 of its
2477 offers are taken. The native-boards check (0.688 [0.618, 0.757] on JSettlers' own boards) agrees with the native
arm, so the injected boards do not flatter our bot. Late answers (0.2%), bank-limit hits (35 in 396 games), fallbacks
(about 0.5 a game, almost all `preroll-card`) and the 8 belief resets are each too small to qualify the result.
