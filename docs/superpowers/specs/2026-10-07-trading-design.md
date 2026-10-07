# Sub-project 3c: domestic trading

Date: 2026-10-07. Builds on 3b (network v2, `net:gen-004` in `runs/3b-v2/nets`, the fast tooling of
`alphasettler fast-train` / `fast-loop`).

The user asked to "move towards implementing trading" and to work without stopping to ask. Every
decision below that the user did not make is marked **Ruling** with its reason, so it can be
overridden.

## 1. Goal and done bar

A bot that offers, answers and confirms player-to-player trades, and is measurably stronger for it.

Done when all three hold, each over 200 fresh seeds (from 2,000,000,000, never used by training or
gates), four seat rotations per seed:

1. **Trading pays against our own bots.** `net:T@600+t` against three `net:T@600` that answer offers
   but never make them, in the trading game (Section 2): win rate above 0.25 with z > 3. `T` is the
   trained trading net.
2. **Trading pays against AlphaBeta.** `net:T@1000+t` against three Catanatron AlphaBeta with domestic
   trades on (AlphaBeta answers our offers with its own search): win rate above `net:T@1000`'s
   (no offers) on the same seeds, paired per seed, z > 2.
3. **No regression without trades.** `net:T@600` against three `net:gen-004@600` with trades off: z > −2.

**Ruling:** criterion 1 uses answer-only opponents rather than a symmetric table. Three copies of the
candidate would score 0.25 by symmetry and measure nothing; opponents that answer but never offer
let the candidate's offers, and only its offers, move the result.

## 2. The trading game

The engine already implements domestic trading (`engine/src/rules/trade.rs`): an offer of one bundle
for another, a response from each opponent in seat order, then the offerer confirms one acceptor or
cancels. The engine does not change.

**Ruling:** the trading game is `max_offers_per_turn: 2`, `max_trade_cards: 1`, so only 1-for-1
offers (20 offer ids). With 2 cards a side a hand can have about 150 legal offers, and an untrained
offer prior spread over 150 moves starves the rest of the search. 1-for-1 covers the commonest trades.
Raising `max_trade_cards` to 2 (adding 2-for-1 and 2-for-2) is the first follow-up once criterion 1
holds.

Name it in code and the CLI as the config `{"max_offers_per_turn": 2, "max_trade_cards": 1}`
(`alphasettler arena --trades`, alongside the existing `--no-trades`).

## 3. Search and bots

- **Offers are searched moves.** `search_actions` keeps offers when the searching bot may offer and
  drops them otherwise. Inside the tree the same rule applies to every seat: a bot that may offer
  assumes its opponents may too.
- **Answers and confirmations are searched.** `IsmctsBot` no longer auto-rejects and auto-cancels. A
  `TradeResponse` or `TradeConfirm` decision is searched like any other, from the mover's own value,
  so a bot accepts exactly when its search says the trade raises its win chance. That holds for every
  searching bot, `ismcts` and `net:` alike. In games where nobody offers, the decision never comes
  up, so 3a's golden digests are unchanged.
- **`+t` lets a bot offer.** `net:NAME@N+t` (and `ismcts@N+t`) may offer; without `+t` a bot answers
  but never offers. The suffix combines with `+rD` and `+sK` in any order after `@N`.
- **World sampling at trade roots.** `WorldSampler` currently refuses `TradeResponse` and
  `TradeConfirm`. It samples them like any other phase. The pending trade is public in the
  observation, and the belief already conditions on `TradeOffered` (the offerer covers its give side)
  and on acceptances (the acceptor covers the get side). A rejection says nothing about the hand.

**Ruling:** no separate trade model. The policy head already has slots for every offer, answer and
confirmation id, and multi-player search with per-seat values already models "accept if it helps
me". A separate acceptance model is only worth building if the search's answers prove too slow or
too weak.

## 4. Network: encoder v3 and the upgrade

The encoder sees the trade phases but not the pending trade. Encoder v3 appends a trade block to the
globals:

| Features | Count |
|---|---|
| give bundle, per resource / `max_trade_cards` | 5 |
| get bundle, per resource / `max_trade_cards` | 5 |
| each seat's response (seat-relative): pending, accepted, rejected | 12 |
| offers made this turn / `max_offers_per_turn` (0 when offers are off) | 1 |
| offers allowed this game (`max_offers_per_turn > 0`) | 1 |

`NET_GLOBAL_WIDTH` goes from 91 to 115, `PUBLIC_LEN` from 1557 to 1581, and `ENCODING_HASH` changes,
so v2 files are refused with the existing message. The weights format stays 2 (its shapes follow the
widths).

**`alphasettler net-upgrade OLD NEW`** turns a v2-encoded net into a v3 one without changing its
moves in no-trade games:

- the global encoder gets zero weights for the 24 new inputs;
- the pooled head's rows for the 20 offer ids, the answer ids (accept, reject) and the confirmation
  ids get zero weights. Their biases are `OFFER_BIAS = -3.0` for offers and 0 for the rest. gen-4
  never trained those rows (offers were masked out and it never answered), so their weights are
  noise.

Test: an upgraded net gives the same outputs as the original on every no-trade position (within
the parity tolerance).

**Ruling:** upgrade rather than retrain. The trading net starts at gen-4's strength (0.585 against
AlphaBeta) instead of from scratch.

## 5. Training data and the loop

- **Offers become policy targets.** `Batch::push` keeps every legal move in the mask and the policy
  target, offers included. 3b's records were all played with offers off, so their batches do not
  change.
- **Answers and confirmations are recorded.** Self-play records each searched trade decision like any
  other (visits aligned with legal moves), so the net learns to answer.
- **The trading loop.** `fast-loop` gains `--trades` (the Section 2 config for self-play and gates,
  every self-play seat `+t`). Generation 1 self-plays with the upgraded gen-4. The window starts with
  3b's last generations, so a mix of no-trade and trade games, and turns over to trade games as
  generations accumulate. The gate is the fast loop's (100 seeds at 100 simulations, promote unless
  z <= −1), played in the trading game with both sides `+t`.

**Ruling:** every self-play seat trades. Trading conventions that only work among copies are a risk,
so done criterion 1 is measured against answer-only opponents, and criterion 2 against AlphaBeta,
not against copies.

## 6. Catanatron trading (criterion 2)

The oracle translation refuses domestic trades (`oracle/translate.py`, `oracle/events.py`). It gains:

- our `OfferTrade` to Catanatron's `OFFER_TRADE` (a 10-tuple of give and get counts);
- Catanatron's `DECIDE_ACCEPTEES` prompt back to our `TradeConfirm` phase, with each opponent's
  response, so our bot confirms or cancels;
- the pending trade and `offers_this_turn` in the snapshot `Game.from_snapshot` reads;
- the trade events (`TradeOffered`, `TradeResponded`, `TradeConfirmed`, `TradeCancelled`) for our
  bot's belief.

AlphaBeta never offers, so our bot is never asked to answer inside Catanatron. The Catanatron arena
gets `--trades`, which turns on the Section 2 rules for our bot's offers. Catanatron's own rules
accept any 1-for-1 offer.

## 7. Out of scope

Counter-offers and negotiation (the engine has neither), offers of more than one card a side (the
first follow-up), trading against humans, a separate opponent model.

## 8. Testing

- Engine: unchanged; `golden_trace` still passes.
- Search: a world sampled at a `TradeResponse` root keeps the pending trade and hands that cover it;
  `search_actions` keeps offers only when allowed.
- Bots: names with `+t`; a searching bot accepts an offer that wins it the game on the spot and
  rejects one that hands an opponent the win; answer-only bots never offer; 3a digests unchanged.
- Net: encoder v3 golden test updated; the upgrade preserves no-trade outputs; offers appear in the
  mask and target of a trade record's batch.
- Oracle: a scripted Catanatron game where our bot offers, AlphaBeta answers and our bot confirms;
  the snapshot and event translation round-trip.
- Python: `fast-loop --trades` runs two tiny generations; `arena --trades` sets the config.
