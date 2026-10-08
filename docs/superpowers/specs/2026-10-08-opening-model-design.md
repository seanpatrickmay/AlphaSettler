# Sub-project 3d-1: opening diagnosis and a separate opening model

Date: 2026-10-08. Builds on 3c (`runs/3c`, trading net `gen-034`, the trading game
`{"max_offers_per_turn": 2, "max_trade_cards": 1}`).

3d is four improvements the user asked for after playing the bots by hand (2026-10-07) and beating
them: 3d-1 openings (this spec), 3d-2 road and endgame-threat features in the encoder, 3d-3 a league
of past nets and scripted styles, 3d-4 compound build moves. Each gets its own spec, plan and build.
The user put 3d ahead of 3c's remaining measurement (Task 6) and chose to build and measure 3d in
the trading game from `gen-034`. Decisions the user did not make are marked **Ruling**, with the
reason, so they can be overridden.

## 1. Why

In gen-4's 10,600 self-play games (42,400 seats), openings that touch no wheat are 14% of all
openings and win 0.137 (all five resources: 0.407; average 0.25); at equal pips (18-20) the gap
holds, 0.134 against 0.377. No-ore openings are 23% and win 0.184. In the hand-played game, two bots
opened without wood and brick, or without wheat, and more search (8x, and `+s30`) chose the same
spots. The user won with a wheat/ore opening the bot would not have picked.

Two questions follow, and the user asked for both: is the bot forced into these openings or does it
choose them, and does it misjudge them (Part 1); and does a model built only to judge openings pick
better ones (Part 2)? The user's view: picking an opening is a different problem from picking a move
in a position.

## 2. Done criteria

1. **Diagnosis recorded** (`docs/bot/results-3d1.md`): the tables of Section 3 and one verdict line:
   *forced* (good alternatives were rarely available), *misjudged* (the value head overrates the
   weak openings), or *misdecided* (the value head rates them right, the search picks them anyway).
2. **Opening model trained and reported:** its held-out cross-entropy on setup positions, beside
   `gen-034`'s value head on the same positions (Section 4.4). Recorded either way; not a gate.
3. **The test** (the headline): `net:gen-034@100+t+o` against three `net:gen-034@100+t` in the
   trading game, by `alphasettler sprt --trades` (p0 0.25, p1 0.30, alpha = beta = 0.05, fresh
   seeds from 2,000,000,000). H1 is a success. If H1, confirm with a fixed run of 200 fresh seeds
   at 600 simulations (`arena --trades`), reported with its z. H0 is recorded as a negative result,
   with the opening rates of Section 3 (D4) for both bots.

## 3. Part 1: diagnosis (`alphasettler opening-diagnose`)

Inputs are recorded self-play games (JSONL, the `selfplay` record format), so no new self-play is
played: the trading loop's `runs/3c/gen-030` .. `gen-034` (each directory holds the games of the loop's
best net when it was played: `start`, the upgraded gen-4, then `gen-030` .. `gen-033`; all trading)
and, for comparison, `runs/3b-v2/gen-005` (played by gen-4, no trades).

An opening is a seat's two setup settlements. Its **group** is the first that applies: *no wood and
brick*, *no wheat*, *no ore*, *no wood or no brick*, *no sheep*, *all five* (Section 1's labels). Its
**pips** are the dice dots (2:1 ... 6:5, 8:5 ... 12:1) of every tile its two settlements touch.

- **D1, forced or chosen.** For each opening in a weak group (*no wheat*, *no ore*, *no wood and
  brick*) and each of its two picks: the chosen spot's pips, and the most pips among spots that were
  legal at that pick and touch the missing resource (for *no wood and brick*, wood or brick). Table:
  per group, the share of openings where such a spot was within 0, 1, 2 and 3 pips of the chosen one
  at either pick, and the share with none at all.
- **D2, misjudged.** A new binding evaluates a position with a net (Section 5). For every seat of
  every game, the net's win probability for that seat at the first `main`-phase position (setup
  done, before the first roll), against the seat's actual result. Table: per group, seats, mean
  predicted win probability, actual win rate, and the gap with its standard error. Calibration is
  read per group, not overall: the question is whether weak groups are overrated relative to
  *all five*.
- **D3, search budget.** On 200 games drawn from `runs/3c/gen-034`, re-choose each setup settlement
  with `gen-034` at 1,000 and 8,000 simulations from the recorded position (setup positions hide nothing,
  so a fresh bot is exact). Table: how often the choice changes, and the weak-group rate of the
  re-chosen openings at each budget.
- **D4, a bot's openings.** For a bot name and N seeds, play only the setup (four copies of the bot)
  and report the group shares and mean pips. Used in Section 2 criterion 3 to compare `+o` with the
  baseline; also on `gen-034@100+t`, `@1000` and `@100+t+o`.

**Verdict rule** (the line in the results file): *forced* if, in the weak groups, a spot within 1 pip
touching the missing resource was available in under a third of openings; otherwise *misjudged* if
D2's gap for the weak groups is more than 2 standard errors above *all five*'s; otherwise
*misdecided*.

## 4. Part 2: the opening model

### 4.1 What it decides

The bot's setup settlements (`Phase::SetupSettlement`): both of each seat's picks. Setup roads, and
every later decision, stay with the net and its search.

**Ruling:** settlements only. The road after each settlement chooses an expansion direction and
matters, but the evidence (Section 1) is about which resources an opening touches, which is the
settlement choice; roads are a follow-up if the model helps.

### 4.2 Inputs (`OPENING_FEATURES` = 78, seat-relative)

Computed in Rust (`bots/src/opening.rs`) from a position, for a perspective seat `p` (the seat that
just picked); seats are named relative to `p` in turn order (0 = `p`, 1 = next, ...), as the net's
inputs name them. One function serves training (through a binding) and play, so they cannot
disagree.

| Features | Count |
|---|---|
| per seat (x4): pips per resource (wood, brick, sheep, wheat, ore) / 15 | 20 |
| per seat: total pips / 30 | 4 |
| per seat: resources touched / 5 | 4 |
| per seat: distinct numbers touched / 6 | 4 |
| per seat: ports held, one each for 3:1 and the five 2:1 kinds | 24 |
| per seat: settlements placed / 2 | 4 |
| per seat: the best spot still legal at road distance 2 from its settlements, pips / 15 (0 if none) | 4 |
| settlements placed in all / 8 | 1 |
| `p` still has a pick to make | 1 |
| per resource: the most pips of that resource on any spot still legal / 5 | 5 |
| the most pips of any spot still legal / 15 | 1 |
| spots still legal with at least 10 pips / 10 | 1 |
| per resource: that resource's pips on the whole board / 25 | 5 |

"Still legal" is the distance rule against every settlement placed. `OPENING_HASH` is a hash of the
feature list's names and order, written into the model file, so a model built for other features is
refused.

**Ruling:** hand-built features and a small MLP rather than a second graph net: the suspected blind
spot is that "which resources do my two settlements cover" is a whole-board property, and these
inputs state it outright.

### 4.3 Model and file

An MLP: 78 inputs, two hidden layers of 64 (ReLU), 4 outputs, softmax over seats (seat-relative,
like the net's value head). The file `<nets_dir>/opening.bin`: magic `ASOPEN\0\0`, format version 1,
`OPENING_HASH`, layer count, each layer's rows and columns, then each layer's f32 weights (row-major,
rows = outputs) and biases, little-endian. The Rust loader checks the magic, version, hash, shapes
(78 in, 4 out, each layer's inputs equal to the previous outputs), exact length and that every float
is finite, and refuses anything else with an error naming the file.

### 4.4 Training (`alphasettler opening-train`)

- **Samples:** every recorded game's setup replayed; after each settlement is placed (before its
  road), the features for the seat that placed it, and the target: the winner relative to that seat
  (a game with no winner adds nothing). Eight samples a game.
- **Data:** `runs/3b-v2/gen-001` .. `gen-005` and `runs/3c/gen-030` .. `gen-034`, about 63,000
  games, about 500,000 samples. **Ruling:** both the no-trade and the trading games: the opening's
  effect on who wins barely depends on whether players trade, and the trading games alone are
  10,000. **Ruling:** not `runs/3b-v2/gen-000` (100,000 games), which the heuristic warm-start bot
  played: who wins from an opening depends on how the rest of the game is played, and that player
  is not ours. Held out: games whose seed % 20 == 0 (the convention of the fast tooling).
- **Fit:** cross-entropy, Adam (lr 1e-3, batch 1,024), until the held-out loss has not improved for
  3 epochs; the best held-out epoch is written.
- **Report:** held-out cross-entropy overall and on the trading games only, against two baselines on
  the same held-out samples: a uniform guess (ln 4 = 1.386) and `gen-034`'s value head on the same
  position (through the Section 5 binding).

Training runs on the PyTorch twin (`train` extra); a parity test checks the Rust forward pass
against PyTorch on random weights and the training features.

### 4.5 Choosing (`+o`)

`+o` is a bot-name suffix like `+t`, `+sK`, `+rD` (any order, at most once): `net:NAME@N+o` loads
`<nets_dir>/opening.bin` beside `NAME.bin`. At a setup-settlement decision, the bot places each legal
settlement on a copy of the position, computes the features for itself, and plays the one with the
highest model probability of its own win; ties go to the first in engine order. No search runs for
that decision. Everything else is unchanged.

**Ruling:** one pick deep, no look-ahead over the remaining draft: the training positions include
"settlements placed" and the spots left for later pickers, so the model's probability already
averages over how the draft usually goes. A draft search with the model at its leaves is the
follow-up if this works and the diagnosis says the draft matters.

**Ruling:** used at play time only. Self-play does not use `+o` in 3d-1; whether the loop should
(so the net trains on better openings) is decided after criterion 3.

## 5. Bindings and CLI

- `opening_features(game, seat) -> list[float]` (78) and `OPENING_FEATURES`, `OPENING_HASH`.
- `net_evaluate(game, viewer, net, nets_dir=None) -> list[float]`: the net's four win
  probabilities, by absolute seat, from `viewer`'s observation of `game`. Errors as `Bot(...)` does
  for a bad name or file.
- `alphasettler opening-diagnose --games DIR [DIR ...] --net NAME [--nets-dir D] [--audit N]
  [--out FILE]` prints D1-D3; `alphasettler opening-diagnose --bot NAME --seeds N` prints D4.
- `alphasettler opening-train --games DIR [DIR ...] --out PATH [--compare-net NAME --nets-dir D]`.

## 6. Out of scope

Setup roads, a draft search, using `+o` in self-play, retraining the main net on openings, 3d-2 to
3d-4.

## 7. Testing

- Rust: features of a hand-built setup position match hand-counted values (pips, ports, reach, the
  draft block); seat-relative rotation (the same position from two perspectives permutes the per-seat
  block); `OPENING_HASH` changes when a feature is renamed or moved; the model file's errors (bad
  magic, version, hash, shapes, length, a non-finite float); `+o` names parse in any order with the
  other suffixes and are refused when repeated; a `+o` bot picks the spot a hand-made model file
  scores highest and leaves non-settlement decisions to the search.
- Python: the parity test (Rust and PyTorch forward passes agree); `opening-train` on two tiny game
  directories writes a file the Rust loader accepts; the diagnosis on a tiny directory prints each
  table, with a hand-checked D1 row on a fixed game; `net_evaluate` returns four probabilities that
  sum to 1 and agrees with the bot's evaluator on a fixed position.
