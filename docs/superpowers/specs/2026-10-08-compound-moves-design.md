# Sub-project 3d-4: compound build moves in search

Date: 2026-10-08. Builds on the ISMCTS search (`search/src/search.rs`: `search_moves`, PUCT and
Gumbel roots), the bots' name suffixes (`+t`, `+sK`, `+rD`, `+g`, `+o`) and the loops' kept
settings (`--trades`, `--gumbel`, `--opening`, `--league`). The user asked (2026-10-07) for
compound moves and how to stop them blowing up the tree, and (2026-10-08) for all of 3d to be done.
Decisions the user did not make are marked **Ruling**.

## 1. Why

A bank or port trade is worth something only for what it lets the player build. Today the search
sees each maritime trade as its own move: to find "trade 4 wheat for a brick, then build a road" it
must pick the right trade among up to twenty, then the right build, two levels deep, with each
trade's value backed up from whatever the tree happened to try after it. With 100 simulations a
move, most of those lines are never tried, so the bot under-trades with the bank (the hand-played
game: no trades, few cities). A compound move puts the trade and the build it enables on one edge.

## 2. Compound moves

At a main-phase decision of the player to move (`Phase::Main`), the search's move set (with `+c`):

- keeps every legal action except bare maritime trades, and
- adds, for each build target that is not affordable now but is after at most two maritime trades,
  one compound move: those trades, then the build. Targets: each legal-by-position road edge,
  settlement node and city node, and a development card (the same position rules the engine's
  build legality uses, ignoring only the hand).
- The trades for a target are chosen deterministically: cover the target's shortfall one card at
  a time, each time giving the resource with the most cards to spare after the target's cost (ties
  to the lower resource index) at the player's best rate for it (`maritime_rate`), never giving a
  resource the target needs. A target that needs more than two trades, or whose trades the bank
  cannot fill, gets no compound move.

A compound move's prior is the net's prior of its first trade, split equally among the compound
moves that start with the same trade; the moves the search keeps take their priors as now; the
priors are renormalised over the move set. Inside the tree a compound edge applies its actions in
order (every one a legal engine action at that point; asserted).

**Ruling:** compounds replace bare maritime trades in search instead of joining them. The point is
a smaller, better-shaped tree: a trade that enables no build within two trades is rarely the best
move at 100 simulations, and the targets cover every build the player could make this turn. Cost if
wrong: the bot never trades with the bank merely to hold cards for a later turn; the SPRT measures
it.

**Ruling:** the prior comes from the first trade, a legal action the policy head was trained on,
not from the build's policy slot, which the head has only ever scored when the build was
affordable.

## 3. Playing a compound move

The bot plays a compound move's first action and keeps the rest as a plan. At its next decisions,
while every observed event since is its own planned action and the plan's next action is legal, it
plays that action without searching; anything else drops the plan and the bot searches as usual.

Self-play records a searched decision as now, with each compound move's visits added to its first
trade's policy id (so the policy learns to make the trade that leads to the build); plan steps
record no decision.

## 4. Bots, loops and done criteria

- `+c` is a bot-name suffix like the others (any order, at most once; `ismcts` and `net:` names).
  Without it nothing changes, so every recorded result and pinned digest stays valid.
- `fast-loop` and `train-loop` gain `--compound` (kept from set-up; self-play and both gate sides
  use `+c`). With `--gumbel` it is refused.

Done criteria:

1. **Play strength, no training:** `net:B@100+t+c` against three `net:B@100+t`, where B is the best
   net of the best arm so far (with `+o` on both sides exactly when that arm used `--opening`):
   `alphasettler sprt --trades`, p0 0.25, p1 0.30, alpha = beta = 0.05, fresh seeds from
   7,000,000,000. Recorded either way.
2. **Training at equal wall time:** arm F, the best arm's settings plus `--compound`, the same
   self-play seeds, 10,800 s of generations after an untimed set-up; then `net:F@100+t+c[+o]`
   against three copies of the best arm's best `@100+t+c[+o]` (both with `+c`, so the test is the
   training, not the search), fresh seeds from 7,001,000,000. H1: `--compound` becomes the default
   for new loops.
3. **Results** in `docs/bot/results-3d4.md`: the share of decisions with at least one compound move
   and the bank trades per game with and without `+c` (from the criterion-1 records), arm F's
   generation table beside its control's, both SPRT lines.

## 5. Out of scope

Compounds of more than two trades or of two builds, domestic offers inside compounds, compound
moves for opponents' decisions inside the tree (they search their own turns with their own move
sets, which include compounds when the bot has `+c`), changes to the engine's action space or the
policy head.

## 6. Testing

- Search: on hand-built positions, the compound moves are exactly the targets the rules above
  allow, with the trades the rule picks (a port rate, a tie, a bank shortage, a target needing three
  trades refused); no bare maritime trade is in a `+c` move set; priors sum to 1 and a shared first
  trade's prior is split; a compound edge applies legally; without `+c` the move set and every
  pinned digest are unchanged.
- Bots: `+c` names parse in any order and once; a `+c` bot plays a plan's steps unsearched and drops
  the plan on an unexpected event; a game with `+c` bots has `belief_resets` 0.
- Self-play and data: a compound's visits land on its first trade's policy id; plan steps record
  nothing; `fast-loop --compound` runs two tiny generations and refuses a resume without it.
