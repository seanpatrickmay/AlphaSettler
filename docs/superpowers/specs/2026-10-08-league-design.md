# Sub-project 3d-3: a league of past nets and scripted styles in self-play

Date: 2026-10-08. Builds on the fast loop (`alphasettler/train/fastloop.py`), 3d-1b
(`--opening`) and 3d-2 (encoder v4); the user asked (2026-10-07) for training "against an ensemble
of styles" after the AlphaStar discussion, and (2026-10-08) for all of 3d to be done. Decisions the
user did not make are marked **Ruling**.

## 1. Why

Pure self-play trains the net against copies of itself, so it learns to beat its own habits. The
hand-played game showed habits a human exploits at once (no cities, no trades, roads that cannot
become the longest road); a net that has only ever met itself never sees an opponent who plays
differently. A league mixes other opponents into self-play: past nets (old habits the current net
must still beat) and scripted styles (strategies the self-play population may never discover).

## 2. Done criteria

1. **League self-play:** `fast-loop` gains `--league` (kept from set-up like `--trades`,
   `--opening`). With it, each self-play game is, independently with probability 1/2, either plain
   self-play (four learner seats, as now) or a league game: two learner seats and two league seats,
   the seats drawn uniformly from the four positions, each league seat's bot drawn uniformly from
   the pool (Section 3). Only learner seats' decisions are recorded (league bots' moves are no
   policy targets); every game's result is a value target as now.
2. **The test, at equal wall time:** arm E: a `fast-loop` with the settings of the best arm so far
   (the control: arm D if 3d-2 was H1, else 3d-2's control), plus `--league`, the same self-play
   seeds, 10,800 s of generations after an untimed set-up. Then `net:E@100+t[+o]` against three
   copies of the control's best (`alphasettler sprt --trades`, p0 0.25, p1 0.30, alpha = beta =
   0.05, fresh seeds from 6,000,000,000). H1: the league becomes the default for new loops.
3. **Robustness, recorded either way:** E's best and the control's best each play 200 fixed fresh
   seeds (from 6,001,000,000) as one seat against three bots drawn from the style half of the pool
   (Section 3), `arena`, reported with z vs 0.25. The question: does the league make the net harder
   to exploit by styles it would not meet in self-play?
4. **Results** in `docs/bot/results-3d3.md`: arm E's generation table beside the control's, the
   share of league games and of each pool member, the SPRT line, the robustness lines.

## 3. The pool

- **Past nets:** the trading loop's `gen-030` .. `gen-034` and the control arm's promoted
  generations at set-up (all at the loop's encoder; a v3 net is upgraded by `net-upgrade` if the
  loop runs v4), plus every net the arm itself promotes, as it promotes it. Each plays as
  `net:NAME@100+t[+o]` (the fast budget, so league seats cost what a fast search costs).
- **Scripted styles:** `ismcts@100` with the heuristic evaluator under three weight sets that
  exaggerate one plan each, over the eight heuristic features (`bots/src/heuristic.rs`:
  vp, production, hand, over_limit, dev_cards, road_length, knights, settlement_spots):
  *roads* (road_length and settlement_spots tripled), *army* (dev_cards and knights tripled),
  *production* (production tripled). Each is `+t` in the trading game.
- **Draws:** half the league seats are past nets, half styles (each half uniform), so the scripted
  styles are a quarter of league seats however many nets the pool holds.

**Ruling:** styles are heuristic ISMCTS bots with tilted weights, not hand-scripted policies:
they reuse the engine-legal search the project trusts, play coherent games, and differ from the
net in kind. They are far weaker than the net (gen-4 beat `ismcts@1000` at 0.48), which is why they
are a quarter of league seats, not half. Cost if wrong: styles too weak to teach anything; the
robustness line shows it.

**Ruling:** league seats search at the fast budget (100 simulations) whatever the learner's
exploration: their moves are not policy targets, so a full search would buy nothing.

## 4. Out of scope

Exploiter agents trained against the learner (AlphaStar's main exploiters), prioritised
(win-rate-weighted) opponent sampling, per-opponent value heads, recording league seats' decisions,
Gumbel self-play with a league (refused, as with `--opening`), 3d-4.

## 5. Testing

- Rust: a league game seats exactly two learner bots and two pool bots at seeded positions,
  records decisions only for learner seats, replays exactly per seed and thread count, and has
  `belief_resets` 0; without a league every pinned self-play digest is unchanged; a style's weights
  are the default weights with exactly its features tripled.
- Python: `fast-loop --league` runs two tiny generations with a two-net pool, adds the promoted net
  to the pool, records the league share in the generation's state, refuses `--league` on resume of
  a loop set up without it (and the reverse), and refuses it with `--gumbel`.
