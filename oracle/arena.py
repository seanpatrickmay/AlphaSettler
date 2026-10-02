"""AlphaSettler bots playing inside Catanatron against Catanatron's bots.

The candidate is rotated through every seat, as in the native arena. The board depends only
on the seed (seating is forced after Catanatron draws it, so every rotation sees the same
map). Dice, steals and Catanatron's bots share one random stream, so common random numbers
hold only for the board and the early rolls (spec, Section 3). AlphaBeta has a 20 s
wall-clock search limit, so its games are not exactly reproducible.
"""

from __future__ import annotations

import multiprocessing as mp
import os

import catanatron.game as catan_game
from catanatron import Color, Game as CatanGame, Player, RandomPlayer
from catanatron.models.actions import generate_playable_actions
from catanatron.players import AlphaBetaPlayer, ValueFunctionPlayer, WeightedRandomPlayer

from alphasettler import Bot, Game
from oracle import hash_seed_zero
from oracle.events import record_events, redact
from oracle.translate import (
    SETUP_TURNS, Untranslatable, action_token, catan_snapshot, move_robber, seat, steal_from,
)

COLORS = [Color.RED, Color.BLUE, Color.ORANGE, Color.WHITE]
CONFIG = {"max_offers_per_turn": 0}
BASELINES = {
    "random": RandomPlayer,
    "weighted": WeightedRandomPlayer,
    "value": ValueFunctionPlayer,
    "alphabeta": AlphaBetaPlayer,
}


class AlphaSettlerPlayer(Player):
    """Plays an AlphaSettler bot. Each decision imports Catanatron's position into our engine,
    asks the bot (which sees only its own observation), and maps the choice back. A choice
    Catanatron does not offer, or a position that will not import, falls back to Catanatron's
    first playable action and is counted in ``fallbacks``."""

    def __init__(self, color, bot_name: str, seed: int):
        super().__init__(color)
        self.bot = Bot(bot_name, seed)
        self.seed = seed
        self.fallbacks = 0

    def decide(self, game, playable_actions):
        by_token = {}
        for a in playable_actions:
            try:
                by_token[action_token(game.state, a)] = a
            except Untranslatable:  # an action we cannot express is simply not choosable
                pass
        try:
            ours = Game.from_snapshot(self.seed, catan_snapshot(game), CONFIG)
            a = self.bot.act(ours)
            if ours.phase == "move_robber":
                after = ours.copy()
                after.apply(a)
                victim = self.bot.act(after) - steal_from(0) if after.phase == "steal" else None
                token = ("robber", a - move_robber(0), victim)
            else:
                token = a
        except BaseException as e:  # noqa: BLE001 - a counted fallback, never fatal to the match
            # A position or choice we cannot translate, or an engine panic (PyO3 raises
            # PanicException, a BaseException). Interrupts still stop the match.
            if isinstance(e, (KeyboardInterrupt, SystemExit)):
                raise
            token = None
        choice = by_token.get(token)
        if choice is None:
            self.fallbacks += 1
            return playable_actions[0]
        return choice

    def observe(self, state, events: list[dict]) -> None:
        """Show the bot the events of one executed action, redacted for its seat."""
        viewer = seat(state, self.color)
        self.bot.observe(viewer, [redact(e, viewer) for e in events])


def _force_seating(game, players) -> None:
    st = game.state
    st.players = list(players)
    st.colors = tuple(p.color for p in players)
    st.color_to_index = {c: i for i, c in enumerate(st.colors)}
    game.playable_actions = generate_playable_actions(st)


def play_game(seed: int, candidate: str, baseline: str, candidate_seat: int) -> dict:
    players = [
        AlphaSettlerPlayer(COLORS[i], candidate, seed * 4 + i) if i == candidate_seat else BASELINES[baseline](COLORS[i])
        for i in range(4)
    ]
    game = CatanGame(players, seed=seed)
    _force_seating(game, players)
    # Catanatron's own Game.play loop (same condition, same end-of-game observer hook), with the
    # candidate shown the events of every executed action, its own included. The empty first
    # feed starts its game, as the native arena's feed does, so a candidate in seat 0 does not
    # make its opening decision with no history at all.
    me = players[candidate_seat]
    me.observe(game.state, [])
    before = catan_snapshot(game)
    while game.winning_color() is None and game.state.num_turns < catan_game.TURNS_LIMIT:
        game.play_tick()
        after = catan_snapshot(game)
        me.observe(game.state, record_events(before, after, game.state.action_records[-1], game.state))
        before = after
    for observer in game.observers:
        observer.after(game)
    st = game.state
    winner = game.winning_color()
    return {
        "seed": seed,
        "candidate_seat": candidate_seat,
        "winner": None if winner is None else seat(st, winner),
        "vp": [st.player_state[f"P{i}_ACTUAL_VICTORY_POINTS"] for i in range(4)],
        "turns": st.num_turns - SETUP_TURNS,
        "actions": len(st.action_records),
        "fallbacks": me.fallbacks,
        "belief_resets": me.bot.diagnostics().get("belief_resets", 0),
    }


def check(candidate: str, baseline: str, seeds: int, seed_start: int = 0) -> None:
    """Raise ValueError for anything `run_match` would reject, without playing a game."""
    if baseline not in BASELINES:
        raise ValueError(f"unknown Catanatron bot {baseline!r}; known: {sorted(BASELINES)}")
    Bot(candidate, 0)  # raises ValueError("unknown bot ...") for a bad name
    if seeds <= 0:
        raise ValueError("seeds must be positive")
    if seed_start < 0 or (seed_start + seeds) * 4 > 2**64:  # bot seeds are seed * 4 + seat, a u64
        raise ValueError(f"seeds {seed_start}..{seed_start + seeds} are outside the bot seed range")


def run_match(candidate: str, baseline: str, seeds: int, seed_start: int = 0, workers: int | None = None) -> list[dict]:
    check(candidate, baseline, seeds, seed_start)
    jobs = [(s, candidate, baseline, k) for s in range(seed_start, seed_start + seeds) for k in range(4)]
    with hash_seed_zero():
        pool = mp.get_context("spawn").Pool(min(workers or os.cpu_count() or 1, len(jobs)))
    with pool:
        records = pool.starmap(play_game, jobs)
    return sorted(records, key=lambda r: (r["seed"], r["candidate_seat"]))
