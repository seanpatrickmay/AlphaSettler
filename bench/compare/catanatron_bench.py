"""Random-player throughput for Catanatron, measured the same way as our engine:
ns per action and games per second on one thread.

Run via bench/compare/run.sh (which installs Catanatron into a gitignored venv).
Catanatron is GPL-3.0; it is imported at runtime here and never vendored.
"""

import random
import sys
import time

from catanatron import Color, Game, RandomPlayer


def main() -> None:
    games = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    colors = [Color.RED, Color.BLUE, Color.ORANGE, Color.WHITE]
    actions = 0
    start = time.perf_counter()
    for seed in range(games):
        random.seed(seed)
        game = Game([RandomPlayer(c) for c in colors], seed=seed)
        while game.winning_color() is None and game.state.num_turns < 1000:
            game.play_tick()
            actions += 1
    elapsed = time.perf_counter() - start
    print(
        f"catanatron 1 thread: {games} games in {elapsed:.2f}s = {games / elapsed:.1f} games/s, "
        f"{actions / games:.0f} actions/game, {elapsed * 1e9 / actions:.0f} ns/action"
    )


if __name__ == "__main__":
    main()
