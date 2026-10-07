"""Play against AlphaSettler bots by hand, in Catanatron's web UI.

Catanatron's own web server stores each game in a database and rebuilds the players from their
registry keys on every request. Our bot keeps a belief built from the events of the whole game, so
it cannot be rebuilt that way. This server serves the endpoints the UI calls (`/api/players`,
`/api/games`, `/api/games/<id>/states/latest`, `/api/games/<id>/actions`) from games held in
memory, and shows our bots the events of every executed action, as `arena.play_game` does.

    python3 -m oracle.play --bot net:gen-004@1000 --nets runs/play/nets

then start Catanatron's UI (`npm start` in its `ui/` folder), which calls http://localhost:5001.
Seat "ALPHASETTLER" in up to three chairs; trades stay off (the UI cannot answer offers).
"""

from __future__ import annotations

import argparse
import random
import re
import sys
import threading
from dataclasses import dataclass, field

from catanatron import Color, Game as CatanGame
from catanatron.models.map import build_map
from catanatron.players.minimax import AlphaBetaPlayer
from catanatron.players.value import ValueFunctionPlayer
from catanatron.serialization import action_from_json, web_view
from flask import Flask, abort, jsonify, request

from alphasettler import Bot
from oracle.arena import CONFIG, AlphaSettlerPlayer
from oracle.events import record_events
from oracle.translate import catan_snapshot

OURS = "ALPHASETTLER"
# Our engine plays Catanatron's standard game only: four seats, the base map, 10 VP, discard at 8+.
STANDARD = {"map_template": "BASE", "vps_to_win": 10, "discard_limit": 7, "friendly_robber": False}
LOCAL_ORIGIN = re.compile(r"http://(localhost|127\.0\.0\.1)(:\d+)?")


class Human(ValueFunctionPlayer):
    """A seat played through the UI (Catanatron's `WebHumanPlayer`, which lives in its web package
    and needs that package's Flask extensions to import). The value function is only the fallback
    for the UI asking the server to move for it."""

    IS_BOT = False


# The seats besides ours, as Catanatron's web server names them; CATANATRON is its AlphaBeta at
# depth 2 with pruning.
SEATS = {
    "HUMAN": ("Human", False, Human),
    "CATANATRON": ("Catanatron", True, lambda c: AlphaBetaPlayer(c, AlphaBetaPlayer.Params(depth=2, prunning=True))),
}


@dataclass
class Table:
    game: CatanGame
    ours: list[AlphaSettlerPlayer]
    before: dict
    lock: threading.Lock = field(default_factory=threading.Lock)

    def step(self, action=None) -> None:
        """Play one action (the current bot's choice when `action` is None) and show it to our bots."""
        fallbacks = sum(p.fallbacks for p in self.ours)
        if action is None:
            self.game.play_tick()
        else:
            self.game.execute(action)
        after = catan_snapshot(self.game)
        events = record_events(self.before, after, self.game.state.action_records[-1], self.game.state)
        for p in self.ours:
            p.observe(self.game.state, events)
        self.before = after
        if sum(p.fallbacks for p in self.ours) > fallbacks:
            print(f"warning: an AlphaSettler seat fell back to Catanatron's first action "
                  f"(game {self.game.id}, action {len(self.game.state.action_records)})", file=sys.stderr)


def create_app(bot_name: str, nets_dir: str | None = None) -> Flask:
    Bot(bot_name, 0, nets_dir)  # a bad name or a missing net fails here, not at the first move
    app = Flask(__name__)
    tables: dict[str, Table] = {}
    app.extensions["alphasettler_tables"] = tables

    @app.after_request
    def allow_local_ui(response):
        origin = request.headers.get("Origin", "")
        if LOCAL_ORIGIN.fullmatch(origin):
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Access-Control-Allow-Headers"] = "Content-Type"
            response.headers["Access-Control-Allow-Methods"] = "GET, POST"
        return response

    @app.get("/api/players")
    def players():
        entries = [{"key": k, "name": name, "description": "", "is_bot": is_bot, "source": "builtin", "params": []}
                   for k, (name, is_bot, _) in SEATS.items()]
        entries.append({"key": OURS, "name": "AlphaSettler", "description": bot_name, "is_bot": True,
                        "source": "builtin", "params": []})
        return jsonify(entries)

    @app.post("/api/games")
    def create_game():
        body = request.get_json(silent=True)
        if not isinstance(body, dict) or not isinstance(body.get("players"), list):
            abort(400, description="Missing or invalid JSON body: 'players' key required")
        keys = [str(k).upper() for k in body["players"]]
        if not 2 <= len(keys) <= 4:
            abort(400, description="'players' must be a list with 2 to 4 entries")
        options = {k: body.get(k, v) for k, v in STANDARD.items()}
        if OURS in keys and (len(keys) != 4 or options != STANDARD):
            abort(400, description="AlphaSettler plays only the standard game: 4 players, the BASE "
                                   "map, 10 VP, discard limit 7, no friendly robber")
        unknown = sorted(set(keys) - set(SEATS) - {OURS})
        if unknown:
            abort(400, description=f"unknown players {unknown}; known: {[*SEATS, OURS]}")
        rng = random.SystemRandom()
        seated = [AlphaSettlerPlayer(c, bot_name, rng.getrandbits(62), nets_dir, CONFIG) if k == OURS
                  else SEATS[k][2](c) for k, c in zip(keys, list(Color))]
        game = CatanGame(players=seated, catan_map=build_map(options["map_template"]),
                         vps_to_win=options["vps_to_win"], discard_limit=options["discard_limit"],
                         friendly_robber=options["friendly_robber"])
        ours = [p for p in seated if isinstance(p, AlphaSettlerPlayer)]
        for p in ours:
            p.observe(game.state, [])  # starts its game, as the arena's empty first feed does
        tables[game.id] = Table(game, ours, catan_snapshot(game))
        return jsonify({"game_id": game.id})

    def table_of(game_id: str) -> Table:
        table = tables.get(game_id)
        if table is None:
            abort(404, description="Resource not found")
        return table

    def view(game):
        color = request.args.get("as")
        if color is not None and color.upper() not in [c.value for c in game.state.colors]:
            abort(400, description="'as' must be one of the seated colors")
        return jsonify(web_view(game, None if color is None else color.upper()))

    @app.get("/api/games/<game_id>/states/<state_index>")
    def state(game_id, state_index):
        table = table_of(game_id)
        if state_index not in ("latest", str(len(table.game.state.action_records))):
            abort(404, description="this server keeps only the latest state of a game")
        return view(table.game)

    @app.post("/api/games/<game_id>/actions")
    def act(game_id):
        table = table_of(game_id)
        with table.lock:
            game = table.game
            if game.winning_color() is None:
                body = request.get_json(silent=True)
                if game.state.current_player().is_bot or not body:
                    table.step()  # a bot's move, or (as Catanatron's server does) the UI asking for one
                else:
                    try:
                        action = action_from_json(body)
                    except (KeyError, IndexError, TypeError, ValueError):
                        abort(400, description="an action is [color, action_type, value]")
                    if action not in game.playable_actions:
                        abort(400, description="that action is not playable now")
                    table.step(action)
            return view(game)

    return app


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="python3 -m oracle.play", description=__doc__.split("\n\n")[0])
    parser.add_argument("--bot", default="net:gen-004@1000", help="our bot's name (default %(default)s)")
    parser.add_argument("--nets", default="runs/play/nets", help="the nets directory (default %(default)s)")
    parser.add_argument("--port", type=int, default=5001, help="the UI calls 5001 (default %(default)s)")
    args = parser.parse_args(argv)
    try:
        app = create_app(args.bot, args.nets)
    except ValueError as error:
        parser.exit(2, f"error: {error}\n")
    app.run(host="127.0.0.1", port=args.port, threaded=True)


if __name__ == "__main__":
    main()
