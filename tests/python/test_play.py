import pytest

pytest.importorskip("catanatron")
pytest.importorskip("flask")

from alphasettler._engine import write_random_net  # noqa: E402
from oracle.play import create_app  # noqa: E402

TABLE = ["HUMAN", "ALPHASETTLER", "CATANATRON", "ALPHASETTLER"]


@pytest.fixture
def client(tmp_path):
    write_random_net(str(tmp_path / "tiny.bin"), 1, 8, 1)
    app = create_app("net:tiny@4", str(tmp_path))
    app.config["TESTING"] = True
    return app.test_client(), app


def new_game(client, players=TABLE, **options):
    response = client.post("/api/games", json={"players": players, **options})
    assert response.status_code == 200, response.json
    return response.json["game_id"]


def test_a_whole_game_through_the_api_with_no_fallbacks_and_no_belief_resets(client):
    client, app = client
    game_id = new_game(client)
    state = client.get(f"/api/games/{game_id}/states/latest").json
    human = next(c for c in state["colors"] if c not in state["bot_colors"])
    for _ in range(5000):
        if state["winning_color"] is not None:
            break
        move = None if state["current_color"] in state["bot_colors"] else state["current_playable_actions"][0]
        response = client.post(f"/api/games/{game_id}/actions?as={human}", json=move)
        assert response.status_code == 200, response.json
        state = response.json
    assert state["winning_color"] is not None
    ours = app.extensions["alphasettler_tables"][game_id].ours
    assert len(ours) == 2
    assert [p.fallbacks for p in ours] == [0, 0]
    assert [p.bot.diagnostics().get("belief_resets", 0) for p in ours] == [0, 0]


def test_a_human_cannot_play_an_action_that_is_not_playable(client):
    client, _ = client
    game_id = new_game(client, ["HUMAN", "CATANATRON", "CATANATRON", "CATANATRON"])
    state = client.get(f"/api/games/{game_id}/states/latest").json
    while state["current_color"] in state["bot_colors"]:
        state = client.post(f"/api/games/{game_id}/actions").json
    human = state["current_color"]
    for bad in ([human, "BUILD_CITY", 0], [human, "NOT_AN_ACTION", None], ["PURPLE", "ROLL", None]):
        assert client.post(f"/api/games/{game_id}/actions", json=bad).status_code == 400


@pytest.mark.parametrize("players, options", [
    (["HUMAN", "ALPHASETTLER"], {}),
    (TABLE, {"map_template": "MINI"}),
    (TABLE, {"vps_to_win": 8}),
    (TABLE, {"friendly_robber": True}),
])
def test_our_bot_is_refused_outside_the_standard_game_with_a_reason(client, players, options):
    client, _ = client
    response = client.post("/api/games", json={"players": players, **options})
    assert response.status_code == 400
    assert "standard game" in response.json["error"]


def test_catanatrons_web_defaults_are_refused_with_a_reason(client):
    client, _ = client
    response = client.post("/api/games", json={"players": TABLE, "vps_to_win": 15, "discard_limit": 9,
                                                "friendly_robber": True})
    assert response.status_code == 400
    assert "10 VP" in response.json["error"]
    unknown = client.post("/api/games", json={"players": ["HUMAN", "WEIGHTED_RANDOM"]})
    assert unknown.status_code == 400 and "WEIGHTED_RANDOM" in unknown.json["error"]


def test_unknown_games_and_old_states_are_not_found(client):
    client, _ = client
    assert client.get("/api/games/nope/states/latest").status_code == 404
    game_id = new_game(client)
    assert client.get(f"/api/games/{game_id}/states/0").status_code == 200
    assert client.get(f"/api/games/{game_id}/states/7").status_code == 404


def test_only_a_local_ui_is_allowed_to_call_the_api(client):
    client, _ = client
    local = client.get("/api/players", headers={"Origin": "http://localhost:3000"})
    assert local.headers["Access-Control-Allow-Origin"] == "http://localhost:3000"
    assert {p["key"] for p in local.json} == {"HUMAN", "CATANATRON", "ALPHASETTLER"}
    elsewhere = client.get("/api/players", headers={"Origin": "http://evil.example"})
    assert "Access-Control-Allow-Origin" not in elsewhere.headers
