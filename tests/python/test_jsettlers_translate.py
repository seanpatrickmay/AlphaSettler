import pytest

from alphasettler import Game
from oracle.jsettlers import tables as T
from oracle.jsettlers.boards import native_board
from oracle.jsettlers.translate import Context, Untranslatable, dev_counts, import_game, snapshot

UNLIMITED = {"max_offers_per_turn": 0, "bank_limit": False}


def player(res=(0, 0, 0, 0, 0), dev=None, played=(0, 0, 0, 0), settlements=(), cities=(), roads=(), discard=0,
           played_dev=False):
    return {"res": list(res), "dev": list(dev or [0] * 10), "played": list(played),
            "settlements": [T.OUR_NODE_TO_JS_NODE[n] for n in settlements],
            "cities": [T.OUR_NODE_TO_JS_NODE[n] for n in cities],
            "roads": [T.OUR_EDGE_TO_JS_EDGE[e] for e in roads], "discard": discard, "played_dev": played_dev,
            "vp": len(settlements) + 2 * len(cities)}


def raw(state, current=0, players=None, deck=None, robber=None, board=None, longest=-1, largest=-1, winner=-1):
    board = board or native_board(3)
    desert = board["tile_resource"].index(None)
    return {"state": state, "current": current,
            "robber": T.OUR_TILE_TO_JS_HEX[desert] if robber is None else robber,
            "deck": [9] * 14 + [1, 1, 2, 2, 3, 3, 4, 5, 6, 7, 8] if deck is None else deck,
            "players": players or [player() for _ in range(4)],
            "longest": longest, "largest": largest, "winner": winner}


def test_the_opening_imports_with_our_player_0_on_jsettlers_first_seat():
    ctx = Context(native_board(3), first=2)
    s = snapshot(raw(T.START1A, current=2), ctx)
    assert (s["phase"], s["current"], s["setup_step"], s["turn"]) == ("setup_settlement", 0, 0, 0)
    assert s["bank"] == [19] * 5 and len(s["dev_deck"]) == 25
    g = import_game(3, raw(T.START1A, current=2), ctx, UNLIMITED)
    assert g.phase == "setup_settlement" and g.current_actor == 0
    assert (ctx.ours(2), ctx.js(0), ctx.ours(1), ctx.js(3)) == (0, 2, 3, 1)


def test_a_setup_road_position_finds_the_settlement_awaiting_its_road():
    ctx = Context(native_board(3), first=0)
    players = [player(settlements=[10]), player(), player(), player()]
    s = snapshot(raw(T.START1B, current=0, players=players), ctx)
    assert (s["phase"], s["setup_road_node"], s["setup_step"]) == ("setup_road", 10, 0)
    import_game(3, raw(T.START1B, current=0, players=players), ctx, UNLIMITED)


def test_hands_seats_cards_and_awards_map_to_ours():
    ctx = Context(native_board(3), first=1, turn=7, rolled=True)
    ctx.drawn = [0, 1, 0, 0, 0]
    dev = [0] * 10
    dev[9], dev[4], dev[1] = 2, 1, 1  # two knights, a VP card, a road building card
    players = [player() for _ in range(4)]
    players[1] = player(res=(1, 2, 3, 4, 5), dev=dev, played=(1, 0, 0, 0), played_dev=True)
    s = snapshot(raw(T.PLAY1, current=1, players=players, longest=1, largest=3), ctx)
    me = s["players"][0]
    assert me["hand"] == [5, 1, 3, 4, 2]
    assert me["dev_hand"] == [2, 1, 1, 0, 0] and dev_counts(dev) == [2, 1, 1, 0, 0]
    assert me["dev_new"] == [0, 1, 0, 0, 0] and s["players"][1]["dev_new"] == [0] * 5
    assert me["dev_played"] == [1, 0, 0, 0, 0] and me["knights_played"] == 1
    assert (s["current"], s["phase"], s["turn"], s["return_phase"]) == (0, "main", 7, "main")
    assert (s["longest_road_owner"], s["largest_army_owner"], s["dev_played_this_turn"]) == (0, 2, True)
    assert s["bank"] == [14, 18, 16, 15, 17]


def test_more_than_19_cards_import_only_without_the_bank_limit():
    ctx = Context(native_board(3), first=0, rolled=True)
    players = [player(res=(0, 0, 0, 0, 12)), player(res=(0, 0, 0, 0, 9)), player(), player()]
    s = snapshot(raw(T.PLAY1, players=players), ctx)
    assert s["bank"][0] == 0 and s["players"][0]["hand"][0] == 12
    import_game(3, raw(T.PLAY1, players=players), ctx, UNLIMITED)
    with pytest.raises(ValueError, match="hands and bank hold"):
        import_game(3, raw(T.PLAY1, players=players), ctx, {"max_offers_per_turn": 0})


@pytest.mark.parametrize("state,phase,extra", [
    (T.ROLL_OR_CARD, "pre_roll", {}), (T.PLACING_ROBBER, "move_robber", {}),
    (T.PLACING_FREE_ROAD1, "road_building", {"roads_left": 2}), (T.PLACING_FREE_ROAD2, "road_building", {"roads_left": 1}),
    (T.OVER, "game_over", {"winner": 3}),
])
def test_states_map_to_our_phases(state, phase, extra):
    ctx = Context(native_board(3), first=1)
    s = snapshot(raw(state, current=1, winner=0), ctx)
    assert s["phase"] == phase and all(s[k] == v for k, v in extra.items())


@pytest.mark.parametrize("state", [T.PLACING_ROAD, T.PLACING_SETTLEMENT, T.PLACING_CITY, T.WAITING_FOR_DISCOVERY,
                                   T.WAITING_FOR_MONOPOLY, 0, 1])
def test_intermediate_states_are_untranslatable(state):
    with pytest.raises(Untranslatable) as e:
        snapshot(raw(state), Context(native_board(3), first=0))
    assert e.value.kind == "intermediate-state"


def test_discards_owed_appear_only_in_the_discard_phase():
    ctx = Context(native_board(3), first=0, rolled=True)
    players = [player(res=(2, 2, 2, 2, 0), discard=4)] + [player() for _ in range(3)]
    s = snapshot(raw(T.WAITING_FOR_DISCARDS, players=players), ctx)
    assert s["phase"] == "discard" and s["players"][0]["discard_remaining"] == 4
    g = import_game(3, raw(T.WAITING_FOR_DISCARDS, players=players), ctx, UNLIMITED)
    assert g.phase == "discard" and g.current_actor == 0


def test_an_imported_position_compares_equal_to_its_own_translation():
    ctx = Context(native_board(3), first=1)
    theirs = snapshot(raw(T.START1A, current=1), ctx)
    assert import_game(3, raw(T.START1A, current=1), ctx, UNLIMITED).snapshot() == theirs
