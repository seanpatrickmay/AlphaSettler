"""Domestic trades inside Catanatron (spec 3c Section 6): our offer, Catanatron's answers (AlphaBeta
by its own search), our confirmation, and the snapshot and events that carry them to our bot."""

import subprocess
import sys

import pytest

pytest.importorskip("catanatron")

from catanatron import Color, Game as CatanGame, Player  # noqa: E402
from catanatron.models.actions import generate_playable_actions  # noqa: E402
from catanatron.models.decks import freqdeck_add, freqdeck_subtract  # noqa: E402
from catanatron.models.enums import Action, ActionPrompt, ActionRecord, ActionType  # noqa: E402
from catanatron.players import AlphaBetaPlayer  # noqa: E402
from catanatron.state_functions import (  # noqa: E402
    get_player_freqdeck, player_freqdeck_add, player_freqdeck_subtract,
)

from alphasettler import describe_action  # noqa: E402
from alphasettler.arena import TRADES  # noqa: E402
from oracle.arena import AlphaSettlerPlayer, _force_seating, let_alphabeta_answer_offers, run_match  # noqa: E402
from oracle.events import record_events, redact  # noqa: E402
from oracle.translate import (  # noqa: E402
    ACCEPT_TRADE, CANCEL_TRADE, REJECT_TRADE, bundle, bundle_index, catan_snapshot, compare, confirm_trade, is_offer,
    offer_trade, offer_value, our_game, our_steps,
)


class StubBot:
    def __init__(self, result):
        self.result = result

    def act(self, game):
        return self.result


class Accepting(Player):
    """Accepts every offer it can; otherwise plays Catanatron's first playable action."""

    def decide(self, game, playable_actions):
        return next((a for a in playable_actions if a.action_type == ActionType.ACCEPT_TRADE), playable_actions[0])


def _set_hand(state, color, hand):
    old = get_player_freqdeck(state, color)
    player_freqdeck_subtract(state, color, old)
    state.resource_freqdeck = freqdeck_add(state.resource_freqdeck, old)
    player_freqdeck_add(state, color, hand)
    state.resource_freqdeck = freqdeck_subtract(state.resource_freqdeck, hand)


def _trading_table(seed=5):
    """Seat 0 is ours (a stub bot), on its turn after a roll of 6, holding one wood. Seat 1
    (AlphaBeta) and seat 2 (accepts everything) hold a brick each; seat 3 (AlphaBeta) holds nothing."""
    ours = AlphaSettlerPlayer(Color.RED, "random", 1, config=TRADES)
    players = [ours, AlphaBetaPlayer(Color.BLUE), Accepting(Color.ORANGE), AlphaBetaPlayer(Color.WHITE)]
    game = CatanGame(players, seed=seed)
    _force_seating(game, players)
    while game.state.is_initial_build_phase:
        game.play_tick(decide_fn=lambda player, g, actions: actions[0])
    st = game.state
    assert st.current_color() == Color.RED and st.current_prompt == ActionPrompt.PLAY_TURN
    roll = Action(Color.RED, ActionType.ROLL, None)
    game.execute(roll, action_record=ActionRecord(action=roll, result=(2, 4)))
    for color, hand in zip(st.colors, ([1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 0, 0, 0])):
        _set_hand(st, color, hand)
    game.playable_actions = generate_playable_actions(st)
    return ours, game


def test_trade_ids_match_the_engine():
    for g in range(20):
        for w in range(20):
            assert describe_action(offer_trade(g, w)) == f"OfferTrade {{ give: {g}, get: {w} }}"
    assert [describe_action(a) for a in (ACCEPT_TRADE, REJECT_TRADE, CANCEL_TRADE)] == [
        "AcceptTrade", "RejectTrade", "CancelTrade"]
    assert [describe_action(confirm_trade(p)) for p in range(4)] == [f"ConfirmTrade({p})" for p in range(4)]
    assert [bundle_index(bundle(i)) for i in range(20)] == list(range(20))
    # Bundle 11 is PAIRS[6] = (brick, sheep): a wood for a brick and a sheep.
    assert offer_value(offer_trade(0, 11)) == (1, 0, 0, 0, 0, 0, 1, 1, 0, 0)


def test_our_offer_is_answered_by_alphabeta_and_confirmed_by_us():
    let_alphabeta_answer_offers()
    ours, game = _trading_table()
    st = game.state
    mirror = our_game(0, game, TRADES)  # our engine, stepped alongside Catanatron
    offer = offer_trade(0, 1)  # one wood for one brick
    assert offer in mirror.legal_actions()
    ours.bot = StubBot(offer)
    chosen = ours.decide(game, game.playable_actions)
    assert chosen == Action(Color.RED, ActionType.OFFER_TRADE, offer_value(offer)) and ours.fallbacks == 0
    events = []
    before = catan_snapshot(game)

    def step(action):
        nonlocal before
        record = game.execute(action)
        for a, chance in our_steps(st, record):
            if chance is None:
                mirror.apply(a)
            else:
                mirror.apply_forced(a, chance)
        after = catan_snapshot(game)
        events.extend(record_events(before, after, record, st))
        before = after

    step(chosen)
    while st.current_prompt == ActionPrompt.DECIDE_TRADE:  # seats 1, 2, 3, in our order too
        step(st.current_player().decide(game, game.playable_actions))
    assert st.current_prompt == ActionPrompt.DECIDE_ACCEPTEES and st.acceptees[2]
    imported = our_game(0, game, TRADES)
    assert imported.phase == mirror.phase == "trade_confirm"
    assert imported.observation(0) == mirror.observation(0)
    assert {confirm_trade(2), CANCEL_TRADE} <= set(imported.legal_actions())
    ours.bot = StubBot(confirm_trade(2))
    confirm = ours.decide(game, game.playable_actions)
    assert confirm.action_type == ActionType.CONFIRM_TRADE and confirm.value[10] == Color.ORANGE
    step(confirm)
    assert ours.fallbacks == 0
    assert get_player_freqdeck(st, Color.RED) == [0, 1, 0, 0, 0]
    assert compare(mirror.snapshot(), catan_snapshot(game)) == []
    assert [e["type"] for e in events] == ["trade_offered"] + ["trade_responded"] * 3 + ["trade_confirmed"]
    for viewer in range(4):
        assert mirror.log(viewer) == [redact(e, viewer) for e in events], viewer


def test_offers_nobody_can_take_are_cancelled_and_a_turn_allows_two():
    ours, game = _trading_table()
    st = game.state
    for n in (1, 2):
        ours.bot = StubBot(offer_trade(0, 4))  # a wood for an ore nobody holds
        before = catan_snapshot(game)
        record = game.execute(ours.decide(game, game.playable_actions))
        events = record_events(before, catan_snapshot(game), record, st)
        while st.current_prompt == ActionPrompt.DECIDE_TRADE:
            before = catan_snapshot(game)
            record = game.execute(st.current_player().decide(game, game.playable_actions))
            events += record_events(before, catan_snapshot(game), record, st)
        assert [e["type"] for e in events] == ["trade_offered"] + ["trade_responded"] * 3 + ["trade_cancelled"]
        assert events[-1] == {"type": "trade_cancelled", "player": 0}
        assert st.current_prompt == ActionPrompt.PLAY_TURN
        assert catan_snapshot(game)["offers_this_turn"] == n
    assert not any(is_offer(a) for a in our_game(0, game, TRADES).legal_actions()), "two offers a turn"
    assert ours.fallbacks == 0


def test_a_trading_bot_plays_whole_games_inside_catanatron(tmp_path):
    records = run_match("ismcts@10+t", "random", seeds=1, workers=4, trades=True)
    assert len(records) == 4
    assert all(r["fallbacks"] == 0 and r["belief_resets"] == 0 for r in records), records
    out = tmp_path / "c.jsonl"
    r = subprocess.run([sys.executable, "-m", "alphasettler", "arena", "--candidate", "ismcts@10+t", "--baseline",
                        "catanatron:random", "--seeds", "1", "--trades", "--out", str(out)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "fallbacks: 0" in r.stdout and "belief resets: 0" in r.stdout


def _trading_table_at(ours_seat, seed=5):
    """`_trading_table` with our bot in `ours_seat`: the seats before ours roll and end their turns."""
    others = iter([AlphaBetaPlayer(Color.BLUE), Accepting(Color.ORANGE), AlphaBetaPlayer(Color.WHITE)])
    ours = AlphaSettlerPlayer(Color.RED, "random", 1, config=TRADES)
    players = [ours if i == ours_seat else next(others) for i in range(4)]
    game = CatanGame(players, seed=seed)
    _force_seating(game, players)
    while game.state.is_initial_build_phase:
        game.play_tick(decide_fn=lambda player, g, actions: actions[0])
    st = game.state
    while True:
        color = st.current_color()
        assert st.current_prompt == ActionPrompt.PLAY_TURN
        roll = Action(color, ActionType.ROLL, None)
        game.execute(roll, action_record=ActionRecord(action=roll, result=(2, 4)))
        if color == Color.RED:
            break
        game.execute(Action(color, ActionType.END_TURN, None))
    hands = iter(([0, 1, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 0, 0, 0]))
    for color in st.colors:
        _set_hand(st, color, [1, 0, 0, 0, 0] if color == Color.RED else next(hands))
    game.playable_actions = generate_playable_actions(st)
    return ours, game


@pytest.mark.parametrize("ours_seat", [1, 3])
def test_catanatron_asking_the_offerer_to_answer_its_own_offer_is_declined_unseen(ours_seat):
    """Catanatron 3.3 picks the next responder by skipping only the one who just answered, so an
    offerer outside seat 0 is asked about its own offer. We decline it without a fallback, and it
    is no answer of ours: no event and no step of our engine. (Catanatron then asks in seat-index
    order, ours in seat order after the offerer, so no mirror is stepped answer by answer here.)"""
    let_alphabeta_answer_offers()
    ours, game = _trading_table_at(ours_seat)
    st = game.state
    ours.bot = StubBot(offer_trade(0, 1))  # one wood for one brick
    events, before, asked = [], catan_snapshot(game), 0

    def step(action):
        nonlocal before
        record = game.execute(action)
        after = catan_snapshot(game)
        events.extend(record_events(before, after, record, st))
        before = after
        return record

    step(ours.decide(game, game.playable_actions))
    while st.current_prompt == ActionPrompt.DECIDE_TRADE:
        if st.current_player() is ours:
            asked += 1
            assert our_steps(st, step(ours.decide(game, game.playable_actions))) == []
        else:
            step(st.current_player().decide(game, game.playable_actions))
    assert asked == 1 and ours.fallbacks == 0
    assert st.current_prompt == ActionPrompt.DECIDE_ACCEPTEES and not st.acceptees[ours_seat]
    imported = our_game(0, game, TRADES)
    assert imported.phase == "trade_confirm"
    assert set(imported.legal_actions()) == {CANCEL_TRADE} | {confirm_trade(q) for q in range(4) if st.acceptees[q]}
    ours.bot = StubBot(CANCEL_TRADE)
    step(ours.decide(game, game.playable_actions))
    assert ours.fallbacks == 0 and st.current_prompt == ActionPrompt.PLAY_TURN
    imported.apply(CANCEL_TRADE)
    assert compare(imported.snapshot(), catan_snapshot(game)) == []
    assert [e["type"] for e in events] == ["trade_offered"] + ["trade_responded"] * 3 + ["trade_cancelled"]
    assert sorted(e["player"] for e in events if e["type"] == "trade_responded") == [
        q for q in range(4) if q != ours_seat]


class Offering:
    """Our bot, except that it offers whenever an offer is legal (one at random): whole games then
    go through offers, answers (Catanatron's), confirmations or cancels (our bot's search)."""

    def __init__(self, bot, rng):
        self.bot, self.rng = bot, rng

    def act(self, game):
        offers = [a for a in game.legal_actions() if is_offer(a)]
        return self.rng.choice(offers) if game.phase == "main" and offers else self.bot.act(game)

    def __getattr__(self, name):
        return getattr(self.bot, name)


@pytest.mark.parametrize("seed, seat, baseline", [(11, 0, "value"), (12, 1, "value"), (13, 2, "value"),
                                                  (14, 3, "value"), (15, 1, "alphabeta")])
def test_whole_games_with_offers_play_through_the_arena_loop(monkeypatch, seed, seat, baseline):
    """play_game's own loop, our bot offering at every chance from each seat: Catanatron's answers
    (AlphaBeta's by its search, and the self-answers it asks of offerers in seats 1-3) cost no
    fallback, and the trade events reach our bot's belief without a reset."""
    import random

    import oracle.arena as arena

    seen = []
    init, observe = AlphaSettlerPlayer.__init__, AlphaSettlerPlayer.observe

    def offering_init(self, *args, **kwargs):
        init(self, *args, **kwargs)
        self.bot = Offering(self.bot, random.Random(seed))

    def recording_observe(self, state, events):
        seen.extend(e["type"] for e in events)
        observe(self, state, events)

    monkeypatch.setattr(AlphaSettlerPlayer, "__init__", offering_init)
    monkeypatch.setattr(AlphaSettlerPlayer, "observe", recording_observe)
    r = arena.play_game(seed, "ismcts@10+t", baseline, seat, trades=True)
    assert seen.count("trade_offered") > 0 and seen.count("trade_responded") == 3 * seen.count("trade_offered")
    assert r["fallbacks"] == 0 and r["belief_resets"] == 0, r
