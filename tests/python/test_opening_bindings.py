"""Opening bindings (spec 3d-1 Section 5): the opening features, the Rust forward pass of an
opening model file, net_evaluate, and bots acting from several Python threads at once."""

import math
import struct
from array import array
from concurrent.futures import ThreadPoolExecutor

import pytest

from alphasettler import Bot, Game
from alphasettler._engine import (
    OPENING_FEATURES, OPENING_FILE, OPENING_FORMAT_VERSION, OPENING_HASH, encode, net_evaluate, net_forward,
    opening_features, opening_forward, write_random_net,
)
from alphasettler.arena import TRADES

SETTLEMENT = 25  # BuildSettlement(n) is action 25 + n


def setup_game(nodes, seed=0):
    """Seed `seed`'s game with settlements on `nodes` in setup order, each but the last followed by
    its first legal road."""
    g = Game(seed, TRADES)
    for k, n in enumerate(nodes):
        g.apply(SETTLEMENT + n)
        if k < len(nodes) - 1:
            g.apply(g.legal_actions()[0])
    return g


def model_bytes(hidden=8, own=0.0, version=OPENING_FORMAT_VERSION, feature_hash=OPENING_HASH):
    """A 78 -> hidden -> 4 opening model file: every weight 0, the own seat's output bias `own`."""
    shapes = [(hidden, OPENING_FEATURES), (4, hidden)]
    out = [b"ASOPEN\0\0", struct.pack("<IQI", version, feature_hash, len(shapes))]
    out += [struct.pack("<II", r, c) for r, c in shapes]
    for r, c in shapes:
        bias = [0.0] * r
        if r == 4:
            bias[0] = own
        out += [array("f", [0.0] * (r * c)).tobytes(), array("f", bias).tobytes()]
    return b"".join(out)


@pytest.fixture
def nets(tmp_path):
    d = tmp_path / "nets"
    d.mkdir()
    write_random_net(str(d / "tiny.bin"), 1, 8, 1)
    return d


def test_the_opening_constants():
    assert (OPENING_FEATURES, OPENING_FORMAT_VERSION, OPENING_FILE) == (78, 1, "opening.bin")
    assert 0 < OPENING_HASH < 2**64


def test_opening_features_of_a_setup_position():
    # Seed 0's board; seats 0 to 3 on nodes 9, 12, 18 and 41; seat 3 to place its road.
    g = setup_game([9, 12, 18, 41])
    assert (g.phase, g.current_actor) == ("setup_road", 3)
    f = opening_features(g, 3)
    assert len(f) == OPENING_FEATURES
    # r0 is seat 3 on node 41: wood 2 (1 pip), sheep 6 (5), wheat 9 (4); per resource / 15.
    assert f[0:5] == pytest.approx([1 / 15, 0, 5 / 15, 4 / 15, 0])
    # Settlements placed in all / 8, and seat 3 still has its second pick.
    assert f[64:66] == pytest.approx([4 / 8, 1.0])
    # The board's pips per resource / 25.
    assert f[73:78] == pytest.approx([7 / 25, 12 / 25, 18 / 25, 14 / 25, 7 / 25])
    # From seat 0's side seat 3 is r3.
    assert opening_features(g, 0)[15:20] == f[0:5]


def test_opening_features_refuse_positions_outside_the_setup_and_bad_seats():
    g = setup_game([9, 12, 18, 41, 31, 10, 4, 29])
    g.apply(g.legal_actions()[0])  # the last road: the setup is over
    assert g.phase == "pre_roll"
    with pytest.raises(ValueError, match="setup position"):
        opening_features(g, 0)
    with pytest.raises(ValueError, match="seat 4 out of range"):
        opening_features(Game(0, TRADES), 4)


def test_opening_forward_reads_a_model_file_as_the_bots_do(tmp_path):
    path = tmp_path / OPENING_FILE
    path.write_bytes(model_bytes(own=math.log(3.0)))
    # Logits [ln 3, 0, 0, 0] whatever the input: the own seat gets 3 / 6.
    for row in opening_forward(str(path), [[0.0] * OPENING_FEATURES, [1.0] * OPENING_FEATURES]):
        assert row == pytest.approx([0.5, 1 / 6, 1 / 6, 1 / 6])
    with pytest.raises(ValueError, match="row 1 has 77 features"):
        opening_forward(str(path), [[0.0] * OPENING_FEATURES, [0.0] * 77])
    stale = tmp_path / "stale.bin"
    stale.write_bytes(model_bytes(feature_hash=OPENING_HASH ^ 1))
    with pytest.raises(ValueError, match=r"stale\.bin.*opening feature hash"):
        opening_forward(str(stale), [[0.0] * OPENING_FEATURES])
    with pytest.raises(ValueError, match="cannot read opening model"):
        opening_forward(str(tmp_path / "missing.bin"), [[0.0] * OPENING_FEATURES])


def test_a_plus_o_bot_needs_a_valid_opening_model(nets):
    with pytest.raises(ValueError, match="cannot read opening model"):
        Bot("net:tiny@10+o", 0, nets_dir=str(nets))
    (nets / OPENING_FILE).write_bytes(model_bytes(version=2))
    with pytest.raises(ValueError, match="format version 2"):
        Bot("net:tiny@10+o", 0, nets_dir=str(nets))
    (nets / OPENING_FILE).write_bytes(model_bytes())
    g = Game(3, TRADES)
    # A model that scores every spot the same plays the first legal one.
    assert Bot("net:tiny@10+t+o", 0, nets_dir=str(nets)).act(g) == g.legal_actions()[0]
    # The Gumbel suffix composes with it: the model still decides, and the root search is Gumbel's.
    gb = Bot("net:tiny@10+g+t+o", 0, nets_dir=str(nets))
    assert gb.diagnostics()["gumbel_root"] == 1 and gb.act(g) == g.legal_actions()[0]


def softmax(xs):
    m = max(xs)
    e = [math.exp(x - m) for x in xs]
    return [x / sum(e) for x in e]


def test_net_evaluate_gives_the_value_head_by_absolute_seat(nets):
    g = Game(6, TRADES)
    for k in range(40):
        legal = g.legal_actions()
        g.apply(legal[(7 * k) % len(legal)])
    for viewer in range(4):
        p = net_evaluate(g, viewer, "tiny", str(nets))
        assert len(p) == 4 and sum(p) == pytest.approx(1.0, abs=1e-6)
    # The actor's view is the Rust forward pass of `encode` (the actor's features), rotated to
    # absolute seats.
    actor = g.current_actor
    public, hidden = encode(g)
    value = array("f")
    value.frombytes(net_forward(str(nets / "tiny.bin"), public, hidden)["value"])
    rel = softmax(list(value))
    want = [rel[(q - actor) % 4] for q in range(4)]
    assert net_evaluate(g, actor, "tiny", str(nets)) == pytest.approx(want, abs=1e-6)
    with pytest.raises(ValueError, match="bad net name"):
        net_evaluate(g, 0, "../tiny", str(nets))
    with pytest.raises(ValueError, match="cannot read"):
        net_evaluate(g, 0, "missing", str(nets))
    with pytest.raises(ValueError, match="viewer 4 out of range"):
        net_evaluate(g, 4, "tiny", str(nets))


def test_bots_act_from_several_threads_as_they_do_one_at_a_time(nets):
    """Bot.act releases the GIL while it decides; the moves still depend only on the seeds."""

    def setup(seed):
        g = Game(seed, TRADES)
        bots = [Bot("net:tiny@20", seed * 4 + p, nets_dir=str(nets)) for p in range(4)]
        moves = []
        for _ in range(16):
            a = bots[g.current_actor].act(g)
            g.apply(a)
            moves.append(a)
        return moves

    one_at_a_time = [setup(s) for s in range(6)]
    with ThreadPoolExecutor(3) as ex:
        assert list(ex.map(setup, range(6))) == one_at_a_time
