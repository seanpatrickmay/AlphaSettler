import subprocess
import sys

import pytest

from alphasettler import Bot, Game
from alphasettler._engine import DEFAULT_NETS_DIR, run_match, write_random_net
from net_names import BAD_NET_NAMES

CONFIG = {"max_offers_per_turn": 0}


def cli(*args):
    return subprocess.run([sys.executable, "-m", "alphasettler", *args], capture_output=True, text=True)


@pytest.fixture
def nets(tmp_path):
    d = tmp_path / "nets"
    d.mkdir()
    write_random_net(str(d / "tiny.bin"), 1, 8, 1)
    stale = bytearray((d / "tiny.bin").read_bytes())
    stale[12] ^= 1
    (d / "stale.bin").write_bytes(bytes(stale))
    return d


def test_the_default_nets_dir():
    assert DEFAULT_NETS_DIR == "runs/3b/nets"


def test_a_net_bot_plays_legal_moves_from_python(nets):
    g = Game(4, CONFIG)
    bot = Bot("net:tiny@10", 1, nets_dir=str(nets))
    assert bot.name == "net"
    for _ in range(80):
        if g.is_over:
            break
        a = bot.act(g)
        assert a in g.legal_actions()
        g.apply(a)


def test_bad_names_missing_files_and_stale_hashes_are_errors(nets):
    for bad in BAD_NET_NAMES:
        with pytest.raises(ValueError, match="unknown bot"):
            Bot(bad, 0, nets_dir=str(nets))
    with pytest.raises(ValueError, match="cannot read"):
        Bot("net:missing", 0, nets_dir=str(nets))
    with pytest.raises(ValueError, match="encoding hash"):
        Bot("net:stale", 0, nets_dir=str(nets))
    with pytest.raises(ValueError, match="encoding hash"):
        run_match("net:stale", "greedy", 0, 1, 1, CONFIG, str(nets))
    with pytest.raises(ValueError, match="width 0 is out of range"):
        write_random_net(str(nets / "x.bin"), 1, 0, 1)
    with pytest.raises(ValueError, match="rounds 0 is out of range"):
        write_random_net(str(nets / "x.bin"), 1, 8, 0)
    mlp = bytearray((nets / "tiny.bin").read_bytes())
    mlp[8:12] = (1).to_bytes(4, "little")  # a version 1 (flat MLP) file
    (nets / "mlp.bin").write_bytes(bytes(mlp))
    with pytest.raises(ValueError, match="format version 1, expected 2"):
        Bot("net:mlp", 0, nets_dir=str(nets))


def test_cli_arena_and_compare_take_net_names(nets, tmp_path):
    out = tmp_path / "a.jsonl"
    r = cli("arena", "--candidate", "net:tiny@5", "--baseline", "greedy", "--seeds", "1", "--no-trades",
            "--nets-dir", str(nets), "--out", str(out))
    assert r.returncode == 0, r.stderr
    assert "belief resets: 0" in r.stdout and len(out.read_text().splitlines()) == 4
    r = cli("compare", "--a", "net:tiny@5", "--b", "greedy", "--baseline", "random", "--seeds", "1",
            "--no-trades", "--nets-dir", str(nets), "--out-dir", str(tmp_path / "cmp"))
    assert r.returncode == 0, r.stderr
    assert "paired net:tiny@5 - greedy:" in r.stdout


def test_cli_rejects_bad_net_names_and_files_before_writing_anything(nets, tmp_path):
    for bad, message in [*[(b, "unknown bot") for b in BAD_NET_NAMES], ("net:missing", "cannot read"),
                         ("net:stale", "encoding hash")]:
        out = tmp_path / "never.jsonl"
        r = cli("arena", "--candidate", bad, "--baseline", "greedy", "--seeds", "1", "--no-trades",
                "--nets-dir", str(nets), "--out", str(out))
        assert r.returncode == 2 and message in r.stderr, (bad, r.stderr)
        assert not out.exists()
        r = cli("compare", "--a", "greedy", "--b", bad, "--baseline", "random", "--seeds", "1",
                "--nets-dir", str(nets), "--out-dir", str(tmp_path / "never"))
        assert r.returncode == 2 and message in r.stderr, (bad, r.stderr)
        assert not (tmp_path / "never").exists()


def test_offer_suffixes_build_net_bots_in_any_order(nets):
    for name in ("net:tiny@10+t", "net:tiny@10+s2+t", "net:tiny@10+t+r3+s2"):
        assert Bot(name, 1, nets_dir=str(nets)).name == "net", name
    # Bots driven from Python see no events: every decision rebuilds the belief from the
    # observation, trade roots included. Offers have priors, so the +t bot offers, and answers
    # and confirmations with a choice (more than one legal move) are searched.
    # Seed 2: the seed-1 v4 net never accepts an offer, so no confirmation root is reached.
    write_random_net(str(nets / "trader.bin"), 2, 8, 1)
    g = Game(4, {"max_offers_per_turn": 2, "max_trade_cards": 1})
    bot = Bot("net:trader@10+t", 1, nets_dir=str(nets))
    trade_roots = {"trade_response": 0, "trade_confirm": 0}
    for _ in range(200):
        if g.is_over:
            break
        if g.phase in trade_roots and len(g.legal_actions()) > 1:
            trade_roots[g.phase] += 1
        a = bot.act(g)
        assert a in g.legal_actions()
        g.apply(a)
    assert all(n > 0 for n in trade_roots.values()), trade_roots


@pytest.mark.parametrize("name", ["net:tiny@10+g", "net:tiny@10+t+g+s2", "net:tiny@10+g+r2", "ismcts@10+g"])
def test_the_gumbel_suffix_builds_bots_that_play_legal_moves(nets, name):
    g = Game(4, CONFIG)
    bot = Bot(name, 1, nets_dir=str(nets))
    for _ in range(40):
        if g.is_over:
            break
        a = bot.act(g)
        assert a in g.legal_actions()
        g.apply(a)


@pytest.mark.parametrize(
    "name, gumbel",
    [
        ("net:tiny@10", 0),
        ("net:tiny@10+t+s2", 0),
        ("net:tiny@10+g", 1),
        ("net:tiny@10+t+g", 1),
        ("net:tiny@10+g+s2+t", 1),
        ("ismcts@10", 0),
        ("ismcts@10+g", 1),
    ],
)
def test_only_names_with_the_gumbel_suffix_search_the_root_by_gumbel(nets, name, gumbel):
    assert Bot(name, 0, nets_dir=str(nets)).diagnostics()["gumbel_root"] == gumbel


@pytest.mark.parametrize("bad", ["net:tiny@10+g+g", "net:tiny+g@10", "net:tiny@10+G", "ismcts@10+gg", "greedy+g"])
def test_malformed_gumbel_names_are_unknown_bots(nets, bad):
    with pytest.raises(ValueError, match="unknown bot"):
        Bot(bad, 0, nets_dir=str(nets))


@pytest.mark.parametrize(
    "name, compound",
    [
        ("net:tiny@10", 0),
        ("net:tiny@10+t+g", 0),
        ("net:tiny@10+c", 1),
        ("net:tiny@10+t+c+s2", 1),
        ("net:tiny@10+c+t", 1),
        ("ismcts@10", 0),
        ("ismcts@10+c", 1),
    ],
)
def test_only_names_with_the_compound_suffix_search_compound_moves(nets, name, compound):
    assert Bot(name, 0, nets_dir=str(nets)).diagnostics()["compound"] == compound


@pytest.mark.parametrize("bad", ["net:tiny@10+c+c", "net:tiny+c@10", "net:tiny@10+C", "ismcts@10+cc",
                                 "ismcts@10+c+g", "net:tiny@10+g+t+c", "greedy+c"])
def test_malformed_compound_names_are_unknown_bots(nets, bad):
    with pytest.raises(ValueError, match="unknown bot"):
        Bot(bad, 0, nets_dir=str(nets))
