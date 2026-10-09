"""Plan 10 Task 4: the Rust forward pass of the net at NET on 200 positions of seeded random trading
games, written to OUT-{policy,value,aux}-{batch,single}.bin (one call of 200, and 200 calls of 1),
for comparing two builds: the engine plays the same games in each. Usage: forward_dump.py NET OUT"""
import random
import sys

from alphasettler._engine import NET_ENCODING_HASH, Game, encode, net_forward

net, out = sys.argv[1], sys.argv[2]
rng = random.Random(1)
public, hidden, seed = [], [], 0
while len(public) < 200:
    g = Game(seed, {"max_offers_per_turn": 2, "max_trade_cards": 1})
    seed += 1
    for k in range(400):
        if g.is_over or len(public) == 200:
            break
        if k % 7 == 0:
            p, h = encode(g)
            public.append(p)
            hidden.append(h)
        g.apply(rng.choice(g.legal_actions()))
batch = net_forward(net, b"".join(public), b"".join(hidden))
single = [net_forward(net, p, h) for p, h in zip(public, hidden)]
for key in ("policy", "value", "aux"):
    with open(f"{out}-{key}-batch.bin", "wb") as f:
        f.write(batch[key])
    with open(f"{out}-{key}-single.bin", "wb") as f:
        f.write(b"".join(r[key] for r in single))
print(f"encoding {NET_ENCODING_HASH:#018x}: {len(public)} positions from {seed} games")
