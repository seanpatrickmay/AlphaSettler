"""3d-3 results: E against the control seed by seed on the robustness arenas, and the styles' own lines.

.venv/bin/python3 runs/3d3/paired.py | tee runs/3d3/robust-paired.log
"""
import json, math, os
from collections import defaultdict
from pathlib import Path

os.chdir(Path(__file__).resolve().parents[2])
ctrl = dict(l.strip().split("=", 1) for l in open("runs/3d3/control.env") if "=" in l)["CTRL"]
STYLES = ("roads", "army", "production")


def load(net, style):
    d = defaultdict(list)
    for l in open(f"runs/3d3/robust-{net}-{style}.jsonl"):
        r = json.loads(l); d[r["seed"]].append(r)
    return d


def rate(rs):
    return sum(r["winner"] == r["candidate_seat"] for r in rs) / len(rs)


def line(label, xs):
    m = sum(xs) / len(xs); sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1)); se = sd / math.sqrt(len(xs))
    print(f"{label}: difference {m:+.4f}, standard error {se:.4f}, z {m / se:+.2f} over {len(xs)} seeds")


per = {net: {st: {s: rate(rs) for s, rs in load(net, st).items()} for st in STYLES} for net in ("E", ctrl)}
seeds = sorted(per["E"]["roads"])
assert all(sorted(per[n][st]) == seeds and all(len(rs) == 4 for rs in load(n, st).values()) for n in per for st in STYLES)
print(f"paired by seed: E's win rate minus {ctrl}'s on the same seed and style (4 games each, one per seat)")
for st in STYLES:
    line(f"vs style:{st}", [per["E"][st][s] - per[ctrl][st][s] for s in seeds])
line("pooled (per seed, the three styles averaged)",
     [sum(per["E"][st][s] - per[ctrl][st][s] for st in STYLES) / 3 for s in seeds])
print()
print("each net's win rate against army minus against production, and against army minus against roads, by seed:")
for net in ("E", ctrl):
    line(f"net:{net} army - production", [per[net]["army"][s] - per[net]["production"][s] for s in seeds])
    line(f"net:{net} army - roads", [per[net]["army"][s] - per[net]["roads"][s] for s in seeds])
print()
print("each net against the three styles, per seed (the three styles averaged), 95% CI by the normal approximation:")
for net in ("E", ctrl):
    xs = [sum(per[net][st][s] for st in STYLES) / 3 for s in seeds]
    m = sum(xs) / len(xs); se = math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) / math.sqrt(len(xs))
    print(f"net:{net}: win rate {m:.3f} [{m - 1.96 * se:.3f}, {m + 1.96 * se:.3f}] over {len(xs)} seeds, z vs 0.25 = {(m - 0.25) / se:.2f}")
print()
print("the styles in these games (style seats only, both nets' arenas pooled):")
print("| style | style seats | style wins | win share per seat | mean VP per seat | mean turns |")
print("|---|---|---|---|---|---|")
for st in STYLES:
    seats = wins = vp = turns = games = 0
    for net in ("E", ctrl):
        for rs in load(net, st).values():
            for r in rs:
                games += 1; turns += r["turns"]
                for s in range(4):
                    if s != r["candidate_seat"]:
                        seats += 1; vp += r["vp"][s]; wins += r["winner"] == s
    print(f"| `style:{st}@100+t` | {seats:,} | {wins:,} | {wins / seats:.3f} | {vp / seats:.2f} | {turns / games:.1f} |")
