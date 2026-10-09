"""3d-3 results: E against A seed by seed on the robustness arenas, and the styles' own lines.

.venv/bin/python3 runs/3d3/paired.py | tee runs/3d3/robust-paired.log
"""
import json, math
from collections import defaultdict
def load(net, style):
    d = defaultdict(list)
    for l in open(f"runs/3d3/robust-{net}-{style}.jsonl"):
        r = json.loads(l); d[r["seed"]].append(r)
    return d
diffs = []
print("paired by seed: E's win rate minus A's on the same seed and style (4 games each, one per seat)")
for style in ("roads", "army", "production"):
    e, a = load("E", style), load("A", style)
    assert e.keys() == a.keys() and all(len(e[s]) == len(a[s]) == 4 for s in e)
    ds = [sum(r["winner"] == r["candidate_seat"] for r in e[s]) / 4 - sum(r["winner"] == r["candidate_seat"] for r in a[s]) / 4 for s in e]
    m = sum(ds) / len(ds); sd = math.sqrt(sum((x - m) ** 2 for x in ds) / (len(ds) - 1)); se = sd / math.sqrt(len(ds))
    print(f"vs style:{style}: difference {m:+.4f}, standard error {se:.4f}, z {m / se:+.2f} over {len(ds)} seeds")
    diffs += ds
m = sum(diffs) / len(diffs); sd = math.sqrt(sum((x - m) ** 2 for x in diffs) / (len(diffs) - 1)); se = sd / math.sqrt(len(diffs))
print(f"pooled (seed x style units): difference {m:+.4f}, standard error {se:.4f}, z {m / se:+.2f} over {len(diffs)} units")
print()
print("the styles in these games (style seats only, both nets' arenas pooled):")
print("| style | style seats | style wins | win share per seat | mean VP per seat | mean turns |")
print("|---|---|---|---|---|---|")
for style in ("roads", "army", "production"):
    seats = wins = vp = turns = games = 0
    for net in ("E", "A"):
        for rs in load(net, style).values():
            for r in rs:
                games += 1; turns += r["turns"]
                for s in range(4):
                    if s != r["candidate_seat"]:
                        seats += 1; vp += r["vp"][s]; wins += r["winner"] == s
    print(f"| `style:{style}@100+t` | {seats:,} | {wins:,} | {wins / seats:.3f} | {vp / seats:.2f} | {turns / games:.1f} |")
