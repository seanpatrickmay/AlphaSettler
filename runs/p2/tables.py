"""P2 criterion 2: each arm's generations finished within the budget, from its state at the cut and
its timestamped log. Usage: .venv/bin/python3 runs/p2/tables.py A B"""
import json
import re
import sys

for arm in sys.argv[1:]:
    state = json.load(open(f"runs/p2/{arm}-state-at-cut.json"))
    stamps = {}
    for line in open(f"runs/p2/{arm}.log"):
        m = re.match(r"(\d+) (gen-\d{3}): (selfplay|train|gate)\b", line)
        if m:
            stamps[(m[2], m[3])] = int(m[1])
    games = seconds = 0
    print(f"### {arm}: best net {state['best']}, {len(state['generations'])} generations finished in the budget")
    print()
    print("| gen | games | decisions | full searches | steps | held-out score | policy | value | aux "
          "| gate wins / games | z | promoted | self-play minutes |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for g in state["generations"]:
        name = f"gen-{g['gen']:03d}"
        sp, tr, gate = g["selfplay"], g["train"], g["gate"]
        secs = stamps[(name, "train")] - stamps[(name, "selfplay")]
        games += sp["games"]
        seconds += secs
        z = "-" if gate["z"] is None else f"{gate['z']:.2f}"
        b = tr["best"]
        print(f"| {g['gen']} | {sp['games']:,} | {sp['decisions']:,} | {sp['full_decisions']:,} | {tr['steps']:,} | "
              f"{b['score']:.4f} | {b['policy']:.4f} | {b['value']:.4f} | {b['aux']:.4f} | "
              f"{gate['wins']} / {gate['games']} | {z} | {'yes' if gate['promoted'] else 'no'} | {secs / 60:.1f} |")
    print()
    if seconds:
        print(f"self-play: {games:,} games in {seconds / 3600:.2f} h of self-play, "
              f"{games / (seconds / 3600):,.0f} games per hour")
    else:
        print("self-play: no generation finished in the budget")
    print()
