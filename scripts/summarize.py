"""Markdown table of street2uav.evaluate results; runs named <setting>_s<seed>[...] are also averaged over seeds
(mean and sample standard deviation).
  python scripts/summarize.py runs/table1/*.json
"""
import json
import re
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

W = ("clear", "fog", "rain", "night")


def row(r):
    return [r["joint"], r["car"]["miou"], r["drone"]["miou"], *(r["drone"]["weather"][w] for w in W)]


def main(paths):
    head = "| run | Joint | Car | Drone | Clear | Fog | Rain | Night |\n|---|---|---|---|---|---|---|---|"
    lines, groups = [head], defaultdict(list)
    for p in sorted(paths):
        r = json.load(open(p))
        if "joint" not in r:
            continue
        v = row(r)
        name = Path(p).stem
        lines.append(f"| {name} | " + " | ".join(f"{x:.2f}" for x in v) + " |")
        groups[re.sub(r"_s\d+", "", name)].append(v)
    for name, vs in groups.items():
        if len(vs) > 1:
            cells = [f"{st.mean(c):.2f} ± {st.stdev(c):.2f}" for c in zip(*vs)]
            lines.append(f"| {name} (mean of {len(vs)} seeds) | " + " | ".join(cells) + " |")
    print("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1:])
