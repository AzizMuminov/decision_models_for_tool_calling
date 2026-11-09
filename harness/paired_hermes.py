"""Apples-to-apples comparison of Hermes conditions: only (task, rep) slots that every compared condition has finished.

Reports, on that common set: pass counts, median and mean tokens and cost per run, and for each condition vs the
baseline the per-slot token ratio (median, and a bootstrap 95% CI of the geometric mean ratio, resampling tasks).

    python harness/paired_hermes.py results/hermes_v3 [--base hermes_default]
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics as st
from collections import defaultdict
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--base", default="hermes_default")
    ap.add_argument("--conditions", default="")
    ap.add_argument("--pairwise", action="store_true", help="compare each condition with the base on the slots both finished")
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(Path(a.dir) / "rows.jsonl")]
    conds = a.conditions.split(",") if a.conditions else sorted({r["condition"] for r in rows})
    if a.pairwise:
        import subprocess, sys
        for c in conds:
            if c != a.base:
                out = subprocess.run([sys.executable, __file__, a.dir, "--base", a.base, "--conditions", f"{a.base},{c}"], capture_output=True, text=True).stdout
                print(out.strip().splitlines()[0] + "\n   " + out.strip().splitlines()[-1])
        return
    by = defaultdict(dict)
    for r in rows:
        by[(r["task"], r["rep"])][r["condition"]] = r
    slots = sorted(k for k, v in by.items() if all(c in v for c in conds))
    tasks = sorted({t for t, _ in slots})
    print(f"{len(slots)} (task, rep) slots finished in all of {conds} ({len(tasks)} tasks)\n")
    print(f"{'condition':18s} {'pass':>6s} {'tok median':>11s} {'tok mean':>9s} {'$/run':>8s} {'searches':>9s}  vs {a.base}: median ratio, geo-mean ratio [95% CI], fewer tokens on")
    rng = random.Random(0)
    for c in conds:
        rs = [by[s][c] for s in slots]
        toks = [r["total_tokens"] for r in rs]
        line = (f"{c:18s} {sum(r['passed'] for r in rs):3d}/{len(rs):<3d} {st.median(toks) / 1e3:9.0f}k {st.mean(toks) / 1e3:8.0f}k "
                f"${st.mean(r['cost'] for r in rs):.4f} {st.mean(r['n_searches'] for r in rs):8.1f}")
        if c != a.base and a.base in conds:
            lr = {s: math.log(max(1, by[s][c]["total_tokens"]) / max(1, by[s][a.base]["total_tokens"])) for s in slots}
            per_task = defaultdict(list)
            for (t, rp), v in lr.items():
                per_task[t].append(v)
            boots = []
            for _ in range(2000):
                pick = [rng.choice(tasks) for _ in tasks]
                vals = [v for t in pick for v in per_task[t]]
                boots.append(math.exp(st.mean(vals)))
            boots.sort()
            ratios = [math.exp(v) for v in lr.values()]
            line += (f"   {st.median(ratios):.2f}x, {math.exp(st.mean(lr.values())):.2f}x [{boots[50]:.2f}, {boots[1949]:.2f}], "
                     f"{sum(v < 0 for v in lr.values())}/{len(lr)}")
        print(line)


if __name__ == "__main__":
    main()
