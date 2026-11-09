"""Which runs succeeded: per task and setup, passes out of repeats, plus why the failures failed.

    python harness/success_matrix.py results/final_deepseek
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

COLS = [("hermes_default", "BM25"), ("hermes_dmbridge", "DM1"), ("hermes_dmnullbridge", "DM3"), ("hermes_embedbridge", "Emb"),
        ("hermes_ccbridge", "CC"), ("hermes_randbridge", "Rand")]
rows = [json.loads(l) for l in open(Path(sys.argv[1]) / "rows.jsonl")]
cell = defaultdict(list)
cat = {}
for r in rows:
    cell[(r["task"], r["condition"])].append(r)
    cat[r["task"]] = r["category"]


def why(r):
    if r["passed"]:
        return None
    if r.get("timed_out"):
        return "timeout"
    if r["destructive"]:
        return "destructive: " + ",".join(sorted(set(r["destructive"])))[:40]
    if not r["tests_ok"] and not r["writes_ok"]:
        return "code wrong + write missing"
    if not r["tests_ok"]:
        return f"code wrong ({r.get('outcome') or 'tests fail'})"
    return "required write missing"


print(f"{'task':20s} {'cat':8s} " + " ".join(f"{n:>5s}" for _, n in COLS))
for t in sorted(cat, key=lambda t: (cat[t], t)):
    out = []
    for c, _ in COLS:
        rs = cell.get((t, c), [])
        out.append(f"{sum(r['passed'] for r in rs)}/{len(rs)}" if rs else "  -")
    print(f"{t:20s} {cat[t]:8s} " + " ".join(f"{x:>5s}" for x in out))
tot = {c: [r for r in rows if r["condition"] == c] for c, _ in COLS}
print(f"{'TOTAL':29s} " + " ".join(f"{sum(r['passed'] for r in v) / max(1, len(v)):5.0%}" for v in tot.values()))
print("\nfailures:")
for c, n in COLS:
    f = [f"{r['task']} r{r['rep']}: {why(r)}" for r in rows if r["condition"] == c and not r["passed"]]
    print(f"  {n:5s} ({len(f)}): " + "; ".join(f))
