"""Summarises runs of runner_real.py: pass rates, context, cost, and why runs failed.

    .venv/bin/python harness/analyze_real.py results/real_v1 [results/real_v1_extra ...]
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

ORDER = ["none", "native", "cc", "bm25", "groups", "embed", "llm", "dm", "random"]


def wilson(k, n, z=1.96):
    if not n:
        return (0, 0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def mcnemar(rows, a, b):
    """Exact two-sided McNemar on runs paired by (task, model, rep)."""
    pa = {(r["task"], r["model"], r["rep"]): r["passed"] for r in rows if r["condition"] == a}
    pb = {(r["task"], r["model"], r["rep"]): r["passed"] for r in rows if r["condition"] == b}
    keys = pa.keys() & pb.keys()
    x = sum(pa[k] and not pb[k] for k in keys)
    y = sum(pb[k] and not pa[k] for k in keys)
    n = x + y
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(0, min(x, y) + 1)) / 2 ** n) if n else 1.0
    return x, y, p, len(keys)


def failure_reason(r):
    if r["passed"]:
        return "pass"
    if r["timed_out"]:
        return "timeout"
    if r["condition"] not in ("none", "native") and r["n_searches"] == 0 and r["n_activations"] == 0 and r["category"] in ("single", "multi"):
        return "never searched"
    if r["category"] in ("single", "multi") and not r["needs_met"]:
        if r["n_searches"] and not any(r.get("search_hit") or []):
            return "search never returned a needed tool"
        return "needed tool not used successfully"
    if r["destructive"]:
        return "destructive call"
    if not r["writes_ok"]:
        return "required write missing"
    if r["category"] == "absent":
        return f"absent: {r.get('outcome')}"
    return "code wrong despite the information"


def main() -> None:
    rows = [json.loads(l) for d in sys.argv[1:] for l in open(Path(d) / "rows.jsonl")]
    conds = [c for c in ORDER if any(r["condition"] == c for r in rows)]
    models = sorted({r["model"] for r in rows})
    print(f"{len(rows)} runs, {len({r['task'] for r in rows})} tasks, models {models}\n")
    print("pass rate on tasks that need a tool (single+multi), with 95% Wilson CI; all-task pass; median largest prompt; mean $/run")
    hdr = f"{'cond':8s}" + "".join(f" | {m:^44s}" for m in models)
    print(hdr)
    for c in conds:
        line = f"{c:8s}"
        for m in models:
            rs = [r for r in rows if r["condition"] == c and r["model"] == m]
            tool = [r for r in rs if r["category"] in ("single", "multi")]
            k, n = sum(r["passed"] for r in tool), len(tool)
            lo, hi = wilson(k, n)
            prompts = sorted(r.get("max_prompt") or 0 for r in rs)
            med = prompts[len(prompts) // 2] if prompts else 0
            cost = sum((r.get("oc_cost") or 0) + r["selector_usd"] for r in rs) / max(1, len(rs))
            allp = sum(r["passed"] for r in rs) / max(1, len(rs))
            line += f" | {k:3d}/{n:<3d} {k / max(1, n):.2f} [{lo:.2f},{hi:.2f}] all {allp:.2f} {med / 1000:5.0f}k ${cost:.4f}"
        print(line)

    print("\nby category (both models pooled): pass rate")
    cats = ["single", "multi", "absent", "control"]
    print(f"{'cond':8s}" + "".join(f"{c:>12s}" for c in cats))
    for c in conds:
        cells = []
        for k in cats:
            sub = [r["passed"] for r in rows if r["condition"] == c and r["category"] == k]
            cells.append(f"{sum(sub) / len(sub):6.2f} ({len(sub):3d})" if sub else f"{'-':>12s}")
        print(f"{c:8s}" + "".join(f"{x:>12s}" for x in cells))

    print("\npaired comparisons (McNemar, runs paired by task/model/rep): A-only passes vs B-only passes")
    for a, b in [("dm", "native"), ("dm", "cc"), ("dm", "bm25"), ("dm", "embed"), ("dm", "llm"), ("dm", "groups"), ("native", "cc")]:
        if a in conds and b in conds:
            x, y, p, n = mcnemar(rows, a, b)
            print(f"  {a:6s} vs {b:7s}: {x:3d} vs {y:3d}  p={p:.3f}  (n={n})")

    print("\nsearch behaviour (search conditions): searches/run, share of searches that returned a needed tool, unknown-tool calls, failed calls/run")
    for c in conds:
        rs = [r for r in rows if r["condition"] == c and r["category"] in ("single", "multi")]
        if not rs:
            continue
        hits = [h for r in rs for h in r.get("search_hit") or []]
        print(f"  {c:8s} searches {sum(r['n_searches'] for r in rs) / len(rs):5.2f}  activations {sum(r['n_activations'] for r in rs) / len(rs):4.2f}  "
              f"hit {sum(hits) / max(1, len(hits)):.2f}  unknown {sum(r['n_unknown_tool'] for r in rs)}  failed/run {sum(r['n_failed_calls'] for r in rs) / len(rs):4.2f}  "
              f"calls/run {sum(r['n_calls'] for r in rs) / len(rs):5.2f}  steps {sum(r.get('steps') or 0 for r in rs) / len(rs):5.1f}")

    print("\nwhy runs failed (count)")
    reasons = defaultdict(lambda: defaultdict(int))
    for r in rows:
        reasons[r["condition"]][failure_reason(r)] += 1
    for c in conds:
        fails = {k: v for k, v in reasons[c].items() if k != "pass"}
        print(f"  {c:8s} " + ", ".join(f"{k}: {v}" for k, v in sorted(fails.items(), key=lambda kv: -kv[1])))

    print("\nabsent tasks: outcome")
    for c in conds:
        oc = defaultdict(int)
        for r in rows:
            if r["condition"] == c and r["category"] == "absent":
                oc[r.get("outcome")] += 1
        print(f"  {c:8s} {dict(oc)}")
    print("\ncontrol tasks: catalog calls made (should be 0):",
          {c: sum(r["n_calls"] for r in rows if r["condition"] == c and r["category"] == "control") for c in conds})
    print("total cost: $%.2f" % sum((r.get("oc_cost") or 0) + r["selector_usd"] for r in rows))


if __name__ == "__main__":
    main()
