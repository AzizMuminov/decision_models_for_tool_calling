"""Summarises harness runs (rows.jsonl from runner.py) -> SUMMARY.md next to it.

    python3 harness/analyze.py results/harness_pilot
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

out = Path(sys.argv[1] if len(sys.argv) > 1 else "results/harness_pilot")
rows = [json.loads(l) for l in open(out / "rows.jsonl")]
conds = [c for c in ("none", "native", "filtered", "universal", "universal_index") if any(r["condition"] == c for r in rows)]
tasks = list(dict.fromkeys(r["task"] for r in rows))
by = {(r["task"], r["condition"]): r for r in rows}

L = ["# Coding-harness pilot: opencode with a 100-tool internal catalog", "",
     f"Model: {rows[0]['model']}. One run per task and condition. Pass = the task's hidden tests pass.", "",
     "## Pass / fail", "", "| task | " + " | ".join(conds) + " |", "|---|" + "---|" * len(conds)]
for t in tasks:
    cells = []
    for c in conds:
        r = by.get((t, c))
        if not r:
            cells.append("–")
            continue
        mark = "pass" if r["passed"] else "fail"
        extra = "" if t == "slug" else (" (found)" if r["needs_met"] else " (missed)")
        cells.append(mark + (extra if c != "none" else ""))
    L.append(f"| {t} | " + " | ".join(cells) + " |")

L += ["", "## Per condition (means over tasks)", "",
      "| condition | passed | needed tools found | steps | largest prompt (tokens) | prompt tokens read per task | output tokens | cost per task (LLM + DM) | find_tool calls |",
      "|---|---|---|---|---|---|---|---|---|"]
for c in conds:
    rs = [r for r in rows if r["condition"] == c]
    n = len(rs)
    need = [r for r in rs if r["task"] != "slug"]
    L.append(f"| {c} | {sum(r['passed'] for r in rs)}/{n} | "
             f"{sum(r['needs_met'] for r in need)}/{len(need)} | {sum(r.get('steps', 0) for r in rs) / n:.1f} | "
             f"{sum(r.get('max_prompt', 0) for r in rs) / n:,.0f} | {sum(r.get('input', 0) + r.get('cache_read', 0) for r in rs) / n:,.0f} | "
             f"{sum(r.get('output', 0) for r in rs) / n:,.0f} | ${sum(r.get('oc_cost', 0) + r['dm_usd'] + r['filter_dm_usd'] for r in rs) / n:.4f} | "
             f"{sum(r['find_calls'] for r in rs) / n:.1f} |")

L += ["", "## What the agent did with the catalog", ""]
for r in rows:
    if r["condition"] == "none":
        continue
    log = out / "runs" / f"{r['task']}__{r['condition']}" / "catalog_calls.jsonl"
    calls = [json.loads(l) for l in open(log)] if log.exists() else []
    seq = []
    for c in calls:
        if c["kind"] == "find_tool":
            seq.append(f"find(\"{c['intent'][:70]}\") → {', '.join(c['ranked'][:3])}")
        else:
            seq.append(f"{c['tool']}{'' if c['ok'] else ' ✗'}")
    filt = f" [filter: {', '.join(r['filter'])}]" if r["condition"] == "filtered" else ""
    L.append(f"- **{r['task']} / {r['condition']}** ({'pass' if r['passed'] else 'fail'}){filt}: " + ("; ".join(seq) if seq else "no catalog calls"))
(out / "SUMMARY.md").write_text("\n".join(L) + "\n")
print("\n".join(L))
