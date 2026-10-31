"""Analysis of the 30-task coding-harness ablation (rows.jsonl from runner.py) -> REPORT.md next to it.

    python3 harness/analyze_v2.py results/harness_v2
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

out = Path(sys.argv[1] if len(sys.argv) > 1 else "results/harness_v2")
rows = [json.loads(l) for l in open(out / "rows.jsonl")]
CONDS = [c for c in ("none", "native", "filtered", "universal", "universal_index") if any(r["condition"] == c for r in rows)]
MODELS = sorted({r["model"] for r in rows})
CATS = ["control", "single", "midtask", "contract", "absent"]
BILL = 1.0  # opencode's estimate matched the OpenRouter bill for this batch (factor 1.01 on $8.84)
rng = np.random.default_rng(0)


def cost(r):
    return r.get("oc_cost", 0) * BILL + r["dm_usd"] + r["filter_dm_usd"]


def task_ci(rs, key=lambda r: r["passed"]):
    """Mean and 95% CI, resampling whole tasks (repeats and models of a task stay together)."""
    by = defaultdict(list)
    for r in rs:
        by[r["task"]].append(float(key(r)))
    tasks = list(by)
    if not tasks:
        return float("nan"), float("nan"), float("nan")
    m = np.mean([v for t in tasks for v in by[t]])
    bs = []
    for _ in range(2000):
        pick = rng.choice(len(tasks), len(tasks))
        bs.append(np.mean([v for i in pick for v in by[tasks[i]]]))
    return m, np.quantile(bs, .025), np.quantile(bs, .975)


L = ["# Coding-harness ablation: opencode + 100-tool catalog", "",
     f"{len(rows)} runs: {len({r['task'] for r in rows})} tasks × {len(CONDS)} conditions × {len(MODELS)} models × "
     f"{max(r['rep'] for r in rows) + 1} repeats. Runs mentioning paths outside the project: "
     f"{sum(bool(r['outside_paths']) for r in rows)} (inspected: pytest's own interpreter path, globs with no matches, and "
     f"directory names listed by agent-written test files; no protected file was read); runs not in their task folder: "
     f"{sum(not r['ran_in_task_dir'] for r in rows)}; timeouts: {sum(r['timed_out'] for r in rows)}. "
     f"Total cost: OpenRouter bill $8.84.", "",
     "## 1. Success, context and cost", "",
     "| model | condition | pass rate [95% CI, task-clustered] | largest prompt (mean) | prompt tokens read per run | steps | cost per run |",
     "|---|---|---|---|---|---|---|"]
for m in MODELS:
    for c in CONDS:
        rs = [r for r in rows if r["model"] == m and r["condition"] == c]
        if not rs:
            continue
        p, lo, hi = task_ci(rs)
        L.append(f"| {m} | {c} | {p:.3f} [{lo:.2f}, {hi:.2f}] | {np.mean([r.get('max_prompt', 0) for r in rs]):,.0f} | "
                 f"{np.mean([r.get('input', 0) + r.get('cache_read', 0) for r in rs]):,.0f} | {np.mean([r.get('steps', 0) for r in rs]):.1f} | "
                 f"${np.mean([cost(r) for r in rs]):.4f} |")

L += ["", "## 2. Pass rate by task category (both models pooled)", "",
      "| category (tasks) | " + " | ".join(CONDS) + " |", "|---|" + "---|" * len(CONDS)]
for cat in CATS:
    n = len({r["task"] for r in rows if r["category"] == cat})
    cells = []
    for c in CONDS:
        rs = [r for r in rows if r["category"] == cat and r["condition"] == c]
        cells.append(f"{np.mean([r['passed'] for r in rs]):.2f}" if rs else "–")
    L.append(f"| {cat} ({n}) | " + " | ".join(cells) + " |")

L += ["", "## 3. Value tasks: what the agent did with the constant", "",
      "Outcome of tasks whose job is to set one value (single, mid-task, absent). On *absent* tasks the value does not "
      "exist, so the right outcome is *abstained* and *wrong* means the agent invented a value.", "",
      "| category | condition | correct | abstained | unchanged (no TODO) | wrong value |", "|---|---|---|---|---|---|"]
for cat in ("single", "midtask", "absent"):
    for c in CONDS:
        rs = [r for r in rows if r["category"] == cat and r["condition"] == c and r.get("outcome")]
        if not rs:
            continue
        cnt = {k: sum(r["outcome"] == k for r in rs) / len(rs) for k in ("correct", "abstained", "unchanged", "wrong")}
        L.append(f"| {cat} | {c} | {cnt['correct']:.2f} | {cnt['abstained']:.2f} | {cnt['unchanged']:.2f} | {cnt['wrong']:.2f} |")

L += ["", "## 4. False positives and extra activity (per run, both models)", "",
      "| condition | state-changing calls | calls to non-existent tools | failed calls | extra catalog calls | control runs that used the catalog | lucky passes (needed tool never called) |",
      "|---|---|---|---|---|---|---|"]
for c in CONDS:
    rs = [r for r in rows if r["condition"] == c]
    ctrl = [r for r in rs if r["category"] == "control"]
    need = [r for r in rs if r["needs"]]
    L.append(f"| {c} | {np.mean([r['fp_mutating_calls'] for r in rs]):.2f} | {np.mean([r['fp_unknown_tool'] for r in rs]):.2f} | "
             f"{np.mean([r['fp_failed_calls'] for r in rs]):.2f} | {np.mean([r['fp_unneeded_calls'] for r in rs]):.2f} | "
             f"{np.mean([r['catalog_calls'] > 0 for r in ctrl]):.2f} | {sum(r['lucky_pass'] for r in need)}/{len(need)} |")

L += ["", "## 5. Did the decision model surface the needed tools?", ""]
for c in ("universal", "universal_index"):
    rs = [r for r in rows if r["condition"] == c and r["needs"]]
    if rs:
        hits = sum(r["find_hits"] for r in rs)
        tot = sum(r["find_hits"] + r["find_irrelevant"] for r in rs)
        never = np.mean([r["find_calls"] == 0 for r in rs])
        L.append(f"- **{c}**: {tot} find_tool calls on tasks that need the catalog; {hits / max(tot, 1):.2f} returned a needed tool in "
                 f"the top 3. In {never:.2f} of these runs the agent never called find_tool at all.")
rs = [r for r in rows if r["condition"] == "filtered" and r["needs"]]
if rs:
    for cat in ("single", "midtask", "contract"):
        s = [r for r in rs if r["category"] == cat]
        if s:
            L.append(f"- **filtered**, {cat}: the DM's top-5 contained every needed tool in {np.mean([r['filter_has_needs'] for r in s]):.2f} of runs.")

L += ["", "## 6. Paired comparisons with native (same task, model and repeat)", "",
      "| condition vs native | pairs | only condition passed / only native passed | McNemar p | prompt tokens read, ratio to native |",
      "|---|---|---|---|---|"]
idx = {(r["task"], r["model"], r["rep"], r["condition"]): r for r in rows}
for c in CONDS:
    if c == "native":
        continue
    pairs = [(idx[k], idx[(k[0], k[1], k[2], "native")]) for k in idx if k[3] == c and (k[0], k[1], k[2], "native") in idx]
    if not pairs:
        continue
    b = sum(x["passed"] and not y["passed"] for x, y in pairs)
    d = sum(y["passed"] and not x["passed"] for x, y in pairs)
    p = binomtest(b, b + d).pvalue if b + d else 1.0
    ratio = np.mean([x.get("input", 0) + x.get("cache_read", 0) for x, _ in pairs]) / np.mean([y.get("input", 0) + y.get("cache_read", 0) for _, y in pairs])
    L.append(f"| {c} | {len(pairs)} | {b} / {d} | {p:.3g} | {ratio:.2f} |")

(out / "REPORT.md").write_text("\n".join(L) + "\n")
print("\n".join(L))
