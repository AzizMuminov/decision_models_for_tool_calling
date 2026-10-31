"""Catalog-size sweep: 100-tool runs (results/harness_v2, repeats 0-1) vs 300-tool runs (results/harness_v2_300).

    python3 harness/analyze_sweep.py results/harness_v2 results/harness_v2_300
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

a, b = Path(sys.argv[1]), Path(sys.argv[2])
CONDS = ["native", "universal", "universal_index"]
r100 = [r for r in map(json.loads, open(a / "rows.jsonl")) if r["condition"] in CONDS and r["rep"] < 2]
r300 = list(map(json.loads, open(b / "rows.jsonl")))
for r in r100:
    r["catalog_size"] = 100
rows = r100 + r300
MODELS = sorted({r["model"] for r in rows})


def cost(r):
    return r.get("oc_cost", 0) + r["dm_usd"] + r["filter_dm_usd"]


L = ["# Catalog-size sweep: 100 vs 300 tools", "",
     f"Same 30 tasks, both models, repeats 0-1 at each size. 300-tool runs: {len(r300)}; runs mentioning paths outside "
     f"the project: {sum(bool(r['outside_paths']) for r in r300)}; timeouts: {sum(r['timed_out'] for r in r300)}. Costs are "
     "opencode's per-run estimates (the OpenRouter bill for the batch is in the run log).", "",
     "| model | condition | tools | pass rate | largest prompt (mean) | prompt tokens read per run | cost per run (est.) | find_tool top-3 hit rate |",
     "|---|---|---|---|---|---|---|---|"]
for m in MODELS:
    for c in CONDS:
        for size in (100, 300):
            rs = [r for r in rows if r["model"] == m and r["condition"] == c and r["catalog_size"] == size]
            if not rs:
                continue
            need = [r for r in rs if r["needs"]]
            tot = sum(r["find_hits"] + r["find_irrelevant"] for r in need)
            hit = f"{sum(r['find_hits'] for r in need) / tot:.2f}" if tot else "–"
            L.append(f"| {m} | {c} | {size} | {np.mean([r['passed'] for r in rs]):.3f} | {np.mean([r.get('max_prompt', 0) for r in rs]):,.0f} | "
                     f"{np.mean([r.get('input', 0) + r.get('cache_read', 0) for r in rs]):,.0f} | ${np.mean([cost(r) for r in rs]):.4f} | {hit} |")

L += ["", "## Pass rate by category (both models)", "", "| category | " + " | ".join(f"{c} {s}" for c in CONDS for s in (100, 300)) + " |",
      "|---|" + "---|" * 6]
for cat in ("control", "single", "midtask", "contract", "absent"):
    cells = []
    for c in CONDS:
        for s in (100, 300):
            rs = [r for r in rows if r["category"] == cat and r["condition"] == c and r["catalog_size"] == s]
            cells.append(f"{np.mean([r['passed'] for r in rs]):.2f}" if rs else "–")
    L.append(f"| {cat} | " + " | ".join(cells) + " |")

L += ["", "## Paired tests (same task, model, repeat)", "", "| comparison | pairs | only A passed / only B passed | McNemar p |", "|---|---|---|---|"]
idx = {(r["task"], r["model"], r["rep"], r["condition"], r["catalog_size"]): r for r in rows}


def paired(name, ka, kb):
    pairs = [(idx[k], idx[kb(k)]) for k in idx if ka(k) and kb(k) in idx]
    if not pairs:
        return
    x = sum(p["passed"] and not q["passed"] for p, q in pairs)
    y = sum(q["passed"] and not p["passed"] for p, q in pairs)
    L.append(f"| {name} | {len(pairs)} | {x} / {y} | {binomtest(x, x + y).pvalue if x + y else 1.0:.3g} |")


paired("universal_index vs native, 300 tools", lambda k: k[3] == "universal_index" and k[4] == 300, lambda k: (k[0], k[1], k[2], "native", 300))
paired("universal vs native, 300 tools", lambda k: k[3] == "universal" and k[4] == 300, lambda k: (k[0], k[1], k[2], "native", 300))
for c in CONDS:
    paired(f"{c}: 300 vs 100 tools", lambda k, c=c: k[3] == c and k[4] == 300, lambda k: (k[0], k[1], k[2], k[3], 100))

(b / "SWEEP.md").write_text("\n".join(L) + "\n")
print("\n".join(L))
