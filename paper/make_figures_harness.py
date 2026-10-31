"""Figure for the coding-harness results, built from the logged runs.

    .venv/bin/python paper/make_figures_harness.py   # -> paper/figures/fig_harness.pdf, figures/harness_figure_data.json
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "figures"
rows = [json.loads(l) for l in open(ROOT / "results/harness_v2/rows.jsonl")]
for r in rows:
    r["catalog_size"] = 100
rows += [json.loads(l) for l in open(ROOT / "results/harness_v2_300/rows.jsonl")]

# fixed categorical order (validated palette from make_figures.py); "none" in neutral gray
C = {"universal_index": "#2a78d6", "native": "#eb6834", "universal": "#1baf7a", "filtered": "#eda100", "none": "#8a8a85"}
LABEL = {"none": "no catalog", "native": "native (all definitions)", "filtered": "filtered (top-5 upfront)",
         "universal": "universal tool", "universal_index": "universal tool + service list"}
INK, MUTED = "#2b2b29", "#6b6b66"
plt.rcParams.update({"font.family": "serif", "font.size": 8.5, "axes.titlesize": 9, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e6e3", "grid.linewidth": 0.6,
                     "legend.frameon": False, "pdf.fonttype": 42})


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


data = {"prompt": {}, "pass": {}}
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.6, 2.75), gridspec_kw={"width_ratios": [1, 1.25]})

# (a) largest prompt vs catalog size
for cond in ("native", "universal_index"):
    for model, ls, mk in (("deepseek", "-", "o"), ("glm", "--", "s")):
        ys = []
        for size in (100, 300):
            rs = [r for r in rows if r["condition"] == cond and r["model"] == model and r["catalog_size"] == size]
            ys.append(np.mean([r["max_prompt"] for r in rs]) / 1000)
        data["prompt"][f"{cond}/{model}"] = [round(y, 1) for y in ys]
        a1.plot([0, 1], ys, color=C[cond], ls=ls, marker=mk, ms=5, lw=1.6)
        a1.annotate(f"{'DeepSeek' if model == 'deepseek' else 'GLM'}", (1, ys[1]), xytext=(6, 4 if model == "deepseek" else -6),
                    textcoords="offset points", fontsize=7, color=INK, va="center")
a1.set_xticks([0, 1]); a1.set_xticklabels(["100 tools", "300 tools"]); a1.set_xlim(-0.25, 1.45)
a1.set_ylabel("largest prompt per run (k tokens)")
a1.set_ylim(0, 195)
a1.set_title("(a) Context vs. catalog size", loc="left", color=INK)
a1.text(0.03, 120, "native", color=INK, fontsize=8)
a1.text(0.03, 25, "universal + list", color=INK, fontsize=8)

# (b) pass rate on tasks that need a catalog tool, 100 tools, both models pooled
order = ["none", "filtered", "universal", "universal_index", "native"]
for i, cond in enumerate(order):
    rs = [r for r in rows if r["condition"] == cond and r["catalog_size"] == 100 and r["category"] not in ("control", "absent")]
    k, n = sum(r["passed"] for r in rs), len(rs)
    lo, hi = wilson(k, n)
    data["pass"][cond] = {"k": k, "n": n, "lo": round(lo, 3), "hi": round(hi, 3)}
    a2.bar(i, k / n, color=C[cond], width=0.62)
    a2.errorbar(i, k / n, yerr=[[k / n - lo], [hi - k / n]], color=INK, capsize=2.5, lw=0.8)
    a2.text(i, hi + 0.03, f"{k}/{n}", ha="center", fontsize=7, color=INK)
a2.set_xticks(range(len(order)))
a2.set_xticklabels(["no\ncatalog", "filtered\nupfront", "universal", "universal\n+ list", "native"], fontsize=7.5)
a2.set_ylim(0, 1.12)
a2.set_ylabel("tasks passed (needs a catalog tool)")
a2.set_title("(b) Accuracy, 100 tools, both models", loc="left", color=INK)
fig.tight_layout()
fig.savefig(OUT / "fig_harness.pdf")
(OUT / "harness_figure_data.json").write_text(json.dumps(data, indent=1))
print(json.dumps(data, indent=1))
