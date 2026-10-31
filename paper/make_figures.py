"""Builds the paper's data figures straight from the result files (every plotted number is traceable).

    .venv/bin/python paper/make_figures.py      # -> paper/figures/*.pdf, paper/figures/figure_data.json
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# validated categorical slots (light): blue, orange, aqua, yellow; control in neutral gray
C = {"MSD1": "#2a78d6", "Jev": "#eb6834", "DeepSeek": "#1baf7a", "Sonnet": "#eda100", "Lexical": "#8a8a85"}
M = {"MSD1": "o", "Jev": "s", "DeepSeek": "^", "Sonnet": "D", "Lexical": "x"}
INK, MUTED = "#2b2b29", "#6b6b66"

plt.rcParams.update({
    "font.family": "serif", "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e6e3",
    "grid.linewidth": 0.6, "legend.frameon": False, "pdf.fonttype": 42,
})


def rows(path: Path):
    return [json.loads(l) for l in open(path)]


def ece(p, y, bins=10):
    p, y = np.asarray(p), np.asarray(y, float)
    edges = np.linspace(0, 1, bins + 1)
    tot = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & ((p < hi) if hi < 1 else (p <= hi))
        if m.any():
            tot += m.mean() * abs(p[m].mean() - y[m].mean())
    return tot


def auroc(p, y):
    pos = [a for a, b in zip(p, y) if b]
    neg = [a for a, b in zip(p, y) if not b]
    return np.mean([(a > b) + 0.5 * (a == b) for a in pos for b in neg])


data = {}

# ---- Figure 2: reliability of the DM gate, BFCL pooled (Study 1, DeepSeek cells; gate is LLM-independent)
fig, ax = plt.subplots(figsize=(3.3, 3.5))
ax.plot([0, 1], [0, 1], color=MUTED, lw=0.8, ls="--", zorder=1)
data["reliability"] = {}
for name, dm in (("MSD1", "msd1"), ("Jev", "jev"), ("Lexical", "lexical")):
    p, y = [], []
    for f in (ROOT / "results/openrouter_v1").glob(f"bfcl__*__dm_gate__deepseek__{dm}.jsonl"):
        for r in rows(f):
            p.append(r["p_tool_call"])
            y.append(bool(r["should_call"]))
    p, y = np.asarray(p), np.asarray(y, float)
    edges = np.linspace(0, 1, 11)
    xs, ys, ns = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & ((p < hi) if hi < 1 else (p <= hi))
        if m.sum() >= 5:
            xs.append(p[m].mean()); ys.append(y[m].mean()); ns.append(int(m.sum()))
    a, e = auroc(p, y), ece(p, y)
    data["reliability"][name] = {"n": int(len(p)), "auroc": round(float(a), 3), "ece": round(float(e), 3),
                                 "bins": list(zip([round(x, 3) for x in xs], [round(v, 3) for v in ys], ns))}
    ax.plot(xs, ys, color=C[name], lw=1.5, marker=M[name], ms=5, label=f"{name}  (AUROC {a:.2f}, ECE {e:.2f})", zorder=3)
ax.set_xlabel("predicted P(tool_call), binned")
ax.set_ylabel("fraction of items that should call a tool")
ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
ax.legend(loc="upper center", fontsize=7, bbox_to_anchor=(0.45, -0.17), ncol=1)  # below the plot, off the data
ax.set_title("Gate calibration, BFCL (n = 860)", loc="left", color=INK)
fig.tight_layout()
fig.savefig(OUT / "fig_reliability.pdf")
plt.close(fig)

# ---- Figure 3: selection accuracy and cost vs holster size (Study 2, padded BFCL)
H = ROOT / "results/holster_v1"
sel = defaultdict(lambda: defaultdict(list))
cost = defaultdict(lambda: defaultdict(list))
for name, f, kind in (("MSD1", "dm__msd1", "dm"), ("Jev", "dm__jev", "dm"), ("Lexical", "lexical__-", "dm"),
                      ("DeepSeek", "llm_native__deepseek", "llm"), ("Sonnet", "llm_native__sonnet", "llm")):
    for r in rows(H / f"{f}.jsonl"):
        if r["src"] != "bfcl_pad":
            continue
        cost[name][r["K"]].append(r.get("cost_usd") or 0.0)
        if not r["should_call"]:
            continue
        ok = r["select_ok"] if kind == "dm" else (bool(r["calls"]) and sorted(next(iter(c)) for c in r["calls"]) == r["gold_tools"])
        sel[name][r["K"]].append(bool(ok))
Ks = [25, 100, 200]
X = {25: 0, 100: 1, 200: 2}  # evenly spaced categories; the sizes are design points, not a continuum
NUDGE = {"MSD1": -6, "Jev": 6, "DeepSeek": 0, "Sonnet": 0, "Lexical": 0}  # keep end labels apart
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.6, 2.7))
data["scaling"] = {}
for name in ("MSD1", "Jev", "DeepSeek", "Sonnet", "Lexical"):
    ks = [k for k in Ks if sel[name].get(k)]
    acc = [np.mean(sel[name][k]) for k in ks]
    n = [len(sel[name][k]) for k in ks]
    cst = [1000 * np.mean(cost[name][k]) for k in ks]
    data["scaling"][name] = {"K": ks, "accuracy": [round(x, 3) for x in acc], "n": n, "cost_per_1k": [round(x, 3) for x in cst]}
    lab = name + (" (n=81)" if name == "Sonnet" else "")
    xs = [X[k] for k in ks]
    a1.plot(xs, acc, color=C[name], lw=1.5, marker=M[name], ms=5, label=lab)
    if name != "Lexical":  # lexical costs nothing; log axis
        a2.plot(xs, cst, color=C[name], lw=1.5, marker=M[name], ms=5)
        a2.annotate(name, (xs[-1], cst[-1]), xytext=(6, NUDGE[name]), textcoords="offset points", va="center", fontsize=7, color=INK)
a1.legend(loc="center", fontsize=7, bbox_to_anchor=(0.66, 0.56))  # 5 series: legend, no end labels
a2.set_yscale("log")
for a in (a1, a2):
    a.set_xticks([0, 1, 2]); a.set_xticklabels([str(k) for k in Ks]); a.set_xlim(-0.2, 2.75)
    a.set_xlabel("tools in the holster (K)")
a1.set_ylabel("right tool picked (should-call items)")
a1.set_ylim(0.35, 1.0)
a1.set_title("(a) Selection accuracy", loc="left", color=INK)
a2.set_ylabel("USD per 1,000 requests (log)")
a2.set_title("(b) Cost of the selecting call", loc="left", color=INK)
fig.tight_layout()
fig.savefig(OUT / "fig_scaling.pdf")
plt.close(fig)

(OUT / "figure_data.json").write_text(json.dumps(data, indent=1))
print(json.dumps(data, indent=1))
