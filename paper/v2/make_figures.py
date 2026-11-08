"""Figures for paper v2, from the logged rows (no API calls).
    python3 paper/v2/make_figures.py   # reads /home/user/dm-toolcall-ablation/results, writes paper/v2/figures/
"""
import json, math, random, statistics as st, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

RES = Path(sys.argv[1] if len(sys.argv) > 1 else "/home/user/dm-toolcall-ablation/results")
OUT = Path(__file__).resolve().parent / "figures"; OUT.mkdir(exist_ok=True)
INK, MUTED = "#2b2b29", "#6b6b66"
BLUE, GRAY, ORANGE = "#2a78d6", "#8a8a85", "#eb6834"
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.titlesize": 9.5, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": INK,
                     "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})


def slots(d, a="hermes_default", b="hermes_dmfrankenbridge"):
    S = {}
    for line in open(RES / d / "rows.jsonl"):
        r = json.loads(line)
        if r["task"] == "k8s_crashloop" or r.get("error") or not r.get("total_tokens"):
            continue
        S.setdefault((r["task"], r["rep"]), {})[r["condition"]] = r
    return {k: math.log(v[b]["total_tokens"] / v[a]["total_tokens"]) for k, v in S.items() if a in v and b in v}


def task_boot(lr):
    """Geometric mean of task-level mean log ratios with a task bootstrap 95% CI (10,000 resamples, seed 1)."""
    per = {}
    for (t, _), v in lr.items():
        per.setdefault(t, []).append(v)
    tm = [sum(v) / len(v) for v in per.values()]
    n, rng = len(tm), random.Random(1)
    bs = sorted(sum(tm[rng.randrange(n)] for _ in range(n)) / n for _ in range(10000))
    return math.exp(sum(tm) / n), math.exp(bs[249]), math.exp(bs[9749]), per


# ---------------------------------------------------------------- main experiment: per-slot token ratios
rows = [("DeepSeek V4.1 Flash\n15 repeats, 555 slots", slots("final_franken")),
        ("GPT-6.1 Sol\n5 repeats, 185 slots", slots("final_franken_sol"))]
fig, ax = plt.subplots(figsize=(6.4, 2.1))
jit = random.Random(0)
for i, (lab, lr) in enumerate(rows):
    gm, lo, hi, per = task_boot(lr)
    ax.scatter([math.exp(v) for v in lr.values()], [i + jit.uniform(-0.28, 0.28) for _ in lr], s=5, color=GRAY,
               alpha=0.45, linewidths=0, zorder=2, label="single slot" if i == 0 else None)
    tms = [math.exp(sum(v) / len(v)) for v in per.values()]
    ax.scatter(tms, [i - 0.36] * len(tms), s=9, marker="|", color=INK, linewidths=0.8, zorder=3,
               label="task geometric mean" if i == 0 else None)
    ax.plot([lo, hi], [i, i], color=BLUE, lw=2.4, zorder=4)
    ax.scatter([gm], [i], color=BLUE, s=26, zorder=5, label="overall, 95% CI" if i == 0 else None)
    ax.text(9.5, i, f"{gm:.3f}\n[{lo:.2f}, {hi:.2f}]", va="center", ha="left", fontsize=8.5, color=INK)
ax.axvline(1, color=MUTED, lw=0.8, zorder=1)
ax.set_xscale("log")
ax.set_xlim(0.12, 24)
ax.set_xticks([0.125, 0.25, 0.5, 1, 2, 4, 8], ["1/8", "1/4", "1/2", "1", "2", "4", "8"])
ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
ax.set_yticks(range(len(rows)), [lab for lab, _ in rows])
ax.set_ylim(len(rows) - 0.5, -0.6)
ax.set_xlabel("tokens per run with selector F relative to BM25 in the same slot")
ax.legend(frameon=False, fontsize=7.5, loc="lower left", ncol=3, bbox_to_anchor=(0.0, 1.0), handletextpad=0.3)
fig.tight_layout()
fig.savefig(OUT / "fig_tokens.pdf")
plt.close(fig)

# ---------------------------------------------------------------- selector study, held-out agent searches
ho = {d["variant"].split()[0]: d for d in json.load(open(RES / "selector_study" / "results_1_3_5b_F_R_B.json"))}
order = [("B", "BM25"), ("1", "1: top service"), ("3", "3: nucleus,\nfixed cards"), ("5b", "5: 3 + none"),
         ("R", "R: earlier\ndesign"), ("F", "F: selector F")]
fig, ax = plt.subplots(figsize=(6.4, 2.0))
xs = range(len(order))
c5 = [ho[k]["agent_cov5"] for k, _ in order]; c15 = [ho[k]["agent_cov15"] for k, _ in order]
ax.bar([x - 0.2 for x in xs], c5, 0.38, color=GRAY, label="first 5 tools")
ax.bar([x + 0.2 for x in xs], c15, 0.38, color=BLUE, label="all 15 tools")
for x, a, b in zip(xs, c5, c15):
    ax.text(x - 0.2, a + 0.004, f"{a:.2f}", ha="center", fontsize=7.5, color=INK)
    ax.text(x + 0.2, b + 0.004, f"{b:.2f}", ha="center", fontsize=7.5, color=INK)
ax.set_xticks(list(xs), [lab for _, lab in order], fontsize=8)
ax.set_ylim(0.65, 0.88)
ax.set_ylabel("coverage of next tools")
ax.legend(frameon=False, fontsize=7.5, loc="upper left", ncol=2)
fig.tight_layout()
fig.savefig(OUT / "fig_selector_heldout.pdf")
plt.close(fig)
print("wrote", OUT)
