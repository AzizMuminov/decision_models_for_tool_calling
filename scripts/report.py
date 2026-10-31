"""Turns an analyzed results directory into a findings report (results/<run>/REPORT.md).

    .venv/bin/python -m dmtc.analyze results/openrouter_v1
    .venv/bin/python scripts/report.py results/openrouter_v1
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

GATE_CATS = {"irrelevance", "live_irrelevance"}
SELECT_CATS = {"multiple", "live_multiple"}
TRIVIAL_SELECT = {"simple_python", "live_simple"}


def load(p: Path) -> list[dict]:
    return list(csv.DictReader(open(p))) if p.exists() else []


def f(x, d=3) -> str:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "–"
    return "–" if np.isnan(v) else f"{v:.{d}f}"


def main() -> None:
    d = Path(sys.argv[1])
    summ, paired, calib, casc = (load(d / n) for n in ("summary.csv", "paired.csv", "calibration.csv", "cascade.csv"))
    L = [f"# Findings: {d.name}", ""]

    # 1. accuracy table: BFCL groups + When2Call, per condition x llm x dm
    groups = {"BFCL gate negatives": GATE_CATS, "BFCL selection (multi-tool)": SELECT_CATS,
              "BFCL single-tool": TRIVIAL_SELECT, "BFCL parallel_multiple": {"parallel_multiple"}}
    cell = defaultdict(dict)
    for r in summ:
        key = (r["condition"], r["llm"], r["dm"])
        if r["benchmark"] == "when2call":
            cell[key]["When2Call"] = (float(r["accuracy"]), int(r["n"]))
            cell[key]["W2C variant-set"] = (float(r["variant_set_acc"]), int(r["n"]))
            cell[key]["W2C macro-F1"] = (float(r["macro_f1"]), int(r["n"]))
        else:
            for g, cats in groups.items():
                if r["category"] in cats:
                    cell[key].setdefault(g, []).append((float(r["accuracy"]), int(r["n"])))
    cols = list(groups) + ["When2Call", "W2C variant-set", "W2C macro-F1"]

    def val(v):
        if isinstance(v, list):  # n-weighted mean over categories in the group
            n = sum(x[1] for x in v)
            return sum(a * k for a, k in v) / n if n else float("nan")
        return v[0] if v else float("nan")

    L += ["## Accuracy by condition", "",
          "Groups are n-weighted over their BFCL categories. `dm_full` uses no LLM.", "",
          "| condition | LLM | DM | " + " | ".join(cols) + " |", "|" + "---|" * (3 + len(cols))]
    order = ["llm_native", "oracle_select", "dm_gate", "dm_select", "dm_gate_select", "dm_full"]
    for key in sorted(cell, key=lambda k: (order.index(k[0]) if k[0] in order else 99, k[1], k[2])):
        L.append(f"| {key[0]} | {key[1]} | {key[2]} | " + " | ".join(f(val(cell[key].get(c))) for c in cols) + " |")

    # 2. paired deltas vs llm_native (pooled over BFCL categories), with Holm-significant cells counted
    L += ["", "## Paired change vs. llm_native (same LLM, same items)", "",
          "| condition | LLM | DM | BFCL Δ acc (pooled) | BFCL cells sig. worse / better (Holm p<0.05) | When2Call Δ |",
          "|---|---|---|---|---|---|"]
    pool = defaultdict(lambda: {"n": 0, "d": 0.0, "worse": 0, "better": 0, "w2c": None})
    for r in paired:
        k = (r["condition"], r["llm"], r["dm"], r["baseline_llm"])
        if r["benchmark"] == "when2call":
            pool[k]["w2c"] = float(r["delta"])
            continue
        n = int(r["n_paired"])
        pool[k]["n"] += n
        pool[k]["d"] += float(r["delta"]) * n
        if float(r.get("p_holm", 1)) < 0.05:
            pool[k]["worse" if float(r["delta"]) < 0 else "better"] += 1
    for k, v in sorted(pool.items(), key=lambda kv: (order.index(kv[0][0]) if kv[0][0] in order else 99, kv[0][1:])):
        llm = k[1] if k[1] != "-" else f"(vs {k[3]})"
        L.append(f"| {k[0]} | {llm} | {k[2]} | {f(v['d'] / v['n'] if v['n'] else float('nan'))} | "
                 f"{v['worse']} / {v['better']} | {f(v['w2c'])} |")

    # 3. gate quality of each DM on its own
    L += ["", "## The DM gate on its own", "",
          "BFCL pooled = AST + relevance items (should call) vs irrelevance items (should not).", "",
          "| DM | benchmark | n | acc@0.5 | AUROC | Brier | ECE |", "|---|---|---|---|---|---|---|"]
    seen = set()
    for r in calib:
        if r["condition"] != "dm_gate":
            continue
        if r["category"] not in ("_pooled", "all"):
            continue
        k = (r["dm"], r["benchmark"])
        if k in seen:
            continue
        seen.add(k)
        L.append(f"| {r['dm']} | {r['benchmark']} | {r['n']} | {f(r['gate_acc@0.5'])} | {f(r['auroc'])} | "
                 f"{f(r['brier'])} | {f(r['ece_binary'])} |")

    # 4. latency + cost per 1k items
    L += ["", "## Latency and cost (all benchmarks pooled)", "",
          "| condition | LLM | DM | p50 latency s | LLM calls/item | LLM out tokens/item | cost per 1k items $ |",
          "|---|---|---|---|---|---|---|"]
    agg = defaultdict(lambda: defaultdict(list))
    for r in summ:
        k = (r["condition"], r["llm"], r["dm"])
        n = int(r["n"])
        for m in ("latency_p50_s", "llm_calls_per_ex", "llm_out_tokens_per_ex", "cost_per_1k_usd"):
            agg[k][m].append((float(r[m]), n))
    for k in sorted(agg, key=lambda k: (order.index(k[0]) if k[0] in order else 99, k[1], k[2])):
        m = {kk: sum(a * n for a, n in v) / sum(n for _, n in v) for kk, v in agg[k].items()}
        L.append(f"| {k[0]} | {k[1]} | {k[2]} | {f(m['latency_p50_s'], 2)} | {f(m['llm_calls_per_ex'], 2)} | "
                 f"{f(m['llm_out_tokens_per_ex'], 0)} | {f(m['cost_per_1k_usd'], 3)} |")

    # 5. cascade frontier (pooled over BFCL categories)
    L += ["", "## Cascade frontier (offline): dm_gate_select when the DM is confident, else llm_native", "",
          "| LLM | DM | τ | fallback rate | BFCL acc | cost per 1k $ |", "|---|---|---|---|---|---|"]
    cf = defaultdict(lambda: [0, 0.0, 0.0, 0.0])
    for r in casc:
        if r["benchmark"] != "bfcl":
            continue
        k = (r["llm"], r["dm"], float(r["tau"]))
        n = int(r["n"])
        cf[k][0] += n
        cf[k][1] += float(r["fallback_rate"]) * n
        cf[k][2] += float(r["accuracy"]) * n
        cf[k][3] += float(r["cost_per_1k_usd"]) * n
    for k in sorted(cf):
        if k[2] in (0.5, 0.7, 0.8, 0.9, 0.95, 1.0):
            n, fb, acc, cost = cf[k]
            L.append(f"| {k[0]} | {k[1]} | {k[2]} | {f(fb / n)} | {f(acc / n)} | {f(cost / n, 3)} |")

    (d / "REPORT.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
