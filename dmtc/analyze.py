"""Summarises a results directory.

    python -m dmtc.analyze results/default

Writes summary.csv / summary.md (one row per cell), paired.csv (McNemar vs llm_native with the same
LLM), calibration.csv (DM gate calibration) and cascade.csv (offline cascade sweep: dm_gate_select
when the DM is confident, llm_native otherwise, simulated from the logged rows of both).
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from dmtc.data import WHEN2CALL_LABELS, iter_jsonl
from dmtc.eval.metrics import auroc, bootstrap_ci, brier, cost_usd, ece, holm, macro_f1, mcnemar


def _prices(cfg: dict[str, Any], llm: str | None, dm: str | None) -> dict[str, Any]:
    p = cfg.get("prices", {})
    return {"llm": p.get(llm, {}) if llm else {}, "dm": p.get(dm, {}) if dm else {}}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as f:
        fields = list(dict.fromkeys(k for r in rows for k in r))
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _fmt(x: Any) -> str:
    return f"{x:.3f}" if isinstance(x, float) else str(x)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir")
    args = ap.parse_args()
    d = Path(args.results_dir)
    cfg = yaml.safe_load(open(d / "config.yaml")) if (d / "config.yaml").exists() else {}

    cells: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for path in sorted(d.glob("*.jsonl")):
        for r in iter_jsonl(path):
            # When2Call categories are its sources; aggregate the benchmark as one cell.
            cat = "all" if r["benchmark"] == "when2call" else r["category"]
            cells[(r["benchmark"], cat, r["condition"], r["llm"], r["dm"])].append(r)

    summary, paired, calib = [], [], []
    for key, rows in sorted(cells.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        bench, cat, cond, llm, dm = key
        prices = _prices(cfg, llm, dm)
        correct = [bool(r["correct"]) for r in rows]
        lo, hi = bootstrap_ci(correct)
        lat = np.array([r["latency_s"] for r in rows])
        sel = [r["selection_correct"] for r in rows if r.get("selection_correct") is not None]
        names = [r["call_names_correct"] for r in rows if r.get("call_names_correct") is not None]
        row = {
            "benchmark": bench, "category": cat, "condition": cond, "llm": llm or "-", "dm": dm or "-",
            "n": len(rows), "accuracy": float(np.mean(correct)), "ci_low": lo, "ci_high": hi,
            "call_decision_acc": float(np.mean([r["call_decision_correct"] for r in rows])),
            "selection_acc": float(np.mean(sel)) if sel else float("nan"),
            "call_names_acc": float(np.mean(names)) if names else float("nan"),
            "latency_p50_s": float(np.median(lat)), "latency_p95_s": float(np.quantile(lat, 0.95)),
            "llm_calls_per_ex": float(np.mean([r["llm_calls"] for r in rows])),
            "llm_out_tokens_per_ex": float(np.mean([r["llm_output_tokens"] for r in rows])),
            "cost_per_1k_usd": 1000 * float(np.mean([cost_usd(r, prices) for r in rows])),
            "error_rate": float(np.mean([bool(r["errors"]) for r in rows])),
        }
        row["macro_f1"] = float("nan")
        row["variant_set_acc"] = float("nan")
        if bench == "when2call":
            # all counterfactual variants of the same BFCL item right (tool kept / removed / info dropped)
            by_src: dict[str, list[bool]] = defaultdict(list)
            for r in rows:
                by_src[r.get("source_id") or r["id"]].append(bool(r["correct"]))
            multi = [all(v) for v in by_src.values() if len(v) > 1]
            row["variant_set_acc"] = float(np.mean(multi)) if multi else float("nan")
            row["macro_f1"] = macro_f1([r["gold_label"] for r in rows], [r["pred_label"] for r in rows],
                                       WHEN2CALL_LABELS)
        summary.append(row)

        # Paired comparison against the all-LLM baseline on the same examples.
        if cond != "llm_native":
            base = {r["id"]: r for r in cells.get((bench, cat, "llm_native", llm, None), [])}
            if llm is None:  # dm_full has no LLM: compare against every llm_native baseline
                bases = [(k[3], {r["id"]: r for r in v}) for k, v in cells.items()
                         if k[:3] == (bench, cat, "llm_native")]
            else:
                bases = [(llm, base)] if base else []
            for base_llm, b in bases:
                ids = [r["id"] for r in rows if r["id"] in b]
                if not ids:
                    continue
                mine = {r["id"]: r for r in rows}
                a_ok = [bool(b[i]["correct"]) for i in ids]
                c_ok = [bool(mine[i]["correct"]) for i in ids]
                t = mcnemar(a_ok, c_ok)
                paired.append({"benchmark": bench, "category": cat, "condition": cond, "llm": llm or "-",
                               "dm": dm or "-", "baseline_llm": base_llm, "n_paired": len(ids),
                               "baseline_acc": float(np.mean(a_ok)), "condition_acc": float(np.mean(c_ok)),
                               "delta": float(np.mean(c_ok) - np.mean(a_ok)), **t})

        gated = [r for r in rows if r.get("p_tool_call") is not None]
        if gated:
            p = [r["p_tool_call"] for r in gated]
            y = [bool(r["should_call"]) for r in gated]
            calib.append({
                "benchmark": bench, "category": cat, "condition": cond, "dm": dm, "n": len(gated),
                "gate_acc@0.5": float(np.mean([(pi >= 0.5) == yi for pi, yi in zip(p, y)])),
                "brier": brier(p, y),
                "auroc": auroc(p, y),
                "ece_binary": ece([max(pi, 1 - pi) for pi in p], [(pi >= 0.5) == yi for pi, yi in zip(p, y)]),
            })

    # Gate quality pooled over categories: within one BFCL category every item has the same
    # should_call, so AUROC is only defined on the pool (AST + relevance vs. irrelevance items).
    pooled: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for (bench, cat, cond, llm, dm), rows in cells.items():
        pooled[(bench, cond, llm, dm)] += [r for r in rows if r.get("p_tool_call") is not None]
    for (bench, cond, llm, dm), rows in sorted(pooled.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        if bench != "bfcl" or not rows:
            continue
        p = [r["p_tool_call"] for r in rows]
        y = [bool(r["should_call"]) for r in rows]
        calib.append({
            "benchmark": bench, "category": "_pooled", "condition": cond, "dm": dm, "n": len(rows),
            "gate_acc@0.5": float(np.mean([(pi >= 0.5) == yi for pi, yi in zip(p, y)])),
            "brier": brier(p, y), "auroc": auroc(p, y),
            "ece_binary": ece([max(pi, 1 - pi) for pi in p], [(pi >= 0.5) == yi for pi, yi in zip(p, y)]),
        })

    # Offline cascade sweep.
    cascade = []
    for (bench, cat, cond, llm, dm), rows in cells.items():
        if cond != "dm_gate_select":
            continue
        base = {r["id"]: r for r in cells.get((bench, cat, "llm_native", llm, None), [])}
        joint = [(r, base[r["id"]]) for r in rows if r["id"] in base]
        if not joint:
            continue
        pr_dm, pr_llm = _prices(cfg, llm, dm), _prices(cfg, llm, None)
        for tau in np.round(np.linspace(0.5, 1.0, 11), 2):
            ok, cost, fb = [], [], 0
            for r, b in joint:
                pt = r.get("p_tool_call") or 0.0
                conf = min(max(pt, 1 - pt), max((r.get("tool_probs") or {"": 1.0}).values()))
                if conf < tau:
                    fb += 1
                    ok.append(bool(b["correct"]))
                    cost.append(cost_usd({**r, "llm_input_tokens": 0, "llm_output_tokens": 0, "llm_cost_usd": 0.0}, pr_dm)
                                + cost_usd(b, pr_llm))
                else:
                    ok.append(bool(r["correct"]))
                    cost.append(cost_usd(r, pr_dm))
            cascade.append({"benchmark": bench, "category": cat, "llm": llm, "dm": dm, "tau": float(tau),
                            "n": len(joint), "fallback_rate": fb / len(joint), "accuracy": float(np.mean(ok)),
                            "cost_per_1k_usd": 1000 * float(np.mean(cost))})

    for row, adj in zip(paired, holm([r["p_value"] for r in paired])):
        row["p_holm"] = adj
    _write_csv(d / "summary.csv", summary)
    _write_csv(d / "paired.csv", paired)
    _write_csv(d / "calibration.csv", calib)
    _write_csv(d / "cascade.csv", cascade)

    cols = ["benchmark", "category", "condition", "llm", "dm", "n", "accuracy", "ci_low", "ci_high",
            "call_decision_acc", "selection_acc", "call_names_acc", "latency_p50_s", "cost_per_1k_usd"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(_fmt(r[c]) for c in cols) + " |" for r in summary]
    (d / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
