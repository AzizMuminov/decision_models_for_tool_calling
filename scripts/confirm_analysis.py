"""Pre-registered analysis of the coding confirmation run (data/synth_confirm/PREREGISTRATION.md).

Written before the results existed. Primary metric "acceptable" on the `full` holster:
  DM (no gate):  the pick is in the item's acceptable set
  LLM:           it called >= 1 tool and every called tool is in the acceptable set (text reply = wrong)
H1: MSD1 > each of deepseek, glm, mimo.   H2: Jev > deepseek, mimo.
Exact two-sided McNemar, Holm over the 5 comparisons; supported if Holm p < 0.05 in the stated direction.

    .venv/bin/python scripts/confirm_analysis.py results/synth_confirm data/synth_confirm
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dmtc.eval.metrics import bootstrap_ci, holm, mcnemar  # noqa: E402


def main() -> None:
    res, data = Path(sys.argv[1]), Path(sys.argv[2])
    mg = json.loads((data / "coding_multigold.json").read_text())["items"]
    ok_ids = {i for i, a in mg.items() if a["gold_in_set"]}
    S: dict[str, dict] = {}
    errs: dict[str, Counter] = {}
    for p in sorted(res.glob("*.jsonl")):
        for l in open(p):
            r = json.loads(l)
            if r["id"] not in ok_ids or r.get("error"):
                continue
            acc = set(mg[r["id"]]["acceptable"])
            if r["system"] == "dm":
                name = {"msd1": "MSD1", "jev": "Jev"}[r["model"]]
                called = {r["pick"]}
            else:
                name = {"deepseek": "DeepSeek", "glm": "GLM", "mimo": "MiMo"}[r["model"]]
                called = {next(iter(c)) for c in r.get("calls") or []}
            a_ok = bool(called) and called <= acc
            b_ok = called == {r["gold_tool"]}
            S.setdefault(name, {}).setdefault(r["K_kind"], {})[r["id"]] = (a_ok, b_ok)
            if not a_ok:
                errs.setdefault(f"{name}/{r['K_kind']}", Counter())["text reply" if not called else "tool outside acceptable set"] += 1

    L = ["# Coding confirmation run: pre-registered analysis", "",
         f"Items analysed: {len(ok_ids)} (original gold inside the reviewer's acceptable set).", "",
         "| system | holster | acceptable (primary) | 95% CI | best (secondary) |", "|---|---|---|---|---|"]
    for name in ("MSD1", "Jev", "GLM", "MiMo", "DeepSeek"):
        for kind in ("full", "family+25"):
            v = S.get(name, {}).get(kind, {})
            if v:
                a = [x[0] for x in v.values()]
                lo, hi = bootstrap_ci(a)
                L.append(f"| {name} | {kind} | {np.mean(a):.3f} | {lo:.3f}–{hi:.3f} | {np.mean([x[1] for x in v.values()]):.3f} |")

    comps = [("H1", "MSD1", "DeepSeek"), ("H1", "MSD1", "GLM"), ("H1", "MSD1", "MiMo"),
             ("H2", "Jev", "DeepSeek"), ("H2", "Jev", "MiMo")]
    rows = []
    for h, dm, llm in comps:
        A, B = S[dm]["full"], S[llm]["full"]
        ids = sorted(set(A) & set(B))
        t = mcnemar([B[i][0] for i in ids], [A[i][0] for i in ids])
        rows.append((h, dm, llm, len(ids), np.mean([A[i][0] for i in ids]), np.mean([B[i][0] for i in ids]), t["p_value"]))
    adj = holm([r[-1] for r in rows])
    L += ["", "## Pre-registered tests (full holster, acceptable metric)", "",
          "| hypothesis | DM | LLM | n | DM | LLM | p | Holm p | supported |", "|---|---|---|---|---|---|---|---|---|"]
    for (h, dm, llm, n, a, b, p), q in zip(rows, adj):
        L.append(f"| {h} | {dm} | {llm} | {n} | {a:.3f} | {b:.3f} | {p:.2g} | {q:.2g} | "
                 f"{'yes' if q < 0.05 and a > b else 'no'} |")
    L += ["", "## Errors on the primary metric", ""] + [f"- {k}: {dict(v)}" for k, v in sorted(errs.items())]
    (res / "CONFIRMATION.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
