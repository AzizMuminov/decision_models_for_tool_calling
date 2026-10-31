"""Multi-gold labels for the coding-agent items of SynthHolster.

A coding agent often has several valid next steps (look at git status before finishing, find the files
before a multi-file patch). Single-gold scoring punishes those. A reviewer that is NOT evaluated sees
the conversation and ALL coding tools (not just the item's family), without the original gold, and
lists every reasonable next step plus the single best one. Items whose original gold is not in that
set are flagged unreliable and excluded from multi-gold scoring.

Scoring needs no new model calls: the saved answers are re-scored.
  strict:     the system's tool(s) == the original gold (as before)
  multi-gold: every tool the system calls is in the acceptable set (DMs: the pick is in the set)

    python -m dmtc.synth.multigold annotate --data data/synth
    python -m dmtc.synth.multigold score results/synth_v2 --data data/synth
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from dmtc.eval.metrics import bootstrap_ci, mcnemar

REVIEWER = "openai/gpt-6.1-sol"
PROMPT = """A coding agent is working in a software repository. Here is the situation at the moment it must choose its next action:

{conv}

Available tools:
{tools}

1. List EVERY tool that a competent coding agent could reasonably call as its very next step here, including
   sensible preparatory or verification steps. Exclude tools that would be useless, risky or off-task.
2. Name the single best next tool.
Return JSON only: {{"acceptable": [tool names], "best": "tool name"}}"""


def annotate(data: Path) -> None:
    import openai

    cli = openai.OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])
    cat = json.loads((data / "catalog.json").read_text())
    dom = next(d for d in cat["domains"] if d["brand"] == "CodingAgent")
    tools = [v for f in dom["families"] for v in f["variants"]]
    listing = "\n".join(f"- {v['name']}: {v['description']}" for v in tools)
    names = {v["name"] for v in tools}
    items = [json.loads(l) for l in open(data / "items.jsonl")][1:]
    items = [i for i in items if i["domain"] == "CodingAgent" and i["label"] == "tool_call"]

    def one(it):
        conv = "\n".join(f"{m['role']}: {m['content']}" for m in it["messages"])
        for k in range(5):
            try:
                r = cli.chat.completions.create(model=REVIEWER, max_tokens=1500, extra_body={"reasoning": {"effort": "low"}},
                                                messages=[{"role": "user", "content": PROMPT.format(conv=conv, tools=listing)}])
                txt = r.choices[0].message.content or ""
                j = json.loads(txt[txt.index("{"): txt.rindex("}") + 1])
                acc = sorted({t for t in j.get("acceptable", []) if t in names})
                return it["id"], {"acceptable": acc, "best": j.get("best"), "gold": it["gold_tool"],
                                  "gold_in_set": it["gold_tool"] in acc}, float(getattr(r.usage, "cost", 0) or 0)
            except Exception:  # noqa: BLE001
                time.sleep(3 * (k + 1))
        return it["id"], None, 0.0

    with ThreadPoolExecutor(8) as ex:
        res = list(ex.map(one, items))
    out = {i: a for i, a, _ in res if a}
    (data / "coding_multigold.json").write_text(json.dumps({"reviewer": REVIEWER, "items": out}, indent=1))
    sizes = [len(a["acceptable"]) for a in out.values()]
    print(f"annotated {len(out)}/{len(items)} coding items; cost ${sum(c for *_, c in res):.4f}")
    print(f"acceptable-set size: mean {np.mean(sizes):.2f}, distribution {dict(sorted(Counter(sizes).items()))}")
    print(f"original gold in acceptable set: {np.mean([a['gold_in_set'] for a in out.values()]):.1%}; "
          f"reviewer's best == original gold: {np.mean([a['best'] == a['gold'] for a in out.values()]):.1%}")


def score(results: Path, data: Path) -> None:
    mg = json.loads((data / "coding_multigold.json").read_text())["items"]
    ok_items = {i for i, a in mg.items() if a["gold_in_set"]}
    systems = {}
    for p in sorted(results.glob("*.jsonl")):
        for l in open(p):
            r = json.loads(l)
            if r["id"] not in ok_items:
                continue
            if r["system"] == "dm":
                key = f"{r['model']} (DM, τ=0.5)"
                strict = r["ok@0.5"]
                called = {r["pick"]} if r["label@0.5"] == "tool_call" else set()
            else:
                key = f"{r['model']} (LLM)"
                called = {next(iter(c)) for c in r.get("calls") or []}
                strict = called == {r["gold_tool"]}
            multi = bool(called) and called <= set(mg[r["id"]]["acceptable"])
            systems.setdefault(key, {}).setdefault(r["K_kind"], {})[r["id"]] = (strict, multi)
    L = [f"Coding items with a reliable gold (original gold in the reviewer's acceptable set): {len(ok_items)}", "",
         "| system | holster | strict (single gold) | multi-gold (any acceptable next step) |", "|---|---|---|---|"]
    for name, by in systems.items():
        for kind in ("family+25", "full"):
            v = by.get(kind, {})
            if v:
                s = [x[0] for x in v.values()]
                m = [x[1] for x in v.values()]
                lo, hi = bootstrap_ci(m)
                L.append(f"| {name} | {kind} | {np.mean(s):.3f} | {np.mean(m):.3f} (±{(hi - lo) / 2:.3f}) |")
    L += ["", "Paired multi-gold, DM vs LLM (McNemar):", ""]
    for dm in [k for k in systems if "DM" in k]:
        for llm in [k for k in systems if "LLM" in k]:
            for kind in ("family+25", "full"):
                A, B = systems[dm].get(kind, {}), systems[llm].get(kind, {})
                ids = sorted(set(A) & set(B))
                if ids:
                    t = mcnemar([B[i][1] for i in ids], [A[i][1] for i in ids])
                    L.append(f"- {dm} vs {llm}, {kind}: {np.mean([A[i][1] for i in ids]):.3f} vs "
                             f"{np.mean([B[i][1] for i in ids]):.3f} (n={len(ids)}, p={t['p_value']:.2g})")
    (results / "CODING_MULTIGOLD.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["annotate", "score"])
    ap.add_argument("results", nargs="?")
    ap.add_argument("--data", default="data/synth")
    a = ap.parse_args()
    annotate(Path(a.data)) if a.cmd == "annotate" else score(Path(a.results), Path(a.data))


if __name__ == "__main__":
    main()
