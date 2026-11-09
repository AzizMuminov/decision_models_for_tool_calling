"""Retrieval benchmark on the real catalog: can each mechanism put a right tool in its results?

Queries: harness/retrieval_queries.py, two per needed step of every task (natural, keywords), 120 in all.
A result is a hit when it contains any tool of that step's acceptable group. Mechanisms: Claude Code's
ToolSearch (cc), BM25, embeddings, LLM router, and the decision model with four ways around its 255-option
limit (hierarchy, tournament, bm25_dm, embed_dm). Copilot-style groups need the agent to pick a group, so
they are measured only in the live runs.

--size extra adds 18 more real MCP servers (1,076 tools in all) to test growth.
Results are cached in results/retrieval_real/, so re-running only pays for what is missing.

    source .or_env && .venv/bin/python harness/eval_real_retrieval.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from real_sim import load_catalog  # noqa: E402
from retrieval_queries import Q  # noqa: E402
from selectors_real import Selectors  # noqa: E402

OUT = HERE.parent / "results" / "retrieval_real"
MECHS = [("cc", {}), ("bm25", {}), ("embed", {}), ("llm", {}), ("dm", {"strategy": "hierarchy"}), ("dm", {"strategy": "tournament"}),
         ("dm", {"strategy": "bm25_dm"}), ("dm", {"strategy": "embed_dm"})]


def label(m, kw):
    return m if not kw else f"dm_{kw['strategy']}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="real", choices=["real", "extra"])
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    tools = load_catalog(extra=a.size == "extra")
    sel = Selectors(tools)
    tasks = {p.name: json.loads((p / "task.json").read_text()) for p in (HERE / "tasks_real").iterdir()}
    cache_f = OUT / f"cache_{a.size}.jsonl"
    cache = {}
    if cache_f.exists():
        for line in cache_f.read_text().splitlines():
            r = json.loads(line)
            cache[(r["mech"], r["query"])] = r
    items = []
    for (task, step), (nat, kw) in Q.items():
        group = tasks[task]["needs"][step]
        for style, q in (("natural", nat), ("keywords", kw)):
            items.append({"task": task, "step": step, "style": style, "query": q, "group": group})
    sel.embed("warm up", 1)  # builds the tool embedding cache once

    def do(job):
        it, (m, kw) = job
        name = label(m, kw)
        if (name, it["query"]) in cache:
            return cache[(name, it["query"])]
        t0 = time.time()
        try:
            ranked, cost = sel.search(m, it["query"], 5, **kw)
            err = None
        except Exception as e:  # noqa: BLE001
            ranked, cost, err = [], 0.0, f"{type(e).__name__}: {e}"
        r = {"mech": name, "query": it["query"], "ranked": ranked, "cost": cost, "latency": round(time.time() - t0, 3), "error": err}
        if not err:
            with open(cache_f, "a") as f:
                f.write(json.dumps(r) + "\n")
        return r

    jobs = [(it, mk) for it in items for mk in MECHS]
    with ThreadPoolExecutor(a.workers) as ex:
        res = list(ex.map(do, jobs))
    rows = []
    for (it, mk), r in zip(jobs, res):
        g = set(it["group"])
        rank = next((i + 1 for i, n in enumerate(r["ranked"]) if n in g), None)
        rows.append({**{k: it[k] for k in ("task", "step", "style", "query")}, "mech": r["mech"], "hit1": rank == 1, "hit3": bool(rank and rank <= 3),
                     "hit5": rank is not None, "rr": 1 / rank if rank else 0.0, "cost": r["cost"], "latency": r["latency"], "error": r["error"]})
    (OUT / f"rows_{a.size}.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    print(f"catalog: {len(tools)} tools, {len(items)} queries")
    print(f"{'mechanism':16s} {'hit@1':>6s} {'hit@3':>6s} {'hit@5':>6s} {'MRR':>6s} {'nat@5':>6s} {'kw@5':>6s} {'$/1k q':>7s} {'p50 s':>6s} err")
    for name in dict.fromkeys(label(m, kw) for m, kw in MECHS):
        rs = [r for r in rows if r["mech"] == name]
        f = lambda k, sub=rs: sum(r[k] for r in sub) / max(1, len(sub))  # noqa: E731
        lat = sorted(r["latency"] for r in rs)[len(rs) // 2]
        print(f"{name:16s} {f('hit1'):6.3f} {f('hit3'):6.3f} {f('hit5'):6.3f} {f('rr'):6.3f} "
              f"{f('hit5', [r for r in rs if r['style'] == 'natural']):6.3f} {f('hit5', [r for r in rs if r['style'] == 'keywords']):6.3f} "
              f"{1000 * f('cost'):7.3f} {lat:6.2f} {sum(bool(r['error']) for r in rs)}")


if __name__ == "__main__":
    main()
