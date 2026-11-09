"""Offline test of DM v3 (flat tournament + null option) against the hierarchy DM and BM25, on the 1,076-tool catalog.

Positive queries: harness/retrieval_queries.Q (120, each has acceptable tools). Negative queries: retrieval_queries.NEG
(40 requests no catalog tool serves). Reports hit@1/hit@5 on positives, how often the selector returned nothing on
positives (false null) and on negatives (true null), the mean number of tools returned, cost and latency.
Cached in results/retrieval_real/null_cache.jsonl.

    source .or_env && .venv/bin/python harness/eval_null.py
"""

from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from real_sim import load_catalog  # noqa: E402
from retrieval_queries import NEG, Q  # noqa: E402
from selectors_real import Selectors  # noqa: E402

OUT = HERE.parent / "results" / "retrieval_real"
MECHS = ["bm25", "dm", "dm3", "dm4", "dm_null"]


def main() -> None:
    sel = Selectors(load_catalog(extra=True))
    tasks = {p.name: json.loads((p / "task.json").read_text()) for p in (HERE / "tasks_real").iterdir()}
    items = [{"query": q, "group": set(tasks[t]["needs"][s])} for (t, s), pair in Q.items() for q in pair]
    items += [{"query": q, "group": None} for q in NEG]
    cache_f = OUT / "null_cache.jsonl"
    cache = {}
    if cache_f.exists():
        for line in cache_f.read_text().splitlines():
            r = json.loads(line)
            cache[(r["mech"], r["query"])] = r

    def do(job):
        it, m = job
        if (m, it["query"]) in cache:
            return cache[(m, it["query"])]
        t0 = time.time()
        if m == "dm_null":
            ranked, cost = sel.dm_null(it["query"], 5, tau=2.0)  # tau>1: never empty; record P(none) to sweep tau below
        else:
            ranked, cost = sel.search(m, it["query"], 5)
        r = {"mech": m, "query": it["query"], "ranked": ranked, "cost": cost, "latency": round(time.time() - t0, 2),
             "p_null": getattr(sel, "last_p_null", None) if m == "dm_null" else None}
        with open(cache_f, "a") as f:
            f.write(json.dumps(r) + "\n")
        return r

    jobs = [(it, m) for it in items for m in MECHS]
    with ThreadPoolExecutor(4) as ex:
        res = list(ex.map(do, jobs))
    print(f"{len(items) - len(NEG)} positive and {len(NEG)} negative queries, 1,076 tools\n")
    print(f"{'selector':8s} {'hit@1':>6s} {'hit@5':>6s} {'false null':>10s} {'true null':>9s} {'tools/answer':>12s} {'$/1k q':>7s} {'p50 s':>6s}")
    for m in MECHS:
        rs = [(it, r) for (it, mm), r in zip(jobs, res) if mm == m]
        pos = [(it, r) for it, r in rs if it["group"]]
        neg = [r for it, r in rs if not it["group"]]
        h1 = sum(bool(r["ranked"]) and r["ranked"][0] in it["group"] for it, r in pos) / len(pos)
        h5 = sum(bool(it["group"] & set(r["ranked"][:5])) for it, r in pos) / len(pos)
        fnull = sum(not r["ranked"] for _, r in pos) / len(pos)
        tnull = sum(not r["ranked"] for r in neg) / len(neg)
        size = sum(len(r["ranked"][:5]) for _, r in rs) / len(rs)
        lat = sorted(r["latency"] for _, r in rs)[len(rs) // 2]
        print(f"{m:8s} {h1:6.3f} {h5:6.3f} {fnull:10.3f} {tnull:9.3f} {size:12.2f} {1000 * sum(r['cost'] for _, r in rs) / len(rs):7.3f} {lat:6.2f}")


def sweep() -> None:
    """dm_null at different tau, from the recorded P(none) (no API calls)."""
    tasks = {p.name: json.loads((p / "task.json").read_text()) for p in (HERE / "tasks_real").iterdir()}
    gold = {q: set(tasks[t]["needs"][s]) for (t, s), pair in Q.items() for q in pair}
    rs = [json.loads(l) for l in open(OUT / "null_cache.jsonl")]
    rs = [r for r in rs if r["mech"] == "dm_null"]
    print("\ndm_null by tau: hit@1  hit@5  false-null  true-null")
    for tau in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        pos = [r for r in rs if r["query"] in gold]
        neg = [r for r in rs if r["query"] not in gold]
        out = lambda r: [] if r["p_null"] >= tau else r["ranked"]  # noqa: E731
        h1 = sum(bool(out(r)) and out(r)[0] in gold[r["query"]] for r in pos) / len(pos)
        h5 = sum(bool(gold[r["query"]] & set(out(r))) for r in pos) / len(pos)
        print(f"  tau {tau:.1f}:   {h1:.3f}  {h5:.3f}   {sum(not out(r) for r in pos) / len(pos):.3f}       {sum(not out(r) for r in neg) / len(neg):.3f}")


if __name__ == "__main__":
    if "--sweep" in sys.argv:
        sweep()
        sys.exit()
    main()
