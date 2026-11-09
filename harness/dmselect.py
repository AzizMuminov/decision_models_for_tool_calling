#!/usr/bin/env python3
"""dmselect: the decision-model tool selector (DM v3) as one standalone command.

    bin/dmselect search "read the failed CI job log on GitLab"      # top tools, P(none), timings
    bin/dmselect search "book a flight" --json                       # machine-readable
    bin/dmselect bench                                               # accuracy + latency on the offline benchmark
    bin/dmselect serve --port 18800                                  # HTTP POST /rank {query, limit} -> {ranked, p_null, ms}

How it selects (same logic as selectors_real.Selectors.dm_null, which the running experiment uses):
  1. One DM question over the services (30 at 1,076 tools) plus a "none of these" option.
     If P(none) >= tau (0.4, set on the benchmark) the answer is empty: no catalog tool fits.
  2. Otherwise, the most likely services until they cover 90% of the probability (at most 3); one DM question per
     service over its tools (chunks of at most 150 options, well under the API's 255).
  3. A tool's score is P(service) x P(tool | service); the top k come back, never more than 5.

Needs OPENROUTER_API_KEY in the environment. Every DM call is timed; `search` prints the breakdown.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from real_sim import SERVER_INFO, load_catalog  # noqa: E402

NULL = "__none__"
MODEL = os.environ.get("DM_MODEL", "microsoft/microsoft-decision-1")


def card(t: dict) -> str:
    props = list(((t["inputSchema"] or {}).get("properties") or {}))
    desc = " ".join(t["description"].split())[:240]
    return f"[{t['server']}] {desc} Parameters: {', '.join(props) or 'none'}."


class DMSelect:
    def __init__(self, extra: bool = True, tau: float = 0.4, cap: int = 5, chunk: int = 150, mass: float = 0.9, max_services: int = 3):
        from dmtc.backends.dm import OpenRouterDM

        self.tools = load_catalog(extra=extra)
        self.by = {t["name"]: t for t in self.tools}
        self.servers = sorted({t["server"] for t in self.tools})
        self.tau, self.cap, self.chunk, self.mass, self.max_services = tau, cap, chunk, mass, max_services
        self.dm = OpenRouterDM(MODEL)

    def _ask(self, query: str, options: dict[str, str], instructions: str):
        from dmtc.backends.dm import Choice

        t0 = time.perf_counter()
        r = self.dm.ask({"what_the_agent_wants": query}, {"x": Choice(instructions, options)})
        a = r.answers.get("x")
        return (a.probabilities if a else {}), r.cost_usd, (time.perf_counter() - t0) * 1000, r.error

    def select(self, query: str, k: int = 5) -> dict:
        t0 = time.perf_counter()
        opts = {s: SERVER_INFO.get(s, s) for s in self.servers}
        opts[NULL] = "None of these services can do what the agent asked."
        ps, cost, ms_service, err = self._ask(query, opts, "Which service has the tool the agent needs? Choose None if no service can do it.")
        p_null = ps.get(NULL, 0.0)
        out = {"query": query, "p_null": round(p_null, 4), "services": [], "ranked": [], "cost_usd": cost,
               "ms": {"service": round(ms_service)}, "error": err}
        if p_null >= self.tau or not ps:
            out["ms"]["total"] = round((time.perf_counter() - t0) * 1000)
            return out
        real = {s: v for s, v in ps.items() if s != NULL}
        tot = sum(real.values()) or 1.0
        acc = 0.0
        for s in sorted(real, key=real.get, reverse=True):
            out["services"].append([s, round(real[s] / tot, 4)])
            acc += real[s] / tot
            if acc >= self.mass or len(out["services"]) >= self.max_services:
                break
        jobs = []
        for s, _ in out["services"]:
            names = [t["name"] for t in self.tools if t["server"] == s]
            jobs += [(s, names[i:i + self.chunk]) for i in range(0, len(names), self.chunk)]
        with ThreadPoolExecutor(len(jobs)) as ex:  # the per-service questions are independent: ask them in parallel
            res = list(ex.map(lambda j: (j[0], *self._ask(query, {n: card(self.by[n]) for n in j[1]}, "Which tool best serves what the agent wants to do?")), jobs))
        score = {}
        for s, pt, c, ms, e in res:
            cost += c
            out["error"] = out["error"] or e
            score.update({n: real[s] * v for n, v in pt.items()})
        out["ms"]["tools"] = round(max(r[3] for r in res)) if res else 0
        out["ranked"] = [[n, round(score[n], 4)] for n in sorted(score, key=score.get, reverse=True)[:min(k, self.cap)]]
        out["cost_usd"] = cost
        out["ms"]["total"] = round((time.perf_counter() - t0) * 1000)
        return out


def cmd_search(a) -> None:
    sel = DMSelect(extra=not a.main_catalog, tau=a.tau)
    r = sel.select(a.query, a.k)
    if a.json:
        print(json.dumps(r, indent=1))
        return
    if not r["ranked"]:
        print(f"no matching tool  (P(none) = {r['p_null']:.2f})")
    else:
        print(f"services: {', '.join(f'{s} {p:.2f}' for s, p in r['services'])}   P(none) = {r['p_null']:.2f}")
        for n, p in r["ranked"]:
            print(f"  {p:.3f}  {n}")
    print(f"latency {r['ms']['total']} ms (service step {r['ms']['service']} ms, tool step {r['ms'].get('tools', 0)} ms), cost ${r['cost_usd']:.5f}")


def cmd_bench(a) -> None:
    from retrieval_queries import NEG, Q

    sel = DMSelect(extra=not a.main_catalog, tau=a.tau)
    tasks = {p.name: json.loads((p / "task.json").read_text()) for p in (HERE / "tasks_real").iterdir()}
    items = [(q, set(tasks[t]["needs"][s])) for (t, s), pair in Q.items() for q in pair] + [(q, None) for q in NEG]
    with ThreadPoolExecutor(a.workers) as ex:
        res = list(ex.map(lambda it: sel.select(it[0], 5), items))
    pos = [(g, r) for (q, g), r in zip(items, res) if g]
    neg = [r for (q, g), r in zip(items, res) if not g]
    names = lambda r: [n for n, _ in r["ranked"]]  # noqa: E731
    h1 = sum(bool(names(r)) and names(r)[0] in g for g, r in pos) / len(pos)
    h5 = sum(bool(g & set(names(r))) for g, r in pos) / len(pos)
    lat = sorted(r["ms"]["total"] for r in res)
    print(f"{len(pos)} positive / {len(neg)} negative queries, {len(sel.tools)} tools, tau {sel.tau}, {a.workers} parallel")
    print(f"hit@1 {h1:.3f}  hit@5 {h5:.3f}  false 'none' {sum(not r['ranked'] for _, r in pos) / len(pos):.3f}  "
          f"correct 'none' {sum(not r['ranked'] for r in neg) / len(neg):.3f}  tools/answer {st.mean(len(r['ranked']) for r in res):.2f}")
    print(f"latency p50 {lat[len(lat) // 2]} ms, p90 {lat[int(.9 * len(lat))]} ms, max {lat[-1]} ms; "
          f"cost ${1000 * st.mean(r['cost_usd'] for r in res):.3f} per 1k searches; errors {sum(bool(r['error']) for r in res)}")


def cmd_serve(a) -> None:
    import uvicorn
    from starlette.applications import Starlette
    from starlette.requests import Request
    from starlette.responses import JSONResponse
    from starlette.routing import Route

    sel = DMSelect(extra=not a.main_catalog, tau=a.tau)

    async def rank(request: Request):
        import asyncio

        b = await request.json()
        r = await asyncio.to_thread(sel.select, str(b.get("query", "")), int(b.get("limit") or 5))
        return JSONResponse({**r, "ranked": [n for n, _ in r["ranked"]]})

    uvicorn.run(Starlette(routes=[Route("/rank", rank, methods=["POST"])]), host="127.0.0.1", port=a.port, log_level="warning")


def main() -> None:
    ap = argparse.ArgumentParser(prog="dmselect", description=__doc__.split("\n\n")[0])
    ap.add_argument("--main-catalog", action="store_true", help="the 613-tool catalog instead of 1,076")
    ap.add_argument("--tau", type=float, default=0.4, help="answer 'none' when P(none) >= tau")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("search")
    s.add_argument("query")
    s.add_argument("--k", type=int, default=5)
    s.add_argument("--json", action="store_true")
    b = sub.add_parser("bench")
    b.add_argument("--workers", type=int, default=4)
    v = sub.add_parser("serve")
    v.add_argument("--port", type=int, default=18800)
    a = ap.parse_args()
    {"search": cmd_search, "bench": cmd_bench, "serve": cmd_serve}[a.cmd](a)


if __name__ == "__main__":
    main()
