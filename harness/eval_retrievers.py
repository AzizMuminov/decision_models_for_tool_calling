"""Offline comparison of tool retrievers on the find_tool queries agents actually wrote in the harness runs.

Every find_tool call logged in results/harness_v2 (100 tools) and results/harness_v2_300 (300 tools) is
replayed against each retriever. The target of a query is the needed-tool group that the agent's next
catalog call fell in (e.g. "a job-log tool" for the first step of m_crypto); when the next call was not a
needed tool, any needed group counts (marked `lenient`). Queries from tasks that need no catalog tool
(control, absent) are left out.

Regex queries: the agents wrote plain-language intents, not regexes. For the regex retriever an LLM
(deepseek-v4.1-flash, the cheap agent-class model) is shown the regex tool's description and rewrites each
intent into the pattern an agent would pass. That is what an agent using regex search has to do itself.

All API results are cached under results/retrieval/, so re-running costs nothing.

    source .or_env && .venv/bin/python harness/eval_retrievers.py
"""

from __future__ import annotations

import json
import os
import random
import re
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))
from catalog import EXTRA_SERVICES, SERVICES, catalog  # noqa: E402
from retrievers import BM25, DMRanker, llm_rank, regex_rank, short_card  # noqa: E402

OUT = REPO / "results" / "retrieval"
OUT.mkdir(parents=True, exist_ok=True)
K = 3
STOP_AT = float(os.environ.get("STOP_AT_USAGE", "99.2"))
GEN = "deepseek/deepseek-v4.1-flash"
ALL_SERVICES = {**SERVICES, **EXTRA_SERVICES}


def usage() -> float:
    r = httpx.get("https://openrouter.ai/api/v1/key", headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"}, timeout=30)
    return float(r.json()["data"]["usage"])


class Cache:
    def __init__(self, name):
        self.path = OUT / f"{name}.jsonl"
        self.d = {}
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                r = json.loads(line)
                self.d[r["key"]] = r

    def get(self, key):
        return self.d.get(key)

    def put(self, key, **v):
        r = {"key": key, **v}
        self.d[key] = r
        with open(self.path, "a") as f:
            f.write(json.dumps(r) + "\n")


def load_queries() -> list[dict]:
    tasks = {p.name: json.loads((p / "task.json").read_text()) for p in (HERE / "tasks_v2").iterdir() if (p / "task.json").exists()}
    qs = []
    for size, root in ((100, REPO / "results/harness_v2/runs"), (300, REPO / "results/harness_v2_300/runs")):
        for run in sorted(root.iterdir()):
            task, cond, model, rep = run.name.split("__")
            if not cond.startswith("universal") or not tasks[task]["needs"]:
                continue
            log = [json.loads(x) for x in (run / "catalog_calls.jsonl").read_text().splitlines()] if (run / "catalog_calls.jsonl").exists() else []
            groups = [set(g) for g in tasks[task]["needs"]]
            for i, e in enumerate(log):
                if e["kind"] != "find_tool":
                    continue
                nxt = next((x["tool"] for x in log[i + 1:] if x["kind"] == "call"), None)
                tgt = [g for g in groups if nxt in g]
                qs.append({"size": size, "task": task, "category": tasks[task]["category"], "cond": cond, "model": model,
                           "rep": rep, "intent": e["intent"], "target": sorted(tgt[0]) if tgt else sorted(set().union(*groups)),
                           "lenient": not tgt, "logged_dm": e["ranked"], "logged_dm_cost": e.get("dm_cost", 0)})
    return qs


REGEX_DESC = ("Search Quillmark's internal tool catalog with a Python regular expression (re.search syntax, max 200 "
              "characters). It is matched against each tool's name, description and parameter names; use (?i) for "
              "case-insensitive matching. Returns the matching tools with their schemas.")


def regex_patterns(items: list[tuple[str, str]], cache: Cache) -> None:
    """items: (key, prompt-context). Writes one regex per intent, 20 intents per request."""
    todo = [(k, ctx) for k, ctx in items if not cache.get(k)]
    for i in range(0, len(todo), 20):
        if usage() > STOP_AT:
            print("budget stop (regex)")
            return
        batch = todo[i:i + 20]
        listing = "\n".join(f"{j + 1}. {json.loads(k)['intent']}" for j, (k, _) in enumerate(batch))
        sys_msg = ("You are a coding agent. Your only way to find catalog tools is this search tool:\n\n" + batch[0][1] +
                   "\n\nFor each need below, write the exact pattern you would pass to the tool. Reply with JSON only: "
                   '{"patterns": ["...", ...]} in the same order.')
        body = {"model": GEN, "temperature": 0, "max_tokens": 2000, "usage": {"include": True},
                "response_format": {"type": "json_object"},
                "messages": [{"role": "system", "content": sys_msg}, {"role": "user", "content": listing}]}
        r = httpx.post("https://openrouter.ai/api/v1/chat/completions", json=body, timeout=120,
                       headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"}).json()
        cost = float(r.get("usage", {}).get("cost") or 0)
        try:
            pats = json.loads(re.search(r"\{.*\}", r["choices"][0]["message"]["content"], re.S).group(0))["patterns"]
        except Exception:  # noqa: BLE001
            pats = []
        for j, (k, _) in enumerate(batch):
            if j < len(pats):
                cache.put(k, pattern=pats[j], cost=cost / len(batch))


def main() -> None:
    qs = load_queries()
    print(len(qs), "queries;", sum(q["lenient"] for q in qs), "lenient")
    cats = {s: catalog(s) for s in (100, 300)}
    bm = {s: BM25(cats[s]) for s in cats}
    bm_short = {s: BM25(cats[s], field=lambda t: f"{t['name'].replace('_', ' ')} {short_card(t)}") for s in cats}
    uniq = {}  # (size, index?, intent) -> representative
    for q in qs:
        uniq.setdefault((q["size"], q["cond"] == "universal_index", q["intent"]), q)
    print(len(uniq), "unique (size, index, intent)")

    # regex patterns: the generator sees the same service list the agent saw in universal_index runs
    rcache = Cache("regex_patterns")
    def ctx(size, idx):
        d = REGEX_DESC
        if idx:
            d += "\n\nThe catalog covers these internal services:\n" + "\n".join(
                f"  - {b}: {w}" for s, (b, w, _, _) in ALL_SERVICES.items() if any(t["service"] == s for t in cats[size]))
        return d
    regex_patterns([(json.dumps({"size": s, "idx": i, "intent": t}), ctx(s, i)) for (s, i, t) in uniq], rcache)

    # LLM router and the DM strategies for >250 tools: API calls, cached, run in parallel
    lcache, dcache = Cache("llm_router"), Cache("dm_strategies")
    rankers = {300: DMRanker(cats[300], ALL_SERVICES)}
    intents = {s: sorted({t for (sz, _, t) in uniq if sz == s}) for s in (100, 300)}

    def do_llm(size, intent):
        key = json.dumps({"size": size, "intent": intent})
        if not lcache.get(key):
            names, cost, err = llm_rank(cats[size], intent, K)
            lcache.put(key, ranked=names, cost=cost, error=err)

    def do_dm(strategy, intent):
        key = json.dumps({"size": 300, "strategy": strategy, "intent": intent})
        if not dcache.get(key):
            names, cost, err = rankers[300].rank(intent, K, strategy)
            dcache.put(key, ranked=names, cost=cost, error=err)

    jobs = [(do_llm, s, t) for s in (100, 300) for t in intents[s]] + \
           [(do_dm, st, t) for st in ("hierarchy", "bm25_dm") for t in intents[300]]
    random.Random(0).shuffle(jobs)
    for i in range(0, len(jobs), 60):
        if usage() > STOP_AT:
            print("budget stop at job", i, "of", len(jobs))
            break
        with ThreadPoolExecutor(8) as ex:
            list(ex.map(lambda j: j[0](j[1], j[2]), jobs[i:i + 60]))

    rows = []
    for q in qs:
        s, idx, t = q["size"], q["cond"] == "universal_index", q["intent"]
        res = {"dm": q["logged_dm"], "bm25": bm[s].rank(t, K), "bm25_short": bm_short[s].rank(t, K)}
        rp = rcache.get(json.dumps({"size": s, "idx": idx, "intent": t}))
        if rp:
            res["regex"] = regex_rank(cats[s], rp["pattern"], K)[0]
        lr = lcache.get(json.dumps({"size": s, "intent": t}))
        if lr and not lr["error"]:
            res["llm"] = lr["ranked"]
        for st in ("hierarchy", "bm25_dm"):
            dr = dcache.get(json.dumps({"size": 300, "strategy": st, "intent": t})) if s == 300 else None
            if dr and not dr["error"]:
                res["dm_" + st] = dr["ranked"]
        for name, ranked in res.items():
            rows.append({**{k: q[k] for k in ("size", "task", "category", "cond", "model", "lenient")}, "retriever": name,
                         "hit1": bool(ranked[:1]) and ranked[0] in q["target"], "hit3": any(r in q["target"] for r in ranked[:K]),
                         "empty": not ranked})
    (OUT / "rows.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    print("cost so far: regex", round(sum(r.get("cost", 0) for r in rcache.d.values()), 4), "llm",
          round(sum(r.get("cost", 0) for r in lcache.d.values()), 4), "dm", round(sum(r.get("cost", 0) for r in dcache.d.values()), 4))


if __name__ == "__main__":
    main()
