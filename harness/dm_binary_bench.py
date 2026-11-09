"""Offline: Microsoft Decision 1 in 'choice' mode (one question, up to 255 options; what the paper uses) vs 'noul' mode
(many yes/no questions in one call, one per option). Same server-then-tool hierarchy; 120 queries (+ negatives), 1,076 tools.
    source .or_env && .venv/bin/python harness/dm_binary_bench.py [--n 10]
"""
import json, sys, time, statistics as st
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
from real_sim import SERVER_INFO, load_catalog
from dmselect import DMSelect, card
from dmtc.backends.dm import Noul
from retrieval_queries import Q

class Binary(DMSelect):
    def _askb(self, query, options, instr):
        t0 = time.perf_counter()
        qs = {f"o{i}": Noul(f"{instr} Option: {desc}") for i, desc in enumerate(options.values())}
        r = self.dm.ask({"what_the_agent_wants": query}, qs)
        keys = list(options)
        p = {keys[int(k[1:])]: a.value for k, a in r.answers.items()}
        return p, r.cost_usd, (time.perf_counter() - t0) * 1000, r.error
    def select(self, query, k=5):
        t0 = time.perf_counter()
        ps, cost, _, err = self._askb(query, {s: SERVER_INFO.get(s, s) for s in self.servers}, "Does this service have the tool the agent needs?")
        top = sorted(ps, key=ps.get, reverse=True)[:3]
        top = [s for s in top if ps[s] >= 0.5 * ps[top[0]]] if top else []
        score = {}
        for s in top:
            names = [t["name"] for t in self.tools if t["server"] == s]
            for i in range(0, len(names), 150):
                pt, c, _, e = self._askb(query, {n: card(self.by[n]) for n in names[i:i + 150]}, "Is this the tool that does what the agent wants?")
                cost += c; err = err or e
                score.update({n: ps[s] * v for n, v in pt.items()})
        return {"ranked": [[n, score[n]] for n in sorted(score, key=score.get, reverse=True)[:k]], "cost_usd": cost,
                "ms": {"total": round((time.perf_counter() - t0) * 1000)}, "error": err}

n = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 10**9
tasks = {p.name: json.loads((p / "task.json").read_text()) for p in (HERE / "tasks_real").iterdir()}
items = [(q, set(tasks[t]["needs"][s])) for (t, s), pair in Q.items() for q in pair][:n]
for name, sel in [("choice (paper)", DMSelect(extra=True, tau=2.0)), ("yes/no", Binary(extra=True))]:
    with ThreadPoolExecutor(4) as ex:
        res = list(ex.map(lambda it: sel.select(it[0], 5), items))
    names = [[x for x, _ in r["ranked"]] for r in res]
    h1 = sum(bool(nm) and nm[0] in g for (q, g), nm in zip(items, names)) / len(items)
    h5 = sum(bool(g & set(nm)) for (q, g), nm in zip(items, names)) / len(items)
    lat = sorted(r["ms"]["total"] for r in res)
    print(f"{name:15s} n={len(items)} hit@1 {h1:.3f} hit@5 {h5:.3f}  ${1000*st.mean(r['cost_usd'] for r in res):.3f}/1k queries  "
          f"latency p50 {lat[len(lat)//2]} ms  errors {sum(bool(r['error']) for r in res)}", flush=True)
