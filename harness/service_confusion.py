"""Offline: how often, and how confidently, does Microsoft Decision 1 pick the wrong SERVICE?
Searches from results/final_rec510 (BM25 runs: labels not influenced by the DM; DM runs: reported separately).
Label: the service of the first NEW tool (not called or shown earlier in the run) the agent successfully used
after the search, before its next search.
    source .or_env && .venv/bin/python harness/service_confusion.py
"""
import json, sys, collections, statistics as st
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
from oracle_review import events
from real_sim import load_catalog, SERVER_INFO
from selectors_real import Selectors

S = Selectors(load_catalog(extra=True)); srv = {t["name"]: t["server"] for t in S.tools}; servers = sorted(set(srv.values()))
items = []
for run in sorted((HERE.parent / "results" / "final_rec510" / "runs").iterdir()):
    task, cond, _, rep = run.name.split("__")
    if task == "k8s_crashloop": continue
    ev = events(run, cond); seen = set()
    for i, e in enumerate(ev):
        if e["kind"] == "call":
            seen.add(e["tool"]); continue
        lab = None  # label: first NEW tool (not called or shown before) the agent used after this search
        for x in ev[i + 1:]:
            if x["kind"] == "search": break
            if x["kind"] == "call" and x.get("ok") and x["tool"] in srv and x["tool"] not in seen:
                lab = srv[x["tool"]]; break
        seen |= set(e.get("ranked") or [])
        if lab: items.append({"cond": cond, "task": task, "query": e["query"], "label": lab})
qs = sorted({it["query"] for it in items})
print(f"{len(items)} labelled searches, {len(qs)} distinct queries", flush=True)
cache_f = HERE.parent / "results" / "final_rec510" / "service_probs.jsonl"
cache = {json.loads(l)["q"]: json.loads(l)["p"] for l in open(cache_f)} if cache_f.exists() else {}
def ask(q):
    p, _ = S._dm_ask(q, {x: SERVER_INFO.get(x, x) for x in servers}, "Which service has the tool the agent needs?")
    with open(cache_f, "a") as f: f.write(json.dumps({"q": q, "p": p}) + "\n")
    return q, p
with ThreadPoolExecutor(8) as ex:
    for q, p in ex.map(ask, [q for q in qs if q not in cache]): cache[q] = p
for cond, lab in (("hermes_default", "BM25 runs (clean labels)"), ("hermes_dmrec510bridge", "DM runs (labels influenced by the DM)")):
    R = [it for it in items if it["cond"] == cond and cache.get(it["query"])]
    top = lambda p: max(p, key=p.get)
    wrong = [it for it in R if top(cache[it["query"]]) != it["label"]]
    conf = lambda it: cache[it["query"]][top(cache[it["query"]])]
    def hedge(it, mass=0.9, k=2):
        p = cache[it["query"]]; acc, ch = 0, []
        for s in sorted(p, key=p.get, reverse=True):
            ch.append(s); acc += p[s]
            if acc >= mass or len(ch) >= k: break
        return it["label"] in ch
    print(f"\n== {lab}: {len(R)} searches")
    print(f"top service wrong: {len(wrong)} ({len(wrong)/len(R):.1%}); of those, DM confidence >= 0.7: {sum(conf(w)>=0.7 for w in wrong)}, >= 0.9: {sum(conf(w)>=0.9 for w in wrong)}")
    print(f"right service among top-2 (until 90% mass): {sum(hedge(it) for it in R)/len(R):.1%} (top-1 only: {1-len(wrong)/len(R):.1%})")
    print(f"P(right service) when top was wrong: median {st.median(cache[w['query']].get(w['label'],0) for w in wrong):.2f}" if wrong else "")
    pairs = collections.Counter((top(cache[w["query"]]), w["label"]) for w in wrong)
    print("most common confusions (picked -> agent actually used):", [(f"{a}->{b}", n) for (a, b), n in pairs.most_common(8)])
    cw = sorted(wrong, key=conf, reverse=True)
    seen = set(); print("most confidently wrong examples:")
    for w in cw:
        if w["query"] in seen: continue
        seen.add(w["query"]); p = cache[w["query"]]; t = top(p)
        print(f"  '{w['query'][:62]}' -> {t} {p[t]:.2f} | used {w['label']} {p.get(w['label'],0):.2f}")
        if len(seen) >= 10: break
