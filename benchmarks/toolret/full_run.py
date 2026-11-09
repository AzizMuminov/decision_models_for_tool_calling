"""Tool-REX full run (ANALYSIS_PLAN.md): methods A and E on all 7,961 queries, with instruction."""
import sys, json, random, collections, time
from concurrent.futures import ThreadPoolExecutor
sys.argv = sys.argv[:2] + ["100000"]
import os, pickle, numpy as np
import improve as I
R = I.R
NOINST = os.environ.get("NOINST") == "1"
if NOINST:
    R.TOP = pickle.load(open(R.HERE / "data" / "bm25_top100.pkl", "rb"))[(False, "all")]
    qtext = lambda q: q["query"] if (q["query"] or "").strip() else "NONE"
    R.ctx = lambda q: {"what_the_user_wants": qtext(q)}
    QE = I.embed([qtext(q) for q in R.Q])
    I.EMB = {q["id"]: [I.ids[j] for j in np.argsort(-(I.E @ v))[:100]] for q, v in zip(R.Q, QE)}
METHODS = os.environ.get("METHODS", "A,E").split(",")
TAG = "noinst" if NOINST else "inst"
P5 = "A retrieval system is ranking tools for a user's task. Which tool is most relevant to it?"
order = lambda cand, s: sorted(cand, key=lambda t: (-s.get(t, 0), cand.index(t)))
FAIL = collections.Counter()
def ask(q, opts, tag):
    p, c = R.ask(R.ctx(q), P5, opts)
    if not p: FAIL[tag] += 1
    return p, c
def A(q):
    cand = I.hybrid(q); p, c = ask(q, {t: R.card(t) for t in cand}, "A"); return order(cand, p), c
full = lambda t: " ".join(str(R.tools[t]).split())[:1500]
def E(q):
    cand = I.hybrid(q); parts = [cand[i:i + 25] for i in range(0, len(cand), 25)]
    with ThreadPoolExecutor(len(parts)) as ex: res = list(ex.map(lambda h: ask(q, {t: full(t) for t in h}, "E"), parts))
    first, fin, cost = {}, [], 0.0
    for h, (p, c) in zip(parts, res): first.update(p); fin += order(h, p)[:6]; cost += c
    p, c = ask(q, {t: full(t) for t in fin}, "E"); cost += c
    return order(fin, p) + order([t for t in cand if t not in fin], first), cost
# pilot ids: same stratified sample as dm_rerank.py (seed 7, 11 per task)
rnd = random.Random(7); by = collections.defaultdict(list)
for q in R.Q: by[q["task"]].append(dict(q))
per = max(1, 400 // len(by)); PILOT = set()
for t, L in sorted(by.items()): rnd.shuffle(L); PILOT |= {x["id"] for x in L[:per]}
def run(name, f):
    out = {}; t0 = time.time()
    with ThreadPoolExecutor(8) as ex:
        for i, (q, r) in enumerate(zip(R.Q, ex.map(f, R.Q)), 1):
            out[q["id"]] = r
            if i % 1000 == 0: print(f"  {name}: {i}/{len(R.Q)} queries, ${R.SP['usd']:.2f} spent, {time.time()-t0:.0f}s", flush=True)
    return out
def report(name, out, ids):
    cat = collections.defaultdict(lambda: collections.defaultdict(list))
    for q in R.Q:
        if q["id"] not in ids: continue
        rel = {str(x["id"]): int(x["relevance"]) for x in json.loads(q["labels"])}; rk = out[q["id"]][0]
        top = set(rk[:10]); m = cat[q["cat"]][q["task"]]
        m.append((R.ndcg10(rk, rel), len(top & set(rel)) / max(1, len(rel)), float(set(rel) <= top)))
    res = {}
    for c, tasks in cat.items():
        res[c] = [100 * sum(sum(v[i] for v in L) / len(L) for L in tasks.values()) / len(tasks) for i in range(3)]
    avg = [sum(res[c][i] for c in res) / 3 for i in range(3)]
    print(f"{name:34s} " + "  ".join(f"{c} {res[c][0]:5.2f}" for c in ("web", "code", "customized")) + f" | avg N@10 {avg[0]:5.2f} R@10 {avg[1]:5.2f} C@10 {avg[2]:5.2f}", flush=True)
    return {"by_category": res, "average": avg}
if __name__ == "__main__":
    print(len(R.Q), "queries;", len(PILOT), "pilot queries;", "without" if NOINST else "with", "instruction; methods", METHODS, flush=True)
    outs = {"BM25": {q["id"]: (R.TOP[q["id"]], 0) for q in R.Q}, "embeddings": {q["id"]: (I.EMB[q["id"]], 0) for q in R.Q}}
    for m in METHODS: outs[m] = run(m, {"A": A, "E": E}[m])
    ALL = {q["id"] for q in R.Q}; HELD = ALL - PILOT; summary = {}
    for label, ids in (("all 7,961 queries", ALL), (f"{len(HELD)} queries not in the pilot", HELD)):
        print(f"\n== {label}"); summary[label] = {n: report(n, o, ids) for n, o in outs.items()}
    print(f"\nfailed DM calls kept first-stage order: {dict(FAIL)}; DM spend ${R.SP['usd']:.2f} over {R.SP['n']} calls")
    json.dump({"summary": summary, "failed": dict(FAIL), "spend": R.SP}, open(R.HERE / "data" / f"full_run_summary_{TAG}.json", "w"), indent=1)
    json.dump({n: {k: v[0][:100] for k, v in o.items()} for n, o in outs.items() if n in METHODS}, open(R.HERE / "data" / f"full_run_rankings_{TAG}.json", "w"))
