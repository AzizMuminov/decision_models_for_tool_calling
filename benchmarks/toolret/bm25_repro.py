"""Reproduce ToolRet's BM25 (bm25s, English stopwords) over the full tool corpus; NDCG@10 and Recall@10 as in
toolret/eval.py (pytrec_eval ndcg_cut: gain = relevance, log2 discount). Saves BM25 top-100 per query for re-ranking."""
import sys, json, math, collections, pickle
from pathlib import Path
import pyarrow.parquet as pq, bm25s
D = Path(sys.argv[1]); OUT = Path(__file__).parent / "data"; OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(D.parent / "toolret_src")); from toolret.config import _TASK_2_CATEGORY as T2C
tools = []
for c in ("code", "customized", "web"): tools += [dict(t, cat=c) for t in pq.read_table(D / f"tools_{c}.parquet").to_pylist()]
ids = [t["id"] for t in tools]; docs = [str(t["documentation"]) for t in tools]
print(len(tools), "tools", flush=True)
def ndcg10(ranked, rel):
    dcg = sum(rel.get(d, 0) / math.log2(i + 2) for i, d in enumerate(ranked[:10]))
    ideal = sorted(rel.values(), reverse=True)[:10]; idcg = sum(r / math.log2(i + 2) for i, r in enumerate(ideal))
    return dcg / idcg if idcg else 0.0
def run(corpus_scope):
    res = {}; cache = {}
    for inst in (True, False):
        per_task = {}
        for qf in sorted((D / "q").glob("*.parquet")):
            task = qf.stem; cat = T2C[task]
            key = cat if corpus_scope == "category" else "all"
            if key not in cache:
                sel = [i for i, t in enumerate(tools) if corpus_scope != "category" or t["cat"] == cat]
                r = bm25s.BM25(); r.index(bm25s.tokenize([docs[i] for i in sel], stopwords="en", show_progress=False), show_progress=False); cache[key] = (r, [ids[i] for i in sel])
            r, cid = cache[key]
            Q = pq.read_table(qf).to_pylist(); nd, rc = [], []
            for q in Q:
                text = (q["instruction"] + " " + q["query"]) if inst else q["query"]
                qt = bm25s.tokenize(text, show_progress=False)
                if len(qt.vocab) == 0: qt = bm25s.tokenize("NONE", stopwords=[], show_progress=False)
                hits, _ = r.retrieve(qt, k=100, show_progress=False); ranked = [cid[h] for h in hits[0]]
                rel = {str(x["id"]): int(x["relevance"]) for x in json.loads(q["labels"])}
                nd.append(ndcg10(ranked, rel)); rc.append(len(set(ranked[:10]) & set(rel)) / len(rel) if rel else 0)
                res.setdefault((inst, corpus_scope), {})[q["id"]] = ranked
            per_task[task] = (cat, len(Q), sum(nd) / len(nd), sum(rc) / len(rc))
        n = sum(v[1] for v in per_task.values())
        qw = sum(v[1] * v[2] for v in per_task.values()) / n
        catm = collections.defaultdict(list)
        for v in per_task.values(): catm[v[0]].append(v)
        cm = {c: sum(x[1] * x[2] for x in L) / sum(x[1] for x in L) for c, L in catm.items()}
        tm = sum(v[2] for v in per_task.values()) / len(per_task)
        print(f"corpus={corpus_scope:8s} inst={'w/ ' if inst else 'w/o'}  queries={n}  NDCG@10: query-weighted {qw*100:.1f} | mean of categories {sum(cm.values())/3*100:.1f} {({c: round(v*100,1) for c, v in cm.items()})} | mean of tasks {tm*100:.1f}", flush=True)
    return res
allres = {}
for scope in ("all", "category"): allres.update(run(scope))
pickle.dump(allres, open(OUT / "bm25_top100.pkl", "wb"))
print("leaderboard BM25: w/ inst 36.0, w/o inst 21.8 (NDCG@10)")
