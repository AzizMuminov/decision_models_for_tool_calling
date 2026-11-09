"""DM re-ranking of BM25's top 100 (full corpus, with instruction) on ToolRet; NDCG@10 / Recall@10 per category, mean of
categories as on the leaderboard. Pilot: N queries stratified over tasks (seed 7). DM calls cached."""
import sys, json, math, random, pickle, hashlib, threading, time, collections
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pyarrow.parquet as pq
HERE = Path(__file__).parent; D = Path(sys.argv[1]); N = int(sys.argv[2]) if len(sys.argv) > 2 else 400
sys.path[:0] = [str(D.parent / "toolret_src"), str(HERE.parent)]
from toolret.config import _TASK_2_CATEGORY as T2C
from dmtc.backends.dm import Choice, OpenRouterDM
tools = {}
for c in ("code", "customized", "web"):
    for t in pq.read_table(D / f"tools_{c}.parquet").to_pylist(): tools[t["id"]] = t["documentation"]
def card(tid):
    """Tool-REX card: name, description, and the expanded profile (function summary, tags), about 350 characters."""
    try: d = json.loads(tools[tid])
    except Exception: return " ".join(str(tools[tid]).split())[:350]
    if not isinstance(d, dict): return " ".join(str(d).split())[:350]
    prof = d.get("tool_profile") or {}
    if not isinstance(prof, dict): prof = {"function": str(prof)}
    parts = [str(d.get("name") or d.get("api_call") or ""), str(d.get("description") or d.get("functionality") or ""),
             str(prof.get("function") or ""), "tags: " + ", ".join(map(str, prof.get("tags") or []))]
    return " | ".join(" ".join(x.split()) for x in parts if x.strip())[:350]
def card_old(tid):
    s = " ".join(str(tools[tid]).split())
    return s[:300]
TOP = pickle.load(open(HERE / "data" / "bm25_top100.pkl", "rb"))[(True, "all")]
Q = []
for qf in sorted((D / "q").glob("*.parquet")):
    for q in pq.read_table(qf).to_pylist(): Q.append(dict(q, task=qf.stem, cat=T2C[qf.stem]))
rnd = random.Random(7); by = collections.defaultdict(list)
for q in Q: by[q["task"]].append(q)
if N < len(Q):
    per = max(1, N // len(by)); S = []
    for t, L in sorted(by.items()): rnd.shuffle(L); S += L[:per]
    Q = S
DM = OpenRouterDM("microsoft/microsoft-decision-1"); CF = HERE / "data" / "dm_cache.jsonl"; CACHE = {}; LOCK = threading.Lock(); SP = {"usd": 0.0, "n": 0}
if CF.exists():
    for l in open(CF): x = json.loads(l); CACHE[x["k"]] = x["p"], x["c"]
def ask(ctx, instr, opts):
    k = hashlib.sha1(json.dumps([ctx, instr, opts], sort_keys=True).encode()).hexdigest()
    if k in CACHE: return CACHE[k]
    for a in range(5):
        r = DM.ask(ctx, {"x": Choice(instr, opts)})
        if not r.error and r.answers.get("x"): break
        time.sleep(2 ** a)
    p = r.answers["x"].probabilities if r.answers.get("x") else {}
    with LOCK:
        SP["usd"] += r.cost_usd; SP["n"] += 1
        if p: CACHE[k] = (p, r.cost_usd); open(CF, "a").write(json.dumps({"k": k, "p": p, "c": r.cost_usd}) + "\n")
    return p, r.cost_usd
BEST, EXPERT = "Which tool best serves what the user wants to do?", "Which tool would an expert engineer call first to do what the user wants?"
def ctx(q): return {"task": q["instruction"], "what_the_user_wants": q["query"]}
def dm_rank(q, instr):
    cand = TOP[q["id"]]; p, c = ask(ctx(q), instr, {t: card(t) for t in cand})
    return sorted(cand, key=lambda t: (-p.get(t, 0), cand.index(t))), c
def ndcg10(ranked, rel):
    dcg = sum(rel.get(d, 0) / math.log2(i + 2) for i, d in enumerate(ranked[:10]))
    ideal = sorted(rel.values(), reverse=True)[:10]; idcg = sum(r / math.log2(i + 2) for i, r in enumerate(ideal))
    return dcg / idcg if idcg else 0.0
def evaluate(name, f):
    with ThreadPoolExecutor(8) as ex: R = list(ex.map(f, Q))
    cats = collections.defaultdict(lambda: [[], []])
    for q, (rk, _) in zip(Q, R):
        rel = {str(x["id"]): int(x["relevance"]) for x in json.loads(q["labels"])}
        cats[q["cat"]][0].append(ndcg10(rk, rel)); cats[q["cat"]][1].append(len(set(rk[:10]) & set(rel)) / max(1, len(rel)))
    nd = {c: 100 * sum(v[0]) / len(v[0]) for c, v in cats.items()}; rc = {c: 100 * sum(v[1]) / len(v[1]) for c, v in cats.items()}
    cost = 1000 * sum(r[1] for r in R) / len(R)
    print(f"{name:34s} NDCG@10 {sum(nd.values())/len(nd):5.1f} (code {nd['code']:.1f}, web {nd['web']:.1f}, customized {nd['customized']:.1f})  Recall@10 {sum(rc.values())/len(rc):5.1f}  ${cost:.2f}/1k", flush=True)
def f_style(q):
    a, c1 = dm_rank(q, BEST); b, c2 = dm_rank(q, EXPERT)
    out = a[:5] + [t for t in b if t not in a[:5]][:3]; out += [t for t in a if t not in out]
    return out, c1 + c2
if __name__ == "__main__":
  print(len(Q), "queries", flush=True)
  evaluate("BM25 (ours, full corpus)", lambda q: (TOP[q["id"]], 0.0))
  evaluate("DM re-rank, best serves", lambda q: dm_rank(q, BEST))
  pass
  print(f"new DM spend ${SP['usd']:.2f} over {SP['n']} calls")
