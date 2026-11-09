"""Two fixes on the pilot. (1) Candidates: union of BM25 top 50 and embedding top 50 (text-embedding-3-small over the
expanded documents), interleaved. (2) Ranking: DM's first choice stays first; positions 2-20 re-ordered by independent
yes/no answers ('Does the user need this tool?') on the DM's top 20."""
import sys, json, os, time, collections, numpy as np, httpx
sys.argv = sys.argv[:3]
import dm_rerank as R
from dmtc.backends.dm import Noul
ids = list(R.tools); EF = R.HERE / "data" / "tool_emb.npy"; H = {"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"}
def embed(texts):
    out = []
    for i in range(0, len(texts), 256):
        for a in range(5):
            try: r = httpx.post("https://openrouter.ai/api/v1/embeddings", headers=H, timeout=120, json={"model": "openai/text-embedding-3-small", "input": texts[i:i + 256]}).json(); out += [d["embedding"] for d in r["data"]]; break
            except Exception: time.sleep(2 ** a)
        else: raise SystemExit("embedding failed")
    v = np.array(out, dtype=np.float32); return v / np.linalg.norm(v, axis=1, keepdims=True)
if not EF.exists(): np.save(EF, embed([" ".join(str(R.tools[t]).split())[:1500] for t in ids]))
E = np.load(EF); QE = embed([q["instruction"] + " " + q["query"] for q in R.Q])
EMB = {q["id"]: [ids[j] for j in np.argsort(-(E @ v))[:100]] for q, v in zip(R.Q, QE)}
def hybrid(q):
    a, b = R.TOP[q["id"]][:50], EMB[q["id"]][:50]; out = []
    for x, y in zip(a, b):
        for t in (x, y):
            if t not in out: out.append(t)
    return out[:100]
def dm_on(q, cand):
    p, c = R.ask(R.ctx(q), R.BEST, {t: R.card(t) for t in cand}); return sorted(cand, key=lambda t: (-p.get(t, 0), cand.index(t))), c
YN = "Does the user need this tool to do what they want? Tool: "
import hashlib
def yn(q, cand):
    k = hashlib.sha1(json.dumps(["yn", R.ctx(q), cand], sort_keys=True).encode()).hexdigest()
    if k in R.CACHE: return R.CACHE[k]
    for a in range(5):
        r = R.DM.ask(R.ctx(q), {f"t{i}": Noul(YN + R.card(t)) for i, t in enumerate(cand)})
        if not r.error: break
        time.sleep(2 ** a)
    p = {t: (r.answers[f"t{i}"].value if r.answers.get(f"t{i}") else 0.0) for i, t in enumerate(cand)}
    with R.LOCK:
        R.SP["usd"] += r.cost_usd; R.SP["n"] += 1
        if not r.error: R.CACHE[k] = (p, r.cost_usd); open(R.CF, "a").write(json.dumps({"k": k, "p": p, "c": r.cost_usd}) + "\n")
    return p, r.cost_usd
def with_yn(q, cand):
    a, c1 = dm_on(q, cand); p, c2 = yn(q, a[:20])
    return [a[0]] + sorted(a[1:20], key=lambda t: -p.get(t, 0)) + a[20:], c1 + c2
def rel_of(q): return {str(x["id"]): int(x["relevance"]) for x in json.loads(q["labels"])}
rec = lambda f: sum(100 * sum(v) / len(v) for v in (lambda d: [d[c] for c in d])(collections.defaultdict(list, {c: [len(set(f(q)) & set(rel_of(q))) / max(1, len(rel_of(q))) for q in R.Q if q["cat"] == c] for c in ("code", "web", "customized")}))) / 3
if __name__ == "__main__":
    print(f"{len(R.Q)} queries. Candidate recall (mean of categories): BM25 top 100 {rec(lambda q: R.TOP[q['id']]):.1f} | embeddings top 100 {rec(lambda q: EMB[q['id']]):.1f} | union of top 50s {rec(hybrid):.1f}", flush=True)
    R.evaluate("embeddings alone", lambda q: (EMB[q["id"]], 0.0))
    R.evaluate("DM on BM25 top 100 (current)", lambda q: dm_on(q, R.TOP[q["id"]]))
    R.evaluate("DM on embeddings top 100", lambda q: dm_on(q, EMB[q["id"]]))
    R.evaluate("DM on BM25+embeddings union", lambda q: dm_on(q, hybrid(q)))
    R.evaluate("DM + yes/no, BM25 top 100", lambda q: with_yn(q, R.TOP[q["id"]]))
    R.evaluate("DM + yes/no, union", lambda q: with_yn(q, hybrid(q)))
    print(f"new DM spend ${R.SP['usd']:.2f} over {R.SP['n']} calls")
