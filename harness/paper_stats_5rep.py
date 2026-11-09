"""Statistics for the paper's agent-run table: GPT-6.1 Sol (5 repeats) and DeepSeek V4.1 Flash (the pre-specified random
5 of 15 repeats per task, seed 20261108). Same methods as the plans: per-task mean log ratio, t-test over 37 tasks,
task bootstrap 95% CI (10,000 resamples, seed 1)."""
import json, math, random, statistics as st, sys, glob
from pathlib import Path
RES = Path(__file__).resolve().parents[1] / "results"
A, B = "hermes_default", "hermes_dmfrankenbridge"
def load(d):
    S = {}
    for l in open(RES / d / "rows.jsonl"):
        r = json.loads(l)
        if r["task"] == "k8s_crashloop" or r.get("error"): continue
        S.setdefault((r["task"], r["rep"]), {})[r["condition"]] = r
    return {k: v for k, v in S.items() if A in v and B in v}
def subsample(S):
    rng = random.Random(20261108); keep = {}
    for t in sorted({k[0] for k in S}):
        for rp in rng.sample(range(15), 5):
            if (t, rp) in S: keep[(t, rp)] = S[(t, rp)]
    return keep
def tstat(tm):
    n = len(tm); m = st.mean(tm); se = st.stdev(tm) / math.sqrt(n); t = m / se
    from math import erf
    # two-sided p from t with n-1 df (scipy-free: use incomplete beta via mpmath-like series) -> use scipy if available
    try:
        from scipy import stats; p = 2 * stats.t.sf(abs(t), n - 1)
    except Exception: p = float("nan")
    return m, p
def ratio(S, key):
    per = {}
    for (t, _), v in S.items():
        a, b = v[A][key], v[B][key]
        if a and b: per.setdefault(t, []).append(math.log(b / a))
    tm = [st.mean(v) for v in per.values()]; m, p = tstat(tm)
    rng = random.Random(1); n = len(tm)
    bs = sorted(sum(tm[rng.randrange(n)] for _ in range(n)) / n for _ in range(10000))
    return math.exp(m), math.exp(bs[249]), math.exp(bs[9749]), p
def diff(S, key):
    per = {}
    for (t, _), v in S.items(): per.setdefault(t, []).append(v[B][key] - v[A][key])
    tm = [st.mean(v) for v in per.values()]; m, p = tstat(tm)
    rng = random.Random(1); n = len(tm); bs = sorted(sum(tm[rng.randrange(n)] for _ in range(n)) / n for _ in range(10000))
    return m, bs[249], bs[9749], p
def mm(S, c, key, scale=1): x = [v[c][key] * scale for v in S.values()]; return st.mean(x), st.median(x)
def report(name, S):
    print(f"== {name}: {len(S)} slots, {len({k[0] for k in S})} tasks")
    pa = sum(v[A]["passed"] for v in S.values()); pb = sum(v[B]["passed"] for v in S.values())
    da = sum(v[A]["passed"] and not v[B]["passed"] for v in S.values()); db = sum(v[B]["passed"] and not v[A]["passed"] for v in S.values())
    from math import comb
    k, n = min(da, db), da + db; pm = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0
    print(f"  passed {pa} vs {pb}; discordant BM25-only {da}, F-only {db}; McNemar p={pm:.3f}")
    for key, lab, sc in (("total_tokens", "tokens (k)", 1e-3), ("api_calls", "LLM calls", 1), ("cost", "cost $", 1), ("wall_s", "wall s", 1)):
        a, b = mm(S, A, key, sc), mm(S, B, key, sc); r = ratio(S, key)
        print(f"  {lab:11s} BM25 {a[0]:.4g} / {a[1]:.4g} | F {b[0]:.4g} / {b[1]:.4g} | ratio {r[0]:.3f} [{r[1]:.2f}, {r[2]:.2f}] p={r[3]:.4f}")
    a, b = mm(S, A, "n_searches"), mm(S, B, "n_searches"); d = diff(S, "n_searches")
    print(f"  searches    BM25 {a[0]:.1f} / {a[1]:.0f} | F {b[0]:.1f} / {b[1]:.0f} | diff {d[0]:.1f} [{d[1]:.1f}, {d[2]:.1f}] p={d[3]:.2g}")
    lr = [math.log(v[B]["total_tokens"] / v[A]["total_tokens"]) for v in S.values()]
    print(f"  SD of per-slot log ratio {st.stdev(lr):.2f}; fewer tokens on {sum(x < 0 for x in lr)} of {len(lr)} slots; range {math.exp(min(lr)):.2f}-{math.exp(max(lr)):.2f}")
    for lab, cat in (("multi", "multi"), ("single", "single")):
        sub = {k: v for k, v in S.items() if str(v[A].get("category", "")).startswith(cat)}
        if sub: print(f"  {lab}-service tasks: {len({k[0] for k in sub})} tasks, tokens ratio {ratio(sub, 'total_tokens')[0]:.2f}")
    for t in ("gh_open_pr", "prom_threshold", "loki_errors"):
        x = [v for k, v in S.items() if k[0] == t]
        print(f"  {t}: BM25 passed {sum(v[A]['passed'] for v in x)}/{len(x)}, F {sum(v[B]['passed'] for v in x)}/{len(x)}")
    fails = {c: sum(not v[c]["passed"] for v in S.values()) for c in (A, B)}; print("  failures", fails)
    # token mix for cost projection
    for c in (A, B):
        print(f"  mix {c}: uncached in {st.mean(v[c]['input'] for v in S.values())/1e3:.1f}k cached {st.mean(v[c]['cache_read'] for v in S.values())/1e3:.1f}k out {st.mean(v[c]['output'] for v in S.values())/1e3:.1f}k")
sol = load("final_franken_sol"); ds = subsample(load("final_franken"))
report("GPT-6.1 Sol, 5 repeats", sol); report("DeepSeek V4.1 Flash, 5 repeats (seed 20261108)", ds)
json.dump(sorted(map(list, ds)), open(RES / "final_franken_sol" / "deepseek_subsample_slots.json", "w"))
