"""Summary of a small paired batch: per condition pass, tokens, searches, LLM calls, cost; per-task token ratios vs BM25."""
import json, math, sys, statistics as st, collections
R = [json.loads(l) for l in open(sys.argv[1] + "/rows.jsonl")]
C = ["hermes_default", "hermes_dmfrankenbridge", "hermes_dmctxbridge"]; lab = {"hermes_default": "BM25", "hermes_dmfrankenbridge": "F", "hermes_dmctxbridge": "one call, context prompt"}
by = {(r["task"], r["rep"], r["condition"]): r for r in R if not r.get("error")}
print(f"{'condition':26s} runs pass  tokens(mean) median  searches llm_calls  $/run")
for c in C:
    x = [r for k, r in by.items() if k[2] == c]
    if not x: continue
    print(f"{lab[c]:26s} {len(x):4d} {sum(r['passed'] for r in x):4d} {st.mean(r['total_tokens'] for r in x):12,.0f} {st.median(r['total_tokens'] for r in x):9,.0f} "
          f"{st.mean(r['n_searches'] for r in x):7.1f} {st.mean(r['api_calls'] for r in x):9.1f} {st.mean(r['cost'] for r in x):8.4f}")
tasks = sorted({k[0] for k in by})
print("\nper-task geometric-mean token ratio vs BM25 (same slots)")
print(f"{'task':20s} {'F':>6s} {'one call':>9s}")
allr = collections.defaultdict(list)
for t in tasks:
    row = []
    for c in C[1:]:
        L = [math.log(by[(t, rp, c)]["total_tokens"] / by[(t, rp, C[0])]["total_tokens"]) for rp in range(9)
             if (t, rp, c) in by and (t, rp, C[0]) in by and by[(t, rp, C[0])]["total_tokens"]]
        row.append(math.exp(st.mean(L)) if L else float("nan")); allr[c].append(st.mean(L) if L else None)
    print(f"{t:20s} " + " ".join(f"{v:8.2f}x" for v in row))
for c in C[1:]:
    L = [v for v in allr[c] if v is not None]
    print(f"overall {lab[c]}: {math.exp(st.mean(L)):.3f}x over {len(L)} tasks")
x = {k: r for k, r in by.items() if k[2] in C[1:]}
L = [math.log(by[(t, rp, C[2])]["total_tokens"] / by[(t, rp, C[1])]["total_tokens"]) for (t, rp, c) in by if c == C[2] and (t, rp, C[1]) in by]
if L: print(f"one call vs F, per slot geometric mean: {math.exp(st.mean(L)):.3f}x over {len(L)} slots")
