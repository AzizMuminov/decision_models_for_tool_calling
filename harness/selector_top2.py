"""No nucleus: always the top-k services (named service first), at least 10 tools; no usage names."""
import sys, statistics as st
from concurrent.futures import ThreadPoolExecutor
HELD = "--heldout" in sys.argv
import selector_study as S
src = open(S.HERE / "selector_pgrid_nopop.py").read().split("hand, agent, named = S.load_sets()")[0]
G = {}; exec(src, G)
def build(pre, k, n, order_mode, pt=None, cap=None):
    if pre is None: return [], 0
    order, ps, sc = pre; ch = order[:k]
    per = {s: sorted((t for t in sc if S.SRV[t] == s), key=sc.get, reverse=True) for s in ch}
    if order_mode == "weighted":
        tools = sorted((t for s in ch for t in per[s]), key=sc.get, reverse=True)
    else:  # alternating: best of each service in turn
        tools = [per[s][i] for i in range(max(len(v) for v in per.values())) for s in ch if i < len(per[s])]
    if pt is None: return tools[:n], k
    tot = sum(sc[t] for t in tools) or 1; out, acc = [], 0.0
    for t in tools:
        out.append(t); acc += sc[t] / tot
        if acc >= pt and len(out) >= n: break
    return out[:cap], k
hand, agent, named = S.load_sets()
if HELD:
    agent = S.load_heldout(); hand = [h for h in hand if h["gold"] is None]; S.TASK_OF.update({a["q"]: a["task"] for a in agent})
qs = sorted({x["q"] for x in hand + agent})
with ThreadPoolExecutor(8) as ex: PRE = dict(zip(qs, ex.map(G["prep"], qs)))
M = lambda xs: (lambda L: st.mean(L) if L else float("nan"))(list(xs)); H = [h for h in hand if h["gold"]]
V = [("ref: nucleus 0.99, top 10, weighted", None),
     ("top 2 services, 10 tools, weighted", (2, 10, "weighted")), ("top 2 services, 10 tools, alternating", (2, 10, "alt")),
     ("top 2 services, 15 tools, alternating", (2, 15, "alt")), ("top 3 services, 10 tools, alternating", (3, 10, "alt")),
     ("top 3 services, 15 tools, alternating", (3, 15, "alt")),
     ("top 2, tool p 0.995, min 10 max 20, weighted", (2, 10, "weighted", 0.995, 20))]
print(f"{'variant':46s} hit1  cov5  cov   all   len")
for name, a in V:
    r = {q: (G["build"](PRE[q], q, 0.99, "off", 10)[0] if a is None else build(PRE[q], *a)[0]) for q in qs}
    print(f"{name:46s} " + (f"{M(bool(r[h['q']]) and r[h['q']][0] in h['gold'] for h in H):.3f} " if H else "  -   ")
          + f"{M(len(x['gold'] & set(r[x['q']][:5])) / len(x['gold']) for x in agent):.3f} {M(len(x['gold'] & set(r[x['q']])) / len(x['gold']) for x in agent):.3f} "
          f"{M(x['gold'] <= set(r[x['q']]) for x in agent):.3f} {M(len(r[x['q']]) for x in agent):5.1f}", flush=True)
