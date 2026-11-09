"""Coverage-first variants. All: F's service step (cards, nucleus 0.9 max 3, named rule, none), 15 tools."""
import sys, statistics as st
from concurrent.futures import ThreadPoolExecutor
HELD = "--heldout" in sys.argv
import selector_study as S
def svc_probs(q):
    opts = {s: S.FIXED.get(s, s) for s in S.SERVERS}; opts[S.NONE] = "None of these services can do what the agent asked."
    ps, _ = S.ask_choice(q, S.Q_SVC, opts)
    if ps.get(S.NONE, 0) >= 0.4: return None
    ps = {s: v for s, v in ps.items() if s != S.NONE}; tot = sum(ps.values()) or 1; ps = {s: v / tot for s, v in ps.items()}
    order = sorted(ps, key=ps.get, reverse=True)
    if (n := S.named_service(q)) and n in ps: order = [n] + [s for s in order if s != n]; ps[n] = max(ps[n], 0.9)
    return order, ps
def nucleus(order, ps, mass=0.9, kmin=1, kmax=3):
    ch, acc = [], 0.0
    for s in order:
        ch.append(s); acc += ps[s]
        if (acc >= mass and len(ch) >= kmin) or len(ch) >= kmax: break
    return {s: ps[s] for s in ch}
def fill(*lists, cap=15):
    out = []
    for L in lists:
        for t in L:
            if t in S.BY and t not in out: out.append(t)
    return out[:cap]
def V(q, mode):
    sp = svc_probs(q)
    if sp is None: return []
    order, ps = sp
    ch = nucleus(order, ps, kmin=2 if mode == "2 services always" else 1)
    P = S._pop_for(q); pop = [t for s in ch for t in P.get(s, [])]
    best, _ = S.tools_choice(q, ch, S.Q_TOOL)
    if mode in ("current: 5 best + usage", "2 services always"): return fill(best[:5], pop, best[5:])
    if mode == "3 best + usage": return fill(best[:3], pop, best[3:])
    if mode == "usage interleaved with best": 
        out = []
        for i in range(15):
            for L in (best, pop):
                if i < len(L) and L[i] in S.BY and L[i] not in out: out.append(L[i])
        return out[:15]
    if mode == "5 best + expert + usage":
        exp, _ = S.tools_choice(q, ch, S.Q_TOOL_EXPERT); return fill(best[:5], exp[:3], pop, best[5:])
    if mode == "usage only (no tool call)": return fill(pop, [t for s in ch for t in S.NAMES_BY[s]])
MODES = ["current: 5 best + usage", "3 best + usage", "usage interleaved with best", "5 best + expert + usage", "2 services always", "usage only (no tool call)"]
def _run():
  pass
if __name__ == "__main__":
 hand, agent, named = S.load_sets()
 if HELD:
     agent = S.load_heldout(); hand = [h for h in hand if h["gold"] is None]; S.TASK_OF.update({a["q"]: a["task"] for a in agent})
 qs = sorted({x["q"] for x in hand + agent})
 M = lambda xs: (lambda L: st.mean(L) if L else float("nan"))(list(xs))
 print(f"{'variant':30s} cov5  cov10 cov15 all15")
 for m in MODES:
     with ThreadPoolExecutor(8) as ex: r = dict(zip(qs, ex.map(lambda q: V(q, m), qs)))
     print(f"{m:30s} " + " ".join(f"{M(len(a['gold'] & set(r[a['q']][:k])) / len(a['gold']) for a in agent):.3f}" for k in (5, 10, 15))
           + f" {M(a['gold'] <= set(r[a['q']]) for a in agent):.3f}", flush=True)
 print("new DM spend $%.3f over %d calls" % (S.COST["usd"], S.COST["calls"]))
