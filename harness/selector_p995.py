"""No usage names, no list limit: service nucleus p_s (max 5 services) and tool nucleus p_t."""
import sys, statistics as st
from concurrent.futures import ThreadPoolExecutor
HELD = "--heldout" in sys.argv
sys.argv = [a for a in sys.argv]
import selector_study as S
import importlib.util
spec = importlib.util.spec_from_file_location("g", str(S.HERE / "selector_pgrid_nopop.py"))
src = open(spec.origin).read().split("hand, agent, named = S.load_sets()")[0]
G = {}; exec(src, G)
hand, agent, named = S.load_sets()
if HELD:
    agent = S.load_heldout(); hand = [h for h in hand if h["gold"] is None]; S.TASK_OF.update({a["q"]: a["task"] for a in agent})
qs = sorted({x["q"] for x in hand + agent})
with ThreadPoolExecutor(8) as ex: PRE = dict(zip(qs, ex.map(G["prep"], qs)))
M = lambda xs: (lambda L: st.mean(L) if L else float("nan"))(list(xs))
H = [h for h in hand if h["gold"]]; N = [h for h in hand if not h["gold"]]
print("p_svc p_tool | hit1  cov5  cov   all   len mean/median/max  svcs  trueNone")
for ps_cut, pt in ((0.99, 0.99), (0.995, 0.995), (0.999, 0.999), (0.995, "off")):
    R = {q: G["build"](PRE[q], q, ps_cut, pt, 10**6) for q in qs}; r = {q: v[0] for q, v in R.items()}
    L = [len(r[a["q"]]) for a in agent]
    print(f"{ps_cut:<5} {str(pt):<6} | " + (f"{M(bool(r[h['q']]) and r[h['q']][0] in h['gold'] for h in H):.3f} " if H else "  -   ")
          + f"{M(len(a['gold'] & set(r[a['q']][:5])) / len(a['gold']) for a in agent):.3f} {M(len(a['gold'] & set(r[a['q']])) / len(a['gold']) for a in agent):.3f} "
          f"{M(a['gold'] <= set(r[a['q']]) for a in agent):.3f} {st.mean(L):5.1f}/{st.median(L):3.0f}/{max(L):4d}  {M(v[1] for v in R.values() if v[1]):.2f}  {M(not r[h['q']] for h in N):.2f}", flush=True)
print("new DM spend $%.3f" % S.COST["usd"])
