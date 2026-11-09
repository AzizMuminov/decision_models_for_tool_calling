"""One tool-step call instead of two: best-serves only, expert-first only, vs F (both). Cache only."""
import sys, statistics as st
from concurrent.futures import ThreadPoolExecutor
HELD = "--heldout" in sys.argv
import selector_study as S
import selector_followup as Fu  # reuses parts(); its main() runs on import, so guard
def lst(p, mode):
    if p is None: return []
    ch, best, exp, pop = p
    first = {"F both": best, "best only": best, "expert only": exp}[mode]
    out = first[:5]
    if mode == "F both": out += [t for t in exp if t not in out][:3]
    for s in ch: out += [t for t in pop.get(s, []) if t in S.BY and t not in out]
    out += [t for t in first[5:] if t not in out]
    return out[:15]
hand, agent, named = S.load_sets()
if HELD:
    agent = S.load_heldout(); hand = [h for h in hand if h["gold"] is None]; S.TASK_OF.update({a["q"]: a["task"] for a in agent})
qs = sorted({x["q"] for x in hand + agent})
with ThreadPoolExecutor(8) as ex: P = dict(zip(qs, ex.map(Fu.parts, qs)))
M = lambda xs: (lambda L: st.mean(L) if L else float("nan"))(list(xs))
H = [h for h in hand if h["gold"]]
print("uncached calls", Fu.MISS["n"]); print("variant       hit1  hit5  cov5  cov15 all15")
for m in ("F both", "best only", "expert only"):
    r = {q: lst(P[q], m) for q in qs}
    print(f"{m:12s} " + " ".join(f"{x:.3f}" for x in [
        M(bool(r[h['q']]) and r[h['q']][0] in h['gold'] for h in H), M(bool(h['gold'] & set(r[h['q']][:5])) for h in H),
        M(len(a['gold'] & set(r[a['q']][:5])) / len(a['gold']) for a in agent),
        M(len(a['gold'] & set(r[a['q']])) / len(a['gold']) for a in agent), M(a['gold'] <= set(r[a['q']]) for a in agent)]))
