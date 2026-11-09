"""One tool-step call: 'best serves' vs the context prompt (CTX_PROMPT), same service step as F, 15 tools, usage names."""
import sys, statistics as st
from concurrent.futures import ThreadPoolExecutor
HELD = "--heldout" in sys.argv
import selector_study as S
from selectors_real import Selectors
CTX = Selectors.CTX_PROMPT
def lst(q, instr):
    ch, _ = S.services(q, S.FIXED, none=True, force_named=True)
    if ch is None: return []
    best, _ = S.tools_choice(q, ch, instr); pop = S._pop_for(q)
    out = best[:5]
    for s in ch: out += [t for t in pop.get(s, []) if t in S.BY and t not in out]
    out += [t for t in best[5:] if t not in out]
    return out[:15]
hand, agent, named = S.load_sets()
if HELD:
    agent = S.load_heldout(); hand = [h for h in hand if h["gold"] is None]; S.TASK_OF.update({a["q"]: a["task"] for a in agent})
qs = sorted({x["q"] for x in hand + agent})
M = lambda xs: (lambda L: st.mean(L) if L else float("nan"))(list(xs)); H = [h for h in hand if h["gold"]]
print("prompt        hit1  hit5  cov5  cov15 all15")
for name, instr in (("best serves", S.Q_TOOL), ("context", CTX)):
    with ThreadPoolExecutor(8) as ex: r = dict(zip(qs, ex.map(lambda q: lst(q, instr), qs)))
    print(f"{name:12s} " + " ".join(f"{x:.3f}" for x in [
        M(bool(r[h['q']]) and r[h['q']][0] in h['gold'] for h in H), M(bool(h['gold'] & set(r[h['q']][:5])) for h in H),
        M(len(a['gold'] & set(r[a['q']][:5])) / len(a['gold']) for a in agent),
        M(len(a['gold'] & set(r[a['q']])) / len(a['gold']) for a in agent), M(a['gold'] <= set(r[a['q']]) for a in agent)]), flush=True)
print("new DM spend $%.3f over %d calls" % (S.COST["usd"], S.COST["calls"]))
