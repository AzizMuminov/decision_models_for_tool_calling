"""Follow-up to selector_study: list size (15 vs 10 vs 8 vs 5), usage names on/off, and the 'none' option on/off.
Uses the selector_study DM cache; --dry counts uncached calls without spending."""
import sys, statistics as st
from concurrent.futures import ThreadPoolExecutor
sys.argv = [sys.argv[0]] + [a for a in sys.argv[1:]]
DRY = "--dry" in sys.argv; HELD = "--heldout" in sys.argv
import selector_study as S

MISS = {"n": 0}
_orig = S.ask_choice
def ask_choice(q, instr, opts):
    if S._key("c", q, instr, opts) not in S.CACHE:
        MISS["n"] += 1
        if DRY: return {}, 0.0
    return _orig(q, instr, opts)
S.ask_choice = ask_choice

def parts(q, none=True):
    ch, c = S.services(q, S.FIXED, none=none, force_named=True)
    if ch is None: return None
    best, _ = S.tools_choice(q, ch, S.Q_TOOL); exp, _ = S.tools_choice(q, ch, S.Q_TOOL_EXPERT)
    return ch, best, exp, S._pop_for(q)

def build(p, cap=15, pop=True, exp_n=3):
    if p is None: return []
    ch, best, exp, popl = p
    out = best[:5] + [t for t in exp if t not in best[:5]][:exp_n]
    if pop:
        for s in ch: out += [t for t in popl.get(s, []) if t in S.BY and t not in out]
    out += [t for t in best[5:] if t not in out]
    return out[:cap]

VARS = {"F 15 (as run)": dict(cap=15), "F 10": dict(cap=10), "F 10, no usage names": dict(cap=10, pop=False),
        "F 8 = 5 best + 3 expert": dict(cap=8), "F 5 best only": dict(cap=5),
        "F 15, no none": dict(cap=15, none=False), "F 10, no none": dict(cap=10, none=False)}

def main():
    hand, agent, named = S.load_sets()
    if HELD:
        agent = S.load_heldout(); hand = [h for h in hand if h["gold"] is None]
        S.TASK_OF.update({a["q"]: a["task"] for a in agent})
    qs = sorted({x["q"] for x in hand + agent + named})
    P = {}
    for none in (True, False):
        with ThreadPoolExecutor(8) as ex: P[none] = dict(zip(qs, ex.map(lambda q: parts(q, none), qs)))
    print("uncached DM calls:", MISS["n"], "spent $%.3f" % S.COST["usd"])
    if DRY: return
    M = lambda xs: (lambda L: st.mean(L) if L else float("nan"))(list(xs))
    H = [h for h in hand if h["gold"]]; N = [h for h in hand if not h["gold"]]
    print(f"{'variant':28s} hit1  hit_list cov  all  named  trueNone falseNone avg_len")
    for name, kw in VARS.items():
        none = kw.pop("none", True); r = {q: build(P[none][q], **kw) for q in qs}; kw["none"] = none
        row = [M(bool(r[h['q']]) and r[h['q']][0] in h['gold'] for h in H) if H else float('nan'),
               M(bool(h['gold'] & set(r[h['q']])) for h in H) if H else float('nan'),
               M(len(a['gold'] & set(r[a['q']])) / len(a['gold']) for a in agent),
               M(a['gold'] <= set(r[a['q']]) for a in agent),
               M(any(S.SRV[t] == n['named'] for t in r[n['q']]) for n in named),
               M(not r[h['q']] for h in N), M(not r[h['q']] for h in H) if H else float('nan'),
               M(len(r[q]) for q in qs)]
        print(f"{name:28s} " + " ".join(f"{x:.3f}" for x in row))
if __name__ == "__main__": main()
