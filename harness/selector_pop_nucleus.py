"""One 'best serves' call per service; usage names on/off x service nucleus 0.90 / 0.95 (max 3 services). 15 tools."""
import sys, statistics as st
from concurrent.futures import ThreadPoolExecutor
HELD = "--heldout" in sys.argv
import selector_study as S
def services(q, mass):
    opts = {s: S.FIXED.get(s, s) for s in S.SERVERS}; opts[S.NONE] = "None of these services can do what the agent asked."
    ps, _ = S.ask_choice(q, S.Q_SVC, opts)
    if ps.get(S.NONE, 0) >= 0.4: return None
    ps = {s: v for s, v in ps.items() if s != S.NONE}; tot = sum(ps.values()) or 1; ps = {s: v / tot for s, v in ps.items()}
    order = sorted(ps, key=ps.get, reverse=True)
    if (n := S.named_service(q)) and n in ps: order = [n] + [s for s in order if s != n]; ps[n] = max(ps[n], 0.9)
    ch, acc = [], 0.0
    for s in order:
        ch.append(s); acc += ps[s]
        if acc >= mass or len(ch) >= 3: break
    return {s: ps[s] for s in ch}
def lst(q, mass, pop):
    ch = services(q, mass)
    if ch is None: return [], 0
    best, _ = S.tools_choice(q, ch, S.Q_TOOL); out = best[:5]
    if pop:
        P = S._pop_for(q)
        for s in ch: out += [t for t in P.get(s, []) if t in S.BY and t not in out]
    out += [t for t in best[5:] if t not in out]
    return out[:15], len(ch)
hand, agent, named = S.load_sets()
if HELD:
    agent = S.load_heldout(); hand = [h for h in hand if h["gold"] is None]; S.TASK_OF.update({a["q"]: a["task"] for a in agent})
qs = sorted({x["q"] for x in hand + agent})
M = lambda xs: (lambda L: st.mean(L) if L else float("nan"))(list(xs)); H = [h for h in hand if h["gold"]]
print("variant                    hit1  hit5  cov5  cov15 all15 services/search")
for mass in (0.9, 0.95):
    for pop in (True, False):
        with ThreadPoolExecutor(8) as ex: R = dict(zip(qs, ex.map(lambda q: lst(q, mass, pop), qs)))
        r = {q: v[0] for q, v in R.items()}
        print(f"p={mass:<4} usage names {'on ' if pop else 'off'}  " + " ".join(f"{x:.3f}" for x in [
            M(bool(r[h['q']]) and r[h['q']][0] in h['gold'] for h in H), M(bool(h['gold'] & set(r[h['q']][:5])) for h in H),
            M(len(a['gold'] & set(r[a['q']][:5])) / len(a['gold']) for a in agent),
            M(len(a['gold'] & set(r[a['q']])) / len(a['gold']) for a in agent), M(a['gold'] <= set(r[a['q']]) for a in agent)])
            + f" {M(v[1] for v in R.values() if v[1]):.2f}", flush=True)
print("new DM spend $%.3f over %d calls" % (S.COST["usd"], S.COST["calls"]))
