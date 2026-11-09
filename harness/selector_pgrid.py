"""p-value grid: service nucleus p_s (max 5 services) x tool nucleus p_t x list cap. One 'best serves' call per service.
List = 5 described + names: the DM's tools until cumulative mass p_t (or all, p_t=off), then usage names, capped."""
import sys, json, statistics as st, itertools
from concurrent.futures import ThreadPoolExecutor
HELD = "--heldout" in sys.argv
import selector_study as S
def svc(q):
    opts = {s: S.FIXED.get(s, s) for s in S.SERVERS}; opts[S.NONE] = "None of these services can do what the agent asked."
    ps, _ = S.ask_choice(q, S.Q_SVC, opts)
    if ps.get(S.NONE, 0) >= 0.4: return None
    ps = {s: v for s, v in ps.items() if s != S.NONE}; tot = sum(ps.values()) or 1; ps = {s: v / tot for s, v in ps.items()}
    order = sorted(ps, key=ps.get, reverse=True)
    if (n := S.named_service(q)) and n in ps: order = [n] + [s for s in order if s != n]; ps[n] = max(ps[n], 0.9)
    return order, ps
def chosen(order, ps, p, kmax=5):
    ch, acc = [], 0.0
    for s in order:
        ch.append(s); acc += ps[s]
        if acc >= p or len(ch) >= kmax: break
    return ch
TOOLP = {}
def tool_scores(q, ch, ps):
    sc = {}
    for s in ch:
        names = S.NAMES_BY[s]; chunks = [names[i:i + 150] for i in range(0, len(names), 150)]
        for c in chunks:
            p, _ = S.ask_choice(q, S.Q_TOOL, {n: S.card(S.BY[n]) for n in c})
            sc.update({n: ps[s] * v / len(chunks) for n, v in p.items()})
    return sc
def prep(q):
    sp = svc(q)
    if sp is None: return None
    order, ps = sp; ch5 = chosen(order, ps, 1.01)          # superset: up to 5 services
    return order, ps, tool_scores(q, ch5, ps)
def build(pre, q, ps_cut, pt, cap):
    if pre is None: return [], 0
    order, ps, sc = pre; ch = chosen(order, ps, ps_cut)
    tools = sorted((n for n in sc if S.SRV[n] in ch), key=sc.get, reverse=True)
    tot = sum(sc[n] for n in tools) or 1
    if pt == "off": dm = tools
    else:
        dm, acc = [], 0.0
        for n in tools:
            dm.append(n); acc += sc[n] / tot
            if acc >= pt: break
    P = S._pop_for(q); pop = [t for s in ch for t in P.get(s, []) if t in S.BY]
    out = list(dict.fromkeys(dm[:5] + pop + dm[5:] if pt == "off" else dm[:5] + dm[5:] + pop))
    return out[:cap], len(ch)
hand, agent, named = S.load_sets()
if HELD:
    agent = S.load_heldout(); hand = [h for h in hand if h["gold"] is None]; S.TASK_OF.update({a["q"]: a["task"] for a in agent})
qs = sorted({x["q"] for x in hand + agent})
with ThreadPoolExecutor(8) as ex: PRE = dict(zip(qs, ex.map(prep, qs)))
print("new DM spend $%.3f over %d calls" % (S.COST["usd"], S.COST["calls"]), flush=True)
M = lambda xs: (lambda L: st.mean(L) if L else float("nan"))(list(xs))
rows = []
print(f"{'p_svc':>5} {'p_tool':>6} {'cap':>3}  cov5  covAll allCov  len  svcs")
for ps_cut, pt, cap in itertools.product((0.9, 0.95, 0.99), ("off", 0.5, 0.9, 0.95, 0.99), (10, 15, 20)):
    R = {q: build(PRE[q], q, ps_cut, pt, cap) for q in qs}; r = {q: v[0] for q, v in R.items()}
    row = dict(p_svc=ps_cut, p_tool=pt, cap=cap,
               cov5=M(len(a["gold"] & set(r[a["q"]][:5])) / len(a["gold"]) for a in agent),
               cov=M(len(a["gold"] & set(r[a["q"]])) / len(a["gold"]) for a in agent),
               allc=M(a["gold"] <= set(r[a["q"]]) for a in agent),
               length=M(len(r[a["q"]]) for a in agent), svcs=M(v[1] for v in R.values() if v[1]))
    rows.append(row)
    print(f"{ps_cut:>5} {str(pt):>6} {cap:>3}  {row['cov5']:.3f} {row['cov']:.3f}  {row['allc']:.3f} {row['length']:5.1f} {row['svcs']:.2f}", flush=True)
json.dump(rows, open(S.OUT / f"pgrid_{'heldout' if HELD else 'dev'}.json", "w"), indent=1)
