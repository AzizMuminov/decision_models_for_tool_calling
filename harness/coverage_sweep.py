"""Offline: on the searches agents made in the refund-incident runs, which strategy puts ALL the new tools the agent
went on to use into an answer of at most 10 tools? "New": not called and not shown before this search.
Resolvable search: all its new tools are on one service, and the query names that service.
    source .or_env && .venv/bin/python harness/coverage_sweep.py
"""
import json, re, sys, statistics as st
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
import subtask_replay as S
from oracle_review import events
from selectors_real import Selectors
from dmtc.backends.dm import Choice, Noul

SEL = Selectors(S.tools)
ALIAS = {"atlassian": ["jira", "confluence", "atlassian"], "grafana": ["grafana", "loki", "oncall", "on-call", "prometheus", "incident"],
         "github": ["github", "pull request", " pr ", "commit"], "notion": ["notion", "runbook"], "postgres": ["postgres", "sql", "database", "db"],
         "sentry": ["sentry"], "slack": ["slack"], "gitlab": ["gitlab"]}

items = []
for run in sorted(S.RUNS.iterdir()):
    cond = run.name.split("__")[1]
    if cond not in ("hermes_default", "hermes_dmbridge"):
        continue
    ev = events(run, cond); seen = set()
    for i, e in enumerate(ev):
        if e["kind"] == "call":
            seen.add(e["tool"]); continue
        used = []
        for x in ev[i + 1:]:
            if x["kind"] == "search": break
            if x["kind"] == "call" and x.get("ok") and x["tool"] not in used: used.append(x["tool"])
        new = [u for u in used if u not in seen]
        seen |= set(e.get("ranked") or [])
        if not new: continue
        srvs = {S.by[n]["server"] for n in new}
        q = " " + e["query"].lower() + " "
        res = len(srvs) == 1 and any(a in q for a in ALIAS.get(next(iter(srvs)), [next(iter(srvs))]))
        items.append({"query": e["query"], "new": new, "server": next(iter(srvs)) if len(srvs) == 1 else None, "resolvable": res})
R = [it for it in items if it["resolvable"]]
print(f"{len(items)} searches led to new tools; resolvable {len(R)} ({len(R)/len(items):.0%}); new tools per resolvable search {st.mean(len(i['new']) for i in R):.2f}")

def pt_for(state, s, question="Which tool best serves what the agent wants to do?"):
    names = [t["name"] for t in S.tools if t["server"] == s]; pt = {}
    for j in range(0, len(names), 150):
        p, _ = S.ask(state, Choice(question, {n: S.card(S.by[n]) for n in names[j:j + 150]})); pt.update(p)
    return pt

def work(it):
    st_ = {"what_the_agent_wants": it["query"]}
    ps, _ = S.ask(st_, Choice("Which service has the tool the agent needs?", {s: S.SERVER_INFO.get(s, s) for s in S.servers}))
    srv = sorted(ps, key=ps.get, reverse=True)
    pt = pt_for(st_, srv[0])
    rank1 = sorted(pt, key=pt.get, reverse=True)
    pt_next = pt_for(st_, srv[0], "Which tool will the agent need for the other steps of this sub-task, besides the obvious first one?")
    rank_next = sorted(pt_next, key=pt_next.get, reverse=True)
    pool = rank1[:20]
    r = S.dm.ask(st_, {f"t{j}": Noul("The agent is searching for the tools of one sub-task. Will it need this tool for that sub-task, now or "
                                       f"in its next few steps? Tool: {S.card(S.by[n])}") for j, n in enumerate(pool)})
    yn = {n: (r.answers[f't{j}'].value if r.answers.get(f't{j}') else 0) for j, n in enumerate(pool)}
    bm = SEL.bm25(it["query"], 20)[0]
    return {"srv": srv, "rank1": rank1, "rank_next": rank_next, "yn": yn, "bm25": bm}

with ThreadPoolExecutor(6) as ex:
    W = list(ex.map(work, R))

def interleave(*lists, cap=10):
    out = []
    for i in range(max(len(l) for l in lists)):
        for l in lists:
            if i < len(l) and l[i] not in out: out.append(l[i])
    return out[:cap]

strategies = {
    "DM top 10": lambda w: w["rank1"][:10],
    "DM top 5": lambda w: w["rank1"][:5],
    "yes/no over top 20, cap 10 (ranked by P(yes))": lambda w: sorted([n for n, v in w["yn"].items() if v >= 0.5], key=lambda n: -w["yn"][n])[:10] or w["rank1"][:1],
    "yes/no over top 20, top 10 by P(yes)": lambda w: sorted(w["yn"], key=lambda n: -w["yn"][n])[:10],
    "DM 'best tool' + DM 'other steps', interleaved, 10": lambda w: interleave(w["rank1"], w["rank_next"]),
    "DM top 5 + BM25 top 5, 10": lambda w: interleave(w["rank1"][:5], w["bm25"][:10]),
    "DM top 7 + yes/no picks, 10": lambda w: interleave(w["rank1"][:7], sorted(w["yn"], key=lambda n: -w["yn"][n])),
    "BM25 top 10": lambda w: w["bm25"][:10],
}
print(f"\n{'strategy (at most 10 tools)':52s} {'all new covered':>15s} {'new covered':>11s} {'tools':>6s}")
for name, f in strategies.items():
    outs = [f(w) for w in W]
    whole = st.mean(set(it["new"]) <= set(o) for it, o in zip(R, outs))
    cov = st.mean(len(set(it["new"]) & set(o)) / len(it["new"]) for it, o in zip(R, outs))
    print(f"{name:52s} {whole:15.2f} {cov:11.2f} {st.mean(len(o) for o in outs):6.1f}")
right = st.mean(w["srv"][0] == it["server"] for it, w in zip(R, W))
print(f"\nDM picked the right service first: {right:.2f}")
best = strategies["DM 'best tool' + DM 'other steps', interleaved, 10"]
miss = [(it, w) for it, w in zip(R, W) if not set(it["new"]) <= set(strategies["DM top 10"](w))]
print(f"\nmisses of 'DM top 10' ({len(miss)}):")
for it, w in miss[:12]:
    lost = [n for n in it["new"] if n not in w["rank1"][:10]]
    print(f"  '{it['query'][:55]}' service picked {w['srv'][0]} | missing {lost} (rank {[w['rank1'].index(n)+1 if n in w['rank1'] else 'other service' for n in lost]})")
json.dump([{**it, **w} for it, w in zip(R, W)], open(HERE.parent / "results" / "long_incident" / "coverage_sweep.json", "w"))
