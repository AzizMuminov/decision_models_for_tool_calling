"""Offline: does one answer stock the agent for the rest of its work on that service?
For each search in the BM25 and DM top-k 5 refund-incident runs: the service the DM picks, and the tools the agent
used on that service for the REST OF THE RUN (from this search on). Scores: share of those covered by the answer,
and whether all are covered (one search would have been enough for that service).
Designs (at most 10 tools, except the menu): top 5, top 10, recommendations (top 3 'best' + top 7 'goes well with'),
menu (all names of the service; the upper bound).
    source .or_env && .venv/bin/python harness/recommend_replay.py
"""
import json, sys, statistics as st
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
import subtask_replay as S
from oracle_review import events
from dmtc.backends.dm import Choice

Q_BEST = "Which tool best serves what the agent wants to do?"
Q_PAIR = ("Which tools go well with what the agent wants to do: tools it will likely need alongside or next, such as "
          "prerequisites that look up IDs, follow-up steps, or closely related tools that work together with the main one? "
          "Not necessarily the exact tool it asked for.")
items = []
for run in sorted(S.RUNS.iterdir()):
    cond = run.name.split("__")[1]
    if cond not in ("hermes_default", "hermes_dmbridge"):
        continue
    ev = events(run, cond)
    for i, e in enumerate(ev):
        if e["kind"] != "search":
            continue
        later = []
        for x in ev[i + 1:]:
            if x["kind"] == "call" and x.get("ok") and x["tool"] not in later:
                later.append(x["tool"])
        items.append({"query": e["query"], "later": later})

def ask_service(state, s, q):
    names = [t["name"] for t in S.tools if t["server"] == s]; p = {}
    for j in range(0, len(names), 150):
        r, _ = S.ask(state, Choice(q, {n: S.card(S.by[n]) for n in names[j:j + 150]})); p.update(r)
    return sorted(p, key=p.get, reverse=True)

def work(it):
    st_ = {"what_the_agent_wants": it["query"]}
    ps, _ = S.ask(st_, Choice("Which service has the tool the agent needs?", {s: S.SERVER_INFO.get(s, s) for s in S.servers}))
    srv = max(ps, key=ps.get)
    best = ask_service(st_, srv, Q_BEST)
    pair = ask_service(st_, srv, Q_PAIR)
    rec = best[:3] + [n for n in pair if n not in best[:3]][:7]
    target = [n for n in it["later"] if S.by[n]["server"] == srv]
    return {"srv": srv, "target": target, "top5": best[:5], "top10": best[:10], "rec": rec,
            "menu": [t["name"] for t in S.tools if t["server"] == srv]}

with ThreadPoolExecutor(6) as ex:
    W = [w for w in ex.map(work, items) if w["target"]]
print(f"{len(W)} searches whose service the agent went on to use; tools used later on that service: mean {st.mean(len(w['target']) for w in W):.1f}")
print(f"{'design':38s} {'share covered':>13s} {'all covered':>12s} {'tools shown':>12s}")
for d, lab in (("top5", "DM top 5"), ("top10", "DM top 10"), ("rec", "recommendations: 3 best + 7 'goes well with'"), ("menu", "menu: all service names (upper bound)")):
    cov = st.mean(len(set(w["target"]) & set(w[d])) / len(w["target"]) for w in W)
    allc = st.mean(set(w["target"]) <= set(w[d]) for w in W)
    print(f"{lab:38s} {cov:13.2f} {allc:12.2f} {st.mean(len(w[d]) for w in W):12.1f}")
json.dump(W, open(HERE.parent / "results" / "long_incident" / "recommend_replay.json", "w"))
