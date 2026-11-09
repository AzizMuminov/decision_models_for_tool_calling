"""Offline: which way of choosing a SET of tools per search covers the sub-task the agent was searching for?
Ground truth per search: the distinct tools the agent successfully called before its next search (its sub-task).
Searches from the BM25 and DM top-k 5 refund-incident runs. Ranking: Microsoft Decision 1, server then tool.
    source .or_env && .venv/bin/python harness/subtask_replay.py
"""
import json, statistics as st, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
from oracle_review import events
from real_sim import SERVER_INFO, load_catalog
from dmselect import card
from dmtc.backends.dm import Choice, Noul, OpenRouterDM

RUNS = HERE.parent / "results" / "long_incident" / "runs"
tools = load_catalog(extra=True); by = {t["name"]: t for t in tools}; servers = sorted({t["server"] for t in tools})
dm = OpenRouterDM("microsoft/microsoft-decision-1")
NONE = "__none__"

items = []
for run in sorted(RUNS.iterdir()):
    cond = run.name.split("__")[1]
    if cond not in ("hermes_default", "hermes_dmbridge"):
        continue
    ev = events(run, cond)
    for i, e in enumerate(ev):
        if e["kind"] != "search":
            continue
        used = []
        for x in ev[i + 1:]:
            if x["kind"] == "search":
                break
            if x["kind"] == "call" and x.get("ok") and x["tool"] not in used:
                used.append(x["tool"])
        if used:
            items.append({"cond": cond, "query": e["query"], "used": used})

def ask(state, q):
    r = dm.ask(state, {"x": q}); a = r.answers.get("x")
    return (a.probabilities if a else {}), r.cost_usd

def rank(state):
    ps, cost = ask(state, Choice("Which service has the tool the agent needs?", {s: SERVER_INFO.get(s, s) for s in servers}))
    chosen, acc = [], 0.0
    for s in sorted(ps, key=ps.get, reverse=True):
        chosen.append(s); acc += ps[s]
        if acc >= 0.9 or len(chosen) >= 3:
            break
    score = {}
    for s in chosen:
        names = [t["name"] for t in tools if t["server"] == s]
        for j in range(0, len(names), 150):
            pt, c = ask(state, Choice("Which tool best serves what the agent wants to do?", {n: card(by[n]) for n in names[j:j + 150]}))
            cost += c; score.update({n: ps[s] * v for n, v in pt.items()})
    return sorted(score, key=score.get, reverse=True), score, cost

def yesno(state, cand, text):
    r = dm.ask(state, {f"t{j}": Noul(f"{text} Tool: {card(by[n])}") for j, n in enumerate(cand)})
    keep = [n for j, n in enumerate(cand) if r.answers.get(f"t{j}") and r.answers[f"t{j}"].value >= 0.5]
    return (keep or cand[:1])[:5], r.cost_usd

def run(it):
    state = {"what_the_agent_wants": it["query"]}
    ranked, score, c0 = rank(state)
    out, cost = {"k5": ranked[:5], "k3": ranked[:3]}, {"k5": c0, "k3": c0}
    tot, acc, tp = sum(score.values()) or 1, 0.0, []
    for n in ranked[:5]:
        tp.append(n); acc += score[n] / tot
        if acc >= 0.9:
            break
    out["topp"], cost["topp"] = tp, c0
    top = score[ranked[0]] if ranked else 0
    rel = [n for n in ranked[:5] if score[n] >= 0.2 * top]
    out["relcut"], cost["relcut"] = (rel if len(rel) >= 2 else ranked[:2]), c0
    cand = ranked[:8]
    out["yn_now"], c = yesno(state, cand, "Does the agent need this tool for what it wants to do now?"); cost["yn_now"] = c0 + c
    out["yn_subtask"], c = yesno(state, cand, "The agent is searching for the tools of one sub-task. Will it need this tool for that sub-task, now or in its next few steps?")
    cost["yn_subtask"] = c0 + c
    seq, c_seq = ranked[:1], 0.0
    while len(seq) < 5:
        rest = [n for n in cand if n not in seq]
        if not rest:
            break
        opts = {n: card(by[n]) for n in rest}; opts[NONE] = "No other tool: the agent has what this sub-task needs."
        p, c = ask({**state, "tools_already_chosen": ", ".join(seq)},
                   Choice("Which other tool will the agent also need for this sub-task? Choose 'no other tool' if none.", opts))
        c_seq += c
        best = max(p, key=p.get) if p else NONE
        if best == NONE or p.get(NONE, 0) >= 0.5:
            break
        seq.append(best)
    out["sequential"], cost["sequential"] = seq, c0 + c_seq
    return {**it, "out": out, "cost": cost}

if __name__ == "__main__":
    print(f"{len(items)} searches followed by at least one successful call", flush=True)
    t0 = time.time()
    with ThreadPoolExecutor(6) as ex:
        res = list(ex.map(run, items))
    (HERE.parent / "results" / "long_incident" / "subtask_replay.jsonl").write_text("\n".join(json.dumps(r) for r in res) + "\n")
    print(f"done in {time.time() - t0:.0f}s; sub-task size mean {st.mean(len(r['used']) for r in res):.2f}\n")
    print(f"{'mode':11s} {'sub-task coverage':>18s} {'whole sub-task':>15s} {'next call':>10s} {'tools mean/median':>18s} {'$/1k':>6s}")
    for v in ("k5", "k3", "topp", "relcut", "yn_now", "yn_subtask", "sequential"):
        cov = st.mean(len(set(r["used"]) & set(r["out"][v])) / len(r["used"]) for r in res)
        whole = st.mean(set(r["used"]) <= set(r["out"][v]) for r in res)
        nxt = st.mean(r["used"][0] in r["out"][v] for r in res)
        sz = [len(r["out"][v]) for r in res]
        print(f"{v:11s} {cov:18.2f} {whole:15.2f} {nxt:10.2f} {st.mean(sz):10.1f} / {st.median(sz):<5} {1000 * st.mean(r['cost'][v] for r in res):6.2f}")
