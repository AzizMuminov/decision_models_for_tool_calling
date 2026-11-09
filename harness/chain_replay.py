"""Offline ablation of DM-aware chaining on the searches agents made in the refund-incident runs.

Variants (Microsoft Decision 1, server-then-tool hierarchy, multiple-choice ranking):
  k5        top 5 by P(server) x P(tool|server), the current DM
  topp      top-p 0.9, cap 5 (reference)
  k5+H      as k5, but the DM's state also holds the agent's last four catalog calls (option 1)
  set       rank as k5, then one yes/no question per top-8 tool, "needed now?"; keep the yeses, at least 1, at most 5 (option 2)
  set+H     both
Ground truth: the tools the agent successfully called next, before its next search (at most 3 distinct).
Scores: next-call hit (the first of them is in the results), coverage (share of them in the results), results size.
    source .or_env && .venv/bin/python harness/chain_replay.py
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
CONDS = ("hermes_default", "hermes_dmbridge")
tools = load_catalog(extra=True); by = {t["name"]: t for t in tools}; servers = sorted({t["server"] for t in tools})
dm = OpenRouterDM("microsoft/microsoft-decision-1")

items = []
for run in sorted(RUNS.iterdir()):
    cond = run.name.split("__")[1]
    if cond not in CONDS:
        continue
    ev = events(run, cond)
    for i, e in enumerate(ev):
        if e["kind"] != "search":
            continue
        hist = [x for x in ev[:i] if x["kind"] == "call"][-4:]
        nxt = []
        for x in ev[i + 1:]:
            if x["kind"] == "search":
                break
            if x["kind"] == "call" and x.get("ok") and x["tool"] not in nxt:
                nxt.append(x["tool"])
        if nxt:
            items.append({"cond": cond, "query": e["query"], "next": nxt[:3],
                          "hist": "; ".join(f"{x['tool']}({json.dumps(x.get('args', {}))[:90]}) -> {'ok' if x.get('ok') else 'failed'}" for x in hist) or "none yet"})
print(f"{len(items)} searches followed by a successful call", flush=True)

def ask(state, q):
    r = dm.ask(state, {"x": q})
    a = r.answers.get("x")
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
            cost += c
            score.update({n: ps[s] * v for n, v in pt.items()})
    return sorted(score, key=score.get, reverse=True), score, cost

def setsel(state, ranked):
    cand = ranked[:8]
    r = dm.ask(state, {f"t{j}": Noul(f"Does the agent need this tool for what it wants to do now? Tool: {card(by[n])}") for j, n in enumerate(cand)})
    keep = [n for j, n in enumerate(cand) if (r.answers.get(f"t{j}") and r.answers[f"t{j}"].value >= 0.5)]
    return (keep or cand[:1])[:5], r.cost_usd

def run(it):
    out, cost = {}, {}
    for h in (False, True):
        state = {"what_the_agent_wants": it["query"], **({"recent_tool_calls": it["hist"]} if h else {})}
        ranked, score, c = rank(state)
        tag = "+H" if h else ""
        out["k5" + tag], cost["k5" + tag] = ranked[:5], c
        if not h:
            tot, acc, tp = sum(score.values()) or 1, 0.0, []
            for n in ranked[:5]:
                tp.append(n); acc += score[n] / tot
                if acc >= 0.9:
                    break
            out["topp"], cost["topp"] = tp, c
        s, c2 = setsel(state, ranked)
        out["set" + tag], cost["set" + tag] = s, c + c2
    return {**it, "out": out, "cost": cost}

t0 = time.time()
with ThreadPoolExecutor(6) as ex:
    res = list(ex.map(run, items))
(HERE.parent / "results" / "long_incident" / "chain_replay.jsonl").write_text("\n".join(json.dumps(r) for r in res) + "\n")
print(f"done in {time.time() - t0:.0f}s\n")
print(f"{'variant':8s} {'next-call hit':>14s} {'coverage':>9s} {'tools mean/median':>18s} {'$/1k searches':>14s}")
for v in ("k5", "topp", "set", "k5+H", "set+H"):
    hit = st.mean(r["next"][0] in r["out"][v] for r in res)
    cov = st.mean(len(set(r["next"]) & set(r["out"][v])) / len(r["next"]) for r in res)
    sz = [len(r["out"][v]) for r in res]
    print(f"{v:8s} {hit:14.2f} {cov:9.2f} {st.mean(sz):10.1f} / {st.median(sz):<5} {1000 * st.mean(r['cost'][v] for r in res):12.2f}")
