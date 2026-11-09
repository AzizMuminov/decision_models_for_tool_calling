"""Offline study of DM tool-selector designs (Microsoft Decision 1), on three query sets:
  HAND   120 hand-written queries (retrieval_queries.Q, task set v1 steps) + 40 with no matching tool (NEG)
  AGENT  400 searches from the BM25 runs of results/final_rec510 (labels not influenced by the DM); label = the NEW tools
         (not called or shown before) the agent used after the search, before its next search
  NAMED  queries that name exactly one service (oversampling git / github / gitlab), from results/final_rec510
Every variant returns a ranked list of at most 15 tools (or nothing, for the 'none' variant). All DM calls are cached
in results/selector_study/dm_cache.jsonl, so reruns are identical and free.
    source .or_env && .venv/bin/python harness/selector_study.py
"""
from __future__ import annotations
import hashlib, json, math, random, re, statistics as st, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
from oracle_review import events
from real_sim import SERVER_INFO, load_catalog
from dmselect import card
from dmtc.backends.dm import Choice, Noul, OpenRouterDM
from retrieval_queries import NEG, Q

OUT = HERE.parent / "results" / "selector_study"; OUT.mkdir(exist_ok=True)
TOOLS = load_catalog(extra=True); BY = {t["name"]: t for t in TOOLS}; SRV = {t["name"]: t["server"] for t in TOOLS}
SERVERS = sorted(set(SRV.values())); NAMES_BY = {s: [t["name"] for t in TOOLS if t["server"] == s] for s in SERVERS}
DM = OpenRouterDM("microsoft/microsoft-decision-1")
FIXED = dict(SERVER_INFO)
FIXED.update({
    "github": "GitHub (remote code hosting): repositories, branches, commits, creating, updating and pushing files, pull requests and reviews, issues, Actions CI runs and logs, releases, code search, security alerts",
    "git": "Git (the local working copy on this machine only; no remote hosting, no pull requests or issues): status, log, diff, local branches, staging and committing",
    "gitlab": "GitLab (remote code hosting, used for some legacy projects): projects, branches, commits, files, merge requests, issues, CI pipelines and job logs, wiki",
})
NONE = "__none__"
Q_SVC = "Which service has the tool the agent needs?"
Q_SVC_NAMED = "Which service has the tool the agent needs? If the request names a service or product, choose that service."
Q_TOOL = "Which tool best serves what the agent wants to do?"
Q_TOOL_EXPERT = "Which tool would an expert engineer call first to do what the agent wants?"
Q_TOOL_RELEVANT = "Which of these tools are relevant to what the agent wants to do?"
Q_YN = "Could the agent use this tool for what it asked, either directly or as a necessary step? Tool: "
ALIAS = {"github": [r"\bgithub\b"], "gitlab": [r"\bgitlab\b"], "git": [r"\bgit\b(?!hub|lab)"], "atlassian": [r"\bjira\b", r"\bconfluence\b", r"\batlassian\b"],
         "grafana": [r"\bgrafana\b", r"\bloki\b", r"\bprometheus\b"], "sentry": [r"\bsentry\b"], "slack": [r"\bslack\b"], "notion": [r"\bnotion\b"],
         "linear": [r"\blinear\b"], "kubernetes": [r"\bkubernetes\b", r"\bkubectl\b", r"\bk8s\b"], "mongodb": [r"\bmongo(db)?\b"],
         "postgres": [r"\bpostgres(ql)?\b"], "aws_cloudwatch": [r"\bcloudwatch\b"], "aws_eks": [r"\beks\b"], "circleci": [r"\bcircleci\b"]}

def named_service(q):
    hits = [s for s, pats in ALIAS.items() if any(re.search(p, q.lower()) for p in pats)]
    return hits[0] if len(hits) == 1 else None

# ------------------------------------------------------------------ cached DM calls
_LOCK = threading.Lock(); CACHE_F = OUT / "dm_cache.jsonl"; CACHE = {}
if CACHE_F.exists():
    for l in open(CACHE_F):
        d = json.loads(l); CACHE[d["k"]] = (d["p"], d["c"])
COST = {"usd": 0.0, "calls": 0}

def _key(*parts): return hashlib.sha1(json.dumps(parts, sort_keys=True).encode()).hexdigest()

def ask_choice(query, instr, options):
    k = _key("c", query, instr, options)
    if k in CACHE:
        return CACHE[k]
    for attempt in range(4):
        r = DM.ask({"what_the_agent_wants": query}, {"x": Choice(instr, options)})
        if not r.error: break
        time.sleep(2 ** attempt)
    a = r.answers.get("x"); p = a.probabilities if a else {}
    with _LOCK:
        CACHE[k] = (p, r.cost_usd); COST["usd"] += r.cost_usd; COST["calls"] += 1
        open(CACHE_F, "a").write(json.dumps({"k": k, "p": p, "c": r.cost_usd}) + "\n")
    return p, r.cost_usd

def ask_yn(query, names, prompt=Q_YN):
    out, cost = {}, 0.0
    for i in range(0, len(names), 250):
        chunk = names[i:i + 250]; k = _key("y", query, prompt, chunk)
        if k not in CACHE:
            for attempt in range(4):
                r = DM.ask({"what_the_agent_wants": query}, {f"t{j}": Noul(prompt + card(BY[n])) for j, n in enumerate(chunk)})
                if not r.error: break
                time.sleep(2 ** attempt)
            p = {n: (r.answers[f"t{j}"].value if r.answers.get(f"t{j}") else 0.0) for j, n in enumerate(chunk)}
            with _LOCK:
                CACHE[k] = (p, r.cost_usd); COST["usd"] += r.cost_usd; COST["calls"] += 1
                open(CACHE_F, "a").write(json.dumps({"k": k, "p": p, "c": r.cost_usd}) + "\n")
        p, c = CACHE[k]; out.update(p); cost += c
    return out, cost

# ------------------------------------------------------------------ selector variants (each -> (ranked list, cost, calls))
def services(query, cards, instr=Q_SVC, nucleus=True, none=False, force_named=False):
    opts = {s: cards.get(s, s) for s in SERVERS}
    if none: opts[NONE] = "None of these services can do what the agent asked."
    ps, c = ask_choice(query, instr, opts)
    if none and ps.get(NONE, 0) >= 0.4:
        return None, c
    ps = {s: v for s, v in ps.items() if s != NONE}; tot = sum(ps.values()) or 1; ps = {s: v / tot for s, v in ps.items()}
    order = sorted(ps, key=ps.get, reverse=True)
    if force_named and (n := named_service(query)) and n in ps:
        order = [n] + [s for s in order if s != n]; ps[n] = max(ps[n], 0.9)
    chosen, acc = [], 0.0
    for s in order:
        chosen.append(s); acc += ps[s]
        if not nucleus or acc >= 0.9 or len(chosen) >= 3: break
    return {s: ps[s] for s in chosen}, c

def tools_choice(query, chosen, instr=Q_TOOL):
    score, cost = {}, 0.0
    for s, w in chosen.items():
        names = NAMES_BY[s]
        for i in range(0, len(names), 150):
            p, c = ask_choice(query, instr, {n: card(BY[n]) for n in names[i:i + 150]}); cost += c
            score.update({n: w * v for n, v in p.items()})
    return sorted(score, key=score.get, reverse=True)[:15], cost

def v_service(cards, **kw):
    def f(q):
        ch, c = services(q, cards, **{k: v for k, v in kw.items() if k != "tool_instr"})
        if ch is None: return [], c
        r, c2 = tools_choice(q, ch, kw.get("tool_instr", Q_TOOL)); return r, c + c2
    return f

def v_yn_in_services(q):
    ch, c = services(q, FIXED)
    names = [n for s in ch for n in NAMES_BY[s]]; p, c2 = ask_yn(q, names)
    return sorted(names, key=lambda n: -(ch[SRV[n]] * p.get(n, 0)))[:15], c + c2

def v_sliding(q):
    names = [t["name"] for t in TOOLS]; pool, cost = [], 0.0
    for i in range(0, len(names), 150):
        p, c = ask_choice(q, Q_TOOL, {n: card(BY[n]) for n in names[i:i + 150]}); cost += c
        pool += sorted(p, key=p.get, reverse=True)[:3]
    p, c = ask_choice(q, Q_TOOL, {n: card(BY[n]) for n in pool}); cost += c
    return sorted(pool, key=lambda n: -p.get(n, 0))[:15], cost

def v_flat_yn(q):
    names = [t["name"] for t in TOOLS]; p, c = ask_yn(q, names)
    return sorted(names, key=lambda n: -p.get(n, 0))[:15], c

def interleave(*lists, cap=15):
    out = []
    for i in range(max(len(l) for l in lists)):
        for l in lists:
            if i < len(l) and l[i] not in out: out.append(l[i])
    return out[:cap]

def v_two_prompts(second):
    def f(q):
        ch, c = services(q, FIXED)
        a, c1 = tools_choice(q, ch, Q_TOOL); b, c2 = tools_choice(q, ch, second)
        return interleave(a, b), c + c1 + c2
    return f

import json as _j
_POP = _j.loads((HERE / "tool_popularity.json").read_text())
_LOO = _j.loads((HERE / "tool_popularity_loo.json").read_text())
TASK_OF = {}  # query -> task, for leave-one-task-out usage lists (held-out set)
def _pop_for(q):
    t = TASK_OF.get(q)
    return _LOO.get(t, _POP) if t else _POP
from selectors_real import Selectors as _Sel
_BM = _Sel(TOOLS)
def v_bm25(q):
    return _BM.bm25(q, 15)[0], 0.0

def v_franken(q):
    """Combined selector: fixed cards, nucleus services (until 90%, max 3), none option, named service always in;
    5 best tools (choice, 'best serves'); up to 3 prerequisites ('expert first'); most-used names up to 15 in total."""
    pop = _pop_for(q)
    ch, c = services(q, FIXED, none=True, force_named=True)
    if ch is None:
        return [], c
    best, c1 = tools_choice(q, ch, Q_TOOL); exp, c2 = tools_choice(q, ch, Q_TOOL_EXPERT)
    out = best[:5]
    out += [t for t in exp if t not in out][:3]
    for s in ch:
        out += [t for t in pop.get(s, []) if t in BY and t not in out]
    out += [t for t in best[5:] if t not in out]
    return out[:15], c + c1 + c2

def v_rec_live(q):
    """Today's live design (DM recommendations 5+10): top-1 service with the original cards, 5 best, then most-used names."""
    pop = _pop_for(q)
    ch, c = services(q, SERVER_INFO, nucleus=False)
    best, c1 = tools_choice(q, ch, Q_TOOL); s = next(iter(ch))
    out = best[:5] + [t for t in pop.get(s, []) + best[5:] if t in BY and t not in best[:5]]
    return list(dict.fromkeys(out))[:15], c + c1

VARIANTS = {
    "1 current (top service)": v_service(SERVER_INFO, nucleus=False),
    "2 nucleus services": v_service(SERVER_INFO),
    "3 nucleus + fixed cards": v_service(FIXED),
    "4 3 + 'choose named service' prompt": v_service(FIXED, instr=Q_SVC_NAMED),
    "5 3 + none option": v_service(FIXED, none=True),
    "6 3 + named-service rule": v_service(FIXED, force_named=True),
    "7 3 + 'expert first' tool prompt": v_service(FIXED, tool_instr=Q_TOOL_EXPERT),
    "8 3 + yes/no per tool": v_yn_in_services,
    "9 sliding window (no service step)": v_sliding,
    "10 yes/no over all tools (no service step)": v_flat_yn,
    "11 3 + 'which tools are relevant' prompt": v_service(FIXED, tool_instr=Q_TOOL_RELEVANT),
    "12 3 + best-serves & expert-first interleaved": v_two_prompts(Q_TOOL_EXPERT),
    "13 3 + best-serves & relevant interleaved": v_two_prompts(Q_TOOL_RELEVANT),
    "F combined selector": v_franken,
    "R live recommendations design": v_rec_live,
    "5b variant 5 (simple candidate)": v_service(FIXED, none=True),
    "B BM25 (reference)": v_bm25,
}
ONLY = [a for a in sys.argv[1:] if not a.startswith("-")]  # run only variants whose number is listed, e.g. 11 12 13

# ------------------------------------------------------------------ query sets
def load_sets(seed=7):
    tasks = {p.name: json.loads((p / "task.json").read_text()) for p in (HERE / "tasks_real").iterdir()}
    hand = [{"q": q, "gold": set(tasks[t]["needs"][s])} for (t, s), pair in Q.items() for q in pair] + [{"q": q, "gold": None} for q in NEG]
    agent, named = [], {}
    for run in sorted((HERE.parent / "results" / "final_rec510" / "runs").iterdir()):
        task, cond, _, _ = run.name.split("__")
        if task == "k8s_crashloop": continue
        ev = events(run, cond); seen = set()
        for i, e in enumerate(ev):
            if e["kind"] == "call":
                seen.add(e["tool"]); continue
            if (s := named_service(e["query"])): named.setdefault(e["query"], s)
            new = []
            for x in ev[i + 1:]:
                if x["kind"] == "search": break
                if x["kind"] == "call" and x.get("ok") and x["tool"] in SRV and x["tool"] not in seen and x["tool"] not in new: new.append(x["tool"])
            seen |= set(e.get("ranked") or [])
            if new and cond == "hermes_default": agent.append({"q": e["query"], "gold": set(new)})
    rnd = random.Random(seed); rnd.shuffle(agent)
    gitfam = [q for q, s in named.items() if s in ("git", "github", "gitlab")]; other = [q for q, s in named.items() if s not in ("git", "github", "gitlab")]
    rnd.shuffle(gitfam); rnd.shuffle(other)
    named_set = [{"q": q, "named": named[q]} for q in gitfam[:200] + other[:150]]
    return hand, agent[:400], named_set

def load_heldout(n=500, seed=11):
    out = []
    for run in sorted((HERE.parent / "results" / "final_deepseek" / "runs").iterdir()):
        task, cond, _, _ = run.name.split("__")
        if cond not in ("hermes_default", "hermes_embedbridge", "hermes_ccbridge") or task == "k8s_crashloop": continue
        ev = events(run, cond); seen = set()
        for i, e in enumerate(ev):
            if e["kind"] == "call":
                seen.add(e["tool"]); continue
            new = []
            for x in ev[i + 1:]:
                if x["kind"] == "search": break
                if x["kind"] == "call" and x.get("ok") and x["tool"] in SRV and x["tool"] not in seen and x["tool"] not in new: new.append(x["tool"])
            seen |= set(e.get("ranked") or [])
            if new: out.append({"q": e["query"], "gold": set(new), "task": task})
    random.Random(seed).shuffle(out); return out[:n]

def main():
    hand, agent, named = load_sets()
    if "--heldout" in sys.argv:
        agent = load_heldout(); hand = [h for h in hand if h["gold"] is None]
        TASK_OF.update({a["q"]: a["task"] for a in agent})  # keep only the no-tool queries from HAND
        print("HELD-OUT agent queries from results/final_deepseek:", len(agent), flush=True)
    print(f"sets: HAND {len(hand)} ({sum(h['gold'] is None for h in hand)} no-tool), AGENT {len(agent)}, NAMED {len(named)} "
          f"({sum(n['named'] in ('git','github','gitlab') for n in named)} git family)", flush=True)
    rows = []
    for name, f in VARIANTS.items():
        if ONLY and name.split()[0] not in ONLY: continue
        t0 = time.time(); c0 = COST["usd"]
        qs = sorted({x["q"] for x in hand + agent + named})
        with ThreadPoolExecutor(8) as ex:
            res = dict(zip(qs, ex.map(f, qs)))
        H = [h for h in hand if h["gold"]]; N = [h for h in hand if not h["gold"]]
        r = lambda q: res[q][0]
        M = lambda xs: (lambda L: st.mean(L) if L else float("nan"))(list(xs))
        row = {"variant": name,
               "hand_hit1": M(bool(r(h["q"])) and r(h["q"])[0] in h["gold"] for h in H),
               "hand_hit5": M(bool(h["gold"] & set(r(h["q"])[:5])) for h in H),
               "hand_hit15": M(bool(h["gold"] & set(r(h["q"])[:15])) for h in H),
               "false_none": M(not r(h["q"]) for h in H), "true_none": M(not r(h["q"]) for h in N),
               "agent_cov5": M(len(a["gold"] & set(r(a["q"])[:5])) / len(a["gold"]) for a in agent),
               "agent_cov15": M(len(a["gold"] & set(r(a["q"])[:15])) / len(a["gold"]) for a in agent),
               "agent_all15": M(a["gold"] <= set(r(a["q"])[:15]) for a in agent),
               "named_top1": M(bool(r(n["q"])) and SRV[r(n["q"])[0]] == n["named"] for n in named),
               "gitfam_top1": M(bool(r(n["q"])) and SRV[r(n["q"])[0]] == n["named"] for n in named if n["named"] in ("git", "github", "gitlab")),
               "named_in15": M(any(SRV[t] == n["named"] for t in r(n["q"])[:15]) for n in named),
               "usd_per_1k": 1000 * M(res[q][1] for q in qs), "new_spend": COST["usd"] - c0, "secs": time.time() - t0}
        rows.append(row)
        print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)
    json.dump(rows, open(OUT / ("results.json" if not ONLY else f"results_{'_'.join(ONLY)}.json"), "w"), indent=1)
    print(f"\ntotal new DM spend ${COST['usd']:.2f} over {COST['calls']} calls")

if __name__ == "__main__":
    main()
