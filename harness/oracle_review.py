"""Oracle review of the searches agents made in the final run (results/final_deepseek).

For every tool search an agent made under Hermes BM25, DM v1, DM v3 (null) and embeddings:
  * the oracle label: which needed tool group the agent was on at that moment (the first group of the task's `needs`
    not yet satisfied by a successful call before the search); tasks that need no tool are labelled "none needed"
  * live hit: did the results the agent actually got contain a tool of that group?
  * replay: the same query sent to every ranker (BM25, DM v1, DM v3, embeddings), so all rankers are judged on
    identical queries; DM v3 answering "none" on a "none needed" search counts as correct
  * qualitative samples: queries where the rankers disagree, written to results/final_deepseek/oracle_review.md

Hermes BM25's searches are read from each run's state.db (Hermes's own bridge); the others from calls.jsonl.
Replays are cached in results/final_deepseek/oracle_replay_cache.jsonl.

    source .or_env && .venv/bin/python harness/oracle_review.py
"""

from __future__ import annotations

import json
import random
import sqlite3
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))
from real_sim import load_catalog  # noqa: E402
from selectors_real import Selectors  # noqa: E402

RUNS = REPO / "results" / "final_deepseek"
TASKS = HERE / "tasks_real_v2"
LIVE = {"hermes_default": "BM25 (Hermes)", "hermes_dmbridge": "DM v1", "hermes_dmnullbridge": "DM v3 (null)", "hermes_embedbridge": "embeddings"}
REPLAY = {"bm25": "BM25", "dm": "DM v1", "dm_null": "DM v3 (null)", "embed": "embeddings"}
PREFIX = "mcp__corp__"


def strip(n: str) -> str:
    return n[len(PREFIX):] if n.startswith(PREFIX) else n


def events(run: Path, cond: str) -> list[dict]:
    """Searches and calls of one run, in time order."""
    ev = [json.loads(l) for l in open(run / "calls.jsonl")] if (run / "calls.jsonl").exists() else []
    if cond == "hermes_default" and (run / "state.db").exists():
        ev = [e for e in ev if e["kind"] == "call"]
        con = sqlite3.connect(run / "state.db")
        rows = con.execute("select role, content, tool_calls, tool_call_id, timestamp from messages order by id").fetchall()
        pending = {}
        for role, content, tc, tcid, ts in rows:
            if role == "assistant" and tc:
                for c in json.loads(tc):
                    f = c.get("function") or {}
                    if f.get("name") == "tool_search":
                        try:
                            pending[c.get("id")] = json.loads(f.get("arguments") or "{}").get("query", "")
                        except json.JSONDecodeError:
                            pending[c.get("id")] = ""
            elif role == "tool" and tcid in pending:
                try:
                    ranked = [strip(m["name"]) for m in json.loads(content).get("matches", [])]
                except Exception:  # noqa: BLE001
                    ranked = []
                ev.append({"t": ts, "kind": "search", "query": pending.pop(tcid), "ranked": ranked})
        con.close()
    return sorted(ev, key=lambda e: e["t"])


def main() -> None:
    sel = Selectors(load_catalog(extra=True))
    searches = []
    for run in sorted((RUNS / "runs").iterdir()):
        task, cond, _, rep = run.name.split("__")
        if cond not in LIVE:
            continue
        meta = json.loads((TASKS / task / "task.json").read_text())
        groups = [set(g) for g in meta["needs"]]
        done = set()
        for e in events(run, cond):
            if e["kind"] == "call":
                if e.get("ok"):
                    done.add(e["tool"])
                continue
            step = next((i for i, g in enumerate(groups) if not g & done), None)
            target = groups[step] if step is not None else None
            searches.append({"task": task, "cond": cond, "rep": rep, "query": e["query"], "live": e["ranked"][:5],
                             "target": sorted(target) if target else None, "none_needed": not groups,
                             "any_needed": sorted(set().union(*groups)) if groups else []})
    print(f"{len(searches)} searches from {len({(s['task'], s['cond'], s['rep']) for s in searches})} runs")

    cache_f = RUNS / "oracle_replay_cache.jsonl"
    cache = {}
    if cache_f.exists():
        for line in cache_f.read_text().splitlines():
            r = json.loads(line)
            cache[(r["mech"], r["query"])] = r["ranked"]
    todo = sorted({(m, s["query"]) for s in searches for m in REPLAY} - set(cache))
    print(f"replaying {len(todo)} (ranker, query) pairs")

    def do(job):
        m, q = job
        try:
            ranked, _ = sel.search(m, q, 5)
        except Exception:  # noqa: BLE001
            return None
        with open(cache_f, "a") as f:
            f.write(json.dumps({"mech": m, "query": q, "ranked": ranked}) + "\n")
        return m, q, ranked

    with ThreadPoolExecutor(6) as ex:
        for r in ex.map(do, todo):
            if r:
                cache[(r[0], r[1])] = r[2]

    def hit(ranked, s):
        if s["none_needed"]:
            return not ranked  # only an empty answer is right when the task needs no tool
        return bool(s["target"]) and bool(set(s["target"]) & set(ranked[:5]))

    # live hit rates (what agents actually got)
    lines = ["# Oracle review of the final run's searches", "",
             "Oracle label = the needed-tool group the agent was on when it searched. Hit = the 5 results contain a tool of that group.", ""]
    lines += ["## Live: the results agents actually got", "", "| setup | searches | hit | searches on 'no tool needed' tasks |", "|---|---|---|---|"]
    for c, name in LIVE.items():
        ss = [s for s in searches if s["cond"] == c and not s["none_needed"] and s["target"]]
        nn = [s for s in searches if s["cond"] == c and s["none_needed"]]
        lines.append(f"| {name} | {len(ss)} | {sum(hit(s['live'], s) for s in ss) / max(1, len(ss)):.2f} | {len(nn)} |")
    # replay
    on_step = [s for s in searches if s["target"]]
    nn = [s for s in searches if s["none_needed"]]
    lines += ["", "## Replay: every query sent to every ranker", "",
              f"{len(on_step)} searches made while a needed tool was still missing; {len(nn)} searches on tasks that need no tool.", "",
              "| ranker | hit on all queries | queries written under BM25 | under DM v1 | under DM v3 | under embeddings | says 'none' when no tool is needed |",
              "|---|---|---|---|---|---|---|"]
    for m, name in REPLAY.items():
        cells = [f"{sum(hit(cache.get((m, s['query']), []), s) for s in on_step) / max(1, len(on_step)):.2f}"]
        for c in LIVE:
            sub = [s for s in on_step if s["cond"] == c]
            cells.append(f"{sum(hit(cache.get((m, s['query']), []), s) for s in sub) / max(1, len(sub)):.2f}")
        cells.append(f"{sum(not cache.get((m, s['query']), ['x']) for s in nn) / max(1, len(nn)):.2f}")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    # qualitative: disagreements between BM25 and DM v1
    rng = random.Random(0)
    dm_only = [s for s in on_step if hit(cache.get(("dm", s["query"]), []), s) and not hit(cache.get(("bm25", s["query"]), []), s)]
    bm_only = [s for s in on_step if hit(cache.get(("bm25", s["query"]), []), s) and not hit(cache.get(("dm", s["query"]), []), s)]
    both_miss = [s for s in on_step if not any(hit(cache.get((m, s["query"]), []), s) for m in REPLAY)]
    lines += ["", f"## Disagreements: DM v1 right, BM25 wrong ({len(dm_only)}), BM25 right, DM v1 wrong ({len(bm_only)}), every ranker wrong ({len(both_miss)})", ""]

    def show(title, items, n=8):
        out = [f"### {title}", ""]
        for s in rng.sample(items, min(n, len(items))):
            out.append(f"- **{s['task']}**, query `{s['query'][:120]}`  \n  needed: {', '.join(s['target'][:3])}  \n"
                       f"  BM25: {', '.join(cache.get(('bm25', s['query']), [])[:3]) or '(nothing)'}  \n"
                       f"  DM v1: {', '.join(cache.get(('dm', s['query']), [])[:3]) or '(nothing)'}  \n"
                       f"  embeddings: {', '.join(cache.get(('embed', s['query']), [])[:3]) or '(nothing)'}")
        return out + [""]

    lines += show("DM v1 right, BM25 wrong", dm_only) + show("BM25 right, DM v1 wrong", bm_only) + show("Every ranker wrong", both_miss)
    # DM v3 'none' answers on tasks that do need tools
    v3_none = [s for s in on_step if not cache.get(("dm_null", s["query"]), ["x"])]
    lines += [f"### DM v3 answered 'none' while a needed tool existed ({len(v3_none)} of {len(on_step)})", ""]
    lines += [f"- {s['task']}: `{s['query'][:120]}` (needed {', '.join(s['target'][:2])})" for s in rng.sample(v3_none, min(10, len(v3_none)))]
    (RUNS / "oracle_review.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:30]))
    print(f"\nfull review: {RUNS / 'oracle_review.md'}")


if __name__ == "__main__":
    main()
