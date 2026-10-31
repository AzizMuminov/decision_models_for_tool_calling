"""Large tool holsters: choose 1 tool out of many.

Two sources:
  - bfcl_pad:  BFCL single-call items (should call) and irrelevance items (should not), padded with
               random distractor tools from BFCL's pool of ~1,270 distinct tools up to K tools.
               Gold answers stay valid, so this gives a scaling curve on the same items.
  - metatool:  MetaTool / ToolE (ICLR 2024): user queries labelled with the one tool (of 199) that
               serves them. Negatives are made by removing the gold tool from the holster.

Systems (decision = call or not + which tool; arguments are scored separately where available):
  llm_native   LLM with all K tools, native tool calling
  dm           one DM request: 4-way gate + Choice over all K tools (DMs accept <= 255 options)
  dm_tourney   chained DM requests for K > max_options: groups -> winners -> final (+ gate)
  lexical      bag-of-words top-1 (non-neural control), same gate as phase 1's lexical DM
  dm+fill      the DM's pick, then the LLM fills arguments seeing only that one tool (BFCL only)

    python -m dmtc.holster run configs/holster.yaml
    python -m dmtc.holster analyze results/holster_v1
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from tqdm import tqdm

from dmtc.backends.dm import Choice, make_dm
from dmtc.backends.llm import OutOfCredits, make_llm
from dmtc.data import load_bfcl
from dmtc.eval.bfcl_checker import score_bfcl
from dmtc.eval.metrics import bootstrap_ci, mcnemar
from dmtc.pipeline import DECISION_CRITERIA, DECISION_INSTRUCTIONS, SELECT_INSTRUCTIONS
from dmtc.schema import NameMap

POS_CATS = ["simple_python", "multiple", "live_simple", "live_multiple"]
NEG_CATS = ["irrelevance", "live_irrelevance"]
POOL_CATS = ["simple_python", "multiple", "live_simple", "live_multiple", "parallel", "parallel_multiple",
             "live_parallel", "live_parallel_multiple", "irrelevance", "live_irrelevance"]


# ---- items ----------------------------------------------------------------------------------
def _card(t: dict) -> dict:
    """Compact tool description for DM criteria (the LLM gets the full schema)."""
    params = list((t.get("parameters") or {}).get("properties", {}))
    return {"description": (t.get("description") or "")[:300], "parameters": params[:12]}


def bfcl_items(n_pos: int, n_neg: int, sizes: list[int], seed: int) -> list[dict]:
    rng = random.Random(seed)
    pool: dict[str, dict] = {}
    for c in POOL_CATS:
        for e in load_bfcl(c):
            for t in e.tools:
                pool.setdefault(t["name"], t)
    pos = [e for c in POS_CATS for e in load_bfcl(c) if e.ground_truth and len(e.ground_truth) == 1]
    neg = [e for c in NEG_CATS for e in load_bfcl(c)]
    pos, neg = rng.sample(pos, n_pos), rng.sample(neg, n_neg)
    items = []
    for e in pos + neg:
        own = {t["name"] for t in e.tools}
        # distractors drawn once per item at the largest size; smaller sizes are prefixes, so every
        # size is nested in the next (same item, more tools)
        others = [t for n, t in pool.items() if n not in own]
        extra = rng.sample(others, max(sizes))
        for k in sizes:
            tools = list(e.tools) + extra[: max(k - len(e.tools), 0)]
            random.Random(f"{seed}:{e.id}:{k}").shuffle(tools)  # gold position is random
            gold = sorted(next(iter(g)) for g in e.ground_truth) if e.should_call else []
            items.append({"id": f"{e.id}@K{k}", "src": "bfcl_pad", "base_id": e.id, "category": e.category,
                          "K": len(tools), "messages": e.messages, "tools": tools, "should_call": e.should_call,
                          "gold_tools": gold, "ground_truth": e.ground_truth})
    return items


def metatool_items(path: Path, n: int, neg_frac: float, seed: int) -> list[dict]:
    rng = random.Random(seed)
    docs = {r["docid"]: r["docstring"] for r in csv.DictReader(open(path / "corpus.tsv"), delimiter="\t")}
    queries = {r["qid"]: r["text"] for r in csv.DictReader(open(path / "queries.tsv"), delimiter="\t")}
    qrels = [(r["qid"], r["docid"]) for r in csv.DictReader(open(path / "qrels/test/single.tsv"), delimiter="\t")]
    tools_all = [{"name": f"tool_{d}", "description": desc,
                  "parameters": {"type": "object", "properties": {"request": {
                      "type": "string", "description": "The user's request, restated for the tool."}},
                      "required": ["request"]}} for d, desc in docs.items()]
    # stratify: spread the sample over the tools rather than over the (skewed) query counts
    by_tool = defaultdict(list)
    for q, d in qrels:
        by_tool[d].append(q)
    picks = []
    order = sorted(by_tool)
    rng.shuffle(order)
    while len(picks) < n:
        for d in order:
            if len(picks) >= n:
                break
            if by_tool[d]:
                picks.append((by_tool[d].pop(rng.randrange(len(by_tool[d]))), d))
    items = []
    for i, (q, d) in enumerate(picks):
        negative = i < int(n * neg_frac)
        tools = [t for t in tools_all if not (negative and t["name"] == f"tool_{d}")]
        random.Random(f"{seed}:{q}").shuffle(tools)
        items.append({"id": f"mt{q}{'-neg' if negative else ''}", "src": "metatool", "base_id": q,
                      "category": "metatool_neg" if negative else "metatool", "K": len(tools),
                      "messages": [{"role": "user", "content": queries[q]}], "tools": tools,
                      "should_call": not negative, "gold_tools": [] if negative else [f"tool_{d}"],
                      "ground_truth": None})
    return items


# ---- systems --------------------------------------------------------------------------------
class Runner:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.llms = {n: make_llm(c) for n, c in cfg["llms"].items()}
        self.dms = {n: make_dm(c) for n, c in cfg["dms"].items()}
        self.max_opts = cfg.get("max_options", 255)
        self.group = cfg.get("tourney_group", 50)

    def _state(self, it: dict, tools: list[dict]) -> dict:
        # tool *names* only: the descriptions are already in the Choice criteria, and sending them twice
        # overflows Microsoft-Decision-1's 32k context at ~200 tools
        return {"conversation": [{"role": m["role"], "content": m["content"]} for m in it["messages"]],
                "available_tools": [t["name"] for t in tools]}

    def dm_once(self, dm, it: dict, tools: list[dict], with_gate: bool):
        """Tool choice and gate as two requests (sent concurrently in a real system).

        The gate must see what the tools *do* to judge whether any fits. Names alone are not enough
        (MetaTool's are `tool_158`), and putting full cards in both the state and the criteria
        overflows a 32k context at ~200 tools. So the gate's state carries short descriptions and the
        tool Choice carries full cards in its criteria.
        """
        from concurrent.futures import ThreadPoolExecutor

        from dmtc.backends.dm import DMResult

        names = NameMap(tools)
        conv = self._state(it, [])["conversation"]
        sel_q = {"tool": Choice(SELECT_INSTRUCTIONS, {names.to_safe[t["name"]]: _card(t) for t in tools})}
        gate_state = {"conversation": conv,
                      "available_tools": [{"name": t["name"], "does": (t.get("description") or "")[:150]} for t in tools]}
        with ThreadPoolExecutor(2) as ex:
            f_sel = ex.submit(dm.ask, {"conversation": conv}, sel_q)
            f_gate = ex.submit(dm.ask, gate_state, {"decision": Choice(DECISION_INSTRUCTIONS, DECISION_CRITERIA)}) \
                if with_gate else None
            rs, rg = f_sel.result(), (f_gate.result() if f_gate else None)
        r = DMResult(answers={**rs.answers, **(rg.answers if rg else {})},
                     latency_s=max(rs.latency_s, rg.latency_s if rg else 0.0),
                     input_tokens=rs.input_tokens + (rg.input_tokens if rg else 0),
                     cost_usd=rs.cost_usd + (rg.cost_usd if rg else 0.0),
                     error=rs.error or (rg.error if rg else None), model=rs.model)
        probs = {names.orig(k): v for k, v in (r.answers["tool"].probabilities.items() if "tool" in r.answers else [])}
        gate = r.answers["decision"].probabilities if "decision" in r.answers else None
        return probs, gate, r

    def run_dm(self, name: str, it: dict) -> dict:
        dm = self.dms[name]
        tools = it["tools"]
        cost = lat = 0.0
        err = None
        single = None
        if len(tools) <= self.max_opts:
            probs, gate, r = self.dm_once(dm, it, tools, True)
            single = r
        if single is not None and not (single.error and "context" in single.error):
            cost, lat, err, rounds = r.cost_usd, r.latency_s, r.error, 1
        else:  # tournament: groups in parallel (latency = slowest group), then a final with the gate
            groups = [tools[i:i + self.group] for i in range(0, len(tools), self.group)]
            winners, glat = [], []
            for g in groups:
                p, _, r = self.dm_once(dm, it, g, False)
                cost += r.cost_usd
                glat.append(r.latency_s)
                err = err or r.error
                winners += [t for t in g if t["name"] in sorted(p, key=p.get, reverse=True)[:2]]
            probs, gate, r = self.dm_once(dm, it, winners, True)
            cost, lat, err, rounds = cost + r.cost_usd, max(glat) + r.latency_s, err or r.error, 2
        pick = max(probs, key=probs.get) if probs else None
        return {"pick": pick, "p_pick": probs.get(pick) if pick else None, "gate": gate,
                "p_tool_call": (gate or {}).get("tool_call"), "cost_usd": cost, "latency_s": lat,
                "error": err, "rounds": rounds,
                "top5": sorted(probs, key=probs.get, reverse=True)[:5]}

    def run_llm(self, name: str, it: dict, tools: list[dict] | None = None, system: str | None = None) -> dict:
        r = self.llms[name].call(it["messages"], tools if tools is not None else it["tools"], system=system)
        return {"calls": r.calls, "text": r.text[:300], "cost_usd": r.cost_usd, "latency_s": r.latency_s,
                "in_tok": r.input_tokens, "error": r.error}


def lexical_pick(it: dict) -> dict:
    from dmtc.backends.dm import LexicalDM

    lex = LexicalDM()
    names = NameMap(it["tools"])
    st = {"conversation": " ".join(m["content"] for m in it["messages"]),
          "tools": [{"name": t["name"], **_card(t)} for t in it["tools"]]}
    r = lex.ask(st, {"tool": Choice("", {names.to_safe[t["name"]]: _card(t) for t in it["tools"]}),
                     "decision": Choice("", DECISION_CRITERIA)})
    probs = {names.orig(k): v for k, v in r.answers["tool"].probabilities.items()}
    pick = max(probs, key=probs.get)
    return {"pick": pick, "p_tool_call": r.answers["decision"].probabilities["tool_call"], "cost_usd": 0.0,
            "latency_s": r.latency_s, "error": None, "rounds": 1, "top5": sorted(probs, key=probs.get, reverse=True)[:5]}


# ---- driver ---------------------------------------------------------------------------------
def run(cfg_path: str, limit: int | None) -> None:
    cfg = yaml.safe_load(open(cfg_path))
    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    items = []
    if "bfcl_pad" in cfg:
        b = cfg["bfcl_pad"]
        items += bfcl_items(b["n_pos"], b["n_neg"], b["sizes"], cfg["seed"])
    if "metatool" in cfg:
        m = cfg["metatool"]
        items += metatool_items(Path(m["path"]), m["n"], m["neg_frac"], cfg["seed"])
    items = items[:limit] if limit else items
    runner = Runner(cfg)
    lock = threading.Lock()
    stop = {"why": None}

    # (system, model, item) tasks; Sonnet only on its configured subset
    tasks = []
    for it in items:
        tasks.append(("lexical", "-", it))
        for d in runner.dms:
            tasks.append(("dm", d, it))
        for l, scope in cfg["llm_scope"].items():
            if it["K"] in scope.get("K", [it["K"]]) and it["src"] in scope.get("src", [it["src"]]) and \
                    random.Random(f"{l}:{it['base_id']}").random() < scope.get("frac", 1.0):
                tasks.append(("llm_native", l, it))
        if it["src"] == "bfcl_pad" and it["should_call"]:
            for d in runner.dms:
                tasks.append(("dm+fill", f"{d}>{cfg['fill_llm']}", it))

    files: dict[str, Any] = {}
    done = set()
    for p in out.glob("*.jsonl"):
        for l in open(p):
            r = json.loads(l)
            done.add((r["system"], r["model"], r["id"]))
    tasks = [t for t in tasks if (t[0], t[1], t[2]["id"]) not in done]

    def work(system: str, model: str, it: dict) -> dict | None:
        if stop["why"]:
            return None
        try:
            if system == "lexical":
                res = lexical_pick(it)
            elif system == "dm":
                res = runner.run_dm(model, it)
            elif system == "llm_native":
                res = runner.run_llm(model, it)
            else:  # dm+fill
                d, l = model.split(">")
                sel = runner.run_dm(d, it)
                chosen = [t for t in it["tools"] if t["name"] == sel["pick"]]
                fill = runner.run_llm(l, it, tools=chosen,
                                      system="A routing step selected the tool below for the user's request. "
                                             "Call it with the correct arguments.")
                res = {**fill, "pick": sel["pick"], "p_tool_call": sel["p_tool_call"],
                       "cost_usd": sel["cost_usd"] + fill["cost_usd"], "latency_s": sel["latency_s"] + fill["latency_s"],
                       "error": sel["error"] or fill["error"]}
        except OutOfCredits as e:
            stop["why"] = str(e)
            return None
        row = {"system": system, "model": model, "id": it["id"], "src": it["src"], "category": it["category"],
               "base_id": it["base_id"], "K": it["K"], "should_call": it["should_call"], "gold_tools": it["gold_tools"],
               **{k: v for k, v in res.items()}}
        if "calls" in res:  # LLM decided
            names = sorted(next(iter(c)) for c in res["calls"])
            row["called"] = bool(res["calls"])
            row["decision_ok"] = (names == it["gold_tools"]) if it["should_call"] else not res["calls"]
            if it["src"] == "bfcl_pad" and it["should_call"]:
                ex = load_bfcl_example(it)
                row["ast_ok"] = score_bfcl(ex, res["calls"])["correct"] if res["calls"] else False
        else:  # DM / lexical: decision at the two pre-registered gate thresholds
            for tau in cfg.get("taus", [0.5, 0.15]):
                open_ = (res["p_tool_call"] or 0) >= tau
                row[f"decision_ok@{tau}"] = (open_ and [res["pick"]] == it["gold_tools"]) if it["should_call"] else not open_
            row["select_ok"] = [res["pick"]] == it["gold_tools"] if it["should_call"] else None
        return row

    with ThreadPoolExecutor(cfg.get("concurrency", 8)) as pool:
        futs = [pool.submit(work, *t) for t in tasks]
        bar = tqdm(as_completed(futs), total=len(futs), desc="holster")
        spent = 0.0
        for fut in bar:
            row = fut.result()
            if row is None:
                continue
            spent += row.get("cost_usd") or 0
            with lock:
                key = f"{row['system']}__{row['model'].replace('>', '-')}"
                if key not in files:
                    files[key] = open(out / f"{key}.jsonl", "a")
                files[key].write(json.dumps(row, default=str) + "\n")
                files[key].flush()
            bar.set_postfix(spend=f"${spent:.2f}")
    for f in files.values():
        f.close()
    print(f"spend this run: ${spent:.4f}" + (f"; STOPPED: {stop['why']}" if stop["why"] else ""))


_BFCL_CACHE: dict[str, Any] = {}


def load_bfcl_example(it: dict):
    if it["category"] not in _BFCL_CACHE:
        _BFCL_CACHE[it["category"]] = {e.id: e for e in load_bfcl(it["category"])}
    ex = _BFCL_CACHE[it["category"]][it["base_id"]]
    # score against the padded holster: the checker looks up the called function's schema there
    from dataclasses import replace

    return replace(ex, tools=it["tools"])


def analyze(d: str) -> None:
    d = Path(d)
    rows = defaultdict(list)
    for p in d.glob("*.jsonl"):
        for l in open(p):
            r = json.loads(l)
            rows[(r["system"], r["model"])].append(r)
    L = [f"# Large tool holster: {d.name}", "",
         "Decision accuracy = called the right single tool when it should call, or did not call when it "
         "should not (arguments ignored). DM gate thresholds τ were fixed in advance from phase 1.", ""]
    for src in ("bfcl_pad", "metatool"):
        Ks = sorted({r["K"] for rs in rows.values() for r in rs if r["src"] == src})
        if not Ks:
            continue
        L += [f"## {src}", "", "| system | model | " + " | ".join(f"K={k}" for k in Ks) +
              " | should-not-call (all K) | p50 latency s | cost per 1k $ |", "|---|---|" + "---|" * (len(Ks) + 3)]
        for (sysname, model), rs in sorted(rows.items()):
            rs = [r for r in rs if r["src"] == src]
            if not rs:
                continue
            variants = [("decision_ok", "")] if sysname in ("llm_native", "dm+fill") else \
                [(f"decision_ok@{t}", f" τ={t}") for t in (0.5, 0.15)]
            for key, lab in variants:
                cells = []
                for k in Ks:
                    v = [r[key] for r in rs if r["K"] == k and key in r]
                    cells.append(f"{np.mean(v):.3f} (n={len(v)})" if v else "–")
                negs = [r[key] for r in rs if not r["should_call"] and key in r]
                neg_cell = f"{np.mean(negs):.3f} (n={len(negs)})" if negs else "–"
                lat = np.median([r["latency_s"] for r in rs])
                cost = 1000 * np.mean([r.get("cost_usd") or 0 for r in rs])
                L.append(f"| {sysname}{lab} | {model} | " + " | ".join(cells) + f" | {neg_cell} | {lat:.2f} | {cost:.3f} |")
        # selection only (should-call items, gate ignored) + end-to-end AST for BFCL
        L += ["", f"### {src}: tool selection only (should-call items, gate ignored)", "",
              "| system | model | " + " | ".join(f"K={k}" for k in Ks) + " |", "|---|---|" + "---|" * len(Ks)]
        for (sysname, model), rs in sorted(rows.items()):
            rs = [r for r in rs if r["src"] == src and r["should_call"]]
            if not rs:
                continue
            cells = []
            for k in Ks:
                rk = [r for r in rs if r["K"] == k]
                if not rk:
                    cells.append("–")
                    continue
                if "select_ok" in rk[0]:
                    v = [r["select_ok"] for r in rk]
                else:
                    v = [sorted(next(iter(c)) for c in r["calls"]) == r["gold_tools"] for r in rk]
                cells.append(f"{np.mean(v):.3f}")
            L.append(f"| {sysname} | {model} | " + " | ".join(cells) + " |")
        if src == "bfcl_pad":
            L += ["", "### bfcl_pad: end-to-end (BFCL AST checker, should-call items)", "",
                  "| system | model | " + " | ".join(f"K={k}" for k in Ks) + " |", "|---|---|" + "---|" * len(Ks)]
            for (sysname, model), rs in sorted(rows.items()):
                rs = [r for r in rs if r["src"] == src and "ast_ok" in r]
                if rs:
                    L.append(f"| {sysname} | {model} | " + " | ".join(
                        f"{np.mean([r['ast_ok'] for r in rs if r['K'] == k]):.3f}" if any(r['K'] == k for r in rs) else "–"
                        for k in Ks) + " |")
    (d / "REPORT.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("config")
    r.add_argument("--limit", type=int)
    a = sub.add_parser("analyze")
    a.add_argument("results_dir")
    args = ap.parse_args()
    run(args.config, args.limit) if args.cmd == "run" else analyze(args.results_dir)


if __name__ == "__main__":
    main()
