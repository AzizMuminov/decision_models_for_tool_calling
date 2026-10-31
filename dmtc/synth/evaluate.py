"""Evaluates DMs and LLMs on SynthHolster (dmtc/synth/build.py).

Every system's output is mapped to When2Call's four labels; a tool_call is right only with the gold tool.
  LLMs: a tool call -> tool_call(name); a text reply -> labelled direct / request_for_info /
        cannot_answer by a judge model that is not evaluated.
  DMs:  4-way gate (P(tool_call) >= tau, tau fixed in advance) + Choice over the holster's tools.

    python -m dmtc.synth.evaluate run configs/synth.yaml
    python -m dmtc.synth.evaluate analyze results/synth_v1
"""

from __future__ import annotations

import argparse
import json
import os
import random
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

from dmtc.backends.llm import OutOfCredits
from dmtc.eval.metrics import bootstrap_ci, mcnemar
from dmtc.holster import Runner

LABELS = ["direct", "tool_call", "request_for_info", "cannot_answer"]


def catalog_tools(cat: dict) -> list[dict]:
    """Tools with >= 1 reviewed cue (the rest were distractor-only; leaving them out keeps the full
    holster under the DMs' 255-option limit)."""
    out = []
    for d in cat["domains"]:
        for f in d["families"]:
            for v in f["variants"]:
                if not v["cues"]:
                    continue
                props = {p: {"type": s["type"], "description": s.get("description", "")} for p, s in v["params"].items()}
                out.append({"name": v["name"], "description": f"[{d['brand']}] {v['description']}",
                            "parameters": {"type": "object", "properties": props, "required": v["required"]}})
    return out


def holster(it: dict, tools: list[dict], k: int | None, seed: int) -> list[dict]:
    """Full catalog (k=None) or the item's family + random distractors up to k. Removed tool excluded."""
    by = {t["name"]: t for t in tools}
    removed = it.get("removed_tool")
    pool = [t for t in tools if t["name"] != removed]
    if k is None:
        # ONE fixed order for every item, so the long tool block is a shared prefix that providers can
        # cache. The gold position still varies, since each item has a different gold tool.
        hs = list(pool)
        random.Random(f"{seed}:full-order").shuffle(hs)
        return hs
    else:
        fam = [by[n] for n in it["family_tools"] if n != removed and n in by]  # cue-less tools are not in the catalog
        rest = [t for t in pool if t["name"] not in set(it["family_tools"])]
        hs = fam + random.Random(f"{seed}:{it['id']}:{k}").sample(rest, max(k - len(fam), 0))
    hs = list(hs)
    random.Random(f"{seed}:{it['id']}:order:{k}").shuffle(hs)
    return hs


JUDGE_PROMPT = """A customer wrote to a support assistant; the assistant replied in text (no tool was used).
Classify the reply:
- direct: it answers or helps directly from general knowledge
- request_for_info: it asks the customer for missing information needed to proceed
- cannot_answer: it says it cannot do this / has no way to do it, or redirects elsewhere

Customer: {user}
Assistant: {reply}
Answer with exactly one word: direct, request_for_info, or cannot_answer."""


def judge_label(cli, model: str, user: str, reply: str) -> tuple[str, float]:
    if not reply.strip():
        return "cannot_answer", 0.0
    import time

    for attempt in range(12):
        try:
            r = cli.chat.completions.create(model=model, max_tokens=400, extra_body={"reasoning": {"effort": "low"}},
                                            messages=[{"role": "user", "content": JUDGE_PROMPT.format(user=user, reply=reply[:2000])}])
            break
        except Exception as e:  # noqa: BLE001
            if getattr(e, "status_code", None) == 402 and "in_flight" in str(e) and attempt < 11:
                time.sleep(min(10 * (attempt + 1), 60))  # reservations clear as in-flight requests finish
                continue
            if getattr(e, "status_code", None) in (402, 403):
                raise OutOfCredits(str(e)[:300]) from e
            if attempt == 11:
                raise
            time.sleep(3)
    txt = (r.choices[0].message.content or "").lower()
    lab = next((l for l in ("request_for_info", "cannot_answer", "direct") if l in txt), "direct")
    return lab, float(getattr(r.usage, "cost", 0) or 0)


def run(cfg_path: str, limit: int | None) -> None:
    import openai

    cfg = yaml.safe_load(open(cfg_path))
    data = Path(cfg["data"])
    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    cat = json.loads((data / "catalog.json").read_text())
    items = [json.loads(l) for l in open(data / "items.jsonl")][1:]  # line 0 is the canary
    items = items[:limit] if limit else items
    tools = catalog_tools(cat)
    runner = Runner(cfg)
    judge = openai.OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])
    sizes = cfg["sizes"]  # e.g. [25, null] (null = full catalog)
    done = set()
    for p in out.glob("*.jsonl"):
        for l in open(p):
            r = json.loads(l)
            done.add((r["system"], r["model"], r["id"], r["K"]))
    tasks = []
    for it in items:
        for k in sizes:
            hs = holster(it, tools, k, cfg["seed"])
            kk = len(hs)
            for d in runner.dms:
                tasks.append(("dm", d, it, hs))
            for l, scope in cfg["llm_scope"].items():
                if (k in scope.get("sizes", sizes)) and random.Random(f"{l}:{it['id']}").random() < scope.get("frac", 1.0):
                    tasks.append(("llm_native", l, it, hs))
    tasks = [t for t in tasks if (t[0], t[1], t[2]["id"], len(t[3])) not in done]
    lock, stop, files = threading.Lock(), {"why": None}, {}

    def work(system, model, it, hs):
        if stop["why"]:
            return None
        row = {"system": system, "model": model, "id": it["id"], "slice": it["slice"], "domain": it["domain"],
               "K": len(hs), "K_kind": "full" if len(hs) > 60 else "family+25", "gold_label": it["label"],
               "gold_tool": it["gold_tool"]}
        prompt_it = {"messages": it["messages"]}
        try:
            if system == "dm":
                res = runner.run_dm(model, {**prompt_it, "tools": hs})
                row.update(cost_usd=res["cost_usd"], latency_s=res["latency_s"], error=res["error"],
                           p_tool_call=res["p_tool_call"], gate=res["gate"], pick=res["pick"], top5=res["top5"])
                for tau in cfg["taus"]:
                    if (res["p_tool_call"] or 0) >= tau:
                        lab, tool = "tool_call", res["pick"]
                    else:
                        g = {a: v for a, v in (res["gate"] or {}).items() if a != "tool_call"}
                        lab, tool = (max(g, key=g.get) if g else "cannot_answer"), None
                    row[f"label@{tau}"] = lab
                    row[f"ok@{tau}"] = lab == it["label"] and (lab != "tool_call" or tool == it["gold_tool"])
            else:
                r = runner.run_llm(model, {**prompt_it, "tools": hs})
                jc = 0.0
                if r["calls"]:
                    lab, tool = "tool_call", next(iter(r["calls"][0]))
                    names = [next(iter(c)) for c in r["calls"]]
                else:
                    user = " | ".join(m["content"] for m in it["messages"] if m["role"] == "user")
                    lab, jc = judge_label(judge, cfg["judge_model"], user, r["text"])
                    tool, names = None, []
                row.update(cost_usd=r["cost_usd"], judge_cost_usd=jc, latency_s=r["latency_s"], error=r["error"],
                           label=lab, calls=r["calls"], text=r["text"][:400])
                # calling the gold tool several times (e.g. two grep patterns) is correct; any other tool is not
                row["ok"] = lab == it["label"] and (lab != "tool_call" or set(names) == {it["gold_tool"]})
        except OutOfCredits as e:
            stop["why"] = str(e)
            return None
        return row

    spent = 0.0
    with ThreadPoolExecutor(cfg.get("concurrency", 6)) as pool:
        futs = [pool.submit(work, *t) for t in tasks]
        bar = tqdm(as_completed(futs), total=len(futs), desc="synth")
        for fut in bar:
            row = fut.result()
            if row is None:
                continue
            spent += (row.get("cost_usd") or 0) + (row.get("judge_cost_usd") or 0)
            key = f"{row['system']}__{row['model']}"
            with lock:
                if key not in files:
                    files[key] = open(out / f"{key}.jsonl", "a")
                files[key].write(json.dumps(row, default=str) + "\n")
                files[key].flush()
            bar.set_postfix(spend=f"${spent:.2f}")
    for f in files.values():
        f.close()
    print(f"spend: ${spent:.4f}" + (f"; STOPPED: {stop['why']}" if stop["why"] else ""))


def analyze(d: str) -> None:
    d = Path(d)
    cfg = yaml.safe_load(open(d / "config.yaml"))
    rows = defaultdict(list)
    for p in d.glob("*.jsonl"):
        for l in open(p):
            r = json.loads(l)
            if r.get("error"):
                continue
            if r["system"] == "dm":
                for tau in cfg["taus"]:
                    rows[(f"dm τ={tau}", r["model"])].append({**r, "ok": r[f"ok@{tau}"], "label": r[f"label@{tau}"]})
            else:
                names = {next(iter(c)) for c in r.get("calls") or []}
                if r["label"] == "tool_call":  # re-score: repeated calls to the gold tool are correct
                    r["ok"] = r["gold_label"] == "tool_call" and names == {r["gold_tool"]}
                rows[("llm", r["model"])].append(r)
                if r["label"] == "tool_call" and r["gold_label"] == "tool_call":
                    r = {**r, "ok": r["gold_tool"] in names}  # lenient: gold among the calls
                rows[("llm lenient", r["model"])].append(r)
    slices = ["implicit", "multiturn", "sibling_neg", "missing", "direct"]
    L = ["# SynthHolster results", "",
         "Correct = right 4-way decision (tool_call / request_for_info / cannot_answer / direct), and for tool_call "
         "the right tool. DM thresholds τ fixed in advance from phase 1.", ""]
    for kind in ("family+25", "full"):
        L += [f"## Holster: {kind}", "", "| system | model | ALL | " + " | ".join(slices) +
              " | call/no-call acc | p50 latency s | cost per 1k $ |", "|---|---|---|" + "---|" * (len(slices) + 3)]
        for (sysn, model), rs in sorted(rows.items()):
            rs = [r for r in rs if r["K_kind"] == kind]
            if not rs:
                continue
            allok = [r["ok"] for r in rs]
            lo, hi = bootstrap_ci(allok)
            cells = []
            for s in slices:
                v = [r["ok"] for r in rs if r["slice"] == s]
                cells.append(f"{np.mean(v):.2f} ({len(v)})" if v else "–")
            callacc = np.mean([(r["label"] == "tool_call") == (r["gold_label"] == "tool_call") for r in rs])
            cost = 1000 * np.mean([(r.get("cost_usd") or 0) for r in rs])
            L.append(f"| {sysn} | {model} | {np.mean(allok):.3f} ±{(hi - lo) / 2:.3f} (n={len(rs)}) | " + " | ".join(cells) +
                     f" | {callacc:.3f} | {np.median([r['latency_s'] for r in rs]):.2f} | {cost:.3f} |")
        # tool selection on should-call items (gate ignored for DMs; LLM: right tool | it called)
        L += ["", f"Tool selection, should-call items ({kind}): DM = its pick (gate ignored); LLM = right tool given it called", ""]
        for (sysn, model), rs in sorted(rows.items()):
            rs = [r for r in rs if r["K_kind"] == kind and r["gold_label"] == "tool_call"]
            if not rs or sysn == "dm τ=0.15":
                continue
            if sysn.startswith("dm"):
                v = [r["pick"] == r["gold_tool"] for r in rs]
                t5 = np.mean([r["gold_tool"] in (r.get("top5") or []) for r in rs])
                L.append(f"- dm/{model}: top-1 {np.mean(v):.3f}, top-5 {t5:.3f} (n={len(v)})")
            else:
                c = [r for r in rs if r["calls"]]
                v = [{next(iter(x)) for x in r["calls"]} == {r["gold_tool"]} for r in c]
                L.append(f"- llm/{model}: {np.mean(v):.3f} of {len(c)} calls (it called on {len(c)}/{len(rs)} should-call items)")
        L.append("")
    # paired: each DM (τ fixed) vs each LLM, on shared items
    L += ["## Paired comparisons (McNemar, same items & holster)", "", "| DM | LLM | holster | n | DM | LLM | p |", "|---|---|---|---|---|---|---|"]
    for (s1, m1), a in rows.items():
        for (s2, m2), b in rows.items():
            if s1.startswith("dm") and s2 in ("llm", "llm lenient"):
                for kind in ("family+25", "full"):
                    A = {r["id"]: r["ok"] for r in a if r["K_kind"] == kind}
                    B = {r["id"]: r["ok"] for r in b if r["K_kind"] == kind}
                    ids = sorted(set(A) & set(B))
                    if ids:
                        t = mcnemar([B[i] for i in ids], [A[i] for i in ids])
                        L.append(f"| {m1} {s1[3:]} | {m2}{' (lenient)' if 'lenient' in s2 else ''} | {kind} | {len(ids)} | {np.mean([A[i] for i in ids]):.3f} | "
                                 f"{np.mean([B[i] for i in ids]):.3f} | {t['p_value']:.2g} |")
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
