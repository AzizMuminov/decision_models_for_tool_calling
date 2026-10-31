"""Phase 2a: next-action replay on resolved OpenHands SWE trajectories.

At a sampled step of a trajectory that resolved its GitHub issue, the system sees the same text
transcript (truncated system prompt, the issue, the last K turns) and must choose the next action.
The action class is the tool plus, for tools with a `command` enum, the command, e.g.
`str_replace_editor:view`. Arguments (code, shell commands) are long-form and stay with the LLM, so
this measures the part a decision model can take over: which action comes next.

Gold is what the original agent (Qwen3-Coder-480B in OpenHands) did in a trajectory that ended up
resolving the issue. Several next actions are often valid, so report this as agreement, not accuracy.

    python -m dmtc.swe.replay fetch --n-traj 60
    python -m dmtc.swe.replay run configs/swe_replay.yaml
    python -m dmtc.swe.replay analyze results/swe_replay_v1
"""

from __future__ import annotations

import argparse
import json
import random
import threading
import time
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from tqdm import tqdm

from dmtc.backends.dm import Choice, make_dm
from dmtc.backends.llm import OutOfCredits, make_llm
from dmtc.eval.metrics import bootstrap_ci, macro_f1, mcnemar

DATASET = "nebius/SWE-rebench-openhands-trajectories"
ROWS_API = "https://datasets-server.huggingface.co/rows?dataset={ds}&config=default&split=train&offset={off}&length={n}"
TOTAL_ROWS = 67074
DATA = Path("data/swe")

ACTION_DESCRIPTIONS = {
    "execute_bash": "Run a shell command: explore the repo (ls, grep, find), run tests or scripts, check git state.",
    "str_replace_editor:view": "Open a file or directory listing to read code.",
    "str_replace_editor:create": "Create a new file (e.g. a reproduction script or a new test).",
    "str_replace_editor:str_replace": "Edit an existing file by replacing an exact snippet with new code.",
    "str_replace_editor:insert": "Insert new lines into an existing file after a given line number.",
    "str_replace_editor:undo_edit": "Revert the last edit made to a file.",
    "think": "Reason privately about the problem, a plan, or a hypothesis without acting.",
    "task_tracker:plan": "Write or update the task plan / checklist.",
    "task_tracker:view": "Look at the current task plan.",
    "finish": "Stop: the issue is resolved and verified, so end the interaction.",
}
ACTIONS = list(ACTION_DESCRIPTIONS)


# ---- data -----------------------------------------------------------------------------------
def _get(url: str, tries: int = 5) -> dict:
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                return json.load(r)
        except Exception:  # noqa: BLE001 - transient HTTP errors from datasets-server
            if i == tries - 1:
                raise
            time.sleep(2 ** i)
    raise RuntimeError("unreachable")


def fetch(n_traj: int, seed: int, page: int = 5) -> Path:
    """Samples resolved trajectories at random offsets, at most one per repo."""
    DATA.mkdir(parents=True, exist_ok=True)
    out = DATA / f"resolved_sample_seed{seed}.jsonl"
    rng = random.Random(seed)
    offsets = list(range(0, TOTAL_ROWS - page, page))
    rng.shuffle(offsets)
    keep, repos = [], set()
    tools = None
    with tqdm(total=n_traj, desc="fetch resolved trajectories") as bar:
        for off in offsets:
            for row in _get(ROWS_API.format(ds=DATASET, off=off, n=page))["rows"]:
                r = row["row"]
                if r["resolved"] != 1 or r["repo"] in repos:
                    continue
                repos.add(r["repo"])
                tools = tools or r["tools"]
                keep.append({k: r[k] for k in ("trajectory_id", "instance_id", "repo", "trajectory")})
                bar.update(1)
                if len(keep) >= n_traj:
                    break
            if len(keep) >= n_traj:
                break
    with open(out, "w") as f:
        for r in keep:
            f.write(json.dumps(r) + "\n")
    (DATA / "tools.json").write_text(json.dumps(tools))
    print(f"wrote {len(keep)} trajectories from {len(repos)} repos to {out}")
    return out


def action_of(tool_call: dict) -> str:
    fn = tool_call["function"]
    args = fn.get("arguments") or {}
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {}
    name = fn["name"]
    if name in ("str_replace_editor", "task_tracker") and isinstance(args, dict) and args.get("command"):
        return f"{name}:{args['command']}"
    return name


def _clip(s: Any, n: int) -> str:
    s = s if isinstance(s, str) else json.dumps(s)
    return s if len(s) <= n else s[: n // 2] + f"\n...[{len(s) - n} chars cut]...\n" + s[-n // 2:]


def render(traj: list[dict], upto: int, k_last: int, cfg: dict) -> str:
    """Text transcript of traj[:upto]: system prompt, issue, and the last k_last messages."""
    sys_msg = next((m for m in traj if m["role"] == "system"), None)
    first_user = next((m for m in traj if m["role"] == "user"), None)
    parts = []
    if sys_msg:
        parts.append("## Agent instructions (truncated)\n" + _clip(sys_msg["content"], cfg["system_chars"]))
    if first_user:
        parts.append("## Task\n" + _clip(first_user["content"], cfg["issue_chars"]))
    hist = [m for m in traj[:upto] if m["role"] in ("assistant", "tool", "user")]
    if first_user in hist:
        hist.remove(first_user)
    recent = hist[-k_last:]
    if len(hist) > len(recent):
        parts.append(f"## Recent history ({len(hist) - len(recent)} earlier messages omitted)")
    else:
        parts.append("## History")
    for m in recent:
        if m["role"] == "assistant":
            calls = "; ".join(f"{c['function']['name']}({_clip(c['function'].get('arguments') or '', cfg['msg_chars'] // 2)})"
                              for c in (m.get("tool_calls") or []))
            text = _clip(m.get("content") or "", cfg["msg_chars"] // 2)
            parts.append(f"[assistant] {text}\n[assistant called] {calls or '(no tool)'}")
        elif m["role"] == "tool":
            parts.append(f"[tool output: {m.get('name')}]\n{_clip(m.get('content') or '', cfg['msg_chars'])}")
        else:
            parts.append(f"[user]\n{_clip(m.get('content') or '', cfg['msg_chars'])}")
    return "\n\n".join(parts)


def build_steps(path: Path, steps_per_traj: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    steps = []
    for line in open(path):
        r = json.loads(line)
        traj = r["trajectory"]
        idx = [i for i, m in enumerate(traj) if m["role"] == "assistant" and m.get("tool_calls")]
        for i in sorted(rng.sample(idx, min(steps_per_traj, len(idx)))):
            prev = [action_of(m["tool_calls"][0]) for m in traj[:i] if m["role"] == "assistant" and m.get("tool_calls")]
            steps.append({"id": f"{r['trajectory_id']}@{i}", "repo": r["repo"], "traj": traj, "pos": i,
                          "rel_pos": idx.index(i) / max(len(idx) - 1, 1),
                          "gold": action_of(traj[i]["tool_calls"][0]), "prev": prev[-1] if prev else None})
    return steps


# ---- conditions -----------------------------------------------------------------------------
def run(cfg_path: str, limit: int | None) -> None:
    cfg = yaml.safe_load(open(cfg_path))
    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    steps = build_steps(Path(cfg["trajectories"]), cfg["steps_per_traj"], cfg["seed"])[:limit]
    tools = json.loads((DATA / "tools.json").read_text())
    tool_docs = [t.get("function", t) for t in tools]
    ctx = cfg["context"]
    budget, spend, lock = cfg.get("budget_usd"), {"usd": 0.0}, threading.Lock()
    for p in out.glob("*.jsonl"):
        for l in open(p):
            r = json.loads(l)
            spend["usd"] += r.get("cost_usd", 0)
    instruction = ("You are the software-engineering agent described above, resuming the task. "
                   "Based on the transcript, take the single next action by calling exactly one tool.")

    stop = {"why": None}

    def work(cond: str, name: str, st: dict) -> dict | None:
        try:
            return _work(cond, name, st)
        except OutOfCredits as e:
            stop["why"] = str(e)
            return None

    def _work(cond: str, name: str, st: dict) -> dict | None:
        if stop["why"] or (budget is not None and spend["usd"] >= budget):
            return None
        text = render(st["traj"], st["pos"], ctx["k_last"], ctx)
        row = {"id": st["id"], "repo": st["repo"], "condition": cond, "model": name, "gold": st["gold"],
               "prev": st["prev"], "rel_pos": st["rel_pos"]}
        if cond == "llm_native":
            r = llms[name].call([{"role": "user", "content": text + "\n\n" + instruction}], tool_docs)
            pred = action_of({"function": {"name": next(iter(r.calls[0])), "arguments": next(iter(r.calls[0].values()))}}) \
                if r.calls else "no_tool"
            row.update(pred=pred, latency_s=r.latency_s, cost_usd=r.cost_usd, in_tok=r.input_tokens,
                       out_tok=r.output_tokens, error=r.error, probs=None)
        else:
            r = dms[name].ask({"transcript": text},
                              {"next": Choice("What should the agent do next?", ACTION_DESCRIPTIONS)})
            a = r.answers.get("next")
            row.update(pred=a.value if a else "error", probs=a.probabilities if a else None,
                       confidence=a.confidence if a else None, latency_s=r.latency_s, cost_usd=r.cost_usd,
                       in_tok=r.input_tokens, out_tok=0, error=r.error)
        with lock:
            spend["usd"] += row["cost_usd"] or 0
        row["correct"] = row["pred"] == row["gold"]
        return row

    llms = {n: make_llm(cfg["llms"][n]) for n in cfg.get("run_llms", [])}
    dms = {n: make_dm(cfg["dms"][n]) for n in cfg.get("run_dms", [])}
    tasks = []
    files = {}
    for cond, names in (("llm_native", llms), ("dm_choice", dms)):
        for n in names:
            path = out / f"{cond}__{n}.jsonl"
            done = {json.loads(l)["id"] for l in open(path)} if path.exists() else set()
            files[(cond, n)] = open(path, "a")
            tasks += [(cond, n, s) for s in steps if s["id"] not in done]
    skipped = 0
    with ThreadPoolExecutor(cfg.get("concurrency", 8)) as pool:
        futs = {pool.submit(work, *t): t[:2] for t in tasks}
        bar = tqdm(as_completed(futs), total=len(futs), desc="swe replay")
        for fut in bar:
            row = fut.result()
            if row is None:
                skipped += 1
                continue
            with lock:
                files[futs[fut]].write(json.dumps(row) + "\n")
                files[futs[fut]].flush()
            bar.set_postfix(spend=f"${spend['usd']:.2f}")
    for f in files.values():
        f.close()
    # Baselines need no model: always the most common action, and repeat the previous action.
    with open(out / "baselines.jsonl", "w") as f:
        for s in steps:
            for cond, pred in (("always_bash", "execute_bash"), ("repeat_prev", s["prev"] or "execute_bash")):
                f.write(json.dumps({"id": s["id"], "condition": cond, "model": "-", "gold": s["gold"], "pred": pred,
                                    "correct": pred == s["gold"], "cost_usd": 0, "latency_s": 0,
                                    "rel_pos": s["rel_pos"]}) + "\n")
    print(f"spend: ${spend['usd']:.4f}" + (f"; {skipped} tasks skipped" if skipped else "")
          + (f"; STOPPED: {stop['why']}" if stop["why"] else ""))


def analyze(d: str) -> None:
    d = Path(d)
    rows = defaultdict(list)
    for p in d.glob("*.jsonl"):
        for l in open(p):
            r = json.loads(l)
            rows[(r["condition"], r["model"])].append(r)
    gold_dist = Counter(r["gold"] for r in next(iter(rows.values())))
    n = sum(gold_dist.values())
    L = [f"# SWE next-action replay: {d.name}", "", f"{n} steps. Gold action mix: " +
         ", ".join(f"{k} {v / n:.1%}" for k, v in gold_dist.most_common()), "",
         "| condition | model | n | agreement | 95% CI | macro-F1 | finish P / R | top-2 (DM) | p50 latency s | cost per 1k steps $ | errors |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for (cond, m), rs in sorted(rows.items()):
        ok = [r["correct"] for r in rs]
        lo, hi = bootstrap_ci(ok)
        tp = sum(r["pred"] == "finish" and r["gold"] == "finish" for r in rs)
        pp = sum(r["pred"] == "finish" for r in rs)
        gp = sum(r["gold"] == "finish" for r in rs)
        top2 = "–"
        if cond == "dm_choice":
            t2 = [r["gold"] in sorted(r["probs"] or {}, key=(r["probs"] or {}).get, reverse=True)[:2] for r in rs]
            top2 = f"{np.mean(t2):.3f}"
        L.append(f"| {cond} | {m} | {len(rs)} | {np.mean(ok):.3f} | {lo:.3f}–{hi:.3f} | "
                 f"{macro_f1([r['gold'] for r in rs], [r['pred'] for r in rs], ACTIONS):.3f} | "
                 f"{(tp / pp if pp else float('nan')):.2f} / {(tp / gp if gp else float('nan')):.2f} | {top2} | "
                 f"{np.median([r['latency_s'] for r in rs]):.2f} | {1000 * np.mean([r['cost_usd'] or 0 for r in rs]):.3f} | "
                 f"{sum(bool(r.get('error')) for r in rs)} |")
    # paired: each DM vs each LLM
    L += ["", "## Paired (McNemar) DM vs LLM", "", "| DM | LLM | n | DM agr. | LLM agr. | p |", "|---|---|---|---|---|---|"]
    for (c1, m1), a in rows.items():
        for (c2, m2), b in rows.items():
            if c1 == "dm_choice" and c2 == "llm_native":
                bi = {r["id"]: r["correct"] for r in b}
                ids = [r["id"] for r in a if r["id"] in bi]
                ai = {r["id"]: r["correct"] for r in a}
                t = mcnemar([bi[i] for i in ids], [ai[i] for i in ids])
                L.append(f"| {m1} | {m2} | {len(ids)} | {np.mean([ai[i] for i in ids]):.3f} | "
                         f"{np.mean([bi[i] for i in ids]):.3f} | {t['p_value']:.3g} |")
    # confusion of the most common errors per system
    L += ["", "## Most common disagreements (gold → predicted)", ""]
    for (cond, m), rs in sorted(rows.items()):
        if cond in ("always_bash",):
            continue
        conf = Counter((r["gold"], r["pred"]) for r in rs if not r["correct"]).most_common(4)
        L.append(f"- **{cond}/{m}**: " + "; ".join(f"{g} → {p} ({c})" for (g, p), c in conf))
    (d / "REPORT.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--n-traj", type=int, default=60)
    f.add_argument("--seed", type=int, default=0)
    r = sub.add_parser("run")
    r.add_argument("config")
    r.add_argument("--limit", type=int)
    a = sub.add_parser("analyze")
    a.add_argument("results_dir")
    args = ap.parse_args()
    if args.cmd == "fetch":
        fetch(args.n_traj, args.seed)
    elif args.cmd == "run":
        run(args.config, args.limit)
    else:
        analyze(args.results_dir)


if __name__ == "__main__":
    main()
