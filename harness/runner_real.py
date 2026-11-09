"""Runs the realistic tasks (harness/tasks_real) through opencode under each tool-selection mechanism and scores them.

Conditions: none, native, cc, bm25, groups, embed, llm, dm (see mcp_server_real.py). The sandbox is the one
described in runner.py: the agent runs as `agentbox` with no network except the LLM proxy and the MCP server,
no credentials, and a shell limited to pytest -- so the services are reachable only through the catalog.

A run passes when (1) the hidden tests pass, (2) every required write happened with the right arguments,
and (3) no destructive call was made (merge, delete, kubectl apply, ...).

    source .or_env && .venv/bin/python harness/runner_real.py --models deepseek,hy4 --repeats 2 --workers 6 \\
        --budget 25 --out results/real_v1
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))
import runner as R  # noqa: E402
from real_sim import _matches, is_destructive, is_write  # noqa: E402

MODELS = {"deepseek": "deepseek/deepseek-v4.1-flash-1017", "hy4": "tencent/hy4", "kimi": "moonshotai/kimi-k3.1-pro", "sol": "openai/gpt-6.1-sol", "sonnet": "anthropic/claude-sonnet-5.5", "haiku": "anthropic/claude-haiku-5.5"}
# USD per token (input, cache read, output) for models opencode has no price for (OpenRouter list prices)
PRICES = {"hy4": (0.417e-6, 0.021e-6, 1.2505e-6)}
CONDITIONS = ["none", "native", "cc", "bm25", "groups", "embed", "llm", "dm", "random"]
PROXY_PORT, MCP_PORT_BASE = 18080, 18190


def probe_outcome(proj: Path, probe: dict | None) -> dict:
    if not probe:
        return {}
    p = dict(probe)
    p["expected"] = p["expected"] if p["expected"] is not None else object()
    out = R.probe_outcome(proj, p)
    try:
        src = (proj / probe["file"]).read_text()
        out["todo"] = "TODO: info not found" in src or "TODO: value not found" in src
        if out.get("outcome") == "unchanged" and out["todo"]:
            out["outcome"] = "abstained"
    except Exception:  # noqa: BLE001
        pass
    return out


def run_one(task_root: Path, task: str, cond: str, mkey: str, rep: int, port: int, out: Path, timeout: int, servers: str) -> dict:
    model = MODELS[mkey]
    task_dir = task_root / task
    meta = json.loads((task_dir / "task.json").read_text())
    tag = f"{task}__{cond}__{mkey}__r{rep}"
    rec = out / "runs" / tag
    if rec.exists():
        shutil.rmtree(rec)
    rec.mkdir(parents=True)
    work = R.WORK / tag
    if work.exists():
        shutil.rmtree(work)
    (work / "xdg").mkdir(parents=True)
    shutil.copytree(task_dir / "repo", work / "proj")
    for sub in ("config", "cache"):
        shutil.copytree(R.XDG_TEMPLATE / sub, work / "xdg" / sub)
    (work / "xdg" / "data").mkdir()
    (work / "xdg" / "state").mkdir()
    cfg = R.config(cond, model, port)
    cfg["provider"]["openrouter"]["options"]["baseURL"] = f"http://127.0.0.1:{PROXY_PORT}/api/v1"
    (work / "xdg" / "config" / "opencode" / "opencode.json").write_text(json.dumps(cfg, indent=1))
    subprocess.run(["chown", "-R", "agentbox:agentbox", str(work)], check=True)
    work.chmod(0o700)

    mcp = None
    if cond != "none":
        env = {**os.environ, "REAL_MODE": cond, "REAL_WORLD": str(task_dir / "world.json"), "REAL_LOG": str(rec / "calls.jsonl"),
               "REAL_HTTP_PORT": str(port), "REAL_SERVERS": servers, "REAL_EXTRA": os.environ.get("REAL_EXTRA", "0"), "PYTHONPATH": f"{HERE}:{REPO}"}
        mcp = subprocess.Popen([str(R.PY), str(HERE / "mcp_server_real.py")], env=env, stdout=subprocess.DEVNULL, stderr=open(rec / "mcp_stderr.txt", "w"))
        R.wait_port(port, 60)

    t0 = time.time()
    try:
        r = R.box([R.BOX_OPENCODE, "run", meta["prompt"]], work, timeout, capture_output=True, text=True)
        rc, timed_out, transcript = r.returncode, False, r.stdout + "\n--- stderr ---\n" + r.stderr
    except subprocess.TimeoutExpired as e:
        rc, timed_out = None, True
        transcript = (e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or "")) + "\n[TIMEOUT]"
        subprocess.run(["pkill", "-u", "agentbox", "-f", str(work)], check=False)
    wall = time.time() - t0
    if mcp:
        mcp.terminate()
        mcp.wait(timeout=10)
    (rec / "stdout.txt").write_text(transcript)
    stats = R.session_stats(work, rec / "export.json")
    log = "".join(p.read_text() for p in (work / "xdg" / "data" / "opencode" / "log").glob("*.log"))
    stats["ran_in_task_dir"] = f"directory={work / 'proj'}" in log

    shutil.copytree(work / "proj", rec / "proj_final", ignore=shutil.ignore_patterns(".opencode", "__pycache__", ".pytest_cache"))
    score = rec / "score"
    shutil.copytree(rec / "proj_final", score)
    (score / "tests").mkdir(exist_ok=True)
    shutil.copy(task_dir / "hidden" / "test_hidden.py", score / "tests" / "test_hidden.py")
    h = subprocess.run([str(R.PY), "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests/test_hidden.py"], cwd=score,
                       capture_output=True, text=True, env={**os.environ, "TZ": "UTC"}, timeout=120)
    shutil.rmtree(work)

    ev = [json.loads(l) for l in open(rec / "calls.jsonl")] if (rec / "calls.jsonl").exists() else []
    calls = [e for e in ev if e["kind"] == "call"]
    searches = [e for e in ev if e["kind"] == "search"]
    used_ok = {c["tool"] for c in calls if c.get("ok")}
    needed = {n for g in meta["needs"] for n in g}
    writes_done = [all_ok for all_ok in (any(c["tool"] in (w["tool"] if isinstance(w["tool"], list) else [w["tool"]]) and _matches(w, c["args"])
                                             for c in calls) for w in meta["writes"])]
    destructive = [c["tool"] for c in calls if is_destructive(c["tool"])]
    tests_ok = h.returncode == 0
    writes_ok = all(writes_done)
    tool_calls = stats.pop("tool_calls", [])
    if not stats.get("oc_cost") and mkey in PRICES:
        pi, pc, po = PRICES[mkey]
        stats["oc_cost"] = stats.get("input", 0) * pi + stats.get("cache_read", 0) * pc + (stats.get("output", 0) + stats.get("reasoning", 0)) * po
        stats["cost_estimated"] = True
    return {"task": task, "category": meta["category"], "condition": cond, "model": mkey, "rep": rep,
            "passed": tests_ok and writes_ok and not destructive, "tests_ok": tests_ok, "writes_ok": writes_ok,
            "writes_done": writes_done, "destructive": destructive, "hidden_tail": h.stdout.strip().splitlines()[-1:],
            "needs_met": all(any(n in used_ok for n in g) for g in meta["needs"]),
            "needs_steps_met": [any(n in used_ok for n in g) for g in meta["needs"]],
            **probe_outcome(rec / "proj_final", meta.get("probe")),
            "n_calls": len(calls), "n_searches": len(searches), "n_activations": len([e for e in ev if e["kind"] == "activate"]),
            "n_failed_calls": len([c for c in calls if not c.get("ok")]), "n_unknown_tool": len([c for c in calls if c.get("unknown")]),
            "n_unneeded_calls": len([c for c in calls if c["tool"] not in needed]),
            "n_writes": len([c for c in calls if is_write(c["tool"])]),
            "search_hit": [bool(needed & set(s["ranked"])) for s in searches], "tools_used": sorted(used_ok),
            "selector_usd": sum(e.get("cost") or 0 for e in searches), "wall_s": round(wall, 1), "timed_out": timed_out, "rc": rc,
            **stats, "builtin_tools_used": dict(Counter(t for t in tool_calls if not str(t).startswith("corp_"))),
            "outside_paths": R.outside_paths(transcript, work / "proj")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks-dir", default=str(HERE / "tasks_real"))
    ap.add_argument("--tasks", default="")
    ap.add_argument("--conditions", default=",".join(CONDITIONS))
    ap.add_argument("--models", default="deepseek")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--rep-offset", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=str(REPO / "results" / "real_v1"))
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--budget", type=float, default=5.0)
    ap.add_argument("--servers", default="", help="comma-separated subset of MCP servers (default: all)")
    ap.add_argument("--mcp-port-base", type=int, default=MCP_PORT_BASE, help="change to run two batches side by side")
    a = ap.parse_args()
    task_root, out = Path(a.tasks_dir).resolve(), Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    tasks = a.tasks.split(",") if a.tasks else sorted(p.name for p in task_root.iterdir() if (p / "task.json").exists())
    done = set()
    if (out / "rows.jsonl").exists():
        done = {(r["task"], r["condition"], r["model"], r["rep"]) for r in map(json.loads, open(out / "rows.jsonl"))}
    jobs = [(t, c, m, r) for r in range(a.rep_offset, a.rep_offset + a.repeats) for m in a.models.split(",") for t in tasks
            for c in a.conditions.split(",") if (t, c, m, r) not in done]
    random.Random(0).shuffle(jobs)
    u0 = R.key_usage()
    print(f"key usage before: ${u0:.4f}; {len(jobs)} runs", flush=True)
    import socket
    with socket.socket() as so:
        proxy_up = so.connect_ex(("127.0.0.1", PROXY_PORT)) == 0  # another batch's proxy is already serving
    proxy = None if proxy_up else subprocess.Popen([str(R.PY), str(HERE / "llm_proxy.py"), "--port", str(PROXY_PORT),
                              "--models", ",".join(MODELS[m] for m in a.models.split(",")), "--log", str(out / "proxy_log.jsonl")],
                             stdout=subprocess.DEVNULL, stderr=open(out / "proxy_stderr.txt", "a"))
    lock = threading.Lock()
    state = {"spent": 0.0, "n": 0, "stop": False}
    ports = list(range(a.mcp_port_base, a.mcp_port_base + a.workers))

    def worker(job):
        if state["stop"]:
            return
        with lock:
            port = ports.pop()
        try:
            row = run_one(task_root, *job, port, out, a.timeout, a.servers)
        except Exception as e:  # noqa: BLE001
            print(f"ERROR {job}: {type(e).__name__}: {e}", flush=True)
            return
        finally:
            with lock:
                ports.append(port)
        with lock:
            state["spent"] += row.get("oc_cost", 0) + row["selector_usd"]
            state["n"] += 1
            if state["spent"] >= a.budget:
                state["stop"] = True
            with open(out / "rows.jsonl", "a") as f:
                f.write(json.dumps(row) + "\n")
            flag = "" if row["ran_in_task_dir"] and not row["outside_paths"] else f"  !! {row['outside_paths'][:3]}"
            print(f"[{state['n']}/{len(jobs)} ~${state['spent']:.2f}] {row['task']:17s} {row['condition']:7s} {row['model']:8s} r{row['rep']} "
                  f"pass={row['passed']!s:5s} tests={row['tests_ok']!s:5s} writes={row['writes_ok']!s:5s} {row.get('outcome', '')} "
                  f"calls={row['n_calls']} search={row['n_searches']} steps={row.get('steps')} prompt_max={row.get('max_prompt')} "
                  f"${row.get('oc_cost', 0):.4f}{flag}", flush=True)

    try:
        R.wait_port(PROXY_PORT)
        with ThreadPoolExecutor(a.workers) as pool:
            list(pool.map(worker, jobs))
        if state["stop"]:
            print(f"budget {a.budget} reached; stopped early", flush=True)
    finally:
        if proxy:
            proxy.terminate()
        time.sleep(20)
        u1 = R.key_usage()
        print(f"key usage after: ${u1:.4f} (batch ${u1 - u0:.4f})", flush=True)


if __name__ == "__main__":
    main()
