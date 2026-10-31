"""Runs coding tasks through the opencode harness under different tool-catalog conditions and scores them.

Conditions:
  none            opencode's built-in tools only (shows which tasks need the catalog)
  native          built-ins + all 100 catalog tools (every definition in every request)
  filtered        built-ins + the DM's top-k catalog tools for this task (chosen once, from the task prompt)
  universal       built-ins + find_tool/call_tool (DM resolves intents at run time); no hint of what exists
  universal_index the same, plus a list of the services the catalog covers in find_tool's description

Sandbox (see sandbox_setup.sh): the agent runs as the unprivileged user `agentbox` in /srv/agentbox/work with an
empty environment. It cannot read this repository (answers, world files) or the API key. The LLM is reached
through llm_proxy.py (root, adds the key, allows the experiment's models only); the catalog through
mcp_server.py over local HTTP (root). opencode permissions additionally limit the shell to running pytest and
deny web access.

Recorded per run, besides pass/fail and tokens: for value tasks the outcome class (correct / abstained /
unchanged / wrong value), and false-positive signals -- unnecessary catalog calls, state-changing calls,
failed calls, calls to tool names that do not exist, DM retrieval misses, and passes without the needed tool.

    source .or_env && .venv/bin/python harness/runner.py --tasks-dir harness/tasks_v2 --models deepseek,glm \\
        --repeats 3 --workers 6 --budget 10 --out results/harness_v2
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import random
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))
from catalog import catalog  # noqa: E402

PY = REPO / ".venv" / "bin" / "python"
BOX_OPENCODE = "/opt/agentbox/opencode/node_modules/.bin/opencode"
BOX_PATH = "/opt/agentbox/py/bin:/usr/bin:/bin"
BOX_HOME = "/srv/agentbox/home"
XDG_TEMPLATE = Path("/opt/agentbox/xdg-template")
WORK = Path("/srv/agentbox/work")
MODELS = {"deepseek": "deepseek/deepseek-v4.1-flash-1017", "glm": "z-ai/glm-5.3"}
DM = "microsoft/microsoft-decision-1"
PROXY_PORT, MCP_PORT_BASE = 18080, 18090
CONDITIONS = ["none", "native", "filtered", "universal", "universal_index"]
CAT_NAMES = {t["name"] for t in catalog(300)}
WRITE_VERBS = ("create", "update", "add", "assign", "retry", "cancel", "set", "archive", "silence", "rollback", "promote",
               "post", "swap", "page", "publish", "link")


def key_usage() -> float:
    r = httpx.get("https://openrouter.ai/api/v1/key", headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"}, timeout=30)
    return float(r.json()["data"]["usage"])


def wait_port(port: int, timeout: float = 30) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.2)
    raise RuntimeError(f"nothing listening on {port}")


def dm_filter(prompt: str, k: int) -> tuple[list[str], float]:
    from dmtc.backends.dm import Choice, OpenRouterDM
    from mcp_server import short_card

    cat = {t["name"]: t for t in catalog()}
    r = OpenRouterDM(DM).ask({"task": prompt}, {"tools": Choice(
        "Which catalog tools will a coding agent need to complete this task?", {n: short_card(t) for n, t in cat.items()})})
    probs = r.answers["tools"].probabilities
    return sorted(probs, key=probs.get, reverse=True)[:k], r.cost_usd


def config(cond: str, model: str, mcp_port: int) -> dict:
    perm = {"*": "deny", "read": "allow", "edit": "allow", "glob": "allow", "grep": "allow", "list": "allow",
            "todowrite": "allow", "todoread": "allow", "corp_*": "allow",
            "bash": {"*": "deny", "python -m pytest*": "allow", "python3 -m pytest*": "allow", "pytest*": "allow"},
            "webfetch": "deny", "websearch": "deny", "external_directory": "deny"}
    cfg = {"$schema": "https://opencode.ai/config.json",
           "provider": {"openrouter": {"options": {"apiKey": "sandbox-no-key", "baseURL": f"http://127.0.0.1:{PROXY_PORT}/api/v1"},
                                       "models": {model: {"name": model}}}},
           "model": f"openrouter/{model}", "small_model": f"openrouter/{model}",
           "permission": perm, "autoupdate": False, "share": "disabled"}
    if cond != "none":
        cfg["mcp"] = {"corp": {"type": "remote", "url": f"http://127.0.0.1:{mcp_port}/mcp", "enabled": True}}
    return cfg


def box(args: list[str], work: Path, timeout: int, **kw):
    """Run a command as the sandbox user with an empty environment (no keys, no proxy settings)."""
    proj = work / "proj"
    env = ["HOME=" + BOX_HOME, "PATH=" + BOX_PATH, f"PWD={proj}", "TZ=UTC", "OPENCODE_DISABLE_AUTOUPDATE=1",
           *(f"XDG_{n}_HOME={work / 'xdg' / n.lower()}" for n in ("CONFIG", "DATA", "CACHE", "STATE"))]
    return subprocess.run(["runuser", "-u", "agentbox", "--", "env", "-i", *env, *args], cwd=proj, timeout=timeout, **kw)


def session_stats(work: Path, dump: Path) -> dict:
    out = box([BOX_OPENCODE, "session", "list", "--format", "json"], work, 60, capture_output=True, text=True)
    try:
        sid = json.loads(out.stdout)[0]["id"]
    except Exception:  # noqa: BLE001
        return {"session": None}
    with open(dump, "w") as fh:
        box([BOX_OPENCODE, "export", sid], work, 120, stdout=fh, stderr=subprocess.DEVNULL)
    raw = dump.read_text()
    data = json.loads(raw[raw.index("{"):])
    st = {"session": sid, "steps": 0, "input": 0, "output": 0, "reasoning": 0, "cache_read": 0, "cache_write": 0, "oc_cost": 0.0,
          "max_prompt": 0, "tool_calls": [], "final_text": ""}
    for m in data.get("messages", []):
        i = m.get("info", {})
        if i.get("role") != "assistant":
            continue
        t = i.get("tokens") or {}
        st["steps"] += 1
        st["input"] += t.get("input", 0)
        st["output"] += t.get("output", 0)
        st["reasoning"] += t.get("reasoning", 0)
        st["cache_read"] += (t.get("cache") or {}).get("read", 0)
        st["cache_write"] += (t.get("cache") or {}).get("write", 0)
        st["oc_cost"] += i.get("cost") or 0
        st["max_prompt"] = max(st["max_prompt"], t.get("input", 0) + (t.get("cache") or {}).get("read", 0))
        for p in m.get("parts", []):
            if p.get("type") == "tool":
                st["tool_calls"].append(p.get("tool"))
            elif p.get("type") == "text" and p.get("text", "").strip():
                st["final_text"] = p["text"][-600:]
    return st


def outside_paths(text: str, proj: Path) -> list[str]:
    """Absolute paths outside the project that a tool call reached successfully (refused attempts are not counted)."""
    lines = [l for l in re.sub(r"\x1b\[[0-9;]*m", "", text).splitlines() if "failed" not in l and "Error" not in l]
    paths = set(re.findall(r"(/(?:home|root|tmp|srv|opt|etc|proc)[^\s'\"`]*)", "\n".join(lines)))
    return sorted(p for p in paths if not p.startswith(str(proj)) and not p.startswith(str(proj.parent / "xdg")))


def probe_outcome(proj: Path, probe: dict | None) -> dict:
    """Classify a value task: correct / abstained (unchanged + TODO) / unchanged / wrong / unreadable."""
    if not probe:
        return {}
    f = proj / probe["file"]
    try:
        src = f.read_text()
        val = None
        for node in ast.parse(src).body:
            if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == probe["name"] for t in node.targets):
                val = ast.literal_eval(node.value)
        norm = lambda v: list(v) if isinstance(v, (tuple, list)) else v  # noqa: E731
        todo = "TODO: value not found" in src
        if norm(val) == norm(probe["expected"]) and norm(val) != norm(probe["original"]):
            cls = "correct"
        elif norm(val) == norm(probe["original"]):
            cls = "abstained" if todo else "unchanged"
        else:
            cls = "wrong"
        return {"final_value": norm(val), "todo": todo, "outcome": cls}
    except Exception as e:  # noqa: BLE001
        return {"final_value": None, "todo": None, "outcome": f"unreadable: {type(e).__name__}"}


def false_positive_signals(calls: list[dict], meta: dict) -> dict:
    needed = {n for g in meta["needs"] for n in g}
    real = [c for c in calls if c.get("kind") == "call"]
    finds = [c for c in calls if c.get("kind") == "find_tool"]
    return {
        "fp_unneeded_calls": len(real) if meta["category"] == "control" else len([c for c in real if c["tool"] not in needed and c.get("ok")]),
        "fp_mutating_calls": len([c for c in real if c["tool"].split("_", 1)[-1].startswith(WRITE_VERBS)]),
        "fp_failed_calls": len([c for c in real if not c.get("ok")]),
        "fp_unknown_tool": len([c for c in real if c["tool"] not in CAT_NAMES]),
        "find_hits": len([f for f in finds if needed & set(f.get("ranked", []))]),
        "find_irrelevant": len([f for f in finds if not needed & set(f.get("ranked", []))]),
    }


def run_one(task_root: Path, task: str, cond: str, mkey: str, rep: int, port: int, out: Path, timeout: int, k: int, size: int = 100) -> dict:
    model = MODELS[mkey]
    task_dir = task_root / task
    meta = json.loads((task_dir / "task.json").read_text())
    meta.setdefault("category", "single")
    tag = f"{task}__{cond}__{mkey}__r{rep}"
    rec = out / "runs" / tag  # root-only record of the run (inside this repo)
    if rec.exists():
        shutil.rmtree(rec)
    rec.mkdir(parents=True)
    work = WORK / tag  # the sandbox user's working copy
    if work.exists():
        shutil.rmtree(work)
    (work / "xdg").mkdir(parents=True)
    shutil.copytree(task_dir / "repo", work / "proj")
    for sub in ("config", "cache"):
        shutil.copytree(XDG_TEMPLATE / sub, work / "xdg" / sub)
    (work / "xdg" / "data").mkdir()
    (work / "xdg" / "state").mkdir()
    (work / "xdg" / "config" / "opencode" / "opencode.json").write_text(json.dumps(config(cond, model, port), indent=1))
    subprocess.run(["chown", "-R", "agentbox:agentbox", str(work)], check=True)
    work.chmod(0o700)

    filt, filt_cost, mcp = [], 0.0, None
    if cond == "filtered":
        filt, filt_cost = dm_filter(meta["prompt"], k)
    if cond != "none":
        env = {**os.environ, "CATALOG_MODE": {"native": "native", "filtered": "filtered"}.get(cond, "universal"),
               "CATALOG_WORLD": str(task_dir / "world.json"), "CATALOG_LOG": str(rec / "catalog_calls.jsonl"),
               "CATALOG_DM": DM, "CATALOG_FILTER": ",".join(filt), "CATALOG_INDEX": "1" if cond == "universal_index" else "0",
               "CATALOG_HTTP_PORT": str(port), "CATALOG_SIZE": str(size), "PYTHONPATH": f"{HERE}:{REPO}"}
        mcp = subprocess.Popen([str(PY), str(HERE / "mcp_server.py")], env=env, stdout=subprocess.DEVNULL, stderr=open(rec / "mcp_stderr.txt", "w"))
        wait_port(port)

    t0 = time.time()
    try:
        r = box([BOX_OPENCODE, "run", meta["prompt"]], work, timeout, capture_output=True, text=True)
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
    stats = session_stats(work, rec / "export.json")
    log = "".join(p.read_text() for p in (work / "xdg" / "data" / "opencode" / "log").glob("*.log"))
    stats["ran_in_task_dir"] = f"directory={work / 'proj'}" in log

    shutil.copytree(work / "proj", rec / "proj_final", ignore=shutil.ignore_patterns(".opencode", "__pycache__", ".pytest_cache"))
    score = rec / "score"
    shutil.copytree(rec / "proj_final", score)
    shutil.copy(task_dir / "hidden" / "test_hidden.py", score / "tests" / "test_hidden.py")
    h = subprocess.run([str(PY), "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests/test_hidden.py"], cwd=score,
                       capture_output=True, text=True, env={**os.environ, "TZ": "UTC"}, timeout=120)
    shutil.rmtree(work)

    calls = [json.loads(l) for l in open(rec / "catalog_calls.jsonl")] if (rec / "catalog_calls.jsonl").exists() else []
    used_ok = {c["tool"] for c in calls if c.get("kind") == "call" and c.get("ok")}
    tool_calls = stats.pop("tool_calls", [])
    needs_met = all(any(n in used_ok for n in g) for g in meta["needs"])
    passed = h.returncode == 0
    return {"task": task, "category": meta["category"], "condition": cond, "model": mkey, "rep": rep, "catalog_size": size, "passed": passed,
            "hidden_tail": h.stdout.strip().splitlines()[-1:], "needs": meta["needs"], "needs_met": needs_met,
            "lucky_pass": passed and bool(meta["needs"]) and not needs_met,
            **probe_outcome(rec / "proj_final", meta.get("probe")),
            "catalog_calls": len([c for c in calls if c.get("kind") == "call"]),
            "find_calls": len([c for c in calls if c.get("kind") == "find_tool"]), "catalog_tools_used": sorted(used_ok),
            **false_positive_signals(calls, meta),
            "filter": filt, "filter_has_needs": all(any(n in filt for n in g) for g in meta["needs"]) if cond == "filtered" else None,
            "wall_s": round(wall, 1), "timed_out": timed_out, "rc": rc, "filter_dm_usd": filt_cost,
            "dm_usd": sum(c.get("dm_cost") or 0 for c in calls), **stats,
            "builtin_tools_used": dict(Counter(t for t in tool_calls if not str(t).startswith("corp_"))),
            "outside_paths": outside_paths(transcript, work / "proj")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks-dir", default=str(HERE / "tasks_v2"))
    ap.add_argument("--tasks", default="")
    ap.add_argument("--conditions", default=",".join(CONDITIONS))
    ap.add_argument("--models", default="deepseek")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", default=str(REPO / "results" / "harness_v2"))
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--budget", type=float, default=5.0)
    ap.add_argument("--catalog-size", type=int, default=100, choices=[100, 300])
    a = ap.parse_args()
    task_root = Path(a.tasks_dir).resolve()
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    tasks = a.tasks.split(",") if a.tasks else sorted(p.name for p in task_root.iterdir() if (p / "task.json").exists())
    done = set()
    if (out / "rows.jsonl").exists():  # resume: skip runs already recorded
        done = {(r["task"], r["condition"], r["model"], r["rep"]) for r in map(json.loads, open(out / "rows.jsonl"))}
    jobs = [(t, c, m, r) for r in range(a.repeats) for m in a.models.split(",") for t in tasks for c in a.conditions.split(",")
            if (t, c, m, r) not in done]
    random.Random(0).shuffle(jobs)
    u0 = key_usage()
    print(f"OpenRouter key usage before batch: ${u0:.4f}; {len(jobs)} runs to do", flush=True)
    proxy = subprocess.Popen([str(PY), str(HERE / "llm_proxy.py"), "--port", str(PROXY_PORT),
                              "--models", ",".join(MODELS[m] for m in a.models.split(",")), "--log", str(out / "proxy_log.jsonl")],
                             stdout=subprocess.DEVNULL, stderr=open(out / "proxy_stderr.txt", "a"))
    lock = threading.Lock()
    state = {"spent": 0.0, "n": 0, "stop": False}
    ports = list(range(MCP_PORT_BASE, MCP_PORT_BASE + a.workers))

    def worker(job):
        if state["stop"]:
            return
        with lock:
            port = ports.pop()
        try:
            row = run_one(task_root, *job, port, out, a.timeout, a.k, a.catalog_size)
        except Exception as e:  # noqa: BLE001
            print(f"ERROR {job}: {type(e).__name__}: {e}", flush=True)
            return
        finally:
            with lock:
                ports.append(port)
        with lock:
            state["spent"] += row.get("oc_cost", 0) + row["dm_usd"] + row["filter_dm_usd"]  # opencode's estimate matched the bill in harness_v2
            state["n"] += 1
            if state["spent"] >= a.budget:
                state["stop"] = True
            with open(out / "rows.jsonl", "a") as f:
                f.write(json.dumps(row) + "\n")
            flag = "" if row["ran_in_task_dir"] and not row["outside_paths"] else f"  !! {row['outside_paths'][:3]}"
            print(f"[{state['n']}/{len(jobs)} ~${state['spent']:.2f}] {row['task']:16s} {row['condition']:15s} {row['model']:8s} r{row['rep']} "
                  f"pass={row['passed']!s:5s} {row.get('outcome', '')} steps={row.get('steps')} prompt_max={row.get('max_prompt')}{flag}", flush=True)

    try:
        wait_port(PROXY_PORT)
        with ThreadPoolExecutor(a.workers) as pool:
            list(pool.map(worker, jobs))
        if state["stop"]:
            print(f"budget {a.budget} reached; stopped early", flush=True)
    finally:
        proxy.terminate()
        time.sleep(30)
        u1 = key_usage()
        print(f"OpenRouter key usage after batch: ${u1:.4f} (batch cost ${u1 - u0:.4f})", flush=True)


if __name__ == "__main__":
    main()
