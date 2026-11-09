"""Runs tasks_real tasks through Hermes Agent (Nous Research, 0.19.0 from PyPI) instead of opencode.

Sandbox hardening (hermes_v3 onward): each run sees only its own work folder, a fresh HOME and a private /tmp; the MCP server answers only requests carrying a
per-run token that Hermes reads from its environment (Hermes strips it from its shell's environment), so the
agent's shell can no longer dump the catalog or call the ranker directly.

  hermes_dm2bridge  Hermes's bridge with the redesigned DM ranker (selectors_real.dm2: exact-name shortcut,
                    stop at 90% probability mass)

Optimised DM variants (added after hermes_v1 showed the DM's cost came from packaging, not ranking)
  hermes_dmbridge   Hermes's own bridge (compact hits, tool_describe, direct typed calls) with the ranking swapped from BM25
                    to the DM: tools/tool_search.py in the Hermes install is patched to POST to the MCP server's /rank
                    endpoint when HERMES_RANK_URL is set (falls back to BM25 on error)
  hermes_randbridge the same bridge with random ranking: does the ranker matter at all?
  hermes_dmcompact  our MCP DM search, made compact: short hits + tool_describe + call_tool

  hermes_ccbridge    Claude Code style in Hermes's bridge: keyword ranker (name hits weigh most, select:<name> for exact
                    names) and every deferred tool's name listed in the prompt, as Claude Code lists them
  hermes_dmnullbridge DM v3: service-then-tool DM with a calibrated "none of these" option (empty answer when
                    P(none) >= 0.4 at the service step), at most 5 results (selectors_real.dm_null)
  hermes_oraclebridge upper bound: the ranker returns the task's needed tools first (then BM25), in Hermes's bridge
  hermes_embedbridge embedding ranker (text-embedding-3-small cosine; Goose vector router / RAG-MCP) in Hermes's bridge

Conditions (Hermes's own tool search is turned off except in hermes_default / hermes_ts)
  hermes_default Hermes as shipped: tool_search "auto" (defers MCP tools once they pass 10% of the context window;
                 613 tools do), i.e. its built-in BM25 search with bridge tools tool_search / tool_describe / tool_call
  hermes_ts      the same with tool_search forced "on"
  hermes_native  all 613 tool definitions in context
  hermes_bm25    our MCP server's BM25 tool_search + service list (Codex CLI / Anthropic BM25 style)
  hermes_cc      our MCP server's Claude Code-style ToolSearch (deferred tool names listed; select:/keyword queries)
  hermes_dm      our MCP server's decision-model tool_search (server, then tool)

Same sandbox as runner.py (user agentbox, empty environment, LLM only through the key-holding proxy). Hermes runs
one-shot: `hermes -z "<prompt>" --yolo` (yolo = no interactive approval prompts; the sandbox is the guard).
Scoring is runner_real's: hidden tests + required writes + no destructive calls. Hermes's session database
(state.db) is kept per run; searches through Hermes's own bridge are counted from it.

    source .or_env && .venv/bin/python harness/run_hermes.py --tasks gh_ci_fail,loki_errors --workers 4
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sqlite3
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))
import runner as R  # noqa: E402
import runner_real as RR  # noqa: E402
from real_sim import _matches, is_destructive  # noqa: E402

HERMES = "/opt/agentbox/hermes/bin/hermes"
PROXY_PORT, MCP_PORT_BASE = 18080, 18390
PRICES = {"deepseek": (0.15e-6, 0.003e-6, 0.6e-6), "hy4": (0.417e-6, 0.021e-6, 1.2505e-6),  # input, cache read, output
          "kimi": (2.00e-6, 0.20e-6, 10.00e-6), "sol": (2.00e-6, 0.10e-6, 10.00e-6), "sonnet": (2.00e-6, 0.10e-6, 10.00e-6), "haiku": (0.10e-6, 0.01e-6, 0.50e-6)}  # kimi: Moonshot AI first-party (the endpoint it is pinned to)
PINNED_PRICES = {"deepseek": (0.098e-6, 0.003e-6, 0.29e-6)}  # DeepInfra fp8, used when --pin-provider is set
CACHE_WRITE_PRICE = {"sol": 2.50e-6, "sonnet": 2.50e-6, "haiku": 0.125e-6}  # per token; providers not listed do not charge cache writes (DeepInfra)
PIN = {"deepseek": "deepinfra/fp8", "kimi": "moonshotai/mxfp4", "sol": "openai", "sonnet": "anthropic", "haiku": "anthropic"}  # kimi: first-party Moonshot (DeepInfra bf16 ran at ~2 tok/s)
# Probing commands blocked by Hermes's own approvals.deny list (an organisation policy); the sandbox isolation is the
# real guard, this keeps agents from spending their run exploring the machine.
DENY = ["*curl *", "*wget *", "ps *", "* ps *", "*ps aux*", "ss *", "*netstat*", "*nmap*", "*/proc/*", "find /", "find / *",
        "*127.0.0.1*", "*localhost*", "*hermes_home*", "*lsof*"]
MCP_MODE = {"hermes_dmnullbridge": "native", "hermes_oraclebridge": "native", "hermes_ccbridge": "native", "hermes_embedbridge": "native", "hermes_dm2bridge": "native", "hermes_dmtoppbridge": "native", "hermes_dmtopp10bridge": "native", "hermes_dmtopp05bridge": "native", "hermes_dmk3bridge": "native", "hermes_dmmenubridge": "native", "hermes_dmynbridge": "native", "hermes_dmrecbridge": "native", "hermes_dmrec510bridge": "native", "hermes_dmfrankenbridge": "native", "hermes_dmctxbridge": "native", "hermes_dmchainbridge": "native", "hermes_defaultchain": "native", "hermes_dmbridge": "native", "hermes_randbridge": "native", "hermes_dmcompact": "dm",
            "hermes_default": "native", "hermes_ts": "native", "hermes_native": "native", "hermes_bm25": "bm25", "hermes_cc": "cc", "hermes_dm": "dm"}
CONDITIONS = ["hermes_default", "hermes_bm25", "hermes_cc", "hermes_dm"]


CHAINED = ("hermes_dmchainbridge", "hermes_defaultchain")  # tool chaining in the bridge, after ranking
BRIDGE_RANKED = ("hermes_dmctxbridge", "hermes_dmfrankenbridge", "hermes_dmrec510bridge", "hermes_dmrecbridge", "hermes_dmynbridge", "hermes_dmmenubridge", "hermes_dmchainbridge", "hermes_dmk3bridge", "hermes_dmtopp05bridge", "hermes_dmtoppbridge", "hermes_dmtopp10bridge", "hermes_dmnullbridge", "hermes_oraclebridge", "hermes_dmbridge", "hermes_dm2bridge", "hermes_randbridge", "hermes_ccbridge", "hermes_embedbridge")


def cc_names() -> str:
    """Claude Code lists every deferred tool by name; hermes_ccbridge gets the same list (Hermes names: mcp__corp__<tool>)."""
    from real_sim import load_catalog
    names = [t["name"] for t in load_catalog(extra=OPTS["extra"])]
    return ("The following deferred tools are available via tool search (query `select:<name>` to load one by exact name, or "
            "search by keywords):\n" + "\n".join(names) + "\n\n")


def config_yaml(cond: str, model: str, port: int, deny: bool = False) -> str:
    on = "tools:\n  tool_search:\n    enabled: \"on\"\n"
    ts = {"hermes_dmnullbridge": on, "hermes_oraclebridge": on, "hermes_ccbridge": on, "hermes_embedbridge": on, "hermes_dm2bridge": on, "hermes_dmtoppbridge": on, "hermes_dmtopp10bridge": on, "hermes_dmtopp05bridge": on, "hermes_dmk3bridge": on, "hermes_dmmenubridge": on, "hermes_dmynbridge": on, "hermes_dmrecbridge": on, "hermes_dmrec510bridge": on, "hermes_dmfrankenbridge": on, "hermes_dmctxbridge": on, "hermes_dmchainbridge": on, "hermes_defaultchain": "", "hermes_dmbridge": on, "hermes_randbridge": on, "hermes_default": "", "hermes_ts": "tools:\n  tool_search:\n    enabled: \"on\"\n"}.get(cond, "tools:\n  tool_search:\n    enabled: \"off\"\n")
    return (f"model:\n  provider: custom\n  default: {model}\n  base_url: http://127.0.0.1:{PROXY_PORT}/api/v1\n  api_key: sandbox-no-key\n"
            f"{ts}mcp_servers:\n  corp:\n    url: http://127.0.0.1:{port}/mcp\n    headers:\n      x-corp-token: \"${{CORP_MCP_TOKEN}}\"\n"
            f"    timeout: 180\n    connect_timeout: 60\n"
            + ("approvals:\n  deny:\n" + "".join(f"    - \"{p}\"\n" for p in DENY) if deny else ""))


def hermes_tool_stats(db: Path) -> dict:
    """Counts of tool calls by name from Hermes's own session DB (includes its tool_search bridge)."""
    if not db.exists():
        return {}
    con = sqlite3.connect(db)
    out = {}
    for (tc,) in con.execute("SELECT tool_calls FROM messages WHERE role='assistant' AND tool_calls IS NOT NULL"):
        try:
            for c in json.loads(tc):
                n = (c.get("function") or {}).get("name") or c.get("name")
                out[n] = out.get(n, 0) + 1
        except Exception:  # noqa: BLE001
            pass
    con.close()
    return out


OPTS = {"tasks_dir": "tasks_real", "extra": False, "validate": False, "deny": False, "pinned": False, "preamble": False}
# One neutral line in front of every task prompt (all conditions alike), standing in for what real harnesses put in the
# system prompt about connected MCP servers (Claude Code lists server instructions and deferred tool names; Hermes's
# bridge only says how many tools are deferred).
PREAMBLE = ("Note: the company's internal services (code hosting, CI, error tracking, metrics and logs, incidents and on-call, "
            "tickets, docs and runbooks, chat, clusters, databases) are connected to you as tools. Some tools are loaded on demand: "
            "if a tool you need is not listed, use tool search to find it.\n\n")


def run_one(task: str, cond: str, mkey: str, rep: int, port: int, out: Path, timeout: int) -> dict:
    td = HERE / OPTS["tasks_dir"] / task
    meta = json.loads((td / "task.json").read_text())
    tag = f"{task}__{cond}__{mkey}__r{rep}"
    rec = out / "runs" / tag
    shutil.rmtree(rec, ignore_errors=True)
    rec.mkdir(parents=True)
    work = R.WORK / tag
    shutil.rmtree(work, ignore_errors=True)
    shutil.copytree(td / "repo", work / "proj")
    home = work / "hermes_home"
    home.mkdir(parents=True)
    (home / "config.yaml").write_text(config_yaml(cond, RR.MODELS[mkey], port, OPTS["deny"]))
    subprocess.run(["chown", "-R", "agentbox:agentbox", str(work)], check=True)
    work.chmod(0o700)
    token = os.urandom(16).hex()
    env = {**os.environ, "REAL_MODE": MCP_MODE[cond], "REAL_WORLD": str(td / "world.json"), "REAL_LOG": str(rec / "calls.jsonl"),
           "REAL_HTTP_PORT": str(port), "PYTHONPATH": f"{HERE}:{REPO}",
           "REAL_EXTRA": "1" if OPTS["extra"] else "0", "REAL_VALIDATE": "1" if OPTS["validate"] else "0",
           "REAL_COMPACT": "1" if cond == "hermes_dmcompact" else "0", "REAL_RANKER": {"hermes_randbridge": "random", "hermes_dm2bridge": "dm2", "hermes_dmtoppbridge": "dm_topp", "hermes_dmtopp10bridge": "dm_topp10", "hermes_dmtopp05bridge": "dm_topp05", "hermes_dmk3bridge": "dm_k3", "hermes_dmmenubridge": "dm_menu", "hermes_dmynbridge": "dm_yn", "hermes_dmrecbridge": "dm_rec", "hermes_dmrec510bridge": "dm_rec510", "hermes_dmfrankenbridge": "dm_franken", "hermes_dmctxbridge": "dm_ctx", "hermes_dmchainbridge": "dm", "hermes_ccbridge": "cc",
                           "hermes_embedbridge": "embed", "hermes_oraclebridge": "oracle", "hermes_dmnullbridge": "dm_null"}.get(cond, "dm"), "REAL_TASK": str(td / "task.json"), "REAL_TOKEN": token}
    mcp = subprocess.Popen([str(R.PY), str(HERE / "mcp_server_real.py")], env=env, stdout=subprocess.DEVNULL, stderr=open(rec / "mcp_stderr.txt", "w"))
    R.wait_port(port, 60)
    benv = [f"HOME={R.BOX_HOME}", f"HERMES_HOME={home}", "PATH=/opt/agentbox/py/bin:/opt/node22/bin:/usr/bin:/bin",
            f"PWD={work / 'proj'}", "TZ=UTC", "NO_COLOR=1", f"CORP_MCP_TOKEN={token}",
            *([f"HERMES_RANK_URL=http://127.0.0.1:{port}/rank"] if cond in BRIDGE_RANKED else []),
            *(["HERMES_CHAIN_GRAPH=/opt/agentbox/hermes/lib/python3.13/site-packages/tools/chain_graph.json"] if cond in CHAINED else [])]
    t0 = time.time()
    try:
        # private /tmp per run (a fresh tmpfs in its own mount namespace): runs cannot see each other's scratch files
        prompt = (PREAMBLE if OPTS["preamble"] else "") + (cc_names() if cond == "hermes_ccbridge" else "") + meta["prompt"]
        inner = ["runuser", "-u", "agentbox", "--", "env", "-i", *benv, HERMES, "-z", prompt, "--yolo", "--usage-file", str(work / "usage.json")]
        # isolation inside a private mount namespace: fresh /tmp, a fresh empty HOME, and /srv/agentbox/work showing only
        # this run's own folder (parallel runs share the agentbox uid, so file permissions alone cannot separate them)
        iso = ('set -e; mount -t tmpfs -o mode=1777 tmpfs /tmp; mkdir /tmp/.own; mount --bind "$W" /tmp/.own; '
               f'mount -t tmpfs -o mode=755 tmpfs {R.WORK}; mkdir -p "$W"; mount --move /tmp/.own "$W"; rmdir /tmp/.own; '
               f'mount -t tmpfs -o mode=700,uid=$(id -u agentbox),gid=$(id -g agentbox) tmpfs {R.BOX_HOME}; cd "$W/proj"; exec "$@"')
        r = subprocess.run(["unshare", "--mount", "--propagation", "private", "--", "env", f"W={work}", "sh", "-c", iso, "sh", *inner],
                           cwd=work / "proj", capture_output=True, text=True, timeout=timeout)
        transcript, timed_out = r.stdout + "\n--- stderr ---\n" + r.stderr, False
    except subprocess.TimeoutExpired as e:
        transcript, timed_out = str(e.stdout or "") + "\n[TIMEOUT]", True
        subprocess.run(["pkill", "-u", "agentbox", "-f", str(work)], check=False)
    wall = time.time() - t0
    mcp.terminate()
    mcp.wait(timeout=10)
    (rec / "stdout.txt").write_text(transcript)
    usage = json.loads((work / "usage.json").read_text()) if (work / "usage.json").exists() else {}
    for f in ("state.db", "logs/agent.log", "logs/errors.log"):
        if (home / f).exists():
            shutil.copy(home / f, rec / Path(f).name)
    htools = hermes_tool_stats(rec / "state.db")
    shutil.copytree(work / "proj", rec / "proj_final", ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))
    score = rec / "score"
    shutil.copytree(rec / "proj_final", score)
    (score / "tests").mkdir(exist_ok=True)
    shutil.copy(td / "hidden" / "test_hidden.py", score / "tests" / "test_hidden.py")
    h = subprocess.run([str(R.PY), "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests/test_hidden.py"], cwd=score, capture_output=True, text=True, timeout=120)
    shutil.rmtree(work)

    ev = [json.loads(l) for l in open(rec / "calls.jsonl")] if (rec / "calls.jsonl").exists() else []
    calls = [e for e in ev if e["kind"] == "call"]
    needed = {n for g in meta["needs"] for n in g}
    writes = [any(c["tool"] in (w["tool"] if isinstance(w["tool"], list) else [w["tool"]]) and _matches(w, c["args"]) for c in calls) for w in meta["writes"]]
    destructive = [c["tool"] for c in calls if is_destructive(c["tool"])]
    used_ok = {c["tool"] for c in calls if c.get("ok")}
    n_search = max(len([e for e in ev if e["kind"] == "search"]), htools.get("tool_search", 0) + htools.get("mcp__corp__tool_search", 0))
    inp, cr = usage.get("input_tokens") or 0, usage.get("cache_read_tokens") or 0
    cw = usage.get("cache_write_tokens") or 0
    if not cw and (rec / "state.db").exists():  # Hermes records cache writes per session in its database
        try:
            _c = sqlite3.connect(rec / "state.db"); cw = _c.execute("select coalesce(sum(cache_write_tokens),0) from sessions").fetchone()[0] or 0; _c.close()
        except Exception:  # noqa: BLE001
            cw = 0
    outp = (usage.get("output_tokens") or 0) + (usage.get("reasoning_tokens") or 0)
    pi, pc, po = (PINNED_PRICES.get(mkey) if OPTS["pinned"] else None) or PRICES[mkey]
    api = usage.get("api_calls") or 0
    return {"task": task, "category": meta["category"], "condition": cond, "model": mkey, "rep": rep,
            "passed": h.returncode == 0 and all(writes) and not destructive, "tests_ok": h.returncode == 0, "writes_ok": all(writes),
            "destructive": destructive, "needs_met": all(any(n in used_ok for n in g) for g in meta["needs"]),
            **RR.probe_outcome(rec / "proj_final", meta.get("probe")),
            "n_calls": len(calls), "n_searches": n_search, "n_describe": htools.get("tool_describe", 0),
            "n_failed_calls": len([c for c in calls if not c.get("ok")]), "n_unneeded_calls": len([c for c in calls if c["tool"] not in needed]),
            "hermes_tools": htools, "api_calls": api, "input": inp, "cache_read": cr, "output": outp,
            "prompt_per_call": round((inp + cr) / api) if api else 0, "total_tokens": usage.get("total_tokens") or 0,
            "cache_write": cw, "cost": inp * pi + cr * pc + outp * po + cw * CACHE_WRITE_PRICE.get(mkey, 0.0) + sum(e.get("cost") or 0 for e in ev if e["kind"] == "search"),
            "wall_s": round(wall, 1), "timed_out": timed_out, "error": "API call failed" in transcript}


def main() -> None:
    global PROXY_PORT
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="gh_ci_fail")
    ap.add_argument("--conditions", default=",".join(CONDITIONS))
    ap.add_argument("--models", default="deepseek")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--budget", type=float, default=3.0)
    ap.add_argument("--out", default=str(REPO / "results" / "hermes_v1"))
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--pin-provider", default="", help="variance control: model=provider pairs; 'default' = DeepInfra pins in PIN")
    ap.add_argument("--tasks-dir", default="tasks_real")
    ap.add_argument("--proxy-port", type=int, default=PROXY_PORT, help="change to run two batches side by side")
    ap.add_argument("--mcp-port-base", type=int, default=MCP_PORT_BASE)
    ap.add_argument("--extra", action="store_true", help="1,076-tool catalog (30 servers)")
    ap.add_argument("--validate", action="store_true", help="simulator validates arguments against the tool schemas")
    ap.add_argument("--deny", action="store_true", help="Hermes approvals.deny list for probing commands")
    ap.add_argument("--preamble", action="store_true", help="prefix every prompt with PREAMBLE (connected services, tool search)")
    ap.add_argument("--temperature", type=float, default=None, help="variance control: forced temperature (+ fixed seed)")
    a = ap.parse_args()
    PROXY_PORT = a.proxy_port
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    OPTS.update(tasks_dir=a.tasks_dir, extra=a.extra, validate=a.validate, deny=a.deny, pinned=bool(a.pin_provider), preamble=a.preamble)
    if a.pin_provider == "default":
        a.pin_provider = ",".join(f"{RR.MODELS[m]}={PIN[m]}" for m in a.models.split(",") if m in PIN)
    (out / "settings.json").write_text(json.dumps({**OPTS, "pin": a.pin_provider, "temperature": a.temperature}, indent=1))
    done = set()
    if (out / "rows.jsonl").exists():
        done = {(r["task"], r["condition"], r["model"], r["rep"]) for r in map(json.loads, open(out / "rows.jsonl"))}
    # order: repeat by repeat; within a repeat, (task, model) slots in random order with all conditions of a slot queued
    # back to back, so the runs being compared share time of day, provider load and pricing
    jobs = []
    for r in range(a.repeats):
        slots = [(t, m) for m in a.models.split(",") for t in a.tasks.split(",")]
        random.Random(r).shuffle(slots)
        for t, m in slots:
            conds = a.conditions.split(",")
            random.Random(f"{t}{r}").shuffle(conds)
            jobs += [(t, c, m, r) for c in conds if (t, c, m, r) not in done]
    u0 = R.key_usage()
    print(f"key usage before ${u0:.4f}; {len(jobs)} runs", flush=True)
    proxy = subprocess.Popen([str(R.PY), str(HERE / "llm_proxy.py"), "--port", str(PROXY_PORT), "--models", ",".join(RR.MODELS[m] for m in a.models.split(",")),
                              "--log", str(out / "proxy_log.jsonl"),
                              *(["--pin-provider", a.pin_provider] if a.pin_provider else []), *(["--temperature", str(a.temperature)] if a.temperature is not None else [])], stdout=subprocess.DEVNULL, stderr=open(out / "proxy_stderr.txt", "a"))
    R.wait_port(PROXY_PORT)
    lock, state = threading.Lock(), {"spent": 0.0, "n": 0, "stop": False}
    ports = list(range(a.mcp_port_base, a.mcp_port_base + a.workers))

    def worker(job):
        if state["stop"]:
            return
        with lock:
            port = ports.pop()
        try:
            row = run_one(*job, port, out, a.timeout)
        except Exception as e:  # noqa: BLE001
            print(f"ERROR {job}: {type(e).__name__}: {e}", flush=True)
            return
        finally:
            with lock:
                ports.append(port)
        with lock:
            state["spent"] += row["cost"]
            state["n"] += 1
            state["stop"] = state["spent"] >= a.budget
            with open(out / "rows.jsonl", "a") as f:
                f.write(json.dumps(row) + "\n")
            print(f"[{state['n']}/{len(jobs)} ~${state['spent']:.2f}] {row['task']:17s} {row['condition']:15s} pass={row['passed']!s:5s} "
                  f"tests={row['tests_ok']!s:5s} writes={row['writes_ok']!s:5s} {row.get('outcome', '')} calls={row['n_calls']} search={row['n_searches']} "
                  f"api={row['api_calls']} prompt/call={row['prompt_per_call'] // 1000}k ${row['cost']:.4f} {row['wall_s']}s{' API-ERROR' if row['error'] else ''}", flush=True)

    try:
        with ThreadPoolExecutor(a.workers) as pool:
            list(pool.map(worker, jobs))
    finally:
        proxy.terminate()
        time.sleep(15)
        print(f"key usage after ${R.key_usage():.4f} (batch ${R.key_usage() - u0:.4f})", flush=True)


if __name__ == "__main__":
    main()
