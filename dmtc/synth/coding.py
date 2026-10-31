"""Coding-harness domain for SynthHolster: the tools coding agents actually have.

Families are written by hand to match real harnesses (file discovery, code search, reading, editing,
execution, git, web/docs, control). Variants inside a family are the near-duplicates agents confuse.
Cues and items are generated (DeepSeek), cues are blind-reviewed (GPT-6.1 Sol), items are checked by
code, exactly like the business domains.

Slices are implicit / multiturn / direct only. sibling_neg and missing are left out on purpose: in a
coding agent a removed tool usually has a working substitute (grep can find a definition) and a
missing value can usually be looked up, so cannot_answer / request_for_info would be wrong gold.

    python -m dmtc.synth.coding --out data/synth
"""

from __future__ import annotations

import argparse
import json
import random
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dmtc.synth.build import CANARY, ask_json, contains_all

STOP = {"the", "and", "for", "file", "files", "with", "without", "from", "into", "that", "this", "what", "which"}


def coding_banned(v: dict, f: dict) -> list[str]:
    """Discriminating tool-name words only (axis values here are phrases, not keywords)."""
    def toks(x):
        return {t for t in x["name"].split("_")[1:] if len(t) > 2}
    shared = set.intersection(*(toks(x) for x in f["variants"]))
    return sorted(t for t in toks(v) - shared if t not in STOP)


def leaks_spec(text: str, v: dict) -> list[str]:
    """4-word phrases copied from the tool's condition or description give the answer away."""
    def grams(s):
        w = re.findall(r"[a-z0-9]+", s.lower())
        return {" ".join(w[i:i + 4]) for i in range(len(w) - 3)}
    return sorted(grams(text) & (grams(v["condition"]) | grams(v["description"])))


def P(t: str, d: str, ex: list) -> dict:
    return {"type": t, "description": d, "examples": ex}


FAMILIES = [
    {"family": "find files", "axis": "what the agent knows about the files it is looking for", "variants": [
        {"name": "ws_list_directory", "axis_value": "a specific folder", "condition": "The agent wants to see what is inside one specific directory.",
         "description": "List the files and subdirectories directly inside a given directory.",
         "params": {"path": P("string", "Directory to list.", ["src/payments", "tests/integration", "docs"])}, "required": ["path"]},
        {"name": "ws_glob_files", "axis_value": "a name pattern", "condition": "The agent knows a filename pattern or extension and wants every match across the tree.",
         "description": "Find all files in the repository whose paths match a glob pattern such as **/*.yaml.",
         "params": {"pattern": P("string", "Glob pattern.", ["**/*.proto", "**/migrations/*.sql", "src/**/test_*.py"])}, "required": ["pattern"]},
        {"name": "ws_find_file_by_name", "axis_value": "an exact file name", "condition": "The agent knows the exact file name but not where it lives.",
         "description": "Locate a file by its exact name anywhere in the repository and return its path(s).",
         "params": {"filename": P("string", "Exact file name.", ["settings_local.py", "Dockerfile.dev", "webpack.config.js"])}, "required": ["filename"]},
    ]},
    {"family": "search code", "axis": "what the agent is searching for", "variants": [
        {"name": "ws_grep_regex", "axis_value": "exact text", "condition": "The agent wants every place a literal string or regex appears (e.g. an error message, a TODO, a config key).",
         "description": "Search file contents for a literal string or regular expression and return matching lines.",
         "params": {"pattern": P("string", "Text or regex.", ["ConnectionResetError", "TODO(perf)", "MAX_RETRIES"]),
                    "path": P("string", "Folder to search.", ["src", ".", "services/api"])}, "required": ["pattern"]},
        {"name": "ws_semantic_code_search", "axis_value": "behaviour described in words", "condition": "The agent can only describe what the code does, not any identifier or exact text in it.",
         "description": "Natural-language search over the codebase: finds code that implements a described behaviour.",
         "params": {"query": P("string", "Description of the behaviour.", ["where retries with exponential backoff happen", "code that signs S3 upload URLs", "where the session cookie is refreshed"])}, "required": ["query"]},
        {"name": "ws_go_to_definition", "axis_value": "where a symbol is defined", "condition": "The agent has a symbol name and wants the single place it is defined.",
         "description": "Jump to the definition of a function, class or variable by name (language-server lookup).",
         "params": {"symbol": P("string", "Symbol name.", ["load_config", "OrderSerializer", "retry_with_jitter"])}, "required": ["symbol"]},
        {"name": "ws_find_references", "axis_value": "every place a symbol is used", "condition": "The agent has a symbol name and needs all call sites / usages (e.g. before renaming or changing its signature).",
         "description": "List every usage of a symbol across the codebase (language-server references).",
         "params": {"symbol": P("string", "Symbol name.", ["parse_timestamp", "UserCache", "emit_metric"])}, "required": ["symbol"]},
    ]},
    {"family": "read", "axis": "what the agent needs to read", "variants": [
        {"name": "ws_read_file", "axis_value": "a whole small file", "condition": "The agent needs the full contents of a short file.",
         "description": "Return the entire contents of a file.",
         "params": {"path": P("string", "File path.", ["setup.cfg", "src/utils/dates.py", ".github/workflows/ci.yml"])}, "required": ["path"]},
        {"name": "ws_read_file_range", "axis_value": "specific lines of a large file", "condition": "The file is large and the agent already knows which line numbers matter.",
         "description": "Return only a range of lines from a file.",
         "params": {"path": P("string", "File path.", ["src/engine/core.py", "legacy/monolith.js", "server/handlers.go"]),
                    "start_line": P("integer", "First line.", [1180, 402, 77]), "end_line": P("integer", "Last line.", [1240, 460, 130])},
         "required": ["path", "start_line", "end_line"]},
        {"name": "ws_read_notebook", "axis_value": "a Jupyter notebook", "condition": "The file is a Jupyter notebook and the agent needs its cells and outputs.",
         "description": "Read a .ipynb notebook: code cells, markdown cells and their outputs.",
         "params": {"path": P("string", "Notebook path.", ["analysis/churn.ipynb", "notebooks/eda_v2.ipynb", "reports/q3.ipynb"])}, "required": ["path"]},
    ]},
    {"family": "edit", "axis": "what kind of change to a file", "variants": [
        {"name": "ws_str_replace_edit", "axis_value": "change an existing snippet", "condition": "An existing file must be changed by replacing one exact, unique snippet.",
         "description": "Edit an existing file by replacing an exact snippet with new text.",
         "params": {"path": P("string", "File path.", ["src/api/routes.py", "lib/format.ts", "app/models/user.rb"]),
                    "old_text": P("string", "Exact text to replace.", ["timeout=30", "return None", "if (!user) {"]),
                    "new_text": P("string", "Replacement.", ["timeout=60", "return []", "if (!user || !user.active) {"])},
         "required": ["path", "old_text", "new_text"]},
        {"name": "ws_write_new_file", "axis_value": "create a file", "condition": "The file does not exist yet and must be created.",
         "description": "Create a new file with the given contents (fails if it exists).",
         "params": {"path": P("string", "New file path.", ["tests/test_refunds.py", "scripts/backfill_ids.sh", "docs/adr/0007-caching.md"]),
                    "content": P("string", "File contents.", ["import pytest", "#!/usr/bin/env bash", "# ADR 7"])}, "required": ["path", "content"]},
        {"name": "ws_insert_lines", "axis_value": "add lines without replacing", "condition": "New lines must be added at a known line number while keeping everything that is there.",
         "description": "Insert new text after a given line number of an existing file, replacing nothing.",
         "params": {"path": P("string", "File path.", ["src/config.py", "Makefile", "requirements.txt"]),
                    "after_line": P("integer", "Insert after this line.", [12, 40, 3]),
                    "text": P("string", "Text to insert.", ["DEBUG = False", "lint:\\n\\truff .", "httpx==0.27.0"])},
         "required": ["path", "after_line", "text"]},
        {"name": "ws_apply_patch", "axis_value": "coordinated change across files", "condition": "One logical change must touch several files at once (e.g. a rename across modules).",
         "description": "Apply a unified diff that may modify several files in one step.",
         "params": {"patch": P("string", "Unified diff.", ["--- a/x.py\\n+++ b/x.py", "--- a/api.ts\\n+++ b/api.ts", "--- a/go.mod\\n+++ b/go.mod"])},
         "required": ["patch"]},
    ]},
    {"family": "run commands", "axis": "what is being run and for how long", "variants": [
        {"name": "ws_run_tests", "axis_value": "the test suite", "condition": "The agent wants to run tests (a test file, a test case or the whole suite) and get pass/fail results.",
         "description": "Run tests with the project's test runner and report failures.",
         "params": {"target": P("string", "Test file, test id or directory.", ["tests/test_api.py::test_refund", "tests/unit", "spec/models"])}, "required": ["target"]},
        {"name": "ws_run_shell", "axis_value": "a short one-off command", "condition": "A quick command that finishes on its own (install a package, build, run a script once).",
         "description": "Run a shell command to completion and return its output.",
         "params": {"command": P("string", "Command.", ["pip install -e .", "npm run build", "python scripts/migrate.py --dry-run"])}, "required": ["command"]},
        {"name": "ws_run_in_background", "axis_value": "a long-running process", "condition": "A process that keeps running (a dev server, a file watcher, a long job) and must not block the agent.",
         "description": "Start a long-running command in the background and return a job id.",
         "params": {"command": P("string", "Command.", ["npm run dev", "uvicorn app:app --reload", "celery -A tasks worker"])}, "required": ["command"]},
        {"name": "ws_check_background_output", "axis_value": "an already-running job", "condition": "A job was started earlier and the agent wants its latest output or status.",
         "description": "Fetch new output and status of a background job by id.",
         "params": {"job_id": P("string", "Job id.", ["job-17", "bg_3", "job-204"])}, "required": ["job_id"]},
    ]},
    {"family": "git", "axis": "which repository question", "variants": [
        {"name": "ws_git_status", "axis_value": "which files changed", "condition": "The agent wants an overview of which files are modified, staged or untracked right now.",
         "description": "Show the working-tree status: modified, staged and untracked files.",
         "params": {"porcelain": P("boolean", "Machine-readable output.", [True, False, True])}, "required": ["porcelain"]},
        {"name": "ws_git_diff", "axis_value": "the exact line changes", "condition": "The agent wants to see exactly which lines changed (e.g. to review its own edits).",
         "description": "Show line-by-line differences of uncommitted changes, optionally for one path.",
         "params": {"path": P("string", "Path to diff.", ["src/parser.py", ".", "web/components/Cart.tsx"])}, "required": ["path"]},
        {"name": "ws_git_log", "axis_value": "the commit history", "condition": "The agent wants the sequence of past commits (e.g. when a behaviour was introduced).",
         "description": "Show recent commit history, optionally limited to a path.",
         "params": {"path": P("string", "Path.", ["src/billing", ".", "CHANGELOG.md"]), "max_count": P("integer", "Commits.", [20, 50, 10])},
         "required": ["path"]},
        {"name": "ws_git_blame", "axis_value": "who last changed specific lines", "condition": "The agent wants to know which commit/author last touched particular lines.",
         "description": "Show, for each line of a file range, the last commit and author that modified it.",
         "params": {"path": P("string", "File.", ["src/auth/tokens.py", "lib/rate_limit.go", "app.js"]),
                    "line": P("integer", "Line number.", [88, 213, 41])}, "required": ["path", "line"]},
    ]},
    {"family": "web and docs", "axis": "where the information is", "variants": [
        {"name": "ws_web_search", "axis_value": "unknown, needs finding", "condition": "The agent needs information it has no link for (an error message online, a changelog, a recent release).",
         "description": "Search the web and return result titles, snippets and URLs.",
         "params": {"query": P("string", "Query.", ["psycopg OperationalError SSL SYSCALL EOF", "pydantic v3 migration breaking changes", "vite 7 release notes"])}, "required": ["query"]},
        {"name": "ws_fetch_url", "axis_value": "a known link", "condition": "The agent already has a specific URL (e.g. from an issue or a stack trace) and wants its content.",
         "description": "Download a web page by URL and return its text.",
         "params": {"url": P("string", "URL.", ["https://github.com/acme/api/issues/482", "https://docs.example.dev/errors/E1042", "https://status.example.io/incidents/91"])}, "required": ["url"]},
        {"name": "ws_read_package_docs", "axis_value": "an installed library's API", "condition": "The agent needs the reference docs of a library the project already depends on.",
         "description": "Look up API reference documentation for an installed package and topic.",
         "params": {"package": P("string", "Package.", ["sqlalchemy", "react-query", "tokio"]), "topic": P("string", "Topic.", ["Session.merge", "useInfiniteQuery", "select!"])},
         "required": ["package", "topic"]},
    ]},
    {"family": "control", "axis": "what the agent should do instead of touching the repo", "variants": [
        {"name": "ws_ask_user", "axis_value": "a decision only the user can make", "condition": "The instructions are ambiguous or a choice needs the user's preference/permission.",
         "description": "Ask the user a clarifying question and wait for the answer.",
         "params": {"question": P("string", "Question.", ["Should the migration drop the old column?", "Target Python 3.11 or 3.13?", "OK to force-push this branch?"])}, "required": ["question"]},
        {"name": "ws_update_todo_list", "axis_value": "track a multi-step plan", "condition": "The task has several steps and the agent wants to record or update its plan/checklist.",
         "description": "Create or update the agent's task checklist.",
         "params": {"items": P("string", "Checklist items.", ["reproduce; fix; add test; update docs", "audit callers; rename; run suite", "write ADR; implement cache; benchmark"])}, "required": ["items"]},
        {"name": "ws_finish", "axis_value": "the task is done", "condition": "The change is complete and verified; the agent should stop and report.",
         "description": "End the task and give the user a summary of what was done.",
         "params": {"summary": P("string", "Summary.", ["Fixed the off-by-one in pagination and added a regression test.", "Renamed UserCache to SessionCache across 14 files; suite green.", "Added retry with jitter to the S3 client; tests pass."])}, "required": ["summary"]},
    ]},
]

CUE_PROMPT = """These are tools of a coding agent working in a software repository. For EACH tool, write 4 short
"situation cues": CONCRETE details that make THIS tool the right next step and not its siblings, without naming
the tool or its distinguishing words. A cue must be either something a developer would literally say
("the dev server needs to stay up while I click around") or a concrete observation
("the stack trace ends with a link to the vendor's error page"). Never describe the rule, and never talk
about "the agent" or "the developer" in the third person.
Family: {family} (axis: {axis})
Tools:
{tools}
Return JSON: {{"cues": {{"<tool name>": ["cue", "cue", "cue", "cue"]}}}}"""

ITEM_PROMPT = """Write a short situation from a coding agent's session in a software repository, at the moment it must
choose its next tool. (For you only, do not paraphrase it into the text: the right next step is: {condition})
Return two parts:
- "task": the developer's original request to the agent (1-2 sentences).
- "progress": the agent's own note of what it has done and observed so far (1-3 sentences, first person).
Rules:
- Work this cue in naturally (in the task or the progress note): "{cue}". Do not state the rule itself.
- Include these exact values verbatim somewhere: {values}
- Do NOT use these words: {banned}. Do not name any tool.
- Do not write code blocks, JSON or braces.
{extra}
Return JSON: {{"task": str, "progress": str}}"""

DIRECT = [
    "What is the difference between git merge and git rebase?", "When should I use a dataclass instead of a dict in Python?",
    "What does the 'Shift left' idea mean in testing?", "Why are flaky tests a problem for CI?",
    "Explain what a race condition is in one paragraph.", "What is semantic versioning?",
    "What's the difference between a unit test and an integration test?", "What does idempotent mean for an HTTP endpoint?",
    "Why would someone prefer composition over inheritance?", "What is a monorepo?",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synth")
    ap.add_argument("--seed", type=int, default=20261026)
    ap.add_argument("--confirm-from", help="reuse this folder's catalog and reviewed cues; write fresh coding "
                                           "items only (confirmation run) into --out")
    a = ap.parse_args()
    out = Path(a.out)
    if a.confirm_from:
        return confirm(Path(a.confirm_from), out, a.seed)
    cat = json.loads((out / "catalog.json").read_text())
    cat["domains"] = [d for d in cat["domains"] if d["brand"] != "CodingAgent"]

    # cues per family (generated), then the same blind review as the business domains
    def cues(f):
        listing = "\n".join(f"- {v['name']}: {v['condition']}" for v in f["variants"])
        data, c = ask_json(CUE_PROMPT.format(family=f["family"], axis=f["axis"], tools=listing), max_tokens=3000)
        for v in f["variants"]:
            v["cues"] = (data.get("cues") or {}).get(v["name"], [])[:4]
        return c

    fams = json.loads(json.dumps(FAMILIES))
    with ThreadPoolExecutor(8) as ex:
        cost = sum(ex.map(cues, fams))
    dom = {"brand": "CodingAgent", "theme": "coding agent working in a software repository", "families": fams,
           "direct_questions": DIRECT}
    cat["domains"].append(dom)
    (out / "catalog.json").write_text(json.dumps(cat, indent=1))
    print(f"coding domain: {len(fams)} families, {sum(len(f['variants']) for f in fams)} tools; cue generation ${cost:.4f}")

    from dmtc.synth.build import review_catalog

    review_catalog(out, only_brand="CodingAgent")
    cat = json.loads((out / "catalog.json").read_text())
    dom = next(d for d in cat["domains"] if d["brand"] == "CodingAgent")

    rng = random.Random(a.seed)
    specs = [(s, f, v) for f in dom["families"] for v in f["variants"] if v["cues"]
             for s in ("implicit", "implicit", "multiturn")]

    def make(spec):
        return make_item(spec, a.seed, str(rng.random()))

    with ThreadPoolExecutor(8) as ex:
        res = list(ex.map(make, specs))
    items = [i for i, _ in res]
    items += [{"slice": "direct", "domain": "CodingAgent", "family": None, "messages": [{"role": "user", "content": q}],
               "label": "direct", "gold_tool": None, "removed_tool": None, "family_tools": [], "check": "none"} for q in DIRECT]
    kept = [i for i in items if i["check"] in ("ok", "none")]
    old = [json.loads(l) for l in open(out / "items.jsonl")][1:]
    old = [i for i in old if i["domain"] != "CodingAgent"]
    allk = old + kept
    for k, it in enumerate(allk):
        it["id"] = f"sh{k:04d}"
    (out / "items.jsonl").write_text(json.dumps({"canary": CANARY}) + "\n" + "\n".join(json.dumps(i) for i in allk) + "\n")
    with open(out / "items_all.jsonl", "a") as f:
        for i in items:
            f.write(json.dumps(i) + "\n")
    from collections import Counter

    print(f"coding items: generated {len(items)}, kept {len(kept)}; cost ${sum(c for _, c in res):.4f}")
    print("kept by slice:", dict(Counter(i['slice'] for i in kept)), "| total benchmark items:", len(allk))


def make_item(spec, seed: int, salt: str):
    """One coding item from (slice, family, variant): generation + construction checks."""
    slice_, f, v = spec
    a_seed = seed
    r = random.Random(f"{a_seed}:{slice_}:{v['name']}:{salt}")
    cue = r.choice(v["cues"])
    vals = {p: str(r.choice(s["examples"])) for p, s in v["params"].items() if p in v["required"] and s["type"] != "boolean"}
    # free-text values (questions, summaries, patches) cannot be required verbatim
    vals = {p: x for p, x in vals.items() if p not in ("question", "summary", "items", "patch", "content", "text", "query")}
    extra = ("- Put the cue ONLY in the task; the progress note must not repeat it." if slice_ == "multiturn" else "")
    data, c = ask_json(ITEM_PROMPT.format(condition=v["condition"], cue=cue, values=json.dumps(vals) if vals else "(none)",
                                          banned=", ".join(coding_banned(v, f)) or "(none)", extra=extra), max_tokens=800)
    task, prog = data.get("task", ""), data.get("progress", "")
    msgs = [{"role": "user", "content": task}, {"role": "assistant", "content": prog},
            {"role": "user", "content": "Continue with the next step."}]
    text = f"{task} {prog}"
    hits = [w for w in coding_banned(v, f) if re.search(rf"\b{re.escape(w)}", text.lower())]
    hits += [f"copied:{g}" for g in leaks_spec(text, v)]
    hits += [x for x in re.findall(r"\bws_[a-z_]+\b", text)]
    if "{" in text or "}" in text:
        hits.append("{}")
    # meta narration states the rule instead of showing the situation
    hits += [f"meta:{m}" for m in re.findall(r"\bthe (?:agent|developer|user asks)\b", text.lower())]
    ok = bool(task.strip() and prog.strip()) and contains_all(text, list(vals.values())) and not hits
    return {"slice": slice_, "domain": "CodingAgent", "family": f["family"], "messages": msgs, "label": "tool_call",
            "gold_tool": v["name"], "removed_tool": None, "gold_args": vals, "withheld": None, "cue": cue,
            "family_tools": [x["name"] for x in f["variants"]], "check": "ok" if ok else f"fail:{hits}"}, c



def confirm(src: Path, out: Path, seed: int) -> None:
    """Pre-registered confirmation run: fresh coding items, same catalog/cues/pipeline, new seed."""
    import shutil
    from collections import Counter

    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(src / "catalog.json", out / "catalog.json")
    cat = json.loads((out / "catalog.json").read_text())
    dom = next(d for d in cat["domains"] if d["brand"] == "CodingAgent")
    specs = [(s, f, v) for f in dom["families"] for v in f["variants"] if v["cues"]
             for s in ("implicit", "implicit", "implicit", "multiturn", "multiturn")]
    with ThreadPoolExecutor(8) as ex:
        res = list(ex.map(lambda k_sp: make_item(k_sp[1], seed, f"confirm{k_sp[0]}"), enumerate(specs)))
    items = [i for i, _ in res]
    kept = [i for i in items if i["check"] == "ok"]
    for k, it in enumerate(kept):
        it["id"] = f"cf{k:04d}"
    (out / "items_all.jsonl").write_text("\n".join(json.dumps(i) for i in items) + "\n")
    (out / "items.jsonl").write_text(json.dumps({"canary": CANARY}) + "\n" + "\n".join(json.dumps(i) for i in kept) + "\n")
    print(f"confirmation items: generated {len(items)}, kept {len(kept)} after code checks; cost ${sum(c for _, c in res):.4f}")
    print("kept by slice:", dict(Counter(i["slice"] for i in kept)))


if __name__ == "__main__":
    main()
