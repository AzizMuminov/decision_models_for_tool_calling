"""Meta-analysis: the coding agent's own tool calls in this project's session transcript.

Turns a Claude Code transcript (.jsonl) into one row per tool call: which tool, what kind of call
(Bash commands are classed by their first program), how large the input and result were, whether it
failed, how much context the model had to read to make that decision, and which user request it served.

    python3 scripts/meta_transcript.py <transcript.jsonl> results/meta
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

PREFIX = re.compile(r"^\s*(?:(?:cd|source|export)\s+[^;&|]+(?:&&|;)\s*|[A-Z_]+=[^\s;]+(?:\s*;\s*|\s+)|time\s+|nohup\s+|timeout\s+\d+\s+|rm -[rf]+ [^;&]+(?:&&|;)\s*|mkdir -p [^;&]+(?:&&|;)\s*)*")


def bash_class(cmd: str) -> str:
    """Intent of a shell command: strip cd/source/VAR= prefixes, then look at the first real program."""
    c = PREFIX.sub("", cmd)
    first = (c.split() or [""])[0]
    body = cmd
    if re.match(r"(\.venv/bin/)?python3?$", first) or first.endswith("/python"):
        if "<<" in c.split("\n")[0] or re.match(r"\S+\s+-c\b", c):  # inline python
            if re.search(r"open\([^)]*,\s*['\"]w['\"]\)|\.write_text\(|def edit\(|\.replace\(.*\)\s*\n.*open\(", body):
                return "edit files (inline python)"
            return "analyse data (inline python)"
        if re.search(r"-m (dmtc|pytest)|scripts/|paper/", c):
            return "run tests" if "pytest" in c else "run pipeline/script"
        return "run pipeline/script"
    if first in ("pdflatex", "bibtex") or re.search(r"\bpdflatex\b", c.split("\n")[0][:200]):
        return "build latex"
    if first in ("pdftoppm", "pdftotext", "pdfinfo"):
        return "inspect pdf"
    if first == "tr" or re.search(r"\.log\b", c.split("\n")[0]) and first in ("tail", "grep", "tr", "cat"):
        return "check run progress"
    if first in ("until", "sleep", "pgrep", "wait"):
        return "check run progress"
    if first in ("curl", "wget") or re.search(r"\bcurl\b", c[:200]):
        return "fetch from web"
    if first in ("pip", "apt-get", "apt", "uv") or re.search(r"\b(pip|apt-get)\b.*install", c[:200]):
        return "install packages"
    if first == "sed" and " -i" in c[:20] or first in ("cat", "tee") and re.match(r"(cat|tee)\s*>", c) or first == "cp" or first == "mv":
        return "edit files (shell)"
    if first in ("grep", "rg", "find"):
        return "search files"
    if first in ("cat", "head", "tail", "sed", "ls", "wc", "du", "column", "diff", "file", "stat"):
        return "view files"
    if first == "git":
        return "git"
    if first in ("docker", "df", "which", "env", "nproc", "free"):
        return "probe environment"
    if first == "for":
        return "loop (mixed)"
    if first == "node":
        return "run pipeline/script"
    return "other shell"


def action(name: str, inp: dict) -> str:
    """A coarse action label, comparable to Study 3's action classes."""
    if name == "Bash":
        return "Bash:" + bash_class(inp.get("command", ""))
    return name


def main() -> None:
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    calls, results, prompts = [], {}, []
    prompt_idx = 0
    for line in open(src):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        msg = r.get("message") or {}
        if r.get("type") == "user" and r.get("origin", {}).get("kind") == "human" and isinstance(msg.get("content"), str):
            prompt_idx += 1
            prompts.append({"idx": prompt_idx, "ts": r.get("timestamp"), "text": msg["content"][:300]})
        if r.get("type") == "user" and isinstance(msg.get("content"), list):
            for b in msg["content"]:
                if b.get("type") == "tool_result":
                    c = b.get("content")
                    text = c if isinstance(c, str) else "".join(x.get("text", "") for x in (c or []) if isinstance(x, dict))
                    results[b["tool_use_id"]] = {"chars": len(text), "is_error": bool(b.get("is_error"))}
        if r.get("type") == "assistant":
            u = msg.get("usage") or {}
            ctx = (u.get("input_tokens") or 0) + (u.get("cache_read_input_tokens") or 0) + (u.get("cache_creation_input_tokens") or 0)
            for b in msg.get("content") or []:
                if b.get("type") == "tool_use":
                    inp = b.get("input") or {}
                    calls.append({"id": b["id"], "ts": r.get("timestamp"), "msg": msg.get("id"), "prompt": prompt_idx,
                                  "tool": b["name"], "action": action(b["name"], inp),
                                  "input_chars": len(json.dumps(inp)), "context_tokens": ctx,
                                  "out_tokens": u.get("output_tokens") or 0,
                                  "cmd": (inp.get("command") or inp.get("file_path") or inp.get("query") or inp.get("pattern") or "")[:160]})
    for c in calls:
        c.update(results.get(c["id"], {"chars": None, "is_error": None}))
        c["result_chars"] = c.pop("chars")
    with open(out / "tool_calls.jsonl", "w") as f:
        for c in calls:
            f.write(json.dumps(c) + "\n")
    (out / "prompts.json").write_text(json.dumps(prompts, indent=1))

    # ---------------------------------------------------------------- summary
    n = len(calls)
    msgs = Counter(c["msg"] for c in calls)
    tools = Counter(c["tool"] for c in calls)
    acts = Counter(c["action"] for c in calls)
    errs = Counter(c["tool"] for c in calls if c["is_error"])
    ctx = [c["context_tokens"] for c in calls]
    L = ["# Meta-analysis: the agent's own tool calls in this session", "",
         f"Source: `{src.name}` ({len(prompts)} user prompts). {n} tool calls in {len(msgs)} assistant messages "
         f"({sum(1 for v in msgs.values() if v > 1)} messages issued parallel calls).", "",
         "## Which tools", "", "| tool | calls | share | errors |", "|---|---|---|---|"]
    L += [f"| {t} | {k} | {k / n:.1%} | {errs.get(t, 0)} |" for t, k in tools.most_common()]
    L += ["", "## What the calls did (Bash split by command type)", "", "| action | calls | share |", "|---|---|---|"]
    L += [f"| {a} | {k} | {k / n:.1%} |" for a, k in acts.most_common()]

    # next-action predictability, as in Study 3 (sequence of distinct decisions = assistant messages)
    seq = []
    for c in calls:
        if not seq or seq[-1][0] != c["msg"]:
            seq.append((c["msg"], c["action"], c["prompt"]))
    labels = [a for _, a, _ in seq]
    maj = Counter(labels).most_common(1)[0]
    rep = sum(1 for i in range(1, len(labels)) if labels[i] == labels[i - 1]) / max(1, len(labels) - 1)
    # bigram predictor trained on the first half, tested on the second half
    half = len(labels) // 2
    big = defaultdict(Counter)
    for a, b in zip(labels[:half], labels[1:half + 1]):
        big[a][b] += 1
    test = list(zip(labels[half:-1], labels[half + 1:]))
    bg = sum(1 for a, b in test if big[a] and big[a].most_common(1)[0][0] == b) / max(1, len(test))
    tools_only = [a.split(":")[0] for a in labels]
    rep_tool = sum(1 for i in range(1, len(tools_only)) if tools_only[i] == tools_only[i - 1]) / max(1, len(tools_only) - 1)
    L += ["", "## How predictable was the next action?", "",
          f"{len(labels)} decisions (assistant messages with ≥1 tool call; first action of each).", "",
          "| predictor | accuracy (action class) |", "|---|---|",
          f"| always `{maj[0]}` (majority) | {maj[1] / len(labels):.3f} |",
          f"| repeat previous action | {rep:.3f} |",
          f"| bigram (fit on first half, tested on second half) | {bg:.3f} |",
          f"| repeat previous *tool* (Bash/Read/Edit/… only) | {rep_tool:.3f} |"]
    trans = Counter(zip(tools_only, tools_only[1:]))
    L += ["", "Most common tool-to-tool transitions:", ""] + [f"- {a} → {b}: {k}" for (a, b), k in trans.most_common(10)]

    # context read per decision
    cs = sorted(ctx)
    q = lambda p: cs[min(len(cs) - 1, int(p * len(cs)))]  # noqa: E731
    in_chars = sorted(c["input_chars"] for c in calls)
    L += ["", "## How much the model read to make each decision", "",
          f"Context per tool-calling message (input + cached tokens): median {q(.5):,}, p90 {q(.9):,}, max {cs[-1]:,} tokens; "
          f"the tool's own input averaged {sum(in_chars) / n:,.0f} characters (median {in_chars[n // 2]:,}).",
          f"Total context tokens read across all tool-calling messages: {sum(c['context_tokens'] for c in {c['msg']: c for c in calls}.values()):,}.",
          "", "## Calls per user request (top 12)", "", "| prompt | calls | first words |", "|---|---|---|"]
    per = Counter(c["prompt"] for c in calls)
    for p, k in per.most_common(12):
        txt = next((x["text"] for x in prompts if x["idx"] == p), "")[:90].replace("|", "/").replace("\n", " ")
        L.append(f"| {p} | {k} | {txt} |")
    ts = [datetime.fromisoformat(c["ts"].replace("Z", "+00:00")) for c in calls if c["ts"]]
    L += ["", f"Span: {ts[0]:%Y-%m-%d %H:%M} → {ts[-1]:%Y-%m-%d %H:%M} UTC."]
    (out / "SUMMARY.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
