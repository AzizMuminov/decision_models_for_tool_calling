"""Could a decision model have made this agent's tool calls? Replays every tool decision in a transcript.

At each assistant message that calls a tool, the DM sees what the agent had just seen -- the user's
latest request and the recent conversation (tool calls and results, truncated) -- and chooses among the
tools the agent had available. Two conditions:
  context  -- state = request + recent history (tests narrowing the catalog / picking the tool)
  intent   -- the same, plus the sentence the agent wrote just before the call, if any (a stand-in for the
              "one universal tool" design: the LLM states what it wants, the DM maps it to a tool)
Scored against the tool the agent actually called (exact) and against an equivalence set (e.g. Bash
`grep` vs the Grep tool), with top-k recall from the DM's probabilities.

    source .or_env && python3 scripts/meta_replay.py <transcript.jsonl> results/meta [--limit N]
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dmtc.backends.dm import Choice, OpenRouterDM  # noqa: E402

DMS = {"msd1": "microsoft/microsoft-decision-1", "jev": "typesafe/jev-1.14"}

# The tools the agent had in this session (core tools, tools loadable through ToolSearch, MCP tools),
# each with a one-line description taken from its definition.
CATALOG = {
    "Bash": "Run a shell command (scripts, python, builds, git, grep/find, cat/head, curl, installs).",
    "Read": "Read a file from disk (text, image or PDF pages).",
    "Write": "Create a new file or overwrite a whole file with given content.",
    "Edit": "Replace an exact string in an existing file.",
    "Glob": "Find files by name pattern.",
    "Grep": "Search file contents with a regular expression.",
    "WebSearch": "Search the web and return result titles and URLs.",
    "WebFetch": "Fetch a URL and return its content.",
    "Monitor": "Watch a running background process or log until a condition is met.",
    "TaskStop": "Stop a running background task.",
    "TaskCreate": "Create an item in the task list to track progress.",
    "TaskUpdate": "Update the status of an item in the task list.",
    "TaskList": "List the items in the task list.",
    "SendUserFile": "Send a finished file (PDF, report, figure) to the user.",
    "AskUserQuestion": "Ask the user a multiple-choice question when a decision is theirs to make.",
    "ToolSearch": "Load the definitions of deferred tools so they can be called.",
    "Skill": "Load a packaged skill's instructions (e.g. data-visualization guidance).",
    "Agent": "Launch a sub-agent to do a task in parallel.",
    "Artifact": "Publish an HTML page as a shareable artifact.",
    "NotebookEdit": "Edit a cell of a Jupyter notebook.",
    "EnterPlanMode": "Switch to planning mode before making changes.",
    "ReadNotifications": "Read queued notifications for the session.",
    "ScheduleWakeup": "Schedule the next iteration of a self-paced loop.",
    "CronCreate": "Create a recurring scheduled job.",
    "PushNotification": "Send a push notification to the user's phone.",
    "SearchPlugins": "Search the plugin catalog.",
    "SuggestSkills": "Suggest installable skills to the user.",
    "ListAgents": "List other agents or sessions that can be messaged.",
    "SendMessage": "Send a message to another agent or session.",
    "ReportFindings": "Report code-review findings as a typed list.",
    "mcp__claude-code-remote__read_documentation": "Read documentation about the cloud environment (network, secrets, setup).",
    "mcp__claude-code-remote__create_session": "Start a new remote coding session.",
    "mcp__claude-code-remote__list_sessions": "List remote coding sessions.",
    "mcp__claude-code-remote__send_later": "Schedule a reminder message to this session.",
    "mcp__claude-code-remote__create_trigger": "Create a scheduled task.",
    "mcp__claude-code-remote__add_repo": "Attach a GitHub repository to the session.",
    "mcp__claude-code-remote__subscribe_pr_activity": "Subscribe to activity on a GitHub pull request.",
    "mcp__Claude_Docs__batch": "Create or edit a shared document.",
}

# Equivalent tools for each kind of action (the same intent can be served by more than one tool).
EQUIV = {
    "Bash:view files": {"Bash", "Read"},
    "Bash:search files": {"Bash", "Grep", "Glob"},
    "Bash:edit files (shell)": {"Bash", "Edit", "Write"},
    "Bash:edit files (inline python)": {"Bash", "Edit", "Write"},
    "Bash:fetch from web": {"Bash", "WebFetch"},
    "Bash:check run progress": {"Bash", "Monitor"},
    "Read": {"Read", "Bash"},
    "Write": {"Write", "Edit", "Bash"},
    "Edit": {"Edit", "Write", "Bash"},
    "WebFetch": {"WebFetch", "Bash"},
    "Monitor": {"Monitor", "Bash"},
}

INSTRUCTIONS = "Which tool should the coding assistant call next to continue the work?"


def clip(s: str, n: int) -> str:
    s = s.strip()
    return s if len(s) <= n else s[: n // 2] + " … " + s[-n // 2:]


def events(path: Path) -> list[dict]:
    """Main-thread events in order: human prompts, assistant text, tool calls, tool results."""
    ev, said = [], {}  # said: text the assistant wrote so far in each message (blocks arrive on separate lines)
    for line in open(path):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("isSidechain"):
            continue
        msg = r.get("message") or {}
        if r.get("type") == "user":
            c = msg.get("content")
            if isinstance(c, str) and r.get("origin", {}).get("kind") == "human":
                ev.append({"kind": "user", "text": c})
            elif isinstance(c, list):
                for b in c:
                    if b.get("type") == "tool_result":
                        x = b.get("content")
                        t = x if isinstance(x, str) else "".join(y.get("text", "") for y in (x or []) if isinstance(y, dict))
                        ev.append({"kind": "result", "text": t})
        elif r.get("type") == "assistant":
            blocks = [b for b in msg.get("content") or [] if b.get("type") in ("text", "tool_use")]
            first_tool = next((i for i, b in enumerate(blocks) if b["type"] == "tool_use"), None)
            mid = msg.get("id")
            if first_tool is not None:
                before = " ".join(b["text"] for b in blocks[:first_tool] if b["type"] == "text")
                intent = (said.get(mid, "") + " " + before).strip()
                ev.append({"kind": "decision", "msg": mid, "tool": blocks[first_tool]["name"], "intent": intent})
            said[mid] = (said.get(mid, "") + " " + " ".join(b["text"] for b in blocks if b["type"] == "text")).strip()
            for b in blocks:
                if b["type"] == "text" and b["text"].strip():
                    ev.append({"kind": "assistant", "text": b["text"]})
                elif b["type"] == "tool_use":
                    inp = b.get("input") or {}
                    ev.append({"kind": "call", "tool": b["name"],
                               "text": inp.get("command") or inp.get("file_path") or inp.get("query") or json.dumps(inp)})
    return ev


def build(ev: list[dict], k_last: int = 8) -> list[dict]:
    """One item per decision; the history excludes the decision message itself."""
    items, last_user, hist, prev_tool = [], "", [], None
    seen = set()
    for e in ev:
        if e["kind"] == "decision":
            if e["msg"] in seen:  # one assistant message can be split over several transcript lines
                continue
            seen.add(e["msg"])
            lines = [f"[{h['kind']}{(' ' + h['tool']) if h.get('tool') else ''}] {clip(h['text'], 700)}" for h in hist[-k_last:]]
            items.append({"msg": e["msg"], "gold": e["tool"], "prev": prev_tool, "intent": clip(e["intent"], 400),
                          "request": clip(last_user, 1500), "history": "\n".join(lines)})
            prev_tool = e["tool"]
            continue
        if e["kind"] == "user":
            last_user = e["text"]
        hist.append(e)
    return items


def main() -> None:
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
    items = build(events(src))[:limit]
    acts = {json.loads(l)["msg"]: json.loads(l)["action"] for l in open(out / "tool_calls.jsonl")}
    q = {"tool": Choice(INSTRUCTIONS, CATALOG)}
    dms = {k: OpenRouterDM(v) for k, v in DMS.items()}
    tasks = [(it, dm, cond) for it in items for dm in DMS for cond in ("context", "intent") if cond == "context" or it["intent"]]
    print(f"{len(items)} decisions, {sum(1 for it in items if it['intent'])} with intent text; {len(tasks)} DM requests", flush=True)

    def one(t):
        it, dm, cond = t
        state = {"user_request": it["request"], "recent_history": it["history"]}
        if cond == "intent":
            state["assistant_says"] = it["intent"]
        r = dms[dm].ask(state, q)
        a = r.answers.get("tool")
        probs = a.probabilities if a else {}
        rank = sorted(probs, key=probs.get, reverse=True)
        return {"msg": it["msg"], "dm": dm, "cond": cond, "gold": it["gold"], "prev": it["prev"],
                "action": acts.get(it["msg"]), "pred": a.value if a else None, "rank": rank[:10],
                "p_gold": probs.get(it["gold"]), "cost_usd": r.cost_usd, "error": r.error}

    with ThreadPoolExecutor(12) as pool, open(out / "replay_rows.jsonl", "w") as f:
        rows = []
        for k, row in enumerate(pool.map(one, tasks), 1):
            rows.append(row)
            f.write(json.dumps(row) + "\n")
            if k % 100 == 0:
                print(f"{k}/{len(tasks)} ${sum(x['cost_usd'] or 0 for x in rows):.3f}", flush=True)
    print(f"done; errors {sum(1 for r in rows if r['error'])}; cost ${sum(r['cost_usd'] or 0 for r in rows):.3f}")


if __name__ == "__main__":
    main()
