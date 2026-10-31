"""MCP stdio server exposing the fictional company catalog to a coding agent, in one of three modes.

  CATALOG_MODE=native     every catalog tool, with its full description and schema
  CATALOG_MODE=filtered   only the tools named in CATALOG_FILTER (comma-separated), chosen beforehand by a DM
  CATALOG_MODE=universal  two tools: find_tool(intent) -- a decision model ranks the catalog against the
                          stated intent and returns the best matches with their schemas -- and
                          call_tool(name, arguments)
  CATALOG_INDEX=1         (universal only) list the services the catalog covers in find_tool's description

Responses come from a per-task world file (CATALOG_WORLD): a list of rules {tool, match, response}; the first
rule whose `match` values all occur (case-insensitively) in the call's arguments answers. Anything else gets
an empty list for list/search tools and NOT_FOUND otherwise, like a real service. Every call is appended to
CATALOG_LOG as JSON.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

import mcp.server.stdio
import mcp.types as types
from mcp.server.lowlevel import Server

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from catalog import EXTRA_SERVICES, SERVICES, catalog  # noqa: E402

MODE = os.environ.get("CATALOG_MODE", "native")
CAT = {t["name"]: t for t in catalog(int(os.environ.get("CATALOG_SIZE", "100")))}
WORLD = json.loads(Path(os.environ["CATALOG_WORLD"]).read_text()) if os.environ.get("CATALOG_WORLD") else []
LOG = Path(os.environ.get("CATALOG_LOG", "/dev/null"))
DM_MODEL = os.environ.get("CATALOG_DM", "microsoft/microsoft-decision-1")
TOPK = int(os.environ.get("CATALOG_TOPK", "3"))


def log(**kw) -> None:
    with open(LOG, "a") as f:
        f.write(json.dumps({"t": time.time(), **kw}) + "\n")


def answer(name: str, args: dict) -> dict:
    blob = json.dumps(args).lower()
    for rule in WORLD:
        if rule["tool"] == name and all(str(v).lower() in blob for v in rule.get("match", {}).values()):
            return {"result": rule["response"], "fetched_at": "2026-10-27T07:00:00Z", "source": CAT[name]["service"]}
    action = name.split("_", 1)[1]
    if action.startswith(("list", "search")):
        return {"result": [], "fetched_at": "2026-10-27T07:00:00Z", "source": CAT[name]["service"], "next_cursor": None}
    return {"error": "NOT_FOUND", "message": f"No matching record for {json.dumps(args)}", "source": CAT[name]["service"]}


def short_card(t: dict) -> str:
    """The first line of the description plus parameter names: what the DM sees per option (fits a 32k context)."""
    params = ", ".join(t["inputSchema"]["properties"]) or "none"
    return f"{t['description'].splitlines()[0]} Parameters: {params}."


_dm = None


def find(intent: str) -> list[dict]:
    global _dm
    from dmtc.backends.dm import Choice, OpenRouterDM

    _dm = _dm or OpenRouterDM(DM_MODEL)
    q = "Which tool from the catalog best serves what the agent wants to do?"

    def ask(names):
        r = _dm.ask({"what_the_agent_wants": intent}, {"tool": Choice(q, {n: short_card(CAT[n]) for n in names})})
        a = r.answers.get("tool")
        return (a.probabilities if a else {}), r.cost_usd, r.error

    names, cost, err = list(CAT), 0.0, None
    if len(names) > 250:  # the DM accepts at most 255 options: shortlist 5 per half, then decide among the 10
        short = []
        for chunk in (names[: len(names) // 2], names[len(names) // 2:]):
            p, c, e = ask(chunk)
            cost, err = cost + c, err or e
            short += sorted(p, key=p.get, reverse=True)[:5]
        names = short
    probs, c, e = ask(names)
    cost, err = cost + c, err or e
    ranked = sorted(probs, key=probs.get, reverse=True)[:TOPK]
    log(kind="find_tool", intent=intent, ranked=ranked, probs=[round(probs[n], 4) for n in ranked], dm_cost=cost, error=err)
    return [{"name": n, "description": CAT[n]["description"], "parameters": CAT[n]["inputSchema"]} for n in ranked]


FIND_DESC = ("Find a tool in Quillmark's internal tool catalog. Describe in plain words what you need to look up or do "
             "(for example: 'get the production value of a configuration key'). Returns the best-matching tools with "
             "their descriptions and parameter schemas. Then run one with call_tool.")
if os.environ.get("CATALOG_INDEX") == "1":
    FIND_DESC += "\n\nThe catalog covers these internal services:\n" + "\n".join(
        f"  - {b}: {w}" for svc, (b, w, _, _) in {**SERVICES, **EXTRA_SERVICES}.items() if any(t["service"] == svc for t in CAT.values()))
CALL_DESC = "Run a catalog tool found with find_tool. Pass its exact name and an arguments object matching its parameters."

def tool_list() -> list[types.Tool]:
    if MODE == "universal":
        return [types.Tool(name="find_tool", description=FIND_DESC,
                           inputSchema={"type": "object", "properties": {"intent": {"type": "string", "description": "What you need, in plain words."}}, "required": ["intent"]}),
                types.Tool(name="call_tool", description=CALL_DESC,
                           inputSchema={"type": "object", "properties": {"name": {"type": "string"}, "arguments": {"type": "object"}}, "required": ["name", "arguments"]})]
    names = os.environ.get("CATALOG_FILTER", "").split(",") if MODE == "filtered" else list(CAT)
    return [types.Tool(name=n, description=CAT[n]["description"], inputSchema=CAT[n]["inputSchema"]) for n in names if n in CAT]


async def on_list_tools(ctx, params) -> types.ListToolsResult:
    return types.ListToolsResult(tools=tool_list())


async def on_call_tool(ctx, params: types.CallToolRequestParams) -> types.CallToolResult:
    name, arguments = params.name, dict(params.arguments or {})
    if MODE == "universal" and name == "find_tool":
        out = await asyncio.to_thread(find, arguments.get("intent", ""))
    elif MODE == "universal" and name == "call_tool":
        target, args = arguments.get("name", ""), arguments.get("arguments") or {}
        if isinstance(args, str):  # some models send the arguments object as a JSON string
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        out = answer(target, args) if target in CAT else {"error": "UNKNOWN_TOOL", "message": f"No tool named {target}; use find_tool."}
        log(kind="call", tool=target, args=args, ok="error" not in out)
    else:
        out = answer(name, arguments) if name in CAT else {"error": "UNKNOWN_TOOL"}
        log(kind="call", tool=name, args=arguments, ok="error" not in out)
    return types.CallToolResult(content=[types.TextContent(type="text", text=json.dumps(out, indent=1))])


server = Server("corp", on_list_tools=on_list_tools, on_call_tool=on_call_tool)


async def main() -> None:
    async with mcp.server.stdio.stdio_server() as (r, w):
        await server.run(r, w, server.create_initialization_options())


if __name__ == "__main__":
    if os.environ.get("CATALOG_HTTP_PORT"):  # sandbox mode: served over local HTTP, so the agent can't read our files
        import uvicorn

        uvicorn.run(server.streamable_http_app(), host="127.0.0.1", port=int(os.environ["CATALOG_HTTP_PORT"]), log_level="warning")
    else:
        asyncio.run(main())
