"""MCP server exposing the real tool catalog (12 MCP servers, 613 tools) to a coding agent under one selection mechanism.

REAL_MODE
  native   every tool's real definition, up front (Cursor, Cline, opencode; Claude Code with tool search off)
  cc       Claude Code: the names of all deferred tools plus each server's one-line purpose are listed in
           tool_search's description; tool_search takes `select:<names>` or keywords (see selectors_real.py)
  bm25     tool_search runs BM25 (Anthropic tool_search_tool_bm25, Codex CLI search_tool_bm25)
  embed    tool_search runs embedding similarity (Goose vector router, RAG-MCP)
  llm      tool_search asks an LLM router (Goose LLM strategy)
  dm       tool_search asks the decision model (server first, then tool)
  random   control: tool_search returns 5 random tools
  groups   Copilot's virtual tools: one activate_<group> tool per server (servers over 40 tools are split
           into parts); activating returns the group's tool definitions

Every mode except native reaches the actual tools through call_tool(name, arguments), so the services and
their answers are identical across modes. The search modes return the top 5 tools with their full definitions.
bm25/embed/llm/dm list the same service overview in tool_search's description (Anthropic recommends naming
the tool categories; Codex shows namespace descriptions).

Env: REAL_MODE, REAL_WORLD (task world file), REAL_LOG (call log), REAL_HTTP_PORT, REAL_SERVERS (optional subset),
REAL_EXTRA=1 (add the 18 extra servers: 1,076 tools).
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

import mcp.types as types
from mcp.server.lowlevel import Server

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from real_sim import SERVER_INFO, answer, load_catalog  # noqa: E402
from selectors_real import Selectors  # noqa: E402

MODE = os.environ.get("REAL_MODE", "native")
SERVERS = [s for s in os.environ.get("REAL_SERVERS", "").split(",") if s] or None
TOOLS = load_catalog(SERVERS, extra=os.environ.get("REAL_EXTRA") == "1")
CAT = {t["name"]: t for t in TOOLS}
WORLD = json.loads(Path(os.environ["REAL_WORLD"]).read_text()) if os.environ.get("REAL_WORLD") else []
LOG = Path(os.environ.get("REAL_LOG", "/dev/null"))
SEL = Selectors(TOOLS)
# oracle ranker: the tools the task needs (REAL_TASK points at its task.json)
ORACLE = ([n for g in json.loads(Path(os.environ["REAL_TASK"]).read_text())["needs"] for n in g] if os.environ.get("REAL_TASK") else [])
GROUP_MAX = 40
COMPACT = os.environ.get("REAL_COMPACT") == "1"  # search returns short hits; full schema via tool_describe (like Hermes's bridge)


def log(**kw) -> None:
    with open(LOG, "a") as f:
        f.write(json.dumps({"t": time.time(), **kw}) + "\n")


def definition(n: str) -> dict:
    return {"name": n, "description": CAT[n]["description"], "input_schema": CAT[n]["inputSchema"]}


SERVICES = "\n".join(f"  - {s}: {SERVER_INFO[s]}" for s in sorted({t["server"] for t in TOOLS}))
CALL = types.Tool(name="call_tool", description="Run a tool you found with tool_search (or loaded by activating a group). Pass its exact name and an arguments object matching its input schema.",
                  inputSchema={"type": "object", "properties": {"name": {"type": "string"}, "arguments": {"type": "object"}}, "required": ["name", "arguments"]})


def search_tool() -> types.Tool:
    if MODE == "cc":
        desc = ("Fetches full schema definitions for deferred tools so they can be called with call_tool. The deferred tools are "
                "listed below by name only; until fetched, only the name is known.\n\nQuery forms:\n"
                "- \"select:github_get_job_logs,slack_post_message\" - fetch these exact tools by name\n"
                "- \"loki logs query\" - keyword search, up to max_results best matches\n"
                "- \"+slack send\" - require \"slack\" in the name, rank by remaining terms\n\n"
                f"Connected MCP servers:\n{SERVICES}\n\nDeferred tools:\n" + "\n".join(t["name"] for t in TOOLS))
    else:
        desc = ("Search the catalog of tools from the connected MCP servers. Describe in plain words what you need to look up or "
                "do; returns the best-matching tools (up to 5) with their full definitions. Then run one with call_tool.\n\n"
                f"Connected MCP servers:\n{SERVICES}")
    return types.Tool(name="tool_search", description=desc, inputSchema={"type": "object", "properties": {
        "query": {"type": "string", "description": "What you need, in plain words" + (" or a select:/keyword query" if MODE == "cc" else "")},
        "max_results": {"type": "integer", "description": "Maximum results (default 5)"}}, "required": ["query"]})


def groups() -> dict[str, list[str]]:
    out = {}
    for s in sorted({t["server"] for t in TOOLS}):
        names = sorted(t["name"] for t in TOOLS if t["server"] == s)
        parts = [names[i:i + GROUP_MAX] for i in range(0, len(names), GROUP_MAX)]
        for j, p in enumerate(parts):
            out[f"activate_{s}" + (f"_part{j + 1}" if len(parts) > 1 else "")] = p
    return out


GROUPS = groups() if MODE == "groups" else {}


DESCRIBE = types.Tool(name="tool_describe", description="Get the full definition (description and input schema) of a tool found with tool_search, before calling it with call_tool.",
                      inputSchema={"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]})


def hit(n: str) -> dict:
    req = (CAT[n]["inputSchema"] or {}).get("required") or []
    return {"name": n, "description": CAT[n]["description"][:400], "required_params": req}


def tool_list() -> list[types.Tool]:
    if MODE == "native":
        return [types.Tool(name=t["name"], description=t["description"], inputSchema=t["inputSchema"]) for t in TOOLS]
    if MODE == "groups":
        out = []
        for g, names in GROUPS.items():
            s = CAT[names[0]]["server"]
            part = f" (part {g.rsplit('part', 1)[1]} of {sum(1 for x in GROUPS if x.startswith(f'activate_{s}'))})" if "_part" in g else ""
            out.append(types.Tool(name=g, inputSchema={"type": "object", "properties": {}},
                                  description=f"Activate a group of {len(names)} tools from {SERVER_INFO[s]}{part}. Returns their full definitions; "
                                              f"run them with call_tool. Tools in this group: {', '.join(names)}"))
        return out + [CALL]
    return [search_tool(), DESCRIBE, CALL] if COMPACT else [search_tool(), CALL]


async def on_list_tools(ctx, params) -> types.ListToolsResult:
    return types.ListToolsResult(tools=tool_list())


def run(name: str, args: dict) -> object:
    if name in CAT:
        out, ok = answer(name, args, WORLD, CAT[name]["inputSchema"])
        log(kind="call", tool=name, args=args, ok=ok)
        return out
    log(kind="call", tool=name, args=args, ok=False, unknown=True)
    return {"error": "UNKNOWN_TOOL", "message": f"No tool named {name}." + ("" if MODE == "native" else " Use tool_search to find tools.")}


async def on_call_tool(ctx, params: types.CallToolRequestParams) -> types.CallToolResult:
    name, args = params.name, dict(params.arguments or {})
    if MODE != "native" and name == "tool_search":
        q, k = str(args.get("query", "")), int(args.get("max_results") or 5)
        t0 = time.time()
        ranked, cost = await asyncio.to_thread(SEL.search, MODE, q, k)
        log(kind="search", query=q, ranked=ranked, cost=cost, latency=round(time.time() - t0, 2))
        if COMPACT:
            out = {"matches": [hit(n) for n in ranked], "next": "call tool_describe(name) for the schema, then call_tool"} if ranked else {"matches": [], "message": "No matching tools."}
        else:
            out = {"tools": [definition(n) for n in ranked]} if ranked else {"tools": [], "message": "No matching tools."}
    elif COMPACT and name == "tool_describe":
        n = str(args.get("name", ""))
        log(kind="describe", tool=n)
        out = definition(n) if n in CAT else {"error": f"No tool named {n}; use tool_search."}
    elif MODE == "groups" and name in GROUPS:
        log(kind="activate", group=name)
        out = {"tools": [definition(n) for n in GROUPS[name]]}
    elif MODE != "native" and name == "call_tool":
        target, a = args.get("name", ""), args.get("arguments") or {}
        if isinstance(a, str):
            try:
                a = json.loads(a)
            except json.JSONDecodeError:
                a = {}
        out = run(target, a)
    else:
        out = run(name, args)
    text = out if isinstance(out, str) else json.dumps(out, indent=1)
    return types.CallToolResult(content=[types.TextContent(type="text", text=text)])


server = Server("corp", on_list_tools=on_list_tools, on_call_tool=on_call_tool)

if __name__ == "__main__":
    import uvicorn

    app = server.streamable_http_app()

    async def rank(request):
        """Ranking endpoint for Hermes's own tool-search bridge (run_hermes.py, hermes_dmbridge): POST {query, limit}."""
        from starlette.responses import JSONResponse

        body = await request.json()
        q, k = str(body.get("query", "")), int(body.get("limit") or 5)
        t0 = time.time()
        mech = os.environ.get("REAL_RANKER", "dm")
        if mech == "oracle":  # upper bound: the task's needed tools first (ordered by BM25 fit to the query), then BM25
            SEL.bm25("warm", 1)
            sc = SEL._bm25.scores(q)
            need = [n for n in ORACLE if n in CAT]
            ranked = sorted(need, key=lambda n: -sc.get(n, 0)) + [n for n in sorted(sc, key=lambda n: -sc[n]) if n not in need and sc[n] > 0]
            ranked, cost = ranked[:k], 0.0
        elif mech in ("dm_franken", "dm_ctx"):
            ranked, extra, cost = await asyncio.to_thread(getattr(SEL, mech), q, k)
            log(kind="search", query=q, ranked=ranked + extra, described=ranked, names_only=extra, none=not (ranked or extra), cost=cost,
                latency=round(time.time() - t0, 2), via="hermes_bridge", mech=mech)
            return JSONResponse({"ranked": ranked, "names": extra, "none": not (ranked or extra)})
        elif mech in ("dm_rec", "dm_rec510"):
            ranked, extra, cost = await asyncio.to_thread(getattr(SEL, mech), q, k)
            log(kind="search", query=q, ranked=ranked + extra, described=ranked, names_only=extra, cost=cost,
                latency=round(time.time() - t0, 2), via="hermes_bridge", mech=mech)
            return JSONResponse({"ranked": ranked, "names": extra})
        elif mech == "dm_menu":
            ranked, menu, cost = await asyncio.to_thread(SEL.dm_menu, q, k)
            log(kind="search", query=q, ranked=ranked, menu={s: len(v) for s, v in menu.items()}, cost=cost,
                latency=round(time.time() - t0, 2), via="hermes_bridge", mech=mech)
            return JSONResponse({"ranked": ranked, "menu": menu})
        else:
            ranked, cost = await asyncio.to_thread(SEL.search, mech, q, k)
        log(kind="search", query=q, ranked=ranked, cost=cost, latency=round(time.time() - t0, 2), via="hermes_bridge", mech=mech)
        return JSONResponse({"ranked": ranked})

    app.add_route("/rank", rank, methods=["POST"])
    token = os.environ.get("REAL_TOKEN")
    if token:  # the agent's shell can reach this port; without the header (known only to the harness process) it gets 401
        from starlette.middleware.base import BaseHTTPMiddleware
        from starlette.responses import PlainTextResponse

        async def guard(request, call_next):
            if request.headers.get("x-corp-token") != token:
                return PlainTextResponse("unauthorized", status_code=401)
            return await call_next(request)

        app.add_middleware(BaseHTTPMiddleware, dispatch=guard)
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ["REAL_HTTP_PORT"]), log_level="warning")
