"""Local OpenRouter proxy for the sandboxed agent (runs as root; the agent never sees the API key).

Listens on 127.0.0.1. Forwards only chat-completion requests for allowed models, adds the Authorization
header, streams the response back unchanged, and logs one line per request.

    source .or_env && .venv/bin/python harness/llm_proxy.py --port 18080 --models deepseek/deepseek-v4.1-flash-1017
"""

from __future__ import annotations

import argparse
import json
import os
import time

import httpx
import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route

UPSTREAM = "https://openrouter.ai/api/v1"
ALLOWED: set[str] = set()
LOG = "/dev/null"
client = httpx.AsyncClient(timeout=httpx.Timeout(300, connect=30))  # honours the container's HTTPS proxy


def log(**kw) -> None:
    with open(LOG, "a") as f:
        f.write(json.dumps({"t": time.time(), **kw}) + "\n")


async def chat(request: Request):
    body = await request.body()
    try:
        model = json.loads(body).get("model")
    except json.JSONDecodeError:
        return JSONResponse({"error": "bad json"}, status_code=400)
    if model not in ALLOWED:
        log(event="blocked", model=model)
        return JSONResponse({"error": {"message": f"model {model} not allowed by the sandbox proxy"}}, status_code=403)
    headers = {"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}", "Content-Type": "application/json"}
    req = client.build_request("POST", f"{UPSTREAM}/chat/completions", content=body, headers=headers)
    resp = await client.send(req, stream=True)
    log(event="forward", model=model, status=resp.status_code)

    async def relay():
        try:
            async for chunk in resp.aiter_raw():
                yield chunk
        finally:
            await resp.aclose()

    passthrough = {k: v for k, v in resp.headers.items() if k.lower() in ("content-type", "cache-control")}
    return StreamingResponse(relay(), status_code=resp.status_code, headers=passthrough)


async def denied(request: Request):
    log(event="denied_path", path=request.url.path)
    return JSONResponse({"error": "not available through the sandbox proxy"}, status_code=404)


app = Starlette(routes=[Route("/api/v1/chat/completions", chat, methods=["POST"]),
                        Route("/{path:path}", denied, methods=["GET", "POST", "PUT", "DELETE"])])

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=18080)
    ap.add_argument("--models", required=True)
    ap.add_argument("--log", default="/dev/null")
    a = ap.parse_args()
    ALLOWED.update(a.models.split(","))
    LOG = a.log
    uvicorn.run(app, host="127.0.0.1", port=a.port, log_level="warning")
