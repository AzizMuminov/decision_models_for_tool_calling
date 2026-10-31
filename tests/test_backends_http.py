"""Backends against a local mock server: checks request serialization and response parsing offline."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from dmtc.backends.dm import Choice, Noul, TypeSafeDM
from dmtc.backends.llm import AnthropicLLM, OpenAICompatLLM

TOOLS = [{"name": "math.hypot", "description": "Hypotenuse", "parameters": {
    "type": "dict", "properties": {"x": {"type": "float"}, "y": {"type": "float"}}, "required": ["x", "y"]}}]
REQUESTS: list[tuple[str, dict]] = []


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        REQUESTS.append((self.path, body))
        if self.path.endswith("/v1/systemone"):
            answers = {}
            for name, q in body["questions"].items():
                if q["type"] == "noul":
                    answers[name] = {"type": "noul", "noul": 0.8}
                else:
                    labels = list(q["criteria"])
                    probs = {l: (0.7 if i == 0 else 0.3 / max(len(labels) - 1, 1)) for i, l in enumerate(labels)}
                    answers[name] = {"type": "choice", "choice": labels[0], "confidence": 0.7, "probabilities": probs}
            out = {"model": body["model"], "usage": {"input_tokens": 42, "output_tokens": 0}, "answers": answers}
        elif self.path.endswith("/v1/messages"):
            out = {"id": "msg_1", "type": "message", "role": "assistant", "model": body["model"],
                   "content": [{"type": "text", "text": "ok"},
                               {"type": "tool_use", "id": "tu_1", "name": "math_hypot", "input": {"x": 3, "y": 4}}],
                   "stop_reason": "tool_use", "stop_sequence": None, "usage": {"input_tokens": 10, "output_tokens": 5}}
        else:  # /v1/chat/completions
            out = {"id": "c", "object": "chat.completion", "created": 0, "model": body["model"],
                   "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
                       "role": "assistant", "content": None, "tool_calls": [{"id": "t", "type": "function", "function": {
                           "name": "math_hypot", "arguments": "{\"x\": 3, \"y\": 4}"}}]}}],
                   "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}
        data = json.dumps(out).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture(scope="module")
def server():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


def test_typesafe_dm(server, monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "test")
    dm = TypeSafeDM(model="jev-latest", base_url=server)
    r = dm.ask({"conversation": "hi"}, {"gate": Noul("Call a tool?", true="yes"),
                                        "tool": Choice("Which?", {"a": {"name": "a.b"}, "b": None})})
    assert r.error is None, r.error
    assert r.answers["gate"].value == pytest.approx(0.8)
    assert r.answers["tool"].value == "a" and r.input_tokens == 42
    path, body = REQUESTS[-1]
    assert path == "/v1/systemone" and body["model"] == "jev-latest"
    assert body["questions"]["gate"] == {"type": "noul", "instructions": "Call a tool?", "criteria": {"true": "yes"}}


def test_anthropic_llm(server, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", server)
    llm = AnthropicLLM(model="claude-opus-5-5", effort="low")
    r = llm.call([{"role": "system", "content": "sys"}, {"role": "user", "content": "3,4?"}], TOOLS)
    assert r.error is None, r.error
    assert r.calls == [{"math.hypot": {"x": 3, "y": 4}}]
    _, body = REQUESTS[-1]
    assert body["system"] == "sys" and body["tools"][0]["name"] == "math_hypot"
    assert body["tools"][0]["input_schema"]["properties"]["x"]["type"] == "number"
    assert body["output_config"] == {"effort": "low"} and "temperature" not in body


def test_openai_compat_llm(server):
    llm = OpenAICompatLLM(model="qwen", base_url=server + "/v1")
    r = llm.call([{"role": "user", "content": "3,4?"}], TOOLS)
    assert r.error is None, r.error
    assert r.calls == [{"math.hypot": {"x": 3, "y": 4}}]
