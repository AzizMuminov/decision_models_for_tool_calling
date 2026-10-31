"""Generative LLM backends with native tool calling.

All backends return calls in BFCL's format: a list of {original_func_name: {param: value}}.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any

from dmtc.data import Example
from dmtc.schema import NameMap, anthropic_tools, openai_tools


class OutOfCredits(RuntimeError):
    """HTTP 402 from the provider: stop the run instead of recording it as a wrong answer."""


@dataclass
class LLMResult:
    calls: list[dict[str, dict[str, Any]]] = field(default_factory=list)
    text: str = ""
    latency_s: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""
    error: str | None = None
    cost_usd: float = 0.0  # provider-reported cost (OpenRouter returns usage.cost)


def _split_system(messages: list[dict[str, Any]], extra_system: str | None) -> tuple[str, list[dict[str, Any]]]:
    system = [m["content"] for m in messages if m["role"] == "system"]
    if extra_system:
        system.append(extra_system)
    rest = [m for m in messages if m["role"] != "system"]
    return "\n\n".join(system), rest


class LLMBackend:
    name = "base"
    model = ""

    def call(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        system: str | None = None,
        example: Example | None = None,
    ) -> LLMResult:
        raise NotImplementedError

    def complete(self, prompt: str, example: Example | None = None) -> LLMResult:
        """Plain text completion, used for label classification (When2Call judge / MCQ baseline)."""
        return self.call([{"role": "user", "content": prompt}], [], example=example)


class AnthropicLLM(LLMBackend):
    name = "anthropic"

    def __init__(self, model: str = "claude-opus-5-5", effort: str | None = "low", max_tokens: int = 16000, **_: Any):
        import anthropic

        self.client = anthropic.Anthropic()
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens

    def call(self, messages, tools, system=None, example=None):
        names = NameMap(tools)
        sys_text, msgs = _split_system(messages, system)
        kwargs: dict[str, Any] = {"model": self.model, "max_tokens": self.max_tokens, "messages": msgs}
        if sys_text:
            kwargs["system"] = sys_text
        if tools:
            kwargs["tools"] = anthropic_tools(tools, names)
            kwargs["tool_choice"] = {"type": "auto"}
        if self.effort:
            kwargs["output_config"] = {"effort": self.effort}
        t0 = time.perf_counter()
        try:
            resp = self.client.messages.create(**kwargs)
        except Exception as e:  # noqa: BLE001 - recorded per example, run continues
            return LLMResult(latency_s=time.perf_counter() - t0, model=self.model, error=f"{type(e).__name__}: {e}")
        out = LLMResult(
            latency_s=time.perf_counter() - t0,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
            model=resp.model,
        )
        if resp.stop_reason == "refusal":
            out.error = "refusal"
        for block in resp.content:
            if block.type == "tool_use":
                out.calls.append({names.orig(block.name): names.orig_args(block.name, dict(block.input))})
            elif block.type == "text":
                out.text += block.text
        return out


class OpenAICompatLLM(LLMBackend):
    """Any OpenAI-compatible chat endpoint: vLLM, SGLang, Ollama, OpenRouter, Together, ..."""

    name = "openai_compat"

    def __init__(self, model: str, base_url: str | None = None, api_key_env: str = "OPENAI_API_KEY",
                 temperature: float | None = 0.0, max_tokens: int = 4096,
                 extra_body: dict[str, Any] | None = None, max_retries: int = 4, **_: Any):
        import openai

        self.client = openai.OpenAI(base_url=base_url, api_key=os.environ.get(api_key_env, "EMPTY"),
                                    max_retries=max_retries)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        # provider-specific fields, e.g. OpenRouter {"reasoning": {"effort": "low"}}
        self.extra_body = extra_body or {}

    def call(self, messages, tools, system=None, example=None):
        names = NameMap(tools)
        sys_text, msgs = _split_system(messages, system)
        if sys_text:
            msgs = [{"role": "system", "content": sys_text}] + msgs
        kwargs: dict[str, Any] = {"model": self.model, "messages": msgs, "max_tokens": self.max_tokens}
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if tools:
            kwargs["tools"] = openai_tools(tools, names)
            kwargs["tool_choice"] = "auto"
        if self.extra_body:
            kwargs["extra_body"] = self.extra_body
        t0 = time.perf_counter()
        for attempt in range(12):
            try:
                resp = self.client.chat.completions.create(**kwargs)
                break
            except Exception as e:  # noqa: BLE001
                if getattr(e, "status_code", None) == 403 and "limit" in str(e).lower():
                    raise OutOfCredits(f"key spending limit reached: {str(e)[:200]}") from e
                if getattr(e, "status_code", None) != 402:
                    return LLMResult(latency_s=time.perf_counter() - t0, model=self.model,
                                     error=f"{type(e).__name__}: {e}")
                # OpenRouter reserves worst-case cost for in-flight requests; that 402 clears as
                # they settle. Any other 402, or one that persists, means the account is empty.
                if "in_flight" not in str(e) or attempt == 11:
                    raise OutOfCredits(str(e)[:300]) from e
                time.sleep(min(10 * (attempt + 1), 60))  # OpenRouter suggests Retry-After up to ~120 s
        if not resp.choices:
            return LLMResult(latency_s=time.perf_counter() - t0, model=self.model, error="no choices in response")
        msg = resp.choices[0].message
        out = LLMResult(
            latency_s=time.perf_counter() - t0,
            input_tokens=getattr(resp.usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(resp.usage, "completion_tokens", 0) or 0,
            model=resp.model or self.model,
            text=msg.content or "",
            cost_usd=float(getattr(resp.usage, "cost", 0.0) or 0.0),
        )
        for tc in msg.tool_calls or []:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                out.error = "invalid_tool_json"
                args = {}
            out.calls.append({names.orig(tc.function.name): names.orig_args(tc.function.name, args)})
        return out


class OracleLLM(LLMBackend):
    """Returns the gold BFCL call(s) restricted to the tools it is shown.

    Not a baseline: it isolates the decision layer (gate/selection) from argument filling, and
    serves as the end-to-end sanity check that a perfect pipeline scores 100% with the checker.
    """

    name = "oracle"
    model = "oracle"

    def call(self, messages, tools, system=None, example=None):
        if example is None or not tools or not example.should_call:
            return LLMResult(text="(no call)", model=self.model)
        shown = {t["name"] for t in tools}
        if not example.ground_truth:
            # Relevance-only items (live_relevance, When2Call): any call to a shown tool is right.
            target = example.meta.get("target_tool")
            target = json.loads(target)["name"] if target else None
            name = target if target in shown else tools[0]["name"]
            return LLMResult(calls=[{name: {}}], model=self.model)

        def pick(options: Any, required: bool) -> tuple[bool, Any]:
            # BFCL lists acceptable values per param; "" means "may be omitted". Omitting an optional
            # param is always accepted, while some listed values (e.g. []) are not.
            if not isinstance(options, list):
                return True, unwrap(options)
            values = [o for o in options if o != ""]
            if not values or ("" in options and not required):
                return False, None
            return True, unwrap(values[0])

        def unwrap(val: Any) -> Any:
            # Acceptable-value lists are nested inside dict-typed values (and dicts inside lists).
            if isinstance(val, dict):
                out = {}
                for k, o in val.items():
                    keep, v = pick(o, required=False)
                    if keep:
                        out[k] = v
                return out
            if isinstance(val, list):
                return [unwrap(x) for x in val]
            return val

        docs = {t["name"]: t for t in tools}
        calls = []
        for gt in example.ground_truth:
            (fname, params), = gt.items()
            if fname not in shown:
                continue
            required = set(docs[fname].get("parameters", {}).get("required", []))
            args = {}
            for k, v in params.items():
                keep, val = pick(v, required=k in required)
                if keep:
                    args[k] = val
            calls.append({fname: args})
        return LLMResult(calls=calls, model=self.model)

    def complete(self, prompt, example=None):
        return LLMResult(text=(example.label if example and example.label else "direct"), model=self.model)


def make_llm(cfg: dict[str, Any]) -> LLMBackend:
    kind = cfg["kind"]
    args = {k: v for k, v in cfg.items() if k != "kind"}
    if kind == "anthropic":
        return AnthropicLLM(**args)
    if kind == "openai_compat":
        return OpenAICompatLLM(**args)
    if kind == "oracle":
        return OracleLLM()
    raise ValueError(f"unknown llm kind {kind!r}")
