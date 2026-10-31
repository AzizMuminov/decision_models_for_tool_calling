"""Decision-model backends.

A decision model answers typed questions about a `state` in one forward pass and generates no
tokens. We use three primitives, mirroring the TypeSafe System One API (also served by Liquid d1):

  Noul   - yes/no, returns P(yes)
  Choice - one label out of a named set, returns per-label probabilities
  Score  - ordinal rubric, returns per-level probabilities and the expected score

Every question in one `ask` call is answered from a single request, so a pipeline stage that
needs several decisions (gate + tool choice + every argument) costs one round trip.
"""

from __future__ import annotations

import math
import os
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Noul:
    instructions: str
    true: str | None = None
    false: str | None = None


@dataclass
class Choice:
    instructions: str
    criteria: dict[str, Any]  # label -> description (text or JSON)


@dataclass
class Score:
    instructions: str
    criteria: list[str]


Question = Noul | Choice | Score


@dataclass
class Answer:
    kind: str  # "noul" | "choice" | "score"
    value: Any  # P(yes) for noul, label for choice, expected score for score
    probabilities: dict[str, float] = field(default_factory=dict)
    confidence: float = 0.0


@dataclass
class DMResult:
    answers: dict[str, Answer] = field(default_factory=dict)
    latency_s: float = 0.0
    input_tokens: int = 0
    model: str = ""
    error: str | None = None
    cost_usd: float = 0.0  # provider-reported cost, when the API returns one


class DMBackend:
    name = "base"
    model = ""

    def ask(self, state: Any, questions: dict[str, Question]) -> DMResult:
        raise NotImplementedError


def split_trivial(questions: dict[str, Question]) -> tuple[dict[str, Question], dict[str, Answer]]:
    """A Choice with one option has nothing to decide (and some DMs reject it: "2-255 options").
    Returns (questions to send, answers decided locally)."""
    send, local = {}, {}
    for k, q in questions.items():
        if isinstance(q, Choice) and len(q.criteria) == 1:
            (label,) = q.criteria
            local[k] = Answer("choice", label, {label: 1.0}, 1.0)
        else:
            send[k] = q
    return send, local


class TypeSafeDM(DMBackend):
    """Hosted decision models behind the System One API.

    - TypeSafe Jev:  base_url https://api.typesafe.ai, model e.g. "jev-latest"
    - Liquid d1:     base_url https://api.liquid.ai/decisions, model e.g. "d1:free" / "d1-ultra"
      (Liquid documents the endpoint as POST https://api.liquid.ai/decisions/v1/systemone and the
      TypeSafe SDK as its client; the SDK appends /v1/systemone to base_url.)
    """

    name = "typesafe"

    def __init__(self, model: str, base_url: str | None = None, api_key_env: str = "TYPESAFE_API_KEY",
                 timeout: float = 30.0, **_: Any):
        from typesafe_sdk import TypeSafeClient

        self.client = TypeSafeClient(api_key=os.environ.get(api_key_env), base_url=base_url, model=model,
                                     timeout=timeout)
        self.model = model

    def ask(self, state, questions):
        from typesafe_sdk import Choice as TChoice, Noul as TNoul, Score as TScore

        questions, local = split_trivial(questions)
        if not questions:
            return DMResult(answers=local, model=self.model)
        wire: dict[str, Any] = {}
        for name, q in questions.items():
            if isinstance(q, Noul):
                crit = {k: v for k, v in (("true", q.true), ("false", q.false)) if v is not None} or None
                wire[name] = TNoul(instructions=q.instructions, criteria=crit)
            elif isinstance(q, Choice):
                wire[name] = TChoice(instructions=q.instructions, criteria=q.criteria)
            elif isinstance(q, Score):
                wire[name] = TScore(instructions=q.instructions, criteria=q.criteria)
        t0 = time.perf_counter()
        try:
            resp = self.client.system_one(state=state, questions=wire)
        except Exception as e:  # noqa: BLE001 - recorded per example
            return DMResult(latency_s=time.perf_counter() - t0, model=self.model, error=f"{type(e).__name__}: {e}")
        out = DMResult(latency_s=time.perf_counter() - t0, input_tokens=resp.usage.input_tokens or 0, model=resp.model)
        for name, a in resp.nouls.items():
            out.answers[name] = Answer("noul", a.noul, {"yes": a.noul, "no": 1 - a.noul}, max(a.noul, 1 - a.noul))
        for name, a in resp.choices.items():
            out.answers[name] = Answer("choice", a.choice, dict(a.probabilities), a.confidence)
        for name, a in resp.scores.items():
            out.answers[name] = Answer("score", a.score, {str(k): v for k, v in a.probabilities.items()}, a.confidence)
        out.answers.update(local)
        return out


class OpenRouterDM(DMBackend):
    """Any decision model in OpenRouter's catalog, via POST /api/alpha/decisions.

    Same question/answer shapes as System One; `model` is the full catalog id, e.g.
    "typesafe/jev-1.14" or "microsoft/microsoft-decision-1". Pin a versioned id, not a "~" alias.
    """

    name = "openrouter"

    def __init__(self, model: str, api_key_env: str = "OPENROUTER_API_KEY",
                 base_url: str = "https://openrouter.ai/api/alpha/decisions", timeout: float = 60.0,
                 max_retries: int = 4, **_: Any):
        import httpx

        self.model = model
        self.url = base_url
        self.max_retries = max_retries
        self.client = httpx.Client(timeout=timeout, headers={
            "Authorization": f"Bearer {os.environ.get(api_key_env, '')}", "Content-Type": "application/json"})

    @staticmethod
    def _wire(q: Question) -> dict[str, Any]:
        if isinstance(q, Noul):
            crit = {k: v for k, v in (("true", q.true), ("false", q.false)) if v is not None}
            return {"type": "noul", "instructions": q.instructions, **({"criteria": crit} if crit else {})}
        if isinstance(q, Choice):
            return {"type": "choice", "instructions": q.instructions, "criteria": q.criteria}
        return {"type": "score", "instructions": q.instructions, "criteria": q.criteria}

    def ask(self, state, questions):
        questions, local = split_trivial(questions)
        if not questions:
            return DMResult(answers=local, model=self.model)
        body = {"model": self.model, "state": state, "questions": {k: self._wire(q) for k, q in questions.items()}}
        t0 = time.perf_counter()
        resp, err = None, None
        for attempt in range(self.max_retries + 1):
            try:
                resp = self.client.post(self.url, json=body)
            except Exception as e:  # noqa: BLE001 - network errors are retried, then recorded
                err = f"{type(e).__name__}: {e}"
                resp = None
            if resp is not None and resp.status_code < 400:
                break
            if resp is not None:
                err = f"HTTP {resp.status_code}: {resp.text[:300]}"
                if resp.status_code == 403 and "limit" in resp.text.lower():
                    from dmtc.backends.llm import OutOfCredits

                    raise OutOfCredits(f"key spending limit reached: {err}")
                if resp.status_code == 402:
                    if "in_flight" in resp.text and attempt < self.max_retries:
                        time.sleep(min(10 * (attempt + 1), 60))  # reservations clear as requests settle
                        continue
                    from dmtc.backends.llm import OutOfCredits

                    raise OutOfCredits(err)
                if resp.status_code not in (408, 429) and resp.status_code < 500:
                    break  # 4xx other than timeout/rate limit will not succeed on retry
            time.sleep(min(2 ** attempt, 20))
        latency = time.perf_counter() - t0
        if resp is None or resp.status_code >= 400:
            return DMResult(latency_s=latency, model=self.model, error=err)
        data = resp.json()
        usage = data.get("usage") or {}
        out = DMResult(latency_s=latency, input_tokens=usage.get("input_tokens") or 0,
                       model=data.get("model", self.model), cost_usd=float(usage.get("cost") or 0.0))
        for name, a in (data.get("answers") or {}).items():
            t = a.get("type")
            if t == "noul":
                p = float(a["noul"])
                out.answers[name] = Answer("noul", p, {"yes": p, "no": 1 - p}, max(p, 1 - p))
            elif t == "choice":
                probs = {k: float(v) for k, v in (a.get("probabilities") or {a["choice"]: 1.0}).items()}
                out.answers[name] = Answer("choice", a["choice"], probs, float(a.get("confidence", probs.get(a["choice"], 0))))
            elif t == "score":
                probs = {str(k): float(v) for k, v in (a.get("probabilities") or {}).items()}
                out.answers[name] = Answer("score", float(a["score"]), probs, float(a.get("confidence", 0)))
        missing = set(questions) - set(out.answers)
        if missing:
            out.error = f"missing answers: {sorted(missing)}"
        out.answers.update(local)
        return out


_TOKEN = re.compile(r"[a-z0-9]+")


def _tokens(x: Any) -> list[str]:
    return _TOKEN.findall(str(x).lower())


def _cosine(a: Counter, b: Counter) -> float:
    num = sum(a[t] * b[t] for t in a.keys() & b.keys())
    den = math.sqrt(sum(v * v for v in a.values())) * math.sqrt(sum(v * v for v in b.values()))
    return num / den if den else 0.0


class LexicalDM(DMBackend):
    """Non-neural control with the same interface: bag-of-words cosine between state and criteria.

    Answers "is the decision model doing more than lexical retrieval?" (cf. the finding that tool
    *selection* is largely retrieval while *abstention* is not). Also lets the whole harness run
    offline for smoke tests.
    """

    name = "lexical"
    model = "lexical-bow"

    def __init__(self, temperature: float = 0.05, gate_threshold: float = 0.12, **_: Any):
        self.temperature = temperature
        self.gate_threshold = gate_threshold

    def _softmax(self, scores: dict[str, float]) -> dict[str, float]:
        m = max(scores.values())
        exp = {k: math.exp((v - m) / self.temperature) for k, v in scores.items()}
        z = sum(exp.values())
        return {k: v / z for k, v in exp.items()}

    def _gate(self, s: Counter, tools: list[Any], labels: list[str]) -> Answer:
        # Retrieval gate: P(tool_call) grows with the best lexical match to any tool. It cannot tell
        # direct from request_for_info, so all remaining mass goes to cannot_answer.
        sim = max((_cosine(s, Counter(_tokens(t))) for t in tools), default=0.0)
        p = 1 / (1 + math.exp(-(sim - self.gate_threshold) / self.temperature))
        probs = {k: 0.0 for k in labels}
        probs["tool_call"], probs["cannot_answer"] = p, 1 - p
        best = max(probs, key=probs.get)
        return Answer("choice", best, probs, probs[best])

    def ask(self, state, questions):
        t0 = time.perf_counter()
        conv = state.get("conversation", state) if isinstance(state, dict) else state
        if isinstance(conv, list):  # chat messages: use the text only, not the role/content keys
            conv = " ".join(str(m.get("content", "")) if isinstance(m, dict) else str(m) for m in conv)
        s = Counter(_tokens(conv))
        tools = state.get("tools") if isinstance(state, dict) else None
        out = DMResult(model=self.model)
        for name, q in questions.items():
            if isinstance(q, Choice) and "tool_call" in q.criteria and tools is not None:
                out.answers[name] = self._gate(s, tools, list(q.criteria))
            elif isinstance(q, Choice) and q.criteria:
                sims = {k: _cosine(s, Counter(_tokens(f"{k} {v}"))) for k, v in q.criteria.items()}
                probs = self._softmax(sims)
                best = max(probs, key=probs.get)
                out.answers[name] = Answer("choice", best, probs, probs[best])
            elif isinstance(q, Noul):
                sim = _cosine(s, Counter(_tokens(q.true or q.instructions)))
                p = 1 / (1 + math.exp(-(sim - self.gate_threshold) / self.temperature))
                out.answers[name] = Answer("noul", p, {"yes": p, "no": 1 - p}, max(p, 1 - p))
            elif isinstance(q, Score):
                sims = {str(i): _cosine(s, Counter(_tokens(c))) for i, c in enumerate(q.criteria)}
                probs = self._softmax(sims)
                exp_score = sum(int(k) * v for k, v in probs.items())
                out.answers[name] = Answer("score", exp_score, probs, max(probs.values()))
        out.latency_s = time.perf_counter() - t0
        return out


def make_dm(cfg: dict[str, Any]) -> DMBackend:
    kind = cfg["kind"]
    args = {k: v for k, v in cfg.items() if k != "kind"}
    if kind == "typesafe":
        return TypeSafeDM(**args)
    if kind == "lexical":
        return LexicalDM(**args)
    if kind == "openrouter":
        return OpenRouterDM(**args)
    raise ValueError(f"unknown dm kind {kind!r}")
