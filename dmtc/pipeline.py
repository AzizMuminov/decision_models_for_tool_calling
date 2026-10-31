"""Ablation conditions: which parts of tool calling are handed to a decision model (DM).

Tool calling decomposes into three decisions, and each condition swaps a different subset of them
from the generative LLM to a DM:

    condition        | call or not? | which tool? | arguments
    -----------------+--------------+-------------+-----------
    llm_native       | LLM          | LLM         | LLM
    dm_gate          | DM           | LLM         | LLM
    dm_select        | LLM          | DM          | LLM (sees only the DM's tools)
    dm_gate_select   | DM           | DM          | LLM (sees only the DM's tools)
    dm_full          | DM           | DM          | DM (enum/bool by Choice, scalars by
                     |              |             |     Choice over spans extracted from the turn)
    cascade          | DM if confident, otherwise falls back to llm_native
    oracle_select    | LLM          | gold        | LLM   (upper bound for selection)

The DM's gate is one 4-way Choice (When2Call's label set) so that the same question is asked on
every benchmark; the gate is open when P(tool_call) >= gate_threshold.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from dmtc.backends.dm import Choice, DMBackend, DMResult, Noul
from dmtc.backends.llm import LLMBackend, LLMResult
from dmtc.data import WHEN2CALL_LABELS, Example
from dmtc.schema import NameMap, tool_card

CONDITIONS = ["llm_native", "dm_gate", "dm_select", "dm_gate_select", "dm_full", "cascade", "oracle_select"]

DECISION_CRITERIA = {
    "direct": "The assistant can fully answer from its own knowledge; no tool is needed.",
    "tool_call": "One or more of the available tools should be called now, and the request contains the "
                 "information the tool's required parameters need.",
    "request_for_info": "A tool is relevant, but a required parameter is missing or ambiguous, so the "
                        "assistant should ask the user for it first.",
    "cannot_answer": "None of the available tools can fulfil the request and it needs capabilities or "
                     "real-time information the assistant does not have.",
}
DECISION_INSTRUCTIONS = "Given the conversation and the available tools, what should the assistant do next?"
SELECT_INSTRUCTIONS = "Which available tool should be called to fulfil the user's latest request?"


@dataclass
class Prediction:
    calls: list[dict[str, dict[str, Any]]] = field(default_factory=list)
    text: str = ""
    # DM-produced decision label (When2Call label space) when the DM made the call/no-call decision
    dm_label: str | None = None
    p_tool_call: float | None = None
    gate_closed: bool | None = None  # None when the DM did not make the call/no-call decision
    decision_probs: dict[str, float] | None = None
    tool_probs: dict[str, float] | None = None
    selected_tools: list[str] | None = None
    unsupported_params: list[str] = field(default_factory=list)
    fell_back: bool = False
    llm_calls: int = 0
    dm_calls: int = 0
    llm_input_tokens: int = 0
    llm_output_tokens: int = 0
    dm_input_tokens: int = 0
    llm_cost_usd: float = 0.0  # provider-reported
    dm_cost_usd: float = 0.0  # provider-reported
    latency_s: float = 0.0  # sum of backend request times (the system's serial latency)
    wall_s: float = 0.0  # wall-clock for the whole example, incl. local work
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _state(ex: Example, tools: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {
        "conversation": [{"role": m["role"], "content": m["content"]} for m in ex.messages],
        "tools": [tool_card(t) for t in (ex.tools if tools is None else tools)],
    }


class Pipeline:
    def __init__(self, condition: str, llm: LLMBackend | None, dm: DMBackend | None, opts: dict[str, Any]):
        if condition not in CONDITIONS:
            raise ValueError(f"unknown condition {condition!r}; choose from {CONDITIONS}")
        self.condition = condition
        self.llm = llm
        self.dm = dm
        self.gate_threshold = opts.get("gate_threshold", 0.5)
        self.cascade_threshold = opts.get("cascade_threshold", 0.8)
        self.select_mode = opts.get("select_mode", "choice")  # "choice" | "multi_noul"
        self.top_k = opts.get("top_k", 1)
        self.max_candidates = opts.get("max_candidates", 120)
        # Used only after the DM opened the gate: the call/no-call decision is already made.
        self.fill_system = opts.get(
            "fill_system",
            "A routing step has already decided that the tool(s) below must be called for the user's "
            "request. Call them with the correct arguments. Do not answer in text.",
        )
        # Used when only the tool list was narrowed (dm_select, oracle_select): the LLM keeps the
        # call/no-call decision, so it must not be told to call. None = no extra instruction, same
        # as llm_native.
        self.restrict_system = opts.get("restrict_system")

    # ---- bookkeeping -------------------------------------------------------------------------
    @staticmethod
    def _add_llm(p: Prediction, r: LLMResult) -> None:
        p.llm_calls += 1
        p.llm_input_tokens += r.input_tokens
        p.llm_output_tokens += r.output_tokens
        p.llm_cost_usd += r.cost_usd
        p.latency_s += r.latency_s
        if r.error:
            p.errors.append(f"llm: {r.error}")

    @staticmethod
    def _add_dm(p: Prediction, r: DMResult) -> None:
        p.dm_calls += 1
        p.dm_input_tokens += r.input_tokens
        p.dm_cost_usd += r.cost_usd
        p.latency_s += r.latency_s
        if r.error:
            p.errors.append(f"dm: {r.error}")

    # ---- stages ------------------------------------------------------------------------------
    def _llm_native(self, ex: Example, p: Prediction, tools: list[dict[str, Any]] | None = None,
                    system: str | None = None) -> None:
        assert self.llm is not None, f"condition {self.condition} needs an LLM backend"
        r = self.llm.call(ex.messages, ex.tools if tools is None else tools, system=system, example=ex)
        self._add_llm(p, r)
        p.calls, p.text = r.calls, r.text

    def _decision_questions(self, ex: Example, with_gate: bool, with_select: bool) -> dict[str, Any]:
        qs: dict[str, Any] = {}
        if with_gate:
            qs["decision"] = Choice(DECISION_INSTRUCTIONS, DECISION_CRITERIA)
        if with_select and ex.tools:
            names = NameMap(ex.tools)
            cards = {names.to_safe[t["name"]]: tool_card(t) for t in ex.tools}
            if self.select_mode == "multi_noul":
                for label, card in cards.items():
                    qs[f"need__{label}"] = Noul(
                        f"The user's request requires calling the tool `{card['name']}`.",
                        true=json.dumps(card),
                    )
            else:
                qs["tool"] = Choice(SELECT_INSTRUCTIONS, cards)
        return qs

    def _apply_decisions(self, ex: Example, p: Prediction, r: DMResult) -> None:
        if "decision" in r.answers:
            a = r.answers["decision"]
            p.decision_probs = a.probabilities
            p.p_tool_call = a.probabilities.get("tool_call", 0.0)
            p.dm_label = a.value
        names = NameMap(ex.tools)
        if "tool" in r.answers:
            a = r.answers["tool"]
            p.tool_probs = {names.orig(k): v for k, v in a.probabilities.items()}
            ranked = sorted(p.tool_probs, key=p.tool_probs.get, reverse=True)
            p.selected_tools = ranked[: self.top_k]
        need = {k[len("need__"):]: a.value for k, a in r.answers.items() if k.startswith("need__")}
        if need:
            p.tool_probs = {names.orig(k): v for k, v in need.items()}
            ranked = sorted(p.tool_probs, key=p.tool_probs.get, reverse=True)
            p.selected_tools = [t for t in ranked if p.tool_probs[t] >= 0.5] or ranked[:1]

    def _gate_open(self, p: Prediction) -> bool:
        return (p.p_tool_call or 0.0) >= self.gate_threshold

    def _fill_with_llm(self, ex: Example, p: Prediction, gate_decided: bool) -> None:
        chosen = [t for t in ex.tools if t["name"] in set(p.selected_tools or [])]
        self._llm_native(ex, p, tools=chosen, system=self.fill_system if gate_decided else self.restrict_system)

    # ---- DM-only argument filling ------------------------------------------------------------
    def _candidates(self, ex: Example) -> list[str]:
        text = ex.user_text
        cands: list[str] = []
        cands += re.findall(r"[\"“]([^\"”]{1,80})[\"”]", text)
        # single quotes only at word boundaries, so apostrophes ("I'm ... what's") don't pair up
        cands += re.findall(r"(?:^|(?<=\s))[‘']([^'’\n]{1,80})['’](?=[\s.,;:!?)]|$)", text)
        cands += re.findall(r"-?\d+(?:\.\d+)?", text)
        # "4%" -> also offer 0.04 (rates are usually passed as fractions)
        cands += [repr(float(x) / 100) for x in re.findall(r"(-?\d+(?:\.\d+)?)\s?%", text)]
        words = re.findall(r"[\w@./:#+-]+", text)
        for n in (1, 2, 3, 4):
            cands += [" ".join(words[i:i + n]).strip(".,;:!?") for i in range(len(words) - n + 1)]
        seen, out = set(), []
        for c in cands:
            if c and c not in seen:
                seen.add(c)
                out.append(c)
        return out[: self.max_candidates]

    @staticmethod
    def _coerce(value: str, typ: str) -> Any:
        try:
            if typ == "integer":
                return int(float(value))
            if typ in ("number", "float"):
                return float(value)
        except ValueError:
            pass
        return value

    def _dm_fill(self, ex: Example, p: Prediction, tool: dict[str, Any]) -> dict[str, Any]:
        assert self.dm is not None
        props = tool.get("parameters", {}).get("properties", {})
        required = set(tool.get("parameters", {}).get("required", []))
        cands = self._candidates(ex)
        qs: dict[str, Any] = {}
        plan: dict[str, tuple[str, Any]] = {}
        for i, (pname, spec) in enumerate(props.items()):
            typ = {"dict": "object", "float": "number", "int": "integer", "bool": "boolean",
                   "str": "string", "tuple": "array", "list": "array"}.get(spec.get("type", "string"), spec.get("type"))
            desc = f"{pname}: {spec.get('description', '')}"
            key = f"arg{i}"
            if pname not in required:
                qs[f"{key}__present"] = Noul(f"The user's request specifies a value for the parameter `{desc}`.")
            if "enum" in spec:
                qs[key] = Choice(f"Which value should the parameter `{desc}` take?",
                                 {f"v{j}": str(v) for j, v in enumerate(spec["enum"])})
                plan[pname] = ("enum", (key, spec["enum"]))
            elif typ == "boolean":
                qs[key] = Choice(f"Which value should the boolean parameter `{desc}` take?",
                                 {"true": "true", "false": "false"})
                plan[pname] = ("bool", key)
            elif typ in ("string", "integer", "number") and cands:
                # the schema default is a legitimate option the DM can pick without generating it
                pc = cands + [str(spec["default"])] if spec.get("default") not in (None, "") and \
                    str(spec["default"]) not in cands else cands
                qs[key] = Choice(f"Which span of the user's request is the value of parameter `{desc}`?",
                                 {f"c{j}": c for j, c in enumerate(pc)})
                plan[pname] = ("span", (key, typ, pc))
            else:
                p.unsupported_params.append(pname)
        if not qs:
            return {}
        r = self.dm.ask({"conversation": _state(ex)["conversation"], "tool": tool_card(tool)}, qs)
        self._add_dm(p, r)
        args: dict[str, Any] = {}
        for pname, (kind, info) in plan.items():
            key = info[0] if isinstance(info, tuple) else info
            present = r.answers.get(f"{key}__present")
            if present is not None and present.value < 0.5:
                continue
            a = r.answers.get(key)
            if a is None:
                continue
            if kind == "enum":
                args[pname] = info[1][int(a.value[1:])]
            elif kind == "bool":
                args[pname] = a.value == "true"
            else:
                args[pname] = self._coerce(info[2][int(a.value[1:])], info[1])
        return args

    # ---- entry point -------------------------------------------------------------------------
    def run(self, ex: Example) -> Prediction:
        p = Prediction()
        t0 = time.perf_counter()
        c = self.condition

        if c == "llm_native":
            self._llm_native(ex, p)

        elif c == "oracle_select":
            gold = {next(iter(g)) for g in (ex.ground_truth or [])}
            if not gold and ex.meta.get("target_tool"):  # When2Call tool_call items
                gold = {json.loads(ex.meta["target_tool"])["name"]}
            p.selected_tools = [t["name"] for t in ex.tools if t["name"] in gold] or [t["name"] for t in ex.tools]
            self._fill_with_llm(ex, p, gate_decided=False)

        elif c in ("dm_gate", "dm_select", "dm_gate_select", "dm_full", "cascade"):
            assert self.dm is not None, f"condition {c} needs a DM backend"
            with_gate = c != "dm_select"
            with_select = c != "dm_gate"
            r = self.dm.ask(_state(ex), self._decision_questions(ex, with_gate, with_select))
            self._add_dm(p, r)
            self._apply_decisions(ex, p, r)
            if with_gate:
                p.gate_closed = not self._gate_open(p)

            if c == "cascade":
                conf = min(max(p.p_tool_call or 0, 1 - (p.p_tool_call or 0)),
                           max((p.tool_probs or {"": 1.0}).values()))
                if conf < self.cascade_threshold:
                    p.fell_back = True
                    p.gate_closed = None  # the LLM made the decision
                    self._llm_native(ex, p)
                elif self._gate_open(p):
                    self._fill_with_llm(ex, p, gate_decided=True)
            elif c == "dm_select":
                self._fill_with_llm(ex, p, gate_decided=False)
            elif not self._gate_open(p):
                pass  # DM closed the gate: no call, p.dm_label carries the decision
            elif c == "dm_gate":
                self._llm_native(ex, p)
            elif c == "dm_gate_select":
                self._fill_with_llm(ex, p, gate_decided=True)
            elif c == "dm_full" and p.selected_tools:
                tool = next(t for t in ex.tools if t["name"] == p.selected_tools[0])
                p.calls = [{tool["name"]: self._dm_fill(ex, p, tool)}]

        p.wall_s = time.perf_counter() - t0
        return p


def when2call_label(p: Prediction, judge: LLMBackend | None, ex: Example) -> tuple[str, bool, float]:
    """Maps a prediction onto When2Call's four labels. Returns (label, used_judge, judge_cost_usd)."""
    if p.calls:
        return "tool_call", False, 0.0
    if p.gate_closed:
        # The DM decided not to call; its most likely non-tool label is the answer. (The argmax can be
        # tool_call with P(tool_call) below gate_threshold, so take the argmax over the other three.)
        probs = {k: v for k, v in (p.decision_probs or {}).items() if k != "tool_call"}
        return (max(probs, key=probs.get) if probs else "cannot_answer"), False, 0.0
    if judge is None:
        return "direct", False, 0.0
    prompt = (
        "Classify the assistant reply to the user's request into exactly one category.\n"
        f"Categories: {', '.join(l for l in WHEN2CALL_LABELS if l != 'tool_call')}\n"
        "- direct: answers the question directly\n"
        "- request_for_info: asks the user for missing information\n"
        "- cannot_answer: says it cannot help or lacks the capability\n\n"
        f"User request: {ex.user_text}\nAssistant reply: {p.text}\n\nReply with the category name only."
    )
    r = judge.complete(prompt, example=ex)
    for label in ("request_for_info", "cannot_answer", "direct"):
        if label in r.text:
            return label, True, r.cost_usd
    return "direct", True, r.cost_usd
