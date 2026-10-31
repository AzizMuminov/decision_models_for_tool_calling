from typing import Any

import pytest

from dmtc.backends.dm import Answer, Choice, DMBackend, DMResult, Noul
from dmtc.backends.llm import OracleLLM
from dmtc.data import load_bfcl
from dmtc.eval.bfcl_checker import score_bfcl
from dmtc.pipeline import Pipeline
from dmtc.schema import NameMap, anthropic_tools, to_json_schema


def test_schema_conversion():
    s = to_json_schema({"type": "dict", "properties": {"x": {"type": "float"}, "p": {"type": "tuple", "items": {"type": "integer"}},
                                                       "a": {"type": "any"}}, "required": ["x"]})
    assert s["type"] == "object"
    assert s["properties"]["x"]["type"] == "number"
    assert s["properties"]["p"]["type"] == "array"
    assert "type" not in s["properties"]["a"]


def test_null_fields_are_dropped():
    from jsonschema import Draft202012Validator

    s = to_json_schema({"type": "object", "properties": {"c": {"type": "string", "enum": None, "items": None},
                                                         "other_tools_param": None},
                        "required": ["c"], "additionalProperties": None})
    Draft202012Validator.check_schema(s)
    assert s["properties"]["c"] == {"type": "string"} and "additionalProperties" not in s


def test_invalid_property_names_roundtrip():
    from dmtc.schema import openai_tools

    tools = [{"name": "cotizar", "parameters": {"type": "dict", "properties": {"año_vehiculo": {"type": "integer"}},
                                                "required": ["año_vehiculo"]}}]
    names = NameMap(tools)
    s = openai_tools(tools, names)[0]["function"]["parameters"]
    assert list(s["properties"]) == ["ano_vehiculo"] and s["required"] == ["ano_vehiculo"]
    assert names.orig_args("cotizar", {"ano_vehiculo": 2020}) == {"año_vehiculo": 2020}


def test_name_map_roundtrip():
    tools = [{"name": "math.hypot"}, {"name": "math_hypot"}]
    m = NameMap(tools)
    assert len(set(m.to_safe.values())) == 2
    for t in tools:
        assert m.orig(m.to_safe[t["name"]]) == t["name"]
    assert all("." not in t["name"] for t in anthropic_tools([{"name": "a.b", "parameters": {}}], NameMap([{"name": "a.b"}])))


@pytest.mark.parametrize("category", ["simple_python", "multiple", "parallel", "parallel_multiple", "live_simple",
                                      "live_multiple", "live_parallel", "live_parallel_multiple"])
def test_oracle_scores_perfectly(category):
    # Gold answers that BFCL's own checker rejects (gold omits a schema-required param, or names a
    # param that is not in the schema). They are unreachable for every system; see README.
    unreachable = {"live_simple_106-63-0", "live_simple_112-68-0", "live_multiple_507-149-4",
                   "live_multiple_862-181-3", "live_multiple_964-207-0"}
    llm = OracleLLM()
    for ex in load_bfcl(category):
        r = llm.call(ex.messages, ex.tools, example=ex)
        assert score_bfcl(ex, r.calls)["correct"] or ex.id in unreachable, (ex.id, r.calls)


def test_wrong_call_is_rejected():
    ex = load_bfcl("simple_python")[0]
    (fname, _), = ex.ground_truth[0].items()
    assert not score_bfcl(ex, [{fname: {}}])["correct"]
    assert not score_bfcl(ex, [])["correct"]


class ScriptedDM(DMBackend):
    """Answers each question with a fixed function of the question, for pipeline unit tests."""

    model = "scripted"

    def __init__(self, fn):
        self.fn = fn

    def ask(self, state: Any, questions):
        return DMResult(answers={k: self.fn(k, q) for k, q in questions.items()})


def test_gate_closed_means_no_llm_call():
    ex = load_bfcl("simple_python")[0]
    dm = ScriptedDM(lambda k, q: Answer("choice", "cannot_answer", {"cannot_answer": 0.9, "tool_call": 0.1}, 0.9)
                    if k == "decision" else Answer("choice", next(iter(q.criteria)), {next(iter(q.criteria)): 1.0}, 1.0))
    p = Pipeline("dm_gate_select", OracleLLM(), dm, {}).run(ex)
    assert p.calls == [] and p.llm_calls == 0 and p.dm_label == "cannot_answer"


def test_dm_full_maps_span_and_enum_answers():
    ex = load_bfcl("simple_python")[0]  # calculate_triangle_area(base=10, height=5, unit="units")
    pipe = Pipeline("dm_full", None, None, {})
    cands = pipe._candidates(ex)
    tool = ex.tools[0]
    gold = {k: v[0] for k, v in ex.ground_truth[0][tool["name"]].items() if v[0] != ""}

    def fn(k, q):
        if isinstance(q, Noul):  # "is optional param X specified?" -> yes iff it is in the gold call
            return Answer("noul", float(q.instructions.split("`")[1].split(":")[0] in gold))
        if k == "decision":
            return Answer("choice", "tool_call", {"tool_call": 1.0}, 1.0)
        if k == "tool":
            label = next(iter(q.criteria))
            return Answer("choice", label, {label: 1.0}, 1.0)
        # pick the candidate matching the gold value for this parameter
        pname = q.instructions.split("`")[1].split(":")[0]
        want = str(gold.get(pname))
        label = next((lab for lab, c in q.criteria.items() if c == want), next(iter(q.criteria)))
        return Answer("choice", label, {label: 1.0}, 1.0)

    pipe.dm = ScriptedDM(fn)
    p = pipe.run(ex)
    assert set(gold) <= set(p.calls[0][tool["name"]]), (p.calls, cands)
    assert score_bfcl(ex, p.calls)["correct"], p.calls


def test_typesafe_questions_build_without_network():
    from typesafe_sdk import Choice as TChoice, Noul as TNoul

    TNoul(instructions="x", criteria={"true": "y"})
    TChoice(instructions="x", criteria={"a": {"name": "a.b", "description": "d"}, "b": None})


# ---- regression tests for the review fixes ---------------------------------------------------

class RecordingLLM(OracleLLM):
    def __init__(self):
        self.systems = []

    def call(self, messages, tools, system=None, example=None):
        self.systems.append(system)
        return super().call(messages, tools, system=system, example=example)


def _first_label_dm(decision_probs):
    def fn(k, q):
        if k == "decision":
            best = max(decision_probs, key=decision_probs.get)
            return Answer("choice", best, decision_probs, decision_probs[best])
        label = next(iter(q.criteria))
        return Answer("choice", label, {label: 1.0}, 1.0)
    return ScriptedDM(fn)


def test_select_only_conditions_do_not_tell_llm_to_call():
    # dm_select / oracle_select narrow the tool list but leave call-or-not to the LLM.
    ex = load_bfcl("irrelevance")[0]
    for cond, dm in (("dm_select", _first_label_dm({"tool_call": 1.0})), ("oracle_select", None)):
        llm = RecordingLLM()
        Pipeline(cond, llm, dm, {}).run(ex)
        assert llm.systems == [None], cond
    llm = RecordingLLM()
    Pipeline("dm_gate_select", llm, _first_label_dm({"tool_call": 0.9, "direct": 0.1}), {}).run(ex)
    assert llm.systems and "must be called" in llm.systems[0]


def test_closed_gate_label_when_argmax_is_tool_call():
    from dmtc.data import load_when2call
    from dmtc.pipeline import when2call_label

    ex = load_when2call(__import__("pathlib").Path("data/When2Call/data/test/when2call_test_mcq.jsonl"))[0]
    probs = {"tool_call": 0.4, "request_for_info": 0.35, "cannot_answer": 0.2, "direct": 0.05}
    p = Pipeline("dm_gate_select", OracleLLM(), _first_label_dm(probs), {"gate_threshold": 0.5}).run(ex)
    assert p.gate_closed and p.calls == []
    assert when2call_label(p, judge=None, ex=ex) == ("request_for_info", False, 0.0)


def test_bfcl_accepts_int_for_float_and_rejects_wrong_value():
    # Find a gold call with an integral float-typed argument, e.g. {"radius": 5.0}.
    for ex in load_bfcl("simple_python"):
        (fname, args), = OracleLLM().call(ex.messages, ex.tools, example=ex).calls[0].items()
        props = ex.tools[0]["parameters"]["properties"]
        fl = [k for k, v in args.items() if props.get(k, {}).get("type") == "float" and float(v).is_integer()]
        if fl:
            break
    k = fl[0]
    assert score_bfcl(ex, [{fname: {**args, k: int(args[k])}}])["correct"], (ex.id, k)
    assert not score_bfcl(ex, [{fname: {**args, k: float(args[k]) + 1234.5}}])["correct"]


def test_all_tool_schemas_are_valid_json_schema():
    from jsonschema import Draft202012Validator

    from dmtc.data import BFCL_CATEGORIES, load_when2call

    exs = [e for c in BFCL_CATEGORIES for e in load_bfcl(c)]
    exs += load_when2call(__import__("pathlib").Path("data/When2Call/data/test/when2call_test_mcq.jsonl"))
    for ex in exs:
        for t in anthropic_tools(ex.tools, NameMap(ex.tools)):
            Draft202012Validator.check_schema(t["input_schema"])
            assert t["input_schema"]["type"] == "object", (ex.id, t["name"])


def test_metrics_against_known_values():
    from dmtc.eval.metrics import auroc, ece, holm, mcnemar

    assert auroc([0.9, 0.8, 0.1], [True, True, False]) == 1.0
    assert auroc([0.5, 0.5], [True, False]) == 0.5
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert ece([1.0, 1.0], [True, False]) == pytest.approx(0.5)
    m = mcnemar([True] * 10 + [False] * 10, [True] * 10 + [True] * 10)
    assert m["b_only"] == 10 and m["a_only"] == 0 and m["p_value"] < 0.01
