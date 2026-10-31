"""Profiles the benchmarks and computes design ceilings for the ablation conditions.

    .venv/bin/python scripts/analyze_benchmarks.py  # writes results/benchmark_analysis.{md,json}

Ceilings use gold answers inside our own question design, so they bound what any DM can reach
under these conditions (e.g. dm_full can only be right if every gold value is an enum/bool or an
extractable span of the user turn).
"""

from __future__ import annotations

import json
import re
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dmtc.backends.dm import Answer, Choice, DMBackend, DMResult, LexicalDM, Noul  # noqa: E402
from dmtc.backends.llm import OracleLLM  # noqa: E402
from dmtc.data import BFCL_AST_CATEGORIES, BFCL_CATEGORIES, WHEN2CALL_LABELS, load_bfcl, load_when2call  # noqa: E402
from dmtc.eval.bfcl_checker import score_bfcl  # noqa: E402
from dmtc.eval.metrics import auroc, macro_f1  # noqa: E402
from dmtc.pipeline import Pipeline, _state  # noqa: E402
from dmtc.schema import NameMap  # noqa: E402

W2C = Path("data/When2Call/data/test/when2call_test_mcq.jsonl")


def std(s: str) -> str:
    # BFCL's string standardisation (ast_checker.standardize_string)
    return re.sub(r"[ \,\.\/\-\_\*\^]", "", str(s)).lower().replace("'", '"')


def gold_values(ex):
    """{func: {param: acceptable values}} for the first gold call."""
    (fname, params), = ex.ground_truth[0].items()
    return fname, params


class GoldArgDM(DMBackend):
    """Answers dm_full's questions with gold: tool, presence, enum, bool and the matching span."""

    model = "gold"

    def __init__(self, ex):
        self.ex = ex
        self.fname, self.params = gold_values(ex)

    def _matches(self, cand: str, accept: list) -> bool:
        for a in accept:
            if a == "":
                continue
            if isinstance(a, (int, float)) and not isinstance(a, bool):
                try:
                    if float(cand) == float(a):
                        return True
                except ValueError:
                    pass
            elif isinstance(a, str) and std(cand) == std(a):
                return True
        return False

    def ask(self, state, questions):
        out = DMResult()
        tool = state.get("tool") if isinstance(state, dict) else None
        props = list(tool["parameters"]) if tool else []  # tool_card keeps param order
        for k, q in questions.items():
            if k == "decision":
                out.answers[k] = Answer("choice", "tool_call", {"tool_call": 1.0}, 1.0)
            elif k == "tool":
                safe = NameMap(self.ex.tools).to_safe[self.fname]
                out.answers[k] = Answer("choice", safe, {safe: 1.0}, 1.0)
            else:
                i = int(re.match(r"arg(\d+)", k).group(1))
                pname = props[i]
                accept = self.params.get(pname, [""])
                if isinstance(q, Noul):  # optional param present?
                    out.answers[k] = Answer("noul", float(not ("" in accept)))
                    continue
                bools = {str(a).lower() for a in accept if isinstance(a, bool)}
                hit = next((lab for lab, c in q.criteria.items() if self._matches(str(c), accept)
                            or str(c).lower() in bools), None)
                if hit is None:
                    # no candidate matches: pick anything, the call will be wrong (what we want to count)
                    hit = next(iter(q.criteria))
                out.answers[k] = Answer("choice", hit, {hit: 1.0}, 1.0)
        return out


def arg_kind(spec: dict, value) -> str:
    if "enum" in spec:
        return "enum"
    t = spec.get("type", "string")
    return {"boolean": "bool", "bool": "bool", "string": "string", "integer": "int", "int": "int",
            "float": "number", "number": "number", "array": "array", "tuple": "array", "list": "array",
            "dict": "object", "object": "object"}.get(t, t)


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def main() -> None:
    report: dict = {"bfcl": {}, "when2call": {}}
    lex = LexicalDM()
    gate_scores, gate_y = [], []

    for cat in BFCL_CATEGORIES:
        exs = load_bfcl(cat)
        ntools = [len(e.tools) for e in exs]
        row = {"n": len(exs), "tools_mean": st.mean(ntools), "tools_median": st.median(ntools), "tools_max": max(ntools),
               "single_tool_items": sum(n == 1 for n in ntools) / len(exs)}

        # Lexical gate score for the pooled AUROC
        for e in exs:
            a = lex.ask(_state(e), {"decision": Choice("", {l: "" for l in WHEN2CALL_LABELS})}).answers["decision"]
            gate_scores.append(a.probabilities["tool_call"])
            gate_y.append(e.should_call)

        if cat in BFCL_AST_CATEGORIES:
            ncalls = [len(e.ground_truth) for e in exs]
            distinct = [len({next(iter(g)) for g in e.ground_truth}) for e in exs]
            row.update({"gold_calls_mean": st.mean(ncalls), "gold_calls_max": max(ncalls),
                        "multi_call_items": sum(n > 1 for n in ncalls) / len(exs),
                        "multi_function_items": sum(d > 1 for d in distinct) / len(exs)})
            # chance + lexical top-1 selection on single-call items
            single = [e for e in exs if len(e.ground_truth) == 1]
            row["chance_selection"] = st.mean(1 / len(e.tools) for e in single) if single else None
            lex_ok = 0
            for e in single:
                names = NameMap(e.tools)
                a = lex.ask(_state(e), Pipeline("dm_select", None, None, {})._decision_questions(e, False, True))
                lex_ok += names.orig(a.answers["tool"].value) == gold_values(e)[0]
            row["lexical_selection"] = lex_ok / len(single) if single else None

            # gold argument profile
            kinds, required_kinds = Counter(), Counter()
            for e in exs:
                for g in e.ground_truth:
                    (fname, params), = g.items()
                    doc = next(t for t in e.tools if t["name"] == fname)
                    props = doc["parameters"].get("properties", {})
                    req = set(doc["parameters"].get("required", []))
                    for p, acc in params.items():
                        k = arg_kind(props.get(p, {}), acc)
                        kinds[k] += 1
                        if p in req:
                            required_kinds[k] += 1
            tot = sum(required_kinds.values())
            row["required_arg_mix"] = {k: v / tot for k, v in required_kinds.most_common()}

            # dm_full ceiling (single-call items only; dm_full emits one call by design)
            ok, why = 0, Counter()
            for e in exs:
                if len(e.ground_truth) != 1:
                    why["needs >1 call"] += 1
                    continue
                pipe = Pipeline("dm_full", None, GoldArgDM(e), {})
                p = pipe.run(e)
                res = score_bfcl(e, p.calls)
                if res["correct"]:
                    ok += 1
                elif p.unsupported_params:
                    why["array/object arg"] += 1
                else:
                    why["value not an extractable span"] += 1
            row["dm_full_ceiling"] = ok / len(exs)
            row["dm_full_fail_reasons"] = {k: v / len(exs) for k, v in why.most_common()}

            # dm_select (top-1 Choice) ceiling: share of items whose gold uses a single function
            row["dm_select_top1_ceiling"] = sum(d == 1 for d in distinct) / len(exs)
        report["bfcl"][cat] = row

    report["bfcl_pooled_lexical_gate_auroc"] = auroc(gate_scores, gate_y)

    # ---- When2Call ------------------------------------------------------------------------------
    w = load_when2call(W2C)
    labels = Counter(e.label for e in w)
    by_source = defaultdict(Counter)
    tools_by_label = defaultdict(list)
    for e in w:
        by_source[e.category][e.label] += 1
        tools_by_label[e.label].append(len(e.tools))
    majority = labels.most_common(1)[0][0]
    raw = [json.loads(l) for l in open(W2C)]
    bfcl_ids = {e.id for c in BFCL_CATEGORIES for e in load_bfcl(c)}
    bfcl_q = {e.user_text.strip().lower() for c in BFCL_CATEGORIES for e in load_bfcl(c)}
    overlap_id = sum(r["source_id"] in bfcl_ids for r in raw)
    overlap_q = sum(r["question"].strip().lower() in bfcl_q for r in raw)
    w_scores = [lex.ask(_state(e), {"decision": Choice("", {l: "" for l in WHEN2CALL_LABELS})})
                .answers["decision"].probabilities["tool_call"] for e in w]
    report["when2call"] = {
        "n": len(w),
        "labels": {k: v / len(w) for k, v in labels.most_common()},
        "majority_baseline_acc": labels[majority] / len(w),
        "majority_baseline_macro_f1": macro_f1([e.label for e in w], [majority] * len(w), WHEN2CALL_LABELS),
        "tools_per_item_by_label": {k: {"mean": st.mean(v), "zero_tools": sum(x == 0 for x in v) / len(v)}
                                    for k, v in tools_by_label.items()},
        "sources": {k: dict(v) for k, v in sorted(by_source.items(), key=lambda kv: -sum(kv[1].values()))},
        "items_whose_source_id_is_in_bfcl_v4": overlap_id / len(w),
        "items_whose_question_text_is_in_bfcl_v4": overlap_q / len(w),
        "lexical_gate_auroc": auroc(w_scores, [e.should_call for e in w]),
    }

    out = Path("results")
    out.mkdir(exist_ok=True)
    (out / "benchmark_analysis.json").write_text(json.dumps(report, indent=2))

    # ---- markdown ---------------------------------------------------------------------------
    L = ["# Benchmark analysis", "", "## BFCL v4 (single-turn)", "",
         "| category | n | tools/item (mean, max) | gold calls >1 | chance sel. | lexical sel. | dm_select top-1 ceiling | dm_full ceiling |",
         "|---|---|---|---|---|---|---|---|"]
    for cat, r in report["bfcl"].items():
        if cat in BFCL_AST_CATEGORIES:
            L.append(f"| {cat} | {r['n']} | {r['tools_mean']:.1f}, {r['tools_max']} | {pct(r['multi_call_items'])} | "
                     f"{pct(r['chance_selection']) if r['chance_selection'] is not None else '–'} | "
                     f"{pct(r['lexical_selection']) if r['lexical_selection'] is not None else '–'} | "
                     f"{pct(r['dm_select_top1_ceiling'])} | {pct(r['dm_full_ceiling'])} |")
        else:
            L.append(f"| {cat} | {r['n']} | {r['tools_mean']:.1f}, {r['tools_max']} | – | – | – | – | – |")
    L += ["", f"Pooled lexical gate AUROC (all BFCL categories, should-call vs not): "
              f"{report['bfcl_pooled_lexical_gate_auroc']:.3f}", "", "### Required-argument mix and dm_full failure reasons", ""]
    for cat, r in report["bfcl"].items():
        if cat in BFCL_AST_CATEGORIES:
            mix = ", ".join(f"{k} {pct(v)}" for k, v in r["required_arg_mix"].items())
            why = ", ".join(f"{k} {pct(v)}" for k, v in r["dm_full_fail_reasons"].items())
            L.append(f"- **{cat}**: {mix}. dm_full misses: {why}")
    wr = report["when2call"]
    L += ["", "## When2Call (MCQ test)", "",
          f"- n = {wr['n']}; labels: " + ", ".join(f"{k} {pct(v)}" for k, v in wr["labels"].items()),
          f"- majority baseline: acc {pct(wr['majority_baseline_acc'])}, macro-F1 {wr['majority_baseline_macro_f1']:.3f}",
          "- tools per item: " + "; ".join(f"{k}: mean {v['mean']:.2f}, zero-tool {pct(v['zero_tools'])}"
                                           for k, v in wr["tools_per_item_by_label"].items()),
          f"- items whose `source_id` is a BFCL v4 id: {pct(wr['items_whose_source_id_is_in_bfcl_v4'])}; "
          f"whose question text appears verbatim in BFCL v4: {pct(wr['items_whose_question_text_is_in_bfcl_v4'])}",
          f"- lexical gate AUROC: {wr['lexical_gate_auroc']:.3f}", "", "Sources:", ""]
    L += [f"- {k}: {dict(v)}" for k, v in wr["sources"].items()]
    (out / "benchmark_analysis.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
