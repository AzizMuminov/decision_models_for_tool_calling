"""Scores predictions with BFCL's own AST checker.

`bfcl_eval.eval_checker.ast_eval.ast_checker` imports `bfcl_eval.constants.model_config`, which in turn
imports every provider handler (vllm, cohere, mistral, google-genai, ...). The checker only uses that
module for `MODEL_CONFIG_MAPPING[model].underscore_to_dot`, so we register a stub before importing it and
install bfcl-eval with --no-deps. Our backends map tool names back to the original dotted names
themselves, so underscore_to_dot is always False here.
"""

from __future__ import annotations

import sys
import types
from collections import defaultdict
from typing import Any

from dmtc.data import Example


def _install_model_config_stub() -> None:
    name = "bfcl_eval.constants.model_config"
    if name in sys.modules:
        return

    class _Cfg:
        underscore_to_dot = False

    stub = types.ModuleType(name)
    stub.MODEL_CONFIG_MAPPING = defaultdict(_Cfg)  # type: ignore[attr-defined]
    sys.modules[name] = stub


_install_model_config_stub()

from bfcl_eval.constants.enums import Language  # noqa: E402
from bfcl_eval.eval_checker.ast_eval.ast_checker import ast_checker  # noqa: E402

CHECKER_MODEL_NAME = "dmtc"


def score_bfcl(ex: Example, calls: list[dict[str, dict[str, Any]]]) -> dict[str, Any]:
    """Returns {"correct": bool, "error_type": str|None}. Mirrors BFCL's per-category rules."""
    if ex.category in ("irrelevance", "live_irrelevance"):
        ok = len(calls) == 0
        return {"correct": ok, "error_type": None if ok else "irrelevance:called_tool"}
    if ex.category == "live_relevance":
        ok = len(calls) > 0
        return {"correct": ok, "error_type": None if ok else "relevance:no_call"}
    if not calls:
        return {"correct": False, "error_type": "ast:no_call"}
    try:
        res = ast_checker(ex.tools, calls, ex.ground_truth, Language.PYTHON, ex.category, CHECKER_MODEL_NAME)
    except Exception as e:  # noqa: BLE001 - malformed output counts as wrong, with the reason kept
        return {"correct": False, "error_type": f"checker_exception:{type(e).__name__}"}
    # BFCL fills error_type with a placeholder ("...:unclear") even for valid calls
    return {"correct": bool(res["valid"]), "error_type": None if res["valid"] else res.get("error_type")}


def tool_selection_correct(ex: Example, selected: list[str] | None) -> bool | None:
    """Did the decision layer pick the gold tool(s)? None when not applicable."""
    if not ex.ground_truth or selected is None:
        return None
    gold = {next(iter(g)) for g in ex.ground_truth}
    return gold <= set(selected)
