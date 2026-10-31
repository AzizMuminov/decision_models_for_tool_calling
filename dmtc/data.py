"""Benchmark loaders. Every loader yields `Example` objects with a common shape."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any, Iterator

# BFCL categories that are scored with the AST checker on a single turn (Python only).
BFCL_AST_CATEGORIES = [
    "simple_python",
    "multiple",
    "parallel",
    "parallel_multiple",
    "live_simple",
    "live_multiple",
    "live_parallel",
    "live_parallel_multiple",
]
# Categories where the only question is "call a tool or not".
BFCL_RELEVANCE_CATEGORIES = ["irrelevance", "live_irrelevance", "live_relevance"]
BFCL_CATEGORIES = BFCL_AST_CATEGORIES + BFCL_RELEVANCE_CATEGORIES

WHEN2CALL_LABELS = ["direct", "tool_call", "request_for_info", "cannot_answer"]


@dataclass
class Example:
    id: str
    benchmark: str  # "bfcl" | "when2call"
    category: str
    messages: list[dict[str, Any]]  # chat history, OpenAI-style {"role", "content"}
    tools: list[dict[str, Any]]  # BFCL-style function docs (name, description, parameters)
    # Gold label for the decision layer: should a tool be called at all?
    should_call: bool
    # BFCL: list of {func_name: {param: [acceptable values]}}; When2Call: None
    ground_truth: Any = None
    # When2Call: one of WHEN2CALL_LABELS
    label: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def user_text(self) -> str:
        return "\n".join(m["content"] for m in self.messages if m["role"] == "user")


def _bfcl_data_dir() -> Path:
    return Path(str(resources.files("bfcl_eval") / "data"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_bfcl(category: str, data_dir: Path | None = None) -> list[Example]:
    if category not in BFCL_CATEGORIES:
        raise ValueError(f"unsupported BFCL category {category!r}; choose from {BFCL_CATEGORIES}")
    data_dir = data_dir or _bfcl_data_dir()
    rows = _read_jsonl(data_dir / f"BFCL_v4_{category}.json")
    answers: dict[str, Any] = {}
    answer_path = data_dir / "possible_answer" / f"BFCL_v4_{category}.json"
    if answer_path.exists():
        answers = {r["id"]: r["ground_truth"] for r in _read_jsonl(answer_path)}

    examples = []
    for r in rows:
        # Single-turn categories hold one turn; keep only that turn's messages.
        turn = r["question"][0]
        messages = [m for m in turn if m["role"] in ("user", "assistant", "system")]
        should_call = category not in ("irrelevance", "live_irrelevance")
        examples.append(
            Example(
                id=r["id"],
                benchmark="bfcl",
                category=category,
                messages=messages,
                tools=r["function"],
                should_call=should_call,
                ground_truth=answers.get(r["id"]),
            )
        )
    return examples


def load_when2call(path: Path) -> list[Example]:
    """Loads the When2Call MCQ test file (data/test/when2call_test_mcq.jsonl)."""
    examples = []
    for r in _read_jsonl(path):
        examples.append(
            Example(
                id=r["uuid"],
                benchmark="when2call",
                category=r["source"],
                messages=[{"role": "user", "content": r["question"]}],
                tools=[json.loads(t) for t in r["tools"]],
                should_call=r["correct_answer"] == "tool_call",
                label=r["correct_answer"],
                # source_id: the BFCL live item this is a counterfactual variant of (shared by its
                # tool_call / cannot_answer / request_for_info versions)
                meta={"target_tool": r.get("target_tool"), "answers": r["answers"], "source_id": r["source_id"]},
            )
        )
    return examples


def load(benchmark: str, category: str | None, cfg: dict[str, Any]) -> list[Example]:
    if benchmark == "bfcl":
        return load_bfcl(category)  # type: ignore[arg-type]
    if benchmark == "when2call":
        return load_when2call(Path(cfg["when2call_path"]))
    raise ValueError(f"unknown benchmark {benchmark!r}")


def subsample(examples: list[Example], limit: int | None, seed: int) -> list[Example]:
    if limit is None or limit >= len(examples):
        return examples
    rng = random.Random(seed)
    return sorted(rng.sample(examples, limit), key=lambda e: e.id)


def subsample_groups(examples: list[Example], limit: int | None, seed: int, key: str = "source_id") -> list[Example]:
    """Samples whole groups (e.g. all When2Call variants of one BFCL item) until ~limit examples."""
    if limit is None or limit >= len(examples):
        return examples
    groups: dict[str, list[Example]] = {}
    for e in examples:
        groups.setdefault(e.meta.get(key) or e.id, []).append(e)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    out: list[Example] = []
    for k in keys:
        if len(out) >= limit:
            break
        out += groups[k]
    return sorted(out, key=lambda e: e.id)


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    if not path.exists():
        return
    with open(path) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)
