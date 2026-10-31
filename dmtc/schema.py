"""Conversions between BFCL function docs and provider tool schemas."""

from __future__ import annotations

import copy
import re
from typing import Any

_BFCL_TO_JSON = {
    "dict": "object",
    "float": "number",
    "tuple": "array",
    "list": "array",
    "int": "integer",
    "str": "string",
    "bool": "boolean",
}


def to_json_schema(params: dict[str, Any]) -> dict[str, Any]:
    """BFCL uses Python-ish type names ("dict", "float", "tuple", "any"); map them to JSON Schema."""
    node = copy.deepcopy(params)

    def walk(n: Any) -> Any:
        if isinstance(n, dict):
            # Columnar exports (parquet / HF datasets-server) fill absent fields with null,
            # e.g. {"enum": null, "items": null}; null is not a valid schema keyword value.
            for k in [k for k, v in n.items() if v is None]:
                del n[k]
            t = n.get("type")
            if t == "any":
                n.pop("type")
            elif isinstance(t, str) and t in _BFCL_TO_JSON:
                n["type"] = _BFCL_TO_JSON[t]
            # BFCL marks optional params with a non-standard flag; JSON Schema ignores it.
            n.pop("optional", None)
            for k in ("properties",):
                if isinstance(n.get(k), dict):
                    # the same exports also list every sibling tool's params as null here
                    n[k] = {pk: walk(pv) for pk, pv in n[k].items() if pv is not None}
            if "items" in n:
                n["items"] = walk(n["items"])
            if n.get("type") == "array" and "items" not in n:
                n["items"] = {}
        return n

    out = walk(node)
    out.setdefault("type", "object")
    out.setdefault("properties", {})
    return out


_PROP_OK = re.compile(r"^[a-zA-Z0-9_.-]{1,64}$")


def safe_prop(name: str) -> str:
    """Anthropic requires property keys to match ^[a-zA-Z0-9_.-]{1,64}$ (e.g. BFCL's 'año_vehiculo')."""
    if _PROP_OK.match(name):
        return name
    import unicodedata

    ascii_ = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return (re.sub(r"[^a-zA-Z0-9_.-]", "_", ascii_) or "param")[:64]


def sanitize_params(schema: dict[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
    """Renames invalid top-level property keys. Returns (schema, {safe: original}) for mapping back."""
    props = schema.get("properties") or {}
    back = {}
    if all(_PROP_OK.match(k) for k in props):
        return schema, back
    out = dict(schema)
    out["properties"] = {}
    for k, v in props.items():
        s = safe_prop(k)
        back[s] = k
        out["properties"][s] = v
    if "required" in out:
        out["required"] = [safe_prop(k) for k in out["required"]]
    return out, back


class NameMap:
    """Provider tool names must match ^[a-zA-Z0-9_-]{1,64}$; BFCL names contain dots."""

    def __init__(self, tools: list[dict[str, Any]]):
        self.to_safe: dict[str, str] = {}
        self.to_orig: dict[str, str] = {}
        self.param_back: dict[str, dict[str, str]] = {}  # safe tool name -> {safe param: original}
        for t in tools:
            safe = re.sub(r"[^a-zA-Z0-9_-]", "_", t["name"])[:64]
            base, i = safe, 1
            while safe in self.to_orig:
                safe = f"{base[:60]}_{i}"
                i += 1
            self.to_safe[t["name"]] = safe
            self.to_orig[safe] = t["name"]

    def orig(self, safe: str) -> str:
        return self.to_orig.get(safe, safe)

    def orig_args(self, safe_tool: str, args: dict[str, Any]) -> dict[str, Any]:
        back = self.param_back.get(safe_tool, {})
        return {back.get(k, k): v for k, v in args.items()} if back else args

    def schema(self, tool: dict[str, Any]) -> dict[str, Any]:
        s, back = sanitize_params(to_json_schema(tool.get("parameters", {})))
        if back:
            self.param_back[self.to_safe[tool["name"]]] = back
        return s


def anthropic_tools(tools: list[dict[str, Any]], names: NameMap) -> list[dict[str, Any]]:
    return [
        {
            "name": names.to_safe[t["name"]],
            "description": t.get("description", "")[:4096],
            "input_schema": names.schema(t),
        }
        for t in tools
    ]


def openai_tools(tools: list[dict[str, Any]], names: NameMap) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": names.to_safe[t["name"]],
                "description": t.get("description", ""),
                "parameters": names.schema(t),
            },
        }
        for t in tools
    ]


def tool_card(tool: dict[str, Any]) -> dict[str, Any]:
    """Compact description of a tool, used as a decision-model criterion."""
    params = tool.get("parameters", {}).get("properties", {})
    return {
        "name": tool["name"],
        "description": tool.get("description", ""),
        "parameters": {k: v.get("description", v.get("type", "")) for k, v in params.items()},
    }
