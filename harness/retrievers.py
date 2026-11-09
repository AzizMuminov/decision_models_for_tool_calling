"""Tool retrievers behind find_tool: the decision model, and the baselines real harnesses use.

  bm25    Okapi BM25 over each tool's name, description and parameter names/descriptions -- the fields
          Anthropic's server-side `tool_search_tool_bm25` searches. The agent's query is plain words.
  regex   the agent writes a Python regular expression (re.search syntax, like `tool_search_tool_regex`);
          tools whose text matches are returned, ranked by matches in the name, then in the description.
  llm     an LLM router: a small chat model reads the whole catalog (one line per tool) and the intent and
          names the best tools. This is the "use a cheap LLM to pick tools" approach.
  dm      the decision model ranks the catalog; it accepts at most 255 options per question. For bigger
          catalogs there are three ways through (see `dm_rank`):
            tournament   split the catalog in halves of <=250, take the top 5 of each, decide among those
            hierarchy    ask the DM which service (32 options), then which tool within the top 2 services
            bm25_dm      BM25 shortlists the top 100, the DM reranks them (one DM call)

Every retriever returns tool names, best first. Cost is returned in USD (0 for the local ones).
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

_TOK = re.compile(r"[a-z0-9]+")


def tool_text(t: dict) -> str:
    props = t["inputSchema"]["properties"]
    params = " ".join(f"{k} {v.get('description', '')}" for k, v in props.items())
    return f"{t['name'].replace('_', ' ')} {t['description']} {params}"


def short_card(t: dict) -> str:
    """The first description line plus parameter names: what the DM and the LLM router see per tool."""
    params = ", ".join(t["inputSchema"]["properties"]) or "none"
    return f"{t['description'].splitlines()[0]} Parameters: {params}."


def toks(s: str) -> list[str]:
    return _TOK.findall(s.lower())


class BM25:
    def __init__(self, tools: list[dict], k1: float = 1.2, b: float = 0.75, field=tool_text):
        self.names = [t["name"] for t in tools]
        self.docs = [Counter(toks(field(t))) for t in tools]
        self.len = [sum(d.values()) for d in self.docs]
        self.avg = sum(self.len) / len(self.len)
        n = len(self.docs)
        df = Counter(w for d in self.docs for w in d)
        self.idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}
        self.k1, self.b = k1, b

    def scores(self, query: str) -> dict[str, float]:
        q = toks(query)
        out = {}
        for name, d, L in zip(self.names, self.docs, self.len):
            s = 0.0
            for w in q:
                f = d.get(w, 0)
                if f:
                    s += self.idf[w] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * L / self.avg))
            out[name] = s
        return out

    def rank(self, query: str, k: int) -> list[str]:
        sc = self.scores(query)
        return [n for n in sorted(sc, key=lambda n: -sc[n])[:k] if sc[n] > 0]


def regex_rank(tools: list[dict], pattern: str, k: int) -> tuple[list[str], str | None]:
    """Python re.search over name and full text; returns (matches, error). Ordering: hits in the name first."""
    if len(pattern) > 200:
        return [], "pattern longer than 200 characters"
    try:
        rx = re.compile(pattern)
    except re.error as e:
        return [], f"invalid regex: {e}"
    hits = []
    for i, t in enumerate(tools):
        in_name = len(rx.findall(t["name"]))
        in_text = len(rx.findall(tool_text(t)))
        if in_text:
            hits.append((-in_name, -in_text, i, t["name"]))
    return [h[3] for h in sorted(hits)[:k]], None


# --------------------------------------------------------------------------- LLM router

LLM_ROUTER = os.environ.get("CATALOG_ROUTER_LLM", "deepseek/deepseek-v4.1-flash")


def llm_rank(tools: list[dict], intent: str, k: int, model: str = LLM_ROUTER) -> tuple[list[str], float, str | None]:
    import httpx

    lines = "\n".join(f"{t['name']}: {short_card(t)}" for t in tools)
    sys_msg = ("You route an agent's request to tools in a catalog. Reply with JSON only: "
               f'{{"tools": [up to {k} exact tool names, best first]}}. Use only names from the catalog.\n\nCATALOG:\n{lines}')
    body = {"model": model, "temperature": 0, "max_tokens": 200, "usage": {"include": True},
            "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": sys_msg}, {"role": "user", "content": f"Request: {intent}"}]}
    err = None
    for attempt in range(4):
        try:
            r = httpx.post("https://openrouter.ai/api/v1/chat/completions", json=body, timeout=60,
                           headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"})
            if r.status_code < 400:
                d = r.json()
                cost = float((d.get("usage") or {}).get("cost") or 0)
                txt = d["choices"][0]["message"]["content"] or ""
                m = re.search(r"\{.*\}", txt, re.S)
                names = json.loads(m.group(0)).get("tools", []) if m else []
                valid = {t["name"] for t in tools}
                return [n for n in names if n in valid][:k], cost, None if names else "unparsable reply"
            err = f"HTTP {r.status_code}: {r.text[:200]}"
            if r.status_code in (402, 403):
                break
        except Exception as e:  # noqa: BLE001
            err = f"{type(e).__name__}: {e}"
    return [], 0.0, err


# --------------------------------------------------------------------------- decision model

MAX_OPTIONS = 250  # the Decisions API accepts at most 255 options per choice question; keep a margin
Q_TOOL = "Which tool from the catalog best serves what the agent wants to do?"
Q_SERVICE = "Which internal service would have the tool the agent needs?"


class DMRanker:
    def __init__(self, tools: list[dict], services: dict, model: str = "microsoft/microsoft-decision-1"):
        from dmtc.backends.dm import OpenRouterDM

        self.tools = {t["name"]: t for t in tools}
        self.services = services
        self.dm = OpenRouterDM(model)
        self.bm25 = BM25(tools)
        self.calls = 0

    def _ask(self, intent: str, options: dict[str, str], q: str = Q_TOOL):
        from dmtc.backends.dm import Choice

        assert len(options) <= 255
        self.calls += 1
        r = self.dm.ask({"what_the_agent_wants": intent}, {"x": Choice(q, options)})
        a = r.answers.get("x")
        return (a.probabilities if a else {}), r.cost_usd, r.error

    def _tools(self, intent, names):
        return self._ask(intent, {n: short_card(self.tools[n]) for n in names})

    def rank(self, intent: str, k: int, strategy: str = "auto") -> tuple[list[str], float, str | None]:
        names = list(self.tools)
        if strategy == "auto":
            strategy = "flat" if len(names) <= MAX_OPTIONS else "tournament"
        cost, err = 0.0, None
        if strategy == "flat":
            assert len(names) <= MAX_OPTIONS, "use a strategy for catalogs over 250 tools"
        elif strategy == "tournament":
            n_chunks = math.ceil(len(names) / MAX_OPTIONS) if len(names) > MAX_OPTIONS else 2
            size = math.ceil(len(names) / n_chunks)
            short = []
            for i in range(0, len(names), size):
                p, c, e = self._tools(intent, names[i:i + size])
                cost, err = cost + c, err or e
                short += sorted(p, key=p.get, reverse=True)[:5]
            names = short
        elif strategy == "hierarchy":
            svcs = sorted({t["service"] for t in self.tools.values()})
            p, c, e = self._ask(intent, {s: f"{self.services[s][0]}: {self.services[s][1]}" for s in svcs}, Q_SERVICE)
            cost, err = cost + c, err or e
            top = sorted(p, key=p.get, reverse=True)[:2]
            names = [n for n, t in self.tools.items() if t["service"] in top]
        elif strategy == "bm25_dm":
            sc = self.bm25.scores(intent)
            names = sorted(sc, key=lambda n: -sc[n])[:100]
        else:
            raise ValueError(strategy)
        p, c, e = self._tools(intent, names)
        cost, err = cost + c, err or e
        return sorted(p, key=p.get, reverse=True)[:k], cost, err
