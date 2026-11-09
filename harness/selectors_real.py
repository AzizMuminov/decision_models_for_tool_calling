"""The tool-selection mechanisms compared on the real MCP catalog, each as a search function query -> tool names.

  cc_search   Claude Code's ToolSearch. The agent also sees every deferred tool's name. Queries are
              `select:a,b` (exact names), or keywords; a `+term` must occur in the tool name. Ranking:
              matches in the name count most, then in the description. (An approximation of the shipped
              tool: Claude Code's exact scoring is not documented.)
  bm25        Okapi BM25 over name, description and parameter names/descriptions: Anthropic's
              tool_search_tool_bm25 and Codex CLI's search_tool_bm25.
  embed       cosine similarity between text-embedding-3-small vectors of the query and of each tool's
              name + description: Goose's vector tool router, RAG-MCP, LangGraph-bigtool.
  llm         an LLM router (deepseek-v4.1-flash) reads one line per tool and names the best ones: Goose's
              LLM tool-selection strategy.
  dm          the decision model. The catalog (613 tools) exceeds the 255 options a decision allows, so it
              ranks hierarchically: first the server (12 options), then the tools of the most likely
              server(s), each at most 198 options. A tool's score is P(server) x P(tool | server).
  random      a control: K tools drawn at random, so we see how much the selection itself matters.
  (groups     Copilot's virtual tools are not a search function: see mcp_server_real.py.)

Search returns the top K names (K=5, Anthropic's default) and the cost in USD.
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from real_sim import SERVER_INFO, load_catalog  # noqa: E402
from retrievers import BM25  # noqa: E402

K = 5
EMBED_MODEL = "openai/text-embedding-3-small"
ROUTER_MODEL = os.environ.get("ROUTER_MODEL", "deepseek/deepseek-v4.1-flash")
DM_MODEL = os.environ.get("DM_MODEL", "microsoft/microsoft-decision-1")
CACHE = HERE.parent / "results" / "cache"
_TOK = re.compile(r"[a-z0-9]+")


def _split(s: str) -> list[str]:
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", s)  # camelCase -> camel Case
    return _TOK.findall(s.lower())


def first_sentences(text: str, n: int = 300) -> str:
    t = " ".join(text.split())
    return t[:n] + ("..." if len(t) > n else "")


def full_text(t: dict) -> str:
    props = (t["inputSchema"] or {}).get("properties") or {}
    params = " ".join(f"{k} {v.get('description', '') if isinstance(v, dict) else ''}" for k, v in props.items())
    return f"{' '.join(_split(t['name']))} {t['description']} {params}"


def card(t: dict) -> str:
    props = list(((t["inputSchema"] or {}).get("properties") or {}))
    return f"[{t['server']}] {first_sentences(t['description'], 240)} Parameters: {', '.join(props) or 'none'}."


def _key() -> dict:
    return {"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"}


class Selectors:
    def __init__(self, tools: list[dict] | None = None):
        self.tools = tools or load_catalog()
        self.by = {t["name"]: t for t in self.tools}
        self.names = [t["name"] for t in self.tools]
        self._bm25 = None
        self._emb = None
        self._dm = None

    # --------------------------------------------------------------- Claude Code ToolSearch
    def cc_search(self, query: str, k: int = K) -> tuple[list[str], float]:
        q = query.strip()
        if q.lower().startswith("select:"):
            want = [w.strip() for w in q[7:].split(",") if w.strip()]
            return [w for w in want if w in self.by][:max(k, len(want))], 0.0
        terms = q.split()
        required = [t[1:].lower() for t in terms if t.startswith("+") and len(t) > 1]
        words = [w for t in terms for w in _split(t.lstrip("+"))]
        scored = []
        for t in self.tools:
            nparts, low_name = _split(t["name"]), t["name"].lower()
            if any(r not in low_name for r in required):
                continue
            desc = set(_split(t["description"]))
            s = sum(4 if w in nparts else 2 if w in low_name else 1 if w in desc else 0 for w in words)
            if s:
                scored.append((-s, t["name"]))
        return [n for _, n in sorted(scored)[:k]], 0.0

    # --------------------------------------------------------------- BM25
    def bm25(self, query: str, k: int = K) -> tuple[list[str], float]:
        if self._bm25 is None:
            self._bm25 = BM25(self.tools, field=full_text)
        return self._bm25.rank(query, k), 0.0

    # --------------------------------------------------------------- embeddings
    def _embed(self, texts: list[str]) -> tuple[list[list[float]], float]:
        out, cost = [], 0.0
        for i in range(0, len(texts), 256):
            r = httpx.post("https://openrouter.ai/api/v1/embeddings", headers=_key(), timeout=60,
                           json={"model": EMBED_MODEL, "input": texts[i:i + 256]}).json()
            out += [d["embedding"] for d in r["data"]]
            cost += float((r.get("usage") or {}).get("cost") or 0)
        return out, cost

    def embed(self, query: str, k: int = K) -> tuple[list[str], float]:
        cost = 0.0
        if self._emb is None:
            CACHE.mkdir(parents=True, exist_ok=True)
            import hashlib
            # stable across processes (Python's hash() is salted per process, which made every MCP server re-embed the catalog)
            f = CACHE / f"emb_{len(self.tools)}_{hashlib.sha1(chr(10).join(self.names).encode()).hexdigest()[:10]}.json"
            if f.exists():
                self._emb = json.loads(f.read_text())
            else:
                vecs, cost = self._embed([f"{t['name']}: {first_sentences(t['description'], 1000)}" for t in self.tools])
                self._emb = dict(zip(self.names, vecs))
                f.write_text(json.dumps(self._emb))
        (qv,), c = self._embed([query])
        cost += c
        qn = math.sqrt(sum(x * x for x in qv))
        sims = {n: sum(a * b for a, b in zip(qv, v)) / (qn * math.sqrt(sum(x * x for x in v))) for n, v in self._emb.items()}
        return sorted(sims, key=sims.get, reverse=True)[:k], cost

    # --------------------------------------------------------------- LLM router
    def llm(self, query: str, k: int = K) -> tuple[list[str], float]:
        lines = "\n".join(f"{t['name']}: {first_sentences(t['description'], 160)}" for t in self.tools)
        body = {"model": ROUTER_MODEL, "temperature": 0, "max_tokens": 300, "usage": {"include": True},
                "response_format": {"type": "json_object"},
                "messages": [{"role": "system", "content": "You route an agent's request to tools. Reply with JSON only: "
                              f'{{"tools": [up to {k} exact tool names from the catalog, best first]}}.\n\nCATALOG:\n{lines}'},
                             {"role": "user", "content": f"Request: {query}"}]}
        for _ in range(3):
            try:
                r = httpx.post("https://openrouter.ai/api/v1/chat/completions", headers=_key(), json=body, timeout=90)
                if r.status_code >= 400:
                    continue
                d = r.json()
                txt = d["choices"][0]["message"]["content"] or ""
                m = re.search(r"\{.*\}", txt, re.S)
                names = json.loads(m.group(0)).get("tools", []) if m else []
                return [n for n in names if n in self.by][:k], float((d.get("usage") or {}).get("cost") or 0)
            except Exception:  # noqa: BLE001
                continue
        return [], 0.0

    # --------------------------------------------------------------- decision model
    def _dm_ask(self, query: str, options: dict[str, str], instructions: str):
        from dmtc.backends.dm import Choice, OpenRouterDM

        self._dm = self._dm or OpenRouterDM(DM_MODEL)
        assert len(options) <= 255
        r = self._dm.ask({"what_the_agent_wants": query}, {"x": Choice(instructions, options)})
        a = r.answers.get("x")
        return (a.probabilities if a else {}), r.cost_usd

    def dm(self, query: str, k: int = K, strategy: str = "hierarchy", mass: float = 0.9, max_servers: int = 3) -> tuple[list[str], float]:
        if len(self.tools) <= 250 and strategy == "hierarchy":
            strategy = "flat"
        if strategy == "flat":
            p, c = self._dm_ask(query, {n: card(self.by[n]) for n in self.names}, "Which tool best serves what the agent wants to do?")
            return sorted(p, key=p.get, reverse=True)[:k], c
        if strategy == "hierarchy":
            servers = sorted({t["server"] for t in self.tools})
            ps, cost = self._dm_ask(query, {s: SERVER_INFO.get(s, s) for s in servers}, "Which service has the tool the agent needs?")
            chosen, acc = [], 0.0
            for s in sorted(ps, key=ps.get, reverse=True):  # most likely servers until 90% of the mass (at most 3)
                chosen.append(s)
                acc += ps[s]
                if acc >= mass or len(chosen) >= max_servers:
                    break
            score = {}
            for s in chosen:
                names = [n for n in self.names if self.by[n]["server"] == s]
                for i in range(0, len(names), 250):
                    pt, c = self._dm_ask(query, {n: card(self.by[n]) for n in names[i:i + 250]}, "Which tool best serves what the agent wants to do?")
                    cost += c
                    score.update({n: ps[s] * v for n, v in pt.items()})
            return sorted(score, key=score.get, reverse=True)[:k], cost
        if strategy == "tournament":
            n_chunks = math.ceil(len(self.names) / 250)
            size = math.ceil(len(self.names) / n_chunks)
            short, cost = [], 0.0
            for i in range(0, len(self.names), size):
                p, c = self._dm_ask(query, {n: card(self.by[n]) for n in self.names[i:i + size]}, "Which tool best serves what the agent wants to do?")
                cost += c
                short += sorted(p, key=p.get, reverse=True)[:k]
            p, c = self._dm_ask(query, {n: card(self.by[n]) for n in short}, "Which tool best serves what the agent wants to do?")
            return sorted(p, key=p.get, reverse=True)[:k], cost + c
        if strategy in ("bm25_dm", "embed_dm"):
            pre, cost = (self.bm25 if strategy == "bm25_dm" else self.embed)(query, 100)
            p, c = self._dm_ask(query, {n: card(self.by[n]) for n in pre}, "Which tool best serves what the agent wants to do?")
            return sorted(p, key=p.get, reverse=True)[:k], cost + c
        raise ValueError(strategy)

    # --------------------------------------------------------------- DM top-p
    def dm_yn(self, query: str, k: int = K, pool: int = 10) -> tuple[list[str], float]:
        """Rank with multiple choice (server, then tool), then one yes/no question per top-10 tool, all in one call:
        "will the agent need this tool for the sub-task it is searching for?" Keep the yeses in rank order (at least 1).
        Choice questions are mutually exclusive, so they cannot say that three tools are all needed; yes/no can."""
        from dmtc.backends.dm import Noul, OpenRouterDM

        servers = sorted({t["server"] for t in self.tools})
        ps, cost = self._dm_ask(query, {s: SERVER_INFO.get(s, s) for s in servers}, "Which service has the tool the agent needs?")
        chosen, acc = [], 0.0
        for s in sorted(ps, key=ps.get, reverse=True):
            chosen.append(s); acc += ps[s]
            if acc >= 0.9 or len(chosen) >= 3:
                break
        score = {}
        for s in chosen:
            names = [n for n in self.names if self.by[n]["server"] == s]
            for i in range(0, len(names), 250):
                pt, c = self._dm_ask(query, {n: card(self.by[n]) for n in names[i:i + 250]}, "Which tool best serves what the agent wants to do?")
                cost += c
                score.update({n: ps[s] * v for n, v in pt.items()})
        cand = sorted(score, key=score.get, reverse=True)[:pool]
        if not cand:
            return [], cost
        self._dm = self._dm or OpenRouterDM(DM_MODEL)
        r = self._dm.ask({"what_the_agent_wants": query}, {f"t{j}": Noul(
            "The agent is searching for the tools of one sub-task. Will it need this tool for that sub-task, now or in its "
            f"next few steps? Tool: {card(self.by[n])}") for j, n in enumerate(cand)})
        cost += r.cost_usd
        keep = [n for j, n in enumerate(cand) if r.answers.get(f"t{j}") and r.answers[f"t{j}"].value >= 0.5]
        return (keep or cand[:1]), cost

    # fixed service cards (selector study): GitHub lists branches/commits/files, Git is local only, GitLab is legacy hosting
    FIXED_CARDS = {
        "github": "GitHub (remote code hosting): repositories, branches, commits, creating, updating and pushing files, pull requests and reviews, issues, Actions CI runs and logs, releases, code search, security alerts",
        "git": "Git (the local working copy on this machine only; no remote hosting, no pull requests or issues): status, log, diff, local branches, staging and committing",
        "gitlab": "GitLab (remote code hosting, used for some legacy projects): projects, branches, commits, files, merge requests, issues, CI pipelines and job logs, wiki",
    }
    NAMED_ALIAS = {"github": [r"\bgithub\b"], "gitlab": [r"\bgitlab\b"], "git": [r"\bgit\b(?!hub|lab)"], "atlassian": [r"\bjira\b", r"\bconfluence\b", r"\batlassian\b"],
                   "grafana": [r"\bgrafana\b", r"\bloki\b", r"\bprometheus\b"], "sentry": [r"\bsentry\b"], "slack": [r"\bslack\b"], "notion": [r"\bnotion\b"],
                   "linear": [r"\blinear\b"], "kubernetes": [r"\bkubernetes\b", r"\bkubectl\b", r"\bk8s\b"], "mongodb": [r"\bmongo(db)?\b"],
                   "postgres": [r"\bpostgres(ql)?\b"], "aws_cloudwatch": [r"\bcloudwatch\b"], "aws_eks": [r"\beks\b"], "circleci": [r"\bcircleci\b"]}

    CTX_PROMPT = ("A coding agent is searching a catalog for the tool it needs next. The tools you rank highest are shown "
                  "to it with descriptions, the next ones by name. Which tool will the agent need to call to do what it wants?")

    def dm_ctx(self, query: str, k: int = K) -> tuple[list[str], list[str], float]:
        """Selector F with one tool-step call per service, using a prompt that explains the situation (CTX_PROMPT),
        instead of the two calls 'best serves' and 'expert first'. Service step, list size and names pool unchanged
        except that the 3 'expert first' names are dropped."""
        return self.dm_franken(query, k, tool_prompts=(("best", self.CTX_PROMPT),))

    def dm_franken(self, query: str, k: int = K, tool_prompts=None) -> tuple[list[str], list[str], float]:
        """Combined selector F (selector study): fixed service cards; services until 90% of the probability (max 3); a
        'none' option (empty answer when P(none) >= 0.4); a service the query names is always included; the 5 tools that
        best serve the query (described) plus, as names, up to 3 prerequisites ('which tool would an expert call first')
        and the services' most-used tools (leave-one-task-out), at most 15 tools in all. Tools already shown in this run
        are not repeated (an already-described top tool comes back as a name)."""
        import json as _json
        from concurrent.futures import ThreadPoolExecutor
        if not hasattr(self, "_pop"):
            here = Path(__file__).resolve().parent
            task = Path(os.environ.get("REAL_TASK", "")).parent.name
            loo = _json.loads((here / "tool_popularity_loo.json").read_text())
            self._pop = loo[task] if task in loo else _json.loads((here / "tool_popularity.json").read_text())
            self._described, self._named = set(), set()
        servers = sorted({t["server"] for t in self.tools})
        cards = {s: self.FIXED_CARDS.get(s, SERVER_INFO.get(s, s)) for s in servers}
        cards["__none__"] = "None of these services can do what the agent asked."
        ps, cost = self._dm_ask(query, cards, "Which service has the tool the agent needs?")
        if ps.get("__none__", 0) >= 0.4:
            return [], [], cost
        ps = {s: v for s, v in ps.items() if s != "__none__"}
        tot = sum(ps.values()) or 1.0
        ps = {s: v / tot for s, v in ps.items()}
        order = sorted(ps, key=ps.get, reverse=True)
        named = [s for s, pats in self.NAMED_ALIAS.items() if any(re.search(p, query.lower()) for p in pats)]
        if len(named) == 1 and named[0] in ps:
            order = [named[0]] + [s for s in order if s != named[0]]
            ps[named[0]] = max(ps[named[0]], 0.9)
        chosen, acc = [], 0.0
        for s in order:
            chosen.append(s); acc += ps[s]
            if acc >= 0.9 or len(chosen) >= 3:
                break
        jobs = []
        for s in chosen:
            names = [n for n in self.names if self.by[n]["server"] == s]
            for i in range(0, len(names), 150):
                for qn, instr in (tool_prompts or (("best", "Which tool best serves what the agent wants to do?"),
                                  ("first", "Which tool would an expert engineer call first to do what the agent wants?"))):
                    jobs.append((s, qn, instr, names[i:i + 150]))
        with ThreadPoolExecutor(len(jobs)) as ex:
            res = list(ex.map(lambda j: (j[0], j[1], *self._dm_ask(query, {n: card(self.by[n]) for n in j[3]}, j[2])), jobs))
        best, first = {}, {}
        for s, qn, p, c in res:
            cost += c
            (best if qn == "best" else first).update({n: ps[s] * v for n, v in p.items()})
        rb = sorted(best, key=best.get, reverse=True); rf = sorted(first, key=first.get, reverse=True)
        top = rb[:5]
        described = [n for n in top if n not in self._described]
        names_out = [n for n in top if n in self._described]  # reminders by name
        pool = [n for n in rf if n not in top][:3]
        for s in chosen:
            pool += [n for n in self._pop.get(s, []) if n in self.by]
        pool += rb[5:]
        for n in dict.fromkeys(pool):
            if len(described) + len(names_out) >= 15:
                break
            if n not in top and n not in self._described and n not in self._named and n not in names_out:
                names_out.append(n)
        self._described |= set(described)
        self._named |= set(names_out)
        return described, names_out, cost

    def dm_rec510(self, query: str, k: int = K) -> tuple[list[str], list[str], float]:
        """dm_rec with the 5 best tools described and up to 10 names."""
        return self.dm_rec(query, k, n_best=5, n_names=10)

    def dm_rec(self, query: str, k: int = K, n_best: int = 3, n_names: int = 7) -> tuple[list[str], list[str], float]:
        """Recommendations: the DM's top 3 tools for the query, with descriptions, plus up to 7 names from a fixed pool
        (the DM's top 10 and the service's 10 most-used tools, harness/tool_popularity.json, built from the 38-task main
        experiment). Aware of what this run has already been shown: a top-3 tool already described comes back as a
        name only, and pool tools already shown are left out, so the answer gets shorter rather than worse."""
        import json as _json
        if not hasattr(self, "_pop"):
            here = Path(__file__).resolve().parent
            task = Path(os.environ.get("REAL_TASK", "")).parent.name
            loo = _json.loads((here / "tool_popularity_loo.json").read_text())
            # leave-one-task-out: a task never sees usage from its own runs (main-suite tasks); other tasks use all 38
            self._pop = loo[task] if task in loo else _json.loads((here / "tool_popularity.json").read_text())
            self._described, self._named = set(), set()
        servers = sorted({t["server"] for t in self.tools})
        ps, cost = self._dm_ask(query, {s: SERVER_INFO.get(s, s) for s in servers}, "Which service has the tool the agent needs?")
        srv = max(ps, key=ps.get)
        names = [n for n in self.names if self.by[n]["server"] == srv]
        pt = {}
        for i in range(0, len(names), 250):
            p, c = self._dm_ask(query, {n: card(self.by[n]) for n in names[i:i + 250]}, "Which tool best serves what the agent wants to do?")
            cost += c
            pt.update(p)
        ranked = sorted(pt, key=pt.get, reverse=True)
        top3 = ranked[:n_best]
        described = [n for n in top3 if n not in self._described]
        renamed = [n for n in top3 if n in self._described]  # already described earlier: a reminder by name
        pool = list(dict.fromkeys([n for n in self._pop.get(srv, [])[:n_names + 3] if n in self.by] + ranked[n_best:n_best + n_names]))  # most-used first
        extra = renamed + [n for n in pool if n not in top3 and n not in self._described and n not in self._named][:max(0, n_names - len(renamed))]
        self._described |= set(described)
        self._named |= set(extra)
        return described, extra, cost

    def dm_menu(self, query: str, k: int = K) -> tuple[list[str], dict[str, list[str]], float]:
        """Menu design: the DM picks the service(s), the agent picks the tool. Returns the top 3 tools (shown with
        descriptions) and, for the one or two most likely services, the names of all their other tools."""
        servers = sorted({t["server"] for t in self.tools})
        ps, cost = self._dm_ask(query, {s: SERVER_INFO.get(s, s) for s in servers}, "Which service has the tool the agent needs?")
        chosen, acc = [], 0.0
        for s in sorted(ps, key=ps.get, reverse=True):
            chosen.append(s); acc += ps[s]
            if acc >= 0.9 or len(chosen) >= 2:
                break
        score = {}
        for s in chosen:
            names = [n for n in self.names if self.by[n]["server"] == s]
            for i in range(0, len(names), 250):
                pt, c = self._dm_ask(query, {n: card(self.by[n]) for n in names[i:i + 250]}, "Which tool best serves what the agent wants to do?")
                cost += c
                score.update({n: ps[s] * v for n, v in pt.items()})
        top = sorted(score, key=score.get, reverse=True)[:3]
        menu = {s: [n for n in self.names if self.by[n]["server"] == s and n not in top] for s in chosen}
        return top, menu, cost

    def dm_k3(self, query: str, k: int = K) -> tuple[list[str], float]:
        """The DM hierarchy, top 3 tools."""
        return self.dm(query, 3)

    def dm_topp05(self, query: str, k: int = K) -> tuple[list[str], float]:
        """dm_topp with p = 0.5 (cap 5)."""
        return self.dm_topp(query, k, p=0.5)

    def dm_topp10(self, query: str, k: int = K) -> tuple[list[str], float]:
        """dm_topp with a cap of 10 instead of 5."""
        return self.dm_topp(query, k, cap=10)

    def dm_topp(self, query: str, k: int = K, p: float = 0.9, cap: int = 5) -> tuple[list[str], float]:
        """The DM hierarchy, returning tools until they cover p of the probability mass (at least 1, at most cap; cap 5 ignores a larger limit, cap 10 overrides a smaller one)."""
        servers = sorted({t["server"] for t in self.tools})
        ps, cost = self._dm_ask(query, {s: SERVER_INFO.get(s, s) for s in servers}, "Which service has the tool the agent needs?")
        chosen, acc = [], 0.0
        for s in sorted(ps, key=ps.get, reverse=True):
            chosen.append(s)
            acc += ps[s]
            if acc >= 0.9 or len(chosen) >= 3:
                break
        score = {}
        for s in chosen:
            names = [n for n in self.names if self.by[n]["server"] == s]
            for i in range(0, len(names), 250):
                pt, c = self._dm_ask(query, {n: card(self.by[n]) for n in names[i:i + 250]}, "Which tool best serves what the agent wants to do?")
                cost += c
                score.update({n: ps[s] * v for n, v in pt.items()})
        total = sum(score.values()) or 1.0
        out, acc = [], 0.0
        for n in sorted(score, key=score.get, reverse=True)[:cap]:
            out.append(n)
            acc += score[n] / total
            if acc >= p:
                break
        return out, cost

    # --------------------------------------------------------------- random (control)
    def random(self, query: str, k: int = K) -> tuple[list[str], float]:
        """K tools drawn at random (seeded by the query, so a repeated query gets the same answer)."""
        import random as _r
        return _r.Random(query).sample(self.names, k), 0.0

    # --------------------------------------------------------------- DM v2 (redesign after reading the Hermes runs)
    def dm2(self, query: str, k: int = K) -> tuple[list[str], float]:
        """The DM ranker, tuned for an agent loop instead of a leaderboard:
        1. exact tool names in the query are returned directly, with no DM call (agents often search for a
           name they already saw, e.g. "create_or_update_file"; v1 then ranked git_add above it);
        2. results stop once they cover 90% of the DM's probability mass (at least 1, at most k): when the DM is
           sure, the agent gets one or two hits instead of five to ten, so less text lands in its context.
        """
        toks = set(re.findall(r"[A-Za-z0-9_\-]+", query))
        compound = lambda w: "_" in w or "-" in w or any(c.isupper() for c in w[1:])  # noqa: E731 - a bare word like "query" is not a name
        exact = [n for n in self.names if n in toks or (self.by[n]["tool"] in toks and compound(self.by[n]["tool"]))]
        if exact:
            return exact[:k], 0.0
        servers = sorted({t["server"] for t in self.tools})
        ps, cost = self._dm_ask(query, {s: SERVER_INFO.get(s, s) for s in servers}, "Which service has the tool the agent needs?")
        chosen, acc = [], 0.0
        for s in sorted(ps, key=ps.get, reverse=True):
            chosen.append(s)
            acc += ps[s]
            if acc >= 0.9 or len(chosen) >= 3:
                break
        score = {}
        for s in chosen:
            names = [n for n in self.names if self.by[n]["server"] == s]
            for i in range(0, len(names), 250):
                pt, c = self._dm_ask(query, {n: card(self.by[n]) for n in names[i:i + 250]}, "Which tool best serves what the agent wants to do?")
                cost += c
                score.update({n: ps[s] * v for n, v in pt.items()})
        ranked = sorted(score, key=score.get, reverse=True)[:k]
        total = sum(score[n] for n in ranked) or 1.0
        out, acc = [], 0.0
        for n in ranked:
            out.append(n)
            acc += score[n] / total
            if acc >= 0.9:
                break
        return out, cost

    # --------------------------------------------------------------- DM v3: flat tournament + null option
    NULL = "__none__"
    NULL_CARD = "None of these tools: nothing listed can do what the agent asked."

    def dm3(self, query: str, k: int = K, chunk: int = 150, per_chunk: int = 3, cap: int = 5) -> tuple[list[str], float]:
        """BM25-shaped interface (flat over all tools, top-k), ranked by the DM, with a null option.

        Round 1: the catalog in chunks of <=150 tools (well under the 255-option limit: long option lists dilute the
        DM's attention), each with the null option; the top 3 real tools of each chunk go on. Round 2: the finalists
        plus null. Tools ranked below null are dropped, so the agent may get fewer than k, or none ("no matching tool").
        At most 5 results however large a limit the agent asks for. Round-1 calls run in parallel.
        """
        from concurrent.futures import ThreadPoolExecutor

        q = "Which tool best serves what the agent wants to do? Choose None if no tool listed can do it."
        chunks = [self.names[i:i + chunk] for i in range(0, len(self.names), chunk)]

        def ask(names):
            opts = {n: card(self.by[n]) for n in names}
            opts[self.NULL] = self.NULL_CARD
            return self._dm_ask(query, opts, q)

        with ThreadPoolExecutor(len(chunks)) as ex:
            res = list(ex.map(ask, chunks))
        cost = sum(c for _, c in res)
        finalists = []
        for p, _ in res:
            finalists += [n for n in sorted(p, key=p.get, reverse=True) if n != self.NULL][:per_chunk]
        p, c = ask(finalists)
        cost += c
        ranked = sorted(p, key=p.get, reverse=True)
        out = []
        for n in ranked:
            if n == self.NULL:
                break
            out.append(n)
        return out[:min(k, cap)], cost

    def dm4(self, query: str, k: int = K, cap: int = 5) -> tuple[list[str], float]:
        """Hierarchy DM (service, then tool) with the null option at both steps: if 'none' is the most likely service the
        answer is empty; otherwise tools ranked below 'none' within the chosen services are dropped. At most 5 results."""
        servers = sorted({t["server"] for t in self.tools})
        opts = {sv: SERVER_INFO.get(sv, sv) for sv in servers}
        opts[self.NULL] = "None of these services can do what the agent asked."
        ps, cost = self._dm_ask(query, opts, "Which service has the tool the agent needs? Choose None if no service can do it.")
        order = sorted(ps, key=ps.get, reverse=True)
        if not order or order[0] == self.NULL:
            return [], cost
        chosen, acc = [], 0.0
        for sv in order:
            if sv == self.NULL:
                break
            chosen.append(sv)
            acc += ps[sv]
            if acc >= 0.9 or len(chosen) >= 3:
                break
        score = {}
        for sv in chosen:
            names = [n for n in self.names if self.by[n]["server"] == sv]
            for i in range(0, len(names), 150):
                o = {n: card(self.by[n]) for n in names[i:i + 150]}
                o[self.NULL] = self.NULL_CARD
                pt, c = self._dm_ask(query, o, "Which tool best serves what the agent wants to do? Choose None if no tool listed can do it.")
                cost += c
                cut = pt.get(self.NULL, 0.0)
                score.update({n: ps[sv] * v for n, v in pt.items() if n != self.NULL and v > cut})
        return sorted(score, key=score.get, reverse=True)[:min(k, cap)], cost

    def dm_null(self, query: str, k: int = K, cap: int = 5, tau: float = 0.4) -> tuple[list[str], float]:
        """Hierarchy DM with a calibrated null: empty answer only when P(none) at the service step is >= tau (set on the
        offline benchmark); otherwise the usual hierarchy ranking, at most 5 results. self.last_p_null records P(none)."""
        servers = sorted({t["server"] for t in self.tools})
        opts = {sv: SERVER_INFO.get(sv, sv) for sv in servers}
        opts[self.NULL] = "None of these services can do what the agent asked."
        ps, cost = self._dm_ask(query, opts, "Which service has the tool the agent needs? Choose None if no service can do it.")
        self.last_p_null = ps.get(self.NULL, 0.0)
        if self.last_p_null >= tau:
            return [], cost
        real = {sv: v for sv, v in ps.items() if sv != self.NULL}
        chosen, acc, tot = [], 0.0, sum(real.values()) or 1.0
        for sv in sorted(real, key=real.get, reverse=True):
            chosen.append(sv)
            acc += real[sv] / tot
            if acc >= 0.9 or len(chosen) >= 3:
                break
        score = {}
        for sv in chosen:
            names = [n for n in self.names if self.by[n]["server"] == sv]
            for i in range(0, len(names), 150):
                pt, c = self._dm_ask(query, {n: card(self.by[n]) for n in names[i:i + 150]}, "Which tool best serves what the agent wants to do?")
                cost += c
                score.update({n: real[sv] * v for n, v in pt.items()})
        return sorted(score, key=score.get, reverse=True)[:min(k, cap)], cost

    def search(self, mech: str, query: str, k: int = K, **kw) -> tuple[list[str], float]:
        return getattr(self, {"cc": "cc_search"}.get(mech, mech))(query, k, **kw)
