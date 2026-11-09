"""Tool-chaining links from the tool schemas, for any ranker: for each tool, the tools on the same server that
produce the IDs it requires (transition_id <- jira_get_transitions, datasourceUid <- list_datasources, ...).
A required parameter is an ID if its name ends in id/Id/Uid/uid/key/Key/slug/Slug; its stem (transition, datasource,
schedule, ...) must appear in the producer's tool name, and producers are list/get/find/search tools.
Written to harness/chain_graph.json: {tool: [producer, ...]} (the best producer per parameter).
"""
import json, re, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from real_sim import load_catalog

ID = re.compile(r"^(.*?)[_-]?(id|Id|ID|uid|Uid|UID|key|Key|slug|Slug|SlugOrId)$")
GENERIC = {"", "owner", "repo", "project", "page", "parent", "user", "account", "organization", "org", "workspace", "team", "connection"}
READ = ("list", "get", "find", "search", "query", "describe", "retrieve", "fetch")

def toks(s):
    s = re.sub(r"([a-z])([A-Z])", r"\1_\2", s).lower()
    return [t for t in re.split(r"[^a-z0-9]+", s) if t]

tools = load_catalog(extra=True)
by_server = {}
for t in tools:
    by_server.setdefault(t["server"], []).append(t)
graph = {}
for t in tools:
    req = (t["inputSchema"] or {}).get("required") or []
    prods = []
    for p in req:
        m = ID.match(p)
        if not m:
            continue
        stem = "_".join(toks(m.group(1)))
        if stem in GENERIC:
            continue
        stem_toks = stem.split("_")
        cands = []
        for u in by_server[t["server"]]:
            if u["name"] == t["name"]:
                continue
            ut = toks(u["tool"])
            if not ut or not (ut[0] in READ or (len(ut) > 1 and ut[1] in READ)):  # jira_get_..., API-get-...
                continue
            words = set(ut) | {w.rstrip("s") for w in ut}
            if all(w in words or w + "s" in words for w in stem_toks):
                cands.append((len(ut), ut[0] != "list", u["name"]))  # prefer short, list-style producers
        prods += [n for *_, n in sorted(cands)[:1]]  # the single best producer per ID parameter
    if prods:
        graph[t["name"]] = sorted(set(prods), key=prods.index)
(HERE / "chain_graph.json").write_text(json.dumps(graph, indent=1))
print(f"{len(graph)} of {len(tools)} tools have producer links")
for k in ["atlassian_jira_transition_issue", "grafana_query_loki_logs", "grafana_get_current_oncall_users", "sentry_update_issue",
          "github_pull_request_read", "slack_post_message", "slack_reply_to_thread", "notion_API-retrieve-page-markdown", "atlassian_confluence_get_page"]:
    print(f"  {k:40s} <- {graph.get(k)}")
