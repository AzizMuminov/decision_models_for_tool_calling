"""Simulated back ends for the real MCP tool catalog (harness/real_catalog/*.json).

The catalog keeps each server's real tool names, descriptions and input schemas; only the answers are simulated.
A task's world is a list of rules {tool, match, response}. The first rule whose `match` values all occur
(case-insensitively) in the named arguments answers; the key "_any" matches anywhere in the arguments. Rules
shared by every task (the organisation's datasources, channels, teams, ...) live in COMMON and are tried after
the task's own rules.

Unmatched calls get what the real service would say about something that does not exist: an empty result for
list/search/query tools, a not-found error for get/read tools. Unmatched writes succeed (they are recorded and
graded). Tool names are `<server>_<tool>`, as opencode names MCP tools, unless the tool already starts with its
server's name (slack_*, linear_*).
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

CAT_DIR = Path(__file__).resolve().parent / "real_catalog"
EXTRA_DIR = Path(__file__).resolve().parent / "real_catalog_extra"  # more real servers, for the catalog-size test

SERVER_INFO = {
    "github": "GitHub: repositories, issues, pull requests, Actions CI runs and logs, releases, code search, security alerts",
    "grafana": "Grafana: dashboards, Prometheus metrics, Loki logs, Tempo traces, alerting, incidents, OnCall schedules",
    "sentry": "Sentry: application errors and exceptions, issues, events, stack traces",
    "atlassian": "Atlassian: Jira issues, sprints and transitions; Confluence pages and spaces",
    "linear": "Linear: issues, projects, cycles, workflow states, documents",
    "slack": "Slack: channels, messages, threads, users",
    "notion": "Notion: pages, databases, blocks, comments (team wiki and runbooks)",
    "kubernetes": "Kubernetes: pods, deployments, logs, events, kubectl operations, Helm",
    "mongodb": "MongoDB: databases, collections, documents, schemas, aggregations",
    "postgres": "PostgreSQL: run read-only SQL against the production database",
    "git": "Git: local repository operations (status, log, diff, branch, commit)",
    "playwright": "Playwright: drive a web browser (navigate, click, fill forms, screenshots)",
    # extra servers (catalog-size test)
    "airtable": "Airtable: bases, tables and records",
    "aws_cloudwatch": "AWS CloudWatch: log groups, Logs Insights queries, metrics and alarms",
    "aws_eks": "AWS EKS: Kubernetes clusters on AWS, pod logs, events, manifests",
    "browserbase": "Browserbase: cloud browser sessions and web automation",
    "circleci": "CircleCI: pipelines, build failure logs, flaky tests, job test results",
    "clickhouse": "ClickHouse: run SQL queries on analytics databases",
    "docker": "Docker: run docker commands",
    "elasticsearch": "Elasticsearch: indices, mappings, search queries",
    "figma": "Figma: design files and images",
    "filesystem": "Filesystem: read, write and search files on the server's disk",
    "gitlab": "GitLab: projects, merge requests, issues, pipelines, wiki, files",
    "heroku": "Heroku: apps, dynos, add-ons, logs, releases",
    "hubspot": "HubSpot CRM: contacts, companies, deals, associations",
    "memory": "Memory: a knowledge graph of entities, relations and observations",
    "monday": "monday.com: boards, items, updates, workspaces",
    "netlify": "Netlify: sites, deploys, teams, forms",
    "salesforce": "Salesforce: orgs, SOQL queries, metadata, Apex, users",
    "terraform": "Terraform Registry: providers, modules, policies",
}

WRITE_WORDS = ("create", "update", "add", "post", "reply", "delete", "merge", "push", "transition", "patch", "apply",
               "scale", "assign", "archive", "remove", "set", "move", "upload", "rotate", "dismiss", "write", "edit",
               "install", "uninstall", "upgrade", "drop", "insert", "rename", "star", "fork", "rollout", "exec", "cleanup",
               "trigger", "request", "manage", "mark", "complete", "duplicate", "convert", "transfer", "subscribe", "link",
               "batch_create", "copy", "rerun", "cancel", "resolve", "close")
DESTRUCTIVE = ("delete", "merge_pull_request", "drop",  # "merge" alone also matched merge-request comments and reads (fixed)
               "archive", "uninstall", "kubectl_apply", "kubectl_scale", "kubectl_patch",
               "kubectl_rollout", "exec_in_pod", "remove", "dismiss", "rotate", "cleanup", "update-many", "insert-many")


def full_name(server: str, tool: str) -> str:
    return tool if tool.lower().startswith(server + "_") else f"{server}_{tool}"


def load_catalog(servers: list[str] | None = None, extra: bool = False) -> list[dict]:
    tools = []
    files = sorted(CAT_DIR.glob("*.json")) + (sorted(EXTRA_DIR.glob("*.json")) if extra else [])
    for f in files:
        server = f.stem
        if servers and server not in servers:
            continue
        for t in json.loads(f.read_text()):
            tools.append({"name": full_name(server, t["name"]), "server": server, "tool": t["name"],
                          "description": t.get("description") or "", "inputSchema": t.get("input_schema") or {"type": "object", "properties": {}}})
    return tools


def is_write(name: str) -> bool:
    action = name.split("_", 1)[1] if "_" in name else name
    action = action.lower().replace("-", "_")
    if action.startswith(("api_get", "api_retrieve", "api_post_search", "api_query", "api_list")):
        return False
    if action.startswith(("api_post", "api_patch", "api_delete", "api_update", "api_create", "api_move")):
        return True
    return any(re.search(rf"(^|_){w}", action) for w in WRITE_WORDS)


def is_destructive(name: str) -> bool:
    n = name.lower().replace("-", "_")
    return any(d.replace("-", "_") in n for d in DESTRUCTIVE)


def _matches(rule: dict, args: dict) -> bool:
    blob = json.dumps(args).lower()
    for k, v in (rule.get("match") or {}).items():
        hay = blob if k == "_any" else json.dumps(args.get(k, None)).lower()
        alts = v if isinstance(v, list) else [v]
        if not any(str(a).lower() in hay for a in alts):
            return False
    return True


_PY = {"string": str, "integer": int, "number": (int, float), "boolean": bool, "array": list, "object": dict}


def validate(args: dict, schema: dict | None) -> str | None:
    """Required arguments and top-level types, as the real servers' input validation (zod / pydantic) checks them."""
    if not schema:
        return None
    props = schema.get("properties") or {}
    missing = [k for k in schema.get("required") or [] if k not in args]
    if missing:
        return f"Invalid arguments: missing required parameter(s) {missing}"
    for k, v in args.items():
        spec = props.get(k)
        if not isinstance(spec, dict):
            continue
        types = spec.get("type")
        types = types if isinstance(types, list) else [types] if types else []
        ok = [t for t in types if t in _PY]
        if v is None and "null" in types:
            continue
        if ok and not any(isinstance(v, _PY[t]) and not (t in ("integer", "number") and isinstance(v, bool)) for t in ok):
            return f"Invalid arguments: parameter '{k}' must be of type {'/'.join(ok)}, got {type(v).__name__}"
        if spec.get("enum") and v not in spec["enum"]:
            return f"Invalid arguments: parameter '{k}' must be one of {spec['enum']}"
    return None


def answer(name: str, args: dict, world: list[dict], schema: dict | None = None) -> tuple[object, bool]:
    """Returns (response, ok)."""
    err = validate(args, schema) if os.environ.get("REAL_VALIDATE") == "1" else None
    if err:
        return {"error": "invalid_arguments", "message": err}, False
    for rule in world + COMMON:
        if rule["tool"] == name and _matches(rule, args):
            r = rule["response"]
            return r, not (isinstance(r, dict) and "error" in r)
    if is_write(name):
        return {"ok": True, "id": f"sim-{abs(hash(json.dumps(args, sort_keys=True))) % 10**6}"}, True
    action = name.split("_", 1)[-1].lower()
    if re.match(r"(list|search|query|find|aggregate|count|get_.*s$)", action) or "search" in action:
        return {"items": [], "total_count": 0}, True
    return {"error": "not_found", "message": f"Resource not found for {json.dumps(args)}"}, False


# Organisation-wide facts every task can discover.
COMMON = [
    {"tool": "github_get_me", "response": {"login": "acme-bot", "name": "Acme CI bot", "orgs": ["acme"]}},
    {"tool": "sentry_find_organizations", "response": "# Organizations\n\n- **acme** (slug: `acme`, region: https://us.sentry.io)"},
    {"tool": "sentry_find_projects", "response": "# Projects in **acme**\n\n- payments-api\n- checkout-web\n- ledger-worker\n- refunds-api\n- billing-cron"},
    {"tool": "grafana_list_datasources", "response": [
        {"uid": "P8E80F9AEF21F6940", "name": "Loki (prod)", "type": "loki", "isDefault": False},
        {"uid": "PBFA97CFB590B2093", "name": "Prometheus (prod)", "type": "prometheus", "isDefault": True},
        {"uid": "P214B5B846CF3925F", "name": "Tempo (prod)", "type": "tempo", "isDefault": False}]},
    {"tool": "grafana_list_oncall_schedules", "response": [
        {"id": "SBM7DV7BKFUYU", "name": "payments-primary", "team": "Payments"},
        {"id": "SQ6F1P3K9WX2A", "name": "platform-primary", "team": "Platform"}]},
    {"tool": "grafana_list_oncall_teams", "response": [{"id": "TI73TDU19W48J", "name": "Payments"}, {"id": "TB2K8D0Y5S7QA", "name": "Platform"}]},
    {"tool": "slack_list_channels", "response": {"ok": True, "channels": [
        {"id": "C07PAYENG01", "name": "payments-eng", "topic": {"value": "Payments engineering"}},
        {"id": "C07DATA0002", "name": "data", "topic": {"value": "Data team"}},
        {"id": "C07INCID003", "name": "incidents", "topic": {"value": "Production incidents - post updates here"}},
        {"id": "C07PLATF004", "name": "platform", "topic": {"value": "Platform team"}},
        {"id": "C07GENRL005", "name": "general", "topic": {"value": "Company-wide"}}]}},
    {"tool": "mongodb_connect", "response": "Successfully connected to MongoDB. connectionId: prod-shop-a1b2"},
    {"tool": "mongodb_list-databases", "response": "Name: shop, Size: 48210944 bytes\nName: admin, Size: 40960 bytes"},
    {"tool": "mongodb_list-collections", "response": "Collections in database shop:\n- orders\n- customers\n- inventory"},
    {"tool": "linear_getTeams", "response": [{"id": "9f1c2e7a-eng", "key": "ENG", "name": "Engineering"}]},
    {"tool": "linear_getWorkflowStates", "match": {"teamId": "9f1c2e7a"}, "response": [
        {"id": "st-backlog-01", "name": "Backlog"}, {"id": "st-todo-02", "name": "Todo"}, {"id": "st-prog-03", "name": "In Progress"},
        {"id": "st-review-04", "name": "In Review"}, {"id": "st-done-05", "name": "Done"}]},
    {"tool": "atlassian_jira_get_transitions", "response": [
        {"id": "11", "name": "To Do"}, {"id": "21", "name": "In Progress"}, {"id": "31", "name": "In Review"}, {"id": "41", "name": "Done"}]},
    {"tool": "kubernetes_kubectl_context", "response": "Current context: prod-eu-1"},
    # existing containers with nothing relevant in them answer empty, not "not found"
    {"tool": "slack_get_channel_history", "match": {"channel_id": "C07"}, "response": {"ok": True, "messages": []}},
    {"tool": "slack_get_thread_replies", "match": {"channel_id": "C07"}, "response": {"ok": True, "messages": []}},
    {"tool": "slack_get_users", "response": {"ok": True, "members": [{"id": "U01OMAR", "name": "omar"}, {"id": "U02LENA", "name": "lena"}, {"id": "U03JDIAZ", "name": "jdiaz"}]}},
]
