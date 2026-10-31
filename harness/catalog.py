"""A fictional company's internal tool catalog, as an MCP-heavy agent would see it.

Twelve invented services ("Quillmark" internal platform), about 100 tools. Each tool has the shape
real MCP tools have: a multi-paragraph description (purpose, when to use, parameters, returns,
errors, limits) and a JSON-schema of typed parameters. Descriptions are generated from templates, so
they are more uniform than hand-written ones; their length (~2.4k characters, ~700 tokens) matches
the MCP tools measured in this project's own agent session.
"""

from __future__ import annotations

import json

SERVICES = {
    "tracklet": ("Tracklet", "the issue tracker (tickets, bugs, sprints)", "issue", "QM-123"),
    "buildwave": ("Buildwave", "the CI system (pipelines, runs, jobs, logs, test reports)", "pipeline run", "run-48211"),
    "flagpole": ("Flagpole", "the feature-flag service (flags, rollouts, segments)", "flag", "checkout.new_flow"),
    "confman": ("Confman", "the runtime configuration service (keys per environment)", "config key", "api.timeout_ms"),
    "lorebook": ("Lorebook", "the internal documentation wiki and API-contract registry", "page", "PAGE-881"),
    "datavault": ("Datavault", "read-only access to the analytics warehouse", "table", "orders_daily"),
    "pulse": ("Pulse", "metrics, dashboards, alerts and SLOs", "metric", "http.p99_latency"),
    "shipyard": ("Shipyard", "deployments, releases and service ownership", "deployment", "dep-2207"),
    "murmur": ("Murmur", "team chat (channels, threads, messages)", "message", "msg-55102"),
    "rota": ("Rota", "on-call schedules and shift swaps", "schedule", "payments-primary"),
    "grepline": ("Grepline", "cross-repository code search", "repository", "quillmark/checkout"),
    "cratedepot": ("Cratedepot", "the internal package registry", "package", "qm-http"),
}

# (service, action, object, extra params: list of (name, type, description, required))
TOOLS = [
    # Tracklet
    ("tracklet", "search_issues", "issues", [("query", "string", "Free-text query over title, description and comments.", True), ("status", "string", "Filter by status: open, in_progress, done.", False), ("limit", "integer", "Maximum results (1-100).", False)]),
    ("tracklet", "get_issue", "an issue", [("issue_key", "string", "Issue key, e.g. QM-412.", True)]),
    ("tracklet", "create_issue", "a new issue", [("title", "string", "Short summary.", True), ("description", "string", "Body in markdown.", False), ("project", "string", "Project key, e.g. QM.", True)]),
    ("tracklet", "update_issue", "an issue's fields", [("issue_key", "string", "Issue key.", True), ("fields", "object", "Map of field name to new value.", True)]),
    ("tracklet", "add_comment", "a comment on an issue", [("issue_key", "string", "Issue key.", True), ("body", "string", "Comment text.", True)]),
    ("tracklet", "list_sprints", "sprints of a board", [("board", "string", "Board name.", True)]),
    ("tracklet", "get_sprint", "a sprint and its issues", [("sprint_id", "string", "Sprint id.", True)]),
    ("tracklet", "assign_issue", "an issue to a person", [("issue_key", "string", "Issue key.", True), ("assignee", "string", "Username.", True)]),
    # Buildwave
    ("buildwave", "list_pipelines", "pipelines", [("repository", "string", "Repository slug.", False)]),
    ("buildwave", "list_pipeline_runs", "recent runs of a pipeline", [("pipeline", "string", "Pipeline name, e.g. reports-nightly.", True), ("status", "string", "Filter: success, failed, running.", False), ("limit", "integer", "Maximum runs.", False)]),
    ("buildwave", "get_pipeline_run", "one pipeline run with its jobs", [("run_id", "string", "Run id.", True)]),
    ("buildwave", "get_job_log", "the log output of a CI job", [("run_id", "string", "Run id.", True), ("job", "string", "Job name within the run.", True), ("tail", "integer", "Only the last N lines.", False)]),
    ("buildwave", "retry_job", "a failed job", [("run_id", "string", "Run id.", True), ("job", "string", "Job name.", True)]),
    ("buildwave", "list_artifacts", "artifacts of a run", [("run_id", "string", "Run id.", True)]),
    ("buildwave", "get_test_report", "the parsed test report of a run", [("run_id", "string", "Run id.", True)]),
    ("buildwave", "cancel_run", "a running pipeline", [("run_id", "string", "Run id.", True)]),
    # Flagpole
    ("flagpole", "list_flags", "feature flags", [("prefix", "string", "Only flags starting with this prefix.", False)]),
    ("flagpole", "get_flag", "a flag's definition and current value", [("flag", "string", "Flag key, e.g. checkout.new_flow.", True), ("environment", "string", "Environment: prod, staging, dev.", False)]),
    ("flagpole", "set_flag", "a flag's value", [("flag", "string", "Flag key.", True), ("value", "string", "New value (JSON-encoded).", True), ("environment", "string", "Environment.", True)]),
    ("flagpole", "get_flag_history", "the change history of a flag", [("flag", "string", "Flag key.", True)]),
    ("flagpole", "list_segments", "user segments", []),
    ("flagpole", "get_rollout", "a flag's percentage rollout", [("flag", "string", "Flag key.", True)]),
    ("flagpole", "evaluate_flag", "a flag for a given user", [("flag", "string", "Flag key.", True), ("user_id", "string", "User id.", True)]),
    ("flagpole", "archive_flag", "a retired flag", [("flag", "string", "Flag key.", True)]),
    # Confman
    ("confman", "get_config_value", "the value of a configuration key", [("key", "string", "Dotted key, e.g. api.timeout_ms.", True), ("environment", "string", "Environment: prod, staging, dev.", True)]),
    ("confman", "list_config_keys", "configuration keys", [("prefix", "string", "Key prefix.", False), ("environment", "string", "Environment.", False)]),
    ("confman", "set_config_value", "a configuration value", [("key", "string", "Dotted key.", True), ("value", "string", "New value.", True), ("environment", "string", "Environment.", True)]),
    ("confman", "get_config_history", "the change history of a key", [("key", "string", "Dotted key.", True), ("environment", "string", "Environment.", True)]),
    ("confman", "diff_environments", "configuration between two environments", [("left", "string", "First environment.", True), ("right", "string", "Second environment.", True), ("prefix", "string", "Key prefix.", False)]),
    ("confman", "list_environments", "environments", []),
    ("confman", "validate_config", "a proposed configuration change", [("key", "string", "Dotted key.", True), ("value", "string", "Proposed value.", True)]),
    ("confman", "export_config", "all keys of an environment", [("environment", "string", "Environment.", True)]),
    # Lorebook
    ("lorebook", "search_docs", "documentation pages", [("query", "string", "Free-text query.", True), ("space", "string", "Restrict to a space.", False)]),
    ("lorebook", "get_page", "a documentation page", [("page_id", "string", "Page id.", True)]),
    ("lorebook", "list_spaces", "documentation spaces", []),
    ("lorebook", "list_api_contracts", "registered API contracts", [("team", "string", "Owning team.", False)]),
    ("lorebook", "get_api_contract", "an API contract (payload schema) by name", [("name", "string", "Contract name, e.g. billing-invoice-v3.", True)]),
    ("lorebook", "get_page_history", "a page's revisions", [("page_id", "string", "Page id.", True)]),
    ("lorebook", "create_page", "a documentation page", [("space", "string", "Space.", True), ("title", "string", "Title.", True), ("body", "string", "Markdown body.", True)]),
    ("lorebook", "update_page", "a documentation page", [("page_id", "string", "Page id.", True), ("body", "string", "New markdown body.", True)]),
    # Datavault
    ("datavault", "list_tables", "warehouse tables", [("schema", "string", "Schema name.", False)]),
    ("datavault", "describe_table", "a table's columns", [("table", "string", "Table name.", True)]),
    ("datavault", "run_query", "a read-only SQL query", [("sql", "string", "SELECT statement.", True), ("limit", "integer", "Row limit.", False)]),
    ("datavault", "get_row_count", "the row count of a table", [("table", "string", "Table name.", True)]),
    ("datavault", "list_migrations", "applied schema migrations", []),
    ("datavault", "get_migration", "one schema migration", [("migration_id", "string", "Migration id.", True)]),
    ("datavault", "explain_query", "the query plan of a SQL query", [("sql", "string", "SELECT statement.", True)]),
    ("datavault", "sample_rows", "sample rows of a table", [("table", "string", "Table name.", True), ("n", "integer", "Number of rows.", False)]),
    # Pulse
    ("pulse", "query_metric", "a metric time series", [("metric", "string", "Metric name.", True), ("window", "string", "Time window, e.g. 1h, 24h.", False), ("service", "string", "Service filter.", False)]),
    ("pulse", "list_dashboards", "dashboards", [("team", "string", "Owning team.", False)]),
    ("pulse", "get_dashboard", "a dashboard and its panels", [("dashboard_id", "string", "Dashboard id.", True)]),
    ("pulse", "list_alerts", "active alerts", [("service", "string", "Service filter.", False)]),
    ("pulse", "get_alert", "an alert's definition and state", [("alert_id", "string", "Alert id.", True)]),
    ("pulse", "silence_alert", "an alert for a period", [("alert_id", "string", "Alert id.", True), ("duration", "string", "e.g. 2h.", True)]),
    ("pulse", "get_slo", "a service level objective", [("slo_id", "string", "SLO id.", True)]),
    ("pulse", "list_slos", "service level objectives", [("service", "string", "Service filter.", False)]),
    # Shipyard
    ("shipyard", "list_deployments", "recent deployments", [("service", "string", "Service name.", False), ("environment", "string", "Environment.", False)]),
    ("shipyard", "get_deployment", "one deployment", [("deployment_id", "string", "Deployment id.", True)]),
    ("shipyard", "rollback", "a service to a previous release", [("service", "string", "Service.", True), ("to_release", "string", "Release tag.", True)]),
    ("shipyard", "promote", "a release to an environment", [("service", "string", "Service.", True), ("release", "string", "Release tag.", True), ("environment", "string", "Target environment.", True)]),
    ("shipyard", "get_release_notes", "release notes of a release", [("service", "string", "Service.", True), ("release", "string", "Release tag.", True)]),
    ("shipyard", "list_services", "deployable services", []),
    ("shipyard", "get_service_owner", "the owning team of a service", [("service", "string", "Service.", True)]),
    ("shipyard", "get_deploy_freeze", "current deploy freezes", [("environment", "string", "Environment.", False)]),
    # Murmur
    ("murmur", "search_messages", "chat messages", [("query", "string", "Free-text query.", True), ("channel", "string", "Channel name.", False)]),
    ("murmur", "get_thread", "a message thread", [("thread_id", "string", "Thread id.", True)]),
    ("murmur", "post_message", "a chat message", [("channel", "string", "Channel.", True), ("text", "string", "Message text.", True)]),
    ("murmur", "list_channels", "chat channels", []),
    ("murmur", "get_user", "a user's profile", [("username", "string", "Username.", True)]),
    ("murmur", "list_users", "users of a team", [("team", "string", "Team.", False)]),
    # Rota
    ("rota", "get_oncall", "who is on call now", [("schedule", "string", "Schedule name.", True)]),
    ("rota", "list_schedules", "on-call schedules", []),
    ("rota", "get_schedule", "an on-call schedule", [("schedule", "string", "Schedule name.", True), ("days", "integer", "Days ahead.", False)]),
    ("rota", "swap_shift", "two on-call shifts", [("shift_a", "string", "Shift id.", True), ("shift_b", "string", "Shift id.", True)]),
    ("rota", "page_oncall", "the on-call engineer", [("schedule", "string", "Schedule name.", True), ("message", "string", "Page text.", True)]),
    # Grepline
    ("grepline", "search_code", "code across all company repositories", [("query", "string", "Literal or regex.", True), ("repository", "string", "Restrict to a repository.", False)]),
    ("grepline", "get_file_at_ref", "a file at a git ref", [("repository", "string", "Repository slug.", True), ("path", "string", "File path.", True), ("ref", "string", "Branch, tag or commit.", False)]),
    ("grepline", "list_repos", "company repositories", [("team", "string", "Owning team.", False)]),
    ("grepline", "get_blame", "line-by-line authorship of a file", [("repository", "string", "Repository slug.", True), ("path", "string", "File path.", True)]),
    ("grepline", "list_commits", "recent commits", [("repository", "string", "Repository slug.", True), ("path", "string", "Path filter.", False)]),
    ("grepline", "get_pull_request", "a pull request", [("repository", "string", "Repository slug.", True), ("number", "integer", "PR number.", True)]),
    # Cratedepot
    ("cratedepot", "get_package", "an internal package", [("package", "string", "Package name.", True)]),
    ("cratedepot", "list_versions", "versions of a package", [("package", "string", "Package name.", True)]),
    ("cratedepot", "get_changelog", "a package's changelog", [("package", "string", "Package name.", True), ("version", "string", "Version.", False)]),
    ("cratedepot", "check_vulnerabilities", "known vulnerabilities of a package version", [("package", "string", "Package name.", True), ("version", "string", "Version.", True)]),
    ("cratedepot", "list_dependents", "packages depending on a package", [("package", "string", "Package name.", True)]),
    ("cratedepot", "publish_package", "a package version", [("package", "string", "Package name.", True), ("version", "string", "Version.", True)]),
    ("cratedepot", "search_packages", "packages by keyword", [("query", "string", "Keyword.", True)]),
    ("tracklet", "link_issues", "two issues with a relation", [("from_key", "string", "Issue key.", True), ("to_key", "string", "Issue key.", True), ("relation", "string", "blocks, relates, duplicates.", True)]),
    ("tracklet", "list_projects", "projects", []),
    ("buildwave", "get_pipeline_config", "a pipeline's YAML definition", [("pipeline", "string", "Pipeline name.", True)]),
    ("buildwave", "list_runners", "CI runners and their status", [("pool", "string", "Runner pool.", False)]),
    ("flagpole", "create_flag", "a new feature flag", [("flag", "string", "Flag key.", True), ("default", "string", "Default value (JSON-encoded).", True)]),
    ("confman", "get_schema", "the type and allowed range of a configuration key", [("key", "string", "Dotted key.", True)]),
    ("lorebook", "get_runbook", "the runbook of a service", [("service", "string", "Service name.", True)]),
    ("datavault", "get_lineage", "upstream and downstream lineage of a table", [("table", "string", "Table name.", True)]),
    ("pulse", "create_annotation", "an annotation on dashboards", [("text", "string", "Annotation text.", True), ("service", "string", "Service.", False)]),
    ("shipyard", "get_service_dependencies", "the runtime dependencies of a service", [("service", "string", "Service.", True)]),
    ("murmur", "list_threads", "recent threads of a channel", [("channel", "string", "Channel.", True), ("limit", "integer", "Maximum threads.", False)]),
    ("grepline", "find_symbol", "the definition site of a symbol across repositories", [("symbol", "string", "Function, class or constant name.", True)]),
]

VERB = {"search": "Searches", "get": "Retrieves", "list": "Lists", "create": "Creates", "update": "Updates", "add": "Adds",
        "assign": "Assigns", "retry": "Retries", "cancel": "Cancels", "set": "Sets", "archive": "Archives", "evaluate": "Evaluates",
        "diff": "Compares", "validate": "Validates", "export": "Exports", "run": "Runs", "explain": "Explains", "sample": "Samples",
        "query": "Queries", "silence": "Silences", "rollback": "Rolls back", "promote": "Promotes", "post": "Posts", "swap": "Swaps",
        "page": "Pages", "check": "Checks", "publish": "Publishes"}
WRITE = {"create", "update", "add", "assign", "retry", "cancel", "set", "archive", "silence", "rollback", "promote", "post", "swap", "page", "publish"}


def describe(svc: str, action: str, obj: str, params: list) -> str:
    brand, what, noun, example = {**SERVICES, **EXTRA_SERVICES}[svc]
    verb = VERB.get(action.split("_")[0], "Operates on")
    writes = action.split("_")[0] in WRITE
    lines = [
        f"{verb} {obj} in {brand}, {what}.",
        "",
        f"Use this tool when the task requires information or actions that live in {brand} and are not available in the "
        f"local repository. {brand} is the system of record for every {noun} at Quillmark; values shown in local code, "
        f"comments or READMEs may be stale, so prefer {brand} when the two disagree.",
        "",
        "Parameters:",
    ]
    for name, typ, desc, req in params:
        lines.append(f"  - {name} ({typ}{', required' if req else ', optional'}): {desc}")
    if not params:
        lines.append("  (none)")
    lines += [
        "",
        f"Returns: a JSON object. On success it contains the requested {noun} data under `result`, plus `fetched_at` "
        f"(ISO-8601 UTC) and `source` (\"{svc}\"). Lists are paginated with `next_cursor`; pass it back to continue.",
        "",
        "Errors:",
        f"  - NOT_FOUND when the referenced {noun} does not exist (check spelling and environment).",
        "  - PERMISSION_DENIED when the service account lacks access; do not retry.",
        "  - RATE_LIMITED with `retry_after_s`; back off before retrying.",
        "",
        "Notes:",
        f"  - Example identifier format: `{example}`.",
        "  - Timestamps are UTC. Environment names are lowercase: prod, staging, dev.",
        ("  - This tool changes state in a shared system. Only call it when the task explicitly asks for that change; "
         "never use it to explore." if writes else "  - Read-only and safe to call repeatedly; results may be cached for up to 60 seconds."),
        f"  - Limits: 60 requests per minute per agent; responses larger than 256 KB are truncated with `truncated: true`.",
        "",
        "Example call:",
        "  " + json.dumps({n: (example if t == "string" and i == 0 else (10 if t == "integer" else ({} if t == "object" else f"<{n}>")))
                          for i, (n, t, _, _) in enumerate(params)}),
        "",
        "Example response:",
        "  " + json.dumps({"result": {"id": example, "kind": noun, "status": "ok", "summary": f"<{noun} details>",
                                      "updated_by": "svc-account", "links": {"self": f"https://{svc}.quillmark.internal/{example}"}},
                           "fetched_at": "2026-10-27T07:00:00Z", "source": svc, "next_cursor": None}),
        "",
        f"See also: other `{svc}_*` tools for related {noun} operations. Authentication is handled by the agent's service "
        "account; never pass credentials as parameters.",
    ]
    return "\n".join(lines)


# ---- 200 extra tools for the catalog-size sweep: 20 more services x 10 tools. Several are deliberately close to
# an original service (a support desk next to the issue tracker, an A/B-testing service next to feature flags,
# a secrets-free settings store next to the config service, a wiki-like knowledge base next to the docs registry).
EXTRA_SERVICES = {
    "helpdesk": ("Helpdesk", "the customer-support ticket system", "support ticket", "SUP-20391"),
    "abacus": ("Abacus", "A/B experiments and variant assignment", "experiment", "exp-checkout-cta"),
    "settingshub": ("Settingshub", "per-customer account settings (not platform configuration)", "account setting", "acct-551.notifications"),
    "kbase": ("Kbase", "the customer-facing knowledge base articles", "article", "KB-1207"),
    "ledgerly": ("Ledgerly", "the finance general ledger and journal entries", "journal entry", "JE-88102"),
    "crmline": ("Crmline", "the sales CRM (accounts, contacts, deals)", "deal", "DEAL-4471"),
    "peopleops": ("Peopleops", "the employee directory and org chart", "employee", "emp-1034"),
    "iamctl": ("Iamctl", "identity and access management (roles, grants)", "role grant", "grant-7781"),
    "bucketd": ("Bucketd", "object storage buckets and files", "object", "reports-archive/2026/10.csv"),
    "queueline": ("Queueline", "message queues and consumer lag", "queue", "orders-events"),
    "mailroom": ("Mailroom", "transactional email templates and sends", "email template", "tmpl-receipt-v4"),
    "indexer": ("Indexer", "search index management (mappings, reindex jobs)", "index", "products-v7"),
    "edgecache": ("Edgecache", "CDN cache rules and purges", "cache rule", "rule-static-assets"),
    "dnsctl": ("Dnsctl", "DNS zones and records", "DNS record", "api.quillmark.com"),
    "etlflow": ("Etlflow", "batch ETL workflows and their runs", "workflow", "wf-orders-daily"),
    "modelreg": ("Modelreg", "the ML model registry and deployments", "model", "fraud-scorer:v12"),
    "statusboard": ("Statusboard", "the public status page and incidents", "incident", "INC-2209"),
    "stockkeep": ("Stockkeep", "warehouse inventory levels", "SKU", "SKU-30021"),
    "auditlog": ("Auditlog", "the security audit log", "audit event", "evt-9918231"),
    "cronsched": ("Cronsched", "scheduled jobs (cron) and their history", "scheduled job", "job-nightly-export"),
}
EXTRA_ACTIONS = [
    ("search_{n}s", "{o}s", [("query", "string", "Free-text query.", True), ("limit", "integer", "Maximum results.", False)]),
    ("get_{n}", "a {o}", [("{n}_id", "string", "Identifier.", True)]),
    ("list_{n}s", "{o}s", [("owner", "string", "Owning team or account.", False)]),
    ("create_{n}", "a new {o}", [("name", "string", "Name or title.", True), ("body", "object", "Fields of the new record.", False)]),
    ("update_{n}", "a {o}", [("{n}_id", "string", "Identifier.", True), ("fields", "object", "Fields to change.", True)]),
    ("delete_{n}", "a {o}", [("{n}_id", "string", "Identifier.", True)]),
    ("get_{n}_history", "the change history of a {o}", [("{n}_id", "string", "Identifier.", True)]),
    ("export_{n}s", "{o}s as CSV", [("since", "string", "ISO date.", False)]),
    ("get_{n}_stats", "usage statistics of a {o}", [("{n}_id", "string", "Identifier.", True), ("window", "string", "e.g. 7d.", False)]),
    ("validate_{n}", "a proposed {o}", [("body", "object", "Proposed record.", True)]),
]


def _extra_tools() -> list:
    out = []
    for svc, (_, _, noun, _) in EXTRA_SERVICES.items():
        n = noun.replace(" ", "_").replace("-", "_").lower()
        for name, obj, params in EXTRA_ACTIONS:
            out.append((svc, name.format(n=n), obj.format(o=noun), [(pn.format(n=n), t, d, r) for pn, t, d, r in params]))
    return out


VERB.update({"delete": "Deletes"})
WRITE.update({"delete"})


def catalog(size: int = 100) -> list[dict]:
    """The original 100 tools, or (size=300) those plus 200 distractors from 20 more services."""
    if size not in (100, 300):
        raise ValueError("catalog size must be 100 or 300")
    out = []
    for svc, action, obj, params in TOOLS + (_extra_tools() if size == 300 else []):
        props = {n: {"type": t, "description": d} for n, t, d, _ in params}
        req = [n for n, _, _, r in params if r]
        out.append({"name": f"{svc}_{action}", "service": svc, "description": describe(svc, action, obj, params),
                    "inputSchema": {"type": "object", "properties": props, "required": req}})
    return out


if __name__ == "__main__":
    import sys

    c = catalog(int(sys.argv[1]) if len(sys.argv) > 1 else 100)
    sizes = [len(json.dumps(t)) for t in c]
    print(len(c), "tools; mean", sum(sizes) // len(c), "chars (~", sum(sizes) // len(c) // 3.44, "tokens); total ~", int(sum(sizes) / 3.44), "tokens")
