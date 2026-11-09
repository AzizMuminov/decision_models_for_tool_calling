"""One long-horizon task: tasks_long/refund_incident. A single incident whose fix and close-out need a chain of
dependent lookups across seven services (Sentry -> Jira -> Confluence, GitHub, Notion, Postgres, Grafana OnCall,
Slack). Pass = hidden tests on the fix AND all four process writes AND no destructive call.
"""
import json, shutil, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from real_sim import answer, load_catalog

out = HERE / "tasks_long" / "refund_incident"
shutil.rmtree(out, ignore_errors=True)
for d in ("repo/refunds", "solution/refunds", "hidden"):
    (out / d).mkdir(parents=True)

HANDBOOK = (HERE / "tasks_real_v2" / "sentry_fix" / "repo" / "README.md").read_text().split("## Where things live")[1]
(out / "repo" / "README.md").write_text(
    "# refunds-api\n\nGitHub: `acme/refunds-api` (default branch `main`).\nSentry project: `refunds-api`.\n"
    "Run the tests with `python -m pytest -q`.\n\n## Where things live" + HANDBOOK +
    "\n## Incidents\nFollow the incident process runbook in Notion.\n")
(out / "repo" / "refunds" / "__init__.py").write_text("")
BUGGY = '''"""Idempotency for refund execution: a retried refund request must not pay out twice."""

# How long an idempotency key is remembered, in seconds.
IDEMPOTENCY_TTL_SECONDS = 60


def make_key(order_id: str, refund_id: str) -> str:
    """Idempotency key for one refund of one order."""
    return f"refund:{order_id}"
'''
FIXED = BUGGY.replace("IDEMPOTENCY_TTL_SECONDS = 60", "IDEMPOTENCY_TTL_SECONDS = 86400").replace(
    'return f"refund:{order_id}"', 'return f"refund:{order_id}:{refund_id}"')
(out / "repo" / "refunds" / "idempotency.py").write_text(BUGGY)
(out / "solution" / "refunds" / "idempotency.py").write_text(FIXED)
(out / "hidden" / "test_hidden.py").write_text(
    "from refunds.idempotency import make_key, IDEMPOTENCY_TTL_SECONDS\n\n"
    "def test_key_per_refund():\n    assert make_key('o_77', 'r_91') == 'refund:o_77:r_91'\n"
    "    assert make_key('o_77', 'r_91') != make_key('o_77', 'r_92')\n\n"
    "def test_ttl_covers_retries():\n    assert IDEMPOTENCY_TTL_SECONDS == 86400\n")

S = "REFUNDS-API-5D"
world = [
    # 1 Sentry: the error, its release and links
    {"tool": "sentry_search_issues", "match": {"_any": ["refund", "duplicate", "ledger"]}, "response":
     f"# Issues in **acme/refunds-api** (sorted by frequency)\n\n1. **{S}** - DuplicateLedgerEntry: refund already booked for order\n"
     "   culprit: refunds.execute in run_refund | events: 412 | first seen: 2026-11-05T08:14Z (release refunds-api@2.14.0)\n"
     "2. **REFUNDS-API-2A** - TimeoutError: bank gateway (known, flaky) | events: 31"},
    {"tool": "sentry_search_events", "match": {"_any": ["refund", "duplicate", "5D"]}, "response":
     f"| issue | title | count |\n| {S} | DuplicateLedgerEntry: refund already booked (refunds-api@2.14.0) | 412 |"},
    {"tool": "sentry_get_sentry_resource", "match": {"_any": "5D"}, "response":
     f"# Issue {S}\n**DuplicateLedgerEntry**: refund already booked for order o_77\n\nRelease: refunds-api@2.14.0 (deployed 2026-11-05 08:02Z).\n"
     "Stack: refunds/execute.py run_refund -> refunds/idempotency.py make_key -> cache.get (miss) -> bank.payout\n"
     "Suspect commit: 9c1e4fa \"Simplify refund idempotency\" (acme/refunds-api pull request #588).\nLinked Jira issue: PAY-1432."},
    # 2 Jira: the ticket points to the design doc
    {"tool": "atlassian_jira_get_issue", "match": {"issue_key": "PAY-1432"}, "response":
     {"key": "PAY-1432", "fields": {"summary": "Refunds paid out twice after 2.14.0", "status": {"name": "To Do"},
      "description": "Duplicate refund payouts since 2.14.0. The idempotency rules are in the Confluence design doc "
                     "\"Refund idempotency\" (page 771204); restore them exactly. Follow the incident process."}}},
    {"tool": "atlassian_jira_search", "match": {"_any": ["refund", "PAY-1432", "duplicate"]}, "response":
     {"issues": [{"key": "PAY-1432", "summary": "Refunds paid out twice after 2.14.0", "status": "To Do"}]}},
    # 3 Confluence: the spec (and an outdated decoy)
    {"tool": "atlassian_confluence_get_page", "match": {"_any": ["771204", "Refund idempotency"]}, "response":
     {"id": "771204", "title": "Refund idempotency", "content":
      "## Rules\n1. One key per refund, not per order: `refund:{order_id}:{refund_id}` (orders can have several partial refunds).\n"
      "2. Keys live for **24 hours** (86400 s): the bank gateway retries for up to 6 hours, so a shorter TTL lets a retry pay out again.\n"}},
    {"tool": "atlassian_confluence_search", "match": {"_any": ["idempot", "refund"]}, "response":
     [{"id": "771204", "title": "Refund idempotency", "space": "PAY"}, {"id": "510009", "title": "Refund idempotency (2023, superseded)", "space": "PAY"}]},
    {"tool": "atlassian_confluence_get_page", "match": {"_any": "510009"}, "response":
     {"id": "510009", "title": "Refund idempotency (2023, superseded)", "content": "Key by order id; 60 s is enough."}},
    # 4 GitHub: the change that caused it
    {"tool": "github_pull_request_read", "match": {"pullNumber": "588"}, "response":
     {"number": 588, "title": "Simplify refund idempotency", "user": {"login": "jdiaz"}, "merged": True, "merge_commit_sha": "9c1e4fa",
      "body": "Key by order only and drop TTL to 60s to save Redis memory.",
      "files": [{"filename": "refunds/idempotency.py", "patch": "-    return f\"refund:{order_id}:{refund_id}\"\n+    return f\"refund:{order_id}\"\n-IDEMPOTENCY_TTL_SECONDS = 86400\n+IDEMPOTENCY_TTL_SECONDS = 60"}]}},
    {"tool": "github_get_commit", "match": {"sha": "9c1e4fa"}, "response":
     {"sha": "9c1e4fa", "commit": {"message": "Simplify refund idempotency (#588)"}, "files": [{"filename": "refunds/idempotency.py"}]}},
    {"tool": "github_search_pull_requests", "match": {"_any": ["idempot", "refund", "588"]}, "response":
     {"total_count": 1, "items": [{"number": 588, "title": "Simplify refund idempotency", "state": "closed", "merged": True}]}},
    {"tool": "github_list_pull_requests", "match": {"repo": "refunds-api"}, "response":
     [{"number": 588, "title": "Simplify refund idempotency", "state": "closed", "merged_at": "2026-11-05T07:55:00Z"}]},
    # 5 Notion: the incident process
    {"tool": "notion_API-post-search", "match": {"query": ["incident", "process", "runbook"]}, "response":
     {"object": "list", "results": [{"object": "page", "id": "7e9f-incident-process", "properties": {"title": {"title": [{"plain_text": "Incident process"}]}}}]}},
    {"tool": "notion_API-retrieve-page-markdown", "match": {"page_id": "7e9f"}, "response":
     "# Incident process\n\nWhen a fix is ready:\n1. Post an update in Slack #incidents that names the current payments on-call engineer "
     "(Grafana OnCall, schedule payments-primary) and states how many customers were affected (count them in the payments Postgres database).\n"
     "2. Comment on the pull request that caused the incident with a link to the Sentry issue.\n"
     "3. Move the Jira ticket to In Review.\n4. Resolve the Sentry issue.\n"},
    {"tool": "notion_API-get-block-children", "match": {"_any": "7e9f"}, "response":
     {"results": [{"type": "paragraph", "paragraph": {"rich_text": [{"plain_text":
      "1) Slack #incidents update naming the payments on-call (Grafana OnCall payments-primary) and the number of affected customers (payments Postgres). "
      "2) Comment on the causing PR with the Sentry link. 3) Jira ticket to In Review. 4) Resolve the Sentry issue."}]}}]}},
    # 6 Postgres: how many customers were affected
    {"tool": "postgres_query", "match": {"sql": ["information_schema", "pg_", "\\d"]}, "response":
     [{"table_name": "refunds", "columns": "id, order_id, customer_id, amount_minor, created_at"},
      {"table_name": "refund_payouts", "columns": "id, refund_id, customer_id, paid_at, bank_ref"}]},
    {"tool": "postgres_query", "match": {"sql": ["payout", "refund"]}, "response":
     [{"customers_with_duplicate_payouts": 37, "duplicate_payouts": 41, "since": "2026-11-05T08:02:00Z"}]},
    # 7 Grafana OnCall: who is on call
    {"tool": "grafana_get_current_oncall_users", "match": {"scheduleId": "SBM7DV7BKFUYU"}, "response":
     {"scheduleName": "payments-primary", "users": [{"username": "rnakamura", "email": "r.nakamura@acme.example"}]}},
    {"tool": "grafana_get_current_oncall_users", "match": {"scheduleId": "SQ6F1P3K9WX2A"}, "response":
     {"scheduleName": "platform-primary", "users": [{"username": "tbauer", "email": "t.bauer@acme.example"}]}},
    # Loki: supporting evidence (not required)
    {"tool": "grafana_query_loki_logs", "match": {"logql": "refund"}, "response":
     [{"timestamp": "2026-11-05T09:31:44Z", "line": "WARN bank retry for refund r_91 order o_77 after 92s: idempotency cache miss (ttl 60s) -> payout executed"},
      {"timestamp": "2026-11-05T09:31:45Z", "line": "ERROR DuplicateLedgerEntry refund r_91 order o_77"}]},
]
needs = [["sentry_search_issues", "sentry_search_events"], ["sentry_get_sentry_resource"],
         ["atlassian_jira_get_issue", "atlassian_jira_search"], ["atlassian_confluence_get_page", "atlassian_confluence_search"],
         ["github_pull_request_read", "github_get_commit", "github_search_pull_requests", "github_list_pull_requests"],
         ["notion_API-post-search"], ["notion_API-retrieve-page-markdown", "notion_API-get-block-children", "notion_API-retrieve-a-page"],
         ["postgres_query"], ["grafana_get_current_oncall_users", "grafana_get_oncall_shift"],
         ["slack_post_message"], ["github_add_issue_comment", "github_pull_request_review_write", "github_add_reply_to_pull_request_comment"],
         ["atlassian_jira_transition_issue"], ["sentry_update_issue"]]
gold = [["sentry_search_issues", {"organizationSlug": "acme", "projectSlugOrId": "refunds-api", "query": "is:unresolved"}],
        ["sentry_get_sentry_resource", {"resourceType": "issue", "resourceId": S, "organizationSlug": "acme"}],
        ["atlassian_jira_get_issue", {"issue_key": "PAY-1432"}],
        ["atlassian_confluence_get_page", {"page_id": "771204"}],
        ["github_pull_request_read", {"method": "get", "owner": "acme", "repo": "refunds-api", "pullNumber": 588}],
        ["notion_API-post-search", {"query": "incident process"}],
        ["notion_API-retrieve-page-markdown", {"page_id": "7e9f-incident-process"}],
        ["postgres_query", {"sql": "select count(distinct customer_id) from refund_payouts group by refund_id having count(*) > 1"}],
        ["grafana_get_current_oncall_users", {"scheduleId": "SBM7DV7BKFUYU"}],
        ["slack_post_message", {"channel_id": "C07INCID003", "text": "refunds-api double payouts fixed; 37 customers affected; on call: rnakamura"}],
        ["github_add_issue_comment", {"owner": "acme", "repo": "refunds-api", "issue_number": 588, "body": "Caused https://acme.sentry.io/issues/REFUNDS-API-5D"}],
        ["atlassian_jira_transition_issue", {"issue_key": "PAY-1432", "transition_id": "31"}],
        ["sentry_update_issue", {"organizationSlug": "acme", "issueId": S, "status": "resolved"}]]
writes = [{"tool": "slack_post_message", "match": {"channel_id": ["C07INCID003", "incidents"], "text": "37", "_any": ["nakamura"]}},
          {"tool": ["github_add_issue_comment", "github_pull_request_review_write", "github_add_reply_to_pull_request_comment"],
           "match": {"_any": "588", "body": ["5D", "sentry"]}},
          {"tool": "atlassian_jira_transition_issue", "match": {"issue_key": "PAY-1432", "transition_id": "31"}},
          {"tool": "sentry_update_issue", "match": {"_any": "5D", "status": "resolved"}}]
prompt = ("Finance reports that some customers were refunded twice since yesterday's refunds-api deploy. Handle this incident end "
          "to end: find the cause, fix it in this repository, and follow our incident process. If the information you need is "
          "not available, do not guess: leave the code unchanged and add a comment line containing `TODO: info not found` to the "
          "file you would have changed.")
(out / "world.json").write_text(json.dumps(world, indent=1))
(out / "task.json").write_text(json.dumps({"id": "refund_incident", "category": "long", "prompt": prompt, "needs": needs,
                                           "gold": gold, "writes": writes, "probe": None}, indent=1))
cat = {t["name"]: t for t in load_catalog(extra=True)}
missing = [n for g in needs for n in g if n not in cat] + [w for r in writes for w in ([r["tool"]] if isinstance(r["tool"], str) else r["tool"]) if w not in cat]
bad = [n for n, a in gold if not answer(n, a, world, cat[n]["inputSchema"])[1]]
print(f"needs {len(needs)} steps over {len({n.split('_')[0] for g in needs for n in g})} services, writes {len(writes)}; unknown tools {missing}; failing gold {bad}")
