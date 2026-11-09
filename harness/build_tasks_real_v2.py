"""Task set v2 (harness/tasks_real_v2): same worlds as tasks_real, closer to how developers actually ask.

Changes from v1 (see the hermes_v3 analysis):
  * symptom-first prompts: the prompt no longer names the service ("check the logs in Loki" -> "customers stopped
    getting e-mails"); the agent has to work out where to look
  * every repo README carries the same short "where things live" handbook (like a CLAUDE.md / AGENTS.md), naming the
    company's tools, including legacy ones (GitLab, CloudWatch, EKS) that double as decoys in the 1,076-tool catalog
  * five new multi-service tasks, three of them on the extra servers (GitLab CI, CloudWatch, EKS), so the decoy
    servers are sometimes the right answer
  * the escape clause is unchanged

    python harness/build_tasks_real_v2.py && python harness/validate_tasks_real.py --dir tasks_real_v2 --extra
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_tasks_real as B  # noqa: E402  (defines B.TASKS on import)
from build_tasks_real import ESCAPE, LOKI, d  # noqa: E402

OUT = HERE / "tasks_real_v2"

HANDBOOK = d("""
    ## Where things live (engineering handbook)
    - Code and reviews: GitHub, org `acme`. A few legacy services are still on GitLab (group `acme`) with GitLab CI; their README says so.
    - CI: GitHub Actions (CircleCI was retired in 2025).
    - Errors: Sentry, org `acme` (one project per service).
    - Metrics, logs, alerts, incidents, on-call: Grafana (Prometheus, Loki, Grafana Incident, Grafana OnCall). Legacy AWS services log to CloudWatch (region eu-west-1).
    - Tickets: Payments uses Jira (project PAY); Engineering uses Linear (team ENG). Design docs are in Confluence; runbooks in Notion.
    - Chat: Slack. Incident updates go to #incidents.
    - Production: Kubernetes (cluster prod-eu-1). The data team's batch jobs run on AWS EKS (cluster `data-prod`).
    - Data: MongoDB (database `shop`), Postgres (payments, read-only).
    """)

VAGUE = {
    "gh_ci_fail": "Pull request #214 in acme/payments can't be merged because a check is red. Get it green and tell the PR author on the PR what you changed.",
    "gh_review": "There is reviewer feedback waiting on pull request #88 in acme/ledger. Address it and answer each piece of feedback where it was left.",
    "gh_issue": "Please implement acme/payments#301. Make sure you account for everything that was discussed on it.",
    "gh_open_pr": "Pagination in acme/catalog is broken (see #57). Fix it and get the change up for review from a branch called `fix/pagination-57` against `main`.",
    "gh_dependabot": "Security says acme/payments has dependencies with known vulnerabilities. Bump whatever needs bumping in requirements.txt to the fixed versions.",
    "gh_release": "We are behind on the ledger API. Point API_VERSION in client/version.py at the newest published ledger-api release (without the leading `v`).",
    "gh_codesearch": "SUPPORTED_CURRENCIES in shop/currencies.py has drifted from the company-wide canonical list, which lives in another repository. Make it match exactly.",
    "sentry_fix": "The noisiest production error of payments-api right now needs to go. Fix it, and close it out where it is tracked.",
    "sentry_events": "Since this morning's deploy some clients get HTTP 500 from POST /refunds. Find out why and fix it.",
    "loki_errors": "Customers stopped getting shipping e-mails at around 06:00 UTC today; the notifier service is the suspect. Find the cause and fix it.",
    "prom_threshold": "Set SLOW_REQUEST_MS in alerts/config.py to twice the checkout service's current p99 request latency, in milliseconds.",
    "grafana_incident": "We are in the middle of an incident about checkout timeouts and the responders agreed on a mitigation. Apply it in checkout/client.py and post an update to the incident that the change is ready.",
    "oncall_contact": "Set ESCALATION_CONTACT in ops/escalation.py to the e-mail address of whoever is on call for the payments team right now.",
    "k8s_crashloop": "ledger-worker keeps restarting in production (namespace `ledger`) since the last config change. Figure out why and fix worker/settings.py.",
    "k8s_oom": "report-builder (namespace `reports`) keeps dying in production. Find out why, then set its memory limit in deploy/report-builder.yaml the way RUNBOOK.md says.",
    "slack_decision": "The payments team agreed on a new webhook retry policy in a chat discussion. Implement it in webhooks/retry.py and confirm in that same discussion once done.",
    "jira_impl": "Pick up PAY-412: implement it, then move it to review.",
    "confluence_limits": "Our public API rate limits changed. Make RATE_LIMITS in api/limits.py match the currently documented limits.",
    "linear_bug": "Fix ENG-77 and move it to In Review.",
    "notion_runbook": "We are rotating the webhook signing keys. Do the code part of the rotation runbook in signing/keys.py.",
    "mongo_shape": "The nightly export has started crashing on recent orders. Make export/parse.py handle whatever orders look like now, without breaking older ones.",
    "pg_migration": "Add migrations/0042_payments_merchant_idx.sql with an index named payments_merchant_created_idx for looking up a merchant's payments by creation time. Careful: production column names do not match our ORM.",
    "multi_alert": "Something is paging for the checkout service. Find out what is wrong, fix checkout/gateway.py, and post a short summary for the incident responders.",
    "multi_jira_conf": "Implement PAY-530 as its design document specifies, then leave a one-line summary on the ticket.",
    "multi_sentry_gh": "A recent change to ledger-worker introduced a reconciliation bug that shows up as a production error. Find the error and the change that caused it, fix worker/rates.py, and flag it on that change with a link to the error.",
    "absent_sentry": "The billing-cron error someone mentioned in standup needs fixing in billing/cron.py.",
    "absent_slack": "Set CACHE_TTL_S in cache/config.py to the value the platform team agreed on.",
    "absent_jira": "Implement PAY-999 in payments/limits.py.",
    "absent_release": "Pin SDK_VERSION in sdk/version.py to the newest release of acme/ghost-sdk (without the leading `v`).",
}

NEW: list[dict] = []

NEW.append(dict(
    id="gl_pipeline", category="multi",
    prompt="The main-branch pipeline of billing-legacy is red. Make it green and explain the fix on the open merge request !12." + ESCAPE,
    files={"README.md": "# billing-legacy\n\nLegacy service. Code: **GitLab** project `acme/billing-legacy` (GitLab CI). Not on GitHub.\n\n" + HANDBOOK,
           "billing/__init__.py": "", "billing/invoice.py": "def invoice_number(year: int, seq: int) -> str:\n    return f\"INV-{year}-{seq}\"\n"},
    solution={"billing/invoice.py": "def invoice_number(year: int, seq: int) -> str:\n    return f\"INV-{year}-{seq:06d}\"\n"},
    hidden=d("""
        from billing.invoice import invoice_number
        def test_padding():
            assert invoice_number(2026, 42) == "INV-2026-000042"
            assert invoice_number(2026, 123456) == "INV-2026-123456"
        """),
    needs=[["gitlab_list_pipelines", "gitlab_get_pipeline", "gitlab_list_pipeline_jobs"], ["gitlab_get_pipeline_job_output"],
           ["gitlab_create_merge_request_thread"]],
    gold=[("gitlab_list_pipelines", {"project_id": "acme/billing-legacy", "ref": "main"}),
          ("gitlab_list_pipeline_jobs", {"project_id": "acme/billing-legacy", "pipeline_id": "8812"}),
          ("gitlab_get_pipeline_job_output", {"project_id": "acme/billing-legacy", "job_id": "77104"}),
          ("gitlab_create_merge_request_thread", {"project_id": "acme/billing-legacy", "merge_request_iid": "12", "body": "Fixed: zero-pad invoice sequence to 6 digits"})],
    writes=[{"tool": "gitlab_create_merge_request_thread", "match": {"merge_request_iid": "12"}}],
    world=[
        {"tool": "gitlab_list_pipelines", "match": {"_any": "billing-legacy"}, "response": [
            {"id": 8812, "iid": 311, "ref": "main", "status": "failed", "sha": "a91c2e0", "web_url": "https://gitlab.com/acme/billing-legacy/-/pipelines/8812"},
            {"id": 8790, "iid": 310, "ref": "main", "status": "success", "sha": "77d01b2"}]},
        {"tool": "gitlab_get_pipeline", "match": {"pipeline_id": "8812"}, "response": {"id": 8812, "status": "failed", "ref": "main", "jobs": [{"id": 77103, "name": "lint", "status": "success"}, {"id": 77104, "name": "test:contract", "status": "failed"}]}},
        {"tool": "gitlab_list_pipeline_jobs", "match": {"pipeline_id": "8812"}, "response": [{"id": 77103, "name": "lint", "stage": "test", "status": "success"}, {"id": 77104, "name": "test:contract", "stage": "test", "status": "failed"}]},
        {"tool": "gitlab_get_pipeline_job_output", "match": {"job_id": "77104"}, "response": d("""
            $ python -m pytest contract/ -q
            F
            ___________________________ test_invoice_number_format ___________________________
                def test_invoice_number_format():
                    # Finance ERP import (FIN-77): sequence part is zero-padded to 6 digits
            >       assert invoice_number(2026, 42) == "INV-2026-000042"
            E       AssertionError: assert 'INV-2026-42' == 'INV-2026-000042'
            FAILED contract/test_invoice.py::test_invoice_number_format
            1 failed in 0.21s
            ERROR: Job failed: exit code 1
            """)},
        {"tool": "gitlab_get_merge_request", "match": {"_any": "12"}, "response": {"iid": 12, "title": "Switch invoices to ERP v2 format", "state": "opened", "source_branch": "erp-v2", "target_branch": "main"}},
    ]))

NEW.append(dict(
    id="cw_reports", category="multi",
    prompt="The legacy reports API has been answering 502 since yesterday evening. Find the cause, fix reports/export.py, and post a short note for the incident responders." + ESCAPE,
    files={"README.md": "# reports-api (legacy)\n\nRuns on AWS (ECS). Logs: CloudWatch log group `/ecs/reports-api` in eu-west-1.\n\n" + HANDBOOK,
           "reports/__init__.py": "", "reports/export.py": "import csv, io\n\ndef export_rows(rows: list[dict]) -> str:\n    out = io.StringIO()\n    w = csv.DictWriter(out, fieldnames=[\"id\", \"amount\"])\n    w.writeheader()\n    for r in rows:\n        w.writerow(r)\n    return out.getvalue()\n"},
    solution={"reports/export.py": "import csv, io\n\ndef export_rows(rows: list[dict]) -> str:\n    out = io.StringIO()\n    w = csv.DictWriter(out, fieldnames=[\"id\", \"amount\"], extrasaction=\"ignore\")\n    w.writeheader()\n    for r in rows:\n        w.writerow(r)\n    return out.getvalue()\n"},
    hidden='from reports.export import export_rows\ndef test_new_field_no_crash():\n    # upstream added `currency`; keeping it as a column or ignoring it are both valid fixes\n    lines = export_rows([{"id": 1, "amount": 5, "currency": "EUR"}]).splitlines()\n    assert lines[0].split(",")[:2] == ["id", "amount"]\n    assert lines[1].split(",")[:2] == ["1", "5"]\ndef test_old_rows_still_work():\n    lines = export_rows([{"id": 2, "amount": 7}]).splitlines()\n    assert lines[1].split(",")[:2] == ["2", "7"]\n',
    needs=[["aws_cloudwatch_describe_log_groups", "aws_cloudwatch_execute_log_insights_query", "aws_cloudwatch_analyze_log_group"],
           ["aws_cloudwatch_execute_log_insights_query", "aws_cloudwatch_get_logs_insight_query_results"], ["slack_list_channels"], ["slack_post_message"]],
    gold=[("aws_cloudwatch_describe_log_groups", {"log_group_name_prefix": "/ecs/reports-api"}),
          ("aws_cloudwatch_execute_log_insights_query", {"log_group_names": ["/ecs/reports-api"], "start_time": "2026-11-04T18:00:00Z", "end_time": "2026-11-05T08:00:00Z", "query_string": "fields @timestamp, @message | filter @message like /ERROR/"}),
          ("aws_cloudwatch_get_logs_insight_query_results", {"query_id": "q-5c1d"}),
          ("slack_list_channels", {}), ("slack_post_message", {"channel_id": "C07INCID003", "text": "reports-api: CSV export now ignores unknown fields"})],
    writes=[{"tool": "slack_post_message", "match": {"channel_id": "C07INCID003"}}],
    world=[
        {"tool": "aws_cloudwatch_describe_log_groups", "response": {"log_groups": [{"logGroupName": "/ecs/reports-api", "storedBytes": 81233411}, {"logGroupName": "/ecs/billing-sync", "storedBytes": 2201933}]}},
        {"tool": "aws_cloudwatch_execute_log_insights_query", "match": {"_any": "reports"}, "response": {"queryId": "q-5c1d", "status": "Complete", "results": [
            [{"field": "@timestamp", "value": "2026-11-04 19:02:11"}, {"field": "@message", "value": "ERROR export failed: ValueError: dict contains fields not in fieldnames: 'currency'"}],
            [{"field": "@timestamp", "value": "2026-11-04 19:02:11"}, {"field": "@message", "value": "INFO upstream ledger 2026.11 added `currency` to every row"}]]}},
        {"tool": "aws_cloudwatch_get_logs_insight_query_results", "match": {"_any": "q-5c1d"}, "response": {"status": "Complete", "results": [[{"field": "@message", "value": "ERROR export failed: ValueError: dict contains fields not in fieldnames: 'currency'"}]]}},
        {"tool": "aws_cloudwatch_analyze_log_group", "match": {"_any": "reports"}, "response": {"top_errors": [{"pattern": "ERROR export failed: ValueError: dict contains fields not in fieldnames: 'currency'", "count": 4120}]}},
    ]))

NEW.append(dict(
    id="eks_importer", category="single",
    prompt="The data team's nightly batch-importer job has failed three nights in a row. Find out why and fix importer/config.py." + ESCAPE,
    files={"README.md": "# batch-importer\n\nData team batch job. Runs on AWS EKS cluster `data-prod`, namespace `batch` (not on the prod-eu-1 Kubernetes cluster).\n\n" + HANDBOOK,
           "importer/__init__.py": "", "importer/config.py": "BATCH_SIZE = 5000\nS3_PREFIX = \"imports/\"\n"},
    solution={"importer/config.py": "BATCH_SIZE = 1000\nS3_PREFIX = \"imports/\"\n"},
    hidden="from importer.config import BATCH_SIZE, S3_PREFIX\ndef test_cfg():\n    assert BATCH_SIZE == 1000 and S3_PREFIX == \"imports/\"\n",
    needs=[["aws_eks_list_k8s_resources", "aws_eks_get_k8s_events"], ["aws_eks_get_pod_logs", "aws_eks_get_cloudwatch_logs"]],
    gold=[("aws_eks_list_k8s_resources", {"cluster_name": "data-prod", "kind": "Pod", "api_version": "v1", "namespace": "batch"}),
          ("aws_eks_get_pod_logs", {"cluster_name": "data-prod", "namespace": "batch", "pod_name": "batch-importer-29188440-x7k2p"})],
    writes=[],
    probe={"file": "importer/config.py", "name": "BATCH_SIZE", "expected": 1000, "original": 5000},
    world=[
        {"tool": "aws_eks_list_k8s_resources", "match": {"_any": "data-prod"}, "response": {"items": [{"metadata": {"name": "batch-importer-29188440-x7k2p", "namespace": "batch"}, "status": {"phase": "Failed"}},
                                                                                                        {"metadata": {"name": "batch-importer-29187000-m2d9q", "namespace": "batch"}, "status": {"phase": "Failed"}}]}},
        {"tool": "aws_eks_get_k8s_events", "match": {"_any": "batch-importer"}, "response": {"events": [{"reason": "BackoffLimitExceeded", "message": "Job has reached the specified backoff limit"}]}},
        {"tool": "aws_eks_get_pod_logs", "match": {"_any": "batch-importer"}, "response": "2026-11-05T02:00:04Z INFO importing 2.3M rows, batch size 5000\n2026-11-05T02:00:09Z ERROR warehouse rejected batch: 413 Payload Too Large (max 1000 rows per request since warehouse v9)\n2026-11-05T02:00:09Z FATAL aborting"},
        {"tool": "aws_eks_get_cloudwatch_logs", "match": {"_any": "data-prod"}, "response": {"events": [{"message": "batch-importer: warehouse rejected batch: 413 Payload Too Large (max 1000 rows per request since warehouse v9)"}]}},
    ]))

NEW.append(dict(
    id="promo_sentry_jira", category="multi",
    prompt="Android users report they can't apply promo codes. Find the production error, check whether there is already a ticket with an agreed fix, implement it, and comment on the ticket." + ESCAPE,
    files={"README.md": "# checkout\n\nSentry project: `checkout-web`. Tickets: Jira PAY.\n\n" + HANDBOOK, "checkout/__init__.py": "",
           "checkout/promo.py": "def normalize_code(code: str) -> str:\n    return code\n"},
    solution={"checkout/promo.py": "def normalize_code(code: str) -> str:\n    return code.strip().upper().replace(\" \", \"\")\n"},
    hidden=d("""
        from checkout.promo import normalize_code
        def test_android_codes():
            assert normalize_code(" autumn24 ") == "AUTUMN24"
            assert normalize_code("AUT UMN24") == "AUTUMN24"
        """),
    needs=[["sentry_search_issues", "sentry_search_events"], ["atlassian_jira_search", "atlassian_jira_get_issue"], ["atlassian_jira_add_comment"]],
    gold=[("sentry_search_issues", {"organizationSlug": "acme", "projectSlugOrId": "checkout-web", "query": "promo"}),
          ("atlassian_jira_search", {"jql": "project = PAY AND text ~ promo"}),
          ("atlassian_jira_get_issue", {"issue_key": "PAY-618"}),
          ("atlassian_jira_add_comment", {"issue_key": "PAY-618", "body": "Implemented: codes are trimmed, upper-cased, spaces removed"})],
    writes=[{"tool": "atlassian_jira_add_comment", "match": {"issue_key": "PAY-618"}}],
    world=[
        {"tool": "sentry_search_issues", "match": {"_any": ["checkout", "promo", "android"]}, "response": "# Issues in **acme/checkout-web**\n\n1. **CHECKOUT-WEB-7Q** - PromoNotFound: code ' autumn24 ' does not exist\n   culprit: checkout.promo in apply | events: 2,940 | tags: os=Android 15, app=android 8.2 | status: unresolved\n   linked ticket: PAY-618"},
        {"tool": "sentry_get_sentry_resource", "match": {"_any": "7Q"}, "response": "# CHECKOUT-WEB-7Q\nPromoNotFound: code ' autumn24 ' does not exist\nAndroid keyboard adds spaces and lower-case. Linked Jira: PAY-618."},
        {"tool": "sentry_search_events", "match": {"_any": ["checkout", "promo"]}, "response": "| issue | title | count |\n| CHECKOUT-WEB-7Q | PromoNotFound ' autumn24 ' (Android; see PAY-618) | 2940 |"},
        {"tool": "atlassian_jira_search", "match": {"_any": ["promo", "PAY-618", "android"]}, "response": {"issues": [{"key": "PAY-618", "summary": "Android promo codes fail", "status": "To Do"}]}},
        {"tool": "atlassian_jira_get_issue", "match": {"issue_key": "PAY-618"}, "response": {"key": "PAY-618", "fields": {"summary": "Android promo codes fail", "description": "Agreed fix (product + mobile): normalise codes before lookup - strip surrounding whitespace, remove inner spaces, upper-case. Implement in checkout.promo.normalize_code."}}},
    ]))

NEW.append(dict(
    id="linear_notion_pr", category="multi",
    prompt="Do ENG-91 the way our API guidelines say, and open a pull request for it from a branch named `eng-91` against `main` in acme/public-api." + ESCAPE,
    files={"README.md": "# public-api\n\nGitHub: `acme/public-api`.\n\n" + HANDBOOK, "api/__init__.py": "",
           "api/headers.py": "def deprecation_headers(sunset_date: str) -> dict:\n    return {}\n"},
    solution={"api/headers.py": "def deprecation_headers(sunset_date: str) -> dict:\n    return {\"Deprecation\": \"true\", \"Sunset\": sunset_date, \"Link\": '<https://developer.acme.example/migrate>; rel=\"deprecation\"'}\n"},
    hidden=d("""
        from api.headers import deprecation_headers
        def test_headers():
            h = deprecation_headers("Wed, 31 Mar 2027 00:00:00 GMT")
            assert h["Deprecation"] == "true"
            assert h["Sunset"] == "Wed, 31 Mar 2027 00:00:00 GMT"
            assert 'rel="deprecation"' in h["Link"] and "developer.acme.example/migrate" in h["Link"]
        """),
    needs=[["linear_getIssueById", "linear_searchIssues"], ["notion_API-post-search"], ["notion_API-retrieve-page-markdown", "notion_API-get-block-children"],
           ["github_create_pull_request"]],
    gold=[("linear_searchIssues", {"query": "ENG-91"}), ("notion_API-post-search", {"query": "API guidelines"}),
          ("notion_API-retrieve-page-markdown", {"page_id": "5e6f-guidelines"}),
          ("github_create_pull_request", {"owner": "acme", "repo": "public-api", "title": "ENG-91 deprecation headers", "head": "eng-91", "base": "main"})],
    writes=[{"tool": "github_create_pull_request", "match": {"head": "eng-91", "base": "main"}}],
    world=[
        {"tool": "linear_getIssueById", "match": {"id": ["ENG-91", "b7"]}, "response": {"id": "b7c1-91", "identifier": "ENG-91", "title": "Add deprecation headers to v1 endpoints", "description": "Implement api.headers.deprecation_headers(sunset_date). Follow the 'API guidelines' page (section Deprecation) in Notion for the exact headers."}},
        {"tool": "linear_searchIssues", "match": {"_any": ["91", "deprecation"]}, "response": [{"id": "b7c1-91", "identifier": "ENG-91", "title": "Add deprecation headers to v1 endpoints", "description": "Implement api.headers.deprecation_headers(sunset_date). Follow the 'API guidelines' page (section Deprecation) in Notion for the exact headers."}]},
        {"tool": "notion_API-post-search", "match": {"query": ["guideline", "api", "deprecation"]}, "response": {"results": [{"object": "page", "id": "5e6f-guidelines", "properties": {"title": {"title": [{"plain_text": "API guidelines"}]}}}]}},
        {"tool": "notion_API-retrieve-page-markdown", "match": {"page_id": "5e6f"}, "response": "# API guidelines\n\n## Deprecation\nDeprecated endpoints return three headers:\n- `Deprecation: true`\n- `Sunset: <HTTP-date>` (the sunset date passed in)\n- `Link: <https://developer.acme.example/migrate>; rel=\"deprecation\"`\n"},
        {"tool": "notion_API-get-block-children", "match": {"_any": "5e6f"}, "response": {"results": [{"paragraph": {"rich_text": [{"plain_text": "Deprecation: headers Deprecation: true; Sunset: <HTTP-date>; Link: <https://developer.acme.example/migrate>; rel=\"deprecation\""}]}}]}},
    ]))


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    tasks = []
    for t in B.TASKS:
        t = dict(t)
        if t["id"] in VAGUE:
            t["prompt"] = VAGUE[t["id"]] + ESCAPE
        files = dict(t["files"])
        files["README.md"] = files.get("README.md", "") + "\n" + HANDBOOK
        t["files"] = files
        tasks.append(t)
    tasks += NEW
    for t in tasks:
        root = OUT / t["id"]
        for sub, files in (("repo", t["files"]), ("solution", t["solution"])):
            for rel, content in files.items():
                p = root / sub / rel
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(content)
        (root / "hidden").mkdir(parents=True)
        (root / "hidden" / "test_hidden.py").write_text(t["hidden"])
        (root / "world.json").write_text(json.dumps(t["world"], indent=1))
        (root / "task.json").write_text(json.dumps({k: t.get(k) for k in ("id", "category", "prompt", "needs", "gold", "writes", "probe")}, indent=1))
    for c in B.CONTROL:
        src = HERE / "tasks_v2" / c
        shutil.copytree(src, OUT / c)
        (OUT / c / "world.json").write_text("[]")
        m = json.loads((src / "task.json").read_text())
        m.update(gold=[], writes=[], probe=None)
        (OUT / c / "task.json").write_text(json.dumps(m, indent=1))
        readme = OUT / c / "repo" / "README.md"
        readme.write_text((readme.read_text() if readme.exists() else "") + "\n" + HANDBOOK)
    print(len(tasks) + len(B.CONTROL), "tasks in", OUT)


if __name__ == "__main__":
    main()
