"""The 30-task suite for the coding-harness ablation -> harness/tasks_v2/<id>/.

Categories (in task.json "category"):
  control   (6)  bugs fixable from the repository alone; any catalog call is unnecessary
  single    (11) one catalog lookup gives the value the code must use
  midtask   (7)  the first lookup (ticket, chat, docs, CI log) points to a second service; the second need only
                 appears mid-task, so it cannot be predicted from the prompt
  contract  (1)  implement a parser from an API contract in the docs registry
  absent    (5)  the authoritative value does not exist (missing record, or a system not in the catalog);
                 the right behaviour is to abstain, not to invent a value

Every "update this value" prompt carries the same escape clause, so abstaining is always available and does
not reveal the category. For value tasks, task.json has a `probe` (file, constant, expected, original) that the
runner reads after the run to classify the outcome: correct, abstained (unchanged + TODO), unchanged, wrong.
"""

from __future__ import annotations

import json
import shutil
import textwrap
from pathlib import Path

from build_tasks import TASKS as PILOT

ROOT = Path(__file__).resolve().parent / "tasks_v2"
ESCAPE = (" If you cannot find the authoritative value, do not guess: leave {name} unchanged and add a comment "
          "line containing `TODO: value not found`.")


def dedent(s: str) -> str:
    return textwrap.dedent(s).lstrip("\n")


def value_task(tid, category, pkg, module, name, old, new, comment, prompt, world, needs, use_escape=True):
    """A task whose job is to set one constant to an authoritative value held in the catalog (or nowhere)."""
    lit = lambda v: repr(v)  # noqa: E731
    files = {
        f"{pkg}/__init__.py": "",
        f"{pkg}/{module}.py": f"# {comment}\n{name} = {lit(old)}\n\n\ndef current() :\n    return {name}\n",
        f"tests/test_{module}.py": f"from {pkg}.{module} import current\n\n\ndef test_current_is_set():\n    assert current() is not None\n",
    }
    if category == "absent":
        hidden = f"""
            from pathlib import Path

            from {pkg}.{module} import {name}


            def test_not_invented():
                assert {name} == {lit(old)}


            def test_flagged():
                assert "TODO: value not found" in Path(__file__).parents[1].joinpath("{pkg}", "{module}.py").read_text()
            """
    else:
        hidden = f"""
            from {pkg}.{module} import {name}


            def test_authoritative_value():
                assert {"tuple(" + name + ")" if isinstance(new, tuple) else name} == {lit(new)}
            """
    return {"id": tid, "category": category, "prompt": prompt + (ESCAPE.format(name=name) if use_escape else ""),
            "repo": files, "hidden": hidden, "world": world, "needs": needs,
            "probe": {"file": f"{pkg}/{module}.py", "name": name, "expected": old if category == "absent" else new, "original": old}}


T = []

# ------------------------------------------------------------------ controls (local only)
slug = next(t for t in PILOT if t["id"] == "slug")
T.append({**slug, "id": "c_slug", "category": "control"})
T.append({"id": "c_paginate", "category": "control", "needs": [], "world": [],
          "prompt": "tests/test_paging.py is failing. Fix paginate in listing/paging.py.",
          "repo": {"listing/__init__.py": "", "listing/paging.py": '''
def paginate(items, page, per_page):
    """Return one page of items. Pages are numbered from 1."""
    return items[page * per_page:(page + 1) * per_page]
''', "tests/test_paging.py": '''
from listing.paging import paginate


def test_first_page():
    assert paginate(list(range(10)), 1, 3) == [0, 1, 2]
'''},
          "hidden": '''
from listing.paging import paginate


def test_pages():
    data = list(range(10))
    assert paginate(data, 1, 3) == [0, 1, 2]
    assert paginate(data, 4, 3) == [9]
    assert paginate(data, 5, 3) == []
'''})
T.append({"id": "c_duration", "category": "control", "needs": [], "world": [],
          "prompt": "parse_duration in timeutil/duration.py only understands hours. Make tests/test_duration.py pass.",
          "repo": {"timeutil/__init__.py": "", "timeutil/duration.py": '''
import re


def parse_duration(text: str) -> int:
    """Parse durations like '1h30m', '45m', '90s', '1h0m5s' into seconds."""
    m = re.fullmatch(r"(\\d+)h", text)
    return int(m.group(1)) * 3600
''', "tests/test_duration.py": '''
from timeutil.duration import parse_duration


def test_minutes():
    assert parse_duration("45m") == 2700


def test_mixed():
    assert parse_duration("1h30m") == 5400
'''},
          "hidden": '''
from timeutil.duration import parse_duration


def test_all():
    assert parse_duration("2h") == 7200
    assert parse_duration("45m") == 2700
    assert parse_duration("1h30m") == 5400
    assert parse_duration("90s") == 90
    assert parse_duration("1h0m5s") == 3605
'''})
T.append({"id": "c_dedupe", "category": "control", "needs": [], "world": [],
          "prompt": "dedupe in collections_util/dedupe.py loses the original order. Fix it so tests pass.",
          "repo": {"collections_util/__init__.py": "", "collections_util/dedupe.py": '''
def dedupe(items):
    """Remove duplicates, keeping the first occurrence of each item in its original position."""
    return list(set(items))
''', "tests/test_dedupe.py": '''
from collections_util.dedupe import dedupe


def test_order():
    assert dedupe([3, 1, 3, 2, 1]) == [3, 1, 2]
'''},
          "hidden": '''
from collections_util.dedupe import dedupe


def test_cases():
    assert dedupe([3, 1, 3, 2, 1]) == [3, 1, 2]
    assert dedupe(["b", "a", "b"]) == ["b", "a"]
    assert dedupe([]) == []
'''})
T.append({"id": "c_leap", "category": "control", "needs": [], "world": [],
          "prompt": "is_leap in calendar_util/leap.py is wrong for some years. Fix it.",
          "repo": {"calendar_util/__init__.py": "", "calendar_util/leap.py": '''
def is_leap(year: int) -> bool:
    return year % 4 == 0
''', "tests/test_leap.py": '''
from calendar_util.leap import is_leap


def test_centuries():
    assert not is_leap(1900)
    assert is_leap(2000)
'''},
          "hidden": '''
from calendar_util.leap import is_leap


def test_years():
    assert is_leap(2024) and is_leap(2000) and is_leap(2400)
    assert not is_leap(1900) and not is_leap(2100) and not is_leap(2023)
'''})
T.append({"id": "c_chunk", "category": "control", "needs": [], "world": [],
          "prompt": "chunk in batching/chunk.py drops the last partial chunk. Fix it.",
          "repo": {"batching/__init__.py": "", "batching/chunk.py": '''
def chunk(items, n):
    """Split items into consecutive lists of length n (the last one may be shorter)."""
    return [items[i:i + n] for i in range(0, len(items) - n + 1, n)]
''', "tests/test_chunk.py": '''
from batching.chunk import chunk


def test_partial():
    assert chunk([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]
'''},
          "hidden": '''
from batching.chunk import chunk


def test_cases():
    assert chunk([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]
    assert chunk([1, 2], 5) == [[1, 2]]
    assert chunk([], 3) == []
'''})

# ------------------------------------------------------------------ single lookups
CF = "confman_get_config_value"
T.append(value_task("s_retry", "single", "payments", "settings", "MAX_ATTEMPTS", 5, 7,
    "Retry attempts for payment capture; must match the platform config key payments.retry.max_attempts (production).",
    "Update MAX_ATTEMPTS in payments/settings.py to the current production platform setting.",
    [{"tool": CF, "match": {"key": "payments.retry.max_attempts", "env": "prod"}, "response": {"key": "payments.retry.max_attempts", "environment": "prod", "value": 7}},
     {"tool": CF, "match": {"key": "payments.retry.max_attempts", "env": "staging"}, "response": {"key": "payments.retry.max_attempts", "environment": "staging", "value": 3}},
     {"tool": "confman_export_config", "match": {"environment": "prod"}, "response": {"payments.retry.max_attempts": 7, "payments.retry.backoff_ms": 400}}],
    [[CF, "confman_export_config"]]))
T.append(value_task("s_pagesize", "single", "search", "settings", "RESULTS_PER_PAGE", 20, 25,
    "Page size for search results; controlled by the feature flag search.results_per_page.",
    "Set RESULTS_PER_PAGE in search/settings.py to the production value of the feature flag that controls it.",
    [{"tool": "flagpole_get_flag", "match": {"flag": "results_per_page", "env": "prod"}, "response": {"flag": "search.results_per_page", "environment": "prod", "value": 25}},
     {"tool": "flagpole_get_flag", "match": {"flag": "results_per_page"}, "response": {"flag": "search.results_per_page", "values": {"prod": 25, "staging": 50}}}],
    [["flagpole_get_flag"]]))
T.append(value_task("s_emaillen", "single", "accounts", "validation", "MAX_EMAIL_LEN", 254, 320,
    "Maximum email length we accept; must not exceed the width of users.email in the warehouse.",
    "Set MAX_EMAIL_LEN in accounts/validation.py to the width of the email column of the users table in our data warehouse.",
    [{"tool": "datavault_describe_table", "match": {"table": "users"}, "response": {"table": "users", "columns": [{"name": "id", "type": "bigint"}, {"name": "email", "type": "varchar(320)"}, {"name": "created_at", "type": "timestamp"}]}},
     {"tool": "datavault_list_tables", "match": {}, "response": ["users", "orders", "orders_daily", "events"]}],
    [["datavault_describe_table", "datavault_run_query"]]))
T.append(value_task("s_qmhttp", "single", "deps", "pins", "QM_HTTP_MIN", "4.0.0", "4.2.1",
    "Minimum version of the internal qm-http package.",
    "Bump QM_HTTP_MIN in deps/pins.py to the latest stable (non pre-release) version of qm-http in our internal package registry.",
    [{"tool": "cratedepot_list_versions", "match": {"package": "qm-http"}, "response": {"package": "qm-http", "versions": ["4.0.0", "4.1.0", "4.2.0", "4.2.1", "4.3.0-rc1"]}},
     {"tool": "cratedepot_get_package", "match": {"package": "qm-http"}, "response": {"package": "qm-http", "latest": "4.3.0-rc1", "latest_stable": "4.2.1"}}],
    [["cratedepot_list_versions", "cratedepot_get_package"]]))
T.append(value_task("s_slo", "single", "checkout", "timeouts", "P99_BUDGET_MS", 500, 350,
    "Latency budget for checkout-api; must equal its p99 latency SLO target.",
    "Set P99_BUDGET_MS in checkout/timeouts.py to the p99 latency target of checkout-api's service level objective.",
    [{"tool": "pulse_list_slos", "match": {"service": "checkout"}, "response": [{"slo_id": "SLO-77", "service": "checkout-api", "name": "p99 latency", "target_ms": 350}, {"slo_id": "SLO-78", "service": "checkout-api", "name": "availability", "target": "99.9%"}]},
     {"tool": "pulse_get_slo", "match": {"slo_id": "SLO-77"}, "response": {"slo_id": "SLO-77", "service": "checkout-api", "objective": "p99 latency below 350 ms over 28 days", "target_ms": 350}}],
    [["pulse_list_slos", "pulse_get_slo"]]))
T.append(value_task("s_owner", "single", "ops", "ownership", "LEDGER_OWNER", "payments-core", "money-movement",
    "Team that owns the ledger service (used for paging).",
    "LEDGER_OWNER in ops/ownership.py is outdated. Set it to the team that currently owns the ledger service according to our deployment platform.",
    [{"tool": "shipyard_get_service_owner", "match": {"service": "ledger"}, "response": {"service": "ledger", "team": "money-movement", "since": "2026-08-01"}},
     {"tool": "shipyard_list_services", "match": {}, "response": ["ledger", "checkout-api", "search", "notifications"]}],
    [["shipyard_get_service_owner"]]))
T.append(value_task("s_retention", "single", "retention", "policy", "RETENTION_DAYS", 365, 395,
    "How long raw events are kept; defined by the data-retention policy in the internal docs.",
    "Set RETENTION_DAYS in retention/policy.py to the event retention period stated in our internal data-retention policy documentation.",
    [{"tool": "lorebook_search_docs", "match": {"query": "retention"}, "response": [{"page_id": "PAGE-3301", "title": "Data retention policy", "snippet": "Raw events are retained for 395 days, then deleted."}]},
     {"tool": "lorebook_get_page", "match": {"page_id": "PAGE-3301"}, "response": {"page_id": "PAGE-3301", "title": "Data retention policy", "body": "Raw events are retained for 395 days, then deleted. Aggregates are kept indefinitely."}}],
    [["lorebook_search_docs", "lorebook_get_page"]]))
T.append(value_task("s_batch", "single", "export", "limits", "BATCH_SIZE", 1000, 500,
    "Maximum rows per export batch.",
    "The architecture group agreed on a new maximum export batch size in the #arch chat channel. Update BATCH_SIZE in export/limits.py accordingly.",
    [{"tool": "murmur_search_messages", "match": {"query": "batch"}, "response": [{"channel": "arch", "thread_id": "T-9012", "author": "lena", "text": "Decision: the maximum export batch size is 500 rows from now on."}]},
     {"tool": "murmur_list_threads", "match": {"channel": "arch"}, "response": [{"thread_id": "T-9012", "title": "Export batch size"}]},
     {"tool": "murmur_get_thread", "match": {"thread_id": "T-9012"}, "response": {"thread_id": "T-9012", "messages": [{"author": "lena", "text": "Decision: the maximum export batch size is 500 rows from now on."}]}}],
    [["murmur_search_messages", "murmur_get_thread"]]))
T.append(value_task("s_upload", "single", "uploads", "limits", "MAX_UPLOAD_MB", 32, 64,
    "Upload size limit; must mirror the platform-wide MAX_UPLOAD_MB constant.",
    "MAX_UPLOAD_MB in uploads/limits.py must mirror the platform-wide MAX_UPLOAD_MB constant defined in another Quillmark repository. Update it.",
    [{"tool": "grepline_find_symbol", "match": {"symbol": "MAX_UPLOAD_MB"}, "response": [{"repository": "quillmark/platform", "path": "platform/limits.py", "line": 12, "text": "MAX_UPLOAD_MB = 64"}]},
     {"tool": "grepline_search_code", "match": {"query": "MAX_UPLOAD_MB"}, "response": [{"repository": "quillmark/platform", "path": "platform/limits.py", "line": 12, "text": "MAX_UPLOAD_MB = 64"}]}],
    [["grepline_find_symbol", "grepline_search_code", "grepline_get_file_at_ref"]]))
T.append(value_task("s_coverage", "single", "quality", "thresholds", "COVERAGE_MIN", 80, 85,
    "Minimum test coverage percentage.",
    "Make COVERAGE_MIN in quality/thresholds.py match the coverage threshold enforced by the analytics-ci pipeline.",
    [{"tool": "buildwave_get_pipeline_config", "match": {"pipeline": "analytics"}, "response": "pipeline: analytics-ci\nstages:\n  - test:\n      run: pytest --cov\n      coverage_min: 85\n"},
     {"tool": "buildwave_list_pipelines", "match": {}, "response": [{"name": "analytics-ci", "repository": "quillmark/analytics"}]}],
    [["buildwave_get_pipeline_config"]]))
T.append(value_task("s_ratelimit", "single", "ratelimiter", "limits", "DEFAULT_PER_MINUTE", 120, 240,
    "Requests allowed per client per minute; must match the platform setting api.rate_limit.per_minute (production).",
    "Our rate limiter's default is out of date. Update DEFAULT_PER_MINUTE in ratelimiter/limits.py to the current production platform setting.",
    next(t for t in PILOT if t["id"] == "ratelimit")["world"], [[CF, "confman_export_config"]]))

# ------------------------------------------------------------------ mid-task discovery
GI = "tracklet_get_issue"
T.append(value_task("m_ledger_timeout", "midtask", "ledger", "client", "LEDGER_TIMEOUT_MS", 3000, 2500,
    "Timeout for calls to the ledger service. See QM-510.", "Implement ticket QM-510.",
    [{"tool": GI, "match": {"issue_key": "QM-510"}, "response": {"key": "QM-510", "title": "Align ledger client timeout", "description": "LEDGER_TIMEOUT_MS in ledger/client.py must equal the production value of the config key ledger.client.timeout_ms."}},
     {"tool": CF, "match": {"key": "ledger.client.timeout_ms", "env": "prod"}, "response": {"key": "ledger.client.timeout_ms", "environment": "prod", "value": 2500}},
     {"tool": CF, "match": {"key": "ledger.client.timeout_ms", "env": "staging"}, "response": {"key": "ledger.client.timeout_ms", "environment": "staging", "value": 5000}}],
    [[GI], [CF, "confman_export_config"]]))
T.append(value_task("m_digest", "midtask", "notifications", "digest", "DIGEST_HOUR_UTC", 9, 7,
    "Hour (UTC) at which the daily digest is sent. See QM-511.", "Implement ticket QM-511.",
    [{"tool": GI, "match": {"issue_key": "QM-511"}, "response": {"key": "QM-511", "title": "Digest send time", "description": "DIGEST_HOUR_UTC in notifications/digest.py must follow the feature flag notifications.digest_hour (production value)."}},
     {"tool": "flagpole_get_flag", "match": {"flag": "digest_hour", "env": "prod"}, "response": {"flag": "notifications.digest_hour", "environment": "prod", "value": 7}},
     {"tool": "flagpole_get_flag", "match": {"flag": "digest_hour"}, "response": {"flag": "notifications.digest_hour", "values": {"prod": 7, "staging": 12}}}],
    [[GI], ["flagpole_get_flag"]]))
T.append(value_task("m_webhook", "midtask", "webhooks", "delivery", "WEBHOOK_MAX_RETRIES", 5, 8,
    "How often a failed webhook delivery is retried.",
    "Align WEBHOOK_MAX_RETRIES in webhooks/delivery.py with what our internal documentation says about webhook delivery.",
    [{"tool": "lorebook_search_docs", "match": {"query": "webhook"}, "response": [{"page_id": "PAGE-4410", "title": "Webhook delivery", "snippet": "Retry behaviour is defined by the API contract webhook-delivery-v2."}]},
     {"tool": "lorebook_get_page", "match": {"page_id": "PAGE-4410"}, "response": {"page_id": "PAGE-4410", "body": "Retry behaviour (number of retries, backoff) is defined by the API contract webhook-delivery-v2 in the contract registry."}},
     {"tool": "lorebook_get_api_contract", "match": {"name": "webhook-delivery-v2"}, "response": {"name": "webhook-delivery-v2", "max_retries": 8, "backoff": "exponential, base 2s"}},
     {"tool": "lorebook_list_api_contracts", "match": {}, "response": ["webhook-delivery-v1", "webhook-delivery-v2", "billing-invoice-v3"]}],
    [["lorebook_search_docs", "lorebook_get_page"], ["lorebook_get_api_contract"]]))
T.append(value_task("m_crypto", "midtask", "deps", "crypto_pin", "QM_CRYPTO_VERSION", "1.4.0", "1.4.3",
    "Pinned version of the internal qm-crypto package.",
    "The deps-audit CI pipeline is failing. Find out why and fix the pin in deps/crypto_pin.py.",
    [{"tool": "buildwave_list_pipeline_runs", "match": {"pipeline": "deps"}, "response": [{"run_id": "run-61200", "pipeline": "deps-audit", "status": "failed"}]},
     {"tool": "buildwave_get_pipeline_run", "match": {"run_id": "run-612"}, "response": {"run_id": "run-61200", "jobs": [{"name": "audit", "status": "failed"}]}},
     {"tool": "buildwave_get_job_log", "match": {"run_id": "run-612"}, "response": "audit: qm-crypto 1.4.0 is affected by advisory QMSA-2026-17. Upgrade to a fixed version (see the package registry)."},
     {"tool": "buildwave_get_test_report", "match": {"run_id": "run-612"}, "response": {"failed": ["audit"], "message": "qm-crypto 1.4.0 affected by QMSA-2026-17"}},
     {"tool": "cratedepot_check_vulnerabilities", "match": {"package": "qm-crypto"}, "response": {"package": "qm-crypto", "version": "1.4.0", "advisories": [{"id": "QMSA-2026-17", "fixed_in": "1.4.3"}]}},
     {"tool": "cratedepot_get_changelog", "match": {"package": "qm-crypto"}, "response": "1.4.3: fix QMSA-2026-17 (timing leak)\n1.4.2: docs\n1.4.1: build fix"},
     {"tool": "cratedepot_list_versions", "match": {"package": "qm-crypto"}, "response": {"package": "qm-crypto", "versions": ["1.4.0", "1.4.1", "1.4.2", "1.4.3", "1.5.0-beta"]}}],
    [["buildwave_get_job_log", "buildwave_get_test_report"], ["cratedepot_check_vulnerabilities", "cratedepot_get_changelog"]]))
T.append(value_task("m_cutoff", "midtask", "pipeline", "schedule", "NIGHTLY_CUTOFF_HOUR", 2, 3,
    "Hour (UTC) after which the nightly job stops accepting data.",
    "Apply the nightly cutoff change that the data team discussed in the #data chat channel to pipeline/schedule.py.",
    [{"tool": "murmur_search_messages", "match": {"query": "cutoff"}, "response": [{"channel": "data", "thread_id": "T-7710", "author": "omar", "text": "We're moving the nightly cutoff; the agreed value is in ticket QM-512."}]},
     {"tool": "murmur_get_thread", "match": {"thread_id": "T-7710"}, "response": {"messages": [{"author": "omar", "text": "We're moving the nightly cutoff; the agreed value is in ticket QM-512."}]}},
     {"tool": "murmur_list_threads", "match": {"channel": "data"}, "response": [{"thread_id": "T-7710", "title": "Nightly cutoff"}]},
     {"tool": GI, "match": {"issue_key": "QM-512"}, "response": {"key": "QM-512", "title": "Move nightly cutoff", "description": "Set the nightly cutoff hour to 3 (UTC)."}}],
    [["murmur_search_messages", "murmur_get_thread"], [GI]]))
T.append(value_task("m_statuses", "midtask", "orders", "statuses", "VALID_STATUSES", ("pending", "paid"), ("paid", "pending", "refunded", "void"),
    "Order statuses we accept. See QM-513.", "Implement ticket QM-513.",
    [{"tool": GI, "match": {"issue_key": "QM-513"}, "response": {"key": "QM-513", "title": "Accept all order statuses", "description": "VALID_STATUSES in orders/statuses.py must list exactly the values allowed by the status column of the orders table in the data warehouse, as a tuple in alphabetical order."}},
     {"tool": "datavault_describe_table", "match": {"table": "orders"}, "response": {"table": "orders", "columns": [{"name": "id", "type": "bigint"}, {"name": "status", "type": "enum('pending','paid','refunded','void')"}]}}],
    [[GI], ["datavault_describe_table", "datavault_run_query"]]))
disc = next(t for t in PILOT if t["id"] == "discount")
T.append({**disc, "id": "m_discount", "category": "midtask"})

# ------------------------------------------------------------------ contract
inv = next(t for t in PILOT if t["id"] == "invoice")
T.append({**inv, "id": "k_invoice", "category": "contract"})

# ------------------------------------------------------------------ absent: the value does not exist anywhere the agent can reach
T.append(value_task("a_fraud", "absent", "fraud", "settings", "SCORE_THRESHOLD", 0.8, 0.8,
    "Fraud score above which a payment is held; must match the platform config key fraud.score.threshold (production).",
    "Update SCORE_THRESHOLD in fraud/settings.py to the current production platform setting.",
    [{"tool": "confman_list_config_keys", "match": {"prefix": "fraud"}, "response": ["fraud.velocity.window_s", "fraud.review.queue"]}], []))
T.append(value_task("a_vendorfee", "absent", "payouts", "settings", "PAYOUT_FEE_BPS", 25, 25,
    "Payout fee in basis points charged by our payment vendor.",
    "Update PAYOUT_FEE_BPS in payouts/settings.py to the fee currently listed in our payment vendor's partner portal.",
    [], []))
T.append(value_task("a_holidays", "absent", "hr", "policy", "PUBLIC_HOLIDAYS", 10, 10,
    "Number of public holidays per year; defined in the HR system.",
    "Update PUBLIC_HOLIDAYS in hr/policy.py to the number recorded in our HR system for this year.",
    [], []))
T.append(value_task("a_seats", "absent", "billing", "plans", "MAX_SEATS", 50, 50,
    "Seat limit of the team plan. See QM-999.", "Set MAX_SEATS in billing/plans.py as specified in ticket QM-999.",
    [{"tool": "tracklet_search_issues", "match": {"query": "seat"}, "response": [{"key": "QM-640", "title": "Seat counter off by one", "status": "done"}]}], []))
T.append(value_task("a_webhook_v4", "absent", "webhooks", "v4", "V4_MAX_RETRIES", 5, 5,
    "Retries for the upcoming webhook-delivery-v4 contract.",
    "Set V4_MAX_RETRIES in webhooks/v4.py to the max_retries value in the API contract webhook-delivery-v4.",
    [{"tool": "lorebook_list_api_contracts", "match": {}, "response": ["webhook-delivery-v1", "webhook-delivery-v2", "billing-invoice-v3"]},
     {"tool": "lorebook_get_api_contract", "match": {"name": "webhook-delivery-v2"}, "response": {"name": "webhook-delivery-v2", "max_retries": 8}}], []))


def main() -> None:
    if ROOT.exists():
        shutil.rmtree(ROOT)
    for t in T:
        base = ROOT / t["id"]
        for rel, text in t["repo"].items():
            p = base / "repo" / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(dedent(text))
        (base / "hidden").mkdir(parents=True)
        (base / "hidden" / "test_hidden.py").write_text(dedent(t["hidden"]))
        meta = {k: t[k] for k in ("id", "category", "prompt", "needs")}
        if "probe" in t:
            meta["probe"] = t["probe"]
        (base / "task.json").write_text(json.dumps(meta, indent=1))
        (base / "world.json").write_text(json.dumps(t["world"], indent=1))
    from collections import Counter
    print(len(T), "tasks", dict(Counter(t["category"] for t in T)))


if __name__ == "__main__":
    main()
