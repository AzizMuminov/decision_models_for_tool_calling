"""Builds the realistic task set (harness/tasks_real/) against the real MCP tool catalog.

Each task is a small Python repository plus a world for the simulated services. The information a task needs
lives only in those services (CI logs, Sentry, Loki, Prometheus, Kubernetes, Slack, Jira, Confluence, Linear,
Notion, MongoDB, Postgres); the sandbox has no network and no credentials, so the shell cannot reach it. Many
tasks also require writing back (comment on the PR, resolve the Sentry issue, move the ticket, reply in the
thread, open a PR), which is graded from the call log.

Per task:
  task.json    prompt, category, needs (ordered groups of acceptable tools), gold calls, required writes, probe
  repo/        the project the agent works in
  hidden/      test_hidden.py, run after the agent finishes
  solution/    files that solve the task (validated: hidden tests fail on repo/, pass with solution/)
  world.json   simulator rules

Categories: single (one service), multi (two or three services), control (no catalog tool needed),
absent (the information does not exist; the right move is to stop and leave a TODO).

    python harness/build_tasks_real.py && python harness/validate_tasks_real.py
"""

from __future__ import annotations

import json
import shutil
import textwrap
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "tasks_real"
ESCAPE = (" If the information you need is not available, do not guess: leave the code unchanged and add a comment "
          "line containing `TODO: info not found` to the file you would have changed.")


def d(s: str) -> str:
    return textwrap.dedent(s).lstrip("\n")


TASKS: list[dict] = []


def task(**kw):
    TASKS.append(kw)


PROJECT = lambda repo, extra="": d(f"""
    # {repo}

    GitHub: `acme/{repo}` (default branch `main`).
    {extra}
    Run the tests with `python -m pytest -q`.
    """)

# ----------------------------------------------------------------------------------------------- GitHub

task(id="gh_ci_fail", category="single",
     prompt="CI is failing on pull request #214 in acme/payments. Find out why, fix the code, and leave a comment on the PR "
            "saying what you changed." + ESCAPE,
     files={"README.md": PROJECT("payments"), "payments/__init__.py": "",
            "payments/fees.py": d("""
                def fee(amount: float) -> float:
                    \"\"\"Processing fee: 1% of the amount, rounded to cents.\"\"\"
                    return round(amount * 0.01 * 100, 2)
                """),
            "tests/test_fees.py": "from payments.fees import fee\n\ndef test_fee_basic():\n    assert fee(100.0) == 100.0\n"},
     solution={"payments/fees.py": d("""
                from decimal import Decimal, ROUND_HALF_UP

                def fee(amount: float) -> float:
                    \"\"\"Processing fee: 1% of the amount, rounded to cents.\"\"\"
                    return float((Decimal(str(amount)) * 100 * Decimal("0.01")).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))
                """)},
     hidden=d("""
        from payments.fees import fee
        def test_half_up():
            assert fee(2.675) == 2.68
            assert fee(1.005) == 1.01
            assert fee(100.0) == 100.0
        """),
     needs=[["github_pull_request_read", "github_actions_list", "github_actions_get"], ["github_get_job_logs"], ["github_add_issue_comment"]],
     gold=[("github_pull_request_read", {"method": "get_check_runs", "owner": "acme", "repo": "payments", "pullNumber": 214}),
           ("github_get_job_logs", {"owner": "acme", "repo": "payments", "job_id": 50112, "return_content": True}),
           ("github_add_issue_comment", {"owner": "acme", "repo": "payments", "issue_number": 214, "body": "Fixed rounding: ROUND_HALF_UP"})],
     writes=[{"tool": "github_add_issue_comment", "match": {"issue_number": "214"}}],
     world=[
        {"tool": "github_pull_request_read", "match": {"pullNumber": "214", "method": "get_check_runs"}, "response": {"total_count": 2, "check_runs": [
            {"id": 50111, "name": "lint", "status": "completed", "conclusion": "success"},
            {"id": 50112, "name": "contract-tests", "status": "completed", "conclusion": "failure", "details_url": "https://github.com/acme/payments/actions/runs/9931/job/50112"}]}},
        {"tool": "github_pull_request_read", "match": {"pullNumber": "214", "method": "get_status"}, "response": {"state": "failure", "statuses": [{"context": "contract-tests", "state": "failure", "target_url": "https://github.com/acme/payments/actions/runs/9931/job/50112"}]}},
        {"tool": "github_pull_request_read", "match": {"pullNumber": "214"}, "response": {"number": 214, "title": "Switch fee calculation to floats", "state": "open", "head": {"ref": "fees-float", "sha": "3f9ac1e"}, "base": {"ref": "main"}, "mergeable_state": "blocked"}},
        {"tool": "github_actions_list", "match": {"repo": "payments", "method": "list_workflow_jobs"}, "response": {"total_count": 2, "jobs": [{"id": 50111, "name": "lint", "conclusion": "success"}, {"id": 50112, "name": "contract-tests", "conclusion": "failure"}]}},
        {"tool": "github_actions_list", "match": {"repo": "payments"}, "response": {"total_count": 2, "workflow_runs": [
            {"id": 9931, "name": "CI", "head_branch": "fees-float", "event": "pull_request", "status": "completed", "conclusion": "failure", "pull_requests": [{"number": 214}]},
            {"id": 9925, "name": "CI", "head_branch": "main", "status": "completed", "conclusion": "success"}]}},
        {"tool": "github_actions_get", "match": {"resource_id": "9931"}, "response": {"id": 9931, "name": "CI", "conclusion": "failure", "jobs_url": "https://api.github.com/repos/acme/payments/actions/runs/9931/jobs", "failed_jobs": [{"id": 50112, "name": "contract-tests"}]}},
        {"tool": "atlassian_jira_get_issue", "match": {"issue_key": "PAY-88"}, "response": {"key": "PAY-88", "fields": {"summary": "Card-network fee rounding contract", "status": {"name": "Done"}, "description": "Fees must be computed on exact decimal values and rounded HALF-UP to cents (e.g. 2.675 -> 2.68, 1.005 -> 1.01). Binary float rounding is not acceptable."}}},
        {"tool": "atlassian_jira_search", "match": {"_any": "PAY-88"}, "response": {"issues": [{"key": "PAY-88", "summary": "Card-network fee rounding contract", "description": "Fees must be computed on exact decimal values and rounded HALF-UP to cents (2.675 -> 2.68)."}]}},
        {"tool": "github_get_job_logs", "match": {"_any": ["50112", "9931"]}, "response": d("""
            2026-11-04T09:12:01Z ============================= test session starts ==============================
            2026-11-04T09:12:02Z collected 41 items
            2026-11-04T09:12:03Z contract/test_fee_contract.py ....F......
            2026-11-04T09:12:03Z =================================== FAILURES ===================================
            2026-11-04T09:12:03Z ____________________________ test_fee_rounding_contract ____________________________
            2026-11-04T09:12:03Z     def test_fee_rounding_contract():
            2026-11-04T09:12:03Z         # Card-network contract (PAY-88): fees are rounded half-up on exact decimal values.
            2026-11-04T09:12:03Z >       assert fee(2.675) == 2.68
            2026-11-04T09:12:03Z E       assert 2.67 == 2.68
            2026-11-04T09:12:03Z E        +  where 2.67 = fee(2.675)
            2026-11-04T09:12:03Z contract/test_fee_contract.py:18: AssertionError
            2026-11-04T09:12:03Z FAILED contract/test_fee_contract.py::test_fee_rounding_contract - assert 2.67 == 2.68
            2026-11-04T09:12:03Z ========================= 1 failed, 40 passed in 1.92s =========================
            ##[error]Process completed with exit code 1.
            """)},
     ])

task(id="gh_review", category="single",
     prompt="Address the review comments on pull request #88 in acme/ledger, then reply to each review comment." + ESCAPE,
     files={"README.md": PROJECT("ledger"), "ledger/__init__.py": "",
            "ledger/entries.py": d("""
                def make_entry(account: str, amt: int) -> dict:
                    return {"account": account, "amount_cents": amt}
                """)},
     solution={"ledger/entries.py": d("""
                def make_entry(account: str, amount_cents: int) -> dict:
                    if amount_cents < 0:
                        raise ValueError("amount_cents must be non-negative")
                    return {"account": account, "amount_cents": amount_cents}
                """)},
     hidden=d("""
        import pytest
        from ledger.entries import make_entry
        def test_rename_and_validate():
            assert make_entry("a", amount_cents=5) == {"account": "a", "amount_cents": 5}
            with pytest.raises(ValueError):
                make_entry("a", amount_cents=-1)
        """),
     needs=[["github_pull_request_read"], ["github_add_reply_to_pull_request_comment", "github_add_issue_comment"]],
     gold=[("github_pull_request_read", {"method": "get_review_comments", "owner": "acme", "repo": "ledger", "pullNumber": 88}),
           ("github_add_reply_to_pull_request_comment", {"owner": "acme", "repo": "ledger", "pullNumber": 88, "commentId": 90011, "body": "Done"})],
     writes=[{"tool": "github_add_reply_to_pull_request_comment", "match": {"commentId": ["90011", "90012"]}}],
     world=[
        {"tool": "github_pull_request_read", "match": {"pullNumber": "88", "method": ["get_review_comments", "get_reviews", "get_comments"]}, "response": [
            {"id": 90011, "path": "ledger/entries.py", "line": 1, "user": {"login": "mkowalski"}, "body": "Please rename `amt` to `amount_cents` - the unit has to be in the name."},
            {"id": 90012, "path": "ledger/entries.py", "line": 2, "user": {"login": "mkowalski"}, "body": "Reject negative amounts with a ValueError; refunds go through a separate entry type."}]},
        {"tool": "github_pull_request_read", "match": {"pullNumber": "88"}, "response": {"number": 88, "title": "Add make_entry helper", "state": "open", "review_comments": 2, "head": {"ref": "entry-helper"}}},
     ])

task(id="gh_issue", category="single",
     prompt="Implement GitHub issue #301 in acme/payments. Read the whole discussion on the issue before you start." + ESCAPE,
     files={"README.md": PROJECT("payments"), "payments/__init__.py": "",
            "payments/currency.py": d("""
                def validate_currency(code: str) -> str:
                    return code
                """)},
     solution={"payments/currency.py": d("""
                ALLOWED = {"EUR", "USD", "GBP", "PLN", "XTS"}

                def validate_currency(code: str) -> str:
                    c = code.strip().upper()
                    if c not in ALLOWED:
                        raise ValueError(f"unsupported currency {code}")
                    return c
                """)},
     hidden=d("""
        import pytest
        from payments.currency import validate_currency
        def test_issue_301():
            assert validate_currency(" eur ") == "EUR"
            assert validate_currency("XTS") == "XTS"
            for bad in ["JPY", "CHF", "EURO"]:
                with pytest.raises(ValueError):
                    validate_currency(bad)
        """),
     needs=[["github_issue_read"]],
     gold=[("github_issue_read", {"method": "get", "owner": "acme", "repo": "payments", "issue_number": 301}),
           ("github_issue_read", {"method": "get_comments", "owner": "acme", "repo": "payments", "issue_number": 301})],
     writes=[],
     world=[
        {"tool": "github_issue_read", "match": {"issue_number": "301", "method": "get_comments"}, "response": [
            {"user": {"login": "jdiaz"}, "body": "Can we also accept the ISO test code XTS? Our sandbox merchants use it."},
            {"user": {"login": "apatel"}, "body": "Yes - add XTS. Everything else stays as in the description."}]},
        {"tool": "github_issue_read", "match": {"issue_number": "301"}, "response": {"number": 301, "title": "validate_currency should reject unsupported codes", "state": "open", "comments": 2,
            "body": "`validate_currency` currently accepts anything.\n\nIt should:\n- strip whitespace and upper-case the code\n- accept only EUR, USD, GBP, PLN\n- raise `ValueError` for anything else\n- return the normalised code"}},
     ])

task(id="gh_open_pr", category="single",
     prompt="Fix the bug described in issue #57 of acme/catalog. Then push your fix to a new branch named `fix/pagination-57` "
            "and open a pull request against `main`." + ESCAPE,
     files={"README.md": PROJECT("catalog"), "catalog/__init__.py": "",
            "catalog/pager.py": d("""
                def page(items: list, number: int, size: int) -> list:
                    start = number * size
                    return items[start:start + size]
                """)},
     solution={"catalog/pager.py": d("""
                def page(items: list, number: int, size: int) -> list:
                    start = (number - 1) * size
                    return items[start:start + size]
                """)},
     hidden=d("""
        from catalog.pager import page
        def test_one_based():
            xs = list(range(10))
            assert page(xs, 1, 3) == [0, 1, 2]
            assert page(xs, 4, 3) == [9]
        """),
     needs=[["github_issue_read"], ["github_create_branch", "github_push_files", "github_create_or_update_file"], ["github_create_pull_request"]],
     gold=[("github_issue_read", {"method": "get", "owner": "acme", "repo": "catalog", "issue_number": 57}),
           ("github_create_branch", {"owner": "acme", "repo": "catalog", "branch": "fix/pagination-57"}),
           ("github_push_files", {"owner": "acme", "repo": "catalog", "branch": "fix/pagination-57", "message": "Fix pagination", "files": [{"path": "catalog/pager.py", "content": "..."}]}),
           ("github_create_pull_request", {"owner": "acme", "repo": "catalog", "title": "Fix pagination (#57)", "head": "fix/pagination-57", "base": "main"})],
     writes=[{"tool": ["github_push_files", "github_create_or_update_file"], "match": {"branch": "fix/pagination-57", "_any": "pager.py"}},
             {"tool": "github_create_pull_request", "match": {"head": "fix/pagination-57", "base": "main"}}],
     world=[
        {"tool": "github_issue_read", "match": {"issue_number": "57"}, "response": {"number": 57, "title": "page() skips the first page", "state": "open",
            "body": "The API documents page numbers as **1-based** (`?page=1` is the first page), but `catalog.pager.page(items, 1, size)` returns the *second* page."}},
     ])

task(id="gh_dependabot", category="single",
     prompt="Resolve the open Dependabot alerts for acme/payments by updating the pins in requirements.txt to the patched versions." + ESCAPE,
     files={"README.md": PROJECT("payments"), "requirements.txt": "requests==2.32.3\nurllib3==2.2.1\njinja2==3.1.3\nclick==8.1.7\n"},
     solution={"requirements.txt": "requests==2.32.3\nurllib3==2.5.0\njinja2==3.1.6\nclick==8.1.7\n"},
     hidden=d("""
        from pathlib import Path
        def pins():
            return dict(l.split("==") for l in Path("requirements.txt").read_text().split() if "==" in l)
        def v(s):
            return tuple(int(x) for x in s.split("."))
        def test_patched():
            p = pins()
            assert v(p["urllib3"]) >= (2, 5, 0)
            assert v(p["jinja2"]) >= (3, 1, 6)
            assert p["requests"] == "2.32.3" and p["click"] == "8.1.7"
        """),
     needs=[["github_list_dependabot_alerts", "github_get_dependabot_alert"]],
     gold=[("github_list_dependabot_alerts", {"owner": "acme", "repo": "payments", "state": "open"})],
     writes=[],
     world=[
        {"tool": "github_list_dependabot_alerts", "match": {"repo": "payments"}, "response": [
            {"number": 12, "state": "open", "dependency": {"package": {"ecosystem": "pip", "name": "urllib3"}, "manifest_path": "requirements.txt"},
             "security_advisory": {"ghsa_id": "GHSA-pq67-6m6q-mj2v", "severity": "medium", "summary": "urllib3 redirects are not disabled when retries are disabled on PoolManager"},
             "security_vulnerability": {"vulnerable_version_range": "< 2.5.0", "first_patched_version": {"identifier": "2.5.0"}}},
            {"number": 11, "state": "open", "dependency": {"package": {"ecosystem": "pip", "name": "jinja2"}, "manifest_path": "requirements.txt"},
             "security_advisory": {"ghsa_id": "GHSA-cpwx-vrp4-4pq7", "severity": "medium", "summary": "Jinja2 vulnerable to sandbox breakout through attr filter"},
             "security_vulnerability": {"vulnerable_version_range": "< 3.1.6", "first_patched_version": {"identifier": "3.1.6"}}}]},
     ])

task(id="gh_release", category="single",
     prompt="client/version.py pins the ledger API version we target. Update API_VERSION to the latest release of acme/ledger-api "
            "(without the leading `v`)." + ESCAPE,
     files={"README.md": PROJECT("ledger-client"), "client/__init__.py": "", "client/version.py": 'API_VERSION = "4.9.0"\n'},
     solution={"client/version.py": 'API_VERSION = "4.12.1"\n'},
     hidden=d("""
        from client.version import API_VERSION
        def test_version():
            assert API_VERSION == "4.12.1"
        """),
     needs=[["github_get_latest_release", "github_list_releases", "github_list_tags"]],
     gold=[("github_get_latest_release", {"owner": "acme", "repo": "ledger-api"})],
     writes=[],
     probe={"file": "client/version.py", "name": "API_VERSION", "expected": "4.12.1", "original": "4.9.0"},
     world=[
        {"tool": "github_get_latest_release", "match": {"repo": "ledger-api"}, "response": {"tag_name": "v4.12.1", "name": "ledger-api 4.12.1", "published_at": "2026-10-29T14:02:11Z", "prerelease": False}},
        {"tool": "github_list_releases", "match": {"repo": "ledger-api"}, "response": [{"tag_name": "v4.13.0-rc.1", "prerelease": True}, {"tag_name": "v4.12.1", "prerelease": False}, {"tag_name": "v4.12.0", "prerelease": False}]},
        {"tool": "github_list_tags", "match": {"repo": "ledger-api"}, "response": [{"name": "v4.13.0-rc.1"}, {"name": "v4.12.1"}, {"name": "v4.12.0"}]},
     ])

task(id="gh_codesearch", category="single",
     prompt="Another repository in the acme organisation holds the canonical list of supported currencies. Find it and make "
            "SUPPORTED_CURRENCIES in shop/currencies.py match it exactly." + ESCAPE,
     files={"README.md": PROJECT("shop"), "shop/__init__.py": "", "shop/currencies.py": 'SUPPORTED_CURRENCIES = ["EUR", "USD"]\n'},
     solution={"shop/currencies.py": 'SUPPORTED_CURRENCIES = ["EUR", "USD", "GBP", "SEK", "PLN", "CZK"]\n'},
     hidden=d("""
        from shop.currencies import SUPPORTED_CURRENCIES
        def test_synced():
            assert sorted(SUPPORTED_CURRENCIES) == sorted(["EUR", "USD", "GBP", "SEK", "PLN", "CZK"])
        """),
     needs=[["github_search_code", "github_search_repositories"], ["github_get_file_contents"]],
     gold=[("github_search_code", {"query": "org:acme supported currencies"}),
           ("github_get_file_contents", {"owner": "acme", "repo": "platform-config", "path": "currencies.yaml"})],
     writes=[],
     world=[
        {"tool": "github_search_code", "match": {"query": ["currenc"]}, "response": {"total_count": 2, "items": [
            {"name": "currencies.yaml", "path": "currencies.yaml", "repository": {"full_name": "acme/platform-config"}},
            {"name": "currencies.py", "path": "shop/currencies.py", "repository": {"full_name": "acme/shop"}}]}},
        {"tool": "github_search_repositories", "match": {"query": ["config", "currenc", "acme"]}, "response": {"total_count": 1, "items": [{"full_name": "acme/platform-config", "description": "Canonical platform configuration: currencies, countries, limits"}]}},
        {"tool": "github_get_file_contents", "match": {"repo": "platform-config", "path": "currencies"}, "response": "# Canonical list - owned by Platform. Do not edit copies by hand.\nsupported_currencies:\n  - EUR\n  - USD\n  - GBP\n  - SEK\n  - PLN\n  - CZK\n"},
        {"tool": "github_get_file_contents", "match": {"repo": "platform-config"}, "response": [{"name": "README.md", "type": "file"}, {"name": "currencies.yaml", "type": "file"}, {"name": "countries.yaml", "type": "file"}]},
     ])

# ----------------------------------------------------------------------------------------------- Sentry

task(id="sentry_fix", category="single",
     prompt="Fix the most frequent unresolved Sentry error in the payments-api project (organisation `acme`) and then mark that "
            "Sentry issue as resolved." + ESCAPE,
     files={"README.md": PROJECT("payments", "Sentry project: `payments-api`."), "payments/__init__.py": "",
            "payments/discounts.py": d("""
                def apply_discount(total_cents: int, coupon: dict) -> int:
                    return total_cents - coupon.get("amount_cents") + 0
                """)},
     solution={"payments/discounts.py": d("""
                def apply_discount(total_cents: int, coupon: dict) -> int:
                    amount = coupon.get("amount_cents") or 0
                    return max(0, total_cents - amount)
                """)},
     hidden=d("""
        from payments.discounts import apply_discount
        def test_expired_coupon():
            assert apply_discount(1000, {"amount_cents": None, "expired": True}) == 1000
            assert apply_discount(1000, {}) == 1000
            assert apply_discount(1000, {"amount_cents": 300}) == 700
        """),
     needs=[["sentry_search_issues", "sentry_search_events"], ["sentry_get_sentry_resource", "sentry_search_events", "sentry_search_issues"], ["sentry_update_issue"]],
     gold=[("sentry_search_issues", {"organizationSlug": "acme", "projectSlugOrId": "payments-api", "query": "is:unresolved", "sort": "freq"}),
           ("sentry_get_sentry_resource", {"resourceType": "issue", "resourceId": "PAYMENTS-API-3K", "organizationSlug": "acme"}),
           ("sentry_update_issue", {"organizationSlug": "acme", "issueId": "PAYMENTS-API-3K", "status": "resolved"})],
     writes=[{"tool": "sentry_update_issue", "match": {"_any": "3K", "status": "resolved"}}],
     world=[
        {"tool": "sentry_search_issues", "match": {"_any": "payments"}, "response": d("""
            # Issues in **acme/payments-api** (sorted by frequency)

            1. **PAYMENTS-API-3K** - TypeError: unsupported operand type(s) for -: 'int' and 'NoneType'
               culprit: payments.discounts in apply_discount | events: 1,843 | users: 611 | status: unresolved | last seen: 4 min ago
            2. **PAYMENTS-API-2Z** - ConnectionResetError: [Errno 104] Connection reset by peer
               culprit: payments.gateway in post | events: 97 | users: 40 | status: unresolved | last seen: 2 h ago
            """)},
        {"tool": "sentry_get_sentry_resource", "match": {"_any": "3K"}, "response": d("""
            # Issue PAYMENTS-API-3K
            **TypeError**: unsupported operand type(s) for -: 'int' and 'NoneType'

            ## Latest event (2026-11-05T04:51:09Z, release payments@2026.11.4)
            ```
            File "payments/checkout.py", line 88, in finalize
              total = apply_discount(total, coupon)
            File "payments/discounts.py", line 2, in apply_discount
              return total_cents - coupon.get("amount_cents") + 0
            TypeError: unsupported operand type(s) for -: 'int' and 'NoneType'
            ```
            ### Local variables (apply_discount)
            total_cents = 1000
            coupon = {'code': 'AUTUMN24', 'amount_cents': None, 'expired': True}

            Tags: environment=prod, release=payments@2026.11.4
            Note from triage: expired coupons come back with amount_cents=None (or without the key); they must not change the total. A discount can never make the total negative.
            """)},
        {"tool": "sentry_search_events", "match": {"_any": ["payments", "3K", "TypeError"]}, "response": "Found 3 events.\n| issue | title | count |\n| PAYMENTS-API-3K | TypeError: unsupported operand type(s) for -: 'int' and 'NoneType' (payments/discounts.py in apply_discount; coupon.amount_cents=None for expired coupons) | 1843 |"},
     ])

task(id="sentry_events", category="single",
     prompt="Since this morning's deploy, POST /refunds returns 500 for some clients. Use Sentry (organisation `acme`, project "
            "refunds-api) to find the error and fix it." + ESCAPE,
     files={"README.md": PROJECT("refunds", "Sentry project: `refunds-api`."), "refunds/__init__.py": "",
            "refunds/handler.py": d("""
                def parse_refund(payload: dict) -> dict:
                    return {"order_id": payload["order_id"], "reason": payload["reason_code"]}
                """)},
     solution={"refunds/handler.py": d("""
                def parse_refund(payload: dict) -> dict:
                    reason = payload.get("reason_code", payload.get("reason"))
                    return {"order_id": payload["order_id"], "reason": reason}
                """)},
     hidden=d("""
        from refunds.handler import parse_refund
        def test_both_field_names():
            assert parse_refund({"order_id": 1, "reason_code": "damaged"})["reason"] == "damaged"
            assert parse_refund({"order_id": 2, "reason": "late"})["reason"] == "late"
        """),
     needs=[["sentry_search_issues", "sentry_search_events"], ["sentry_get_sentry_resource", "sentry_search_events", "sentry_search_issues"]],
     gold=[("sentry_search_issues", {"organizationSlug": "acme", "projectSlugOrId": "refunds-api", "query": "is:unresolved firstSeen:-24h"}),
           ("sentry_get_sentry_resource", {"resourceType": "issue", "resourceId": "REFUNDS-API-1C", "organizationSlug": "acme"})],
     writes=[],
     world=[
        {"tool": "sentry_search_issues", "match": {"_any": "refund"}, "response": "# Issues in **acme/refunds-api**\n\n1. **REFUNDS-API-1C** - KeyError: 'reason_code'\n   culprit: refunds.handler in parse_refund | events: 412 | first seen: 06:14 UTC today | status: unresolved"},
        {"tool": "sentry_get_sentry_resource", "match": {"_any": "1C"}, "response": d("""
            # Issue REFUNDS-API-1C
            **KeyError**: 'reason_code'
            ```
            File "refunds/handler.py", line 2, in parse_refund
              return {"order_id": payload["order_id"], "reason": payload["reason_code"]}
            KeyError: 'reason_code'
            ```
            Request body (scrubbed): {"order_id": 88123, "reason": "damaged"}
            Tags: client=ios-app 5.0.2, environment=prod
            Breadcrumb: mobile clients >= 5.0 send the field as `reason`; web and older apps still send `reason_code`. Both must keep working.
            """)},
        {"tool": "sentry_search_events", "match": {"_any": ["refund", "1C", "KeyError"]}, "response": "| issue | title | client | count |\n| REFUNDS-API-1C | KeyError: 'reason_code' in parse_refund; body has `reason` instead (ios-app 5.x sends `reason`, web sends `reason_code`) | ios-app 5.0.2 | 412 |"},
     ])

# ----------------------------------------------------------------------------------------------- Grafana

LOKI, PROM = "P8E80F9AEF21F6940", "PBFA97CFB590B2093"

task(id="loki_errors", category="single",
     prompt="The notifier service in prod has been logging errors since about 06:00 UTC and customers are not getting "
            "shipping e-mails. Check the service's logs in Loki and fix the cause." + ESCAPE,
     files={"README.md": PROJECT("notifier", "Logs: Loki, label `app=\"notifier\"`."), "notifier/__init__.py": "",
            "notifier/templates.py": d("""
                TEMPLATES = {"order_shipped": "order_shipped_v2", "order_refunded": "order_refunded_v1"}

                def template_for(event: str) -> str:
                    return TEMPLATES[event]
                """)},
     solution={"notifier/templates.py": d("""
                TEMPLATES = {"order_shipped": "order_shipped_v3", "order_refunded": "order_refunded_v1"}

                def template_for(event: str) -> str:
                    return TEMPLATES[event]
                """)},
     hidden=d("""
        from notifier.templates import template_for
        def test_template():
            assert template_for("order_shipped") == "order_shipped_v3"
            assert template_for("order_refunded") == "order_refunded_v1"
        """),
     needs=[["grafana_list_datasources"], ["grafana_query_loki_logs", "grafana_find_error_pattern_logs", "grafana_query_loki_patterns"]],
     gold=[("grafana_list_datasources", {"name": "loki"}),
           ("grafana_query_loki_logs", {"datasourceUid": LOKI, "logql": '{app="notifier"} |= "error"', "limit": 50})],
     writes=[],
     world=[
        {"tool": "grafana_query_loki_logs", "match": {"datasourceUid": LOKI, "logql": "notifier"}, "response": [
            {"timestamp": "2026-11-05T06:01:12Z", "labels": {"app": "notifier", "level": "error"}, "line": "ERROR render failed event=order_shipped template=order_shipped_v2: TemplateNotFound (available: order_shipped, order_shipped_v3, order_refunded_v1)"},
            {"timestamp": "2026-11-05T06:01:13Z", "labels": {"app": "notifier", "level": "info"}, "line": "INFO template store migrated to v3 layouts; v2 templates removed (CHG-2231)"},
            {"timestamp": "2026-11-05T06:03:40Z", "labels": {"app": "notifier", "level": "error"}, "line": "ERROR render failed event=order_shipped template=order_shipped_v2: TemplateNotFound (available: order_shipped, order_shipped_v3, order_refunded_v1)"}]},
        {"tool": "grafana_query_loki_logs", "match": {"datasourceUid": PROM}, "response": {"error": "datasource PBFA97CFB590B2093 is of type prometheus, not loki"}},
        {"tool": "grafana_query_loki_logs", "response": {"error": "datasource not found or not a Loki datasource"}},
        {"tool": "atlassian_jira_get_issue", "match": {"issue_key": "CHG-2231"}, "response": {"key": "CHG-2231", "fields": {"summary": "Template store: migrate to v3 layouts", "status": {"name": "Done"}, "description": "All v2 templates were removed on 2026-11-05 06:00 UTC. Services must reference the v3 names (order_shipped_v3). order_refunded_v1 is unchanged."}}},
        {"tool": "atlassian_jira_search", "match": {"_any": "CHG-2231"}, "response": {"issues": [{"key": "CHG-2231", "summary": "Template store: migrate to v3 layouts", "description": "v2 templates removed; use order_shipped_v3. order_refunded_v1 unchanged."}]}},
        {"tool": "grafana_find_error_pattern_logs", "match": {"_any": "notifier"}, "response": {"patterns": [{"pattern": "ERROR render failed event=order_shipped template=order_shipped_v2: TemplateNotFound (available: order_shipped, order_shipped_v3, order_refunded_v1)", "count": 3120}]}},
        {"tool": "grafana_query_loki_patterns", "match": {"datasourceUid": LOKI, "_any": "notifier"}, "response": [{"pattern": "ERROR render failed event=<_> template=order_shipped_v2: TemplateNotFound (available: order_shipped, order_shipped_v3, order_refunded_v1)", "samples": 3120}]},
     ])

task(id="prom_threshold", category="single",
     prompt="Set SLOW_REQUEST_MS in alerts/config.py to twice the current p99 request latency of the checkout service, in "
            "milliseconds, using the `http_request_duration_seconds` histogram in Prometheus." + ESCAPE,
     files={"README.md": PROJECT("alerts"), "alerts/__init__.py": "", "alerts/config.py": "SLOW_REQUEST_MS = 1000\n"},
     solution={"alerts/config.py": "SLOW_REQUEST_MS = 824\n"},
     hidden=d("""
        from alerts.config import SLOW_REQUEST_MS
        def test_threshold():
            assert SLOW_REQUEST_MS == 824
        """),
     needs=[["grafana_list_datasources"], ["grafana_query_prometheus", "grafana_query_prometheus_histogram"]],
     gold=[("grafana_list_datasources", {"name": "prom"}),
           ("grafana_query_prometheus", {"datasourceUid": PROM, "expr": 'histogram_quantile(0.99, sum by (le) (rate(http_request_duration_seconds_bucket{service="checkout"}[5m])))', "endTime": "now", "queryType": "instant"})],
     writes=[],
     probe={"file": "alerts/config.py", "name": "SLOW_REQUEST_MS", "expected": 824, "original": 1000},
     world=[
        {"tool": "grafana_query_prometheus", "match": {"datasourceUid": PROM, "expr": "checkout"}, "response": {"resultType": "vector", "result": [{"metric": {"service": "checkout"}, "value": [1793770000, "0.412"]}]}},
        {"tool": "grafana_query_prometheus_histogram", "match": {"datasourceUid": PROM, "_any": "checkout"}, "response": {"quantile": 0.99, "result": [{"metric": {"service": "checkout"}, "value": "0.412"}]}},
        {"tool": "grafana_query_prometheus", "response": {"error": "datasource not found or query returned no series; check datasourceUid and labels"}},
     ])

task(id="grafana_incident", category="single",
     prompt="There is an active Grafana incident about checkout timeouts. Apply the mitigation that was decided in the incident "
            "to checkout/client.py, then add an update to the incident saying the change is ready." + ESCAPE,
     files={"README.md": PROJECT("checkout"), "checkout/__init__.py": "", "checkout/client.py": "UPSTREAM_TIMEOUT_MS = 3000\nRETRY_BUDGET = 0\n"},
     solution={"checkout/client.py": "UPSTREAM_TIMEOUT_MS = 800\nRETRY_BUDGET = 2\n"},
     hidden=d("""
        from checkout import client
        def test_mitigation():
            assert client.UPSTREAM_TIMEOUT_MS == 800
            assert client.RETRY_BUDGET == 2
        """),
     needs=[["grafana_list_incidents"], ["grafana_get_incident"], ["grafana_add_activity_to_incident"]],
     gold=[("grafana_list_incidents", {"status": "active"}), ("grafana_get_incident", {"id": "INC-1182"}),
           ("grafana_add_activity_to_incident", {"incidentId": "INC-1182", "body": "Mitigation ready: timeout 800ms, retry budget 2"})],
     writes=[{"tool": "grafana_add_activity_to_incident", "match": {"incidentId": "1182"}}],
     world=[
        {"tool": "grafana_list_incidents", "response": [{"incidentID": "INC-1182", "title": "Checkout timeouts on payment confirmation", "status": "active", "severity": "major"},
                                                         {"incidentID": "INC-1171", "title": "Search latency", "status": "resolved"}]},
        {"tool": "grafana_get_incident", "match": {"id": "1182"}, "response": {"incidentID": "INC-1182", "title": "Checkout timeouts on payment confirmation", "status": "active",
            "activity": [{"time": "08:02", "user": "lena", "body": "Gateway p99 is 650ms; our 3s timeout makes threads pile up."},
                         {"time": "08:20", "user": "omar", "body": "DECISION (mitigation): set the upstream timeout to 800 ms and allow a retry budget of 2. Owner: checkout team."}]}},
     ])

task(id="oncall_contact", category="single",
     prompt="Set ESCALATION_CONTACT in ops/escalation.py to the e-mail address of whoever is on call right now for the payments "
            "team in Grafana OnCall." + ESCAPE,
     files={"README.md": PROJECT("ops"), "ops/__init__.py": "", "ops/escalation.py": 'ESCALATION_CONTACT = "payments-oncall@acme.example"\n'},
     solution={"ops/escalation.py": 'ESCALATION_CONTACT = "r.nakamura@acme.example"\n'},
     hidden=d("""
        from ops.escalation import ESCALATION_CONTACT
        def test_contact():
            assert ESCALATION_CONTACT == "r.nakamura@acme.example"
        """),
     needs=[["grafana_list_oncall_schedules", "grafana_list_oncall_teams"], ["grafana_get_current_oncall_users", "grafana_get_oncall_shift"]],
     gold=[("grafana_list_oncall_schedules", {}), ("grafana_get_current_oncall_users", {"scheduleId": "SBM7DV7BKFUYU"})],
     writes=[],
     probe={"file": "ops/escalation.py", "name": "ESCALATION_CONTACT", "expected": "r.nakamura@acme.example", "original": "payments-oncall@acme.example"},
     world=[
        {"tool": "grafana_get_current_oncall_users", "match": {"scheduleId": "SBM7DV7BKFUYU"}, "response": {"scheduleId": "SBM7DV7BKFUYU", "scheduleName": "payments-primary", "users": [{"id": "U4X3", "username": "rnakamura", "email": "r.nakamura@acme.example"}]}},
        {"tool": "grafana_get_current_oncall_users", "match": {"scheduleId": "SQ6F1P3K9WX2A"}, "response": {"scheduleName": "platform-primary", "users": [{"username": "tbauer", "email": "t.bauer@acme.example"}]}},
        {"tool": "grafana_get_oncall_shift", "match": {"_any": "SBM7"}, "response": {"schedule": "payments-primary", "users": [{"email": "r.nakamura@acme.example"}], "start": "2026-11-05T00:00:00Z"}},
     ])

# ----------------------------------------------------------------------------------------------- Kubernetes

task(id="k8s_crashloop", category="single",
     prompt="The ledger-worker deployment in the `ledger` namespace has been crash-looping since the last config change. Find out "
            "why from the cluster and fix worker/settings.py." + ESCAPE,
     files={"README.md": PROJECT("ledger-worker", "Deployed to Kubernetes namespace `ledger`."), "worker/__init__.py": "",
            "worker/settings.py": d("""
                import os

                def db_url() -> str:
                    return os.environ["LEDGER_DB_URL"]
                """)},
     solution={"worker/settings.py": d("""
                import os

                def db_url() -> str:
                    return os.environ.get("DATABASE_URL") or os.environ["LEDGER_DB_URL"]
                """)},
     hidden=d("""
        import os
        from worker.settings import db_url
        def test_env(monkeypatch):
            monkeypatch.delenv("LEDGER_DB_URL", raising=False)
            monkeypatch.setenv("DATABASE_URL", "postgres://x")
            assert db_url() == "postgres://x"
        """),
     needs=[["kubernetes_kubectl_get", "kubernetes_kubectl_describe"], ["kubernetes_kubectl_logs"]],
     gold=[("kubernetes_kubectl_get", {"resourceType": "pods", "namespace": "ledger"}),
           ("kubernetes_kubectl_logs", {"resourceType": "pod", "name": "ledger-worker-7d9f6c5b8-xk2pq", "namespace": "ledger", "previous": True})],
     writes=[],
     world=[
        {"tool": "kubernetes_kubectl_get", "match": {"_any": "configmap"}, "response": "apiVersion: v1\nkind: ConfigMap\nmetadata:\n  name: ledger-config\n  namespace: ledger\n  annotations:\n    change: \"rev 12 - platform migration: LEDGER_DB_URL renamed to DATABASE_URL for all services\"\ndata:\n  LOG_LEVEL: info\n"},
        {"tool": "kubernetes_kubectl_get", "match": {"_any": "ledger"}, "response": "NAME                            READY   STATUS             RESTARTS        AGE\nledger-worker-7d9f6c5b8-xk2pq   0/1     CrashLoopBackOff   14 (2m ago)     48m\nledger-api-5c8d7f9b6-2mzl4      1/1     Running            0               3d"},
        {"tool": "kubernetes_kubectl_describe", "match": {"_any": "ledger-worker"}, "response": "Name: ledger-worker-7d9f6c5b8-xk2pq\nNamespace: ledger\nContainers:\n  worker:\n    State: Waiting (CrashLoopBackOff)\n    Last State: Terminated, Reason: Error, Exit Code: 1\n    Environment From: ledger-config (ConfigMap, revision 12)\n    Environment:\n      DATABASE_URL: <set from secret ledger-db>\nEvents:\n  Warning BackOff 2m kubelet Back-off restarting failed container worker"},
        {"tool": "kubernetes_kubectl_logs", "match": {"_any": "ledger-worker"}, "response": "2026-11-05T05:02:11Z INFO starting ledger-worker 2026.11.2\nTraceback (most recent call last):\n  File \"/app/worker/main.py\", line 9, in <module>\n    engine = create_engine(settings.db_url())\n  File \"/app/worker/settings.py\", line 4, in db_url\n    return os.environ[\"LEDGER_DB_URL\"]\nKeyError: 'LEDGER_DB_URL'\n# note: ConfigMap ledger-config rev 12 (platform migration) renamed LEDGER_DB_URL -> DATABASE_URL for all services"},
     ])

task(id="k8s_oom", category="single",
     prompt="The report-builder pods in namespace `reports` keep getting killed. Check the cluster, then set the container memory "
            "limit in deploy/report-builder.yaml as our runbook (RUNBOOK.md) says." + ESCAPE,
     files={"README.md": PROJECT("reports"), "RUNBOOK.md": "Memory limits: take the peak memory the killed container reached, and round it UP to the next multiple of 256Mi.\n",
            "deploy/report-builder.yaml": "apiVersion: apps/v1\nkind: Deployment\nmetadata:\n  name: report-builder\nspec:\n  template:\n    spec:\n      containers:\n        - name: builder\n          image: acme/report-builder:2026.10\n          resources:\n            limits:\n              memory: 512Mi\n"},
     solution={"deploy/report-builder.yaml": "apiVersion: apps/v1\nkind: Deployment\nmetadata:\n  name: report-builder\nspec:\n  template:\n    spec:\n      containers:\n        - name: builder\n          image: acme/report-builder:2026.10\n          resources:\n            limits:\n              memory: 1024Mi\n"},
     hidden=d("""
        import re
        from pathlib import Path
        def test_limit():
            m = re.search(r"memory:\\s*\\"?(\\d+)(Mi|Gi)", Path("deploy/report-builder.yaml").read_text())
            n = int(m.group(1)) * (1024 if m.group(2) == "Gi" else 1)
            assert n == 1024
        """),
     needs=[["kubernetes_kubectl_get", "kubernetes_kubectl_describe"], ["kubernetes_kubectl_describe", "kubernetes_kubectl_logs", "kubernetes_kubectl_get"]],
     gold=[("kubernetes_kubectl_get", {"resourceType": "pods", "namespace": "reports"}),
           ("kubernetes_kubectl_describe", {"resourceType": "pod", "name": "report-builder-6b7c9d-q8w2e", "namespace": "reports"})],
     writes=[],
     world=[
        {"tool": "kubernetes_kubectl_get", "match": {"_any": "reports"}, "response": "NAME                         READY   STATUS      RESTARTS      AGE\nreport-builder-6b7c9d-q8w2e   0/1     OOMKilled   7 (40s ago)   1h"},
        {"tool": "kubernetes_kubectl_describe", "match": {"_any": "report-builder"}, "response": "Name: report-builder-6b7c9d-q8w2e\nNamespace: reports\nContainers:\n  builder:\n    Limits: memory: 512Mi\n    Last State: Terminated\n      Reason: OOMKilled\n      Exit Code: 137\nEvents:\n  Warning OOMKilling 41s kernel-monitor Memory cgroup out of memory: Killed process 4121 (python) total-vm:1301220kB, anon-rss:811008kB (peak 792Mi)"},
        {"tool": "kubernetes_kubectl_logs", "match": {"_any": "report-builder"}, "response": "2026-11-05T05:40:02Z INFO building monthly report (rows=2.1M)\n2026-11-05T05:40:31Z INFO memory high-water mark 792Mi\n(container killed: OOMKilled)"},
     ])

# ----------------------------------------------------------------------------------------------- Slack

task(id="slack_decision", category="single",
     prompt="The team settled on a new webhook retry policy in the #payments-eng Slack channel. Implement it in "
            "webhooks/retry.py and confirm in that Slack thread once done." + ESCAPE,
     files={"README.md": PROJECT("webhooks"), "webhooks/__init__.py": "",
            "webhooks/retry.py": "MAX_RETRIES = 3\nBASE_DELAY_S = 1\nMAX_DELAY_S = 30\n\ndef delay(attempt: int) -> int:\n    return min(MAX_DELAY_S, BASE_DELAY_S * 2 ** attempt)\n"},
     solution={"webhooks/retry.py": "MAX_RETRIES = 4\nBASE_DELAY_S = 3\nMAX_DELAY_S = 60\n\ndef delay(attempt: int) -> int:\n    return min(MAX_DELAY_S, BASE_DELAY_S * 2 ** attempt)\n"},
     hidden=d("""
        from webhooks import retry
        def test_policy():
            assert retry.MAX_RETRIES == 4 and retry.BASE_DELAY_S == 3 and retry.MAX_DELAY_S == 60
            assert retry.delay(0) == 3 and retry.delay(5) == 60
        """),
     needs=[["slack_list_channels"], ["slack_get_channel_history"], ["slack_get_thread_replies"], ["slack_reply_to_thread"]],
     gold=[("slack_list_channels", {}), ("slack_get_channel_history", {"channel_id": "C07PAYENG01", "limit": 50}),
           ("slack_get_thread_replies", {"channel_id": "C07PAYENG01", "thread_ts": "1793741112.004100"}),
           ("slack_reply_to_thread", {"channel_id": "C07PAYENG01", "thread_ts": "1793741112.004100", "text": "Implemented: 4 retries, 3s base, 60s cap"})],
     writes=[{"tool": "slack_reply_to_thread", "match": {"thread_ts": "1793741112.004100"}}],
     world=[
        {"tool": "slack_get_channel_history", "match": {"channel_id": "C07PAYENG01"}, "response": {"ok": True, "messages": [
            {"ts": "1793755001.000200", "user": "U02LENA", "text": "deploy of payments 2026.11.4 done"},
            {"ts": "1793741112.004100", "user": "U01OMAR", "text": "Proposal for webhook retries: 5 retries, 2s base delay, exponential, cap 30s. Thoughts?", "reply_count": 4, "thread_ts": "1793741112.004100"}]}},
        {"tool": "slack_get_thread_replies", "match": {"thread_ts": "1793741112.004100"}, "response": {"ok": True, "messages": [
            {"ts": "1793741112.004100", "user": "U01OMAR", "text": "Proposal for webhook retries: 5 retries, 2s base delay, exponential, cap 30s. Thoughts?"},
            {"ts": "1793741300.004200", "user": "U03JDIAZ", "text": "5 is too many, merchants complain about duplicates"},
            {"ts": "1793741422.004300", "user": "U02LENA", "text": "and 2s base is too aggressive for their rate limits"},
            {"ts": "1793742001.004400", "user": "U01OMAR", "text": "FINAL: 4 retries, base delay 3s, exponential, cap 60s. Please implement."},
            {"ts": "1793742100.004500", "user": "U03JDIAZ", "text": "+1"}]}},
     ])

# ----------------------------------------------------------------------------------------------- Atlassian

task(id="jira_impl", category="single",
     prompt="Implement Jira ticket PAY-412 and then move the ticket to In Review." + ESCAPE,
     files={"README.md": PROJECT("payments", "Jira project: PAY."), "payments/__init__.py": "",
            "payments/masking.py": "def mask_iban(iban: str) -> str:\n    return iban\n"},
     solution={"payments/masking.py": d("""
                def mask_iban(iban: str) -> str:
                    s = iban.replace(" ", "")
                    return s[:2] + "*" * (len(s) - 6) + s[-4:]
                """)},
     hidden=d("""
        from payments.masking import mask_iban
        def test_mask():
            assert mask_iban("DE89 3704 0044 0532 0130 00") == "DE" + "*" * 16 + "3000"
            assert mask_iban("PL61109010140000071219812874") == "PL" + "*" * 22 + "2874"
        """),
     needs=[["atlassian_jira_get_issue", "atlassian_jira_search"], ["atlassian_jira_get_transitions"], ["atlassian_jira_transition_issue"]],
     gold=[("atlassian_jira_get_issue", {"issue_key": "PAY-412"}), ("atlassian_jira_get_transitions", {"issue_key": "PAY-412"}),
           ("atlassian_jira_transition_issue", {"issue_key": "PAY-412", "transition_id": "31"})],
     writes=[{"tool": "atlassian_jira_transition_issue", "match": {"issue_key": "PAY-412", "transition_id": "31"}}],
     world=[
        {"tool": "atlassian_jira_get_issue", "match": {"issue_key": "PAY-412"}, "response": {"key": "PAY-412", "fields": {"summary": "Mask IBANs in logs", "status": {"name": "In Progress"},
            "description": "Acceptance criteria:\n* remove spaces\n* keep the 2-letter country code and the last 4 characters\n* replace every other character with `*`\n* function: payments.masking.mask_iban"}}},
        {"tool": "atlassian_jira_search", "match": {"_any": "412"}, "response": {"issues": [{"key": "PAY-412", "summary": "Mask IBANs in logs", "description": "keep the 2-letter country code and the last 4 characters, remove spaces, replace every other character with `*`"}]}},
     ])

task(id="confluence_limits", category="single",
     prompt="Update RATE_LIMITS in api/limits.py to match the 'Public API rate limits' page in Confluence." + ESCAPE,
     files={"README.md": PROJECT("public-api"), "api/__init__.py": "", "api/limits.py": 'RATE_LIMITS = {"free": 60, "pro": 600}\n'},
     solution={"api/limits.py": 'RATE_LIMITS = {"free": 30, "pro": 900, "enterprise": 6000}\n'},
     hidden=d("""
        from api.limits import RATE_LIMITS
        def test_limits():
            assert RATE_LIMITS == {"free": 30, "pro": 900, "enterprise": 6000}
        """),
     needs=[["atlassian_confluence_search"], ["atlassian_confluence_get_page"]],
     gold=[("atlassian_confluence_search", {"query": "Public API rate limits"}), ("atlassian_confluence_get_page", {"page_id": "884213"})],
     writes=[],
     world=[
        {"tool": "atlassian_confluence_search", "match": {"query": ["rate", "limit"]}, "response": [{"id": "884213", "title": "Public API rate limits", "space": "API", "url": "/wiki/spaces/API/pages/884213"},
                                                                                                  {"id": "771002", "title": "Rate limiting design (2024, superseded)", "space": "ARCH"}]},
        {"tool": "atlassian_confluence_get_page", "match": {"_any": ["884213", "Public API rate limits"]}, "response": {"id": "884213", "title": "Public API rate limits", "version": 7,
            "content": "| Plan | Requests per minute |\n|---|---|\n| free | 30 |\n| pro | 900 |\n| enterprise | 6000 |\n\nEffective 2026-11-01. Keys in code are the lowercase plan names."}},
        {"tool": "atlassian_confluence_get_page", "match": {"_any": "771002"}, "response": {"id": "771002", "title": "Rate limiting design (2024, superseded)", "content": "free 60/min, pro 600/min. SUPERSEDED - see 'Public API rate limits'."}},
     ])

# ----------------------------------------------------------------------------------------------- Linear, Notion, databases

task(id="linear_bug", category="single",
     prompt="Fix Linear issue ENG-77 and set its status to In Review in Linear." + ESCAPE,
     files={"README.md": PROJECT("finance-export", "Issues are tracked in Linear (team ENG)."), "export/__init__.py": "",
            "export/csv_out.py": d("""
                def to_csv(rows: list[list[str]]) -> str:
                    return "\\n".join(",".join(r) for r in rows)
                """)},
     solution={"export/csv_out.py": d("""
                def to_csv(rows: list[list[str]]) -> str:
                    return "\\ufeff" + "\\n".join(";".join(r) for r in rows)
                """)},
     hidden=d("""
        from export.csv_out import to_csv
        def test_finance_format():
            out = to_csv([["a", "b"], ["1", "2"]])
            assert out.startswith("\\ufeff")
            assert out[1:] == "a;b\\n1;2"
        """),
     needs=[["linear_getIssueById", "linear_searchIssues", "linear_getIssues"], ["linear_getWorkflowStates", "linear_getTeams"], ["linear_updateIssue"]],
     gold=[("linear_searchIssues", {"query": "ENG-77"}), ("linear_getWorkflowStates", {"teamId": "9f1c2e7a-eng"}),
           ("linear_updateIssue", {"id": "ENG-77", "stateId": "st-review-04"})],
     writes=[{"tool": "linear_updateIssue", "match": {"_any": ["ENG-77", "a41e0c9d"], "stateId": "st-review-04"}}],
     world=[
        {"tool": "linear_getIssueById", "match": {"id": ["ENG-77", "a41e0c9d"]}, "response": {"id": "a41e0c9d-77", "identifier": "ENG-77", "title": "Finance CSV export opens garbled in Excel", "team": {"id": "9f1c2e7a-eng", "key": "ENG"}, "state": {"name": "In Progress"},
            "description": "Finance (EU locale Excel) needs:\n1. `;` as the separator instead of `,`\n2. a UTF-8 byte-order mark (U+FEFF) at the very start of the file\nNothing else changes."}},
        {"tool": "linear_searchIssues", "match": {"_any": ["77", "csv", "export"]}, "response": [{"id": "a41e0c9d-77", "identifier": "ENG-77", "title": "Finance CSV export opens garbled in Excel", "team": {"id": "9f1c2e7a-eng"},
            "description": "Finance (EU locale Excel) needs: 1. `;` as the separator instead of `,` 2. a UTF-8 byte-order mark (U+FEFF) at the very start of the file. Nothing else changes."}]},
     ])

task(id="notion_runbook", category="single",
     prompt="Follow the 'Webhook signing key rotation' runbook in Notion and make the code change it asks for in signing/keys.py." + ESCAPE,
     files={"README.md": PROJECT("webhooks"), "signing/__init__.py": "",
            "signing/keys.py": 'ACTIVE_KEY_IDS = ["whsec_2026_05"]\n\ndef accepts(key_id: str) -> bool:\n    return key_id in ACTIVE_KEY_IDS\n'},
     solution={"signing/keys.py": 'ACTIVE_KEY_IDS = ["whsec_2026_11", "whsec_2026_05"]\n\ndef accepts(key_id: str) -> bool:\n    return key_id in ACTIVE_KEY_IDS\n'},
     hidden=d("""
        from signing.keys import accepts, ACTIVE_KEY_IDS
        def test_grace_period():
            assert accepts("whsec_2026_11") and accepts("whsec_2026_05")
            assert not accepts("whsec_2025_11")
            assert ACTIVE_KEY_IDS[0] == "whsec_2026_11"
        """),
     needs=[["notion_API-post-search"], ["notion_API-retrieve-page-markdown", "notion_API-get-block-children", "notion_API-retrieve-a-page"]],
     gold=[("notion_API-post-search", {"query": "Webhook signing key rotation"}), ("notion_API-retrieve-page-markdown", {"page_id": "1a2b3c4d-runbook"})],
     writes=[],
     world=[
        {"tool": "notion_API-post-search", "match": {"query": ["signing", "rotation", "webhook"]}, "response": {"object": "list", "results": [{"object": "page", "id": "1a2b3c4d-runbook", "properties": {"title": {"title": [{"plain_text": "Webhook signing key rotation"}]}}, "url": "https://notion.so/acme/Webhook-signing-key-rotation-1a2b3c4d"}]}},
        {"tool": "notion_API-retrieve-page-markdown", "match": {"page_id": "1a2b3c4d"}, "response": "# Webhook signing key rotation\n\n**Current rotation (Nov 2026)**\n\n1. New key id: `whsec_2026_11`. It becomes the primary key: put it FIRST in `ACTIVE_KEY_IDS`.\n2. Keep the previous key `whsec_2026_05` accepted during the 7-day grace period (second in the list).\n3. Remove any older keys.\n"},
        {"tool": "notion_API-get-block-children", "match": {"_any": "1a2b3c4d"}, "response": {"results": [{"type": "paragraph", "paragraph": {"rich_text": [{"plain_text": "New key id whsec_2026_11 becomes primary (first in ACTIVE_KEY_IDS); keep previous whsec_2026_05 accepted (second) for the 7-day grace period; remove older keys."}]}}]}},
        {"tool": "notion_API-retrieve-a-page", "match": {"_any": "1a2b3c4d"}, "response": {"id": "1a2b3c4d-runbook", "properties": {"title": {"title": [{"plain_text": "Webhook signing key rotation"}]}}, "note": "page content is in its blocks"}},
     ])

task(id="mongo_shape", category="single",
     prompt="The nightly export crashes on new documents in the `orders` collection of the prod MongoDB (database `shop`, "
            "connection string in README). Inspect the current document shape and update export/parse.py so both old and new "
            "documents work." + ESCAPE,
     files={"README.md": PROJECT("shop-export", "Prod MongoDB (read-only user): `mongodb+srv://export-ro@shop-prod.acme.internal/shop`."), "export/__init__.py": "",
            "export/parse.py": d("""
                def parse_order(doc: dict) -> dict:
                    return {"email": doc["customer_email"], "total": float(doc["total"]), "currency": doc.get("currency", "EUR")}
                """)},
     solution={"export/parse.py": d("""
                def parse_order(doc: dict) -> dict:
                    if isinstance(doc.get("customer"), dict):
                        email = doc["customer"]["email"]
                    else:
                        email = doc["customer_email"]
                    total = doc["total"]
                    if isinstance(total, dict):
                        return {"email": email, "total": float(total["amount"]), "currency": total["currency"]}
                    return {"email": email, "total": float(total), "currency": doc.get("currency", "EUR")}
                """)},
     hidden=d("""
        from export.parse import parse_order
        def test_shapes():
            old = {"customer_email": "a@x.io", "total": "12.5", "currency": "EUR"}
            new = {"customer": {"id": "c1", "email": "b@x.io"}, "total": {"amount": "7.25", "currency": "SEK"}}
            assert parse_order(old) == {"email": "a@x.io", "total": 12.5, "currency": "EUR"}
            assert parse_order(new) == {"email": "b@x.io", "total": 7.25, "currency": "SEK"}
        """),
     needs=[["mongodb_connect"], ["mongodb_collection-schema", "mongodb_find", "mongodb_aggregate"]],
     gold=[("mongodb_connect", {"connectionString": "mongodb+srv://export-ro@shop-prod.acme.internal/shop"}),
           ("mongodb_find", {"connectionId": "prod-shop-a1b2", "database": "shop", "collection": "orders", "limit": 5, "sort": {"_id": -1}})],
     writes=[],
     world=[
        {"tool": "mongodb_collection-schema", "match": {"collection": "orders"}, "response": "Schema of shop.orders (sample 1000):\n- _id: ObjectId\n- customer_email: String (31%)\n- customer: Document (69%)\n  - customer.id: String\n  - customer.email: String\n- total: String (31%) | Document (69%)\n  - total.amount: String\n  - total.currency: String\n- currency: String (31%)\n- created_at: Date"},
        {"tool": "mongodb_find", "match": {"collection": "orders"}, "response": "Found 2 documents:\n{\"_id\": {\"$oid\": \"6729f0\"}, \"customer\": {\"id\": \"c_8812\", \"email\": \"m.rossi@example.com\"}, \"total\": {\"amount\": \"49.90\", \"currency\": \"EUR\"}, \"created_at\": {\"$date\": \"2026-11-04T22:10:00Z\"}}\n{\"_id\": {\"$oid\": \"6729a1\"}, \"customer_email\": \"j.li@example.com\", \"total\": \"12.00\", \"currency\": \"SEK\", \"created_at\": {\"$date\": \"2026-10-28T09:00:00Z\"}}"},
        {"tool": "mongodb_aggregate", "match": {"collection": "orders"}, "response": "[{\"customer\": {\"id\": \"c_8812\", \"email\": \"m.rossi@example.com\"}, \"total\": {\"amount\": \"49.90\", \"currency\": \"EUR\"}}, {\"customer_email\": \"j.li@example.com\", \"total\": \"12.00\", \"currency\": \"SEK\"}]"},
     ])

task(id="pg_migration", category="single",
     prompt="Write migrations/0042_payments_merchant_idx.sql creating an index named payments_merchant_created_idx that serves "
            "lookups of a merchant's payments ordered by creation time. Check the real production schema first (read-only "
            "Postgres): column names there differ from our ORM." + ESCAPE,
     files={"README.md": PROJECT("payments-db"), "migrations/0041_init.sql": "-- see ORM models\n", "models.py": "class Payment:\n    merchant_id: str\n    created_at: str\n"},
     solution={"migrations/0042_payments_merchant_idx.sql": "CREATE INDEX payments_merchant_created_idx ON payments (merchant_ref, created_ts);\n"},
     hidden=d("""
        import re
        from pathlib import Path
        def test_index():
            p = Path("migrations/0042_payments_merchant_idx.sql")
            sql = re.sub(r"\\s+", " ", p.read_text().lower())
            assert "payments_merchant_created_idx" in sql
            m = re.search(r"on (?:public\\.)?payments\\s*(?:using btree\\s*)?\\(([^)]*)\\)", sql)
            cols = [c.strip().split()[0].strip('"') for c in m.group(1).split(",")]
            assert cols == ["merchant_ref", "created_ts"]
        """),
     needs=[["postgres_query"]],
     gold=[("postgres_query", {"sql": "SELECT column_name, data_type FROM information_schema.columns WHERE table_name = 'payments'"})],
     writes=[],
     world=[
        {"tool": "postgres_query", "match": {"sql": ["information_schema", "pg_attribute", "\\d"]}, "response": [
            {"column_name": "id", "data_type": "uuid"}, {"column_name": "merchant_ref", "data_type": "text"}, {"column_name": "amount_minor", "data_type": "bigint"},
            {"column_name": "currency", "data_type": "character(3)"}, {"column_name": "created_ts", "data_type": "timestamp with time zone"}, {"column_name": "status", "data_type": "text"}]},
        {"tool": "postgres_query", "match": {"sql": "payments"}, "response": [{"id": "5f0c...", "merchant_ref": "m_1029", "amount_minor": 4990, "currency": "EUR", "created_ts": "2026-11-04T21:00:00Z", "status": "captured"}]},
     ])

# ----------------------------------------------------------------------------------------------- multi-service

task(id="multi_alert", category="multi",
     prompt="An alert is firing for the checkout service. Use the alert and the service's logs to find out what is wrong, fix "
            "checkout/gateway.py, and post a short summary of the fix in the #incidents Slack channel." + ESCAPE,
     files={"README.md": PROJECT("checkout", "Logs: Loki `app=\"checkout\"`. Alerts: Grafana OnCall."), "checkout/__init__.py": "",
            "checkout/gateway.py": "def charge_payload(amount_eur: float) -> dict:\n    return {\"amount\": amount_eur, \"currency\": \"EUR\"}\n"},
     solution={"checkout/gateway.py": "def charge_payload(amount_eur: float) -> dict:\n    return {\"amount\": int(round(amount_eur * 100)), \"currency\": \"EUR\"}\n"},
     hidden=d("""
        from checkout.gateway import charge_payload
        def test_minor_units():
            assert charge_payload(12.34) == {"amount": 1234, "currency": "EUR"}
            assert charge_payload(0.1 + 0.2)["amount"] == 30
        """),
     needs=[["grafana_list_alert_groups", "grafana_get_alert_group"], ["grafana_list_datasources"], ["grafana_query_loki_logs", "grafana_find_error_pattern_logs"],
            ["slack_list_channels"], ["slack_post_message"]],
     gold=[("grafana_list_alert_groups", {"labels": ["service:checkout"]}), ("grafana_list_datasources", {"name": "loki"}),
           ("grafana_query_loki_logs", {"datasourceUid": LOKI, "logql": '{app="checkout"} |= "gateway"', "limit": 50}),
           ("slack_list_channels", {}), ("slack_post_message", {"channel_id": "C07INCID003", "text": "checkout: amounts now sent in minor units"})],
     writes=[{"tool": "slack_post_message", "match": {"channel_id": "C07INCID003"}}],
     world=[
        {"tool": "grafana_list_alert_groups", "response": [{"id": "AG-7731", "title": "[FIRING] PaymentGatewayErrors checkout", "state": "firing", "labels": {"service": "checkout", "severity": "critical"},
                                                            "summary": "Gateway 4xx rate 38% for service=checkout since 07:41 UTC. See logs for gateway error bodies."}]},
        {"tool": "grafana_get_alert_group", "match": {"_any": "7731"}, "response": {"id": "AG-7731", "title": "[FIRING] PaymentGatewayErrors checkout", "labels": {"service": "checkout"}, "summary": "Gateway 4xx rate 38% for service=checkout since 07:41 UTC. See Loki logs app=checkout for the gateway error bodies."}},
        {"tool": "grafana_query_loki_logs", "match": {"datasourceUid": LOKI, "logql": "checkout"}, "response": [
            {"timestamp": "2026-11-05T07:41:03Z", "line": "WARN gateway rejected charge status=400 body={\"error\":\"invalid_amount\",\"message\":\"amount must be an integer in minor units (cents), got 12.34\"}"},
            {"timestamp": "2026-11-05T07:41:05Z", "line": "INFO gateway api version switched to 2026-11 (strict amounts)"}]},
        {"tool": "grafana_query_loki_logs", "response": {"error": "datasource not found or not a Loki datasource"}},
        {"tool": "grafana_find_error_pattern_logs", "match": {"_any": "checkout"}, "response": {"patterns": [{"pattern": "gateway rejected charge status=400 invalid_amount: amount must be an integer in minor units (cents), got <_>", "count": 9124}]}},
     ])

task(id="multi_jira_conf", category="multi",
     prompt="Implement Jira ticket PAY-530 using the design document linked from the ticket, then comment on the ticket with a "
            "one-line summary of what you implemented." + ESCAPE,
     files={"README.md": PROJECT("refunds", "Jira project: PAY. Design docs live in Confluence."), "refunds/__init__.py": "",
            "refunds/policy.py": "def refund_window_days(category: str) -> int:\n    return 30\n"},
     solution={"refunds/policy.py": "def refund_window_days(category: str) -> int:\n    return 14 if category == \"digital\" else 45\n"},
     hidden=d("""
        from refunds.policy import refund_window_days
        def test_policy_v3():
            assert refund_window_days("digital") == 14
            assert refund_window_days("apparel") == 45
            assert refund_window_days("electronics") == 45
        """),
     needs=[["atlassian_jira_get_issue"], ["atlassian_confluence_get_page", "atlassian_confluence_search"], ["atlassian_jira_add_comment"]],
     gold=[("atlassian_jira_get_issue", {"issue_key": "PAY-530"}), ("atlassian_confluence_get_page", {"page_id": "902117"}),
           ("atlassian_jira_add_comment", {"issue_key": "PAY-530", "body": "Implemented refund policy v3"})],
     writes=[{"tool": "atlassian_jira_add_comment", "match": {"issue_key": "PAY-530"}}],
     world=[
        {"tool": "atlassian_jira_get_issue", "match": {"issue_key": "PAY-530"}, "response": {"key": "PAY-530", "fields": {"summary": "Refund policy v3", "description": "Implement refund policy v3 in refunds.policy.refund_window_days. All the rules are in the design doc: https://acme.atlassian.net/wiki/spaces/PAY/pages/902117/Refund+policy+v3"}}},
        {"tool": "atlassian_confluence_get_page", "match": {"_any": ["902117", "Refund policy v3"]}, "response": {"id": "902117", "title": "Refund policy v3", "content": "## Refund windows\n- Default: **45 days** from delivery (was 30).\n- Digital goods (`digital`): **14 days**.\n- No other exceptions."}},
        {"tool": "atlassian_confluence_search", "match": {"query": ["refund"]}, "response": [{"id": "902117", "title": "Refund policy v3", "space": "PAY"}, {"id": "640001", "title": "Refund policy v2 (old)", "space": "PAY"}]},
        {"tool": "atlassian_confluence_get_page", "match": {"_any": "640001"}, "response": {"id": "640001", "title": "Refund policy v2 (old)", "content": "30 days for everything."}},
     ])

task(id="multi_sentry_gh", category="multi",
     prompt="A Sentry issue in ledger-worker (organisation `acme`) was introduced by a recent pull request in acme/ledger-worker. "
            "Find the issue and the PR that caused it, fix the regression in worker/rates.py, and comment on that PR with a link "
            "to the Sentry issue." + ESCAPE,
     files={"README.md": PROJECT("ledger-worker", "Sentry project: `ledger-worker`."), "worker/__init__.py": "",
            "worker/rates.py": "def convert(amount_minor: int, rate: float) -> int:\n    return int(amount_minor * rate)\n"},
     solution={"worker/rates.py": "def convert(amount_minor: int, rate: float) -> int:\n    return round(amount_minor * rate)\n"},
     hidden=d("""
        from worker.rates import convert
        def test_round_not_truncate():
            assert convert(1000, 1.0999) == 1100
            assert convert(999, 0.5) == 500 or convert(999, 0.5) == 499
            assert convert(3, 0.333334) == 1
        """),
     needs=[["sentry_search_issues", "sentry_search_events"], ["sentry_get_sentry_resource"], ["github_pull_request_read", "github_list_pull_requests", "github_search_pull_requests"], ["github_add_issue_comment"]],
     gold=[("sentry_search_issues", {"organizationSlug": "acme", "projectSlugOrId": "ledger-worker", "query": "is:unresolved"}),
           ("sentry_get_sentry_resource", {"resourceType": "issue", "resourceId": "LEDGER-WORKER-9A", "organizationSlug": "acme"}),
           ("github_pull_request_read", {"method": "get", "owner": "acme", "repo": "ledger-worker", "pullNumber": 412}),
           ("github_add_issue_comment", {"owner": "acme", "repo": "ledger-worker", "issue_number": 412, "body": "Regression: https://acme.sentry.io/issues/LEDGER-WORKER-9A"})],
     writes=[{"tool": "github_add_issue_comment", "match": {"issue_number": "412", "body": ["sentry", "LEDGER-WORKER-9A"]}}],
     world=[
        {"tool": "sentry_search_issues", "match": {"_any": "ledger"}, "response": "# Issues in **acme/ledger-worker**\n\n1. **LEDGER-WORKER-9A** - ReconciliationMismatch: converted totals off by 1 minor unit\n   events: 2,201 | first seen: release ledger-worker@2026.11.3 | status: unresolved"},
        {"tool": "sentry_get_sentry_resource", "match": {"_any": "9A"}, "response": "# Issue LEDGER-WORKER-9A\n**ReconciliationMismatch**: converted totals off by 1 minor unit (expected 1100, got 1099)\n\nFirst seen in release ledger-worker@2026.11.3.\n**Suspect commit**: 8c2f1d0 \"Speed up FX conversion\" by tkim - from pull request acme/ledger-worker#412.\nIn worker/rates.py the conversion now truncates (`int(...)`) instead of rounding to the nearest minor unit."},
        {"tool": "sentry_search_events", "match": {"_any": ["ledger", "9A"]}, "response": "| issue | title | release |\n| LEDGER-WORKER-9A | ReconciliationMismatch: off by 1 minor unit; suspect commit 8c2f1d0 from PR acme/ledger-worker#412 (truncates instead of rounding) | 2026.11.3 |"},
        {"tool": "github_get_commit", "match": {"_any": "8c2f1d0"}, "response": {"sha": "8c2f1d0", "commit": {"message": "Speed up FX conversion (#412)"}, "files": [{"filename": "worker/rates.py", "patch": "-    return round(amount_minor * rate)\n+    return int(amount_minor * rate)"}]}},
        {"tool": "github_pull_request_read", "match": {"pullNumber": "412"}, "response": {"number": 412, "title": "Speed up FX conversion", "state": "closed", "merged": True, "user": {"login": "tkim"}, "body": "Replaced Decimal rounding with int() for speed."}},
        {"tool": "github_list_pull_requests", "match": {"repo": "ledger-worker"}, "response": [{"number": 415, "title": "Bump deps", "state": "open"}, {"number": 412, "title": "Speed up FX conversion", "state": "closed", "merged_at": "2026-11-03T10:00:00Z"}]},
        {"tool": "github_search_pull_requests", "match": {"_any": ["ledger-worker", "fx", "8c2f1d0"]}, "response": {"items": [{"number": 412, "title": "Speed up FX conversion", "repository_url": "acme/ledger-worker"}]}},
     ])

# ----------------------------------------------------------------------------------------------- absent

task(id="absent_sentry", category="absent",
     prompt="Fix the Sentry error in the billing-cron project (organisation `acme`) that was mentioned in standup, in "
            "billing/cron.py." + ESCAPE,
     files={"README.md": PROJECT("billing", "Sentry project: `billing-cron`."), "billing/__init__.py": "", "billing/cron.py": "BATCH_SIZE = 500\n"},
     solution={"billing/cron.py": "BATCH_SIZE = 500\n# TODO: info not found\n"},
     hidden="from billing.cron import BATCH_SIZE\ndef test_unchanged():\n    assert BATCH_SIZE == 500\n",
     needs=[], gold=[("sentry_search_issues", {"organizationSlug": "acme", "projectSlugOrId": "billing-cron", "query": "is:unresolved"})], writes=[],
     probe={"file": "billing/cron.py", "name": "BATCH_SIZE", "expected": None, "original": 500, "todo_file": "billing/cron.py"},
     world=[{"tool": "sentry_search_issues", "match": {"_any": "billing"}, "response": "# Issues in **acme/billing-cron**\n\nNo unresolved issues found."}])

task(id="absent_slack", category="absent",
     prompt="Set CACHE_TTL_S in cache/config.py to the value the platform team agreed on in the #platform Slack channel." + ESCAPE,
     files={"README.md": PROJECT("cache"), "cache/__init__.py": "", "cache/config.py": "CACHE_TTL_S = 300\n"},
     solution={"cache/config.py": "CACHE_TTL_S = 300\n# TODO: info not found\n"},
     hidden="from cache.config import CACHE_TTL_S\ndef test_unchanged():\n    assert CACHE_TTL_S == 300\n",
     needs=[], gold=[("slack_get_channel_history", {"channel_id": "C07PLATF004"})], writes=[],
     probe={"file": "cache/config.py", "name": "CACHE_TTL_S", "expected": None, "original": 300, "todo_file": "cache/config.py"},
     world=[{"tool": "slack_get_channel_history", "match": {"channel_id": "C07PLATF004"}, "response": {"ok": True, "messages": [
         {"ts": "1793740000.000100", "user": "U09TBAUER", "text": "should we revisit the cache TTL? 300s feels long"},
         {"ts": "1793740100.000200", "user": "U08NINA", "text": "let's discuss next week, no decision yet"}]}}])

task(id="absent_jira", category="absent",
     prompt="Implement Jira ticket PAY-999 in payments/limits.py." + ESCAPE,
     files={"README.md": PROJECT("payments", "Jira project: PAY."), "payments/__init__.py": "", "payments/limits.py": "DAILY_LIMIT_EUR = 10000\n"},
     solution={"payments/limits.py": "DAILY_LIMIT_EUR = 10000\n# TODO: info not found\n"},
     hidden="from payments.limits import DAILY_LIMIT_EUR\ndef test_unchanged():\n    assert DAILY_LIMIT_EUR == 10000\n",
     needs=[], gold=[("atlassian_jira_get_issue", {"issue_key": "PAY-999"})], writes=[],
     probe={"file": "payments/limits.py", "name": "DAILY_LIMIT_EUR", "expected": None, "original": 10000, "todo_file": "payments/limits.py"},
     world=[{"tool": "atlassian_jira_get_issue", "match": {"issue_key": "PAY-999"}, "response": {"error": "Issue does not exist or you do not have permission to see it.", "status": 404}}])

task(id="absent_release", category="absent",
     prompt="Pin SDK_VERSION in sdk/version.py to the latest release of acme/ghost-sdk (without the leading `v`)." + ESCAPE,
     files={"README.md": PROJECT("sdk-consumer"), "sdk/__init__.py": "", "sdk/version.py": 'SDK_VERSION = "1.2.0"\n'},
     solution={"sdk/version.py": 'SDK_VERSION = "1.2.0"\n# TODO: info not found\n'},
     hidden='from sdk.version import SDK_VERSION\ndef test_unchanged():\n    assert SDK_VERSION == "1.2.0"\n',
     needs=[], gold=[("github_get_latest_release", {"owner": "acme", "repo": "ghost-sdk"})], writes=[],
     probe={"file": "sdk/version.py", "name": "SDK_VERSION", "expected": None, "original": "1.2.0", "todo_file": "sdk/version.py"},
     world=[{"tool": "github_get_latest_release", "match": {"repo": "ghost-sdk"}, "response": {"error": "Not Found", "status": 404, "message": "repository acme/ghost-sdk has no releases"}}])

# ----------------------------------------------------------------------------------------------- control (copied from tasks_v2)

CONTROL = ["c_chunk", "c_dedupe", "c_duration", "c_leap"]


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    for t in TASKS:
        root = OUT / t["id"]
        for sub, files in (("repo", t["files"]), ("solution", t["solution"])):
            for rel, content in files.items():
                p = root / sub / rel
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(content)
        (root / "hidden").mkdir(parents=True)
        (root / "hidden" / "test_hidden.py").write_text(t["hidden"])
        (root / "world.json").write_text(json.dumps(t["world"], indent=1))
        meta = {k: t.get(k) for k in ("id", "category", "prompt", "needs", "gold", "writes", "probe")}
        (root / "task.json").write_text(json.dumps(meta, indent=1))
    for c in CONTROL:
        src = HERE / "tasks_v2" / c
        shutil.copytree(src, OUT / c)
        (OUT / c / "world.json").write_text("[]")
        m = json.loads((src / "task.json").read_text())
        m.update(gold=[], writes=[], probe=None)
        (OUT / c / "task.json").write_text(json.dumps(m, indent=1))
    print(len(TASKS) + len(CONTROL), "tasks in", OUT)


if __name__ == "__main__":
    main()
