"""`needs` is a list of groups; a task's needs are met when, for every group, at least one of its tools was
called successfully.

Writes the pilot coding tasks to harness/tasks/<id>/: repo/ (what the agent gets), hidden/ (scoring tests),
task.json (prompt, the catalog tools the task needs, and the world rules the catalog server answers from)."""

from __future__ import annotations

import json
import shutil
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "tasks"


def write(base: Path, files: dict[str, str]) -> None:
    for rel, text in files.items():
        p = base / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(textwrap.dedent(text).lstrip("\n"))


TASKS = []

# ---------------------------------------------------------------- 1. control: everything is local
TASKS.append({
    "id": "slug", "needs": [],
    "prompt": "The tests in tests/test_slug.py are failing. Fix textkit/slug.py so they pass.",
    "repo": {
        "textkit/__init__.py": "",
        "textkit/slug.py": '''
            import re


            def slugify(text: str) -> str:
                """Turn a title into a URL slug."""
                text = text.lower()
                text = re.sub(r"[^a-z0-9]+", "-", text)
                return text
            ''',
        "tests/test_slug.py": '''
            from textkit.slug import slugify


            def test_punctuation():
                assert slugify("Hello, World!") == "hello-world"


            def test_accents():
                assert slugify("  Crème Brûlée ") == "creme-brulee"


            def test_repeated_separators():
                assert slugify("a -- b") == "a-b"
            ''',
    },
    "hidden": '''
        from textkit.slug import slugify


        def test_visible_cases():
            assert slugify("Hello, World!") == "hello-world"
            assert slugify("  Crème Brûlée ") == "creme-brulee"
            assert slugify("a -- b") == "a-b"


        def test_more():
            assert slugify("Ça va?") == "ca-va"
            assert slugify("") == ""
        ''',
    "world": [],
})

# ---------------------------------------------------------------- 2. config service
TASKS.append({
    "id": "ratelimit", "needs": [["confman_get_config_value", "confman_export_config"]],
    "prompt": "Our rate limiter's default is out of date. Update DEFAULT_PER_MINUTE in ratelimiter/limits.py to the "
              "current production platform setting.",
    "repo": {
        "ratelimiter/__init__.py": "",
        "ratelimiter/limits.py": '''
            # Requests allowed per client per minute.
            # Must match the platform setting `api.rate_limit.per_minute` (production).
            DEFAULT_PER_MINUTE = 120


            def allow(count_this_minute: int) -> bool:
                return count_this_minute < DEFAULT_PER_MINUTE
            ''',
        "tests/test_limits.py": '''
            from ratelimiter.limits import DEFAULT_PER_MINUTE, allow


            def test_allow_below_limit():
                assert allow(DEFAULT_PER_MINUTE - 1)
                assert not allow(DEFAULT_PER_MINUTE)
            ''',
    },
    "hidden": '''
        from ratelimiter.limits import DEFAULT_PER_MINUTE, allow


        def test_matches_production():
            assert DEFAULT_PER_MINUTE == 240


        def test_allow():
            assert allow(239) and not allow(240)
        ''',
    "world": [
        {"tool": "confman_get_config_value", "match": {"key": "api.rate_limit.per_minute", "env": "prod"},
         "response": {"key": "api.rate_limit.per_minute", "environment": "prod", "value": 240, "updated_at": "2026-09-30T12:00:00Z"}},
        {"tool": "confman_get_config_value", "match": {"key": "api.rate_limit.per_minute", "env": "staging"},
         "response": {"key": "api.rate_limit.per_minute", "environment": "staging", "value": 300}},
        {"tool": "confman_get_config_value", "match": {"key": "api.rate_limit.per_minute", "env": "dev"},
         "response": {"key": "api.rate_limit.per_minute", "environment": "dev", "value": 1000}},
        {"tool": "confman_list_config_keys", "match": {"prefix": "api"},
         "response": ["api.rate_limit.per_minute", "api.rate_limit.burst", "api.timeout_ms"]},
        {"tool": "confman_export_config", "match": {"environment": "prod"},
         "response": {"api.rate_limit.per_minute": 240, "api.rate_limit.burst": 40, "api.timeout_ms": 3000}},
    ],
})

# ---------------------------------------------------------------- 3. API contract in the docs registry
TASKS.append({
    "id": "invoice", "needs": [["lorebook_get_api_contract"]],
    "prompt": "Implement parse_invoice in billing_client/invoice.py for the billing service's v3 invoice payload, "
              "following the billing team's published API contract.",
    "repo": {
        "billing_client/__init__.py": "",
        "billing_client/invoice.py": '''
            from dataclasses import dataclass
            from datetime import datetime
            from decimal import Decimal


            @dataclass
            class Invoice:
                id: str
                amount: Decimal  # in major units, e.g. Decimal("12.34")
                currency: str  # ISO 4217 code, upper case
                issued_at: datetime  # timezone-aware, UTC


            def parse_invoice(payload: dict) -> Invoice:
                """Parse a v3 invoice payload from the billing service (see the billing team's API contract)."""
                raise NotImplementedError
            ''',
        "tests/test_invoice.py": '''
            import pytest

            from billing_client.invoice import parse_invoice


            def test_rejects_empty_payload():
                with pytest.raises(Exception):
                    parse_invoice({})
            ''',
    },
    "hidden": '''
        from datetime import datetime, timezone
        from decimal import Decimal

        from billing_client.invoice import parse_invoice


        def test_contract_mapping():
            inv = parse_invoice({"inv_ref": "INV-9", "amt_minor": 1234, "ccy": "eur", "issued_ts": 1700000000})
            assert inv.id == "INV-9"
            assert inv.amount == Decimal("12.34")
            assert inv.currency == "EUR"
            assert inv.issued_at == datetime(2023, 11, 14, 22, 13, 20, tzinfo=timezone.utc)
        ''',
    "world": [
        {"tool": "lorebook_get_api_contract", "match": {"name": "invoice"},
         "response": {"name": "billing-invoice-v3", "owner": "billing", "fields": {
             "inv_ref": "string. The invoice identifier.",
             "amt_minor": "integer. Amount in minor currency units (cents).",
             "ccy": "string. ISO 4217 currency code, lower case.",
             "issued_ts": "integer. Issue time as Unix epoch seconds (UTC)."},
             "example": {"inv_ref": "INV-1001", "amt_minor": 50000, "ccy": "usd", "issued_ts": 1767225600}}},
        {"tool": "lorebook_list_api_contracts", "match": {},
         "response": ["billing-invoice-v2", "billing-invoice-v3", "ledger-entry-v1", "payout-v2"]},
        {"tool": "lorebook_search_docs", "match": {"query": "invoice"},
         "response": [{"page_id": "PAGE-2210", "title": "Billing API contracts", "snippet": "The current invoice payload is the contract `billing-invoice-v3` in the API-contract registry."}]},
    ],
})

# ---------------------------------------------------------------- 4. CI log
TASKS.append({
    "id": "ci_tz", "needs": [["buildwave_get_job_log", "buildwave_get_test_report"]],
    "prompt": "The nightly CI pipeline for the reports package is failing, but the tests pass on my machine. "
              "Find out why and fix the code.",
    "repo": {
        "reports/__init__.py": "",
        "reports/buckets.py": '''
            from datetime import datetime


            def day_bucket(ts: int) -> str:
                """Return the reporting day (YYYY-MM-DD) that a Unix timestamp belongs to."""
                return datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
            ''',
        "tests/test_buckets.py": '''
            from reports.buckets import day_bucket


            def test_bucket():
                assert day_bucket(1700000000) == "2023-11-14"
            ''',
    },
    "hidden": '''
        import os
        import time

        from reports.buckets import day_bucket


        def _tz(name):
            os.environ["TZ"] = name
            time.tzset()


        def test_auckland_runner():
            _tz("Pacific/Auckland")
            assert day_bucket(1700000000) == "2023-11-14"


        def test_los_angeles():
            _tz("America/Los_Angeles")
            assert day_bucket(1700020000) == "2023-11-15"
            _tz("UTC")
        ''',
    "world": [
        {"tool": "buildwave_list_pipeline_runs", "match": {"pipeline": "report"},
         "response": [{"run_id": "run-50117", "pipeline": "reports-nightly", "status": "failed", "started_at": "2026-10-27T02:00:00Z"},
                      {"run_id": "run-50094", "pipeline": "reports-nightly", "status": "failed", "started_at": "2026-10-26T02:00:00Z"}]},
        {"tool": "buildwave_list_pipelines", "match": {},
         "response": [{"name": "reports-nightly", "repository": "quillmark/reports"}, {"name": "checkout-ci", "repository": "quillmark/checkout"}]},
        {"tool": "buildwave_get_pipeline_run", "match": {"run_id": "run-50"},
         "response": {"run_id": "run-50117", "status": "failed", "jobs": [{"name": "lint", "status": "success"}, {"name": "unit-tests", "status": "failed"}]}},
        {"tool": "buildwave_get_job_log", "match": {"run_id": "run-50"},
         "response": "runner: nz-akl-04  env: TZ=Pacific/Auckland PYTHONHASHSEED=0\n$ python -m pytest -q\n"
                     "F\nFAILED tests/test_buckets.py::test_bucket - AssertionError: assert '2023-11-15' == '2023-11-14'\n"
                     "1 failed in 0.03s\nnote: reporting days are defined as UTC calendar days (see reports README in Lorebook)."},
        {"tool": "buildwave_get_test_report", "match": {"run_id": "run-50"},
         "response": {"failed": ["tests/test_buckets.py::test_bucket"], "message": "assert '2023-11-15' == '2023-11-14'", "runner_env": {"TZ": "Pacific/Auckland"}}},
    ],
})

# ---------------------------------------------------------------- 5. issue tracker + feature flag
TASKS.append({
    "id": "discount", "needs": [["tracklet_get_issue"], ["flagpole_get_flag"]],
    "prompt": "Implement ticket QM-412 in checkout/pricing.py.",
    "repo": {
        "checkout/__init__.py": "",
        "checkout/pricing.py": '''
            from decimal import Decimal


            def apply_discount(total: Decimal, pct: int) -> Decimal:
                """Return the total after a percentage discount. See QM-412 for the required behaviour."""
                return total - total * Decimal(pct) / 100
            ''',
        "tests/test_pricing.py": '''
            from decimal import Decimal

            from checkout.pricing import apply_discount


            def test_no_discount():
                assert apply_discount(Decimal("19.99"), 0) == Decimal("19.99")
            ''',
    },
    "hidden": '''
        from decimal import Decimal

        import pytest

        from checkout.pricing import apply_discount


        def test_cap_from_flag():
            assert apply_discount(Decimal("10.00"), 50) == Decimal("7.00")


        def test_bankers_rounding():
            assert apply_discount(Decimal("0.25"), 10) == Decimal("0.22")


        def test_negative_rejected():
            with pytest.raises(ValueError):
                apply_discount(Decimal("100"), -1)


        def test_zero():
            assert apply_discount(Decimal("19.99"), 0) == Decimal("19.99")
        ''',
    "world": [
        {"tool": "tracklet_get_issue", "match": {"issue_key": "QM-412"},
         "response": {"key": "QM-412", "title": "Discount calculation rules", "status": "open", "description":
                      "1. Round discounted totals to 2 decimal places using round-half-to-even (banker's rounding).\n"
                      "2. Cap the discount percentage at the maximum allowed by the feature flag `checkout.max_discount_pct` "
                      "(use the production value; marketing changes it occasionally, so hard-code the current value for now "
                      "and leave a comment).\n3. A negative percentage is a programming error: raise ValueError."}},
        {"tool": "tracklet_search_issues", "match": {"query": "412"},
         "response": [{"key": "QM-412", "title": "Discount calculation rules", "status": "open"}]},
        {"tool": "flagpole_get_flag", "match": {"flag": "max_discount", "environment": "prod"},
         "response": {"flag": "checkout.max_discount_pct", "environment": "prod", "value": 30}},
        {"tool": "flagpole_get_flag", "match": {"flag": "max_discount", "environment": "staging"},
         "response": {"flag": "checkout.max_discount_pct", "environment": "staging", "value": 50}},
        {"tool": "flagpole_get_flag", "match": {"flag": "max_discount"},
         "response": {"flag": "checkout.max_discount_pct", "values": {"prod": 30, "staging": 50, "dev": 90}}},
    ],
})


def main() -> None:
    if ROOT.exists():
        shutil.rmtree(ROOT)
    for t in TASKS:
        base = ROOT / t["id"]
        write(base / "repo", t["repo"])
        write(base / "hidden", {"test_hidden.py": t["hidden"]})
        (base / "task.json").write_text(json.dumps({k: t[k] for k in ("id", "prompt", "needs")}, indent=1))
        (base / "world.json").write_text(json.dumps(t["world"], indent=1))
    print("wrote", [t["id"] for t in TASKS])


if __name__ == "__main__":
    main()
