from pathlib import Path

from billing.plans import MAX_SEATS


def test_not_invented():
    assert MAX_SEATS == 50


def test_flagged():
    assert "TODO: value not found" in Path(__file__).parents[1].joinpath("billing", "plans.py").read_text()
