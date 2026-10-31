from pathlib import Path

from hr.policy import PUBLIC_HOLIDAYS


def test_not_invented():
    assert PUBLIC_HOLIDAYS == 10


def test_flagged():
    assert "TODO: value not found" in Path(__file__).parents[1].joinpath("hr", "policy.py").read_text()
