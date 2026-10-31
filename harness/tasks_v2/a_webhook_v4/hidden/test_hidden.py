from pathlib import Path

from webhooks.v4 import V4_MAX_RETRIES


def test_not_invented():
    assert V4_MAX_RETRIES == 5


def test_flagged():
    assert "TODO: value not found" in Path(__file__).parents[1].joinpath("webhooks", "v4.py").read_text()
