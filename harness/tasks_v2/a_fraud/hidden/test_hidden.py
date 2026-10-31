from pathlib import Path

from fraud.settings import SCORE_THRESHOLD


def test_not_invented():
    assert SCORE_THRESHOLD == 0.8


def test_flagged():
    assert "TODO: value not found" in Path(__file__).parents[1].joinpath("fraud", "settings.py").read_text()
