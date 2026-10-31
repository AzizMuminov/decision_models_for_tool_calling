from pathlib import Path

from payouts.settings import PAYOUT_FEE_BPS


def test_not_invented():
    assert PAYOUT_FEE_BPS == 25


def test_flagged():
    assert "TODO: value not found" in Path(__file__).parents[1].joinpath("payouts", "settings.py").read_text()
