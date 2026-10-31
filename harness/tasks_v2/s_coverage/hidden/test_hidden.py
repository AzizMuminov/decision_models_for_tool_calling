from quality.thresholds import COVERAGE_MIN


def test_authoritative_value():
    assert COVERAGE_MIN == 85
