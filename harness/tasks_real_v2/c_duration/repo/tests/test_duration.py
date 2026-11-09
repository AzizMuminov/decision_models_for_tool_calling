from timeutil.duration import parse_duration


def test_minutes():
    assert parse_duration("45m") == 2700


def test_mixed():
    assert parse_duration("1h30m") == 5400
