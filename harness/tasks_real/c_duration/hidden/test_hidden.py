from timeutil.duration import parse_duration


def test_all():
    assert parse_duration("2h") == 7200
    assert parse_duration("45m") == 2700
    assert parse_duration("1h30m") == 5400
    assert parse_duration("90s") == 90
    assert parse_duration("1h0m5s") == 3605
