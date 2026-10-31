from ratelimiter.limits import DEFAULT_PER_MINUTE, allow


def test_matches_production():
    assert DEFAULT_PER_MINUTE == 240


def test_allow():
    assert allow(239) and not allow(240)
