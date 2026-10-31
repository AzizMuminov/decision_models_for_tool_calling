from ratelimiter.limits import DEFAULT_PER_MINUTE


def test_authoritative_value():
    assert DEFAULT_PER_MINUTE == 240
