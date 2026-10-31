from ratelimiter.limits import DEFAULT_PER_MINUTE, allow


def test_allow_below_limit():
    assert allow(DEFAULT_PER_MINUTE - 1)
    assert not allow(DEFAULT_PER_MINUTE)
