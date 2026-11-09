from api.limits import RATE_LIMITS
def test_limits():
    assert RATE_LIMITS == {"free": 30, "pro": 900, "enterprise": 6000}
