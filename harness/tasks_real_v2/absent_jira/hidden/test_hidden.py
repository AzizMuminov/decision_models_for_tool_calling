from payments.limits import DAILY_LIMIT_EUR
def test_unchanged():
    assert DAILY_LIMIT_EUR == 10000
