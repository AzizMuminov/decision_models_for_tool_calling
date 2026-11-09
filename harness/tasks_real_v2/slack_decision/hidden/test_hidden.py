from webhooks import retry
def test_policy():
    assert retry.MAX_RETRIES == 4 and retry.BASE_DELAY_S == 3 and retry.MAX_DELAY_S == 60
    assert retry.delay(0) == 3 and retry.delay(5) == 60
