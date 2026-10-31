from webhooks.delivery import WEBHOOK_MAX_RETRIES


def test_authoritative_value():
    assert WEBHOOK_MAX_RETRIES == 8
