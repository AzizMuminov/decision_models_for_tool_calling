from retention.policy import RETENTION_DAYS


def test_authoritative_value():
    assert RETENTION_DAYS == 395
