from notifications.digest import DIGEST_HOUR_UTC


def test_authoritative_value():
    assert DIGEST_HOUR_UTC == 7
