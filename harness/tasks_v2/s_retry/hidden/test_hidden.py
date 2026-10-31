from payments.settings import MAX_ATTEMPTS


def test_authoritative_value():
    assert MAX_ATTEMPTS == 7
