from accounts.validation import MAX_EMAIL_LEN


def test_authoritative_value():
    assert MAX_EMAIL_LEN == 320
