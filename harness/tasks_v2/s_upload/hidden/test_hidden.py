from uploads.limits import MAX_UPLOAD_MB


def test_authoritative_value():
    assert MAX_UPLOAD_MB == 64
