from export.limits import BATCH_SIZE


def test_authoritative_value():
    assert BATCH_SIZE == 500
