from calendar_util.leap import is_leap


def test_centuries():
    assert not is_leap(1900)
    assert is_leap(2000)
