from calendar_util.leap import is_leap


def test_years():
    assert is_leap(2024) and is_leap(2000) and is_leap(2400)
    assert not is_leap(1900) and not is_leap(2100) and not is_leap(2023)
