from notifications.digest import current


def test_current_is_set():
    assert current() is not None
