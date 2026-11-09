from alerts.config import SLOW_REQUEST_MS
def test_threshold():
    assert SLOW_REQUEST_MS == 824
