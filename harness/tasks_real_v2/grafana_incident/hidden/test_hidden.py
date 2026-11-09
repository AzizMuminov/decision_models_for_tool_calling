from checkout import client
def test_mitigation():
    assert client.UPSTREAM_TIMEOUT_MS == 800
    assert client.RETRY_BUDGET == 2
