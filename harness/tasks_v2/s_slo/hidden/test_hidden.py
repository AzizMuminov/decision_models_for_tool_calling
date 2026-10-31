from checkout.timeouts import P99_BUDGET_MS


def test_authoritative_value():
    assert P99_BUDGET_MS == 350
