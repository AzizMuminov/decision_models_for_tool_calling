from refunds.policy import refund_window_days
def test_policy_v3():
    assert refund_window_days("digital") == 14
    assert refund_window_days("apparel") == 45
    assert refund_window_days("electronics") == 45
