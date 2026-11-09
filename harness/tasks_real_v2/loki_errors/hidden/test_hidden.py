from notifier.templates import template_for
def test_template():
    assert template_for("order_shipped") == "order_shipped_v3"
    assert template_for("order_refunded") == "order_refunded_v1"
