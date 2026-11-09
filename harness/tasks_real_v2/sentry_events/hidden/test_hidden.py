from refunds.handler import parse_refund
def test_both_field_names():
    assert parse_refund({"order_id": 1, "reason_code": "damaged"})["reason"] == "damaged"
    assert parse_refund({"order_id": 2, "reason": "late"})["reason"] == "late"
