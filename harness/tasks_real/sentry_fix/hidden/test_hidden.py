from payments.discounts import apply_discount
def test_expired_coupon():
    assert apply_discount(1000, {"amount_cents": None, "expired": True}) == 1000
    assert apply_discount(1000, {}) == 1000
    assert apply_discount(1000, {"amount_cents": 300}) == 700
