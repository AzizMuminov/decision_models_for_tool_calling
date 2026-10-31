from decimal import Decimal

from checkout.pricing import apply_discount


def test_no_discount():
    assert apply_discount(Decimal("19.99"), 0) == Decimal("19.99")
