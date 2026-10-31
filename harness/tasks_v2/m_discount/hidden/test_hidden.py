from decimal import Decimal

import pytest

from checkout.pricing import apply_discount


def test_cap_from_flag():
    assert apply_discount(Decimal("10.00"), 50) == Decimal("7.00")


def test_bankers_rounding():
    assert apply_discount(Decimal("0.25"), 10) == Decimal("0.22")


def test_negative_rejected():
    with pytest.raises(ValueError):
        apply_discount(Decimal("100"), -1)


def test_zero():
    assert apply_discount(Decimal("19.99"), 0) == Decimal("19.99")
