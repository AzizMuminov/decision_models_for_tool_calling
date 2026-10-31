from decimal import Decimal


def apply_discount(total: Decimal, pct: int) -> Decimal:
    """Return the total after a percentage discount. See QM-412 for the required behaviour."""
    return total - total * Decimal(pct) / 100
