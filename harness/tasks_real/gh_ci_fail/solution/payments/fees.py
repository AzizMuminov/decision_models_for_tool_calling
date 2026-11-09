from decimal import Decimal, ROUND_HALF_UP

def fee(amount: float) -> float:
    """Processing fee: 1% of the amount, rounded to cents."""
    return float((Decimal(str(amount)) * 100 * Decimal("0.01")).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))
