def fee(amount: float) -> float:
    """Processing fee: 1% of the amount, rounded to cents."""
    return round(amount * 0.01 * 100, 2)
