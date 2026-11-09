def apply_discount(total_cents: int, coupon: dict) -> int:
    amount = coupon.get("amount_cents") or 0
    return max(0, total_cents - amount)
