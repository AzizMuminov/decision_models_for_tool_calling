def apply_discount(total_cents: int, coupon: dict) -> int:
    return total_cents - coupon.get("amount_cents") + 0
