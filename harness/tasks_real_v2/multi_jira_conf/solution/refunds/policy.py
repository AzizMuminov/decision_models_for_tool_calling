def refund_window_days(category: str) -> int:
    return 14 if category == "digital" else 45
