ALLOWED = {"EUR", "USD", "GBP", "PLN", "XTS"}

def validate_currency(code: str) -> str:
    c = code.strip().upper()
    if c not in ALLOWED:
        raise ValueError(f"unsupported currency {code}")
    return c
