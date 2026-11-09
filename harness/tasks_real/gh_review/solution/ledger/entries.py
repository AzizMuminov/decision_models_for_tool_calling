def make_entry(account: str, amount_cents: int) -> dict:
    if amount_cents < 0:
        raise ValueError("amount_cents must be non-negative")
    return {"account": account, "amount_cents": amount_cents}
