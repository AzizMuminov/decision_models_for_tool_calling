def charge_payload(amount_eur: float) -> dict:
    return {"amount": int(round(amount_eur * 100)), "currency": "EUR"}
