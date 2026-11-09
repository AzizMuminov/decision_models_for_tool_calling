def parse_order(doc: dict) -> dict:
    if isinstance(doc.get("customer"), dict):
        email = doc["customer"]["email"]
    else:
        email = doc["customer_email"]
    total = doc["total"]
    if isinstance(total, dict):
        return {"email": email, "total": float(total["amount"]), "currency": total["currency"]}
    return {"email": email, "total": float(total), "currency": doc.get("currency", "EUR")}
