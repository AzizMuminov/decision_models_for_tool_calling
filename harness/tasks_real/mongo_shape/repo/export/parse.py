def parse_order(doc: dict) -> dict:
    return {"email": doc["customer_email"], "total": float(doc["total"]), "currency": doc.get("currency", "EUR")}
