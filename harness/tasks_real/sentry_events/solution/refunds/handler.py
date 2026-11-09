def parse_refund(payload: dict) -> dict:
    reason = payload.get("reason_code", payload.get("reason"))
    return {"order_id": payload["order_id"], "reason": reason}
