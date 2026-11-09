def parse_refund(payload: dict) -> dict:
    return {"order_id": payload["order_id"], "reason": payload["reason_code"]}
