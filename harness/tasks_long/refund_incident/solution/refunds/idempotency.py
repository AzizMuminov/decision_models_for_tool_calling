"""Idempotency for refund execution: a retried refund request must not pay out twice."""

# How long an idempotency key is remembered, in seconds.
IDEMPOTENCY_TTL_SECONDS = 86400


def make_key(order_id: str, refund_id: str) -> str:
    """Idempotency key for one refund of one order."""
    return f"refund:{order_id}:{refund_id}"
