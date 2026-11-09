TEMPLATES = {"order_shipped": "order_shipped_v2", "order_refunded": "order_refunded_v1"}

def template_for(event: str) -> str:
    return TEMPLATES[event]
