from datetime import datetime, timezone
from decimal import Decimal

from billing_client.invoice import parse_invoice


def test_contract_mapping():
    inv = parse_invoice({"inv_ref": "INV-9", "amt_minor": 1234, "ccy": "eur", "issued_ts": 1700000000})
    assert inv.id == "INV-9"
    assert inv.amount == Decimal("12.34")
    assert inv.currency == "EUR"
    assert inv.issued_at == datetime(2023, 11, 14, 22, 13, 20, tzinfo=timezone.utc)
