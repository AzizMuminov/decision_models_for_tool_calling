from billing.invoice import invoice_number
def test_padding():
    assert invoice_number(2026, 42) == "INV-2026-000042"
    assert invoice_number(2026, 123456) == "INV-2026-123456"
