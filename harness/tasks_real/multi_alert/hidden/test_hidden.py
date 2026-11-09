from checkout.gateway import charge_payload
def test_minor_units():
    assert charge_payload(12.34) == {"amount": 1234, "currency": "EUR"}
    assert charge_payload(0.1 + 0.2)["amount"] == 30
