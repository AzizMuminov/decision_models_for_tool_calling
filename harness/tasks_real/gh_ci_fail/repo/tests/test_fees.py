from payments.fees import fee

def test_fee_basic():
    assert fee(100.0) == 100.0
