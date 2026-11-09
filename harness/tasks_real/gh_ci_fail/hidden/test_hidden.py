from payments.fees import fee
def test_half_up():
    assert fee(2.675) == 2.68
    assert fee(1.005) == 1.01
    assert fee(100.0) == 100.0
