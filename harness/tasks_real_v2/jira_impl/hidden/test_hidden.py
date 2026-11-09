from payments.masking import mask_iban
def test_mask():
    assert mask_iban("DE89 3704 0044 0532 0130 00") == "DE" + "*" * 16 + "3000"
    assert mask_iban("PL61109010140000071219812874") == "PL" + "*" * 22 + "2874"
