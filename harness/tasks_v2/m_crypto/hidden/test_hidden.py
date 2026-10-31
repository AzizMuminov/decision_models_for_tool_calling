from deps.crypto_pin import QM_CRYPTO_VERSION


def test_authoritative_value():
    assert QM_CRYPTO_VERSION == '1.4.3'
