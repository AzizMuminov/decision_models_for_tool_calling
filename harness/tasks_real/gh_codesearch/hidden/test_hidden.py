from shop.currencies import SUPPORTED_CURRENCIES
def test_synced():
    assert sorted(SUPPORTED_CURRENCIES) == sorted(["EUR", "USD", "GBP", "SEK", "PLN", "CZK"])
