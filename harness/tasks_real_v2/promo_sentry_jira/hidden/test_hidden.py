from checkout.promo import normalize_code
def test_android_codes():
    assert normalize_code(" autumn24 ") == "AUTUMN24"
    assert normalize_code("AUT UMN24") == "AUTUMN24"
