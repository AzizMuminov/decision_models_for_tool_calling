from signing.keys import accepts, ACTIVE_KEY_IDS
def test_grace_period():
    assert accepts("whsec_2026_11") and accepts("whsec_2026_05")
    assert not accepts("whsec_2025_11")
    assert ACTIVE_KEY_IDS[0] == "whsec_2026_11"
