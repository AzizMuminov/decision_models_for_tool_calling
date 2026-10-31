from ledger.client import LEDGER_TIMEOUT_MS


def test_authoritative_value():
    assert LEDGER_TIMEOUT_MS == 2500
