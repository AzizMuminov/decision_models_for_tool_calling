from ops.ownership import LEDGER_OWNER


def test_authoritative_value():
    assert LEDGER_OWNER == 'money-movement'
