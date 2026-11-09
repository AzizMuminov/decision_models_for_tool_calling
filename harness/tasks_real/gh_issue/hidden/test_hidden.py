import pytest
from payments.currency import validate_currency
def test_issue_301():
    assert validate_currency(" eur ") == "EUR"
    assert validate_currency("XTS") == "XTS"
    for bad in ["JPY", "CHF", "EURO"]:
        with pytest.raises(ValueError):
            validate_currency(bad)
