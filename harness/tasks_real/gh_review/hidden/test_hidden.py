import pytest
from ledger.entries import make_entry
def test_rename_and_validate():
    assert make_entry("a", amount_cents=5) == {"account": "a", "amount_cents": 5}
    with pytest.raises(ValueError):
        make_entry("a", amount_cents=-1)
