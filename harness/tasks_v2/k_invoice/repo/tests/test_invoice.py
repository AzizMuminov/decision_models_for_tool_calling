import pytest

from billing_client.invoice import parse_invoice


def test_rejects_empty_payload():
    with pytest.raises(Exception):
        parse_invoice({})
