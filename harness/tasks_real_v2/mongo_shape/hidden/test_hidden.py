from export.parse import parse_order
def test_shapes():
    old = {"customer_email": "a@x.io", "total": "12.5", "currency": "EUR"}
    new = {"customer": {"id": "c1", "email": "b@x.io"}, "total": {"amount": "7.25", "currency": "SEK"}}
    assert parse_order(old) == {"email": "a@x.io", "total": 12.5, "currency": "EUR"}
    assert parse_order(new) == {"email": "b@x.io", "total": 7.25, "currency": "SEK"}
