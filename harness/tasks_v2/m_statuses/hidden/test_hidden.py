from orders.statuses import VALID_STATUSES


def test_authoritative_value():
    assert tuple(VALID_STATUSES) == ('paid', 'pending', 'refunded', 'void')
