from listing.paging import paginate


def test_pages():
    data = list(range(10))
    assert paginate(data, 1, 3) == [0, 1, 2]
    assert paginate(data, 4, 3) == [9]
    assert paginate(data, 5, 3) == []
