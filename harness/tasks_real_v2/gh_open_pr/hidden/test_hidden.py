from catalog.pager import page
def test_one_based():
    xs = list(range(10))
    assert page(xs, 1, 3) == [0, 1, 2]
    assert page(xs, 4, 3) == [9]
