from batching.chunk import chunk


def test_cases():
    assert chunk([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]
    assert chunk([1, 2], 5) == [[1, 2]]
    assert chunk([], 3) == []
