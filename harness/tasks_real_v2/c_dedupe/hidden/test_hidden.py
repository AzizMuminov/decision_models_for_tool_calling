from collections_util.dedupe import dedupe


def test_cases():
    assert dedupe([3, 1, 3, 2, 1]) == [3, 1, 2]
    assert dedupe(["b", "a", "b"]) == ["b", "a"]
    assert dedupe([]) == []
