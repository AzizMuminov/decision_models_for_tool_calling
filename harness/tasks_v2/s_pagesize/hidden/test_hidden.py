from search.settings import RESULTS_PER_PAGE


def test_authoritative_value():
    assert RESULTS_PER_PAGE == 25
