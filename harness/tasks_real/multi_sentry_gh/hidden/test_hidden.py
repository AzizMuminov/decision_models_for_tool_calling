from worker.rates import convert
def test_round_not_truncate():
    assert convert(1000, 1.0999) == 1100
    assert convert(999, 0.5) == 500 or convert(999, 0.5) == 499
    assert convert(3, 0.333334) == 1
