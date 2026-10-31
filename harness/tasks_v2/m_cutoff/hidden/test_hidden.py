from pipeline.schedule import NIGHTLY_CUTOFF_HOUR


def test_authoritative_value():
    assert NIGHTLY_CUTOFF_HOUR == 3
