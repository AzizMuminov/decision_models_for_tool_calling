from reports.buckets import day_bucket


def test_bucket():
    assert day_bucket(1700000000) == "2023-11-14"
