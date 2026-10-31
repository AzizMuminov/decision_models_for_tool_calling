import os
import time

from reports.buckets import day_bucket


def _tz(name):
    os.environ["TZ"] = name
    time.tzset()


def test_auckland_runner():
    _tz("Pacific/Auckland")
    assert day_bucket(1700000000) == "2023-11-14"


def test_los_angeles():
    _tz("America/Los_Angeles")
    assert day_bucket(1700020000) == "2023-11-15"
    _tz("UTC")
