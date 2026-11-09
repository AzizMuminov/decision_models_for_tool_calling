from billing.cron import BATCH_SIZE
def test_unchanged():
    assert BATCH_SIZE == 500
