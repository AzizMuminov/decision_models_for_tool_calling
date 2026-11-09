from refunds.idempotency import make_key, IDEMPOTENCY_TTL_SECONDS

def test_key_per_refund():
    assert make_key('o_77', 'r_91') == 'refund:o_77:r_91'
    assert make_key('o_77', 'r_91') != make_key('o_77', 'r_92')

def test_ttl_covers_retries():
    assert IDEMPOTENCY_TTL_SECONDS == 86400
