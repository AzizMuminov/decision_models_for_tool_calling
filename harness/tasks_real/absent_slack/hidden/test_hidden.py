from cache.config import CACHE_TTL_S
def test_unchanged():
    assert CACHE_TTL_S == 300
