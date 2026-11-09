MAX_RETRIES = 3
BASE_DELAY_S = 1
MAX_DELAY_S = 30

def delay(attempt: int) -> int:
    return min(MAX_DELAY_S, BASE_DELAY_S * 2 ** attempt)
