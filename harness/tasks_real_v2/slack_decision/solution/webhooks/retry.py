MAX_RETRIES = 4
BASE_DELAY_S = 3
MAX_DELAY_S = 60

def delay(attempt: int) -> int:
    return min(MAX_DELAY_S, BASE_DELAY_S * 2 ** attempt)
