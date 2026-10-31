# Retry attempts for payment capture; must match the platform config key payments.retry.max_attempts (production).
MAX_ATTEMPTS = 5


def current() :
    return MAX_ATTEMPTS
