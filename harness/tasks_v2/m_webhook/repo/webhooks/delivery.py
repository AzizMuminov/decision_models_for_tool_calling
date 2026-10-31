# How often a failed webhook delivery is retried.
WEBHOOK_MAX_RETRIES = 5


def current() :
    return WEBHOOK_MAX_RETRIES
