# Requests allowed per client per minute; must match the platform setting api.rate_limit.per_minute (production).
DEFAULT_PER_MINUTE = 120


def current() :
    return DEFAULT_PER_MINUTE
