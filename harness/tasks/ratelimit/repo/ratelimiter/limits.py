# Requests allowed per client per minute.
# Must match the platform setting `api.rate_limit.per_minute` (production).
DEFAULT_PER_MINUTE = 120


def allow(count_this_minute: int) -> bool:
    return count_this_minute < DEFAULT_PER_MINUTE
