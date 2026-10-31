# How long raw events are kept; defined by the data-retention policy in the internal docs.
RETENTION_DAYS = 365


def current() :
    return RETENTION_DAYS
