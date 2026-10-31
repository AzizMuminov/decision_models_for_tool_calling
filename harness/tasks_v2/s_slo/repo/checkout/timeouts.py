# Latency budget for checkout-api; must equal its p99 latency SLO target.
P99_BUDGET_MS = 500


def current() :
    return P99_BUDGET_MS
