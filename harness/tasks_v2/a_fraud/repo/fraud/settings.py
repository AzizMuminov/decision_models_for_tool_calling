# Fraud score above which a payment is held; must match the platform config key fraud.score.threshold (production).
SCORE_THRESHOLD = 0.8


def current() :
    return SCORE_THRESHOLD
