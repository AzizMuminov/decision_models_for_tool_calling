from datetime import datetime


def day_bucket(ts: int) -> str:
    """Return the reporting day (YYYY-MM-DD) that a Unix timestamp belongs to."""
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
