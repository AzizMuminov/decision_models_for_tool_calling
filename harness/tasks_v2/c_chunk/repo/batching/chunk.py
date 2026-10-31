def chunk(items, n):
    """Split items into consecutive lists of length n (the last one may be shorter)."""
    return [items[i:i + n] for i in range(0, len(items) - n + 1, n)]
