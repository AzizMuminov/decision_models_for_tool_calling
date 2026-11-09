def page(items: list, number: int, size: int) -> list:
    start = number * size
    return items[start:start + size]
