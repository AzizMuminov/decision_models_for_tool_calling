def page(items: list, number: int, size: int) -> list:
    start = (number - 1) * size
    return items[start:start + size]
