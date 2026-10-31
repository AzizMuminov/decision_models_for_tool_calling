def paginate(items, page, per_page):
    """Return one page of items. Pages are numbered from 1."""
    return items[page * per_page:(page + 1) * per_page]
