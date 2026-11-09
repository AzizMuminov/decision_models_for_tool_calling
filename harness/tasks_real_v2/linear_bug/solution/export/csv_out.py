def to_csv(rows: list[list[str]]) -> str:
    return "\ufeff" + "\n".join(";".join(r) for r in rows)
