def to_csv(rows: list[list[str]]) -> str:
    return "\n".join(",".join(r) for r in rows)
