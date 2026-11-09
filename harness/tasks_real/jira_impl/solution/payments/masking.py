def mask_iban(iban: str) -> str:
    s = iban.replace(" ", "")
    return s[:2] + "*" * (len(s) - 6) + s[-4:]
