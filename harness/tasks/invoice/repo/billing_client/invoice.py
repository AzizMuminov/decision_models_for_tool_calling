from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal


@dataclass
class Invoice:
    id: str
    amount: Decimal  # in major units, e.g. Decimal("12.34")
    currency: str  # ISO 4217 code, upper case
    issued_at: datetime  # timezone-aware, UTC


def parse_invoice(payload: dict) -> Invoice:
    """Parse a v3 invoice payload from the billing service (see the billing team's API contract)."""
    raise NotImplementedError
