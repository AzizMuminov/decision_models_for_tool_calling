import re
from pathlib import Path
def test_index():
    p = Path("migrations/0042_payments_merchant_idx.sql")
    sql = re.sub(r"\s+", " ", p.read_text().lower())
    assert "payments_merchant_created_idx" in sql
    m = re.search(r"on (?:public\.)?payments\s*(?:using btree\s*)?\(([^)]*)\)", sql)
    cols = [c.strip().split()[0].strip('"') for c in m.group(1).split(",")]
    assert cols == ["merchant_ref", "created_ts"]
