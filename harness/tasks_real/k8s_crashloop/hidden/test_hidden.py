import os
from worker.settings import db_url
def test_env(monkeypatch):
    monkeypatch.delenv("LEDGER_DB_URL", raising=False)
    monkeypatch.setenv("DATABASE_URL", "postgres://x")
    assert db_url() == "postgres://x"
