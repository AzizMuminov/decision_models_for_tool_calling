import os

def db_url() -> str:
    return os.environ.get("DATABASE_URL") or os.environ["LEDGER_DB_URL"]
