import os

def db_url() -> str:
    return os.environ["LEDGER_DB_URL"]
