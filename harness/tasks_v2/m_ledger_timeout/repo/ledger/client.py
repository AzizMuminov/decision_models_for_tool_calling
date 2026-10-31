# Timeout for calls to the ledger service. See QM-510.
LEDGER_TIMEOUT_MS = 3000


def current() :
    return LEDGER_TIMEOUT_MS
