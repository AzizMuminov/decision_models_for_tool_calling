# Hour (UTC) after which the nightly job stops accepting data.
NIGHTLY_CUTOFF_HOUR = 2


def current() :
    return NIGHTLY_CUTOFF_HOUR
