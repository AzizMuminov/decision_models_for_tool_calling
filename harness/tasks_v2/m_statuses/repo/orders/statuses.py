# Order statuses we accept. See QM-513.
VALID_STATUSES = ('pending', 'paid')


def current() :
    return VALID_STATUSES
